import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime as dt
import json
import logging
import requests
from typing import Optional

import discord
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


@dataclass
class MsgNode:
    data: dict = field(default_factory=dict)
    next_msg: Optional[discord.Message] = None

    too_much_text: bool = False
    too_many_images: bool = False
    has_bad_attachments: bool = False
    fetch_next_failed: bool = False

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class LLMCordBot:
    def __init__(self, config):
        self.config = config
        self.msg_nodes = {}
        self.last_task_time = None

        self.LLM_ACCEPTS_IMAGES = any(x in self.config["model"] for x in ("gpt-4-turbo", "gpt-4o", "claude-3", "gemini", "llava", "vision"))
        self.LLM_ACCEPTS_NAMES = "openai/" in self.config["model"]

        self.ALLOWED_FILE_TYPES = ("image", "text")
        self.ALLOWED_CHANNEL_TYPES = (discord.ChannelType.text, discord.ChannelType.public_thread, discord.ChannelType.private_thread, discord.ChannelType.private)
        self.ALLOWED_CHANNEL_IDS = self.config["allowed_channel_ids"]
        self.ALLOWED_ROLE_IDS = self.config["allowed_role_ids"]

        self.MAX_TEXT = self.config["max_text"]
        self.MAX_IMAGES = self.config["max_images"] if self.LLM_ACCEPTS_IMAGES else 0
        self.MAX_MESSAGES = self.config["max_messages"]

        self.USE_PLAIN_RESPONSES = self.config["use_plain_responses"]

        self.EMBED_COLOR_COMPLETE = discord.Color.dark_green()
        self.EMBED_COLOR_INCOMPLETE = discord.Color.orange()
        self.STREAMING_INDICATOR = " ⚪"
        self.EDIT_DELAY_SECONDS = 1
        self.MAX_MESSAGE_LENGTH = 2000 if self.USE_PLAIN_RESPONSES else (4096 - len(self.STREAMING_INDICATOR))
        self.MAX_MESSAGE_NODES = 100

        provider, model = self.config["model"].split("/", 1)
        self.base_url = self.config["providers"][provider]["base_url"]
        self.api_key = self.config["providers"][provider].get("api_key", "None")
        self.openai_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

        intents = discord.Intents.default()
        intents.message_content = True
        activity = discord.CustomActivity(name=self.config["status_message"][:128] or "github.com/jakobdylanc/llmcord.py")
        self.discord_client = discord.Client(intents=intents, activity=activity)

        @self.discord_client.event
        async def on_message(new_msg):
            await self.handle_message(new_msg)

    def get_system_prompt(self):
        system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
        if self.LLM_ACCEPTS_NAMES:
            system_prompt_extras += ["User's names are their Discord IDs and should be typed as '<@ID>'."]

        return {
            "role": "system",
            "content": "\n".join([self.config["system_prompt"]] + system_prompt_extras),
        }

    async def handle_message(self, new_msg):
        # Filter out unwanted messages
        if (
            new_msg.channel.type not in self.ALLOWED_CHANNEL_TYPES
            or (new_msg.channel.type != discord.ChannelType.private and self.discord_client.user not in new_msg.mentions)
            or (self.ALLOWED_CHANNEL_IDS and not any(id in self.ALLOWED_CHANNEL_IDS for id in (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None))))
            or (self.ALLOWED_ROLE_IDS and (new_msg.channel.type == discord.ChannelType.private or not any(role.id in self.ALLOWED_ROLE_IDS for role in new_msg.author.roles)))
        ):
            return
        
        # Wait for 10 seconds before collecting messages
        await asyncio.sleep(10)

        # Fetch full channel history with author tags
        channel_history = []
        async for message in new_msg.channel.history(limit=None):
            author_tag = f"<@{message.author.id}>"
            content = f"\nauthor: {author_tag}\n{message.content}\n---\n"
            channel_history.append(content)

        context = "\n".join(reversed(channel_history))

        logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}:\n{new_msg.content}")

        # Handle image attachments
        if self.LLM_ACCEPTS_IMAGES and new_msg.attachments:
            for attachment in new_msg.attachments:
                if attachment.filename.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'webp')):
                    image_data = await attachment.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    context += f"\n[Image: {attachment.filename}](data:image/png;base64,{image_base64})\n"

        # Generate and send response message(s)
        response_msgs = []
        response_contents = []
        prev_chunk = None
        edit_task = None
        messages = [self.get_system_prompt(), {"role": "user", "content": context}]
        kwargs = dict(model=self.config["model"], messages=messages, stream=True, extra_body=self.config["extra_api_parameters"])
        try:
            async with new_msg.channel.typing():
                async for curr_chunk in await self.openai_client.chat.completions.create(**kwargs):
                    if prev_chunk:
                        prev_content = prev_chunk.choices[0].delta.content or ""
                        curr_content = curr_chunk.choices[0].delta.content or ""

                        if response_contents or prev_content:
                            if not response_contents or len(response_contents[-1] + prev_content) > self.MAX_MESSAGE_LENGTH:
                                response_contents += [""]

                                if not self.USE_PLAIN_RESPONSES:
                                    embed = discord.Embed(description=(prev_content + self.STREAMING_INDICATOR), color=self.EMBED_COLOR_INCOMPLETE)
                                    response_msg = await new_msg.channel.send(embed=embed)
                                    self.msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                                    await self.msg_nodes[response_msg.id].lock.acquire()
                                    self.last_task_time = dt.now().timestamp()
                                    response_msgs += [response_msg]

                            response_contents[-1] += prev_content

                            if not self.USE_PLAIN_RESPONSES:
                                is_final_edit = curr_chunk.choices[0].finish_reason != None or len(response_contents[-1] + curr_content) > self.MAX_MESSAGE_LENGTH

                                if is_final_edit or ((not edit_task or edit_task.done()) and dt.now().timestamp() - self.last_task_time >= self.EDIT_DELAY_SECONDS):
                                    while edit_task and not edit_task.done():
                                        await asyncio.sleep(0)
                                    embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + self.STREAMING_INDICATOR)
                                    embed.color = self.EMBED_COLOR_COMPLETE if is_final_edit else self.EMBED_COLOR_INCOMPLETE
                                    edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                                    self.last_task_time = dt.now().timestamp()

                    prev_chunk = curr_chunk

            if self.USE_PLAIN_RESPONSES:
                full_response = "".join(response_contents)
                split_responses = full_response.split("\n\n")
                for content in split_responses:
                    response_msg = await new_msg.channel.send(content=content)
                    self.msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                    await self.msg_nodes[response_msg.id].lock.acquire()
                    response_msgs += [response_msg]
        except:
            logging.exception("Error while generating response")

        # Create MsgNode data for response messages
        data = {
            "content": "".join(response_contents),
            "role": "assistant",
        }
        if self.LLM_ACCEPTS_NAMES:
            data["name"] = str(self.discord_client.user.id)

        for msg in response_msgs:
            self.msg_nodes[msg.id].data = data
            self.msg_nodes[msg.id].lock.release()

        # Delete oldest MsgNodes (lowest message IDs) from the cache
        if (num_nodes := len(self.msg_nodes)) > self.MAX_MESSAGE_NODES:
            for msg_id in sorted(self.msg_nodes.keys())[: num_nodes - self.MAX_MESSAGE_NODES]:
                async with self.msg_nodes.setdefault(msg_id, MsgNode()).lock:
                    del self.msg_nodes[msg_id]

    async def start(self):
        await self.discord_client.start(self.config["bot_token"])
