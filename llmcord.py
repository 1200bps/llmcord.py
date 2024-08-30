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

        self.user_cooldowns = {}  # Dictionary to store user cooldowns
        self.COOLDOWN_SECONDS = 10  # Define your cooldown period here

        provider, self.model_name = self.config["model"].split("/", 1)
        self.base_url = self.config["providers"][provider]["base_url"]
        self.api_key = self.config["providers"][provider].get("api_key", "None")
        self.openai_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        activity = discord.CustomActivity(name=self.config["status_message"][:128] or "github.com/jakobdylanc/llmcord.py")
        self.discord_client = discord.Client(intents=intents, activity=activity)

        @self.discord_client.event
        async def on_message(new_msg):
            await self.handle_message(new_msg)

    def get_system_prompt(self):
        system_prompt_extras = [f"File attachments are provided to you inline, at the bottom of the context.\nCurrent datetime: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}."]
        if self.LLM_ACCEPTS_NAMES:
            system_prompt_extras += ["User's names are their Discord IDs and should be typed as '<@ID>'."]

        return {
            "role": "system",
            "content": "\n".join([self.config["system_prompt"]] + system_prompt_extras),
        }

    async def handle_message(self, new_msg):
        # Prevent the bot from pinging itself
        if new_msg.author == self.discord_client.user:
            return

        # Check if the user is on cooldown
        current_time = dt.now().timestamp()
        last_ping_time = self.user_cooldowns.get(new_msg.author.id)
        if last_ping_time and (current_time - last_ping_time) < self.COOLDOWN_SECONDS:
            logging.info(f"User {new_msg.author.id} is on cooldown.")
            return

        # Update the user's last ping time
        self.user_cooldowns[new_msg.author.id] = current_time

        # Filter out unwanted messages
        if (
            new_msg.channel.type not in self.ALLOWED_CHANNEL_TYPES
            or (new_msg.channel.type != discord.ChannelType.private and self.discord_client.user not in new_msg.mentions)
            or (self.ALLOWED_CHANNEL_IDS and not any(id in self.ALLOWED_CHANNEL_IDS for id in (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None))))
            or (self.ALLOWED_ROLE_IDS and (new_msg.channel.type == discord.ChannelType.private or not any(role.id in self.ALLOWED_ROLE_IDS for role in new_msg.author.roles)))
        ):
            return
        
        # Wait before collecting messages
        await asyncio.sleep(1)

        # Fetch full channel history with author tags and metadata
        channel_history = []
        async for message in new_msg.channel.history(limit=None):
            author_tag = f"<@{message.author.id}>"
            if isinstance(message.channel, discord.DMChannel) or isinstance(message.channel, discord.GroupChannel):
                author_name = message.author.display_name
            else:
                member = message.author.guild.get_member(message.author.id)
                if member.nick:
                    author_name = member.nick
                else:
                    author_name = message.author.name
            # content = f"{message.content}\n<|begin_metadata|>\nAuthor: {author_name} ({message.author.name})\nAuthor ID: {author_tag}\nTime: {message.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n<|end_metadata|>\n\n\n"
            content = f"{message.content}\n<metadata>\n<author_nick>{author_name}</author_nick>\n<author_name>{message.author.name}</author_name>\n<author_id>{author_tag}</author_id>\n<datetime>{message.created_at.strftime('%Y-%m-%d %H:%M:%S')}</datetime>\n</metadata>\n\n\n"
            channel_history.append(content)

        context = "\n".join(reversed(channel_history))
        
        logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}:\n{new_msg.content}")

        # Initialize this here to avoid AttributeError
        self.images = []

        # Handle attachments
        if new_msg.attachments:
            image_count = 0
            for attachment in new_msg.attachments:
                file_type = attachment.filename.split('.')[-1].lower()
                if file_type in ['png', 'jpg', 'jpeg', 'gif', 'webp'] and self.LLM_ACCEPTS_IMAGES:
                    image_count += 1
                    if image_count > self.MAX_IMAGES:
                        logging.warning(f"Too many images attached by user {new_msg.author.id}")
                        # Set the too_many_images flag in the MsgNode for this message
                        msg_node = self.msg_nodes.get(new_msg.id, MsgNode())
                        msg_node.too_many_images = True
                        self.msg_nodes[new_msg.id] = msg_node
                        break   # Stop processing of remaining images, move on to other attachments
                    else:
                        self.images += [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{attachment.content_type};base64,{base64.b64encode(requests.get(attachment.url).content).decode('utf-8')}"},
                            }
                        ]
                        logging.info(f"Added image attachment: {attachment.filename}")
                elif file_type in ['txt', 'md', 'c', 'cpp', 'py', 'json']:
                    file_content = await attachment.read()
                    file_content_str = file_content.decode('utf-8')
                    context += f"\n<file name=\"{attachment.filename}\">\n```\n{file_content_str}\n```\n</file>\n"
                    logging.info(f"Added text/source file attachment: {attachment.filename}")
                else:
                    logging.warning(f"Unsupported file type: {attachment.filename}")

        logging.info(context)

        # Generate and send response message(s)
        response_msgs = []
        response_contents = [""]  # Initialize with an empty string to avoid IndexError
        prev_chunk = None
        edit_task = None

        messages = [self.get_system_prompt()]
        if context:
            messages.append({"role": "user", "content": [{"type": "text", "text": context}]})
        if self.images:
            for image in self.images:
                messages[-1]["content"].append(image)
                logging.info(f"Image added to content dictionary in messages list successfully")
        kwargs = dict(model=self.model_name, messages=messages, stream=True, extra_body=self.config["extra_api_parameters"])
        try:
            async with new_msg.channel.typing():
                async for curr_chunk in await self.openai_client.chat.completions.create(**kwargs):

                    if prev_chunk:
                        prev_content = prev_chunk.choices[0].delta.content or ""
                        curr_content = curr_chunk.choices[0].delta.content or ""

                        if response_contents or prev_content:
                            if len(response_contents[-1] + prev_content) > self.MAX_MESSAGE_LENGTH:
                                response_contents += [""]

                                if not self.USE_PLAIN_RESPONSES:
                                    embed = discord.Embed(description=(prev_content + self.STREAMING_INDICATOR), color=self.EMBED_COLOR_INCOMPLETE)
                                    response_msg = await new_msg.channel.send(embed=embed)
                                    self.msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                                    await self.msg_nodes[response_msg.id].lock.acquire()
                                    self.last_task_time = dt.now().timestamp()
                                    response_msgs += [response_msg]

                        response_contents[-1] += prev_content

                        if "<metadata>" in response_contents[-1]:
                            # Stop inference upon encountering hallucinated metadata
                            break

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
                    while len(content) > self.MAX_MESSAGE_LENGTH:
                        # Send the first 2000 characters and remove them from content
                        await new_msg.channel.send(content=content[:self.MAX_MESSAGE_LENGTH])
                        content = content[self.MAX_MESSAGE_LENGTH:]
                    if content:  # Ensure the content is not empty
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
