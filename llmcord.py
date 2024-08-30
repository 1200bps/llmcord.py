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

with open("config.json", "r") as file:
    config = {k: v for d in json.load(file).values() for k, v in d.items()}

LLM_ACCEPTS_IMAGES: bool = any(x in config["model"] for x in ("gpt-4-turbo", "gpt-4o", "claude-3", "gemini", "llava", "vision"))
LLM_ACCEPTS_NAMES: bool = "openai/" in config["model"]

ALLOWED_FILE_TYPES = ("image", "text")
ALLOWED_CHANNEL_TYPES = (discord.ChannelType.text, discord.ChannelType.public_thread, discord.ChannelType.private_thread, discord.ChannelType.private)
ALLOWED_CHANNEL_IDS = config["allowed_channel_ids"]
ALLOWED_ROLE_IDS = config["allowed_role_ids"]

MAX_TEXT = config["max_text"]
MAX_IMAGES = config["max_images"] if LLM_ACCEPTS_IMAGES else 0
MAX_MESSAGES = config["max_messages"]

USE_PLAIN_RESPONSES: bool = config["use_plain_responses"]

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1
MAX_MESSAGE_LENGTH = 2000 if USE_PLAIN_RESPONSES else (4096 - len(STREAMING_INDICATOR))
MAX_MESSAGE_NODES = 100

provider, model = config["model"].split("/", 1)
base_url = config["providers"][provider]["base_url"]
api_key = config["providers"][provider].get("api_key", "None")
openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=config["status_message"][:128] or "github.com/jakobdylanc/llmcord.py")
discord_client = discord.Client(intents=intents, activity=activity)

msg_nodes = {}
last_task_time = None

if config["client_id"] != 123456789:
    print(f"\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={config['client_id']}&permissions=412317273088&scope=bot\n")


@dataclass
class MsgNode:
    data: dict = field(default_factory=dict)
    next_msg: Optional[discord.Message] = None

    too_much_text: bool = False
    too_many_images: bool = False
    has_bad_attachments: bool = False
    fetch_next_failed: bool = False

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def get_system_prompt():
    system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
    if LLM_ACCEPTS_NAMES:
        system_prompt_extras += ["User's names are their Discord IDs and should be typed as '<@ID>'."]

    return {
        "role": "system",
        "content": "\n".join([config["system_prompt"]] + system_prompt_extras),
    }


@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time

    # Filter out unwanted messages
    if (
        new_msg.channel.type not in ALLOWED_CHANNEL_TYPES
        or (new_msg.channel.type != discord.ChannelType.private and discord_client.user not in new_msg.mentions)
        or (ALLOWED_CHANNEL_IDS and not any(id in ALLOWED_CHANNEL_IDS for id in (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None))))
        or (ALLOWED_ROLE_IDS and (new_msg.channel.type == discord.ChannelType.private or not any(role.id in ALLOWED_ROLE_IDS for role in new_msg.author.roles)))
        or new_msg.author.bot
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

    # Generate and send response message(s)
    response_msgs = []
    response_contents = []
    prev_chunk = None
    edit_task = None
    messages = [get_system_prompt(), {"role": "user", "content": context}]
    kwargs = dict(model=model, messages=messages, stream=True, extra_body=config["extra_api_parameters"])
    try:
        async with new_msg.channel.typing():
            async for curr_chunk in await openai_client.chat.completions.create(**kwargs):
                if prev_chunk:
                    prev_content = prev_chunk.choices[0].delta.content or ""
                    curr_content = curr_chunk.choices[0].delta.content or ""

                    if response_contents or prev_content:
                        if not response_contents or len(response_contents[-1] + prev_content) > MAX_MESSAGE_LENGTH:
                            response_contents += [""]

                            if not USE_PLAIN_RESPONSES:
                                embed = discord.Embed(description=(prev_content + STREAMING_INDICATOR), color=EMBED_COLOR_INCOMPLETE)
                                response_msg = await new_msg.channel.send(embed=embed)
                                msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                                await msg_nodes[response_msg.id].lock.acquire()
                                last_task_time = dt.now().timestamp()
                                response_msgs += [response_msg]

                        response_contents[-1] += prev_content

                        if not USE_PLAIN_RESPONSES:
                            is_final_edit = curr_chunk.choices[0].finish_reason != None or len(response_contents[-1] + curr_content) > MAX_MESSAGE_LENGTH

                            if is_final_edit or ((not edit_task or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS):
                                while edit_task and not edit_task.done():
                                    await asyncio.sleep(0)
                                embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                                embed.color = EMBED_COLOR_COMPLETE if is_final_edit else EMBED_COLOR_INCOMPLETE
                                edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                                last_task_time = dt.now().timestamp()

                prev_chunk = curr_chunk

        if USE_PLAIN_RESPONSES:
            full_response = "".join(response_contents)
            split_responses = full_response.split("\n\n")
            for content in split_responses:
                response_msg = await new_msg.channel.send(content=content)
                msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                await msg_nodes[response_msg.id].lock.acquire()
                response_msgs += [response_msg]
    except:
        logging.exception("Error while generating response")

    # Create MsgNode data for response messages
    data = {
        "content": "".join(response_contents),
        "role": "assistant",
    }
    if LLM_ACCEPTS_NAMES:
        data["name"] = str(discord_client.user.id)

    for msg in response_msgs:
        msg_nodes[msg.id].data = data
        msg_nodes[msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                del msg_nodes[msg_id]


async def main():
    await discord_client.start(config["bot_token"])


asyncio.run(main())
