import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
import requests
import sys
import re
from typing import Optional, List, Dict, Any, AsyncGenerator
from bs4 import BeautifulSoup

import discord
from openai import AsyncOpenAI

print("Starting llmcord.py")

# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("llmcord.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Logging initialized")

@dataclass
class MsgNode:
    data: dict = field(default_factory=dict)
    next_msg: Optional[discord.Message] = None
    too_much_text: bool = False
    too_many_images: bool = False
    has_bad_attachments: bool = False
    fetch_next_failed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

class APIClient:
    def __init__(self, config: Dict[str, Any]):
        logging.info("Initializing APIClient")
        provider, self.model_name = config['model'].split("/", 1)
        self.base_url = config['providers'][provider]['base_url']
        self.api_key = config['providers'][provider].get('api_key', 'None')
        self.openai_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        logging.info(f"APIClient initialized with model: {self.model_name}")

    async def generate_response(self, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[Any, None]:
        logging.info("Generating response")
        return await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            **kwargs
        )

class LLMCordBot:
    def __init__(self, config: Dict[str, Any]):
        logging.info("Initializing LLMCordBot")
        self.config = config
        self.msg_nodes: Dict[int, MsgNode] = {}
        self.last_task_time: Optional[float] = None
        self.api_client = APIClient(config)
        self.context = ""

        self.LLM_ACCEPTS_IMAGES = any(x in self.config['model'] for x in ("gpt-4-turbo", "gpt-4o", "claude-3", "gemini", "llava", "vision"))
        self.LLM_ACCEPTS_NAMES = "openai/" in self.config['model']

        self.ALLOWED_FILE_TYPES = ("image", "text")
        self.ALLOWED_CHANNEL_TYPES = (discord.ChannelType.text, discord.ChannelType.public_thread, discord.ChannelType.private_thread, discord.ChannelType.private)
        self.ALLOWED_CHANNEL_IDS = self.config.get('allowed_channel_ids', [])
        self.ALLOWED_ROLE_IDS = self.config.get('allowed_role_ids', [])

        self.MAX_TEXT = self.config.get('max_text', 10000)
        self.MAX_IMAGES = self.config.get('max_images', 5) if self.LLM_ACCEPTS_IMAGES else 0
        self.MAX_MESSAGES = self.config.get('max_messages', 50)

        self.USE_PLAIN_RESPONSES = self.config.get('use_plain_responses', True)

        self.EMBED_COLOR_COMPLETE = discord.Color.dark_green()
        self.EMBED_COLOR_INCOMPLETE = discord.Color.orange()
        self.STREAMING_INDICATOR = " ⚪"
        self.EDIT_DELAY_SECONDS = 1
        self.MAX_MESSAGE_LENGTH = 2000 if self.USE_PLAIN_RESPONSES else (4096 - len(self.STREAMING_INDICATOR))
        self.MAX_MESSAGE_NODES = 200

        self.user_cooldowns: Dict[int, float] = {}
        self.COOLDOWN_SECONDS = 10

        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        activity = discord.CustomActivity(name=self.config.get('status_message', '')[:128] or "github.com/jakobdylanc/llmcord.py")
        self.discord_client = discord.Client(intents=intents, activity=activity)

        @self.discord_client.event
        async def on_message(new_msg: discord.Message):
            await self.handle_message(new_msg)

        logging.info("LLMCordBot initialization complete")

    def get_system_prompt(self) -> Dict[str, str]:
        system_prompt_extras = [
            f"The current UTC date and time are {dt.now().strftime('%Y-%m-%d %H:%M:%S')}."
        ]
        return {
            "role": "system",
            "content": "\n".join([self.config.get('system_prompt', '')] + system_prompt_extras),
        }

    async def handle_message(self, new_msg: discord.Message):
        if new_msg.author == self.discord_client.user:
            return

        await asyncio.sleep(0.1)  # Small delay to reduce likelihood of duplicate message handling

        if not self._is_message_allowed(new_msg):
            return

        if self._is_user_on_cooldown(new_msg.author.id):
            logging.info(f"User {new_msg.author.id} is on cooldown.")
            return

        self._update_user_cooldown(new_msg.author.id)

        logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}:\n{new_msg.content}")

        self.context = await self._fetch_channel_history(new_msg.channel)
        
        self.images = []
        await self._handle_attachments(new_msg)

        await self._generate_and_send_response(new_msg, self.context)

    def _is_message_allowed(self, msg: discord.Message) -> bool:
        allowed = (
            msg.channel.type in self.ALLOWED_CHANNEL_TYPES
            and (msg.channel.type == discord.ChannelType.private or self.discord_client.user.id in [user.id for user in msg.mentions])
            and (not self.ALLOWED_CHANNEL_IDS or any(id in self.ALLOWED_CHANNEL_IDS for id in (msg.channel.id, getattr(msg.channel, "parent_id", None))))
            and (not self.ALLOWED_ROLE_IDS or (msg.channel.type != discord.ChannelType.private and any(role.id in self.ALLOWED_ROLE_IDS for role in msg.author.roles)))
        )
        if not allowed:
            # TODO: fix logspam
            # logging.info(f"Message not allowed: channel_type={msg.channel.type}, mentioned={self.discord_client.user.id in [user.id for user in msg.mentions]}, channel_id={msg.channel.id}")
            pass
        return allowed

    def _is_user_on_cooldown(self, user_id: int) -> bool:
        last_ping_time = self.user_cooldowns.get(user_id)
        return last_ping_time and (dt.now().timestamp() - last_ping_time) < self.COOLDOWN_SECONDS

    def _update_user_cooldown(self, user_id: int):
        self.user_cooldowns[user_id] = dt.now().timestamp()

    async def _fetch_channel_history(self, channel: discord.abc.Messageable) -> str:
        channel_name = getattr(channel, 'name', 'Direct Message')
        logging.info(f"Fetching channel history for channel: {channel_name}")
        channel_history = []
        async for message in channel.history(limit=self.MAX_MESSAGES):
            channel_history.append(message)

        grouped_messages = []
        current_group = []
        last_author_id = None

        for message in reversed(channel_history):
            if message.author.id != last_author_id:
                if current_group:
                    grouped_messages.append(current_group)
                current_group = []
            current_group.append(message)
            last_author_id = message.author.id

        if current_group:
            grouped_messages.append(current_group)

        final_history = []
        for group in grouped_messages:
            author_tag = f"<@{group[0].author.id}>"
            author_name = self._get_author_name(group[0])
            content = "\n\n".join([message.content for message in group])
            metadata = f"<metadata>\n<author_nick>{author_name}</author_nick>\n<author_name>{group[0].author.name}</author_name>\n<author_id>{author_tag}</author_id>\n<datetime>{group[-1].created_at.strftime('%Y-%m-%d %H:%M:%S')}</datetime>\n</metadata>\n\n\n\n"
            final_history.append(f"{content}\n\n{metadata}")

        logging.info(f"Fetched {len(final_history)} grouped messages from channel history")
        return "\n".join(final_history)

    def _get_author_name(self, message: discord.Message) -> str:
        if isinstance(message.channel, (discord.DMChannel, discord.GroupChannel)):
            return message.author.display_name
        member = message.guild.get_member(message.author.id)
        return member.nick if member and member.nick else message.author.name

    async def _handle_attachments(self, msg: discord.Message):
        logging.info(f"Handling attachments and URLs for message: {msg.id}")
        image_count = 0
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', msg.content)
        for attachment in msg.attachments:
            file_type = attachment.filename.split('.')[-1].lower()
            if file_type in ['png', 'jpg', 'jpeg', 'gif', 'webp'] and self.LLM_ACCEPTS_IMAGES:
                image_count += 1
                if image_count > self.MAX_IMAGES:
                    logging.warning(f"Too many images attached by user {msg.author.id}")
                    msg_node = self.msg_nodes.get(msg.id, MsgNode())
                    msg_node.too_many_images = True
                    self.msg_nodes[msg.id] = msg_node
                    break
                else:
                    self.images.append(self._create_image_data(attachment))
                    logging.info(f"Added image attachment: {attachment.filename}")
            elif file_type in ['txt', 'md', 'c', 'cpp', 'py', 'json']:
                file_content = await attachment.read()
                file_content_str = file_content.decode('utf-8')
                self.context += f"\n<file name=\"{attachment.filename}\">\n```\n{file_content_str}\n```\n</file>\n"
                logging.info(f"Added text/source file attachment: {attachment.filename}")
            else:
                logging.warning(f"Unsupported file type: {attachment.filename}")
        for url in urls:
            url_text = await self._extract_text_from_url(url)
            if url_text:
                self.context += f"\n<webpage>\n<url>{url}</url>\n<text>\n{url_text}\n</text>\n</webpage>"
                logging.info(f"Added webpage attachment: {url}")

    async def _extract_text_from_url(self, url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            # Limit to first 2000 words
            words = text.split()[:2000]
            logging.debug(' '.join(words))
            return ' '.join(words)
        except Exception as e:
            logging.error(f"Failed to extract text from URL {url}: {str(e)}")
            return ""

    def _create_image_data(self, attachment: discord.Attachment) -> Dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{attachment.content_type};base64,{base64.b64encode(requests.get(attachment.url).content).decode('utf-8')}"},
        }

    async def _generate_and_send_response(self, new_msg: discord.Message, context: str):
        logging.info("Generating and sending response")
        response_msgs = []
        response_contents = [""]
        prev_chunk = None
        edit_task = None

        messages = [self.get_system_prompt()]
        if context:
            context += f"\n<response user=\"{self.discord_client.user.name}\">"
            logging.debug(context)
            messages.append({"role": "user", "content": [{"type": "text", "text": context}]})
        for image in self.images:
            messages[-1]["content"].append(image)
            logging.debug(f"Image added to content dictionary in messages list")
        
        kwargs = dict(extra_body=self.config.get("extra_api_parameters", {}))

        try:
            async with new_msg.channel.typing():
                async for curr_chunk in await self.api_client.generate_response(messages, **kwargs):
                    should_continue = await self._process_response_chunk(curr_chunk, prev_chunk, response_contents, response_msgs, new_msg, edit_task)
                    if not should_continue:
                        break
                    prev_chunk = curr_chunk

            if self.USE_PLAIN_RESPONSES:
                await self._send_plain_responses(response_contents, new_msg)
            else:
                # Ensure the final message is sent
                if response_contents[-1]:
                    embed = discord.Embed(description=response_contents[-1], color=self.EMBED_COLOR_COMPLETE)
                    await new_msg.channel.send(embed=embed)

        except asyncio.TimeoutError:
            await self._handle_timeout_error(new_msg)
        except Exception as e:
            await self._handle_general_error(new_msg, e)

        await self._update_msg_nodes(response_msgs, response_contents)

    async def _process_response_chunk(self, curr_chunk, prev_chunk, response_contents, response_msgs, new_msg, edit_task):
        if prev_chunk:
            prev_content = prev_chunk.choices[0].delta.content or ""
            curr_content = curr_chunk.choices[0].delta.content or ""

            # Accumulate content
            response_contents[-1] += prev_content

            # Check for metadata or any XML-like tags in the accumulated content
            if re.search(r'<\s*metadata\b', response_contents[-1]):
                logging.warning("Detected hallucinated metadata in LLM response. Stopping inference.")
                return False

            if len(response_contents[-1]) > self.MAX_MESSAGE_LENGTH:
                response_contents += [""]

                if not self.USE_PLAIN_RESPONSES:
                    embed = discord.Embed(description=(response_contents[-1] + self.STREAMING_INDICATOR), color=self.EMBED_COLOR_INCOMPLETE)
                    response_msg = await new_msg.channel.send(embed=embed)
                    self.msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                    await self.msg_nodes[response_msg.id].lock.acquire()
                    self.last_task_time = dt.now().timestamp()
                    response_msgs += [response_msg]

            if not self.USE_PLAIN_RESPONSES:
                is_final_edit = curr_chunk.choices[0].finish_reason is not None or len(response_contents[-1] + curr_content) > self.MAX_MESSAGE_LENGTH

                if is_final_edit or ((not edit_task or edit_task.done()) and dt.now().timestamp() - self.last_task_time >= self.EDIT_DELAY_SECONDS):
                    while edit_task and not edit_task.done():
                        await asyncio.sleep(0)
                    embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + self.STREAMING_INDICATOR)
                    embed.color = self.EMBED_COLOR_COMPLETE if is_final_edit else self.EMBED_COLOR_INCOMPLETE
                    edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                    self.last_task_time = dt.now().timestamp()

        return True

    async def _send_plain_responses(self, response_contents, new_msg):
        logging.info("Sending plain responses")
        full_response = "".join(response_contents)
        split_responses = full_response.split("\n\n")
        for content in split_responses:
            while len(content) > self.MAX_MESSAGE_LENGTH:
                await new_msg.channel.send(content=content[:self.MAX_MESSAGE_LENGTH])
                content = content[self.MAX_MESSAGE_LENGTH:]
            if content:
                response_msg = await new_msg.channel.send(content=content)
                self.msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                await self.msg_nodes[response_msg.id].lock.acquire()

    async def _handle_timeout_error(self, new_msg):
        logging.error("API request timed out")
        error_message = "[ The API request timed out—please try again later ]"
        if not self.USE_PLAIN_RESPONSES:
            embed = discord.Embed(description=error_message, color=discord.Color.red())
            await new_msg.channel.send(embed=embed)
        else:
            await new_msg.channel.send(content=error_message)

    async def _handle_general_error(self, new_msg, error):
        logging.exception("Error while generating response")
        error_message = f"[ An error occurred while generating the response: {str(error)} ]"
        if not self.USE_PLAIN_RESPONSES:
            embed = discord.Embed(description=error_message, color=discord.Color.red())
            await new_msg.channel.send(embed=embed)
        else:
            await new_msg.channel.send(content=error_message)

    async def _update_msg_nodes(self, response_msgs, response_contents):
        logging.info("Updating message nodes")
        data = {
            "content": "".join(response_contents),
            "role": "assistant",
        }
        if self.LLM_ACCEPTS_NAMES:
            data["name"] = str(self.discord_client.user.id)

        for msg in response_msgs:
            self.msg_nodes[msg.id].data = data
            self.msg_nodes[msg.id].lock.release()

        await self._prune_msg_nodes()

    async def _prune_msg_nodes(self):
        logging.debug("Pruning message nodes")
        if (num_nodes := len(self.msg_nodes)) > self.MAX_MESSAGE_NODES:
            for msg_id in sorted(self.msg_nodes.keys())[: num_nodes - self.MAX_MESSAGE_NODES]:
                async with self.msg_nodes[msg_id].lock:
                    del self.msg_nodes[msg_id]

    async def start(self):
        logging.info("Starting LLMCordBot")
        try:
            await self.discord_client.start(self.config['bot_token'])
        except Exception as e:
            logging.exception(f"Error starting LLMCordBot: {str(e)}")
            raise

logging.info("llmcord.py module loaded")
