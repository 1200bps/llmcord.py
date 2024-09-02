<h1 align="center">
  llmcord.py – discount janus edition
</h1>

<h3 align="center"><i>
  Talk to LLMs with your LLMs!
</i></h3>

<p align="center">
  <img src="https://files.catbox.moe/pzb1un.png" alt="">
</p>

llmcord.py lets you (and your friends, who are probably LLMs themselves) chat with LLMs directly in Discord. It works with practically any LLM, remote or locally hosted.

This fork is heavily inspired by Janus' (@repligate@twitter) [Chapter II](https://ampdot.mesh.host/chapter2) project.

# Features
## Channel-based context system
Just @ the bot to start a conversation and @ again to continue. The full contents of the channel (up to a specified message limit) comprise the context of the conversation!

You can do things like:
- Have free-flowing conversations with multiple human and nonhuman agents

Additionally:
- All messages in the context are tagged with author metadata, so the LLMs know who said what

## Attachment and web browsing support
The bot supports text file attachments (.md, .c, .py, .json, etcetera) and image attachments (when using vision models). The bot can view webpages, too—ping it with a link and it will see the full text contents of the page (up to 2,000 words).

`TODO: web search support, webpage caching/RAG`

## Choose any LLM
llmcord.py supports remote models from:
- [OpenAI API](https://platform.openai.com/docs/models)
- [Mistral API](https://docs.mistral.ai/platform/endpoints)
- [Groq API](https://console.groq.com/docs/models)
- [OpenRouter API](https://openrouter.ai/docs/models)

Or run a local model with:
- [ollama](https://ollama.com)
- [oobabooga](https://github.com/oobabooga/text-generation-webui)
- [Jan](https://jan.ai)
- [LM Studio](https://lmstudio.ai)

Or use any other OpenAI compatible API server.

## And more:
- Customizable system prompts per-agent
- DM for private access (no @ required)
- User identity aware
- Streamed responses (turns green when complete, automatically splits into separate messages when too long)
- Displays helpful warning messages when appropriate (like "⚠️ Only using last 25 messages" when the customizable message limit is exceeded)
- Caches message data in a size-managed (no memory leaks) and mutex-protected (no race conditions) global dictionary to maximize efficiency and minimize Discord API calls
- Fully asynchronous
- 2 Python files, ??? lines of code (and growing!)

# Instructions
Before you start, install Python and clone this git repo.

1. Install Python requirements: `pip install -U discord.py openai`

2. Create a copy of "config-example.json" named "config.json" and set it up (see below)

3. Run the bot: `nohup python3 main.py &` (the invite URL will print to the console)

## LLM settings:

| Setting | Description |
| --- | --- |
| **providers** | Add the LLM providers you want to use, each with a `base_url` and optional `api_key` entry. Common providers (`openai`, `ollama`, etc.) are already included. **Only supports OpenAI compatible APIs.** |
| **model** | Set to `<provider name>/<model name>`, e.g:<br /><br />-`openai/gpt-4o`<br />-`ollama/llama3.1`<br />-`openrouter/anthropic/claude-3.5-sonnet` |
| **extra_api_parameters** | Extra API parameters for your LLM. Add more entries as needed.<br />(Default: `max_tokens=4096, temperature=1.0`) |
| **system_prompt** | Write anything you want to customize the bot's behavior! |

## Discord settings:

| Setting | Description |
| --- | --- |
| **bot_token** | Create a new Discord bot at [discord.com/developers/applications](https://discord.com/developers/applications) and generate a token under the "Bot" tab. Also enable "MESSAGE CONTENT INTENT" and "SERVER MEMBERS INTENT". |
| **client_id** | Found under the "OAuth2" tab of the Discord bot you just made. |
| **status_message** | Set a custom message that displays on the bot's Discord profile. **Max 128 characters.** |
| **allowed_channel_ids** | A list of Discord channel IDs where the bot can be used. **Leave empty to allow all channels.** |
| **allowed_role_ids** | A list of Discord role IDs that can use the bot. **Leave empty to allow everyone. Specifying at least one role also disables DMs.** |
| **max_text** | The maximum amount of text allowed in a single message, including text from file attachments.<br />(Default: `100,000`) |
| **max_images** | The maximum number of image attachments allowed in a single message. **Only applicable when using a vision model.**<br />(Default: `5`) |
| **max_messages** | The maximum number of messages from the channel that are turned into context.<br />(Default: `75`) |
| **use_plain_responses** | When set to `true` the bot's messages appear more like a regular user message. **This disables embeds, streamed responses and warning messages**.<br />(Default: `false`) |

# Notes
- If you're having issues, try my suggestions [here](https://github.com/jakobdylanc/llmcord.py/issues/19)

- Only models from OpenAI are "user identity aware" because only OpenAI API supports the message "name" property. Other models are given message metadata (including author) inline with the messages themselves.

- PRs are welcome :)
