<h1 align="center">
  llmcord.py
</h1>

<h3 align="center"><i>
  Talk to LLMs with your friends!
</i></h3>

<p align="center">
  <img src="https://github.com/jakobdylanc/discord-llm-chatbot/assets/38699060/789d49fe-ef5c-470e-b60e-48ac03057443" alt="">
</p>

llmcord.py lets you (and your friends) chat with LLMs directly in Discord. It works with practically any LLM, remote or locally hosted.

## Features
### Reply-based chat system
Just @ the bot to start a conversation and reply to continue. Build conversations with reply chains!

You can do things like:
- Continue your own conversation or someone else's
- "Rewind" a conversation by simply replying to an older message
- @ the bot while replying to any message in your server to ask a question about it

Additionally:
- Back-to-back messages from the same user are automatically chained together. Just reply to the latest one and the bot will see all of them.
- You can seamlessly move any conversation into a [thread](https://support.discord.com/hc/en-us/articles/4403205878423-Threads-FAQ). Just create a thread from any message and @ the bot inside to continue.

### Choose any LLM
llmcord.py supports remote models from:
- [OpenAI API](https://platform.openai.com/docs/models)
- [Mistral API](https://docs.mistral.ai/platform/endpoints)
- [Groq API](https://console.groq.com/docs/models)

Or run a local model with:
- [ollama](https://ollama.com)
- [oobabooga](https://github.com/oobabooga/text-generation-webui)
- [Jan](https://jan.ai)
- [LM Studio](https://lmstudio.ai)

Or use any other OpenAI compatible API server.

### And more:
- Supports image attachments when using a vision model (like gpt-4o, llava, etc.)
- Supports text file attachments (.txt, .py, .c, etc.)
- Customizable system prompt
- DM for private access (no @ required)
- User identity aware (OpenAI API only)
- Streamed responses (turns green when complete, automatically splits into separate messages when too long)
- Displays helpful warning messages when appropriate (like "⚠️ Only using last 25 messages" when the customizable message limit is exceeded)
- Caches message data in a size-managed (no memory leaks) and mutex-protected (no race conditions) global dictionary to maximize efficiency and minimize Discord API calls
- Fully asynchronous
- 1 Python file, ~200 lines of code

## Instructions
Before you start, install Python and clone this git repo.

1. Install Python requirements: `pip install -U -r requirements.txt`

2. Create a copy of "config-example.json" named "config.json" and set it up (see below)

3. Run the bot: `python llmcord.py` (the invite URL will print to the console)

### LLM settings:

| Setting | Description |
| --- | --- |
| **providers** | Add the LLM providers you want to use, each with a `base_url` and `api_key` entry. Common providers (`openai`, `ollama`, etc.) are already included. **Only supports OpenAI compatible APIs.** |
| **model** | Set to `<provider name>/<model name>`, e.g. `openai/gpt-4o` or `ollama/llama3.1`.<br /><br />For some providers the model name doesn't matter, in which case set to `<provider name>/model` or `<provider name>/vision-model` if it's a vision model. |
| **extra_api_parameters** | Extra API parameters for your LLM. Add more entries as needed.<br />(Default: `max_tokens=4096, temperature=1.0`) |
| **system_prompt** | Write anything you want to customize the bot's behavior! |

### Discord settings:

| Setting | Description |
| --- | --- |
| **bot_token** | Create a new Discord bot at [discord.com/developers/applications](https://discord.com/developers/applications) and generate a token under the "Bot" tab. Also enable "MESSAGE CONTENT INTENT". |
| **client_id** | Found under the "OAuth2" tab of the Discord bot you just made. |
| **status_message** | Set a custom message that displays on the bot's Discord profile. **Max 128 characters.** |
| **allowed_channel_ids** | A list of Discord channel IDs where the bot can be used. **Leave empty to allow all channels.** |
| **allowed_role_ids** | A list of Discord role IDs that can use the bot. **Leave empty to allow everyone. Specifying at least one role also disables DMs.** |
| **max_text** | The maximum amount of text allowed in a single message, including text from file attachments.<br />(Default: `100,000`) |
| **max_images** | The maximum number of image attachments allowed in a single message. **Only applicable when using a vision model.**<br />(Default: `5`) |
| **max_messages** | The maximum number of messages allowed in a reply chain.<br />(Default: `25`) |
| **use_plain_responses** | When set to `true` the bot's messages appear more like a regular user message. **This disables embeds, streamed responses and warning messages**.<br />(Default: `false`) |

## Notes
- If you're having issues, try my suggestions [here](https://github.com/jakobdylanc/discord-llm-chatbot/issues/19)

- Only models from OpenAI are "user identity aware" because only OpenAI API supports the message "name" property. Hopefully others support this in the future.

- PRs are welcome :)

## Star History
<a href="https://star-history.com/#jakobdylanc/discord-llm-chatbot&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/discord-llm-chatbot&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/discord-llm-chatbot&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jakobdylanc/discord-llm-chatbot&type=Date" />
  </picture>
</a>
