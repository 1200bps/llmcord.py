import asyncio
import json
from llmcord import LLMCordBot

async def main():
    with open("config.json", "r") as file:
        configs = json.load(file)

    tasks = []
    for config in configs:
        # Adjust the config to match the expected structure in LLMCordBot
        adjusted_config = {**config.get('llm_settings', {}), **config.get('discord_settings', {})}
        bot = LLMCordBot(adjusted_config)
        tasks.append(bot.start())

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())