from openai import AsyncOpenAI, RateLimitError
from dotenv import load_dotenv
import os
from halo import Halo
import asyncio
import backoff
import time
from pydantic import BaseModel
from rich.markdown import Markdown
from rich.console import Console

load_dotenv()

# setup the OpenAI Client
client = AsyncOpenAI()

# Azure OpenAI variables from .env file
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")


class ModelResponse(BaseModel):
    text: str
    model: str


def open_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as infile:
        return infile.read()
    
def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


###  OpenAI chat completions call with backoff for rate limits
@backoff.on_exception(backoff.expo, RateLimitError)
async def chat(**kwargs):
    try:
        spinner = Halo(text="Reasoning...", spinner="dots")
        spinner.start()
        # print(kwargs)

        start_time = time.time()  # Record the start time
        response = await client.beta.chat.completions.parse(**kwargs)
        end_time = time.time()  # Record the end time

        elapsed_time = end_time - start_time  # Calculate the elapsed time in seconds
        minutes, seconds = divmod(
            elapsed_time, 60
        )  # Convert seconds to minutes and seconds
        formatted_time = (
            f"{int(minutes)} minutes and {seconds:.2f} seconds"  # Format the time
        )

        text = response.choices[0].message.parsed.text
        model = response.model
        tokens = response.usage

        spinner.stop()

        return text, model, tokens, formatted_time
    except Exception as yikes:
        print(f'\n\nError communicating with OpenAI: "{yikes}"')
        exit(0)


async def main():

    # Create conversation
    conversation = list()
    # conversation.append({'role': 'system', 'content': open_file('./prompts/system.md')})
    conversation.append(
        {
            "role": "system",
            "content": open_file("./prompts/spr_unpack.xml"),
        }
    )
    conversation.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": open_file("./docs/truein/spr/truein_spr.md")}],
        }
    )

    text, model, tokens, formatted_time = await chat(
        model=OPENAI_MODEL,
        messages=conversation,
        max_completion_tokens=4000,
        temperature=1,
        response_format=ModelResponse,
    )

    console = Console()
    console.print(Markdown(text))
    print(f"\nModel used: {model}")
    print(f"Your question took a total of: {tokens.total_tokens} tokens")
    print(f"Your question took: {tokens.completion_tokens_details.reasoning_tokens} reasoning tokens")
    print(f"Your question prompt used: {tokens.prompt_tokens_details}")
    print(f"Time elapsed: {formatted_time}")

    save_file("./docs/truein/spr/truein_spr_unpack.md", text)


if __name__ == "__main__":
    asyncio.run(main())
