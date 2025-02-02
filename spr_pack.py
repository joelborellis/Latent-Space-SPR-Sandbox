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
import tiktoken

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
    # Get the directory name from the filepath
    directory = os.path.dirname(filepath)

    # If there's a directory specified and it doesn't exist, create it
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Write the content to the file
    with open(filepath, "a", encoding="utf-8") as outfile:
        outfile.write(content)
        outfile.write("\n\n")  # Add extra newline for separation


def count_openai_tokens(input_string: str, model: str = "gpt-4o") -> int:
    """
    Calculate the total number of OpenAI tokens a string will consume.

    Args:
        input_string (str): The string to tokenize.
        model (str): The OpenAI model to use for tokenization.
                     Defaults to "gpt-3.5-turbo".

    Returns:
        int: The total number of tokens consumed by the string.
    """
    # Load the tokenizer for the specified model
    encoding = tiktoken.encoding_for_model(model)

    # Encode the input string to tokens
    tokens = encoding.encode(input_string)

    # Return the token count
    return len(tokens)


###  OpenAI chat completions call with backoff for rate limits
@backoff.on_exception(backoff.expo, RateLimitError)
async def chat(**kwargs):
    try:
        spinner = Halo(text="Packing SPR...", spinner="dots")
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


async def process_files(
    directory: str,
    prompt_file: str,
    model=OPENAI_MODEL,
    max_completion_tokens: int = 4000,
    temperature: float = 1,
    response_format=ModelResponse,
):

    # Read the system prompt once
    system_prompt = open_file(prompt_file)

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Process only text files
        if os.path.isfile(file_path) and filename.lower().endswith(".txt"):
            # Read the file content
            doc = open_file(file_path)

            # Optionally count tokens and log the count
            context_tokens = count_openai_tokens(doc)
            print(f"Processing '{filename}' - Context tokens: {context_tokens}")

            # Build the conversation for this file
            conversation = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": doc}],
                },
            ]

            # Send the conversation to the chat function and await a response
            text, model_used, tokens, formatted_time = await chat(
                model=model,
                messages=conversation,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                response_format=response_format,
            )

            # Print or process the response as needed
            #print(f"Response for '{filename}':\n{text}\n")
            console = Console()
            console.print(Markdown(text))
            print(f"\nModel used: {model_used}")
            print(f"Your question took a total of: {tokens.total_tokens} tokens")
            print(
                f"Your question took: {tokens.completion_tokens_details.reasoning_tokens} reasoning tokens"
            )
            print(f"Your question prompt used: {tokens.prompt_tokens_details}")
            print(f"Time elapsed: {formatted_time}")

            save_file("./docs/langone/spr/langone_spr.md", text)


if __name__ == "__main__":

    # Define the directory containing the text files and the path to the prompt file
    docs_directory = "./docs/langone/split"
    prompt_file = "./prompts/spr_pack.xml"

    asyncio.run(process_files(docs_directory, prompt_file))
