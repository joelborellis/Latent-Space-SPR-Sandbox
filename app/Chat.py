import streamlit as st
from openai import AsyncOpenAI, RateLimitError
import os
from dotenv import load_dotenv
import asyncio
import backoff
import time
from pydantic import BaseModel

load_dotenv()

# Azure OpenAI model from .env file
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")
# Azure OpenAI key from .env file
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Set OpenAI API key from Streamlit secrets
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

class ModelResponse(BaseModel):
    text: str
    model: str

@backoff.on_exception(backoff.expo, RateLimitError)
async def chat(**kwargs):
    try:
        start_time = time.time()  # Record the start time
        print(kwargs)
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

        return text, model, tokens, formatted_time
    except Exception as yikes:
        print(f'\n\nError communicating with OpenAI: "{yikes}"')
        exit(0)

# Asynchronous main function
async def main():
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Accept user input
    with st.sidebar:
        st.header("Reason with o1 Model", divider="gray")
        messages = st.container(height=300)
        if prompt := st.chat_input("Say something"):
            # Here is there we will add the token stuff
            messages.chat_message("user").write(prompt)
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )
    # Display user message in chat message container

    # with st.chat_message("user"):
    # st.markdown(prompt)

    # Display assistant response in chat message container
    if st.session_state.messages:
        with st.status("Reasoning and thinking...", expanded=False) as o1_status:
            text, model, tokens, formatted_time = await chat(
                model=OPENAI_MODEL,
                messages=[
                    {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                    for m in st.session_state.messages
                ],
                max_completion_tokens=4000,
                temperature=1,
                response_format=ModelResponse,
            )
            o1_status.update(label="Complete!", state="complete", expanded=True)

            st.write(f"used model: {model}")
            st.write(f"Your question took a total of: {tokens.total_tokens} tokens")
            st.write(f"Your question took: {tokens.completion_tokens_details.reasoning_tokens} reasoning tokens")
            st.write(f"Your question prompt used: {tokens.prompt_tokens_details}")
            st.write(formatted_time)
 
            st.markdown(text)

            st.session_state.messages.append({"role": "assistant", "content": text})


# Run the asynchronous main function
if __name__ == "__main__":
    asyncio.run(main())
