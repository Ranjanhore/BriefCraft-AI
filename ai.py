from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_with_ai(messages):

    system_prompt = """
    You are an expert AI Exhibition & Event Design Assistant.

    You must:
    - Ask clarifying questions if brief is incomplete
    - Request missing assets (logo, layout, references)
    - Guide user step-by-step
    - Output structured design direction
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}] + messages
    )

    return response.choices[0].message.content
