from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_concepts(brief: str):

    prompt = f"""
    You are an expert exhibition booth designer.

    Create 3 concepts from this brief:

    {brief}

    Return JSON:
    [
      {{
        "title": "",
        "one_liner": "",
        "style": "",
        "zones": [],
        "visual_prompt": ""
      }}
    ]
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(res.choices[0].message.content)
