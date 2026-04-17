from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Brief(BaseModel):
    text: str


@app.get("/")
def home():
    return {"status": "AI Studio Engine Running"}


@app.post("/generate-concepts")
def generate_concepts(data: Brief):

    prompt = f"""
You are an expert exhibition designer AI.

Convert this brief into 3 design concepts:

Brief:
{data.text}

Return JSON:
[
  {{
    "title": "",
    "description": "",
    "style": "",
    "layout_idea": ""
  }}
]
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"concepts": res.choices[0].message.content}


@app.post("/generate-image-prompt")
def image_prompt(data: dict):

    concept = data.get("concept")

    prompt = f"""
Convert this concept into a hyper realistic 3D render prompt:

Concept:
{concept}

Return only image prompt for AI rendering.
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"prompt": res.choices[0].message.content}
