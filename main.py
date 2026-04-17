from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import os

app = FastAPI()

# 🔑 SET YOUR KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------
# MODELS
# -----------------------
class ChatRequest(BaseModel):
    message: str

class BriefRequest(BaseModel):
    brief: str

class Concept(BaseModel):
    title: str
    description: str

# -----------------------
# HEALTH
# -----------------------
@app.get("/")
def root():
    return {"message": "AI Creative Studio Backend Running"}

# -----------------------
# CHAT API
# -----------------------
@app.post("/chat")
def chat(req: ChatRequest):
    response = client.chat.completions.create(
        model="gpt-5.3",
        messages=[
            {"role": "system", "content": "You are a creative AI assistant for event, booth and stage design."},
            {"role": "user", "content": req.message}
        ]
    )

    return {"reply": response.choices[0].message.content}

# -----------------------
# GENERATE CONCEPTS
# -----------------------
@app.post("/generate-concepts")
def generate_concepts(req: BriefRequest):

    prompt = f"""
    Convert this creative brief into 3 unique high-quality design concepts.

    Brief:
    {req.brief}

    Return:
    - Concept Name
    - Short Description
    """

    response = client.chat.completions.create(
        model="gpt-5.3",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"concepts": response.choices[0].message.content}

# -----------------------
# NEXT STEPS PLACEHOLDERS
# -----------------------

@app.post("/generate-cad")
def cad():
    return {"status": "CAD layout generated"}

@app.post("/generate-2d")
def two_d():
    return {"status": "2D graphics generated"}

@app.post("/generate-3d")
def three_d():
    return {"status": "3D render generated"}

@app.post("/generate-production")
def production():
    return {"status": "Production drawings ready"}
