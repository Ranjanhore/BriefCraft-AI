from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

# -------------------------
# APP SETUP
# -------------------------
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# INPUT MODEL
# -------------------------
class Input(BaseModel):
    text: str

# -------------------------
# LLM BRAIN
# -------------------------
CREATIVE_BRAIN = """
You are an elite creative AI studio.

You specialize in:
- event design
- exhibition booths
- stage design

Think like:
- creative director
- spatial designer

Always give:
- premium ideas
- structured outputs
"""

def llm(user_input):
    res = client.chat.completions.create(
        model="gpt-5.3",
        messages=[
            {"role": "system", "content": CREATIVE_BRAIN},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7
    )
    return res.choices[0].message.content

# -------------------------
# STATE (VERY SIMPLE MEMORY)
# -------------------------
state = {
    "brief": None,
    "analysis": None,
    "concepts": None
}

# -------------------------
# AGENTS (SIMPLE)
# -------------------------
def analysis_agent(brief):
    return llm(f"""
Analyze this brief:

{brief}

Give:
- Audience
- Goal
- Tone
- Key requirements
""")

def concept_agent(analysis):
    return llm(f"""
Based on this:

{analysis}

Create 3 creative concepts.

Each with:
- Name
- Idea
- Experience
""")

# -------------------------
# ORCHESTRATOR
# -------------------------
@app.post("/run")
def run(data: Input):

    # Step 1: Save brief
    if not state["brief"]:
        state["brief"] = data.text

    # Step 2: Analysis
    if not state["analysis"]:
        state["analysis"] = analysis_agent(state["brief"])
        return {
            "stage": "analysis",
            "data": state["analysis"]
        }

    # Step 3: Concepts
    if not state["concepts"]:
        state["concepts"] = concept_agent(state["analysis"])
        return {
            "stage": "concepts",
            "data": state["concepts"]
        }

    # Final
    return {
        "stage": "done",
        "data": "Workflow complete"
    }
