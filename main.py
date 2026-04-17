from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
import json

# -------------------------
# APP SETUP
# -------------------------
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# INPUT MODELS
# -------------------------
class Input(BaseModel):
    text: str

class SelectConcept(BaseModel):
    index: int

# -------------------------
# LLM BRAIN
# -------------------------
CREATIVE_BRAIN = """
You are an elite creative studio AI.

You specialize in:
- event design
- exhibition experiences
- stage design
- brand activations
- immersive environments

You think like:
- creative director
- experience designer
- visual designer

Your job is to:
- deeply understand the brief
- create strong creative directions
- define mood, visuals, and experience

Outputs must include:
- concept thinking
- mood direction
- visual ideas (lighting, materials, references)

Avoid:
- generic answers
- only booth thinking

Always think in full experience, not just structure.
"""

def llm(prompt):
    res = client.chat.completions.create(
        model="gpt-5.3",
        messages=[
            {"role": "system", "content": CREATIVE_BRAIN},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return res.choices[0].message.content

# -------------------------
# STATE (MEMORY)
# -------------------------
state = {
    "brief": None,
    "analysis": None,
    "concepts": None,
    "selected": None
}

# -------------------------
# AGENTS
# -------------------------
def analysis_agent(brief):
    return llm(f"""
You are a creative strategist.

Understand this creative brief:

{brief}

Return STRICTLY in this structure:

Event Type:
Audience:
Objective:
Experience Goal:
Visual Direction:
Mood & Feel:
Key Elements:
Reference Style:
Constraints:
""")

def concept_agent(analysis):
    prompt = f"""
Based on this:

{analysis}

Create EXACTLY 3 creative concepts.

Each concept must include:
- name
- idea
- experience
- visual
- reference

Return in JSON format:

[
  {{"name":"", "idea":"", "experience":"", "visual":"", "reference":""}},
  {{"name":"", "idea":"", "experience":"", "visual":"", "reference":""}},
  {{"name":"", "idea":"", "experience":"", "visual":"", "reference":""}}
]
"""
    response = llm(prompt)

    try:
        return json.loads(response)
    except:
        return []

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

    # Step 4: WAIT FOR SELECTION
    if not state["selected"]:
        return {
            "stage": "select_concept",
            "options": state["concepts"]
        }

    # Step 5: Confirm selection
    return {
        "stage": "selected",
        "data": state["selected"]
    }

# -------------------------
# SELECT CONCEPT API
# -------------------------
@app.post("/select")
def select_concept(req: SelectConcept):

    if not state["concepts"]:
        return {"error": "No concepts available"}

    if req.index < 0 or req.index > 2:
        return {"error": "Invalid selection"}

    state["selected"] = state["concepts"][req.index]

    return {
        "message": "Concept selected",
        "selected": state["selected"]
    }
