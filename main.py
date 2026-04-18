from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from fastapi.responses import FileResponse
import os, json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# MODELS
# -------------------------
class Input(BaseModel):
    text: str

class SelectConcept(BaseModel):
    index: int

# -------------------------
# STRONG LLM BRAIN
# -------------------------
SYSTEM_BRAIN = """
You are an elite creative studio AI.

Think like:
- creative director
- spatial designer
- visual artist

Rules:
- Always give structured output
- Be specific (materials, lighting, layout)
- Avoid generic ideas
- Think in real-world execution

IMPORTANT:
If JSON is requested → return ONLY valid JSON
No explanation outside JSON
"""

def llm(prompt):
    res = client.chat.completions.create(
        model="gpt-5.3",
        messages=[
            {"role": "system", "content": SYSTEM_BRAIN},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )
    return res.choices[0].message.content.strip()

# -------------------------
# STATE
# -------------------------
state = {
    "brief": None,
    "analysis": None,
    "concepts": None,
    "selected": None,
    "moodboard": None,
    "images": None,
    "render3d": None,
    "cad": None,
    "pdf": None
}

# -------------------------
# HELPERS
# -------------------------
def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        return []

def build_presentation_text():
    return f"""
BRIEF:
{state['brief']}

ANALYSIS:
{state['analysis']}

SELECTED CONCEPT:
{state['selected']}

MOODBOARD:
{state['moodboard']}

3D RENDER:
{state['render3d']}

CAD:
{state['cad']}
"""

def generate_pdf():
    file_path = "presentation.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []
    for line in build_presentation_text().split("\n"):
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 8))

    doc.build(content)
    return file_path

# -------------------------
# AGENTS
# -------------------------
def analysis_agent(brief):
    return llm(f"""
Analyze this creative brief:

{brief}

Return:
Event Type:
Audience:
Objective:
Experience Goal:
Visual Direction:
Mood & Feel:
Key Elements:
Constraints:
""")

def concept_agent(analysis):
    response = llm(f"""
Based on this:

{analysis}

Return EXACTLY this JSON:

[
  {{
    "name": "",
    "idea": "",
    "experience": "",
    "visual": "",
    "reference": ""
  }},
  {{
    "name": "",
    "idea": "",
    "experience": "",
    "visual": "",
    "reference": ""
  }},
  {{
    "name": "",
    "idea": "",
    "experience": "",
    "visual": "",
    "reference": ""
  }}
]
""")
    return safe_json_parse(response)

def moodboard_agent(concept):
    return llm(f"""
Create a moodboard:

{concept}

Include:
- Colors
- Materials
- Lighting
- Style
- Image prompts
""")

def render3d_agent(concept):
    return llm(f"""
Create 3D render setup:

{concept}

Include:
- Scene
- Camera
- Lighting
- Materials
""")

def cad_agent(concept):
    return llm(f"""
Create CAD layout:

{concept}

Include:
- Dimensions
- Zoning
- Placement
- Technical notes
""")

def image_agent(concept):
    images = []
    for _ in range(2):
        img = client.images.generate(
            model="gpt-image-1",
            prompt=f"{concept}, cinematic lighting, ultra realistic, 4k",
            size="1024x1024"
        )
        images.append(img.data[0].url)
    return images

# -------------------------
# ORCHESTRATOR
# -------------------------
@app.post("/run")
def run(data: Input):

    if not state["brief"]:
        state["brief"] = data.text

    if not state["analysis"]:
        state["analysis"] = analysis_agent(state["brief"])
        return {"stage": "analysis"}

    if not state["concepts"]:
        state["concepts"] = concept_agent(state["analysis"])
        return {"stage": "concepts", "data": state["concepts"]}

    if not state["selected"]:
        return {"stage": "select_concept", "options": state["concepts"]}

    if not state["moodboard"]:
        state["moodboard"] = moodboard_agent(state["selected"])
        return {"stage": "moodboard"}

    if not state["images"]:
        state["images"] = image_agent(state["selected"])
        return {"stage": "images", "data": state["images"]}

    if not state["render3d"]:
        state["render3d"] = render3d_agent(state["selected"])
        return {"stage": "3d_render"}

    if not state["cad"]:
        state["cad"] = cad_agent(state["selected"])
        return {"stage": "cad"}

    if not state["pdf"]:
        state["pdf"] = generate_pdf()
        return {
            "stage": "pdf_ready",
            "download_link": "http://localhost:8000/download"
        }

    return {"stage": "done"}

# -------------------------
# SELECT CONCEPT
# -------------------------
@app.post("/select")
def select(req: SelectConcept):
    state["selected"] = state["concepts"][req.index]
    return {"message": "selected"}

# -------------------------
# DOWNLOAD
# -------------------------
@app.get("/download")
def download():
    return FileResponse("presentation.pdf", media_type='application/pdf', filename="presentation.pdf")
