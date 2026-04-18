from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from fastapi.responses import FileResponse
import os, json, uuid
import psycopg2
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# DATABASE CONNECTION
# -------------------------
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
conn.autocommit = True

# -------------------------
# MODELS
# -------------------------
class Input(BaseModel):
    text: str
    project_id: str = None

class SelectConcept(BaseModel):
    index: int
    project_id: str

# -------------------------
# LLM
# -------------------------
SYSTEM_BRAIN = """
You are an elite creative studio AI.
Give structured, high-quality outputs.
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
# DB HELPERS
# -------------------------
def create_project():
    project_id = str(uuid.uuid4())
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO projects (id) VALUES (%s)",
        (project_id,)
    )
    return project_id

def get_project(project_id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects WHERE id=%s", (project_id,))
    row = cur.fetchone()
    if not row:
        return None

    return {
        "id": row[0],
        "brief": row[1],
        "analysis": row[2],
        "concepts": row[3],
        "selected": row[4],
        "moodboard": row[5],
        "images": row[6],
        "render3d": row[7],
        "cad": row[8]
    }

def update_project(project_id, field, value):
    cur = conn.cursor()
    cur.execute(f"UPDATE projects SET {field}=%s WHERE id=%s", (json.dumps(value) if isinstance(value,(list,dict)) else value, project_id))

# -------------------------
# AGENTS
# -------------------------
def analysis_agent(b): return llm(f"Analyze:\n{b}")
def concept_agent(a):
    try:
        return json.loads(llm(f"3 concepts JSON:\n{a}"))
    except:
        return []
def moodboard_agent(c): return llm(f"Moodboard:\n{c}")
def render3d_agent(c): return llm(f"3D setup:\n{c}")
def cad_agent(c): return llm(f"CAD layout:\n{c}")

def image_agent(c):
    imgs=[]
    for _ in range(2):
        img = client.images.generate(
            model="gpt-image-1",
            prompt=f"{c}, realistic event render",
            size="1024x1024"
        )
        imgs.append(img.data[0].url)
    return imgs

# -------------------------
# ORCHESTRATOR (DB BASED)
# -------------------------
@app.post("/run")
def run(data: Input):

    # create or load project
    project_id = data.project_id or create_project()
    project = get_project(project_id)

    if not project["brief"]:
        update_project(project_id, "brief", data.text)
        return {"stage":"brief_saved","project_id":project_id}

    if not project["analysis"]:
        result = analysis_agent(project["brief"])
        update_project(project_id,"analysis",result)
        return {"stage":"analysis","project_id":project_id}

    if not project["concepts"]:
        result = concept_agent(project["analysis"])
        update_project(project_id,"concepts",result)
        return {"stage":"concepts","data":result,"project_id":project_id}

    if not project["selected"]:
        return {"stage":"select_concept","options":project["concepts"],"project_id":project_id}

    if not project["moodboard"]:
        result = moodboard_agent(project["selected"])
        update_project(project_id,"moodboard",result)
        return {"stage":"moodboard","project_id":project_id}

    if not project["images"]:
        result = image_agent(project["selected"])
        update_project(project_id,"images",result)
        return {"stage":"images","data":result,"project_id":project_id}

    if not project["render3d"]:
        result = render3d_agent(project["selected"])
        update_project(project_id,"render3d",result)
        return {"stage":"3d_render","project_id":project_id}

    if not project["cad"]:
        result = cad_agent(project["selected"])
        update_project(project_id,"cad",result)
        return {"stage":"cad","project_id":project_id}

    return {"stage":"complete","project_id":project_id}

# -------------------------
# SELECT CONCEPT
# -------------------------
@app.post("/select")
def select(req: SelectConcept):
    project = get_project(req.project_id)
    selected = project["concepts"][req.index]
    update_project(req.project_id,"selected",selected)
    return {"message":"selected"}

# -------------------------
# GET PROJECT
# -------------------------
@app.get("/project/{project_id}")
def fetch(project_id: str):
    return get_project(project_id)
