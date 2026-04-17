from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uuid

from ai import generate_concepts
from db import PROJECTS, BRIEFS, CONCEPTS, OUTPUTS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PROJECT ----------------
@app.post("/project/create")
def create_project():
    pid = str(uuid.uuid4())
    PROJECTS[pid] = {"id": pid}
    return {"project_id": pid}

# ---------------- BRIEF ----------------
@app.post("/project/{pid}/brief")
def save_brief(pid: str, data: dict):
    BRIEFS[pid] = data["brief"]
    return {"status": "saved"}

# ---------------- CONCEPTS ----------------
@app.post("/project/{pid}/concepts")
def concepts(pid: str):

    brief = BRIEFS.get(pid, "")

    data = generate_concepts(brief)

    CONCEPTS[pid] = data

    return {"items": data}

# ---------------- OUTPUTS ----------------
@app.post("/project/{pid}/outputs")
def outputs(pid: str):

    CONCEPTS[pid]

    result = {
        "outputs": [
            {"type": "2D Layout", "url": "/mock/2d"},
            {"type": "3D Render", "url": "/mock/3d"},
            {"type": "CAD", "url": "/mock/cad"}
        ]
    }

    OUTPUTS[pid] = result

    return result
