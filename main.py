from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid

app = FastAPI(title="BriefCraft AI Engine")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DATABASE (IN-MEMORY FOR MVP) ----------------
PROJECTS = {}
BRIEFS = {}
CONCEPTS = {}
OUTPUTS = {}
FEEDBACK = {}

# ---------------- MODELS ----------------
class ProjectCreate(BaseModel):
    name: str
    project_type: str

class BriefCreate(BaseModel):
    notes: str

class ConceptEdit(BaseModel):
    title: Optional[str] = None
    one_liner: Optional[str] = None
    style: Optional[str] = None

class FeedbackCreate(BaseModel):
    target_type: str   # concept / output / layout / cad
    target_id: str
    rating: int
    comment: str

# ---------------- HEALTH ----------------
@app.get("/health")
def health():
    return {"status": "ok", "app": "BriefCraft AI Engine"}

# ---------------- PROJECT ----------------
@app.post("/v1/projects")
def create_project(payload: ProjectCreate):
    pid = str(uuid.uuid4())

    PROJECTS[pid] = {
        "id": pid,
        "name": payload.name,
        "project_type": payload.project_type
    }

    return {"project_id": pid}

@app.get("/v1/projects")
def list_projects():
    return {"items": list(PROJECTS.values())}

# ---------------- BRIEF ----------------
@app.post("/v1/projects/{project_id}/brief")
def create_brief(project_id: str, payload: BriefCreate):

    if project_id not in PROJECTS:
        raise HTTPException(404, "Project not found")

    BRIEFS[project_id] = payload.notes

    return {"message": "Brief saved"}

# ---------------- CONCEPT ENGINE ----------------
@app.post("/v1/projects/{project_id}/concepts/generate")
def generate_concepts(project_id: str):

    if project_id not in PROJECTS:
        raise HTTPException(404, "Project not found")

    concepts = [
        {
            "id": str(uuid.uuid4()),
            "title": "AI Command Center Booth",
            "one_liner": "Futuristic control room style immersive booth",
            "style": "dark neon tech",
            "board_url": "/mock/3d1"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Smart City Dome Experience",
            "one_liner": "360° immersive dome with AI city simulation",
            "style": "immersive LED dome",
            "board_url": "/mock/3d2"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Robotics Live Arena",
            "one_liner": "Open demo arena for robot interaction",
            "style": "industrial futuristic",
            "board_url": "/mock/3d3"
        }
    ]

    CONCEPTS[project_id] = concepts

    return {"items": concepts}

# ---------------- EDIT CONCEPT ----------------
@app.put("/v1/concepts/{concept_id}")
def edit_concept(concept_id: str, payload: ConceptEdit):

    for project_id, concepts in CONCEPTS.items():
        for c in concepts:
            if c["id"] == concept_id:

                if payload.title:
                    c["title"] = payload.title
                if payload.one_liner:
                    c["one_liner"] = payload.one_liner
                if payload.style:
                    c["style"] = payload.style

                return {"message": "Concept updated", "concept": c}

    raise HTTPException(404, "Concept not found")

# ---------------- OUTPUT GENERATION ----------------
@app.post("/v1/projects/{project_id}/generate-all-from-selected-concept")
def generate_outputs(project_id: str):

    outputs = [
        {
            "id": str(uuid.uuid4()),
            "type": "2D Layout",
            "file_url": "/mock/layout2d.png"
        },
        {
            "id": str(uuid.uuid4()),
            "type": "3D Render",
            "file_url": "/mock/render3d.png"
        },
        {
            "id": str(uuid.uuid4()),
            "type": "CAD Production Drawing",
            "file_url": "/mock/cad.pdf"
        }
    ]

    OUTPUTS[project_id] = outputs

    return {"outputs": outputs}

# ---------------- FEEDBACK SYSTEM ----------------
@app.post("/v1/feedback")
def add_feedback(payload: FeedbackCreate):

    fid = str(uuid.uuid4())

    FEEDBACK[fid] = {
        "id": fid,
        "target_type": payload.target_type,
        "target_id": payload.target_id,
        "rating": payload.rating,
        "comment": payload.comment
    }

    return {"message": "Feedback saved", "id": fid}

@app.get("/v1/feedback")
def get_feedback():
    return {"items": list(FEEDBACK.values())}

# ---------------- MOCK ROUTES (FOR PREVIEW) ----------------
@app.get("/mock/{file_id}")
def mock_files(file_id: str):
    return {
        "preview": f"Mock preview for {file_id}"
    }
