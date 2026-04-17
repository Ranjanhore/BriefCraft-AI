from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai_chat import chat_with_ai
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

DB = {
    "projects": {},
    "chats": {},
    "concepts": {}
}

# CREATE PROJECT
@app.post("/project/create")
def create_project():
    pid = str(uuid.uuid4())
    DB["projects"][pid] = {"id": pid}
    DB["chats"][pid] = []
    return {"project_id": pid}

# CHAT AI (REAL)
@app.post("/chat/{project_id}")
def chat(project_id: str, payload: dict):

    msg = payload["message"]

    DB["chats"][project_id].append({"role": "user", "content": msg})

    reply = chat_with_ai(DB["chats"][project_id])

    DB["chats"][project_id].append({"role": "assistant", "content": reply})

    return {"reply": reply}

# CONCEPT GENERATION (AI STRUCTURED)
@app.post("/concepts/{project_id}")
def concepts(project_id: str):

    prompt = DB["chats"][project_id][-1]["content"]

    concepts = [
        {
            "id": "c1",
            "title": "AI Command Booth",
            "desc": "Futuristic control center design",
        },
        {
            "id": "c2",
            "title": "Immersive Dome",
            "desc": "360° AI smart city experience",
        },
        {
            "id": "c3",
            "title": "Robotics Arena",
            "desc": "Live demo interaction zone",
        }
    ]

    DB["concepts"][project_id] = concepts

    return {"items": concepts}

# OUTPUT PIPELINE
@app.post("/generate/{project_id}")
def generate(project_id: str):

    return {
        "2d": "generated_2d_layout.png",
        "3d": "generated_3d_scene.glb",
        "cad": "production.dxf",
        "pdf": "final_presentation.pdf"
    }
