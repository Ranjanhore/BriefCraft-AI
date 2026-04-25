
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from openai import OpenAI
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator

load_dotenv()

APP_NAME = os.getenv("APP_TITLE", "AICreative Studio API").strip() or "AICreative Studio API"
APP_VERSION = "4.0.1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY", "")).strip()
SECRET_KEY = (os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY", "change-me-32char-secret-key-xx")).strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1").strip() or "gpt-image-1"
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts"
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe").strip() or "gpt-4o-mini-transcribe"
TTS_VOICE = os.getenv("TTS_VOICE", "coral").strip() or "coral"
PORT = int(os.getenv("PORT", "10000"))
ALGORITHM = "HS256"
TOKEN_HOURS = int(os.getenv("TOKEN_HOURS", "72"))
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = (BASE_DIR / "exports").resolve()
UPLOAD_DIR = (BASE_DIR / "uploads").resolve()
MEDIA_DIR = (BASE_DIR / "media").resolve()
RENDER_OUTPUT_DIR = (BASE_DIR / "renders").resolve()
VOICE_DIR = (MEDIA_DIR / "voice").resolve()
for _path in (EXPORT_DIR, UPLOAD_DIR, MEDIA_DIR, RENDER_OUTPUT_DIR, VOICE_DIR):
    _path.mkdir(parents=True, exist_ok=True)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://([a-zA-Z0-9-]+\.)?lovable\.(app|dev)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/exports", StaticFiles(directory=str(EXPORT_DIR)), name="exports")
app.mount("/renders", StaticFiles(directory=str(RENDER_OUTPUT_DIR)), name="renders")

try:
    from supabase import create_client
    _sb = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
except Exception:
    _sb = None

_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer(auto_error=False)

_LOCAL: Dict[str, Dict[str, Dict[str, Any]]] = {
    "users": {},
    "projects": {},
}

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def absolute_public_url(relative_url: str) -> str:
    host = os.getenv("RENDER_EXTERNAL_HOSTNAME", "").strip()
    if relative_url.startswith("http://") or relative_url.startswith("https://"):
        return relative_url
    if not host:
        return relative_url
    if not relative_url.startswith("/"):
        relative_url = "/" + relative_url
    return f"https://{host}{relative_url}"

def relative_public_url(path: Path) -> str:
    path = path.resolve()
    for root, prefix in (
        (MEDIA_DIR, "/media"),
        (UPLOAD_DIR, "/uploads"),
        (EXPORT_DIR, "/exports"),
        (RENDER_OUTPUT_DIR, "/renders"),
    ):
        try:
            rel = path.relative_to(root)
            return f"{prefix}/{str(rel).replace(os.sep, '/')}"
        except Exception:
            continue
    return str(path)

def db_insert(table: str, data: Dict[str, Any]) -> Dict[str, Any]:
    if _sb:
        response = _sb.table(table).insert(data).execute()
        return response.data[0] if response.data else data
    row_id = str(data.get("id") or uuid.uuid4())
    row = {**data, "id": row_id}
    _LOCAL.setdefault(table, {})[row_id] = row
    return row

def db_get(table: str, **filters: Any) -> Optional[Dict[str, Any]]:
    if _sb:
        query = _sb.table(table).select("*")
        for key, value in filters.items():
            query = query.eq(key, value)
        response = query.limit(1).execute()
        return response.data[0] if response.data else None
    for row in _LOCAL.get(table, {}).values():
        if all(row.get(k) == v for k, v in filters.items()):
            return row
    return None

def db_list(table: str, limit: int = 100, **filters: Any) -> List[Dict[str, Any]]:
    if _sb:
        query = _sb.table(table).select("*")
        for key, value in filters.items():
            query = query.eq(key, value)
        response = query.limit(limit).execute()
        return response.data or []
    out: List[Dict[str, Any]] = []
    for row in _LOCAL.get(table, {}).values():
        if all(row.get(k) == v for k, v in filters.items()):
            out.append(row)
    return out[:limit]

def db_update(table: str, row_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    payload = {**data, "updated_at": now_iso()}
    if _sb:
        response = _sb.table(table).update(payload).eq("id", row_id).execute()
        if response.data:
            return response.data[0]
        current = db_get(table, id=row_id)
        return {**(current or {}), **payload, "id": row_id}
    current = _LOCAL.setdefault(table, {}).get(row_id)
    if not current:
        raise HTTPException(status_code=404, detail=f"{table} row not found")
    current.update(payload)
    return current

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)

def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> Dict[str, Any]:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    user = db_get("users", id=user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def synthesize_speech(text: str, voice: Optional[str] = None) -> Dict[str, str]:
    if not _openai_client:
        raise HTTPException(status_code=500, detail="OpenAI not configured")
    out_path = VOICE_DIR / f"tts_{uuid.uuid4().hex}.mp3"
    with _openai_client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=(voice or TTS_VOICE),
        input=text[:4096],
        response_format="mp3",
    ) as response:
        response.stream_to_file(out_path)
    rel = relative_public_url(out_path)
    return {"audio_url": absolute_public_url(rel), "audio_path": rel}

def transcribe_audio_file(path: Path) -> str:
    if not _openai_client:
        return "Transcription unavailable: OpenAI not configured."
    with path.open("rb") as file_obj:
        result = _openai_client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=file_obj)
    return getattr(result, "text", "") or ""

def create_pdf(title: str, sections: List[Dict[str, str]], prefix: str) -> Dict[str, str]:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    filename = EXPORT_DIR / f"{prefix}_{uuid.uuid4().hex}.pdf"
    c = canvas.Canvas(str(filename), pagesize=A4)
    width, height = A4
    y = height - 20 * mm
    c.setFont("Helvetica-Bold", 18)
    c.drawString(18 * mm, y, title[:90])
    y -= 14 * mm
    for section in sections:
        if y < 25 * mm:
            c.showPage()
            y = height - 20 * mm
        heading = str(section.get("heading", "Section"))
        body = str(section.get("body", ""))
        c.setFont("Helvetica-Bold", 12)
        c.drawString(18 * mm, y, heading[:90])
        y -= 8 * mm
        c.setFont("Helvetica", 10)
        for paragraph in body.splitlines() or [""]:
            words = paragraph.split()
            line = ""
            for word in words:
                candidate = (line + " " + word).strip()
                if c.stringWidth(candidate, "Helvetica", 10) < width - 36 * mm:
                    line = candidate
                else:
                    c.drawString(18 * mm, y, line)
                    y -= 5 * mm
                    line = word
            if line:
                c.drawString(18 * mm, y, line)
                y -= 5 * mm
        y -= 4 * mm
    c.save()
    rel = relative_public_url(filename)
    return {"pdf_path": rel, "pdf_url": absolute_public_url(rel)}

def infer_event_type(text: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    lower = (text or "").lower()
    if "launch" in lower:
        return "brand launch"
    if "award" in lower:
        return "award show"
    if "wedding" in lower:
        return "wedding"
    return "generic"

def build_analysis(brief: str, event_type: Optional[str]) -> Dict[str, Any]:
    inferred = infer_event_type(brief, event_type)
    return {
        "summary": brief[:300],
        "event_type": inferred,
        "objectives": ["Translate brief into concept", "Prepare department outputs", "Estimate production scope"],
        "audience": "Stakeholders, clients, production team, audience",
        "risks": ["Venue, timelines, or technical constraints may be incomplete"],
        "assumptions": ["Planning-level outputs only until venue details are confirmed"],
    }

def build_concepts(brief: str, analysis: Dict[str, Any], event_type: Optional[str]) -> List[Dict[str, Any]]:
    inferred = infer_event_type(brief, event_type or analysis.get("event_type"))
    return [
        {
            "name": "Cinematic Signature",
            "summary": f"Premium reveal-led concept for {inferred}.",
            "style": "immersive premium",
            "colors": ["black", "gold", "warm white"],
        },
        {
            "name": "Modern Tech Grid",
            "summary": f"Futuristic sharp concept for {inferred}.",
            "style": "futuristic sharp",
            "colors": ["midnight blue", "cyan", "silver"],
        },
        {
            "name": "Elegant Minimal Luxe",
            "summary": f"Refined editorial concept for {inferred}.",
            "style": "clean editorial",
            "colors": ["ivory", "champagne", "graphite"],
        },
    ]

def build_sound_plan() -> Dict[str, Any]:
    return {
        "pdf_sections": [
            {"heading": "Sound Overview", "body": "Planning-level sound system design."},
            {"heading": "Input List", "body": "MC, playback, guest, ambient and redundancy."},
        ],
    }

def build_lighting_plan() -> Dict[str, Any]:
    return {
        "pdf_sections": [
            {"heading": "Lighting Overview", "body": "Concept-driven lighting plan with transitions."},
            {"heading": "Cue Intent", "body": "Opening, transitions, and finale lighting beats."},
        ],
    }

def build_showrunner_plan() -> Dict[str, Any]:
    return {
        "pdf_sections": [
            {"heading": "Show Running", "body": "Cue-based show running script and sequence."},
        ],
        "console_cues": [
            {"cue_no": 1, "name": "Standby", "cue_type": "standby", "actions": []},
            {"cue_no": 2, "name": "House to Half", "cue_type": "lighting", "actions": [{"protocol": "lighting", "target": "house_lights", "value": "half"}]},
            {"cue_no": 3, "name": "Opening AV", "cue_type": "av", "actions": [{"protocol": "av", "target": "screen", "value": "play_opener"}]},
            {"cue_no": 4, "name": "MC Welcome", "cue_type": "sound", "actions": [{"protocol": "sound", "target": "mc_mic", "value": "on"}]},
        ],
    }

def build_scene_json(project: Dict[str, Any]) -> Dict[str, Any]:
    selected = project.get("selected_concept") or {}
    return {
        "venue_type": project.get("event_type", "event"),
        "concept_name": selected.get("name"),
        "stage": {"width": 18000, "depth": 9000, "height": 1200},
    }

def console_state(project: Dict[str, Any]) -> Dict[str, Any]:
    state = project.get("department_outputs") or {}
    if not isinstance(state, dict):
        state = {}
    state.setdefault("armed", False)
    state.setdefault("hold", False)
    state.setdefault("console_index", 0)
    state.setdefault("execution_log", [])
    return state

def log_console_event(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    history = list(state.get("execution_log") or [])
    history.append({"time": now_iso(), **event})
    state["execution_log"] = history[-200:]
    state["last_status"] = event.get("status")
    return state

class SignupIn(BaseModel):
    email: str
    password: str = Field(min_length=8)
    full_name: Optional[str] = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not EMAIL_RE.match(value):
            raise ValueError("Invalid email")
        return value

class LoginIn(BaseModel):
    email: str
    password: str

class ProjectIn(BaseModel):
    title: Optional[str] = None
    name: Optional[str] = None
    brief: Optional[str] = None
    event_type: Optional[str] = None
    style_direction: Optional[str] = None
    style_theme: Optional[str] = None

class RunIn(BaseModel):
    text: str = Field(min_length=3)
    project_id: Optional[str] = None
    name: Optional[str] = None
    event_type: Optional[str] = None
    style_direction: Optional[str] = None

class RunProjectIn(BaseModel):
    text: Optional[str] = None
    event_type: Optional[str] = None

class SelectIn(BaseModel):
    project_id: str
    index: int = Field(ge=0, le=2)

class DeptPDFIn(BaseModel):
    title: Optional[str] = None

class ArmIn(BaseModel):
    armed: bool = True

class TTSIn(BaseModel):
    text: str = Field(min_length=1, max_length=4096)
    voice: Optional[str] = None

@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.get("/")
def root():
    return {"message": f"{APP_NAME} running", "version": APP_VERSION, "time": now_iso(), "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "time": now_iso(), "openai": bool(_openai_client), "supabase": bool(_sb), "port": PORT}

@app.post("/signup")
def signup(payload: SignupIn):
    if db_get("users", email=payload.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = str(uuid.uuid4())
    user = db_insert(
        "users",
        {
            "id": user_id,
            "email": payload.email,
            "password": hash_password(payload.password),
            "full_name": payload.full_name or payload.email.split("@")[0],
            "created_at": now_iso(),
        },
    )
    token = create_access_token(user_id)
    safe_user = {k: v for k, v in user.items() if k != "password"}
    return {"message": "User created", "user_id": user_id, "access_token": token, "token": token, "token_type": "bearer", "user": safe_user}

@app.post("/login")
def login(payload: LoginIn):
    user = db_get("users", email=payload.email.strip().lower())
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    if not verify_password(payload.password, user.get("password", "")):
        raise HTTPException(status_code=400, detail="Wrong password")
    token = create_access_token(str(user["id"]))
    safe_user = {k: v for k, v in user.items() if k != "password"}
    return {"access_token": token, "token": token, "token_type": "bearer", "user_id": str(user["id"]), "user": safe_user}

@app.get("/me")
def me(user: Dict[str, Any] = Depends(get_current_user)):
    return {"user": {k: v for k, v in user.items() if k != "password"}}

@app.get("/projects")
def list_projects(user: Dict[str, Any] = Depends(get_current_user)):
    return {"projects": db_list("projects", user_id=str(user["id"]))}

@app.post("/projects")
def create_project(payload: ProjectIn, user: Dict[str, Any] = Depends(get_current_user)):
    return db_insert("projects", {"id": str(uuid.uuid4()), "user_id": str(user["id"]), "project_name": (payload.title or payload.name or "Untitled").strip(), "brief_text": payload.brief, "event_type": payload.event_type, "style_direction": payload.style_direction, "style_theme": payload.style_theme or "luxury", "status": "draft", "created_at": now_iso(), "updated_at": now_iso()})

@app.get("/projects/{project_id}")
@app.get("/project/{project_id}")
def get_project(project_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    project = db_get("projects", id=project_id, user_id=str(user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": project}

def run_logic(project: Dict[str, Any], text: str, event_type: Optional[str]) -> Dict[str, Any]:
    if text and project.get("brief_text") != text:
        project = db_update("projects", str(project["id"]), {"brief_text": text})
    if event_type and not project.get("event_type"):
        project = db_update("projects", str(project["id"]), {"event_type": event_type})
    analysis = project.get("analysis")
    if not analysis:
        analysis = build_analysis(project.get("brief_text") or text, project.get("event_type") or event_type)
        project = db_update("projects", str(project["id"]), {"analysis": analysis})
    concepts = project.get("concepts")
    if not concepts:
        concepts = build_concepts(project.get("brief_text") or text, analysis, project.get("event_type") or event_type)
        project = db_update("projects", str(project["id"]), {"concepts": concepts, "status": "concepts_ready"})
    return {"message": "Pipeline completed", "project_id": str(project["id"]), "status": "concepts_ready", "analysis": analysis, "concepts": concepts, "project": project}

@app.post("/run")
def run_pipeline(payload: RunIn, user: Dict[str, Any] = Depends(get_current_user)):
    if payload.project_id:
        project = db_get("projects", id=payload.project_id, user_id=str(user["id"]))
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
    else:
        project = db_insert("projects", {"id": str(uuid.uuid4()), "user_id": str(user["id"]), "project_name": (payload.name or payload.text[:50]).strip(), "brief_text": payload.text, "event_type": payload.event_type, "style_direction": payload.style_direction, "status": "draft", "created_at": now_iso(), "updated_at": now_iso()})
    return run_logic(project, payload.text, payload.event_type)

@app.post("/projects/{project_id}/run")
def run_project(project_id: str, payload: RunProjectIn, user: Dict[str, Any] = Depends(get_current_user)):
    project = db_get("projects", id=project_id, user_id=str(user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    text = (payload.text or project.get("brief_text") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text required")
    return run_logic(project, text, payload.event_type or project.get("event_type"))

@app.post("/select")
def select_concept(payload: SelectIn, user: Dict[str, Any] = Depends(get_current_user)):
    project = db_get("projects", id=payload.project_id, user_id=str(user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    concepts = project.get("concepts") or []
    if not concepts:
        raise HTTPException(status_code=400, detail="Run pipeline first")
    if payload.index >= len(concepts):
        raise HTTPException(status_code=400, detail=f"Only {len(concepts)} concepts")
    selected = concepts[payload.index]
    project = db_update("projects", payload.project_id, {"selected_concept": selected, "status": "concept_selected"})
    return {"message": "Concept selected", "index": payload.index, "selected": selected, "project": project}

@app.post("/project/{project_id}/departments/build")
@app.post("/projects/{project_id}/generate-departments")
def build_departments(project_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    project = db_get("projects", id=project_id, user_id=str(user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.get("selected_concept"):
        raise HTTPException(status_code=400, detail="Select a concept first")
    sound = build_sound_plan()
    lighting = build_lighting_plan()
    showrunner = build_showrunner_plan()
    state = console_state(project)
    state.update({"sound_ready": True, "lighting_ready": True, "showrunner_ready": True})
    project = db_update("projects", project_id, {"sound_data": sound, "lighting_data": lighting, "showrunner_data": showrunner, "department_outputs": state, "scene_json": build_scene_json(project), "status": "departments_ready"})
    return {"message": "Departments generated", "project_id": project_id, "sound_data": sound, "lighting_data": lighting, "showrunner_data": showrunner, "project": project}

@app.get("/project/{project_id}/show-console")
def show_console_status(project_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    project = db_get("projects", id=project_id, user_id=str(user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    cues = (project.get("showrunner_data") or {}).get("console_cues") or []
    state = console_state(project)
    idx = min(int(state.get("console_index", 0)), max(len(cues) - 1, 0)) if cues else 0
    return {"project_id": project_id, "armed": bool(state.get("armed")), "hold": bool(state.get("hold")), "cue_index": idx, "cue": cues[idx] if cues else None, "available_cues": cues}

@app.post("/project/{project_id}/show-console/arm")
def show_console_arm(project_id: str, payload: ArmIn, user: Dict[str, Any] = Depends(get_current_user)):
    project = db_get("projects", id=project_id, user_id=str(user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    state = console_state(project)
    state["armed"] = bool(payload.armed)
    state = log_console_event(state, {"status": "armed" if payload.armed else "disarmed"})
    db_update("projects", project_id, {"department_outputs": state})
    return {"message": "Console updated", "armed": state["armed"]}

@app.post("/voice/tts")
@app.post("/tts")
def tts(payload: TTSIn, _: Dict[str, Any] = Depends(get_current_user)):
    audio = synthesize_speech(payload.text, voice=payload.voice)
    return {"message": "Audio generated", "text": payload.text, **audio, "voice": payload.voice or TTS_VOICE, "disclosure": "AI-generated voice"}

@app.post("/voice/transcribe")
async def voice_transcribe(audio_file: UploadFile = File(...), _: Dict[str, Any] = Depends(get_current_user)):
    suffix = Path(audio_file.filename or "audio.webm").suffix or ".webm"
    saved_path = UPLOAD_DIR / f"transcribe_{uuid.uuid4().hex}{suffix}"
    content = await audio_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
    saved_path.write_bytes(content)
    transcript = transcribe_audio_file(saved_path)
    return {"message": "Transcription completed", "transcript": transcript, "input_audio_url": absolute_public_url(relative_public_url(saved_path))}

@app.post("/projects/{project_id}/departments/pdf/sound")
def pdf_sound(project_id: str, payload: DeptPDFIn, user: Dict[str, Any] = Depends(get_current_user)):
    project = db_get("projects", id=project_id, user_id=str(user["id"]))
    if not project or not project.get("sound_data"):
        raise HTTPException(status_code=404, detail="Sound data not found. Build departments first.")
    sections = project["sound_data"].get("pdf_sections") or [{"heading": "Sound", "body": json.dumps(project["sound_data"], indent=2)}]
    return {"project_id": project_id, **create_pdf(payload.title or "Sound Design Manual", sections, "sound_manual")}
