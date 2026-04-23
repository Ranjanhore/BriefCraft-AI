from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel
from supabase import Client, create_client

load_dotenv()

APP_NAME = "Creative Brief to Concept & Execution API"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000").strip() or "*"
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", "./exports")).resolve()
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads")).resolve()

def _split_origins(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]

ALLOWED_ORIGINS = _split_origins(
    os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,https://briefly-sparkle.lovable.app",
    )
)

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
if not SUPABASE_URL:
    raise RuntimeError("Missing SUPABASE_URL")
if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_SERVICE_ROLE_KEY")

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_NAME)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
# ------------------------------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------------------------------
class UserInput(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)
    full_name: Optional[str] = Field(default=None, max_length=120)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        email = value.strip().lower()
        if not EMAIL_RE.match(email):
            raise ValueError("Invalid email address")
        return email

    @field_validator("full_name")
    @classmethod
    def normalize_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class LoginInput(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        email = value.strip().lower()
        if not EMAIL_RE.match(email):
            raise ValueError("Invalid email address")
        return email


class ProjectCreateInput(BaseModel):
    title: Optional[str] = None
    name: Optional[str] = None
    brief: Optional[str] = None
    event_type: Optional[str] = None
    style_direction: Optional[str] = None


class RunInput(BaseModel):
    text: str = Field(min_length=3)
    project_id: Optional[str] = None
    name: Optional[str] = None
    event_type: Optional[str] = None
    style_direction: Optional[str] = None


class RunProjectInput(BaseModel):
    text: Optional[str] = None
    name: Optional[str] = None
    event_type: Optional[str] = None
    style_direction: Optional[str] = None


class SelectConceptInput(BaseModel):
    project_id: str
    index: int = Field(ge=0, le=2)


class SelectConceptCompatInput(BaseModel):
    concept_index: Optional[int] = Field(default=None, ge=0, le=2)
    index: Optional[int] = Field(default=None, ge=0, le=2)


class CommentInput(BaseModel):
    project_id: str
    section: str
    comment_text: str


class UpdateProjectInput(BaseModel):
    project_id: str
    field: str
    value: Any


class DepartmentPDFRequest(BaseModel):
    title: Optional[str] = None


class VoiceSessionCreateInput(BaseModel):
    project_id: Optional[str] = None
    title: Optional[str] = None
    system_prompt: Optional[str] = None
    voice: Optional[str] = None


class VoiceTextInput(BaseModel):
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    text: str = Field(min_length=1)
    voice: Optional[str] = None
    voice_instructions: Optional[str] = None
    title: Optional[str] = None
    system_prompt: Optional[str] = None


class TTSInput(BaseModel):
    text: str = Field(min_length=1, max_length=4096)
    voice: Optional[str] = None
    instructions: Optional[str] = None


class ControlActionInput(BaseModel):
    target: Optional[str] = None
    protocol: str
    ip: Optional[str] = None
    port: Optional[int] = None
    address: Optional[str] = None
    args: Optional[List[Any]] = None
    base_url: Optional[str] = None
    path: Optional[str] = None
    method: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    body: Optional[Dict[str, Any]] = None
    output_name: Optional[str] = None
    message_type: Optional[str] = None
    channel: Optional[int] = None
    note: Optional[int] = None
    velocity: Optional[int] = None
    control: Optional[int] = None
    value: Optional[int] = None
    program: Optional[int] = None
    data_bytes: Optional[List[int]] = None


class ArmInput(BaseModel):
    armed: bool = True


class CueJumpInput(BaseModel):
    cue_index: Optional[int] = Field(default=None, ge=0)
    cue_no: Optional[int] = None


class VisualPolicyInput(BaseModel):
    preview_size: Optional[str] = None
    master_size: Optional[str] = None
    print_size: Optional[str] = None
    aspect_ratio: Optional[str] = None


class AssetCreateInput(BaseModel):
    asset_type: str
    title: str
    prompt: str = Field(min_length=3)
    section: Optional[str] = None
    job_kind: Optional[str] = None
    source_asset_id: Optional[str] = None
    generate_now: bool = True


class MoodboardGenerateInput(BaseModel):
    concept_index: Optional[int] = Field(default=None, ge=0, le=2)
    count: int = Field(default=3, ge=1, le=6)
    generate_now: bool = True


class JobQueueInput(BaseModel):
    agent_type: str
    job_type: str
    title: Optional[str] = None
    priority: int = Field(default=5, ge=1, le=10)
    input_data: Optional[Dict[str, Any]] = None


class OrchestrateInput(BaseModel):
    auto_generate_moodboard: bool = True
    queue_3d: bool = True
    queue_video: bool = True
    queue_cad: bool = True
    queue_manuals: bool = True


class ReferenceUploadMeta(BaseModel):
    title: Optional[str] = None
    section: Optional[str] = None


class ElementSheetGenerateInput(BaseModel):
    include_sound: bool = True
    include_lighting: bool = True
    include_scenic: bool = True
    include_power_summary: bool = True
    include_xlsx: bool = True
    sheet_title: Optional[str] = None


class ShowTrialGenerateInput(BaseModel):
    include_walkthrough: bool = True
    include_audio_video: bool = True
    include_camera_pan: bool = True
    queue_render_jobs: bool = True
    draft_name: Optional[str] = None


class ShowTrialUpdateInput(BaseModel):
    trial_data: Dict[str, Any]


class ShowTrialFinalizeInput(BaseModel):
    use_trial_cues: bool = True
    mark_ready: bool = True


# ------------------------------------------------------------------------------
# APP
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    if OPENAI_API_KEY and OpenAI is not None:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception:
            client = None
    if DATABASE_URL:
        create_tables()
    yield


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https:\/\/([a-zA-Z0-9-]+\.)?lovable\.(app|dev)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/renders", StaticFiles(directory=str(RENDER_OUTPUT_DIR)), name="renders")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": f"{APP_TITLE} is running",
        "time": now_iso(),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    db_ok = False
    db_error = None
    if DATABASE_URL:
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("select 1 as ok")
                    db_ok = cur.fetchone()["ok"] == 1
        except Exception as e:
            db_error = str(e)

    return {
        "status": "ok",
        "time": now_iso(),
        "database": {"configured": bool(DATABASE_URL), "ok": db_ok, "error": db_error},
        "openai": {
            "configured": bool(OPENAI_API_KEY),
            "tts_model": TTS_MODEL,
            "transcribe_model": TRANSCRIBE_MODEL,
            "default_voice": TTS_VOICE,
        },
        "show_control": {
            "gateway_configured": bool(SHOW_GATEWAY_URL),
            "companion_configured": bool(COMPANION_BASE_URL),
            "resolume_configured": bool(RESOLUME_BASE_URL),
            "qlab_osc_configured": bool(QLAB_OSC_IP and QLAB_OSC_PORT),
            "eos_osc_configured": bool(EOS_OSC_IP and EOS_OSC_PORT),
        },
    }


@app.post("/signup")
def signup(payload: UserInput):
    user = create_user(payload.email, payload.password, payload.full_name)
    token = create_access_token(str(user["id"]))
    return {
        "message": "User created",
        "user_id": str(user["id"]),
        "access_token": token,
        "token": token,
        "token_type": "bearer",
    }


@app.post("/login")
def login(payload: LoginInput):
    user = get_user_by_email(payload.email)
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    password_hash = user.get("password")
    if not password_hash or not verify_password(payload.password, password_hash):
        raise HTTPException(status_code=400, detail="Wrong password")

    token = create_access_token(str(user["id"]))
    return {
        "access_token": token,
        "token": token,
        "token_type": "bearer",
        "user_id": str(user["id"]),
    }


@app.post("/logout")
def logout(_: Dict[str, Any] = Depends(get_current_user)):
    return {"message": "Logged out on server side. Remove the bearer token on the frontend/client."}


@app.get("/me")
def me(current_user: Dict[str, Any] = Depends(get_current_user)):
    return {"user": current_user}


@app.get("/projects")
def get_projects(current_user: Dict[str, Any] = Depends(get_current_user)):
    projects = list_projects(str(current_user["id"]))
    return {"projects": projects}


@app.post("/projects")
def create_project_endpoint(payload: ProjectCreateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    name = (payload.title or payload.name or "Untitled Project").strip()
    project = create_project(
        str(current_user["id"]),
        name,
        payload.brief,
        payload.event_type,
        payload.style_direction,
    )
    snapshot_project_version(str(project["id"]), str(current_user["id"]), "Project created")
    return project


@app.get("/projects/{project_id}")
def get_project_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": project}


@app.get("/project/{project_id}")
def get_project_alias(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.post("/project/update")
def update_project_endpoint(payload: UpdateProjectInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = update_project_fields(
        payload.project_id,
        str(current_user["id"]),
        {payload.field: payload.value},
    )
    snapshot_project_version(payload.project_id, str(current_user["id"]), f"Updated field: {payload.field}")
    return {"message": "Project updated", "project": project}


@app.post("/comment")
def create_comment_endpoint(payload: CommentInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(payload.project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    comment = add_comment(payload.project_id, str(current_user["id"]), payload.section, payload.comment_text)
    return {"message": "Comment added", "comment": comment}


@app.get("/comments/{project_id}")
def comments_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    return {"comments": list_comments(project_id, str(current_user["id"]))}


def _run_pipeline_logic(project: Dict[str, Any], text: str, event_type: Optional[str], user_id: str) -> Dict[str, Any]:
    project_id = str(project["id"])

    updates: Dict[str, Any] = {}
    if text and (not project.get("brief") or project.get("brief") != text):
        updates["brief"] = text
    if event_type and not project.get("event_type"):
        updates["event_type"] = event_type
    if updates:
        project = update_project_fields(project_id, user_id, updates)

    analysis = project.get("analysis")
    if not analysis:
        analysis = analyze_brief(project.get("brief") or text, project.get("event_type") or event_type)
        project = update_project_fields(project_id, user_id, {"analysis": analysis})

    concepts = load_json(project.get("concepts"))
    if not concepts:
        concepts = generate_concepts(project.get("brief") or text, analysis, project.get("event_type") or event_type)
        project = update_project_fields(project_id, user_id, {"concepts": concepts, "status": "concepts_ready"})

    snapshot_project_version(project_id, user_id, "Pipeline completed to concepts stage")
    project = get_project_by_id(project_id, user_id=user_id) or project

    return {
        "message": "Pipeline completed",
        "project_id": project_id,
        "status": "concepts_ready",
        "brief": project.get("brief"),
        "analysis": project.get("analysis"),
        "concepts": project.get("concepts"),
        "project": project,
    }


@app.post("/run")
def run_pipeline(payload: RunInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    if payload.project_id:
        project = get_project_by_id(payload.project_id, user_id=user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
    else:
        project = create_project(
            user_id,
            (payload.name or "Untitled Project").strip(),
            payload.text,
            payload.event_type,
            payload.style_direction,
        )
        snapshot_project_version(str(project["id"]), user_id, "Project created from /run")
    return _run_pipeline_logic(project, payload.text, payload.event_type, user_id)


@app.post("/projects/{project_id}/run")
def run_pipeline_alias(project_id: str, payload: RunProjectInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    text = (payload.text or project.get("brief") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Field 'text' is required if the project brief is empty")

    extra_updates: Dict[str, Any] = {}
    if payload.name:
        extra_updates["name"] = payload.name.strip()
    if payload.event_type:
        extra_updates["event_type"] = payload.event_type
    if payload.style_direction:
        extra_updates["style_direction"] = payload.style_direction
    if extra_updates:
        project = update_project_fields(project_id, user_id, extra_updates)

    return _run_pipeline_logic(project, text, payload.event_type or project.get("event_type"), user_id)


@app.post("/select")
def select_concept(payload: SelectConceptInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(payload.project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    concepts = load_json(project.get("concepts")) or []
    if not isinstance(concepts, list) or not concepts:
        raise HTTPException(status_code=400, detail="Run pipeline first to generate concepts")
    if payload.index < 0 or payload.index >= len(concepts):
        raise HTTPException(status_code=400, detail="Invalid concept index")

    selected = concepts[payload.index]
    project = update_project_fields(payload.project_id, user_id, {"selected": selected, "status": "concept_selected"})
    snapshot_project_version(payload.project_id, user_id, f"Selected concept {payload.index}")
    moodboard_assets: List[Dict[str, Any]] = []
    moodboard_error = None
    if AUTO_GENERATE_MOODBOARD_ON_SELECT:
        try:
            result = generate_moodboards_endpoint(payload.project_id, MoodboardGenerateInput(concept_index=payload.index, count=3, generate_now=True), current_user)
            moodboard_assets = result.get("assets") or []
        except Exception as exc:
            moodboard_error = str(exc)

    return {
        "message": "Concept selected",
        "index": payload.index,
        "selected": selected,
        "project": project,
        "moodboard_assets": moodboard_assets,
        "moodboard_error": moodboard_error,
    }


@app.post("/projects/{project_id}/select-concept")
def select_concept_alias(project_id: str, payload: SelectConceptCompatInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    index = payload.concept_index if payload.concept_index is not None else payload.index
    if index is None:
        raise HTTPException(status_code=422, detail="concept_index is required")
    return select_concept(SelectConceptInput(project_id=project_id, index=index), current_user)


def _build_departments_logic(project_id: str, user_id: str) -> Dict[str, Any]:
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not load_json(project.get("selected")):
        raise HTTPException(status_code=400, detail="Select a concept first")

    sound_data = generate_sound_department(project)
    lighting_data = generate_lighting_department(project)
    showrunner_data = generate_showrunner_department(project)

    outputs = get_console_state(project)
    outputs.update({
        "sound_ready": True,
        "lighting_ready": True,
        "showrunner_ready": True,
        "console_index": 0,
        "hold": False,
        "last_status": "departments_ready",
    })

    project = update_project_fields(
        project_id,
        user_id,
        {
            "sound_data": sound_data,
            "lighting_data": lighting_data,
            "showrunner_data": showrunner_data,
            "department_outputs": outputs,
            "status": "departments_ready",
        },
    )
    snapshot_project_version(project_id, user_id, "Departments generated")

    return {
        "message": "Departments generated",
        "project_id": project_id,
        "sound_data": sound_data,
        "lighting_data": lighting_data,
        "showrunner_data": showrunner_data,
        "department_outputs": outputs,
        "project": project,
    }


@app.post("/project/{project_id}/departments/build")
def build_departments(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    return _build_departments_logic(project_id, str(current_user["id"]))


@app.post("/projects/{project_id}/generate-departments")
def build_departments_alias(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    return _build_departments_logic(project_id, str(current_user["id"]))


@app.get("/project/{project_id}/departments/manuals")
def department_manuals(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        "project_id": project_id,
        "sound_data": project.get("sound_data"),
        "lighting_data": project.get("lighting_data"),
        "showrunner_data": project.get("showrunner_data"),
    }


@app.post("/project/{project_id}/departments/pdf/sound")
def export_sound_pdf(project_id: str, payload: DepartmentPDFRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.get("sound_data"):
        raise HTTPException(status_code=404, detail="Sound data not found. Build departments first.")

    pdf = create_simple_pdf(
        payload.title or "Sound Design Manual",
        (project["sound_data"] or {}).get("pdf_sections") or project["sound_data"],
        "sound_manual",
    )
    return {"project_id": project_id, **pdf}


@app.post("/project/{project_id}/departments/pdf/lighting")
def export_lighting_pdf(project_id: str, payload: DepartmentPDFRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.get("lighting_data"):
        raise HTTPException(status_code=404, detail="Lighting data not found. Build departments first.")

    pdf = create_simple_pdf(
        payload.title or "Lighting Design Manual",
        (project["lighting_data"] or {}).get("pdf_sections") or project["lighting_data"],
        "lighting_manual",
    )
    return {"project_id": project_id, **pdf}


@app.post("/project/{project_id}/departments/pdf/showrunner")
def export_showrunner_pdf(project_id: str, payload: DepartmentPDFRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.get("showrunner_data"):
        raise HTTPException(status_code=404, detail="Show runner data not found. Build departments first.")

    pdf = create_simple_pdf(
        payload.title or "Show Running Script",
        (project["showrunner_data"] or {}).get("pdf_sections") or project["showrunner_data"],
        "showrunner_manual",
    )
    return {"project_id": project_id, **pdf}


# ------------------------------------------------------------------------------
# REAL SHOW CONSOLE ROUTES
# ------------------------------------------------------------------------------
@app.post("/control/execute")
def execute_control(payload: ControlActionInput, _: Dict[str, Any] = Depends(get_current_user)):
    result = execute_control_action(payload.model_dump(exclude_none=True))
    return {"message": "Action executed", "result": result}


@app.get("/project/{project_id}/show-console")
def show_console_status(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    showrunner = load_json(project.get("showrunner_data")) or {}
    cues = showrunner.get("console_cues") or []
    state = get_console_state(project)
    idx = min(int(state.get("console_index", 0)), max(len(cues) - 1, 0)) if cues else 0

    return {
        "project_id": project_id,
        "armed": bool(state.get("armed")),
        "hold": bool(state.get("hold")),
        "cue_index": idx,
        "cue": cues[idx] if cues else None,
        "next_cue": cues[idx + 1] if cues and idx + 1 < len(cues) else None,
        "available_cues": cues,
        "department_outputs": state,
    }


@app.post("/project/{project_id}/show-console/arm")
def show_console_arm(project_id: str, payload: ArmInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    state = get_console_state(project)
    state["armed"] = bool(payload.armed)
    state = log_console_event(state, {"status": "armed" if payload.armed else "disarmed", "message": "Console arm state updated"})
    project = save_console_state(project_id, user_id, state)
    return {"message": "Console updated", "armed": state["armed"], "department_outputs": project.get("department_outputs")}


@app.post("/project/{project_id}/show-console/standby")
def show_console_standby(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    state = get_console_state(project)
    state["hold"] = False
    state = log_console_event(state, {"status": "standby", "message": "Standby called"})
    project = save_console_state(project_id, user_id, state)
    return {"message": "Standby called", "department_outputs": project.get("department_outputs")}


@app.post("/project/{project_id}/show-console/hold")
def show_console_hold(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    state = get_console_state(project)
    state["hold"] = True
    state = log_console_event(state, {"status": "hold", "message": "Show hold engaged"})
    project = save_console_state(project_id, user_id, state)
    return {"message": "Hold engaged", "department_outputs": project.get("department_outputs")}


@app.post("/project/{project_id}/show-console/back")
def show_console_back(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    showrunner = load_json(project.get("showrunner_data")) or {}
    cues = showrunner.get("console_cues") or []
    state = get_console_state(project)
    idx = max(int(state.get("console_index", 0)) - 1, 0)
    state["console_index"] = idx
    state = log_console_event(state, {"status": "back", "message": "Moved back one cue", "cue_index": idx, "cue": cues[idx] if cues else None})
    project = save_console_state(project_id, user_id, state)
    return {"message": "Moved back", "cue_index": idx, "cue": cues[idx] if cues else None, "department_outputs": project.get("department_outputs")}


@app.post("/project/{project_id}/show-console/next")
def show_console_next(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    showrunner = load_json(project.get("showrunner_data")) or {}
    cues = showrunner.get("console_cues") or []
    state = get_console_state(project)
    if not cues:
        raise HTTPException(status_code=400, detail="No console cues found")
    idx = min(int(state.get("console_index", 0)) + 1, len(cues) - 1)
    state["console_index"] = idx
    state = log_console_event(state, {"status": "next", "message": "Moved to next cue", "cue_index": idx, "cue": cues[idx]})
    project = save_console_state(project_id, user_id, state)
    return {"message": "Moved to next cue", "cue_index": idx, "cue": cues[idx], "department_outputs": project.get("department_outputs")}


@app.post("/project/{project_id}/show-console/go")
def show_console_go(project_id: str, execute: bool = Query(True), current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    showrunner = load_json(project.get("showrunner_data")) or {}
    cues = showrunner.get("console_cues") or []
    if not cues:
        raise HTTPException(status_code=400, detail="No console cues found")

    state = get_console_state(project)
    if not state.get("armed"):
        raise HTTPException(status_code=400, detail="Console is not armed")
    if state.get("hold"):
        raise HTTPException(status_code=400, detail="Console is on hold")

    idx = min(int(state.get("console_index", 0)), len(cues) - 1)
    cue = cues[idx]
    actions = cue.get("actions") or []
    results = []

    if execute:
        for action in actions:
            results.append(execute_control_action(action))

    next_idx = min(idx + 1, len(cues) - 1)
    state["console_index"] = next_idx
    state["last_run_cue"] = cue.get("cue_no")
    state["last_run_at"] = now_iso()
    state = log_console_event(
        state,
        {
            "status": "go",
            "message": "Cue executed" if execute else "Cue previewed",
            "cue_index": idx,
            "cue": cue,
            "results": results,
        },
    )
    project = save_console_state(project_id, user_id, state)

    return {
        "message": "Cue executed" if execute else "Cue previewed",
        "executed": execute,
        "cue_index": idx,
        "cue": cue,
        "results": results,
        "next_index": next_idx,
        "department_outputs": project.get("department_outputs"),
    }


@app.post("/project/{project_id}/show-console/panic")
def show_console_panic(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    state = get_console_state(project)
    state["hold"] = True
    state["armed"] = False
    state = log_console_event(state, {"status": "panic", "message": "Panic triggered. Console disarmed and hold engaged."})
    project = save_console_state(project_id, user_id, state)
    return {"message": "Panic triggered", "department_outputs": project.get("department_outputs")}


# ------------------------------------------------------------------------------
# VOICE ROUTES
# ------------------------------------------------------------------------------
@app.post("/voice/sessions")
def create_voice_session_endpoint(payload: VoiceSessionCreateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project_id = ensure_uuid(payload.project_id, "project_id") if payload.project_id else None
    if project_id:
        project = get_project_by_id(project_id, user_id=str(current_user["id"]))
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
    session = create_voice_session(
        str(current_user["id"]),
        project_id,
        payload.title,
        payload.system_prompt,
        payload.voice,
    )
    return {"session": session}


@app.get("/voice/sessions")
def list_voice_sessions_endpoint(current_user: Dict[str, Any] = Depends(get_current_user)):
    return {"sessions": list_voice_sessions(str(current_user["id"]))}


@app.get("/voice/sessions/{session_id}")
def get_voice_session_endpoint(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    session = get_voice_session_by_id(session_id, str(current_user["id"]))
    if not session:
        raise HTTPException(status_code=404, detail="Voice session not found")
    messages = get_voice_messages(session_id, str(current_user["id"]), limit=100)
    return {"session": session, "messages": messages}


@app.post("/voice/tts")
def voice_tts(payload: TTSInput, _: Dict[str, Any] = Depends(get_current_user)):
    audio = synthesize_speech(payload.text, voice=payload.voice, instructions=payload.instructions, filename_prefix="tts")
    return {
        "message": "Audio generated",
        "text": payload.text,
        **audio,
        "disclosure": "AI-generated voice",
    }


@app.post("/voice/chat")
def voice_chat_text(payload: VoiceTextInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = None
    project_id = ensure_uuid(payload.project_id, "project_id") if payload.project_id else None
    if project_id:
        project = get_project_by_id(project_id, user_id=user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    session = None
    session_id = ensure_uuid(payload.session_id, "session_id") if payload.session_id else None
    if session_id:
        session = get_voice_session_by_id(session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Voice session not found")
    else:
        session = create_voice_session(user_id, project_id, payload.title or "Voice Chat", payload.system_prompt, payload.voice)
        session_id = str(session["id"])

    prior_messages = get_voice_messages(session_id, user_id, limit=30)
    user_text = payload.text.strip()

    add_voice_message(session_id, "user", user_text, transcript=user_text, meta={"input_type": "text"})

    reply = generate_voice_reply(current_user, session, project, user_text, prior_messages)
    audio = synthesize_speech(
        reply,
        voice=payload.voice or session.get("voice") or TTS_VOICE,
        instructions=payload.voice_instructions or VOICE_DEFAULT_INSTRUCTIONS,
        filename_prefix="assistant_reply",
    )
    assistant_message = add_voice_message(
        session_id,
        "assistant",
        reply,
        transcript=reply,
        audio_url=audio["audio_url"],
        meta={"voice": audio["voice"], "response_format": audio["response_format"]},
    )
    touch_voice_session(session_id, user_id, voice=audio["voice"])

    return {
        "message": "Voice response generated",
        "session_id": session_id,
        "project_id": project_id,
        "user_text": user_text,
        "assistant_text": reply,
        "assistant_message": assistant_message,
        **audio,
        "disclosure": "AI-generated voice",
    }


@app.post("/voice/chat-audio")
async def voice_chat_audio(
    audio_file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    project_id: Optional[str] = Form(default=None),
    voice: Optional[str] = Form(default=None),
    voice_instructions: Optional[str] = Form(default=None),
    title: Optional[str] = Form(default=None),
    system_prompt: Optional[str] = Form(default=None),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = str(current_user["id"])
    project = None
    safe_project_id = ensure_uuid(project_id, "project_id") if project_id else None
    if safe_project_id:
        project = get_project_by_id(safe_project_id, user_id=user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    session = None
    safe_session_id = ensure_uuid(session_id, "session_id") if session_id else None
    if safe_session_id:
        session = get_voice_session_by_id(safe_session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Voice session not found")
    else:
        session = create_voice_session(user_id, safe_project_id, title or "Voice Chat", system_prompt, voice)
        safe_session_id = str(session["id"])

    suffix = Path(audio_file.filename or "audio.webm").suffix or ".webm"
    saved_path = UPLOAD_DIR / f"voice_input_{uuid.uuid4().hex}{suffix}"
    content = await audio_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
    saved_path.write_bytes(content)

    transcript = transcribe_audio_file(saved_path)
    prior_messages = get_voice_messages(safe_session_id, user_id, limit=30)
    add_voice_message(
        safe_session_id,
        "user",
        transcript,
        transcript=transcript,
        meta={
            "input_type": "audio",
            "uploaded_filename": audio_file.filename,
            "saved_input_path": relative_public_url(saved_path),
        },
    )

    reply = generate_voice_reply(current_user, session, project, transcript, prior_messages)
    audio = synthesize_speech(
        reply,
        voice=voice or session.get("voice") or TTS_VOICE,
        instructions=voice_instructions or VOICE_DEFAULT_INSTRUCTIONS,
        filename_prefix="assistant_reply",
    )
    assistant_message = add_voice_message(
        safe_session_id,
        "assistant",
        reply,
        transcript=reply,
        audio_url=audio["audio_url"],
        meta={"voice": audio["voice"], "response_format": audio["response_format"]},
    )
    touch_voice_session(safe_session_id, user_id, voice=audio["voice"])

    return {
        "message": "Voice response generated from uploaded audio",
        "session_id": safe_session_id,
        "project_id": safe_project_id,
        "transcript": transcript,
        "assistant_text": reply,
        "assistant_message": assistant_message,
        "input_audio_url": absolute_public_url(relative_public_url(saved_path)),
        **audio,
        "disclosure": "AI-generated voice",
    }


@app.post("/voice/transcribe")
async def voice_transcribe(
    audio_file: UploadFile = File(...),
    _: Dict[str, Any] = Depends(get_current_user),
):
    suffix = Path(audio_file.filename or "audio.webm").suffix or ".webm"
    saved_path = UPLOAD_DIR / f"transcribe_{uuid.uuid4().hex}{suffix}"
    content = await audio_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
    saved_path.write_bytes(content)
    transcript = transcribe_audio_file(saved_path)
    return {
        "message": "Transcription completed",
        "transcript": transcript,
        "input_audio_url": absolute_public_url(relative_public_url(saved_path)),
    }


# ------------------------------------------------------------------------------
# COMPATIBILITY SHOW CONSOLE ROUTE
# ------------------------------------------------------------------------------
@app.post("/project/{project_id}/show-console")
def show_console_compat(project_id: str, command: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    cmd = (command or "").strip().lower()
    if cmd in {"next", "next cue"}:
        return show_console_next(project_id, current_user)
    if cmd in {"back", "prev", "previous"}:
        return show_console_back(project_id, current_user)
    if cmd in {"go", "run", "play"}:
        return show_console_go(project_id, execute=True, current_user=current_user)
    if cmd in {"hold", "pause"}:
        return show_console_hold(project_id, current_user)
    if cmd in {"standby"}:
        return show_console_standby(project_id, current_user)
    if cmd in {"panic", "stop"}:
        return show_console_panic(project_id, current_user)
    return show_console_status(project_id, current_user)


# ------------------------------------------------------------------------------
# ADVANCED ASSETS / AGENT JOBS / 16:9 VISUAL PIPELINE
# ------------------------------------------------------------------------------
def asset_row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(row)
    if "meta" in item:
        item["meta"] = load_json(item.get("meta"))
    return item


def job_row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(row)
    for key in ("input_data", "output_data"):
        if key in item:
            item[key] = load_json(item.get(key))
    return item


def activity_row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(row)
    if "meta" in item:
        item["meta"] = load_json(item.get("meta"))
    return item


def parse_size(value: str, fallback: Tuple[int, int]) -> Tuple[int, int]:
    try:
        left, right = str(value).lower().split("x", 1)
        w, h = int(left), int(right)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return fallback


def normalize_visual_size(size: Optional[str], fallback: str) -> str:
    raw = (size or fallback or "3840x2160").lower().strip()
    w, h = parse_size(raw, parse_size(fallback, (3840, 2160)))
    if h == 0:
        return fallback
    ratio = round(w / h, 4)
    if abs(ratio - (16 / 9)) > 0.02:
        w = int(round(h * 16 / 9))
    if w > 3840 or h > 3840:
        scale = min(3840 / max(w, 1), 3840 / max(h, 1))
        w = max(512, int(w * scale))
        h = max(512, int(h * scale))
        w = int(round(h * 16 / 9))
    return f"{w}x{h}"

def normalize_visual_size(size: Optional[str], fallback: str) -> str:
    raw = (size or fallback or "3840x2160").lower().strip()
    w, h = parse_size(raw, parse_size(fallback, (3840, 2160)))
    if h == 0:
        return fallback
    ratio = round(w / h, 4)
    if abs(ratio - (16 / 9)) > 0.02:
        w = int(round(h * 16 / 9))
    if w > 3840 or h > 3840:
        scale = min(3840 / max(w, 1), 3840 / max(h, 1))
        w = max(512, int(w * scale))
        h = max(512, int(h * scale))
        w = int(round(h * 16 / 9))
    return f"{w}x{h}"


def choose_openai_image_size(policy: Dict[str, Any]) -> str:
    target = normalize_visual_size(
        str(policy.get("master_size") or VISUAL_MASTER_SIZE),
        VISUAL_MASTER_SIZE,
    )
    w, h = parse_size(target, (3840, 2160))

    if w == h:
        return "1024x1024"
    if w > h:
        return "1536x1024"
    return "1024x1536"


def default_visual_policy() -> Dict[str, Any]:
    preview = normalize_visual_size(VISUAL_PREVIEW_SIZE, "1920x1080")
    master = normalize_visual_size(VISUAL_MASTER_SIZE, "3840x2160")
    ...


def default_visual_policy() -> Dict[str, Any]:
    preview = normalize_visual_size(VISUAL_PREVIEW_SIZE, "1920x1080")
    master = normalize_visual_size(VISUAL_MASTER_SIZE, "3840x2160")
    pw, ph = parse_size(VISUAL_PRINT_SIZE, (5760, 3240))
    if abs((pw / max(ph, 1)) - (16 / 9)) > 0.02:
        pw = int(round(ph * 16 / 9))
    return {
        "aspect_ratio": VISUAL_ASPECT_RATIO or "16:9",
        "preview_size": preview,
        "master_size": master,
        "print_size": f"{pw}x{ph}",
        "preview_format": "jpg",
        "master_format": "png",
        "print_format": "png",
        "quality": IMAGE_QUALITY or "high",
        "printable": True,
        "created_from": "default_policy",
    }


def merged_visual_policy(project: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    policy = default_visual_policy()
    if project:
        custom = load_json(project.get("visual_policy")) or {}
        if isinstance(custom, dict):
            policy.update({k: v for k, v in custom.items() if v not in (None, "")})
    policy["preview_size"] = normalize_visual_size(policy.get("preview_size"), default_visual_policy()["preview_size"])
    policy["master_size"] = normalize_visual_size(policy.get("master_size"), default_visual_policy()["master_size"])
    return policy


def ensure_visual_policy(project_id: str, user_id: str) -> Dict[str, Any]:
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    policy = merged_visual_policy(project)
    if load_json(project.get("visual_policy")) != policy:
        update_project_fields(project_id, user_id, {"visual_policy": policy})
    return policy


def safe_filename(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name or "asset")).strip("_")
    return value or "asset"


@with_db
def create_project_asset(cur, project_id: str, user_id: str, asset_type: str, title: str, prompt: Optional[str] = None, section: Optional[str] = None, job_kind: Optional[str] = None, status: str = "queued", preview_url: Optional[str] = None, master_url: Optional[str] = None, print_url: Optional[str] = None, source_file_url: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    asset_id = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.project_assets (
            id, project_id, user_id, asset_type, section, job_kind, title, prompt, status,
            preview_url, master_url, print_url, source_file_url, meta
        ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (asset_id, project_id, user_id, asset_type, section, job_kind, title, prompt, status, preview_url, master_url, print_url, source_file_url, dump_json(meta or {})),
    )
    cur.execute("select * from public.project_assets where id = %s", (asset_id,))
    return asset_row_to_dict(cur.fetchone())


@with_db
def update_project_asset(cur, asset_id: str, user_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
    if not values:
        cur.execute("select * from public.project_assets where id = %s and user_id = %s", (asset_id, user_id))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Asset not found")
        return asset_row_to_dict(row)
    allowed = {"asset_type", "section", "job_kind", "title", "prompt", "status", "preview_url", "master_url", "print_url", "source_file_url", "meta"}
    clean = {k: v for k, v in values.items() if k in allowed}
    if not clean:
        raise HTTPException(status_code=400, detail="No valid asset fields supplied")
    assignments = []
    params: List[Any] = []
    for key, value in clean.items():
        assignments.append(f"{key} = %s")
        params.append(dump_json(value) if isinstance(value, (dict, list)) else value)
    assignments.append("updated_at = now()")
    params.extend([asset_id, user_id])
    cur.execute(f"update public.project_assets set {', '.join(assignments)} where id = %s and user_id = %s", params)
    cur.execute("select * from public.project_assets where id = %s and user_id = %s", (asset_id, user_id))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset_row_to_dict(row)


@with_db
def list_project_assets(cur, project_id: str, user_id: str, section: Optional[str] = None) -> List[Dict[str, Any]]:
    if section:
        cur.execute(
            """
            select * from public.project_assets
            where project_id = %s and user_id = %s and section = %s
            order by created_at desc
            """,
            (project_id, user_id, section),
        )
    else:
        cur.execute(
            """
            select * from public.project_assets
            where project_id = %s and user_id = %s
            order by created_at desc
            """,
            (project_id, user_id),
        )
    return [asset_row_to_dict(r) for r in cur.fetchall()]


@with_db
def get_project_asset_by_id(cur, asset_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    cur.execute("select * from public.project_assets where id = %s and user_id = %s limit 1", (asset_id, user_id))
    row = cur.fetchone()
    return asset_row_to_dict(row) if row else None


@with_db
def create_agent_job(cur, project_id: str, user_id: str, agent_type: str, job_type: str, title: str, priority: int = 5, input_data: Optional[Dict[str, Any]] = None, status: str = "queued", parent_job_id: Optional[str] = None) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.agent_jobs (
            id, project_id, user_id, agent_type, job_type, title, status, priority, input_data, parent_job_id
        ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (job_id, project_id, user_id, agent_type, job_type, title, status, priority, dump_json(input_data or {}), parent_job_id),
    )
    cur.execute("select * from public.agent_jobs where id = %s", (job_id,))
    return job_row_to_dict(cur.fetchone())


@with_db
def update_agent_job(cur, job_id: str, user_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {"agent_type", "job_type", "title", "status", "priority", "progress", "input_data", "output_data", "error_text", "parent_job_id", "started_at", "completed_at"}
    clean = {k: v for k, v in values.items() if k in allowed}
    if not clean:
        cur.execute("select * from public.agent_jobs where id = %s and user_id = %s", (job_id, user_id))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        return job_row_to_dict(row)
    assignments = []
    params: List[Any] = []
    for key, value in clean.items():
        assignments.append(f"{key} = %s")
        params.append(dump_json(value) if isinstance(value, (dict, list)) else value)
    assignments.append("updated_at = now()")
    params.extend([job_id, user_id])
    cur.execute(f"update public.agent_jobs set {', '.join(assignments)} where id = %s and user_id = %s", params)
    cur.execute("select * from public.agent_jobs where id = %s and user_id = %s", (job_id, user_id))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_row_to_dict(row)


@with_db
def list_agent_jobs(cur, project_id: str, user_id: str) -> List[Dict[str, Any]]:
    cur.execute(
        """
        select * from public.agent_jobs
        where project_id = %s and user_id = %s
        order by created_at desc
        """,
        (project_id, user_id),
    )
    return [job_row_to_dict(r) for r in cur.fetchall()]


@with_db
def add_project_activity(cur, project_id: str, user_id: Optional[str], activity_type: str, title: str, detail: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    activity_id = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.project_activity_logs (id, project_id, user_id, activity_type, title, detail, meta)
        values (%s, %s, %s, %s, %s, %s, %s)
        """,
        (activity_id, project_id, user_id, activity_type, title, detail, dump_json(meta or {})),
    )
    cur.execute("select * from public.project_activity_logs where id = %s", (activity_id,))
    return activity_row_to_dict(cur.fetchone())


@with_db
def list_project_activity(cur, project_id: str, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    cur.execute(
        """
        select * from public.project_activity_logs
        where project_id = %s and (user_id = %s or user_id is null)
        order by created_at desc
        limit %s
        """,
        (project_id, user_id, max(1, min(limit, 500))),
    )
    return [activity_row_to_dict(r) for r in cur.fetchall()]


def save_binary_image_versions(image_bytes: bytes, title: str, policy: Dict[str, Any], folder_name: str = "visuals") -> Dict[str, str]:
    folder = MEDIA_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_filename(title)}_{uuid.uuid4().hex}"
    master_path = folder / f"{stem}_master.png"
    master_path.write_bytes(image_bytes)

    preview_format = str(policy.get("preview_format") or "jpg").lower()
    preview_path = folder / f"{stem}_preview.{preview_format}"
    print_format = str(policy.get("print_format") or "png").lower()
    print_path = folder / f"{stem}_print.{print_format}"

    try:
        from PIL import Image
        from PIL import ImageOps

        master_img = Image.open(master_path).convert("RGB")

        pw, ph = parse_size(str(policy.get("preview_size") or "1920x1080"), (1920, 1080))
        mw, mh = parse_size(str(policy.get("master_size") or "3840x2160"), master_img.size)
        tw, th = parse_size(str(policy.get("print_size") or "5760x3240"), (5760, 3240))

        if master_img.size != (mw, mh):
            master_img = master_img.resize((mw, mh), Image.LANCZOS)
            master_img.save(master_path, format="PNG")

        preview_img = master_img.resize((pw, ph), Image.LANCZOS)
        if preview_format in {"jpg", "jpeg"}:
            preview_img.save(preview_path, format="JPEG", quality=92, optimize=True)
        else:
            preview_img.save(preview_path, format=preview_format.upper())

        print_img = master_img.resize((tw, th), Image.LANCZOS)
        if print_format in {"jpg", "jpeg"}:
            print_img.save(print_path, format="JPEG", quality=98, optimize=True)
        else:
            print_img.save(print_path, format=print_format.upper())
    except Exception:
        preview_path.write_bytes(master_path.read_bytes())
        print_path.write_bytes(master_path.read_bytes())

    preview_rel = relative_public_url(preview_path)
    master_rel = relative_public_url(master_path)
    print_rel = relative_public_url(print_path)
    return {
        "preview_url": absolute_public_url(preview_rel),
        "master_url": absolute_public_url(master_rel),
        "print_url": absolute_public_url(print_rel),
        "preview_path": preview_rel,
        "master_path": master_rel,
        "print_path": print_rel,
    }


def generate_image_asset_sync(prompt: str, title: str, policy: Dict[str, Any], reference_images: Optional[List[str]] = None) -> Dict[str, str]:
    api = get_openai_client()
    if api is None:
        raise HTTPException(status_code=500, detail="OpenAI not configured for image generation")
    master_size = normalize_visual_size(str(policy.get("master_size") or VISUAL_MASTER_SIZE), VISUAL_MASTER_SIZE)
    try:
        result = api.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            size=master_size,
            quality=policy.get("quality") or IMAGE_QUALITY or "high",
        )
        b64 = result.data[0].b64_json
        if not b64:
            raise ValueError("Image API returned no b64_json")
        image_bytes = base64.b64decode(b64)
        return save_binary_image_versions(image_bytes, title=title, policy=policy, folder_name="visuals")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {exc}")


def build_concept_visual_prompts(project: Dict[str, Any], concept: Dict[str, Any], count: int = 3) -> List[str]:
    count = max(1, min(count, 6))
    base_prompt = f"""
Create {count} distinct premium event moodboard prompts as JSON.
Return a JSON array of strings only.
Every prompt must target a printable 16:9 concept visual for an event presentation.
Project name: {project.get('name') or 'Untitled Project'}
Event type: {project.get('event_type') or 'Event'}
Selected concept: {dump_json(concept)}
Brief: {project.get('brief') or ''}
Analysis: {project.get('analysis') or ''}
""".strip()
    try:
        data = llm_json(
            "You are a world-class experiential design visual prompt strategist. Return JSON only.",
            base_prompt,
        )
        if isinstance(data, list):
            prompts = [str(x).strip() for x in data if str(x).strip()]
            if prompts:
                return prompts[:count]
    except Exception:
        pass
    summary = concept.get("summary") or project.get("brief") or "premium event concept"
    style = concept.get("style") or project.get("style_direction") or "cinematic premium"
    colors = ", ".join(concept.get("colors") or []) or "black, silver, warm white"
    materials = ", ".join(concept.get("materials") or []) or "premium scenic materials"
    return [
        f"Premium 16:9 printable moodboard hero visual for {project.get('name') or 'event'}, inspired by {concept.get('name') or 'selected concept'}, {summary}, style {style}, materials {materials}, colors {colors}, realistic event scenography, rich lighting, high-detail presentation frame, no watermark, no text.",
        f"Premium 16:9 printable wide-angle event concept visual showing full venue ambience for {project.get('name') or 'event'}, concept {concept.get('name') or 'selected concept'}, cinematic production design, stage, scenic, screen, audience mood, realistic materials, high detail, no text.",
        f"Premium 16:9 printable close-up concept frame showing materiality, lighting mood, scenic edges, furniture, and brand detailing for {project.get('name') or 'event'}, concept {concept.get('name') or 'selected concept'}, photorealistic presentation quality, no text.",
    ][:count]


def sync_create_visual_asset(project: Dict[str, Any], user_id: str, asset_type: str, title: str, prompt: str, section: Optional[str] = None, job_kind: Optional[str] = None, source_file_url: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    policy = ensure_visual_policy(str(project["id"]), user_id)
    asset = create_project_asset(
        str(project["id"]),
        user_id,
        asset_type=asset_type,
        title=title,
        prompt=prompt,
        section=section,
        job_kind=job_kind,
        source_file_url=source_file_url,
        status="running",
        meta={**(meta or {}), "visual_policy": policy},
    )
    generated = generate_image_asset_sync(prompt=prompt, title=title, policy=policy)
    asset = update_project_asset(
        asset["id"],
        user_id,
        {
            "status": "completed",
            "preview_url": generated["preview_url"],
            "master_url": generated["master_url"],
            "print_url": generated["print_url"],
            "meta": {**(meta or {}), "visual_policy": policy, "storage": generated},
        },
    )
    add_project_activity(str(project["id"]), user_id, "asset.completed", title, detail=f"{asset_type} generated", meta={"asset_id": asset["id"], "section": section})
    return asset


def queue_agent_job_with_activity(project_id: str, user_id: str, agent_type: str, job_type: str, title: str, priority: int = 5, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    job = create_agent_job(project_id, user_id, agent_type=agent_type, job_type=job_type, title=title, priority=priority, input_data=input_data or {})
    add_project_activity(project_id, user_id, "job.queued", title, detail=f"{agent_type} queued", meta={"job_id": job["id"], "job_type": job_type})
    return job


def create_master_event_manual(project: Dict[str, Any]) -> Dict[str, str]:
    sections = [
        {"heading": "Project Overview", "body": f"Project: {project.get('name') or 'Untitled Project'}\nEvent Type: {project.get('event_type') or 'Event'}\nStatus: {project.get('status') or 'draft'}"},
        {"heading": "Brief", "body": project.get("brief") or "No brief available."},
        {"heading": "Analysis", "body": project.get("analysis") or "No analysis available."},
    ]
    selected = load_json(project.get("selected")) or {}
    if selected:
        sections.append({"heading": "Selected Concept", "body": dump_json(selected)})
    for heading, key in [
        ("Sound Department", "sound_data"),
        ("Lighting Department", "lighting_data"),
        ("Show Runner", "showrunner_data"),
    ]:
        data = load_json(project.get(key))
        if not data:
            continue
        pdf_sections = data.get("pdf_sections") if isinstance(data, dict) else None
        if pdf_sections:
            sections.extend(_normalize_pdf_sections(pdf_sections, heading))
        else:
            sections.append({"heading": heading, "body": dump_json(data)})
    assets = list_project_assets(str(project["id"]), str(project["user_id"]))
    if assets:
        body_lines = []
        for a in assets[:100]:
            body_lines.append(f"{a.get('asset_type')}: {a.get('title')}\nPreview: {a.get('preview_url') or '-'}\nMaster: {a.get('master_url') or '-'}\nPrint: {a.get('print_url') or '-'}")
        sections.append({"heading": "Generated Assets", "body": "\n\n".join(body_lines)})
    return create_simple_pdf(f"Master Event Manual - {project.get('name') or 'Project'}", sections, "master_event_manual")


def update_project_media_rollups(project_id: str, user_id: str) -> Dict[str, Any]:
    assets = list_project_assets(project_id, user_id)
    images = [
        {
            "id": a["id"],
            "title": a.get("title"),
            "preview_url": a.get("preview_url"),
            "master_url": a.get("master_url"),
            "print_url": a.get("print_url"),
            "status": a.get("status"),
            "asset_type": a.get("asset_type"),
            "section": a.get("section"),
        }
        for a in assets if a.get("asset_type") in {"moodboard", "image", "2d_graphic", "reference"}
    ]
    render3d = [
        {
            "id": a["id"],
            "title": a.get("title"),
            "preview_url": a.get("preview_url"),
            "master_url": a.get("master_url"),
            "print_url": a.get("print_url"),
            "status": a.get("status"),
            "asset_type": a.get("asset_type"),
            "section": a.get("section"),
        }
        for a in assets if a.get("asset_type") in {"3d_render", "scene_preview"}
    ]
    return update_project_fields(project_id, user_id, {"images": images, "render3d": render3d})


@app.get("/projects/{project_id}/visual-policy")
def get_visual_policy_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    policy = ensure_visual_policy(project_id, str(current_user["id"]))
    return {"project_id": project_id, "visual_policy": policy}


@app.post("/projects/{project_id}/visual-policy")
def set_visual_policy_endpoint(project_id: str, payload: VisualPolicyInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    policy = merged_visual_policy(project)
    if payload.preview_size:
        policy["preview_size"] = normalize_visual_size(payload.preview_size, policy["preview_size"])
    if payload.master_size:
        policy["master_size"] = normalize_visual_size(payload.master_size, policy["master_size"])
    if payload.print_size:
        pw, ph = parse_size(payload.print_size, parse_size(policy.get("print_size") or "5760x3240", (5760, 3240)))
        if abs((pw / max(ph, 1)) - (16 / 9)) > 0.02:
            pw = int(round(ph * 16 / 9))
        policy["print_size"] = f"{pw}x{ph}"
    if payload.aspect_ratio:
        policy["aspect_ratio"] = payload.aspect_ratio
    project = update_project_fields(project_id, str(current_user["id"]), {"visual_policy": policy})
    add_project_activity(project_id, str(current_user["id"]), "visual_policy.updated", "Visual policy updated", meta={"visual_policy": policy})
    return {"message": "Visual policy updated", "project": project, "visual_policy": policy}


@app.get("/projects/{project_id}/assets")
def list_assets_endpoint(project_id: str, section: Optional[str] = Query(default=None), current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"assets": list_project_assets(project_id, str(current_user["id"]), section=section)}


@app.get("/projects/{project_id}/jobs")
def list_jobs_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"jobs": list_agent_jobs(project_id, str(current_user["id"]))}


@app.get("/projects/{project_id}/activity")
def list_activity_endpoint(project_id: str, limit: int = Query(default=100, ge=1, le=500), current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"activity": list_project_activity(project_id, str(current_user["id"]), limit=limit)}


@app.post("/projects/{project_id}/assets/upload-reference")
async def upload_reference_asset(
    project_id: str,
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    section: Optional[str] = Form(default="references"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    suffix = Path(file.filename or "reference.bin").suffix or ".bin"
    saved_path = UPLOAD_DIR / f"reference_{uuid.uuid4().hex}{suffix}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    saved_path.write_bytes(content)
    rel = relative_public_url(saved_path)
    asset = create_project_asset(
        project_id,
        str(current_user["id"]),
        asset_type="reference",
        title=title or (file.filename or "Reference Asset"),
        prompt="",
        section=section,
        job_kind="upload",
        status="completed",
        preview_url=absolute_public_url(rel),
        master_url=absolute_public_url(rel),
        print_url=absolute_public_url(rel),
        source_file_url=absolute_public_url(rel),
        meta={"filename": file.filename, "content_type": file.content_type, "size_bytes": len(content)},
    )
    update_project_media_rollups(project_id, str(current_user["id"]))
    add_project_activity(project_id, str(current_user["id"]), "asset.uploaded", asset["title"], detail="Reference asset uploaded", meta={"asset_id": asset["id"]})
    return {"message": "Reference uploaded", "asset": asset}


@app.post("/projects/{project_id}/assets/generate")
def generate_asset_endpoint(project_id: str, payload: AssetCreateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if payload.generate_now and payload.asset_type in {"moodboard", "image", "2d_graphic", "scene_preview", "3d_render"}:
        asset = sync_create_visual_asset(project, user_id, payload.asset_type, payload.title, payload.prompt, section=payload.section, job_kind=payload.job_kind)
        update_project_media_rollups(project_id, user_id)
        return {"message": "Asset generated", "asset": asset}
    asset = create_project_asset(project_id, user_id, payload.asset_type, payload.title, payload.prompt, payload.section, payload.job_kind, status="queued", meta={"queued_only": True})
    job = queue_agent_job_with_activity(project_id, user_id, agent_type=payload.asset_type, job_type=payload.job_kind or "asset_generation", title=payload.title, input_data={"asset_id": asset["id"], "prompt": payload.prompt})
    return {"message": "Asset queued", "asset": asset, "job": job}


@app.post("/projects/{project_id}/moodboards/generate")
def generate_moodboards_endpoint(project_id: str, payload: MoodboardGenerateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    concepts = load_json(project.get("concepts")) or []
    if not concepts:
        raise HTTPException(status_code=400, detail="Run pipeline first to generate concepts")
    idx = payload.concept_index if payload.concept_index is not None else 0
    if idx < 0 or idx >= len(concepts):
        raise HTTPException(status_code=400, detail="Invalid concept index")
    concept = concepts[idx]
    prompts = build_concept_visual_prompts(project, concept, count=payload.count)
    assets = []
    queued_jobs = []
    for i, prompt in enumerate(prompts, start=1):
        title = f"{concept.get('name') or 'Concept'} Moodboard {i}"
        if payload.generate_now:
            assets.append(sync_create_visual_asset(project, user_id, "moodboard", title, prompt, section="moodboards", job_kind="concept_moodboard", meta={"concept_index": idx}))
        else:
            asset = create_project_asset(project_id, user_id, "moodboard", title, prompt, section="moodboards", job_kind="concept_moodboard", status="queued", meta={"concept_index": idx})
            assets.append(asset)
            queued_jobs.append(queue_agent_job_with_activity(project_id, user_id, "moodboard_agent", "concept_moodboard", title, input_data={"asset_id": asset["id"], "prompt": prompt, "concept_index": idx}))
    update_project_media_rollups(project_id, user_id)
    update_project_fields(project_id, user_id, {"moodboard": assets[0].get("preview_url") if assets else project.get("moodboard")})
    return {"message": "Moodboards processed", "assets": assets, "jobs": queued_jobs}


@app.post("/projects/{project_id}/orchestrate")
def orchestrate_project_endpoint(project_id: str, payload: OrchestrateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    ensure_visual_policy(project_id, user_id)
    orchestration_summary = {
        "queued_at": now_iso(),
        "auto_generate_moodboard": payload.auto_generate_moodboard,
        "queue_3d": payload.queue_3d,
        "queue_video": payload.queue_video,
        "queue_cad": payload.queue_cad,
        "queue_manuals": payload.queue_manuals,
    }
    jobs = []
    if payload.auto_generate_moodboard:
        try:
            moodboard_result = generate_moodboards_endpoint(project_id, MoodboardGenerateInput(generate_now=True, count=3), current_user)
            orchestration_summary["moodboards"] = [a.get("id") for a in moodboard_result.get("assets", [])]
        except Exception as exc:
            orchestration_summary["moodboards_error"] = str(exc)
    if payload.queue_3d:
        jobs.append(queue_agent_job_with_activity(project_id, user_id, "scene_agent", "scene_json", "Generate 3D scene JSON", input_data={"project_id": project_id}))
        jobs.append(queue_agent_job_with_activity(project_id, user_id, "render_agent", "blender_render", "Generate multi-angle 3D renders", input_data={"project_id": project_id, "target_worker": BLENDER_QUEUE_URL or None, "required_views": ["hero_front", "hero_three_quarter", "top_angle", "wide_venue", "closeup_detail"]}))
    if payload.queue_video:
        jobs.append(queue_agent_job_with_activity(project_id, user_id, "video_agent", "screen_movie", "Generate screen movie and sound bed", input_data={"project_id": project_id, "target_worker": VIDEO_QUEUE_URL or None}))
    if payload.queue_cad:
        jobs.append(queue_agent_job_with_activity(project_id, user_id, "cad_agent", "layout_trace", "Trace layout and create CAD package", input_data={"project_id": project_id, "target_worker": CAD_QUEUE_URL or None}))
    if payload.queue_manuals:
        jobs.append(queue_agent_job_with_activity(project_id, user_id, "manual_agent", "master_manual", "Generate master event manual", input_data={"project_id": project_id}))
    project = update_project_fields(project_id, user_id, {"orchestration_data": orchestration_summary})
    add_project_activity(project_id, user_id, "orchestration.updated", "Project orchestration updated", meta=orchestration_summary)
    return {"message": "Project orchestration upgraded", "project": project, "jobs": jobs, "orchestration": orchestration_summary}


@app.post("/projects/{project_id}/jobs")
def create_job_endpoint(project_id: str, payload: JobQueueInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    job = queue_agent_job_with_activity(project_id, user_id, payload.agent_type, payload.job_type, payload.title or f"{payload.agent_type} - {payload.job_type}", priority=payload.priority, input_data=payload.input_data or {})
    return {"message": "Job queued", "job": job}


@app.post("/projects/{project_id}/manuals/master/pdf")
def export_master_manual_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    pdf = create_master_event_manual(project)
    add_project_activity(project_id, str(current_user["id"]), "manual.generated", "Master event manual generated", meta=pdf)
    return {"project_id": project_id, **pdf}


@app.post("/project/{project_id}/show-console/jump")
def show_console_jump(project_id: str, payload: CueJumpInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    showrunner = project.get("showrunner_data") or {}
    cues = showrunner.get("console_cues") or []
    if not cues:
        raise HTTPException(status_code=400, detail="No console cues found")
    idx = payload.cue_index
    if payload.cue_no is not None:
        matches = [i for i, cue in enumerate(cues) if str(cue.get("cue_no")) == str(payload.cue_no)]
        if not matches:
            raise HTTPException(status_code=404, detail="Cue number not found")
        idx = matches[0]
    if idx is None:
        raise HTTPException(status_code=422, detail="cue_index or cue_no is required")
    if idx < 0 or idx >= len(cues):
        raise HTTPException(status_code=400, detail="Cue index out of range")
    state = get_console_state(project)
    state["console_index"] = idx
    state = log_console_event(state, {"status": "jump", "message": "Jumped to cue", "cue_index": idx, "cue": cues[idx]})
    project = save_console_state(project_id, user_id, state)
    return {"message": "Jumped to cue", "cue_index": idx, "cue": cues[idx], "department_outputs": project.get("department_outputs")}


@app.get("/project/{project_id}/show-console/history")
def show_console_history(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    state = get_console_state(project)
    return {"execution_log": state.get("execution_log") or []}



def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _power_row(
    element_type: str,
    name: str,
    qty: float,
    unit: str,
    width_mm: float = 0.0,
    height_mm: float = 0.0,
    depth_mm: float = 0.0,
    watts_each: float = 0.0,
    notes: str = "",
) -> Dict[str, Any]:
    total_watts = round(qty * watts_each, 2)
    total_kw = round(total_watts / 1000.0, 3)
    amps_230v = round(total_watts / 230.0, 2) if total_watts > 0 else 0.0
    return {
        "element_type": element_type,
        "name": name,
        "qty": qty,
        "unit": unit,
        "width_mm": round(width_mm, 2),
        "height_mm": round(height_mm, 2),
        "depth_mm": round(depth_mm, 2),
        "watts_each": round(watts_each, 2),
        "total_watts": total_watts,
        "total_kw": total_kw,
        "amps_230v": amps_230v,
        "wiring_points": max(1, int(qty)) if total_watts > 0 else 0,
        "notes": notes,
    }


def generate_element_sheet(
    project: Dict[str, Any],
    include_sound: bool = True,
    include_lighting: bool = True,
    include_scenic: bool = True,
    include_power_summary: bool = True,
) -> Dict[str, Any]:
    scene = load_json(project.get("scene_json")) or {}
    lighting = load_json(project.get("lighting_data")) or {}
    sound = load_json(project.get("sound_data")) or {}
    selected = load_json(project.get("selected")) or {}

    rows: List[Dict[str, Any]] = []

    if include_scenic:
        stage = scene.get("stage") or {}
        rows.append(
            _power_row(
                element_type="scenic",
                name="Main Stage Deck",
                qty=1,
                unit="set",
                width_mm=_to_float(stage.get("width"), 18000),
                depth_mm=_to_float(stage.get("depth"), 9000),
                height_mm=_to_float(stage.get("height"), 1200),
                watts_each=0,
                notes="Primary performance stage from scene_json.",
            )
        )

        for screen in scene.get("screens") or []:
            width = _to_float(screen.get("width"), 0)
            height = _to_float(screen.get("height"), 0)
            sqm = (width / 1000.0) * (height / 1000.0)
            watts_each = round(sqm * 650.0, 2)
            rows.append(
                _power_row(
                    element_type="led",
                    name=str(screen.get("name") or "LED Screen"),
                    qty=1,
                    unit="pc",
                    width_mm=width,
                    height_mm=height,
                    depth_mm=_to_float(screen.get("depth"), 0),
                    watts_each=watts_each,
                    notes=f"Estimated from {round(sqm, 2)} sqm LED area at 650 W/sqm average load.",
                )
            )

        for scenic in scene.get("scenic_elements") or []:
            rows.append(
                _power_row(
                    element_type="scenic",
                    name=str(scenic.get("name") or scenic.get("type") or "Scenic Element"),
                    qty=1,
                    unit="pc",
                    width_mm=_to_float(scenic.get("width"), 0),
                    height_mm=_to_float(scenic.get("height"), 0),
                    depth_mm=_to_float(scenic.get("depth"), 0),
                    watts_each=_to_float(scenic.get("watts"), 0),
                    notes="Scenic element from scene_json.",
                )
            )

    if include_lighting:
        fixture_list = lighting.get("fixture_list") or []
        default_watt_map = {
            "moving heads (spot/profile)": 550,
            "wash fixtures": 350,
            "led battens / linears": 120,
            "audience blinders": 650,
            "pinspots / specials": 75,
        }
        for item in fixture_list:
            label = str(item).strip()
            key = label.lower()
            qty = 4 if "moving" in key else 8 if "wash" in key else 12 if "battens" in key or "linears" in key else 6 if "blinder" in key else 8
            watts = default_watt_map.get(key, 150)
            rows.append(
                _power_row(
                    element_type="lighting",
                    name=label,
                    qty=qty,
                    unit="pc",
                    watts_each=watts,
                    notes="Estimated quantity/load for concept-stage planning.",
                )
            )

    if include_sound:
        inputs = sound.get("input_list") or []
        rows.append(
            _power_row(
                element_type="audio",
                name="FOH Console",
                qty=1,
                unit="pc",
                watts_each=350,
                notes=str((sound.get("system_design") or {}).get("console") or "FOH mixing console"),
            )
        )
        rows.append(
            _power_row(
                element_type="audio",
                name="Playback Rack / Show Laptop",
                qty=1,
                unit="pc",
                watts_each=250,
                notes="Primary playback and show control audio.",
            )
        )
        if inputs:
            rows.append(
                _power_row(
                    element_type="audio",
                    name="Wireless Mic / RF Package",
                    qty=max(2, len(inputs)),
                    unit="ch",
                    watts_each=25,
                    notes="Estimated power for receivers, antenna distro, and charging.",
                )
            )

    total_watts = round(sum(_to_float(r.get("total_watts")) for r in rows), 2)
    total_kw = round(total_watts / 1000.0, 3)
    amps_230v = round(total_watts / 230.0, 2) if total_watts > 0 else 0.0
    recommended_with_headroom_kw = round(total_kw * 1.25, 3)

    summary = {
        "project_id": str(project.get("id")),
        "project_name": project.get("name") or "Untitled Project",
        "concept_name": selected.get("name"),
        "generated_at": now_iso(),
        "rows": rows,
        "totals": {
            "element_count": len(rows),
            "total_watts": total_watts,
            "total_kw": total_kw,
            "amps_230v": amps_230v,
            "recommended_with_25pct_headroom_kw": recommended_with_headroom_kw,
        },
        "notes": [
            "All power figures are planning estimates and must be validated by production vendors.",
            "LED screen load is estimated from visible area and average LED wall consumption.",
            "Fixture quantities for lighting are preliminary planning allowances unless exact counts exist in design data.",
        ],
    }

    if include_power_summary:
        summary["power_summary"] = {
            "connected_load_kw": total_kw,
            "recommended_generator_or_mains_kw": recommended_with_headroom_kw,
            "single_phase_230v_amps": amps_230v,
            "recommended_wiring_points": sum(int(_to_float(r.get("wiring_points"))) for r in rows),
        }

    return summary


def export_element_sheet_xlsx(element_sheet: Dict[str, Any], filename_prefix: str = "element_sheet") -> Dict[str, str]:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"XLSX dependency missing: {exc}")

    out_dir = MEDIA_DIR / "spreadsheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename_prefix}_{uuid.uuid4().hex}.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.title = "Element Sheet"

    headers = [
        "element_type", "name", "qty", "unit", "width_mm", "height_mm", "depth_mm",
        "watts_each", "total_watts", "total_kw", "amps_230v", "wiring_points", "notes",
    ]
    ws.append(headers)

    header_fill = PatternFill(fill_type="solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row in element_sheet.get("rows") or []:
        ws.append([row.get(h) for h in headers])

    widths = {
        "A": 16, "B": 34, "C": 8, "D": 10, "E": 12, "F": 12, "G": 12,
        "H": 12, "I": 14, "J": 12, "K": 12, "L": 14, "M": 48,
    }
    for col, width in widths.items():
        ws.column_dimensions[col].width = width

    totals_ws = wb.create_sheet("Totals")
    totals_ws.append(["metric", "value"])
    for cell in totals_ws[1]:
        cell.fill = header_fill
        cell.font = header_font

    totals = element_sheet.get("totals") or {}
    power_summary = element_sheet.get("power_summary") or {}
    for key, value in totals.items():
        totals_ws.append([key, value])

    if power_summary:
        totals_ws.append([])
        totals_ws.append(["power_metric", "value"])
        for cell in totals_ws[totals_ws.max_row]:
            cell.fill = header_fill
            cell.font = header_font
        for key, value in power_summary.items():
            totals_ws.append([key, value])

    meta_ws = wb.create_sheet("Project Meta")
    meta_ws.append(["field", "value"])
    for cell in meta_ws[1]:
        cell.fill = header_fill
        cell.font = header_font
    meta_ws.append(["project_id", element_sheet.get("project_id")])
    meta_ws.append(["project_name", element_sheet.get("project_name")])
    meta_ws.append(["concept_name", element_sheet.get("concept_name")])
    meta_ws.append(["generated_at", element_sheet.get("generated_at")])

    wb.save(out_path)
    rel = relative_public_url(out_path)
    return {"xlsx_path": rel, "xlsx_url": absolute_public_url(rel)}


def build_walkthrough_timeline(project: Dict[str, Any], scene: Dict[str, Any], showrunner: Dict[str, Any], include_camera_pan: bool = True) -> List[Dict[str, Any]]:
    cameras = scene.get("cameras") or []
    cues = showrunner.get("console_cues") or []
    timeline: List[Dict[str, Any]] = []

    base_views = cameras[:] or [
        {"view": "hero", "label": "Front Hero View"},
        {"view": "wide", "label": "Wide Venue View"},
        {"view": "top", "label": "Top View"},
    ]

    cue_durations = [12, 14, 16, 18, 14, 12]
    t = 0
    for idx, cue in enumerate(cues or [{"cue_no": 1, "name": "Overview"}]):
        cam = base_views[idx % len(base_views)]
        duration = cue_durations[idx % len(cue_durations)]
        segment = {
            "segment_no": idx + 1,
            "cue_no": cue.get("cue_no", idx + 1),
            "cue_name": cue.get("name", f"Cue {idx + 1}"),
            "start_sec": t,
            "duration_sec": duration,
            "camera": {
                "view": cam.get("view", "hero"),
                "label": cam.get("label", "Camera"),
                "motion": "pan_orbit" if include_camera_pan else "static",
            },
            "audio_mode": "demo_mix",
            "video_mode": "screen_preview",
            "notes": cue.get("go") or cue.get("standby") or "",
        }
        if include_camera_pan:
            segment["camera"]["path"] = [
                {"x": -6.0, "y": 1.8, "z": 18.0},
                {"x": 0.0, "y": 2.2, "z": 14.0},
                {"x": 6.0, "y": 1.8, "z": 18.0},
            ]
        timeline.append(segment)
        t += duration

    return timeline


def build_show_trial_package(
    project: Dict[str, Any],
    include_walkthrough: bool = True,
    include_audio_video: bool = True,
    include_camera_pan: bool = True,
) -> Dict[str, Any]:
    scene = load_json(project.get("scene_json")) or build_scene_3d_json(project)
    sound = load_json(project.get("sound_data")) or _default_sound_plan(project)
    lighting = load_json(project.get("lighting_data")) or _default_lighting_plan(project)
    showrunner = load_json(project.get("showrunner_data")) or _default_showrunner_plan(project)
    selected = load_json(project.get("selected")) or {}

    cues = showrunner.get("console_cues") or _default_showrunner_plan(project)["console_cues"]
    timeline = build_walkthrough_timeline(project, scene, showrunner, include_camera_pan=include_camera_pan) if include_walkthrough else []

    cue_sheet = []
    elapsed = 0
    for idx, cue in enumerate(cues, start=1):
        duration = 15 if idx == 1 else 18
        cue_sheet.append({
            "cue_no": cue.get("cue_no", idx),
            "name": cue.get("name", f"Cue {idx}"),
            "standby": cue.get("standby", ""),
            "go": cue.get("go", ""),
            "cue_type": cue.get("cue_type", "show"),
            "actions": cue.get("actions", []),
            "est_start_sec": elapsed,
            "est_duration_sec": duration,
        })
        elapsed += duration

    package = {
        "project_id": str(project.get("id")),
        "project_name": project.get("name") or "Untitled Project",
        "concept_name": selected.get("name"),
        "generated_at": now_iso(),
        "mode": "trial",
        "walkthrough_enabled": include_walkthrough,
        "audio_video_enabled": include_audio_video,
        "camera_pan_enabled": include_camera_pan,
        "scene_overview": {
            "venue_type": scene.get("venue_type"),
            "stage": scene.get("stage"),
            "screen_count": len(scene.get("screens") or []),
            "camera_count": len(scene.get("cameras") or []),
        },
        "walkthrough_timeline": timeline,
        "cue_sheet": cue_sheet,
        "trial_script": [
            f"Welcome to the pre-show trial for {project.get('name') or 'your event'}.",
            "This simulation walks the client through venue mood, screen content, cue rhythm, and reveal timing.",
            "Use edit mode to change cue text, pacing, actions, or camera timing before final show lock.",
        ],
        "audio_preview": {
            "enabled": include_audio_video,
            "playback_cues": sound.get("playback_cues") or [],
        },
        "video_preview": {
            "enabled": include_audio_video,
            "screens": scene.get("screens") or [],
        },
        "lighting_preview": {
            "scene_cues": lighting.get("scene_cues") or [],
        },
        "edit_options": [
            "Edit cue copy",
            "Edit timing",
            "Edit actions",
            "Edit camera path",
            "Finalize for live show run",
        ],
        "final_show_ready": False,
    }
    return package


@app.post("/projects/{project_id}/element-sheet/generate")
def generate_element_sheet_endpoint(
    project_id: str,
    payload: ElementSheetGenerateInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    element_sheet = generate_element_sheet(
        project,
        include_sound=payload.include_sound,
        include_lighting=payload.include_lighting,
        include_scenic=payload.include_scenic,
        include_power_summary=payload.include_power_summary,
    )

    xlsx = {}
    if payload.include_xlsx:
        xlsx = export_element_sheet_xlsx(
            element_sheet,
            filename_prefix=safe_filename(payload.sheet_title or f"{project.get('name') or 'project'}_element_sheet"),
        )
        element_sheet["xlsx_url"] = xlsx["xlsx_url"]
        element_sheet["xlsx_path"] = xlsx["xlsx_path"]

    updated_project = update_project_fields(project_id, user_id, {"element_sheet": element_sheet})
    add_project_activity(
        project_id,
        user_id,
        "element_sheet.generated",
        payload.sheet_title or "Element sheet generated",
        detail="Element sizing, wiring points, and power load summary generated.",
        meta={"xlsx_url": xlsx.get("xlsx_url"), "row_count": len(element_sheet.get("rows") or [])},
    )

    return {
        "message": "Element sheet generated",
        "project_id": project_id,
        "element_sheet": element_sheet,
        "project": updated_project,
        **xlsx,
    }


@app.get("/projects/{project_id}/element-sheet")
def get_element_sheet_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project_id": project_id, "element_sheet": load_json(project.get("element_sheet"))}


@app.post("/projects/{project_id}/show-trial/generate")
def generate_show_trial_endpoint(
    project_id: str,
    payload: ShowTrialGenerateInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.get("scene_json"):
        project = update_project_fields(project_id, user_id, {"scene_json": build_scene_3d_json(project)})
    if not project.get("showrunner_data"):
        project = update_project_fields(project_id, user_id, {"showrunner_data": generate_showrunner_department(project)})
    if not project.get("sound_data"):
        project = update_project_fields(project_id, user_id, {"sound_data": generate_sound_department(project)})
    if not project.get("lighting_data"):
        project = update_project_fields(project_id, user_id, {"lighting_data": generate_lighting_department(project)})

    project = get_project_by_id(project_id, user_id=user_id) or project
    trial = build_show_trial_package(
        project,
        include_walkthrough=payload.include_walkthrough,
        include_audio_video=payload.include_audio_video,
        include_camera_pan=payload.include_camera_pan,
    )
    trial["draft_name"] = payload.draft_name or f"{project.get('name') or 'Project'} Trial"

    orchestration = load_json(project.get("orchestration_data")) or {}
    orchestration["show_trial"] = trial
    project = update_project_fields(project_id, user_id, {"orchestration_data": orchestration})

    queued_jobs = []
    if payload.queue_render_jobs:
        walkthrough_asset = create_project_asset(
            project_id, user_id, "walkthrough_preview", trial["draft_name"] + " Walkthrough",
            prompt="3D walkthrough preview with camera pan, AV preview and cue-driven simulation",
            section="show_trial", job_kind="walkthrough_trial", status="queued", meta={"trial": True},
        )
        av_asset = create_project_asset(
            project_id, user_id, "show_trial_preview", trial["draft_name"] + " Live Trial",
            prompt="Live show trial preview with cue sheet, show running script, AV sync and console timing",
            section="show_trial", job_kind="live_show_trial", status="queued", meta={"trial": True},
        )
        queued_jobs.append(queue_agent_job_with_activity(project_id, user_id, "walkthrough_agent", "walkthrough_trial", walkthrough_asset["title"], input_data={"asset_id": walkthrough_asset["id"], "trial": trial}))
        queued_jobs.append(queue_agent_job_with_activity(project_id, user_id, "show_trial_agent", "live_show_trial", av_asset["title"], input_data={"asset_id": av_asset["id"], "trial": trial}))

    add_project_activity(
        project_id, user_id, "show_trial.generated", trial["draft_name"],
        detail="3D walkthrough + live show trial prepared for pre-show review.",
        meta={"queued_jobs": [j.get("id") for j in queued_jobs]},
    )

    return {
        "message": "Show trial generated",
        "project_id": project_id,
        "show_trial": trial,
        "jobs": queued_jobs,
        "project": project,
    }


@app.get("/projects/{project_id}/show-trial")
def get_show_trial_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    orchestration = load_json(project.get("orchestration_data")) or {}
    return {"project_id": project_id, "show_trial": orchestration.get("show_trial")}


@app.post("/projects/{project_id}/show-trial/update")
def update_show_trial_endpoint(
    project_id: str,
    payload: ShowTrialUpdateInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    orchestration = load_json(project.get("orchestration_data")) or {}
    existing = orchestration.get("show_trial") or {}
    updated_trial = {**existing, **(payload.trial_data or {})}
    orchestration["show_trial"] = updated_trial
    project = update_project_fields(project_id, user_id, {"orchestration_data": orchestration})

    add_project_activity(project_id, user_id, "show_trial.updated", updated_trial.get("draft_name") or "Show trial updated")
    return {"message": "Show trial updated", "show_trial": updated_trial, "project": project}


@app.post("/projects/{project_id}/show-trial/finalize")
def finalize_show_trial_endpoint(
    project_id: str,
    payload: ShowTrialFinalizeInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    orchestration = load_json(project.get("orchestration_data")) or {}
    trial = orchestration.get("show_trial")
    if not trial:
        raise HTTPException(status_code=404, detail="Show trial not found")

    showrunner = load_json(project.get("showrunner_data")) or _default_showrunner_plan(project)
    if payload.use_trial_cues and trial.get("cue_sheet"):
        showrunner["console_cues"] = [
            {
                "cue_no": cue.get("cue_no", idx + 1),
                "name": cue.get("name", f"Cue {idx + 1}"),
                "cue_type": cue.get("cue_type", "show"),
                "standby": cue.get("standby", ""),
                "go": cue.get("go", ""),
                "actions": cue.get("actions", []),
            }
            for idx, cue in enumerate(trial.get("cue_sheet") or [])
        ]

    trial["final_show_ready"] = True
    trial["finalized_at"] = now_iso()
    orchestration["show_trial"] = trial

    updates = {"showrunner_data": showrunner, "orchestration_data": orchestration}
    if payload.mark_ready:
        updates["status"] = "show_ready"

    project = update_project_fields(project_id, user_id, updates)

    add_project_activity(
        project_id, user_id, "show_trial.finalized", trial.get("draft_name") or "Show trial finalized",
        detail="Trial cues promoted to final show running option.",
    )
    return {"message": "Show trial finalized", "show_trial": trial, "project": project}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)
