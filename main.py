import os
import re
import json
import uuid
import base64
import shutil
import tempfile
import datetime
import subprocess
from pathlib import Path
from typing import Optional, Any, Dict, List, Literal
from contextlib import asynccontextmanager
from urllib.request import Request, urlopen

import psycopg
from psycopg.rows import dict_row

from dotenv import load_dotenv
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator

from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    Depends,
    UploadFile,
    File,
    Request as FastAPIRequest,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import PlainTextResponse

from openai import OpenAI


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
SECRET_KEY = os.getenv("SECRET_KEY", "").strip() or "change-me-in-render"

TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4.1").strip()
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1").strip()
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-transcribe").strip()
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts").strip()
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime").strip()

BLENDER_PATH = os.getenv("BLENDER_PATH", "blender").strip()
BLENDER_SCRIPT = os.getenv("BLENDER_SCRIPT", "blender_script.py").strip()

RENDER_OUTPUT_DIR = Path(os.getenv("RENDER_OUTPUT_DIR", "/tmp/ai_creative_renders"))
MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "/tmp/ai_creative_media"))
ACCESS_TOKEN_HOURS = int(os.getenv("ACCESS_TOKEN_HOURS", "24"))
JWT_ALGORITHM = "HS256"
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()] or ["*"]

RENDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

client: Optional[OpenAI] = None
pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


def require_db_url() -> str:
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL missing")
    return DATABASE_URL


def get_conn():
    return psycopg.connect(require_db_url(), row_factory=dict_row, autocommit=True)


def with_db(fn):
    def wrapper(*args, **kwargs):
        with get_conn() as conn:
            with conn.cursor() as cur:
                return fn(cur, *args, **kwargs)
    return wrapper


def create_tables() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("create extension if not exists pgcrypto;")
            cur.execute("""
                create table if not exists public.users (
                  id uuid primary key,
                  email text unique not null,
                  password text not null,
                  full_name text,
                  role text default 'user',
                  is_active boolean default true,
                  created_at timestamptz default now()
                );
            """)
            cur.execute("""
                create table if not exists public.projects (
                  id uuid primary key,
                  user_id uuid not null references public.users(id) on delete cascade,
                  name text default 'Untitled Project',
                  event_type text,
                  status text default 'draft',
                  brief text,
                  analysis text,
                  concepts jsonb,
                  selected jsonb,
                  moodboard text,
                  images jsonb,
                  render3d jsonb,
                  cad text,
                  scene_json jsonb,
                  deliverables jsonb,
                  dimensions jsonb,
                  brand_data jsonb,
                  presentation_data jsonb,
                  sound_data jsonb,
                  lighting_data jsonb,
                  showrunner_data jsonb,
                  department_outputs jsonb,
                  created_at timestamptz default now(),
                  updated_at timestamptz default now()
                );
            """)
            cur.execute("""
                create table if not exists public.project_versions (
                  id uuid primary key,
                  project_id uuid not null references public.projects(id) on delete cascade,
                  user_id uuid not null references public.users(id) on delete cascade,
                  version_no int not null,
                  snapshot jsonb,
                  note text,
                  created_at timestamptz default now()
                );
            """)
            cur.execute("""
                create table if not exists public.project_comments (
                  id uuid primary key,
                  project_id uuid not null references public.projects(id) on delete cascade,
                  user_id uuid not null references public.users(id) on delete cascade,
                  section text,
                  comment_text text,
                  status text default 'open',
                  created_at timestamptz default now()
                );
            """)
            cur.execute("""
                create table if not exists public.render_jobs (
                  id uuid primary key,
                  project_id uuid not null references public.projects(id) on delete cascade,
                  user_id uuid not null references public.users(id) on delete cascade,
                  job_type text not null,
                  status text default 'queued',
                  input_json jsonb,
                  output_json jsonb,
                  error_text text,
                  started_at timestamptz,
                  finished_at timestamptz,
                  created_at timestamptz default now()
                );
            """)
            cur.execute("""
                create table if not exists public.project_files (
                  id uuid primary key,
                  project_id uuid not null references public.projects(id) on delete cascade,
                  user_id uuid references public.users(id) on delete set null,
                  file_type text,
                  file_name text,
                  file_url text,
                  meta jsonb,
                  created_at timestamptz default now()
                );
            """)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
    if DATABASE_URL:
        create_tables()
    yield


app = FastAPI(title="AI Creative Studio API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/renders", StaticFiles(directory=str(RENDER_OUTPUT_DIR)), name="renders")


def require_openai() -> OpenAI:
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI not configured")
    return client


def safe_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def json_dumps_safe(value: Any) -> str:
    return json.dumps(value, default=str)


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def hash_password(password: str) -> str:
    return pwd.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd.verify(password, hashed)


def create_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=ACCESS_TOKEN_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return str(user_id)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_current_user_id(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> str:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return decode_token(credentials.credentials)


def build_public_url(saved_path: Path) -> str:
    try:
        rel = saved_path.relative_to(MEDIA_DIR)
        return f"/media/{rel.as_posix()}"
    except Exception:
        rel = saved_path.relative_to(RENDER_OUTPUT_DIR)
        return f"/renders/{rel.as_posix()}"


def save_b64_to_media(b64_data: str, suffix: str = ".png", prefix: str = "img") -> str:
    folder = MEDIA_DIR / datetime.datetime.utcnow().strftime("%Y%m%d")
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"{prefix}_{uuid.uuid4().hex}{suffix}"
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    return build_public_url(file_path)


def local_or_remote_to_temp_file(url_or_path: str, suffix: str = ".png") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    if url_or_path.startswith("/media/"):
        src = MEDIA_DIR / url_or_path.replace("/media/", "", 1)
        shutil.copyfile(src, tmp_path)
        return tmp_path

    if url_or_path.startswith("/renders/"):
        src = RENDER_OUTPUT_DIR / url_or_path.replace("/renders/", "", 1)
        shutil.copyfile(src, tmp_path)
        return tmp_path

    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        req = Request(url_or_path, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as r, open(tmp_path, "wb") as f:
            f.write(r.read())
        return tmp_path

    src_path = Path(url_or_path)
    if src_path.exists():
        shutil.copyfile(src_path, tmp_path)
        return tmp_path

    raise HTTPException(status_code=400, detail="Unable to resolve image path/url")


def best_project_name_from_prompt(prompt: str) -> str:
    text = (prompt or "").strip().splitlines()
    return text[0][:80].strip() if text and text[0].strip() else "Untitled Project"


def create_simple_pdf(title: str, sections: List[Dict[str, Any]], filename_prefix: str) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    out_dir = MEDIA_DIR / "pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename_prefix}_{uuid.uuid4().hex}.pdf"

    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

    for section in sections:
        story.append(Paragraph(str(section.get("heading", "")), styles["Heading2"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(str(section.get("body", "")).replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 12))

    SimpleDocTemplate(str(out_path), pagesize=A4).build(story)
    return build_public_url(out_path)


class UserInput(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None


class RunInput(BaseModel):
    text: str = Field(min_length=3)
    project_id: Optional[str] = None
    name: Optional[str] = None
    event_type: Optional[str] = None


class SelectConceptInput(BaseModel):
    project_id: str
    index: int


class CommentInput(BaseModel):
    project_id: str
    section: str
    comment_text: str


class UpdateProjectInput(BaseModel):
    project_id: str
    field: str
    value: Any


class Generate3DInput(BaseModel):
    project_id: str
    width: int = 1920
    height: int = 1080


class DepartmentPDFRequest(BaseModel):
    title: Optional[str] = None


class ProjectCreateInput(BaseModel):
    title: Optional[str] = None
    brief: str = Field(min_length=3)
    style_direction: Optional[str] = None
    event_type: Optional[str] = None


class EmptyInput(BaseModel):
    pass


class SelectConceptCompatInput(BaseModel):
    concept_index: int


class ShowConsoleInput(BaseModel):
    command: str = "next"


PROJECT_ALLOWED_FIELDS = {
    "name", "event_type", "status", "brief", "analysis", "concepts", "selected",
    "moodboard", "images", "render3d", "cad", "scene_json", "deliverables",
    "dimensions", "brand_data", "presentation_data", "sound_data", "lighting_data",
    "showrunner_data", "department_outputs",
}


@with_db
def create_user(cur, email: str, password: str, full_name: Optional[str] = None) -> str:
    user_id = str(uuid.uuid4())
    cur.execute("""
        insert into public.users (id, email, password, full_name)
        values (%s, %s, %s, %s)
        returning id
    """, (user_id, email, hash_password(password), full_name))
    return str(cur.fetchone()["id"])


@with_db
def get_user_by_email(cur, email: str):
    cur.execute("""
        select id, email, password, full_name, role, is_active, created_at
        from public.users
        where email = %s
    """, (email,))
    row = cur.fetchone()
    return dict(row) if row else None


@with_db
def get_user_by_id(cur, user_id: str):
    cur.execute("""
        select id, email, full_name, role, is_active, created_at
        from public.users
        where id = %s
    """, (user_id,))
    row = cur.fetchone()
    return dict(row) if row else None


@with_db
def create_project(cur, user_id: str, name: Optional[str] = None, event_type: Optional[str] = None) -> str:
    project_id = str(uuid.uuid4())
    cur.execute("""
        insert into public.projects (id, user_id, name, event_type, status)
        values (%s, %s, %s, %s, %s)
        returning id
    """, (project_id, user_id, name or "Untitled Project", event_type, "draft"))
    return str(cur.fetchone()["id"])


@with_db
def get_project(cur, project_id: str) -> Optional[Dict[str, Any]]:
    cur.execute("select * from public.projects where id = %s", (project_id,))
    row = cur.fetchone()
    if not row:
        return None

    project = dict(row)
    for key in [
        "concepts", "selected", "images", "render3d", "scene_json", "deliverables",
        "dimensions", "brand_data", "presentation_data", "sound_data", "lighting_data",
        "showrunner_data", "department_outputs",
    ]:
        project[key] = safe_json(project.get(key))
    return project


@with_db
def list_projects_for_user(cur, user_id: str) -> List[Dict[str, Any]]:
    cur.execute("""
        select
            id,
            name as project_name,
            event_type,
            status,
            brief,
            analysis,
            concepts,
            selected,
            sound_data,
            lighting_data,
            showrunner_data,
            department_outputs,
            created_at,
            updated_at
        from public.projects
        where user_id = %s
        order by created_at desc
    """, (user_id,))
    rows = []
    for row in cur.fetchall():
        item = dict(row)
        for key in ["concepts", "selected", "sound_data", "lighting_data", "showrunner_data", "department_outputs"]:
            item[key] = safe_json(item.get(key))
        rows.append(item)
    return rows


@with_db
def update_project_field(cur, project_id: str, field: str, value: Any) -> None:
    if field not in PROJECT_ALLOWED_FIELDS:
        raise HTTPException(status_code=400, detail=f"Invalid project field: {field}")
    db_value = json_dumps_safe(value) if isinstance(value, (dict, list)) else value
    cur.execute(
        f"update public.projects set {field} = %s, updated_at = now() where id = %s",
        (db_value, project_id),
    )


@with_db
def snapshot_project_version(cur, project_id: str, user_id: str, note: str = "") -> None:
    project = get_project(project_id)
    if not project:
        return

    cur.execute(
        "select coalesce(max(version_no), 0) + 1 as next_version from public.project_versions where project_id = %s",
        (project_id,),
    )
    next_version = int(cur.fetchone()["next_version"])
    snapshot_payload = jsonable(project)

    cur.execute(
        """
        insert into public.project_versions (id, project_id, user_id, version_no, snapshot, note)
        values (%s, %s, %s, %s, %s, %s)
        """,
        (
            str(uuid.uuid4()),
            project_id,
            user_id,
            next_version,
            json_dumps_safe(snapshot_payload),
            note,
        ),
    )


@with_db
def add_comment(cur, project_id: str, user_id: str, section: str, comment_text: str) -> str:
    comment_id = str(uuid.uuid4())
    cur.execute("""
        insert into public.project_comments (id, project_id, user_id, section, comment_text, status)
        values (%s, %s, %s, %s, %s, %s)
        returning id
    """, (comment_id, project_id, user_id, section, comment_text, "open"))
    return str(cur.fetchone()["id"])


@with_db
def get_comments(cur, project_id: str) -> List[Dict[str, Any]]:
    cur.execute("select * from public.project_comments where project_id = %s order by created_at desc", (project_id,))
    return [dict(r) for r in cur.fetchall()]


@with_db
def create_render_job(cur, project_id: str, user_id: str, job_type: str, input_json: Dict[str, Any]) -> str:
    job_id = str(uuid.uuid4())
    cur.execute("""
        insert into public.render_jobs (id, project_id, user_id, job_type, status, input_json)
        values (%s, %s, %s, %s, %s, %s)
        returning id
    """, (job_id, project_id, user_id, job_type, "queued", json_dumps_safe(input_json)))
    return str(cur.fetchone()["id"])


@with_db
def update_render_job(cur, job_id: str, status: str, output_json: Any = None, error_text: Optional[str] = None) -> None:
    started_at = datetime.datetime.utcnow() if status == "running" else None
    finished_at = datetime.datetime.utcnow() if status in {"done", "failed"} else None
    cur.execute("""
        update public.render_jobs
        set status = %s,
            output_json = coalesce(%s, output_json),
            error_text = coalesce(%s, error_text),
            started_at = coalesce(%s, started_at),
            finished_at = coalesce(%s, finished_at)
        where id = %s
    """, (
        status,
        json_dumps_safe(output_json) if output_json is not None else None,
        error_text,
        started_at,
        finished_at,
        job_id,
    ))


@with_db
def get_render_job(cur, job_id: str) -> Optional[Dict[str, Any]]:
    cur.execute("select * from public.render_jobs where id = %s", (job_id,))
    row = cur.fetchone()
    if not row:
        return None
    result = dict(row)
    result["input_json"] = safe_json(result.get("input_json"))
    result["output_json"] = safe_json(result.get("output_json"))
    return result


def require_project_owner(project_id: str, user_id: str) -> Dict[str, Any]:
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if str(project["user_id"]) != str(user_id):
        raise HTTPException(status_code=403, detail="Not allowed")
    return project


def llm_text(system_prompt: str, user_prompt: str, temperature: float = 0.5) -> str:
    api = require_openai()
    response = api.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def llm_json(system_prompt: str, user_prompt: str) -> Any:
    raw = llm_text(system_prompt, user_prompt, temperature=0.2)
    cleaned = re.sub(r"\s*```$", "", re.sub(r"^```json\s*", "", raw.strip()))
    return json.loads(cleaned)


def analyze_brief(brief: str) -> str:
    return llm_text(
        "You are an expert AI Creative Studio planner for exhibitions, activations, launches, concerts, government events, stage productions, sound design, lighting design, and show operations.",
        f"""
Analyze this creative brief.
Return a structured practical analysis covering:
1. event objective
2. audience profile
3. event type
4. key deliverables
5. branding requirements
6. probable dimensions if missing
7. missing clarifications
8. production concerns
9. sound considerations
10. lighting considerations
11. show flow considerations

Brief:
{brief}
""",
    )


def generate_concepts(analysis: str) -> List[Dict[str, Any]]:
    result = llm_json(
        "Return only valid JSON.",
        f"""
Return exactly 3 strong concepts as a JSON array.
Each object must contain:
- name
- summary
- style
- materials
- colors
- lighting
- stage_elements
- camera_style

Analysis:
{analysis}
""",
    )
    if not isinstance(result, list):
        raise HTTPException(status_code=500, detail="Concept generation failed")
    return result


def generate_moodboard(selected_concept: Any) -> str:
    return llm_text(
        "Create polished moodboard directions for premium event, stage, exhibition, and activation concepts.",
        f"Concept:\n{json.dumps(selected_concept, indent=2, default=str)}",
    )


def generate_scene_json(selected_concept: Any, brief: str) -> Dict[str, Any]:
    result = llm_json(
        "Return only valid JSON.",
        f"""
Create structured JSON for Blender scene generation.
Brief:
{brief}
Concept:
{json.dumps(selected_concept, indent=2, default=str)}
""",
    )
    if not isinstance(result, dict):
        result = {}

    result.setdefault("units", "feet")
    result.setdefault("stage", {"width": 60, "depth": 24, "height": 4})
    result.setdefault("led_wall", {"type": "curved", "width": 40, "height": 12})
    result.setdefault("truss", {"height": 18})
    result.setdefault("audience", {"rows": 10, "cols": 20})
    result.setdefault("colors", {"primary": "#1A5DFF", "secondary": "#A855F7"})
    result.setdefault("lighting", {"style": "futuristic"})
    result.setdefault("branding", {"notes": "premium branded stage"})
    result.setdefault("camera_target", [0, 0, 6])
    result.setdefault("render", {"ratio": "16:9"})
    return result


def generate_sound_department(brief: str, project: Dict[str, Any]) -> Dict[str, Any]:
    return llm_json(
        "Return only valid JSON. You are a senior live event sound engineer.",
        f"""
Create a full sound department plan for this event.
Return JSON with:
- concept
- system_design
- speaker_plan
- input_list
- mic_plan
- patch_sheet
- playback_cues
- staffing
- rehearsal_notes
- risk_notes
- pdf_sections

Project:
{json.dumps(project, indent=2, default=str)}

Brief:
{brief}
""",
    )


def generate_lighting_department(brief: str, project: Dict[str, Any]) -> Dict[str, Any]:
    return llm_json(
        "Return only valid JSON. You are a senior live event lighting designer.",
        f"""
Create a full lighting department plan for this event.
Return JSON with:
- concept
- fixture_list
- truss_plan
- dmx_notes
- scene_cues
- looks
- operator_notes
- rehearsal_notes
- fallback_plan
- pdf_sections

Project:
{json.dumps(project, indent=2, default=str)}

Brief:
{brief}
""",
    )


def generate_showrunner_department(brief: str, project: Dict[str, Any]) -> Dict[str, Any]:
    return llm_json(
        "Return only valid JSON. You are a show runner and stage manager.",
        f"""
Create a full show running plan for this event.
Return JSON with:
- running_order
- cue_script
- standby_calls
- go_calls
- departmental_dependencies
- delay_protocol
- emergency_protocol
- rehearsal_flow
- console_cues
- pdf_sections

Project:
{json.dumps(project, indent=2, default=str)}

Brief:
{brief}
""",
    )


def run_blender_multi_angle(scene_json: Dict[str, Any], job_id: str, width: int, height: int,
                            project_id: str, user_id: str) -> None:
    try:
        update_render_job(job_id, "running")
        job_dir = RENDER_OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        scene_json_path = job_dir / "scene.json"
        with open(scene_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scene": scene_json,
                    "render": {"width": width, "height": height, "output_dir": str(job_dir)}
                },
                f,
                indent=2,
            )

        completed = subprocess.run(
            [BLENDER_PATH, "-b", "-P", BLENDER_SCRIPT, "--", str(scene_json_path)],
            capture_output=True,
            text=True,
        )

        if completed.returncode != 0:
            update_render_job(job_id, "failed", error_text=(completed.stderr or completed.stdout)[:4000])
            return

        render_map = {
            "front_wide": build_public_url(job_dir / "front_wide.png") if (job_dir / "front_wide.png").exists() else None,
            "front_center": build_public_url(job_dir / "front_center.png") if (job_dir / "front_center.png").exists() else None,
            "left_perspective": build_public_url(job_dir / "left_perspective.png") if (job_dir / "left_perspective.png").exists() else None,
            "right_perspective": build_public_url(job_dir / "right_perspective.png") if (job_dir / "right_perspective.png").exists() else None,
            "top_plan": build_public_url(job_dir / "top_plan.png") if (job_dir / "top_plan.png").exists() else None,
            "audience_view": build_public_url(job_dir / "audience_view.png") if (job_dir / "audience_view.png").exists() else None,
            "glb": build_public_url(job_dir / "scene.glb") if (job_dir / "scene.glb").exists() else None,
            "manifest": build_public_url(job_dir / "manifest.json") if (job_dir / "manifest.json").exists() else None,
        }

        update_render_job(job_id, "done", output_json=render_map)
        update_project_field(project_id, "render3d", render_map)
        snapshot_project_version(project_id, user_id, note="Multi-angle render batch completed")
    except Exception as e:
        update_render_job(job_id, "failed", error_text=str(e))


@app.get("/")
def root():
    return {
        "message": "AI Creative Studio API is running",
        "time": now_iso(),
        "api_base_ready": True,
    }


@app.get("/health")
def health():
    return {"ok": True, "time": now_iso()}


@app.post("/signup")
def signup(payload: UserInput):
    existing = get_user_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    user_id = create_user(payload.email, payload.password, payload.full_name)
    token = create_token(user_id)

    return {
        "message": "User created",
        "user_id": user_id,
        "access_token": token,
        "token": token,
        "token_type": "bearer",
    }


@app.post("/login")
def login(payload: UserInput):
    user = get_user_by_email(payload.email)
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    if not verify_password(payload.password, user["password"]):
        raise HTTPException(status_code=400, detail="Wrong password")

    token = create_token(str(user["id"]))
    return {
        "access_token": token,
        "token": token,
        "token_type": "bearer",
        "user_id": str(user["id"]),
    }


@app.get("/me")
def me(user_id: str = Depends(get_current_user_id)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.get("/projects")
def projects_list(user_id: str = Depends(get_current_user_id)):
    return {"projects": list_projects_for_user(user_id)}


@app.post("/projects")
def projects_create(payload: ProjectCreateInput, user_id: str = Depends(get_current_user_id)):
    project_id = create_project(
        user_id=user_id,
        name=payload.title or best_project_name_from_prompt(payload.brief),
        event_type=payload.event_type,
    )
    update_project_field(project_id, "brief", payload.brief)
    update_project_field(project_id, "status", "brief_received")
    snapshot_project_version(project_id, user_id, note="Project created from frontend")

    return {
        "id": project_id,
        "project_id": project_id,
        "title": payload.title or best_project_name_from_prompt(payload.brief),
        "brief": payload.brief,
        "status": "brief_received",
    }


@app.get("/projects/{project_id}")
def projects_detail(project_id: str, user_id: str = Depends(get_current_user_id)):
    project = require_project_owner(project_id, user_id)
    return {"project": project}


@app.post("/projects/{project_id}/run")
def projects_run(project_id: str, payload: EmptyInput = EmptyInput(), user_id: str = Depends(get_current_user_id)):
    project = require_project_owner(project_id, user_id)
    brief_text = project.get("brief") or ""
    if not brief_text:
        raise HTTPException(status_code=400, detail="Project brief missing")

    # Move the old staged pipeline until concepts are ready.
    for _ in range(4):
        fresh = get_project(project_id)
        if fresh and fresh.get("concepts"):
            break
        run_pipeline(
            RunInput(
                text=brief_text,
                project_id=project_id,
                name=project.get("name"),
                event_type=project.get("event_type"),
            ),
            user_id,
        )

    project = get_project(project_id)
    return {
        "project_id": project_id,
        "analysis": project.get("analysis"),
        "concepts": project.get("concepts") or [],
        "status": project.get("status"),
    }


@app.post("/projects/{project_id}/select-concept")
def projects_select_concept(project_id: str, payload: SelectConceptCompatInput, user_id: str = Depends(get_current_user_id)):
    require_project_owner(project_id, user_id)
    return select_concept(
        SelectConceptInput(project_id=project_id, index=payload.concept_index),
        user_id,
    )


@app.post("/projects/{project_id}/generate-departments")
def projects_generate_departments(project_id: str, payload: EmptyInput = EmptyInput(), user_id: str = Depends(get_current_user_id)):
    require_project_owner(project_id, user_id)
    return build_departments(project_id, user_id)


@app.post("/projects/{project_id}/show-console")
def projects_show_console(project_id: str, payload: ShowConsoleInput, user_id: str = Depends(get_current_user_id)):
    require_project_owner(project_id, user_id)
    return show_console(project_id, payload.command, user_id)


@app.get("/project/{project_id}")
def project_detail(project_id: str, user_id: str = Depends(get_current_user_id)):
    return require_project_owner(project_id, user_id)


@app.post("/project/update")
def project_update(payload: UpdateProjectInput, user_id: str = Depends(get_current_user_id)):
    require_project_owner(payload.project_id, user_id)
    update_project_field(payload.project_id, payload.field, payload.value)
    snapshot_project_version(payload.project_id, user_id, note=f"Updated field: {payload.field}")
    return {"message": "Project updated"}


@app.post("/comment")
def create_comment(payload: CommentInput, user_id: str = Depends(get_current_user_id)):
    require_project_owner(payload.project_id, user_id)
    comment_id = add_comment(payload.project_id, user_id, payload.section, payload.comment_text)
    return {"message": "Comment added", "comment_id": comment_id}


@app.get("/comments/{project_id}")
def list_comments(project_id: str, user_id: str = Depends(get_current_user_id)):
    require_project_owner(project_id, user_id)
    return {"comments": get_comments(project_id)}


@app.post("/run")
def run_pipeline(payload: RunInput, user_id: str = Depends(get_current_user_id)):
    project_id = payload.project_id or create_project(
        user_id=user_id,
        name=payload.name or best_project_name_from_prompt(payload.text),
        event_type=payload.event_type,
    )

    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if str(project["user_id"]) != str(user_id):
        raise HTTPException(status_code=403, detail="Not allowed")

    if not project.get("brief"):
        update_project_field(project_id, "brief", payload.text)
        update_project_field(project_id, "status", "brief_received")
        snapshot_project_version(project_id, user_id, note="Brief saved")
        return {"stage": "brief_saved", "project_id": project_id}

    if not project.get("analysis"):
        analysis = analyze_brief(project["brief"])
        update_project_field(project_id, "analysis", analysis)
        update_project_field(project_id, "status", "analysis_ready")
        snapshot_project_version(project_id, user_id, note="Analysis generated")
        return {"stage": "analysis_ready", "project_id": project_id, "analysis": analysis}

    if not project.get("concepts"):
        concepts = generate_concepts(project["analysis"])
        update_project_field(project_id, "concepts", concepts)
        update_project_field(project_id, "status", "concepts_ready")
        snapshot_project_version(project_id, user_id, note="Concept options generated")
        return {"stage": "concepts_ready", "project_id": project_id, "concepts": concepts}

    if not project.get("selected"):
        return {"stage": "awaiting_concept_selection", "project_id": project_id, "options": project["concepts"]}

    if not project.get("moodboard"):
        moodboard = generate_moodboard(project["selected"])
        update_project_field(project_id, "moodboard", moodboard)
        update_project_field(project_id, "status", "moodboard_ready")
        snapshot_project_version(project_id, user_id, note="Moodboard generated")
        return {"stage": "moodboard_ready", "project_id": project_id, "moodboard": moodboard}

    return {"stage": "ready_for_departments", "project_id": project_id, "message": "Concept approved. Ready for department generation."}


@app.post("/select")
def select_concept(payload: SelectConceptInput, user_id: str = Depends(get_current_user_id)):
    project = require_project_owner(payload.project_id, user_id)
    concepts = project.get("concepts")

    if not isinstance(concepts, list) or payload.index < 0 or payload.index >= len(concepts):
        raise HTTPException(status_code=400, detail="Invalid concept index")

    selected = concepts[payload.index]
    update_project_field(payload.project_id, "selected", selected)
    update_project_field(payload.project_id, "status", "concept_selected")
    snapshot_project_version(payload.project_id, user_id, note=f"Concept {payload.index} selected")
    return {"message": "Concept selected", "selected": selected}


@app.post("/generate-multi-angle")
def generate_multi_angle(payload: Generate3DInput, background_tasks: BackgroundTasks, user_id: str = Depends(get_current_user_id)):
    project = require_project_owner(payload.project_id, user_id)
    if not project.get("selected"):
        raise HTTPException(status_code=400, detail="Select concept first")

    scene_json = generate_scene_json(project["selected"], project.get("brief") or "")
    update_project_field(payload.project_id, "scene_json", scene_json)

    job_id = create_render_job(
        project_id=payload.project_id,
        user_id=user_id,
        job_type="multi_angle_render",
        input_json={"scene_json": scene_json, "width": payload.width, "height": payload.height},
    )

    background_tasks.add_task(
        run_blender_multi_angle,
        scene_json,
        job_id,
        payload.width,
        payload.height,
        payload.project_id,
        user_id,
    )

    return {"message": "Multi-angle render queued", "job_id": job_id, "project_id": payload.project_id}


@app.get("/job/{job_id}")
def job_status(job_id: str, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    job = get_render_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/project/{project_id}/departments/build")
def build_departments(project_id: str, user_id: str = Depends(get_current_user_id)):
    project = require_project_owner(project_id, user_id)
    if not project.get("selected"):
        raise HTTPException(status_code=400, detail="Select concept first")

    brief = project.get("brief") or ""
    sound = generate_sound_department(brief, project)
    lighting = generate_lighting_department(brief, project)
    showrunner = generate_showrunner_department(brief, project)

    update_project_field(project_id, "sound_data", sound)
    update_project_field(project_id, "lighting_data", lighting)
    update_project_field(project_id, "showrunner_data", showrunner)

    outputs = project.get("department_outputs") or {}
    outputs.update({
        "sound_ready": True,
        "lighting_ready": True,
        "showrunner_ready": True,
        "console_index": 0,
    })
    update_project_field(project_id, "department_outputs", outputs)
    update_project_field(project_id, "status", "departments_ready")

    snapshot_project_version(project_id, user_id, note="Built sound, lighting, and showrunner departments")
    return {
        "message": "Departments generated",
        "project_id": project_id,
        "sound_data": sound,
        "lighting_data": lighting,
        "showrunner_data": showrunner,
        "department_outputs": outputs,
    }


@app.get("/project/{project_id}/departments/manuals")
def get_department_manuals(project_id: str, user_id: str = Depends(get_current_user_id)):
    project = require_project_owner(project_id, user_id)
    return {
        "project_id": project_id,
        "manual_payload": {
            "project_name": project.get("name"),
            "event_type": project.get("event_type"),
            "brief": project.get("brief"),
            "sound_data": project.get("sound_data"),
            "lighting_data": project.get("lighting_data"),
            "showrunner_data": project.get("showrunner_data"),
        },
    }


@app.post("/project/{project_id}/departments/pdf/sound")
def export_sound_pdf(project_id: str, payload: DepartmentPDFRequest, user_id: str = Depends(get_current_user_id)):
    project = require_project_owner(project_id, user_id)
    if not project.get("sound_data"):
        raise HTTPException(status_code=404, detail="Sound data not found")
    sections = project["sound_data"].get("pdf_sections") or [{"heading": "Sound Design Manual", "body": json.dumps(project["sound_data"], indent=2, default=str)}]
    return {"project_id": project_id, "pdf_url": create_simple_pdf(payload.title or "Sound Design Manual", sections, "sound_manual")}


@app.post("/project/{project_id}/departments/pdf/lighting")
def export_lighting_pdf(project_id: str, payload: DepartmentPDFRequest, user_id: str = Depends(get_current_user_id)):
    project = require_project_owner(project_id, user_id)
    if not project.get("lighting_data"):
        raise HTTPException(status_code=404, detail="Lighting data not found")
    sections = project["lighting_data"].get("pdf_sections") or [{"heading": "Lighting Design Manual", "body": json.dumps(project["lighting_data"], indent=2, default=str)}]
    return {"project_id": project_id, "pdf_url": create_simple_pdf(payload.title or "Lighting Design Manual", sections, "lighting_manual")}


@app.post("/project/{project_id}/departments/pdf/showrunner")
def export_showrunner_pdf(project_id: str, payload: DepartmentPDFRequest, user_id: str = Depends(get_current_user_id)):
    project = require_project_owner(project_id, user_id)
    if not project.get("showrunner_data"):
        raise HTTPException(status_code=404, detail="Show runner data not found")
    sections = project["showrunner_data"].get("pdf_sections") or [{"heading": "Show Running Script", "body": json.dumps(project["showrunner_data"], indent=2, default=str)}]
    return {"project_id": project_id, "pdf_url": create_simple_pdf(payload.title or "Show Running Script", sections, "showrunner_manual")}


@app.post("/project/{project_id}/show-console")
def show_console(project_id: str, command: str, user_id: str = Depends(get_current_user_id)):
    project = require_project_owner(project_id, user_id)

    show_data = project.get("showrunner_data") or {}
    cues = show_data.get("console_cues") or []
    outputs = project.get("department_outputs") or {}
    current = int(outputs.get("console_index", 0))

    cmd = (command or "").strip().lower()

    if cmd in {"next", "next cue"}:
        if cues:
            next_index = min(current + 1, len(cues) - 1)
            outputs["console_index"] = next_index
            update_project_field(project_id, "department_outputs", outputs)
            return {
                "project_id": project_id,
                "status": "advanced",
                "cue_index": next_index,
                "cue": cues[next_index],
            }
        return {
            "project_id": project_id,
            "status": "advanced",
            "cue_index": 0,
            "cue": None,
        }

    if cmd in {"current", "status"}:
        return {
            "project_id": project_id,
            "status": "current",
            "cue_index": current,
            "cue": cues[current] if cues and current < len(cues) else None,
        }

    return {
        "project_id": project_id,
        "command": command,
        "status": "simulated",
        "console_cues": cues[:10],
    }
