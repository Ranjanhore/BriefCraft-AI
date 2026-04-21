import os
import re
import io
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

import requests
from dotenv import load_dotenv
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_validator

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
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from openai import OpenAI

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


# ============================================================
# ENV
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
SECRET_KEY = os.getenv("SECRET_KEY", "").strip()

TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-5.2").strip()
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1").strip()
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts").strip()
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-transcribe").strip()
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime").strip()

BLENDER_PATH = os.getenv("BLENDER_PATH", "blender").strip()
BLENDER_SCRIPT = os.getenv("BLENDER_SCRIPT", "blender_script.py").strip()

RENDER_OUTPUT_DIR = Path(os.getenv("RENDER_OUTPUT_DIR", "/tmp/ai_creative_renders"))
MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "/tmp/ai_creative_media"))

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_HOURS = int(os.getenv("ACCESS_TOKEN_HOURS", "24"))
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]

RENDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# APP STATE
# ============================================================

client: Optional[OpenAI] = None
db_pool: Optional[ConnectionPool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, db_pool

    print("=== STARTUP ===")
    print("OPENAI:", bool(OPENAI_API_KEY))
    print("DB:", bool(DATABASE_URL))
    print("SECRET:", bool(SECRET_KEY))

    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)

    if DATABASE_URL:
        db_pool = ConnectionPool(
            conninfo=DATABASE_URL,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row, "autocommit": True},
            open=True,
        )
        print("DB pool connected")

    yield

    if db_pool:
        db_pool.close()
        print("DB pool closed")


app = FastAPI(title="AI Creative Studio API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/renders", StaticFiles(directory=str(RENDER_OUTPUT_DIR)), name="renders")


# ============================================================
# SECURITY
# ============================================================

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return pwd.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd.verify(password, hashed)


def create_token(user_id: str) -> str:
    if not SECRET_KEY:
        raise HTTPException(status_code=500, detail="SECRET_KEY missing")
    payload = {
        "user_id": user_id,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=ACCESS_TOKEN_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> str:
    if not SECRET_KEY:
        raise HTTPException(status_code=500, detail="SECRET_KEY missing")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return str(user_id)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return decode_token(credentials.credentials)


# ============================================================
# HELPERS
# ============================================================

def require_openai() -> OpenAI:
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI not configured")
    return client


def require_db() -> ConnectionPool:
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database not connected")
    return db_pool


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


def with_db(fn):
    def wrapper(*args, **kwargs):
        pool = require_db()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                return fn(cur, *args, **kwargs)
    return wrapper


def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return name or str(uuid.uuid4())


def build_public_media_url(saved_path: Path) -> str:
    try:
        rel = saved_path.relative_to(MEDIA_DIR)
        return f"/media/{rel.as_posix()}"
    except Exception:
        rel = saved_path.relative_to(RENDER_OUTPUT_DIR)
        return f"/renders/{rel.as_posix()}"


def save_b64_to_media(b64_data: str, suffix: str = ".png", prefix: str = "img") -> str:
    folder = MEDIA_DIR / datetime.datetime.utcnow().strftime("%Y%m%d")
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"{prefix}_{uuid.uuid4().hex}{suffix}"
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    return build_public_media_url(out_path)


def image_response_to_urls(image_response: Any, prefix: str = "gen") -> List[str]:
    urls: List[str] = []
    data = getattr(image_response, "data", None) or []
    for item in data:
        item_url = getattr(item, "url", None)
        item_b64 = getattr(item, "b64_json", None)
        if item_url:
            urls.append(item_url)
        elif item_b64:
            urls.append(save_b64_to_media(item_b64, ".png", prefix))
    return urls


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

    src = Path(url_or_path)
    if src.exists():
        shutil.copyfile(src, tmp_path)
        return tmp_path

    raise HTTPException(status_code=400, detail="Unable to resolve image path/url")


def best_project_name(text: str) -> str:
    line = (text or "").strip().splitlines()[0][:80].strip()
    return line or "Untitled Project"


# ============================================================
# Pydantic Models
# ============================================================

class UserInput(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=200)
    full_name: Optional[str] = None


class RunInput(BaseModel):
    text: str = Field(min_length=3)
    project_id: Optional[str] = None
    name: Optional[str] = None
    event_type: Optional[str] = None


class SelectConceptInput(BaseModel):
    project_id: str
    index: int


class Generate3DInput(BaseModel):
    project_id: str
    width: int = 1920
    height: int = 1080


class CommentInput(BaseModel):
    project_id: str
    section: str
    comment_text: str


class UpdateProjectInput(BaseModel):
    project_id: str
    field: str
    value: Any


class GenerateImageInput(BaseModel):
    prompt: str = Field(min_length=3)
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    event_type: Optional[str] = None
    num_images: int = Field(default=4, ge=1, le=4)
    size: Literal["1024x1024", "1536x1024", "1024x1536"] = "1536x1024"
    style_strength: Literal["low", "medium", "high"] = "medium"
    reference_image_url: Optional[str] = None


class VariationInput(BaseModel):
    generation_id: str
    image_index: int = Field(ge=0)
    strength: Literal["subtle", "strong"] = "subtle"
    num_images: int = Field(default=4, ge=1, le=4)
    size: Literal["1024x1024", "1536x1024", "1024x1536"] = "1536x1024"
    remix_prompt: Optional[str] = None


class EditImageInput(BaseModel):
    generation_id: Optional[str] = None
    source_image_url: Optional[str] = None
    image_index: Optional[int] = None
    prompt: str = Field(min_length=3)
    size: Literal["1024x1024", "1536x1024", "1024x1536"] = "1536x1024"


class ProjectOnlyInput(BaseModel):
    project_id: str


class PresentationCreateInput(BaseModel):
    project_id: str
    title: Optional[str] = None
    presenter_name: str = "AI Presenter"
    presenter_style: str = "corporate"
    voice: str = "marin"


class PresentationSlideInput(BaseModel):
    presentation_id: str
    slide_no: int
    title: str
    body: str
    notes: Optional[str] = None
    asset_url: Optional[str] = None


class PresentationNarrateInput(BaseModel):
    presentation_id: str
    slide_no: int
    language: str = "en"
    audience_mode: str = "client"


class PresentationAskInput(BaseModel):
    session_id: str
    question: str
    language: str = "en"


class PresentationCommandInput(BaseModel):
    session_id: str
    command: str


class RealtimeSessionUpdateInput(BaseModel):
    session_id: str
    presentation_id: str
    slide_no: int
    audience_mode: str = "client"
    language: str = "en"


# ============================================================
# DB: USERS
# ============================================================

@with_db
def get_user_by_email(cur, email: str) -> Optional[Dict[str, Any]]:
    cur.execute(
        """
        select id, email, password, full_name, role, is_active
        from public.users
        where email = %s
        """,
        (email,),
    )
    return cur.fetchone()


@with_db
def create_user(cur, email: str, password: str, full_name: Optional[str]) -> str:
    user_id = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.users (id, email, password, full_name)
        values (%s, %s, %s, %s)
        returning id
        """,
        (user_id, email, hash_password(password), full_name),
    )
    return str(cur.fetchone()["id"])


# ============================================================
# DB: PROJECTS
# ============================================================

PROJECT_ALLOWED_FIELDS = {
    "name",
    "event_type",
    "status",
    "brief",
    "analysis",
    "concepts",
    "selected",
    "moodboard",
    "images",
    "render3d",
    "cad",
    "scene_json",
    "deliverables",
    "dimensions",
    "brand_data",
    "presentation_data",
    "brand_strategy",
    "campaign_plan",
    "ad_creatives",
    "video_scripts",
    "packaging_data",
    "retail_display_data",
    "boq_data",
    "audio_plan",
    "video_plan",
}


@with_db
def create_project(cur, user_id: str, name: str, event_type: Optional[str]) -> str:
    project_id = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.projects (id, user_id, name, event_type, status)
        values (%s, %s, %s, %s, %s)
        returning id
        """,
        (project_id, user_id, name, event_type, "draft"),
    )
    return str(cur.fetchone()["id"])


@with_db
def get_project(cur, project_id: str) -> Optional[Dict[str, Any]]:
    cur.execute("select * from public.projects where id = %s", (project_id,))
    row = cur.fetchone()
    if not row:
        return None
    item = dict(row)
    json_fields = [
        "concepts", "selected", "images", "render3d", "scene_json", "deliverables",
        "dimensions", "brand_data", "presentation_data", "brand_strategy",
        "campaign_plan", "ad_creatives", "video_scripts", "packaging_data",
        "retail_display_data", "boq_data", "audio_plan", "video_plan"
    ]
    for key in json_fields:
        item[key] = safe_json(item.get(key))
    return item


@with_db
def update_project_field(cur, project_id: str, field: str, value: Any) -> None:
    if field not in PROJECT_ALLOWED_FIELDS:
        raise HTTPException(status_code=400, detail="Invalid project field")
    db_value = json.dumps(value) if isinstance(value, (dict, list)) else value
    cur.execute(f"update public.projects set {field} = %s where id = %s", (db_value, project_id))


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
    cur.execute(
        """
        insert into public.project_versions (id, project_id, user_id, version_no, snapshot, note)
        values (%s, %s, %s, %s, %s, %s)
        """,
        (str(uuid.uuid4()), project_id, user_id, next_version, json.dumps(project), note),
    )


# ============================================================
# DB: COMMENTS
# ============================================================

@with_db
def add_comment(cur, project_id: str, user_id: str, section: str, comment_text: str) -> str:
    comment_id = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.project_comments (id, project_id, user_id, section, comment_text, status)
        values (%s, %s, %s, %s, %s, %s)
        returning id
        """,
        (comment_id, project_id, user_id, section, comment_text, "open"),
    )
    return str(cur.fetchone()["id"])


@with_db
def get_comments(cur, project_id: str) -> List[Dict[str, Any]]:
    cur.execute(
        """
        select * from public.project_comments
        where project_id = %s
        order by created_at desc
        """,
        (project_id,),
    )
    return [dict(r) for r in cur.fetchall()]


# ============================================================
# DB: RENDER JOBS
# ============================================================

@with_db
def create_render_job(cur, project_id: str, user_id: str, job_type: str, input_json: Dict[str, Any]) -> str:
    job_id = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.render_jobs (id, project_id, user_id, job_type, status, input_json)
        values (%s, %s, %s, %s, %s, %s)
        returning id
        """,
        (job_id, project_id, user_id, job_type, "queued", json.dumps(input_json)),
    )
    return str(cur.fetchone()["id"])


@with_db
def update_render_job(cur, job_id: str, status: str, output_json: Any = None, error_text: Optional[str] = None):
    started_at = datetime.datetime.utcnow() if status == "running" else None
    finished_at = datetime.datetime.utcnow() if status in {"done", "failed"} else None
    cur.execute(
        """
        update public.render_jobs
        set
            status = %s,
            output_json = coalesce(%s, output_json),
            error_text = coalesce(%s, error_text),
            started_at = coalesce(%s, started_at),
            finished_at = coalesce(%s, finished_at)
        where id = %s
        """,
        (
            status,
            json.dumps(output_json) if output_json is not None else None,
            error_text,
            started_at,
            finished_at,
            job_id,
        ),
    )


@with_db
def get_render_job(cur, job_id: str) -> Optional[Dict[str, Any]]:
    cur.execute("select * from public.render_jobs where id = %s", (job_id,))
    row = cur.fetchone()
    if not row:
        return None
    item = dict(row)
    item["input_json"] = safe_json(item.get("input_json"))
    item["output_json"] = safe_json(item.get("output_json"))
    return item


# ============================================================
# DB: IMAGE GENERATIONS
# ============================================================

@with_db
def create_generation(
    cur,
    project_id: str,
    user_id: str,
    prompt: str,
    expanded_prompt: str,
    model: str,
    size: str,
    action_type: str,
    parent_generation_id: Optional[str] = None,
    source_image_url: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    generation_id = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.generations
        (id, project_id, user_id, parent_generation_id, prompt, expanded_prompt, model, size, action_type, source_image_url, settings, status)
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        returning id
        """,
        (
            generation_id, project_id, user_id, parent_generation_id, prompt,
            expanded_prompt, model, size, action_type, source_image_url,
            json.dumps(settings or {}), "running"
        ),
    )
    return str(cur.fetchone()["id"])


@with_db
def update_generation_prompt(cur, generation_id: str, expanded_prompt: str):
    cur.execute(
        "update public.generations set expanded_prompt = %s where id = %s",
        (expanded_prompt, generation_id),
    )


@with_db
def complete_generation(cur, generation_id: str, image_urls: List[str]):
    cur.execute(
        """
        update public.generations
        set status = %s, output_images = %s, completed_at = now()
        where id = %s
        """,
        ("done", json.dumps(image_urls), generation_id),
    )


@with_db
def fail_generation(cur, generation_id: str, error_text: str):
    cur.execute(
        """
        update public.generations
        set status = %s, error_text = %s, completed_at = now()
        where id = %s
        """,
        ("failed", error_text[:4000], generation_id),
    )


@with_db
def add_generation_images(cur, generation_id: str, image_urls: List[str]):
    for idx, url in enumerate(image_urls):
        cur.execute(
            """
            insert into public.generation_images (id, generation_id, image_index, image_url)
            values (%s, %s, %s, %s)
            """,
            (str(uuid.uuid4()), generation_id, idx, url),
        )


@with_db
def get_generation(cur, generation_id: str) -> Optional[Dict[str, Any]]:
    cur.execute("select * from public.generations where id = %s", (generation_id,))
    row = cur.fetchone()
    if not row:
        return None
    item = dict(row)
    item["settings"] = safe_json(item.get("settings"))
    item["output_images"] = safe_json(item.get("output_images"))
    return item


@with_db
def list_generations(cur, project_id: str) -> List[Dict[str, Any]]:
    cur.execute(
        "select * from public.generations where project_id = %s order by created_at desc",
        (project_id,),
    )
    rows = []
    for row in cur.fetchall():
        item = dict(row)
        item["settings"] = safe_json(item.get("settings"))
        item["output_images"] = safe_json(item.get("output_images"))
        rows.append(item)
    return rows


# ============================================================
# DB: PRESENTATIONS
# ============================================================

@with_db
def create_presentation(cur, project_id: str, user_id: str, title: str, presenter_name: str, presenter_style: str, voice: str) -> str:
    pid = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.presentations
        (id, project_id, user_id, title, presenter_name, presenter_style, voice)
        values (%s, %s, %s, %s, %s, %s, %s)
        returning id
        """,
        (pid, project_id, user_id, title, presenter_name, presenter_style, voice),
    )
    return str(cur.fetchone()["id"])


@with_db
def add_or_update_slide(cur, presentation_id: str, slide_no: int, title: str, body: str, notes: Optional[str], asset_url: Optional[str]):
    cur.execute(
        """
        insert into public.presentation_slides
        (id, presentation_id, slide_no, title, body, notes, asset_url)
        values (%s, %s, %s, %s, %s, %s, %s)
        on conflict (presentation_id, slide_no)
        do update set
            title = excluded.title,
            body = excluded.body,
            notes = excluded.notes,
            asset_url = excluded.asset_url
        """,
        (str(uuid.uuid4()), presentation_id, slide_no, title, body, notes, asset_url),
    )


@with_db
def get_presentation(cur, presentation_id: str) -> Optional[Dict[str, Any]]:
    cur.execute("select * from public.presentations where id = %s", (presentation_id,))
    row = cur.fetchone()
    return dict(row) if row else None


@with_db
def list_presentation_slides(cur, presentation_id: str) -> List[Dict[str, Any]]:
    cur.execute(
        "select * from public.presentation_slides where presentation_id = %s order by slide_no asc",
        (presentation_id,),
    )
    return [dict(r) for r in cur.fetchall()]


@with_db
def get_slide(cur, presentation_id: str, slide_no: int) -> Optional[Dict[str, Any]]:
    cur.execute(
        "select * from public.presentation_slides where presentation_id = %s and slide_no = %s",
        (presentation_id, slide_no),
    )
    row = cur.fetchone()
    return dict(row) if row else None


@with_db
def create_presentation_session(cur, presentation_id: str, user_id: str, current_slide_no: int = 1) -> str:
    sid = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.presentation_sessions
        (id, presentation_id, user_id, current_slide_no, status)
        values (%s, %s, %s, %s, %s)
        returning id
        """,
        (sid, presentation_id, user_id, current_slide_no, "live"),
    )
    return str(cur.fetchone()["id"])


@with_db
def get_presentation_session(cur, session_id: str) -> Optional[Dict[str, Any]]:
    cur.execute("select * from public.presentation_sessions where id = %s", (session_id,))
    row = cur.fetchone()
    return dict(row) if row else None


@with_db
def update_session_slide(cur, session_id: str, slide_no: int):
    cur.execute(
        """
        update public.presentation_sessions
        set current_slide_no = %s, updated_at = now()
        where id = %s
        """,
        (slide_no, session_id),
    )


@with_db
def add_voice_log(cur, session_id: str, user_id: str, role: str, message_text: str, audio_url: Optional[str]):
    cur.execute(
        """
        insert into public.voice_interaction_logs
        (id, session_id, user_id, role, message_text, audio_url)
        values (%s, %s, %s, %s, %s, %s)
        """,
        (str(uuid.uuid4()), session_id, user_id, role, message_text, audio_url),
    )


@with_db
def get_recent_voice_logs(cur, session_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    cur.execute(
        """
        select role, message_text
        from public.voice_interaction_logs
        where session_id = %s
        order by created_at desc
        limit %s
        """,
        (session_id, limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    rows.reverse()
    return rows


# ============================================================
# LLM HELPERS
# ============================================================

def llm_text(system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    api = require_openai()
    response = api.responses.create(
        model=TEXT_MODEL,
        instructions=system_prompt,
        input=user_prompt,
        temperature=temperature,
    )
    text = getattr(response, "output_text", None)
    if not text:
        raise HTTPException(status_code=500, detail="Empty model response")
    return text.strip()


def llm_json(system_prompt: str, user_prompt: str) -> Any:
    raw = llm_text(system_prompt, user_prompt, temperature=0.2)
    cleaned = raw.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


# ============================================================
# AI CORE MODULES
# ============================================================

def analyze_brief(brief: str) -> str:
    return llm_text(
        "You are an expert AI Creative Studio planner for exhibitions, activations, road shows, concerts, government events, stage productions, retail displays and campaigns. Be precise, structured and production-focused.",
        f"""
Analyze this creative brief.

Return:
1. objective
2. audience
3. event type
4. key deliverables
5. branding requirements
6. likely dimensions if missing
7. missing clarifications
8. production concerns

Brief:
{brief}
""",
    )


def generate_concepts(analysis: str) -> List[Dict[str, Any]]:
    result = llm_json(
        "Return only valid JSON.",
        f"""
Return exactly 3 concept objects in a JSON array.

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
        "You create premium visual directions for events, exhibitions, retail and brand environments.",
        f"""
Create a polished moodboard direction.

Include:
- color palette
- material palette
- lighting language
- branding placement logic
- finish recommendations
- premium visual notes

Concept:
{json.dumps(selected_concept, indent=2)}
""",
    )


def generate_brand_strategy(project: Dict[str, Any]) -> Dict[str, Any]:
    return llm_json(
        "Return only valid JSON.",
        f"""
Create a brand strategy object.

Fields:
- positioning
- target_audience
- audience_insights
- tone_of_voice
- messaging_pillars
- brand_personality
- value_proposition
- competitor_direction
- visual_identity_cues

Project brief:
{project.get('brief') or ''}
Analysis:
{project.get('analysis') or ''}
""",
    )


def generate_campaign_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    return llm_json(
        "Return only valid JSON.",
        f"""
Create a campaign planning object.

Fields:
- objective
- launch_campaign
- festival_campaigns
- monthly_content_calendar
- channels
- ad_themes
- activations
- KPI_focus

Project brief:
{project.get('brief') or ''}
Brand strategy:
{json.dumps(project.get('brand_strategy') or {}, indent=2)}
""",
    )


def generate_ad_creatives(project: Dict[str, Any]) -> List[Dict[str, Any]]:
    return llm_json(
        "Return only valid JSON.",
        f"""
Return a JSON array of 6 ad creative objects.

Each object must contain:
- platform
- headline
- primary_text
- cta
- visual_prompt
- angle
- variation_label

Project brief:
{project.get('brief') or ''}
Brand strategy:
{json.dumps(project.get('brand_strategy') or {}, indent=2)}
Campaign:
{json.dumps(project.get('campaign_plan') or {}, indent=2)}
""",
    )


def generate_video_scripts(project: Dict[str, Any]) -> List[Dict[str, Any]]:
    return llm_json(
        "Return only valid JSON.",
        f"""
Return a JSON array of 3 short video script objects.

Each object must contain:
- platform
- duration
- hook
- concept
- scenes
- voiceover
- on_screen_text
- cta

Project brief:
{project.get('brief') or ''}
Brand strategy:
{json.dumps(project.get('brand_strategy') or {}, indent=2)}
Campaign:
{json.dumps(project.get('campaign_plan') or {}, indent=2)}
""",
    )


def generate_packaging_data(project: Dict[str, Any]) -> Dict[str, Any]:
    return llm_json(
        "Return only valid JSON.",
        f"""
Create packaging strategy JSON.

Fields:
- packaging_concept
- pack_structure
- materials
- finish
- shelf_impact_logic
- label_copy
- visual_direction
- premium_notes

Project brief:
{project.get('brief') or ''}
Brand strategy:
{json.dumps(project.get('brand_strategy') or {}, indent=2)}
""",
    )


def generate_retail_display_data(project: Dict[str, Any]) -> Dict[str, Any]:
    return llm_json(
        "Return only valid JSON.",
        f"""
Create retail display JSON.

Fields:
- retail_concept
- display_type
- layout_logic
- shopper_flow
- branding_surfaces
- product_zones
- materials
- lighting
- fabrication_notes

Project brief:
{project.get('brief') or ''}
Packaging:
{json.dumps(project.get('packaging_data') or {}, indent=2)}
""",
    )


def generate_scene_json(project: Dict[str, Any]) -> Dict[str, Any]:
    selected = project.get("selected") or {}
    result = llm_json(
        "Return only valid JSON.",
        f"""
Create scene JSON for Blender.

Fields:
- units
- stage
- led_wall
- truss
- audience
- colors
- lighting
- branding
- camera_target
- render
- zones
- objects
- materials

Brief:
{project.get('brief') or ''}

Selected concept:
{json.dumps(selected, indent=2)}

Retail display:
{json.dumps(project.get('retail_display_data') or {}, indent=2)}
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
    result.setdefault("lighting", {"style": "premium"})
    result.setdefault("branding", {"notes": "premium branded environment"})
    result.setdefault("camera_target", [0, 0, 6])
    result.setdefault("render", {"ratio": "16:9"})
    return result


def generate_boq(project: Dict[str, Any]) -> Dict[str, Any]:
    scene = project.get("scene_json") or {}
    return llm_json(
        "Return only valid JSON.",
        f"""
Generate a BOQ/costing JSON.

Fields:
- print_area_estimate
- flooring_area_estimate
- led_area_estimate
- truss_estimate
- material_summary
- fixture_summary
- estimated_cost_band
- assumptions

Scene:
{json.dumps(scene, indent=2)}
Brief:
{project.get('brief') or ''}
""",
    )


def generate_audio_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    return llm_json(
        "Return only valid JSON.",
        f"""
Generate an audio plan JSON.

Fields:
- presenter_voice_style
- speaker_zones
- ambience_style
- announcement_style
- multilingual_notes
- cue_notes

Brief:
{project.get('brief') or ''}
Presentation data:
{json.dumps(project.get('presentation_data') or {}, indent=2)}
""",
    )


def generate_video_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    return llm_json(
        "Return only valid JSON.",
        f"""
Generate a video walkthrough plan JSON.

Fields:
- opening_shot
- camera_path
- key_shots
- transitions
- overlays
- voiceover_style
- output_formats

Scene:
{json.dumps(project.get('scene_json') or {}, indent=2)}
Brand strategy:
{json.dumps(project.get('brand_strategy') or {}, indent=2)}
""",
    )


def expand_image_prompt(prompt: str, style_strength: str, event_type: Optional[str]) -> str:
    return llm_text(
        "You are a senior visual prompt engineer. Return only one polished image prompt.",
        f"""
Rewrite this user prompt into a stronger image generation prompt.

User prompt:
{prompt}

Rules:
- preserve intent
- improve composition, materials, camera, lighting, realism
- style strength: {style_strength}
- event type: {event_type or 'not specified'}
- no markdown

Return only the final prompt.
""",
        temperature=0.6,
    )


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_images_core(
    prompt: str,
    num_images: int,
    size: str,
    style_strength: str,
    reference_image_url: Optional[str],
    event_type: Optional[str],
) -> Dict[str, Any]:
    api = require_openai()
    expanded_prompt = expand_image_prompt(prompt, style_strength, event_type)

    if reference_image_url:
        temp_path = local_or_remote_to_temp_file(reference_image_url, ".png")
        try:
            urls: List[str] = []
            for _ in range(num_images):
                with open(temp_path, "rb") as image_file:
                    result = api.images.edit(
                        model=IMAGE_MODEL,
                        image=image_file,
                        prompt=expanded_prompt,
                        size=size,
                    )
                urls.extend(image_response_to_urls(result, "edit_ref"))
            return {"expanded_prompt": expanded_prompt, "image_urls": urls[:num_images]}
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    urls: List[str] = []
    while len(urls) < num_images:
        result = api.images.generate(
            model=IMAGE_MODEL,
            prompt=expanded_prompt,
            size=size,
        )
        urls.extend(image_response_to_urls(result, "gen"))
    return {"expanded_prompt": expanded_prompt, "image_urls": urls[:num_images]}


def variation_prompt(base_prompt: str, strength: str, remix_prompt: Optional[str]) -> str:
    if remix_prompt:
        return f"Create a {strength} variation of the source image using this remix direction: {remix_prompt}\n\nBase intent:\n{base_prompt}"
    if strength == "subtle":
        return f"Create a subtle variation of the source image. Keep composition and identity close. Change only polish, detail, atmosphere and styling.\n\nBase intent:\n{base_prompt}"
    return f"Create a strong variation of the source image. Keep the core subject, but explore clearly different style, materials, lighting and drama.\n\nBase intent:\n{base_prompt}"


# ============================================================
# AUDIO / PRESENTER HELPERS
# ============================================================

def tts_to_file(text: str, voice: str = "marin") -> str:
    api = require_openai()
    folder = MEDIA_DIR / "presenter_audio"
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"{uuid.uuid4().hex}.mp3"

    speech = api.audio.speech.create(
        model=TTS_MODEL,
        voice=voice,
        input=text,
    )
    speech.stream_to_file(str(out_path))
    return build_public_media_url(out_path)


def transcribe_upload(upload: UploadFile) -> str:
    api = require_openai()
    suffix = Path(upload.filename or "audio.webm").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(upload.file.read())
        tmp.close()
        with open(tmp.name, "rb") as f:
            transcript = api.audio.transcriptions.create(
                model=STT_MODEL,
                file=f,
            )
        return getattr(transcript, "text", "").strip()
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def build_slide_context(presentation_id: str, slide_no: int) -> str:
    presentation = get_presentation(presentation_id)
    slide = get_slide(presentation_id, slide_no)
    slides = list_presentation_slides(presentation_id)
    if not presentation or not slide:
        raise HTTPException(status_code=404, detail="Presentation or slide not found")

    outline = [f"Slide {s['slide_no']}: {s['title']}" for s in slides]
    return f"""
Presentation title: {presentation.get('title')}
Presenter name: {presentation.get('presenter_name')}
Presenter style: {presentation.get('presenter_style')}

Deck outline:
{chr(10).join(outline)}

Current slide number: {slide['slide_no']}
Current slide title: {slide['title']}
Current slide body:
{slide['body']}

Presenter notes:
{slide.get('notes') or ''}
""".strip()


def generate_slide_narration(presentation_id: str, slide_no: int, language: str, audience_mode: str) -> str:
    context = build_slide_context(presentation_id, slide_no)
    return llm_text(
        "You are a live AI presenter. Speak naturally and clearly. Be engaging, persuasive and concise.",
        f"""
Use this context:
{context}

Narrate this slide for a live audience.

Rules:
- language: {language}
- audience mode: {audience_mode}
- 90 to 160 words
- sound like a polished human presenter
- explain, do not robotically read
- return narration text only
""",
        temperature=0.6,
    )


def answer_presentation_question(session_id: str, user_question: str, language: str) -> str:
    session = get_presentation_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    presentation = get_presentation(session["presentation_id"])
    slide = get_slide(session["presentation_id"], session["current_slide_no"])
    history = get_recent_voice_logs(session_id)
    history_text = "\n".join([f"{x['role']}: {x['message_text']}" for x in history])

    return llm_text(
        "You are a live AI presenter answering audience questions. Use current slide context first.",
        f"""
Presentation title: {presentation.get('title')}
Presenter style: {presentation.get('presenter_style')}
Language: {language}

Current slide number: {slide.get('slide_no')}
Current slide title: {slide.get('title')}
Current slide body:
{slide.get('body')}

Notes:
{slide.get('notes') or ''}

Recent history:
{history_text}

Audience question:
{user_question}

Rules:
- answer naturally
- 40 to 140 words
- use current slide first
- simplify if helpful
- no markdown

Return only the answer.
""",
        temperature=0.5,
    )


def run_presentation_command(session_id: str, command: str) -> Dict[str, Any]:
    session = get_presentation_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    slides = list_presentation_slides(session["presentation_id"])
    total = len(slides)
    current = int(session["current_slide_no"])
    cmd = command.strip().lower()

    if cmd in {"next", "next slide"}:
        new_slide = min(current + 1, total)
        update_session_slide(session_id, new_slide)
        return {"action": "slide_changed", "slide_no": new_slide}

    if cmd in {"previous", "prev", "previous slide"}:
        new_slide = max(current - 1, 1)
        update_session_slide(session_id, new_slide)
        return {"action": "slide_changed", "slide_no": new_slide}

    m = re.search(r"slide\s+(\d+)", cmd)
    if m:
        n = int(m.group(1))
        if 1 <= n <= total:
            update_session_slide(session_id, n)
            return {"action": "slide_changed", "slide_no": n}

    return {"action": "no_change", "slide_no": current}


def build_presenter_realtime_instructions(presentation_id: str, slide_no: int, audience_mode: str = "client", language: str = "en") -> str:
    presentation = get_presentation(presentation_id)
    slide = get_slide(presentation_id, slide_no)
    slides = list_presentation_slides(presentation_id)
    if not presentation or not slide:
        raise HTTPException(status_code=404, detail="Presentation or slide not found")

    outline = [f"Slide {s['slide_no']}: {s['title']}" for s in slides]
    return f"""
You are {presentation.get('presenter_name') or 'AI Presenter'}, a live voice presenter.

Presentation title: {presentation.get('title') or 'Untitled Presentation'}
Presenter style: {presentation.get('presenter_style') or 'corporate'}
Audience mode: {audience_mode}
Language: {language}

Deck outline:
{chr(10).join(outline)}

Current slide number: {slide['slide_no']}
Current slide title: {slide['title']}
Current slide body:
{slide['body']}

Presenter notes:
{slide.get('notes') or ''}

Behavior:
- speak naturally
- present the current slide clearly
- answer interruptions briefly and confidently
- use current slide context first
- do not claim to click buttons yourself
""".strip()


# ============================================================
# BLENDER EXECUTION
# ============================================================

def run_blender_multi_angle(scene_json: Dict[str, Any], job_id: str, width: int, height: int, project_id: str, user_id: str):
    try:
        update_render_job(job_id, "running")

        job_dir = RENDER_OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        scene_path = job_dir / "scene.json"

        with open(scene_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scene": scene_json,
                    "render": {"width": width, "height": height, "output_dir": str(job_dir)},
                },
                f,
                indent=2,
            )

        cmd = [BLENDER_PATH, "-b", "-P", BLENDER_SCRIPT, "--", str(scene_path)]
        completed = subprocess.run(cmd, capture_output=True, text=True)

        if completed.returncode != 0:
            update_render_job(job_id, "failed", error_text=completed.stderr[:4000])
            return

        render_map = {
            "front_wide": build_public_media_url(job_dir / "front_wide.png") if (job_dir / "front_wide.png").exists() else None,
            "front_center": build_public_media_url(job_dir / "front_center.png") if (job_dir / "front_center.png").exists() else None,
            "left_perspective": build_public_media_url(job_dir / "left_perspective.png") if (job_dir / "left_perspective.png").exists() else None,
            "right_perspective": build_public_media_url(job_dir / "right_perspective.png") if (job_dir / "right_perspective.png").exists() else None,
            "top_plan": build_public_media_url(job_dir / "top_plan.png") if (job_dir / "top_plan.png").exists() else None,
            "audience_view": build_public_media_url(job_dir / "audience_view.png") if (job_dir / "audience_view.png").exists() else None,
            "glb": build_public_media_url(job_dir / "scene.glb") if (job_dir / "scene.glb").exists() else None,
        }

        update_render_job(job_id, "done", output_json=render_map)
        update_project_field(project_id, "render3d", render_map)
        snapshot_project_version(project_id, user_id, note="3D multi-angle render completed")

    except Exception as e:
        update_render_job(job_id, "failed", error_text=str(e))


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {"message": "AI Creative Studio API is running", "time": now_iso()}


# ============================================================
# AUTH ROUTES
# ============================================================

@app.post("/signup")
def signup(payload: UserInput):
    existing = get_user_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    user_id = create_user(payload.email, payload.password, payload.full_name)
    token = create_token(user_id)
    return {"message": "User created", "user_id": user_id, "token": token}


@app.post("/login")
def login(payload: UserInput):
    user = get_user_by_email(payload.email)
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    if not verify_password(payload.password, user["password"]):
        raise HTTPException(status_code=400, detail="Wrong password")
    token = create_token(str(user["id"]))
    return {"token": token, "user_id": str(user["id"])}


# ============================================================
# PROJECT ROUTES
# ============================================================

@app.get("/project/{project_id}")
def project_detail(project_id: str, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.post("/project/update")
def project_update(payload: UpdateProjectInput, user_id: str = Depends(get_current_user_id)):
    update_project_field(payload.project_id, payload.field, payload.value)
    snapshot_project_version(payload.project_id, user_id, note=f"Updated field: {payload.field}")
    return {"message": "Project updated"}


# ============================================================
# COMMENTS
# ============================================================

@app.post("/comment")
def create_comment(payload: CommentInput, user_id: str = Depends(get_current_user_id)):
    comment_id = add_comment(payload.project_id, user_id, payload.section, payload.comment_text)
    return {"message": "Comment added", "comment_id": comment_id}


@app.get("/comments/{project_id}")
def comments(project_id: str, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    return {"comments": get_comments(project_id)}


# ============================================================
# MAIN BRIEF PIPELINE
# ============================================================

@app.post("/run")
def run_pipeline(payload: RunInput, user_id: str = Depends(get_current_user_id)):
    project_id = payload.project_id or create_project(
        user_id=user_id,
        name=payload.name or best_project_name(payload.text),
        event_type=payload.event_type,
    )

    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

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
        snapshot_project_version(project_id, user_id, note="Concepts generated")
        return {"stage": "concepts_ready", "project_id": project_id, "concepts": concepts}

    if not project.get("selected"):
        return {"stage": "awaiting_concept_selection", "project_id": project_id, "options": project["concepts"]}

    if not project.get("moodboard"):
        moodboard = generate_moodboard(project["selected"])
        update_project_field(project_id, "moodboard", moodboard)
        update_project_field(project_id, "status", "moodboard_ready")
        snapshot_project_version(project_id, user_id, note="Moodboard generated")
        return {"stage": "moodboard_ready", "project_id": project_id, "moodboard": moodboard}

    return {"stage": "core_strategy_ready", "project_id": project_id}


@app.post("/select")
def select_concept(payload: SelectConceptInput, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    concepts = project.get("concepts")
    if not isinstance(concepts, list) or payload.index < 0 or payload.index >= len(concepts):
        raise HTTPException(status_code=400, detail="Invalid concept index")
    selected = concepts[payload.index]
    update_project_field(payload.project_id, "selected", selected)
    update_project_field(payload.project_id, "status", "concept_selected")
    snapshot_project_version(payload.project_id, user_id, note=f"Concept {payload.index} selected")
    return {"message": "Concept selected", "selected": selected}


# ============================================================
# STRATEGY / CAMPAIGN / CREATIVE MODULES
# ============================================================

@app.post("/brand-strategy/generate")
def brand_strategy_generate(payload: ProjectOnlyInput, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    data = generate_brand_strategy(project)
    update_project_field(payload.project_id, "brand_strategy", data)
    snapshot_project_version(payload.project_id, user_id, note="Brand strategy generated")
    return {"project_id": payload.project_id, "brand_strategy": data}


@app.post("/campaign-plan/generate")
def campaign_plan_generate(payload: ProjectOnlyInput, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    data = generate_campaign_plan(project)
    update_project_field(payload.project_id, "campaign_plan", data)
    snapshot_project_version(payload.project_id, user_id, note="Campaign plan generated")
    return {"project_id": payload.project_id, "campaign_plan": data}


@app.post("/ads/generate")
def ads_generate(payload: ProjectOnlyInput, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    data = generate_ad_creatives(project)
    update_project_field(payload.project_id, "ad_creatives", data)
    snapshot_project_version(payload.project_id, user_id, note="Ad creatives generated")
    return {"project_id": payload.project_id, "ad_creatives": data}


@app.post("/video-script/generate")
def video_script_generate(payload: ProjectOnlyInput, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    data = generate_video_scripts(project)
    update_project_field(payload.project_id, "video_scripts", data)
    snapshot_project_version(payload.project_id, user_id, note="Video scripts generated")
    return {"project_id": payload.project_id, "video_scripts": data}


@app.post("/packaging/generate")
def packaging_generate(payload: ProjectOnlyInput, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    data = generate_packaging_data(project)
    update_project_field(payload.project_id, "packaging_data", data)
    snapshot_project_version(payload.project_id, user_id, note="Packaging generated")
    return {"project_id": payload.project_id, "packaging_data": data}


@app.post("/retail-display/generate")
def retail_display_generate(payload: ProjectOnlyInput, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    data = generate_retail_display_data(project)
    update_project_field(payload.project_id, "retail_display_data", data)
    snapshot_project_version(payload.project_id, user_id, note="Retail display generated")
    return {"project_id": payload.project_id, "retail_display_data": data}


@app.post("/scene/generate")
def scene_generate(payload: ProjectOnlyInput, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    data = generate_scene_json(project)
    update_project_field(payload.project_id, "scene_json", data)
    snapshot_project_version(payload.project_id, user_id, note="Scene JSON generated")
    return {"project_id": payload.project_id, "scene_json": data}


@app.post("/boq/generate")
def boq_generate(payload: ProjectOnlyInput, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    data = generate_boq(project)
    update_project_field(payload.project_id, "boq_data", data)
    snapshot_project_version(payload.project_id, user_id, note="BOQ generated")
    return {"project_id": payload.project_id, "boq_data": data}


@app.post("/audio/plan")
def audio_plan_generate(payload: ProjectOnlyInput, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    data = generate_audio_plan(project)
    update_project_field(payload.project_id, "audio_plan", data)
    snapshot_project_version(payload.project_id, user_id, note="Audio plan generated")
    return {"project_id": payload.project_id, "audio_plan": data}


@app.post("/video/plan")
def video_plan_generate(payload: ProjectOnlyInput, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    data = generate_video_plan(project)
    update_project_field(payload.project_id, "video_plan", data)
    snapshot_project_version(payload.project_id, user_id, note="Video plan generated")
    return {"project_id": payload.project_id, "video_plan": data}


# ============================================================
# IMAGE STUDIO
# ============================================================

@app.post("/generate")
def generate_images(payload: GenerateImageInput, user_id: str = Depends(get_current_user_id)):
    project_id = payload.project_id or create_project(
        user_id=user_id,
        name=payload.project_name or best_project_name(payload.prompt),
        event_type=payload.event_type,
    )

    generation_id = create_generation(
        project_id=project_id,
        user_id=user_id,
        prompt=payload.prompt,
        expanded_prompt="",
        model=IMAGE_MODEL,
        size=payload.size,
        action_type="generate",
        source_image_url=payload.reference_image_url,
        settings={"num_images": payload.num_images, "style_strength": payload.style_strength},
    )

    try:
        result = generate_images_core(
            prompt=payload.prompt,
            num_images=payload.num_images,
            size=payload.size,
            style_strength=payload.style_strength,
            reference_image_url=payload.reference_image_url,
            event_type=payload.event_type,
        )
        update_generation_prompt(generation_id, result["expanded_prompt"])
        add_generation_images(generation_id, result["image_urls"])
        complete_generation(generation_id, result["image_urls"])
        snapshot_project_version(project_id, user_id, note="Images generated")
        return {
            "message": "Images generated",
            "project_id": project_id,
            "generation_id": generation_id,
            "expanded_prompt": result["expanded_prompt"],
            "images": result["image_urls"],
        }
    except Exception as e:
        fail_generation(generation_id, str(e))
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


@app.post("/variation")
def create_variation(payload: VariationInput, user_id: str = Depends(get_current_user_id)):
    source = get_generation(payload.generation_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source generation not found")
    output_images = source.get("output_images") or []
    if payload.image_index >= len(output_images):
        raise HTTPException(status_code=400, detail="Invalid image index")

    source_image_url = output_images[payload.image_index]
    base_prompt = source.get("expanded_prompt") or source.get("prompt") or ""
    new_prompt = variation_prompt(base_prompt, payload.strength, payload.remix_prompt)

    generation_id = create_generation(
        project_id=source["project_id"],
        user_id=user_id,
        prompt=source.get("prompt") or "",
        expanded_prompt=new_prompt,
        model=IMAGE_MODEL,
        size=payload.size,
        action_type="variation",
        parent_generation_id=source["id"],
        source_image_url=source_image_url,
        settings={"strength": payload.strength, "num_images": payload.num_images, "source_index": payload.image_index},
    )

    try:
        temp_path = local_or_remote_to_temp_file(source_image_url, ".png")
        urls: List[str] = []
        api = require_openai()
        try:
            for _ in range(payload.num_images):
                with open(temp_path, "rb") as image_file:
                    result = api.images.edit(
                        model=IMAGE_MODEL,
                        image=image_file,
                        prompt=new_prompt,
                        size=payload.size,
                    )
                urls.extend(image_response_to_urls(result, "variation"))
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        urls = urls[:payload.num_images]
        add_generation_images(generation_id, urls)
        complete_generation(generation_id, urls)
        snapshot_project_version(source["project_id"], user_id, note="Image variation created")
        return {
            "message": "Variation created",
            "project_id": source["project_id"],
            "generation_id": generation_id,
            "images": urls,
            "expanded_prompt": new_prompt,
        }
    except Exception as e:
        fail_generation(generation_id, str(e))
        raise HTTPException(status_code=500, detail=f"Variation failed: {str(e)}")


@app.post("/edit-image")
def edit_image(payload: EditImageInput, user_id: str = Depends(get_current_user_id)):
    source_image_url = payload.source_image_url
    project_id: Optional[str] = None
    parent_generation_id: Optional[str] = None

    if payload.generation_id:
        source = get_generation(payload.generation_id)
        if not source:
            raise HTTPException(status_code=404, detail="Generation not found")
        idx = payload.image_index or 0
        output_images = source.get("output_images") or []
        if idx >= len(output_images):
            raise HTTPException(status_code=400, detail="Invalid image index")
        source_image_url = output_images[idx]
        project_id = source["project_id"]
        parent_generation_id = source["id"]

    if not source_image_url:
        raise HTTPException(status_code=400, detail="source_image_url or generation_id is required")

    if not project_id:
        project_id = create_project(user_id=user_id, name=best_project_name(payload.prompt), event_type=None)

    generation_id = create_generation(
        project_id=project_id,
        user_id=user_id,
        prompt=payload.prompt,
        expanded_prompt=payload.prompt,
        model=IMAGE_MODEL,
        size=payload.size,
        action_type="edit",
        parent_generation_id=parent_generation_id,
        source_image_url=source_image_url,
        settings={"image_index": payload.image_index},
    )

    try:
        temp_path = local_or_remote_to_temp_file(source_image_url, ".png")
        try:
            api = require_openai()
            with open(temp_path, "rb") as image_file:
                result = api.images.edit(
                    model=IMAGE_MODEL,
                    image=image_file,
                    prompt=payload.prompt,
                    size=payload.size,
                )
            urls = image_response_to_urls(result, "edit")
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        if not urls:
            raise HTTPException(status_code=500, detail="No image returned")

        add_generation_images(generation_id, urls[:1])
        complete_generation(generation_id, urls[:1])
        snapshot_project_version(project_id, user_id, note="Edited image")
        return {"message": "Image edited", "project_id": project_id, "generation_id": generation_id, "image": urls[0]}
    except Exception as e:
        fail_generation(generation_id, str(e))
        raise HTTPException(status_code=500, detail=f"Edit failed: {str(e)}")


@app.get("/generation/{generation_id}")
def generation_detail(generation_id: str, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    item = get_generation(generation_id)
    if not item:
        raise HTTPException(status_code=404, detail="Generation not found")
    return item


@app.get("/generations/{project_id}")
def project_generations(project_id: str, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    return {"project_id": project_id, "generations": list_generations(project_id)}


# ============================================================
# 3D RENDER
# ============================================================

@app.post("/generate-multi-angle")
def generate_multi_angle(payload: Generate3DInput, background_tasks: BackgroundTasks, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.get("scene_json"):
        scene_json = generate_scene_json(project)
        update_project_field(payload.project_id, "scene_json", scene_json)
    else:
        scene_json = project["scene_json"]

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

    return {"message": "3D render queued", "job_id": job_id, "project_id": payload.project_id}


@app.get("/job/{job_id}")
def job_status(job_id: str, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    job = get_render_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ============================================================
# PRESENTATIONS
# ============================================================

@app.post("/presentation/create")
def presentation_create(payload: PresentationCreateInput, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    pid = create_presentation(
        project_id=payload.project_id,
        user_id=user_id,
        title=payload.title or f"{project.get('name', 'Project')} Presentation",
        presenter_name=payload.presenter_name,
        presenter_style=payload.presenter_style,
        voice=payload.voice,
    )
    snapshot_project_version(payload.project_id, user_id, note="Presentation created")
    return {"message": "Presentation created", "presentation_id": pid}


@app.post("/presentation/slide")
def presentation_slide(payload: PresentationSlideInput, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    presentation = get_presentation(payload.presentation_id)
    if not presentation:
        raise HTTPException(status_code=404, detail="Presentation not found")
    add_or_update_slide(payload.presentation_id, payload.slide_no, payload.title, payload.body, payload.notes, payload.asset_url)
    return {"message": "Slide saved"}


@app.get("/presentation/{presentation_id}")
def presentation_detail(presentation_id: str, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    presentation = get_presentation(presentation_id)
    if not presentation:
        raise HTTPException(status_code=404, detail="Presentation not found")
    slides = list_presentation_slides(presentation_id)
    return {"presentation": presentation, "slides": slides}


@app.post("/presentation/start-live/{presentation_id}")
def presentation_start_live(presentation_id: str, user_id: str = Depends(get_current_user_id)):
    presentation = get_presentation(presentation_id)
    if not presentation:
        raise HTTPException(status_code=404, detail="Presentation not found")
    session_id = create_presentation_session(presentation_id, user_id, current_slide_no=1)
    return {"message": "Live session started", "session_id": session_id, "current_slide_no": 1}


@app.post("/presentation/narrate-slide")
def presentation_narrate_slide(payload: PresentationNarrateInput, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    presentation = get_presentation(payload.presentation_id)
    if not presentation:
        raise HTTPException(status_code=404, detail="Presentation not found")
    narration = generate_slide_narration(payload.presentation_id, payload.slide_no, payload.language, payload.audience_mode)
    audio_url = tts_to_file(narration, voice=presentation.get("voice") or "marin")
    return {"slide_no": payload.slide_no, "text": narration, "audio_url": audio_url}


@app.post("/presentation/command")
def presentation_command(payload: PresentationCommandInput, user_id: str = Depends(get_current_user_id)):
    result = run_presentation_command(payload.session_id, payload.command)
    add_voice_log(payload.session_id, user_id, "user", payload.command, None)
    return result


@app.post("/presentation/ask")
def presentation_ask(payload: PresentationAskInput, user_id: str = Depends(get_current_user_id)):
    session = get_presentation_session(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    presentation = get_presentation(session["presentation_id"])
    answer = answer_presentation_question(payload.session_id, payload.question, payload.language)
    audio_url = tts_to_file(answer, voice=presentation.get("voice") or "marin")
    add_voice_log(payload.session_id, user_id, "user", payload.question, None)
    add_voice_log(payload.session_id, user_id, "assistant", answer, audio_url)
    return {"current_slide_no": session["current_slide_no"], "answer": answer, "audio_url": audio_url}


@app.post("/presentation/ask-audio/{session_id}")
def presentation_ask_audio(
    session_id: str,
    file: UploadFile = File(...),
    language: str = "en",
    user_id: str = Depends(get_current_user_id),
):
    session = get_presentation_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    presentation = get_presentation(session["presentation_id"])
    transcript = transcribe_upload(file)
    if not transcript:
        raise HTTPException(status_code=400, detail="Could not transcribe audio")
    answer = answer_presentation_question(session_id, transcript, language)
    audio_url = tts_to_file(answer, voice=presentation.get("voice") or "marin")
    add_voice_log(session_id, user_id, "user", transcript, None)
    add_voice_log(session_id, user_id, "assistant", answer, audio_url)
    return {"transcript": transcript, "answer": answer, "audio_url": audio_url, "current_slide_no": session["current_slide_no"]}


@app.get("/presentation/session/{session_id}")
def presentation_session_detail(session_id: str, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    session = get_presentation_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    slide = get_slide(session["presentation_id"], session["current_slide_no"])
    logs = get_recent_voice_logs(session_id, limit=20)
    return {"session": session, "current_slide": slide, "logs": logs}


# ============================================================
# REALTIME WEBRTC PRESENTER
# ============================================================

@app.post("/presentation/realtime/session", response_class=PlainTextResponse)
async def presentation_realtime_session(
    request: FastAPIRequest,
    presentation_id: str,
    slide_no: int = 1,
    voice: str = "marin",
    audience_mode: str = "client",
    language: str = "en",
    user_id: str = Depends(get_current_user_id),
):
    presentation = get_presentation(presentation_id)
    if not presentation:
        raise HTTPException(status_code=404, detail="Presentation not found")

    sdp_offer = (await request.body()).decode("utf-8", errors="ignore").strip()
    if not sdp_offer:
        raise HTTPException(status_code=400, detail="Missing SDP offer")

    session_id = create_presentation_session(presentation_id, user_id, slide_no)
    instructions = build_presenter_realtime_instructions(presentation_id, slide_no, audience_mode, language)

    session_config = {
        "type": "realtime",
        "model": REALTIME_MODEL,
        "audio": {
            "output": {
                "voice": voice
            }
        },
        "instructions": instructions,
    }

    files = {
        "sdp": (None, sdp_offer),
        "session": (None, json.dumps(session_config)),
    }

    try:
        r = requests.post(
            "https://api.openai.com/v1/realtime/calls",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files=files,
            timeout=60,
        )
        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail=f"Realtime failed: {r.text[:1000]}")

        add_voice_log(session_id, user_id, "system", f"Realtime session started on slide {slide_no}", None)
        return PlainTextResponse(r.text)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Realtime connection error: {str(e)}")


@app.post("/presentation/realtime/instructions")
def presentation_realtime_instructions(payload: RealtimeSessionUpdateInput, user_id: str = Depends(get_current_user_id)):
    presentation = get_presentation(payload.presentation_id)
    if not presentation:
        raise HTTPException(status_code=404, detail="Presentation not found")
    session = get_presentation_session(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    update_session_slide(payload.session_id, payload.slide_no)
    instructions = build_presenter_realtime_instructions(
        payload.presentation_id,
        payload.slide_no,
        payload.audience_mode,
        payload.language,
    )

    add_voice_log(payload.session_id, user_id, "system", f"Realtime instructions updated for slide {payload.slide_no}", None)

    return {
        "session_update": {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": REALTIME_MODEL,
                "output_modalities": ["audio"],
                "audio": {"output": {"voice": presentation.get("voice") or "marin"}},
                "instructions": instructions,
            },
        }
    }


# ============================================================
# OPTIONAL AUTO PRESENTATION DECK FROM PROJECT
# ============================================================

@app.post("/presentation/auto-build/{project_id}")
def auto_build_presentation(project_id: str, user_id: str = Depends(get_current_user_id)):
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    presentation_id = create_presentation(
        project_id=project_id,
        user_id=user_id,
        title=f"{project.get('name', 'Project')} Presentation",
        presenter_name="AI Presenter",
        presenter_style="corporate",
        voice="marin",
    )

    slides = [
        {
            "slide_no": 1,
            "title": "Project Overview",
            "body": project.get("brief") or "No brief available.",
            "notes": "Introduce the project and the high-level objective.",
            "asset_url": None,
        },
        {
            "slide_no": 2,
            "title": "Brand Strategy",
            "body": json.dumps(project.get("brand_strategy") or {}, indent=2),
            "notes": "Explain positioning, audience and tone.",
            "asset_url": None,
        },
        {
            "slide_no": 3,
            "title": "Campaign Plan",
            "body": json.dumps(project.get("campaign_plan") or {}, indent=2),
            "notes": "Explain launch and channel approach.",
            "asset_url": None,
        },
        {
            "slide_no": 4,
            "title": "Selected Concept",
            "body": json.dumps(project.get("selected") or {}, indent=2),
            "notes": "Present the approved concept direction.",
            "asset_url": None,
        },
        {
            "slide_no": 5,
            "title": "3D / Render Summary",
            "body": json.dumps(project.get("render3d") or project.get("scene_json") or {}, indent=2),
            "notes": "Talk through the scene, layout and render viewpoints.",
            "asset_url": None,
        },
        {
            "slide_no": 6,
            "title": "BOQ / Cost Summary",
            "body": json.dumps(project.get("boq_data") or {}, indent=2),
            "notes": "Summarize cost assumptions and production estimates.",
            "asset_url": None,
        },
    ]

    for s in slides:
        add_or_update_slide(
            presentation_id=presentation_id,
            slide_no=s["slide_no"],
            title=s["title"],
            body=s["body"],
            notes=s["notes"],
            asset_url=s["asset_url"],
        )

    snapshot_project_version(project_id, user_id, note="Auto presentation built")
    return {"message": "Presentation built", "presentation_id": presentation_id}