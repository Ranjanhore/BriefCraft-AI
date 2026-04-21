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

import requests
import psycopg
from psycopg.rows import dict_row

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
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import PlainTextResponse

from openai import OpenAI


# ============================================================
# ENV
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
SECRET_KEY = os.getenv("SECRET_KEY", "").strip()

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

ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]

RENDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GLOBALS
# ============================================================

client: Optional[OpenAI] = None
pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


# ============================================================
# DB
# ============================================================

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


@with_db
def create_project(cur, user_id, name=None, event_type=None):
    uid = str(uuid.uuid4())
    cur.execute(
        """
        insert into users (id, email, password, full_name)
        values (%s, %s, %s, %s)
        """,
        (uid, email, hash_password(password), full_name)
    )
    return uid


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

            cur.execute("""
                create table if not exists public.generations (
                  id uuid primary key,
                  project_id uuid not null references public.projects(id) on delete cascade,
                  user_id uuid not null references public.users(id) on delete cascade,
                  parent_generation_id uuid references public.generations(id) on delete set null,
                  prompt text,
                  expanded_prompt text,
                  model text,
                  size text,
                  action_type text,
                  source_image_url text,
                  settings jsonb,
                  output_images jsonb,
                  status text default 'queued',
                  error_text text,
                  created_at timestamptz default now(),
                  completed_at timestamptz
                );
            """)

            cur.execute("""
                create table if not exists public.generation_images (
                  id uuid primary key,
                  generation_id uuid not null references public.generations(id) on delete cascade,
                  image_index int not null,
                  image_url text not null,
                  created_at timestamptz default now()
                );
            """)

            cur.execute("""
                create table if not exists public.presentations (
                  id uuid primary key,
                  project_id uuid not null references public.projects(id) on delete cascade,
                  user_id uuid not null references public.users(id) on delete cascade,
                  title text,
                  presenter_name text default 'AI Presenter',
                  presenter_style text default 'corporate',
                  voice text default 'marin',
                  created_at timestamptz default now()
                );
            """)

            cur.execute("""
                create table if not exists public.presentation_slides (
                  id uuid primary key,
                  presentation_id uuid not null references public.presentations(id) on delete cascade,
                  slide_no int not null,
                  title text not null,
                  body text not null,
                  notes text,
                  asset_url text,
                  created_at timestamptz default now(),
                  unique (presentation_id, slide_no)
                );
            """)

            cur.execute("""
                create table if not exists public.presentation_sessions (
                  id uuid primary key,
                  presentation_id uuid not null references public.presentations(id) on delete cascade,
                  user_id uuid not null references public.users(id) on delete cascade,
                  current_slide_no int default 1,
                  status text default 'live',
                  created_at timestamptz default now(),
                  updated_at timestamptz default now()
                );
            """)

            cur.execute("""
                create table if not exists public.voice_interaction_logs (
                  id uuid primary key,
                  session_id uuid not null references public.presentation_sessions(id) on delete cascade,
                  user_id uuid not null references public.users(id) on delete cascade,
                  role text not null,
                  message_text text,
                  audio_url text,
                  created_at timestamptz default now()
                );
            """)

            cur.execute("create index if not exists idx_projects_user_id on public.projects(user_id);")
            cur.execute("create index if not exists idx_render_jobs_project_id on public.render_jobs(project_id);")
            cur.execute("create index if not exists idx_generations_project_id on public.generations(project_id);")
            cur.execute("create index if not exists idx_generations_parent_id on public.generations(parent_generation_id);")
            cur.execute("create index if not exists idx_generation_images_generation_id on public.generation_images(generation_id);")
            cur.execute("create index if not exists idx_presentations_project_id on public.presentations(project_id);")
            cur.execute("create index if not exists idx_presentation_slides_presentation_id on public.presentation_slides(presentation_id);")
            cur.execute("create index if not exists idx_presentation_sessions_presentation_id on public.presentation_sessions(presentation_id);")
            cur.execute("create index if not exists idx_voice_logs_session_id on public.voice_interaction_logs(session_id);")


# ============================================================
# APP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client

    print("=== STARTUP ===")
    print("OPENAI:", bool(OPENAI_API_KEY))
    print("DATABASE_URL:", bool(DATABASE_URL))
    print("SECRET_KEY:", bool(SECRET_KEY))

    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)

    if DATABASE_URL:
        create_tables()

    yield


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
# GENERAL HELPERS
# ============================================================

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


def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


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


def image_response_to_urls(image_response: Any, prefix: str = "gen") -> List[str]:
    urls: List[str] = []
    data = getattr(image_response, "data", None) or []
    for item in data:
        item_url = getattr(item, "url", None)
        item_b64 = getattr(item, "b64_json", None)
        if item_url:
            urls.append(item_url)
        elif item_b64:
            urls.append(save_b64_to_media(item_b64, suffix=".png", prefix=prefix))
    return urls


def best_project_name_from_prompt(prompt: str) -> str:
    text = (prompt or "").strip().splitlines()
    if not text:
        return "Untitled Project"
    return text[0][:80].strip() or "Untitled Project"


# ============================================================
# MODELS
# ============================================================

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
    style_profile_name: Optional[str] = None
    seed_note: Optional[str] = None


class VariationInput(BaseModel):
    generation_id: str
    image_index: int = Field(ge=0)
    strength: Literal["subtle", "strong"] = "subtle"
    num_images: int = Field(default=4, ge=1, le=4)
    size: Literal["1024x1024", "1536x1024", "1024x1536"] = "1536x1024"
    remix_prompt: Optional[str] = None


class UpscaleInput(BaseModel):
    generation_id: str
    image_index: int = Field(ge=0)
    size: Literal["1024x1024", "1536x1024", "1024x1536"] = "1536x1024"
    mode: Literal["subtle", "creative"] = "creative"


class EditImageInput(BaseModel):
    generation_id: Optional[str] = None
    source_image_url: Optional[str] = None
    image_index: Optional[int] = None
    prompt: str = Field(min_length=3)
    size: Literal["1024x1024", "1536x1024", "1024x1536"] = "1536x1024"

    @field_validator("source_image_url")
    @classmethod
    def clean_source(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if isinstance(v, str) else v


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
# DB FUNCTIONS
# ============================================================

PROJECT_ALLOWED_FIELDS = {
    "name", "event_type", "status", "brief", "analysis", "concepts", "selected",
    "moodboard", "images", "render3d", "cad", "scene_json", "deliverables",
    "dimensions", "brand_data", "presentation_data"
}



@with_db
def get_user_by_email(cur, email):
    cur.execute("""
        select id, email, password, full_name, role, is_active
        from public.users
        where email = %s
    """, (email,))
    return cur.fetchone()


@with_db
def create_user(cur, email, password, full_name=None):
    uid = str(uuid.uuid4())
    cur.execute(
        """
        insert into users (id, email, password, full_name)
        values (%s, %s, %s, %s)
        """,
        (uid, email, hash_password(password), full_name)
    )
    return uid
@with_db
def create_project(cur, user_id, name=None, event_type=None):
    project_id = str(uuid.uuid4())
    cur.execute("""
        insert into public.projects (id, user_id, name, event_type, status)
        values (%s, %s, %s, %s, %s)
        returning id
    """, (project_id, user_id, name or "Untitled Project", event_type, "draft"))
    return str(cur.fetchone()["id"])


@with_db
def update_project_field(cur, project_id, field, value):
    cur.execute("select * from public.projects where id = %s", (project_id,))
    row = cur.fetchone()
    if not row:
        return None
    project = dict(row)
    for key in [
        "concepts", "selected", "images", "render3d", "scene_json",
        "deliverables", "dimensions", "brand_data", "presentation_data"
    ]:
        project[key] = safe_json(project.get(key))
    return project


@with_db
def update_project_field(cur, project_id: str, field: str, value: Any) -> None:
    if field not in PROJECT_ALLOWED_FIELDS:
        raise HTTPException(status_code=400, detail="Invalid project field")
    db_value = json.dumps(value) if isinstance(value, (dict, list)) else value
    cur.execute(
        f"update public.projects set {field} = %s, updated_at = now() where id = %s",
        (db_value, project_id),
    )


@with_db
def snapshot_project_version(cur, project_id, user_id, note=""):
    project = get_project(project_id)
    if not project:
        return
    cur.execute("""
        select coalesce(max(version_no), 0) + 1 as next_version
        from public.project_versions
        where project_id = %s
    """, (project_id,))
    next_version = int(cur.fetchone()["next_version"])
    cur.execute("""
        insert into public.project_versions (id, project_id, user_id, version_no, snapshot, note)
        values (%s, %s, %s, %s, %s, %s)
    """, (str(uuid.uuid4()), project_id, user_id, next_version, json.dumps(project), note))


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
    cur.execute("""
        select *
        from public.project_comments
        where project_id = %s
        order by created_at desc
    """, (project_id,))
    return [dict(r) for r in cur.fetchall()]


@with_db
def create_render_job(cur, project_id: str, user_id: str, job_type: str, input_json: Dict[str, Any]) -> str:
    job_id = str(uuid.uuid4())
    cur.execute("""
        insert into public.render_jobs (id, project_id, user_id, job_type, status, input_json)
        values (%s, %s, %s, %s, %s, %s)
        returning id
    """, (job_id, project_id, user_id, job_type, "queued", json.dumps(input_json)))
    return str(cur.fetchone()["id"])


@with_db
def update_render_job(cur, job_id: str, status: str, output_json: Any = None, error_text: Optional[str] = None):
    started_at = datetime.datetime.utcnow() if status == "running" else None
    finished_at = datetime.datetime.utcnow() if status in {"done", "failed"} else None

    cur.execute("""
        update public.render_jobs
        set
          status = %s,
          output_json = coalesce(%s, output_json),
          error_text = coalesce(%s, error_text),
          started_at = coalesce(%s, started_at),
          finished_at = coalesce(%s, finished_at)
        where id = %s
    """, (
        status,
        json.dumps(output_json) if output_json is not None else None,
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


@with_db
def add_project_file(cur, project_id: str, user_id: Optional[str], file_type: str, file_name: str, file_url: str, meta: Optional[dict] = None):
    cur.execute("""
        insert into public.project_files (id, project_id, user_id, file_type, file_name, file_url, meta)
        values (%s, %s, %s, %s, %s, %s, %s)
    """, (
        str(uuid.uuid4()), project_id, user_id, file_type, file_name, file_url, json.dumps(meta or {})
    ))


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
    cur.execute("""
        insert into public.generations (
          id, project_id, user_id, parent_generation_id, prompt, expanded_prompt,
          model, size, action_type, source_image_url, settings, status
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        returning id
    """, (
        generation_id, project_id, user_id, parent_generation_id, prompt, expanded_prompt,
        model, size, action_type, source_image_url, json.dumps(settings or {}), "running"
    ))
    return str(cur.fetchone()["id"])


@with_db
def patch_generation_prompt(cur, generation_id: str, expanded_prompt: str):
    cur.execute("""
        update public.generations
        set expanded_prompt = %s
        where id = %s
    """, (expanded_prompt, generation_id))


@with_db
def add_generation_images(cur, generation_id: str, image_urls: List[str]) -> None:
    for idx, url in enumerate(image_urls):
        cur.execute("""
            insert into public.generation_images (id, generation_id, image_index, image_url)
            values (%s, %s, %s, %s)
        """, (str(uuid.uuid4()), generation_id, idx, url))


@with_db
def complete_generation(cur, generation_id: str, image_urls: List[str]) -> None:
    cur.execute("""
        update public.generations
        set status = %s, output_images = %s, completed_at = now()
        where id = %s
    """, ("done", json.dumps(image_urls), generation_id))


@with_db
def fail_generation(cur, generation_id: str, error_text: str) -> None:
    cur.execute("""
        update public.generations
        set status = %s, error_text = %s, completed_at = now()
        where id = %s
    """, ("failed", error_text[:4000], generation_id))


@with_db
def get_generation(cur, generation_id: str) -> Optional[Dict[str, Any]]:
    cur.execute("select * from public.generations where id = %s", (generation_id,))
    row = cur.fetchone()
    if not row:
        return None
    result = dict(row)
    result["settings"] = safe_json(result.get("settings"))
    result["output_images"] = safe_json(result.get("output_images"))
    return result


@with_db
def list_generations(cur, project_id: str) -> List[Dict[str, Any]]:
    cur.execute("""
        select *
        from public.generations
        where project_id = %s
        order by created_at desc
    """, (project_id,))
    rows = []
    for row in cur.fetchall():
        item = dict(row)
        item["settings"] = safe_json(item.get("settings"))
        item["output_images"] = safe_json(item.get("output_images"))
        rows.append(item)
    return rows


@with_db
def create_presentation(cur, project_id: str, user_id: str, title: Optional[str], presenter_name: str, presenter_style: str, voice: str) -> str:
    presentation_id = str(uuid.uuid4())
    cur.execute("""
        insert into public.presentations
        (id, project_id, user_id, title, presenter_name, presenter_style, voice)
        values (%s, %s, %s, %s, %s, %s, %s)
        returning id
    """, (presentation_id, project_id, user_id, title or "AI Presentation", presenter_name, presenter_style, voice))
    return str(cur.fetchone()["id"])


@with_db
def get_presentation(cur, presentation_id: str) -> Optional[Dict[str, Any]]:
    cur.execute("select * from public.presentations where id = %s", (presentation_id,))
    row = cur.fetchone()
    return dict(row) if row else None


@with_db
def add_or_update_slide(cur, presentation_id: str, slide_no: int, title: str, body: str, notes: Optional[str], asset_url: Optional[str]):
    cur.execute("""
        insert into public.presentation_slides
        (id, presentation_id, slide_no, title, body, notes, asset_url)
        values (%s, %s, %s, %s, %s, %s, %s)
        on conflict (presentation_id, slide_no)
        do update set
          title = excluded.title,
          body = excluded.body,
          notes = excluded.notes,
          asset_url = excluded.asset_url
    """, (str(uuid.uuid4()), presentation_id, slide_no, title, body, notes, asset_url))


@with_db
def list_presentation_slides(cur, presentation_id: str) -> List[Dict[str, Any]]:
    cur.execute("""
        select *
        from public.presentation_slides
        where presentation_id = %s
        order by slide_no asc
    """, (presentation_id,))
    return [dict(r) for r in cur.fetchall()]


@with_db
def get_slide(cur, presentation_id: str, slide_no: int) -> Optional[Dict[str, Any]]:
    cur.execute("""
        select *
        from public.presentation_slides
        where presentation_id = %s and slide_no = %s
    """, (presentation_id, slide_no))
    row = cur.fetchone()
    return dict(row) if row else None


@with_db
def create_presentation_session(cur, presentation_id: str, user_id: str, current_slide_no: int = 1) -> str:
    session_id = str(uuid.uuid4())
    cur.execute("""
        insert into public.presentation_sessions
        (id, presentation_id, user_id, current_slide_no, status)
        values (%s, %s, %s, %s, %s)
        returning id
    """, (session_id, presentation_id, user_id, current_slide_no, "live"))
    return str(cur.fetchone()["id"])


@with_db
def get_presentation_session(cur, session_id: str) -> Optional[Dict[str, Any]]:
    cur.execute("select * from public.presentation_sessions where id = %s", (session_id,))
    row = cur.fetchone()
    return dict(row) if row else None


@with_db
def update_session_slide(cur, session_id: str, slide_no: int) -> None:
    cur.execute("""
        update public.presentation_sessions
        set current_slide_no = %s, updated_at = now()
        where id = %s
    """, (slide_no, session_id))


@with_db
def add_voice_log(cur, session_id: str, user_id: str, role: str, message_text: str, audio_url: Optional[str] = None) -> None:
    cur.execute("""
        insert into public.voice_interaction_logs
        (id, session_id, user_id, role, message_text, audio_url)
        values (%s, %s, %s, %s, %s, %s)
    """, (str(uuid.uuid4()), session_id, user_id, role, message_text, audio_url))


@with_db
def get_recent_voice_logs(cur, session_id: str, limit: int = 8) -> List[Dict[str, Any]]:
    cur.execute("""
        select role, message_text
        from public.voice_interaction_logs
        where session_id = %s
        order by created_at desc
        limit %s
    """, (session_id, limit))
    rows = [dict(r) for r in cur.fetchall()]
    rows.reverse()
    return rows


# ============================================================
# AI HELPERS
# ============================================================

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
    cleaned = raw.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        raise HTTPException(status_code=500, detail=f"Model returned invalid JSON: {cleaned[:500]}")


def analyze_brief(brief: str) -> str:
    return llm_text(
        "You are an expert AI Creative Studio planner for exhibitions, activations, road shows, concerts, government events, and stage productions. Be precise, structured, and production-focused.",
        f"""
Analyze this creative brief for production planning.

Need:
1. event objective
2. audience profile
3. event type
4. key deliverables
5. branding requirements
6. probable dimensions if missing
7. missing clarifications
8. production concerns

Brief:
{brief}
"""
    )


def generate_concepts(analysis: str) -> List[Dict[str, Any]]:
    result = llm_json(
        "Return only valid JSON. No markdown.",
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
"""
    )
    if not isinstance(result, list):
        raise HTTPException(status_code=500, detail="Concept generation failed")
    return result


def generate_moodboard(selected_concept: Any) -> str:
    return llm_text(
        "You create polished moodboard directions for premium event, stage, exhibition, and activation concepts.",
        f"""
Create a polished moodboard and visual direction.

Include:
- material palette
- color palette
- lighting language
- finish recommendations
- branding placement logic
- premium visual notes

Concept:
{json.dumps(selected_concept, indent=2)}
"""
    )


def generate_scene_json(selected_concept: Any, brief: str) -> Dict[str, Any]:
    result = llm_json(
        "Return only valid JSON. No markdown.",
        f"""
Create structured JSON for Blender scene generation.

Return valid JSON only.

Requirements:
- use feet
- render ratio 16:9
- stage dimensions
- led wall dimensions
- truss height
- audience rows and cols
- primary and secondary brand colors
- camera target

Brief:
{brief}

Concept:
{json.dumps(selected_concept, indent=2)}
"""
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


def expand_image_prompt(user_prompt: str, style_strength: str, style_profile_name: Optional[str], event_type: Optional[str]) -> str:
    return llm_text(
        "You are a senior visual prompt engineer for high-end image generation. Return only a single polished generation prompt. No bullets.",
        f"""
Rewrite this prompt into a highly visual, production-quality image prompt.

User prompt:
{user_prompt}

Constraints:
- preserve the original intent
- improve composition, materials, lighting, camera framing, realism, and detail
- style strength: {style_strength}
- event type: {event_type or "not specified"}
- style profile: {style_profile_name or "none"}

Return only the final prompt.
""",
        temperature=0.6,
    )


def generate_concept_images(selected_concept: Any) -> List[str]:
    api = require_openai()
    prompt = f"""
High-end event stage or exhibition concept in 16:9 ratio.

Use this concept:
{json.dumps(selected_concept, indent=2)}

Requirements:
- cinematic
- premium
- realistic materials
- polished design presentation
- 16:9 framing
"""
    result = api.images.generate(model=IMAGE_MODEL, prompt=prompt, size="1536x1024")
    return image_response_to_urls(result, prefix="concept")


def generate_images_core(
    prompt: str,
    num_images: int,
    size: str,
    style_strength: str,
    reference_image_url: Optional[str],
    style_profile_name: Optional[str],
    event_type: Optional[str],
) -> Dict[str, Any]:
    api = require_openai()
    expanded_prompt = expand_image_prompt(prompt, style_strength, style_profile_name, event_type)

    if reference_image_url:
        temp_path = local_or_remote_to_temp_file(reference_image_url, suffix=".png")
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
                urls.extend(image_response_to_urls(result, prefix="edit_ref"))
            return {"expanded_prompt": expanded_prompt, "image_urls": urls[:num_images]}
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    urls: List[str] = []
    while len(urls) < num_images:
        result = api.images.generate(model=IMAGE_MODEL, prompt=expanded_prompt, size=size)
        urls.extend(image_response_to_urls(result, prefix="gen"))

    return {"expanded_prompt": expanded_prompt, "image_urls": urls[:num_images]}


def variation_prompt(base_prompt: str, strength: str, remix_prompt: Optional[str]) -> str:
    if remix_prompt:
        return f"""
Create a {strength} variation of the source image while applying this remix direction:
{remix_prompt}

Base intent:
{base_prompt}
""".strip()

    if strength == "subtle":
        return f"""
Create a subtle variation of the source image.
Preserve composition, identity, design language, and overall intent.
Only change styling, micro-details, materials, atmosphere, lighting nuance, and polish.

Base intent:
{base_prompt}
""".strip()

    return f"""
Create a strong variation of the source image.
Keep the same core subject/theme, but explore noticeably different styling,
camera feel, materials, lighting, and visual drama.

Base intent:
{base_prompt}
""".strip()


def upscale_prompt(base_prompt: str, mode: str) -> str:
    if mode == "subtle":
        return f"""
Recreate this image at a more polished, higher-detail, upscale presentation quality.
Stay very close to the source composition and design.

Base intent:
{base_prompt}
""".strip()

    return f"""
Create a more premium, creatively upscaled version of this image with richer details,
better materials, stronger lighting, and a more polished design presentation.

Base intent:
{base_prompt}
""".strip()


# ============================================================
# BLENDER
# ============================================================

def run_blender_multi_angle(scene_json: Dict[str, Any], job_id: str, width: int, height: int, project_id: str, user_id: str) -> None:
    try:
        update_render_job(job_id, "running")

        job_dir = RENDER_OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        scene_json_path = job_dir / "scene.json"
        with open(scene_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scene": scene_json,
                    "render": {"width": width, "height": height, "output_dir": str(job_dir)},
                },
                f,
                indent=2,
            )

        cmd = [BLENDER_PATH, "-b", "-P", BLENDER_SCRIPT, "--", str(scene_json_path)]
        completed = subprocess.run(cmd, capture_output=True, text=True)

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


# ============================================================
# AUDIO / PRESENTATION HELPERS
# ============================================================

def tts_to_file(text: str, voice: str = "marin") -> str:
    api = require_openai()
    out_dir = MEDIA_DIR / "presenter_audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uuid.uuid4().hex}.mp3"

    speech = api.audio.speech.create(model=TTS_MODEL, voice=voice, input=text)
    speech.stream_to_file(str(out_path))
    return build_public_url(out_path)


def transcribe_upload(upload: UploadFile) -> str:
    api = require_openai()
    suffix = Path(upload.filename or "audio.webm").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = upload.file.read()
        tmp.write(content)
        tmp.close()
        with open(tmp.name, "rb") as f:
            transcript = api.audio.transcriptions.create(model=STT_MODEL, file=f)
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
        "You are a live AI presenter. Speak naturally and clearly. Be engaging, concise, and slide-aware. Do not read bullet points robotically. Present like a polished human presenter.",
        f"""
Use this context:
{context}

Narrate this slide for a live audience.

Rules:
- language: {language}
- audience mode: {audience_mode}
- 90 to 160 words
- explain the key message clearly
- sound live, not robotic
- end with a smooth transition or invite for question if suitable

Return narration text only.
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
    history_text = "\n".join([f"{item['role']}: {item['message_text']}" for item in history])

    return llm_text(
        "You are a live AI presenter answering audience questions during a presentation. You know the full project context, but you are especially aware of the current slide. Be confident, clear, and presentation-friendly.",
        f"""
Presentation title: {presentation.get('title')}
Presenter style: {presentation.get('presenter_style')}
Language: {language}

Current slide number: {slide.get('slide_no')}
Current slide title: {slide.get('title')}
Current slide body:
{slide.get('body')}

Current slide notes:
{slide.get('notes') or ''}

Recent interaction history:
{history_text}

Audience question:
{user_question}

Rules:
- answer in a natural spoken way
- 40 to 140 words
- use current slide context first
- simplify if needed

Return answer text only.
""",
        temperature=0.5,
    )


def run_presentation_command(session_id: str, command: str) -> Dict[str, Any]:
    session = get_presentation_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    presentation_id = session["presentation_id"]
    slides = list_presentation_slides(presentation_id)
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
        requested = int(m.group(1))
        if 1 <= requested <= total:
            update_session_slide(session_id, requested)
            return {"action": "slide_changed", "slide_no": requested}

    return {"action": "no_change", "slide_no": current}


def build_presenter_realtime_instructions(
    presentation_id: str,
    slide_no: int,
    audience_mode: str = "client",
    language: str = "en",
) -> str:
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

Behavior rules:
- Speak naturally like a polished live presenter.
- Explain the current slide clearly and persuasively.
- If the audience interrupts, answer based on the current slide first.
- If asked to simplify, explain in simpler language.
- Keep answers concise and spoken.
- Do not pretend to physically change slides yourself.
""".strip()


def create_or_get_live_presentation_session(
    presentation_id: str,
    user_id: str,
    current_slide_no: int = 1,
    existing_session_id: Optional[str] = None,
) -> str:
    if existing_session_id:
        existing = get_presentation_session(existing_session_id)
        if existing:
            return existing_session_id
    return create_presentation_session(presentation_id, user_id, current_slide_no)


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"message": "AI Creative Studio API is running", "time": now_iso()}


@app.post("/signup")
def signup(data: UserInput):
    try:
        user_id = create_user(data.email, data.password, data.full_name)
        token = create_token(user_id)
        return {
            "message": "User created",
            "user_id": user_id,
            "token": token
        }
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

@app.post("/login")
def login(payload: UserInput):
    user = get_user_by_email(payload.email)
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    if not verify_password(payload.password, user["password"]):
        raise HTTPException(status_code=400, detail="Wrong password")
    token = create_token(str(user["id"]))
    return {"token": token, "user_id": str(user["id"])}


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


@app.post("/comment")
def create_comment(payload: CommentInput, user_id: str = Depends(get_current_user_id)):
    comment_id = add_comment(payload.project_id, user_id, payload.section, payload.comment_text)
    return {"message": "Comment added", "comment_id": comment_id}


@app.get("/comments/{project_id}")
def list_comments(project_id: str, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    return {"comments": get_comments(project_id)}


@app.post("/run")
def run_pipeline(payload: RunInput, user_id: str = Depends(get_current_user_id)):
    try:
        project_id = payload.project_id or create_project(
            user_id=user_id,
            name=payload.name or best_project_name_from_prompt(payload.text),
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
            snapshot_project_version(project_id, user_id, note="Concept options generated")
            return {"stage": "concepts_ready", "project_id": project_id, "concepts": concepts}

        if not project.get("selected"):
            return {
                "stage": "awaiting_concept_selection",
                "project_id": project_id,
                "options": project["concepts"],
            }

        if not project.get("moodboard"):
            moodboard = generate_moodboard(project["selected"])
            update_project_field(project_id, "moodboard", moodboard)
            update_project_field(project_id, "status", "moodboard_ready")
            snapshot_project_version(project_id, user_id, note="Moodboard generated")
            return {"stage": "moodboard_ready", "project_id": project_id, "moodboard": moodboard}

        if not project.get("images"):
            images = generate_concept_images(project["selected"])
            update_project_field(project_id, "images", images)
            update_project_field(project_id, "status", "concept_images_ready")
            snapshot_project_version(project_id, user_id, note="Concept images generated")
            return {"stage": "concept_images_ready", "project_id": project_id, "images": images}

        return {
            "stage": "ready_for_multi_angle_3d",
            "project_id": project_id,
            "message": "Concept approved. Ready for multi-angle render.",
        }
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

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


@app.post("/generate-multi-angle")
def generate_multi_angle(payload: Generate3DInput, background_tasks: BackgroundTasks, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
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


@app.post("/generate")
def generate_images(payload: GenerateImageInput, user_id: str = Depends(get_current_user_id)):
    project_id = payload.project_id or create_project(
        user_id=user_id,
        name=payload.project_name or best_project_name_from_prompt(payload.prompt),
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
        settings={
            "num_images": payload.num_images,
            "style_strength": payload.style_strength,
            "style_profile_name": payload.style_profile_name,
            "seed_note": payload.seed_note,
        },
        source_image_url=payload.reference_image_url,
    )

    try:
        result = generate_images_core(
            prompt=payload.prompt,
            num_images=payload.num_images,
            size=payload.size,
            style_strength=payload.style_strength,
            reference_image_url=payload.reference_image_url,
            style_profile_name=payload.style_profile_name,
            event_type=payload.event_type,
        )

        expanded_prompt = result["expanded_prompt"]
        image_urls = result["image_urls"]

        patch_generation_prompt(generation_id, expanded_prompt)
        add_generation_images(generation_id, image_urls)
        complete_generation(generation_id, image_urls)

        add_project_file(
            project_id=project_id,
            user_id=user_id,
            file_type="generated_images",
            file_name=f"generation_{generation_id}",
            file_url=image_urls[0],
            meta={"generation_id": generation_id, "count": len(image_urls)},
        )

        snapshot_project_version(project_id, user_id, note="Generated image set")

        return {
            "message": "Images generated",
            "project_id": project_id,
            "generation_id": generation_id,
            "expanded_prompt": expanded_prompt,
            "images": image_urls,
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
        settings={
            "strength": payload.strength,
            "num_images": payload.num_images,
            "remix_prompt": payload.remix_prompt,
            "source_index": payload.image_index,
        },
    )

    try:
        temp_path = local_or_remote_to_temp_file(source_image_url, suffix=".png")
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
                urls.extend(image_response_to_urls(result, prefix="variation"))
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        urls = urls[:payload.num_images]
        add_generation_images(generation_id, urls)
        complete_generation(generation_id, urls)
        snapshot_project_version(source["project_id"], user_id, note="Created image variation")

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


@app.post("/upscale")
def upscale_image(payload: UpscaleInput, user_id: str = Depends(get_current_user_id)):
    source = get_generation(payload.generation_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source generation not found")

    output_images = source.get("output_images") or []
    if payload.image_index >= len(output_images):
        raise HTTPException(status_code=400, detail="Invalid image index")

    source_image_url = output_images[payload.image_index]
    new_prompt = upscale_prompt(source.get("expanded_prompt") or source.get("prompt") or "", payload.mode)

    generation_id = create_generation(
        project_id=source["project_id"],
        user_id=user_id,
        prompt=source.get("prompt") or "",
        expanded_prompt=new_prompt,
        model=IMAGE_MODEL,
        size=payload.size,
        action_type="upscale",
        parent_generation_id=source["id"],
        source_image_url=source_image_url,
        settings={"mode": payload.mode, "source_index": payload.image_index},
    )

    try:
        temp_path = local_or_remote_to_temp_file(source_image_url, suffix=".png")
        try:
            api = require_openai()
            with open(temp_path, "rb") as image_file:
                result = api.images.edit(
                    model=IMAGE_MODEL,
                    image=image_file,
                    prompt=new_prompt,
                    size=payload.size,
                )
            urls = image_response_to_urls(result, prefix="upscale")
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        if not urls:
            raise HTTPException(status_code=500, detail="No image returned")

        add_generation_images(generation_id, urls[:1])
        complete_generation(generation_id, urls[:1])
        snapshot_project_version(source["project_id"], user_id, note="Upscaled selected image")

        return {
            "message": "Upscale created",
            "project_id": source["project_id"],
            "generation_id": generation_id,
            "image": urls[0],
            "expanded_prompt": new_prompt,
        }
    except Exception as e:
        fail_generation(generation_id, str(e))
        raise HTTPException(status_code=500, detail=f"Upscale failed: {str(e)}")


@app.post("/edit-image")
def edit_image(payload: EditImageInput, user_id: str = Depends(get_current_user_id)):
    source_image_url: Optional[str] = payload.source_image_url
    project_id: Optional[str] = None
    parent_generation_id: Optional[str] = None

    if payload.generation_id:
        source = get_generation(payload.generation_id)
        if not source:
            raise HTTPException(status_code=404, detail="Generation not found")
        output_images = source.get("output_images") or []
        idx = payload.image_index or 0
        if idx >= len(output_images):
            raise HTTPException(status_code=400, detail="Invalid image index")
        source_image_url = output_images[idx]
        project_id = source["project_id"]
        parent_generation_id = source["id"]

    if not source_image_url:
        raise HTTPException(status_code=400, detail="source_image_url or generation_id is required")

    if not project_id:
        project_id = create_project(user_id=user_id, name=best_project_name_from_prompt(payload.prompt), event_type=None)

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
        temp_path = local_or_remote_to_temp_file(source_image_url, suffix=".png")
        try:
            api = require_openai()
            with open(temp_path, "rb") as image_file:
                result = api.images.edit(
                    model=IMAGE_MODEL,
                    image=image_file,
                    prompt=payload.prompt,
                    size=payload.size,
                )
            urls = image_response_to_urls(result, prefix="edit")
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        if not urls:
            raise HTTPException(status_code=500, detail="No image returned")

        add_generation_images(generation_id, urls[:1])
        complete_generation(generation_id, urls[:1])
        snapshot_project_version(project_id, user_id, note="Edited image from source")

        return {
            "message": "Image edited",
            "project_id": project_id,
            "generation_id": generation_id,
            "image": urls[0],
        }
    except Exception as e:
        fail_generation(generation_id, str(e))
        raise HTTPException(status_code=500, detail=f"Edit failed: {str(e)}")


@app.get("/generations/{project_id}")
def get_project_generations(project_id: str, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    return {"project_id": project_id, "generations": list_generations(project_id)}


@app.get("/generation/{generation_id}")
def generation_detail(generation_id: str, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    generation = get_generation(generation_id)
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")
    return generation


@app.post("/presentation/create")
def presentation_create(payload: PresentationCreateInput, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    presentation_id = create_presentation(
        project_id=payload.project_id,
        user_id=user_id,
        title=payload.title or f"{project.get('name', 'Project')} Presentation",
        presenter_name=payload.presenter_name,
        presenter_style=payload.presenter_style,
        voice=payload.voice,
    )
    snapshot_project_version(payload.project_id, user_id, note="Presentation created")
    return {"message": "Presentation created", "presentation_id": presentation_id}


@app.post("/presentation/slide")
def presentation_slide_upsert(payload: PresentationSlideInput, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    presentation = get_presentation(payload.presentation_id)
    if not presentation:
        raise HTTPException(status_code=404, detail="Presentation not found")

    add_or_update_slide(
        presentation_id=payload.presentation_id,
        slide_no=payload.slide_no,
        title=payload.title,
        body=payload.body,
        notes=payload.notes,
        asset_url=payload.asset_url,
    )
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

    session_id = create_presentation_session(presentation_id=presentation_id, user_id=user_id, current_slide_no=1)
    return {"message": "Live presentation session started", "session_id": session_id, "current_slide_no": 1}


@app.post("/presentation/narrate-slide")
def presentation_narrate_slide(payload: PresentationNarrateInput, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    presentation = get_presentation(payload.presentation_id)
    if not presentation:
        raise HTTPException(status_code=404, detail="Presentation not found")

    narration = generate_slide_narration(
        presentation_id=payload.presentation_id,
        slide_no=payload.slide_no,
        language=payload.language,
        audience_mode=payload.audience_mode,
    )
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

    return {
        "current_slide_no": session["current_slide_no"],
        "answer": answer,
        "audio_url": audio_url,
    }


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
    addVoiceRole = "assistant"
    add_voice_log(session_id, user_id, addVoiceRole, answer, audio_url)

    return {
        "transcript": transcript,
        "answer": answer,
        "audio_url": audio_url,
        "current_slide_no": session["current_slide_no"],
    }


@app.get("/presentation/session/{session_id}")
def presentation_session_detail(session_id: str, user_id: str = Depends(get_current_user_id)):
    _ = user_id
    session = get_presentation_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    slide = get_slide(session["presentation_id"], session["current_slide_no"])
    logs = get_recent_voice_logs(session_id, limit=20)

    return {"session": session, "current_slide": slide, "logs": logs}


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

    session_id = create_or_get_live_presentation_session(
        presentation_id=presentation_id,
        user_id=user_id,
        current_slide_no=slide_no,
    )

    instructions = build_presenter_realtime_instructions(
        presentation_id=presentation_id,
        slide_no=slide_no,
        audience_mode=audience_mode,
        language=language,
    )

    session_config = {
        "type": "realtime",
        "model": REALTIME_MODEL,
        "audio": {"output": {"voice": voice}},
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
            raise HTTPException(status_code=500, detail=f"Realtime session failed: {r.text[:1000]}")

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
        presentation_id=payload.presentation_id,
        slide_no=payload.slide_no,
        audience_mode=payload.audience_mode,
        language=payload.language,
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


@app.post("/presentation/realtime/command")
def presentation_realtime_command(payload: PresentationCommandInput, user_id: str = Depends(get_current_user_id)):
    result = run_presentation_command(payload.session_id, payload.command)
    add_voice_log(payload.session_id, user_id, "user", payload.command, None)
    return result


@app.get("/presentation/realtime/token")
def presentation_realtime_token(
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

    instructions = build_presenter_realtime_instructions(
        presentation_id=presentation_id,
        slide_no=slide_no,
        audience_mode=audience_mode,
        language=language,
    )

    payload = {
        "session": {
            "type": "realtime",
            "model": REALTIME_MODEL,
            "audio": {"output": {"voice": voice}},
            "instructions": instructions,
        }
    }

    r = requests.post(
        "https://api.openai.com/v1/realtime/client_secrets",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )

    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=r.text[:1000])

    return r.json()
