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
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


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
# HELPERS
# ============================================================

def require_openai() -> OpenAI:
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI not configured")
    return client


def require_db() -> str:
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="Database not connected")
    return DATABASE_URL


def get_conn():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row, autocommit=True)


def with_db(fn):
    def wrapper(*args, **kwargs):
        require_db()
        with get_conn() as conn:
            with conn.cursor() as cur:
                return fn(cur, *args, **kwargs)
    return wrapper


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
# DB SETUP
# ============================================================

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


# ============================================================
# DB FUNCTIONS
# ============================================================

PROJECT_ALLOWED_FIELDS = {
    "name", "event_type", "status", "brief", "analysis", "concepts", "selected",
    "moodboard", "images", "render3d", "cad", "scene_json", "deliverables",
    "dimensions", "brand_data", "presentation_data"
}


@with_db
def get_user_by_email(cur, email: str):
    cur.execute("""
        select id, email, password, full_name, role, is_active
        from public.users
        where email = %s
    """, (email,))
    return cur.fetchone()


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
def snapshot_project_version(cur, project_id: str, user_id: str, note: str = "") -> None:
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
        on conflict