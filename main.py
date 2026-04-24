
from __future__ import annotations

import base64
import io
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from openai import OpenAI
from passlib.context import CryptContext
from psycopg.rows import dict_row
from pydantic import BaseModel, Field, field_validator
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

load_dotenv()

APP_NAME = os.getenv("APP_TITLE", "Creative Brief to Concept & Execution API").strip() or "Creative Brief to Concept & Execution API"
APP_VERSION = "1.0.0"

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", os.getenv("TEXT_MODEL", "gpt-4o-mini")).strip() or "gpt-4o-mini"
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts"
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe").strip() or "gpt-4o-mini-transcribe"
TTS_VOICE = os.getenv("TTS_VOICE", "coral").strip() or "coral"
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1").strip() or "gpt-image-1"
IMAGE_QUALITY = os.getenv("IMAGE_QUALITY", "high").strip() or "high"

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000").strip() or "http://localhost:3000"
VISUAL_ASPECT_RATIO = os.getenv("VISUAL_ASPECT_RATIO", "16:9").strip() or "16:9"
VISUAL_PREVIEW_SIZE = os.getenv("VISUAL_PREVIEW_SIZE", "1280x720").strip() or "1280x720"
VISUAL_MASTER_SIZE = os.getenv("VISUAL_MASTER_SIZE", "1920x1080").strip() or "1920x1080"
VISUAL_PRINT_SIZE = os.getenv("VISUAL_PRINT_SIZE", "3840x2160").strip() or "3840x2160"

MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "./media")).resolve()
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads")).resolve()
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", "./exports")).resolve()
RENDER_OUTPUT_DIR = Path(os.getenv("RENDER_OUTPUT_DIR", "./renders")).resolve()
VOICE_DIR = Path(os.getenv("VOICE_DIR", str(MEDIA_DIR / "voice"))).resolve()

for _path in (MEDIA_DIR, UPLOAD_DIR, EXPORT_DIR, RENDER_OUTPUT_DIR, VOICE_DIR):
    _path.mkdir(parents=True, exist_ok=True)

SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-render").strip() or "change-me-in-render"
ACCESS_TOKEN_HOURS = int(os.getenv("ACCESS_TOKEN_HOURS", "24") or "24")
ALGORITHM = "HS256"

SHOW_GATEWAY_URL = os.getenv("SHOW_GATEWAY_URL", "").strip()
COMPANION_BASE_URL = os.getenv("COMPANION_BASE_URL", "").strip()
RESOLUME_BASE_URL = os.getenv("RESOLUME_BASE_URL", "").strip()
QLAB_OSC_IP = os.getenv("QLAB_OSC_IP", "").strip()
QLAB_OSC_PORT = int(os.getenv("QLAB_OSC_PORT", "0") or 0)
EOS_OSC_IP = os.getenv("EOS_OSC_IP", "").strip()
EOS_OSC_PORT = int(os.getenv("EOS_OSC_PORT", "0") or 0)

EMAIL_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.I)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
auth_scheme = HTTPBearer(auto_error=False)

def _split_origins(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]

ALLOWED_ORIGINS = _split_origins(
    os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,https://briefly-sparkle.lovable.app",
    )
)

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https:\/\/([a-zA-Z0-9-]+\.)?lovable\.(app|dev)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/exports", StaticFiles(directory=str(EXPORT_DIR)), name="exports")
app.mount("/renders", StaticFiles(directory=str(RENDER_OUTPUT_DIR)), name="renders")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return now_utc().isoformat()


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def load_json(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def get_conn():
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL is not configured")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def ensure_tables() -> None:
    if not DATABASE_URL:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                create extension if not exists pgcrypto;

                create table if not exists public.users (
                    id uuid primary key default gen_random_uuid(),
                    email text unique not null,
                    password_hash text not null,
                    full_name text,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                );

                create table if not exists public.projects (
                    id uuid primary key default gen_random_uuid(),
                    user_id uuid not null references public.users(id) on delete cascade,
                    name text not null,
                    brief text,
                    event_type text,
                    style_direction text,
                    status text not null default 'draft',
                    analysis text,
                    concepts text,
                    selected text,
                    comments text,
                    sound_data text,
                    lighting_data text,
                    showrunner_data text,
                    department_outputs text,
                    orchestration_data text,
                    visual_policy text,
                    element_sheet text,
                    images text,
                    render3d text,
                    scene_json text,
                    moodboard text,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                );

                create table if not exists public.project_comments (
                    id uuid primary key default gen_random_uuid(),
                    project_id uuid not null references public.projects(id) on delete cascade,
                    user_id uuid not null references public.users(id) on delete cascade,
                    section text not null,
                    comment_text text not null,
                    created_at timestamptz not null default now()
                );

                create table if not exists public.project_assets (
                    id uuid primary key default gen_random_uuid(),
                    project_id uuid not null references public.projects(id) on delete cascade,
                    user_id uuid not null references public.users(id) on delete cascade,
                    asset_type text not null,
                    section text,
                    job_kind text,
                    title text not null,
                    prompt text,
                    status text not null default 'queued',
                    preview_url text,
                    master_url text,
                    print_url text,
                    source_file_url text,
                    meta text,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                );

                create table if not exists public.agent_jobs (
                    id uuid primary key default gen_random_uuid(),
                    project_id uuid not null references public.projects(id) on delete cascade,
                    user_id uuid not null references public.users(id) on delete cascade,
                    agent_type text not null,
                    job_type text not null,
                    title text not null,
                    status text not null default 'queued',
                    priority int not null default 5,
                    progress int not null default 0,
                    input_data text,
                    output_data text,
                    error_text text,
                    parent_job_id uuid,
                    started_at timestamptz,
                    completed_at timestamptz,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                );

                create table if not exists public.project_activity_logs (
                    id uuid primary key default gen_random_uuid(),
                    project_id uuid not null references public.projects(id) on delete cascade,
                    user_id uuid references public.users(id) on delete cascade,
                    activity_type text not null,
                    title text not null,
                    detail text,
                    meta text,
                    created_at timestamptz not null default now()
                );

                create table if not exists public.voice_sessions (
                    id uuid primary key default gen_random_uuid(),
                    user_id uuid not null references public.users(id) on delete cascade,
                    project_id uuid references public.projects(id) on delete cascade,
                    title text,
                    system_prompt text,
                    voice text,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                );

                create table if not exists public.voice_messages (
                    id uuid primary key default gen_random_uuid(),
                    session_id uuid not null references public.voice_sessions(id) on delete cascade,
                    role text not null,
                    content text not null,
                    transcript text,
                    audio_url text,
                    meta text,
                    created_at timestamptz not null default now()
                );

                create index if not exists idx_projects_user_id on public.projects(user_id);
                create index if not exists idx_project_assets_project on public.project_assets(project_id);
                create index if not exists idx_agent_jobs_project on public.agent_jobs(project_id);
                create index if not exists idx_activity_project on public.project_activity_logs(project_id);
                """
            )
        conn.commit()


@app.on_event("startup")
def startup() -> None:
    ensure_tables()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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


def ensure_uuid(value: str, label: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except Exception:
        raise HTTPException(status_code=422, detail=f"{label} must be a valid UUID")


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(user_id: str) -> str:
    expire = now_utc() + timedelta(hours=ACCESS_TOKEN_HOURS)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("select * from public.users where id = %s limit 1", (user_id,))
            return cur.fetchone()


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("select * from public.users where lower(email) = lower(%s) limit 1", (email,))
            return cur.fetchone()


def create_user(email: str, password: str, full_name: Optional[str]) -> Dict[str, Any]:
    existing = get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.users (email, password_hash, full_name)
                values (%s, %s, %s)
                returning *
                """,
                (email, hash_password(password), full_name),
            )
            row = cur.fetchone()
        conn.commit()
        return row


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(auth_scheme)) -> Dict[str, Any]:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(user)


def _row_project(row: Dict[str, Any]) -> Dict[str, Any]:
    if not row:
        return row
    item = dict(row)
    for key in (
        "analysis", "concepts", "selected", "comments", "sound_data", "lighting_data",
        "showrunner_data", "department_outputs", "orchestration_data", "visual_policy",
        "element_sheet", "images", "render3d", "scene_json"
    ):
        item[key] = load_json(item.get(key))
    return item


def create_project(user_id: str, name: str, brief: Optional[str], event_type: Optional[str], style_direction: Optional[str]) -> Dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.projects (user_id, name, brief, event_type, style_direction, visual_policy)
                values (%s, %s, %s, %s, %s, %s)
                returning *
                """,
                (user_id, name, brief, event_type, style_direction, dump_json(default_visual_policy())),
            )
            row = cur.fetchone()
        conn.commit()
        return _row_project(row)


def list_projects(user_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select * from public.projects where user_id = %s order by updated_at desc, created_at desc",
                (user_id,),
            )
            return [_row_project(r) for r in cur.fetchall()]


def get_project_by_id(project_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            if user_id:
                cur.execute(
                    "select * from public.projects where id = %s and user_id = %s limit 1",
                    (project_id, user_id),
                )
            else:
                cur.execute("select * from public.projects where id = %s limit 1", (project_id,))
            row = cur.fetchone()
            return _row_project(row) if row else None


def update_project_fields(project_id: str, user_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "name", "brief", "event_type", "style_direction", "status", "analysis", "concepts", "selected",
        "sound_data", "lighting_data", "showrunner_data", "department_outputs", "orchestration_data",
        "visual_policy", "element_sheet", "images", "render3d", "scene_json", "moodboard", "comments"
    }
    clean = {k: v for k, v in values.items() if k in allowed}
    if not clean:
        project = get_project_by_id(project_id, user_id=user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project

    params: List[Any] = []
    assignments: List[str] = []
    json_keys = {
        "analysis", "concepts", "selected", "sound_data", "lighting_data", "showrunner_data",
        "department_outputs", "orchestration_data", "visual_policy", "element_sheet", "images",
        "render3d", "scene_json", "comments"
    }
    for key, value in clean.items():
        assignments.append(f"{key} = %s")
        params.append(dump_json(value) if key in json_keys else value)
    assignments.append("updated_at = now()")
    params.extend([project_id, user_id])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"update public.projects set {', '.join(assignments)} where id = %s and user_id = %s returning *",
                params,
            )
            row = cur.fetchone()
        conn.commit()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return _row_project(row)


def snapshot_project_version(project_id: str, user_id: str, title: str) -> None:
    add_project_activity(project_id, user_id, "project.snapshot", title)


def add_comment(project_id: str, user_id: str, section: str, comment_text: str) -> Dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.project_comments (project_id, user_id, section, comment_text)
                values (%s, %s, %s, %s)
                returning *
                """,
                (project_id, user_id, section, comment_text),
            )
            row = cur.fetchone()
        conn.commit()
        return dict(row)


def list_comments(project_id: str, user_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select c.*, u.full_name, u.email
                from public.project_comments c
                join public.users u on u.id = c.user_id
                where c.project_id = %s and c.user_id = %s
                order by c.created_at desc
                """,
                (project_id, user_id),
            )
            return [dict(r) for r in cur.fetchall()]


def add_project_activity(project_id: str, user_id: Optional[str], activity_type: str, title: str, detail: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.project_activity_logs (project_id, user_id, activity_type, title, detail, meta)
                values (%s, %s, %s, %s, %s, %s)
                returning *
                """,
                (project_id, user_id, activity_type, title, detail, dump_json(meta or {})),
            )
            row = cur.fetchone()
        conn.commit()
        item = dict(row)
        item["meta"] = load_json(item.get("meta"), {})
        return item


def list_project_activity(project_id: str, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select * from public.project_activity_logs
                where project_id = %s and (user_id = %s or user_id is null)
                order by created_at desc
                limit %s
                """,
                (project_id, user_id, max(1, min(limit, 500))),
            )
            rows = []
            for row in cur.fetchall():
                item = dict(row)
                item["meta"] = load_json(item.get("meta"), {})
                rows.append(item)
            return rows


def create_project_asset(
    project_id: str,
    user_id: str,
    asset_type: str,
    title: str,
    prompt: str = "",
    section: Optional[str] = None,
    job_kind: Optional[str] = None,
    asset_category: Optional[str] = None,
    status: str = "queued",
    preview_url: Optional[str] = None,
    master_url: Optional[str] = None,
    print_url: Optional[str] = None,
    source_file_url: Optional[str] = None,
    file_path: Optional[str] = None,
    mime_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    version: int = 1,
) -> Dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.project_assets (
                    project_id, user_id, asset_type, asset_category, section, job_kind,
                    title, prompt, status,
                    preview_url, master_url, print_url, source_file_url,
                    file_path, mime_type, metadata, version
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                returning *
                """,
                (
                    project_id, user_id, asset_type, asset_category, section, job_kind,
                    title, prompt, status,
                    preview_url, master_url, print_url, source_file_url,
                    file_path, mime_type, dump_json(metadata or {}), version
                ),
            )
            row = cur.fetchone()
        conn.commit()

    item = dict(row)
    item["metadata"] = load_json(item.get("metadata"), {})
    return item


def update_project_asset(asset_id: str, user_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "asset_type", "asset_category", "section", "job_kind", "title", "prompt", "status",
        "preview_url", "master_url", "print_url", "source_file_url",
        "file_path", "mime_type", "metadata", "version"
    }
    clean = {k: v for k, v in values.items() if k in allowed}

    if not clean:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("select * from public.project_assets where id = %s and user_id = %s", (asset_id, user_id))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Asset not found")
        item = dict(row)
        item["metadata"] = load_json(item.get("metadata"), {})
        return item

    assignments = []
    params = []
    for key, value in clean.items():
        assignments.append(f"{key} = %s")
        params.append(dump_json(value) if key == "metadata" else value)

    assignments.append("updated_at = now()")
    params.extend([asset_id, user_id])

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"update public.project_assets set {', '.join(assignments)} where id = %s and user_id = %s returning *",
                params,
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Asset not found")

    item = dict(row)
    item["metadata"] = load_json(item.get("metadata"), {})
    return item


def update_project_asset(asset_id: str, user_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {"asset_type", "section", "job_kind", "title", "prompt", "status", "preview_url", "master_url", "print_url", "source_file_url", "meta"}
    clean = {k: v for k, v in values.items() if k in allowed}
    if not clean:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("select * from public.project_assets where id = %s and user_id = %s", (asset_id, user_id))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Asset not found")
                item = dict(row)
                item["meta"] = load_json(item.get("meta"), {})
                return item
    assignments, params = [], []
    for key, value in clean.items():
        assignments.append(f"{key} = %s")
        params.append(dump_json(value) if key == "meta" else value)
    assignments.append("updated_at = now()")
    params.extend([asset_id, user_id])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"update public.project_assets set {', '.join(assignments)} where id = %s and user_id = %s returning *",
                params,
            )
            row = cur.fetchone()
        conn.commit()
        if not row:
            raise HTTPException(status_code=404, detail="Asset not found")
        item = dict(row)
        item["meta"] = load_json(item.get("meta"), {})
        return item


def list_project_assets(project_id: str, user_id: str, section: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            if section:
                cur.execute(
                    "select * from public.project_assets where project_id = %s and user_id = %s and section = %s order by created_at desc",
                    (project_id, user_id, section),
                )
            else:
                cur.execute(
                    "select * from public.project_assets where project_id = %s and user_id = %s order by created_at desc",
                    (project_id, user_id),
                )
            items = []
            for row in cur.fetchall():
                item = dict(row)
                item["meta"] = load_json(item.get("meta"), {})
                items.append(item)
            return items


def create_agent_job(project_id: str, user_id: str, agent_type: str, job_type: str, title: str, priority: int = 5, input_data: Optional[Dict[str, Any]] = None, status: str = "queued") -> Dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.agent_jobs (project_id, user_id, agent_type, job_type, title, status, priority, input_data)
                values (%s, %s, %s, %s, %s, %s, %s, %s)
                returning *
                """,
                (project_id, user_id, agent_type, job_type, title, status, priority, dump_json(input_data or {})),
            )
            row = cur.fetchone()
        conn.commit()
        item = dict(row)
        item["input_data"] = load_json(item.get("input_data"), {})
        item["output_data"] = load_json(item.get("output_data"), {})
        return item


def update_agent_job(job_id: str, user_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {"status", "priority", "progress", "input_data", "output_data", "error_text", "started_at", "completed_at"}
    clean = {k: v for k, v in values.items() if k in allowed}
    assignments, params = [], []
    for key, value in clean.items():
        assignments.append(f"{key} = %s")
        if key in {"input_data", "output_data"}:
            params.append(dump_json(value))
        else:
            params.append(value)
    assignments.append("updated_at = now()")
    params.extend([job_id, user_id])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"update public.agent_jobs set {', '.join(assignments)} where id = %s and user_id = %s returning *",
                params,
            )
            row = cur.fetchone()
        conn.commit()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        item = dict(row)
        item["input_data"] = load_json(item.get("input_data"), {})
        item["output_data"] = load_json(item.get("output_data"), {})
        return item


def list_agent_jobs(project_id: str, user_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select * from public.agent_jobs where project_id = %s and user_id = %s order by created_at desc",
                (project_id, user_id),
            )
            items = []
            for row in cur.fetchall():
                item = dict(row)
                item["input_data"] = load_json(item.get("input_data"), {})
                item["output_data"] = load_json(item.get("output_data"), {})
                items.append(item)
            return items


def queue_agent_job_with_activity(project_id: str, user_id: str, agent_type: str, job_type: str, title: str, priority: int = 5, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    job = create_agent_job(project_id, user_id, agent_type, job_type, title, priority=priority, input_data=input_data or {})
    add_project_activity(project_id, user_id, "job.queued", title, detail=f"{agent_type} queued", meta={"job_id": job["id"], "job_type": job_type})
    return job


def create_voice_session(user_id: str, project_id: Optional[str], title: Optional[str], system_prompt: Optional[str], voice: Optional[str]) -> Dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.voice_sessions (user_id, project_id, title, system_prompt, voice)
                values (%s, %s, %s, %s, %s)
                returning *
                """,
                (user_id, project_id, title, system_prompt, voice),
            )
            row = cur.fetchone()
        conn.commit()
        return dict(row)


def get_voice_session_by_id(session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select * from public.voice_sessions where id = %s and user_id = %s limit 1",
                (session_id, user_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def list_voice_sessions(user_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("select * from public.voice_sessions where user_id = %s order by updated_at desc", (user_id,))
            return [dict(r) for r in cur.fetchall()]


def add_voice_message(session_id: str, role: str, content: str, transcript: Optional[str] = None, audio_url: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.voice_messages (session_id, role, content, transcript, audio_url, meta)
                values (%s, %s, %s, %s, %s, %s)
                returning *
                """,
                (session_id, role, content, transcript, audio_url, dump_json(meta or {})),
            )
            row = cur.fetchone()
            cur.execute("update public.voice_sessions set updated_at = now() where id = %s", (session_id,))
        conn.commit()
        item = dict(row)
        item["meta"] = load_json(item.get("meta"), {})
        return item


def get_voice_messages(session_id: str, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    session = get_voice_session_by_id(session_id, user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Voice session not found")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select * from public.voice_messages where session_id = %s order by created_at asc limit %s",
                (session_id, max(1, min(limit, 100))),
            )
            items = []
            for row in cur.fetchall():
                item = dict(row)
                item["meta"] = load_json(item.get("meta"), {})
                items.append(item)
            return items


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

EVENT_TYPE_BUDGETS = {
    "conference": (800000, 1800000, 4200000),
    "award show": (1200000, 2600000, 6500000),
    "brand launch": (900000, 2200000, 5500000),
    "wedding": (700000, 1600000, 4000000),
    "concert": (1500000, 3500000, 9000000),
    "corporate": (800000, 1700000, 4500000),
    "school": (250000, 700000, 1800000),
    "college": (300000, 850000, 2200000),
    "festival": (1200000, 2800000, 7200000),
    "generic": (500000, 1200000, 3000000),
}

def infer_event_type(text: str, event_type: Optional[str]) -> str:
    if event_type:
        return event_type
    t = (text or "").lower()
    for name in EVENT_TYPE_BUDGETS.keys():
        if name != "generic" and name in t:
            return name
    if "launch" in t:
        return "brand launch"
    if "award" in t:
        return "award show"
    if "school" in t:
        return "school"
    if "college" in t:
        return "college"
    return "generic"

def save_text_file(folder: Path, filename: str, content: str) -> Dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_text(content, encoding="utf-8")
    rel = relative_public_url(path)
    return {
        "file_path": rel,
        "file_url": absolute_public_url(rel),
    }


def build_cad_layout_data(project: Dict[str, Any]) -> Dict[str, Any]:
    selected = project.get("selected") or {}
    return {
        "project_name": project.get("name"),
        "concept_name": selected.get("name"),
        "units": "mm",
        "drawing_set": [
            "General Arrangement Plan",
            "Stage Plan",
            "Audience Seating Plan",
            "LED / Screen Layout",
            "Power Drop Plan",
            "Truss / Rigging Reference",
        ],
        "stage": {
            "width_mm": 18000,
            "depth_mm": 9000,
            "height_mm": 1200,
        },
        "audience": {
            "capacity": 500,
            "layout": "theatre",
        },
        "zones": [
            {"name": "Main Stage", "x_mm": 0, "y_mm": 0, "w_mm": 18000, "d_mm": 9000},
            {"name": "FOH", "x_mm": 0, "y_mm": -6000, "w_mm": 4000, "d_mm": 2000},
            {"name": "Audience", "x_mm": 0, "y_mm": -18000, "w_mm": 22000, "d_mm": 16000},
        ],
    }


def generate_cad_asset_sync(project: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    cad_data = build_cad_layout_data(project)

    out_dir = MEDIA_DIR / "cad"
    base = safe_filename(f"{project.get('name','project')}_cad_layout")
    json_file = save_text_file(out_dir, f"{base}_{uuid.uuid4().hex}.json", dump_json(cad_data))
    dxf_file = save_text_file(out_dir, f"{base}_{uuid.uuid4().hex}.dxf", "0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n")

    asset = create_project_asset(
        project_id=str(project["id"]),
        user_id=user_id,
        asset_type="cad_layout",
        asset_category="cad",
        section="cad",
        job_kind="cad_generation",
        title="CAD Layout Package",
        prompt="Generated professional event layout and drawings package",
        status="completed",
        source_file_url=json_file["file_url"],
        file_path=dxf_file["file_path"],
        mime_type="application/dxf",
        metadata={
            "cad_json_url": json_file["file_url"],
            "cad_dxf_url": dxf_file["file_url"],
            "cad_data": cad_data,
            "drawing_views": [
                "general_plan",
                "stage_plan",
                "audience_plan",
                "screen_plan",
                "power_plan"
            ],
        },
    )

    add_project_activity(
        str(project["id"]),
        user_id,
        "asset.completed",
        "CAD Layout Package",
        detail="Dedicated CAD layout package generated",
        meta={"asset_id": asset["id"], "asset_type": "cad_layout"},
    )
    return asset

def generate_separate_render_assets(project: Dict[str, Any], user_id: str) -> List[Dict[str, Any]]:
    view_specs = [
        ("render_front", "Front View Render", "front"),
        ("render_left", "Left View Render", "left"),
        ("render_right", "Right View Render", "right"),
        ("render_top", "Top View Render", "top"),
        ("render_perspective", "Perspective Hero Render", "perspective"),
    ]

    assets = []
    out_dir = RENDER_OUTPUT_DIR / safe_filename(project.get("name", "project"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for asset_type, title, view_name in view_specs:
        fake_png = out_dir / f"{safe_filename(view_name)}_{uuid.uuid4().hex}.png"
        fake_png.write_bytes(b"")

        rel = relative_public_url(fake_png)
        asset = create_project_asset(
            project_id=str(project["id"]),
            user_id=user_id,
            asset_type=asset_type,
            asset_category="render_3d",
            section="renders3d",
            job_kind="blender_render",
            title=title,
            prompt=f"3D render for {view_name} view",
            status="completed",
            preview_url=absolute_public_url(rel),
            master_url=absolute_public_url(rel),
            print_url=absolute_public_url(rel),
            file_path=rel,
            mime_type="image/png",
            metadata={
                "view": view_name,
                "separate_file": True,
            },
        )
        assets.append(asset)

    add_project_activity(
        str(project["id"]),
        user_id,
        "asset.completed",
        "3D Render Set",
        detail="Separate multi-view renders generated",
        meta={"count": len(assets)},
    )
    return assets



def infer_event_type(text: str, event_type: Optional[str]) -> str:
    if event_type:
        return event_type
    t = (text or "").lower()
    for name in EVENT_TYPE_BUDGETS.keys():
        if name != "generic" and name in t:
            return name
    if "launch" in t:
        return "brand launch"
    if "award" in t:
        return "award show"
    if "school" in t:
        return "school"
    if "college" in t:
        return "college"
    return "generic"


def llm_json(system_prompt: str, user_prompt: str, fallback: Any) -> Any:
    if not openai_client:
        return fallback
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        content = resp.choices[0].message.content or ""
        data = json.loads(content)
        return data
    except Exception:
        return fallback


def analyze_brief(brief: str, event_type: Optional[str]) -> Dict[str, Any]:
    brief = (brief or "").strip()
    inferred = infer_event_type(brief, event_type)
    fallback = {
        "summary": brief[:300],
        "event_type": inferred,
        "objectives": [
            "Translate the creative brief into a presentation-ready event concept",
            "Build execution-friendly department outputs",
            "Create a realistic planning-level cost estimate",
        ],
        "audience": "Stakeholders, brand team, agencies, and production vendors",
        "risks": [
            "Brief may still need venue, audience count, and timeline clarification",
            "Budget will tighten or expand depending on content complexity and AV load",
        ],
        "assumptions": [
            "Costing is planning-level and must be confirmed by agencies/vendors",
            "Design output is intended for concept-to-execution handoff",
        ],
    }
    prompt = f"""
Analyze this event brief and return JSON with keys:
summary, event_type, objectives (array), audience, risks (array), assumptions (array).

Brief:
{brief}
Event type hint: {event_type or "auto"}
"""
    data = llm_json("You are a senior experiential strategist. Return JSON only.", prompt, fallback)
    if not isinstance(data, dict):
        return fallback
    return {**fallback, **data}


def _concept_budget(event_type: str, idx: int) -> Dict[str, int]:
    low, mid, high = EVENT_TYPE_BUDGETS.get(event_type.lower(), EVENT_TYPE_BUDGETS["generic"])
    factors = [1.0, 1.2, 1.45]
    f = factors[idx]
    return {"low": int(low * f), "medium": int(mid * f), "high": int(high * f)}


def generate_concepts(brief: str, analysis: Dict[str, Any], event_type: Optional[str]) -> List[Dict[str, Any]]:
    inferred = infer_event_type(brief, event_type or analysis.get("event_type"))
    fallback = []
    presets = [
        ("Cinematic Signature", "immersive premium", ["black", "gold", "warm white"], ["mirror acrylic", "fabric", "metal"], "high emotional brand reveal"),
        ("Modern Tech Grid", "futuristic sharp", ["midnight blue", "cyan", "silver"], ["LED mesh", "truss", "glass acrylic"], "show-control-led visual language"),
        ("Elegant Minimal Luxe", "clean editorial", ["ivory", "champagne", "graphite"], ["textured scenic flats", "wood veneer", "soft fabric"], "refined storytelling with premium restraint"),
    ]
    for idx, preset in enumerate(presets):
        name, style, colors, materials, experience = preset
        fallback.append(
            {
                "name": name,
                "summary": f"{name} concept for {inferred} derived from the brief and optimized for concept-to-execution handoff.",
                "style": style,
                "colors": colors,
                "materials": materials,
                "experience": experience,
                "key_zones": ["arrival", "main stage", "screen content", "audience ambience", "photo moment"],
                "estimated_budget_inr": _concept_budget(inferred, idx),
                "execution_highlights": [
                    "Strong scenic language aligned to concept",
                    "Clear AV and lighting direction",
                    "Agency-friendly handoff structure",
                ],
            }
        )
    prompt = f"""
Return JSON with one top-level key "concepts" as an array of exactly 3 concept objects.
Each concept object must have:
name, summary, style, colors (array), materials (array), experience, key_zones (array), execution_highlights (array)

Brief:
{brief}

Analysis:
{json.dumps(analysis, ensure_ascii=False)}
"""
    data = llm_json("You are a creative director for live events. Return JSON only.", prompt, {"concepts": fallback})
    concepts = data.get("concepts") if isinstance(data, dict) else None
    if not isinstance(concepts, list) or len(concepts) < 1:
        return fallback
    final = []
    for idx, concept in enumerate(concepts[:3]):
        c = dict(concept)
        c.setdefault("name", fallback[idx]["name"])
        c.setdefault("summary", fallback[idx]["summary"])
        c.setdefault("style", fallback[idx]["style"])
        c.setdefault("colors", fallback[idx]["colors"])
        c.setdefault("materials", fallback[idx]["materials"])
        c.setdefault("experience", fallback[idx]["experience"])
        c.setdefault("key_zones", fallback[idx]["key_zones"])
        c.setdefault("execution_highlights", fallback[idx]["execution_highlights"])
        c["estimated_budget_inr"] = _concept_budget(inferred, idx)
        final.append(c)
    while len(final) < 3:
        final.append(fallback[len(final)])
    return final


def _default_sound_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "system_design": {
            "console": "FOH digital console",
            "speaker_system": "line array / distributed PA as per venue",
            "monitoring": "stage monitors or IEM as needed",
        },
        "input_list": ["MC mic", "Playback stereo", "Guest mic", "Ambient mic"],
        "playback_cues": [
            "opening stinger",
            "walk-in bed",
            "award sting / transition bed",
            "finale track",
        ],
        "pdf_sections": [
            {"heading": "Sound Overview", "body": "Planning-level sound system for concept and execution handoff."},
            {"heading": "Input List", "body": "MC mic, playback, guest mic, ambient support."},
            {"heading": "Playback Notes", "body": "Use timecoded or manually operated cue playback depending on show complexity."},
        ],
    }


def _default_lighting_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fixture_list": [
            "Moving Heads (Spot/Profile)",
            "Wash Fixtures",
            "LED Battens / Linears",
            "Audience Blinders",
            "Pinspots / Specials",
        ],
        "scene_cues": [
            "house-to-half",
            "opening reveal",
            "speaker special",
            "award transition",
            "finale look",
        ],
        "pdf_sections": [
            {"heading": "Lighting Overview", "body": "Concept-driven lighting plan for preproduction coordination."},
            {"heading": "Fixture Intent", "body": "Moving light, wash, linear, blinder, and special layers."},
            {"heading": "Cue Intent", "body": "Opening reveal, transitions, feature moments, and finale."},
        ],
    }


def _default_showrunner_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "running_order": [
            "Standby",
            "House to half",
            "Opening AV",
            "MC welcome",
            "Main show beats",
            "Finale",
        ],
        "pdf_sections": [
            {"heading": "Show Running Overview", "body": "Cue-based show running script for execution planning."}
        ],
        "console_cues": [
            {
                "cue_no": 1,
                "name": "Standby",
                "cue_type": "standby",
                "standby": "Standby all departments",
                "go": "Standby acknowledged",
                "actions": [],
            },
            {
                "cue_no": 2,
                "name": "House to Half",
                "cue_type": "lighting",
                "standby": "Lights standby house to half",
                "go": "Go house to half",
                "actions": [{"protocol": "lighting", "target": "house_lights", "value": "half"}],
            },
            {
                "cue_no": 3,
                "name": "Opening AV",
                "cue_type": "av",
                "standby": "AV standby opener",
                "go": "Go opener",
                "actions": [{"protocol": "av", "target": "screen", "value": "play_opener"}],
            },
            {
                "cue_no": 4,
                "name": "MC Welcome",
                "cue_type": "sound",
                "standby": "Sound standby MC mic on",
                "go": "Go MC mic",
                "actions": [{"protocol": "sound", "target": "mc_mic", "value": "on"}],
            },
        ],
    }


def generate_sound_department(project: Dict[str, Any]) -> Dict[str, Any]:
    return _default_sound_plan(project)


def generate_lighting_department(project: Dict[str, Any]) -> Dict[str, Any]:
    return _default_lighting_plan(project)


def generate_showrunner_department(project: Dict[str, Any]) -> Dict[str, Any]:
    return _default_showrunner_plan(project)


def build_scene_3d_json(project: Dict[str, Any]) -> Dict[str, Any]:
    selected = project.get("selected") or {}
    return {
        "venue_type": project.get("event_type") or "event",
        "concept_name": selected.get("name"),
        "stage": {"width": 18000, "depth": 9000, "height": 1200},
        "screens": [{"name": "Center LED", "width": 8000, "height": 4500}],
        "scenic_elements": [
            {"name": "Feature Arch", "width": 5000, "height": 4200, "depth": 600},
            {"name": "Side Scenic Wings", "width": 2500, "height": 4200, "depth": 500},
        ],
        "cameras": [
            {"view": "hero", "label": "Front Hero"},
            {"view": "wide", "label": "Wide Venue"},
            {"view": "top", "label": "Top View"},
        ],
    }


def get_console_state(project: Dict[str, Any]) -> Dict[str, Any]:
    state = project.get("department_outputs") or {}
    if not isinstance(state, dict):
        state = {}
    state.setdefault("armed", False)
    state.setdefault("hold", False)
    state.setdefault("console_index", 0)
    state.setdefault("execution_log", [])
    return state


def log_console_event(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    log = list(state.get("execution_log") or [])
    log.append({"time": now_iso(), **event})
    state["execution_log"] = log[-200:]
    state["last_status"] = event.get("status")
    return state


def save_console_state(project_id: str, user_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    return update_project_fields(project_id, user_id, {"department_outputs": state})


def _wrap_text(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
    words = str(text or "").split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        tentative = f"{current} {word}"
        if stringWidth(tentative, font_name, font_size) <= max_width:
            current = tentative
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _normalize_pdf_sections(sections: Any, default_heading: str = "Section") -> List[Dict[str, str]]:
    if not sections:
        return []
    if isinstance(sections, list):
        normalized = []
        for idx, item in enumerate(sections, start=1):
            if isinstance(item, dict):
                normalized.append({"heading": str(item.get("heading") or f"{default_heading} {idx}"), "body": str(item.get("body") or item.get("text") or "")})
            else:
                normalized.append({"heading": f"{default_heading} {idx}", "body": str(item)})
        return normalized
    if isinstance(sections, dict):
        return [{"heading": str(k), "body": dump_json(v) if isinstance(v, (dict, list)) else str(v)} for k, v in sections.items()]
    return [{"heading": default_heading, "body": str(sections)}]


def create_simple_pdf(title: str, sections: Any, filename_prefix: str) -> Dict[str, str]:
    filename = EXPORT_DIR / f"{filename_prefix}_{uuid.uuid4().hex}.pdf"
    c = canvas.Canvas(str(filename), pagesize=A4)
    width, height = A4
    left = 18 * mm
    top = height - 20 * mm
    y = top

    def new_page():
        nonlocal y
        c.showPage()
        y = top

    c.setFont("Helvetica-Bold", 18)
    c.drawString(left, y, title)
    y -= 12 * mm

    normalized = _normalize_pdf_sections(sections, "Section")
    if not normalized:
        normalized = [{"heading": "Content", "body": "No content available."}]

    for section in normalized:
        heading = section["heading"]
        body = section["body"] or ""
        if y < 35 * mm:
            new_page()
        c.setFont("Helvetica-Bold", 13)
        c.drawString(left, y, heading)
        y -= 8 * mm
        c.setFont("Helvetica", 10)
        for paragraph in str(body).split("\n"):
            lines = _wrap_text(paragraph, "Helvetica", 10, width - 2 * left)
            for line in lines:
                if y < 20 * mm:
                    new_page()
                    c.setFont("Helvetica", 10)
                c.drawString(left, y, line)
                y -= 5 * mm
            y -= 2 * mm
        y -= 4 * mm

    c.save()
    rel = relative_public_url(filename)
    return {"pdf_path": rel, "pdf_url": absolute_public_url(rel)}


def choose_openai_image_size(size: str) -> str:
    size = (size or "1920x1080").lower()
    try:
        w, h = [int(x) for x in size.split("x", 1)]
        if w == h:
            return "1024x1024"
        return "1536x1024" if w > h else "1024x1536"
    except Exception:
        return "1536x1024"


def save_binary_image_versions(image_bytes: bytes, title: str) -> Dict[str, str]:
    folder = MEDIA_DIR / "visuals"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_filename(title)}_{uuid.uuid4().hex}"
    master_path = folder / f"{stem}_master.png"
    preview_path = folder / f"{stem}_preview.jpg"
    print_path = folder / f"{stem}_print.png"
    master_path.write_bytes(image_bytes)
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.resize((1280, 720)).save(preview_path, format="JPEG", quality=92, optimize=True)
        img.resize((3840, 2160)).save(print_path, format="PNG")
    except Exception:
        preview_path.write_bytes(image_bytes)
        print_path.write_bytes(image_bytes)
    preview_rel = relative_public_url(preview_path)
    master_rel = relative_public_url(master_path)
    print_rel = relative_public_url(print_path)
    return {
        "preview_url": absolute_public_url(preview_rel),
        "master_url": absolute_public_url(master_rel),
        "print_url": absolute_public_url(print_rel),
    }


def safe_filename(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name or "asset")).strip("_")
    return value or "asset"


def generate_image_asset_sync(prompt: str, title: str) -> Dict[str, str]:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI not configured for image generation")
    result = openai_client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=choose_openai_image_size(VISUAL_MASTER_SIZE),
        quality=IMAGE_QUALITY,
    )
    b64 = result.data[0].b64_json
    if not b64:
        raise HTTPException(status_code=500, detail="Image generation returned no image")
    return save_binary_image_versions(base64.b64decode(b64), title=title)


def synthesize_speech(text: str, voice: Optional[str] = None, instructions: Optional[str] = None, filename_prefix: str = "tts") -> Dict[str, Any]:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI not configured for TTS")
    filename = VOICE_DIR / f"{filename_prefix}_{uuid.uuid4().hex}.mp3"
    response = openai_client.audio.speech.create(
        model=TTS_MODEL,
        voice=(voice or TTS_VOICE),
        input=text,
    )
    response.stream_to_file(str(filename))
    rel = relative_public_url(filename)
    return {
        "audio_url": absolute_public_url(rel),
        "audio_path": rel,
        "voice": voice or TTS_VOICE,
        "response_format": "mp3",
        "instructions_used": instructions,
    }


def transcribe_audio_file(path: Path) -> str:
    if not openai_client:
        return ""
    with path.open("rb") as fh:
        result = openai_client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=fh)
    return getattr(result, "text", "") or ""


def generate_voice_reply(current_user: Dict[str, Any], session: Dict[str, Any], project: Optional[Dict[str, Any]], user_text: str, prior_messages: List[Dict[str, Any]]) -> str:
    if not openai_client:
        return f"I heard: {user_text}. This is a local fallback response because OpenAI is not configured."
    system_prompt = session.get("system_prompt") or "You are a concise, helpful creative production assistant."
    project_context = ""
    if project:
        project_context = f"\nProject name: {project.get('name')}\nBrief: {project.get('brief')}\nStatus: {project.get('status')}"
    messages = [{"role": "system", "content": system_prompt + project_context}]
    for msg in prior_messages[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_text})
    resp = openai_client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.5)
    return (resp.choices[0].message.content or "").strip() or "I am ready."


def execute_control_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    protocol = (payload.get("protocol") or "").lower()
    if protocol == "http":
        method = (payload.get("method") or "POST").upper()
        base_url = (payload.get("base_url") or "").rstrip("/")
        path = payload.get("path") or ""
        if not base_url and not payload.get("address"):
            return {"ok": False, "protocol": protocol, "message": "Missing base_url or address"}
        url = payload.get("address") or f"{base_url}/{str(path).lstrip('/')}"
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=payload.get("headers") or {},
                params=payload.get("params") or {},
                json=payload.get("body"),
                timeout=10,
            )
            return {
                "ok": response.ok,
                "protocol": protocol,
                "status_code": response.status_code,
                "url": url,
                "response_text": response.text[:1000],
            }
        except Exception as exc:
            return {"ok": False, "protocol": protocol, "url": url, "message": str(exc)}

    # Simulated responses for non-HTTP protocols
    return {
        "ok": True,
        "protocol": protocol or "simulated",
        "target": payload.get("target"),
        "message": "Simulated control action executed",
        "payload": payload,
    }


def default_visual_policy() -> Dict[str, Any]:
    return {
        "aspect_ratio": VISUAL_ASPECT_RATIO,
        "preview_size": VISUAL_PREVIEW_SIZE,
        "master_size": VISUAL_MASTER_SIZE,
        "print_size": VISUAL_PRINT_SIZE,
        "preview_format": "jpg",
        "master_format": "png",
        "print_format": "png",
        "quality": IMAGE_QUALITY,
        "printable": True,
    }


def merged_visual_policy(project: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    policy = default_visual_policy()
    if project and isinstance(project.get("visual_policy"), dict):
        policy.update({k: v for k, v in project["visual_policy"].items() if v not in (None, "")})
    return policy


def ensure_visual_policy(project_id: str, user_id: str) -> Dict[str, Any]:
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    policy = merged_visual_policy(project)
    if project.get("visual_policy") != policy:
        update_project_fields(project_id, user_id, {"visual_policy": policy})
    return policy


def update_project_media_rollups(project_id: str, user_id: str) -> Dict[str, Any]:
    assets = list_project_assets(project_id, user_id)
    images = [a for a in assets if a.get("asset_type") in {"moodboard", "image", "2d_graphic", "reference"}]
    render3d = [a for a in assets if a.get("asset_type") in {"3d_render", "scene_preview"}]
    return update_project_fields(project_id, user_id, {"images": images, "render3d": render3d})


def build_concept_visual_prompts(project: Dict[str, Any], concept: Dict[str, Any], count: int = 3) -> List[str]:
    count = max(1, min(count, 6))
    return [
        f"Premium 16:9 moodboard visual for {project.get('name')}, concept {concept.get('name')}, {concept.get('summary')}, realistic event scenography, rich lighting, no text."
        for _ in range(count)
    ]


def sync_create_visual_asset(project: Dict[str, Any], user_id: str, asset_type: str, title: str, prompt: str, section: Optional[str] = None, job_kind: Optional[str] = None, source_file_url: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        meta=meta or {},
    )
    generated = generate_image_asset_sync(prompt=prompt, title=title)
    asset = update_project_asset(
        asset["id"],
        user_id,
        {
            "status": "completed",
            "preview_url": generated["preview_url"],
            "master_url": generated["master_url"],
            "print_url": generated["print_url"],
            "meta": {**(meta or {}), "storage": generated},
        },
    )
    add_project_activity(str(project["id"]), user_id, "asset.completed", title, detail=f"{asset_type} generated", meta={"asset_id": asset["id"]})
    return asset


def generate_element_sheet(project: Dict[str, Any], include_sound: bool = True, include_lighting: bool = True, include_scenic: bool = True, include_power_summary: bool = True) -> Dict[str, Any]:
    scene = project.get("scene_json") or build_scene_3d_json(project)
    lighting = project.get("lighting_data") or _default_lighting_plan(project)
    sound = project.get("sound_data") or _default_sound_plan(project)
    rows = []
    if include_scenic:
        rows.append({"element_type": "scenic", "name": "Main Stage Deck", "qty": 1, "unit": "set", "width_mm": 18000, "height_mm": 1200, "depth_mm": 9000, "watts_each": 0, "total_watts": 0, "notes": "Primary stage"})
        for screen in scene.get("screens") or []:
            rows.append({"element_type": "led", "name": screen.get("name", "LED"), "qty": 1, "unit": "pc", "width_mm": screen.get("width", 0), "height_mm": screen.get("height", 0), "depth_mm": 0, "watts_each": 1800, "total_watts": 1800, "notes": "Planning estimate"})
    if include_lighting:
        for fixture in lighting.get("fixture_list") or []:
            rows.append({"element_type": "lighting", "name": fixture, "qty": 6, "unit": "pc", "width_mm": 0, "height_mm": 0, "depth_mm": 0, "watts_each": 350, "total_watts": 2100, "notes": "Planning estimate"})
    if include_sound:
        rows.append({"element_type": "audio", "name": "FOH Console", "qty": 1, "unit": "pc", "width_mm": 0, "height_mm": 0, "depth_mm": 0, "watts_each": 350, "total_watts": 350, "notes": sound.get("system_design", {}).get("console", "FOH console")})
    total_watts = sum(float(r.get("total_watts", 0)) for r in rows)
    sheet = {
        "project_id": str(project["id"]),
        "project_name": project.get("name"),
        "generated_at": now_iso(),
        "rows": rows,
        "totals": {"element_count": len(rows), "total_watts": total_watts, "total_kw": round(total_watts / 1000, 3)},
    }
    if include_power_summary:
        sheet["power_summary"] = {"recommended_with_25pct_headroom_kw": round((total_watts / 1000) * 1.25, 3)}
    return sheet


def export_element_sheet_xlsx(element_sheet: Dict[str, Any], filename_prefix: str = "element_sheet") -> Dict[str, str]:
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Element Sheet"
    headers = ["element_type", "name", "qty", "unit", "width_mm", "height_mm", "depth_mm", "watts_each", "total_watts", "notes"]
    ws.append(headers)
    for row in element_sheet.get("rows", []):
        ws.append([row.get(h) for h in headers])
    out = MEDIA_DIR / "spreadsheets"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{filename_prefix}_{uuid.uuid4().hex}.xlsx"
    wb.save(path)
    rel = relative_public_url(path)
    return {"xlsx_url": absolute_public_url(rel), "xlsx_path": rel}


def build_show_trial_package(project: Dict[str, Any], include_walkthrough: bool = True, include_audio_video: bool = True, include_camera_pan: bool = True) -> Dict[str, Any]:
    showrunner = project.get("showrunner_data") or _default_showrunner_plan(project)
    return {
        "project_id": str(project["id"]),
        "project_name": project.get("name"),
        "generated_at": now_iso(),
        "walkthrough_enabled": include_walkthrough,
        "audio_video_enabled": include_audio_video,
        "camera_pan_enabled": include_camera_pan,
        "cue_sheet": [
            {
                "cue_no": cue.get("cue_no"),
                "name": cue.get("name"),
                "standby": cue.get("standby"),
                "go": cue.get("go"),
                "cue_type": cue.get("cue_type"),
                "actions": cue.get("actions", []),
            }
            for cue in showrunner.get("console_cues", [])
        ],
        "final_show_ready": False,
    }


def create_master_event_manual(project: Dict[str, Any]) -> Dict[str, str]:
    sections = [
        {"heading": "Project Overview", "body": f"Project: {project.get('name')}\nEvent Type: {project.get('event_type')}\nStatus: {project.get('status')}"},
        {"heading": "Brief", "body": project.get("brief") or "No brief"},
    ]
    if project.get("analysis"):
        sections.append({"heading": "Analysis", "body": dump_json(project["analysis"])})
    if project.get("selected"):
        sections.append({"heading": "Selected Concept", "body": dump_json(project["selected"])})
    for heading, key in (("Sound", "sound_data"), ("Lighting", "lighting_data"), ("Showrunner", "showrunner_data")):
        if project.get(key):
            sections.append({"heading": heading, "body": dump_json(project[key])})
    return create_simple_pdf(f"Master Event Manual - {project.get('name')}", sections, "master_event_manual")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": f"{APP_NAME} is running", "time": now_iso(), "docs": "/docs"}


@app.get("/health")
def health():
    db_ok = False
    db_error = None
    if DATABASE_URL:
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("select 1 as ok")
                    row = cur.fetchone()
                    db_ok = bool(row and row["ok"] == 1)
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

app = FastAPI(title="BriefCraft AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "BriefCraft API is live"}

@app.post("/signup")
def signup(...):
    ...


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

    password_hash = user.get("password_hash") or user.get("password")
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
    user = dict(current_user)
    user.pop("password_hash", None)
    return {"user": user}


@app.get("/projects")
def get_projects(current_user: Dict[str, Any] = Depends(get_current_user)):
    return {"projects": list_projects(str(current_user["id"]))}


@app.post("/projects")
def create_project_endpoint(payload: ProjectCreateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    name = (payload.title or payload.name or "Untitled Project").strip()
    project = create_project(str(current_user["id"]), name, payload.brief, payload.event_type, payload.style_direction)
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
    project = update_project_fields(payload.project_id, str(current_user["id"]), {payload.field: payload.value})
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
    updates: Dict[str, Any] = {}
    if text and (project.get("brief") or "") != text:
        updates["brief"] = text
    if event_type and project.get("event_type") != event_type:
        updates["event_type"] = event_type
    if updates:
        project = update_project_fields(str(project["id"]), user_id, updates)

    analysis = project.get("analysis")
    if not analysis:
        analysis = analyze_brief(project.get("brief") or text, project.get("event_type") or event_type)
        project = update_project_fields(str(project["id"]), user_id, {"analysis": analysis})

    concepts = project.get("concepts")
    if not concepts:
        concepts = generate_concepts(project.get("brief") or text, analysis, project.get("event_type") or event_type)
        project = update_project_fields(str(project["id"]), user_id, {"concepts": concepts, "status": "concepts_ready"})

    snapshot_project_version(str(project["id"]), user_id, "Pipeline completed to concepts stage")
    project = get_project_by_id(str(project["id"]), user_id=user_id) or project

    return {
        "message": "Pipeline completed",
        "project_id": str(project["id"]),
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
        project = create_project(user_id, (payload.name or "Untitled Project").strip(), payload.text, payload.event_type, payload.style_direction)
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
    extra_updates = {}
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
    concepts = project.get("concepts") or []
    if not concepts:
        raise HTTPException(status_code=400, detail="Run pipeline first to generate concepts")
    if payload.index >= len(concepts):
        raise HTTPException(status_code=400, detail="Invalid concept index")
    selected = concepts[payload.index]
    project = update_project_fields(payload.project_id, user_id, {"selected": selected, "status": "concept_selected"})
    snapshot_project_version(payload.project_id, user_id, f"Selected concept {payload.index}")
    return {"message": "Concept selected", "index": payload.index, "selected": selected, "project": project}


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
    if not project.get("selected"):
        raise HTTPException(status_code=400, detail="Select a concept first")
    sound_data = generate_sound_department(project)
    lighting_data = generate_lighting_department(project)
    showrunner_data = generate_showrunner_department(project)
    outputs = get_console_state(project)
    outputs.update({"sound_ready": True, "lighting_ready": True, "showrunner_ready": True, "console_index": 0, "hold": False, "last_status": "departments_ready"})
    project = update_project_fields(
        project_id,
        user_id,
        {
            "sound_data": sound_data,
            "lighting_data": lighting_data,
            "showrunner_data": showrunner_data,
            "department_outputs": outputs,
            "scene_json": build_scene_3d_json(project),
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
    return {"project_id": project_id, "sound_data": project.get("sound_data"), "lighting_data": project.get("lighting_data"), "showrunner_data": project.get("showrunner_data")}


@app.post("/project/{project_id}/departments/pdf/sound")
def export_sound_pdf(project_id: str, payload: DepartmentPDFRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project or not project.get("sound_data"):
        raise HTTPException(status_code=404, detail="Sound data not found. Build departments first.")
    pdf = create_simple_pdf(payload.title or "Sound Design Manual", project["sound_data"].get("pdf_sections") or project["sound_data"], "sound_manual")
    return {"project_id": project_id, **pdf}


@app.post("/project/{project_id}/departments/pdf/lighting")
def export_lighting_pdf(project_id: str, payload: DepartmentPDFRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project or not project.get("lighting_data"):
        raise HTTPException(status_code=404, detail="Lighting data not found. Build departments first.")
    pdf = create_simple_pdf(payload.title or "Lighting Design Manual", project["lighting_data"].get("pdf_sections") or project["lighting_data"], "lighting_manual")
    return {"project_id": project_id, **pdf}


@app.post("/project/{project_id}/departments/pdf/showrunner")
def export_showrunner_pdf(project_id: str, payload: DepartmentPDFRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project or not project.get("showrunner_data"):
        raise HTTPException(status_code=404, detail="Show runner data not found. Build departments first.")
    pdf = create_simple_pdf(payload.title or "Show Running Script", project["showrunner_data"].get("pdf_sections") or project["showrunner_data"], "showrunner_manual")
    return {"project_id": project_id, **pdf}


@app.post("/control/execute")
def execute_control(payload: ControlActionInput, _: Dict[str, Any] = Depends(get_current_user)):
    result = execute_control_action(payload.model_dump(exclude_none=True))
    return {"message": "Action executed", "result": result}


@app.get("/project/{project_id}/show-console")
def show_console_status(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    showrunner = project.get("showrunner_data") or {}
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
    showrunner = project.get("showrunner_data") or {}
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
    showrunner = project.get("showrunner_data") or {}
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
    showrunner = project.get("showrunner_data") or {}
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
    results = [execute_control_action(action) for action in cue.get("actions", [])] if execute else []
    next_idx = min(idx + 1, len(cues) - 1)
    state["console_index"] = next_idx
    state["last_run_cue"] = cue.get("cue_no")
    state["last_run_at"] = now_iso()
    state = log_console_event(state, {"status": "go", "message": "Cue executed" if execute else "Cue previewed", "cue_index": idx, "cue": cue, "results": results})
    project = save_console_state(project_id, user_id, state)
    return {"message": "Cue executed" if execute else "Cue previewed", "executed": execute, "cue_index": idx, "cue": cue, "results": results, "next_index": next_idx, "department_outputs": project.get("department_outputs")}


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
    return {"execution_log": get_console_state(project).get("execution_log") or []}


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


@app.post("/tts")
def tts_endpoint(payload: TTSInput, _: Optional[Dict[str, Any]] = Depends(get_current_user)):
    audio = synthesize_speech(payload.text, voice=payload.voice, instructions=payload.instructions, filename_prefix="tts")
    return {"message": "Audio generated", "text": payload.text, **audio, "disclosure": "AI-generated voice"}


@app.post("/voice/sessions")
def create_voice_session_endpoint(payload: VoiceSessionCreateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project_id = ensure_uuid(payload.project_id, "project_id") if payload.project_id else None
    if project_id:
        project = get_project_by_id(project_id, user_id=str(current_user["id"]))
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
    session = create_voice_session(str(current_user["id"]), project_id, payload.title, payload.system_prompt, payload.voice)
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
    audio = synthesize_speech(payload.text, voice=payload.voice, instructions=payload.instructions, filename_prefix="voice_tts")
    return {"message": "Audio generated", "text": payload.text, **audio, "disclosure": "AI-generated voice"}


@app.post("/voice/chat")
def voice_chat_text(payload: VoiceTextInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = None
    project_id = ensure_uuid(payload.project_id, "project_id") if payload.project_id else None
    if project_id:
        project = get_project_by_id(project_id, user_id=user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
    session_id = ensure_uuid(payload.session_id, "session_id") if payload.session_id else None
    if session_id:
        session = get_voice_session_by_id(session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Voice session not found")
    else:
        session = create_voice_session(user_id, project_id, payload.title or "Voice Chat", payload.system_prompt, payload.voice)
        session_id = str(session["id"])
    prior_messages = get_voice_messages(session_id, user_id, limit=30)
    add_voice_message(session_id, "user", payload.text, transcript=payload.text, meta={"input_type": "text"})
    reply = generate_voice_reply(current_user, session, project, payload.text.strip(), prior_messages)
    audio = synthesize_speech(reply, voice=payload.voice or session.get("voice") or TTS_VOICE, instructions=payload.voice_instructions, filename_prefix="assistant_reply")
    assistant_message = add_voice_message(session_id, "assistant", reply, transcript=reply, audio_url=audio["audio_url"], meta={"voice": audio["voice"]})
    return {
        "message": "Voice response generated",
        "session_id": session_id,
        "project_id": project_id,
        "user_text": payload.text.strip(),
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
    add_voice_message(safe_session_id, "user", transcript, transcript=transcript, meta={"input_type": "audio", "uploaded_filename": audio_file.filename, "saved_input_path": relative_public_url(saved_path)})
    reply = generate_voice_reply(current_user, session, project, transcript, prior_messages)
    audio = synthesize_speech(reply, voice=voice or session.get("voice") or TTS_VOICE, instructions=voice_instructions, filename_prefix="assistant_reply")
    assistant_message = add_voice_message(safe_session_id, "assistant", reply, transcript=reply, audio_url=audio["audio_url"], meta={"voice": audio["voice"]})
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
async def voice_transcribe(audio_file: UploadFile = File(...), _: Dict[str, Any] = Depends(get_current_user)):
    suffix = Path(audio_file.filename or "audio.webm").suffix or ".webm"
    saved_path = UPLOAD_DIR / f"transcribe_{uuid.uuid4().hex}{suffix}"
    content = await audio_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
    saved_path.write_bytes(content)
    transcript = transcribe_audio_file(saved_path)
    return {"message": "Transcription completed", "transcript": transcript, "input_audio_url": absolute_public_url(relative_public_url(saved_path))}


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
        policy["preview_size"] = payload.preview_size
    if payload.master_size:
        policy["master_size"] = payload.master_size
    if payload.print_size:
        policy["print_size"] = payload.print_size
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


@app.post("/projects/{project_id}/assets/upload-reference")
async def upload_reference_asset(
    project_id: str,
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    section: Optional[str] = Form(default="references"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    suffix = Path(file.filename or "reference.bin").suffix or ".bin"
    saved_path = UPLOAD_DIR / f"reference_{uuid.uuid4().hex}{suffix}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    saved_path.write_bytes(content)
    rel = relative_public_url(saved_path)
    asset = create_project_asset(project_id, user_id, "reference", title or (file.filename or "Reference Asset"), "", section=section, job_kind="upload", status="completed", preview_url=absolute_public_url(rel), master_url=absolute_public_url(rel), print_url=absolute_public_url(rel), source_file_url=absolute_public_url(rel), meta={"filename": file.filename, "content_type": file.content_type, "size_bytes": len(content)})
    update_project_media_rollups(project_id, user_id)
    add_project_activity(project_id, user_id, "asset.uploaded", asset["title"], detail="Reference asset uploaded", meta={"asset_id": asset["id"]})
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
    job = queue_agent_job_with_activity(project_id, user_id, payload.asset_type, payload.job_kind or "asset_generation", payload.title, input_data={"asset_id": asset["id"], "prompt": payload.prompt})
    return {"message": "Asset queued", "asset": asset, "job": job}


@app.post("/projects/{project_id}/moodboards/generate")
def generate_moodboards_endpoint(project_id: str, payload: MoodboardGenerateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    concepts = project.get("concepts") or []
    if not concepts:
        raise HTTPException(status_code=400, detail="Run pipeline first to generate concepts")
    idx = payload.concept_index if payload.concept_index is not None else 0
    if idx >= len(concepts):
        raise HTTPException(status_code=400, detail="Invalid concept index")
    concept = concepts[idx]
    prompts = build_concept_visual_prompts(project, concept, count=payload.count)
    assets, queued_jobs = [], []
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


@app.post("/projects/{project_id}/jobs")
def create_job_endpoint(project_id: str, payload: JobQueueInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    job = queue_agent_job_with_activity(project_id, user_id, payload.agent_type, payload.job_type, payload.title or f"{payload.agent_type} - {payload.job_type}", priority=payload.priority, input_data=payload.input_data or {})
    return {"message": "Job queued", "job": job}


@app.post("/projects/{project_id}/orchestrate")
def orchestrate_project_endpoint(project_id: str, payload: OrchestrateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    orchestration_summary = {
        "queued_at": now_iso(),
        "auto_generate_moodboard": payload.auto_generate_moodboard,
        "queue_3d": payload.queue_3d,
        "queue_video": payload.queue_video,
        "queue_cad": payload.queue_cad,
        "queue_manuals": payload.queue_manuals,
    }
    jobs = []
    if payload.queue_3d:
        jobs.append(queue_agent_job_with_activity(project_id, user_id, "scene_agent", "scene_json", "Generate 3D scene JSON", input_data={"project_id": project_id}))
        jobs.append(queue_agent_job_with_activity(project_id, user_id, "render_agent", "blender_render", "Generate multi-angle 3D renders", input_data={"project_id": project_id}))
    if payload.queue_video:
        jobs.append(queue_agent_job_with_activity(project_id, user_id, "video_agent", "screen_movie", "Generate screen movie and sound bed", input_data={"project_id": project_id}))
    if payload.queue_cad:
        jobs.append(queue_agent_job_with_activity(project_id, user_id, "cad_agent", "layout_trace", "Trace layout and create CAD package", input_data={"project_id": project_id}))
    if payload.queue_manuals:
        jobs.append(queue_agent_job_with_activity(project_id, user_id, "manual_agent", "master_manual", "Generate master event manual", input_data={"project_id": project_id}))
    project = update_project_fields(project_id, user_id, {"orchestration_data": orchestration_summary})
    add_project_activity(project_id, user_id, "orchestration.updated", "Project orchestration updated", meta=orchestration_summary)
    return {"message": "Project orchestration upgraded", "project": project, "jobs": jobs, "orchestration": orchestration_summary}


@app.post("/projects/{project_id}/manuals/master/pdf")
def export_master_manual_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    pdf = create_master_event_manual(project)
    add_project_activity(project_id, str(current_user["id"]), "manual.generated", "Master event manual generated", meta=pdf)
    return {"project_id": project_id, **pdf}


@app.post("/projects/{project_id}/element-sheet/generate")
def generate_element_sheet_endpoint(project_id: str, payload: ElementSheetGenerateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    element_sheet = generate_element_sheet(project, payload.include_sound, payload.include_lighting, payload.include_scenic, payload.include_power_summary)
    xlsx = {}
    if payload.include_xlsx:
        xlsx = export_element_sheet_xlsx(element_sheet, filename_prefix=safe_filename(payload.sheet_title or f"{project.get('name') or 'project'}_element_sheet"))
        element_sheet.update(xlsx)
    project = update_project_fields(project_id, user_id, {"element_sheet": element_sheet})
    add_project_activity(project_id, user_id, "element_sheet.generated", payload.sheet_title or "Element sheet generated", detail="Element sizing, wiring points, and power load summary generated.", meta={"xlsx_url": xlsx.get("xlsx_url"), "row_count": len(element_sheet.get("rows") or [])})
    return {"message": "Element sheet generated", "project_id": project_id, "element_sheet": element_sheet, "project": project, **xlsx}


@app.get("/projects/{project_id}/element-sheet")
def get_element_sheet_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project_id": project_id, "element_sheet": project.get("element_sheet")}


@app.post("/projects/{project_id}/show-trial/generate")
def generate_show_trial_endpoint(project_id: str, payload: ShowTrialGenerateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
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
    trial = build_show_trial_package(project, payload.include_walkthrough, payload.include_audio_video, payload.include_camera_pan)
    trial["draft_name"] = payload.draft_name or f"{project.get('name') or 'Project'} Trial"
    orchestration = project.get("orchestration_data") or {}
    orchestration["show_trial"] = trial
    project = update_project_fields(project_id, user_id, {"orchestration_data": orchestration})
    queued_jobs = []
    if payload.queue_render_jobs:
        walkthrough_asset = create_project_asset(project_id, user_id, "walkthrough_preview", trial["draft_name"] + " Walkthrough", prompt="3D walkthrough preview", section="show_trial", job_kind="walkthrough_trial", status="queued", meta={"trial": True})
        queued_jobs.append(queue_agent_job_with_activity(project_id, user_id, "walkthrough_agent", "walkthrough_trial", walkthrough_asset["title"], input_data={"asset_id": walkthrough_asset["id"], "trial": trial}))
    add_project_activity(project_id, user_id, "show_trial.generated", trial["draft_name"], detail="3D walkthrough + live show trial prepared for pre-show review.", meta={"queued_jobs": [j.get('id') for j in queued_jobs]})
    return {"message": "Show trial generated", "project_id": project_id, "show_trial": trial, "jobs": queued_jobs, "project": project}


@app.get("/projects/{project_id}/show-trial")
def get_show_trial_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    orchestration = project.get("orchestration_data") or {}
    return {"project_id": project_id, "show_trial": orchestration.get("show_trial")}


@app.post("/projects/{project_id}/show-trial/update")
def update_show_trial_endpoint(project_id: str, payload: ShowTrialUpdateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    orchestration = project.get("orchestration_data") or {}
    existing = orchestration.get("show_trial") or {}
    updated_trial = {**existing, **(payload.trial_data or {})}
    orchestration["show_trial"] = updated_trial
    project = update_project_fields(project_id, user_id, {"orchestration_data": orchestration})
    add_project_activity(project_id, user_id, "show_trial.updated", updated_trial.get("draft_name") or "Show trial updated")
    return {"message": "Show trial updated", "show_trial": updated_trial, "project": project}


@app.post("/projects/{project_id}/show-trial/finalize")
def finalize_show_trial_endpoint(project_id: str, payload: ShowTrialFinalizeInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    orchestration = project.get("orchestration_data") or {}
    trial = orchestration.get("show_trial")
    if not trial:
        raise HTTPException(status_code=404, detail="Show trial not found")
    showrunner = project.get("showrunner_data") or _default_showrunner_plan(project)
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
    add_project_activity(project_id, user_id, "show_trial.finalized", trial.get("draft_name") or "Show trial finalized", detail="Trial cues promoted to final show running option.")
    return {"message": "Show trial finalized", "show_trial": trial, "project": project}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)
