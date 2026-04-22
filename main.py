import os
import re
import json
import uuid
import socket
import struct
import datetime as dt
from pathlib import Path
from contextlib import asynccontextmanager
from html import escape
from typing import Any, Dict, List, Optional, Tuple

import requests
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator
from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Form,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ------------------------------------------------------------------------------
# ENV
# ------------------------------------------------------------------------------
load_dotenv()

APP_TITLE = os.getenv("APP_TITLE", "AI Creative Studio API").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
SECRET_KEY = os.getenv("SECRET_KEY", "").strip()
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4.1").strip()
ACCESS_TOKEN_HOURS = int(os.getenv("ACCESS_TOKEN_HOURS", "24") or "24")
JWT_ALGORITHM = "HS256"

ALLOWED_ORIGINS = [
    x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()
] or ["*"]

MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "/tmp/ai_creative_media")).resolve()
RENDER_OUTPUT_DIR = Path(os.getenv("RENDER_OUTPUT_DIR", "/tmp/ai_creative_renders")).resolve()
VOICE_DIR = Path(os.getenv("VOICE_DIR", str(MEDIA_DIR / "voice"))).resolve()
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(MEDIA_DIR / "uploads"))).resolve()
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
RENDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_BASE_URL = (
    os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    or (f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME', '').strip()}" if os.getenv("RENDER_EXTERNAL_HOSTNAME") else "")
)

EMAIL_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.I)

TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts").strip()
TTS_VOICE = os.getenv("TTS_VOICE", "coral").strip()
TTS_RESPONSE_FORMAT = os.getenv("TTS_RESPONSE_FORMAT", "mp3").strip().lower()
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe").strip()
VOICE_DEFAULT_INSTRUCTIONS = os.getenv(
    "VOICE_DEFAULT_INSTRUCTIONS",
    "Speak like a warm, realistic female AI assistant with natural pacing, clear diction, friendly tone, and smooth conversational delivery. Keep the speech confident, premium, and human-like.",
).strip()
VOICE_SYSTEM_PROMPT = os.getenv(
    "VOICE_SYSTEM_PROMPT",
    "You are the voice assistant for an AI creative studio and live show control app. Be helpful, warm, concise, production-aware, and practical. Speak like an experienced female creative producer who can help with concepts, show flow, light, sound, screen control, and production decisions.",
).strip()

SHOW_GATEWAY_URL = os.getenv("SHOW_GATEWAY_URL", "").strip().rstrip("/")
COMPANION_BASE_URL = os.getenv("COMPANION_BASE_URL", "").strip().rstrip("/")
QLAB_OSC_IP = os.getenv("QLAB_OSC_IP", "").strip()
QLAB_OSC_PORT = int(os.getenv("QLAB_OSC_PORT", "0") or "0")
EOS_OSC_IP = os.getenv("EOS_OSC_IP", "").strip()
EOS_OSC_PORT = int(os.getenv("EOS_OSC_PORT", "0") or "0")
RESOLUME_BASE_URL = os.getenv("RESOLUME_BASE_URL", "").strip().rstrip("/")

client = None
pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def now_iso() -> str:
    return utcnow().isoformat()


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def load_json(value: Any) -> Any:
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


def prepare_database_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if not url:
        raise HTTPException(status_code=500, detail="DATABASE_URL missing")
    if not url.startswith("postgresql://") and not url.startswith("postgres://"):
        raise HTTPException(status_code=500, detail="DATABASE_URL must be a PostgreSQL connection string")
    if "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


def require_secret_key() -> str:
    if not SECRET_KEY:
        raise HTTPException(status_code=500, detail="SECRET_KEY missing")
    return SECRET_KEY


def get_conn():
    return psycopg.connect(
        prepare_database_url(DATABASE_URL),
        row_factory=dict_row,
        autocommit=True,
    )


def with_db(fn):
    def wrapper(*args, **kwargs):
        with get_conn() as conn:
            with conn.cursor() as cur:
                return fn(cur, *args, **kwargs)
    return wrapper


def hash_password(password: str) -> str:
    return pwd.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd.verify(password, hashed)


def create_access_token(user_id: str) -> str:
    secret = require_secret_key()
    payload = {
        "user_id": str(user_id),
        "iat": int(utcnow().timestamp()),
        "exp": int((utcnow() + dt.timedelta(hours=ACCESS_TOKEN_HOURS)).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, require_secret_key(), algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return str(user_id)


def relative_public_url(file_path: Path) -> str:
    resolved = file_path.resolve()
    for base, prefix in [
        (MEDIA_DIR, "/media"),
        (RENDER_OUTPUT_DIR, "/renders"),
    ]:
        try:
            rel = resolved.relative_to(base)
            return f"{prefix}/{rel.as_posix()}"
        except Exception:
            continue
    raise HTTPException(status_code=500, detail="Unable to create public URL for file")


def absolute_public_url(relative_url: str) -> str:
    if relative_url.startswith("http://") or relative_url.startswith("https://"):
        return relative_url
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}{relative_url}"
    return relative_url


def ensure_uuid(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(uuid.UUID(str(value)))
    except Exception:
        raise HTTPException(status_code=422, detail=f"{field_name} must be a valid UUID")


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def project_row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(row)
    for key in (
        "concepts", "selected", "images", "render3d", "scene_json", "deliverables",
        "dimensions", "brand_data", "presentation_data", "sound_data", "lighting_data",
        "showrunner_data", "department_outputs",
    ):
        if key in item:
            item[key] = load_json(item.get(key))
    return item


def voice_message_row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    return dict(row)


# ------------------------------------------------------------------------------
# DATABASE SETUP
# ------------------------------------------------------------------------------
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
                    role text not null default 'user',
                    is_active boolean not null default true,
                    created_at timestamptz not null default now()
                );
            """)

            cur.execute("""
                create table if not exists public.projects (
                    id uuid primary key,
                    user_id uuid not null references public.users(id) on delete cascade,
                    name text not null default 'Untitled Project',
                    event_type text,
                    style_direction text,
                    status text not null default 'draft',
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
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                );
            """)

            for stmt in [
                "alter table public.projects add column if not exists style_direction text;",
                "alter table public.projects add column if not exists brief text;",
                "alter table public.projects add column if not exists analysis text;",
                "alter table public.projects add column if not exists concepts jsonb;",
                "alter table public.projects add column if not exists selected jsonb;",
                "alter table public.projects add column if not exists sound_data jsonb;",
                "alter table public.projects add column if not exists lighting_data jsonb;",
                "alter table public.projects add column if not exists showrunner_data jsonb;",
                "alter table public.projects add column if not exists department_outputs jsonb;",
                "alter table public.projects add column if not exists updated_at timestamptz not null default now();",
            ]:
                cur.execute(stmt)

            cur.execute("""
                create table if not exists public.project_versions (
                    id uuid primary key,
                    project_id uuid not null references public.projects(id) on delete cascade,
                    user_id uuid not null references public.users(id) on delete cascade,
                    version_no int not null,
                    snapshot jsonb,
                    note text,
                    created_at timestamptz not null default now()
                );
            """)

            cur.execute("""
                create table if not exists public.project_comments (
                    id uuid primary key,
                    project_id uuid not null references public.projects(id) on delete cascade,
                    user_id uuid not null references public.users(id) on delete cascade,
                    section text,
                    comment_text text,
                    status text not null default 'open',
                    created_at timestamptz not null default now()
                );
            """)

            cur.execute("""
                create table if not exists public.voice_sessions (
                    id uuid primary key,
                    user_id uuid not null references public.users(id) on delete cascade,
                    project_id uuid references public.projects(id) on delete set null,
                    title text not null default 'Voice Session',
                    system_prompt text,
                    voice text,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                );
            """)

            cur.execute("""
                create table if not exists public.voice_messages (
                    id uuid primary key,
                    session_id uuid not null references public.voice_sessions(id) on delete cascade,
                    role text not null,
                    content text,
                    transcript text,
                    audio_url text,
                    meta jsonb,
                    created_at timestamptz not null default now()
                );
            """)


# ------------------------------------------------------------------------------
# AUTH + USERS
# ------------------------------------------------------------------------------
@with_db
def get_user_by_email(cur, email: str) -> Optional[Dict[str, Any]]:
    cur.execute(
        """
        select id, email, password, full_name, role, is_active, created_at
        from public.users
        where lower(email) = lower(%s)
        limit 1
        """,
        (email,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


@with_db
def get_user_by_id(cur, user_id: str) -> Optional[Dict[str, Any]]:
    cur.execute(
        """
        select id, email, full_name, role, is_active, created_at
        from public.users
        where id = %s
        limit 1
        """,
        (user_id,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


@with_db
def create_user(cur, email: str, password: str, full_name: Optional[str]) -> Dict[str, Any]:
    existing = get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    user_id = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.users (id, email, password, full_name)
        values (%s, %s, %s, %s)
        """,
        (user_id, email, hash_password(password), full_name),
    )
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=500, detail="User creation failed")
    return user


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> Dict[str, Any]:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    user_id = decode_access_token(credentials.credentials)
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists for this token")
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="User is inactive")
    return user


# ------------------------------------------------------------------------------
# PROJECTS
# ------------------------------------------------------------------------------
@with_db
def create_project(cur, user_id: str, name: str, brief: Optional[str], event_type: Optional[str], style_direction: Optional[str]) -> Dict[str, Any]:
    project_id = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.projects (id, user_id, name, brief, event_type, style_direction, status)
        values (%s, %s, %s, %s, %s, %s, %s)
        """,
        (project_id, user_id, name or "Untitled Project", brief, event_type, style_direction, "draft"),
    )
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=500, detail="Project creation failed")
    return project


@with_db
def list_projects(cur, user_id: str) -> List[Dict[str, Any]]:
    cur.execute(
        """
        select *
        from public.projects
        where user_id = %s
        order by created_at desc
        """,
        (user_id,),
    )
    return [project_row_to_dict(r) for r in cur.fetchall()]


@with_db
def get_project_by_id(cur, project_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if user_id:
        cur.execute(
            """
            select *
            from public.projects
            where id = %s and user_id = %s
            limit 1
            """,
            (project_id, user_id),
        )
    else:
        cur.execute(
            """
            select *
            from public.projects
            where id = %s
            limit 1
            """,
            (project_id,),
        )
    row = cur.fetchone()
    return project_row_to_dict(row) if row else None


@with_db
def update_project_fields(cur, project_id: str, user_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
    if not values:
        project = get_project_by_id(project_id, user_id=user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project

    allowed = {
        "name", "event_type", "style_direction", "status", "brief", "analysis", "concepts", "selected",
        "moodboard", "images", "render3d", "cad", "scene_json", "deliverables", "dimensions",
        "brand_data", "presentation_data", "sound_data", "lighting_data", "showrunner_data",
        "department_outputs",
    }

    clean = {k: v for k, v in values.items() if k in allowed}
    if not clean:
        raise HTTPException(status_code=400, detail="No valid project fields supplied")

    assignments = []
    params: List[Any] = []
    for key, value in clean.items():
        assignments.append(f"{key} = %s")
        params.append(dump_json(value) if isinstance(value, (dict, list)) else value)

    assignments.append("updated_at = now()")
    params.extend([project_id, user_id])

    cur.execute(
        f"update public.projects set {', '.join(assignments)} where id = %s and user_id = %s",
        params,
    )

    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@with_db
def snapshot_project_version(cur, project_id: str, user_id: str, note: str = "") -> None:
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        return
    cur.execute(
        """
        select coalesce(max(version_no), 0) + 1 as next_version
        from public.project_versions
        where project_id = %s
        """,
        (project_id,),
    )
    next_version = int(cur.fetchone()["next_version"])
    cur.execute(
        """
        insert into public.project_versions (id, project_id, user_id, version_no, snapshot, note)
        values (%s, %s, %s, %s, %s, %s)
        """,
        (str(uuid.uuid4()), project_id, user_id, next_version, dump_json(project), note),
    )


@with_db
def add_comment(cur, project_id: str, user_id: str, section: str, comment_text: str) -> Dict[str, Any]:
    comment_id = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.project_comments (id, project_id, user_id, section, comment_text)
        values (%s, %s, %s, %s, %s)
        """,
        (comment_id, project_id, user_id, section, comment_text),
    )
    cur.execute("select * from public.project_comments where id = %s", (comment_id,))
    return dict(cur.fetchone())


@with_db
def list_comments(cur, project_id: str, user_id: str) -> List[Dict[str, Any]]:
    cur.execute(
        """
        select c.*
        from public.project_comments c
        join public.projects p on p.id = c.project_id
        where c.project_id = %s and p.user_id = %s
        order by c.created_at desc
        """,
        (project_id, user_id),
    )
    return [dict(r) for r in cur.fetchall()]


# ------------------------------------------------------------------------------
# VOICE SESSIONS
# ------------------------------------------------------------------------------
@with_db
def create_voice_session(cur, user_id: str, project_id: Optional[str], title: Optional[str], system_prompt: Optional[str], voice: Optional[str]) -> Dict[str, Any]:
    session_id = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.voice_sessions (id, user_id, project_id, title, system_prompt, voice)
        values (%s, %s, %s, %s, %s, %s)
        """,
        (
            session_id,
            user_id,
            project_id,
            title or "Voice Session",
            system_prompt or VOICE_SYSTEM_PROMPT,
            voice or TTS_VOICE,
        ),
    )
    return get_voice_session_by_id(session_id, user_id)


@with_db
def get_voice_session_by_id(cur, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    cur.execute(
        """
        select *
        from public.voice_sessions
        where id = %s and user_id = %s
        limit 1
        """,
        (session_id, user_id),
    )
    row = cur.fetchone()
    return dict(row) if row else None


@with_db
def touch_voice_session(cur, session_id: str, user_id: str, voice: Optional[str] = None) -> None:
    if voice:
        cur.execute(
            """
            update public.voice_sessions
            set updated_at = now(), voice = %s
            where id = %s and user_id = %s
            """,
            (voice, session_id, user_id),
        )
    else:
        cur.execute(
            """
            update public.voice_sessions
            set updated_at = now()
            where id = %s and user_id = %s
            """,
            (session_id, user_id),
        )


@with_db
def add_voice_message(cur, session_id: str, role: str, content: Optional[str], transcript: Optional[str] = None, audio_url: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    message_id = str(uuid.uuid4())
    cur.execute(
        """
        insert into public.voice_messages (id, session_id, role, content, transcript, audio_url, meta)
        values (%s, %s, %s, %s, %s, %s, %s)
        """,
        (message_id, session_id, role, content, transcript, audio_url, dump_json(meta or {})),
    )
    cur.execute("select * from public.voice_messages where id = %s", (message_id,))
    return voice_message_row_to_dict(cur.fetchone())


@with_db
def get_voice_messages(cur, session_id: str, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    cur.execute(
        """
        select m.*
        from public.voice_messages m
        join public.voice_sessions s on s.id = m.session_id
        where m.session_id = %s and s.user_id = %s
        order by m.created_at asc
        limit %s
        """,
        (session_id, user_id, limit),
    )
    return [voice_message_row_to_dict(r) for r in cur.fetchall()]


@with_db
def list_voice_sessions(cur, user_id: str) -> List[Dict[str, Any]]:
    cur.execute(
        """
        select *
        from public.voice_sessions
        where user_id = %s
        order by updated_at desc, created_at desc
        limit 100
        """,
        (user_id,),
    )
    return [dict(r) for r in cur.fetchall()]


# ------------------------------------------------------------------------------
# AI / LLM HELPERS
# ------------------------------------------------------------------------------
def get_openai_client():
    global client
    if client is not None:
        return client
    if OPENAI_API_KEY and OpenAI is not None:
        client = OpenAI(api_key=OPENAI_API_KEY)
    return client


def llm_text(system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    api = get_openai_client()
    if api is None:
        raise RuntimeError("OpenAI not configured")
    response = api.chat.completions.create(
        model=TEXT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def llm_json(system_prompt: str, user_prompt: str) -> Any:
    raw = llm_text(system_prompt, user_prompt, temperature=0.2)
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def llm_chat(history: List[Dict[str, str]], temperature: float = 0.6) -> str:
    api = get_openai_client()
    if api is None:
        raise RuntimeError("OpenAI not configured")
    response = api.chat.completions.create(
        model=TEXT_MODEL,
        temperature=temperature,
        messages=history,
    )
    return (response.choices[0].message.content or "").strip()


def fallback_analysis(brief: str, event_type: Optional[str]) -> str:
    et = event_type or "Event"
    return (
        f"**Project Analysis: {et}**\n\n"
        f"**1. Event Objective**\nCreate a premium, high-impact experience aligned to the brief.\n\n"
        f"**2. Audience Profile**\nVIP guests, media, partners, and invited stakeholders.\n\n"
        f"**3. Creative Direction**\nSophisticated, immersive, brand-led, and reveal-focused.\n\n"
        f"**4. Production Needs**\nStage, LED content, sound cues, lighting cues, show flow, and backstage coordination.\n\n"
        f"**5. Key Brief Input**\n{brief}\n"
    )


def analyze_brief(brief: str, event_type: Optional[str]) -> str:
    prompt = f"""
Analyze this event brief and return a practical structured write-up with:
1. Event objective
2. Audience profile
3. Brand tone
4. Spatial and staging needs
5. Sound implications
6. Lighting implications
7. Show-flow implications
8. Production risks
9. Recommended next steps

Event Type: {event_type or 'Not specified'}
Brief:
{brief}
    """.strip()

    try:
        return llm_text(
            "You are a senior experiential creative director for premium events, launches, exhibitions, and stage productions.",
            prompt,
            temperature=0.5,
        )
    except Exception:
        return fallback_analysis(brief, event_type)


def fallback_concepts(brief: str, event_type: Optional[str]) -> List[Dict[str, Any]]:
    base_event = event_type or "Event"
    return [
        {
            "name": "Infinite Reflection",
            "summary": f"A 360° immersive {base_event.lower()} environment with mirrored surfaces, glossy finishes, choreographed light, and premium reveal energy.",
            "style": "Futuristic minimalism with high-gloss finishes and clean lines",
            "materials": ["mirror acrylic", "gloss laminate", "smoked glass", "black truss"],
            "colors": ["black", "silver", "white"],
            "lighting": "Programmable LED, moving heads, architectural uplighting, pinspot accents",
            "stage_elements": ["hero reveal portal", "curved LED wall", "central turntable", "VIP runway"],
            "camera_style": "Wide cinematic hero shots with dramatic reveal timing",
        },
        {
            "name": "Velocity Chamber",
            "summary": f"A high-energy {base_event.lower()} concept inspired by speed, precision, and automotive engineering.",
            "style": "Bold industrial luxury with linear lighting and sculpted geometry",
            "materials": ["brushed metal", "matte black panels", "LED linears", "fabric walling"],
            "colors": ["gunmetal", "black", "electric white"],
            "lighting": "Sharp beam language, chase effects, and rhythm-based cueing",
            "stage_elements": ["angular proscenium", "kinetic LED strips", "engineered product plinth"],
            "camera_style": "Dynamic motion-driven angles with strong contrast",
        },
        {
            "name": "Prestige Horizon",
            "summary": f"A refined premium hospitality-led {base_event.lower()} concept blending luxury lounge mood with a hero reveal moment.",
            "style": "Luxury premium with warm textures, clean architecture, and immersive storytelling",
            "materials": ["wood veneer", "champagne metal", "velvet seating", "translucent panels"],
            "colors": ["champagne gold", "warm white", "deep charcoal"],
            "lighting": "Layered warm washes, scenic highlights, reveal specials, and audience ambience",
            "stage_elements": ["hospitality pods", "hero façade", "premium backdrop", "content-driven reveal zone"],
            "camera_style": "Elegant premium framing with hospitality atmosphere",
        },
    ]


def generate_concepts(brief: str, analysis: str, event_type: Optional[str]) -> List[Dict[str, Any]]:
    prompt = f"""
Return exactly 3 strong creative concepts as a JSON array.
Each concept object must contain:
name, summary, style, materials, colors, lighting, stage_elements, camera_style

Event Type: {event_type or 'Not specified'}

Analysis:
{analysis}

Brief:
{brief}
    """.strip()

    try:
        result = llm_json(
            "You are a senior creative strategist. Return JSON only.",
            prompt,
        )
        if not isinstance(result, list) or len(result) < 3:
            raise ValueError("Invalid concepts JSON")
        return result[:3]
    except Exception:
        return fallback_concepts(brief, event_type)


def _default_sound_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    concept = load_json(project.get("selected")) or {}
    project_name = project.get("name") or "Untitled Project"
    event_type = project.get("event_type") or "Event"
    return {
        "concept": {
            "name": concept.get("name", "Sound Direction"),
            "summary": f"Premium immersive sound design for {project_name}, tightly supporting reveal moments, VIP arrivals, walk-ins, speeches, and hero content."
        },
        "system_design": {
            "console": "Yamaha CL5 Digital Mixing Console (FOH) with Dante network",
            "pa": "L/R line array with front fills, delay speakers if venue depth requires, and controlled sub reinforcement",
            "monitoring": "Presenter foldback + IFB / cue wedge for stage manager"
        },
        "speaker_plan": [
            "Main L/R hangs aligned to guest bowl",
            "Front fills across stage apron",
            "Sub array centered or cardioid depending venue restrictions",
            "Delay zones for rear audience if required"
        ],
        "input_list": [
            "Host wireless handheld x2",
            "Panel/lapel wireless x6",
            "Playback L/R from show laptop",
            "Emergency backup playback source",
            "Press mult box feed"
        ],
        "mic_plan": [
            "Primary wireless handheld",
            "Backup wireless handheld",
            "Lapel mics for presenters",
            "Podium mic if podium is used"
        ],
        "patch_sheet": [
            "CH1 Host HH A",
            "CH2 Host HH B",
            "CH3-8 Lavalier pack inputs",
            "CH9-10 Playback L/R",
            "CH11-12 Backup playback L/R"
        ],
        "playback_cues": [
            "Walk-in ambience",
            "VIP arrival sting",
            "Opening cue",
            "Reveal impact cue",
            "Speech beds",
            "Exit music"
        ],
        "staffing": [
            "FOH engineer",
            "Monitor / systems tech",
            "RF tech",
            "Playback operator"
        ],
        "rehearsal_notes": [
            "Soundcheck all presenter mics",
            "Verify cue timing with lighting and show caller",
            "Run full reveal sequence twice"
        ],
        "risk_notes": [
            "Keep redundant playback systems ready",
            "Maintain spare RF channels and batteries",
            "Hold emergency speech mic side-stage"
        ],
        "pdf_sections": [
            {"heading": "Overview", "body": f"Sound design manual for {project_name} ({event_type})."},
            {"heading": "System Design", "body": json.dumps({
                "system_design": {
                    "console": "Yamaha CL5 Digital Mixing Console (FOH) with Dante network",
                    "pa": "L/R line array with fills and subs",
                    "monitoring": "Presenter foldback + cue monitor"
                }
            }, indent=2)},
            {"heading": "Cue Plan", "body": "Walk-in ambience\nVIP sting\nOpening cue\nReveal impact cue\nSpeech beds\nExit music"},
            {"heading": "Risk Management", "body": "Backup playback, spare RF, emergency handheld, and coordinated rehearsal timing."},
        ],
    }


def _default_lighting_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    concept = load_json(project.get("selected")) or {}
    project_name = project.get("name") or "Untitled Project"
    return {
        "concept": {
            "name": concept.get("name", "Lighting Direction"),
            "summary": f"Layered lighting design for {project_name} with premium arrival mood, reveal impact, and clean presenter visibility."
        },
        "fixture_list": [
            "Moving heads (spot/profile)",
            "Wash fixtures",
            "LED battens / linears",
            "Audience blinders",
            "Pinspots / specials"
        ],
        "truss_plan": [
            "Front truss for key light",
            "Mid truss for texture and movement",
            "Upstage truss for reveal silhouette and content integration"
        ],
        "dmx_notes": [
            "Separate universes for movers, washes, LED scenic, and practicals",
            "Address reveal fixtures on dedicated cue group"
        ],
        "scene_cues": [
            "Guest arrival look",
            "Pre-show hold",
            "Opening cue",
            "Reveal blackout to impact",
            "Vehicle hero look",
            "Speech state",
            "Media photo state",
            "Exit state"
        ],
        "looks": [
            "Warm premium ambience",
            "Cool reveal contrast",
            "Crisp presenter white",
            "High-impact movement for hero moments"
        ],
        "operator_notes": [
            "Keep presenter key light stable",
            "Coordinate blackout timings with sound and video",
            "Hold manual override for reveal sequence"
        ],
        "rehearsal_notes": [
            "Program cues with show caller",
            "Validate sightlines and camera exposure",
            "Test emergency full worklight state"
        ],
        "fallback_plan": [
            "Safe white speech look",
            "Backup reveal cue",
            "Manual fader page for emergency operation"
        ],
        "pdf_sections": [
            {"heading": "Overview", "body": f"Lighting design manual for {project_name}."},
            {"heading": "Fixture Strategy", "body": "Moving heads, washes, linears, blinders, and architectural scenic accents."},
            {"heading": "Cue Sequence", "body": "Arrival look\nOpening cue\nReveal impact\nHero look\nSpeech state\nExit state"},
            {"heading": "Fallback Plan", "body": "Safe white state, manual override, worklight state, and backup reveal cue."},
        ],
    }


def _default_showrunner_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    project_name = project.get("name") or "Untitled Project"
    return {
        "running_order": [
            "Guest arrivals",
            "Pre-show hold",
            "Opening welcome",
            "Brand film",
            "Hero reveal",
            "Leadership speech",
            "Media/photo moment",
            "Networking / hospitality"
        ],
        "cue_script": [
            "Standby sound, lights, video for guest-open",
            "Go guest-open",
            "Standby reveal sequence",
            "Go blackout",
            "Go reveal audio",
            "Go hero light",
            "Go brand film",
            "Go CEO walk-on"
        ],
        "standby_calls": [
            "Standby LX 1 / SND 1 / VDO 1",
            "Standby reveal sequence",
            "Standby speech state"
        ],
        "go_calls": [
            "Go LX 1",
            "Go SND 1",
            "Go reveal",
            "Go speech state"
        ],
        "departmental_dependencies": {
            "sound": "Reveal audio cue must lock to lighting and video",
            "lighting": "Blackout and hero specials must follow stage clear",
            "video": "Brand content and reveal animation must stay in sync with show caller"
        },
        "delay_protocol": [
            "Hold audience in ambient loop",
            "Notify departments on comms",
            "Shift to backup intro if hero asset is delayed"
        ],
        "emergency_protocol": [
            "Cut to safe worklight and stop playback if stage safety is compromised",
            "Stage manager takes hard control over all departments"
        ],
        "rehearsal_flow": [
            "Technical rehearsal",
            "Cue-to-cue",
            "Full dress run",
            "VIP walk-through"
        ],
        "console_cues": [
            {
                "cue_no": 1,
                "name": "Guest Open",
                "standby": "Standby ambience and venue open",
                "go": "Open venue ambience",
                "actions": []
            },
            {
                "cue_no": 2,
                "name": "Opening",
                "standby": "Standby opening cue",
                "go": "Start opening cue",
                "actions": []
            },
            {
                "cue_no": 3,
                "name": "Reveal",
                "standby": "Standby synchronized reveal",
                "go": "Execute synchronized reveal",
                "actions": []
            },
            {
                "cue_no": 4,
                "name": "Speech",
                "standby": "Standby speech state",
                "go": "Set speech state",
                "actions": []
            },
        ],
        "pdf_sections": [
            {"heading": "Overview", "body": f"Show running script for {project_name}."},
            {"heading": "Running Order", "body": "Guest arrivals\nOpening\nBrand film\nHero reveal\nSpeech\nMedia moment\nNetworking"},
            {"heading": "Cue Script", "body": "Standby departments\nGo opening\nStandby reveal\nGo reveal\nGo speech state"},
            {"heading": "Delay & Emergency", "body": "Use ambient hold, notify all departments, switch to safe state if needed."},
        ],
    }


def generate_sound_department(project: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
Return JSON only.
Create a complete sound department plan for this event and selected concept.
Project:
{dump_json(project)}
    """.strip()
    try:
        result = llm_json("You are a senior live event sound engineer. Return valid JSON only.", prompt)
        if not isinstance(result, dict):
            raise ValueError("Invalid sound JSON")
        result.setdefault("pdf_sections", _default_sound_plan(project)["pdf_sections"])
        return result
    except Exception:
        return _default_sound_plan(project)


def generate_lighting_department(project: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
Return JSON only.
Create a complete lighting department plan for this event and selected concept.
Project:
{dump_json(project)}
    """.strip()
    try:
        result = llm_json("You are a senior lighting designer. Return valid JSON only.", prompt)
        if not isinstance(result, dict):
            raise ValueError("Invalid lighting JSON")
        result.setdefault("pdf_sections", _default_lighting_plan(project)["pdf_sections"])
        return result
    except Exception:
        return _default_lighting_plan(project)


def generate_showrunner_department(project: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
Return JSON only.
Create a complete show running plan for this event and selected concept.
Project:
{dump_json(project)}
    """.strip()
    try:
        result = llm_json("You are a senior show caller and stage manager. Return valid JSON only.", prompt)
        if not isinstance(result, dict):
            raise ValueError("Invalid showrunner JSON")
        result.setdefault("pdf_sections", _default_showrunner_plan(project)["pdf_sections"])
        if not result.get("console_cues"):
            result["console_cues"] = _default_showrunner_plan(project)["console_cues"]
        for cue in result.get("console_cues", []):
            cue.setdefault("actions", [])
        return result
    except Exception:
        return _default_showrunner_plan(project)


def build_voice_system_prompt(current_user: Dict[str, Any], session: Optional[Dict[str, Any]] = None, project: Optional[Dict[str, Any]] = None) -> str:
    parts = [VOICE_SYSTEM_PROMPT]
    if current_user.get("full_name"):
        parts.append(f"User name: {current_user['full_name']}")
    if session and session.get("system_prompt"):
        parts.append(str(session["system_prompt"]))
    if project:
        parts.append(f"Current project name: {project.get('name') or 'Untitled Project'}")
        if project.get("event_type"):
            parts.append(f"Event type: {project['event_type']}")
        if project.get("style_direction"):
            parts.append(f"Style direction: {project['style_direction']}")
        if project.get("brief"):
            parts.append(f"Project brief: {project['brief']}")
        if project.get("analysis"):
            parts.append(f"Project analysis: {project['analysis']}")
        selected = load_json(project.get("selected")) or {}
        if selected:
            parts.append(f"Selected concept: {dump_json(selected)}")
    return "\n\n".join([p for p in parts if p])


def transcribe_audio_file(file_path: Path) -> str:
    api = get_openai_client()
    if api is None:
        raise HTTPException(status_code=500, detail="OpenAI not configured for transcription")
    with open(file_path, "rb") as audio_file:
        result = api.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=audio_file,
            response_format="text",
        )
    text = getattr(result, "text", None) or (result if isinstance(result, str) else str(result))
    text = str(text).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Transcription returned empty text")
    return text


def synthesize_speech(text: str, voice: Optional[str] = None, instructions: Optional[str] = None, filename_prefix: str = "assistant") -> Dict[str, str]:
    api = get_openai_client()
    if api is None:
        raise HTTPException(status_code=500, detail="OpenAI not configured for text-to-speech")

    final_voice = (voice or TTS_VOICE or "coral").strip()
    final_instructions = (instructions or VOICE_DEFAULT_INSTRUCTIONS).strip()
    ext = TTS_RESPONSE_FORMAT if TTS_RESPONSE_FORMAT in {"mp3", "wav", "opus", "aac", "flac", "pcm"} else "mp3"
    out_path = VOICE_DIR / f"{filename_prefix}_{uuid.uuid4().hex}.{ext}"

    with api.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=final_voice,
        input=text[:4096],
        instructions=final_instructions,
        response_format=ext,
    ) as response:
        response.stream_to_file(out_path)

    rel = relative_public_url(out_path)
    return {
        "audio_path": rel,
        "audio_url": absolute_public_url(rel),
        "voice": final_voice,
        "response_format": ext,
    }


def build_chat_history(system_prompt: str, prior_messages: List[Dict[str, Any]], user_text: str) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in prior_messages[-20:]:
        role = (msg.get("role") or "user").strip().lower()
        if role not in {"user", "assistant", "system"}:
            continue
        content = clean_text(msg.get("content") or msg.get("transcript"))
        if content:
            history.append({"role": role, "content": content})
    history.append({"role": "user", "content": user_text})
    return history


def generate_voice_reply(current_user: Dict[str, Any], session: Optional[Dict[str, Any]], project: Optional[Dict[str, Any]], user_text: str, history_rows: List[Dict[str, Any]]) -> str:
    system_prompt = build_voice_system_prompt(current_user, session=session, project=project)
    messages = build_chat_history(system_prompt, history_rows, user_text)
    try:
        return llm_chat(messages, temperature=0.7)
    except Exception:
        return f"I understood you said: {user_text}. I can help you with creative planning, show flow, lighting, sound, screen control, and production tasks."


# ------------------------------------------------------------------------------
# SHOW CONTROL HELPERS
# ------------------------------------------------------------------------------
def _osc_pad(data: bytes) -> bytes:
    return data + (b"\x00" * ((4 - len(data) % 4) % 4))


def send_osc_message(ip: str, port: int, address: str, args: Optional[List[Any]] = None) -> Dict[str, Any]:
    if not ip or not port or not address:
        raise HTTPException(status_code=400, detail="OSC action requires ip, port, and address")

    args = args or []
    packet = _osc_pad(address.encode("utf-8") + b"\x00")
    type_tags = ","
    encoded_args = b""

    for arg in args:
        if isinstance(arg, bool):
            type_tags += "T" if arg else "F"
        elif isinstance(arg, int):
            type_tags += "i"
            encoded_args += struct.pack(">i", arg)
        elif isinstance(arg, float):
            type_tags += "f"
            encoded_args += struct.pack(">f", arg)
        else:
            type_tags += "s"
            encoded_args += _osc_pad(str(arg).encode("utf-8") + b"\x00")

    packet += _osc_pad(type_tags.encode("utf-8") + b"\x00") + encoded_args
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(packet, (ip, int(port)))
    finally:
        sock.close()
    return {"ok": True, "protocol": "osc", "ip": ip, "port": int(port), "address": address, "args": args}


def default_target_connection(target: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    t = (target or "").strip().lower()
    if t == "qlab":
        return (QLAB_OSC_IP or None, QLAB_OSC_PORT or None, None)
    if t == "eos":
        return (EOS_OSC_IP or None, EOS_OSC_PORT or None, None)
    if t == "resolume":
        return (None, None, RESOLUME_BASE_URL or None)
    if t == "companion":
        return (None, None, COMPANION_BASE_URL or None)
    return (None, None, None)


def execute_control_action(action: Dict[str, Any]) -> Dict[str, Any]:
    protocol = str(action.get("protocol") or "").strip().lower()
    target = str(action.get("target") or "").strip().lower()

    default_ip, default_port, default_base_url = default_target_connection(target)

    if protocol == "osc":
        ip = action.get("ip") or default_ip
        port = int(action.get("port") or default_port or 0)
        address = action.get("address")
        args = action.get("args") or []
        return send_osc_message(ip=ip, port=port, address=address, args=args)

    if protocol in {"rest", "http"}:
        base_url = (action.get("base_url") or default_base_url or "").rstrip("/")
        path = str(action.get("path") or "").strip()
        if not base_url:
            raise HTTPException(status_code=400, detail=f"REST action for target '{target}' requires base_url or matching environment variable")
        if not path.startswith("/"):
            path = f"/{path}" if path else ""
        method = str(action.get("method") or "POST").upper()
        headers = action.get("headers") or {}
        params = action.get("params") or None
        body = action.get("body")
        response = requests.request(
            method,
            f"{base_url}{path}",
            headers=headers,
            params=params,
            json=body,
            timeout=8,
        )
        return {
            "ok": response.ok,
            "protocol": "rest",
            "status_code": response.status_code,
            "target": target,
            "url": f"{base_url}{path}",
            "response_text": response.text[:500],
        }

    if protocol == "gateway":
        if not SHOW_GATEWAY_URL:
            raise HTTPException(status_code=400, detail="SHOW_GATEWAY_URL missing for gateway actions")
        response = requests.post(f"{SHOW_GATEWAY_URL}/execute", json=action, timeout=8)
        return {
            "ok": response.ok,
            "protocol": "gateway",
            "status_code": response.status_code,
            "target": target,
            "response_text": response.text[:500],
        }

    raise HTTPException(status_code=400, detail=f"Unsupported control protocol: {protocol}")


def get_console_state(project: Dict[str, Any]) -> Dict[str, Any]:
    outputs = load_json(project.get("department_outputs")) or {}
    outputs.setdefault("console_index", 0)
    outputs.setdefault("armed", False)
    outputs.setdefault("hold", False)
    outputs.setdefault("last_status", "idle")
    outputs.setdefault("execution_log", [])
    return outputs


def save_console_state(project_id: str, user_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    return update_project_fields(project_id, user_id, {"department_outputs": state})


def log_console_event(state: Dict[str, Any], event: Dict[str, Any], max_items: int = 50) -> Dict[str, Any]:
    log = state.get("execution_log") or []
    log.append({"at": now_iso(), **event})
    state["execution_log"] = log[-max_items:]
    state["last_status"] = event.get("status") or state.get("last_status") or "idle"
    return state


# ------------------------------------------------------------------------------
# PDF HELPERS
# ------------------------------------------------------------------------------
def _normalize_pdf_sections(sections: Any, fallback_heading: str) -> List[Dict[str, str]]:
    if isinstance(sections, dict):
        sections = [{"heading": fallback_heading, "body": sections}]
    if not isinstance(sections, list) or not sections:
        return [{"heading": fallback_heading, "body": dump_json(sections)}]

    normalized: List[Dict[str, str]] = []
    for i, section in enumerate(sections, start=1):
        if isinstance(section, dict):
            heading = str(section.get("heading") or section.get("title") or f"{fallback_heading} {i}")
            body_raw = section.get("body") if "body" in section else section.get("content", "")
        else:
            heading = f"{fallback_heading} {i}"
            body_raw = section

        if isinstance(body_raw, (dict, list)):
            body = dump_json(body_raw)
        else:
            body = str(body_raw or "")
        normalized.append({"heading": heading, "body": body})
    return normalized


def create_simple_pdf(title: str, sections: Any, filename_prefix: str) -> Dict[str, str]:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF dependency missing: {exc}. Add reportlab to requirements.txt")

    out_dir = MEDIA_DIR / "pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename_prefix}_{uuid.uuid4().hex}.pdf"

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.fontName = "Helvetica"
    body_style.fontSize = 10
    body_style.leading = 14

    mono_style = ParagraphStyle(
        "MonoBody",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8,
        leading=10,
    )

    story = [Paragraph(escape(str(title)), title_style), Spacer(1, 10)]

    for section in _normalize_pdf_sections(sections, title):
        story.append(Paragraph(escape(section["heading"]), heading_style))
        story.append(Spacer(1, 4))
        body = str(section["body"]).strip()

        if body.startswith("{") or body.startswith("["):
            story.append(Preformatted(body, mono_style))
        else:
            safe_body = escape(body).replace("\n", "<br/>")
            story.append(Paragraph(safe_body, body_style))

        story.append(Spacer(1, 8))

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    doc.build(story)

    rel = relative_public_url(out_path)
    return {"pdf_path": rel, "pdf_url": absolute_public_url(rel)}


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


class ArmInput(BaseModel):
    armed: bool = True


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


app = FastAPI(title=APP_TITLE, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
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

    return {
        "message": "Concept selected",
        "index": payload.index,
        "selected": selected,
        "project": project,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)
