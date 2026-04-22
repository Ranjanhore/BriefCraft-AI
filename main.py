import os
import re
import json
import uuid
import datetime
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg.errors import UniqueViolation, IntegrityError

from dotenv import load_dotenv
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator

from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =============================================================================
# ENV
# =============================================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
SECRET_KEY = os.getenv("SECRET_KEY", "").strip()

TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4.1").strip()
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1").strip()
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-transcribe").strip()
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts").strip()
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime").strip()

ACCESS_TOKEN_HOURS = int(os.getenv("ACCESS_TOKEN_HOURS", "24"))
JWT_ALGORITHM = "HS256"

ALLOWED_ORIGINS = [
    x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()
] or ["*"]


# =============================================================================
# GLOBALS
# =============================================================================

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)
client = None


# =============================================================================
# UTILS
# =============================================================================

def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def now_iso() -> str:
    return now_utc().isoformat()


def require_env(value: str, name: str) -> str:
    if not value:
        raise RuntimeError(f"{name} is missing")
    return value


def require_db_url() -> str:
    return require_env(DATABASE_URL, "DATABASE_URL")


def require_secret_key() -> str:
    return require_env(SECRET_KEY, "SECRET_KEY")


def get_conn():
    return psycopg.connect(
        require_db_url(),
        row_factory=dict_row,
        autocommit=True,
    )


def clean_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [clean_jsonable(v) for v in value]
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return value


def parse_json_value(value: Any) -> Any:
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


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def create_access_token(user_id: str, email: str) -> str:
    payload = {
        "user_id": str(user_id),
        "email": email,
        "iat": int(now_utc().timestamp()),
        "exp": int((now_utc() + datetime.timedelta(hours=ACCESS_TOKEN_HOURS)).timestamp()),
    }
    return jwt.encode(payload, require_secret_key(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, require_secret_key(), algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def normalize_email(email: str) -> str:
    return email.strip().lower()


def is_valid_email(email: str) -> bool:
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return bool(re.match(pattern, email))


def with_db(fn):
    def wrapper(*args, **kwargs):
        with get_conn() as conn:
            with conn.cursor() as cur:
                return fn(cur, *args, **kwargs)
    return wrapper


# =============================================================================
# DB SCHEMA
# =============================================================================

def ensure_schema() -> None:
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
                    user_id uuid not null,
                    name text not null default 'Untitled Project',
                    event_type text,
                    style_direction text,
                    status text not null default 'draft',
                    brief text,
                    analysis text,
                    concepts jsonb,
                    selected jsonb,
                    sound_data jsonb,
                    lighting_data jsonb,
                    showrunner_data jsonb,
                    department_outputs jsonb,
                    images jsonb,
                    render3d jsonb,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                );
            """)

            cur.execute("alter table public.projects add column if not exists style_direction text;")
            cur.execute("alter table public.projects add column if not exists brief text;")
            cur.execute("alter table public.projects add column if not exists analysis text;")
            cur.execute("alter table public.projects add column if not exists concepts jsonb;")
            cur.execute("alter table public.projects add column if not exists selected jsonb;")
            cur.execute("alter table public.projects add column if not exists sound_data jsonb;")
            cur.execute("alter table public.projects add column if not exists lighting_data jsonb;")
            cur.execute("alter table public.projects add column if not exists showrunner_data jsonb;")
            cur.execute("alter table public.projects add column if not exists department_outputs jsonb;")
            cur.execute("alter table public.projects add column if not exists images jsonb;")
            cur.execute("alter table public.projects add column if not exists render3d jsonb;")
            cur.execute("alter table public.projects add column if not exists created_at timestamptz not null default now();")
            cur.execute("alter table public.projects add column if not exists updated_at timestamptz not null default now();")

            cur.execute("""
                create table if not exists public.project_comments (
                    id uuid primary key,
                    project_id uuid not null,
                    user_id uuid,
                    section text,
                    comment_text text,
                    status text not null default 'open',
                    created_at timestamptz not null default now()
                );
            """)

            cur.execute("""
                create table if not exists public.project_versions (
                    id uuid primary key,
                    project_id uuid not null,
                    user_id uuid,
                    version_no int not null,
                    snapshot jsonb,
                    note text,
                    created_at timestamptz not null default now()
                );
            """)

            cur.execute("""
                create table if not exists public.render_jobs (
                    id uuid primary key,
                    project_id uuid not null,
                    user_id uuid,
                    job_type text not null,
                    status text not null default 'queued',
                    input_json jsonb,
                    output_json jsonb,
                    error_text text,
                    created_at timestamptz not null default now()
                );
            """)

            cur.execute("alter table if exists public.projects drop constraint if exists projects_user_id_fkey;")
            cur.execute("alter table if exists public.project_comments drop constraint if exists project_comments_user_id_fkey;")
            cur.execute("alter table if exists public.project_versions drop constraint if exists project_versions_user_id_fkey;")
            cur.execute("alter table if exists public.render_jobs drop constraint if exists render_jobs_user_id_fkey;")

            cur.execute("create index if not exists idx_users_email on public.users (email);")
            cur.execute("create index if not exists idx_projects_user_id on public.projects (user_id);")
            cur.execute("create index if not exists idx_projects_created_at on public.projects (created_at desc);")


# =============================================================================
# OPENAI
# =============================================================================

def require_openai():
    if client is None:
        raise RuntimeError("OpenAI client not configured")
    return client


def llm_text(system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    api = require_openai()
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
    raw = llm_text(system_prompt, user_prompt, temperature=0.2).strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# =============================================================================
# AI FALLBACKS
# =============================================================================

def brief_guest_count(brief: str) -> Optional[int]:
    match = re.search(r"(\d{2,5})\s*(guest|guests|pax|people|attendees)", brief, flags=re.I)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def fallback_analysis(brief: str, event_type: Optional[str], project_name: Optional[str]) -> str:
    guests = brief_guest_count(brief)
    audience_line = f"Estimated audience: approximately {guests} guests." if guests else "Audience size needs final confirmation."
    return (
        f"Project: {project_name or 'Untitled Project'}\n"
        f"Event Type: {event_type or 'To be confirmed'}\n\n"
        f"1. Objective\n"
        f"Create a premium event experience based on the submitted brief, with strong visual impact and coordinated production delivery.\n\n"
        f"2. Audience Profile\n"
        f"{audience_line}\n\n"
        f"3. Creative Direction\n"
        f"The brief suggests a polished, premium, immersive presentation style with strong reveal moments, clean branding integration, and controlled guest flow.\n\n"
        f"4. Production Priorities\n"
        f"- Clear guest arrival and VIP movement\n"
        f"- Strong hero reveal zone\n"
        f"- Branded scenic architecture\n"
        f"- Reliable cue-based show execution\n\n"
        f"5. Sound Considerations\n"
        f"- Speech intelligibility must remain high\n"
        f"- Playback cues should be tightly controlled\n"
        f"- Reveal moment needs stronger dynamic range than general ambience\n\n"
        f"6. Lighting Considerations\n"
        f"- Layered stage looks\n"
        f"- Pre-show ambience, reveal hit, presenter mode, and photo-friendly states\n"
        f"- Practical focus on brand colors and scenic depth\n\n"
        f"7. Show Flow Considerations\n"
        f"- Guest entry\n"
        f"- Pre-show holding\n"
        f"- Opening cue\n"
        f"- Main reveal cue\n"
        f"- Presentation segment\n"
        f"- Closing / media moment\n\n"
        f"8. Missing Clarifications\n"
        f"- Venue dimensions\n"
        f"- Ceiling height / rigging availability\n"
        f"- LED size and resolution\n"
        f"- Final run-of-show duration\n"
        f"- Presenter count and mic count\n"
    )


def fallback_concepts(brief: str) -> List[Dict[str, Any]]:
    return [
        {
            "name": "Aura Reveal",
            "summary": "A premium hero stage with immersive reveal layering and cinematic presentation flow.",
            "style": "Luxury / Premium",
            "materials": ["Gloss scenic panels", "Brushed metal trims", "LED reveal surface"],
            "colors": ["Black", "Gold", "Cool white"],
            "lighting": "Elegant pre-show ambience with dramatic reveal hits",
            "stage_elements": ["Hero portal", "Main stage deck", "VIP sightline focus"],
            "camera_style": "Cinematic wide shots and reveal pushes",
        },
        {
            "name": "Velocity Frame",
            "summary": "A sharper, high-energy launch environment with bold geometry and performance-driven transitions.",
            "style": "Futuristic / Tech",
            "materials": ["Angular scenic shells", "Backlit frames", "Integrated LED surfaces"],
            "colors": ["Black", "Electric blue", "Silver"],
            "lighting": "Fast cue transitions and high-contrast accents",
            "stage_elements": ["Angular stage frame", "Reveal runway", "Side feature towers"],
            "camera_style": "Dynamic tracking and strong diagonals",
        },
        {
            "name": "Signature Grand",
            "summary": "A balanced premium event architecture with strong branding zones and executive presentation polish.",
            "style": "Corporate Luxury",
            "materials": ["Fabric finishes", "Metal trims", "Textured scenic cladding"],
            "colors": ["Charcoal", "Champagne gold", "Warm white"],
            "lighting": "Executive keynote looks with premium scenic depth",
            "stage_elements": ["Central presentation wall", "Brand integration zones", "Photo-op zone"],
            "camera_style": "Clean frontal compositions and brand-led framing",
        },
    ]


def fallback_sound_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "concept": "Premium launch audio design prioritizing speech clarity and controlled reveal playback.",
        "system_design": {
            "foh": "L/R main PA with center image support where possible",
            "fill": "Front fills for first rows",
            "delay": "Optional delay speakers for deeper audience zones",
            "subs": "Controlled low-end reinforcement for reveal moments",
        },
        "speaker_plan": [
            "Main L/R",
            "Front fill",
            "Optional VIP fill",
            "Sub reinforcement",
        ],
        "input_list": [
            "Presenter wireless handheld x2",
            "Lapel / headset x2",
            "Playback L/R",
            "Video feed backup stereo",
        ],
        "mic_plan": [
            "Primary presenter mic",
            "Backup presenter mic",
            "VIP Q&A mic",
        ],
        "patch_sheet": [
            "CH1 Presenter HH",
            "CH2 Backup HH",
            "CH3 Headset",
            "CH9/10 Playback",
        ],
        "playback_cues": [
            "Pre-show ambience",
            "Opening sting",
            "Reveal hit",
            "Walk-on music",
            "Exit music",
        ],
        "staffing": [
            "FOH Engineer",
            "System Tech",
            "Playback Operator",
        ],
        "rehearsal_notes": [
            "Confirm presenter gain structure",
            "Check reveal cue level",
            "Check backup playback source",
        ],
        "risk_notes": [
            "Keep backup playback device ready",
            "Maintain spare wireless channel",
        ],
        "pdf_sections": [
            {"heading": "Sound Concept", "body": "Premium launch audio design prioritizing intelligibility and reveal impact."},
            {"heading": "Input List", "body": "Presenter mics, playback feeds, and backup audio paths."},
            {"heading": "Cue Notes", "body": "Pre-show ambience, opening, reveal, walk-on, and exit music."},
        ],
    }


def fallback_lighting_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "concept": "Premium scenic lighting with distinct keynote, reveal, and photo moments.",
        "fixture_list": [
            "Profile spots",
            "Wash fixtures",
            "Beam accents",
            "Audience blinders",
            "Followspot or key front light",
        ],
        "truss_plan": [
            "Upstage truss",
            "Mid-stage accent positions",
            "Front keylight positions",
        ],
        "dmx_notes": [
            "Separate keynote and reveal universes if scale requires",
            "Keep emergency white look available on master",
        ],
        "scene_cues": [
            "Pre-show ambience",
            "Opening dark hold",
            "Reveal hit",
            "Presentation mode",
            "Photo mode",
            "Closing mode",
        ],
        "looks": [
            "Warm executive keynote",
            "High-impact reveal",
            "Clean brand color wash",
        ],
        "operator_notes": [
            "Keep reveal timing locked to playback",
            "Avoid over-bright audience spill during keynote",
        ],
        "rehearsal_notes": [
            "Confirm presenter keylight positions",
            "Check camera white balance look",
        ],
        "fallback_plan": [
            "Emergency white stage wash",
            "Manual cue fallback page",
        ],
        "pdf_sections": [
            {"heading": "Lighting Concept", "body": "Layered lighting for keynote, reveal, and media-friendly coverage."},
            {"heading": "Cue Structure", "body": "Pre-show, opening, reveal, keynote, photo, and close."},
            {"heading": "Operator Notes", "body": "Maintain clean timing and protect presenter visibility."},
        ],
    }


def fallback_showrunner_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    cues = [
        {"cue_no": 1, "label": "House Open", "call": "Open doors and run guest ambience"},
        {"cue_no": 2, "label": "Standby Opening", "call": "Standby audio opening sting and lights opening look"},
        {"cue_no": 3, "label": "GO Opening", "call": "GO opening sting and lights"},
        {"cue_no": 4, "label": "Standby Reveal", "call": "Standby reveal video, reveal audio, reveal lights"},
        {"cue_no": 5, "label": "GO Reveal", "call": "GO reveal"},
        {"cue_no": 6, "label": "Presenter Walk-on", "call": "GO presenter walk-on music"},
        {"cue_no": 7, "label": "Close", "call": "GO closing music and transition to networking"},
    ]
    return {
        "running_order": [
            "Guest entry",
            "Pre-show hold",
            "Opening",
            "Reveal",
            "Presenter segment",
            "Closing",
        ],
        "cue_script": cues,
        "standby_calls": [
            "Standby audio",
            "Standby lighting",
            "Standby playback",
            "Standby stage management",
        ],
        "go_calls": [
            "GO Opening",
            "GO Reveal",
            "GO Walk-on",
            "GO Closing",
        ],
        "departmental_dependencies": {
            "audio": "Playback must be armed before reveal",
            "lighting": "Reveal cue follows playback lock",
            "stage_management": "Talent clear before GO reveal",
        },
        "delay_protocol": [
            "Hold opening cue until all departments confirm",
            "Use standby page if talent delay exceeds 60 seconds",
        ],
        "emergency_protocol": [
            "Move to safe white stage look",
            "Cut to MC handheld announcement if playback fails",
        ],
        "rehearsal_flow": [
            "Technical rehearsal",
            "Department cue-to-cue",
            "Full dress rehearsal",
        ],
        "console_cues": cues,
        "pdf_sections": [
            {"heading": "Running Order", "body": "Guest entry, opening, reveal, presentation, and close."},
            {"heading": "Cue Script", "body": "Standby and GO call structure for live execution."},
            {"heading": "Emergency Protocol", "body": "Fallback show control actions if playback or cueing fails."},
        ],
    }


def generate_analysis(brief: str, event_type: Optional[str], name: Optional[str]) -> str:
    if not OPENAI_API_KEY or client is None:
        return fallback_analysis(brief, event_type, name)
    try:
        return llm_text(
            "You are a senior event creative strategist. Return a practical production-ready brief analysis.",
            f"""
Project name: {name or 'Untitled Project'}
Event type: {event_type or 'Not specified'}

Analyze this brief in a practical way.
Cover:
1. event objective
2. audience profile
3. creative direction
4. production priorities
5. sound considerations
6. lighting considerations
7. show flow considerations
8. missing clarifications

Brief:
{brief}
""",
            temperature=0.4,
        )
    except Exception:
        return fallback_analysis(brief, event_type, name)


def generate_concepts(brief: str, analysis: str) -> List[Dict[str, Any]]:
    if not OPENAI_API_KEY or client is None:
        return fallback_concepts(brief)
    try:
        result = llm_json(
            "Return only valid JSON.",
            f"""
Return exactly 3 concept options as a JSON array.
Each object must contain:
- name
- summary
- style
- materials
- colors
- lighting
- stage_elements
- camera_style

Brief:
{brief}

Analysis:
{analysis}
""",
        )
        if isinstance(result, list) and result:
            return result
        return fallback_concepts(brief)
    except Exception:
        return fallback_concepts(brief)


def generate_sound_department(project: Dict[str, Any]) -> Dict[str, Any]:
    if not OPENAI_API_KEY or client is None:
        return fallback_sound_plan(project)
    try:
        result = llm_json(
            "Return only valid JSON. You are a senior live event sound engineer.",
            f"""
Create a production-ready sound department plan.
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
{json.dumps(clean_jsonable(project), indent=2)}
"""
        )
        return result if isinstance(result, dict) else fallback_sound_plan(project)
    except Exception:
        return fallback_sound_plan(project)


def generate_lighting_department(project: Dict[str, Any]) -> Dict[str, Any]:
    if not OPENAI_API_KEY or client is None:
        return fallback_lighting_plan(project)
    try:
        result = llm_json(
            "Return only valid JSON. You are a senior live event lighting designer.",
            f"""
Create a production-ready lighting department plan.
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
{json.dumps(clean_jsonable(project), indent=2)}
"""
        )
        return result if isinstance(result, dict) else fallback_lighting_plan(project)
    except Exception:
        return fallback_lighting_plan(project)


def generate_showrunner_department(project: Dict[str, Any]) -> Dict[str, Any]:
    if not OPENAI_API_KEY or client is None:
        return fallback_showrunner_plan(project)
    try:
        result = llm_json(
            "Return only valid JSON. You are a show caller / show runner / stage manager.",
            f"""
Create a production-ready show running plan.
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
{json.dumps(clean_jsonable(project), indent=2)}
"""
        )
        return result if isinstance(result, dict) else fallback_showrunner_plan(project)
    except Exception:
        return fallback_showrunner_plan(project)


# =============================================================================
# Pydantic Models
# =============================================================================

class UserInput(BaseModel):
    email: str
    password: str = Field(min_length=8)
    full_name: Optional[str] = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = normalize_email(v)
        if not is_valid_email(v):
            raise ValueError("Invalid email address")
        return v


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


class SelectConceptInput(BaseModel):
    index: Optional[int] = None
    concept_index: Optional[int] = None


class CommandInput(BaseModel):
    command: str


# =============================================================================
# DB OPERATIONS
# =============================================================================

@with_db
def db_get_user_by_email(cur, email: str) -> Optional[Dict[str, Any]]:
    cur.execute("""
        select id, email, password, full_name, role, is_active, created_at
        from public.users
        where email = %s
        limit 1
    """, (normalize_email(email),))
    row = cur.fetchone()
    return clean_jsonable(dict(row)) if row else None


@with_db
def db_get_user_by_id(cur, user_id: str) -> Optional[Dict[str, Any]]:
    cur.execute("""
        select id, email, password, full_name, role, is_active, created_at
        from public.users
        where id = %s
        limit 1
    """, (user_id,))
    row = cur.fetchone()
    return clean_jsonable(dict(row)) if row else None


@with_db
def db_create_user(cur, email: str, password: str, full_name: Optional[str]) -> Dict[str, Any]:
    user_id = str(uuid.uuid4())
    cur.execute("""
        insert into public.users (id, email, password, full_name)
        values (%s, %s, %s, %s)
        returning id, email, full_name, role, is_active, created_at
    """, (
        user_id,
        normalize_email(email),
        hash_password(password),
        full_name.strip() if full_name else None,
    ))
    row = cur.fetchone()
    return clean_jsonable(dict(row))


@with_db
def db_list_projects(cur, user_id: str) -> List[Dict[str, Any]]:
    cur.execute("""
        select id, user_id, name, event_type, style_direction, status, brief,
               analysis, concepts, selected, sound_data, lighting_data,
               showrunner_data, department_outputs, images, render3d,
               created_at, updated_at
        from public.projects
        where user_id = %s
        order by created_at desc
    """, (user_id,))
    rows = []
    for row in cur.fetchall():
        item = dict(row)
        for key in [
            "concepts", "selected", "sound_data", "lighting_data",
            "showrunner_data", "department_outputs", "images", "render3d"
        ]:
            item[key] = parse_json_value(item.get(key))
        rows.append(clean_jsonable(item))
    return rows


@with_db
def db_get_project(cur, project_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if user_id:
        cur.execute("""
            select id, user_id, name, event_type, style_direction, status, brief,
                   analysis, concepts, selected, sound_data, lighting_data,
                   showrunner_data, department_outputs, images, render3d,
                   created_at, updated_at
            from public.projects
            where id = %s and user_id = %s
            limit 1
        """, (project_id, user_id))
    else:
        cur.execute("""
            select id, user_id, name, event_type, style_direction, status, brief,
                   analysis, concepts, selected, sound_data, lighting_data,
                   showrunner_data, department_outputs, images, render3d,
                   created_at, updated_at
            from public.projects
            where id = %s
            limit 1
        """, (project_id,))
    row = cur.fetchone()
    if not row:
        return None
    item = dict(row)
    for key in [
        "concepts", "selected", "sound_data", "lighting_data",
        "showrunner_data", "department_outputs", "images", "render3d"
    ]:
        item[key] = parse_json_value(item.get(key))
    return clean_jsonable(item)


@with_db
def db_create_project(
    cur,
    user_id: str,
    name: Optional[str],
    brief: Optional[str],
    event_type: Optional[str],
    style_direction: Optional[str],
) -> Dict[str, Any]:
    cur.execute("select id from public.users where id = %s limit 1", (user_id,))
    existing_user = cur.fetchone()
    if not existing_user:
        raise HTTPException(
            status_code=400,
            detail="Authenticated user was not found in users table. Please login again."
        )

    project_id = str(uuid.uuid4())
    cur.execute("""
        insert into public.projects (
            id, user_id, name, brief, event_type, style_direction, status
        )
        values (%s, %s, %s, %s, %s, %s, %s)
        returning id, user_id, name, event_type, style_direction, status, brief,
                  analysis, concepts, selected, sound_data, lighting_data,
                  showrunner_data, department_outputs, images, render3d,
                  created_at, updated_at
    """, (
        project_id,
        user_id,
        (name or "Untitled Project").strip(),
        brief.strip() if brief else None,
        event_type.strip() if event_type else None,
        style_direction.strip() if style_direction else None,
        "draft",
    ))
    row = cur.fetchone()
    item = dict(row)
    return clean_jsonable(item)


@with_db
def db_update_project(cur, project_id: str, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    allowed = {
        "name", "brief", "event_type", "style_direction", "status", "analysis",
        "concepts", "selected", "sound_data", "lighting_data", "showrunner_data",
        "department_outputs", "images", "render3d"
    }

    parts = []
    values = []
    for key, value in updates.items():
        if key not in allowed:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        parts.append(f"{key} = %s")
        values.append(value)

    parts.append("updated_at = now()")
    values.extend([project_id, user_id])

    cur.execute(
        f"""
        update public.projects
        set {", ".join(parts)}
        where id = %s and user_id = %s
        returning id, user_id, name, event_type, style_direction, status, brief,
                  analysis, concepts, selected, sound_data, lighting_data,
                  showrunner_data, department_outputs, images, render3d,
                  created_at, updated_at
        """,
        tuple(values)
    )

    row = cur.fetchone()
    if not row:
        return None
    item = dict(row)
    for key in [
        "concepts", "selected", "sound_data", "lighting_data",
        "showrunner_data", "department_outputs", "images", "render3d"
    ]:
        item[key] = parse_json_value(item.get(key))
    return clean_jsonable(item)


# =============================================================================
# AUTH DEPENDENCIES
# =============================================================================

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> Dict[str, Any]:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    payload = decode_access_token(credentials.credentials)
    user_id = str(payload["user_id"])
    user = db_get_user_by_id(user_id)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user


# =============================================================================
# APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    if OPENAI_API_KEY and OpenAI is not None:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception:
            client = None
    ensure_schema()
    yield


app = FastAPI(
    title="AI Creative Studio API",
    version="2.0.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# INTERNAL PIPELINE
# =============================================================================

def run_pipeline_internal(project: Dict[str, Any], incoming_text: Optional[str], incoming_name: Optional[str], incoming_event_type: Optional[str], current_user: Dict[str, Any]) -> Dict[str, Any]:
    project_id = str(project["id"])
    user_id = str(current_user["id"])

    brief = (incoming_text or project.get("brief") or "").strip()
    if not brief:
        raise HTTPException(status_code=422, detail="Field 'text' is required")

    name = (incoming_name or project.get("name") or "Untitled Project").strip()
    event_type = (incoming_event_type or project.get("event_type") or "").strip() or None

    analysis = project.get("analysis")
    concepts = project.get("concepts")

    updates: Dict[str, Any] = {
        "name": name,
        "brief": brief,
        "event_type": event_type,
    }

    if not analysis:
        analysis = generate_analysis(brief, event_type, name)
        updates["analysis"] = analysis

    if not concepts:
        concepts = generate_concepts(brief, analysis)
        updates["concepts"] = concepts

    updates["status"] = "concepts_ready"

    updated = db_update_project(project_id, user_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")

    return {
        "message": "Pipeline completed",
        "project_id": project_id,
        "status": updated.get("status"),
        "brief": updated.get("brief"),
        "analysis": updated.get("analysis"),
        "concepts": updated.get("concepts") or [],
        "project": updated,
    }


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/")
def root():
    return {
        "message": "AI Creative Studio API is running",
        "time": now_iso(),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": now_iso(),
    }


@app.post("/signup")
def signup(payload: UserInput):
    email = normalize_email(payload.email)
    existing = db_get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    try:
        user = db_create_user(email, payload.password, payload.full_name)
    except UniqueViolation:
        raise HTTPException(status_code=400, detail="User already exists")

    token = create_access_token(str(user["id"]), user["email"])
    return {
        "message": "User created",
        "user_id": user["id"],
        "access_token": token,
        "token": token,
        "token_type": "bearer",
    }


@app.post("/login")
def login(payload: UserInput):
    email = normalize_email(payload.email)
    user = db_get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    if not verify_password(payload.password, user["password"]):
        raise HTTPException(status_code=400, detail="Wrong password")

    token = create_access_token(str(user["id"]), user["email"])
    return {
        "access_token": token,
        "token": token,
        "token_type": "bearer",
        "user_id": user["id"],
    }


@app.get("/me")
def me(current_user: Dict[str, Any] = Depends(get_current_user)):
    safe_user = dict(current_user)
    safe_user.pop("password", None)
    return {"user": clean_jsonable(safe_user)}


@app.get("/projects")
def list_projects(current_user: Dict[str, Any] = Depends(get_current_user)):
    projects = db_list_projects(str(current_user["id"]))
    return {"projects": projects}


@app.post("/projects")
def create_project(
    payload: ProjectCreateInput = Body(default=ProjectCreateInput()),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        project = db_create_project(
            user_id=str(current_user["id"]),
            name=payload.title or payload.name,
            brief=payload.brief,
            event_type=payload.event_type,
            style_direction=payload.style_direction,
        )
        return clean_jsonable(project)
    except HTTPException:
        raise
    except IntegrityError as e:
        raise HTTPException(status_code=500, detail=f"Project creation failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Project creation failed: {str(e)}")


@app.get("/projects/{project_id}")
def get_project_route(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = db_get_project(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": project}


@app.get("/project/{project_id}")
def get_project_alias(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = db_get_project(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return clean_jsonable(project)


@app.post("/run")
def run_pipeline(
    payload: RunInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        project: Optional[Dict[str, Any]] = None

        if payload.project_id:
            project = db_get_project(payload.project_id, str(current_user["id"]))
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
        else:
            project = db_create_project(
                user_id=str(current_user["id"]),
                name=payload.name,
                brief=payload.text,
                event_type=payload.event_type,
                style_direction=None,
            )

        return run_pipeline_internal(
            project=project,
            incoming_text=payload.text,
            incoming_name=payload.name,
            incoming_event_type=payload.event_type,
            current_user=current_user,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


@app.post("/projects/{project_id}/run")
def run_pipeline_for_project(
    project_id: str,
    payload: Optional[RunInput] = Body(default=None),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    project = db_get_project(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    body = payload or RunInput(text=project.get("brief") or "Untitled project brief")
    return run_pipeline_internal(
        project=project,
        incoming_text=body.text if body.text else project.get("brief"),
        incoming_name=body.name,
        incoming_event_type=body.event_type,
        current_user=current_user,
    )


@app.post("/select")
def select_concept_alias(
    payload: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    project_id = payload.get("project_id")
    index = payload.get("index", payload.get("concept_index"))
    if not project_id:
        raise HTTPException(status_code=422, detail="project_id is required")
    if index is None:
        raise HTTPException(status_code=422, detail="index is required")

    project = db_get_project(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    concepts = project.get("concepts") or []
    if not isinstance(concepts, list) or not concepts:
        raise HTTPException(status_code=400, detail="No concepts available")

    if index < 0 or index >= len(concepts):
        raise HTTPException(status_code=400, detail="Invalid concept index")

    selected = concepts[index]
    updated = db_update_project(
        project_id,
        str(current_user["id"]),
        {"selected": selected, "status": "concept_selected"},
    )
    return {
        "message": "Concept selected",
        "index": index,
        "selected": selected,
        "project": updated,
    }


@app.post("/projects/{project_id}/select-concept")
def select_concept(
    project_id: str,
    payload: SelectConceptInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    project = db_get_project(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    concepts = project.get("concepts") or []
    if not isinstance(concepts, list) or not concepts:
        raise HTTPException(status_code=400, detail="No concepts available")

    index = payload.index if payload.index is not None else payload.concept_index
    if index is None:
        raise HTTPException(status_code=422, detail="index or concept_index is required")

    if index < 0 or index >= len(concepts):
        raise HTTPException(status_code=400, detail="Invalid concept index")

    selected = concepts[index]
    updated = db_update_project(
        project_id,
        str(current_user["id"]),
        {"selected": selected, "status": "concept_selected"},
    )
    return {
        "message": "Concept selected",
        "index": index,
        "selected": selected,
        "project": updated,
    }


@app.post("/projects/{project_id}/generate-departments")
def generate_departments(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = db_get_project(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    selected = project.get("selected")
    if not selected:
        concepts = project.get("concepts") or []
        if concepts:
            selected = concepts[0]
            project = db_update_project(project_id, str(current_user["id"]), {"selected": selected, "status": "concept_selected"}) or project
        else:
            raise HTTPException(status_code=400, detail="Select a concept first")

    sound = generate_sound_department(project)
    lighting = generate_lighting_department(project)
    showrunner = generate_showrunner_department(project)

    department_outputs = {
        "sound_ready": True,
        "lighting_ready": True,
        "showrunner_ready": True,
        "generated_at": now_iso(),
    }

    updated = db_update_project(
        project_id,
        str(current_user["id"]),
        {
            "sound_data": sound,
            "lighting_data": lighting,
            "showrunner_data": showrunner,
            "department_outputs": department_outputs,
            "status": "departments_ready",
        },
    )

    return {
        "message": "Departments generated",
        "project_id": project_id,
        "sound_data": sound,
        "lighting_data": lighting,
        "showrunner_data": showrunner,
        "department_outputs": department_outputs,
        "project": updated,
    }


@app.post("/projects/{project_id}/show-console")
def show_console(
    project_id: str,
    payload: CommandInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    project = db_get_project(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    command = (payload.command or "").strip()
    if not command:
        raise HTTPException(status_code=422, detail="command is required")

    showrunner_data = project.get("showrunner_data") or {}
    console_cues = showrunner_data.get("console_cues") or []

    lower = command.lower()

    if lower in {"next", "next cue"}:
        reply = "Advancing to the next cue."
    elif lower in {"status", "show status"}:
        reply = f"Show console ready. Total stored cues: {len(console_cues)}."
    elif "cue" in lower:
        reply = f"Console acknowledged command: {command}"
    else:
        reply = f"Command received: {command}"

    return {
        "reply": reply,
        "command": command,
        "console_cues": console_cues[:10],
    }
