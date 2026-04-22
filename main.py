import os
import re
import json
import uuid
import datetime as dt
from pathlib import Path
from contextlib import asynccontextmanager
from html import escape
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, Depends
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
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
RENDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_BASE_URL = (
    os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    or (f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME', '').strip()}" if os.getenv("RENDER_EXTERNAL_HOSTNAME") else "")
)

EMAIL_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.I)

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
    try:
        rel = file_path.resolve().relative_to(MEDIA_DIR)
        return f"/media/{rel.as_posix()}"
    except Exception:
        rel = file_path.resolve().relative_to(RENDER_OUTPUT_DIR)
        return f"/renders/{rel.as_posix()}"


def absolute_public_url(relative_url: str) -> str:
    if relative_url.startswith("http://") or relative_url.startswith("https://"):
        return relative_url
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}{relative_url}"
    return relative_url


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


# ------------------------------------------------------------------------------
# AUTH + USERS
# ------------------------------------------------------------------------------
@with_db
def get_user_by_email(cur, email: str) -> Optional[Dict[str, Any]]:
    cur.execute("""
        select id, email, password, full_name, role, is_active, created_at
        from public.users
        where lower(email) = lower(%s)
        limit 1
    """, (email,))
    row = cur.fetchone()
    return dict(row) if row else None


@with_db
def get_user_by_id(cur, user_id: str) -> Optional[Dict[str, Any]]:
    cur.execute("""
        select id, email, full_name, role, is_active, created_at
        from public.users
        where id = %s
        limit 1
    """, (user_id,))
    row = cur.fetchone()
    return dict(row) if row else None


@with_db
def create_user(cur, email: str, password: str, full_name: Optional[str]) -> Dict[str, Any]:
    existing = get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    user_id = str(uuid.uuid4())
    cur.execute("""
        insert into public.users (id, email, password, full_name)
        values (%s, %s, %s, %s)
    """, (user_id, email, hash_password(password), full_name))
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
    cur.execute("""
        insert into public.projects (id, user_id, name, brief, event_type, style_direction, status)
        values (%s, %s, %s, %s, %s, %s, %s)
    """, (project_id, user_id, name or "Untitled Project", brief, event_type, style_direction, "draft"))
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=500, detail="Project creation failed")
    return project


@with_db
def list_projects(cur, user_id: str) -> List[Dict[str, Any]]:
    cur.execute("""
        select *
        from public.projects
        where user_id = %s
        order by created_at desc
    """, (user_id,))
    return [project_row_to_dict(r) for r in cur.fetchall()]


@with_db
def get_project_by_id(cur, project_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if user_id:
        cur.execute("""
            select *
            from public.projects
            where id = %s and user_id = %s
            limit 1
        """, (project_id, user_id))
    else:
        cur.execute("""
            select *
            from public.projects
            where id = %s
            limit 1
        """, (project_id,))
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
        params
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
    cur.execute("""
        select coalesce(max(version_no), 0) + 1 as next_version
        from public.project_versions
        where project_id = %s
    """, (project_id,))
    next_version = int(cur.fetchone()["next_version"])
    cur.execute("""
        insert into public.project_versions (id, project_id, user_id, version_no, snapshot, note)
        values (%s, %s, %s, %s, %s, %s)
    """, (str(uuid.uuid4()), project_id, user_id, next_version, dump_json(project), note))


@with_db
def add_comment(cur, project_id: str, user_id: str, section: str, comment_text: str) -> Dict[str, Any]:
    comment_id = str(uuid.uuid4())
    cur.execute("""
        insert into public.project_comments (id, project_id, user_id, section, comment_text)
        values (%s, %s, %s, %s, %s)
    """, (comment_id, project_id, user_id, section, comment_text))
    cur.execute("select * from public.project_comments where id = %s", (comment_id,))
    return dict(cur.fetchone())


@with_db
def list_comments(cur, project_id: str, user_id: str) -> List[Dict[str, Any]]:
    cur.execute("""
        select c.*
        from public.project_comments c
        join public.projects p on p.id = c.project_id
        where c.project_id = %s and p.user_id = %s
        order by c.created_at desc
    """, (project_id, user_id))
    return [dict(r) for r in cur.fetchall()]


# ------------------------------------------------------------------------------
# AI / FALLBACK GENERATION
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
            {"cue_no": 1, "name": "Guest Open", "go": "Open venue ambience"},
            {"cue_no": 2, "name": "Opening", "go": "Start opening cue"},
            {"cue_no": 3, "name": "Reveal", "go": "Execute synchronized reveal"},
            {"cue_no": 4, "name": "Speech", "go": "Set speech state"},
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
        return result
    except Exception:
        return _default_showrunner_plan(project)


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
        "openai": {"configured": bool(OPENAI_API_KEY)},
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

    outputs = {
        "sound_ready": True,
        "lighting_ready": True,
        "showrunner_ready": True,
        "console_index": 0,
    }

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


@app.post("/project/{project_id}/show-console")
def show_console(project_id: str, command: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, user_id=str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    showrunner = project.get("showrunner_data") or {}
    cues = showrunner.get("console_cues") or []
    outputs = project.get("department_outputs") or {}
    current_index = int(outputs.get("console_index", 0))

    cmd = (command or "").strip().lower()
    if cmd in {"next", "go", "next cue"} and cues:
        next_index = min(current_index + 1, len(cues) - 1)
        outputs["console_index"] = next_index
        update_project_fields(project_id, str(current_user["id"]), {"department_outputs": outputs})
        return {
            "message": "Cue advanced",
            "cue_index": next_index,
            "cue": cues[next_index],
        }

    return {
        "message": "Console status",
        "cue_index": current_index,
        "cue": cues[current_index] if cues else None,
        "available_cues": cues,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)
