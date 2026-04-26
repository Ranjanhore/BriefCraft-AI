
from __future__ import annotations

import base64
import io
import json
import os
from datetime import datetime, timedelta, timezone
from jose import jwt
from openai import OpenAI
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-render")
ALGORITHM = "HS256"
TOKEN_HOURS = int(os.getenv("ACCESS_TOKEN_HOURS", "72"))

def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
    payload = {
        "sub": user_id,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from jose import jwt

SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-render")
ALGORITHM = "HS256"
TOKEN_HOURS = 72

def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
    payload = {
        "sub": user_id,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

if __name__ == "__main__":
    user_id = "693d347e-b791-4dfe-b275-b5fff2de3df7"
    print(create_access_token(user_id))

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile

try:
    from supabase import create_client
except Exception:
    create_client = None

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from openai import OpenAI
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException

load_dotenv()

APP_NAME = os.getenv("APP_TITLE", "AICreative Studio API").strip() or "AICreative Studio API"
APP_VERSION = "4.2.0"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY and OpenAI else None

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()

SUPABASE_SERVICE_ROLE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    or os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    or os.getenv("SUPABASE_KEY", "").strip()
)

SUPABASE_STORAGE_BUCKET = (
    os.getenv("SUPABASE_STORAGE_BUCKET", "").strip()
    or os.getenv("STORAGE_BUCKET", "").strip()
    or "briefcraft-assets"
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY", "")).strip()
SECRET_KEY = (os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY", "change-me-32char-secret-key-xx")).strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1").strip() or "gpt-image-1"
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts"
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe").strip() or "gpt-4o-mini-transcribe"
TTS_VOICE = os.getenv("TTS_VOICE", "coral").strip() or "coral"
VOICE_DEFAULT_INSTRUCTIONS = os.getenv(
    "VOICE_DEFAULT_INSTRUCTIONS",
    "Warm, polished Indian event presenter voice. Natural pacing. Clear diction.",
).strip()
PORT = int(os.getenv("PORT", "10000"))
ALGORITHM = "HS256"
TOKEN_HOURS = int(os.getenv("TOKEN_HOURS", "72"))
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ------------------------------------------------------------------------------
# Supabase / local fallback / password hashing
# ------------------------------------------------------------------------------

_LOCAL: Dict[str, Dict[str, Dict[str, Any]]] = {}

_sb = None
if SUPABASE_URL and SUPABASE_KEY and create_client:
    try:
        _sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        _sb = None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = (BASE_DIR / "exports").resolve()
UPLOAD_DIR = (BASE_DIR / "uploads").resolve()
MEDIA_DIR = (BASE_DIR / "media").resolve()
RENDER_OUTPUT_DIR = (BASE_DIR / "renders").resolve()
VOICE_DIR = (MEDIA_DIR / "voice").resolve()
PDF_DIR = (MEDIA_DIR / "pdfs").resolve()
CAD_DIR = (MEDIA_DIR / "cad").resolve()
for _path in (EXPORT_DIR, UPLOAD_DIR, MEDIA_DIR, RENDER_OUTPUT_DIR, VOICE_DIR, PDF_DIR, CAD_DIR):
    _path.mkdir(parents=True, exist_ok=True)

def _split_origins(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]

ALLOWED_ORIGINS = _split_origins(
    os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,https://briefly-sparkle.lovable.app,https://81db4809-ba40-464a-bd03-42e7f872691c.lovableproject.com",
    )
)

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://([a-zA-Z0-9-]+\.)?(lovable\.(app|dev)|lovableproject\.com)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/exports", StaticFiles(directory=str(EXPORT_DIR)), name="exports")
app.mount("/renders", StaticFiles(directory=str(RENDER_OUTPUT_DIR)), name="renders")


# ------------------------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)

def load_json(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default

def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "file"))
    return cleaned.strip("._") or "file"

def ensure_uuid(value: Optional[str], field_name: str) -> str:
    if not value:
        raise HTTPException(status_code=422, detail=f"{field_name} required")
    try:
        return str(uuid.UUID(str(value)))
    except Exception:
        raise HTTPException(status_code=422, detail=f"Invalid {field_name}")

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

def parse_size(value: str, fallback: Tuple[int, int]) -> Tuple[int, int]:
    try:
        left, right = str(value).lower().split("x", 1)
        width = int(left)
        height = int(right)
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return fallback


# ------------------------------------------------------------------------------
# Local/Supabase persistence helpers
# ------------------------------------------------------------------------------

def _deserialize_row(table: str, row: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(row)
    if table in {"projects", "project_assets", "agent_jobs", "project_activity_logs", "voice_sessions", "voice_messages", "cad_layouts", "show_trials"}:
        for key in ("analysis", "concepts", "selected_concept", "sound_data", "lighting_data", "showrunner_data",
                    "department_outputs", "scene_json", "visual_policy", "input_data", "output_data", "meta", "trial_data"):
            if key in item:
                item[key] = load_json(item.get(key), item.get(key))
    return item

def _serialize_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            out[key] = dump_json(value)
        else:
            out[key] = value
    return out

def db_insert(table: str, data: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(data)
    payload.setdefault("id", str(uuid.uuid4()))
    payload.setdefault("created_at", now_iso())
    payload.setdefault("updated_at", now_iso())
    if _sb:
        response = _sb.table(table).insert(_serialize_payload(payload)).execute()
        if response.data:
            return _deserialize_row(table, response.data[0])
        return payload
    row = dict(payload)
    _LOCAL.setdefault(table, {})[str(row["id"])] = row
    return row

def db_get(table: str, **filters: Any) -> Optional[Dict[str, Any]]:
    if _sb:
        query = _sb.table(table).select("*")
        for key, value in filters.items():
            query = query.eq(key, value)
        response = query.limit(1).execute()
        return _deserialize_row(table, response.data[0]) if response.data else None
    for row in _LOCAL.get(table, {}).values():
        if all(row.get(k) == v for k, v in filters.items()):
            return dict(row)
    return None

def db_list(table: str, limit: int = 100, order_key: str = "created_at", desc: bool = True, **filters: Any) -> List[Dict[str, Any]]:
    if _sb:
        query = _sb.table(table).select("*")
        for key, value in filters.items():
            query = query.eq(key, value)
        try:
            query = query.order(order_key, desc=desc)
        except Exception:
            pass
        response = query.limit(limit).execute()
        return [_deserialize_row(table, row) for row in (response.data or [])]
    rows = []
    for row in _LOCAL.get(table, {}).values():
        if all(row.get(k) == v for k, v in filters.items()):
            rows.append(dict(row))
    rows.sort(key=lambda item: str(item.get(order_key, "")), reverse=desc)
    return rows[:limit]

def db_update(table: str, row_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(data)
    payload["updated_at"] = now_iso()
    if _sb:
        response = _sb.table(table).update(_serialize_payload(payload)).eq("id", row_id).execute()
        if response.data:
            return _deserialize_row(table, response.data[0])
        current = db_get(table, id=row_id) or {"id": row_id}
        current.update(payload)
        return current
    current = _LOCAL.setdefault(table, {}).get(row_id)
    if not current:
        raise HTTPException(status_code=404, detail=f"{table} row not found")
    current.update(payload)
    return dict(current)


# ------------------------------------------------------------------------------
# Auth helpers
# ------------------------------------------------------------------------------

def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    return db_get("users", id=user_id)    
    
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


bearer = HTTPBearer(auto_error=False)
def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> Dict[str, Any]:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# ------------------------------------------------------------------------------
# OpenAI / AI helpers
# ------------------------------------------------------------------------------

def llm_json(system: str, user_prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not _openai_client:
        return fallback
    try:
        response = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else fallback
    except Exception:
        return fallback

def synthesize_speech(text: str, voice: Optional[str] = None, instructions: Optional[str] = None, filename_prefix: str = "tts") -> Dict[str, str]:
    if not _openai_client:
        raise HTTPException(status_code=500, detail="OpenAI not configured")
    out_path = VOICE_DIR / f"{filename_prefix}_{uuid.uuid4().hex}.mp3"
    with _openai_client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=(voice or TTS_VOICE),
        input=text[:4096],
        instructions=instructions or VOICE_DEFAULT_INSTRUCTIONS,
        response_format="mp3",
    ) as response:
        response.stream_to_file(out_path)
    rel = relative_public_url(out_path)
    return {
        "audio_url": absolute_public_url(rel),
        "audio_path": rel,
        "voice": voice or TTS_VOICE,
        "response_format": "mp3",
    }

def transcribe_audio_file(path: Path) -> str:
    if not _openai_client:
        return "Transcription unavailable: OpenAI not configured."
    with path.open("rb") as file_obj:
        result = _openai_client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=file_obj)
    return getattr(result, "text", "") or ""

def generate_image_data_url(prompt: str, size: str = "1536x1024", quality: str = "high") -> Optional[str]:
    if not _openai_client:
        return None
    try:
        response = _openai_client.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            size=size,
            quality=quality,
        )
        b64 = getattr(response.data[0], "b64_json", None)
        if b64:
            return f"data:image/png;base64,{b64}"
        url = getattr(response.data[0], "url", None)
        return url
    except Exception:
        return None

def persist_data_url_image(data_url: str, target_dir: Path, prefix: str) -> Tuple[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    out_path = target_dir / filename
    if data_url.startswith("data:image"):
        header, b64 = data_url.split(",", 1)
        out_path.write_bytes(base64.b64decode(b64))
    else:
        raise HTTPException(status_code=500, detail="Image response was not a base64 data URL")
    rel = relative_public_url(out_path)
    return rel, absolute_public_url(rel)


# ------------------------------------------------------------------------------
# PDF helpers
# ------------------------------------------------------------------------------

def _normalize_pdf_sections(sections: Any, fallback_heading: str) -> List[Dict[str, str]]:
    if isinstance(sections, dict):
        sections = [{"heading": fallback_heading, "body": sections}]
    if not isinstance(sections, list) or not sections:
        return [{"heading": fallback_heading, "body": dump_json(sections)}]

    normalized: List[Dict[str, str]] = []
    for index, section in enumerate(sections, start=1):
        if isinstance(section, dict):
            heading = str(section.get("heading") or section.get("title") or f"{fallback_heading} {index}")
            body_raw = section.get("body") if "body" in section else section.get("content", "")
        else:
            heading = f"{fallback_heading} {index}"
            body_raw = section
        body = dump_json(body_raw) if isinstance(body_raw, (dict, list)) else str(body_raw or "")
        normalized.append({"heading": heading, "body": body})
    return normalized

def create_simple_pdf(title: str, sections: Any, filename_prefix: str) -> Dict[str, str]:
    from html import escape
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

    out_path = PDF_DIR / f"{filename_prefix}_{uuid.uuid4().hex}.pdf"
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.fontName = "Helvetica"
    body_style.fontSize = 10
    body_style.leading = 14
    mono_style = ParagraphStyle("MonoBody", parent=styles["Code"], fontName="Courier", fontSize=8, leading=10)

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
# Domain helpers
# ------------------------------------------------------------------------------

EVENT_BUDGETS = {
    "conference": (800000, 1800000, 4200000),
    "award show": (1200000, 2600000, 6500000),
    "brand launch": (900000, 2200000, 5500000),
    "wedding": (700000, 1600000, 4000000),
    "concert": (1500000, 3500000, 9000000),
    "festival": (1200000, 2800000, 7200000),
    "corporate": (800000, 1700000, 4500000),
    "generic": (500000, 1200000, 3000000),
}

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
    if "festival" in lower:
        return "festival"
    if "conference" in lower:
        return "conference"
    return "generic"

def analyze_brief(brief: str, event_type: Optional[str]) -> Dict[str, Any]:
    inferred = infer_event_type(brief, event_type)
    fallback = {
        "summary": brief[:300],
        "event_type": inferred,
        "objectives": ["Translate brief into concept", "Prepare department outputs", "Estimate production scope"],
        "audience": "Stakeholders, clients, production team, audience",
        "risks": ["Venue, timelines, or technical constraints may be incomplete"],
        "assumptions": ["Planning-level outputs only until venue details are confirmed"],
    }
    prompt = (
        "Analyze this event brief and return JSON with keys "
        "summary, event_type, objectives, audience, risks, assumptions.\n\n"
        f"Brief:\n{brief}"
    )
    return {**fallback, **llm_json("You are a senior experiential event strategist. Return JSON only.", prompt, fallback)}

def generate_concepts(brief: str, analysis: Dict[str, Any], event_type: Optional[str]) -> List[Dict[str, Any]]:
    inferred = infer_event_type(brief, event_type or analysis.get("event_type"))
    low, mid, high = EVENT_BUDGETS.get(inferred, EVENT_BUDGETS["generic"])
    fallback = [
        {
            "name": "Cinematic Signature",
            "summary": f"Premium reveal-led concept for {inferred}.",
            "style": "immersive premium",
            "colors": ["black", "gold", "warm white"],
            "materials": ["mirror acrylic", "fabric", "metal"],
            "experience": "emotional brand reveal",
            "key_zones": ["arrival", "stage", "screen", "audience"],
            "estimated_budget_inr": {"low": low, "medium": mid, "high": high},
        },
        {
            "name": "Modern Tech Grid",
            "summary": f"Futuristic sharp concept for {inferred}.",
            "style": "futuristic sharp",
            "colors": ["midnight blue", "cyan", "silver"],
            "materials": ["LED mesh", "truss", "glass"],
            "experience": "show-control visual language",
            "key_zones": ["arrival", "stage", "screen", "audience"],
            "estimated_budget_inr": {"low": int(low * 1.15), "medium": int(mid * 1.15), "high": int(high * 1.15)},
        },
        {
            "name": "Elegant Minimal Luxe",
            "summary": f"Refined editorial concept for {inferred}.",
            "style": "clean editorial",
            "colors": ["ivory", "champagne", "graphite"],
            "materials": ["wood veneer", "soft fabric", "textured flats"],
            "experience": "refined storytelling",
            "key_zones": ["arrival", "stage", "screen", "audience"],
            "estimated_budget_inr": {"low": int(low * 1.35), "medium": int(mid * 1.35), "high": int(high * 1.35)},
        },
    ]
    prompt = (
        "Generate exactly 3 event concepts as JSON under a key named concepts. "
        "Each concept should include name, summary, style, colors, materials, experience, key_zones.\n\n"
        f"Brief:\n{brief}"
    )
    data = llm_json("You are a senior live-experience creative director. Return JSON only.", prompt, {"concepts": fallback})
    concepts = data.get("concepts")
    if isinstance(concepts, list) and len(concepts) >= 3:
        merged = []
        for idx in range(3):
            base = fallback[idx]
            item = concepts[idx] if isinstance(concepts[idx], dict) else {}
            merged.append({**base, **item, "estimated_budget_inr": base["estimated_budget_inr"]})
        return merged
    return fallback

def default_sound_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "system_design": {"console": "FOH digital console", "speaker_system": "Line array PA"},
        "input_list": ["MC mic", "Playback stereo", "Guest mic"],
        "playback_cues": ["opening stinger", "walk-in bed", "finale"],
        "pdf_sections": [
            {"heading": "Sound Overview", "body": f"Planning-level sound design for {project.get('project_name', 'event')}."},
            {"heading": "Input List", "body": "MC, playback, guest, ambient and redundancy."},
        ],
    }

def default_lighting_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fixture_list": ["Moving Heads", "Wash Fixtures", "LED Battens", "Audience Blinders", "Pinspots"],
        "scene_cues": ["house-to-half", "opening reveal", "speaker special", "finale"],
        "pdf_sections": [
            {"heading": "Lighting Overview", "body": f"Concept-driven lighting plan for {project.get('project_name', 'event')}."},
            {"heading": "Cue Intent", "body": "Opening, transitions, and finale lighting beats."},
        ],
    }

def default_showrunner_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "running_order": ["Standby", "House to half", "Opening AV", "MC welcome", "Finale"],
        "pdf_sections": [
            {"heading": "Show Running", "body": f"Cue-based show running script for {project.get('project_name', 'event')}."},
        ],
        "console_cues": [
            {"cue_no": 1, "name": "Standby", "cue_type": "standby", "standby": "All standby", "go": "Standby acknowledged", "actions": []},
            {"cue_no": 2, "name": "House to Half", "cue_type": "lighting", "standby": "Lights standby", "go": "Go half", "actions": [{"protocol": "lighting", "target": "house_lights", "value": "half"}]},
            {"cue_no": 3, "name": "Opening AV", "cue_type": "av", "standby": "AV standby", "go": "Go opener", "actions": [{"protocol": "av", "target": "screen", "value": "play_opener"}]},
            {"cue_no": 4, "name": "MC Welcome", "cue_type": "sound", "standby": "Mic standby", "go": "Go MC", "actions": [{"protocol": "sound", "target": "mc_mic", "value": "on"}]},
        ],
    }

def build_scene_json(project: Dict[str, Any]) -> Dict[str, Any]:
    selected = project.get("selected_concept") or {}
    return {
        "venue_type": project.get("event_type", "event"),
        "concept_name": selected.get("name"),
        "stage": {"width": 18000, "depth": 9000, "height": 1200},
        "screens": [{"name": "Center LED", "width": 8000, "height": 4500}],
        "cameras": [{"view": "hero"}, {"view": "wide"}, {"view": "top"}],
    }


# ------------------------------------------------------------------------------
# Assets / jobs / activity helpers
# ------------------------------------------------------------------------------

def asset_row_to_dict(row):
    if not row:
        return {}

    item = dict(row)

    return {
        "id": item.get("id"),
        "project_id": item.get("project_id"),
        "user_id": item.get("user_id"),
        "asset_type": item.get("asset_type"),
        "section": item.get("section"),
        "job_kind": item.get("job_kind"),
        "title": item.get("title"),
        "prompt": item.get("prompt"),
        "status": item.get("status"),
        "preview_url": item.get("preview_url"),
        "master_url": item.get("master_url"),
        "print_url": item.get("print_url"),
        "source_file_url": item.get("source_file_url"),
        "meta": load_json(item.get("meta")),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
    }

def job_row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(row)
    item["input_data"] = load_json(item.get("input_data"), {})
    item["output_data"] = load_json(item.get("output_data"), {})
    return item

def activity_row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(row)
    item["meta"] = load_json(item.get("meta"), {})
    return item

def merged_visual_policy(project: Dict[str, Any]) -> Dict[str, Any]:
    default_policy = {
        "aspect_ratio": "16:9",
        "preview_size": "1920x1080",
        "master_size": "1536x1024",
        "print_size": "3840x2160",
    }
    current = load_json(project.get("visual_policy"), {}) or {}
    default_policy.update(current)
    return default_policy

def ensure_visual_policy(project_id: str, user_id: str) -> Dict[str, Any]:
    project = db_get("projects", id=project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    policy = merged_visual_policy(project)
    if load_json(project.get("visual_policy"), None) != policy:
        db_update("projects", project_id, {"visual_policy": policy})
    return policy

def add_project_activity(project_id: str, user_id: str, event_type: str, title: str, detail: str = "", meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row = db_insert(
        "project_activity_logs",
        {
            "project_id": project_id,
            "user_id": user_id,
            "event_type": event_type,
            "activity_type": event_type,
            "title": title,
            "detail": detail,
            "meta": meta or {},
        },
    )
    return activity_row_to_dict(row)

def list_project_activity(project_id: str, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    rows = db_list("project_activity_logs", project_id=project_id, user_id=user_id, limit=limit)
    return [activity_row_to_dict(row) for row in rows]

def create_project_asset(
    project_id: str,
    user_id: str,
    asset_type: str,
    title: str,
    prompt: str,
    section: Optional[str] = None,
    job_kind: Optional[str] = None,
    status: str = "completed",
    preview_url: Optional[str] = None,
    master_url: Optional[str] = None,
    print_url: Optional[str] = None,
    source_file_url: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = db_insert(
        "project_assets",
        {
            "project_id": project_id,
            "user_id": user_id,
            "asset_type": asset_type,
            "title": title,
            "prompt": prompt,
            "section": section or "general",
            "job_kind": job_kind or "asset_generation",
            "status": status,
            "preview_url": preview_url,
            "master_url": master_url,
            "print_url": print_url,
            "source_file_url": source_file_url,
            "meta": meta or {},
        },
    )
    return asset_row_to_dict(row)

def asset_row_to_dict(row):
    if not row:
        return {}

    item = dict(row)

    return {
        "id": item.get("id"),
        "project_id": item.get("project_id"),
        "user_id": item.get("user_id"),
        "asset_type": item.get("asset_type"),
        "section": item.get("section"),
        "job_kind": item.get("job_kind"),
        "title": item.get("title"),
        "prompt": item.get("prompt"),
        "status": item.get("status"),
        "preview_url": item.get("preview_url"),
        "master_url": item.get("master_url"),
        "print_url": item.get("print_url"),
        "source_file_url": item.get("source_file_url"),
        "meta": load_json(item.get("meta")),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
    }


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

def queue_agent_job_with_activity(
    project_id: str,
    user_id: str,
    agent_type: str,
    job_type: str,
    title: str,
    input_data: Optional[Dict[str, Any]] = None,
    priority: int = 5,
) -> Dict[str, Any]:
    job = db_insert(
        "agent_jobs",
        {
            "project_id": project_id,
            "user_id": user_id,
            "agent_type": agent_type,
            "job_type": job_type,
            "title": title,
            "priority": priority,
            "status": "queued",
            "input_data": input_data or {},
            "output_data": {},
        },
    )
    add_project_activity(project_id, user_id, "job.queued", title, detail=f"{agent_type} job queued", meta={"job_id": job["id"], "job_type": job_type})
    return job_row_to_dict(job)

def list_agent_jobs(project_id: str, user_id: str) -> List[Dict[str, Any]]:
    return [job_row_to_dict(row) for row in db_list("agent_jobs", project_id=project_id, user_id=user_id, limit=500)]

def update_project_media_rollups(project_id: str, user_id: str) -> Dict[str, Any]:
    assets = list_project_assets(project_id, user_id)
    rollup = {
        "assets_total": len(assets),
        "moodboards": len([a for a in assets if a.get("asset_type") == "moodboard"]),
        "renders": len([a for a in assets if a.get("asset_type") in {"3d_render", "scene_preview"}]),
        "cad": len([a for a in assets if a.get("asset_type") == "cad_layout"]),
        "references": len([a for a in assets if a.get("asset_type") == "reference"]),
    }
    return db_update("projects", project_id, {"media_rollup": rollup})

def build_concept_visual_prompts(project: Dict[str, Any], concept: Dict[str, Any], count: int = 3) -> List[str]:
    base = (
        f"Create a premium 16:9 event design visual for project '{project.get('project_name')}'. "
        f"Concept: {concept.get('name')}. "
        f"Summary: {concept.get('summary')}. "
        f"Colors: {', '.join(concept.get('colors') or [])}. "
        f"Style: {concept.get('style')}. "
        "High-end stage design, scenic depth, branded environment, realistic lighting, production design."
    )
    prompt_variations = [
        base + " Moodboard board with textures, references, scenic material samples, typography cues.",
        base + " Hero stage render perspective from audience center, cinematic reveal angle.",
        base + " Wide venue overview showing entry, stage, LED surfaces, and guest flow.",
        base + " Top-down stage + audience composition board with lighting mood and scenic composition.",
        base + " VIP experience corner, premium entry, hospitality, and branded feature moments.",
        base + " Alternate hero perspective with stronger lighting contrast and immersive screens.",
    ]
    return prompt_variations[: max(1, min(count, len(prompt_variations)))]

def sync_create_visual_asset(
    project: Dict[str, Any],
    user_id: str,
    asset_type: str,
    title: str,
    prompt: str,
    section: Optional[str] = None,
    job_kind: Optional[str] = None,
    size: str = "1536x1024",
    quality: str = "high",
) -> Dict[str, Any]:
    data_url = generate_image_data_url(prompt, size=size, quality=quality)
    if not data_url:
        raise HTTPException(status_code=500, detail="Image generation failed")
    rel, public_url = persist_data_url_image(data_url, RENDER_OUTPUT_DIR, safe_filename(asset_type))
    asset = create_project_asset(
        str(project["id"]),
        user_id,
        asset_type=asset_type,
        title=title,
        prompt=prompt,
        section=section,
        job_kind=job_kind or "sync_generation",
        status="completed",
        preview_url=public_url,
        master_url=public_url,
        print_url=public_url,
        source_file_url=public_url,
        meta={"saved_path": rel, "size": size, "quality": quality},
    )
    add_project_activity(str(project["id"]), user_id, "asset.generated", title, detail=f"{asset_type} generated", meta={"asset_id": asset["id"]})
    return asset

def create_separate_render_view_assets(project: Dict[str, Any], user_id: str, concept: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompts = [
        ("Hero Perspective", f"Create a premium 16:9 hero camera view for event concept {concept.get('name')} with cinematic audience-facing perspective."),
        ("Wide Perspective", f"Create a premium 16:9 wide venue render for event concept {concept.get('name')} showing full stage and audience arrangement."),
        ("Top View", f"Create a premium 16:9 aerial top-view render for event concept {concept.get('name')} showing layout and zoning."),
        ("Side View", f"Create a premium 16:9 side perspective render for event concept {concept.get('name')} showing scenic depth and LED surfaces."),
    ]
    assets = []
    for title, short_prompt in prompts:
        full_prompt = (
            f"{short_prompt} Project: {project.get('project_name')}. "
            f"Style: {concept.get('style')}. Colors: {', '.join(concept.get('colors') or [])}. "
            "Realistic event production design, scenic details, truss, lighting, premium look."
        )
        assets.append(
            sync_create_visual_asset(
                project,
                user_id,
                asset_type="3d_render",
                title=f"{concept.get('name')} {title}",
                prompt=full_prompt,
                section="renders",
                job_kind="separate_render_view",
            )
        )
    update_project_media_rollups(str(project["id"]), user_id)
    return assets


# ------------------------------------------------------------------------------
# CAD helpers
# ------------------------------------------------------------------------------

def default_cad_layout(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "venue_width_m": 40,
        "venue_depth_m": 30,
        "ceiling_height_m": 8,
        "dimensions": "40m x 30m",
        "total_area": "1200 sqm",
        "capacity": "500 pax",
        "zones": [
            {"name": "Main Stage", "zone_type": "stage", "x_m": 11, "y_m": 2, "width_m": 18, "depth_m": 8, "area_m2": 144},
            {"name": "Audience Seating", "zone_type": "seating", "x_m": 6, "y_m": 12, "width_m": 28, "depth_m": 12, "area_m2": 336},
            {"name": "VIP Lounge", "zone_type": "vip", "x_m": 31, "y_m": 12, "width_m": 6, "depth_m": 6, "area_m2": 36},
            {"name": "Registration", "zone_type": "registration", "x_m": 2, "y_m": 24, "width_m": 8, "depth_m": 4, "area_m2": 32},
            {"name": "Circulation Spine", "zone_type": "circulation", "x_m": 18, "y_m": 10, "width_m": 4, "depth_m": 16, "area_m2": 64},
            {"name": "Backstage Service", "zone_type": "service", "x_m": 29, "y_m": 2, "width_m": 7, "depth_m": 6, "area_m2": 42},
        ],
    }

def create_cad_layout(project: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    selected = project.get("selected_concept") or {}
    layout = default_cad_layout(project)
    layout["concept_name"] = selected.get("name")
    layout["project_name"] = project.get("project_name")
    row = db_insert(
        "cad_layouts",
        {
            "project_id": str(project["id"]),
            "user_id": user_id,
            "layout_type": "professional_event_layout",
            "layout_data": layout,
            "title": f"{project.get('project_name', 'Project')} CAD Layout",
        },
    )
    add_project_activity(str(project["id"]), user_id, "cad.generated", row.get("title", "CAD layout"), detail="CAD layout generated")
    return {**row, "layout_data": layout}

def _gen_dxf(layout: Dict[str, Any], project_name: str) -> str:
    venue_width = float(layout.get("venue_width_m") or 40)
    venue_depth = float(layout.get("venue_depth_m") or 30)
    zones = layout.get("zones") or []
    scale = 1000
    handle = [0]
    output: List[str] = []

    def nxt() -> str:
        handle[0] += 1
        return format(handle[0], "X")

    def g(code: int, value: Any) -> None:
        output.append(f"{code:>3}")
        output.append(str(value))

    def line(x1: float, y1: float, x2: float, y2: float, layer: str = "0", color: int = 7) -> None:
        g(0, "LINE"); g(5, nxt()); g(100, "AcDbEntity"); g(8, layer); g(62, color); g(100, "AcDbLine")
        g(10, x1); g(20, y1); g(30, 0); g(11, x2); g(21, y2); g(31, 0)

    def polyline(points: List[Tuple[float, float]], layer: str = "0", color: int = 7) -> None:
        g(0, "LWPOLYLINE"); g(5, nxt()); g(100, "AcDbEntity"); g(8, layer); g(62, color); g(100, "AcDbPolyline")
        g(90, len(points)); g(70, 1)
        for x_val, y_val in points:
            g(10, x_val); g(20, y_val)

    def text(x_val: float, y_val: float, height: float, value: str, layer: str = "0", color: int = 7) -> None:
        g(0, "TEXT"); g(5, nxt()); g(100, "AcDbEntity"); g(8, layer); g(62, color); g(100, "AcDbText")
        g(10, x_val); g(20, y_val); g(30, 0); g(40, height); g(1, value[:120])

    g(0, "SECTION"); g(2, "HEADER"); g(9, "$ACADVER"); g(1, "AC1024"); g(0, "ENDSEC")
    g(0, "SECTION"); g(2, "ENTITIES")
    polyline([(0, 0), (venue_width * scale, 0), (venue_width * scale, venue_depth * scale), (0, venue_depth * scale)], "VENUE", 7)
    x = scale
    while x < venue_width * scale:
        line(x, 0, x, venue_depth * scale, "GRID", 9)
        x += scale
    y = scale
    while y < venue_depth * scale:
        line(0, y, venue_width * scale, y, "GRID", 9)
        y += scale

    zone_colors = {"stage": 1, "seating": 3, "vip": 4, "circulation": 8, "service": 6, "registration": 5, "catering": 2}
    for zone in zones:
        x1 = float(zone.get("x_m", 0)) * scale
        y1 = float(zone.get("y_m", 0)) * scale
        width = float(zone.get("width_m", 5)) * scale
        depth = float(zone.get("depth_m", 5)) * scale
        x2 = x1 + width
        y2 = y1 + depth
        layer = f"ZONE-{str(zone.get('zone_type', 'general')).upper()}"
        color = zone_colors.get(str(zone.get("zone_type", "general")).lower(), 7)
        polyline([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], layer, color)
        text(x1 + width / 2, y1 + depth / 2, 180, str(zone.get("name", "Zone")), "TEXT", 7)

    text(venue_width * scale + 1000, venue_depth * scale - 1500, 220, project_name.upper(), "TITLE", 7)
    text(venue_width * scale + 1000, venue_depth * scale - 2200, 120, f"{venue_width:.1f}m x {venue_depth:.1f}m", "TITLE", 7)
    g(0, "ENDSEC"); g(0, "EOF")
    return "\n".join(output)

def export_dxf_for_project(project: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    rows = db_list("cad_layouts", project_id=str(project["id"]), user_id=user_id, limit=1)
    if rows:
        row = rows[0]
        layout = load_json(row.get("layout_data"), {}) or {}
    else:
        row = create_cad_layout(project, user_id)
        layout = row.get("layout_data") or {}
    project_name = project.get("project_name") or "Project"
    dxf_content = _gen_dxf(layout, project_name)
    filename = f"{safe_filename(project_name)}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.dxf"
    out_path = CAD_DIR / filename
    out_path.write_text(dxf_content, encoding="utf-8")
    rel = relative_public_url(out_path)
    return {"filename": filename, "download_url": absolute_public_url(rel), "relative_url": rel, "size_bytes": len(dxf_content.encode("utf-8"))}


# ------------------------------------------------------------------------------
# Voice session helpers
# ------------------------------------------------------------------------------

def get_project_by_id(project_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    return db_get("projects", id=project_id, user_id=user_id)

def create_voice_session(user_id: str, project_id: Optional[str], title: str, system_prompt: Optional[str], voice: Optional[str]) -> Dict[str, Any]:
    row = db_insert(
        "voice_sessions",
        {
            "user_id": user_id,
            "project_id": project_id,
            "title": title,
            "system_prompt": system_prompt or "You are a helpful live-event creative assistant.",
            "voice": voice or TTS_VOICE,
            "last_used_at": now_iso(),
        },
    )
    return row

def get_voice_session_by_id(session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    return db_get("voice_sessions", id=session_id, user_id=user_id)

def list_voice_sessions(user_id: str, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if project_id:
        return db_list("voice_sessions", user_id=user_id, project_id=project_id, limit=100)
    return db_list("voice_sessions", user_id=user_id, limit=100)

def touch_voice_session(session_id: str, user_id: str, voice: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"last_used_at": now_iso()}
    if voice:
        payload["voice"] = voice
    session = get_voice_session_by_id(session_id, user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Voice session not found")
    return db_update("voice_sessions", session_id, payload)

def add_voice_message(
    session_id: str,
    role: str,
    text: str,
    transcript: Optional[str] = None,
    audio_url: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = db_insert(
        "voice_messages",
        {
            "session_id": session_id,
            "role": role,
            "text": text,
            "transcript": transcript,
            "audio_url": audio_url,
            "meta": meta or {},
        },
    )
    return row

def get_voice_messages(session_id: str, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    session = get_voice_session_by_id(session_id, user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Voice session not found")
    return db_list("voice_messages", session_id=session_id, limit=limit, desc=False)

def generate_voice_reply(current_user: Dict[str, Any], session: Dict[str, Any], project: Optional[Dict[str, Any]], user_text: str, prior_messages: List[Dict[str, Any]]) -> str:
    fallback = (
        f"Got it, {current_user.get('full_name') or 'there'}. "
        f"I understood: {user_text[:180]}. "
        "I can help refine the concept, build assets, and continue the show-planning flow."
    )
    if not _openai_client:
        return fallback

    convo_lines = []
    for msg in prior_messages[-10:]:
        convo_lines.append(f"{msg.get('role', 'user')}: {msg.get('text') or msg.get('transcript') or ''}")
    convo = "\n".join(convo_lines)
    project_context = dump_json(
        {
            "project_name": project.get("project_name") if project else None,
            "event_type": project.get("event_type") if project else None,
            "selected_concept": project.get("selected_concept") if project else None,
        }
    )

    try:
        response = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": session.get("system_prompt") or "You are a helpful live-event creative assistant."},
                {"role": "user", "content": f"Project context:\n{project_context}\n\nConversation so far:\n{convo}\n\nLatest user message:\n{user_text}"},
            ],
            temperature=0.6,
            max_tokens=500,
        )
        return (response.choices[0].message.content or fallback).strip()
    except Exception:
        return fallback


# ------------------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------------------

@field_validator("password")
@classmethod
def validate_password_length(cls, value: str) -> str:
    if len(value.encode("utf-8")) > 72:
        raise ValueError("Password must be 72 bytes or less")
    return value

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        email = value.strip().lower()
        if not EMAIL_RE.match(email):
            raise ValueError("Invalid email address")
        return email

class SignupIn(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=72)
    full_name: Optional[str] = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        email = value.strip().lower()
        if not EMAIL_RE.match(email):
            raise ValueError("Invalid email address")
        return email

    @field_validator("password")
    @classmethod
    def validate_password_length(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be 72 bytes or less")
        return value

class LoginIn(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()

class ProjectCreateInput(BaseModel):
    title: Optional[str] = None
    name: Optional[str] = None
    brief: Optional[str] = None
    event_type: Optional[str] = None
    style_direction: Optional[str] = None
    style_theme: Optional[str] = None

class RunInput(BaseModel):
    text: str = Field(min_length=3)
    project_id: Optional[str] = None
    name: Optional[str] = None
    event_type: Optional[str] = None
    style_direction: Optional[str] = None

class RunProjectInput(BaseModel):
    text: Optional[str] = None
    event_type: Optional[str] = None
    style_direction: Optional[str] = None

class SelectConceptInput(BaseModel):
    project_id: str
    index: int = Field(ge=0, le=2)

class SelectCompatInput(BaseModel):
    concept_index: Optional[int] = Field(default=None, ge=0, le=2)
    index: Optional[int] = Field(default=None, ge=0, le=2)

class CommentInput(BaseModel):
    project_id: str
    section: str = "general"
    comment_text: str

class UpdateProjectFieldInput(BaseModel):
    project_id: str
    field: str
    value: Any

class DepartmentPdfInput(BaseModel):
    title: Optional[str] = None

class ArmInput(BaseModel):
    armed: bool = True

class CueJumpInput(BaseModel):
    cue_index: Optional[int] = Field(default=None, ge=0)
    cue_no: Optional[int] = None

class AssetCreateInput(BaseModel):
    asset_type: str
    title: str
    prompt: str = Field(min_length=3)
    section: Optional[str] = None
    job_kind: Optional[str] = None
    generate_now: bool = True

class MoodboardGenerateInput(BaseModel):
    concept_index: Optional[int] = Field(default=None, ge=0, le=2)
    count: int = Field(default=3, ge=1, le=6)
    generate_now: bool = True

class JobCreateInput(BaseModel):
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

class GraphicsGenerateInput(BaseModel):
    concept_index: Optional[int] = 0
    count: int = Field(default=3, ge=1, le=6)
    generate_now: bool = True
    feedback: Optional[str] = None

@app.post("/projects/{project_id}/graphics/generate")
def generate_2d_graphics_endpoint(
    project_id: str,
    payload: GraphicsGenerateInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    concepts = project.get("concepts") or []
    if not concepts:
        raise HTTPException(status_code=400, detail="Run pipeline first to generate concepts")

    idx = payload.concept_index or 0
    if idx >= len(concepts):
        raise HTTPException(status_code=400, detail="Invalid concept index")

    concept = concepts[idx]
    concept_name = concept.get("name") or "Concept"
    feedback = (payload.feedback or "").strip()

    prompts = []
    for i in range(1, payload.count + 1):
        prompts.append(
            f"Create a premium 2D event graphic for project {project.get('project_name') or project.get('name') or 'Project'}, "
            f"concept {concept_name}. {concept.get('summary') or ''} "
            f"{'Feedback: ' + feedback if feedback else ''} "
            "Luxury event branding, typography, signage, key visual, high-end presentation style."
        )

    assets: List[Dict[str, Any]] = []
    for i, prompt in enumerate(prompts, start=1):
        title = f"{concept_name} 2D Graphic {i}"
        if payload.generate_now:
            asset = sync_create_visual_asset(
                project,
                user_id,
                asset_type="2d_graphic",
                title=title,
                prompt=prompt,
                section="2d_graphics",
                job_kind="concept_2d_graphic",
            )
            assets.append(asset)
        else:
            asset = create_project_asset(
                project_id,
                user_id,
                asset_type="2d_graphic",
                title=title,
                prompt=prompt,
                section="2d_graphics",
                job_kind="concept_2d_graphic",
                status="queued",
                meta={"queued_only": True},
            )
            assets.append(asset)

    update_project_media_rollups(project_id, user_id)
    add_project_activity(
        project_id,
        user_id,
        "graphics.generated",
        "2D graphics generated",
        detail=f"{len(assets)} 2D graphics created",
        meta={"concept_name": concept_name},
    )

    return {"message": "2D graphics generated", "assets": assets}

class Generate2DCompatInput(BaseModel):
    project_id: str
    concept_id: Optional[str] = None
    concept_index: Optional[int] = 0
    prompt: Optional[str] = None
    format: Optional[str] = "poster"
    size: Optional[str] = "1536x1024"
    count: int = Field(default=1, ge=1, le=6)
    feedback: Optional[str] = None


@app.post("/ai/generate-2d")
def ai_generate_2d_compat(
    payload: Generate2DCompatInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = str(current_user["id"])
    project = get_project_by_id(payload.project_id, user_id=user_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if payload.prompt and payload.prompt.strip():
        asset = sync_create_visual_asset(
            project,
            user_id,
            asset_type="2d_graphic",
            title="AI Generated 2D Graphic",
            prompt=payload.prompt.strip(),
            section="2d_graphics",
            job_kind="ai_generate_2d",
            size=payload.size or "1536x1024",
            quality="high",
        )

        update_project_media_rollups(payload.project_id, user_id)

        add_project_activity(
            payload.project_id,
            user_id,
            "graphics.generated",
            "2D graphic generated",
            detail="Generated from /ai/generate-2d",
            meta={
                "asset_id": asset.get("id"),
                "format": payload.format,
                "concept_id": payload.concept_id,
            },
        )

        return {
            "ok": True,
            "message": "2D graphic generated",
            "asset": asset,
            "assets": [asset],
        }

    return generate_2d_graphics_endpoint(
        payload.project_id,
        GraphicsGenerateInput(
            concept_index=payload.concept_index or 0,
            count=payload.count,
            generate_now=True,
            feedback=payload.feedback,
        ),
        current_user,
    )

class VisualPolicyInput(BaseModel):
    preview_size: Optional[str] = None
    master_size: Optional[str] = None
    print_size: Optional[str] = None
    aspect_ratio: Optional[str] = None

class DXFInput(BaseModel):
    project_id: str

class PDFInput(BaseModel):
    project_id: str
    template: Optional[str] = "executive"

class ImageGenerateInput(BaseModel):
    prompt: str
    size: str = "1536x1024"
    quality: str = "high"
    project_id: Optional[str] = None
    section: Optional[str] = None

class TTSInput(BaseModel):
    text: str = Field(min_length=1, max_length=4096)
    voice: Optional[str] = None
    instructions: Optional[str] = None

class VoiceSessionCreateInput(BaseModel):
    project_id: Optional[str] = None
    title: Optional[str] = None
    system_prompt: Optional[str] = None
    voice: Optional[str] = None

class VoiceChatTextInput(BaseModel):
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    text: str = Field(min_length=1)
    voice: Optional[str] = None
    voice_instructions: Optional[str] = None
    title: Optional[str] = None
    system_prompt: Optional[str] = None


# ------------------------------------------------------------------------------
# Error handler
# ------------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ------------------------------------------------------------------------------
#  / auth routes
# ------------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": f"{APP_NAME} running",
        "version": APP_VERSION,
        "time": now_iso(),
        "docs": "/docs",
        "openai": bool(_openai_client),
        "supabase": bool(_sb),
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "BriefCraft-AI backend",
        "openai": _openai_client is not None,
        "supabase": _sb is not None,
        "supabase_env": bool(SUPABASE_URL and SUPABASE_KEY),
        "storage_bucket": SUPABASE_STORAGE_BUCKET or None,
    }

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
            "plan": "studio_pro",
            "projects_used": 0,
            "projects_limit": 100,
        },
    )

    token = create_access_token(user_id)
    safe_user = {k: v for k, v in user.items() if k != "password"}

    return {
        "message": "User created",
        "user_id": user_id,
        "access_token": token,
        "token": token,
        "token_type": "bearer",
        "user": safe_user,
    }

@app.post("/login")
def login(payload: LoginIn):
    user = db_get("users", email=payload.email)
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    if not verify_password(payload.password, user.get("password", "")):
        raise HTTPException(status_code=400, detail="Wrong password")
    token = create_access_token(str(user["id"]))
    safe_user = {k: v for k, v in user.items() if k != "password"}
    return {
        "access_token": token,
        "token": token,
        "token_type": "bearer",
        "user_id": str(user["id"]),
        "user": safe_user,
    }

@app.post("/logout")
def logout(_: Dict[str, Any] = Depends(get_current_user)):
    return {"message": "Logged out. Remove bearer token on client."}

@app.get("/me")
def me(user: Dict[str, Any] = Depends(get_current_user)):
    return {"user": {k: v for k, v in user.items() if k != "password"}}


# ------------------------------------------------------------------------------
# Project routes
# ------------------------------------------------------------------------------

@app.get("/projects")
def list_projects(current_user: Dict[str, Any] = Depends(get_current_user)):
    return {"projects": db_list("projects", user_id=str(current_user["id"]), limit=500)}

@app.post("/projects")
def create_project(payload: ProjectCreateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    return db_insert(
        "projects",
        {
            "user_id": str(current_user["id"]),
            "project_name": (payload.title or payload.name or "Untitled").strip(),
            "brief_text": payload.brief,
            "event_type": payload.event_type,
            "style_direction": payload.style_direction,
            "style_theme": payload.style_theme or "luxury",
            "status": "draft",
            "visual_policy": merged_visual_policy({}),
        },
    )

@app.get("/projects/{project_id}")
@app.get("/project/{project_id}")
def get_project(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": project}

@app.post("/project/update")
def update_project_field(payload: UpdateProjectFieldInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(payload.project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    updated = db_update("projects", payload.project_id, {payload.field: payload.value})
    return {"message": "Updated", "project": updated}

def _run_logic(project: Dict[str, Any], text: str, event_type: Optional[str], user_id: str) -> Dict[str, Any]:
    if text and project.get("brief_text") != text:
        project = db_update("projects", str(project["id"]), {"brief_text": text})
    if event_type and not project.get("event_type"):
        project = db_update("projects", str(project["id"]), {"event_type": event_type})

    analysis = project.get("analysis")
    if not analysis:
        analysis = analyze_brief(project.get("brief_text") or text, project.get("event_type") or event_type)
        project = db_update("projects", str(project["id"]), {"analysis": analysis})

    concepts = project.get("concepts")
    if not concepts:
        concepts = generate_concepts(project.get("brief_text") or text, analysis, project.get("event_type") or event_type)
        project = db_update("projects", str(project["id"]), {"concepts": concepts, "status": "concepts_ready"})
        add_project_activity(str(project["id"]), user_id, "concepts.generated", "Concepts generated", detail="Three creative concepts prepared")

    return {
        "message": "Pipeline completed",
        "project_id": str(project["id"]),
        "status": "concepts_ready",
        "brief": project.get("brief_text"),
        "analysis": analysis,
        "concepts": concepts,
        "project": project,
    }

@app.post("/run")
def run_pipeline(payload: RunInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    if payload.project_id:
        project = get_project_by_id(payload.project_id, user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
    else:
        project = db_insert(
            "projects",
            {
                "user_id": user_id,
                "project_name": (payload.name or payload.text[:50]).strip(),
                "brief_text": payload.text,
                "event_type": payload.event_type,
                "style_direction": payload.style_direction,
                "status": "draft",
                "visual_policy": merged_visual_policy({}),
            },
        )
    return _run_logic(project, payload.text, payload.event_type, user_id)

@app.post("/projects/{project_id}/run")
def run_project(project_id: str, payload: RunProjectInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    text = (payload.text or project.get("brief_text") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text required")
    return _run_logic(project, text, payload.event_type or project.get("event_type"), str(current_user["id"]))

@app.post("/select")
def select_concept(payload: SelectConceptInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(payload.project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    concepts = project.get("concepts") or []
    if not concepts:
        raise HTTPException(status_code=400, detail="Run pipeline first")
    if payload.index >= len(concepts):
        raise HTTPException(status_code=400, detail=f"Only {len(concepts)} concepts")
    selected = concepts[payload.index]
    updated = db_update("projects", payload.project_id, {"selected_concept": selected, "status": "concept_selected"})
    add_project_activity(payload.project_id, str(current_user["id"]), "concept.selected", selected.get("name", "Concept"), detail="Concept selected")
    return {"message": "Concept selected", "index": payload.index, "selected": selected, "project": updated}

@app.post("/projects/{project_id}/select-concept")
def select_concept_compat(project_id: str, payload: SelectCompatInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    idx = payload.concept_index if payload.concept_index is not None else payload.index
    if idx is None:
        raise HTTPException(status_code=422, detail="concept_index required")
    return select_concept(SelectConceptInput(project_id=project_id, index=idx), current_user)


# ------------------------------------------------------------------------------
# Comments
# ------------------------------------------------------------------------------

@app.post("/comment")
@app.post("/comments")
def add_comment(payload: CommentInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(payload.project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    comment = db_insert(
        "project_comments",
        {
            "project_id": payload.project_id,
            "user_id": str(current_user["id"]),
            "section": payload.section,
            "content": payload.comment_text,
            "author_name": current_user.get("full_name") or "Anonymous",
        },
    )
    add_project_activity(payload.project_id, str(current_user["id"]), "comment.added", "Comment added", detail=payload.section)
    return {"message": "Comment added", "comment": comment}

@app.get("/comments/{project_id}")
def list_comments(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"comments": db_list("project_comments", project_id=project_id, limit=200)}


# ------------------------------------------------------------------------------
# Departments
# ------------------------------------------------------------------------------
def console_state(project: Dict[str, Any]) -> Dict[str, Any]:
    state = project.get("department_outputs") or {}

    if isinstance(state, str):
        state = load_json(state, {}) or {}

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


def build_departments_logic(project_id: str, user_id: str) -> Dict[str, Any]:
    project = get_project_by_id(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.get("selected_concept"):
        raise HTTPException(status_code=400, detail="Select a concept first")

    sound_data = default_sound_plan(project)
    lighting_data = default_lighting_plan(project)
    showrunner_data = default_showrunner_plan(project)

    state = console_state(project)
    state.update({
        "sound_ready": True,
        "lighting_ready": True,
        "showrunner_ready": True,
        "console_index": 0,
        "hold": False,
    })

    updated = db_update(
        "projects",
        project_id,
        {
            "sound_data": sound_data,
            "lighting_data": lighting_data,
            "showrunner_data": showrunner_data,
            "department_outputs": state,
            "scene_json": build_scene_json(project),
            "status": "departments_ready",
        },
    )

    add_project_activity(
        project_id,
        user_id,
        "departments.generated",
        "Departments generated",
        detail="Sound, lighting, showrunner ready",
    )

    return {
        "message": "Departments generated",
        "project_id": project_id,
        "sound_data": sound_data,
        "lighting_data": lighting_data,
        "showrunner_data": showrunner_data,
        "project": updated,
    }


def build_departments_logic(project_id: str, user_id: str) -> Dict[str, Any]:
    project = get_project_by_id(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.get("selected_concept"):
        raise HTTPException(status_code=400, detail="Select a concept first")

    sound_data = default_sound_plan(project)
    lighting_data = default_lighting_plan(project)
    showrunner_data = default_showrunner_plan(project)
    state = console_state(project)
    state.update({"sound_ready": True, "lighting_ready": True, "showrunner_ready": True, "console_index": 0, "hold": False})
    updated = db_update(
        "projects",
        project_id,
        {
            "sound_data": sound_data,
            "lighting_data": lighting_data,
            "showrunner_data": showrunner_data,
            "department_outputs": state,
            "scene_json": build_scene_json(project),
            "status": "departments_ready",
        },
    )
    add_project_activity(project_id, user_id, "departments.generated", "Departments generated", detail="Sound, lighting, showrunner ready")
    return {
        "message": "Departments generated",
        "project_id": project_id,
        "sound_data": sound_data,
        "lighting_data": lighting_data,
        "showrunner_data": showrunner_data,
        "project": updated,
    }

@app.post("/project/{project_id}/departments/build")
@app.post("/projects/{project_id}/generate-departments")
def build_departments(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    return build_departments_logic(project_id, str(current_user["id"]))

@app.get("/project/{project_id}/departments/manuals")
def get_department_manuals(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        "project_id": project_id,
        "sound_data": project.get("sound_data"),
        "lighting_data": project.get("lighting_data"),
        "showrunner_data": project.get("showrunner_data"),
    }

@app.post("/projects/{project_id}/departments/pdf/sound")
@app.post("/project/{project_id}/departments/pdf/sound")
def pdf_sound(project_id: str, payload: DepartmentPdfInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project or not project.get("sound_data"):
        raise HTTPException(status_code=404, detail="Sound data not found. Build departments first.")
    sections = project["sound_data"].get("pdf_sections") or [{"heading": "Sound", "body": dump_json(project["sound_data"])}]
    return {"project_id": project_id, **create_simple_pdf(payload.title or "Sound Design Manual", sections, "sound_manual")}

@app.post("/projects/{project_id}/departments/pdf/lighting")
@app.post("/project/{project_id}/departments/pdf/lighting")
def pdf_lighting(project_id: str, payload: DepartmentPdfInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project or not project.get("lighting_data"):
        raise HTTPException(status_code=404, detail="Lighting data not found. Build departments first.")
    sections = project["lighting_data"].get("pdf_sections") or [{"heading": "Lighting", "body": dump_json(project["lighting_data"])}]
    return {"project_id": project_id, **create_simple_pdf(payload.title or "Lighting Design Manual", sections, "lighting_manual")}

@app.post("/projects/{project_id}/departments/pdf/showrunner")
@app.post("/project/{project_id}/departments/pdf/showrunner")
def pdf_showrunner(project_id: str, payload: DepartmentPdfInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project or not project.get("showrunner_data"):
        raise HTTPException(status_code=404, detail="Showrunner data not found. Build departments first.")
    sections = project["showrunner_data"].get("pdf_sections") or [{"heading": "Showrunner", "body": dump_json(project["showrunner_data"])}]
    return {"project_id": project_id, **create_simple_pdf(payload.title or "Show Running Script", sections, "showrunner_manual")}

@app.post("/projects/{project_id}/manuals/master/pdf")
def master_manual_pdf(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    project_name = project.get("project_name") or "Project"
    sections = [
        {"heading": "Overview", "body": f"Project: {project_name}\nStatus: {project.get('status', 'draft')}"},
        {"heading": "Brief", "body": project.get("brief_text") or ""},
        {"heading": "Analysis", "body": dump_json(project.get("analysis") or {})},
        {"heading": "Sound", "body": dump_json(project.get("sound_data") or {})},
        {"heading": "Lighting", "body": dump_json(project.get("lighting_data") or {})},
        {"heading": "Showrunner", "body": dump_json(project.get("showrunner_data") or {})},
    ]
    return create_simple_pdf(f"Master Manual - {project_name}", sections, "master_manual")


# ------------------------------------------------------------------------------
# Show console
# ------------------------------------------------------------------------------

def _control_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    protocol = str(payload.get("protocol") or "simulated").lower()
    return {
        "ok": True,
        "protocol": protocol,
        "target": payload.get("target"),
        "value": payload.get("value"),
        "message": "Simulated action executed",
    }

@app.get("/project/{project_id}/show-console")
def show_console_status(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    cues = (project.get("showrunner_data") or {}).get("console_cues") or []
    state = console_state(project)
    idx = min(int(state.get("console_index", 0)), max(len(cues) - 1, 0)) if cues else 0
    return {
        "project_id": project_id,
        "armed": bool(state.get("armed")),
        "hold": bool(state.get("hold")),
        "cue_index": idx,
        "cue": cues[idx] if cues else None,
        "available_cues": cues,
        "history": state.get("execution_log") or [],
    }

@app.post("/project/{project_id}/show-console/arm")
def show_console_arm(project_id: str, payload: ArmInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    state = console_state(project)
    state["armed"] = bool(payload.armed)
    state = log_console_event(state, {"status": "armed" if payload.armed else "disarmed"})
    db_update("projects", project_id, {"department_outputs": state})
    return {"message": "Console updated", "armed": state["armed"]}

@app.post("/project/{project_id}/show-console/go")
def show_console_go(project_id: str, execute: bool = Query(True), current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    cues = (project.get("showrunner_data") or {}).get("console_cues") or []
    if not cues:
        raise HTTPException(status_code=400, detail="No cues found")
    state = console_state(project)
    if not state.get("armed"):
        raise HTTPException(status_code=400, detail="Console not armed")
    if state.get("hold"):
        raise HTTPException(status_code=400, detail="Console on hold")
    idx = min(int(state.get("console_index", 0)), len(cues) - 1)
    cue = cues[idx]
    results = [_control_action(action) for action in cue.get("actions", [])] if execute else []
    state["console_index"] = min(idx + 1, len(cues) - 1)
    state = log_console_event(state, {"status": "go", "cue_index": idx, "cue_name": cue.get("name")})
    db_update("projects", project_id, {"department_outputs": state})
    return {"message": "Cue executed", "cue_index": idx, "cue": cue, "results": results}

@app.post("/project/{project_id}/show-console/next")
def show_console_next(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    cues = (project.get("showrunner_data") or {}).get("console_cues") or []
    if not cues:
        raise HTTPException(status_code=400, detail="No cues")
    state = console_state(project)
    idx = min(int(state.get("console_index", 0)) + 1, len(cues) - 1)
    state["console_index"] = idx
    db_update("projects", project_id, {"department_outputs": state})
    return {"cue_index": idx, "cue": cues[idx]}

@app.post("/project/{project_id}/show-console/back")
def show_console_back(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    cues = (project.get("showrunner_data") or {}).get("console_cues") or []
    state = console_state(project)
    idx = max(int(state.get("console_index", 0)) - 1, 0)
    state["console_index"] = idx
    db_update("projects", project_id, {"department_outputs": state})
    return {"cue_index": idx, "cue": cues[idx] if cues else None}

@app.post("/project/{project_id}/show-console/hold")
def show_console_hold(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    state = console_state(project)
    state["hold"] = True
    db_update("projects", project_id, {"department_outputs": state})
    return {"message": "Hold engaged"}

@app.post("/project/{project_id}/show-console/standby")
def show_console_standby(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    state = console_state(project)
    state["hold"] = False
    db_update("projects", project_id, {"department_outputs": state})
    return {"message": "Standby"}

@app.post("/project/{project_id}/show-console/panic")
def show_console_panic(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    state = console_state(project)
    state["hold"] = True
    state["armed"] = False
    state = log_console_event(state, {"status": "panic"})
    db_update("projects", project_id, {"department_outputs": state})
    return {"message": "Panic - console disarmed and hold engaged"}

@app.post("/project/{project_id}/show-console/jump")
def show_console_jump(project_id: str, payload: CueJumpInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    cues = (project.get("showrunner_data") or {}).get("console_cues") or []
    if not cues:
        raise HTTPException(status_code=400, detail="No cues")
    idx = payload.cue_index
    if payload.cue_no is not None:
        matches = [i for i, cue in enumerate(cues) if str(cue.get("cue_no")) == str(payload.cue_no)]
        if not matches:
            raise HTTPException(status_code=404, detail="Cue not found")
        idx = matches[0]
    if idx is None:
        raise HTTPException(status_code=422, detail="cue_index or cue_no required")
    if idx < 0 or idx >= len(cues):
        raise HTTPException(status_code=400, detail="Index out of range")
    state = console_state(project)
    state["console_index"] = idx
    db_update("projects", project_id, {"department_outputs": state})
    return {"cue_index": idx, "cue": cues[idx]}

@app.get("/project/{project_id}/show-console/history")
def show_console_history(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"execution_log": console_state(project).get("execution_log") or []}

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
# Visual policy / assets / jobs / activity
# ------------------------------------------------------------------------------

@app.get("/projects/{project_id}/visual-policy")
def get_visual_policy_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    policy = ensure_visual_policy(project_id, str(current_user["id"]))
    return {"project_id": project_id, "visual_policy": policy}

@app.post("/projects/{project_id}/visual-policy")
def set_visual_policy_endpoint(project_id: str, payload: VisualPolicyInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
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
    updated = db_update("projects", project_id, {"visual_policy": policy})
    add_project_activity(project_id, str(current_user["id"]), "visual_policy.updated", "Visual policy updated", meta={"visual_policy": policy})
    return {"message": "Visual policy updated", "project": updated, "visual_policy": policy}

@app.get("/projects/{project_id}/assets")
def list_assets_endpoint(project_id: str, section: Optional[str] = Query(default=None), current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"assets": list_project_assets(project_id, str(current_user["id"]), section=section)}

@app.get("/projects/{project_id}/jobs")
def list_jobs_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"jobs": list_agent_jobs(project_id, str(current_user["id"]))}

@app.get("/projects/{project_id}/activity")
def list_activity_endpoint(project_id: str, limit: int = Query(default=100, ge=1, le=500), current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
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
    asset = create_project_asset(
        project_id,
        user_id,
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
    job = queue_agent_job_with_activity(project_id, user_id, agent_type=payload.asset_type, job_type=payload.job_kind or "asset_generation", title=payload.title, input_data={"asset_id": asset["id"], "prompt": payload.prompt})
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
    assets: List[Dict[str, Any]] = []
    queued_jobs: List[Dict[str, Any]] = []
    for i, prompt in enumerate(prompts, start=1):
        title = f"{concept.get('name') or 'Concept'} Moodboard {i}"
        if payload.generate_now:
            assets.append(sync_create_visual_asset(project, user_id, "moodboard", title, prompt, section="moodboard", job_kind="concept_moodboard"))
        else:
            asset = create_project_asset(project_id, user_id, "moodboard", title, prompt, "moodboard", "concept_moodboard", status="queued")
            job = queue_agent_job_with_activity(project_id, user_id, "moodboard", "concept_moodboard", title, input_data={"asset_id": asset["id"], "prompt": prompt})
            queued_jobs.append(job)
            assets.append(asset)
    update_project_media_rollups(project_id, user_id)
    return {"message": "Moodboards processed", "assets": assets, "jobs": queued_jobs}


def generate_separated_renders_logic(project_id: str, user_id: str):
    project = get_project_by_id(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    selected = project.get("selected_concept") or {}
    concept_name = selected.get("name") or "Event Concept"

    render_views = [
        {
            "title": f"{concept_name} Hero Perspective",
            "prompt": f"Create a premium 16:9 hero camera view for event concept {concept_name} with dramatic stage reveal, polished materials, cinematic lighting, ultra realistic 3D render."
        },
        {
            "title": f"{concept_name} Side View",
            "prompt": f"Create a premium 16:9 side perspective render for event concept {concept_name} showing scenic depth, LED surfaces, stage layout, ultra realistic 3D render."
        },
        {
            "title": f"{concept_name} Top View",
            "prompt": f"Create a premium 16:9 top or elevated view render for event concept {concept_name} showing stage zoning, guest layout, scenic structure, ultra realistic 3D render."
        },
    ]

    assets: List[Dict[str, Any]] = []

    for view in render_views:
        title = view["title"]
        prompt = view["prompt"]

        asset = sync_create_visual_asset(
            project,
            user_id,
            "3d_render",
            title,
            prompt,
            section="renders",
            job_kind="separate_render_view",
        )

        assets.append(asset_row_to_dict(asset))

    add_project_activity(
        project_id,
        user_id,
        "renders.generated",
        "Separate render views generated",
        detail=f"{len(assets)} separated render views created",
    )

    update_project_media_rollups(project_id, user_id)

    return {
        "message": "Separate render views generated",
        "assets": assets,
    }


@app.post("/projects/{project_id}/renders/generate-separated")
def generate_separated_renders(
    project_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return generate_separated_renders_logic(project_id, str(current_user["id"]))
    



@app.post("/projects/{project_id}/jobs/queue")
def queue_job_endpoint(project_id: str, payload: JobCreateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    job = queue_agent_job_with_activity(project_id, user_id, payload.agent_type, payload.job_type, payload.title or payload.job_type, payload.input_data or {}, priority=payload.priority)
    return {"message": "Job queued", "job": job}

@app.post("/projects/{project_id}/orchestrate")
def orchestrate_project_endpoint(project_id: str, payload: OrchestrateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    summary = {
        "queued_at": now_iso(),
        "queue_3d": payload.queue_3d,
        "queue_video": payload.queue_video,
        "queue_cad": payload.queue_cad,
        "queue_manuals": payload.queue_manuals,
        "auto_generate_moodboard": payload.auto_generate_moodboard,
    }
    jobs = []
    if payload.queue_cad:
        jobs.append(queue_agent_job_with_activity(project_id, user_id, "cad_layout", "cad_generation", "Professional CAD layout", {"project_id": project_id}))
    if payload.queue_3d:
        jobs.append(queue_agent_job_with_activity(project_id, user_id, "3d_render", "separate_render_views", "Separate 3D render views", {"project_id": project_id}))
    if payload.queue_video:
        jobs.append(queue_agent_job_with_activity(project_id, user_id, "show_trial", "show_trial_generation", "Show trial walkthrough", {"project_id": project_id}))
    if payload.queue_manuals:
        jobs.append(queue_agent_job_with_activity(project_id, user_id, "manuals", "manual_pack", "Department manual pack", {"project_id": project_id}))
    updated = db_update("projects", project_id, {"orchestration_data": summary})
    add_project_activity(project_id, user_id, "orchestration.updated", "Project orchestration updated", meta=summary)
    return {"message": "Orchestration updated", "project": updated, "orchestration": summary, "jobs": jobs}


# ------------------------------------------------------------------------------
# CAD / exports
# ------------------------------------------------------------------------------

@app.post("/projects/{project_id}/cad/generate")
def generate_cad_layout_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    layout = create_cad_layout(project, str(current_user["id"]))
    asset = create_project_asset(
        project_id,
        str(current_user["id"]),
        asset_type="cad_layout",
        title=layout.get("title") or "CAD Layout",
        prompt="Professional event CAD layout",
        section="cad",
        job_kind="cad_generation",
        status="completed",
        meta={"layout_data": layout.get("layout_data")},
    )
    update_project_media_rollups(project_id, str(current_user["id"]))
    return {"message": "CAD layout generated", "layout": layout, "asset": asset}

@app.get("/projects/{project_id}/cad")
def get_cad_layout_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    rows = db_list("cad_layouts", project_id=project_id, user_id=str(current_user["id"]), limit=20)
    return {"layouts": rows}

@app.post("/export/dxf")
def export_dxf(payload: DXFInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(payload.project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return export_dxf_for_project(project, str(current_user["id"]))

@app.post("/export/pdf")
def export_pdf(payload: PDFInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    project = get_project_by_id(payload.project_id, str(current_user["id"]))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    project_name = project.get("project_name") or "Project"
    sections = [
        {"heading": "Overview", "body": f"Project: {project_name}\nStatus: {project.get('status', 'draft')}"},
        {"heading": "Brief", "body": project.get("brief_text") or ""},
        {"heading": "Analysis", "body": dump_json(project.get("analysis") or {})},
    ]
    concepts = project.get("concepts") or []
    if concepts:
        sections.append({"heading": "Creative Concepts", "body": "\n\n".join(f"{c.get('name', '')}: {c.get('summary', '')}" for c in concepts)})
    return {"template": payload.template, **create_simple_pdf(project_name, sections, "project_export")}

@app.get("/exports/{filename}")
def download_export(filename: str):
    filename = safe_filename(filename)
    for root in (EXPORT_DIR, PDF_DIR, CAD_DIR):
        path = root / filename
        if path.exists():
            media_type = "application/pdf" if path.suffix.lower() == ".pdf" else "application/octet-stream"
            return FileResponse(path, media_type=media_type, filename=path.name)
    raise HTTPException(status_code=404, detail="File not found")


# ------------------------------------------------------------------------------
# Voice routes
# ------------------------------------------------------------------------------

@app.post("/voice/sessions")
def create_voice_session_endpoint(payload: VoiceSessionCreateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = None
    safe_project_id = None
    if payload.project_id:
        safe_project_id = ensure_uuid(payload.project_id, "project_id")
        project = get_project_by_id(safe_project_id, user_id=user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
    session = create_voice_session(user_id, safe_project_id, payload.title or "Voice Chat", payload.system_prompt, payload.voice)
    return {"message": "Voice session created", "session": session}

@app.get("/voice/sessions")
def list_voice_sessions_endpoint(project_id: Optional[str] = Query(default=None), current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    if project_id:
        project_id = ensure_uuid(project_id, "project_id")
    return {"sessions": list_voice_sessions(user_id, project_id=project_id)}

@app.get("/voice/sessions/{session_id}")
def get_voice_session_endpoint(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    session = get_voice_session_by_id(ensure_uuid(session_id, "session_id"), str(current_user["id"]))
    if not session:
        raise HTTPException(status_code=404, detail="Voice session not found")
    messages = get_voice_messages(str(session["id"]), str(current_user["id"]), limit=100)
    return {"session": session, "messages": messages}

@app.post("/voice/chat-text")
def voice_chat_text(payload: VoiceChatTextInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = None
    safe_project_id = ensure_uuid(payload.project_id, "project_id") if payload.project_id else None
    if safe_project_id:
        project = get_project_by_id(safe_project_id, user_id=user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    safe_session_id = ensure_uuid(payload.session_id, "session_id") if payload.session_id else None
    if safe_session_id:
        session = get_voice_session_by_id(safe_session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Voice session not found")
    else:
        session = create_voice_session(user_id, safe_project_id, payload.title or "Voice Chat", payload.system_prompt, payload.voice)
        safe_session_id = str(session["id"])

    prior_messages = get_voice_messages(safe_session_id, user_id, limit=30)
    add_voice_message(safe_session_id, "user", payload.text.strip(), transcript=payload.text.strip(), meta={"input_type": "text"})
    reply = generate_voice_reply(current_user, session, project, payload.text.strip(), prior_messages)
    audio = synthesize_speech(reply, voice=payload.voice or session.get("voice") or TTS_VOICE, instructions=payload.voice_instructions, filename_prefix="assistant_reply")
    assistant_message = add_voice_message(safe_session_id, "assistant", reply, transcript=reply, audio_url=audio["audio_url"], meta={"voice": audio["voice"]})
    touch_voice_session(safe_session_id, user_id, voice=audio["voice"])

    return {
        "message": "Voice response generated",
        "session_id": safe_session_id,
        "project_id": safe_project_id,
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
async def voice_transcribe(audio_file: UploadFile = File(...), _: Dict[str, Any] = Depends(get_current_user)):
    suffix = Path(audio_file.filename or "audio.webm").suffix or ".webm"
    saved_path = UPLOAD_DIR / f"transcribe_{uuid.uuid4().hex}{suffix}"
    content = await audio_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
    saved_path.write_bytes(content)
    transcript = transcribe_audio_file(saved_path)
    return {"message": "Transcription completed", "transcript": transcript, "input_audio_url": absolute_public_url(relative_public_url(saved_path))}


# ------------------------------------------------------------------------------
# Direct image generation endpoint
# ------------------------------------------------------------------------------

@app.post("/generate/image")
def generate_image_endpoint(payload: ImageGenerateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    image_data = generate_image_data_url(payload.prompt, payload.size, payload.quality)
    if not image_data:
        raise HTTPException(status_code=500, detail="Image generation failed")
    if image_data.startswith("data:image"):
        rel, public_url = persist_data_url_image(image_data, RENDER_OUTPUT_DIR, "generated_image")
        result = {"data_url": image_data, "saved_url": public_url, "saved_path": rel}
    else:
        result = {"image_url": image_data}
    if payload.project_id:
        create_project_asset(
            payload.project_id,
            str(current_user["id"]),
            asset_type="image",
            title=payload.prompt[:60],
            prompt=payload.prompt,
            section=payload.section or "general",
            job_kind="direct_image_generation",
            status="completed",
            preview_url=result.get("saved_url") or result.get("image_url"),
            master_url=result.get("saved_url") or result.get("image_url"),
            print_url=result.get("saved_url") or result.get("image_url"),
            source_file_url=result.get("saved_url") or result.get("image_url"),
            meta={"size": payload.size, "quality": payload.quality},
        )
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)
