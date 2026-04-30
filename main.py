from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, Request

import base64
import io
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from supabase import create_client
except Exception:
    create_client = None


# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------

load_dotenv()

APP_NAME = os.getenv("APP_TITLE", "BriefCraft-AI backend").strip() or "BriefCraft-AI backend"
APP_VERSION = "4.3.0"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY and OpenAI else None

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    or os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    or os.getenv("SUPABASE_KEY", "").strip()
)
SUPABASE_KEY = SUPABASE_SERVICE_ROLE_KEY

SUPABASE_STORAGE_BUCKET = (
    os.getenv("SUPABASE_STORAGE_BUCKET", "").strip()
    or os.getenv("STORAGE_BUCKET", "").strip()
    or "briefcraft-assets"
)

SECRET_KEY = (
    os.getenv("JWT_SECRET", "").strip()
    or os.getenv("SECRET_KEY", "").strip()
    or "change-me-32char-secret-key-xx"
)

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
    except Exception as e:
        print("Supabase client init failed:", e)
        _sb = None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ------------------------------------------------------------------------------
# Files / folders
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# FastAPI app / CORS
# ------------------------------------------------------------------------------

def _split_origins(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


ALLOWED_ORIGINS = _split_origins(
    os.getenv(
        "ALLOWED_ORIGINS",
        ",".join(
            [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "https://briefly-sparkle.lovable.app",
                "https://81db4809-ba40-464a-bd03-42e7f872691c.lovableproject.com",
            ]
        ),
    )
)

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://lovable.dev",
        "https://briefly-sparkle.lovable.app",
    ],
    allow_origin_regex=r"https://.*\.lovable\.app|https://.*\.lovableproject\.com",
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

    # --------------------------------------------------------------------------
    # CRITICAL FIX:
    # Never return Render-local /renders URLs for generated files.
    # Render local disk is temporary and causes 404.
    # --------------------------------------------------------------------------
    try:
        path.relative_to(RENDER_OUTPUT_DIR)

        if path.exists() and path.is_file():
            import mimetypes
            import time

            content_type = mimetypes.guess_type(str(path))[0] or "image/png"
            ext = path.suffix or ".png"
            clean_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", path.stem).strip("-._") or "render"

            storage_path = (
                f"projects/shared/renders/"
                f"{int(time.time())}-{uuid.uuid4().hex[:10]}-{clean_name}{ext}"
            )

            public_url = _upload_image_bytes_to_supabase(
                image_bytes=path.read_bytes(),
                storage_path=storage_path,
                content_type=content_type,
            )

            return public_url

        raise RuntimeError(f"Render output file does not exist: {path}")

    except ValueError:
        # Not inside RENDER_OUTPUT_DIR, continue normal public URL handling.
        pass

    for root, prefix in (
        (MEDIA_DIR, "/media"),
        (UPLOAD_DIR, "/uploads"),
        (EXPORT_DIR, "/exports"),
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


SECTION_NORMALIZER = {
    "moodboards": "moodboard",
    "mood_board": "moodboard",
    "mood-boards": "moodboard",
    "graphics_2d": "2d_graphics",
    "2d": "2d_graphics",
    "2d_graphic": "2d_graphics",
    "2d-graphics": "2d_graphics",
    "renders_3d": "renders",
    "3d": "renders",
    "3d_renders": "renders",
    "render": "renders",
}


def normalize_section(section: Optional[str], fallback: str = "general") -> str:
    raw = section or fallback
    return SECTION_NORMALIZER.get(raw, raw)


# ------------------------------------------------------------------------------
# Local/Supabase persistence helpers
# ------------------------------------------------------------------------------

def _deserialize_row(table: str, row: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(row)
    json_keys = {
        "analysis",
        "concepts",
        "selected_concept",
        "sound_data",
        "lighting_data",
        "showrunner_data",
        "department_outputs",
        "scene_json",
        "visual_policy",
        "input_data",
        "output_data",
        "meta",
        "trial_data",
        "media_rollup",
        "orchestration_data",
        "layout_data",
    }
    if table in {
        "projects",
        "project_assets",
        "agent_jobs",
        "project_activity_logs",
        "voice_sessions",
        "voice_messages",
        "cad_layouts",
        "show_trials",
    }:
        for key in json_keys:
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


def db_list(
    table: str,
    limit: int = 100,
    order_key: str = "created_at",
    desc: bool = True,
    **filters: Any,
) -> List[Dict[str, Any]]:
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

    rows: List[Dict[str, Any]] = []
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
    if not password_hash:
        return False
    try:
        return pwd_context.verify(password, password_hash)
    except Exception:
        return False


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

    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = get_user_by_id(str(user_id))
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


def synthesize_speech(
    text: str,
    voice: Optional[str] = None,
    instructions: Optional[str] = None,
    filename_prefix: str = "tts",
) -> Dict[str, str]:
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


# ------------------------------------------------------------------------------
# OpenAI image generation + Supabase Storage helpers
# ------------------------------------------------------------------------------

def _safe_storage_name(value: str) -> str:
    value = str(value or "asset").strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-._")
    return value or "asset"


def _extract_openai_image_payload(response: Any) -> str:
    """
    Extract image payload from OpenAI image response.

    Supports:
    - response.data[0].b64_json
    - response.data[0].url
    - response.output image_generation_call result
    """
    data = getattr(response, "data", None)
    if not data and isinstance(response, dict):
        data = response.get("data")

    if data:
        first = data[0]
        b64 = getattr(first, "b64_json", None) or (
            first.get("b64_json") if isinstance(first, dict) else None
        )
        url = getattr(first, "url", None) or (
            first.get("url") if isinstance(first, dict) else None
        )

        if b64:
            return f"data:image/png;base64,{b64}"

        if url:
            return str(url)

    output = getattr(response, "output", None)
    if not output and isinstance(response, dict):
        output = response.get("output")

    if output:
        for item in output:
            typ = getattr(item, "type", None) or (
                item.get("type") if isinstance(item, dict) else ""
            )
            result = getattr(item, "result", None) or (
                item.get("result") if isinstance(item, dict) else None
            )

            if typ == "image_generation_call" and result:
                return f"data:image/png;base64,{result}"

    raise RuntimeError(
        "OpenAI image generation returned no b64_json, URL, or image_generation_call result."
    )


def _decode_image_payload(image_payload: str) -> Tuple[bytes, str, str]:
    """
    Accepts:
    - data:image/png;base64,...
    - raw base64
    - http/https image URL

    Returns:
    - image_bytes
    - content_type
    - extension
    """
    import mimetypes
    import requests

    if not image_payload:
        raise ValueError("No image payload received")

    payload = str(image_payload).strip()

    if payload.startswith("http://") or payload.startswith("https://"):
        res = requests.get(payload, timeout=120)
        res.raise_for_status()

        content_type = (
            res.headers.get("content-type") or "image/png"
        ).split(";", 1)[0].strip()

        if not content_type.startswith("image/"):
            raise ValueError(
                f"Image URL did not return an image. content-type={content_type}"
            )

        ext = mimetypes.guess_extension(content_type) or ".png"
        return res.content, content_type, ext

    content_type = "image/png"
    b64 = payload

    match = re.match(
        r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$",
        payload,
        flags=re.S,
    )

    if match:
        content_type = match.group(1)
        b64 = match.group(2)

    b64 = re.sub(r"\s+", "", b64)

    image_bytes = base64.b64decode(b64, validate=False)

    if len(image_bytes) < 1000:
        raise ValueError(
            "Decoded image payload is too small; refusing to save broken/placeholder image."
        )

    ext = mimetypes.guess_extension(content_type) or ".png"
    return image_bytes, content_type, ext


def _upload_image_bytes_to_supabase(
    image_bytes: bytes,
    storage_path: str,
    content_type: str = "image/png",
) -> str:
    """
    Upload generated image bytes to Supabase Storage and return a public URL.
    """
    import requests

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set on Render."
        )

    bucket = SUPABASE_STORAGE_BUCKET or "briefcraft-assets"
    storage_path = storage_path.lstrip("/")

    upload_url = (
        f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/"
        f"{bucket}/{storage_path}"
    )

    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": content_type,
        "Cache-Control": "31536000",
        "x-upsert": "true",
    }

    res = requests.post(
        upload_url,
        headers=headers,
        data=image_bytes,
        timeout=180,
    )

    if res.status_code not in (200, 201):
        raise RuntimeError(
            f"Supabase Storage upload failed {res.status_code}: {res.text[:500]}"
        )

    return (
        f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/"
        f"{bucket}/{storage_path}"
    )


def generate_image_data_url(
    prompt: str,
    size: str = "1536x1024",
    quality: str = "high",
) -> str:
    """
    Generate one real image.

    No fake placeholder.
    No DALL-E fallback unless IMAGE_MODEL is intentionally set to DALL-E.
    If OpenAI fails, return the real error to Render logs.
    """
    if not _openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not configured")

    requested_model = (
        os.getenv("OPENAI_IMAGE_MODEL")
        or os.getenv("IMAGE_MODEL")
        or IMAGE_MODEL
        or "gpt-image-1"
    ).strip()

    requested_size = (
        size
        or os.getenv("VISUAL_MASTER_SIZE")
        or os.getenv("VISUAL_PREVIEW_SIZE")
        or "1536x1024"
    ).strip()

    requested_quality = (
        quality
        or os.getenv("IMAGE_QUALITY")
        or "high"
    ).strip()

    try:
        if requested_model.startswith("gpt-image"):
            safe_size = requested_size if requested_size in {
                "1024x1024",
                "1536x1024",
                "1024x1536",
                "auto",
            } else "1536x1024"

            safe_quality = requested_quality if requested_quality in {
                "low",
                "medium",
                "high",
                "auto",
            } else "high"

            response = _openai_client.images.generate(
                model=requested_model,
                prompt=prompt,
                size=safe_size,
                quality=safe_quality,
                output_format="png",
                n=1,
            )

        else:
            safe_size = requested_size if requested_size in {
                "1024x1024",
                "1792x1024",
                "1024x1792",
            } else "1024x1024"

            safe_quality = "hd" if requested_quality in {"high", "hd"} else "standard"

            response = _openai_client.images.generate(
                model=requested_model,
                prompt=prompt,
                size=safe_size,
                quality=safe_quality,
                response_format="b64_json",
                n=1,
            )

        return _extract_openai_image_payload(response)

    except Exception as exc:
        print("OpenAI image generation failed full error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI image generation failed: {repr(exc)}",
        )


def persist_data_url_image(
    data_url: str,
    target_dir: Path,
    prefix: str,
) -> Tuple[str, str]:
    """
    Drop-in replacement for old local /renders saver.

    Existing callers can still call:
        persist_data_url_image(data_url, RENDER_OUTPUT_DIR, prefix)

    But this now uploads to Supabase Storage and returns:
        storage_path, public_url
    """
    import time

    image_bytes, content_type, ext = _decode_image_payload(data_url)

    ext = ext if ext.startswith(".") else f".{ext}"

    clean_prefix = _safe_storage_name(prefix or "generated-image")

    storage_path = (
        f"projects/shared/{clean_prefix}/"
        f"{int(time.time())}-{uuid.uuid4().hex[:10]}{ext}"
    )

    public_url = _upload_image_bytes_to_supabase(
        image_bytes=image_bytes,
        storage_path=storage_path,
        content_type=content_type,
    )

    return storage_path, public_url


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
    try:
        from html import escape
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer
    except Exception:
        raise HTTPException(status_code=500, detail="reportlab is not installed. Add reportlab to requirements.txt")

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
    if "concert" in lower:
        return "concert"
    return "generic"


def analyze_brief(brief: str, event_type: Optional[str]) -> Dict[str, Any]:
    inferred = infer_event_type(brief, event_type)
    fallback = {
        "summary": brief[:300],
        "event_type": inferred,
        "objectives": [
            "Translate brief into concept",
            "Prepare department outputs",
            "Estimate production scope",
        ],
        "audience": "Stakeholders, clients, production team, audience",
        "risks": ["Venue, timelines, or technical constraints may be incomplete"],
        "assumptions": ["Planning-level outputs only until venue details are confirmed"],
    }
    prompt = (
        "Analyze this event brief and return JSON with keys "
        "summary, event_type, objectives, audience, risks, assumptions.\n\n"
        f"Brief:\n{brief}"
    )
    return {
        **fallback,
        **llm_json(
            "You are a senior experiential event strategist. Return JSON only.",
            prompt,
            fallback,
        ),
    }


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
            "estimated_budget_inr": {
                "low": int(low * 1.15),
                "medium": int(mid * 1.15),
                "high": int(high * 1.15),
            },
        },
        {
            "name": "Elegant Minimal Luxe",
            "summary": f"Refined editorial concept for {inferred}.",
            "style": "clean editorial",
            "colors": ["ivory", "champagne", "graphite"],
            "materials": ["wood veneer", "soft fabric", "textured flats"],
            "experience": "refined storytelling",
            "key_zones": ["arrival", "stage", "screen", "audience"],
            "estimated_budget_inr": {
                "low": int(low * 1.35),
                "medium": int(mid * 1.35),
                "high": int(high * 1.35),
            },
        },
    ]

    prompt = (
        "Generate exactly 3 event concepts as JSON under a key named concepts. "
        "Each concept should include name, summary, style, colors, materials, experience, key_zones.\n\n"
        f"Brief:\n{brief}"
    )
    data = llm_json(
        "You are a senior live-experience creative director. Return JSON only.",
        prompt,
        {"concepts": fallback},
    )
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
            {
                "heading": "Sound Overview",
                "body": f"Planning-level sound design for {project.get('project_name', 'event')}.",
            },
            {"heading": "Input List", "body": "MC, playback, guest, ambient and redundancy."},
        ],
    }


def default_lighting_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fixture_list": ["Moving Heads", "Wash Fixtures", "LED Battens", "Audience Blinders", "Pinspots"],
        "scene_cues": ["house-to-half", "opening reveal", "speaker special", "finale"],
        "pdf_sections": [
            {
                "heading": "Lighting Overview",
                "body": f"Concept-driven lighting plan for {project.get('project_name', 'event')}.",
            },
            {"heading": "Cue Intent", "body": "Opening, transitions, and finale lighting beats."},
        ],
    }


def default_showrunner_plan(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "running_order": ["Standby", "House to half", "Opening AV", "MC welcome", "Finale"],
        "pdf_sections": [
            {
                "heading": "Show Running",
                "body": f"Cue-based show running script for {project.get('project_name', 'event')}.",
            }
        ],
        "console_cues": [
            {
                "cue_no": 1,
                "name": "Standby",
                "cue_type": "standby",
                "standby": "All standby",
                "go": "Standby acknowledged",
                "actions": [],
            },
            {
                "cue_no": 2,
                "name": "House to Half",
                "cue_type": "lighting",
                "standby": "Lights standby",
                "go": "Go half",
                "actions": [{"protocol": "lighting", "target": "house_lights", "value": "half"}],
            },
            {
                "cue_no": 3,
                "name": "Opening AV",
                "cue_type": "av",
                "standby": "AV standby",
                "go": "Go opener",
                "actions": [{"protocol": "av", "target": "screen", "value": "play_opener"}],
            },
            {
                "cue_no": 4,
                "name": "MC Welcome",
                "cue_type": "sound",
                "standby": "Mic standby",
                "go": "Go MC",
                "actions": [{"protocol": "sound", "target": "mc_mic", "value": "on"}],
            },
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

def asset_row_to_dict(row: Any) -> Dict[str, Any]:
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
        "meta": load_json(item.get("meta"), {}),
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


def add_project_activity(
    project_id: str,
    user_id: str,
    event_type: str,
    title: str,
    detail: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
            "section": normalize_section(section),
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


def list_project_assets(
    project_id: str,
    user_id: str,
    section: Optional[str] = None,
) -> List[Dict[str, Any]]:
    filters: Dict[str, Any] = {"project_id": project_id, "user_id": user_id}
    if section:
        filters["section"] = normalize_section(section)

    rows = db_list(
        "project_assets",
        limit=500,
        order_key="created_at",
        desc=True,
        **filters,
    )
    return [asset_row_to_dict(row) for row in rows]


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
    add_project_activity(
        project_id,
        user_id,
        "job.queued",
        title,
        detail=f"{agent_type} job queued",
        meta={"job_id": job["id"], "job_type": job_type},
    )
    return job_row_to_dict(job)


def list_agent_jobs(project_id: str, user_id: str) -> List[Dict[str, Any]]:
    return [
        job_row_to_dict(row)
        for row in db_list("agent_jobs", project_id=project_id, user_id=user_id, limit=500)
    ]


def update_project_media_rollups(project_id: str, user_id: str) -> Dict[str, Any]:
    assets = list_project_assets(project_id, user_id)
    rollup = {
        "assets_total": len(assets),
        "moodboards": len([a for a in assets if a.get("asset_type") == "moodboard"]),
        "renders": len([a for a in assets if a.get("asset_type") in {"3d_render", "scene_preview"}]),
        "graphics_2d": len([a for a in assets if a.get("asset_type") == "2d_graphic"]),
        "cad": len([a for a in assets if a.get("asset_type") == "cad_layout"]),
        "references": len([a for a in assets if a.get("asset_type") == "reference"]),
    }
    return db_update("projects", project_id, {"media_rollup": rollup})


def build_concept_visual_prompts(
    project: Dict[str, Any],
    concept: Dict[str, Any],
    count: int = 3,
) -> List[str]:
    project_name = project.get("project_name") or project.get("name") or "Premium Event"
    brief = project.get("brief_text") or ""
    concept_name = concept.get("name") or "Selected Concept"
    concept_summary = (
        concept.get("summary")
        or concept.get("oneLiner")
        or "Premium visual direction"
    )
    style = concept.get("style") or "premium futuristic cinematic"
    colors = ", ".join(
        concept.get("colors")
        or ["black", "deep blue", "gold", "warm white", "chrome"]
    )

    base = f"""
Create a real image-led professional event mood board collage in 16:9 landscape format.

Project:
{project_name}

Selected concept:
{concept_name}

Concept summary:
{concept_summary}

Brief context:
{brief[:900]}

Style:
{style}

Color palette:
{colors}

The final image must be dominated by photoreal event visuals, not paragraphs or prompt text.

Create a premium AI / luxury event design reference sheet with thin gold dividers and many real-looking visual panels.

Include visual panels for:
- cinematic keynote stage with large LED wall and audience
- futuristic immersive entry tunnel
- luxury registration / welcome desk
- networking lounge
- demo zone / interactive AI screens
- photo opportunity zone
- materials and texture references: black marble, brushed gold, glass, chrome, LED pixels
- lighting mood references: blue beams, warm gold accents, haze, reflections
- experience touchpoints and guest journey

Visual style:
black, deep blue, gold, warm white, chrome, glass, cinematic reflections, premium technology, luxurious event production.

IMPORTANT NEGATIVE INSTRUCTIONS:
Do not create a black text card.
Do not write the prompt into the image.
Do not create long paragraphs inside the image.
Do not create empty gradient panels.
Do not create a UI screen.
Do not create a simple poster.
Do not make the image mostly typography.
The final output must look like a finished visual mood board image with real mood references.
""".strip()

    variations = [
        base + "\n\nComposition: full mood board reference sheet, multiple panels, color palette, visual language, materials, lighting mood, experience touchpoints.",
        base + "\n\nComposition: focus on guest journey, entry tunnel, registration, demo zone, lounge, photo-op, with material and lighting strips.",
        base + "\n\nComposition: focus on hero keynote stage, LED wall, audience reveal, premium show lighting, cinematic brand world, with supporting reference tiles.",
        base + "\n\nComposition: refined luxury-tech board with large hero stage image, immersive tunnel image, lounge image, and small swatches for materials and lighting.",
        base + "\n\nComposition: production-ready visual direction board for designers and 3D render artists, photo-rich and high-end.",
        base + "\n\nComposition: editorial presentation board, very premium, deeply cinematic, dramatic lighting and elegant spacing.",
    ]

    return variations[: max(1, min(count, len(variations)))]

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
    image_payload = generate_image_data_url(
        prompt=prompt,
        size=size,
        quality=quality,
    )

    if not image_payload:
        raise HTTPException(
            status_code=500,
            detail="Image generation failed: empty image payload",
        )

    storage_path, public_url = persist_data_url_image(
    image_payload,
    RENDER_OUTPUT_DIR,
    safe_filename(asset_type),
)

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
        meta={
            "storage_path": storage_path,
            "storage_bucket": SUPABASE_STORAGE_BUCKET,
            "size": size,
            "quality": quality,
            "image_model": (
                os.getenv("OPENAI_IMAGE_MODEL")
                or os.getenv("IMAGE_MODEL")
                or IMAGE_MODEL
            ),
        },
    )

    add_project_activity(
        str(project["id"]),
        user_id,
        "asset.generated",
        title,
        detail=f"{asset_type} generated",
        meta={
            "asset_id": asset["id"],
            "storage_path": storage_path,
            "storage_bucket": SUPABASE_STORAGE_BUCKET,
        },
    )

    return asset

def create_separate_render_view_assets(
    project: Dict[str, Any],
    user_id: str,
    concept: Dict[str, Any],
) -> List[Dict[str, Any]]:
    prompts = [
        (
            "Hero Perspective",
            f"Create a premium 16:9 hero camera view for event concept {concept.get('name')} with cinematic audience-facing perspective.",
        ),
        (
            "Wide Perspective",
            f"Create a premium 16:9 wide venue render for event concept {concept.get('name')} showing full stage and audience arrangement.",
        ),
        (
            "Top View",
            f"Create a premium 16:9 aerial top-view render for event concept {concept.get('name')} showing layout and zoning.",
        ),
        (
            "Side View",
            f"Create a premium 16:9 side perspective render for event concept {concept.get('name')} showing scenic depth and LED surfaces.",
        ),
    ]

    assets: List[Dict[str, Any]] = []
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
        g(0, "LINE")
        g(5, nxt())
        g(100, "AcDbEntity")
        g(8, layer)
        g(62, color)
        g(100, "AcDbLine")
        g(10, x1)
        g(20, y1)
        g(30, 0)
        g(11, x2)
        g(21, y2)
        g(31, 0)

    def polyline(points: List[Tuple[float, float]], layer: str = "0", color: int = 7) -> None:
        g(0, "LWPOLYLINE")
        g(5, nxt())
        g(100, "AcDbEntity")
        g(8, layer)
        g(62, color)
        g(100, "AcDbPolyline")
        g(90, len(points))
        g(70, 1)
        for x_val, y_val in points:
            g(10, x_val)
            g(20, y_val)

    def text(x_val: float, y_val: float, height: float, value: str, layer: str = "0", color: int = 7) -> None:
        g(0, "TEXT")
        g(5, nxt())
        g(100, "AcDbEntity")
        g(8, layer)
        g(62, color)
        g(100, "AcDbText")
        g(10, x_val)
        g(20, y_val)
        g(30, 0)
        g(40, height)
        g(1, value[:120])

    g(0, "SECTION")
    g(2, "HEADER")
    g(9, "$ACADVER")
    g(1, "AC1024")
    g(0, "ENDSEC")
    g(0, "SECTION")
    g(2, "ENTITIES")

    polyline(
        [(0, 0), (venue_width * scale, 0), (venue_width * scale, venue_depth * scale), (0, venue_depth * scale)],
        "VENUE",
        7,
    )

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

    g(0, "ENDSEC")
    g(0, "EOF")
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

    return {
        "filename": filename,
        "download_url": absolute_public_url(rel),
        "relative_url": rel,
        "size_bytes": len(dxf_content.encode("utf-8")),
    }


# ------------------------------------------------------------------------------
# Voice helpers
# ------------------------------------------------------------------------------

def get_project_by_id(project_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    return db_get("projects", id=project_id, user_id=user_id)


def create_voice_session(
    user_id: str,
    project_id: Optional[str],
    title: str,
    system_prompt: Optional[str],
    voice: Optional[str],
) -> Dict[str, Any]:
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


def generate_voice_reply(
    current_user: Dict[str, Any],
    session: Dict[str, Any],
    project: Optional[Dict[str, Any]],
    user_text: str,
    prior_messages: List[Dict[str, Any]],
) -> str:
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
                {
                    "role": "system",
                    "content": session.get("system_prompt") or "You are a helpful live-event creative assistant.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Project context:\n{project_context}\n\n"
                        f"Conversation so far:\n{chr(10).join(convo_lines)}\n\n"
                        f"Latest user message:\n{user_text}"
                    ),
                },
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


class Generate2DCompatInput(BaseModel):
    project_id: str
    concept_id: Optional[str] = None
    concept_index: Optional[int] = 0
    prompt: Optional[str] = None
    format: Optional[str] = "poster"
    size: Optional[str] = "1536x1024"
    count: int = Field(default=1, ge=1, le=6)
    feedback: Optional[str] = None


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

class BriefIntakeInput(BaseModel):
    brief_text: Optional[str] = None
    apply_all_suggestions: bool = False
    selected_suggestion_ids: Optional[List[str]] = None
    user_notes: Optional[str] = None

class BriefConfirmInput(BaseModel):
    approved_brief: Dict[str, Any]
    user_note: Optional[str] = None
    start_concepts: bool = True

class WorkflowNextInput(BaseModel):
    action: Optional[str] = None
    auto_generate_moodboard: bool = False
    auto_generate_2d: bool = False
    auto_generate_3d: bool = False
    auto_generate_cad: bool = False
    auto_build_departments: bool = False


# ------------------------------------------------------------------------------
# Error handler
# ------------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ------------------------------------------------------------------------------
# Root / auth routes
# ------------------------------------------------------------------------------

@app.get("/")
def root():
    supabase_ok = globals().get("_sb") is not None
    openai_ok = globals().get("_openai_client") is not None

    return {
        "message": f"{APP_NAME} running",
        "version": APP_VERSION,
        "time": now_iso(),
        "docs": "/docs",
        "openai": openai_ok,
        "supabase": supabase_ok,
    }


@app.get("/health")
def health():
    supabase_ok = globals().get("_sb") is not None
    openai_ok = globals().get("_openai_client") is not None

    return {
        "ok": True,
        "service": "BriefCraft-AI backend",
        "openai": openai_ok,
        "supabase": supabase_ok,
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

    stored_password = user.get("password") or user.get("password_hash") or ""
    if not verify_password(payload.password, stored_password):
        raise HTTPException(status_code=400, detail="Wrong password")

    token = create_access_token(str(user["id"]))
    safe_user = {k: v for k, v in user.items() if k not in {"password", "password_hash"}}

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
    return {"user": {k: v for k, v in user.items() if k not in {"password", "password_hash"}}}


# ------------------------------------------------------------------------------
# Project routes
# ------------------------------------------------------------------------------

@app.post("/projects/{project_id}/brief/intake")
def brief_intake_endpoint(
    project_id: str,
    payload: BriefIntakeInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return brief_intake_agent(
        project=project,
        user_id=user_id,
        raw_brief=payload.brief_text,
        uploaded_sources=[],
        apply_all_suggestions=payload.apply_all_suggestions,
        selected_suggestion_ids=payload.selected_suggestion_ids,
        user_notes=payload.user_notes,
    )


@app.post("/projects/{project_id}/brief/intake-with-files")
async def brief_intake_with_files_endpoint(
    project_id: str,
    brief_text: Optional[str] = Form(default=None),
    apply_all_suggestions: bool = Form(default=False),
    user_notes: Optional[str] = Form(default=None),
    files: List[UploadFile] = File(default=[]),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    uploaded_sources: List[Dict[str, Any]] = []

    for file in files:
        content = await file.read()
        if not content:
            continue

        suffix = Path(file.filename or "upload.bin").suffix or ".bin"
        saved_path = UPLOAD_DIR / f"brief_source_{uuid.uuid4().hex}{suffix}"
        saved_path.write_bytes(content)

        extracted = extract_text_from_uploaded_bytes(file.filename or saved_path.name, file.content_type, content)
        rel = relative_public_url(saved_path)
        public_url = absolute_public_url(rel)

        asset = create_project_asset(
            project_id,
            user_id,
            asset_type="reference",
            title=file.filename or "Brief Source",
            prompt=extracted.get("text", "")[:1000],
            section="references",
            job_kind="brief_source_upload",
            status="completed",
            preview_url=public_url,
            master_url=public_url,
            print_url=public_url,
            source_file_url=public_url,
            meta={
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(content),
                "extraction_notes": extracted.get("notes", []),
                "extracted_text_preview": extracted.get("text", "")[:2000],
            },
        )

        uploaded_sources.append(
            {
                "asset_id": asset.get("id"),
                "filename": file.filename,
                "content_type": file.content_type,
                "source_file_url": public_url,
                "extracted_text": extracted.get("text", "")[:12000],
                "notes": extracted.get("notes", []),
            }
        )

    update_project_media_rollups(project_id, user_id)

    return brief_intake_agent(
        project=project,
        user_id=user_id,
        raw_brief=brief_text,
        uploaded_sources=uploaded_sources,
        apply_all_suggestions=apply_all_suggestions,
        selected_suggestion_ids=[],
        user_notes=user_notes,
    )


@app.post("/projects/{project_id}/brief/confirm")
def confirm_brief_endpoint(
    project_id: str,
    payload: BriefConfirmInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = str(current_user["id"])
    confirmed = confirm_structured_brief(
        project_id,
        user_id,
        payload.approved_brief,
        payload.user_note,
    )

    if payload.start_concepts:
        project = confirmed["project"]
        run_result = _run_logic(
            project,
            project.get("brief_text") or "",
            project.get("event_type"),
            user_id,
        )
        confirmed["concept_generation"] = run_result
        confirmed["next_step"] = "User should select a final concept"

    return confirmed


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
        add_project_activity(
            str(project["id"]),
            user_id,
            "concepts.generated",
            "Concepts generated",
            detail="Three creative concepts prepared",
        )

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
def run_project(
    project_id: str,
    payload: RunProjectInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = str(current_user["id"])

    project = get_project_by_id(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    text = (
        payload.text
        or getattr(payload, "brief", None)
        or project.get("brief_text")
        or project.get("brief")
        or ""
    ).strip()

    if not text:
        raise HTTPException(status_code=422, detail="text required")

    if project.get("status") == "brief_needs_confirmation" and not brief_ready_for_concepts(project):
        approved_brief = (
            project.get("approved_brief")
            or project.get("structured_brief")
            or project.get("brief_intake")
            or project.get("analysis")
            or {
                "summary": text,
                "event_type": payload.event_type
                or project.get("event_type")
                or project.get("campaign_type")
                or "event",
                "style_direction": project.get("style_direction")
                or project.get("style_theme")
                or "premium creative",
                "ready_for_concepts": True,
            }
        )

        try:
            project = db_update(
                "projects",
                project_id,
                {
                    "approved_brief": approved_brief,
                    "status": "brief_confirmed",
                    "current_stage": "brief_confirmed",
                },
            ) or project
        except Exception as e:
            print("Auto brief confirm failed, continuing run:", e)
            project["approved_brief"] = approved_brief
            project["status"] = "brief_confirmed"
            project["current_stage"] = "brief_confirmed"

    return _run_logic(
        project,
        text,
        payload.event_type or project.get("event_type"),
        user_id,
    )
# =============================================================================
# Compatibility route: frontend expects POST /projects
# =============================================================================

@app.get("/projects")
def list_projects_plural(current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])

    rows = db_list(
        "projects",
        limit=200,
        order_key="created_at",
        desc=True,
        user_id=user_id,
    )

    return {
        "projects": rows,
        "count": len(rows),
    }


@app.post("/projects")
async def create_project_plural(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = str(current_user["id"])

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    brief = (
        payload.get("brief")
        or payload.get("brief_text")
        or payload.get("text")
        or ""
    ).strip()

    if not brief:
        raise HTTPException(status_code=422, detail="brief required")

    title = (
        payload.get("title")
        or payload.get("name")
        or payload.get("project_name")
        or brief[:80]
        or "New Creative Project"
    ).strip()

    event_type = (
        payload.get("event_type")
        or payload.get("campaign_type")
        or "event"
    )

    style_direction = (
        payload.get("style_direction")
        or payload.get("style_theme")
        or "premium creative"
    )

    row = db_insert(
        "projects",
        {
            "user_id": user_id,
            "project_name": title,
            "brief_text": brief,
            "campaign_type": event_type,
            "status": "draft",
            "style_theme": style_direction,
        },
    )

    project_id = row.get("id")

    return {
        **row,
        "id": project_id,
        "project_id": project_id,
        "title": title,
        "brief": brief,
        "brief_text": brief,
        "event_type": event_type,
        "style_direction": style_direction,
        "message": "Project created",
    }


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
    updated = db_update(
        "projects",
        payload.project_id,
        {"selected_concept": selected, "status": "concept_selected"},
    )

    add_project_activity(
        payload.project_id,
        str(current_user["id"]),
        "concept.selected",
        selected.get("name", "Concept"),
        detail="Concept selected",
    )

    return {"message": "Concept selected", "index": payload.index, "selected": selected, "project": updated}


@app.post("/projects/{project_id}/select-concept")
def select_concept_compat(project_id: str, payload: SelectCompatInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    idx = payload.concept_index if payload.concept_index is not None else payload.index
    if idx is None:
        raise HTTPException(status_code=422, detail="concept_index required")
    return select_concept(SelectConceptInput(project_id=project_id, index=idx), current_user)


# ------------------------------------------------------------------------------
# Brief Intake Agent
# ------------------------------------------------------------------------------

def extract_text_from_uploaded_bytes(filename: str, content_type: Optional[str], data: bytes) -> Dict[str, Any]:
    filename = filename or "upload"
    content_type = content_type or ""

    result = {
        "filename": filename,
        "content_type": content_type,
        "text": "",
        "notes": [],
    }

    suffix = Path(filename).suffix.lower()

    try:
        if suffix in {".txt", ".md", ".csv", ".json"} or content_type.startswith("text/"):
            result["text"] = data.decode("utf-8", errors="ignore")[:12000]
            return result

        if suffix == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(io.BytesIO(data))
                pages = []
                for page in reader.pages[:20]:
                    pages.append(page.extract_text() or "")
                result["text"] = "\n".join(pages)[:20000]
                return result
            except Exception as e:
                result["notes"].append(f"PDF text extraction unavailable: {repr(e)}")
                return result

        if suffix in {".docx"}:
            try:
                import docx

                document = docx.Document(io.BytesIO(data))
                result["text"] = "\n".join([p.text for p in document.paragraphs])[:20000]
                return result
            except Exception as e:
                result["notes"].append(f"DOCX text extraction unavailable: {repr(e)}")
                return result

        if content_type.startswith("image/") or suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            result["notes"].append("Image uploaded. Stored as source reference. Visual extraction can be added later.")
            return result

        result["notes"].append("Unsupported file type for text extraction. Stored as source reference.")
        return result

    except Exception as e:
        result["notes"].append(f"Extraction failed: {repr(e)}")
        return result


def project_source_context(project: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    project_id = str(project["id"])
    assets = list_project_assets(project_id, user_id)

    references = [
        {
            "title": a.get("title"),
            "asset_type": a.get("asset_type"),
            "section": a.get("section"),
            "prompt": a.get("prompt"),
            "meta": a.get("meta"),
            "preview_url": a.get("preview_url"),
            "source_file_url": a.get("source_file_url"),
        }
        for a in assets
        if a.get("asset_type") in {"reference", "moodboard", "2d_graphic", "3d_render"}
    ]

    return {
        "project_id": project_id,
        "project_name": project.get("project_name"),
        "existing_brief": project.get("brief_text"),
        "event_type": project.get("event_type"),
        "style_direction": project.get("style_direction"),
        "style_theme": project.get("style_theme"),
        "references": references[:30],
    }


def fallback_structured_brief(project: Dict[str, Any], raw_brief: str, uploaded_sources: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    event_type = infer_event_type(raw_brief or project.get("brief_text") or "", project.get("event_type"))

    return {
        "brief_title": project.get("project_name") or "Untitled Event Brief",
        "clean_brief": raw_brief or project.get("brief_text") or "",
        "event_type": event_type,
        "objective": "Create a premium event experience based on the provided user brief.",
        "audience": "Guests, stakeholders, brand team, and production team.",
        "brand_direction": project.get("style_direction") or project.get("style_theme") or "premium contemporary",
        "tone_and_style": ["premium", "cinematic", "clear", "production-ready"],
        "key_requirements": [
            "Create strong creative concepts",
            "Prepare moodboard and visual assets",
            "Prepare 2D graphics and 3D renders",
            "Prepare CAD layout and department manuals",
            "Prepare show running flow",
        ],
        "known_details": {
            "project_name": project.get("project_name"),
            "event_type": event_type,
            "style_direction": project.get("style_direction"),
        },
        "missing_information": [
            {
                "id": "venue",
                "question": "What is the venue name, city, indoor/outdoor condition, and approximate dimensions?",
                "importance": "high",
                "default_assumption": "Indoor premium event venue with stage, audience, and LED screen.",
            },
            {
                "id": "audience_count",
                "question": "What is expected guest/audience count?",
                "importance": "high",
                "default_assumption": "Planning for 300–500 guests.",
            },
            {
                "id": "budget",
                "question": "What is the approximate production budget range?",
                "importance": "medium",
                "default_assumption": "Premium but cost-conscious production budget.",
            },
            {
                "id": "brand_assets",
                "question": "Do you have logo, brand guideline, color palette, or reference images?",
                "importance": "high",
                "default_assumption": "Use elegant futuristic AI brand styling until brand assets are uploaded.",
            },
            {
                "id": "timeline",
                "question": "What is the event date and production timeline?",
                "importance": "medium",
                "default_assumption": "Standard pre-production timeline with staged approvals.",
            },
        ],
        "suggested_improvements": [
            {
                "id": "add_objective",
                "title": "Add clear event objective",
                "reason": "Concepts become stronger when the purpose is clear.",
                "suggested_text": "The event should launch the AI creative studio as a premium, futuristic, reliable creative technology brand.",
            },
            {
                "id": "add_audience",
                "title": "Define audience profile",
                "reason": "Audience profile affects entry, seating, show flow, and visual tone.",
                "suggested_text": "Primary audience includes brand partners, investors, enterprise clients, creators, and media guests.",
            },
            {
                "id": "add_experience_flow",
                "title": "Add guest journey",
                "reason": "A strong event needs arrival, reveal, engagement, and finale moments.",
                "suggested_text": "Guest journey should include premium entry, immersive brand tunnel, cinematic reveal, AI demo zone, networking lounge, and finale moment.",
            },
        ],
        "recommended_next_questions": [
            "Do you want a luxury cinematic direction or a futuristic technology direction?",
            "Should the event feel more premium, emotional, corporate, or experimental?",
            "Should the design prioritize stage impact, brand storytelling, or guest journey?",
        ],
        "sources_used": uploaded_sources or [],
        "approval_status": "needs_user_confirmation",
    }


def brief_intake_agent(
    project: Dict[str, Any],
    user_id: str,
    raw_brief: Optional[str],
    uploaded_sources: Optional[List[Dict[str, Any]]] = None,
    apply_all_suggestions: bool = False,
    selected_suggestion_ids: Optional[List[str]] = None,
    user_notes: Optional[str] = None,
) -> Dict[str, Any]:
    raw_brief = (raw_brief or project.get("brief_text") or "").strip()
    if not raw_brief and not uploaded_sources:
        raise HTTPException(status_code=422, detail="brief_text or upload source required")

    source_context = project_source_context(project, user_id)
    source_context["uploaded_sources"] = uploaded_sources or []

    fallback = fallback_structured_brief(project, raw_brief, uploaded_sources)

    system = """
You are BriefCraft-AI's Brief Intake Agent.

Your job:
1. Convert messy user text and upload context into a complete professional event brief.
2. Keep user intent intact.
3. Extract structured project information.
4. Detect missing information.
5. If missing, create practical assumptions.
6. Suggest brief improvements.
7. Give options the user can approve.
8. Do not start concept generation. Only prepare the brief for user confirmation.

Return JSON only with:
brief_title,
clean_brief,
event_type,
objective,
audience,
brand_direction,
tone_and_style,
key_requirements,
known_details,
missing_information,
suggested_improvements,
recommended_next_questions,
sources_used,
ready_for_concepts,
approval_status.
"""

    prompt = f"""
PROJECT CONTEXT:
{dump_json(source_context)}

RAW USER BRIEF:
{raw_brief}

USER NOTES:
{user_notes or ""}

APPLY ALL SUGGESTIONS:
{apply_all_suggestions}

SELECTED SUGGESTION IDS:
{selected_suggestion_ids or []}

Create a polished, structured, complete event brief.
If information is missing, use clear assumptions but mark them as assumptions.
Do not invent exact venue, budget, date, brand logo, or legal facts unless provided.
"""

    structured = llm_json(system, prompt, fallback)

    if not isinstance(structured, dict):
        structured = fallback

    structured.setdefault("approval_status", "needs_user_confirmation")
    structured.setdefault("ready_for_concepts", False)
    structured.setdefault("sources_used", uploaded_sources or [])

    job = db_insert(
        "agent_jobs",
        {
            "project_id": str(project["id"]),
            "user_id": user_id,
            "agent_type": "brief_intake_agent",
            "job_type": "brief_structuring",
            "title": "Brief Intake Agent",
            "status": "completed",
            "input_data": {
                "raw_brief": raw_brief,
                "uploaded_sources": uploaded_sources or [],
                "apply_all_suggestions": apply_all_suggestions,
                "selected_suggestion_ids": selected_suggestion_ids or [],
                "user_notes": user_notes,
            },
            "output_data": structured,
        },
    )

    updated_analysis = project.get("analysis") or {}
    if not isinstance(updated_analysis, dict):
        updated_analysis = load_json(updated_analysis, {}) or {}

    updated_analysis["brief_intake_agent"] = structured

    updated = db_update(
        "projects",
        str(project["id"]),
        {
            "brief_text": structured.get("clean_brief") or raw_brief,
            "analysis": updated_analysis,
            "status": "brief_needs_confirmation",
        },
    )

    add_project_activity(
        str(project["id"]),
        user_id,
        "brief.intake.completed",
        "Brief Intake Agent completed",
        detail="Structured brief prepared and waiting for user confirmation",
        meta={"job_id": job["id"], "approval_status": structured.get("approval_status")},
    )

    return {
        "message": "Structured brief prepared",
        "project": updated,
        "structured_brief": structured,
        "job": job_row_to_dict(job),
        "requires_confirmation": True,
        "next_step": "User should review, edit, and confirm the brief before concept generation.",
    }


def brief_ready_for_concepts(project: Dict[str, Any]) -> bool:
    analysis = project.get("analysis") or {}
    if not isinstance(analysis, dict):
        analysis = load_json(analysis, {}) or {}

    intake = analysis.get("brief_intake_agent") or {}
    return intake.get("approval_status") == "confirmed"


def confirm_structured_brief(
    project_id: str,
    user_id: str,
    approved_brief: Dict[str, Any],
    user_note: Optional[str] = None,
) -> Dict[str, Any]:
    project = get_project_by_id(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not isinstance(approved_brief, dict):
        raise HTTPException(status_code=422, detail="approved_brief must be an object")

    approved_brief["approval_status"] = "confirmed"
    approved_brief["confirmed_at"] = now_iso()
    if user_note:
        approved_brief["confirmation_note"] = user_note

    analysis = project.get("analysis") or {}
    if not isinstance(analysis, dict):
        analysis = load_json(analysis, {}) or {}

    analysis["brief_intake_agent"] = approved_brief

    clean_brief = approved_brief.get("clean_brief") or project.get("brief_text") or ""
    event_type = approved_brief.get("event_type") or project.get("event_type")

    updated = db_update(
        "projects",
        project_id,
        {
            "brief_text": clean_brief,
            "event_type": event_type,
            "analysis": analysis,
            "status": "brief_confirmed",
        },
    )

    add_project_activity(
        project_id,
        user_id,
        "brief.confirmed",
        "Brief confirmed",
        detail="User confirmed structured brief. Concept generation can start.",
    )

    return {
        "message": "Brief confirmed",
        "project": updated,
        "structured_brief": approved_brief,
        "next_step": "Generate concepts",
    }


def workflow_next_logic(project_id: str, user_id: str, payload: WorkflowNextInput) -> Dict[str, Any]:
    project = get_project_by_id(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    status = project.get("status") or "draft"

    if status in {"draft", "brief_needs_confirmation"} and not brief_ready_for_concepts(project):
        return {
            "message": "Brief confirmation required before concept generation",
            "status": status,
            "required_action": "confirm_brief",
            "endpoint": f"/projects/{project_id}/brief/confirm",
        }

    if status in {"brief_confirmed", "draft"} or not project.get("concepts"):
        result = _run_logic(project, project.get("brief_text") or "", project.get("event_type"), user_id)
        return {
            "message": "Concept generation completed",
            "stage": "concepts",
            "result": result,
            "next_step": "User should select final concept",
        }

    if project.get("concepts") and not project.get("selected_concept"):
        return {
            "message": "Concepts are ready. User must select a final concept.",
            "stage": "concept_selection",
            "concepts": project.get("concepts"),
            "endpoint": f"/projects/{project_id}/select-concept",
        }

    if project.get("selected_concept") and payload.auto_build_departments:
        departments = build_departments_logic(project_id, user_id)
        return {
            "message": "Departments generated",
            "stage": "departments",
            "departments": departments,
        }

    if project.get("selected_concept"):
        return {
            "message": "Final concept selected. Downstream generation is available.",
            "stage": "downstream_ready",
            "available_actions": [
                "Generate moodboard",
                "Generate 2D graphics",
                "Generate 3D renders",
                "Generate CAD layout",
                "Build departments",
                "Generate manuals",
                "Run show console",
            ],
        }

    return {
        "message": "No workflow action performed",
        "status": status,
    }


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
    state.update(
        {
            "sound_ready": True,
            "lighting_ready": True,
            "showrunner_ready": True,
            "console_index": 0,
            "hold": False,
        }
    )

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
def show_console_go(
    project_id: str,
    execute: bool = Query(True),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
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
def show_console_compat(
    project_id: str,
    command: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
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
    add_project_activity(
        project_id,
        str(current_user["id"]),
        "visual_policy.updated",
        "Visual policy updated",
        meta={"visual_policy": policy},
    )

    return {"message": "Visual policy updated", "project": updated, "visual_policy": policy}


@app.get("/projects/{project_id}/assets")
def list_assets_endpoint(
    project_id: str,
    section: Optional[str] = Query(default=None),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
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
def list_activity_endpoint(
    project_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
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
    public_url = absolute_public_url(rel)

    asset = create_project_asset(
        project_id,
        user_id,
        asset_type="reference",
        title=title or (file.filename or "Reference Asset"),
        prompt="",
        section=section,
        job_kind="upload",
        status="completed",
        preview_url=public_url,
        master_url=public_url,
        print_url=public_url,
        source_file_url=public_url,
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
        asset = sync_create_visual_asset(
            project,
            user_id,
            payload.asset_type,
            payload.title,
            payload.prompt,
            section=payload.section,
            job_kind=payload.job_kind,
        )
        update_project_media_rollups(project_id, user_id)
        return {"message": "Asset generated", "asset": asset}

    asset = create_project_asset(
        project_id,
        user_id,
        payload.asset_type,
        payload.title,
        payload.prompt,
        payload.section,
        payload.job_kind,
        status="queued",
        meta={"queued_only": True},
    )

    job = queue_agent_job_with_activity(
        project_id,
        user_id,
        agent_type=payload.asset_type,
        job_type=payload.job_kind or "asset_generation",
        title=payload.title,
        input_data={"asset_id": asset["id"], "prompt": payload.prompt},
    )

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
            assets.append(
                sync_create_visual_asset(
                    project,
                    user_id,
                    "moodboard",
                    title,
                    prompt,
                    section="moodboard",
                    job_kind="concept_moodboard",
                )
            )
        else:
            asset = create_project_asset(
                project_id,
                user_id,
                "moodboard",
                title,
                prompt,
                "moodboard",
                "concept_moodboard",
                status="queued",
            )
            job = queue_agent_job_with_activity(
                project_id,
                user_id,
                "moodboard",
                "concept_moodboard",
                title,
                input_data={"asset_id": asset["id"], "prompt": prompt},
            )
            queued_jobs.append(job)
            assets.append(asset)

    update_project_media_rollups(project_id, user_id)
    return {"message": "Moodboards processed", "assets": assets, "jobs": queued_jobs}


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

    assets: List[Dict[str, Any]] = []
    for i in range(1, payload.count + 1):
        prompt = (
            f"Create a premium 2D event graphic for project {project.get('project_name') or project.get('name') or 'Project'}, "
            f"concept {concept_name}. {concept.get('summary') or ''} "
            f"{'Feedback: ' + feedback if feedback else ''} "
            "Luxury event branding, typography, signage, key visual, high-end presentation style."
        )
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


@app.post("/ai/generate-2d")
def ai_generate_2d_compat(payload: Generate2DCompatInput, current_user: Dict[str, Any] = Depends(get_current_user)):
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
            meta={"asset_id": asset.get("id"), "format": payload.format, "concept_id": payload.concept_id},
        )

        return {"ok": True, "message": "2D graphic generated", "asset": asset, "assets": [asset]}

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


def generate_separated_renders_logic(project_id: str, user_id: str) -> Dict[str, Any]:
    project = get_project_by_id(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    selected = project.get("selected_concept") or {}
    if not selected:
        concepts = project.get("concepts") or []
        selected = concepts[0] if concepts else {}

    concept_name = selected.get("name") or "Event Concept"
    render_views = [
        {
            "title": f"{concept_name} Hero Perspective",
            "prompt": f"Create a premium 16:9 hero camera view for event concept {concept_name} with dramatic stage reveal, polished materials, cinematic lighting, ultra realistic 3D render.",
        },
        {
            "title": f"{concept_name} Side View",
            "prompt": f"Create a premium 16:9 side perspective render for event concept {concept_name} showing scenic depth, LED surfaces, stage layout, ultra realistic 3D render.",
        },
        {
            "title": f"{concept_name} Top View",
            "prompt": f"Create a premium 16:9 top or elevated view render for event concept {concept_name} showing stage zoning, guest layout, scenic structure, ultra realistic 3D render.",
        },
    ]

    assets: List[Dict[str, Any]] = []
    for view in render_views:
        assets.append(
            sync_create_visual_asset(
                project,
                user_id,
                "3d_render",
                view["title"],
                view["prompt"],
                section="renders",
                job_kind="separate_render_view",
            )
        )

    add_project_activity(
        project_id,
        user_id,
        "renders.generated",
        "Separate render views generated",
        detail=f"{len(assets)} separated render views created",
    )
    update_project_media_rollups(project_id, user_id)

    return {"message": "Separate render views generated", "assets": assets}


@app.post("/projects/{project_id}/renders/generate-separated")
def generate_separated_renders(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    return generate_separated_renders_logic(project_id, str(current_user["id"]))


@app.post("/projects/{project_id}/jobs/queue")
def queue_job_endpoint(project_id: str, payload: JobCreateInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id=user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    job = queue_agent_job_with_activity(
        project_id,
        user_id,
        payload.agent_type,
        payload.job_type,
        payload.title or payload.job_type,
        payload.input_data or {},
        priority=payload.priority,
    )
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
    user_id = str(current_user["id"])
    project = get_project_by_id(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    layout = create_cad_layout(project, user_id)
    asset = create_project_asset(
        project_id,
        user_id,
        asset_type="cad_layout",
        title=layout.get("title") or "CAD Layout",
        prompt="Professional event CAD layout",
        section="cad",
        job_kind="cad_generation",
        status="completed",
        meta={"layout_data": layout.get("layout_data")},
    )
    update_project_media_rollups(project_id, user_id)

    return {"message": "CAD layout generated", "layout": layout, "asset": asset}


@app.get("/projects/{project_id}/cad")
def get_cad_layout_endpoint(project_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    return {"layouts": db_list("cad_layouts", project_id=project_id, user_id=str(current_user["id"]), limit=20)}


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
        sections.append(
            {
                "heading": "Creative Concepts",
                "body": "\n\n".join(f"{c.get('name', '')}: {c.get('summary', '')}" for c in concepts),
            }
        )

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
    audio = synthesize_speech(
        reply,
        voice=payload.voice or session.get("voice") or TTS_VOICE,
        instructions=payload.voice_instructions,
        filename_prefix="assistant_reply",
    )

    assistant_message = add_voice_message(
        safe_session_id,
        "assistant",
        reply,
        transcript=reply,
        audio_url=audio["audio_url"],
        meta={"voice": audio["voice"]},
    )
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
    return {
        "message": "Transcription completed",
        "transcript": transcript,
        "input_audio_url": absolute_public_url(relative_public_url(saved_path)),
    }


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
        url = result.get("saved_url") or result.get("image_url")
        create_project_asset(
            payload.project_id,
            str(current_user["id"]),
            asset_type="image",
            title=payload.prompt[:60],
            prompt=payload.prompt,
            section=payload.section or "general",
            job_kind="direct_image_generation",
            status="completed",
            preview_url=url,
            master_url=url,
            print_url=url,
            source_file_url=url,
            meta={"size": payload.size, "quality": payload.quality},
        )

    return result


# =============================================================================
# UCD AGENT — Universal Creative Director Agent Backend Brain
# =============================================================================

from pydantic import BaseModel
from typing import Optional, Any, Dict, List
import datetime as dt
import json
import re


# ---------------------------------------------------------------------
# UCD Models
# ---------------------------------------------------------------------

class UcdThinkRequest(BaseModel):
    message: str = ""
    project_id: Optional[str] = None
    user_name: Optional[str] = None
    current_tab: Optional[str] = None
    brief: Optional[str] = None
    project_state: Optional[Dict[str, Any]] = None


class UcdDeliverableConfirmRequest(BaseModel):
    deliverables: List[Dict[str, Any]] = []
    user_note: Optional[str] = None


class UcdDepartmentStartRequest(BaseModel):
    department: str
    run_type: str = "standard"
    input_payload: Dict[str, Any] = {}


class UcdReviewSnapshotRequest(BaseModel):
    snapshot_type: str
    title: str
    content: Dict[str, Any] = {}
    html_content: Optional[str] = None
    markdown_content: Optional[str] = None


# ---------------------------------------------------------------------
# UCD DB helpers
# ---------------------------------------------------------------------

def _ucd_now_iso():
    return dt.datetime.utcnow().isoformat() + "Z"


def _ucd_user_id(current_user: Dict[str, Any]) -> str:
    uid = (
        current_user.get("id")
        or current_user.get("user_id")
        or current_user.get("sub")
        or current_user.get("uid")
    )
    if not uid:
        raise HTTPException(status_code=401, detail="User id missing from auth")
    return str(uid)


def _ucd_user_name(current_user: Dict[str, Any], fallback: Optional[str] = None) -> str:
    return (
        fallback
        or current_user.get("full_name")
        or current_user.get("name")
        or current_user.get("email")
        or "there"
    )


def _ucd_db_execute(query: str, params: tuple = (), fetch: str = "none"):
    """
    Uses existing psycopg DATABASE_URL.
    This assumes DATABASE_URL is already present in your main.py.
    """
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL missing")

    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)

            if fetch == "one":
                row = cur.fetchone()
                conn.commit()
                return row

            if fetch == "all":
                rows = cur.fetchall()
                conn.commit()
                return rows

            conn.commit()
            return None


def _ucd_safe_json(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


def _ucd_get_or_create_memory(user_id: str, user_name: str):
    row = _ucd_db_execute(
        """
        insert into public.ucd_user_memory (user_id, user_name)
        values (%s, %s)
        on conflict (user_id) do update set
          user_name = coalesce(excluded.user_name, public.ucd_user_memory.user_name),
          updated_at = now()
        returning *;
        """,
        (user_id, user_name),
        fetch="one",
    )
    return row


def _ucd_get_or_create_project_state(project_id: str, user_id: str):
    row = _ucd_db_execute(
        """
        insert into public.ucd_project_state (project_id, user_id)
        values (%s, %s)
        on conflict (project_id) do update set
          updated_at = now()
        returning *;
        """,
        (project_id, user_id),
        fetch="one",
    )
    return row


def _ucd_log_chat(
    user_id: str,
    project_id: Optional[str],
    sender: str,
    message: str,
    intent: Optional[str] = None,
    action_taken: Optional[str] = None,
    department: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    _ucd_db_execute(
        """
        insert into public.ucd_chat_messages
        (project_id, user_id, sender, message, intent, action_taken, department, metadata)
        values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb);
        """,
        (
            project_id,
            user_id,
            sender,
            message,
            intent,
            action_taken,
            department,
            _ucd_safe_json(metadata or {}),
        ),
    )


def _ucd_log_decision(
    user_id: str,
    project_id: Optional[str],
    decision_type: str,
    decision_title: str,
    decision_summary: str,
    reason: str,
    confidence: float = 0.85,
    before_state: Optional[Dict[str, Any]] = None,
    after_state: Optional[Dict[str, Any]] = None,
):
    _ucd_db_execute(
        """
        insert into public.ucd_decision_log
        (user_id, project_id, decision_type, decision_title, decision_summary, reason, confidence, before_state, after_state)
        values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb);
        """,
        (
            user_id,
            project_id,
            decision_type,
            decision_title,
            decision_summary,
            reason,
            confidence,
            _ucd_safe_json(before_state or {}),
            _ucd_safe_json(after_state or {}),
        ),
    )


# ---------------------------------------------------------------------
# UCD Brain logic
# ---------------------------------------------------------------------

def _ucd_detect_intent(message: str) -> str:
    m = (message or "").lower().strip()

    if any(x in m for x in ["next step", "what now", "continue", "guide me", "what should i do", "what's missing", "whats missing"]):
        return "advise_next_step"

    if any(x in m for x in ["deliverable", "deliverables", "what output", "what do we need"]):
        return "recommend_deliverables"

    if any(x in m for x in ["brief", "refine brief", "improve brief", "make brief"]):
        return "brief_refinement"

    if any(x in m for x in ["concept", "idea", "creative route"]):
        return "concept_direction"

    if any(x in m for x in ["moodboard", "mood board", "visual direction"]):
        return "moodboard_direction"

    if any(x in m for x in ["2d", "graphic", "key visual", "print"]):
        return "graphics_2d_direction"

    if any(x in m for x in ["3d", "render", "stage", "booth", "stall", "environment"]):
        return "render_3d_direction"

    if any(x in m for x in ["cost", "credit", "token", "price", "budget"]):
        return "credit_estimate"

    return "general_ucd_chat"


def _ucd_time_greeting(user_name: str = "there") -> str:
    hour = dt.datetime.now().hour
    if hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    if user_name and user_name not in ["there", "None"]:
        return f"{greeting}, {user_name}."
    return f"{greeting}."


def _ucd_recommend_deliverables_from_brief(brief: str) -> List[Dict[str, Any]]:
    text = (brief or "").lower()

    deliverables = [
        {
            "deliverable_key": "structured_brief",
            "deliverable_title": "Refined Structured Brief",
            "deliverable_type": "content",
            "department": "brief",
            "priority": 1,
            "is_required": True,
            "ucd_reason": "Every project should start with a clean, formal, client-ready brief.",
        },
        {
            "deliverable_key": "creative_concepts",
            "deliverable_title": "3 Creative Concept Options",
            "deliverable_type": "content",
            "department": "concept",
            "priority": 2,
            "is_required": True,
            "ucd_reason": "The user needs strong creative routes before visual production begins.",
        },
    ]

    if any(x in text for x in ["launch", "brand", "premium", "corporate", "conference", "activation", "exhibition", "mall", "product", "car", "automobile"]):
        deliverables.append({
            "deliverable_key": "moodboard",
            "deliverable_title": "Moodboard / Visual Direction",
            "deliverable_type": "visual",
            "department": "moodboard",
            "priority": 3,
            "is_required": True,
            "ucd_reason": "A premium event needs a visual language before design and render generation.",
        })

    if any(x in text for x in ["graphic", "poster", "kv", "key visual", "print", "banner", "standee", "backdrop", "social"]):
        deliverables.append({
            "deliverable_key": "graphics_2d",
            "deliverable_title": "2D Graphic Design Set",
            "deliverable_type": "design",
            "department": "graphics_2d",
            "priority": 4,
            "is_required": True,
            "ucd_reason": "The brief indicates a need for print or digital graphic assets.",
        })

    if any(x in text for x in ["3d", "stage", "stall", "booth", "exhibition", "space", "render", "venue", "walkthrough", "set"]):
        deliverables.append({
            "deliverable_key": "renders_3d",
            "deliverable_title": "3D Render Views",
            "deliverable_type": "render",
            "department": "render_3d",
            "priority": 5,
            "is_required": True,
            "ucd_reason": "Spatial/event setup work needs 3D visualization for approval.",
        })

    if any(x in text for x in ["show", "performance", "stage", "artist", "launch moment", "reveal"]):
        deliverables.append({
            "deliverable_key": "show_running_script",
            "deliverable_title": "Show Running Script",
            "deliverable_type": "show_document",
            "department": "show_running",
            "priority": 6,
            "is_required": False,
            "ucd_reason": "A live show or launch moment benefits from cue-based show running.",
        })

    return deliverables


def _ucd_next_step_message(state: Dict[str, Any], brief: str, user_name: str) -> Dict[str, Any]:
    current_stage = state.get("current_stage") if state else None
    brief_status = state.get("brief_status") if state else "draft"
    concept_status = state.get("concept_status") if state else "pending"
    moodboard_status = state.get("moodboard_status") if state else "pending"
    graphics_status = state.get("graphics_2d_status") if state else "pending"
    render_status = state.get("render_3d_status") if state else "pending"

    if not brief or len(brief.strip()) < 30:
        return {
            "message": "Let’s first make the brief stronger. Share the event type, brand/product, audience, venue, objective, preferred style, and tentative budget. Then I’ll convert it into a polished client-ready brief.",
            "next_action": "collect_brief_details",
            "department": "brief",
        }

    if brief_status in ["draft", "pending"]:
        return {
            "message": "The direction is good. Next I should refine this into a formal structured brief, fill the missing parts with smart assumptions, and show it in fullscreen for your review.",
            "next_action": "refine_brief",
            "department": "brief",
        }

    if concept_status in ["pending", "draft"]:
        return {
            "message": "Now we are ready for concepts. I’ll keep only the Brief and Concept departments active, create 3 strong concept routes, and then you can select one final route for further production.",
            "next_action": "generate_concepts",
            "department": "concept",
        }

    if moodboard_status in ["pending", "draft"]:
        return {
            "message": "The selected concept should now move into visual direction. Moodboard is the right next department, unless you want to directly create only 2D graphics or 3D renders.",
            "next_action": "generate_moodboard",
            "department": "moodboard",
        }

    if graphics_status in ["pending", "draft"]:
        return {
            "message": "Next we can move into 2D graphics. I’ll keep this department focused on clean thumbnails, fullscreen review, feedback, and download-ready outputs.",
            "next_action": "generate_2d_graphics",
            "department": "graphics_2d",
        }

    if render_status in ["pending", "draft"]:
        return {
            "message": "Next we can move into 3D renders. I’ll prepare it like a professional visualization workflow: progress state, render cards, fullscreen review, feedback, and final download.",
            "next_action": "generate_3d_renders",
            "department": "render_3d",
        }

    return {
        "message": "The project is moving well. Next I’d prepare the final presentation/export layer so the work becomes client-ready instead of staying only as separate assets.",
        "next_action": "prepare_presentation",
        "department": "presentation",
    }


def _ucd_credit_estimate(departments: List[str]) -> Dict[str, Any]:
    if not departments:
        departments = ["brief", "concept"]

    rows = _ucd_db_execute(
        """
        select department, display_name, hourly_rate_inr, credits_per_hour, average_minutes_per_output
        from public.ucd_credit_rates
        where department = any(%s)
          and is_active = true;
        """,
        (departments,),
        fetch="all",
    )

    total_credits = 0
    total_inr = 0
    items = []

    for r in rows:
        minutes = r.get("average_minutes_per_output") or 60
        credits = round((float(r.get("credits_per_hour") or 0) / 60) * minutes, 2)
        amount = round((float(r.get("hourly_rate_inr") or 0) / 60) * minutes, 2)

        total_credits += credits
        total_inr += amount

        items.append({
            "department": r["department"],
            "display_name": r["display_name"],
            "estimated_minutes": minutes,
            "estimated_credits": credits,
            "estimated_inr": amount,
        })

    return {
        "items": items,
        "total_credits": round(total_credits, 2),
        "total_inr": round(total_inr, 2),
    }


def _ucd_format_reply(intent: str, user_name: str, message: str, project_state: Dict[str, Any], brief: str):
    greeting = _ucd_time_greeting(user_name)

    if intent == "recommend_deliverables":
        deliverables = _ucd_recommend_deliverables_from_brief(brief or message)
        departments = [d["department"] for d in deliverables]
        credit = _ucd_credit_estimate(departments)
        return {
            "message": f"{greeting} I’ve mapped the required deliverables. My suggestion is to keep the workflow focused: Brief → Concept → Moodboard/Visual Direction → required production departments only. This avoids wasting credits on departments we don’t need.",
            "intent": intent,
            "action": "show_deliverables",
            "deliverables": deliverables,
            "credit_estimate": credit,
        }

    if intent == "advise_next_step":
        step = _ucd_next_step_message(project_state or {}, brief or message, user_name)
        return {
            "message": step["message"],
            "intent": intent,
            "action": step["next_action"],
            "department": step["department"],
        }

    if intent == "brief_refinement":
        return {
            "message": "Interesting direction. I’ll treat this like a premium creative brief, not just a text note. First I’ll check what is missing: objective, audience, venue, brand tone, deliverables, budget scale, and success expectation. Then I’ll refine it into a professional client-ready format.",
            "intent": intent,
            "action": "refine_brief",
            "department": "brief",
        }

    if intent == "concept_direction":
        return {
            "message": "Good, now we are entering the creative zone. For concepts, I should create 3 strong routes with proper storytelling, event logic, audience emotion, brand/product relevance, reveal moments, visual language, and practical execution notes.",
            "intent": intent,
            "action": "generate_concepts",
            "department": "concept",
        }

    if intent == "moodboard_direction":
        return {
            "message": "For moodboard, I’ll keep it structured like a visual presentation: 16:9 frames, image direction, color language, material mood, lighting feel, reference explanation, and why each visual supports the concept.",
            "intent": intent,
            "action": "generate_moodboard",
            "department": "moodboard",
        }

    if intent == "graphics_2d_direction":
        return {
            "message": "For 2D graphics, I’ll push toward premium print-ready thinking: key visual, layout direction, typography mood, brand placement, thumbnail gallery, fullscreen review, feedback loop, and export/download flow.",
            "intent": intent,
            "action": "generate_2d_graphics",
            "department": "graphics_2d",
        }

    if intent == "render_3d_direction":
        return {
            "message": "For 3D, I’ll handle it like a studio render pipeline: scene planning, camera angles, material direction, lighting mood, render progress, fullscreen preview, feedback, and final selected views.",
            "intent": intent,
            "action": "generate_3d_renders",
            "department": "render_3d",
        }

    if intent == "credit_estimate":
        departments = ["brief", "concept", "moodboard", "graphics_2d", "render_3d"]
        credit = _ucd_credit_estimate(departments)
        return {
            "message": f"Here’s the working estimate. The smart way is to activate only the departments required for this project, so the user does not burn unnecessary credits.",
            "intent": intent,
            "action": "show_credit_estimate",
            "credit_estimate": credit,
        }

    return {
        "message": "I’m with you. Tell me what you want to improve next: brief, deliverables, concepts, moodboard, 2D graphics, 3D renders, or final presentation.",
        "intent": intent,
        "action": "general_reply",
    }


# ---------------------------------------------------------------------
# UCD API Routes
# ---------------------------------------------------------------------

@app.get("/ucd/health")
def ucd_health():
    return {
        "ok": True,
        "agent": "Universal Creative Director Agent",
        "short_name": "UCD Agent",
        "status": "ready",
        "time": _ucd_now_iso(),
    }

# ---------------------------------------------------------------------
# UCD Root Compatibility Route
# Frontend calls: POST /ucd/think
# This route does NOT require auth or DATABASE_URL.
# ---------------------------------------------------------------------

def _ucd_root_pick_action(message: str, state: Dict[str, Any]) -> Dict[str, Any]:
    text = (message or "").lower().strip()

    has_brief = bool(state.get("has_brief"))
    concepts_count = int(state.get("concepts_count") or 0)
    has_final_concept = bool(state.get("has_final_concept"))
    outputs_ready = bool(state.get("outputs_ready"))

    if any(x in text for x in ["3d", "render", "renders", "show 3d", "open 3d"]):
        return {
            "action": "open_3d_renders",
            "message": "Opened 3D Renders. These are ready for review — you can view, open, download, or copy the asset links.",
        }

    if any(x in text for x in ["moodboard", "mood board", "visual board"]):
        return {
            "action": "open_moodboard",
            "message": "Opened Mood Board. This is where we shape the visual language before moving deeper into production.",
        }

    if any(x in text for x in ["concept", "concepts", "idea options", "creative options"]):
        return {
            "action": "open_concepts",
            "message": "Opened Concepts. Review the three options and mark the strongest one as final so I can guide the next departments properly.",
        }

    if any(x in text for x in ["brief", "project brief", "open brief", "start a brief"]):
        return {
            "action": "open_project_brief",
            "message": "Opened Project Brief. This is the foundation — I’ll use it to control the creative direction and department flow.",
        }

    if any(x in text for x in ["2d", "graphic", "graphics", "print design"]):
        return {
            "action": "open_2d_graphics",
            "message": "Opened 2D Graphics. This department should focus on polished, print-ready and presentation-ready visuals.",
        }

    if any(x in text for x in ["lighting", "light", "lighting cues"]):
        return {
            "action": "open_lighting",
            "message": "Opened Lighting. Once the final concept is locked, this becomes the lighting design manual and cue direction.",
        }

    if any(x in text for x in ["sound", "audio"]):
        return {
            "action": "open_sound",
            "message": "Opened Sound. This department should handle sonic mood, music direction, SFX, and show atmosphere.",
        }

    if any(x in text for x in ["show runner", "showrunner", "run of show", "ros"]):
        return {
            "action": "open_show_runner",
            "message": "Opened Show Runner. This is where we control the live execution script, cue stack, and show-flow.",
        }

    if any(x in text for x in ["all outputs", "outputs", "downloads", "download"]):
        return {
            "action": "open_all_outputs",
            "message": "Opened All Outputs. Here you can review everything generated across departments.",
        }

    if any(x in text for x in [
        "what should i do next",
        "what next",
        "next step",
        "continue",
        "guide me",
        "what is missing",
        "what's missing",
        "what now",
        "next"
    ]):
        if not has_brief:
            return {
                "action": "open_project_brief",
                "message": "First, let’s strengthen the project brief. Add the event idea, audience, venue, brand/product, objective, style direction, and budget if available.",
            }

        if has_brief and concepts_count <= 0:
            return {
                "action": "open_project_brief",
                "message": "The brief is ready enough to move forward. Next, we should generate three strong concept options based on the event objective, audience, brand tone, and deliverables.",
            }

        if concepts_count > 0 and not has_final_concept:
            return {
                "action": "open_concepts",
                "message": "Your concepts are ready. Open the Concepts tab, choose the strongest option, then mark it as Final. After that I’ll activate only the required departments.",
            }

        if has_final_concept and not outputs_ready:
            return {
                "action": "open_moodboard",
                "message": "Nice, now we are moving in the right direction. Since the final concept is selected, the next smart step is Mood Board or the specific department required by the deliverables.",
            }

        return {
            "action": "open_all_outputs",
            "message": "Production outputs are ready. Next, review each department output, give feedback where needed, then export the final package.",
        }

    if any(x in text for x in ["generate concepts", "create concepts", "make concepts"]):
        return {
            "action": "generate_concepts",
            "message": "I’ll move into concept development. The right approach is to create three strong options, then lock one final direction before activating other departments.",
        }

    if any(x in text for x in ["generate moodboard", "create moodboard", "make moodboard"]):
        return {
            "action": "generate_moodboard",
            "message": "I’ll prepare the mood board direction. It should feel like a proper visual story, not random references.",
        }

    if any(x in text for x in ["generate 3d", "create 3d", "make 3d render"]):
        return {
            "action": "generate_3d_renders",
            "message": "I’ll prepare 3D render generation with cinematic angles, spatial depth, stage layout, branding, and audience experience.",
        }

    return {
        "action": None,
        "message": "I’m ready. Tell me what you want to do next — brief, concepts, mood board, 2D graphics, 3D renders, lighting, sound, show runner, or final downloads.",
    }


@app.post("/ucd/think")
async def ucd_think_root(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    message = payload.get("message") or ""
    user_name = (payload.get("user_name") or "").strip()
    project_id = payload.get("project_id")
    user_id = payload.get("user_id")

    state = (
        payload.get("state")
        or payload.get("project_state")
        or {}
    )

    if not isinstance(state, dict):
        state = {}

    if payload.get("current_tab") and not state.get("current_tab"):
        state["current_tab"] = payload.get("current_tab")

    decision = _ucd_root_pick_action(message, state)

    reply_message = decision.get("message") or ""

    if user_name and any(x in message.lower() for x in ["start", "next", "continue", "what"]):
        reply_message = f"{user_name}, {reply_message}"

    return {
        "ok": True,
        "agent": "Universal Creative Director Agent",
        "short_name": "UCD Agent",
        "action": decision.get("action"),
        "message": reply_message,
        "payload": {
            "project_id": project_id,
            "user_id": user_id,
            "state": state,
        },
    }

@app.get("/ucd/rates")
def ucd_rates(current_user: Dict[str, Any] = Depends(get_current_user)):
    rows = _ucd_db_execute(
        """
        select department, display_name, hourly_rate_inr, credits_per_hour, average_minutes_per_output, description
        from public.ucd_credit_rates
        where is_active = true
        order by hourly_rate_inr asc;
        """,
        fetch="all",
    )
    return {"ok": True, "rates": rows}


@app.post("/projects/{project_id}/ucd/init")
def ucd_project_init(
    project_id: str,
    body: UcdThinkRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = _ucd_user_id(current_user)
    user_name = _ucd_user_name(current_user, body.user_name)

    memory = _ucd_get_or_create_memory(user_id, user_name)
    state = _ucd_get_or_create_project_state(project_id, user_id)

    _ucd_log_chat(
        user_id=user_id,
        project_id=project_id,
        sender="ucd",
        message=f"{_ucd_time_greeting(user_name)} I’m ready to guide this project like your creative director.",
        intent="init",
        action_taken="project_ucd_initialized",
    )

    return {
        "ok": True,
        "memory": memory,
        "state": state,
        "message": f"{_ucd_time_greeting(user_name)} I’m ready. Share the brief and I’ll start shaping it properly.",
    }


@app.get("/projects/{project_id}/ucd/state")
def ucd_project_state(
    project_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = _ucd_user_id(current_user)
    state = _ucd_get_or_create_project_state(project_id, user_id)

    deliverables = _ucd_db_execute(
        """
        select *
        from public.ucd_project_deliverables
        where project_id = %s and user_id = %s
        order by priority asc, created_at asc;
        """,
        (project_id, user_id),
        fetch="all",
    )

    runs = _ucd_db_execute(
        """
        select *
        from public.ucd_department_runs
        where project_id = %s and user_id = %s
        order by created_at desc
        limit 20;
        """,
        (project_id, user_id),
        fetch="all",
    )

    return {
        "ok": True,
        "state": state,
        "deliverables": deliverables,
        "department_runs": runs,
    }


@app.post("/projects/{project_id}/ucd/think")
def ucd_think(
    project_id: str,
    body: UcdThinkRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = _ucd_user_id(current_user)
    user_name = _ucd_user_name(current_user, body.user_name)

    _ucd_get_or_create_memory(user_id, user_name)
    state = _ucd_get_or_create_project_state(project_id, user_id)

    message = body.message or ""
    brief = body.brief or ""
    intent = _ucd_detect_intent(message)

    _ucd_log_chat(
        user_id=user_id,
        project_id=project_id,
        sender="user",
        message=message,
        intent=intent,
    )

    reply = _ucd_format_reply(
        intent=intent,
        user_name=user_name,
        message=message,
        project_state=state or {},
        brief=brief,
    )

    _ucd_log_chat(
        user_id=user_id,
        project_id=project_id,
        sender="ucd",
        message=reply.get("message", ""),
        intent=intent,
        action_taken=reply.get("action"),
        department=reply.get("department"),
        metadata=reply,
    )

    _ucd_log_decision(
        user_id=user_id,
        project_id=project_id,
        decision_type="ucd_think",
        decision_title=f"UCD handled intent: {intent}",
        decision_summary=reply.get("message", ""),
        reason="Intent was detected from user chat and routed to UCD decision logic.",
        after_state={"reply": reply},
    )

    return {
        "ok": True,
        "agent": "UCD Agent",
        **reply,
    }


@app.post("/projects/{project_id}/ucd/deliverables/recommend")
def ucd_recommend_deliverables(
    project_id: str,
    body: UcdThinkRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = _ucd_user_id(current_user)
    user_name = _ucd_user_name(current_user, body.user_name)

    _ucd_get_or_create_project_state(project_id, user_id)

    brief = body.brief or body.message or ""
    deliverables = _ucd_recommend_deliverables_from_brief(brief)
    departments = [d["department"] for d in deliverables]
    credit = _ucd_credit_estimate(departments)

    return {
        "ok": True,
        "message": "I’ve recommended the required deliverables and estimated the credits.",
        "deliverables": deliverables,
        "credit_estimate": credit,
    }


@app.post("/projects/{project_id}/ucd/deliverables/confirm")
def ucd_confirm_deliverables(
    project_id: str,
    body: UcdDeliverableConfirmRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = _ucd_user_id(current_user)

    for idx, d in enumerate(body.deliverables):
        _ucd_db_execute(
            """
            insert into public.ucd_project_deliverables
            (project_id, user_id, deliverable_key, deliverable_title, deliverable_type, department, priority,
             status, is_required, is_user_confirmed, ucd_reason, user_note, metadata)
            values (%s, %s, %s, %s, %s, %s, %s, 'planned', %s, true, %s, %s, %s::jsonb);
            """,
            (
                project_id,
                user_id,
                d.get("deliverable_key") or d.get("key") or f"deliverable_{idx+1}",
                d.get("deliverable_title") or d.get("title") or f"Deliverable {idx+1}",
                d.get("deliverable_type") or "creative_output",
                d.get("department") or "brief",
                d.get("priority") or idx + 1,
                bool(d.get("is_required", True)),
                d.get("ucd_reason"),
                body.user_note,
                _ucd_safe_json(d),
            ),
        )

    departments = list({d.get("department") or "brief" for d in body.deliverables})

    _ucd_db_execute(
        """
        update public.ucd_project_state
        set deliverables_status = 'confirmed',
            active_departments = %s::jsonb,
            ucd_next_action = 'start_required_department_workflow',
            updated_at = now()
        where project_id = %s and user_id = %s;
        """,
        (_ucd_safe_json(departments), project_id, user_id),
    )

    return {
        "ok": True,
        "message": "Deliverables confirmed and stored. I’ll now activate only the required departments.",
        "active_departments": departments,
    }


@app.post("/projects/{project_id}/ucd/department/start")
def ucd_department_start(
    project_id: str,
    body: UcdDepartmentStartRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = _ucd_user_id(current_user)

    credit = _ucd_credit_estimate([body.department])
    item = credit["items"][0] if credit.get("items") else {}

    row = _ucd_db_execute(
        """
        insert into public.ucd_department_runs
        (project_id, user_id, department, run_type, status, progress, progress_label, progress_effect,
         input_payload, estimated_minutes, estimated_credits, started_at)
        values (%s, %s, %s, %s, 'running', 1, %s, %s, %s::jsonb, %s, %s, now())
        returning *;
        """,
        (
            project_id,
            user_id,
            body.department,
            body.run_type,
            "Starting department workflow",
            _ucd_department_effect(body.department),
            _ucd_safe_json(body.input_payload),
            item.get("estimated_minutes", 60),
            item.get("estimated_credits", 0),
        ),
        fetch="one",
    )

    return {
        "ok": True,
        "message": f"{body.department} department started.",
        "department_run": row,
    }


def _ucd_department_effect(department: str) -> str:
    effects = {
        "brief": "typing_writer_progress",
        "concept": "story_writing_progress",
        "moodboard": "thinking_visual_board_progress",
        "graphics_2d": "sketch_to_design_progress",
        "render_3d": "vray_render_bucket_progress",
        "light_design": "lighting_cue_progress",
        "sound_design": "audio_waveform_progress",
        "show_running": "cue_sheet_progress",
        "presentation": "deck_build_progress",
        "export": "file_export_progress",
        "account": "credit_calculation_progress",
    }
    return effects.get(department, "standard_progress")


@app.post("/projects/{project_id}/ucd/review-snapshot")
def ucd_create_review_snapshot(
    project_id: str,
    body: UcdReviewSnapshotRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = _ucd_user_id(current_user)

    row = _ucd_db_execute(
        """
        insert into public.ucd_review_snapshots
        (project_id, user_id, snapshot_type, title, content, html_content, markdown_content)
        values (%s, %s, %s, %s, %s::jsonb, %s, %s)
        returning *;
        """,
        (
            project_id,
            user_id,
            body.snapshot_type,
            body.title,
            _ucd_safe_json(body.content),
            body.html_content,
            body.markdown_content,
        ),
        fetch="one",
    )

    return {
        "ok": True,
        "message": "Review snapshot saved.",
        "snapshot": row,
    }

# existing main.py code above...

# ---------------------------------------------------------------------------
# Moodboard generation API
# ---------------------------------------------------------------------------

class MoodboardGenerateRequest(BaseModel):
    project_id: str
    brief: Optional[str] = ""
    concept_id: Optional[str] = ""
    concept_title: Optional[str] = ""
    concept_summary: Optional[str] = ""
    concept: Optional[Dict[str, Any]] = None
    count: int = 4


def _moodboard_prompt_pack(payload: MoodboardGenerateRequest) -> List[Dict[str, str]]:
    concept = payload.concept or {}
    title = (
        payload.concept_title
        or concept.get("title")
        or concept.get("name")
        or "Selected Event Concept"
    )
    summary = (
        payload.concept_summary
        or concept.get("summary")
        or concept.get("description")
        or payload.brief
        or ""
    )

    base = f"""
Create a premium event moodboard image for this concept.

Concept title: {title}
Concept summary: {summary}
Brief: {payload.brief}

Style requirements:
- real visual mood board image, not text-only
- cinematic luxury event design
- premium stage, entrance, lighting, materials, atmosphere
- no written text, no labels, no logo, no watermark
- no gradient placeholder, no abstract flat card
- realistic 3D/event design photography look
- landscape composition suitable for 16:9 frontend display
""".strip()

    return [
        {
            "title": f"{title} — Spatial Atmosphere",
            "explanation": "Shows the overall venue mood, scale, lighting, and premium environment direction.",
            "prompt": base + "\nFocus on full venue atmosphere, entrance vista, guest arrival energy, wide cinematic composition.",
        },
        {
            "title": f"{title} — Stage & Focal Moment",
            "explanation": "Defines the hero stage language and the main visual moment for the concept.",
            "prompt": base + "\nFocus on stage design, LED backdrop, layered scenic elements, lighting beams, hero presentation moment.",
        },
        {
            "title": f"{title} — Materials & Decor Palette",
            "explanation": "Captures the material mood, textures, floral/decor details, and premium finish.",
            "prompt": base + "\nFocus on luxury materials, decor textures, furniture, floral accents, metallic details, ambient light.",
        },
        {
            "title": f"{title} — Guest Experience Mood",
            "explanation": "Shows how guests feel inside the experience and why the mood suits the selected concept.",
            "prompt": base + "\nFocus on immersive guest experience, lounge ambience, hospitality zones, warm premium celebration mood.",
        },
        {
            "title": f"{title} — Hero Detail Mood",
            "explanation": "Highlights premium focal details, finishes, lighting accents, and branded atmosphere.",
            "prompt": base + "\nFocus on close-up premium details, refined decor, stage materials, lighting texture, cinematic brand atmosphere.",
        },
        {
            "title": f"{title} — Production Design Direction",
            "explanation": "Shows the production-ready design language for render artists and event designers.",
            "prompt": base + "\nFocus on production design references, scenic layers, LED language, spatial composition, event visualization quality.",
        },
    ]


def _asset_image_url(row: Dict[str, Any]) -> Optional[str]:
    return (
        row.get("preview_url")
        or row.get("master_url")
        or row.get("print_url")
        or row.get("source_file_url")
        or row.get("public_url")
        or row.get("image_url")
    )


@app.post("/api/moodboard/generate")
def generate_project_moodboard(payload: MoodboardGenerateRequest):
    project_id = ensure_uuid(payload.project_id, "project_id")

    project = db_get("projects", id=project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    asset_user_id = (
        project.get("user_id")
        or project.get("owner_id")
        or project.get("created_by")
    )

    if not asset_user_id:
        raise HTTPException(
            status_code=422,
            detail="Project has no user_id/owner_id/created_by, cannot save moodboard asset.",
        )

    count = max(1, min(int(payload.count or 4), 6))
    prompt_pack = _moodboard_prompt_pack(payload)[:count]

    created_assets: List[Dict[str, Any]] = []

    for index, item in enumerate(prompt_pack, start=1):
        image_payload = generate_image_data_url(
            item["prompt"],
            size="1536x1024",
            quality="medium",
        )

        storage_path, public_url = persist_data_url_image(
            image_payload,
            RENDER_OUTPUT_DIR,
            f"moodboard-{project_id}-{index}",
        )

        asset = db_insert(
            "project_assets",
            {
                "project_id": project_id,
                "user_id": asset_user_id,
                "asset_type": "moodboard",
                "section": "moodboard",
                "job_kind": "concept_moodboard",
                "title": item["title"],
                "prompt": item["prompt"],
                "status": "completed",
                "preview_url": public_url,
                "master_url": public_url,
                "print_url": public_url,
                "source_file_url": public_url,
                "output_data": {
                    "prompt": item["prompt"],
                    "explanation": item["explanation"],
                    "storage_path": storage_path,
                    "public_url": public_url,
                    "aspect_display": "16:9",
                    "image_size": "1536x1024",
                },
                "meta": {
                    "generated_by": "moodboard_generate_api",
                    "concept_id": payload.concept_id,
                    "concept_title": payload.concept_title,
                    "index": index,
                    "storage_path": storage_path,
                    "storage_bucket": SUPABASE_STORAGE_BUCKET,
                },
            },
        )

        asset["image_url"] = _asset_image_url(asset) or public_url
        created_assets.append(asset)

    try:
        update_project_media_rollups(project_id, asset_user_id)
    except Exception as e:
        print("Moodboard media rollup update failed:", repr(e))

    return {
        "ok": True,
        "project_id": project_id,
        "section": "moodboard",
        "count": len(created_assets),
        "assets": created_assets,
    }


@app.get("/api/projects/{project_id}/moodboard")
def list_project_moodboard(project_id: str):
    project_id = ensure_uuid(project_id, "project_id")

    rows = db_list(
        "project_assets",
        limit=100,
        order_key="created_at",
        desc=True,
        project_id=project_id,
    )

    assets: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("asset_type") == "moodboard" or row.get("section") == "moodboard":
            image_url = _asset_image_url(row)
            if image_url:
                row["image_url"] = image_url
                assets.append(row)

    return {
        "ok": True,
        "project_id": project_id,
        "section": "moodboard",
        "count": len(assets),
        "assets": assets,
    }

# ==============================================================================
# CAD BACKEND V3 PATCH - Professional DXF/SVG/PDF generator
# Paste this block at the very bottom of main.py, after all existing routes.
# ==============================================================================

import math as _cad_v3_math
import mimetypes as _cad_v3_mimetypes
import time as _cad_v3_time
from typing import Any as _CadV3Any, Dict as _CadV3Dict, List as _CadV3List, Optional as _CadV3Optional, Tuple as _CadV3Tuple

from fastapi import File as _CadV3File, Form as _CadV3Form, UploadFile as _CadV3UploadFile
from pydantic import BaseModel as _CadV3BaseModel, Field as _CadV3Field


class CadV3GenerateRequest(_CadV3BaseModel):
    project_id: _CadV3Optional[str] = _CadV3Field(default="demo-cad-project")
    title: _CadV3Optional[str] = _CadV3Field(default="BriefCraft CAD Layout V3")
    brief: _CadV3Optional[str] = _CadV3Field(default="")
    concept: _CadV3Optional[str] = _CadV3Field(default="")
    unit: str = _CadV3Field(default="meters")
    width: _CadV3Optional[float] = None
    height: _CadV3Optional[float] = None
    venue_width: _CadV3Optional[float] = None
    venue_height: _CadV3Optional[float] = None
    audience_count: _CadV3Optional[int] = None
    include_svg: bool = True
    include_pdf: bool = True
    include_dxf: bool = True
    notes: _CadV3Optional[str] = ""


_CAD_V3_LAYER_COLORS: _CadV3Dict[str, int] = {
    "BC_TITLE_BLOCK": 7,
    "BC_VENUE_BOUNDARY": 8,
    "BC_STAGE": 1,
    "BC_LED_SCREEN": 5,
    "BC_AUDIENCE": 3,
    "BC_VIP": 2,
    "BC_FOH": 4,
    "BC_BOH": 6,
    "BC_CHAIRS": 30,
    "BC_POWER": 1,
    "BC_CAMERA": 5,
    "BC_CABLE_SIGNAL": 6,
    "BC_DIMENSIONS": 7,
    "BC_TEXT": 7,
    "BC_GRID": 9,
}


def _cad_v3_safe_name(value: _CadV3Any, fallback: str = "cad") -> str:
    value = str(value or fallback).strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-._")
    return value or fallback


def _cad_v3_num(value: _CadV3Any, default: float) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _cad_v3_int(value: _CadV3Any, default: int) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _cad_v3_unit_name(unit: str) -> str:
    raw = str(unit or "meters").strip().lower()
    if raw in {"m", "meter", "metre", "meters", "metres"}:
        return "meters"
    if raw in {"mm", "millimeter", "millimetre", "millimeters", "millimetres"}:
        return "millimeters"
    if raw in {"ft", "feet", "foot"}:
        return "feet"
    return "meters"


def _cad_v3_unit_suffix(unit: str) -> str:
    return {"meters": "m", "millimeters": "mm", "feet": "ft"}.get(_cad_v3_unit_name(unit), "m")


def _cad_v3_insunits(unit: str) -> int:
    # AutoCAD INSUNITS: 4=millimeters, 6=meters, 2=feet
    return {"millimeters": 4, "meters": 6, "feet": 2}.get(_cad_v3_unit_name(unit), 6)


def _cad_v3_parse_brief_dimensions(brief: str) -> _CadV3Tuple[_CadV3Optional[float], _CadV3Optional[float], _CadV3Optional[int]]:
    text = str(brief or "")
    width = None
    height = None
    audience = None

    dim_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meter|metre|ft|feet)?\s*[x×]\s*(\d+(?:\.\d+)?)", text, re.I)
    if dim_match:
        width = float(dim_match.group(1))
        height = float(dim_match.group(2))

    aud_match = re.search(r"(\d{2,6})\s*(?:pax|people|guests|audience|chairs|seats|seating)", text, re.I)
    if aud_match:
        audience = int(aud_match.group(1))

    return width, height, audience


def _cad_v3_rect(x: float, y: float, w: float, h: float, layer: str, label: str, kind: str = "zone") -> _CadV3Dict[str, _CadV3Any]:
    return {
        "id": f"{kind}_{_cad_v3_safe_name(label)}_{uuid.uuid4().hex[:6]}",
        "type": "rect",
        "kind": kind,
        "label": label,
        "layer": layer,
        "x": round(float(x), 3),
        "y": round(float(y), 3),
        "w": round(float(w), 3),
        "h": round(float(h), 3),
    }


def _cad_v3_line(x1: float, y1: float, x2: float, y2: float, layer: str, label: str, kind: str = "line") -> _CadV3Dict[str, _CadV3Any]:
    return {
        "id": f"{kind}_{_cad_v3_safe_name(label)}_{uuid.uuid4().hex[:6]}",
        "type": "line",
        "kind": kind,
        "label": label,
        "layer": layer,
        "x1": round(float(x1), 3),
        "y1": round(float(y1), 3),
        "x2": round(float(x2), 3),
        "y2": round(float(y2), 3),
    }


def _cad_v3_text(x: float, y: float, text: str, layer: str = "BC_TEXT", height: float = 0.38, kind: str = "text") -> _CadV3Dict[str, _CadV3Any]:
    return {
        "id": f"{kind}_{uuid.uuid4().hex[:6]}",
        "type": "text",
        "kind": kind,
        "label": text,
        "text": text,
        "layer": layer,
        "x": round(float(x), 3),
        "y": round(float(y), 3),
        "height": round(float(height), 3),
    }


def _cad_v3_insert(block: str, x: float, y: float, layer: str, label: str, rotation: float = 0.0, scale: float = 1.0, kind: str = "symbol") -> _CadV3Dict[str, _CadV3Any]:
    return {
        "id": f"{kind}_{_cad_v3_safe_name(label)}_{uuid.uuid4().hex[:6]}",
        "type": "insert",
        "kind": kind,
        "label": label,
        "block": block,
        "layer": layer,
        "x": round(float(x), 3),
        "y": round(float(y), 3),
        "rotation": round(float(rotation), 3),
        "scale": round(float(scale), 3),
    }


def _cad_v3_build_layout(payload: CadV3GenerateRequest) -> _CadV3Dict[str, _CadV3Any]:
    brief = str(payload.brief or "")
    parsed_w, parsed_h, parsed_audience = _cad_v3_parse_brief_dimensions(brief)
    unit = _cad_v3_unit_name(payload.unit)
    unit_s = _cad_v3_unit_suffix(unit)

    w = _cad_v3_num(payload.venue_width or payload.width or parsed_w, 36.0)
    h = _cad_v3_num(payload.venue_height or payload.height or parsed_h, 22.0)
    w = max(18.0, min(w, 180.0))
    h = max(12.0, min(h, 120.0))
    audience = max(20, min(_cad_v3_int(payload.audience_count or parsed_audience, 300), 1200))

    title = str(payload.title or "BriefCraft CAD Layout V3")
    project_id = str(payload.project_id or "demo-cad-project")

    stage = _cad_v3_rect(w * 0.055, h * 0.31, w * 0.205, h * 0.38, "BC_STAGE", "STAGE", "stage")
    led = _cad_v3_rect(stage["x"] + stage["w"] * 0.06, stage["y"] + stage["h"] - max(0.65, h * 0.035), stage["w"] * 0.88, max(0.55, h * 0.028), "BC_LED_SCREEN", "LED SCREEN", "led")
    vip = _cad_v3_rect(w * 0.285, h * 0.36, w * 0.105, h * 0.28, "BC_VIP", "VIP SEATING", "vip")
    audience_zone = _cad_v3_rect(w * 0.41, h * 0.16, w * 0.35, h * 0.68, "BC_AUDIENCE", "AUDIENCE SEATING", "audience")
    foh = _cad_v3_rect(w * 0.80, h * 0.41, w * 0.075, h * 0.18, "BC_FOH", "FOH CONTROL", "foh")
    boh = _cad_v3_rect(w * 0.045, h * 0.075, w * 0.18, h * 0.14, "BC_BOH", "BOH / GREEN ROOM", "boh")
    db = _cad_v3_insert("BC_POWER_DB", w * 0.90, h * 0.14, "BC_POWER", "MAIN POWER DB", 0, 1.15, "power")

    objects: _CadV3List[_CadV3Dict[str, _CadV3Any]] = []
    objects.append(_cad_v3_rect(0, 0, w, h, "BC_VENUE_BOUNDARY", "VENUE BOUNDARY", "boundary"))
    objects.extend([stage, led, vip, audience_zone, foh, boh, db])

    # Chairs: real INSERT references, capped for DXF performance but scheduled with full requested count.
    seat_count_to_draw = min(audience, 520)
    cols = max(4, int(_cad_v3_math.sqrt(seat_count_to_draw * audience_zone["w"] / max(audience_zone["h"], 1))))
    rows = max(1, int(_cad_v3_math.ceil(seat_count_to_draw / cols)))
    chair_dx = audience_zone["w"] / max(cols, 1)
    chair_dy = audience_zone["h"] / max(rows, 1)
    chair_scale = max(0.42, min(0.72, min(chair_dx, chair_dy) * 0.75))
    drawn = 0
    for r in range(rows):
        for c in range(cols):
            if drawn >= seat_count_to_draw:
                break
            x = audience_zone["x"] + chair_dx * (c + 0.5)
            y = audience_zone["y"] + chair_dy * (r + 0.5)
            objects.append(_cad_v3_insert("BC_CHAIR", x, y, "BC_CHAIRS", f"Chair {drawn + 1}", 0, chair_scale, "chair"))
            drawn += 1
        if drawn >= seat_count_to_draw:
            break

    # Cameras and technical symbols.
    cameras = [
        _cad_v3_insert("BC_CAMERA", foh["x"] + foh["w"] * 0.5, foh["y"] + foh["h"] + h * 0.06, "BC_CAMERA", "CAM-1 CENTER", 180, 1.0, "camera"),
        _cad_v3_insert("BC_CAMERA", w * 0.32, h * 0.82, "BC_CAMERA", "CAM-2 LEFT", 220, 1.0, "camera"),
        _cad_v3_insert("BC_CAMERA", w * 0.32, h * 0.18, "BC_CAMERA", "CAM-3 RIGHT", 140, 1.0, "camera"),
    ]
    objects.extend(cameras)

    cable_routes = [
        _cad_v3_line(db["x"], db["y"], stage["x"] + stage["w"] * 0.5, stage["y"], "BC_CABLE_SIGNAL", "POWER TO STAGE", "cable"),
        _cad_v3_line(db["x"], db["y"], led["x"] + led["w"] * 0.5, led["y"], "BC_CABLE_SIGNAL", "POWER TO LED", "cable"),
        _cad_v3_line(foh["x"], foh["y"], led["x"] + led["w"], led["y"], "BC_CABLE_SIGNAL", "FOH SIGNAL TO LED", "signal"),
        _cad_v3_line(foh["x"], foh["y"] + foh["h"], stage["x"] + stage["w"], stage["y"] + stage["h"] * 0.5, "BC_CABLE_SIGNAL", "AUDIO / COMMS ROUTE", "signal"),
    ]
    objects.extend(cable_routes)

    # Dimension graphics as CAD-readable LINE + TEXT entities.
    dim_offset = max(0.9, h * 0.045)
    objects.append(_cad_v3_line(0, -dim_offset, w, -dim_offset, "BC_DIMENSIONS", "VENUE WIDTH DIM", "dimension"))
    objects.append(_cad_v3_line(0, -dim_offset * 1.25, 0, 0, "BC_DIMENSIONS", "WIDTH LEFT EXT", "dimension"))
    objects.append(_cad_v3_line(w, -dim_offset * 1.25, w, 0, "BC_DIMENSIONS", "WIDTH RIGHT EXT", "dimension"))
    objects.append(_cad_v3_text(w * 0.45, -dim_offset * 1.5, f"Venue width: {w:g}{unit_s}", "BC_DIMENSIONS", max(0.32, h * 0.018), "dimension_text"))
    objects.append(_cad_v3_line(-dim_offset, 0, -dim_offset, h, "BC_DIMENSIONS", "VENUE HEIGHT DIM", "dimension"))
    objects.append(_cad_v3_line(-dim_offset * 1.25, 0, 0, 0, "BC_DIMENSIONS", "HEIGHT BOTTOM EXT", "dimension"))
    objects.append(_cad_v3_line(-dim_offset * 1.25, h, 0, h, "BC_DIMENSIONS", "HEIGHT TOP EXT", "dimension"))
    objects.append(_cad_v3_text(-dim_offset * 1.8, h * 0.48, f"Venue depth: {h:g}{unit_s}", "BC_DIMENSIONS", max(0.32, h * 0.018), "dimension_text"))

    # Labels.
    for rect in [stage, led, vip, audience_zone, foh, boh]:
        objects.append(_cad_v3_text(rect["x"] + 0.25, rect["y"] + rect["h"] * 0.52, rect["label"], rect["layer"], max(0.32, h * 0.018), "label"))
    objects.append(_cad_v3_text(db["x"] + 0.65, db["y"] + 0.15, "MAIN DB", "BC_POWER", max(0.28, h * 0.016), "label"))
    for cam in cameras:
        objects.append(_cad_v3_text(cam["x"] + 0.45, cam["y"] + 0.2, cam["label"], "BC_CAMERA", max(0.28, h * 0.016), "label"))

    element_list = [
        {"name": "Stage", "qty": 1, "layer": "BC_STAGE", "size": f"{stage['w']:g} x {stage['h']:g} {unit_s}"},
        {"name": "LED screen", "qty": 1, "layer": "BC_LED_SCREEN", "size": f"{led['w']:g} x {led['h']:g} {unit_s}"},
        {"name": "Audience chairs", "qty": audience, "layer": "BC_CHAIRS", "size": "symbol insert"},
        {"name": "VIP zone", "qty": 1, "layer": "BC_VIP", "size": f"{vip['w']:g} x {vip['h']:g} {unit_s}"},
        {"name": "FOH control", "qty": 1, "layer": "BC_FOH", "size": f"{foh['w']:g} x {foh['h']:g} {unit_s}"},
        {"name": "BOH", "qty": 1, "layer": "BC_BOH", "size": f"{boh['w']:g} x {boh['h']:g} {unit_s}"},
        {"name": "Camera positions", "qty": 3, "layer": "BC_CAMERA", "size": "symbol insert"},
        {"name": "Main power DB", "qty": 1, "layer": "BC_POWER", "size": "symbol insert"},
    ]
    power_schedule = [
        {"circuit": "P-01", "load": "LED wall + processor", "source": "Main DB", "route": "DB to LED", "status": "planned"},
        {"circuit": "P-02", "load": "Stage lighting / truss", "source": "Main DB", "route": "DB to stage", "status": "planned"},
        {"circuit": "S-01", "load": "FOH video signal", "source": "FOH", "route": "FOH to LED", "status": "planned"},
        {"circuit": "A-01", "load": "Audio / comms", "source": "FOH", "route": "FOH to stage", "status": "planned"},
    ]

    return {
        "version": "CAD_V3",
        "project_id": project_id,
        "title": title,
        "unit": unit,
        "unit_suffix": unit_s,
        "venue": {"width": round(w, 3), "height": round(h, 3), "area": round(w * h, 3)},
        "audience_count_requested": audience,
        "audience_symbols_drawn": seat_count_to_draw,
        "brief": brief,
        "concept": str(payload.concept or ""),
        "layers": _CAD_V3_LAYER_COLORS,
        "objects": objects,
        "element_list": element_list,
        "power_schedule": power_schedule,
        "notes": str(payload.notes or ""),
        "created_at": now_iso(),
    }


def _cad_v3_dxf_pair(code: int, value: _CadV3Any) -> str:
    if isinstance(value, float):
        value = round(value, 6)
    return f"{code}\n{value}\n"


def _cad_v3_dxf_escape(text: _CadV3Any) -> str:
    return str(text or "").replace("\r", " ").replace("\n", "\\P")[:900]


def _cad_v3_dxf_lwpolyline(points: _CadV3List[_CadV3Tuple[float, float]], layer: str, closed: bool = True) -> _CadV3List[str]:
    out = [
        _cad_v3_dxf_pair(0, "LWPOLYLINE"),
        _cad_v3_dxf_pair(8, layer),
        _cad_v3_dxf_pair(90, len(points)),
        _cad_v3_dxf_pair(70, 1 if closed else 0),
    ]
    for x, y in points:
        out.append(_cad_v3_dxf_pair(10, float(x)))
        out.append(_cad_v3_dxf_pair(20, float(y)))
    return out


def _cad_v3_dxf_rect(obj: _CadV3Dict[str, _CadV3Any]) -> _CadV3List[str]:
    x, y, w, h = float(obj["x"]), float(obj["y"]), float(obj["w"]), float(obj["h"])
    return _cad_v3_dxf_lwpolyline([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], str(obj.get("layer") or "0"), True)


def _cad_v3_dxf_line(obj: _CadV3Dict[str, _CadV3Any]) -> _CadV3List[str]:
    return [
        _cad_v3_dxf_pair(0, "LINE"),
        _cad_v3_dxf_pair(8, obj.get("layer") or "0"),
        _cad_v3_dxf_pair(10, float(obj.get("x1", 0))),
        _cad_v3_dxf_pair(20, float(obj.get("y1", 0))),
        _cad_v3_dxf_pair(30, 0),
        _cad_v3_dxf_pair(11, float(obj.get("x2", 0))),
        _cad_v3_dxf_pair(21, float(obj.get("y2", 0))),
        _cad_v3_dxf_pair(31, 0),
    ]


def _cad_v3_dxf_text(obj: _CadV3Dict[str, _CadV3Any]) -> _CadV3List[str]:
    return [
        _cad_v3_dxf_pair(0, "TEXT"),
        _cad_v3_dxf_pair(8, obj.get("layer") or "BC_TEXT"),
        _cad_v3_dxf_pair(10, float(obj.get("x", 0))),
        _cad_v3_dxf_pair(20, float(obj.get("y", 0))),
        _cad_v3_dxf_pair(30, 0),
        _cad_v3_dxf_pair(40, float(obj.get("height", 0.35))),
        _cad_v3_dxf_pair(1, _cad_v3_dxf_escape(obj.get("text") or obj.get("label") or "")),
        _cad_v3_dxf_pair(7, "STANDARD"),
    ]


def _cad_v3_dxf_mtext(x: float, y: float, width: float, height: float, text: str, layer: str = "BC_TEXT") -> _CadV3List[str]:
    return [
        _cad_v3_dxf_pair(0, "MTEXT"),
        _cad_v3_dxf_pair(8, layer),
        _cad_v3_dxf_pair(10, float(x)),
        _cad_v3_dxf_pair(20, float(y)),
        _cad_v3_dxf_pair(30, 0),
        _cad_v3_dxf_pair(40, float(height)),
        _cad_v3_dxf_pair(41, float(width)),
        _cad_v3_dxf_pair(71, 1),
        _cad_v3_dxf_pair(1, _cad_v3_dxf_escape(text)),
    ]


def _cad_v3_dxf_insert(obj: _CadV3Dict[str, _CadV3Any]) -> _CadV3List[str]:
    scale = float(obj.get("scale", 1) or 1)
    return [
        _cad_v3_dxf_pair(0, "INSERT"),
        _cad_v3_dxf_pair(8, obj.get("layer") or "0"),
        _cad_v3_dxf_pair(2, obj.get("block") or "BC_SYMBOL"),
        _cad_v3_dxf_pair(10, float(obj.get("x", 0))),
        _cad_v3_dxf_pair(20, float(obj.get("y", 0))),
        _cad_v3_dxf_pair(30, 0),
        _cad_v3_dxf_pair(41, scale),
        _cad_v3_dxf_pair(42, scale),
        _cad_v3_dxf_pair(43, 1),
        _cad_v3_dxf_pair(50, float(obj.get("rotation", 0) or 0)),
    ]


def _cad_v3_dxf_blocks() -> _CadV3List[str]:
    out: _CadV3List[str] = []

    def start_block(name: str, layer: str) -> None:
        out.extend([
            _cad_v3_dxf_pair(0, "BLOCK"),
            _cad_v3_dxf_pair(8, layer),
            _cad_v3_dxf_pair(2, name),
            _cad_v3_dxf_pair(70, 0),
            _cad_v3_dxf_pair(10, 0),
            _cad_v3_dxf_pair(20, 0),
            _cad_v3_dxf_pair(30, 0),
            _cad_v3_dxf_pair(3, name),
        ])

    def end_block() -> None:
        out.extend([_cad_v3_dxf_pair(0, "ENDBLK"), _cad_v3_dxf_pair(8, "0")])

    start_block("BC_CHAIR", "BC_CHAIRS")
    out.extend(_cad_v3_dxf_lwpolyline([(-0.22, -0.18), (0.22, -0.18), (0.22, 0.16), (-0.22, 0.16)], "BC_CHAIRS", True))
    out.extend([
        _cad_v3_dxf_pair(0, "LINE"), _cad_v3_dxf_pair(8, "BC_CHAIRS"),
        _cad_v3_dxf_pair(10, -0.24), _cad_v3_dxf_pair(20, 0.22), _cad_v3_dxf_pair(11, 0.24), _cad_v3_dxf_pair(21, 0.22),
    ])
    end_block()

    start_block("BC_POWER_DB", "BC_POWER")
    out.extend(_cad_v3_dxf_lwpolyline([(-0.35, -0.28), (0.35, -0.28), (0.35, 0.28), (-0.35, 0.28)], "BC_POWER", True))
    out.extend(_cad_v3_dxf_mtext(-0.20, -0.08, 0.5, 0.16, "DB", "BC_POWER"))
    end_block()

    start_block("BC_CAMERA", "BC_CAMERA")
    out.extend(_cad_v3_dxf_lwpolyline([(-0.25, -0.18), (0.20, 0), (-0.25, 0.18)], "BC_CAMERA", True))
    out.extend([
        _cad_v3_dxf_pair(0, "LINE"), _cad_v3_dxf_pair(8, "BC_CAMERA"),
        _cad_v3_dxf_pair(10, 0.20), _cad_v3_dxf_pair(20, 0), _cad_v3_dxf_pair(11, 0.45), _cad_v3_dxf_pair(21, 0),
    ])
    end_block()

    return out


def _cad_v3_render_dxf(layout: _CadV3Dict[str, _CadV3Any]) -> str:
    unit = str(layout.get("unit") or "meters")
    venue = layout.get("venue") or {}
    w = float(venue.get("width", 36))
    h = float(venue.get("height", 22))
    title_x = w + max(2.0, w * 0.05)
    title_w = max(9.0, w * 0.22)

    out: _CadV3List[str] = []
    out.extend([
        _cad_v3_dxf_pair(0, "SECTION"), _cad_v3_dxf_pair(2, "HEADER"),
        _cad_v3_dxf_pair(9, "$ACADVER"), _cad_v3_dxf_pair(1, "AC1027"),
        _cad_v3_dxf_pair(9, "$INSUNITS"), _cad_v3_dxf_pair(70, _cad_v3_insunits(unit)),
        _cad_v3_dxf_pair(0, "ENDSEC"),
    ])

    out.extend([_cad_v3_dxf_pair(0, "SECTION"), _cad_v3_dxf_pair(2, "TABLES")])
    out.extend([_cad_v3_dxf_pair(0, "TABLE"), _cad_v3_dxf_pair(2, "LAYER"), _cad_v3_dxf_pair(70, len(_CAD_V3_LAYER_COLORS) + 1)])
    out.extend([_cad_v3_dxf_pair(0, "LAYER"), _cad_v3_dxf_pair(2, "0"), _cad_v3_dxf_pair(70, 0), _cad_v3_dxf_pair(62, 7), _cad_v3_dxf_pair(6, "CONTINUOUS")])
    for layer, color in _CAD_V3_LAYER_COLORS.items():
        out.extend([_cad_v3_dxf_pair(0, "LAYER"), _cad_v3_dxf_pair(2, layer), _cad_v3_dxf_pair(70, 0), _cad_v3_dxf_pair(62, color), _cad_v3_dxf_pair(6, "CONTINUOUS")])
    out.extend([_cad_v3_dxf_pair(0, "ENDTAB"), _cad_v3_dxf_pair(0, "ENDSEC")])

    out.extend([_cad_v3_dxf_pair(0, "SECTION"), _cad_v3_dxf_pair(2, "BLOCKS")])
    out.extend(_cad_v3_dxf_blocks())
    out.extend([_cad_v3_dxf_pair(0, "ENDSEC")])

    out.extend([_cad_v3_dxf_pair(0, "SECTION"), _cad_v3_dxf_pair(2, "ENTITIES")])

    for obj in layout.get("objects", []):
        typ = obj.get("type")
        if typ == "rect":
            out.extend(_cad_v3_dxf_rect(obj))
        elif typ == "line":
            out.extend(_cad_v3_dxf_line(obj))
        elif typ == "text":
            out.extend(_cad_v3_dxf_text(obj))
        elif typ == "insert":
            out.extend(_cad_v3_dxf_insert(obj))

    # Title block, power schedule, and element list inside DXF.
    title_h = max(12.0, h * 0.72)
    out.extend(_cad_v3_dxf_lwpolyline([(title_x, 0), (title_x + title_w, 0), (title_x + title_w, title_h), (title_x, title_h)], "BC_TITLE_BLOCK", True))
    for y_ratio in [0.86, 0.73, 0.60, 0.47, 0.34, 0.21, 0.10]:
        y = title_h * y_ratio
        out.extend(_cad_v3_dxf_line({"layer": "BC_TITLE_BLOCK", "x1": title_x, "y1": y, "x2": title_x + title_w, "y2": y}))

    title_text = (
        f"BRIEFCRAFT-AI CAD V3\\P"
        f"Project: {layout.get('project_id')}\\P"
        f"Title: {layout.get('title')}\\P"
        f"Venue: {w:g} x {h:g} {layout.get('unit_suffix')}\\P"
        f"Audience: {layout.get('audience_count_requested')}"
    )
    out.extend(_cad_v3_dxf_mtext(title_x + 0.35, title_h - 0.75, title_w - 0.7, max(0.32, h * 0.018), title_text, "BC_TITLE_BLOCK"))

    power_lines = ["POWER / SIGNAL SCHEDULE"]
    for item in layout.get("power_schedule", []):
        power_lines.append(f"{item.get('circuit')}: {item.get('load')} | {item.get('route')}")
    out.extend(_cad_v3_dxf_mtext(title_x + 0.35, title_h * 0.58, title_w - 0.7, max(0.26, h * 0.014), "\\P".join(power_lines), "BC_TEXT"))

    element_lines = ["ELEMENT LIST"]
    for item in layout.get("element_list", []):
        element_lines.append(f"{item.get('qty')}x {item.get('name')} - {item.get('size')}")
    out.extend(_cad_v3_dxf_mtext(title_x + 0.35, title_h * 0.28, title_w - 0.7, max(0.24, h * 0.013), "\\P".join(element_lines), "BC_TEXT"))

    out.extend([_cad_v3_dxf_pair(0, "ENDSEC"), _cad_v3_dxf_pair(0, "EOF")])
    return "".join(out)


def _cad_v3_svg_escape(value: _CadV3Any) -> str:
    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _cad_v3_render_svg(layout: _CadV3Dict[str, _CadV3Any]) -> str:
    venue = layout.get("venue") or {}
    w = float(venue.get("width", 36))
    h = float(venue.get("height", 22))
    vb_w = 1600
    vb_h = 900
    margin = 70
    title_w = 310
    scale = min((vb_w - margin * 2 - title_w) / max(w, 1), (vb_h - margin * 2) / max(h, 1))

    def sx(x: float) -> float:
        return margin + x * scale

    def sy(y: float) -> float:
        return vb_h - margin - y * scale

    def rect(obj: _CadV3Dict[str, _CadV3Any], css: str) -> str:
        x = float(obj["x"]); y = float(obj["y"]); rw = float(obj["w"]); rh = float(obj["h"])
        return f'<rect x="{sx(x):.2f}" y="{sy(y + rh):.2f}" width="{rw * scale:.2f}" height="{rh * scale:.2f}" class="{css}" />'

    def line(obj: _CadV3Dict[str, _CadV3Any], css: str) -> str:
        return f'<line x1="{sx(float(obj["x1"])):.2f}" y1="{sy(float(obj["y1"])):.2f}" x2="{sx(float(obj["x2"])):.2f}" y2="{sy(float(obj["y2"])):.2f}" class="{css}" />'

    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{vb_w}" height="{vb_h}" viewBox="0 0 {vb_w} {vb_h}">
<style>
.bg{{fill:#070911}} .grid{{stroke:#202433;stroke-width:1}} .boundary{{fill:none;stroke:#f6e7b1;stroke-width:3}} .stage{{fill:#301619;stroke:#ff6b6b;stroke-width:2}} .led{{fill:#10263f;stroke:#58a6ff;stroke-width:2}} .audience{{fill:#102a18;stroke:#76d275;stroke-width:2}} .vip{{fill:#2f2611;stroke:#ffd166;stroke-width:2}} .foh{{fill:#101e33;stroke:#80bfff;stroke-width:2}} .boh{{fill:#221832;stroke:#d0a5ff;stroke-width:2}} .chair{{fill:#d8c27c;stroke:#332c18;stroke-width:.6}} .power{{fill:#401414;stroke:#ff5757;stroke-width:2}} .camera{{fill:#101b32;stroke:#8fd3ff;stroke-width:2}} .cable{{stroke:#f6c453;stroke-width:3;stroke-dasharray:9 7;fill:none}} .dim{{stroke:#d9d9d9;stroke-width:1.4;fill:none}} .text{{fill:#f7f3dd;font-family:Arial,Helvetica,sans-serif;font-size:14px}} .small{{font-size:11px}} .title{{font-size:22px;font-weight:700}}
</style>
<rect class="bg" width="100%" height="100%"/>''']

    # Background grid.
    grid_step = max(1.0, round(min(w, h) / 10, 1))
    gx = 0.0
    while gx <= w:
        parts.append(f'<line x1="{sx(gx):.2f}" y1="{sy(0):.2f}" x2="{sx(gx):.2f}" y2="{sy(h):.2f}" class="grid"/>')
        gx += grid_step
    gy = 0.0
    while gy <= h:
        parts.append(f'<line x1="{sx(0):.2f}" y1="{sy(gy):.2f}" x2="{sx(w):.2f}" y2="{sy(gy):.2f}" class="grid"/>')
        gy += grid_step

    css_by_kind = {"boundary": "boundary", "stage": "stage", "led": "led", "audience": "audience", "vip": "vip", "foh": "foh", "boh": "boh"}
    for obj in layout.get("objects", []):
        typ = obj.get("type")
        kind = obj.get("kind")
        if typ == "rect":
            parts.append(rect(obj, css_by_kind.get(kind, "boundary")))
        elif typ == "line":
            parts.append(line(obj, "cable" if kind in {"cable", "signal"} else "dim"))
        elif typ == "insert":
            x = sx(float(obj.get("x", 0))); y = sy(float(obj.get("y", 0)))
            if obj.get("block") == "BC_CHAIR":
                s = max(4, float(obj.get("scale", 1)) * scale * 0.45)
                parts.append(f'<rect x="{x - s / 2:.2f}" y="{y - s / 2:.2f}" width="{s:.2f}" height="{s:.2f}" rx="1.2" class="chair"/>')
            elif obj.get("block") == "BC_POWER_DB":
                parts.append(f'<rect x="{x - 9:.2f}" y="{y - 9:.2f}" width="18" height="18" class="power"/><text x="{x + 12:.2f}" y="{y + 4:.2f}" class="text small">DB</text>')
            elif obj.get("block") == "BC_CAMERA":
                parts.append(f'<polygon points="{x - 10:.2f},{y - 8:.2f} {x + 10:.2f},{y:.2f} {x - 10:.2f},{y + 8:.2f}" class="camera"/>')
        elif typ == "text":
            cls = "text small" if obj.get("kind") != "label" else "text"
            parts.append(f'<text x="{sx(float(obj.get("x", 0))):.2f}" y="{sy(float(obj.get("y", 0))):.2f}" class="{cls}">{_cad_v3_svg_escape(obj.get("text") or obj.get("label"))}</text>')

    tx = vb_w - title_w + 18
    parts.append(f'<rect x="{vb_w - title_w}" y="0" width="{title_w}" height="{vb_h}" fill="#0d1018" stroke="#b99a4d"/>')
    parts.append(f'<text x="{tx}" y="42" class="text title">BriefCraft CAD V3</text>')
    parts.append(f'<text x="{tx}" y="72" class="text small">Project: {_cad_v3_svg_escape(layout.get("project_id"))}</text>')
    parts.append(f'<text x="{tx}" y="94" class="text small">Venue: {w:g} x {h:g} {_cad_v3_svg_escape(layout.get("unit_suffix"))}</text>')
    parts.append(f'<text x="{tx}" y="116" class="text small">Audience: {layout.get("audience_count_requested")}</text>')
    parts.append(f'<text x="{tx}" y="152" class="text">Power Schedule</text>')
    y = 176
    for item in layout.get("power_schedule", []):
        parts.append(f'<text x="{tx}" y="{y}" class="text small">{_cad_v3_svg_escape(item.get("circuit"))}: {_cad_v3_svg_escape(item.get("load"))}</text>')
        y += 20
    y += 22
    parts.append(f'<text x="{tx}" y="{y}" class="text">Element List</text>')
    y += 24
    for item in layout.get("element_list", []):
        parts.append(f'<text x="{tx}" y="{y}" class="text small">{item.get("qty")}x {_cad_v3_svg_escape(item.get("name"))}</text>')
        y += 20
    parts.append("</svg>")
    return "\n".join(parts)


def _cad_v3_render_pdf(layout: _CadV3Dict[str, _CadV3Any], out_path: Path) -> None:
    try:
        from reportlab.lib.pagesizes import A3, landscape
        from reportlab.pdfgen import canvas
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"reportlab is required for CAD PDF export: {exc}")

    venue = layout.get("venue") or {}
    w = float(venue.get("width", 36))
    h = float(venue.get("height", 22))
    c = canvas.Canvas(str(out_path), pagesize=landscape(A3))
    page_w, page_h = landscape(A3)
    margin = 34
    title_w = 205
    scale = min((page_w - margin * 2 - title_w) / max(w, 1), (page_h - margin * 2) / max(h, 1))

    def px(x: float) -> float:
        return margin + x * scale

    def py(y: float) -> float:
        return margin + y * scale

    c.setTitle(str(layout.get("title") or "BriefCraft CAD V3"))
    c.setFont("Helvetica-Bold", 15)
    c.drawString(margin, page_h - 26, str(layout.get("title") or "BriefCraft CAD V3"))
    c.setFont("Helvetica", 8)

    # Grid.
    c.setLineWidth(0.25)
    grid_step = max(1.0, round(min(w, h) / 10, 1))
    gx = 0.0
    while gx <= w:
        c.line(px(gx), py(0), px(gx), py(h))
        gx += grid_step
    gy = 0.0
    while gy <= h:
        c.line(px(0), py(gy), px(w), py(gy))
        gy += grid_step

    for obj in layout.get("objects", []):
        typ = obj.get("type")
        if typ == "rect":
            c.setLineWidth(1.0 if obj.get("kind") != "boundary" else 1.8)
            c.rect(px(float(obj["x"])), py(float(obj["y"])), float(obj["w"]) * scale, float(obj["h"]) * scale, stroke=1, fill=0)
        elif typ == "line":
            c.setLineWidth(0.8 if obj.get("kind") in {"dimension", "dimension_text"} else 1.1)
            c.line(px(float(obj.get("x1", 0))), py(float(obj.get("y1", 0))), px(float(obj.get("x2", 0))), py(float(obj.get("y2", 0))))
        elif typ == "insert":
            x = px(float(obj.get("x", 0))); y = py(float(obj.get("y", 0)))
            if obj.get("block") == "BC_CHAIR":
                s = max(2.5, float(obj.get("scale", 1)) * scale * 0.45)
                c.rect(x - s / 2, y - s / 2, s, s, stroke=1, fill=0)
            elif obj.get("block") == "BC_POWER_DB":
                c.rect(x - 5, y - 5, 10, 10, stroke=1, fill=0)
                c.drawString(x + 7, y - 2, "DB")
            elif obj.get("block") == "BC_CAMERA":
                p = c.beginPath()
                p.moveTo(x - 6, y - 5); p.lineTo(x + 7, y); p.lineTo(x - 6, y + 5); p.close()
                c.drawPath(p, stroke=1, fill=0)
        elif typ == "text":
            c.setFont("Helvetica", max(5, min(10, float(obj.get("height", 0.35)) * scale * 0.5)))
            c.drawString(px(float(obj.get("x", 0))), py(float(obj.get("y", 0))), str(obj.get("text") or obj.get("label") or "")[:80])

    # Title block.
    tx = page_w - title_w + 16
    c.setLineWidth(1)
    c.rect(page_w - title_w, margin, title_w - 18, page_h - margin * 2, stroke=1, fill=0)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(tx, page_h - margin - 18, "BRIEFCRAFT-AI CAD V3")
    c.setFont("Helvetica", 8)
    lines = [
        f"Project: {layout.get('project_id')}",
        f"Venue: {w:g} x {h:g} {layout.get('unit_suffix')}",
        f"Audience: {layout.get('audience_count_requested')}",
        f"Generated: {layout.get('created_at')}",
        "",
        "POWER / SIGNAL SCHEDULE",
    ]
    for item in layout.get("power_schedule", []):
        lines.append(f"{item.get('circuit')}: {item.get('load')}")
    lines.extend(["", "ELEMENT LIST"])
    for item in layout.get("element_list", []):
        lines.append(f"{item.get('qty')}x {item.get('name')} - {item.get('size')}")

    y = page_h - margin - 42
    for line_text in lines:
        c.drawString(tx, y, str(line_text)[:42])
        y -= 12
        if y < margin + 16:
            break

    c.showPage()
    c.save()


def _cad_v3_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _cad_v3_upload_public_file(path: Path, project_id: str, content_type: _CadV3Optional[str] = None) -> str:
    content_type = content_type or _cad_v3_mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    try:
        if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
            import requests
            storage_name = f"cad/v3/{_cad_v3_safe_name(project_id)}/{int(_cad_v3_time.time())}-{uuid.uuid4().hex[:10]}-{_cad_v3_safe_name(path.name)}"
            upload_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/{SUPABASE_STORAGE_BUCKET}/{storage_name}"
            headers = {
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Content-Type": content_type,
                "Cache-Control": "31536000",
                "x-upsert": "true",
            }
            res = requests.post(upload_url, headers=headers, data=path.read_bytes(), timeout=120)
            if res.status_code in (200, 201):
                return f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{SUPABASE_STORAGE_BUCKET}/{storage_name}"
            print("CAD V3 Supabase upload failed:", res.status_code, res.text[:400])
    except Exception as exc:
        print("CAD V3 public upload fallback:", repr(exc))

    return absolute_public_url(relative_public_url(path))


def _cad_v3_generate_files(layout: _CadV3Dict[str, _CadV3Any], include_svg: bool = True, include_pdf: bool = True, include_dxf: bool = True) -> _CadV3Dict[str, _CadV3Any]:
    project_id = str(layout.get("project_id") or "demo-cad-project")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"cad_v3_{_cad_v3_safe_name(project_id)}_{stamp}_{uuid.uuid4().hex[:8]}"
    out_dir = CAD_DIR / _cad_v3_safe_name(project_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{base_name}.json"
    _cad_v3_write_text(json_path, dump_json(layout))
    urls: _CadV3Dict[str, _CadV3Any] = {
        "json_path": str(json_path),
        "json_url": _cad_v3_upload_public_file(json_path, project_id, "application/json"),
    }

    if include_svg:
        svg_path = out_dir / f"{base_name}.svg"
        _cad_v3_write_text(svg_path, _cad_v3_render_svg(layout))
        urls["svg_path"] = str(svg_path)
        urls["svg_url"] = _cad_v3_upload_public_file(svg_path, project_id, "image/svg+xml")

    if include_dxf:
        dxf_path = out_dir / f"{base_name}.dxf"
        _cad_v3_write_text(dxf_path, _cad_v3_render_dxf(layout))
        urls["dxf_path"] = str(dxf_path)
        urls["dxf_url"] = _cad_v3_upload_public_file(dxf_path, project_id, "application/dxf")

    if include_pdf:
        pdf_path = out_dir / f"{base_name}.pdf"
        _cad_v3_render_pdf(layout, pdf_path)
        urls["pdf_path"] = str(pdf_path)
        urls["pdf_url"] = _cad_v3_upload_public_file(pdf_path, project_id, "application/pdf")

    return urls


def _cad_v3_response_payload(layout: _CadV3Dict[str, _CadV3Any], urls: _CadV3Optional[_CadV3Dict[str, _CadV3Any]] = None, row: _CadV3Optional[_CadV3Dict[str, _CadV3Any]] = None) -> _CadV3Dict[str, _CadV3Any]:
    urls = urls or {}
    return {
        "ok": True,
        "version": "CAD_V3",
        "id": (row or {}).get("id"),
        "project_id": layout.get("project_id"),
        "status": (row or {}).get("status", "ready"),
        "layout_data": layout,
        "svg_url": urls.get("svg_url") or (row or {}).get("svg_url"),
        "pdf_url": urls.get("pdf_url") or (row or {}).get("pdf_url"),
        "dxf_url": urls.get("dxf_url") or (row or {}).get("dxf_url"),
        "json_url": urls.get("json_url") or (row or {}).get("json_url"),
        "urls": {k: v for k, v in urls.items() if k.endswith("_url")},
    }


@app.get("/api/cad/v3/sample")
def cad_v3_sample() -> _CadV3Dict[str, _CadV3Any]:
    sample = CadV3GenerateRequest(
        project_id="cad-v3-sample",
        title="Sample Event CAD V3",
        brief="36 x 22 meter corporate launch for 300 pax with stage, LED, VIP, FOH, BOH, cameras and power routes.",
        unit="meters",
        width=36,
        height=22,
        audience_count=300,
        concept="Premium tech launch layout",
    )
    layout = _cad_v3_build_layout(sample)
    return {
        "ok": True,
        "version": "CAD_V3",
        "message": "CAD V3 backend is loaded. Use POST /api/cad/v3/generate to create DXF/SVG/PDF/JSON files.",
        "sample_request": sample.model_dump() if hasattr(sample, "model_dump") else sample.dict(),
        "layout_data": layout,
    }


@app.post("/api/cad/v3/calculate")
def cad_v3_calculate(payload: CadV3GenerateRequest) -> _CadV3Dict[str, _CadV3Any]:
    layout = _cad_v3_build_layout(payload)
    return {"ok": True, "version": "CAD_V3", "layout_data": layout}


@app.post("/api/cad/v3/generate")
def cad_v3_generate(payload: CadV3GenerateRequest) -> _CadV3Dict[str, _CadV3Any]:
    layout = _cad_v3_build_layout(payload)
    urls = _cad_v3_generate_files(
        layout,
        include_svg=bool(payload.include_svg),
        include_pdf=bool(payload.include_pdf),
        include_dxf=bool(payload.include_dxf),
    )

    row: _CadV3Dict[str, _CadV3Any] = {}
    try:
        row = db_insert(
            "cad_layouts",
            {
                "project_id": str(layout.get("project_id")),
                "layout_data": layout,
                "svg_url": urls.get("svg_url"),
                "pdf_url": urls.get("pdf_url"),
                "dxf_url": urls.get("dxf_url"),
                "json_url": urls.get("json_url"),
                "status": "ready",
            },
        )
    except Exception as exc:
        print("CAD V3 db_insert failed. Files generated but row was not saved:", repr(exc))
        row = {"id": None, "status": "ready", "db_warning": str(exc)}

    response = _cad_v3_response_payload(layout, urls, row)
    if row.get("db_warning"):
        response["db_warning"] = row.get("db_warning")
    return response


@app.post("/api/cad/v3/upload")
async def cad_v3_upload(
    file: _CadV3UploadFile = _CadV3File(...),
    project_id: str = _CadV3Form("demo-cad-project"),
    brief: str = _CadV3Form(""),
    unit: str = _CadV3Form("meters"),
    width: _CadV3Optional[float] = _CadV3Form(None),
    height: _CadV3Optional[float] = _CadV3Form(None),
    audience_count: _CadV3Optional[int] = _CadV3Form(None),
) -> _CadV3Dict[str, _CadV3Any]:
    original_name = file.filename or "uploaded-cad-reference"
    ext = Path(original_name).suffix.lower() or ".bin"
    upload_dir = UPLOAD_DIR / "cad_v3" / _cad_v3_safe_name(project_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / f"{int(_cad_v3_time.time())}-{uuid.uuid4().hex[:8]}-{_cad_v3_safe_name(original_name)}"
    saved_path.write_bytes(await file.read())
    upload_url = _cad_v3_upload_public_file(saved_path, project_id, file.content_type or _cad_v3_mimetypes.guess_type(str(saved_path))[0] or "application/octet-stream")

    payload = CadV3GenerateRequest(
        project_id=project_id,
        title="Uploaded Reference CAD Layout V3",
        brief=brief or f"CAD layout from uploaded reference file: {original_name}",
        unit=unit,
        width=width,
        height=height,
        audience_count=audience_count,
    )
    layout = _cad_v3_build_layout(payload)
    layout["source_upload"] = {
        "filename": original_name,
        "content_type": file.content_type,
        "extension": ext,
        "url": upload_url,
        "note": "V3 stores the upload as a reference. Real DWG/DXF parsing/import is reserved for the next CAD import phase.",
    }

    urls = _cad_v3_generate_files(layout, include_svg=True, include_pdf=True, include_dxf=True)
    row: _CadV3Dict[str, _CadV3Any] = {}
    try:
        row = db_insert(
            "cad_layouts",
            {
                "project_id": str(project_id),
                "layout_data": layout,
                "svg_url": urls.get("svg_url"),
                "pdf_url": urls.get("pdf_url"),
                "dxf_url": urls.get("dxf_url"),
                "json_url": urls.get("json_url"),
                "status": "ready",
            },
        )
    except Exception as exc:
        print("CAD V3 upload db_insert failed:", repr(exc))
        row = {"id": None, "status": "ready", "db_warning": str(exc)}

    response = _cad_v3_response_payload(layout, urls, row)
    response["upload_url"] = upload_url
    if row.get("db_warning"):
        response["db_warning"] = row.get("db_warning")
    return response


@app.get("/api/cad/v3/{project_id}/latest")
def cad_v3_latest(project_id: str) -> _CadV3Dict[str, _CadV3Any]:
    try:
        rows = db_list("cad_layouts", limit=1, order_key="created_at", desc=True, project_id=str(project_id))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read cad_layouts table: {exc}")

    if not rows:
        raise HTTPException(status_code=404, detail="No CAD V3 layout found for this project_id")

    row = rows[0]
    layout = load_json(row.get("layout_data"), row.get("layout_data") or {})
    return _cad_v3_response_payload(layout, {}, row)

# ==============================================================================
# END CAD BACKEND V3 PATCH
# ==============================================================================
