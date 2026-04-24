"""
AICreative Studio — BriefCraft AI Backend  (APR-25-CLEAN)
Render.com: https://briefcraft-ai.onrender.com
"""
from __future__ import annotations

# ── Standard library ───────────────────────────────────────────────────────────
import base64, io, json, os, re, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Third-party ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from openai import OpenAI
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator
from supabase import Client, create_client

print("MAIN.PY BUILD: APR-25-CLEAN")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL     = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY     = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY", "")
SECRET_KEY       = os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY", "change-me-32-char-secret-key-here")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
IMAGE_MODEL      = os.getenv("IMAGE_MODEL", "dall-e-3").strip()
IMAGE_QUALITY    = os.getenv("IMAGE_QUALITY", "standard").strip()
TTS_MODEL        = os.getenv("TTS_MODEL", "tts-1").strip()
TTS_VOICE        = os.getenv("TTS_VOICE", "alloy").strip()
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1").strip()
PORT             = int(os.getenv("PORT", "10000"))
ALGORITHM        = "HS256"
ACCESS_TOKEN_HOURS = 72
APP_NAME         = "AICreative Studio API"
EMAIL_RE         = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

# Visual defaults
VISUAL_ASPECT_RATIO = "16:9"
VISUAL_PREVIEW_SIZE = "1920x1080"
VISUAL_MASTER_SIZE  = "1536x1024"
VISUAL_PRINT_SIZE   = "3840x2160"

BASE_DIR          = Path(__file__).resolve().parent
EXPORT_DIR        = (BASE_DIR / os.getenv("EXPORT_DIR", "exports")).resolve()
UPLOAD_DIR        = (BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")).resolve()
MEDIA_DIR         = (BASE_DIR / os.getenv("MEDIA_DIR",  "media")).resolve()
RENDER_OUTPUT_DIR = (BASE_DIR / "renders").resolve()
VOICE_DIR         = (MEDIA_DIR / "voice")
for _d in [EXPORT_DIR, UPLOAD_DIR, MEDIA_DIR, RENDER_OUTPUT_DIR, VOICE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# CORS
_BASE_ORIGINS = [
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:5173", "http://127.0.0.1:5173",
    "https://briefly-sparkle.lovable.app",
    "https://aicreative.studio",
]
for _o in os.getenv("ALLOWED_ORIGINS","").split(","):
    _o = _o.strip()
    if _o and _o not in _BASE_ORIGINS:
        _BASE_ORIGINS.append(_o)
ALLOWED_ORIGINS = _BASE_ORIGINS

# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title=APP_NAME, version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://([a-zA-Z0-9\-]+\.)?lovable\.(app|dev)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media",   StaticFiles(directory=str(MEDIA_DIR)),         name="media")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)),        name="uploads")
app.mount("/exports", StaticFiles(directory=str(EXPORT_DIR)),        name="exports")
app.mount("/renders", StaticFiles(directory=str(RENDER_OUTPUT_DIR)), name="renders")

@app.exception_handler(Exception)
async def _exc(_, e: Exception):
    if isinstance(e, HTTPException):
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    return JSONResponse(status_code=500, content={"detail": str(e)})

# ══════════════════════════════════════════════════════════════════════════════
# CLIENTS
# ══════════════════════════════════════════════════════════════════════════════
openai_client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
_sb: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer     = HTTPBearer(auto_error=False)

def hash_password(pw: str) -> str:   return pwd_context.hash(pw)
def verify_password(pw: str, h: str) -> bool: return pwd_context.verify(pw, h)

def create_access_token(uid: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_HOURS)
    return jwt.encode({"sub": uid, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(tok: str) -> str:
    try:
        return jwt.decode(tok, SECRET_KEY, algorithms=[ALGORITHM])["sub"]
    except JWTError:
        raise HTTPException(401, "Invalid or expired token")

def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> Dict[str, Any]:
    if not creds or not creds.credentials:
        raise HTTPException(401, "Authorization header required")
    uid = decode_access_token(creds.credentials)
    if not _sb:
        return {"id": uid, "email": "demo@aicreative.studio", "full_name": "Demo User"}
    r = _sb.table("users").select("*").eq("id", uid).maybe_single().execute()
    if not (r and r.data):
        raise HTTPException(401, "User not found")
    return dict(r.data)

# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def db_get(table: str, **kw) -> Optional[Dict]:
    if not _sb: return None
    q = _sb.table(table).select("*")
    for k, v in kw.items(): q = q.eq(k, v)
    r = q.maybe_single().execute()
    return r.data if r else None

def db_list(table: str, order="created_at", desc=True, limit=100, **kw) -> List[Dict]:
    if not _sb: return []
    q = _sb.table(table).select("*").order(order, desc=desc).limit(limit)
    for k, v in kw.items(): q = q.eq(k, v)
    r = q.execute()
    return r.data or []

def db_insert(table: str, data: Dict) -> Dict:
    if not _sb: return data
    r = _sb.table(table).insert(data).execute()
    return r.data[0] if (r and r.data) else data

def db_update(table: str, row_id: str, data: Dict) -> Dict:
    if not _sb: return data
    data["updated_at"] = datetime.utcnow().isoformat()
    r = _sb.table(table).update(data).eq("id", row_id).execute()
    return r.data[0] if (r and r.data) else data

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def now_iso() -> str: return datetime.now(timezone.utc).isoformat()
def dump_json(v: Any) -> str: return json.dumps(v, ensure_ascii=False, default=str)
def load_json(v: Any, default: Any = None) -> Any:
    if v in (None, ""): return default
    if isinstance(v, (dict, list)): return v
    try: return json.loads(v)
    except: return default

def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+","_",str(name or "asset")).strip("_") or "asset"

def absolute_public_url(rel: str) -> str:
    host = os.getenv("RENDER_EXTERNAL_HOSTNAME","").strip()
    if rel.startswith("http"): return rel
    if not host: return rel
    return f"https://{host}/{rel.lstrip('/')}"

def relative_public_url(path: Path) -> str:
    path = path.resolve()
    for root, prefix in [(MEDIA_DIR,"/media"),(UPLOAD_DIR,"/uploads"),(EXPORT_DIR,"/exports"),(RENDER_OUTPUT_DIR,"/renders")]:
        try: return f"{prefix}/{path.relative_to(root)}"
        except: pass
    return str(path)

# ══════════════════════════════════════════════════════════════════════════════
# LLM / IMAGE
# ══════════════════════════════════════════════════════════════════════════════
def llm(system: str, user: str, json_mode=False) -> str:
    if not openai_client: return "{}" if json_mode else "OpenAI not configured."
    kw: Dict[str,Any] = {"model":OPENAI_MODEL,"max_tokens":3000,"temperature":0.75,
                          "messages":[{"role":"system","content":system},{"role":"user","content":user}]}
    if json_mode: kw["response_format"] = {"type":"json_object"}
    return openai_client.chat.completions.create(**kw).choices[0].message.content.strip()

def parse_json(raw: str, fallback: Any = None) -> Any:
    try: return json.loads(raw)
    except:
        m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', raw)
        if m:
            try: return json.loads(m.group(1))
            except: pass
    return fallback if fallback is not None else {}

def generate_image_b64(prompt: str, size="1024x1024", quality="standard") -> Optional[str]:
    if not openai_client: return None
    r = openai_client.images.generate(model=IMAGE_MODEL, prompt=prompt, n=1,
                                       size=size, quality=quality, style="vivid",
                                       response_format="b64_json")
    b64 = r.data[0].b64_json
    return f"data:image/png;base64,{b64}" if b64 else None

def save_image_versions(image_bytes: bytes, title: str) -> Dict[str,str]:
    folder = MEDIA_DIR / "visuals"; folder.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_filename(title)}_{uuid.uuid4().hex}"
    master = folder / f"{stem}_master.png"; master.write_bytes(image_bytes)
    preview = folder / f"{stem}_preview.jpg"; print_f = folder / f"{stem}_print.png"
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.resize((1280,720)).save(preview,"JPEG",quality=90)
        img.resize((3840,2160)).save(print_f,"PNG")
    except:
        preview.write_bytes(image_bytes); print_f.write_bytes(image_bytes)
    return {k: absolute_public_url(relative_public_url(p))
            for k,p in [("preview_url",preview),("master_url",master),("print_url",print_f)]}

def synthesize_speech(text: str, voice: Optional[str]=None, filename_prefix: str="tts") -> Dict[str,Any]:
    if not openai_client: raise HTTPException(500,"OpenAI not configured")
    path = VOICE_DIR / f"{filename_prefix}_{uuid.uuid4().hex}.mp3"
    openai_client.audio.speech.create(model=TTS_MODEL, voice=voice or TTS_VOICE, input=text).stream_to_file(str(path))
    rel = relative_public_url(path)
    return {"audio_url": absolute_public_url(rel), "audio_path": rel, "voice": voice or TTS_VOICE, "response_format":"mp3"}

def transcribe_audio_file(path: Path) -> str:
    if not openai_client: return ""
    with path.open("rb") as fh:
        return getattr(openai_client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=fh),"text","")

# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS  (all after imports — no more NameError)
# ══════════════════════════════════════════════════════════════════════════════
class UserInput(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)
    full_name: Optional[str] = Field(default=None, max_length=120)
    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        e = v.strip().lower()
        if not EMAIL_RE.match(e): raise ValueError("Invalid email")
        return e

class LoginInput(BaseModel):
    email: str
    password: str
    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str: return v.strip().lower()

class ProjectCreateInput(BaseModel):
    title: Optional[str] = None
    name:  Optional[str] = None
    brief: Optional[str] = None
    event_type:      Optional[str] = None
    style_direction: Optional[str] = None
    style_theme:     Optional[str] = None

class RunInput(BaseModel):
    text: str = Field(min_length=3)
    project_id:      Optional[str] = None
    name:            Optional[str] = None
    event_type:      Optional[str] = None
    style_direction: Optional[str] = None

class RunProjectInput(BaseModel):
    text:            Optional[str] = None
    name:            Optional[str] = None
    event_type:      Optional[str] = None
    style_direction: Optional[str] = None

class SelectConceptInput(BaseModel):
    project_id: str
    index: int = Field(ge=0, le=2)

class SelectConceptCompatInput(BaseModel):
    concept_index: Optional[int] = Field(default=None, ge=0, le=2)
    index:         Optional[int] = Field(default=None, ge=0, le=2)

class CommentInput(BaseModel):
    project_id:   str
    section:      str
    comment_text: str

class UpdateProjectInput(BaseModel):
    project_id: str
    field: str
    value: Any

class DepartmentPDFRequest(BaseModel):
    title: Optional[str] = None

class VoiceSessionCreateInput(BaseModel):
    project_id:    Optional[str] = None
    title:         Optional[str] = None
    system_prompt: Optional[str] = None
    voice:         Optional[str] = None

class VoiceTextInput(BaseModel):
    session_id:         Optional[str] = None
    project_id:         Optional[str] = None
    text: str = Field(min_length=1)
    voice:              Optional[str] = None
    voice_instructions: Optional[str] = None
    title:              Optional[str] = None
    system_prompt:      Optional[str] = None

class TTSInput(BaseModel):
    text: str = Field(min_length=1, max_length=4096)
    voice:        Optional[str] = None
    instructions: Optional[str] = None

class ControlActionInput(BaseModel):
    protocol: str
    target:   Optional[str] = None
    base_url: Optional[str] = None
    path:     Optional[str] = None
    method:   Optional[str] = None
    headers:  Optional[Dict[str,str]] = None
    params:   Optional[Dict[str,Any]] = None
    body:     Optional[Dict[str,Any]] = None
    address:  Optional[str] = None
    ip:       Optional[str] = None
    port:     Optional[int] = None
    args:     Optional[List[Any]] = None

class ArmInput(BaseModel):
    armed: bool = True

class CueJumpInput(BaseModel):
    cue_index: Optional[int] = Field(default=None, ge=0)
    cue_no:    Optional[int] = None

class VisualPolicyInput(BaseModel):
    preview_size: Optional[str] = None
    master_size:  Optional[str] = None
    print_size:   Optional[str] = None
    aspect_ratio: Optional[str] = None

class AssetCreateInput(BaseModel):
    asset_type:    str
    title:         str
    prompt: str = Field(min_length=3)
    section:       Optional[str] = None
    job_kind:      Optional[str] = None
    generate_now:  bool = True

class MoodboardGenerateInput(BaseModel):
    concept_index: Optional[int] = Field(default=None, ge=0, le=2)
    count: int = Field(default=3, ge=1, le=6)
    generate_now:  bool = True

class JobQueueInput(BaseModel):
    agent_type: str
    job_type:   str
    title:      Optional[str] = None
    priority:   int = Field(default=5, ge=1, le=10)
    input_data: Optional[Dict[str,Any]] = None

class OrchestrateInput(BaseModel):
    auto_generate_moodboard: bool = True
    queue_3d:      bool = True
    queue_video:   bool = True
    queue_cad:     bool = True
    queue_manuals: bool = True

class ElementSheetGenerateInput(BaseModel):
    include_sound:         bool = True
    include_lighting:      bool = True
    include_scenic:        bool = True
    include_power_summary: bool = True
    include_xlsx:          bool = True
    sheet_title:           Optional[str] = None

class ShowTrialGenerateInput(BaseModel):
    include_walkthrough:  bool = True
    include_audio_video:  bool = True
    include_camera_pan:   bool = True
    queue_render_jobs:    bool = True
    draft_name:           Optional[str] = None

class ShowTrialUpdateInput(BaseModel):
    trial_data: Dict[str,Any]

class ShowTrialFinalizeInput(BaseModel):
    use_trial_cues: bool = True
    mark_ready:     bool = True

class DXFRequest(BaseModel):
    project_id: str

class PDFRequest(BaseModel):
    project_id: str
    template:   Optional[str] = "executive"

class ImageGenRequest(BaseModel):
    prompt:     str
    size:       str = "1024x1024"
    quality:    str = "standard"
    project_id: Optional[str] = None
    section:    Optional[str] = None

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN LOGIC — Defaults
# ══════════════════════════════════════════════════════════════════════════════
def default_visual_policy() -> Dict[str,Any]:
    return {"aspect_ratio":VISUAL_ASPECT_RATIO,"preview_size":VISUAL_PREVIEW_SIZE,
            "master_size":VISUAL_MASTER_SIZE,"print_size":VISUAL_PRINT_SIZE,
            "preview_format":"jpg","master_format":"png","print_format":"png",
            "quality":IMAGE_QUALITY,"printable":True}

def _default_sound_plan(_: Dict) -> Dict[str,Any]:
    return {"system_design":{"console":"FOH digital console","speaker_system":"Line array PA","monitoring":"IEM/stage monitors"},
            "input_list":["MC mic","Playback stereo","Guest mic","Ambient mic"],
            "playback_cues":["opening stinger","walk-in bed","transition bed","finale"],
            "pdf_sections":[{"heading":"Sound Overview","body":"Planning-level sound system."},
                            {"heading":"Input List","body":"MC, playback, guest, ambient."},
                            {"heading":"Playback","body":"Timecoded or manual cue playback."}]}

def _default_lighting_plan(_: Dict) -> Dict[str,Any]:
    return {"fixture_list":["Moving Heads (Spot/Profile)","Wash Fixtures","LED Battens / Linears","Audience Blinders","Pinspots / Specials"],
            "scene_cues":["house-to-half","opening reveal","speaker special","award transition","finale"],
            "pdf_sections":[{"heading":"Lighting Overview","body":"Concept-driven lighting plan."},
                            {"heading":"Fixture Intent","body":"Moving, wash, linear, blinder, special layers."},
                            {"heading":"Cue Intent","body":"Opening, transitions, finale."}]}

def _default_showrunner_plan(_: Dict) -> Dict[str,Any]:
    return {"running_order":["Standby","House to half","Opening AV","MC welcome","Main beats","Finale"],
            "pdf_sections":[{"heading":"Show Running","body":"Cue-based show running script."}],
            "console_cues":[
                {"cue_no":1,"name":"Standby","cue_type":"standby","standby":"All departments standby","go":"Standby acknowledged","actions":[]},
                {"cue_no":2,"name":"House to Half","cue_type":"lighting","standby":"Lights standby","go":"Go house to half","actions":[{"protocol":"lighting","target":"house_lights","value":"half"}]},
                {"cue_no":3,"name":"Opening AV","cue_type":"av","standby":"AV standby opener","go":"Go opener","actions":[{"protocol":"av","target":"screen","value":"play_opener"}]},
                {"cue_no":4,"name":"MC Welcome","cue_type":"sound","standby":"Sound standby MC","go":"Go MC mic","actions":[{"protocol":"sound","target":"mc_mic","value":"on"}]},
            ]}

def generate_sound_department(project: Dict) -> Dict: return _default_sound_plan(project)
def generate_lighting_department(project: Dict) -> Dict: return _default_lighting_plan(project)
def generate_showrunner_department(project: Dict) -> Dict: return _default_showrunner_plan(project)

def build_scene_3d_json(project: Dict) -> Dict:
    sel = project.get("selected_concept") or project.get("selected") or {}
    return {"venue_type": project.get("event_type","event"), "concept_name": sel.get("name"),
            "stage":{"width":18000,"depth":9000,"height":1200},
            "screens":[{"name":"Center LED","width":8000,"height":4500}],
            "scenic_elements":[{"name":"Feature Arch","width":5000,"height":4200,"depth":600}],
            "cameras":[{"view":"hero","label":"Front Hero"},{"view":"wide","label":"Wide Venue"},{"view":"top","label":"Top View"}]}

def get_console_state(project: Dict) -> Dict:
    state = project.get("department_outputs") or {}
    if not isinstance(state, dict): state = {}
    state.setdefault("armed", False); state.setdefault("hold", False)
    state.setdefault("console_index", 0); state.setdefault("execution_log", [])
    return state

def log_console_event(state: Dict, event: Dict) -> Dict:
    log = list(state.get("execution_log") or [])
    log.append({"time": now_iso(), **event}); state["execution_log"] = log[-200:]
    state["last_status"] = event.get("status"); return state

def save_console_state(pid: str, uid: str, state: Dict) -> Dict:
    return db_update("projects", pid, {"department_outputs": state})

# ══════════════════════════════════════════════════════════════════════════════
# AI GENERATION
# ══════════════════════════════════════════════════════════════════════════════
EVENT_BUDGETS = {"conference":(800000,1800000,4200000),"award show":(1200000,2600000,6500000),
                 "brand launch":(900000,2200000,5500000),"wedding":(700000,1600000,4000000),
                 "concert":(1500000,3500000,9000000),"festival":(1200000,2800000,7200000),
                 "corporate":(800000,1700000,4500000),"generic":(500000,1200000,3000000)}

def infer_event_type(text: str, event_type: Optional[str]) -> str:
    if event_type: return event_type
    t = (text or "").lower()
    for n in EVENT_BUDGETS:
        if n != "generic" and n in t: return n
    if "launch" in t: return "brand launch"
    if "award" in t:  return "award show"
    return "generic"

def analyze_brief(brief: str, event_type: Optional[str]) -> Dict:
    inferred = infer_event_type(brief, event_type)
    fallback = {"summary":brief[:300],"event_type":inferred,
                "objectives":["Translate brief into presentation-ready concept",
                               "Build execution-friendly department outputs",
                               "Realistic planning-level cost estimate"],
                "audience":"Stakeholders, brand team, agencies, vendors",
                "risks":["Brief may need venue/timeline clarification","Budget needs vendor confirmation"],
                "assumptions":["Costing is planning-level","Design is concept-to-execution handoff"]}
    data = parse_json(llm("Senior experiential strategist. Return JSON only.",
        f'Analyze brief. Return JSON: summary, event_type, objectives(array), audience, risks(array), assumptions(array).\nBrief: {brief}\nEvent type: {event_type or "auto"}',
        json_mode=True), fallback)
    return {**fallback, **data} if isinstance(data, dict) else fallback

def generate_concepts(brief: str, analysis: Dict, event_type: Optional[str]) -> List[Dict]:
    inferred = infer_event_type(brief, event_type or analysis.get("event_type",""))
    budgets = EVENT_BUDGETS.get(inferred, EVENT_BUDGETS["generic"])
    fallback = [
        {"name":"Cinematic Signature","summary":f"Premium cinematic concept for {inferred}.",
         "style":"immersive premium","colors":["black","gold","warm white"],
         "materials":["mirror acrylic","fabric","metal"],"experience":"high emotional brand reveal",
         "key_zones":["arrival","main stage","screen content","audience","photo moment"],
         "estimated_budget_inr":{"low":budgets[0],"medium":budgets[1],"high":budgets[2]}},
        {"name":"Modern Tech Grid","summary":f"Futuristic tech concept for {inferred}.",
         "style":"futuristic sharp","colors":["midnight blue","cyan","silver"],
         "materials":["LED mesh","truss","glass acrylic"],"experience":"show-control-led visual language",
         "key_zones":["arrival","main stage","screen content","audience","photo moment"],
         "estimated_budget_inr":{"low":int(budgets[0]*1.2),"medium":int(budgets[1]*1.2),"high":int(budgets[2]*1.2)}},
        {"name":"Elegant Minimal Luxe","summary":f"Refined minimal concept for {inferred}.",
         "style":"clean editorial","colors":["ivory","champagne","graphite"],
         "materials":["textured scenic flats","wood veneer","soft fabric"],"experience":"refined storytelling",
         "key_zones":["arrival","main stage","screen content","audience","photo moment"],
         "estimated_budget_inr":{"low":int(budgets[0]*1.45),"medium":int(budgets[1]*1.45),"high":int(budgets[2]*1.45)}},
    ]
    data = parse_json(llm("Creative director for live events. Return JSON only.",
        f'Generate 3 concepts. Return {{"concepts":[{{"name":"...","summary":"...","style":"...","colors":[],"materials":[],"experience":"...","key_zones":[],"execution_highlights":[]}}]}}\nBrief: {brief}',
        json_mode=True))
    raw = data.get("concepts",[]) if isinstance(data,dict) else []
    result = []
    for i,fb in enumerate(fallback):
        c = dict(raw[i]) if i < len(raw) and isinstance(raw[i],dict) else {}
        result.append({**fb, **c, "estimated_budget_inr": fb["estimated_budget_inr"]})
    return result

def create_simple_pdf(title: str, sections: Any, filename_prefix: str) -> Dict[str,str]:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as rl_canvas
    except ImportError:
        fname = EXPORT_DIR / f"{filename_prefix}_{uuid.uuid4().hex}.txt"
        fname.write_text(f"{title}\n\n{str(sections)}", encoding="utf-8")
        rel = relative_public_url(fname)
        return {"pdf_path": rel, "pdf_url": absolute_public_url(rel)}

    filename = EXPORT_DIR / f"{filename_prefix}_{uuid.uuid4().hex}.pdf"
    c = rl_canvas.Canvas(str(filename), pagesize=A4)
    w, h = A4; left = 18*mm; y = h - 20*mm

    def new_page():
        nonlocal y; c.showPage(); y = h - 20*mm

    c.setFont("Helvetica-Bold", 18); c.drawString(left, y, title); y -= 12*mm
    secs = sections if isinstance(sections, list) else [{"heading":"Content","body":str(sections)}]
    for sec in secs:
        heading = str(sec.get("heading","") if isinstance(sec,dict) else sec)
        body    = str(sec.get("body","") if isinstance(sec,dict) else "")
        if y < 35*mm: new_page()
        c.setFont("Helvetica-Bold", 13); c.drawString(left, y, heading); y -= 8*mm
        c.setFont("Helvetica", 10)
        for para in body.split("\n"):
            words = para.split(); line = ""
            for word in words:
                test = (line + " " + word).strip()
                if c.stringWidth(test,"Helvetica",10) < (w - 2*left): line = test
                else:
                    if y < 20*mm: new_page(); c.setFont("Helvetica",10)
                    c.drawString(left, y, line); y -= 5*mm; line = word
            if line:
                if y < 20*mm: new_page(); c.setFont("Helvetica",10)
                c.drawString(left, y, line); y -= 5*mm
            y -= 2*mm
        y -= 4*mm
    c.save()
    rel = relative_public_url(filename)
    return {"pdf_path": rel, "pdf_url": absolute_public_url(rel)}

def execute_control_action(payload: Dict) -> Dict:
    protocol = (payload.get("protocol") or "").lower()
    if protocol == "http":
        import requests
        method = (payload.get("method") or "POST").upper()
        url = payload.get("address") or f"{(payload.get('base_url') or '').rstrip('/')}/{str(payload.get('path','') or '').lstrip('/')}"
        if not url: return {"ok":False,"protocol":"http","message":"Missing URL"}
        try:
            r = requests.request(method=method, url=url, headers=payload.get("headers") or {},
                                  params=payload.get("params") or {}, json=payload.get("body"), timeout=10)
            return {"ok":r.ok,"protocol":"http","status_code":r.status_code,"url":url,"response_text":r.text[:500]}
        except Exception as e:
            return {"ok":False,"protocol":"http","url":url,"message":str(e)}
    return {"ok":True,"protocol":protocol or "simulated","target":payload.get("target"),
            "message":"Simulated control action executed"}

def generate_element_sheet(project: Dict, include_sound=True, include_lighting=True,
                            include_scenic=True, include_power_summary=True) -> Dict:
    scene    = project.get("scene_json") or build_scene_3d_json(project)
    lighting = project.get("lighting_data") or _default_lighting_plan(project)
    sound    = project.get("sound_data")    or _default_sound_plan(project)
    rows = []
    if include_scenic:
        rows.append({"element_type":"scenic","name":"Main Stage Deck","qty":1,"unit":"set",
                     "width_mm":18000,"height_mm":1200,"depth_mm":9000,"watts_each":0,"total_watts":0,"notes":"Primary stage"})
        for scr in (scene.get("screens") or []):
            rows.append({"element_type":"led","name":scr.get("name","LED"),"qty":1,"unit":"pc",
                         "width_mm":scr.get("width",0),"height_mm":scr.get("height",0),"depth_mm":0,
                         "watts_each":1800,"total_watts":1800,"notes":"Planning estimate"})
    if include_lighting:
        for fix in (lighting.get("fixture_list") or []):
            rows.append({"element_type":"lighting","name":fix,"qty":6,"unit":"pc",
                         "width_mm":0,"height_mm":0,"depth_mm":0,"watts_each":350,"total_watts":2100,"notes":"Estimate"})
    if include_sound:
        rows.append({"element_type":"audio","name":"FOH Console","qty":1,"unit":"pc",
                     "width_mm":0,"height_mm":0,"depth_mm":0,"watts_each":350,"total_watts":350,
                     "notes":str((sound.get("system_design") or {}).get("console","FOH console"))})
    total_w = sum(float(r.get("total_watts",0)) for r in rows)
    sheet = {"project_id":str(project["id"]),"project_name":project.get("project_name") or project.get("name"),
             "generated_at":now_iso(),"rows":rows,
             "totals":{"element_count":len(rows),"total_watts":total_w,"total_kw":round(total_w/1000,3)}}
    if include_power_summary:
        sheet["power_summary"] = {"recommended_with_25pct_headroom_kw": round((total_w/1000)*1.25,3)}
    return sheet

def export_element_sheet_xlsx(sheet: Dict, prefix: str="element_sheet") -> Dict[str,str]:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        return {}
    wb = Workbook(); ws = wb.active; ws.title = "Element Sheet"
    headers = ["element_type","name","qty","unit","width_mm","height_mm","depth_mm","watts_each","total_watts","notes"]
    ws.append(headers)
    hdr_fill = PatternFill("solid",fgColor="1F4E78"); hdr_font = Font(color="FFFFFF",bold=True)
    for cell in ws[1]: cell.fill=hdr_fill; cell.font=hdr_font
    for row in sheet.get("rows",[]): ws.append([row.get(h) for h in headers])
    out = MEDIA_DIR / "spreadsheets"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"{prefix}_{uuid.uuid4().hex}.xlsx"; wb.save(path)
    rel = relative_public_url(path)
    return {"xlsx_url": absolute_public_url(rel), "xlsx_path": rel}

# ══════════════════════════════════════════════════════════════════════════════
# DXF EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def generate_dxf(layout: Dict, project_name: str="Event Layout") -> str:
    vw = float(layout.get("venue_width_m") or 40)
    vd = float(layout.get("venue_depth_m") or 30)
    zones = layout.get("zones") or []
    S = 1000  # mm per metre
    _h = [0]
    def nxt(): _h[0]+=1; return format(_h[0],'X')
    out: List[str] = []
    def g(code,val): out.append(f"{code:>3}"); out.append(str(val))

    ZONE_LAYERS = {"stage":("ZONE-STAGE",1),"seating":("ZONE-SEATING",3),"vip":("ZONE-VIP",4),
                   "circulation":("ZONE-CIRCULATION",8),"service":("ZONE-SERVICE",6),
                   "registration":("ZONE-REGISTRATION",5),"catering":("ZONE-CATERING",2),
                   "backstage":("ZONE-BACKSTAGE",6),"emergency":("ZONE-EMERGENCY",14)}

    # HEADER
    g(0,"SECTION"); g(2,"HEADER"); g(9,"$ACADVER"); g(1,"AC1024"); g(9,"$INSUNITS"); g(70,4)
    g(9,"$EXTMIN"); g(10,-2000); g(20,-2000); g(30,0)
    g(9,"$EXTMAX"); g(10,vw*S+6000); g(20,vd*S+6000); g(30,0); g(0,"ENDSEC")

    # TABLES
    g(0,"SECTION"); g(2,"TABLES")
    layers = [("0",7,"CONTINUOUS"),("VENUE-PERIMETER",7,"CONTINUOUS"),
              ("ZONE-STAGE",1,"CONTINUOUS"),("ZONE-SEATING",3,"CONTINUOUS"),
              ("ZONE-VIP",4,"CONTINUOUS"),("ZONE-CIRCULATION",8,"CONTINUOUS"),
              ("ZONE-SERVICE",6,"CONTINUOUS"),("ZONE-REGISTRATION",5,"CONTINUOUS"),
              ("ZONE-CATERING",2,"CONTINUOUS"),("ZONE-BACKSTAGE",6,"DASHED"),
              ("ZONE-EMERGENCY",14,"DASHED"),("WALLS",7,"CONTINUOUS"),
              ("DOORS",5,"CONTINUOUS"),("POWER-POINTS",1,"CONTINUOUS"),
              ("GRID",9,"DOTTED"),("DIMENSIONS",7,"CONTINUOUS"),
              ("TEXT-LABELS",7,"CONTINUOUS"),("TITLE-BLOCK",7,"CONTINUOUS")]
    g(0,"TABLE"); g(2,"LAYER"); g(5,nxt()); g(100,"AcDbSymbolTable"); g(70,len(layers))
    for ln,lc,lt in layers:
        g(0,"LAYER"); g(5,nxt()); g(100,"AcDbSymbolTableRecord"); g(100,"AcDbLayerTableRecord")
        g(2,ln); g(70,0); g(62,lc); g(6,lt)
    g(0,"ENDTAB"); g(0,"ENDSEC")

    # ENTITIES
    g(0,"SECTION"); g(2,"ENTITIES")

    def line(x1,y1,x2,y2,layer,color=7):
        g(0,"LINE"); g(5,nxt()); g(100,"AcDbEntity"); g(100,"AcDbLine")
        g(8,layer); g(62,color); g(10,x1); g(20,y1); g(30,0); g(11,x2); g(21,y2); g(31,0)

    def pline(pts,layer,color=7,closed=True):
        g(0,"LWPOLYLINE"); g(5,nxt()); g(100,"AcDbEntity"); g(100,"AcDbPolyline")
        g(8,layer); g(62,color); g(90,len(pts)); g(70,1 if closed else 0); g(43,0)
        for px,py in pts: g(10,px); g(20,py)

    def text(tx,ty,height,txt,layer,color=7,align=0):
        g(0,"TEXT"); g(5,nxt()); g(100,"AcDbEntity"); g(100,"AcDbText")
        g(8,layer); g(62,color); g(10,tx); g(20,ty); g(30,0)
        g(40,height); g(1,str(txt)[:80]); g(72,align)
        if align: g(11,tx); g(21,ty); g(31,0)

    # Venue perimeter
    pline([(0,0),(vw*S,0),(vw*S,vd*S),(0,vd*S)],"VENUE-PERIMETER",7)
    # Grid
    x=S
    while x<vw*S: line(x,0,x,vd*S,"GRID",9); x+=S
    y=S
    while y<vd*S: line(0,y,vw*S,y,"GRID",9); y+=S
    # Zones
    for z in zones:
        zt=(z.get("zone_type") or "").lower()
        layer,color=ZONE_LAYERS.get(zt,("ZONE-CIRCULATION",8))
        if z.get("x_m") is not None:
            x1=float(z["x_m"])*S; y1=float(z["y_m"])*S
            w=float(z.get("width_m") or z.get("width",5))*S
            h=float(z.get("depth_m") or z.get("height",5))*S
        else:
            x1=float(str(z.get("left_pct","5%")).rstrip("%"))/100*vw*S
            y1=float(str(z.get("top_pct","5%")).rstrip("%"))/100*vd*S
            w=float(str(z.get("width_pct","20%")).rstrip("%"))/100*vw*S
            h=float(str(z.get("height_pct","20%")).rstrip("%"))/100*vd*S
        x2,y2=x1+w,y1+h; cx,cy=x1+w/2,y1+h/2
        area=z.get("area_m2") or round(w/S*h/S,1)
        pline([(x1,y1),(x2,y1),(x2,y2),(x1,y2)],layer,color)
        th=max(100.0,min(300.0,w/8))
        text(cx,cy+th*.6,th,z.get("name","Zone").upper(),"TEXT-LABELS",7,1)
        text(cx,cy-th*.6,th*.5,f"{area:.0f} m²","TEXT-LABELS",9,1)
    # Walls
    for w in (layout.get("walls") or []):
        x1=float(w.get("x1_m",0))*S; y1=float(w.get("y1_m",0))*S
        x2=float(w.get("x2_m",0))*S; y2=float(w.get("y2_m",0))*S
        line(x1,y1,x2,y2,"WALLS",7)
    # Dimension annotations
    off=800.0
    line(0,-off,vw*S,-off,"DIMENSIONS",7)
    text(vw*S/2,-off-250,200,f"WIDTH: {vw:.1f}m","DIMENSIONS",7,1)
    line(-off,0,-off,vd*S,"DIMENSIONS",7)
    text(-off-300,vd*S/2,200,f"DEPTH: {vd:.1f}m","DIMENSIONS",7,1)
    text(vw*S/2,-off-550,150,f"TOTAL AREA: {layout.get('total_area','')}  |  CAPACITY: {layout.get('capacity','')}","DIMENSIONS",7,1)
    # North arrow
    na_x,na_y=vw*S+1200,vd*S-800
    line(na_x,na_y-600,na_x,na_y+600,"TITLE-BLOCK",7)
    text(na_x,na_y+700,180,"N","TITLE-BLOCK",7,1)
    # Title block
    tb_x,tb_y,tb_w,tb_h=vw*S+800,0,5000,3000
    pline([(tb_x,tb_y),(tb_x+tb_w,tb_y),(tb_x+tb_w,tb_y+tb_h),(tb_x,tb_y+tb_h)],"TITLE-BLOCK",7,False)
    for tx,ty,th,txt in [
        (tb_x+200,tb_y+2600,220,project_name.upper()[:40]),
        (tb_x+200,tb_y+2000,140,f"{layout.get('dimensions','')} | Ceiling: {layout.get('ceiling_height','')}"),
        (tb_x+200,tb_y+1400,120,f"Area: {layout.get('total_area','')} | Capacity: {layout.get('capacity','')}"),
        (tb_x+200,tb_y+900,100,"Scale 1:100 | Units: mm | Grid: 1m"),
        (tb_x+200,tb_y+400,90,"AICreative Studio | Production Drawing"),
        (tb_x+200,tb_y+200,80,f"Date: {datetime.utcnow().strftime('%Y-%m-%d')} | DXF R2010"),
    ]: text(tx,ty,th,txt,"TITLE-BLOCK",7)

    g(0,"ENDSEC"); g(0,"EOF")
    return "\n".join(out)

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root(): return {"message":f"{APP_NAME} is running","time":now_iso(),"docs":"/docs"}

@app.get("/health")
def health():
    return {"status":"ok","time":now_iso(),"openai":bool(openai_client),
            "supabase":bool(_sb),"cors_origins":ALLOWED_ORIGINS,
            "model":OPENAI_MODEL,"image_model":IMAGE_MODEL}

# ── Auth ───────────────────────────────────────────────────────────────────────
@app.post("/signup")
def signup(payload: UserInput):
    if _sb:
        ex = _sb.table("users").select("id").eq("email",payload.email).execute()
        if ex and ex.data: raise HTTPException(400,"Email already registered")
    uid = str(uuid.uuid4())
    user = db_insert("users",{"id":uid,"email":payload.email,"password":hash_password(payload.password),
                               "full_name":payload.full_name or payload.email.split("@")[0],
                               "plan":"studio_pro","projects_used":0,"projects_limit":100})
    token = create_access_token(uid)
    safe = {k:v for k,v in user.items() if k!="password"}
    return {"message":"User created","user_id":uid,"access_token":token,"token":token,
            "token_type":"bearer","user":safe}

@app.post("/login")
def login(payload: LoginInput):
    user = db_get("users", email=payload.email)
    if not user: raise HTTPException(400,"User not found")
    pw_hash = user.get("password") or user.get("password_hash","")
    if not verify_password(payload.password, pw_hash): raise HTTPException(400,"Wrong password")
    token = create_access_token(str(user["id"]))
    safe = {k:v for k,v in user.items() if k not in ("password","password_hash")}
    return {"access_token":token,"token":token,"token_type":"bearer","user_id":str(user["id"]),"user":safe}

@app.post("/logout")
def logout(_=Depends(get_current_user)): return {"message":"Logged out. Remove bearer token on client."}

@app.get("/me")
def me(u=Depends(get_current_user)):
    return {"user":{k:v for k,v in u.items() if k not in ("password","password_hash")}}

# ── Projects ───────────────────────────────────────────────────────────────────
@app.get("/projects")
def list_projects(u=Depends(get_current_user)):
    return {"projects": db_list("projects", user_id=str(u["id"]))}

@app.post("/projects")
def create_project(p: ProjectCreateInput, u=Depends(get_current_user)):
    name = (p.title or p.name or "Untitled Project").strip()
    row = db_insert("projects",{"id":str(uuid.uuid4()),"user_id":str(u["id"]),
                                 "project_name":name,"brief_text":p.brief,
                                 "event_type":p.event_type,"style_direction":p.style_direction,
                                 "style_theme":p.style_theme or "luxury","status":"draft"})
    return row

@app.get("/projects/{pid}")
@app.get("/project/{pid}")
def get_project(pid: str, u=Depends(get_current_user)):
    proj = db_get("projects", id=pid, user_id=str(u["id"]))
    if not proj: raise HTTPException(404,"Project not found")
    return {"project": proj}

@app.post("/project/update")
def update_project(p: UpdateProjectInput, u=Depends(get_current_user)):
    proj = db_update("projects", p.project_id, {p.field: p.value})
    return {"message":"Updated","project":proj}

# ── Pipeline ───────────────────────────────────────────────────────────────────
def _run_logic(project: Dict, text: str, event_type: Optional[str], uid: str) -> Dict:
    pid = str(project["id"])
    updates: Dict[str,Any] = {}
    if text and project.get("brief_text","") != text: updates["brief_text"] = text
    if event_type and not project.get("event_type"): updates["event_type"] = event_type
    if updates: project = db_update("projects", pid, updates)
    analysis = project.get("analysis") or {}
    if not analysis:
        analysis = analyze_brief(project.get("brief_text") or text, project.get("event_type") or event_type)
        project = db_update("projects", pid, {"analysis": analysis})
    concepts_raw = project.get("concepts")
    concepts = load_json(concepts_raw) if isinstance(concepts_raw,str) else (concepts_raw or [])
    if not concepts:
        concepts = generate_concepts(project.get("brief_text") or text, analysis, project.get("event_type") or event_type)
        project = db_update("projects", pid, {"concepts": concepts, "status":"concepts_ready"})
    return {"message":"Pipeline completed","project_id":pid,"status":"concepts_ready",
            "brief":project.get("brief_text"),"analysis":analysis,"concepts":concepts,"project":project}

@app.post("/run")
def run_pipeline(p: RunInput, u=Depends(get_current_user)):
    uid = str(u["id"])
    if p.project_id:
        proj = db_get("projects", id=p.project_id, user_id=uid)
        if not proj: raise HTTPException(404,"Project not found")
    else:
        proj = db_insert("projects",{"id":str(uuid.uuid4()),"user_id":uid,
                                      "project_name":(p.name or p.text[:50]).strip(),
                                      "brief_text":p.text,"event_type":p.event_type,
                                      "style_direction":p.style_direction,"status":"draft"})
    return _run_logic(proj, p.text, p.event_type, uid)

@app.post("/projects/{pid}/run")
def run_project(pid: str, p: RunProjectInput, u=Depends(get_current_user)):
    uid = str(u["id"])
    proj = db_get("projects", id=pid, user_id=uid)
    if not proj: raise HTTPException(404,"Project not found")
    text = (p.text or proj.get("brief_text","")).strip()
    if not text: raise HTTPException(422,"text is required")
    return _run_logic(proj, text, p.event_type or proj.get("event_type"), uid)

@app.post("/select")
def select_concept(p: SelectConceptInput, u=Depends(get_current_user)):
    uid = str(u["id"])
    proj = db_get("projects", id=p.project_id, user_id=uid)
    if not proj: raise HTTPException(404,"Project not found")
    raw = proj.get("concepts") or "[]"
    concepts = load_json(raw) if isinstance(raw,str) else (raw or [])
    if not concepts: raise HTTPException(400,"Run pipeline first")
    if p.index >= len(concepts): raise HTTPException(400,f"Only {len(concepts)} concepts available")
    selected = concepts[p.index]
    proj = db_update("projects", p.project_id, {"selected_concept":selected,"status":"concept_selected"})
    return {"message":"Concept selected","index":p.index,"selected":selected,"project":proj}

@app.post("/projects/{pid}/select-concept")
def select_concept_alias(pid: str, p: SelectConceptCompatInput, u=Depends(get_current_user)):
    idx = p.concept_index if p.concept_index is not None else p.index
    if idx is None: raise HTTPException(422,"concept_index required")
    return select_concept(SelectConceptInput(project_id=pid, index=idx), u)

# ── Comments ───────────────────────────────────────────────────────────────────
@app.post("/comment")
@app.post("/comments")
def add_comment(p: CommentInput, u=Depends(get_current_user)):
    row = db_insert("project_comments",{"id":str(uuid.uuid4()),"project_id":p.project_id,
                                         "user_id":str(u["id"]),"section":p.section,
                                         "content":p.comment_text,"author_name":u.get("full_name","Anonymous")})
    return {"message":"Comment added","comment":row}

@app.get("/comments/{pid}")
def list_comments(pid: str, u=Depends(get_current_user)):
    return {"comments": db_list("project_comments", project_id=pid)}

# ── Departments ────────────────────────────────────────────────────────────────
def _build_departments(pid: str, uid: str) -> Dict:
    proj = db_get("projects", id=pid, user_id=uid)
    if not proj: raise HTTPException(404,"Project not found")
    if not (proj.get("selected_concept") or proj.get("selected")): raise HTTPException(400,"Select a concept first")
    sound = generate_sound_department(proj)
    lighting = generate_lighting_department(proj)
    showrunner = generate_showrunner_department(proj)
    state = get_console_state(proj)
    state.update({"sound_ready":True,"lighting_ready":True,"showrunner_ready":True,"console_index":0,"hold":False})
    proj = db_update("projects", pid, {"sound_data":sound,"lighting_data":lighting,"showrunner_data":showrunner,
                                        "department_outputs":state,"scene_json":build_scene_3d_json(proj),"status":"departments_ready"})
    return {"message":"Departments generated","project_id":pid,"sound_data":sound,
            "lighting_data":lighting,"showrunner_data":showrunner,"department_outputs":state,"project":proj}

@app.post("/project/{pid}/departments/build")
@app.post("/projects/{pid}/generate-departments")
def build_departments(pid: str, u=Depends(get_current_user)):
    return _build_departments(pid, str(u["id"]))

@app.get("/project/{pid}/departments/manuals")
def dept_manuals(pid: str, u=Depends(get_current_user)):
    proj = db_get("projects", id=pid, user_id=str(u["id"]))
    if not proj: raise HTTPException(404,"Project not found")
    return {"project_id":pid,"sound_data":proj.get("sound_data"),
            "lighting_data":proj.get("lighting_data"),"showrunner_data":proj.get("showrunner_data")}

@app.post("/project/{pid}/departments/pdf/sound")
def pdf_sound(pid: str, p: DepartmentPDFRequest, u=Depends(get_current_user)):
    proj = db_get("projects", id=pid, user_id=str(u["id"]))
    if not proj or not proj.get("sound_data"): raise HTTPException(404,"Sound data not found")
    sd = proj["sound_data"]; secs = sd.get("pdf_sections") or sd if isinstance(sd,dict) else sd
    return {"project_id":pid, **create_simple_pdf(p.title or "Sound Design Manual", secs, "sound_manual")}

@app.post("/project/{pid}/departments/pdf/lighting")
def pdf_lighting(pid: str, p: DepartmentPDFRequest, u=Depends(get_current_user)):
    proj = db_get("projects", id=pid, user_id=str(u["id"]))
    if not proj or not proj.get("lighting_data"): raise HTTPException(404,"Lighting data not found")
    ld = proj["lighting_data"]; secs = ld.get("pdf_sections") or ld if isinstance(ld,dict) else ld
    return {"project_id":pid, **create_simple_pdf(p.title or "Lighting Design Manual", secs, "lighting_manual")}

@app.post("/project/{pid}/departments/pdf/showrunner")
def pdf_showrunner(pid: str, p: DepartmentPDFRequest, u=Depends(get_current_user)):
    proj = db_get("projects", id=pid, user_id=str(u["id"]))
    if not proj or not proj.get("showrunner_data"): raise HTTPException(404,"Showrunner data not found")
    sd = proj["showrunner_data"]; secs = sd.get("pdf_sections") or sd if isinstance(sd,dict) else sd
    return {"project_id":pid, **create_simple_pdf(p.title or "Show Running Script", secs, "showrunner_manual")}

# ── Show Console ───────────────────────────────────────────────────────────────
@app.get("/project/{pid}/show-console")
def console_status(pid: str, u=Depends(get_current_user)):
    proj = db_get("projects", id=pid, user_id=str(u["id"]))
    if not proj: raise HTTPException(404,"Project not found")
    sd = proj.get("showrunner_data") or {}; cues = sd.get("console_cues") or []
    state = get_console_state(proj); idx = min(int(state.get("console_index",0)), max(len(cues)-1,0)) if cues else 0
    return {"project_id":pid,"armed":bool(state.get("armed")),"hold":bool(state.get("hold")),
            "cue_index":idx,"cue":cues[idx] if cues else None,
            "next_cue":cues[idx+1] if cues and idx+1<len(cues) else None,
            "available_cues":cues,"department_outputs":state}

@app.post("/project/{pid}/show-console/arm")
def console_arm(pid: str, p: ArmInput, u=Depends(get_current_user)):
    proj = db_get("projects", id=pid, user_id=str(u["id"]))
    if not proj: raise HTTPException(404,"Project not found")
    state = get_console_state(proj); state["armed"] = bool(p.armed)
    state = log_console_event(state,{"status":"armed" if p.armed else "disarmed"})
    save_console_state(pid, str(u["id"]), state)
    return {"message":"Console updated","armed":state["armed"]}

@app.post("/project/{pid}/show-console/go")
def console_go(pid: str, execute: bool=Query(True), u=Depends(get_current_user)):
    uid = str(u["id"])
    proj = db_get("projects", id=pid, user_id=uid)
    if not proj: raise HTTPException(404,"Project not found")
    sd = proj.get("showrunner_data") or {}; cues = sd.get("console_cues") or []
    if not cues: raise HTTPException(400,"No cues found")
    state = get_console_state(proj)
    if not state.get("armed"): raise HTTPException(400,"Console not armed")
    if state.get("hold"): raise HTTPException(400,"Console is on hold")
    idx = min(int(state.get("console_index",0)), len(cues)-1)
    cue = cues[idx]; results=[]
    if execute: results=[execute_control_action(a) for a in cue.get("actions",[])]
    state["console_index"] = min(idx+1, len(cues)-1)
    state = log_console_event(state,{"status":"go","cue_index":idx,"cue":cue})
    save_console_state(pid, uid, state)
    return {"message":"Cue executed","cue_index":idx,"cue":cue,"results":results}

@app.post("/project/{pid}/show-console/next")
def console_next(pid: str, u=Depends(get_current_user)):
    uid=str(u["id"]); proj=db_get("projects",id=pid,user_id=uid)
    if not proj: raise HTTPException(404,"Project not found")
    cues=(proj.get("showrunner_data") or {}).get("console_cues") or []
    if not cues: raise HTTPException(400,"No cues")
    state=get_console_state(proj); idx=min(int(state.get("console_index",0))+1,len(cues)-1)
    state["console_index"]=idx; state=log_console_event(state,{"status":"next","cue_index":idx})
    save_console_state(pid,uid,state)
    return {"message":"Next cue","cue_index":idx,"cue":cues[idx]}

@app.post("/project/{pid}/show-console/back")
def console_back(pid: str, u=Depends(get_current_user)):
    uid=str(u["id"]); proj=db_get("projects",id=pid,user_id=uid)
    if not proj: raise HTTPException(404,"Project not found")
    cues=(proj.get("showrunner_data") or {}).get("console_cues") or []
    state=get_console_state(proj); idx=max(int(state.get("console_index",0))-1,0)
    state["console_index"]=idx; state=log_console_event(state,{"status":"back","cue_index":idx})
    save_console_state(pid,uid,state)
    return {"message":"Previous cue","cue_index":idx,"cue":cues[idx] if cues else None}

@app.post("/project/{pid}/show-console/hold")
def console_hold(pid: str, u=Depends(get_current_user)):
    uid=str(u["id"]); proj=db_get("projects",id=pid,user_id=uid)
    if not proj: raise HTTPException(404,"Project not found")
    state=get_console_state(proj); state["hold"]=True
    state=log_console_event(state,{"status":"hold"}); save_console_state(pid,uid,state)
    return {"message":"Hold engaged"}

@app.post("/project/{pid}/show-console/standby")
def console_standby(pid: str, u=Depends(get_current_user)):
    uid=str(u["id"]); proj=db_get("projects",id=pid,user_id=uid)
    if not proj: raise HTTPException(404,"Project not found")
    state=get_console_state(proj); state["hold"]=False
    state=log_console_event(state,{"status":"standby"}); save_console_state(pid,uid,state)
    return {"message":"Standby"}

@app.post("/project/{pid}/show-console/panic")
def console_panic(pid: str, u=Depends(get_current_user)):
    uid=str(u["id"]); proj=db_get("projects",id=pid,user_id=uid)
    if not proj: raise HTTPException(404,"Project not found")
    state=get_console_state(proj); state["hold"]=True; state["armed"]=False
    state=log_console_event(state,{"status":"panic"}); save_console_state(pid,uid,state)
    return {"message":"Panic — console disarmed and hold engaged"}

@app.post("/project/{pid}/show-console/jump")
def console_jump(pid: str, p: CueJumpInput, u=Depends(get_current_user)):
    uid=str(u["id"]); proj=db_get("projects",id=pid,user_id=uid)
    if not proj: raise HTTPException(404,"Project not found")
    cues=(proj.get("showrunner_data") or {}).get("console_cues") or []
    if not cues: raise HTTPException(400,"No cues")
    idx=p.cue_index
    if p.cue_no is not None:
        matches=[i for i,c in enumerate(cues) if str(c.get("cue_no"))==str(p.cue_no)]
        if not matches: raise HTTPException(404,"Cue not found")
        idx=matches[0]
    if idx is None: raise HTTPException(422,"cue_index or cue_no required")
    if idx<0 or idx>=len(cues): raise HTTPException(400,"Index out of range")
    state=get_console_state(proj); state["console_index"]=idx
    state=log_console_event(state,{"status":"jump","cue_index":idx}); save_console_state(pid,uid,state)
    return {"message":"Jumped","cue_index":idx,"cue":cues[idx]}

@app.get("/project/{pid}/show-console/history")
def console_history(pid: str, u=Depends(get_current_user)):
    proj=db_get("projects",id=pid,user_id=str(u["id"]))
    if not proj: raise HTTPException(404,"Project not found")
    return {"execution_log": get_console_state(proj).get("execution_log") or []}

@app.post("/control/execute")
def ctrl_execute(p: ControlActionInput, _=Depends(get_current_user)):
    return {"message":"Action executed","result": execute_control_action(p.model_dump(exclude_none=True))}

# ── Assets / Images ────────────────────────────────────────────────────────────
@app.get("/projects/{pid}/assets")
def list_assets(pid: str, section: Optional[str]=Query(None), u=Depends(get_current_user)):
    proj=db_get("projects",id=pid,user_id=str(u["id"]))
    if not proj: raise HTTPException(404,"Project not found")
    assets=db_list("project_assets",project_id=pid)
    if section: assets=[a for a in assets if a.get("section")==section]
    return {"assets":assets}

@app.post("/generate/image")
@app.post("/projects/{pid}/assets/generate-image")
def gen_image_endpoint(p: ImageGenRequest, u=Depends(get_current_user)):
    url = generate_image_b64(p.prompt, p.size, p.quality)
    if not url: raise HTTPException(500,"Image generation failed")
    if p.project_id and _sb:
        db_insert("project_assets",{"id":str(uuid.uuid4()),"project_id":p.project_id,
                                     "user_id":str(u["id"]),"asset_type":"generated_image",
                                     "title":p.prompt[:60],"section":p.section or "general","status":"completed"})
    return {"dataUrl":url,"size":p.size,"quality":p.quality}

# ── Export ─────────────────────────────────────────────────────────────────────
@app.post("/export/dxf")
def export_dxf(p: DXFRequest, u=Depends(get_current_user)):
    proj=db_get("projects",id=p.project_id,user_id=str(u["id"]))
    if not proj: raise HTTPException(404,"Project not found")
    rows=db_list("cad_layouts",project_id=p.project_id,limit=1)
    if not rows: raise HTTPException(400,"Generate CAD layout first")
    layout=rows[0]; pname=proj.get("project_name") or proj.get("name","Event")
    dxf=generate_dxf(layout,pname)
    fname=f"{safe_filename(pname)}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.dxf"
    (EXPORT_DIR/fname).write_text(dxf,encoding="utf-8")
    return {"filename":fname,"download_url":f"/exports/{fname}","size_bytes":len(dxf.encode()),"message":"DXF ready"}

@app.post("/export/pdf")
def export_pdf_endpoint(p: PDFRequest, u=Depends(get_current_user)):
    proj=db_get("projects",id=p.project_id,user_id=str(u["id"]))
    if not proj: raise HTTPException(404,"Project not found")
    pname=proj.get("project_name") or proj.get("name","Project")
    secs=[{"heading":"Project Overview","body":f"Name: {pname}\nStatus: {proj.get('status','draft')}\nEvent: {proj.get('event_type','')}\nStyle: {proj.get('style_direction','')}"}]
    if proj.get("analysis"):
        a=proj["analysis"]; body=a.get("summary","") if isinstance(a,dict) else str(a)
        secs.append({"heading":"Brief Analysis","body":body})
    raw=proj.get("concepts")
    concepts=load_json(raw) if isinstance(raw,str) else (raw or [])
    if concepts:
        secs.append({"heading":"Creative Concepts","body":"\n\n".join([f"{c.get('name','')}: {c.get('summary','')}" for c in concepts])})
    fname=f"{safe_filename(pname)}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.pdf"
    result=create_simple_pdf(pname, secs, safe_filename(pname))
    return {"filename":fname,**result,"template":p.template}

@app.get("/exports/{fname}")
def dl_export(fname: str):
    path=EXPORT_DIR/fname
    if not path.exists(): raise HTTPException(404,"File not found")
    mt="application/dxf" if fname.endswith(".dxf") else "application/pdf"
    return FileResponse(path,media_type=mt,filename=fname)

# ── Voice ──────────────────────────────────────────────────────────────────────
@app.post("/voice/tts")
@app.post("/tts")
def tts(p: TTSInput, _=Depends(get_current_user)):
    audio=synthesize_speech(p.text,voice=p.voice,filename_prefix="tts")
    return {"message":"Audio generated","text":p.text,**audio,"disclosure":"AI-generated voice"}

@app.post("/voice/transcribe")
async def transcribe(audio_file: UploadFile=File(...), _=Depends(get_current_user)):
    suffix=Path(audio_file.filename or "audio.webm").suffix or ".webm"
    path=UPLOAD_DIR/f"transcribe_{uuid.uuid4().hex}{suffix}"
    content=await audio_file.read()
    if not content: raise HTTPException(400,"Empty file")
    path.write_bytes(content)
    return {"transcript": transcribe_audio_file(path)}

# ── Element sheet ──────────────────────────────────────────────────────────────
@app.post("/projects/{pid}/element-sheet/generate")
def gen_element_sheet(pid: str, p: ElementSheetGenerateInput, u=Depends(get_current_user)):
    uid=str(u["id"]); proj=db_get("projects",id=pid,user_id=uid)
    if not proj: raise HTTPException(404,"Project not found")
    sheet=generate_element_sheet(proj,p.include_sound,p.include_lighting,p.include_scenic,p.include_power_summary)
    xlsx={}
    if p.include_xlsx: xlsx=export_element_sheet_xlsx(sheet,prefix=safe_filename(p.sheet_title or f"{proj.get('project_name','project')}_elements"))
    sheet.update(xlsx)
    db_update("projects",pid,{"element_sheet":sheet})
    return {"message":"Element sheet generated","element_sheet":sheet,**xlsx}

@app.get("/projects/{pid}/element-sheet")
def get_element_sheet(pid: str, u=Depends(get_current_user)):
    proj=db_get("projects",id=pid,user_id=str(u["id"]))
    if not proj: raise HTTPException(404,"Project not found")
    return {"element_sheet": proj.get("element_sheet")}

# ── Master manual PDF ──────────────────────────────────────────────────────────
@app.post("/projects/{pid}/manuals/master/pdf")
def master_manual_pdf(pid: str, u=Depends(get_current_user)):
    proj=db_get("projects",id=pid,user_id=str(u["id"]))
    if not proj: raise HTTPException(404,"Project not found")
    secs=[{"heading":"Overview","body":f"Project: {proj.get('project_name')}\nStatus: {proj.get('status')}"},
          {"heading":"Brief","body":proj.get("brief_text") or proj.get("brief","")},
          {"heading":"Analysis","body":dump_json(proj.get("analysis") or {})}]
    return create_simple_pdf(f"Master Manual — {proj.get('project_name','Project')}", secs, "master_manual")

# ── Visual policy ──────────────────────────────────────────────────────────────
@app.get("/projects/{pid}/visual-policy")
def get_visual_policy(pid: str, u=Depends(get_current_user)):
    proj=db_get("projects",id=pid,user_id=str(u["id"]))
    if not proj: raise HTTPException(404,"Project not found")
    return {"project_id":pid,"visual_policy":default_visual_policy()}

@app.post("/projects/{pid}/visual-policy")
def set_visual_policy(pid: str, p: VisualPolicyInput, u=Depends(get_current_user)):
    proj=db_get("projects",id=pid,user_id=str(u["id"]))
    if not proj: raise HTTPException(404,"Project not found")
    pol=default_visual_policy()
    if p.preview_size: pol["preview_size"]=p.preview_size
    if p.master_size:  pol["master_size"]=p.master_size
    if p.print_size:   pol["print_size"]=p.print_size
    if p.aspect_ratio: pol["aspect_ratio"]=p.aspect_ratio
    db_update("projects",pid,{"visual_policy":pol})
    return {"visual_policy":pol}

# ── Orchestrate ────────────────────────────────────────────────────────────────
@app.post("/projects/{pid}/orchestrate")
def orchestrate(pid: str, p: OrchestrateInput, u=Depends(get_current_user)):
    uid=str(u["id"]); proj=db_get("projects",id=pid,user_id=uid)
    if not proj: raise HTTPException(404,"Project not found")
    summary={"queued_at":now_iso(),"queue_3d":p.queue_3d,"queue_video":p.queue_video,
             "queue_cad":p.queue_cad,"queue_manuals":p.queue_manuals}
    db_update("projects",pid,{"orchestration_data":summary})
    return {"message":"Orchestration updated","orchestration":summary}

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False, workers=1)
