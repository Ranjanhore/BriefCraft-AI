"""
BriefCraft AI backend
"""
from **future** import annotations
import base64, io, json, os, re, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── dotenv (optional) ──────────────────────────────────────────────────────────

try:
from dotenv import load_dotenv
load_dotenv()
except ImportError:
pass

print("BUILD: APR-25-V4-FINAL - starting up")

# ── Config ─────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY", "")).strip()
SECRET_KEY = (os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY", "change-me-32char-secret-key-xx")).strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "dall-e-3").strip()
IMAGE_QUALITY = os.getenv("IMAGE_QUALITY", "standard").strip()
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1").strip()
TTS_VOICE = os.getenv("TTS_VOICE", "alloy").strip()
TRANSCRIBE_MDL = os.getenv("TRANSCRIBE_MODEL", "whisper-1").strip()
PORT = int(os.getenv("PORT", "10000"))
ALGORITHM = "HS256"
TOKEN_HOURS = 72
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = (BASE_DIR / "exports").resolve()
UPLOAD_DIR = (BASE_DIR / "uploads").resolve()
MEDIA_DIR = (BASE_DIR / "media").resolve()
RENDER_DIR = (BASE_DIR / "renders").resolve()
VOICE_DIR = MEDIA_DIR / "voice"

for _d in [EXPORT_DIR, UPLOAD_DIR, MEDIA_DIR, RENDER_DIR, VOICE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
ALLOWED_ORIGINS = [
“http://localhost:3000”, “http://127.0.0.1:3000”,
“http://localhost:5173”, “http://127.0.0.1:5173”,
“https://briefly-sparkle.lovable.app”,
“https://aicreative.studio”,
]
for _o in os.getenv(“ALLOWED_ORIGINS”, “”).split(”,”):
_o = _o.strip()
if _o and _o not in ALLOWED_ORIGINS:
ALLOWED_ORIGINS.append(_o)

# ── Optional heavy imports ──────────────────────────────────────────────────────

try:
from openai import OpenAI as _OpenAIClass
_openai_client = _OpenAIClass(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except ImportError:
_openai_client = None

try:
from supabase import create_client as _sb_create
_sb = _sb_create(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
except ImportError:
_sb = None

try:
from jose import JWTError, jwt as _jwt
_JWT_OK = True
except ImportError:
_JWT_OK = False
class JWTError(Exception): pass

try:
from passlib.context import CryptContext
_pwd = CryptContext(schemes=[“bcrypt”], deprecated=“auto”)
_PWD_OK = True
except ImportError:
_pwd = None
_PWD_OK = False

# ── FastAPI (required — in requirements.txt) ────────────────────────────────────

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# ══════════════════════════════════════════════════════════════════════════════

# APP

# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title=“AICreative Studio API”, version=“4.0.0”)

app.add_middleware(
CORSMiddleware,
allow_origins=ALLOWED_ORIGINS,
allow_origin_regex=r”^https?://([a-zA-Z0-9-]+.)?lovable.(app|dev)$”,
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

for _mount, _dir, _name in [(”/media”, MEDIA_DIR, “media”),
(”/uploads”, UPLOAD_DIR, “uploads”),
(”/exports”, EXPORT_DIR, “exports”),
(”/renders”, RENDER_DIR, “renders”)]:
app.mount(_mount, StaticFiles(directory=str(_dir)), name=_name)

@app.exception_handler(Exception)
async def *exc_handler(*, e: Exception):
if isinstance(e, HTTPException):
return JSONResponse(status_code=e.status_code, content={“detail”: e.detail})
return JSONResponse(status_code=500, content={“detail”: str(e)})

# ══════════════════════════════════════════════════════════════════════════════

# AUTH HELPERS

# ══════════════════════════════════════════════════════════════════════════════

def _hash_pw(pw: str) -> str:
if _PWD_OK: return _pwd.hash(pw)
import hashlib
return hashlib.sha256((pw + SECRET_KEY).encode()).hexdigest()

def _verify_pw(pw: str, hashed: str) -> bool:
if _PWD_OK: return _pwd.verify(pw, hashed)
import hashlib
return hashlib.sha256((pw + SECRET_KEY).encode()).hexdigest() == hashed

def _make_token(uid: str) -> str:
exp = datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
payload = {“sub”: uid, “exp”: exp}
if _JWT_OK: return _jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
import base64 as _b64
return _b64.urlsafe_b64encode(json.dumps({“sub”: uid, “exp”: exp.timestamp()}).encode()).decode()

def _decode_token(tok: str) -> str:
if _JWT_OK:
try: return _jwt.decode(tok, SECRET_KEY, algorithms=[ALGORITHM])[“sub”]
except: raise HTTPException(401, “Invalid or expired token”)
try:
import base64 as _b64
data = json.loads(_b64.urlsafe_b64decode(tok.encode() + b”==”))
if data.get(“exp”, 0) < datetime.now(timezone.utc).timestamp():
raise HTTPException(401, “Token expired”)
return data[“sub”]
except HTTPException: raise
except: raise HTTPException(401, “Invalid token”)

_bearer = HTTPBearer(auto_error=False)

def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> Dict[str, Any]:
if not creds or not creds.credentials:
raise HTTPException(401, “Authorization header required”)
uid = _decode_token(creds.credentials)
if _sb:
r = _sb.table(“users”).select(”*”).eq(“id”, uid).maybe_single().execute()
if r and r.data: return dict(r.data)
return {“id”: uid, “email”: “user@aicreative.studio”, “full_name”: “User”}

# ══════════════════════════════════════════════════════════════════════════════

# DB HELPERS

# ══════════════════════════════════════════════════════════════════════════════

def db_get(table: str, **kw) -> Optional[Dict]:
if not _sb: return None
q = _sb.table(table).select(”*”)
for k, v in kw.items(): q = q.eq(k, v)
r = q.maybe_single().execute()
return r.data if (r and r.data) else None

def db_list(table: str, order=“created_at”, desc=True, limit=100, **kw) -> List[Dict]:
if not _sb: return []
q = _sb.table(table).select(”*”).order(order, desc=desc).limit(limit)
for k, v in kw.items(): q = q.eq(k, v)
r = q.execute()
return r.data or []

def db_insert(table: str, data: Dict) -> Dict:
if not _sb: return data
r = _sb.table(table).insert(data).execute()
return (r.data[0] if r.data else data) if r else data

def db_update(table: str, row_id: str, data: Dict) -> Dict:
if not _sb: return data
data[“updated_at”] = datetime.utcnow().isoformat()
r = _sb.table(table).update(data).eq(“id”, row_id).execute()
return (r.data[0] if r.data else data) if r else data

# ══════════════════════════════════════════════════════════════════════════════

# UTILITIES

# ══════════════════════════════════════════════════════════════════════════════

def now_iso() -> str: return datetime.now(timezone.utc).isoformat()
def safe_fn(name: str) -> str:
return re.sub(r”[^A-Za-z0-9_-]+”,”*”,str(name or “file”)).strip(”*”) or “file”

def abs_url(rel: str) -> str:
host = os.getenv(“RENDER_EXTERNAL_HOSTNAME”,””).strip()
if rel.startswith(“http”): return rel
if not host: return rel
return f”https://{host}/{rel.lstrip(’/’)}”

def rel_url(path: Path) -> str:
path = path.resolve()
for root, prefix in [(MEDIA_DIR,”/media”),(UPLOAD_DIR,”/uploads”),
(EXPORT_DIR,”/exports”),(RENDER_DIR,”/renders”)]:
try: return f”{prefix}/{path.relative_to(root)}”
except: pass
return str(path)

def load_json(v: Any, default: Any = None) -> Any:
if v in (None, “”): return default
if isinstance(v, (dict, list)): return v
try: return json.loads(v)
except: return default

# ══════════════════════════════════════════════════════════════════════════════

# LLM

# ══════════════════════════════════════════════════════════════════════════════

def llm(system: str, user: str, json_mode=False) -> str:
if not _openai_client: return “{}” if json_mode else “OpenAI not configured.”
kw: Dict[str,Any] = {“model”:OPENAI_MODEL,“max_tokens”:3000,“temperature”:0.75,
“messages”:[{“role”:“system”,“content”:system},
{“role”:“user”,“content”:user}]}
if json_mode: kw[“response_format”] = {“type”:“json_object”}
return _openai_client.chat.completions.create(**kw).choices[0].message.content.strip()

def parse_j(raw: str, fb: Any = None) -> Any:
try: return json.loads(raw)
except:
m = re.search(r’({[\s\S]*}|[[\s\S]*])’, raw)
if m:
try: return json.loads(m.group(1))
except: pass
return fb if fb is not None else {}

def gen_img(prompt: str, size=“1024x1024”, quality=“standard”) -> Optional[str]:
if not _openai_client: return None
r = _openai_client.images.generate(model=IMAGE_MODEL, prompt=prompt, n=1,
size=size, quality=quality, style=“vivid”,
response_format=“b64_json”)
b = r.data[0].b64_json
return f”data:image/png;base64,{b}” if b else None

# ══════════════════════════════════════════════════════════════════════════════

# PYDANTIC MODELS

# ══════════════════════════════════════════════════════════════════════════════

class SignupIn(BaseModel):
email: str; password: str = Field(min_length=8); full_name: Optional[str] = None
@field_validator(“email”)
@classmethod
def _e(cls, v): e=v.strip().lower(); assert EMAIL_RE.match(e),“Invalid email”; return e

class LoginIn(BaseModel):
email: str; password: str
@field_validator(“email”)
@classmethod
def _e(cls, v): return v.strip().lower()

class ProjectIn(BaseModel):
title: Optional[str]=None; name: Optional[str]=None; brief: Optional[str]=None
event_type: Optional[str]=None; style_direction: Optional[str]=None
style_theme: Optional[str]=None

class RunIn(BaseModel):
text: str = Field(min_length=3); project_id: Optional[str]=None
name: Optional[str]=None; event_type: Optional[str]=None
style_direction: Optional[str]=None

class RunProjectIn(BaseModel):
text: Optional[str]=None; name: Optional[str]=None
event_type: Optional[str]=None; style_direction: Optional[str]=None

class SelectIn(BaseModel):
project_id: str; index: int = Field(ge=0, le=2)

class SelectCompatIn(BaseModel):
concept_index: Optional[int]=Field(default=None,ge=0,le=2)
index: Optional[int]=Field(default=None,ge=0,le=2)

class CommentIn(BaseModel):
project_id: str; section: str = “general”; comment_text: str

class UpdateIn(BaseModel):
project_id: str; field: str; value: Any

class DeptPDFIn(BaseModel):
title: Optional[str]=None

class ArmIn(BaseModel):
armed: bool = True

class CueJumpIn(BaseModel):
cue_index: Optional[int]=Field(default=None,ge=0); cue_no: Optional[int]=None

class AssetIn(BaseModel):
asset_type: str; title: str; prompt: str=Field(min_length=3)
section: Optional[str]=None; job_kind: Optional[str]=None; generate_now: bool=True

class MoodboardIn(BaseModel):
concept_index: Optional[int]=Field(default=None,ge=0,le=2)
count: int=Field(default=3,ge=1,le=6); generate_now: bool=True

class JobIn(BaseModel):
agent_type: str; job_type: str; title: Optional[str]=None
priority: int=Field(default=5,ge=1,le=10); input_data: Optional[Dict[str,Any]]=None

class OrchestrateIn(BaseModel):
auto_generate_moodboard: bool=True; queue_3d: bool=True
queue_video: bool=True; queue_cad: bool=True; queue_manuals: bool=True

class ElemSheetIn(BaseModel):
include_sound: bool=True; include_lighting: bool=True
include_scenic: bool=True; include_power_summary: bool=True
include_xlsx: bool=True; sheet_title: Optional[str]=None

class ShowTrialIn(BaseModel):
include_walkthrough: bool=True; include_audio_video: bool=True
include_camera_pan: bool=True; queue_render_jobs: bool=True
draft_name: Optional[str]=None

class ShowTrialUpdateIn(BaseModel): trial_data: Dict[str,Any]
class ShowTrialFinalIn(BaseModel): use_trial_cues: bool=True; mark_ready: bool=True

class TTSIn(BaseModel):
text: str=Field(min_length=1,max_length=4096)
voice: Optional[str]=None; instructions: Optional[str]=None

class VoiceSessionIn(BaseModel):
project_id: Optional[str]=None; title: Optional[str]=None
system_prompt: Optional[str]=None; voice: Optional[str]=None

class VoiceTextIn(BaseModel):
session_id: Optional[str]=None; project_id: Optional[str]=None
text: str=Field(min_length=1); voice: Optional[str]=None
voice_instructions: Optional[str]=None; title: Optional[str]=None
system_prompt: Optional[str]=None

class CtrlIn(BaseModel):
protocol: str; target: Optional[str]=None; base_url: Optional[str]=None
path: Optional[str]=None; method: Optional[str]=None
headers: Optional[Dict[str,str]]=None; params: Optional[Dict[str,Any]]=None
body: Optional[Dict[str,Any]]=None; address: Optional[str]=None
ip: Optional[str]=None; port: Optional[int]=None; args: Optional[List[Any]]=None

class VisualPolicyIn(BaseModel):
preview_size: Optional[str]=None; master_size: Optional[str]=None
print_size: Optional[str]=None; aspect_ratio: Optional[str]=None

class DXFIn(BaseModel): project_id: str
class PDFIn(BaseModel): project_id: str; template: Optional[str]=“executive”
class ImgIn(BaseModel):
prompt: str; size: str=“1024x1024”; quality: str=“standard”
project_id: Optional[str]=None; section: Optional[str]=None

# ══════════════════════════════════════════════════════════════════════════════

# DOMAIN DEFAULTS

# ══════════════════════════════════════════════════════════════════════════════

def *sound_plan(*=None):
return {“system_design”:{“console”:“FOH digital console”,“speaker_system”:“Line array PA”},
“input_list”:[“MC mic”,“Playback stereo”,“Guest mic”],
“playback_cues”:[“opening stinger”,“walk-in bed”,“finale”],
“pdf_sections”:[{“heading”:“Sound Overview”,“body”:“Planning-level sound system.”},
{“heading”:“Input List”,“body”:“MC, playback, guest, ambient.”}]}

def *lighting_plan(*=None):
return {“fixture_list”:[“Moving Heads”,“Wash Fixtures”,“LED Battens”,“Audience Blinders”,“Pinspots”],
“scene_cues”:[“house-to-half”,“opening reveal”,“speaker special”,“finale”],
“pdf_sections”:[{“heading”:“Lighting Overview”,“body”:“Concept-driven lighting plan.”},
{“heading”:“Cue Intent”,“body”:“Opening, transitions, finale.”}]}

def *showrunner_plan(*=None):
return {“running_order”:[“Standby”,“House to half”,“Opening AV”,“MC welcome”,“Finale”],
“pdf_sections”:[{“heading”:“Show Running”,“body”:“Cue-based show running script.”}],
“console_cues”:[
{“cue_no”:1,“name”:“Standby”,“cue_type”:“standby”,“standby”:“All standby”,“go”:“Standby ack”,“actions”:[]},
{“cue_no”:2,“name”:“House to Half”,“cue_type”:“lighting”,“standby”:“Lights standby”,“go”:“Go half”,“actions”:[{“protocol”:“lighting”,“target”:“house_lights”,“value”:“half”}]},
{“cue_no”:3,“name”:“Opening AV”,“cue_type”:“av”,“standby”:“AV standby”,“go”:“Go opener”,“actions”:[{“protocol”:“av”,“target”:“screen”,“value”:“play_opener”}]},
{“cue_no”:4,“name”:“MC Welcome”,“cue_type”:“sound”,“standby”:“Mic standby”,“go”:“Go MC”,“actions”:[{“protocol”:“sound”,“target”:“mc_mic”,“value”:“on”}]},
]}

def _scene_json(project: Dict) -> Dict:
sel = project.get(“selected_concept”) or project.get(“selected”) or {}
return {“venue_type”:project.get(“event_type”,“event”),“concept_name”:sel.get(“name”),
“stage”:{“width”:18000,“depth”:9000,“height”:1200},
“screens”:[{“name”:“Center LED”,“width”:8000,“height”:4500}],
“cameras”:[{“view”:“hero”},{“view”:“wide”},{“view”:“top”}]}

def _console_state(project: Dict) -> Dict:
s = project.get(“department_outputs”) or {}
if not isinstance(s, dict): s = {}
s.setdefault(“armed”,False); s.setdefault(“hold”,False)
s.setdefault(“console_index”,0); s.setdefault(“execution_log”,[])
return s

def _log_cue(state: Dict, event: Dict) -> Dict:
log = list(state.get(“execution_log”) or [])
log.append({“time”:now_iso(),**event}); state[“execution_log”] = log[-200:]
state[“last_status”] = event.get(“status”); return state

EVENT_BUDGETS = {
“conference”:(800000,1800000,4200000),“award show”:(1200000,2600000,6500000),
“brand launch”:(900000,2200000,5500000),“wedding”:(700000,1600000,4000000),
“concert”:(1500000,3500000,9000000),“festival”:(1200000,2800000,7200000),
“corporate”:(800000,1700000,4500000),“generic”:(500000,1200000,3000000),
}

def _infer_type(text: str, et: Optional[str]) -> str:
if et: return et
t = (text or “”).lower()
for n in EVENT_BUDGETS:
if n!=“generic” and n in t: return n
if “launch” in t: return “brand launch”
if “award” in t: return “award show”
return “generic”

def _analyze(brief: str, event_type: Optional[str]) -> Dict:
inferred = _infer_type(brief, event_type)
fb = {“summary”:brief[:300],“event_type”:inferred,
“objectives”:[“Translate brief into concept”,“Build department outputs”,“Cost estimate”],
“audience”:“Stakeholders, brand team, agencies, vendors”,
“risks”:[“Brief needs venue/timeline detail”],
“assumptions”:[“Costing is planning-level”]}
data = parse_j(llm(“Senior experiential strategist. Return JSON only.”,
f’Analyze brief. Return JSON: summary, event_type, objectives(array), audience, risks(array), assumptions(array).\nBrief: {brief}’,
json_mode=True), fb)
return {**fb, **data} if isinstance(data,dict) else fb

def _gen_concepts(brief: str, analysis: Dict, event_type: Optional[str]) -> List[Dict]:
inferred = _infer_type(brief, event_type or analysis.get(“event_type”,””))
b = EVENT_BUDGETS.get(inferred, EVENT_BUDGETS[“generic”])
fb = [
{“name”:“Cinematic Signature”,“summary”:f”Premium concept for {inferred}.”,“style”:“immersive premium”,
“colors”:[“black”,“gold”,“warm white”],“materials”:[“mirror acrylic”,“fabric”,“metal”],
“experience”:“emotional brand reveal”,“key_zones”:[“arrival”,“stage”,“screen”,“audience”],
“estimated_budget_inr”:{“low”:b[0],“medium”:b[1],“high”:b[2]}},
{“name”:“Modern Tech Grid”,“summary”:f”Futuristic concept for {inferred}.”,“style”:“futuristic sharp”,
“colors”:[“midnight blue”,“cyan”,“silver”],“materials”:[“LED mesh”,“truss”,“glass”],
“experience”:“show-control visual language”,“key_zones”:[“arrival”,“stage”,“screen”,“audience”],
“estimated_budget_inr”:{“low”:int(b[0]*1.2),“medium”:int(b[1]*1.2),“high”:int(b[2]*1.2)}},
{“name”:“Elegant Minimal Luxe”,“summary”:f”Refined concept for {inferred}.”,“style”:“clean editorial”,
“colors”:[“ivory”,“champagne”,“graphite”],“materials”:[“textured flats”,“wood veneer”,“soft fabric”],
“experience”:“refined storytelling”,“key_zones”:[“arrival”,“stage”,“screen”,“audience”],
“estimated_budget_inr”:{“low”:int(b[0]*1.45),“medium”:int(b[1]*1.45),“high”:int(b[2]*1.45)}},
]
data = parse_j(llm(“Creative director for live events. Return JSON only.”,
f’Generate 3 concepts. Return {{“concepts”:[{{“name”:”…”,“summary”:”…”,“style”:”…”,“colors”:[],“materials”:[],“experience”:”…”,“key_zones”:[]}}]}}\nBrief: {brief}’,
json_mode=True))
raw = data.get(“concepts”,[]) if isinstance(data,dict) else []
return [{**fb[i], **(raw[i] if i<len(raw) and isinstance(raw[i],dict) else {}),
“estimated_budget_inr”:fb[i][“estimated_budget_inr”]} for i in range(3)]

def *create_pdf(title: str, sections: Any, prefix: str) -> Dict[str,str]:
try:
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_c
fname = EXPORT_DIR / f”{prefix}*{uuid.uuid4().hex}.pdf”
c = rl_c.Canvas(str(fname), pagesize=A4)
w, h = A4; left=18*mm; y=h-20*mm
def np():
nonlocal y; c.showPage(); y=h-20*mm
c.setFont(“Helvetica-Bold”,18); c.drawString(left,y,title); y-=12*mm
secs = sections if isinstance(sections,list) else [{“heading”:“Content”,“body”:str(sections)}]
for sec in secs:
hd = str(sec.get(“heading”,””) if isinstance(sec,dict) else sec)
bd = str(sec.get(“body”,””) if isinstance(sec,dict) else “”)
if y<35*mm: np()
c.setFont(“Helvetica-Bold”,13); c.drawString(left,y,hd[:80]); y-=8*mm
c.setFont(“Helvetica”,10)
for para in bd.split(”\n”):
words=para.split(); line=””
for word in words:
t=(line+” “+word).strip()
if c.stringWidth(t,“Helvetica”,10)<(w-2*left): line=t
else:
if y<20*mm: np(); c.setFont(“Helvetica”,10)
c.drawString(left,y,line); y-=5*mm; line=word
if line:
if y<20*mm: np(); c.setFont(“Helvetica”,10)
c.drawString(left,y,line); y-=5*mm
y-=2*mm
y-=4*mm
c.save()
r = rel_url(fname)
return {“pdf_path”:r,“pdf_url”:abs_url(r)}
except ImportError:
fname = EXPORT_DIR / f”{prefix}_{uuid.uuid4().hex}.txt”
secs = sections if isinstance(sections,list) else [{“heading”:“Content”,“body”:str(sections)}]
fname.write_text(”\n\n”.join([f”## {s.get(‘heading’,’’)}\n{s.get(‘body’,’’)}”
if isinstance(s,dict) else str(s) for s in secs]), encoding=“utf-8”)
r = rel_url(fname)
return {“pdf_path”:r,“pdf_url”:abs_url(r)}

def _ctrl(payload: Dict) -> Dict:
proto = (payload.get(“protocol”) or “”).lower()
if proto == “http”:
try:
import requests as _req
method = (payload.get(“method”) or “POST”).upper()
url = payload.get(“address”) or f”{(payload.get(‘base_url’) or ‘’).rstrip(’/’)}/{str(payload.get(‘path’,’’) or ‘’).lstrip(’/’)}”
if not url: return {“ok”:False,“message”:“Missing URL”}
r = _req.request(method=method,url=url,headers=payload.get(“headers”) or {},
params=payload.get(“params”) or {},json=payload.get(“body”),timeout=10)
return {“ok”:r.ok,“status_code”:r.status_code,“url”:url,“response”:r.text[:300]}
except Exception as e:
return {“ok”:False,“message”:str(e)}
return {“ok”:True,“protocol”:proto or “simulated”,“message”:“Simulated”,“target”:payload.get(“target”)}

def _gen_dxf(layout: Dict, project_name: str) -> str:
vw=float(layout.get(“venue_width_m”) or 40); vd=float(layout.get(“venue_depth_m”) or 30)
zones=layout.get(“zones”) or []; S=1000
h=[0]
def nxt(): h[0]+=1; return format(h[0],‘X’)
out=[]
def g(c,v): out.append(f”{c:>3}”); out.append(str(v))
ZLAYERS={“stage”:(“ZONE-STAGE”,1),“seating”:(“ZONE-SEATING”,3),“vip”:(“ZONE-VIP”,4),
“circulation”:(“ZONE-CIRCULATION”,8),“service”:(“ZONE-SERVICE”,6),
“registration”:(“ZONE-REGISTRATION”,5),“catering”:(“ZONE-CATERING”,2)}
g(0,“SECTION”);g(2,“HEADER”);g(9,”$ACADVER”);g(1,“AC1024”);g(9,”$INSUNITS”);g(70,4)
g(9,”$EXTMIN”);g(10,-2000);g(20,-2000);g(30,0)
g(9,”$EXTMAX”);g(10,vw*S+8000);g(20,vd*S+6000);g(30,0);g(0,“ENDSEC”)
g(0,“SECTION”);g(2,“TABLES”)
layers=[(“0”,7),(“VENUE-PERIMETER”,7),(“ZONE-STAGE”,1),(“ZONE-SEATING”,3),
(“ZONE-VIP”,4),(“ZONE-CIRCULATION”,8),(“ZONE-SERVICE”,6),
(“ZONE-REGISTRATION”,5),(“ZONE-CATERING”,2),(“WALLS”,7),
(“GRID”,9),(“DIMENSIONS”,7),(“TEXT-LABELS”,7),(“TITLE-BLOCK”,7)]
g(0,“TABLE”);g(2,“LAYER”);g(5,nxt());g(100,“AcDbSymbolTable”);g(70,len(layers))
for ln,lc in layers:
g(0,“LAYER”);g(5,nxt());g(100,“AcDbSymbolTableRecord”);g(100,“AcDbLayerTableRecord”)
g(2,ln);g(70,0);g(62,lc);g(6,“CONTINUOUS”)
g(0,“ENDTAB”);g(0,“ENDSEC”)
g(0,“SECTION”);g(2,“ENTITIES”)
def line(x1,y1,x2,y2,layer,color=7):
g(0,“LINE”);g(5,nxt());g(100,“AcDbEntity”);g(100,“AcDbLine”)
g(8,layer);g(62,color);g(10,x1);g(20,y1);g(30,0);g(11,x2);g(21,y2);g(31,0)
def pline(pts,layer,color=7):
g(0,“LWPOLYLINE”);g(5,nxt());g(100,“AcDbEntity”);g(100,“AcDbPolyline”)
g(8,layer);g(62,color);g(90,len(pts));g(70,1);g(43,0)
for px,py in pts: g(10,px);g(20,py)
def txt(tx,ty,ht,t,layer,color=7,a=0):
g(0,“TEXT”);g(5,nxt());g(100,“AcDbEntity”);g(100,“AcDbText”)
g(8,layer);g(62,color);g(10,tx);g(20,ty);g(30,0)
g(40,ht);g(1,str(t)[:80]);g(72,a)
if a: g(11,tx);g(21,ty);g(31,0)
pline([(0,0),(vw*S,0),(vw*S,vd*S),(0,vd*S)],“VENUE-PERIMETER”,7)
x=float(S)
while x<vw*S: line(x,0,x,vd*S,“GRID”,9); x+=S
y=float(S)
while y<vd*S: line(0,y,vw*S,y,“GRID”,9); y+=S
for z in zones:
zt=(z.get(“zone_type”) or “”).lower()
layer,color=ZLAYERS.get(zt,(“ZONE-CIRCULATION”,8))
if z.get(“x_m”) is not None:
x1=float(z[“x_m”])*S; y1=float(z[“y_m”])*S
w=float(z.get(“width_m”) or z.get(“width”,5))*S
hh=float(z.get(“depth_m”) or z.get(“height”,5))*S
else:
x1=float(str(z.get(“left_pct”,“5%”)).rstrip(”%”))/100*vw*S
y1=float(str(z.get(“top_pct”,“5%”)).rstrip(”%”))/100*vd*S
w=float(str(z.get(“width_pct”,“20%”)).rstrip(”%”))/100*vw*S
hh=float(str(z.get(“height_pct”,“20%”)).rstrip(”%”))/100*vd*S
x2,y2=x1+w,y1+hh; cx,cy=x1+w/2,y1+hh/2
area=z.get(“area_m2”) or round(w/S*hh/S,1)
pline([(x1,y1),(x2,y1),(x2,y2),(x1,y2)],layer,color)
th=max(100.0,min(300.0,w/8))
txt(cx,cy+th*.6,th,z.get(“name”,“Zone”).upper(),“TEXT-LABELS”,7,1)
txt(cx,cy-th*.6,th*.5,f”{area:.0f}m\u00b2”,“TEXT-LABELS”,9,1)
off=800.0
line(0,-off,vw*S,-off,“DIMENSIONS”,7)
txt(vw*S/2,-off-260,200,f”WIDTH: {vw:.1f}m”,“DIMENSIONS”,7,1)
line(-off,0,-off,vd*S,“DIMENSIONS”,7)
txt(-off-300,vd*S/2,200,f”DEPTH: {vd:.1f}m”,“DIMENSIONS”,7,1)
tb_x,tb_y,tb_w,tb_h=vw*S+800,0,5000,3000
pline([(tb_x,tb_y),(tb_x+tb_w,tb_y),(tb_x+tb_w,tb_y+tb_h),(tb_x,tb_y+tb_h)],“TITLE-BLOCK”,7)
for tx,ty,ht,t in [
(tb_x+200,tb_y+2600,220,project_name.upper()[:40]),
(tb_x+200,tb_y+2000,140,f”{layout.get(‘dimensions’,’’)} | Ceiling: {layout.get(‘ceiling_height’,’’)}”),
(tb_x+200,tb_y+1400,120,f”Area: {layout.get(‘total_area’,’’)} | Cap: {layout.get(‘capacity’,’’)}”),
(tb_x+200,tb_y+900,100,“Scale 1:100 | Units: mm”),
(tb_x+200,tb_y+400,90,“AICreative Studio”),
(tb_x+200,tb_y+200,80,f”Date: {datetime.utcnow().strftime(’%Y-%m-%d’)} | DXF R2010”),
]: txt(tx,ty,ht,t,“TITLE-BLOCK”,7)
g(0,“ENDSEC”);g(0,“EOF”)
return “\n”.join(out)

# ══════════════════════════════════════════════════════════════════════════════

# ROUTES

# ══════════════════════════════════════════════════════════════════════════════

@app.get(”/”)
def root():
return {“message”:“AICreative Studio API v4.0 — running”,“time”:now_iso(),“docs”:”/docs”,
“openai”:bool(_openai_client),“supabase”:bool(_sb)}

@app.get(”/health”)
def health():
return {“status”:“ok”,“time”:now_iso(),“openai”:bool(_openai_client),
“supabase”:bool(_sb),“port”:PORT,“cors_origins”:ALLOWED_ORIGINS}

# ─── Auth ──────────────────────────────────────────────────────────────────────

@app.post(”/signup”)
def signup(p: SignupIn):
if _sb:
ex = _sb.table(“users”).select(“id”).eq(“email”,p.email).execute()
if ex and ex.data: raise HTTPException(400,“Email already registered”)
uid = str(uuid.uuid4())
user = db_insert(“users”,{“id”:uid,“email”:p.email,“password”:_hash_pw(p.password),
“full_name”:p.full_name or p.email.split(”@”)[0],
“plan”:“studio_pro”,“projects_used”:0,“projects_limit”:100})
token = _make_token(uid)
safe = {k:v for k,v in user.items() if k!=“password”}
return {“message”:“User created”,“user_id”:uid,“access_token”:token,
“token”:token,“token_type”:“bearer”,“user”:safe}

@app.post(”/login”)
def login(p: LoginIn):
user = db_get(“users”,email=p.email)
if not user: raise HTTPException(400,“User not found”)
pw_hash = user.get(“password”) or user.get(“password_hash”,””)
if not _verify_pw(p.password, pw_hash): raise HTTPException(400,“Wrong password”)
token = _make_token(str(user[“id”]))
safe = {k:v for k,v in user.items() if k not in (“password”,“password_hash”)}
return {“access_token”:token,“token”:token,“token_type”:“bearer”,
“user_id”:str(user[“id”]),“user”:safe}

@app.post(”/logout”)
def logout(_=Depends(get_current_user)):
return {“message”:“Logged out. Remove bearer token on client.”}

@app.get(”/me”)
def me(u=Depends(get_current_user)):
return {“user”:{k:v for k,v in u.items() if k not in (“password”,“password_hash”)}}

# ─── Projects ──────────────────────────────────────────────────────────────────

@app.get(”/projects”)
def list_projects(u=Depends(get_current_user)):
return {“projects”:db_list(“projects”,user_id=str(u[“id”]))}

@app.post(”/projects”)
def create_project(p: ProjectIn, u=Depends(get_current_user)):
return db_insert(“projects”,{“id”:str(uuid.uuid4()),“user_id”:str(u[“id”]),
“project_name”:(p.title or p.name or “Untitled”).strip(),
“brief_text”:p.brief,“event_type”:p.event_type,
“style_direction”:p.style_direction,“style_theme”:p.style_theme or “luxury”,“status”:“draft”})

@app.get(”/projects/{pid}”)
@app.get(”/project/{pid}”)
def get_project(pid: str, u=Depends(get_current_user)):
proj = db_get(“projects”,id=pid,user_id=str(u[“id”]))
if not proj: raise HTTPException(404,“Project not found”)
return {“project”:proj}

@app.post(”/project/update”)
def update_project(p: UpdateIn, u=Depends(get_current_user)):
return {“message”:“Updated”,“project”:db_update(“projects”,p.project_id,{p.field:p.value})}

# ─── Pipeline ──────────────────────────────────────────────────────────────────

def _run_logic(project: Dict, text: str, event_type: Optional[str], uid: str) -> Dict:
pid = str(project[“id”])
updates: Dict[str,Any] = {}
if text and project.get(“brief_text”,””)!=text: updates[“brief_text”]=text
if event_type and not project.get(“event_type”): updates[“event_type”]=event_type
if updates: project=db_update(“projects”,pid,updates)
analysis = project.get(“analysis”) or {}
if not analysis:
analysis=_analyze(project.get(“brief_text”) or text, project.get(“event_type”) or event_type)
project=db_update(“projects”,pid,{“analysis”:analysis})
raw=project.get(“concepts”); concepts=load_json(raw) if isinstance(raw,str) else (raw or [])
if not concepts:
concepts=_gen_concepts(project.get(“brief_text”) or text,analysis,project.get(“event_type”) or event_type)
project=db_update(“projects”,pid,{“concepts”:concepts,“status”:“concepts_ready”})
return {“message”:“Pipeline completed”,“project_id”:pid,“status”:“concepts_ready”,
“brief”:project.get(“brief_text”),“analysis”:analysis,“concepts”:concepts,“project”:project}

@app.post(”/run”)
def run_pipeline(p: RunIn, u=Depends(get_current_user)):
uid=str(u[“id”])
if p.project_id:
proj=db_get(“projects”,id=p.project_id,user_id=uid)
if not proj: raise HTTPException(404,“Project not found”)
else:
proj=db_insert(“projects”,{“id”:str(uuid.uuid4()),“user_id”:uid,
“project_name”:(p.name or p.text[:50]).strip(),“brief_text”:p.text,
“event_type”:p.event_type,“style_direction”:p.style_direction,“status”:“draft”})
return _run_logic(proj,p.text,p.event_type,uid)

@app.post(”/projects/{pid}/run”)
def run_project(pid: str, p: RunProjectIn, u=Depends(get_current_user)):
uid=str(u[“id”]); proj=db_get(“projects”,id=pid,user_id=uid)
if not proj: raise HTTPException(404,“Project not found”)
text=(p.text or proj.get(“brief_text”,””)).strip()
if not text: raise HTTPException(422,“text required”)
return _run_logic(proj,text,p.event_type or proj.get(“event_type”),uid)

@app.post(”/select”)
def select_concept(p: SelectIn, u=Depends(get_current_user)):
uid=str(u[“id”]); proj=db_get(“projects”,id=p.project_id,user_id=uid)
if not proj: raise HTTPException(404,“Project not found”)
raw=proj.get(“concepts”); concepts=load_json(raw) if isinstance(raw,str) else (raw or [])
if not concepts: raise HTTPException(400,“Run pipeline first”)
if p.index>=len(concepts): raise HTTPException(400,f”Only {len(concepts)} concepts”)
selected=concepts[p.index]
proj=db_update(“projects”,p.project_id,{“selected_concept”:selected,“status”:“concept_selected”})
return {“message”:“Concept selected”,“index”:p.index,“selected”:selected,“project”:proj}

@app.post(”/projects/{pid}/select-concept”)
def select_compat(pid: str, p: SelectCompatIn, u=Depends(get_current_user)):
idx=p.concept_index if p.concept_index is not None else p.index
if idx is None: raise HTTPException(422,“concept_index required”)
return select_concept(SelectIn(project_id=pid,index=idx),u)

# ─── Comments ──────────────────────────────────────────────────────────────────

@app.post(”/comment”)
@app.post(”/comments”)
def add_comment(p: CommentIn, u=Depends(get_current_user)):
row=db_insert(“project_comments”,{“id”:str(uuid.uuid4()),“project_id”:p.project_id,
“user_id”:str(u[“id”]),“section”:p.section,
“content”:p.comment_text,“author_name”:u.get(“full_name”,“Anonymous”)})
return {“message”:“Comment added”,“comment”:row}

@app.get(”/comments/{pid}”)
def list_comments(pid: str, _=Depends(get_current_user)):
return {“comments”:db_list(“project_comments”,project_id=pid)}

# ─── Departments ───────────────────────────────────────────────────────────────

def _build_depts(pid: str, uid: str) -> Dict:
proj=db_get(“projects”,id=pid,user_id=uid)
if not proj: raise HTTPException(404,“Project not found”)
sel=proj.get(“selected_concept”) or proj.get(“selected”)
if not sel: raise HTTPException(400,“Select a concept first”)
sound=_sound_plan(proj); lighting=_lighting_plan(proj); showrunner=_showrunner_plan(proj)
state=_console_state(proj)
state.update({“sound_ready”:True,“lighting_ready”:True,“showrunner_ready”:True,
“console_index”:0,“hold”:False})
proj=db_update(“projects”,pid,{“sound_data”:sound,“lighting_data”:lighting,
“showrunner_data”:showrunner,“department_outputs”:state,
“scene_json”:_scene_json(proj),“status”:“departments_ready”})
return {“message”:“Departments generated”,“project_id”:pid,“sound_data”:sound,
“lighting_data”:lighting,“showrunner_data”:showrunner,“project”:proj}

@app.post(”/project/{pid}/departments/build”)
@app.post(”/projects/{pid}/generate-departments”)
def build_departments(pid: str, u=Depends(get_current_user)):
return _build_depts(pid,str(u[“id”]))

@app.get(”/project/{pid}/departments/manuals”)
def dept_manuals(pid: str, u=Depends(get_current_user)):
proj=db_get(“projects”,id=pid,user_id=str(u[“id”]))
if not proj: raise HTTPException(404,“Project not found”)
return {“project_id”:pid,“sound_data”:proj.get(“sound_data”),
“lighting_data”:proj.get(“lighting_data”),“showrunner_data”:proj.get(“showrunner_data”)}

@app.post(”/project/{pid}/departments/pdf/sound”)
def pdf_sound(pid: str, p: DeptPDFIn, u=Depends(get_current_user)):
proj=db_get(“projects”,id=pid,user_id=str(u[“id”]))
if not proj or not proj.get(“sound_data”): raise HTTPException(404,“Sound data not found. Build departments first.”)
sd=proj[“sound_data”]; secs=sd.get(“pdf_sections”) or sd if isinstance(sd,dict) else sd
return {“project_id”:pid,**_create_pdf(p.title or “Sound Design Manual”,secs,“sound_manual”)}

@app.post(”/project/{pid}/departments/pdf/lighting”)
def pdf_lighting(pid: str, p: DeptPDFIn, u=Depends(get_current_user)):
proj=db_get(“projects”,id=pid,user_id=str(u[“id”]))
if not proj or not proj.get(“lighting_data”): raise HTTPException(404,“Lighting data not found. Build departments first.”)
ld=proj[“lighting_data”]; secs=ld.get(“pdf_sections”) or ld if isinstance(ld,dict) else ld
return {“project_id”:pid,**_create_pdf(p.title or “Lighting Design Manual”,secs,“lighting_manual”)}

@app.post(”/project/{pid}/departments/pdf/showrunner”)
def pdf_showrunner(pid: str, p: DeptPDFIn, u=Depends(get_current_user)):
proj=db_get(“projects”,id=pid,user_id=str(u[“id”]))
if not proj or not proj.get(“showrunner_data”): raise HTTPException(404,“Showrunner data not found.”)
sd=proj[“showrunner_data”]; secs=sd.get(“pdf_sections”) or sd if isinstance(sd,dict) else sd
return {“project_id”:pid,**_create_pdf(p.title or “Show Running Script”,secs,“showrunner_manual”)}

# ─── Show Console ───────────────────────────────────────────────────────────────

@app.get(”/project/{pid}/show-console”)
def console_status(pid: str, u=Depends(get_current_user)):
proj=db_get(“projects”,id=pid,user_id=str(u[“id”]))
if not proj: raise HTTPException(404,“Project not found”)
sd=proj.get(“showrunner_data”) or {}; cues=sd.get(“console_cues”) or []
state=_console_state(proj); idx=min(int(state.get(“console_index”,0)),max(len(cues)-1,0)) if cues else 0
return {“project_id”:pid,“armed”:bool(state.get(“armed”)),“hold”:bool(state.get(“hold”)),
“cue_index”:idx,“cue”:cues[idx] if cues else None,“available_cues”:cues}

@app.post(”/project/{pid}/show-console/arm”)
def console_arm(pid: str, p: ArmIn, u=Depends(get_current_user)):
uid=str(u[“id”]); proj=db_get(“projects”,id=pid,user_id=uid)
if not proj: raise HTTPException(404,“Project not found”)
state=_console_state(proj); state[“armed”]=bool(p.armed)
state=_log_cue(state,{“status”:“armed” if p.armed else “disarmed”})
db_update(“projects”,pid,{“department_outputs”:state})
return {“message”:“Console updated”,“armed”:state[“armed”]}

@app.post(”/project/{pid}/show-console/go”)
def console_go(pid: str, execute: bool=Query(True), u=Depends(get_current_user)):
uid=str(u[“id”]); proj=db_get(“projects”,id=pid,user_id=uid)
if not proj: raise HTTPException(404,“Project not found”)
cues=(proj.get(“showrunner_data”) or {}).get(“console_cues”) or []
if not cues: raise HTTPException(400,“No cues found”)
state=_console_state(proj)
if not state.get(“armed”): raise HTTPException(400,“Console not armed”)
if state.get(“hold”): raise HTTPException(400,“Console on hold”)
idx=min(int(state.get(“console_index”,0)),len(cues)-1)
cue=cues[idx]; results=[_ctrl(a) for a in cue.get(“actions”,[])] if execute else []
state[“console_index”]=min(idx+1,len(cues)-1)
state=_log_cue(state,{“status”:“go”,“cue_index”:idx})
db_update(“projects”,pid,{“department_outputs”:state})
return {“message”:“Cue executed”,“cue_index”:idx,“cue”:cue,“results”:results}

@app.post(”/project/{pid}/show-console/next”)
def console_next(pid: str, u=Depends(get_current_user)):
uid=str(u[“id”]); proj=db_get(“projects”,id=pid,user_id=uid)
if not proj: raise HTTPException(404,“Project not found”)
cues=(proj.get(“showrunner_data”) or {}).get(“console_cues”) or []
if not cues: raise HTTPException(400,“No cues”)
state=_console_state(proj); idx=min(int(state.get(“console_index”,0))+1,len(cues)-1)
state[“console_index”]=idx; db_update(“projects”,pid,{“department_outputs”:state})
return {“cue_index”:idx,“cue”:cues[idx]}

@app.post(”/project/{pid}/show-console/back”)
def console_back(pid: str, u=Depends(get_current_user)):
uid=str(u[“id”]); proj=db_get(“projects”,id=pid,user_id=uid)
if not proj: raise HTTPException(404,“Project not found”)
cues=(proj.get(“showrunner_data”) or {}).get(“console_cues”) or []
state=_console_state(proj); idx=max(int(state.get(“console_index”,0))-1,0)
state[“console_index”]=idx; db_update(“projects”,pid,{“department_outputs”:state})
return {“cue_index”:idx,“cue”:cues[idx] if cues else None}

@app.post(”/project/{pid}/show-console/hold”)
def console_hold(pid: str, u=Depends(get_current_user)):
uid=str(u[“id”]); proj=db_get(“projects”,id=pid,user_id=uid)
if not proj: raise HTTPException(404,“Project not found”)
state=_console_state(proj); state[“hold”]=True
db_update(“projects”,pid,{“department_outputs”:state})
return {“message”:“Hold engaged”}

@app.post(”/project/{pid}/show-console/standby”)
def console_standby(pid: str, u=Depends(get_current_user)):
uid=str(u[“id”]); proj=db_get(“projects”,id=pid,user_id=uid)
if not proj: raise HTTPException(404,“Project not found”)
state=_console_state(proj); state[“hold”]=False
db_update(“projects”,pid,{“department_outputs”:state})
return {“message”:“Standby”}

@app.post(”/project/{pid}/show-console/panic”)
def console_panic(pid: str, u=Depends(get_current_user)):
uid=str(u[“id”]); proj=db_get(“projects”,id=pid,user_id=uid)
if not proj: raise HTTPException(404,“Project not found”)
state=_console_state(proj); state[“hold”]=True; state[“armed”]=False
db_update(“projects”,pid,{“department_outputs”:state})
return {“message”:“Panic — console disarmed and hold engaged”}

@app.post(”/project/{pid}/show-console/jump”)
def console_jump(pid: str, p: CueJumpIn, u=Depends(get_current_user)):
uid=str(u[“id”]); proj=db_get(“projects”,id=pid,user_id=uid)
if not proj: raise HTTPException(404,“Project not found”)
cues=(proj.get(“showrunner_data”) or {}).get(“console_cues”) or []
if not cues: raise HTTPException(400,“No cues”)
idx=p.cue_index
if p.cue_no is not None:
m=[i for i,c in enumerate(cues) if str(c.get(“cue_no”))==str(p.cue_no)]
if not m: raise HTTPException(404,“Cue not found”)
idx=m[0]
if idx is None: raise HTTPException(422,“cue_index or cue_no required”)
if idx<0 or idx>=len(cues): raise HTTPException(400,“Index out of range”)
state=_console_state(proj); state[“console_index”]=idx
db_update(“projects”,pid,{“department_outputs”:state})
return {“cue_index”:idx,“cue”:cues[idx]}

@app.get(”/project/{pid}/show-console/history”)
def console_history(pid: str, u=Depends(get_current_user)):
proj=db_get(“projects”,id=pid,user_id=str(u[“id”]))
if not proj: raise HTTPException(404,“Project not found”)
return {“execution_log”:_console_state(proj).get(“execution_log”) or []}

@app.post(”/control/execute”)
def ctrl_execute(p: CtrlIn, _=Depends(get_current_user)):
return {“message”:“Action executed”,“result”:_ctrl(p.model_dump(exclude_none=True))}

# ─── Assets / Images ───────────────────────────────────────────────────────────

@app.get(”/projects/{pid}/assets”)
def list_assets(pid: str, section: Optional[str]=Query(None), _=Depends(get_current_user)):
assets=db_list(“project_assets”,project_id=pid)
if section: assets=[a for a in assets if a.get(“section”)==section]
return {“assets”:assets}

@app.post(”/generate/image”)
def gen_image_ep(p: ImgIn, u=Depends(get_current_user)):
url=gen_img(p.prompt,p.size,p.quality)
if not url: raise HTTPException(500,“Image generation failed — check OPENAI_API_KEY”)
if p.project_id and _sb:
db_insert(“project_assets”,{“id”:str(uuid.uuid4()),“project_id”:p.project_id,
“user_id”:str(u[“id”]),“asset_type”:“generated_image”,
“title”:p.prompt[:60],“section”:p.section or “general”,“status”:“completed”})
return {“dataUrl”:url,“size”:p.size,“quality”:p.quality}

# ─── Exports ───────────────────────────────────────────────────────────────────

@app.post(”/export/dxf”)
def export_dxf(p: DXFIn, u=Depends(get_current_user)):
proj=db_get(“projects”,id=p.project_id,user_id=str(u[“id”]))
if not proj: raise HTTPException(404,“Project not found”)
rows=db_list(“cad_layouts”,project_id=p.project_id,limit=1)
if not rows: raise HTTPException(400,“Generate CAD layout first”)
layout=rows[0]; pname=proj.get(“project_name”) or proj.get(“name”,“Event”)
dxf=*gen_dxf(layout,pname)
fname=f”{safe_fn(pname)}*{datetime.utcnow().strftime(’%Y%m%d_%H%M’)}.dxf”
(EXPORT_DIR/fname).write_text(dxf,encoding=“utf-8”)
return {“filename”:fname,“download_url”:f”/exports/{fname}”,“size_bytes”:len(dxf.encode())}

@app.post(”/export/pdf”)
def export_pdf(p: PDFIn, u=Depends(get_current_user)):
proj=db_get(“projects”,id=p.project_id,user_id=str(u[“id”]))
if not proj: raise HTTPException(404,“Project not found”)
pname=proj.get(“project_name”) or proj.get(“name”,“Project”)
secs=[{“heading”:“Overview”,“body”:f”Project: {pname}\nStatus: {proj.get(‘status’,‘draft’)}”},
{“heading”:“Brief”,“body”:proj.get(“brief_text”) or proj.get(“brief”,””)},
{“heading”:“Analysis”,“body”:json.dumps(proj.get(“analysis”) or {},indent=2)}]
raw=proj.get(“concepts”); concepts=load_json(raw) if isinstance(raw,str) else (raw or [])
if concepts:
secs.append({“heading”:“Creative Concepts”,
“body”:”\n\n”.join([f”{c.get(‘name’,’’)}: {c.get(‘summary’,’’)}” for c in concepts])})
result=_create_pdf(pname,secs,safe_fn(pname))
return {“template”:p.template,**result}

@app.get(”/exports/{fname}”)
def dl_export(fname: str):
path=EXPORT_DIR/fname
if not path.exists(): raise HTTPException(404,“File not found”)
mt=“application/dxf” if fname.endswith(”.dxf”) else “application/pdf”
return FileResponse(path,media_type=mt,filename=fname)

# ─── Voice ─────────────────────────────────────────────────────────────────────

@app.post(”/voice/tts”)
@app.post(”/tts”)
def tts(p: TTSIn, _=Depends(get_current_user)):
if not *openai_client: raise HTTPException(500,“OpenAI not configured”)
path=VOICE_DIR/f”tts*{uuid.uuid4().hex}.mp3”
_openai_client.audio.speech.create(model=TTS_MODEL,voice=p.voice or TTS_VOICE,
input=p.text).stream_to_file(str(path))
r=rel_url(path)
return {“audio_url”:abs_url(r),“voice”:p.voice or TTS_VOICE,“disclosure”:“AI-generated voice”}

@app.post(”/voice/transcribe”)
async def transcribe(audio_file: UploadFile=File(…), _=Depends(get_current_user)):
if not *openai_client: raise HTTPException(500,“OpenAI not configured”)
suffix=Path(audio_file.filename or “audio.webm”).suffix or “.webm”
path=UPLOAD_DIR/f”transcribe*{uuid.uuid4().hex}{suffix}”
content=await audio_file.read()
if not content: raise HTTPException(400,“Empty file”)
path.write_bytes(content)
with path.open(“rb”) as fh:
result=_openai_client.audio.transcriptions.create(model=TRANSCRIBE_MDL,file=fh)
return {“transcript”:getattr(result,“text”,””)}

# ─── Manual PDF ─────────────────────────────────────────────────────────────────

@app.post(”/projects/{pid}/manuals/master/pdf”)
def master_pdf(pid: str, u=Depends(get_current_user)):
proj=db_get(“projects”,id=pid,user_id=str(u[“id”]))
if not proj: raise HTTPException(404,“Project not found”)
pname=proj.get(“project_name”) or proj.get(“name”,“Project”)
secs=[{“heading”:“Overview”,“body”:f”Project: {pname}\nStatus: {proj.get(‘status’)}”},
{“heading”:“Brief”,“body”:proj.get(“brief_text”) or “”},
{“heading”:“Analysis”,“body”:json.dumps(proj.get(“analysis”) or {},indent=2)},
{“heading”:“Sound”,“body”:json.dumps(proj.get(“sound_data”) or {},indent=2)},
{“heading”:“Lighting”,“body”:json.dumps(proj.get(“lighting_data”) or {},indent=2)},
{“heading”:“Showrunner”,“body”:json.dumps(proj.get(“showrunner_data”) or {},indent=2)}]
return _create_pdf(f”Master Manual — {pname}”,secs,“master_manual”)

# ─── Visual policy ─────────────────────────────────────────────────────────────

@app.get(”/projects/{pid}/visual-policy”)
def get_vp(pid: str, _=Depends(get_current_user)):
return {“project_id”:pid,“visual_policy”:{“aspect_ratio”:“16:9”,
“preview_size”:“1920x1080”,“master_size”:“1536x1024”,“print_size”:“3840x2160”}}

@app.post(”/projects/{pid}/visual-policy”)
def set_vp(pid: str, p: VisualPolicyIn, _=Depends(get_current_user)):
pol={“aspect_ratio”:p.aspect_ratio or “16:9”,“preview_size”:p.preview_size or “1920x1080”,
“master_size”:p.master_size or “1536x1024”,“print_size”:p.print_size or “3840x2160”}
db_update(“projects”,pid,{“visual_policy”:pol})
return {“visual_policy”:pol}

@app.get(”/projects/{pid}/activity”)
def list_activity(pid: str, limit: int=Query(100,ge=1,le=500), _=Depends(get_current_user)):
return {“activity”:db_list(“project_activity_logs”,project_id=pid,limit=limit)}

@app.get(”/projects/{pid}/jobs”)
def list_jobs(pid: str, _=Depends(get_current_user)):
return {“jobs”:db_list(“agent_jobs”,project_id=pid)}

@app.post(”/projects/{pid}/orchestrate”)
def orchestrate(pid: str, p: OrchestrateIn, u=Depends(get_current_user)):
uid=str(u[“id”]); proj=db_get(“projects”,id=pid,user_id=uid)
if not proj: raise HTTPException(404,“Project not found”)
summary={“queued_at”:now_iso(),“queue_3d”:p.queue_3d,“queue_video”:p.queue_video,
“queue_cad”:p.queue_cad,“queue_manuals”:p.queue_manuals}
db_update(“projects”,pid,{“orchestration_data”:summary})
return {“message”:“Orchestration updated”,“orchestration”:summary}

# ══════════════════════════════════════════════════════════════════════════════

# ENTRY POINT — must bind to PORT for Render

# ══════════════════════════════════════════════════════════════════════════════

if **name** == “**main**”:
import uvicorn
print(f”Starting server on 0.0.0.0:{PORT}”)
uvicorn.run(app, host=“0.0.0.0”, port=PORT, workers=1)