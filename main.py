import os
import re
import json
import uuid
import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager

import psycopg
from psycopg.rows import dict_row

from dotenv import load_dotenv
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles

from openai import OpenAI


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
SECRET_KEY = os.getenv("SECRET_KEY", "").strip()
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4.1").strip()
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()] or ["*"]
ACCESS_TOKEN_HOURS = int(os.getenv("ACCESS_TOKEN_HOURS", "24"))
JWT_ALGORITHM = "HS256"

MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "/tmp/ai_creative_media"))
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

client: Optional[OpenAI] = None
pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


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


def require_db_url() -> str:
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL missing")
    return DATABASE_URL


def get_conn():
    return psycopg.connect(require_db_url(), row_factory=dict_row, autocommit=True)


def db_json(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, (dict, list)) else value


def create_tables() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                create table if not exists public.users (
                    id uuid primary key,
                    email text unique not null,
                    password text not null,
                    full_name text,
                    role text default 'user',
                    is_active boolean default true,
                    created_at timestamptz default now()
                );
                """
            )
            cur.execute(
                """
                create table if not exists public.projects (
                    id uuid primary key,
                    user_id uuid not null references public.users(id) on delete cascade,
                    name text default 'Untitled Project',
                    event_type text,
                    style_direction text,
                    status text default 'draft',
                    brief text,
                    analysis text,
                    concepts jsonb,
                    selected jsonb,
                    sound_data jsonb,
                    lighting_data jsonb,
                    showrunner_data jsonb,
                    department_outputs jsonb,
                    created_at timestamptz default now(),
                    updated_at timestamptz default now()
                );
                """
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
    create_tables()
    yield


app = FastAPI(title="AI Creative Studio API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


def require_openai() -> OpenAI:
    if not client:
        raise RuntimeError("OpenAI not configured")
    return client


def hash_password(password: str) -> str:
    return pwd.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd.verify(password, hashed)


def create_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "iat": int(now_utc().timestamp()),
        "exp": int((now_utc() + datetime.timedelta(hours=ACCESS_TOKEN_HOURS)).timestamp()),
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


def get_current_user_id(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> str:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return decode_token(credentials.credentials)


class UserInput(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None


class CreateProjectInput(BaseModel):
    title: Optional[str] = None
    brief: str = Field(min_length=3)
    event_type: Optional[str] = None
    style_direction: Optional[str] = None


class RunInput(BaseModel):
    text: str = Field(min_length=3)
    project_id: Optional[str] = None
    name: Optional[str] = None
    event_type: Optional[str] = None


class SelectConceptInput(BaseModel):
    project_id: str
    index: int


class DepartmentPDFRequest(BaseModel):
    title: Optional[str] = None


def create_user(email: str, password: str, full_name: Optional[str]) -> Dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            user_id = str(uuid.uuid4())
            cur.execute(
                """
                insert into public.users (id, email, password, full_name)
                values (%s, %s, %s, %s)
                returning id, email, full_name, role, is_active, created_at
                """,
                (user_id, email.lower().strip(), hash_password(password), full_name),
            )
            return dict(cur.fetchone())


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select id, email, password, full_name, role, is_active, created_at
                from public.users
                where lower(email) = lower(%s)
                """,
                (email.strip(),),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select id, email, full_name, role, is_active, created_at
                from public.users
                where id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def create_project(user_id: str, name: str, brief: str, event_type: Optional[str], style_direction: Optional[str] = None) -> Dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            project_id = str(uuid.uuid4())
            cur.execute(
                """
                insert into public.projects (
                    id, user_id, name, event_type, style_direction, status, brief
                )
                values (%s, %s, %s, %s, %s, %s, %s)
                returning *
                """,
                (project_id, user_id, name or "Untitled Project", event_type, style_direction, "draft", brief),
            )
            return normalize_project(dict(cur.fetchone()))


def update_project(project_id: str, updates: Dict[str, Any]) -> None:
    if not updates:
        return

    allowed = {
        "name",
        "event_type",
        "style_direction",
        "status",
        "brief",
        "analysis",
        "concepts",
        "selected",
        "sound_data",
        "lighting_data",
        "showrunner_data",
        "department_outputs",
    }

    pairs = []
    values = []

    for key, value in updates.items():
        if key not in allowed:
            continue
        pairs.append(f"{key} = %s")
        values.append(db_json(value))

    if not pairs:
        return

    pairs.append("updated_at = now()")
    values.append(project_id)

    query = f"update public.projects set {', '.join(pairs)} where id = %s"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, tuple(values))


def normalize_project(project: Dict[str, Any]) -> Dict[str, Any]:
    for key in ["concepts", "selected", "sound_data", "lighting_data", "showrunner_data", "department_outputs"]:
        project[key] = safe_json(project.get(key))
    return project


def get_project(project_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("select * from public.projects where id = %s", (project_id,))
            row = cur.fetchone()
            return normalize_project(dict(row)) if row else None


def list_projects(user_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select * from public.projects where user_id = %s order by created_at desc",
                (user_id,),
            )
            return [normalize_project(dict(r)) for r in cur.fetchall()]


def best_project_name_from_text(text: str) -> str:
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return first[:80] if first else "Untitled Project"


def fallback_analysis(brief: str, name: str, event_type: Optional[str]) -> str:
    return (
        f"**Project Analysis: {name}**\n"
        f"- **Event Type:** {event_type or 'Not specified'}\n"
        f"- **Primary Goal:** Create a high-impact, immersive experience based on the brief.\n"
        f"- **Audience:** VIP guests, media, brand partners, and invited attendees.\n"
        f"- **Creative Direction:** Premium staging, controlled show flow, strong sound and lighting.\n"
        f"- **Production Needs:** Stage, reveal moment, content screens, cue management, rehearsal planning.\n"
        f"- **Sound Needs:** Music playback, announcement mics, reveal cues, audience coverage.\n"
        f"- **Lighting Needs:** Entrance looks, reveal looks, keynote looks, ambient luxury mood.\n"
        f"- **Show Flow Needs:** Tight run-of-show, standby calls, transition timing, risk handling.\n"
        f"- **Brief:** {brief}"
    )


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
    raw = llm_text(system_prompt, user_prompt, temperature=0.3)
    cleaned = raw.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def analyze_brief(brief: str, name: str, event_type: Optional[str]) -> str:
    try:
        return llm_text(
            "You are a senior event creative director. Be structured, practical, and concise.",
            f"""
Analyze this event brief and return:
1. event objective
2. audience profile
3. creative direction
4. production requirements
5. sound requirements
6. lighting requirements
7. show-running requirements
8. risks and missing clarifications

Project name: {name}
Event type: {event_type or 'Not specified'}
Brief:
{brief}
""",
            temperature=0.4,
        )
    except Exception:
        return fallback_analysis(brief, name, event_type)


def fallback_concepts() -> List[Dict[str, Any]]:
    return [
        {
            "name": "Prestige Reveal",
            "summary": "Luxury black-and-gold reveal stage with cinematic product entry and VIP-focused visual language.",
            "style": "Luxury / Premium",
            "materials": ["Gloss stage", "Metal trims", "LED surfaces"],
            "colors": ["Black", "Gold", "Warm white"],
            "lighting": "Controlled beams, soft ambient luxury wash, dramatic reveal burst",
            "stage_elements": ["Hero platform", "Curved LED", "Brand portal", "VIP entry axis"],
            "camera_style": "Cinematic, slow push-ins, wide reveal frames",
        },
        {
            "name": "Future Motion",
            "summary": "High-tech immersive launch environment with clean geometry, sharp motion content, and futuristic cue timing.",
            "style": "Futuristic / Tech",
            "materials": ["LED mesh", "Matte metallic", "Gloss acrylic"],
            "colors": ["Electric blue", "White", "Graphite"],
            "lighting": "Precision beams, cool edge light, synchronized motion looks",
            "stage_elements": ["Angular stage", "Motion tunnel", "Side wings", "Content spine"],
            "camera_style": "Dynamic tracking, symmetrical wides, fast hero cuts",
        },
        {
            "name": "Iconic Brand Theatre",
            "summary": "Balanced premium stage system combining keynote, reveal, media moments, and clear show control.",
            "style": "Brand Theatre",
            "materials": ["Scenic fascia", "Integrated LED", "Textured scenic panels"],
            "colors": ["Brand-led palette", "Neutral dark base", "Highlight accents"],
            "lighting": "Flexible keynote base looks with scalable reveal moments",
            "stage_elements": ["Main stage", "Presenter zone", "Reveal center", "Photo-op area"],
            "camera_style": "Broadcast-friendly, keynote-safe, media-ready angles",
        },
    ]


def generate_concepts(analysis: str) -> List[Dict[str, Any]]:
    try:
        result = llm_json(
            "Return only valid JSON.",
            f"""
Based on this event analysis, return exactly 3 concept options as a JSON array.
Each object must contain:
name, summary, style, materials, colors, lighting, stage_elements, camera_style

Analysis:
{analysis}
""",
        )
        if isinstance(result, list) and result:
            return result[:3]
    except Exception:
        pass
    return fallback_concepts()


def fallback_sound(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "concept": "Premium launch audio system for speeches, reveal cues, playback, and audience coverage.",
        "system_design": "Main PA + front fills + delay strategy as per venue depth.",
        "speaker_plan": ["L/R mains", "Front fills", "Stage monitors", "Playback control"],
        "input_list": ["Host mic", "Presenter mic", "Playback system", "Backup playback"],
        "mic_plan": ["2 wireless handheld", "2 lavaliers", "1 podium backup"],
        "patch_sheet": ["CH1 Host HH", "CH2 Presenter HH", "CH3 Lav 1", "CH4 Lav 2", "CH5 Playback L", "CH6 Playback R"],
        "playback_cues": ["Walk-in music", "Brand film", "Reveal sting", "Closing music"],
        "staffing": ["FOH Engineer", "Monitor/Playback Operator", "Audio Technician"],
        "rehearsal_notes": ["Check speech levels", "Confirm reveal sting timing", "Run full cue rehearsal"],
        "risk_notes": ["Keep spare mic ready", "Maintain backup playback source", "Pre-ring venue before doors"],
        "pdf_sections": [
            {"heading": "Sound Concept", "body": "Premium launch audio design covering speeches, media, and reveal moments."},
            {"heading": "System Design", "body": "Use a distributed PA strategy, clear speech reinforcement, and backup playback routing."},
            {"heading": "Cue Notes", "body": "Rehearse walk-in, reveal sting, keynote transitions, and closing music with stage manager."},
        ],
    }


def fallback_lighting(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "concept": "Luxury keynote + reveal lighting system with high contrast and polished scenic emphasis.",
        "fixture_list": ["Profile fixtures", "Wash fixtures", "Beam fixtures", "Audience blinders", "House practical integration"],
        "truss_plan": ["Front truss", "Mid truss", "Back truss", "Side ladders"],
        "dmx_notes": ["Separate keynote universes from reveal effects", "Keep house-light integration isolated"],
        "scene_cues": ["Guest ingress", "Host opening", "Keynote look", "Reveal blackout", "Reveal hit", "Photo-op"],
        "looks": ["Warm premium wash", "Clean keynote white", "Dramatic silhouette", "Reveal burst"],
        "operator_notes": ["Keep presenter faces clean", "Avoid screen spill", "Time reveal hit to audio sting"],
        "rehearsal_notes": ["Focus key presenter positions", "Check camera-safe levels", "Run reveal timing with audio"],
        "fallback_plan": ["Safe keynote look", "Manual reveal cue", "Reduced effect mode if content sync fails"],
        "pdf_sections": [
            {"heading": "Lighting Concept", "body": "Premium beam architecture with clean keynote looks and dramatic reveal moments."},
            {"heading": "Cue Structure", "body": "Build separate looks for ingress, keynote, blackout, reveal hit, and media interaction."},
            {"heading": "Fallback Plan", "body": "Maintain a safe keynote look and manual reveal trigger if timecode or content sync fails."},
        ],
    }


def fallback_showrunner(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "running_order": ["Doors open", "Guest seating", "Opening AV", "Host intro", "Leadership speech", "Reveal cue", "Product walkaround", "Media moment", "Closing"],
        "cue_script": ["Standby opening AV", "Go opening AV", "Standby host walk-in", "Go host", "Standby reveal", "Go reveal"],
        "standby_calls": ["Standby sound", "Standby lights", "Standby playback", "Standby presenters"],
        "go_calls": ["Go AV", "Go host", "Go reveal", "Go closing track"],
        "departmental_dependencies": ["Reveal depends on audio sting + blackout + scenic clear", "Media moment depends on photo light state"],
        "delay_protocol": ["Hold guests in lounge", "Loop ambience track", "Move keynote earlier if reveal setup delays"],
        "emergency_protocol": ["Mic failure swap", "Playback backup source", "Safe worklight state if technical stop needed"],
        "rehearsal_flow": ["Tech check", "Presenter rehearsal", "Cue-to-cue", "Full dress run"],
        "console_cues": ["Cue 1 Doors", "Cue 2 Host", "Cue 3 Speech", "Cue 4 Reveal", "Cue 5 Closing"],
        "pdf_sections": [
            {"heading": "Running Order", "body": "Build a disciplined run-of-show with clear standby and go calls for each transition."},
            {"heading": "Cue Calling", "body": "All reveal-critical cues must be rehearsed with playback, lighting, and stage reset timing."},
            {"heading": "Risk Handling", "body": "Prepare hold cues, backup playback, spare microphones, and manual show continuation logic."},
        ],
    }


def generate_sound_department(brief: str, project: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = llm_json(
            "Return only valid JSON. You are a senior live event sound engineer.",
            f"""
Create a full sound department plan for this project.
Return JSON with:
concept, system_design, speaker_plan, input_list, mic_plan, patch_sheet, playback_cues, staffing, rehearsal_notes, risk_notes, pdf_sections

Project:
{json.dumps(project, default=str)}
Brief:
{brief}
""",
        )
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return fallback_sound(project)


def generate_lighting_department(brief: str, project: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = llm_json(
            "Return only valid JSON. You are a senior live event lighting designer.",
            f"""
Create a full lighting department plan for this project.
Return JSON with:
concept, fixture_list, truss_plan, dmx_notes, scene_cues, looks, operator_notes, rehearsal_notes, fallback_plan, pdf_sections

Project:
{json.dumps(project, default=str)}
Brief:
{brief}
""",
        )
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return fallback_lighting(project)


def generate_showrunner_department(brief: str, project: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = llm_json(
            "Return only valid JSON. You are a senior show caller and stage manager.",
            f"""
Create a full show-running plan for this project.
Return JSON with:
running_order, cue_script, standby_calls, go_calls, departmental_dependencies, delay_protocol, emergency_protocol, rehearsal_flow, console_cues, pdf_sections

Project:
{json.dumps(project, default=str)}
Brief:
{brief}
""",
        )
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return fallback_showrunner(project)


def create_simple_pdf(title: str, sections: List[Dict[str, Any]], filename_prefix: str) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    out_dir = MEDIA_DIR / "pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename_prefix}_{uuid.uuid4().hex}.pdf"

    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

    for section in sections:
        story.append(Paragraph(str(section.get("heading", "")), styles["Heading2"]))
        story.append(Spacer(1, 6))
        body = str(section.get("body", "")).replace("\n", "<br/>")
        story.append(Paragraph(body, styles["BodyText"]))
        story.append(Spacer(1, 12))

    SimpleDocTemplate(str(out_path), pagesize=A4).build(story)
    return f"/media/pdfs/{out_path.name}"


@app.get("/")
def root():
    return {"message": "AI Creative Studio API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/signup")
def signup(payload: UserInput):
    existing = get_user_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    user = create_user(payload.email, payload.password, payload.full_name)
    token = create_token(str(user["id"]))

    return {
        "message": "User created",
        "user_id": str(user["id"]),
        "access_token": token,
        "token": token,
        "token_type": "bearer",
    }


@app.post("/login")
def login(payload: UserInput):
    user = get_user_by_email(payload.email)
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    if not verify_password(payload.password, user["password"]):
        raise HTTPException(status_code=400, detail="Wrong password")

    token = create_token(str(user["id"]))

    return {
        "access_token": token,
        "token": token,
        "token_type": "bearer",
        "user_id": str(user["id"]),
    }


@app.get("/me")
def me(user_id: str = Depends(get_current_user_id)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user": user}


@app.post("/projects")
def projects_create(payload: CreateProjectInput, user_id: str = Depends(get_current_user_id)):
    project = create_project(
        user_id=user_id,
        name=payload.title or best_project_name_from_text(payload.brief),
        brief=payload.brief,
        event_type=payload.event_type,
        style_direction=payload.style_direction,
    )
    return project


@app.get("/projects")
def projects_list(user_id: str = Depends(get_current_user_id)):
    return {"projects": list_projects(user_id)}


@app.get("/projects/{project_id}")
@app.get("/project/{project_id}")
def project_detail(project_id: str, user_id: str = Depends(get_current_user_id)):
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if str(project["user_id"]) != str(user_id):
        raise HTTPException(status_code=403, detail="Not allowed")

    return {"project": project}


@app.post("/run")
@app.post("/project/run")
def run_pipeline(payload: RunInput, user_id: str = Depends(get_current_user_id)):
    project = None

    if payload.project_id:
        project = get_project(payload.project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        if str(project["user_id"]) != str(user_id):
            raise HTTPException(status_code=403, detail="Not allowed")
        project_id = payload.project_id
        if payload.text and not project.get("brief"):
            update_project(project_id, {"brief": payload.text})
            project = get_project(project_id)
    else:
        project = create_project(
            user_id=user_id,
            name=payload.name or best_project_name_from_text(payload.text),
            brief=payload.text,
            event_type=payload.event_type,
            style_direction=None,
        )
        project_id = str(project["id"])

    if not project:
        project = get_project(project_id)

    try:
        analysis = project.get("analysis")
        if not analysis:
            analysis = analyze_brief(
                project.get("brief") or payload.text,
                project.get("name") or payload.name or "Untitled Project",
                project.get("event_type") or payload.event_type,
            )
            update_project(project_id, {"analysis": analysis, "status": "analysis_ready"})

        project = get_project(project_id)
        concepts = project.get("concepts")
        if not concepts:
            concepts = generate_concepts(project.get("analysis") or analysis)
            update_project(project_id, {"concepts": concepts, "status": "concepts_ready"})

        project = get_project(project_id)

        return {
            "message": "Pipeline completed",
            "project_id": project_id,
            "status": "concepts_ready",
            "brief": project.get("brief"),
            "analysis": project.get("analysis"),
            "concepts": project.get("concepts") or [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")


@app.post("/select")
@app.post("/project/select")
def select_concept(payload: SelectConceptInput, user_id: str = Depends(get_current_user_id)):
    project = get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if str(project["user_id"]) != str(user_id):
        raise HTTPException(status_code=403, detail="Not allowed")

    concepts = project.get("concepts") or []
    if not isinstance(concepts, list) or payload.index < 0 or payload.index >= len(concepts):
        raise HTTPException(status_code=400, detail="Invalid concept index")

    selected = concepts[payload.index]
    update_project(payload.project_id, {"selected": selected, "status": "concept_selected"})
    project = get_project(payload.project_id)

    return {
        "message": "Concept selected",
        "index": payload.index,
        "selected": selected,
        "project": project,
    }


@app.post("/project/{project_id}/departments/build")
@app.post("/project/project/{project_id}/departments/build")
def build_departments(project_id: str, user_id: str = Depends(get_current_user_id)):
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if str(project["user_id"]) != str(user_id):
        raise HTTPException(status_code=403, detail="Not allowed")

    if not project.get("selected"):
        raise HTTPException(status_code=400, detail="Select a concept first")

    brief = project.get("brief") or ""
    sound = generate_sound_department(brief, project)
    lighting = generate_lighting_department(brief, project)
    showrunner = generate_showrunner_department(brief, project)
    outputs = {
        "sound_ready": True,
        "lighting_ready": True,
        "showrunner_ready": True,
        "console_index": 0,
    }

    update_project(
        project_id,
        {
            "sound_data": sound,
            "lighting_data": lighting,
            "showrunner_data": showrunner,
            "department_outputs": outputs,
            "status": "departments_ready",
        },
    )

    return {
        "message": "Departments generated",
        "project_id": project_id,
        "sound_data": sound,
        "lighting_data": lighting,
        "showrunner_data": showrunner,
        "department_outputs": outputs,
    }


@app.post("/project/{project_id}/departments/pdf/sound")
@app.post("/project/project/{project_id}/departments/pdf/sound")
def export_sound_pdf(
    project_id: str,
    payload: Optional[DepartmentPDFRequest] = Body(default=None),
    user_id: str = Depends(get_current_user_id),
):
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if str(project["user_id"]) != str(user_id):
        raise HTTPException(status_code=403, detail="Not allowed")

    if not project.get("sound_data"):
        raise HTTPException(status_code=404, detail="Sound data not found. Build departments first.")

    title = (payload.title if payload else None) or "Sound Design Manual"
    sections = project["sound_data"].get("pdf_sections") or [
        {"heading": "Sound Design Manual", "body": json.dumps(project["sound_data"], indent=2)}
    ]

    return {"project_id": project_id, "pdf_url": create_simple_pdf(title, sections, "sound_manual")}


@app.post("/project/{project_id}/departments/pdf/lighting")
@app.post("/project/project/{project_id}/departments/pdf/lighting")
def export_lighting_pdf(
    project_id: str,
    payload: Optional[DepartmentPDFRequest] = Body(default=None),
    user_id: str = Depends(get_current_user_id),
):
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if str(project["user_id"]) != str(user_id):
        raise HTTPException(status_code=403, detail="Not allowed")

    if not project.get("lighting_data"):
        raise HTTPException(status_code=404, detail="Lighting data not found. Build departments first.")

    title = (payload.title if payload else None) or "Lighting Design Manual"
    sections = project["lighting_data"].get("pdf_sections") or [
        {"heading": "Lighting Design Manual", "body": json.dumps(project["lighting_data"], indent=2)}
    ]

    return {"project_id": project_id, "pdf_url": create_simple_pdf(title, sections, "lighting_manual")}


@app.post("/project/{project_id}/departments/pdf/showrunner")
@app.post("/project/project/{project_id}/departments/pdf/showrunner")
def export_showrunner_pdf(
    project_id: str,
    payload: Optional[DepartmentPDFRequest] = Body(default=None),
    user_id: str = Depends(get_current_user_id),
):
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if str(project["user_id"]) != str(user_id):
        raise HTTPException(status_code=403, detail="Not allowed")

    if not project.get("showrunner_data"):
        raise HTTPException(status_code=404, detail="Show runner data not found. Build departments first.")

    title = (payload.title if payload else None) or "Show Running Script"
    sections = project["showrunner_data"].get("pdf_sections") or [
        {"heading": "Show Running Script", "body": json.dumps(project["showrunner_data"], indent=2)}
    ]

    return {"project_id": project_id, "pdf_url": create_simple_pdf(title, sections, "showrunner_manual")}
