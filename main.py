# =============================================================================
# BriefCraftAI Brain  —  Production Backend
# Perfect Indian Teacher: Intro → Chapter Intro → Teaching → Quiz → Done
# =============================================================================
import os
import re
import time
import uuid
import json
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── FastAPI ────────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── OpenAI  (pre-init all vars — safe on any Python version) ──────────────
_OpenAI = None          # type: ignore
_openai_client = None
try:
    from openai import OpenAI as _OpenAI
except Exception as _e:
    print(f"[WARN] openai import failed: {_e}")

# ── Supabase  (pre-init all vars — safe on any Python version) ────────────
_SUPABASE_OK = False
_sb_create = None       # type: ignore
try:
    from supabase import create_client as _sb_create
    _SUPABASE_OK = True
except Exception as _e:
    print(f"[WARN] supabase import failed: {_e}")

# =============================================================================
# ENV CONFIG
# =============================================================================
AUDIO_DIR     = os.getenv("AUDIO_DIR", "audio_files")
CAD_PRO_DIR   = os.getenv("CAD_PRO_DIR", "media/cad_pro")
GENERATED_DIR = os.getenv("GENERATED_DIR", "media/generated")
RENDER_DOMAIN = os.getenv("RENDER_EXTERNAL_HOSTNAME", "localhost:8000")
OPENAI_KEY    = os.getenv("OPENAI_API_KEY", "")
SB_URL        = os.getenv("SUPABASE_URL", "")
SB_KEY        = os.getenv("SUPABASE_SERVICE_KEY", "")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(CAD_PRO_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

# Initialise OpenAI client safely
def _get_openai():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    if _OpenAI and OPENAI_KEY:
        try:
            _openai_client = _OpenAI(api_key=OPENAI_KEY)
        except Exception as e:
            print(f"[WARN] OpenAI init: {e}")
    return _openai_client

# Initialise Supabase client safely
_sb_client = None
def _get_sb():
    global _sb_client
    if _sb_client is not None:
        return _sb_client
    if _SUPABASE_OK and _sb_create and SB_URL and SB_KEY:
        try:
            _sb_client = _sb_create(SB_URL, SB_KEY)
            print("[DB] Supabase connected")
        except Exception as e:
            print(f"[WARN] Supabase init: {e}")
    return _sb_client

# =============================================================================
# APP  (middleware BEFORE mount to avoid Starlette ordering issues)
# =============================================================================
app = FastAPI(title="BriefCraftAI Brain", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
app.mount("/media/cad_pro", StaticFiles(directory=CAD_PRO_DIR), name="cad_pro_media")
app.mount("/media/generated", StaticFiles(directory=GENERATED_DIR), name="generated_media")

try:
    from cad_engine_pro import router as cad_pro_router
    app.include_router(cad_pro_router)
    print("[CAD] BriefCraft-AI Professional CAD router mounted at /api/cad/pro")
except Exception as _e:
    print(f"[WARN] CAD Pro router not mounted: {_e}")

# =============================================================================
# SESSION STORE
# =============================================================================
SESSIONS: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class StartSession(BaseModel):
    session_id:          Optional[str] = None
    board:               Optional[str] = None
    class_level:         Optional[str] = None
    subject:             Optional[str] = None
    book_name:           Optional[str] = None
    chapter_name:        Optional[str] = None
    chapter_title:       Optional[str] = None
    chapter_code:        Optional[str] = None
    syllabus_chapter_id: Optional[str] = None
    language:            Optional[str] = None

class StudentReply(BaseModel):
    session_id:          str
    message:             str
    board:               Optional[str] = None
    class_level:         Optional[str] = None
    subject:             Optional[str] = None
    book_name:           Optional[str] = None
    chapter_name:        Optional[str] = None
    chapter_title:       Optional[str] = None
    chapter_code:        Optional[str] = None
    syllabus_chapter_id: Optional[str] = None

class TTSRequest(BaseModel):
    text:             str
    voice:            Optional[str] = None
    model:            Optional[str] = None
    instructions:     Optional[str] = None
    subject:          Optional[str] = None
    teacher_gender:   Optional[str] = "female"
    preferred_language: Optional[str] = None

class HomeworkRequest(BaseModel):
    session_id: str
    question:   str
    subject:    Optional[str] = None
    board:      Optional[str] = None
    class_level: Optional[str] = None
    chapter_name: Optional[str] = None
    chapter_title: Optional[str] = None
    chapter_code: Optional[str] = None
    syllabus_chapter_id: Optional[str] = None
    book_name: Optional[str] = None

class UCDFileMeta(BaseModel):
    filename: Optional[str] = None
    content_type: Optional[str] = None
    url: Optional[str] = None
    size_bytes: Optional[int] = None

class UCDChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    project_id: Optional[str] = "demo-project"
    title: Optional[str] = None
    file: Optional[UCDFileMeta] = None
    context: Dict[str, Any] = {}

class UCDChatResponse(BaseModel):
    ok: bool = True
    session_id: str
    intent: str
    message: str
    questions: List[str] = []
    ui: Dict[str, Any] = {}
    agent: Dict[str, Any] = {}
    next_actions: List[Dict[str, Any]] = []


class AgentRunRequest(BaseModel):
    agent_id: str
    message: Optional[str] = ""
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Dict[str, Any] = {}


class AccountAgentRequest(BaseModel):
    message: Optional[str] = ""
    context: Dict[str, Any] = {}


class CreditConsumeRequest(BaseModel):
    amount: int
    reason: Optional[str] = "usage"
    agent_id: Optional[str] = None
    project_id: Optional[str] = None


class CheckoutRequest(BaseModel):
    package_id: str
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None


class ProjectCreateRequest(BaseModel):
    title: Optional[str] = None
    project_name: Optional[str] = None
    brief: Optional[str] = None
    brief_text: Optional[str] = None
    event_type: Optional[str] = None
    style_direction: Optional[str] = None


class ProjectRunRequest(BaseModel):
    text: Optional[str] = None
    brief: Optional[str] = None
    event_type: Optional[str] = None
    style_direction: Optional[str] = None
    user_name: Optional[str] = None
    deliverables: List[str] = []
    budget_range: Optional[str] = None
    context: Dict[str, Any] = {}


class ConceptSelectRequest(BaseModel):
    concept_index: int = 0


class MoodboardGenerateRequest(BaseModel):
    project_id: str
    concept_index: Optional[int] = 0
    count: Optional[int] = 6
    brief: Optional[str] = None
    concept: Dict[str, Any] = {}


class AssetGenerateRequest(BaseModel):
    project_id: str
    concept_index: Optional[int] = 0
    prompt: Optional[str] = None
    format: Optional[str] = None
    size: Optional[str] = None
    count: Optional[int] = 1
    context: Dict[str, Any] = {}


class PresentationBuildRequest(BaseModel):
    project_id: str
    concept_index: Optional[int] = 0
    brand_logo_url: Optional[str] = None
    template: Optional[str] = "premium pitch"
    context: Dict[str, Any] = {}


# =============================================================================
# ACCOUNT, CREDIT, PACKAGE, AND AGENT REGISTRY
# =============================================================================
# These helpers are intentionally self-contained and graceful without a database.
# If Supabase tables are added later, the endpoint contracts can remain stable and
# the persistence layer can be swapped underneath.
DEFAULT_CREDIT_GRANT = int(os.getenv("DEFAULT_CREDIT_GRANT", "2500"))
PACKAGE_PAYMENT_LINKS: Dict[str, str] = {}
try:
    import json as _json
    PACKAGE_PAYMENT_LINKS = _json.loads(os.getenv("PACKAGE_PAYMENT_LINKS", "{}") or "{}")
except Exception:
    PACKAGE_PAYMENT_LINKS = {}

ACCOUNT_STORE: Dict[str, Dict[str, Any]] = {}
CREDIT_LEDGER: Dict[str, List[Dict[str, Any]]] = {}
AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {}
PROJECT_STORE: Dict[str, Dict[str, Any]] = {}
PROJECT_ASSETS: Dict[str, List[Dict[str, Any]]] = {}
PROJECT_JOBS: Dict[str, List[Dict[str, Any]]] = {}

AGENT_HOURLY_RATES_INR: Dict[str, int] = {
    "CONCEPT_AGENT": 3000,
    "MOODBOARD_AGENT": 5000,
    "GRAPHICS_2D_AGENT": 7000,
    "RENDER_3D_AGENT": 10000,
    "SOUND_AGENT": 5000,
    "ELECTRIC_ENGINEER_AGENT": 5000,
    "LIGHTING_AGENT": 5000,
    "CAD_AGENT": 5000,
    "AV_AGENT": 5000,
    "SHOW_RUNNER_AGENT": 5000,
    "PRESENTATION_AGENT": 7000,
    "UCD_AGENT": 8000,
}

LOW_BALANCE_THRESHOLD = int(os.getenv("LOW_BALANCE_THRESHOLD", "1000"))

PACKAGE_PLANS: List[Dict[str, Any]] = [
    {
        "id": "individual_starter",
        "audience": "individual",
        "name": "Individual Starter",
        "price_inr": 999,
        "billing": "monthly",
        "credits": 12000,
        "features": [
            "AI Studio access",
            "Project Brief, Concepts, Moodboard",
            "2D/3D/CAD generations with credit limits",
            "Standard support",
        ],
        "recommended": False,
    },
    {
        "id": "individual_pro",
        "audience": "individual",
        "name": "Individual Pro",
        "price_inr": 2499,
        "billing": "monthly",
        "credits": 40000,
        "features": [
            "Everything in Starter",
            "Show Runner, Sound, Lighting agents",
            "Priority generation queue",
            "Export-ready package downloads",
        ],
        "recommended": True,
    },
    {
        "id": "institution_team",
        "audience": "institution",
        "name": "Institution Team",
        "price_inr": 14999,
        "billing": "monthly",
        "credits": 300000,
        "features": [
            "Multi-user institution workspace",
            "Shared credit wallet",
            "Agent access controls",
            "Monthly usage report",
        ],
        "recommended": True,
    },
    {
        "id": "institution_enterprise",
        "audience": "institution",
        "name": "Institution Enterprise",
        "price_inr": None,
        "billing": "custom",
        "credits": 1500000,
        "features": [
            "Dedicated workspace",
            "Custom backend agents",
            "Higher credit limits",
            "Onboarding and priority support",
        ],
        "recommended": False,
    },
]


def _account_user_id(
    request: Request,
    x_user_id: Optional[str] = None,
    authorization: Optional[str] = None,
) -> str:
    """Return a stable account id without assuming a specific auth provider."""
    explicit = (x_user_id or "").strip()
    if explicit:
        return explicit
    token = (authorization or request.headers.get("authorization") or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if token:
        return "token-" + uuid.uuid5(uuid.NAMESPACE_URL, token).hex[:24]
    return "guest"


def _ensure_account(user_id: str) -> Dict[str, Any]:
    if user_id not in ACCOUNT_STORE:
        ACCOUNT_STORE[user_id] = {
            "user_id": user_id,
            "credit_balance": DEFAULT_CREDIT_GRANT,
            "plan_id": "free_trial",
            "plan_name": "Free Trial",
            "account_type": "individual",
            "currency": "INR",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        CREDIT_LEDGER[user_id] = [
            {
                "id": str(uuid.uuid4()),
                "type": "grant",
                "amount": DEFAULT_CREDIT_GRANT,
                "balance_after": DEFAULT_CREDIT_GRANT,
                "reason": "Welcome credits",
                "ts": time.time(),
            }
        ]
    return ACCOUNT_STORE[user_id]


def _grant_credits(user_id: str, amount: int, reason: str, package_id: Optional[str] = None) -> Dict[str, Any]:
    acct = _ensure_account(user_id)
    acct["credit_balance"] = int(acct.get("credit_balance", 0)) + max(0, int(amount))
    acct["updated_at"] = time.time()
    row = {
        "id": str(uuid.uuid4()),
        "type": "grant",
        "amount": max(0, int(amount)),
        "balance_after": acct["credit_balance"],
        "reason": reason,
        "package_id": package_id,
        "ts": time.time(),
    }
    CREDIT_LEDGER.setdefault(user_id, []).append(row)
    return acct


def _consume_credits(user_id: str, amount: int, reason: str, agent_id: Optional[str] = None, project_id: Optional[str] = None) -> Dict[str, Any]:
    amount = max(0, int(amount))
    acct = _ensure_account(user_id)
    balance = int(acct.get("credit_balance", 0))
    if amount > balance:
        raise HTTPException(
            status_code=402,
            detail={
                "message": "Insufficient credits.",
                "required": amount,
                "balance": balance,
                "upgrade_endpoint": "/account/packages",
            },
        )
    acct["credit_balance"] = balance - amount
    acct["updated_at"] = time.time()
    CREDIT_LEDGER.setdefault(user_id, []).append({
        "id": str(uuid.uuid4()),
        "type": "debit",
        "amount": -amount,
        "balance_after": acct["credit_balance"],
        "reason": reason,
        "agent_id": agent_id,
        "project_id": project_id,
        "ts": time.time(),
    })
    return acct


def register_backend_agent(
    agent_id: str,
    name: str,
    description: str,
    endpoint: str,
    capabilities: Optional[List[str]] = None,
    credit_cost: int = 25,
    category: str = "creative",
    enabled: bool = True,
    hourly_rate_inr: Optional[int] = None,
) -> None:
    """Register an agent so the frontend can discover it automatically."""
    AGENT_REGISTRY[agent_id] = {
        "id": agent_id,
        "name": name,
        "description": description,
        "endpoint": endpoint,
        "capabilities": capabilities or [],
        "credit_cost": credit_cost,
        "hourly_rate_inr": hourly_rate_inr if hourly_rate_inr is not None else AGENT_HOURLY_RATES_INR.get(agent_id, 0),
        "category": category,
        "enabled": enabled,
    }


def _register_default_agents() -> None:
    agents = [
        {
            "agent_id": "ACCOUNT_AGENT",
            "name": "Account Agent",
            "description": "Manages user credits, balances, plans, upgrades, billing, and package recommendations.",
            "endpoint": "/account/agent",
            "capabilities": ["credits", "balance", "packages", "upgrade", "billing"],
            "credit_cost": 0,
            "category": "account",
        },
        {
            "agent_id": "UCD_AGENT",
            "name": "Universal Creative Director / Orchestrator",
            "description": "60-year senior creative director. Understands the brief, asks missing questions, fixes deliverables and budget logic, then supervises every specialist agent step by step.",
            "endpoint": "/ucd/chat",
            "capabilities": ["brief", "concept", "handoff", "orchestration", "agent-routing", "budget-thinking", "human-chat"],
            "credit_cost": 15,
            "category": "orchestrator",
        },
        {
            "agent_id": "BRIEF_AGENT",
            "name": "Project Brief Agent",
            "description": "Turns raw client input into a production-ready event brief with missing questions and structured requirements.",
            "endpoint": "/agents/run",
            "capabilities": ["brief", "requirements", "missing-inputs", "project-structure"],
            "credit_cost": 20,
            "category": "creative",
        },
        {
            "agent_id": "CONCEPT_AGENT",
            "name": "Concept Agent",
            "description": "45-year senior creative director. Creates multiple radically different creative routes for events, exhibitions, weddings, activations, concerts, launches, corporate events, birthdays, and any experience-led brief.",
            "endpoint": "/agents/run",
            "capabilities": ["concept", "creative-routes", "guest-journey", "hero-moments", "cross-industry-thinking", "distinct-options"],
            "credit_cost": 60,
            "category": "creative",
        },
        {
            "agent_id": "MOODBOARD_AGENT",
            "name": "Moodboard Agent",
            "description": "45-year creative director for moodboards. Explains mood, materials, lighting, ambience, seating, stage look, textures, palette and why each visual is right for the chosen concept.",
            "endpoint": "/api/moodboard/generate",
            "capabilities": ["moodboard", "visual-direction", "image-prompts", "references", "materials", "lighting", "ambience"],
            "credit_cost": 120,
            "category": "visual",
        },
        {
            "agent_id": "GRAPHICS_2D_AGENT",
            "name": "2D Graphics Art Director",
            "description": "45-year art director. Creates invite copy, registration backdrop, table facade, stage backdrop, key visual, print sizes, screen graphics and brand copy guidance.",
            "endpoint": "/agents/run",
            "capabilities": ["2d", "graphics", "key-visual", "print", "screen-content", "wayfinding"],
            "credit_cost": 100,
            "category": "visual",
        },
        {
            "agent_id": "RENDER_3D_AGENT",
            "name": "3D Render Design Agent",
            "description": "40-year 3D designer. Designs contemporary, complicated structures with accurate dimensions from CAD, stage/exhibition/wedding/event geometry and render handoff.",
            "endpoint": "/agents/run",
            "capabilities": ["3d", "renders", "scene-prompts", "spatial-visualization", "render-handoff", "dimension-accuracy", "structures"],
            "credit_cost": 180,
            "category": "visual",
        },
        {
            "agent_id": "CAD_AGENT",
            "name": "Professional CAD Agent",
            "description": "Generates or traces production CAD layouts with SVG/PDF/DXF outputs.",
            "endpoint": "/api/cad/pro/generate",
            "capabilities": ["cad", "layout", "trace", "dxf", "pdf", "svg"],
            "credit_cost": 250,
            "category": "production",
        },
        {
            "agent_id": "AV_AGENT",
            "name": "Audio Visual Agent",
            "description": "Plans LED, projection, content playback, camera, switching, and technical AV requirements.",
            "endpoint": "/agents/run",
            "capabilities": ["av", "led", "projection", "camera", "switching", "playback"],
            "credit_cost": 80,
            "category": "production",
        },
        {
            "agent_id": "SOUND_AGENT",
            "name": "Sound Agent",
            "description": "Creates sound design direction, cue notes, music beds, mic planning, and show audio requirements.",
            "endpoint": "/agents/run",
            "capabilities": ["sound", "audio", "music", "mics", "cueing"],
            "credit_cost": 70,
            "category": "production",
        },
        {
            "agent_id": "LIGHTING_AGENT",
            "name": "Senior Lighting Designer",
            "description": "Creates lighting design direction, scene looks, cue stacks, reveal moments, and fixture logic.",
            "endpoint": "/agents/run",
            "capabilities": ["lighting", "cues", "fixture-logic", "reveal", "atmosphere"],
            "credit_cost": 90,
            "category": "production",
        },
        {
            "agent_id": "ELECTRIC_ENGINEER_AGENT",
            "name": "Electric Engineer",
            "description": "Plans power distribution, DB positions, cable routes, load notes, safety checks and electrical coordination.",
            "endpoint": "/agents/run",
            "capabilities": ["electrical", "power", "db", "cable-routes", "load-calculation"],
            "credit_cost": 90,
            "category": "production",
        },
        {
            "agent_id": "PRESENTATION_AGENT",
            "name": "Client Pitch Presentation Agent",
            "description": "Senior presentation creator and content writer. Builds client-pitch decks with brief summary, concept story, moodboard, 2D, 3D, flow, logo/template placement and persuasive presenter copy.",
            "endpoint": "/projects/{project_id}/presentation/build",
            "capabilities": ["presentation", "pitch-deck", "content-writing", "client-story", "ppt"],
            "credit_cost": 120,
            "category": "export",
        },
        {
            "agent_id": "SHOW_RUNNER_AGENT",
            "name": "Show Runner Agent",
            "description": "Builds run-of-show, cue sheets, presenter flow, backstage notes, and live show calling structure.",
            "endpoint": "/agents/run",
            "capabilities": ["show-runner", "run-of-show", "cue-sheet", "stage-management", "calling-script"],
            "credit_cost": 90,
            "category": "production",
        },
        {
            "agent_id": "DOWNLOADS_AGENT",
            "name": "Downloads Agent",
            "description": "Packages approved outputs into export-ready files, decks, PDFs, briefs, and production download bundles.",
            "endpoint": "/agents/run",
            "capabilities": ["downloads", "exports", "pdf", "ppt", "production-package"],
            "credit_cost": 40,
            "category": "export",
        },
    ]

    for agent in agents:
        register_backend_agent(**agent)


_register_default_agents()


def _account_agent_reply(user_id: str, message: str) -> Dict[str, Any]:
    acct = _ensure_account(user_id)
    text = (message or "").lower()
    low_balance = int(acct.get("credit_balance", 0)) <= LOW_BALANCE_THRESHOLD
    if any(w in text for w in ["package", "plan", "upgrade", "price", "pay", "institution", "individual"]):
        msg = (
            "Here are the available packages. Individuals usually start with Individual Pro; "
            "institutions should compare Institution Team and Enterprise based on team size and monthly generation volume."
        )
        return {"message": msg, "account": acct, "packages": PACKAGE_PLANS, "hourly_rates_inr": AGENT_HOURLY_RATES_INR, "low_balance": low_balance, "action": "show_packages"}
    if any(w in text for w in ["ledger", "history", "usage", "spent", "rate", "rates", "cost"]):
        return {
            "message": "Here is your credit usage history and the current agent hourly rate card.",
            "account": acct,
            "ledger": CREDIT_LEDGER.get(user_id, [])[-50:],
            "hourly_rates_inr": AGENT_HOURLY_RATES_INR,
            "low_balance": low_balance,
            "low_balance_threshold": LOW_BALANCE_THRESHOLD,
            "action": "show_ledger",
        }
    return {
        "message": (
            f"Your current credit balance is {acct['credit_balance']} tokens on {acct['plan_name']}. "
            + ("Your balance is low, so please recharge or top up before starting heavy creative jobs." if low_balance else "I will keep refreshing this while agents work.")
        ),
        "account": acct,
        "packages": PACKAGE_PLANS,
        "hourly_rates_inr": AGENT_HOURLY_RATES_INR,
        "low_balance": low_balance,
        "low_balance_threshold": LOW_BALANCE_THRESHOLD,
        "action": "show_balance",
    }


def _dump_model(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


# =============================================================================
# BRIEFCRAFT CREATIVE ENGINE
# =============================================================================
CREATIVE_INDUSTRIES = {
    "wedding": ["wedding", "sangeet", "mehendi", "reception", "haldi", "mandap"],
    "exhibition": ["exhibition", "expo", "stall", "booth", "pavilion", "trade show"],
    "concert": ["concert", "festival", "music", "artist", "gig", "performance"],
    "corporate": ["corporate", "conference", "summit", "townhall", "award", "annual"],
    "activation": ["activation", "mall", "roadshow", "launch", "sampling", "consumer"],
    "birthday": ["birthday", "party", "kids", "anniversary"],
    "event": ["event", "gala", "show", "experience", "ceremony"],
}

CONCEPT_ARCHETYPES = [
    {
        "key": "cinematic_reveal",
        "name": "Cinematic Reveal Theatre",
        "logic": "a tightly scripted arrival-to-reveal journey with one unforgettable hero moment",
        "spatial": "central stage, controlled sightlines, strong reveal aperture, layered audience focus",
    },
    {
        "key": "immersive_district",
        "name": "Immersive Brand District",
        "logic": "a walk-through experience where every zone tells a different part of the story",
        "spatial": "entry portal, gallery moments, interaction zones, social-content anchors",
    },
    {
        "key": "living_gallery",
        "name": "Living Gallery of Moments",
        "logic": "an editorial, art-gallery style experience where guests move through composed scenes",
        "spatial": "modular scenic frames, slow discovery path, curated seating and photo compositions",
    },
    {
        "key": "ritual_celebration",
        "name": "Ritual-Led Celebration",
        "logic": "a human-centered ceremony or hospitality experience built around emotion and belonging",
        "spatial": "warm arrival, central ritual zone, layered lounges, family/VIP hospitality clusters",
    },
    {
        "key": "future_lab",
        "name": "Future Lab Experience",
        "logic": "a technology-led lab or innovation narrative with interactive reveals and luminous surfaces",
        "spatial": "LED portals, demo islands, kinetic light, high-contrast stage geometry",
    },
    {
        "key": "marketplace_festival",
        "name": "Premium Festival Marketplace",
        "logic": "a high-energy multi-touchpoint format mixing entertainment, food, product and social scenes",
        "spatial": "activity pods, performance pocket, hospitality lane, signage spine, photo-stop clusters",
    },
    {
        "key": "monumental_installation",
        "name": "Monumental Installation Story",
        "logic": "a sculptural center-piece drives the entire experience and becomes the visual memory",
        "spatial": "large central form, radial circulation, layered viewing points, controlled lighting focus",
    },
]

MOODBOARD_LENSES = [
    ("Mood & Emotion", "the emotional temperature, pacing and guest feeling"),
    ("Material Palette", "surface finishes, textures, structural finishes and tactile cues"),
    ("Lighting & Ambience", "key light, fill, practical glow, haze, temperature and reveal states"),
    ("Seating & Guest Comfort", "seating posture, lounges, VIP clusters, dining or audience ergonomics"),
    ("Stage / Spatial Look", "stage language, scenic objects, portal, backdrop and focal hierarchy"),
    ("Brand / Graphic Behaviour", "logo hierarchy, typography scale, signage, invitation and backdrop logic"),
]


def _stable_seed(*parts: Any) -> int:
    raw = "||".join(str(p or "") for p in parts)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)


def _rng_for(*parts: Any) -> random.Random:
    return random.Random(_stable_seed(*parts))


def _public_url(path: str) -> str:
    host = RENDER_DOMAIN
    scheme = "https" if host and host != "localhost:8000" else "http"
    return f"{scheme}://{host}/{path.lstrip('/')}"


def _safe_title(text: str, fallback: str = "Creative Project") -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return fallback
    text = re.split(r"[.!?\n]", text)[0].strip()
    return (text[:82].rstrip() + "…") if len(text) > 84 else text


def _detect_industry(brief: str) -> str:
    low = (brief or "").lower()
    for industry, words in CREATIVE_INDUSTRIES.items():
        if any(w in low for w in words):
            return industry
    return "event"


def _extract_guest_count(brief: str) -> Optional[str]:
    m = re.search(r"\b(\d{2,6})\s*(pax|people|guests|attendees|audience|seats)\b", brief or "", re.I)
    return f"{m.group(1)} {m.group(2)}" if m else None


def _extract_dimensions(brief: str) -> Optional[str]:
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(m|meter|meters|ft|feet|mm|in)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(m|meter|meters|ft|feet|mm|in)?\b", brief or "", re.I)
    if not m:
        return None
    return f"{m.group(1)} x {m.group(3)} {m.group(4) or m.group(2) or 'm'}"


def _extract_budget(brief: str, fallback: Optional[str] = None) -> str:
    m = re.search(r"\b(?:budget|cost|spend)\D{0,10}([\d,.]+)\s*(lakh|lakhs|cr|crore|k|rs|inr|aed|usd)?", brief or "", re.I)
    if m:
        return " ".join(x for x in m.groups() if x)
    return fallback or "Budget to be confirmed"


def _missing_brief_questions(brief: str) -> List[str]:
    checks = [
        (r"\b\d{2,6}\s*(pax|people|guests|attendees|audience|seats)\b", "How many guests / visitors / seats should I plan for?"),
        (r"\b\d+(?:\.\d+)?\s*(m|meter|meters|ft|feet|mm|in)?\s*[x×]\s*\d+", "What are the final venue or stall dimensions?"),
        (r"\b(budget|cost|spend|lakh|crore|premium|luxury|standard)\b", "What budget level should I assume: luxury, premium, standard or cost-controlled?"),
        (r"\b(logo|brand|guideline|palette|sponsor|copy|content)\b", "Do you have brand guidelines, logo files, mandatory copy or sponsor hierarchy?"),
        (r"\b(date|time|schedule|duration|show|opening|closing)\b", "What is the date, guest arrival time, show time and total duration?"),
        (r"\b(deliverable|moodboard|2d|3d|cad|sound|lighting|presentation|deck|layout)\b", "Which deliverables do you need: concepts, moodboard, 2D, 3D, CAD, sound, lighting, show runner, presentation or downloads?"),
    ]
    out = [q for pat, q in checks if not re.search(pat, brief or "", re.I)]
    return out[:5]


def _brief_context(brief: str, req: Optional[ProjectRunRequest] = None) -> Dict[str, Any]:
    industry = _detect_industry(brief)
    return {
        "title": _safe_title(brief, "Production-Ready Creative Brief"),
        "industry": industry,
        "guest_count": _extract_guest_count(brief) or "To be confirmed",
        "dimensions": _extract_dimensions(brief) or "To be confirmed",
        "budget": _extract_budget(brief, getattr(req, "budget_range", None) if req else None),
        "style_direction": (getattr(req, "style_direction", None) if req else None) or "Premium creative",
        "event_type": (getattr(req, "event_type", None) if req else None) or industry,
        "missing_questions": _missing_brief_questions(brief),
        "user_name": (getattr(req, "user_name", None) if req else None) or "there",
    }


def _creative_director_prompt(role: str, years: int, industry: str) -> str:
    return (
        f"You are a {years}-year senior {role} for {industry}, events, exhibitions, weddings, "
        "activations, concerts, corporate events, parties and brand experiences. Think like a real "
        "creative director: first understand the brief, then make precise, executable, emotionally "
        "strong decisions. Do not reuse generic concepts. Each output must feel brief-specific, "
        "commercially useful and production-aware."
    )


def _generate_concepts(brief: str, ctx: Dict[str, Any], count: int = 3) -> List[Dict[str, Any]]:
    rng = _rng_for("concepts", brief, ctx.get("industry"), ctx.get("style_direction"))
    archetypes = CONCEPT_ARCHETYPES[:]
    rng.shuffle(archetypes)
    selected = archetypes[:max(3, min(5, count))]
    verbs = ["orchestrates", "transforms", "stages", "curates", "dramatizes", "humanises", "amplifies"]
    material_sets = [
        ["brushed metal", "ribbed glass", "deep velvet", "warm champagne trims"],
        ["raw timber", "limewash texture", "linen drape", "soft brass accents"],
        ["mirror acrylic", "translucent polycarbonate", "pixel LED", "black gloss floor"],
        ["handcrafted floral structures", "embroidered textiles", "warm wood", "ceramic details"],
        ["modular truss skin", "mesh fabric", "neon edge light", "matte graphite surfaces"],
    ]
    concepts: List[Dict[str, Any]] = []
    for i, arch in enumerate(selected):
        verb = rng.choice(verbs)
        materials = material_sets[(i + rng.randrange(len(material_sets))) % len(material_sets)]
        name = f"{arch['name']} — {ctx['title'][:34].strip()}"
        one_liner = (
            f"A {ctx['style_direction']} {ctx['industry']} direction that {verb} the brief through "
            f"{arch['logic']}."
        )
        flow = [
            f"Arrival: a clear first-view moment establishes {ctx['style_direction']} tone and guest orientation.",
            f"Build-up: guests move through {arch['spatial']} with a controlled rhythm.",
            "Hero moment: lighting, sound, scenic structure and screen content converge into one memorable peak.",
            "After-moment: hospitality, photo capture and interaction zones convert the concept into social memory.",
        ]
        concepts.append({
            "id": f"concept_{i+1}_{arch['key']}",
            "name": name,
            "title": name,
            "style": arch["name"],
            "one_liner": one_liner,
            "summary": one_liner,
            "big_idea": (
                f"The concept takes the user brief and turns it into {arch['logic']}. "
                f"It avoids generic decor by giving every department one shared story: "
                f"{ctx['industry']} guests should feel the experience building from arrival to a designed peak."
            ),
            "why_best": (
                "This route is strong because it is creatively distinctive, still executable, and gives CAD, "
                "2D, 3D, lighting, sound and show-running a clear production brief. It can scale up or down "
                f"depending on the budget ({ctx['budget']})."
            ),
            "experience_flow": flow,
            "design_language": (
                f"{', '.join(materials)} with a {ctx['style_direction']} palette, controlled negative space, "
                "clear brand hierarchy and stage-picture discipline."
            ),
            "materials": materials,
            "hero_moments": [
                "First-look entry composition",
                "Main reveal or central ceremony peak",
                "High-value photo / social capture moment",
            ],
            "cad_direction": (
                f"Plan dimensions around {ctx['dimensions']}. Reserve clean guest flow, FOH sightline, "
                "BOH/service access, power routes and stage/feature object footprint before 3D starts."
            ),
            "graphics_2d_brief": {
                "invite_copy": f"You are invited to experience {ctx['title']}",
                "registration_backdrop": "Large brand mark, concept pattern, directional welcome copy and sponsor lockup.",
                "table_facade": "Low-height branded facade with premium material texture and subtle edge lighting.",
                "stage_backdrop": "Hero concept title, layered depth graphics and screen-safe logo hierarchy.",
                "sizes": ["Invite 1080x1920", "Backdrop 16ft x 8ft", "Table facade 8ft x 3ft", "Stage backdrop as per CAD"],
            },
            "render_3d_brief": (
                "Use CAD dimensions first. Model main scenic geometry, entry, seating/audience zones, "
                "stage or booth structure, lighting truss positions and camera-safe hero views."
            ),
            "lighting_brief": "Use warm key/fill for premium emotion, sharper moving light at reveal, practical glow in guest zones and dimmable states for show flow.",
            "sound_brief": "Build subtle arrival ambience, gradual musical tension, hero stinger at reveal and controlled post-moment lounge bed.",
            "production_watchouts": [
                "Confirm venue ceiling height and rigging load.",
                "Lock exact brand assets and mandatory copy.",
                "Approve power, BOH and emergency access routes before render finalization.",
            ],
        })
    return concepts


def _build_structured_brief(brief: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": ctx["title"],
        "industry": ctx["industry"],
        "executive_summary": (
            f"This is a {ctx['style_direction']} {ctx['industry']} brief. The UCD should first confirm "
            "missing commercial and technical details, then create concepts, select deliverables, and supervise "
            "specialist agents in a production-safe order."
        ),
        "objective": "Create a memorable, commercially useful, production-ready creative experience with clear deliverables and department handoffs.",
        "guest_count": ctx["guest_count"],
        "dimensions": ctx["dimensions"],
        "budget": ctx["budget"],
        "missing_questions": ctx["missing_questions"],
        "recommended_deliverables": [
            "Project Brief",
            "Concepts",
            "Mood Board",
            "CAD Layout",
            "2D Graphics",
            "3D Renders",
            "Lighting",
            "Sound",
            "Show Runner",
            "Client Pitch Presentation",
        ],
        "agent_sequence": [
            "UCD Agent confirms missing details, deliverables and budget thinking.",
            "Concept Agent creates multiple different creative routes.",
            "Moodboard Agent expands selected concept into visual/material/lighting ambience.",
            "CAD Agent fixes dimensions and layout logic before 3D.",
            "2D Art Director creates copy and graphics requirements.",
            "3D Agent renders from CAD dimensions and concept geometry.",
            "Lighting, Sound, Electrical and Show Runner agents finalize production details.",
            "Presentation Agent packages everything for client pitch.",
        ],
        "original_brief": brief,
    }


def _svg_asset(project_id: str, title: str, subtitle: str, palette: List[str], section: str, idx: int) -> str:
    fname = f"{project_id}_{section}_{idx}_{uuid.uuid4().hex[:8]}.svg"
    path = Path(GENERATED_DIR) / fname
    color_a = palette[0] if palette else "#0b0d14"
    color_b = palette[1] if len(palette) > 1 else "#c9a84c"
    color_c = palette[2] if len(palette) > 2 else "#f6d27a"
    def x(v: Any) -> str:
        return str(v or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    safe_section = x(section.upper())
    safe_title = x(str(title)[:34])
    safe_subtitle = x(str(subtitle)[:210])
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
<defs><linearGradient id="g" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stop-color="{color_a}"/><stop offset="0.55" stop-color="{color_b}"/><stop offset="1" stop-color="{color_c}"/></linearGradient></defs>
<rect width="1600" height="900" fill="#07080d"/>
<rect x="48" y="48" width="1504" height="804" rx="34" fill="url(#g)" opacity="0.86"/>
<circle cx="1260" cy="170" r="260" fill="#ffffff" opacity="0.08"/>
<circle cx="250" cy="760" r="300" fill="#000000" opacity="0.20"/>
<path d="M160 670 C420 450 640 740 890 470 C1100 245 1260 385 1450 245" fill="none" stroke="#fff7d6" stroke-width="10" opacity="0.34"/>
<text x="110" y="180" font-family="Arial, sans-serif" font-size="32" font-weight="700" fill="#fff7d6" letter-spacing="5">{safe_section}</text>
<text x="110" y="295" font-family="Arial, sans-serif" font-size="74" font-weight="800" fill="#ffffff">{safe_title}</text>
<foreignObject x="110" y="335" width="1180" height="240"><div xmlns="http://www.w3.org/1999/xhtml" style="font: 34px Arial, sans-serif; line-height:1.35; color:#fff7d6;">{safe_subtitle}</div></foreignObject>
<text x="110" y="800" font-family="Arial, sans-serif" font-size="24" fill="#fff7d6" opacity="0.82">BriefCraftAI Creative Engine</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return _public_url(f"media/generated/{fname}")


def _store_asset(project_id: str, section: str, asset_type: str, title: str, description: str, url: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    asset = {
        "id": str(uuid.uuid4()),
        "project_id": project_id,
        "section": section,
        "asset_type": asset_type,
        "title": title,
        "description": description,
        "prompt": description,
        "status": "ready",
        "preview_url": url,
        "master_url": url,
        "source_file_url": url,
        "created_at": time.time(),
        "meta": meta or {},
    }
    PROJECT_ASSETS.setdefault(project_id, []).append(asset)
    return asset


def _selected_concept(project_id: str, concept_index: Optional[int] = 0) -> Dict[str, Any]:
    project = PROJECT_STORE.get(project_id, {})
    concepts = project.get("concepts") or []
    idx = concept_index if concept_index is not None else project.get("selected_concept_index", 0)
    try:
        idx = max(0, min(len(concepts) - 1, int(idx)))
    except Exception:
        idx = 0
    return concepts[idx] if concepts else {}


def _build_moodboard_assets(project_id: str, concept_index: int = 0, count: int = 6) -> List[Dict[str, Any]]:
    concept = _selected_concept(project_id, concept_index)
    rng = _rng_for("moodboard", project_id, concept.get("name"), count)
    palettes = [
        ["#111827", "#c9a84c", "#f6d27a"],
        ["#141414", "#7b2d26", "#f4c6a6"],
        ["#08111f", "#3aa6ff", "#d8f2ff"],
        ["#102015", "#7bb274", "#dceecf"],
        ["#210b2c", "#c14a5a", "#ffd6df"],
    ]
    frames: List[Dict[str, Any]] = []
    details = [
        ("Overall Mood", "sets the emotional climate and first visual promise"),
        ("Materials", "translates concept into tactile finishes and structural surfaces"),
        ("Lighting", "defines contrast, reveal cue, warmth, shadow and ambience"),
        ("Seating", "shows guest comfort, VIP hierarchy and social behaviour"),
        ("Stage Look", "explains backdrop, scenic volume, screen focus and focal depth"),
        ("Brand Details", "shows logo behaviour, typography, facade, invitation and wayfinding tone"),
    ][: max(1, min(8, count))]
    for i, (title, role) in enumerate(details):
        palette = palettes[(i + rng.randrange(len(palettes))) % len(palettes)]
        desc = (
            f"{title} for {concept.get('name', 'selected concept')}: this frame {role}. "
            f"It is best for the concept because it supports: {concept.get('why_best', concept.get('summary', 'the creative route'))}"
        )
        url = _svg_asset(project_id, title, desc, palette, "moodboard", i + 1)
        frames.append(_store_asset(project_id, "moodboard", "moodboard", title, desc, url, {
            "concept_index": concept_index,
            "mood": title,
            "creative_director_note": desc,
        }))
    return frames


def _build_2d_assets(project_id: str, concept_index: int = 0) -> List[Dict[str, Any]]:
    concept = _selected_concept(project_id, concept_index)
    briefs = concept.get("graphics_2d_brief") or {}
    outputs = [
        ("Invite / Digital Save-the-Date", briefs.get("invite_copy", "Premium invitation headline and event details")),
        ("Registration Backdrop", briefs.get("registration_backdrop", "Welcome backdrop with brand hierarchy")),
        ("Table Facade", briefs.get("table_facade", "Reception desk facade with material and lighting detail")),
        ("Stage Backdrop", briefs.get("stage_backdrop", "Main stage graphic system and LED-safe composition")),
    ]
    assets = []
    for i, (title, desc) in enumerate(outputs):
        url = _svg_asset(project_id, title, desc, ["#0b0d14", "#c9a84c", "#ffffff"], "2d_graphics", i + 1)
        assets.append(_store_asset(project_id, "2d_graphics", "2d_graphic", title, desc, url, {
            "concept_index": concept_index,
            "art_director_note": "45-year art director output: copy, hierarchy, size and material logic included.",
            "sizes": briefs.get("sizes", []),
        }))
    return assets


def _build_3d_assets(project_id: str, concept_index: int = 0) -> List[Dict[str, Any]]:
    concept = _selected_concept(project_id, concept_index)
    views = [
        ("Entry Portal 3D View", "dimensioned first-view arrival portal based on CAD flow"),
        ("Main Stage 3D View", "stage/backdrop/LED/scenic structure with accurate footprint from CAD"),
        ("Experience Zone 3D View", "guest interaction zone with seating, circulation and light mood"),
    ]
    assets = []
    for i, (title, desc) in enumerate(views):
        full_desc = f"{desc}. 3D agent must follow CAD dimensions first: {concept.get('cad_direction', 'confirm dimensions before render')}."
        url = _svg_asset(project_id, title, full_desc, ["#07111f", "#3a6bff", "#dbeafe"], "renders", i + 1)
        assets.append(_store_asset(project_id, "renders", "3d_render", title, full_desc, url, {
            "concept_index": concept_index,
            "render_agent_note": concept.get("render_3d_brief", ""),
        }))
    return assets


def _department_outputs(project_id: str, concept_index: int = 0) -> Dict[str, Any]:
    concept = _selected_concept(project_id, concept_index)
    return {
        "sound_data": {
            "designer": "Senior sound engineer",
            "direction": concept.get("sound_brief", ""),
            "cue_logic": ["arrival bed", "anticipation build", "hero stinger", "post-reveal lounge bed"],
            "mic_plan": "Confirm MC, presenter, panel and backup microphone count.",
        },
        "lighting_data": {
            "designer": "Senior lighting designer",
            "direction": concept.get("lighting_brief", ""),
            "looks": ["arrival warmth", "brand wash", "reveal contrast", "photo moment glow"],
            "notes": "Confirm rigging height, fixture inventory, haze permission and power load.",
        },
        "showrunner_data": {
            "show_caller": "Senior show runner",
            "run_of_show": [
                "Guest arrival and holding ambience",
                "Opening cue and host welcome",
                "Concept story / brand introduction",
                "Hero reveal / central moment",
                "Guest interaction and closing handover",
            ],
            "backstage_notes": "Lock cue numbers only after final venue tech check.",
        },
        "electrical_data": {
            "engineer": "Electric engineer",
            "scope": "DB positions, power routing, emergency clearance, LED/stage/sound load assumptions.",
            "watchouts": ["separate sound and lighting power where possible", "protect guest cable crossings", "confirm generator backup"],
        },
    }


def _presentation_deck(project_id: str, concept_index: int = 0) -> Dict[str, Any]:
    project = PROJECT_STORE.get(project_id, {})
    concept = _selected_concept(project_id, concept_index)
    brief = project.get("structured_brief", {})
    return {
        "title": f"Client Pitch — {project.get('project_name', 'Creative Experience')}",
        "template": "premium dark-gold pitch",
        "slides": [
            {"title": "Brief Summary", "body": brief.get("executive_summary", project.get("brief", ""))},
            {"title": "Creative Challenge", "body": "Turn the client ask into a memorable, executable and commercially valuable experience."},
            {"title": f"Concept: {concept.get('name', 'Selected Concept')}", "body": concept.get("big_idea", "")},
            {"title": "Mood & Material Direction", "body": "Palette, ambience, materials, lighting, seating and stage treatment from the moodboard agent."},
            {"title": "2D / Brand Graphics", "body": json.dumps(concept.get("graphics_2d_brief", {}), ensure_ascii=False)},
            {"title": "CAD to 3D Logic", "body": concept.get("cad_direction", "")},
            {"title": "Show Flow", "body": "Arrival, build-up, hero moment, interaction and closing handover."},
            {"title": "Production Notes", "body": "; ".join(concept.get("production_watchouts", []))},
        ],
        "presenter_note": "Speak like a senior creative presenter: clear, persuasive, never over-explain, and always connect each output back to the client objective.",
    }

# =============================================================================
# TEXT UTILITIES
# =============================================================================
_EMOJI = ["😊","🙂","😂","😄","👍","😃","🎉","✅","❌","🎓","📚","💡","•","→","►"]
_PLACEHOLDER_FRAGMENTS = [
    "अब इस अंश को सरल भाषा में समझते हैं",
    "इस अध्याय का विषय है",
    "suno dhyan se",
]

def clean_text(text: str) -> str:
    """Strip markdown, emoji, extra whitespace for TTS."""
    if not text:
        return ""
    for ch in _EMOJI + ["*","_","#","`","~","**","__"]:
        text = text.replace(ch, "")
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def is_placeholder(text: Optional[str]) -> bool:
    if not text or len(text.strip()) < 25:
        return True
    t = text.lower()
    return any(p.lower() in t for p in _PLACEHOLDER_FRAGMENTS)

def is_krutidev(text: Optional[str]) -> bool:
    """Detect Krutidev legacy font encoding (ASCII-faked Hindi)."""
    if not text or len(text) < 15:
        return True
    dev = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    asc = sum(1 for c in text if c.isascii() and c.isalpha())
    return asc > 8 and dev < 4

# =============================================================================
# DATABASE HELPERS  (all return empty gracefully when DB unavailable)
# =============================================================================
def db_chapter_meta(chapter_id: str) -> dict:
    sb = _get_sb()
    if not sb or not chapter_id:
        return {}
    try:
        r = (sb.table("syllabus_chapters")
               .select("id,chapter_title,chapter_code,book_name,author,board,class_level,subject,language")
               .eq("id", chapter_id).limit(1).execute())
        return (r.data or [{}])[0]
    except Exception as e:
        print(f"[DB] chapter_meta: {e}"); return {}

def db_teacher(subject: str, board: str, class_level: str) -> dict:
    sb = _get_sb()
    if not sb:
        return {}
    try:
        # Try subject-specific teacher first via teacher_subject_map
        r = (sb.table("teacher_subject_map")
               .select("teacher_id, teacher_profiles(teacher_code,teacher_name,gender,base_language,accent_style,persona_prompt,teaching_pattern,voice_provider,voice_id)")
               .eq("subject", subject).eq("board", board).eq("active", True)
               .limit(1).execute())
        rows = r.data or []
        if rows and rows[0].get("teacher_profiles"):
            return rows[0]["teacher_profiles"]
        # Fallback: any active teacher
        r2 = (sb.table("teacher_profiles")
                .select("teacher_code,teacher_name,gender,base_language,accent_style,persona_prompt,teaching_pattern,voice_provider,voice_id")
                .eq("active", True).limit(1).execute())
        return (r2.data or [{}])[0]
    except Exception as e:
        print(f"[DB] teacher: {e}"); return {}

def db_live_chunks(chapter_id: str) -> List[dict]:
    sb = _get_sb()
    if not sb or not chapter_id:
        return []
    try:
        r = (sb.table("live_teaching_chunks")
               .select("id,live_order,chunk_kind,stage_type,cleaned_text,read_text,explain_text,ask_text,practice_text,recap_text,keywords")
               .eq("chapter_id", chapter_id).eq("is_active", True)
               .order("live_order").limit(300).execute())
        return r.data or []
    except Exception as e:
        print(f"[DB] live_chunks: {e}"); return []

def db_quiz_questions(chapter_id: str) -> List[dict]:
    sb = _get_sb()
    if not sb or not chapter_id:
        return []
    try:
        # Try chapter_quiz_questions via chapter_parts
        r = (sb.table("chapter_quiz_questions")
               .select("question_text,options,correct_answer,explanation,question_type")
               .eq("is_active", True).limit(8).execute())
        return r.data or []
    except Exception as e:
        print(f"[DB] quiz: {e}"); return []

# =============================================================================
# TTS  — Human Indian Teacher Voice
# =============================================================================
# OpenAI voice personalities:
# nova   → warm, clear, female  ← best for Indian English teacher
# shimmer → soft, gentle, female
# alloy  → neutral, balanced
# echo   → male, warm
# onyx   → male, deep
# fable  → male, storytelling

_TTS_VOICE_MAP = {
    "female": "nova",
    "male":   "echo",
    "nova": "nova", "shimmer": "shimmer", "alloy": "alloy",
    "echo": "echo", "onyx": "onyx", "fable": "fable",
    "ash": "echo", "coral": "shimmer", "sage": "alloy",
}

def _pick_voice(session: dict) -> str:
    teacher = session.get("teacher", {})
    vid = teacher.get("voice_id", "")
    if vid and vid in _TTS_VOICE_MAP:
        return _TTS_VOICE_MAP[vid]
    gender = (teacher.get("gender", "") or "female").lower()
    return _TTS_VOICE_MAP.get(gender, "nova")

def _tts_instructions(lang: str) -> str:
    """Build voice instructions for perfect Indian accent based on student language."""
    lang_low = (lang or "").lower()

    # Regional language pronunciation guidance
    if "bengali" in lang_low or "bangla" in lang_low:
        lang_note = (
            "Pronounce Bengali words with authentic softness — 'bhaalo', 'tumi', 'keno', 'shikha' "
            "should sound natural, not anglicised. Switch warmly between Bengali and Hindi."
        )
    elif "hindi" in lang_low or "hinglish" in lang_low:
        lang_note = (
            "Pronounce Hindi words with full authentic Devanagari sounds. "
            "'maa' not 'mah', 'padhna' not 'paadna', 'pyaar' with proper retroflex. "
            "Hindi should sound like a native Hindi speaker, not translated English."
        )
    elif "tamil" in lang_low:
        lang_note = "Mix Tamil words naturally. Pronounce Tamil with authentic retroflex sounds."
    elif "telugu" in lang_low:
        lang_note = "Mix Telugu words naturally with authentic pronunciation."
    elif "marathi" in lang_low:
        lang_note = "Mix Marathi naturally. Pronounce Marathi words authentically."
    elif "gujarati" in lang_low:
        lang_note = "Mix Gujarati naturally with authentic pronunciation."
    elif "kannada" in lang_low:
        lang_note = "Mix Kannada naturally with authentic pronunciation."
    elif "malayalam" in lang_low:
        lang_note = "Mix Malayalam naturally with authentic pronunciation."
    elif "punjabi" in lang_low:
        lang_note = "Mix Punjabi naturally. Pronounce Punjabi with authentic sounds."
    else:
        lang_note = (
            "Use natural Indian English pronunciation — warm Indian accent, "
            "not American or British."
        )

    return f"""You are Priya Ma'am — a warm, energetic, experienced Indian school teacher.

ACCENT AND VOICE:
- Speak with a NATURAL INDIAN ACCENT — NOT American, NOT British, NOT robotic
- Volume: speak LOUDLY and CLEARLY like a classroom teacher — not soft or mumbled
- Pace: slightly slower than conversation, pause between key ideas for emphasis
- Energy: HIGH and GENUINE — excited about teaching, proud of students
- Warmth: every sentence carries genuine care for the student

LANGUAGE PRONUNCIATION:
- {lang_note}
- English technical words: pronounce with Indian English (not American accent)
- Names: Ranjan, Priya, Malhar — authentic Indian pronunciation always
- NEVER anglicise Indian words — keep them authentic

EMOTIONAL EXPRESSION:
- "Arey waah!" — genuinely excited, voice lifts up
- "Ekdum sahi!" — proud and celebratory tone
- "Yaad rakhna..." — slower, serious, emphasised
- Questions: voice rises naturally at the end
- Corrections: warm and encouraging, never harsh
- Build suspense before revealing answers — slight pause then reveal"""


def generate_audio(text: str, voice: str = "nova", lang: str = "Hinglish") -> Optional[str]:
    """Generate TTS audio with Indian accent instructions. Returns public URL or None."""
    oai = _get_openai()
    if not oai:
        return None
    clean = clean_text(text)
    if not clean or len(clean) < 3:
        return None
    try:
        # Try gpt-4o-mini-tts first (supports instructions for accent control)
        try:
            resp = oai.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=clean[:4000],
                instructions=_tts_instructions(lang),
            )
        except Exception:
            # Fallback to tts-1-hd if gpt-4o-mini-tts unavailable
            resp = oai.audio.speech.create(
                model="tts-1-hd",
                voice=voice,
                input=clean[:4000],
                speed=0.92,
            )
        fname = f"{uuid.uuid4()}.mp3"
        fpath = os.path.join(AUDIO_DIR, fname)
        resp.stream_to_file(fpath)
        return f"https://{RENDER_DOMAIN}/audio/{fname}"
    except Exception as e:
        print(f"[TTS] error: {e}")
        return None

# =============================================================================
# UCD ORCHESTRATOR - human-style chat + CAD agent handoff
# =============================================================================
CAD_INTENT_WORDS = {
    "cad", "trace", "layout", "floor", "plan", "drawing", "dxf", "dwg",
    "convert", "blueprint", "venue", "stage", "seating", "autocad",
}
IMAGE_FILE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
CAD_FILE_EXTS = IMAGE_FILE_EXTS | {".pdf", ".dxf", ".dwg"}

def _ucd_session_id(raw_sid: Optional[str]) -> str:
    sid = raw_sid or f"ucd-{uuid.uuid4()}"
    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "phase": "UCD",
            "history": [],
            "pending": {},
        }
    return sid

def _ucd_detect_intent(message: str, file_meta: Optional[UCDFileMeta] = None) -> str:
    text = (message or "").lower()
    ext = Path(file_meta.filename or "").suffix.lower() if file_meta else ""
    if ext in CAD_FILE_EXTS:
        return "cad_trace_file"
    if any(word in text for word in CAD_INTENT_WORDS):
        if any(word in text for word in {"trace", "convert", "upload", "image", "file", "pdf", "dwg", "dxf"}):
            return "cad_trace_file"
        return "cad_generate_layout"
    if any(word in text for word in {"concept", "idea", "creative route", "theme", "story"}) and "mood" not in text:
        return "concept_strategy"
    if any(word in text for word in {"moodboard", "mood board", "materials", "ambience", "palette"}):
        return "moodboard_direction"
    if any(word in text for word in {"invite", "backdrop", "facade", "key visual", "2d", "graphic"}):
        return "graphics_2d_direction"
    if any(word in text for word in {"3d", "render", "structure", "sketchup", "spatial"}):
        return "render_3d_direction"
    if any(word in text for word in {"presentation", "pitch", "deck", "ppt", "proposal"}):
        return "presentation_direction"
    if any(word in text for word in {"deliverable", "deliverables", "budget", "brief", "requirement", "requirements"}):
        return "brief_orchestration"
    if any(word in text for word in {"help", "how", "what", "can you"}):
        return "general_help"
    return "conversation"

def _ucd_missing_for_cad(intent: str, message: str, file_meta: Optional[UCDFileMeta]) -> List[str]:
    text = message or ""
    questions: List[str] = []
    has_dims = parse_dim_hint(text) is not None
    if intent == "cad_trace_file" and not file_meta:
        questions.append("Please upload the layout/image/PDF/DXF file you want me to trace into CAD.")
    if intent in {"cad_trace_file", "cad_generate_layout"} and not has_dims and not file_meta:
        questions.append("What is the approximate venue size, for example 40m x 30m or 120ft x 80ft?")
    if intent == "cad_generate_layout" and not re.search(r"\b(concert|conference|wedding|exhibition|award|launch|expo|summit)\b", text, re.I):
        questions.append("What type of event is this: conference, concert, exhibition, wedding, product launch, or generic?")
    if intent in {"cad_trace_file", "cad_generate_layout"} and not re.search(r"\b\d{2,5}\s*(pax|people|audience|seats|guests)\b", text, re.I):
        questions.append("How many people or seats should I plan for?")
    return questions[:3]

def parse_dim_hint(text: str) -> Optional[Dict[str, Any]]:
    try:
        from cad_engine_pro import parse_dim_string
        dims = parse_dim_string(text or "", "m")
        if not dims:
            return None
        return {"width_mm": dims[0], "depth_mm": dims[1], "unit": dims[2]}
    except Exception:
        return None

def _ucd_cad_ui_contract(intent: str, has_file: bool) -> Dict[str, Any]:
    return {
        "mode": "cad_fullscreen",
        "panel": "cad",
        "show_chat": True,
        "show_progress": True,
        "preview": "live_svg",
        "progress_steps": [
            {"id": "receive", "label": "Reading the uploaded file", "status": "pending"},
            {"id": "analyze", "label": "Detecting dimensions, rooms, zones, and labels", "status": "pending"},
            {"id": "draft", "label": "Creating CAD geometry layer by layer", "status": "pending"},
            {"id": "annotate", "label": "Adding dimensions, title block, scale bar, and symbols", "status": "pending"},
            {"id": "export", "label": "Exporting SVG preview, PDF, and DXF when available", "status": "pending"},
        ],
        "reason": "CAD work benefits from a large canvas and visible progress.",
        "requires_upload": intent == "cad_trace_file" and not has_file,
    }

def _ucd_build_cad_agent_message(req: UCDChatRequest, intent: str) -> Dict[str, Any]:
    file_dict = _dump_model(req.file) if req.file else None
    return {
        "target_agent": "CAD_AGENT",
        "intent": intent,
        "instruction": (
            "Trace or generate this layout as a professional CAD drawing. "
            "Keep the user updated while creating boundary, zones, symbols, dimensions, "
            "title block, scale bar, north arrow, SVG preview, PDF, and DXF when possible."
        ),
        "user_message": req.message,
        "project_id": req.project_id or "demo-project",
        "title": req.title or "CAD Layout",
        "file": file_dict,
        "preferred_endpoint": "/api/cad/pro/trace" if intent == "cad_trace_file" else "/api/cad/pro/generate",
    }

def _ucd_human_message(intent: str, questions: List[str], has_file: bool) -> str:
    if questions:
        return "I can do that. I just need a couple of details so the CAD result is accurate."
    if intent == "cad_trace_file" and has_file:
        return "I will open the CAD workspace full screen and send this file to the CAD agent for tracing. You will see the drawing build up step by step."
    if intent == "cad_generate_layout":
        return "I will create a professional CAD layout from your brief and show the CAD workspace full screen while it is generated."
    if intent == "general_help":
        return "Tell me what you want to create, improve or upload, and I will route it to the right specialist agent."
    if intent == "concept_strategy":
        return "I will treat this like a senior creative director: first I will understand the brief, then create multiple concept routes that are genuinely different from each other."
    if intent == "moodboard_direction":
        return "I will build the moodboard around the selected concept with detailed logic for mood, material, lighting, ambience, seating and stage look."
    if intent == "graphics_2d_direction":
        return "I will brief the 2D art director with exact outputs, copy direction, size logic and visual hierarchy."
    if intent == "render_3d_direction":
        return "I will make CAD and dimensions guide the 3D render agent before any spatial design is visualized."
    if intent == "presentation_direction":
        return "I will package the brief, concept, moodboard, 2D, 3D and flow into a client-ready pitch presentation."
    if intent == "brief_orchestration":
        return "I will first clarify the brief, missing details, deliverables and budget thinking, then start the agents in the correct order."
    return "I am here with you. Tell me what you want to build, convert, trace, improve, or present."

def _ucd_response(req: UCDChatRequest) -> UCDChatResponse:
    sid = _ucd_session_id(req.session_id)
    intent = _ucd_detect_intent(req.message, req.file)
    questions = _ucd_missing_for_cad(intent, req.message, req.file) if intent.startswith("cad_") else _missing_brief_questions(req.message or "")
    has_file = req.file is not None
    ui: Dict[str, Any] = {}
    agent: Dict[str, Any] = {}
    next_actions: List[Dict[str, Any]] = []

    if intent.startswith("cad_"):
        ui = _ucd_cad_ui_contract(intent, has_file)
        agent = _ucd_build_cad_agent_message(req, intent)
        if questions:
            next_actions.append({"type": "ask_user", "questions": questions})
        elif intent == "cad_trace_file":
            next_actions.append({"type": "open_cad_fullscreen"})
            next_actions.append({"type": "send_to_agent", "agent": "CAD_AGENT", "endpoint": "/api/cad/pro/trace"})
        else:
            next_actions.append({"type": "open_cad_fullscreen"})
            next_actions.append({"type": "send_to_agent", "agent": "CAD_AGENT", "endpoint": "/api/cad/pro/generate"})
    elif intent in {"concept_strategy", "brief_orchestration", "conversation"}:
        agent = {
            "target_agent": "CONCEPT_AGENT",
            "supervised_by": "UCD_AGENT",
            "instruction": _creative_director_prompt("Concept Agent", 45)
            + " Generate at least three very different concept routes; no repeated generic fallbacks.",
            "preferred_endpoint": "/projects/{project_id}/run",
        }
        if questions:
            next_actions.append({"type": "ask_user", "questions": questions})
        next_actions.append({"type": "send_to_agent", "agent": "CONCEPT_AGENT", "endpoint": "/projects/{project_id}/run"})
    elif intent == "moodboard_direction":
        agent = {"target_agent": "MOODBOARD_AGENT", "supervised_by": "UCD_AGENT", "preferred_endpoint": "/api/moodboard/generate"}
        next_actions.append({"type": "send_to_agent", "agent": "MOODBOARD_AGENT", "endpoint": "/api/moodboard/generate"})
    elif intent == "graphics_2d_direction":
        agent = {"target_agent": "GRAPHICS_2D_AGENT", "supervised_by": "UCD_AGENT", "preferred_endpoint": "/ai/generate-2d"}
        next_actions.append({"type": "send_to_agent", "agent": "GRAPHICS_2D_AGENT", "endpoint": "/ai/generate-2d"})
    elif intent == "render_3d_direction":
        agent = {"target_agent": "RENDER_3D_AGENT", "supervised_by": "UCD_AGENT", "preferred_endpoint": "/projects/{project_id}/renders/generate-separated"}
        next_actions.append({"type": "send_to_agent", "agent": "CAD_AGENT", "endpoint": "/api/cad/pro/generate"})
        next_actions.append({"type": "send_to_agent", "agent": "RENDER_3D_AGENT", "endpoint": "/projects/{project_id}/renders/generate-separated"})
    elif intent == "presentation_direction":
        agent = {"target_agent": "PRESENTATION_AGENT", "supervised_by": "UCD_AGENT", "preferred_endpoint": "/projects/{project_id}/presentation/build"}
        next_actions.append({"type": "send_to_agent", "agent": "PRESENTATION_AGENT", "endpoint": "/projects/{project_id}/presentation/build"})

    response = UCDChatResponse(
        session_id=sid,
        intent=intent,
        message=_ucd_human_message(intent, questions, has_file),
        questions=questions,
        ui=ui,
        agent=agent,
        next_actions=next_actions,
    )
    SESSIONS[sid].setdefault("history", []).append({
        "user": req.message,
        "intent": intent,
        "questions": questions,
        "ts": time.time(),
    })
    return response

# =============================================================================
# GPT  — Brain of the teacher
# =============================================================================
def _gpt(messages: List[dict], max_tokens: int = 400, temp: float = 0.85) -> str:
    oai = _get_openai()
    if not oai:
        return "Chalo aage badhte hain!"
    try:
        r = oai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GPT] error: {e}")
        return "Thoda technical issue hai, lekin chalo aage badhte hain!"

# =============================================================================
# TEACHER PERSONA  — Legacy education persona utilities
# =============================================================================
def _lang_rules(lang: str) -> str:
    """Return strict language mixing rules based on student preference."""
    l = (lang or "").lower()

    if "bengali" in l or "bangla" in l:
        return """LANGUAGE — BENGALI + HINDI MIX (MANDATORY):
You MUST speak in Bengali + Hindi mix throughout. No pure English unless it's a technical term.
Bengali words to use naturally: "bhalo" (good), "keno" (why), "ki" (what), "tumi" (you),
"amra" (we), "bujhecho?" (understood?), "darun!" (excellent!), "thik ache" (okay),
"shono" (listen), "dekho" (look/see), "janao" (tell me), "boro" (big), "choto" (small).
Celebrate: "Darun! Khub bhalo!" | Questions: "Tumi ki bujhecho?" | Encouragement: "Cheshta koro!"
Mix example: "Dekho, jemon amar ma amader care koren — sheikhanei 'Maa' ar 'Matrubhumi' eki."
NEVER speak in only English. Bengali first, Hindi support, English only for terms."""

    elif "hindi" in l or "hinglish" in l:
        return """LANGUAGE — HINDI + ENGLISH MIX (HINGLISH):
Speak natural Hinglish — Hindi base with English terms mixed.
Use full Hindi sentences: "Dekho beta, yeh concept bahut important hai."
Hindi celebrations: "Ekdum sahi!", "Wah wah!", "Bahut achha!"
Questions: "Samajh aaya?", "Batao, kya socha?", "Koi sawaal?"
English only for technical terms — not everyday words."""

    elif "tamil" in l:
        return """LANGUAGE — TAMIL + HINDI + ENGLISH MIX:
Use Tamil words: "nalla" (good), "enna" (what), "aamaa" (yes), "purinjucha?" (understood?),
"super!" (great), "romba nalla" (very good), "paaru" (look/see).
Mix: Tamil + Hindi + English technical terms naturally."""

    elif "telugu" in l:
        return """LANGUAGE — TELUGU + HINDI + ENGLISH MIX:
Use Telugu words: "bagundi" (good), "enti" (what), "cheppandi" (tell me), "artham ayinda?" (understood?),
"chala bagundi!" (very good!), "chudandi" (look/see).
Mix: Telugu + Hindi + English technical terms naturally."""

    elif "marathi" in l:
        return """LANGUAGE — MARATHI + HINDI + ENGLISH MIX:
Use Marathi words: "chan" (good), "kay" (what), "sangaa" (tell me), "samajle?" (understood?),
"shabas!" (bravo!), "khup chan!" (very good!), "bagha" (look/see).
Mix: Marathi + Hindi + English technical terms naturally."""

    elif "gujarati" in l:
        return """LANGUAGE — GUJARATI + HINDI + ENGLISH MIX:
Use Gujarati words: "saras" (good), "shu" (what), "samjayu?" (understood?),
"shabash!" (bravo!), "khub saras!" (very good!), "juo" (look/see).
Mix: Gujarati + Hindi + English technical terms naturally."""

    elif "kannada" in l:
        return """LANGUAGE — KANNADA + HINDI + ENGLISH MIX:
Use Kannada words: "chennagide" (good), "yenu" (what), "arthavaytu?" (understood?),
"shabash!" (bravo!), "tumba chennagide!" (very good!), "nodi" (look/see).
Mix: Kannada + Hindi + English technical terms naturally."""

    elif "malayalam" in l:
        return """LANGUAGE — MALAYALAM + HINDI + ENGLISH MIX:
Use Malayalam words: "nannayittundu" (good), "enthanu" (what), "manasilaayo?" (understood?),
"kalaakki!" (excellent!), "valare nannayittundu!" (very good!).
Mix: Malayalam + Hindi + English technical terms naturally."""

    else:
        return """LANGUAGE — INDIAN ENGLISH + HINDI MIX:
Speak warm Indian English with natural Hindi phrases mixed in.
Hindi phrases: "Dekho", "Samajh aaya?", "Bilkul sahi!", "Bahut achha!", "Koi doubt?"
Keep it conversational and Indian — not formal British/American English."""


def _system_prompt(session: dict) -> str:
    st   = session["student"]
    meta = session["meta"]
    tea  = session.get("teacher", {})

    name    = st.get("name", "beta")
    lang    = st.get("language", "Hinglish")
    subject = meta.get("subject", "")
    chapter = meta.get("chapter_title", "")
    book    = meta.get("book_name", "")
    board   = meta.get("board", "CBSE")
    cls     = meta.get("class_level", "")
    author  = meta.get("author", "")

    t_name    = tea.get("teacher_name", "Priya Ma'am")
    t_pattern = tea.get("teaching_pattern") or "Warm, patient, concept-first."
    t_persona = tea.get("persona_prompt") or "Speak naturally like a warm Indian school teacher."

    lang_rules = _lang_rules(lang)

    return f"""You are {t_name}, a brilliant and beloved Indian school teacher with 20+ years of experience.

━━━ STUDENT CONTEXT ━━━
Name: {name}  |  Class: {cls}  |  Board: {board}
Subject: {subject}  |  Book: {book}{f'  |  Author: {author}' if author else ''}
Chapter: {chapter}

━━━ {lang_rules}

━━━ YOUR PERSONALITY ━━━
{t_persona}
Teaching Style: {t_pattern}

━━━ HOW YOU TEACH ━━━
• SHORT sentences — max 2 lines per thought (this is SPOKEN audio, not text)
• Vary ENERGY: excited for new concepts, suspenseful before answers, warm always
• REAL Indian examples: chai, cricket, roti, rickshaw, mango, festivals, Bollywood
• "Yaad rakhna..." before KEY exam points (slow down, emphasise)
• After every explanation: check with "Samajh aaya?" or language equivalent
• Build SUSPENSE: "Toh batao... kya socha?" (pause effect)
• Playfully strict: "Dhyan se! Yeh exam mein zaroor aayega!"

━━━ EMOTIONAL REACTIONS ━━━
Correct answer  → CELEBRATE loudly: "AREY WAAH! Ekdum sahi! Mujhe pata tha!"
Almost right    → "Bahut close hai! Aur thoda socho..."
Wrong answer    → Warm: "Achha try! Sahi answer hai actually... [explain with analogy]"
Doubt asked     → "Bahut achha sawaal {name}! [answer enthusiastically with example]"
Quiet/shy       → "Koi baat nahi, aaram se. Main yahan hoon."

━━━ STRICT SPEECH RULES ━━━
1. NO bullet points, NO asterisks, NO markdown — PURE SPOKEN WORDS ONLY
2. NO robotic monotone — every sentence has ENERGY and EMOTION
3. Keep under 100 words per response — short bursts work better in audio
4. ALWAYS end with either a question OR a clear "Chalo aage badhte hain!"
5. Use {name}'s name at least once per response — personal connection matters"""

def teacher_say(session: dict, instruction: str, max_tokens: int = 350) -> str:
    """Ask GPT to generate teacher speech, maintaining conversation history."""
    history = session["teaching"].get("history", [])
    msgs = [{"role": "system", "content": _system_prompt(session)}]
    msgs += history[-14:]          # last 7 exchanges = 14 messages
    msgs.append({"role": "user", "content": instruction})
    reply = _gpt(msgs, max_tokens=max_tokens)
    # Update history
    session["teaching"].setdefault("history", [])
    session["teaching"]["history"].append({"role": "user",      "content": instruction})
    session["teaching"]["history"].append({"role": "assistant", "content": reply})
    return reply

# =============================================================================
# RESPONSE BUILDER
# =============================================================================
def make_resp(
    text: str,
    phase: str,
    next_step: str,
    session: dict,
    session_id: Optional[str] = None,
    should_listen: bool = False,
    extra_meta: Optional[dict] = None,
) -> dict:
    voice    = _pick_voice(session)
    lang     = session.get("student", {}).get("language", "Hinglish")
    tea      = session.get("teacher", {})
    meta     = session.get("meta", {})
    teaching = session.get("teaching", {})
    audio_url = generate_audio(text, voice=voice, lang=lang)

    return {
        "ok":                True,
        "session_id":        session_id,
        "text":              text,
        "tts_text":          clean_text(text),
        "audio_url":         audio_url,
        "phase":             phase,
        "next":              next_step,
        "should_listen":     should_listen,
        "student_name":      session["student"].get("name"),
        "preferred_language":session["student"].get("language"),
        "meta": {
            "teacher_name":          tea.get("teacher_name", ""),
            "teacher_code":          tea.get("teacher_code", ""),
            "teacher_gender":        tea.get("gender", "female"),
            "teacher_voice_id":      voice,
            "teacher_voice_provider":"openai",
            "teacher_base_language": tea.get("base_language", ""),
            "chapter_name":          meta.get("chapter_title", ""),
            "book_name":             meta.get("book_name", ""),
            "chapter_author":        meta.get("author", ""),
            "subject":               meta.get("subject", ""),
            "current_chunk_index":   teaching.get("chunk_idx", 0),
            "current_chunk_stage":   teaching.get("stage", ""),
            "current_chunk_title":   teaching.get("chunk_title", ""),
            "waiting_for_reply":     should_listen,
            "read_line_text":        teaching.get("read_line", ""),
            "explanation_text":      teaching.get("explain_line", ""),
            "bottom_ticker_text":    teaching.get("read_line", ""),
            **(extra_meta or {}),
        },
    }

# =============================================================================
# SESSION BOOTSTRAP  — pull DB data once at session start
# =============================================================================
def bootstrap(session: dict):
    """Load chapter data, teacher profile, and teaching chunks from DB."""
    meta       = session["meta"]
    chapter_id = meta.get("syllabus_chapter_id", "")

    # 1. Enrich chapter metadata
    if chapter_id:
        row = db_chapter_meta(chapter_id)
        if row:
            meta.setdefault("chapter_title", row.get("chapter_title", ""))
            meta.setdefault("book_name",     row.get("book_name", ""))
            meta.setdefault("author",        row.get("author", ""))
            meta.setdefault("subject",       row.get("subject", ""))
            meta.setdefault("board",         row.get("board", "CBSE"))

    # 2. Load teacher profile
    teacher = db_teacher(
        meta.get("subject", ""),
        meta.get("board", "CBSE"),
        str(meta.get("class_level", "")),
    )
    session["teacher"] = teacher if teacher else _default_teacher()

    # 3. Load live chunks (filter out encoded/garbage ones)
    all_chunks = db_live_chunks(chapter_id)
    teach_chunks = [
        c for c in all_chunks
        if c.get("stage_type") in {"READ", "EXPLAIN", "TEACH", "STORY"}
        and not is_krutidev(c.get("cleaned_text") or c.get("read_text") or "")
        and (
            not is_placeholder(c.get("explain_text"))
            or not is_placeholder(c.get("cleaned_text") or c.get("read_text") or "")
        )
    ]
    session["teaching"]["db_chunks"]   = teach_chunks
    session["teaching"]["db_all"]      = all_chunks

    # 4. Load quiz questions
    session["teaching"]["quiz_pool"]   = db_quiz_questions(chapter_id)

    print(f"[bootstrap] chapter_id={chapter_id} chunks={len(teach_chunks)} teacher={session['teacher'].get('teacher_name','?')}")

def _default_teacher() -> dict:
    return {
        "teacher_name":    "Priya Ma'am",
        "teacher_code":    "PRIYA_01",
        "gender":          "female",
        "voice_id":        "nova",
        "voice_provider":  "openai",
        "base_language":   "Hinglish",
        "persona_prompt":  "Warm, energetic, experienced Indian teacher. Mix Hindi and English naturally.",
        "teaching_pattern":"Read line → Explain simply → Indian example → Check understanding",
    }

# =============================================================================
# PHASE 1: INTRO  — Name, Language, School → personal connection
# =============================================================================
def phase_intro(session: dict, msg: str, sid: str) -> dict:
    step = session.get("step", "ASK_NAME")
    st   = session["student"]
    meta = session["meta"]

    # Guard empty/garbage input — re-ask the same question
    if not msg or msg.lower().strip() in {"undefined", "null", "next", "continue", ""}:
        reprompts = {
            "ASK_NAME":     "Hmm, main sun nahi paya! Apna naam phir se batao?",
            "ASK_LANGUAGE": "Kaunsi bhasha mein padhna chahoge? Hindi, English, ya Hinglish?",
            "ASK_SCHOOL":   "Aur school ka naam? Batao please!",
        }
        text = reprompts.get(step, "Zara phir se batao please!")
        return make_resp(text, "INTRO", step, session, sid, should_listen=True)

    # ── ASK_NAME ──────────────────────────────────────────────────────────
    if step == "ASK_NAME":
        name = msg.strip().title()
        st["name"] = name
        session["step"] = "ASK_LANGUAGE"

        text = teacher_say(session,
            f"Student introduced themselves as '{name}'. "
            "Greet them warmly with their name, say something nice about it, "
            "then ask which language they prefer: Hindi, English, Hinglish, or their regional language. "
            "Keep it friendly and fun — 3 sentences max.",
            max_tokens=120)
        session["teaching"]["stage"] = "ASK_LANGUAGE"
        return make_resp(text, "INTRO", "ASK_LANGUAGE", session, sid, should_listen=True)

    # ── ASK_LANGUAGE ──────────────────────────────────────────────────────
    if step == "ASK_LANGUAGE":
        lang = msg.strip()
        st["language"] = lang
        session["step"] = "ASK_SCHOOL"

        text = teacher_say(session,
            f"Student said they prefer '{lang}'. "
            "Confirm this enthusiastically in 1 sentence, then ask which school they study in.",
            max_tokens=80)
        session["teaching"]["stage"] = "ASK_SCHOOL"
        return make_resp(text, "INTRO", "ASK_SCHOOL", session, sid, should_listen=True)

    # ── ASK_SCHOOL ────────────────────────────────────────────────────────
    if step == "ASK_SCHOOL":
        school = msg.strip()
        st["school"] = school
        session["step"]  = "DONE"
        session["phase"] = "CHAPTER_INTRO"

        name    = st.get("name", "beta")
        lang    = st.get("language", "Hinglish")
        chapter = meta.get("chapter_title", "aaj ka chapter")
        subject = meta.get("subject", "")
        book    = meta.get("book_name", "")
        author  = meta.get("author", "")
        cls     = meta.get("class_level", "")
        board   = meta.get("board", "CBSE")

        # GPT generates a warm, personalised chapter intro hook
        hook = teacher_say(session,
            f"Student {name} from '{school}' is ready to study. "
            f"Now deliver a perfect chapter introduction for '{chapter}' ({subject}, {board} Class {cls}, book: {book}{', by ' + author if author else ''}). "
            "Do this in exactly this sequence: "
            "1) Celebrate that school warmly (1 short line). "
            "2) Build CURIOSITY with one surprising real-world fact or relatable situation connected to this chapter topic (2 lines). "
            "3) Tell them today's chapter name and what exciting things they'll learn (2 lines). "
            "4) End with: 'Toh chalte hain, shuru karte hain aaj ki class!' "
            f"Speak naturally in {lang}/Hinglish. Total: 5-6 sentences only.",
            max_tokens=250)

        # Find chapter overview from DB chunks if available
        db_all = session["teaching"].get("db_all", [])
        overview_chunk = next(
            (c for c in db_all if c.get("stage_type") == "CHAPTER_INTRO"), None)
        overview_text = ""
        if overview_chunk and not is_placeholder(overview_chunk.get("explain_text")):
            overview_text = overview_chunk.get("explain_text", "")

        session["teaching"]["stage"] = "chapter_intro"
        return make_resp(hook, "CHAPTER_INTRO", "START_TEACHING", session, sid,
            should_listen=False,
            extra_meta={
                "current_chunk_stage":   "chapter_intro",
                "chapter_overview_text": overview_text,
            })

    # Fallback
    session["phase"] = "TEACHING"
    return _start_teaching(session, sid)

# =============================================================================
# PHASE 2: CHAPTER INTRO  → transitions to TEACHING automatically
# =============================================================================
def phase_chapter_intro(session: dict, msg: str, sid: str) -> dict:
    session["phase"] = "TEACHING"
    session["teaching"]["chunk_idx"] = 0
    session["teaching"]["stage"]     = "read"
    return _start_teaching(session, sid)

def _start_teaching(session: dict, sid: str) -> dict:
    """Deliver a short 'let's begin' bridge before first chunk."""
    name    = session["student"].get("name", "beta")
    chapter = session["meta"].get("chapter_title", "chapter")
    subject = session["meta"].get("subject", "")

    bridge = teacher_say(session,
        f"Say a short 1-sentence 'let's begin' line for chapter '{chapter}' ({subject}). "
        "Energetic, warm, in Hinglish. Then immediately dive into the first chunk.",
        max_tokens=60)
    # Don't wait for student here — auto-advance to first chunk
    return _teach_next_chunk(session, sid, preamble=bridge)

# =============================================================================
# PHASE 3: TEACHING  — the core loop
# =============================================================================
def phase_teaching(session: dict, msg: str, sid: str) -> dict:
    teaching = session["teaching"]
    stage    = teaching.get("stage", "read")
    msg_low  = (msg or "").lower().strip()

    # ── Student answered the teacher's comprehension question ──────────────
    if stage == "waiting_answer":
        if not msg_low or msg_low in {"next","aage","skip","continue","chalo","pass"}:
            # Skipped — give quick answer then IMMEDIATELY go to next chunk
            lang_s    = session["student"].get("language", "Hinglish")
            skip_text = teacher_say(session,
                f"Student skipped. In 1 sentence of {lang_s}/Hinglish: "
                "give the correct answer briefly, then say 'Chalo aage badhte hain!'",
                max_tokens=70)
            teaching["stage"] = "read"
            teaching["xp"]    = teaching.get("xp", 0) + 5
            return _teach_next_chunk(session, sid, preamble=skip_text)

        return _eval_answer(session, msg, sid)

    # ── Student asked a doubt/question ────────────────────────────────────
    is_doubt = (
        "?" in msg and
        any(w in msg_low for w in [
            "kya","kyun","kaise","what","why","how","means","matlab",
            "samajh","doubt","explain","bata","pata nahi","confused",
            "nahi samjha","iska","ka matlab","difference","define",
        ])
    )
    if is_doubt:
        return _handle_doubt(session, msg, sid)

    # ── Continue signal ────────────────────────────────────────────────────
    continue_words = {"next","aage","chalo","ok","okay","haan","yes","ji",
                      "samjha","samajh gaya","samajh aayi","theek hai","got it",
                      "understood","clear","continue","aur batao","batao"}
    if msg_low in continue_words or not msg_low:
        return _teach_next_chunk(session, sid)

    # ── Emotional/general student message ─────────────────────────────────
    general_text = teacher_say(session,
        f"Student said: '{msg}'. "
        "Respond warmly and briefly (1-2 sentences), acknowledge what they said, "
        "then redirect back to the lesson naturally.",
        max_tokens=100)
    return make_resp(general_text, "TEACHING", "CONTINUE", session, sid,
        should_listen=False)

def _teach_next_chunk(session: dict, sid: str, preamble: str = "") -> dict:
    """Deliver the next DB chunk or GPT-generated teaching turn."""
    teaching   = session["teaching"]
    db_chunks  = teaching.get("db_chunks", [])
    idx        = teaching.get("chunk_idx", 0)
    meta       = session["meta"]
    st         = session["student"]
    name       = st.get("name", "beta")
    lang       = st.get("language", "Hinglish")
    chapter    = meta.get("chapter_title", "")
    subject    = meta.get("subject", "")

    # ── End of all DB chunks ───────────────────────────────────────────────
    if idx >= len(db_chunks):
        # If no DB chunks at all, use GPT to teach up to 8 parts
        if not db_chunks and idx < 8:
            return _gpt_teach_chunk(session, sid, idx, preamble)
        return _wrap_up(session, sid, preamble=preamble)

    chunk = db_chunks[idx]
    teaching["chunk_idx"]   = idx + 1
    teaching["chunk_title"] = f"Part {idx + 1}"

    # Extract fields
    raw_text   = (chunk.get("cleaned_text") or chunk.get("read_text") or "").strip()
    explain    = (chunk.get("explain_text") or "").strip()
    ask_text   = (chunk.get("ask_text") or "").strip()

    # Skip still-encoded chunks (shouldn't happen after bootstrap filter)
    if is_krutidev(raw_text) and is_krutidev(explain):
        teaching["chunk_idx"] = idx + 1
        return _teach_next_chunk(session, sid, preamble)

    parts = []
    if preamble:
        parts.append(preamble)

    # 1. READ the original line (only if it's clean Unicode)
    if raw_text and not is_krutidev(raw_text) and len(raw_text) > 20:
        read_line = raw_text[:350]
        teaching["read_line"] = read_line
        parts.append(f"Suno — '{read_line}'")
    else:
        teaching["read_line"] = ""

    # 2. EXPLAIN — use DB value if rich, otherwise GPT
    if not is_placeholder(explain) and not is_krutidev(explain):
        teaching["explain_line"] = explain
        parts.append(explain)
    else:
        gpt_explain = teacher_say(session,
            f"Just explained this line from '{chapter}' ({subject}) to {name}: "
            f"'{(raw_text or explain)[:300]}'. "
            f"Explain it simply in 2-3 sentences of {lang}/Hinglish. "
            "Use one relatable Indian daily-life example. Be warm and enthusiastic.",
            max_tokens=180)
        teaching["explain_line"] = gpt_explain
        parts.append(gpt_explain)

    # 3. ASK — DB question or GPT-generated
    if ask_text and not is_placeholder(ask_text) and len(ask_text) > 10:
        teaching["pending_question"] = ask_text
        parts.append(ask_text)
    else:
        gpt_q = _gpt([
            {"role": "system", "content": _system_prompt(session)},
            {"role": "user",   "content":
                f"After explaining '{(raw_text or explain)[:150]}' to {name}, "
                f"ask ONE short, clear comprehension question in {lang}/Hinglish. "
                "Make it simple, encouraging, and directly related to what was just explained. "
                "Just the question — no preamble."}
        ], max_tokens=70)
        teaching["pending_question"] = gpt_q
        parts.append(gpt_q)

    teaching["stage"]    = "waiting_answer"
    full_text = " ".join(p for p in parts if p)

    return make_resp(full_text, "TEACHING", "AWAIT_ANSWER", session, sid,
        should_listen=True,
        extra_meta={
            "current_chunk_stage": "question",
            "current_chunk_title": f"Part {idx + 1} / {len(db_chunks)}",
        })

def _gpt_teach_chunk(session: dict, sid: str, chunk_num: int, preamble: str = "") -> dict:
    """Pure GPT teaching when no DB chunks available."""
    meta    = session["meta"]
    st      = session["student"]
    name    = st.get("name", "beta")
    lang    = st.get("language", "Hinglish")
    chapter = meta.get("chapter_title", "")
    subject = meta.get("subject", "")
    book    = meta.get("book_name", "")
    cls     = meta.get("class_level", "")
    board   = meta.get("board", "CBSE")

    teaching = session["teaching"]
    teaching["chunk_idx"] = chunk_num + 1

    chunk_text = teacher_say(session,
        f"Teach chunk #{chunk_num + 1} of '{chapter}' ({subject}, {board} Class {cls}, {book}) to {name}. "
        "Follow this EXACT sequence in one response: "
        "1) State one key concept or sentence from this chapter part (quote-style). "
        "2) Explain it simply in 2 sentences with a real Indian daily-life example. "
        "3) Ask ONE clear comprehension question to check understanding. "
        f"Speak naturally in {lang}/Hinglish. Total: 5-6 sentences. End on the question.",
        max_tokens=300)

    teaching["stage"]    = "waiting_answer"
    teaching["chunk_title"] = f"Part {chunk_num + 1}"
    text = (preamble + " " + chunk_text).strip() if preamble else chunk_text

    return make_resp(text, "TEACHING", "AWAIT_ANSWER", session, sid,
        should_listen=True,
        extra_meta={"current_chunk_stage": "question",
                    "current_chunk_title": f"Part {chunk_num + 1}"})

def _eval_answer(session: dict, student_answer: str, sid: str) -> dict:
    """Evaluate answer, give feedback, then IMMEDIATELY advance to next chunk.
    Combining both into one response eliminates the frontend auto-continue dependency."""
    teaching = session["teaching"]
    name     = session["student"].get("name", "beta")
    lang     = session["student"].get("language", "Hinglish")
    q        = teaching.get("pending_question", "the question")
    context  = teaching.get("read_line") or teaching.get("explain_line", "")

    teaching["questions_asked"] = teaching.get("questions_asked", 0) + 1
    teaching["xp"]              = teaching.get("xp", 0) + 20

    # Generate warm feedback (short — 1-2 sentences only)
    feedback = teacher_say(session,
        f"Student {name} answered: '{student_answer}'.\n"
        f"The question was: '{q}'\n"
        f"Context: '{context[:150]}'\n\n"
        f"In {lang}/Hinglish, give EXACTLY 1-2 sentences of warm feedback: "
        "If correct: celebrate loudly with their name, add 1 bonus fact. "
        "If wrong: gently say the right answer with a simple analogy. "
        "End with ONLY: 'Chalo, aage badhte hain!'",
        max_tokens=100)

    teaching["stage"] = "read"

    # Check if all chunks done
    db_chunks = teaching.get("db_chunks", [])
    idx       = teaching.get("chunk_idx", 0)
    teach_chunks = [c for c in db_chunks
                    if c.get("stage_type") in {"READ","EXPLAIN","TEACH","STORY"}
                    and not is_krutidev(c.get("cleaned_text") or c.get("read_text") or "")]

    if idx >= len(teach_chunks) and (teach_chunks or idx >= 6):
        # Chapter complete — return feedback + wrap up
        return _wrap_up(session, sid, preamble=feedback)

    # Immediately advance to next chunk with feedback as preamble
    return _teach_next_chunk(session, sid, preamble=feedback)

def _handle_doubt(session: dict, question: str, sid: str) -> dict:
    """Handle a student doubt mid-lesson."""
    name    = session["student"].get("name", "beta")
    lang    = session["student"].get("language", "Hinglish")
    meta    = session["meta"]

    answer = teacher_say(session,
        f"Student {name} has a doubt: '{question}'.\n"
        f"Subject: {meta.get('subject','')} | Chapter: {meta.get('chapter_title','')}\n\n"
        f"In {lang}/Hinglish (3 sentences max): "
        "1) Praise their curiosity warmly. "
        "2) Answer clearly using a real Indian daily-life analogy. "
        "3) End with: 'Ab samajh aaya? Koi aur doubt?'",
        max_tokens=200)

    return make_resp(answer, "TEACHING", "DOUBT_RESOLVED", session, sid,
        should_listen=True,
        extra_meta={"current_chunk_stage": "doubt_answer"})

def _wrap_up(session: dict, sid: str, preamble: str = "") -> dict:
    """Chapter complete — summary and quiz offer."""
    name    = session["student"].get("name", "beta")
    chapter = session["meta"].get("chapter_title", "")
    xp      = session["teaching"].get("xp", 0)
    asked   = session["teaching"].get("questions_asked", 0)

    summary = teacher_say(session,
        f"The entire chapter '{chapter}' is now complete for {name}. "
        f"They answered {asked} questions and earned {xp} XP. "
        "In 3-4 sentences: "
        "1) Recap the 2-3 most important things learned today (specific to this chapter). "
        "2) Celebrate their effort genuinely and personally with their name. "
        "3) Say: 'Kya tum ek quick revision quiz dena chahoge?' "
        "Be very proud and energetic!",
        max_tokens=200)

    full_text = (preamble + " " + summary).strip() if preamble else summary
    session["phase"]             = "DONE"
    session["teaching"]["stage"] = "done"
    return make_resp(full_text, "DONE", "CHAPTER_COMPLETE", session, sid,
        should_listen=True,
        extra_meta={"current_chunk_stage": "chapter_complete"})

# =============================================================================
# PHASE 4: QUIZ
# =============================================================================
def phase_quiz_ask(session: dict, sid: str) -> dict:
    """Ask the next quiz question."""
    teaching  = session["teaching"]
    pool      = teaching.get("quiz_pool", [])
    q_idx     = teaching.get("quiz_idx", 0)
    name      = session["student"].get("name", "beta")
    meta      = session["meta"]

    if q_idx >= len(pool) or not pool:
        # GPT-generated quiz question
        q_text = teacher_say(session,
            f"Ask {name} one clear revision question about '{meta.get('chapter_title','')}' "
            f"({meta.get('subject','')}). Make it interesting and challenging but fair. "
            "Hinglish, short, end with '— batao!'",
            max_tokens=80)
        teaching["quiz_answer"] = None
    else:
        q = pool[q_idx]
        q_text = (
            f"Quiz time {name}! Dhyan se suno. "
            f"{q.get('question_text','')} — socho aur batao!"
        )
        teaching["quiz_answer"] = q.get("correct_answer")

    teaching["quiz_idx"] = q_idx + 1
    teaching["stage"]    = "quiz_answer"

    return make_resp(q_text, "QUIZ", "AWAIT_QUIZ_ANSWER", session, sid,
        should_listen=True,
        extra_meta={"current_chunk_stage": "quiz"})

def phase_quiz_eval(session: dict, msg: str, sid: str) -> dict:
    """Evaluate quiz answer and ask if they want another."""
    teaching = session["teaching"]
    correct  = teaching.get("quiz_answer")
    name     = session["student"].get("name", "beta")

    if correct:
        fb_prompt = (
            f"Student {name} answered '{msg}'. Correct answer: '{correct}'. "
            "2 sentences of Hinglish feedback — celebrate if right, gently correct if wrong. "
            "End with: 'Ek aur question?'"
        )
    else:
        fb_prompt = (
            f"Student {name} answered '{msg}'. "
            "Give 1-sentence encouraging feedback. End with: 'Ek aur question?'"
        )
    feedback = teacher_say(session, fb_prompt, max_tokens=120)
    teaching["stage"] = "quiz_wait"
    teaching["xp"]    = teaching.get("xp", 0) + 15
    return make_resp(feedback, "QUIZ", "NEXT_QUIZ_Q", session, sid,
        should_listen=True,
        extra_meta={"current_chunk_stage": "quiz_feedback"})

# =============================================================================
# FAREWELL
# =============================================================================
def phase_farewell(session: dict, sid: str) -> dict:
    name    = session["student"].get("name", "beta")
    chapter = session["meta"].get("chapter_title", "")
    xp      = session["teaching"].get("xp", 0)

    text = teacher_say(session,
        f"Say a warm, energetic farewell to {name} who completed '{chapter}' "
        f"and earned {xp} XP today. "
        "3 sentences: 1) Congratulate them specifically, "
        "2) Give 1 quick memory tip for this chapter, "
        "3) Say goodbye warmly.",
        max_tokens=150)
    return make_resp(text, "DONE", "FAREWELL", session, sid)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/agents")
def list_agents():
    """
    Frontend discovery endpoint.
    Any backend module can call register_backend_agent(...) and it will appear here
    without changing the studio HTML.
    """
    return {
        "ok": True,
        "agents": sorted(AGENT_REGISTRY.values(), key=lambda a: (a.get("category", ""), a.get("name", ""))),
        "hourly_rates_inr": AGENT_HOURLY_RATES_INR,
    }


@app.post("/projects")
def create_project(req: ProjectCreateRequest):
    brief = (req.brief or req.brief_text or "").strip()
    project_id = str(uuid.uuid4())
    now = time.time()
    project = {
        "id": project_id,
        "project_id": project_id,
        "project_name": req.project_name or req.title or _safe_title(brief, "New Creative Project"),
        "title": req.title or req.project_name or _safe_title(brief, "New Creative Project"),
        "brief": brief,
        "event_type": req.event_type or _detect_industry(brief),
        "style_direction": req.style_direction or "Premium creative",
        "status": "draft",
        "concepts": [],
        "selected_concept_index": None,
        "created_at": now,
        "updated_at": now,
    }
    PROJECT_STORE[project_id] = project
    PROJECT_ASSETS.setdefault(project_id, [])
    PROJECT_JOBS.setdefault(project_id, [])
    return project


@app.get("/projects")
def list_projects():
    projects = sorted(PROJECT_STORE.values(), key=lambda p: p.get("updated_at", 0), reverse=True)
    return {"ok": True, "projects": projects}


@app.get("/projects/{project_id}")
def get_project(project_id: str):
    project = PROJECT_STORE.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    return {"ok": True, "project": project}


@app.post("/projects/{project_id}/run")
def run_project_pipeline(
    project_id: str,
    req: ProjectRunRequest,
    request: Request,
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    project = PROJECT_STORE.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    brief = (req.brief or req.text or project.get("brief") or "").strip()
    if not brief:
        raise HTTPException(status_code=400, detail="Brief text is required.")
    ctx = _brief_context(brief, req)
    structured = _build_structured_brief(brief, ctx)
    concepts = _generate_concepts(brief, ctx, count=3)
    project.update({
        "brief": brief,
        "event_type": ctx["event_type"],
        "style_direction": ctx["style_direction"],
        "structured_brief": structured,
        "analysis": structured["executive_summary"],
        "concepts": concepts,
        "concept_options": concepts,
        "status": "concepts_ready",
        "updated_at": time.time(),
    })
    PROJECT_JOBS.setdefault(project_id, []).append({
        "id": str(uuid.uuid4()),
        "job_kind": "concept",
        "section": "concepts",
        "status": "completed",
        "progress": 100,
        "created_at": time.time(),
        "updated_at": time.time(),
    })
    user_id = _account_user_id(request, x_user_id=x_user_id, authorization=authorization)
    acct = _consume_credits(user_id, AGENT_REGISTRY["CONCEPT_AGENT"]["credit_cost"], "Concept generation", agent_id="CONCEPT_AGENT", project_id=project_id)
    return {
        "ok": True,
        "project_id": project_id,
        "analysis": structured["executive_summary"],
        "structured_brief": structured,
        "concepts": concepts,
        "concept_options": concepts,
        "missing_questions": ctx["missing_questions"],
        "ucd_message": (
            f"{ctx['user_name']}, I have understood the brief at a senior creative director level. "
            "Please review the missing questions, confirm deliverables and choose the concept route before I send work to specialist agents."
        ),
        "account": acct,
        "hourly_rates_inr": AGENT_HOURLY_RATES_INR,
    }


@app.get("/projects/{project_id}/concepts")
def get_project_concepts(project_id: str):
    project = PROJECT_STORE.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    return {"ok": True, "concepts": project.get("concepts", [])}


@app.post("/projects/{project_id}/select-concept")
def select_project_concept(project_id: str, req: ConceptSelectRequest):
    project = PROJECT_STORE.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    concepts = project.get("concepts") or []
    if not concepts:
        raise HTTPException(status_code=400, detail="No concepts generated yet.")
    idx = max(0, min(len(concepts) - 1, int(req.concept_index)))
    project["selected_concept_index"] = idx
    project["selected_concept"] = concepts[idx]
    project["status"] = "concept_selected"
    project["updated_at"] = time.time()
    return {"ok": True, "project_id": project_id, "selected_concept_index": idx, "selected_concept": concepts[idx]}


@app.get("/projects/{project_id}/assets")
def list_project_assets(project_id: str, section: Optional[str] = None):
    assets = PROJECT_ASSETS.get(project_id, [])
    if section:
        sec = section.lower()
        assets = [a for a in assets if (a.get("section") or "").lower() == sec]
    return {"ok": True, "assets": assets}


@app.get("/projects/{project_id}/jobs")
def list_project_jobs(project_id: str):
    return {"ok": True, "jobs": PROJECT_JOBS.get(project_id, [])}


@app.get("/api/projects/{project_id}/moodboard")
def get_api_project_moodboard(project_id: str):
    assets = [a for a in PROJECT_ASSETS.get(project_id, []) if a.get("section") == "moodboard"]
    return {"ok": True, "project_id": project_id, "assets": assets, "moodboard": assets, "images": assets}


@app.post("/api/moodboard/generate")
def generate_moodboard(
    req: MoodboardGenerateRequest,
    request: Request,
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    if req.project_id not in PROJECT_STORE:
        raise HTTPException(status_code=404, detail="Project not found.")
    user_id = _account_user_id(request, x_user_id=x_user_id, authorization=authorization)
    acct = _consume_credits(user_id, AGENT_REGISTRY["MOODBOARD_AGENT"]["credit_cost"], "Moodboard generation", agent_id="MOODBOARD_AGENT", project_id=req.project_id)
    PROJECT_JOBS.setdefault(req.project_id, []).append({
        "id": str(uuid.uuid4()),
        "job_kind": "moodboard",
        "section": "moodboard",
        "status": "completed",
        "progress": 100,
        "created_at": time.time(),
        "updated_at": time.time(),
    })
    assets = _build_moodboard_assets(req.project_id, req.concept_index or 0, req.count or 6)
    PROJECT_STORE[req.project_id]["updated_at"] = time.time()
    return {
        "ok": True,
        "project_id": req.project_id,
        "assets": assets,
        "account": acct,
        "message": "Moodboard generated by a 45-year creative director with detailed mood, material, lighting, ambience, seating and stage logic.",
    }


@app.post("/ai/generate-2d")
def generate_2d_assets(
    req: AssetGenerateRequest,
    request: Request,
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    if req.project_id not in PROJECT_STORE:
        raise HTTPException(status_code=404, detail="Project not found.")
    user_id = _account_user_id(request, x_user_id=x_user_id, authorization=authorization)
    acct = _consume_credits(user_id, AGENT_REGISTRY["GRAPHICS_2D_AGENT"]["credit_cost"], "2D graphics generation", agent_id="GRAPHICS_2D_AGENT", project_id=req.project_id)
    assets = _build_2d_assets(req.project_id, req.concept_index or 0)
    PROJECT_JOBS.setdefault(req.project_id, []).append({
        "id": str(uuid.uuid4()),
        "job_kind": "2d_graphics",
        "section": "2d_graphics",
        "status": "completed",
        "progress": 100,
        "created_at": time.time(),
        "updated_at": time.time(),
    })
    return {"ok": True, "project_id": req.project_id, "assets": assets, "account": acct}


@app.post("/projects/{project_id}/renders/generate-separated")
def generate_3d_renders(
    project_id: str,
    request: Request,
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    if project_id not in PROJECT_STORE:
        raise HTTPException(status_code=404, detail="Project not found.")
    user_id = _account_user_id(request, x_user_id=x_user_id, authorization=authorization)
    acct = _consume_credits(user_id, AGENT_REGISTRY["RENDER_3D_AGENT"]["credit_cost"], "3D render generation", agent_id="RENDER_3D_AGENT", project_id=project_id)
    idx = PROJECT_STORE[project_id].get("selected_concept_index") or 0
    assets = _build_3d_assets(project_id, idx)
    PROJECT_JOBS.setdefault(project_id, []).append({
        "id": str(uuid.uuid4()),
        "job_kind": "3d_renders",
        "section": "renders",
        "status": "completed",
        "progress": 100,
        "created_at": time.time(),
        "updated_at": time.time(),
    })
    return {"ok": True, "project_id": project_id, "assets": assets, "account": acct}


@app.post("/project/{project_id}/departments/build")
def build_departments(project_id: str):
    if project_id not in PROJECT_STORE:
        raise HTTPException(status_code=404, detail="Project not found.")
    idx = PROJECT_STORE[project_id].get("selected_concept_index") or 0
    outputs = _department_outputs(project_id, idx)
    PROJECT_STORE[project_id].update(outputs)
    PROJECT_STORE[project_id]["department_outputs"] = outputs
    PROJECT_STORE[project_id]["updated_at"] = time.time()
    return {"ok": True, "project_id": project_id, **outputs}


@app.post("/projects/{project_id}/presentation/build")
def build_presentation(
    project_id: str,
    req: PresentationBuildRequest,
    request: Request,
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    if project_id not in PROJECT_STORE:
        raise HTTPException(status_code=404, detail="Project not found.")
    user_id = _account_user_id(request, x_user_id=x_user_id, authorization=authorization)
    acct = _consume_credits(user_id, AGENT_REGISTRY["PRESENTATION_AGENT"]["credit_cost"], "Presentation build", agent_id="PRESENTATION_AGENT", project_id=project_id)
    deck = _presentation_deck(project_id, req.concept_index or PROJECT_STORE[project_id].get("selected_concept_index") or 0)
    PROJECT_STORE[project_id]["presentation"] = deck
    PROJECT_STORE[project_id]["updated_at"] = time.time()
    return {"ok": True, "project_id": project_id, "presentation": deck, "account": acct}


@app.get("/briefcraft_backend_connector.js")
def briefcraft_backend_connector():
    path = Path("briefcraft_backend_connector.js")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frontend connector not found.")
    return FileResponse(path, media_type="application/javascript")


@app.get("/account/balance")
def account_balance(
    request: Request,
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    user_id = _account_user_id(request, x_user_id=x_user_id, authorization=authorization)
    acct = _ensure_account(user_id)
    return {
        "ok": True,
        "account": acct,
        "balance": acct["credit_balance"],
        "unit": "tokens",
        "ledger": CREDIT_LEDGER.get(user_id, [])[-10:],
        "hourly_rates_inr": AGENT_HOURLY_RATES_INR,
        "low_balance_threshold": LOW_BALANCE_THRESHOLD,
        "low_balance": int(acct["credit_balance"]) <= LOW_BALANCE_THRESHOLD,
    }


@app.get("/account/rates")
def account_rates():
    return {
        "ok": True,
        "currency": "INR",
        "hourly_rates_inr": AGENT_HOURLY_RATES_INR,
        "low_balance_threshold": LOW_BALANCE_THRESHOLD,
    }


@app.get("/account/packages")
def account_packages():
    return {
        "ok": True,
        "currency": "INR",
        "packages": PACKAGE_PLANS,
        "payment_links_configured": bool(PACKAGE_PAYMENT_LINKS),
    }


@app.post("/account/credits/consume")
def account_consume_credits(
    req: CreditConsumeRequest,
    request: Request,
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    user_id = _account_user_id(request, x_user_id=x_user_id, authorization=authorization)
    acct = _consume_credits(
        user_id,
        req.amount,
        req.reason or "usage",
        agent_id=req.agent_id,
        project_id=req.project_id,
    )
    return {"ok": True, "account": acct, "balance": acct["credit_balance"], "unit": "tokens"}


@app.post("/account/checkout")
def account_checkout(
    req: CheckoutRequest,
    request: Request,
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    user_id = _account_user_id(request, x_user_id=x_user_id, authorization=authorization)
    pkg = next((p for p in PACKAGE_PLANS if p["id"] == req.package_id), None)
    if not pkg:
        raise HTTPException(status_code=404, detail="Package not found.")

    # Real payment processors can be connected by putting hosted checkout links
    # in PACKAGE_PAYMENT_LINKS={"individual_pro":"https://..."}.
    configured_url = PACKAGE_PAYMENT_LINKS.get(req.package_id)
    checkout_id = str(uuid.uuid4())
    if configured_url:
        checkout_url = configured_url
        mode = "payment_link"
    else:
        # Development fallback: creates an auditable checkout object and grants
        # credits immediately so local/demo usage keeps working.
        _grant_credits(user_id, int(pkg.get("credits") or 0), f"Demo activation: {pkg['name']}", package_id=req.package_id)
        acct = _ensure_account(user_id)
        acct["plan_id"] = pkg["id"]
        acct["plan_name"] = pkg["name"]
        acct["account_type"] = pkg["audience"]
        checkout_url = req.success_url or f"/account/checkout/{checkout_id}/success"
        mode = "demo_grant"

    return {
        "ok": True,
        "checkout_id": checkout_id,
        "checkout_url": checkout_url,
        "mode": mode,
        "package": pkg,
        "message": (
            "Payment link created."
            if configured_url
            else "Demo checkout activated. Configure PACKAGE_PAYMENT_LINKS for real payments."
        ),
    }


@app.post("/account/agent")
def account_agent(
    req: AccountAgentRequest,
    request: Request,
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    user_id = _account_user_id(request, x_user_id=x_user_id, authorization=authorization)
    return {"ok": True, "agent_id": "ACCOUNT_AGENT", **_account_agent_reply(user_id, req.message or "")}


@app.post("/agents/run")
def run_agent(
    req: AgentRunRequest,
    request: Request,
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    user_id = _account_user_id(request, x_user_id=x_user_id, authorization=authorization)
    agent_id = (req.agent_id or "").strip().upper()
    agent = AGENT_REGISTRY.get(agent_id)
    if not agent or not agent.get("enabled", True):
        raise HTTPException(status_code=404, detail="Agent not found or disabled.")

    cost = int(agent.get("credit_cost") or 0)
    if cost:
        _consume_credits(user_id, cost, f"Agent run: {agent['name']}", agent_id=agent_id, project_id=req.project_id)

    if agent_id == "ACCOUNT_AGENT":
        payload = _account_agent_reply(user_id, req.message or "")
    elif agent_id == "UCD_AGENT":
        payload = _dump_model(_ucd_response(UCDChatRequest(
            session_id=req.session_id,
            message=req.message or "",
            project_id=req.project_id or "demo-project",
            context=req.context or {},
        )))
    elif agent_id == "CAD_AGENT":
        payload = {
            "message": "CAD Agent is ready. Use /api/cad/pro/generate or /api/cad/pro/trace with the returned endpoint.",
            "endpoint": agent["endpoint"],
            "project_id": req.project_id,
            "context": req.context,
        }
    else:
        payload = {
            "message": f"{agent['name']} accepted the request.",
            "endpoint": agent["endpoint"],
            "context": req.context,
        }

    acct = _ensure_account(user_id)
    return {
        "ok": True,
        "agent": agent,
        "result": payload,
        "account": {
            "credit_balance": acct["credit_balance"],
            "plan_id": acct["plan_id"],
            "plan_name": acct["plan_name"],
            "unit": "tokens",
        },
    }

@app.post("/ucd/chat", response_model=UCDChatResponse)
def ucd_chat(req: UCDChatRequest):
    """
    Human-style UCD orchestrator.
    It understands user intent, asks for missing info, and returns UI + agent actions.
    """
    if not (req.message or "").strip():
        raise HTTPException(status_code=400, detail="Message is required.")
    return _ucd_response(req)


@app.post("/ucd/cad/upload-intent", response_model=UCDChatResponse)
async def ucd_cad_upload_intent(
    file: UploadFile = File(...),
    message: str = Form("Trace this file in CAD"),
    session_id: Optional[str] = Form(None),
    project_id: str = Form("demo-project"),
    title: Optional[str] = Form(None),
):
    """
    Upload-aware UCD handoff.
    The frontend can call this before /api/cad/pro/trace to switch CAD fullscreen,
    show progress, and dispatch the uploaded file to the CAD agent.
    """
    data = await file.read()
    meta = UCDFileMeta(
        filename=file.filename,
        content_type=file.content_type,
        size_bytes=len(data),
    )
    req = UCDChatRequest(
        session_id=session_id,
        message=message,
        project_id=project_id,
        title=title,
        file=meta,
    )
    response = _ucd_response(req)
    response.agent["upload_ready"] = True
    response.agent["upload_filename"] = file.filename
    response.agent["trace_endpoint"] = "/api/cad/pro/trace"
    return response


@app.post("/start-session")
def start_session(req: StartSession):
    sid = str(uuid.uuid4())

    session: Dict[str, Any] = {
        "phase": "INTRO",
        "step":  "ASK_NAME",
        "student": {
            "name":     "",
            "language": req.language or "Hinglish",
            "school":   "",
        },
        "meta": {
            "board":               req.board or "CBSE",
            "class_level":         str(req.class_level or ""),
            "subject":             req.subject or "",
            "book_name":           req.book_name or "",
            "chapter_title":       req.chapter_title or req.chapter_name or "",
            "chapter_code":        req.chapter_code or "",
            "syllabus_chapter_id": req.syllabus_chapter_id or "",
            "author":              "",
        },
        "teacher":  {},
        "teaching": {
            "history":          [],
            "db_chunks":        [],
            "db_all":           [],
            "quiz_pool":        [],
            "chunk_idx":        0,
            "quiz_idx":         0,
            "stage":            "ASK_NAME",
            "read_line":        "",
            "explain_line":     "",
            "chunk_title":      "",
            "pending_question": "",
            "questions_asked":  0,
            "xp":               0,
        },
    }

    # Load DB data — non-blocking, errors are caught
    try:
        bootstrap(session)
    except Exception as e:
        print(f"[bootstrap] error: {e}")
        session["teacher"] = _default_teacher()

    SESSIONS[sid] = session

    # Opening greeting from teacher
    t_name  = session["teacher"].get("teacher_name", "Priya Ma'am")
    chapter = session["meta"].get("chapter_title", "")
    subject = session["meta"].get("subject", "")

    greeting = teacher_say(session,
        f"You are {t_name}. A new student is starting class for '{chapter}' ({subject}). "
        "Deliver a warm, energetic opening greeting: "
        "1) Introduce yourself briefly (1 sentence). "
        "2) Express excitement about today's class (1 sentence). "
        "3) Ask the student their name with warmth. "
        "Hinglish, natural, human. 3 sentences total.",
        max_tokens=120)

    session["teaching"]["stage"] = "ASK_NAME"
    return make_resp(greeting, "INTRO", "ASK_NAME", session, sid, should_listen=True)


@app.post("/student-reply")
def student_reply(req: StudentReply):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404,
            detail="Session not found. Please tap Start Class again.")

    session = SESSIONS[req.session_id]
    msg     = (req.message or "").strip()
    phase   = session.get("phase", "INTRO")

    # ── INTRO phase ────────────────────────────────────────────────────────
    if phase == "INTRO":
        return phase_intro(session, msg, req.session_id)

    # ── CHAPTER_INTRO phase ────────────────────────────────────────────────
    if phase == "CHAPTER_INTRO":
        return phase_chapter_intro(session, msg, req.session_id)

    # ── TEACHING phase ─────────────────────────────────────────────────────
    if phase == "TEACHING":
        return phase_teaching(session, msg, req.session_id)

    # ── DONE phase (after all chunks — quiz offer) ──────────────────────────
    if phase == "DONE":
        msg_low = msg.lower()
        want_quiz = any(w in msg_low for w in
            ["haan","yes","quiz","ok","sure","chalo","haan ji","bilkul","please"])
        if want_quiz:
            session["phase"] = "QUIZ"
            session["teaching"]["stage"] = "quiz_ask"
            return phase_quiz_ask(session, req.session_id)
        return phase_farewell(session, req.session_id)

    # ── QUIZ phase ─────────────────────────────────────────────────────────
    if phase == "QUIZ":
        stage = session["teaching"].get("stage", "")
        if stage == "quiz_answer":
            return phase_quiz_eval(session, msg, req.session_id)
        # After feedback — student said "yes more" or "no"
        msg_low = msg.lower()
        want_more = any(w in msg_low for w in
            ["haan","yes","ok","sure","chalo","aur","more","ek aur"])
        if want_more:
            session["teaching"]["stage"] = "quiz_ask"
            return phase_quiz_ask(session, req.session_id)
        return phase_farewell(session, req.session_id)

    # Fallback
    return make_resp("Chalo aage badhte hain!", phase, "CONTINUE",
        session, req.session_id)


@app.post("/student/homework-help")
def homework_help(req: HomeworkRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = SESSIONS[req.session_id]
    name    = session["student"].get("name", "beta")
    lang    = session["student"].get("language", "Hinglish")
    meta    = session["meta"]

    text = teacher_say(session,
        f"Student {name} needs homework help: '{req.question}'\n"
        f"Subject: {req.subject or meta.get('subject','')} | "
        f"Chapter: {req.chapter_title or req.chapter_name or meta.get('chapter_title','')}\n\n"
        f"In {lang}/Hinglish (3 sentences): "
        "1) Acknowledge warmly. "
        "2) Give a HINT using a daily-life Indian analogy — do NOT give the direct answer. "
        "3) Say: 'Ab try karo — kya lagta hai tumhe?'",
        max_tokens=200)

    return make_resp(text,
        session.get("phase", "TEACHING"), "HOMEWORK_HELP",
        session, req.session_id, should_listen=True,
        extra_meta={"current_chunk_stage": "homework"})


@app.post("/tts")
def tts_endpoint(req: TTSRequest):
    clean = clean_text(req.text or "")
    if not clean:
        return JSONResponse({"ok": False, "error": "Empty text"}, status_code=400)

    # Determine voice
    voice = "nova"
    if req.voice and req.voice in _TTS_VOICE_MAP:
        voice = _TTS_VOICE_MAP[req.voice]
    elif req.teacher_gender:
        gender = req.teacher_gender.lower()
        voice  = _TTS_VOICE_MAP.get(gender, "nova")

    url = generate_audio(clean, voice=voice)
    if not url:
        return JSONResponse({"ok": False, "error": "TTS generation failed"}, status_code=500)

    return {
        "ok":        True,
        "audio_url": url,
        "provider":  "openai",
        "model":     "tts-1-hd",
        "voice":     voice,
    }


@app.get("/health")
def health():
    oai_ok = _get_openai() is not None
    sb_ok  = _get_sb() is not None
    return {
        "ok":                True,
        "service":           "BriefCraftAI Brain v4",
        "status":            "running",
        "render_domain":     RENDER_DOMAIN,
        "active_sessions":   len(SESSIONS),
        "audio_dir":         AUDIO_DIR,
        "audio_dir_exists":  os.path.isdir(AUDIO_DIR),
        "openai_ready":      oai_ok,
        "openai_key_set":    bool(OPENAI_KEY),
        "supabase_pkg":      _SUPABASE_OK,
        "supabase_ready":    sb_ok,
        "supabase_url_set":  bool(SB_URL),
        "supabase_key_set":  bool(SB_KEY),
        "python_version":    __import__("sys").version,
    }

try:
    import fitz  # PyMuPDF
except Exception as exc:
    fitz = None
    print(f"[WARN] PyMuPDF import failed; PDF extraction disabled: {exc}")

def extract_text_from_pdf(pdf_path):
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Install PyMuPDF to enable PDF extraction.")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def create_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i+chunk_size])
        chunks.append({
            "type": "learning",
            "text": chunk_text,
            "question": "इस भाग से आपने क्या सीखा?"
        })

    return chunks


def process_full_book(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"Processing: {pdf_path}")

                text = extract_text_from_pdf(pdf_path)
                chunks = create_chunks(text)

                print(f"✅ Chunks created: {len(chunks)}")
