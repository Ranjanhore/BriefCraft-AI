import hashlib
import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from supabase import create_client as create_supabase_client
except Exception:
    create_supabase_client = None


# =============================================================================
# BRIEFCRAFTAI - CLEAN CREATIVE STUDIO BACKEND
# =============================================================================

APP_NAME = "BriefCraftAI"
API_VERSION = "briefcraft-clean-v1"
RENDER_DOMAIN = os.getenv("RENDER_EXTERNAL_HOSTNAME", "localhost:8000")
MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "media"))
GENERATED_DIR = Path(os.getenv("GENERATED_DIR", "media/generated"))
CAD_PRO_DIR = Path(os.getenv("CAD_PRO_DIR", "media/cad_pro"))
for folder in [MEDIA_DIR, GENERATED_DIR, CAD_PRO_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_NAME, version=API_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

try:
    from cad_engine_pro import router as cad_pro_router

    app.include_router(cad_pro_router)
except Exception as exc:
    print(f"[WARN] CAD Pro router unavailable: {exc}")


# =============================================================================
# DATA MODELS
# =============================================================================

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


class AccountAgentRequest(BaseModel):
    message: Optional[str] = ""


class AgentRunRequest(BaseModel):
    agent_id: str
    message: Optional[str] = ""
    project_id: Optional[str] = None
    session_id: Optional[str] = None
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
    brand: Optional[str] = None
    venue: Optional[str] = None


class ProjectRunRequest(BaseModel):
    text: Optional[str] = None
    brief: Optional[str] = None
    event_type: Optional[str] = None
    style_direction: Optional[str] = None
    user_name: Optional[str] = None
    brand: Optional[str] = None
    venue: Optional[str] = None
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


class ResearchRunRequest(BaseModel):
    project_id: Optional[str] = None
    brief: Optional[str] = None
    brand: Optional[str] = None
    venue: Optional[str] = None
    event_type: Optional[str] = None
    query: Optional[str] = None
    max_results: int = 5


class JobCreateRequest(BaseModel):
    project_id: str
    job_kind: str
    agent_id: Optional[str] = None
    payload: Dict[str, Any] = {}
    run_async: bool = True


# =============================================================================
# IN-MEMORY STORES
# =============================================================================

PROJECT_STORE: Dict[str, Dict[str, Any]] = {}
PROJECT_ASSETS: Dict[str, List[Dict[str, Any]]] = {}
PROJECT_JOBS: Dict[str, List[Dict[str, Any]]] = {}
JOB_STORE: Dict[str, Dict[str, Any]] = {}
RESEARCH_STORE: Dict[str, List[Dict[str, Any]]] = {}
ACCOUNT_STORE: Dict[str, Dict[str, Any]] = {}
CREDIT_LEDGER: Dict[str, List[Dict[str, Any]]] = {}
SESSIONS: Dict[str, Dict[str, Any]] = {}
_SUPABASE_CLIENT = None

DEFAULT_CREDIT_GRANT = int(os.getenv("DEFAULT_CREDIT_GRANT", "2500"))
LOW_BALANCE_THRESHOLD = int(os.getenv("LOW_BALANCE_THRESHOLD", "1000"))

AGENT_HOURLY_RATES_INR: Dict[str, int] = {
    "UCD_AGENT": 8000,
    "CONCEPT_AGENT": 3000,
    "MOODBOARD_AGENT": 5000,
    "GRAPHICS_2D_AGENT": 7000,
    "RENDER_3D_AGENT": 10000,
    "CAD_AGENT": 5000,
    "AV_AGENT": 5000,
    "SOUND_AGENT": 5000,
    "ELECTRIC_ENGINEER_AGENT": 5000,
    "LIGHTING_AGENT": 5000,
    "SHOW_RUNNER_AGENT": 5000,
    "PRESENTATION_AGENT": 7000,
}

PACKAGE_PLANS = [
    {
        "id": "free_trial",
        "name": "Free Trial",
        "audience": "individual",
        "price_inr": 0,
        "credits": DEFAULT_CREDIT_GRANT,
        "billing": "trial",
        "features": ["Try UCD, concept, moodboard, CAD and visual agents"],
        "recommended": False,
    },
    {
        "id": "individual_pro",
        "name": "Individual Pro",
        "audience": "individual",
        "price_inr": 9999,
        "credits": 25000,
        "billing": "monthly",
        "features": ["More credits", "Creative agents", "CAD/PDF exports", "Pitch support"],
        "recommended": True,
    },
    {
        "id": "studio_agency",
        "name": "Studio / Agency",
        "audience": "institution",
        "price_inr": 49999,
        "credits": 150000,
        "billing": "monthly",
        "features": ["Team production usage", "Advanced CAD", "Presentation workflows", "Priority capacity"],
        "recommended": False,
    },
]


AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_backend_agent(
    agent_id: str,
    name: str,
    description: str,
    endpoint: str,
    capabilities: Optional[List[str]] = None,
    credit_cost: int = 25,
    category: str = "creative",
) -> None:
    AGENT_REGISTRY[agent_id] = {
        "id": agent_id,
        "name": name,
        "description": description,
        "endpoint": endpoint,
        "capabilities": capabilities or [],
        "credit_cost": credit_cost,
        "hourly_rate_inr": AGENT_HOURLY_RATES_INR.get(agent_id, 0),
        "category": category,
        "enabled": True,
    }


def register_default_agents() -> None:
    agents = [
        ("ACCOUNT_AGENT", "Account Agent", "Manages credits, packages, billing, rate cards and low-balance alerts.", "/account/agent", ["credits", "billing", "rates"], 0, "account"),
        ("UCD_AGENT", "Universal Creative Director", "60-year senior creative director. Understands brand, brief, venue, category, budget and routes single-job or full-project workflows.", "/ucd/chat", ["brief", "brand", "orchestration", "single-job-routing", "cad-fullscreen"], 15, "orchestrator"),
        ("CONCEPT_AGENT", "Concept Agent", "45-year creative director. Thinks twice, researches brand/category/technology and creates multiple distinct product-launch and event ideas.", "/projects/{project_id}/run", ["concept", "product-launch", "technology", "research"], 60, "creative"),
        ("RESEARCH_AGENT", "Research Agent", "Researches brand history, past events, category references, venue images/layout clues and useful technologies before creative output.", "/research/run", ["research", "brand", "venue", "category", "technology"], 40, "creative"),
        ("MOODBOARD_AGENT", "Moodboard Agent", "45-year creative director for mood, materials, lighting, ambience, seating, stage look and visual explanation.", "/api/moodboard/generate", ["moodboard", "materials", "lighting", "ambience"], 120, "visual"),
        ("GRAPHICS_2D_AGENT", "2D Graphics Art Director", "45-year art director for invites, badges, registration backdrop, table facade, stage backdrop, copy and print sizes.", "/ai/generate-2d", ["2d", "invite", "badge", "backdrop", "copy"], 100, "visual"),
        ("RENDER_3D_AGENT", "3D Render Design Agent", "40-year 3D designer. Creates 3D scenes, structures and realistic renders using CAD dimensions and venue references.", "/projects/{project_id}/renders/generate-separated", ["3d", "renders", "venue", "structures"], 180, "visual"),
        ("CAD_AGENT", "Professional CAD Agent", "Creates or traces layouts, dimensions, production drawings, SVG, PDF and DXF.", "/api/cad/pro/generate", ["cad", "layout", "trace", "pdf", "dxf"], 250, "production"),
        ("AV_AGENT", "Audio Visual Agent", "Plans LED, projection, playback, cameras, switching and show tech.", "/agents/run", ["av", "led", "projection"], 80, "production"),
        ("SOUND_AGENT", "Sound Engineer", "Plans sound design, music beds, mic plan, cueing and audio flow.", "/agents/run", ["sound", "mics", "cueing"], 70, "production"),
        ("ELECTRIC_ENGINEER_AGENT", "Electric Engineer", "Plans DB, power loads, cable routes, generator backup and electrical safety.", "/agents/run", ["power", "db", "electrical"], 90, "production"),
        ("LIGHTING_AGENT", "Lighting Designer", "Plans lighting looks, reveal cues, atmosphere, fixtures and show states.", "/agents/run", ["lighting", "cues", "fixtures"], 90, "production"),
        ("SHOW_RUNNER_AGENT", "Show Runner Agent", "Builds run of show, cue sheet, backstage notes and show caller flow.", "/agents/run", ["show-runner", "cue-sheet"], 90, "production"),
        ("PRESENTATION_AGENT", "Client Pitch Presentation Agent", "Builds client pitch decks with strategy, concept, moodboard, 2D, 3D, flow and presenter copy.", "/projects/{project_id}/presentation/build", ["presentation", "pitch", "deck"], 120, "export"),
    ]
    for item in agents:
        register_backend_agent(*item)


register_default_agents()


# =============================================================================
# HELPERS
# =============================================================================

CREATIVE_INDUSTRIES = {
    "product launch": ["launch", "product", "automotive", "car", "phone", "watch", "device"],
    "exhibition": ["exhibition", "expo", "stall", "booth", "trade show", "pavilion"],
    "wedding": ["wedding", "sangeet", "mehendi", "reception", "mandap"],
    "activation": ["activation", "mall", "roadshow", "sampling", "pop-up"],
    "concert": ["concert", "festival", "music", "artist", "performance"],
    "corporate event": ["corporate", "conference", "summit", "award", "agm", "dealer"],
    "birthday party": ["birthday", "party", "celebration"],
}

CONCEPT_ARCHETYPES = [
    {"key": "reveal_lab", "name": "Reveal Lab", "logic": "technology-led anticipation, precision reveal and hands-on discovery", "spatial": "a controlled lab-like journey"},
    {"key": "brand_city", "name": "Brand City", "logic": "turning the brand into a walkable district with zones and story streets", "spatial": "district-style guest circulation"},
    {"key": "ritual_to_future", "name": "Ritual to Future", "logic": "combining cultural emotion with future-facing technology", "spatial": "ceremonial arrival into futuristic reveal"},
    {"key": "cinematic_sequence", "name": "Cinematic Sequence", "logic": "a film-like sequence with acts, light, sound and hero close-up", "spatial": "a theatre-like reveal arc"},
    {"key": "social_engine", "name": "Social Engine", "logic": "content capture, interaction and shareable moments as the main experience", "spatial": "a camera-first guest flow"},
    {"key": "crafted_luxury", "name": "Crafted Luxury", "logic": "materials, tactility and service detail carrying the brand story", "spatial": "gallery-like hospitality and reveal"},
]


def now_ts() -> float:
    return time.time()


def get_supabase():
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT
    if not create_supabase_client:
        return None
    url = os.getenv("SUPABASE_URL", "").strip()
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.getenv("SUPABASE_SERVICE_KEY", "").strip()
        or os.getenv("SUPABASE_KEY", "").strip()
    )
    if not url or not key:
        return None
    try:
        _SUPABASE_CLIENT = create_supabase_client(url, key)
        return _SUPABASE_CLIENT
    except Exception as exc:
        print(f"[WARN] Supabase client unavailable: {exc}")
        return None


def sb_table_insert(table: str, row: Dict[str, Any]) -> None:
    sb = get_supabase()
    if not sb:
        return
    try:
        sb.table(table).insert(row).execute()
    except Exception as exc:
        print(f"[WARN] Supabase insert {table} failed: {exc}")


def sb_table_upsert(table: str, row: Dict[str, Any]) -> None:
    sb = get_supabase()
    if not sb:
        return
    try:
        sb.table(table).upsert(row).execute()
    except Exception as exc:
        print(f"[WARN] Supabase upsert {table} failed: {exc}")


def sb_table_select_one(table: str, key: str, value: Any) -> Optional[Dict[str, Any]]:
    sb = get_supabase()
    if not sb:
        return None
    try:
        res = sb.table(table).select("*").eq(key, value).limit(1).execute()
        data = getattr(res, "data", None) or []
        return data[0] if data else None
    except Exception as exc:
        print(f"[WARN] Supabase select {table} failed: {exc}")
        return None


def persist_project(project: Dict[str, Any]) -> None:
    sb_table_upsert("bc_projects", {
        "id": project.get("id") or project.get("project_id"),
        "project_id": project.get("project_id") or project.get("id"),
        "project_name": project.get("project_name") or project.get("title"),
        "title": project.get("title") or project.get("project_name"),
        "brief": project.get("brief"),
        "event_type": project.get("event_type"),
        "brand": project.get("brand"),
        "venue": project.get("venue"),
        "style_direction": project.get("style_direction"),
        "status": project.get("status"),
        "data": project,
        "updated_at": project.get("updated_at") or now_ts(),
    })


def persist_asset(asset: Dict[str, Any]) -> None:
    sb_table_upsert("bc_assets", {
        "id": asset.get("id"),
        "project_id": asset.get("project_id"),
        "section": asset.get("section"),
        "asset_type": asset.get("asset_type"),
        "title": asset.get("title"),
        "description": asset.get("description"),
        "preview_url": asset.get("preview_url") or asset.get("image_url"),
        "status": asset.get("status", "ready"),
        "data": asset,
        "created_at": asset.get("created_at") or now_ts(),
    })


def persist_job(job: Dict[str, Any]) -> None:
    sb_table_upsert("bc_jobs", {
        "id": job.get("id"),
        "project_id": job.get("project_id"),
        "job_kind": job.get("job_kind"),
        "agent_id": job.get("agent_id"),
        "section": job.get("section"),
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "data": job,
        "created_at": job.get("created_at") or now_ts(),
        "updated_at": job.get("updated_at") or now_ts(),
    })


def persist_research(item: Dict[str, Any]) -> None:
    sb_table_upsert("bc_research", {
        "id": item.get("id"),
        "project_id": item.get("project_id"),
        "brand": item.get("brand"),
        "venue": item.get("venue"),
        "event_type": item.get("event_type"),
        "query": item.get("query"),
        "source": item.get("source"),
        "data": item,
        "created_at": item.get("created_at") or now_ts(),
    })


def load_project(project_id: str) -> Optional[Dict[str, Any]]:
    project = PROJECT_STORE.get(project_id)
    if project:
        return project
    row = sb_table_select_one("bc_projects", "project_id", project_id) or sb_table_select_one("bc_projects", "id", project_id)
    if row:
        data = row.get("data") if isinstance(row.get("data"), dict) else row
        PROJECT_STORE[project_id] = data
        return data
    return None


def dump_model(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def safe_title(text: str, fallback: str = "Creative Project") -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return fallback
    title = re.split(r"[.!?\n]", text)[0].strip()
    return (title[:82].rstrip() + "...") if len(title) > 84 else title


def stable_seed(*parts: Any) -> int:
    raw = "||".join(str(p or "") for p in parts)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12], 16)


def rng_for(*parts: Any) -> random.Random:
    return random.Random(stable_seed(*parts))


def public_url(rel_path: str) -> str:
    rel = rel_path.replace("\\", "/").lstrip("/")
    if RENDER_DOMAIN.startswith("localhost") or RENDER_DOMAIN.startswith("127.0.0.1"):
        return f"http://{RENDER_DOMAIN}/{rel}"
    return f"https://{RENDER_DOMAIN}/{rel}"


def xml_escape(value: Any) -> str:
    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def detect_industry(brief: str) -> str:
    low = (brief or "").lower()
    for industry, words in CREATIVE_INDUSTRIES.items():
        if any(w in low for w in words):
            return industry
    return "creative experience"


def extract_guest_count(brief: str) -> Optional[str]:
    m = re.search(r"\b(\d{2,6})\s*(pax|people|guests|attendees|audience|seats)\b", brief or "", re.I)
    return f"{m.group(1)} {m.group(2)}" if m else None


def extract_dimensions(brief: str) -> Optional[str]:
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(m|meter|meters|ft|feet|mm|in)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(m|meter|meters|ft|feet|mm|in)?\b", brief or "", re.I)
    if not m:
        return None
    return f"{m.group(1)} x {m.group(3)} {m.group(4) or m.group(2) or 'm'}"


def extract_budget(brief: str, fallback: Optional[str] = None) -> str:
    m = re.search(r"\b(?:budget|cost|spend)\D{0,10}([\d,.]+)\s*(lakh|lakhs|cr|crore|k|rs|inr|aed|usd)?", brief or "", re.I)
    if m:
        return " ".join(x for x in m.groups() if x)
    return fallback or "Budget to be confirmed"


def extract_brand(brief: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.strip()
    patterns = [
        r"(?:brand|client|company)\s*(?:is|:|-)?\s*([A-Z][A-Za-z0-9& .'-]{2,48})",
        r"(?:for|by)\s+([A-Z][A-Za-z0-9& .'-]{2,48})\s+(?:brand|launch|event|exhibition|wedding|activation)",
    ]
    for pat in patterns:
        m = re.search(pat, brief or "")
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip(" .,-")
    return "Brand to be confirmed"


def extract_venue(brief: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.strip()
    patterns = [
        r"(?:venue|location)\s*(?:is|:|-)?\s*([A-Z][A-Za-z0-9& .,'-]{3,70})",
        r"\bat\s+([A-Z][A-Za-z0-9& .,'-]{3,70})\s+(?:in|for|with|on)\b",
    ]
    for pat in patterns:
        m = re.search(pat, brief or "")
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip(" .,-")
    return "Venue to be confirmed"


def missing_brief_questions(brief: str) -> List[str]:
    checks = [
        (r"\b\d{2,6}\s*(pax|people|guests|attendees|audience|seats)\b", "How many guests / visitors / seats should I plan for?"),
        (r"\b\d+(?:\.\d+)?\s*(m|meter|meters|ft|feet|mm|in)?\s*[x×]\s*\d+", "What are the final venue or stall dimensions?"),
        (r"\b(budget|cost|spend|lakh|crore|premium|luxury|standard)\b", "What budget level should I assume: luxury, premium, standard or cost-controlled?"),
        (r"\b(logo|brand|guideline|palette|sponsor|copy|content)\b", "Do you have brand guidelines, logo files, mandatory copy or sponsor hierarchy?"),
        (r"\b(date|time|schedule|duration|show|opening|closing)\b", "What is the date, guest arrival time, show time and total duration?"),
        (r"\b(deliverable|moodboard|2d|3d|cad|sound|lighting|presentation|deck|layout)\b", "Which deliverables do you need: concepts, moodboard, 2D, 3D, CAD, sound, lighting, show runner, presentation or downloads?"),
    ]
    return [q for pat, q in checks if not re.search(pat, brief or "", re.I)][:5]


def research_directives(ctx: Dict[str, Any]) -> Dict[str, Any]:
    brand = ctx.get("brand") or "the brand"
    venue = ctx.get("venue") or "the venue"
    category = ctx.get("event_type") or ctx.get("industry") or "creative event"
    return {
        "brand_research": [
            f"Research {brand}'s brand world, product design language, tone, campaigns and past event style.",
            f"Look for launches, exhibitions, activations or ceremonies by {brand}.",
            "Identify what the brand must never look like: wrong colours, gimmicks, low-quality material cues or off-category language.",
        ],
        "category_research": [
            f"Research best global {category} examples from luxury, technology, automotive, fashion, entertainment and culture.",
            "Study reveal mechanics, guest journey, interaction, stage design, content capture and hospitality logic.",
            "Extract strategy only; never copy a reference literally.",
        ],
        "technology_research": [
            "Consider RFID/NFC invites or name badges, QR guest journey, AR filters, projection mapping, kinetic lighting, LED mesh, hologauze, transparent OLED, spatial audio, live polling, generative visuals, interactive touch tables, sensor-triggered reveal, drone/camera live feed and automated content capture.",
            "Recommend technology only when it improves story, operations, guest comfort or measurable content value.",
        ],
        "venue_research": [
            f"If venue images or plans are uploaded, extract dimensions, sightlines, entry path, rigging, circulation and production risks.",
            f"If not uploaded, search public images, capacity sheets, layout PDFs and location clues for {venue}.",
            "For outdoor venues, use map/earth-style location research to estimate usable footprint, access, audience zones, backstage, power, emergency paths and weather risk. Mark estimated dimensions clearly.",
        ],
    }


def search_web(query: str, max_results: int = 5) -> Dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {"provider": "none", "query": query, "results": []}
    max_results = max(1, min(10, int(max_results or 5)))
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    serper_key = os.getenv("SERPER_API_KEY", "").strip()
    try:
        if tavily_key:
            res = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": tavily_key, "query": query, "max_results": max_results, "search_depth": "advanced"},
                timeout=20,
            )
            res.raise_for_status()
            data = res.json()
            return {
                "provider": "tavily",
                "query": query,
                "results": [
                    {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("content")}
                    for r in (data.get("results") or [])[:max_results]
                ],
            }
        if serper_key:
            res = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                json={"q": query, "num": max_results},
                timeout=20,
            )
            res.raise_for_status()
            data = res.json()
            organic = data.get("organic") or []
            return {
                "provider": "serper",
                "query": query,
                "results": [
                    {"title": r.get("title"), "url": r.get("link"), "snippet": r.get("snippet")}
                    for r in organic[:max_results]
                ],
            }
    except Exception as exc:
        print(f"[WARN] research search failed for {query!r}: {exc}")
    return {
        "provider": "fallback",
        "query": query,
        "results": [],
        "note": "Set TAVILY_API_KEY or SERPER_API_KEY to enable live web research.",
    }


def run_research_pack(ctx: Dict[str, Any], max_results: int = 5, project_id: Optional[str] = None) -> Dict[str, Any]:
    brand = ctx.get("brand") or "brand"
    venue = ctx.get("venue") or "venue"
    event_type = ctx.get("event_type") or ctx.get("industry") or "event"
    queries = {
        "brand_past_events": f"{brand} past events product launches activations exhibitions",
        "category_global_references": f"best global {event_type} event experience product launch technology ideas",
        "technology_references": f"latest event technology RFID NFC AR projection mapping interactive guest experience",
        "venue_layout_references": f"{venue} venue layout capacity images floor plan",
    }
    pack = {
        "id": str(uuid.uuid4()),
        "project_id": project_id,
        "brand": brand,
        "venue": venue,
        "event_type": event_type,
        "source": "research_agent",
        "created_at": now_ts(),
        "queries": queries,
        "results": {key: search_web(query, max_results) for key, query in queries.items()},
        "strategy_notes": [
            "Use references for strategy and production intelligence only; never copy visuals literally.",
            "Convert brand research into tone, material, technology and guest-experience decisions.",
            "Convert venue research into dimension assumptions, sightline risks, entry flow and CAD checks.",
        ],
    }
    if project_id:
        RESEARCH_STORE.setdefault(project_id, []).append(pack)
    persist_research(pack)
    return pack


def brief_context(brief: str, req: Optional[ProjectRunRequest] = None) -> Dict[str, Any]:
    industry = (req.event_type if req and req.event_type else None) or detect_industry(brief)
    ctx = {
        "title": safe_title(brief, "Production-Ready Creative Brief"),
        "industry": industry,
        "event_type": industry,
        "brand": extract_brand(brief, req.brand if req else None),
        "venue": extract_venue(brief, req.venue if req else None),
        "guest_count": extract_guest_count(brief) or "To be confirmed",
        "dimensions": extract_dimensions(brief) or "To be confirmed",
        "budget": extract_budget(brief, req.budget_range if req else None),
        "style_direction": (req.style_direction if req and req.style_direction else "Premium creative"),
        "user_name": (req.user_name if req and req.user_name else "there"),
        "deliverables": (req.deliverables if req else []) or [],
        "missing_questions": missing_brief_questions(brief),
    }
    ctx["research"] = research_directives(ctx)
    return ctx


def create_svg_asset(project_id: str, title: str, subtitle: str, palette: List[str], section: str, idx: int) -> str:
    filename = f"{project_id}_{section}_{idx}_{uuid.uuid4().hex[:8]}.svg"
    path = GENERATED_DIR / filename
    color_a = palette[0] if palette else "#07080d"
    color_b = palette[1] if len(palette) > 1 else "#c9a84c"
    color_c = palette[2] if len(palette) > 2 else "#f6d27a"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
<defs><linearGradient id="g" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stop-color="{color_a}"/><stop offset="0.56" stop-color="{color_b}"/><stop offset="1" stop-color="{color_c}"/></linearGradient></defs>
<rect width="1600" height="900" fill="#07080d"/>
<rect x="48" y="48" width="1504" height="804" rx="34" fill="url(#g)" opacity="0.86"/>
<circle cx="1260" cy="170" r="260" fill="#ffffff" opacity="0.08"/>
<circle cx="250" cy="760" r="300" fill="#000000" opacity="0.20"/>
<path d="M160 670 C420 450 640 740 890 470 C1100 245 1260 385 1450 245" fill="none" stroke="#fff7d6" stroke-width="10" opacity="0.34"/>
<text x="110" y="180" font-family="Arial, sans-serif" font-size="32" font-weight="700" fill="#fff7d6" letter-spacing="5">{xml_escape(section.upper())}</text>
<text x="110" y="295" font-family="Arial, sans-serif" font-size="74" font-weight="800" fill="#ffffff">{xml_escape(title[:34])}</text>
<foreignObject x="110" y="335" width="1180" height="260"><div xmlns="http://www.w3.org/1999/xhtml" style="font: 34px Arial, sans-serif; line-height:1.35; color:#fff7d6;">{xml_escape(subtitle[:220])}</div></foreignObject>
<text x="110" y="800" font-family="Arial, sans-serif" font-size="24" fill="#fff7d6" opacity="0.82">BriefCraftAI Creative Engine</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return public_url(f"media/generated/{filename}")


def store_asset(project_id: str, section: str, asset_type: str, title: str, description: str, url: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        "image_url": url,
        "master_url": url,
        "source_file_url": url,
        "created_at": now_ts(),
        "meta": meta or {},
    }
    PROJECT_ASSETS.setdefault(project_id, []).append(asset)
    persist_asset(asset)
    return asset


def selected_concept(project_id: str, concept_index: Optional[int] = 0) -> Dict[str, Any]:
    project = PROJECT_STORE.get(project_id, {})
    concepts = project.get("concepts") or []
    if not concepts:
        return {}
    idx = concept_index if concept_index is not None else project.get("selected_concept_index", 0)
    try:
        idx = max(0, min(len(concepts) - 1, int(idx)))
    except Exception:
        idx = 0
    return concepts[idx]


# =============================================================================
# CREATIVE ENGINE
# =============================================================================

def generate_concepts(brief: str, ctx: Dict[str, Any], count: int = 3) -> List[Dict[str, Any]]:
    rng = rng_for("concepts", brief, ctx.get("brand"), ctx.get("industry"), ctx.get("style_direction"))
    archetypes = CONCEPT_ARCHETYPES[:]
    rng.shuffle(archetypes)
    selected = archetypes[: max(3, min(5, count))]
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
        materials = material_sets[(i + rng.randrange(len(material_sets))) % len(material_sets)]
        name = f"{arch['name']} - {ctx['brand'] if ctx['brand'] != 'Brand to be confirmed' else ctx['title'][:32]}"
        one_liner = f"A {ctx['style_direction']} {ctx['industry']} direction that {rng.choice(verbs)} the brief through {arch['logic']}."
        concepts.append({
            "id": f"concept_{i + 1}_{arch['key']}",
            "name": name,
            "title": name,
            "style": arch["name"],
            "one_liner": one_liner,
            "summary": one_liner,
            "brand_understanding": f"Understand {ctx['brand']}'s product language, past campaigns, audience expectation, visual discipline and category codes before approving this route.",
            "research_plan": ctx["research"],
            "global_reference_angle": f"Research global {ctx['event_type']} and competitor/category experiences. Use strategy, not copied visuals.",
            "technology_opportunities": [
                "RFID/NFC invite or name badge for guest check-in and personalized content triggers",
                "Projection mapping or LED mesh for the hero reveal if venue sightline supports it",
                "AR/social filter or generative visual capture moment for shareable content",
                "Sensor-triggered lighting/audio cue for product reveal or VIP movement",
            ],
            "big_idea": f"The concept turns the brief into {arch['logic']}. It gives all departments one shared story from arrival to the designed peak moment.",
            "why_best": f"This route is distinctive, executable and scalable for budget: {ctx['budget']}. It gives CAD, 2D, 3D, lighting, sound and show-running a clear brief.",
            "experience_flow": [
                f"Arrival: first-view moment establishes {ctx['style_direction']} tone and guest orientation.",
                f"Build-up: guests move through {arch['spatial']} with controlled rhythm.",
                "Hero moment: lighting, sound, scenic structure, screen content and technology converge into one memorable peak.",
                "After-moment: hospitality, photo capture, interaction and content value extend the experience.",
            ],
            "design_language": f"{', '.join(materials)} with disciplined brand hierarchy, controlled negative space and camera-safe stage composition.",
            "materials": materials,
            "hero_moments": ["First-look entry composition", "Main reveal / ceremony peak", "High-value photo and content capture moment"],
            "cad_direction": f"Plan dimensions around {ctx['dimensions']}. Reserve guest flow, FOH sightline, BOH/service access, power routes and stage/feature footprint before 3D starts.",
            "graphics_2d_brief": {
                "invite_copy": f"You are invited to experience {ctx['title']}",
                "registration_backdrop": "Large brand mark, concept pattern, directional welcome copy and sponsor lockup.",
                "table_facade": "Low-height branded facade with premium material texture and subtle edge lighting.",
                "stage_backdrop": "Hero concept title, layered depth graphics and screen-safe logo hierarchy.",
                "sizes": ["Invite 1080x1920", "Name badge 90mm x 55mm", "Backdrop 16ft x 8ft", "Table facade 8ft x 3ft", "Stage backdrop as per CAD"],
                "handoff_to_3d": "Render invite card, name badge, registration counter skin, facade texture, signages and stage panels as realistic 3D objects when physical.",
                "handoff_to_cad": "Extract print size, bleed, frame depth, install height, viewing distance and mounting method for fabrication.",
            },
            "render_3d_brief": f"Use CAD dimensions first. Match actual venue images if uploaded. If not, research public images/layout clues for {ctx['venue']} and label assumptions.",
            "lighting_brief": "Use warm key/fill for premium emotion, sharper moving light at reveal, practical glow in guest zones and dimmable states for show flow.",
            "sound_brief": "Build subtle arrival ambience, gradual musical tension, hero stinger at reveal and controlled post-moment lounge bed.",
            "production_watchouts": ["Confirm venue ceiling height and rigging load", "Lock exact brand assets and mandatory copy", "Approve power, BOH and emergency access before final render"],
        })
    return concepts


def build_structured_brief(brief: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": ctx["title"],
        "industry": ctx["industry"],
        "brand": ctx["brand"],
        "venue": ctx["venue"],
        "guest_count": ctx["guest_count"],
        "dimensions": ctx["dimensions"],
        "budget": ctx["budget"],
        "research_directives": ctx["research"],
        "executive_summary": (
            f"This is a {ctx['style_direction']} {ctx['industry']} brief for {ctx['brand']}. "
            "UCD must understand brand, category, venue, audience, commercial objective and production risk before output. "
            "If the user asks for one specific job, route directly to that specialist instead of forcing the full workflow."
        ),
        "objective": "Create a memorable, commercially useful, production-ready creative experience with clear deliverables and department handoffs.",
        "missing_questions": ctx["missing_questions"],
        "recommended_deliverables": ["Project Brief", "Concepts", "Mood Board", "CAD Layout", "2D Graphics", "3D Renders", "Lighting", "Sound", "Show Runner", "Client Pitch Presentation"],
        "agent_sequence": [
            "UCD confirms missing details, brand, venue, deliverables and budget thinking.",
            "Concept Agent researches brand/category/technology and creates multiple distinct routes.",
            "Moodboard Agent expands the selected concept into mood/material/lighting/ambience/seating/stage detail.",
            "CAD Agent fixes dimensions and layout logic before 3D.",
            "2D Art Director creates copy, print sizes and physical graphic handoff.",
            "3D Agent renders from CAD dimensions and venue references.",
            "3D approvals return to CAD for top/right/left/front production drawings and PDF/DXF export.",
            "Presentation Agent packages everything for client pitch.",
        ],
        "original_brief": brief,
    }


def build_moodboard_assets(project_id: str, concept_index: int = 0, count: int = 6) -> List[Dict[str, Any]]:
    concept = selected_concept(project_id, concept_index)
    frames = [
        ("Overall Mood", "emotional climate and first visual promise"),
        ("Materials", "tactile finishes, surfaces, drapes, flooring, facade and scenic material logic"),
        ("Lighting", "contrast, reveal cue, warmth, shadow, photo tone and ambience"),
        ("Seating", "guest comfort, VIP hierarchy, circulation and social behaviour"),
        ("Stage Look", "backdrop, scenic volume, screen focus, reveal line and focal depth"),
        ("Brand Details", "logo behaviour, invite, badge, typography, facade, wayfinding and content moments"),
    ][: max(1, min(8, count))]
    palettes = [["#111827", "#c9a84c", "#f6d27a"], ["#141414", "#7b2d26", "#f4c6a6"], ["#08111f", "#3aa6ff", "#d8f2ff"], ["#102015", "#7bb274", "#dceecf"], ["#210b2c", "#c14a5a", "#ffd6df"]]
    assets = []
    for i, (title, role) in enumerate(frames):
        desc = f"{title} for {concept.get('name', 'selected concept')}: this frame explains {role}. It supports the concept because: {concept.get('why_best', concept.get('summary', 'the creative route'))}"
        url = create_svg_asset(project_id, title, desc, palettes[i % len(palettes)], "moodboard", i + 1)
        assets.append(store_asset(project_id, "moodboard", "moodboard", title, desc, url, {"concept_index": concept_index, "creative_director_note": desc}))
    return assets


def build_2d_assets(project_id: str, concept_index: int = 0) -> List[Dict[str, Any]]:
    concept = selected_concept(project_id, concept_index)
    briefs = concept.get("graphics_2d_brief") or {}
    outputs = [
        ("Invite / Digital Save-the-Date", briefs.get("invite_copy", "Premium invitation headline and event details")),
        ("Name Badge", "Guest name badge with RFID/NFC/QR logic, role coding and premium finish."),
        ("Registration Backdrop", briefs.get("registration_backdrop", "Welcome backdrop with brand hierarchy")),
        ("Table Facade", briefs.get("table_facade", "Reception desk facade with material and lighting detail")),
        ("Stage Backdrop", briefs.get("stage_backdrop", "Main stage graphic system and LED-safe composition")),
    ]
    assets = []
    for i, (title, desc) in enumerate(outputs):
        url = create_svg_asset(project_id, title, desc, ["#0b0d14", "#c9a84c", "#ffffff"], "2d_graphics", i + 1)
        assets.append(store_asset(project_id, "2d_graphics", "2d_graphic", title, desc, url, {
            "concept_index": concept_index,
            "sizes": briefs.get("sizes", []),
            "handoff_to_3d": briefs.get("handoff_to_3d", "Render approved physical graphics as realistic 3D objects."),
            "handoff_to_cad": briefs.get("handoff_to_cad", "Extract fabrication dimensions."),
        }))
    return assets


def build_3d_assets(project_id: str, concept_index: int = 0) -> List[Dict[str, Any]]:
    concept = selected_concept(project_id, concept_index)
    project = PROJECT_STORE.get(project_id, {})
    venue = (project.get("structured_brief") or {}).get("venue") or "venue to be researched"
    views = [
        ("3D Invite / Name Badge Render", "realistic product-style render of invite/name badge/printed collateral using approved 2D graphics"),
        ("Entry Portal 3D View", "dimensioned first-view arrival portal based on CAD flow"),
        ("Main Stage 3D View", "stage/backdrop/LED/scenic structure with accurate footprint from CAD"),
        ("Experience Zone 3D View", "guest interaction zone with seating, circulation and light mood"),
    ]
    assets = []
    for i, (title, desc) in enumerate(views):
        full_desc = f"{desc}. 3D agent must follow CAD dimensions first: {concept.get('cad_direction', 'confirm dimensions before render')}."
        url = create_svg_asset(project_id, title, full_desc, ["#07111f", "#3a6bff", "#dbeafe"], "renders", i + 1)
        assets.append(store_asset(project_id, "renders", "3d_render", title, full_desc, url, {
            "concept_index": concept_index,
            "render_agent_note": concept.get("render_3d_brief", ""),
            "venue_reference_instruction": f"Use uploaded venue images if available. If not, research public images/layout clues for {venue}; match proportions and label assumptions.",
            "cad_feedback_loop": "After 3D approval, CAD must generate top, right, left, front and section production drawings.",
        }))
    return assets


def build_pdf_assets(project_id: str, concept_index: int = 0) -> List[Dict[str, Any]]:
    concept = selected_concept(project_id, concept_index)
    drawings = [
        ("CAD Production Drawing - Top View", "top-view production drawing with dimensions, zones, print panels, stage, seating and power routes"),
        ("CAD Production Drawing - Right Elevation", "right elevation showing heights, truss, stage backdrop, signages and install levels"),
        ("CAD Production Drawing - Left Elevation", "left elevation showing scenic depth, access, cable routes and finish notes"),
        ("CAD Production Drawing - Front Elevation", "client-facing elevation of main stage or booth facade"),
    ]
    assets = []
    for i, (title, desc) in enumerate(drawings):
        note = f"{desc}. Generated after CAD/3D coordination for {concept.get('name', 'selected concept')}."
        url = create_svg_asset(project_id, title, note, ["#10131c", "#9f7a28", "#f6d27a"], "pdf", i + 1)
        assets.append(store_asset(project_id, "pdf", "production_pdf", title, note, url, {"concept_index": concept_index, "drawing_view": title}))
    return assets


def department_outputs(project_id: str, concept_index: int = 0) -> Dict[str, Any]:
    concept = selected_concept(project_id, concept_index)
    return {
        "sound_data": {"designer": "Senior sound engineer", "direction": concept.get("sound_brief", ""), "cue_logic": ["arrival bed", "anticipation build", "hero stinger", "post-reveal lounge bed"], "mic_plan": "Confirm MC, presenter, panel and backup microphone count."},
        "lighting_data": {"designer": "Senior lighting designer", "direction": concept.get("lighting_brief", ""), "looks": ["arrival warmth", "brand wash", "reveal contrast", "photo moment glow"], "notes": "Confirm rigging height, fixture inventory, haze permission and power load."},
        "showrunner_data": {"show_caller": "Senior show runner", "run_of_show": ["Guest arrival", "Opening cue", "Brand story", "Hero reveal", "Guest interaction", "Closing handover"], "backstage_notes": "Lock cue numbers after final venue tech check."},
        "electrical_data": {"engineer": "Electric engineer", "scope": "DB positions, power routing, emergency clearance, LED/stage/sound load assumptions.", "watchouts": ["separate sound and lighting power where possible", "protect guest cable crossings", "confirm generator backup"]},
        "cad_pdf_direction": {"views_required": ["top", "right elevation", "left elevation", "front elevation", "section where needed"], "save_section": "pdf", "rule": "Whatever 3D creates must return to CAD for production drawings, dimensions, fabrication notes and PDF/DXF export."},
    }


def presentation_deck(project_id: str, concept_index: int = 0) -> Dict[str, Any]:
    project = PROJECT_STORE.get(project_id, {})
    concept = selected_concept(project_id, concept_index)
    brief = project.get("structured_brief", {})
    return {
        "title": f"Client Pitch - {project.get('project_name', 'Creative Experience')}",
        "template": "premium dark-gold pitch",
        "slides": [
            {"title": "Brief Summary", "body": brief.get("executive_summary", project.get("brief", ""))},
            {"title": "Brand & Category Understanding", "body": json.dumps(brief.get("research_directives", {}), ensure_ascii=False)},
            {"title": f"Concept: {concept.get('name', 'Selected Concept')}", "body": concept.get("big_idea", "")},
            {"title": "Technology Opportunities", "body": "; ".join(concept.get("technology_opportunities", []))},
            {"title": "Mood & Materials", "body": "Moodboard direction with materials, lighting, ambience, seating and stage treatment."},
            {"title": "2D / Brand Graphics", "body": json.dumps(concept.get("graphics_2d_brief", {}), ensure_ascii=False)},
            {"title": "CAD to 3D Logic", "body": concept.get("cad_direction", "")},
            {"title": "Show Flow", "body": "Arrival, build-up, hero moment, interaction and closing handover."},
            {"title": "Production Notes", "body": "; ".join(concept.get("production_watchouts", []))},
        ],
        "presenter_note": "Speak like a senior creative presenter: clear, persuasive and connected to the client objective.",
    }


# =============================================================================
# ACCOUNT HELPERS
# =============================================================================

def account_user_id(request: Request, x_user_id: Optional[str] = None, authorization: Optional[str] = None) -> str:
    if x_user_id:
        return x_user_id
    auth = authorization or request.headers.get("authorization") or ""
    if auth:
        return hashlib.sha256(auth.encode("utf-8")).hexdigest()[:24]
    return request.client.host if request.client else "anonymous"


def ensure_account(user_id: str) -> Dict[str, Any]:
    if user_id not in ACCOUNT_STORE:
        ACCOUNT_STORE[user_id] = {
            "user_id": user_id,
            "credit_balance": DEFAULT_CREDIT_GRANT,
            "plan_id": "free_trial",
            "plan_name": "Free Trial",
            "account_type": "individual",
            "currency": "INR",
            "created_at": now_ts(),
            "updated_at": now_ts(),
        }
        CREDIT_LEDGER[user_id] = [{"id": str(uuid.uuid4()), "type": "grant", "amount": DEFAULT_CREDIT_GRANT, "balance_after": DEFAULT_CREDIT_GRANT, "reason": "Welcome credits", "ts": now_ts()}]
    return ACCOUNT_STORE[user_id]


def grant_credits(user_id: str, amount: int, reason: str, package_id: Optional[str] = None) -> Dict[str, Any]:
    acct = ensure_account(user_id)
    acct["credit_balance"] = int(acct.get("credit_balance", 0)) + max(0, int(amount))
    acct["updated_at"] = now_ts()
    CREDIT_LEDGER.setdefault(user_id, []).append({"id": str(uuid.uuid4()), "type": "grant", "amount": amount, "balance_after": acct["credit_balance"], "reason": reason, "package_id": package_id, "ts": now_ts()})
    return acct


def consume_credits(
    user_id: str,
    amount: int,
    reason: str,
    agent_id: Optional[str] = None,
    project_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    acct = ensure_account(user_id)
    amount = max(0, int(amount))
    request_id = request_id or str(uuid.uuid4())
    sb = get_supabase()
    if sb:
        try:
            rpc_res = sb.rpc("bc_consume_credits", {
                "p_user_key": user_id,
                "p_amount": amount,
                "p_reason": reason,
                "p_agent_id": agent_id,
                "p_project_id": project_id,
                "p_endpoint": endpoint,
                "p_request_id": request_id,
            }).execute()
            data = getattr(rpc_res, "data", None)
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict):
                balance = data.get("balance") or data.get("credit_balance") or data.get("balance_after")
                if balance is not None:
                    acct["credit_balance"] = int(balance)
                    acct["updated_at"] = now_ts()
                CREDIT_LEDGER.setdefault(user_id, []).append({
                    "id": str(uuid.uuid4()),
                    "type": "debit",
                    "amount": amount,
                    "balance_after": acct["credit_balance"],
                    "reason": reason,
                    "agent_id": agent_id,
                    "project_id": project_id,
                    "endpoint": endpoint,
                    "request_id": request_id,
                    "source": "supabase_rpc",
                    "ts": now_ts(),
                })
                return acct
        except Exception as exc:
            print(f"[WARN] Supabase bc_consume_credits failed; falling back to local store: {exc}")

    balance = int(acct.get("credit_balance", 0))
    if amount > balance:
        raise HTTPException(status_code=402, detail={"message": "Insufficient credits.", "required": amount, "balance": balance, "upgrade_endpoint": "/account/packages"})
    acct["credit_balance"] = balance - amount
    acct["updated_at"] = now_ts()
    CREDIT_LEDGER.setdefault(user_id, []).append({"id": str(uuid.uuid4()), "type": "debit", "amount": amount, "balance_after": acct["credit_balance"], "reason": reason, "agent_id": agent_id, "project_id": project_id, "endpoint": endpoint, "request_id": request_id, "source": "local_store", "ts": now_ts()})
    return acct


def account_payload(user_id: str) -> Dict[str, Any]:
    acct = ensure_account(user_id)
    return {
        "account": acct,
        "balance": acct["credit_balance"],
        "unit": "tokens",
        "packages": PACKAGE_PLANS,
        "agents": sorted(AGENT_REGISTRY.values(), key=lambda a: (a.get("category", ""), a.get("name", ""))),
        "hourly_rates_inr": AGENT_HOURLY_RATES_INR,
        "low_balance_threshold": LOW_BALANCE_THRESHOLD,
        "low_balance": int(acct["credit_balance"]) <= LOW_BALANCE_THRESHOLD,
        "ledger": CREDIT_LEDGER.get(user_id, [])[-10:],
        "currency": "INR",
    }


# =============================================================================
# UCD ORCHESTRATOR
# =============================================================================

CAD_INTENT_WORDS = {"cad", "trace", "layout", "floor", "plan", "drawing", "dxf", "dwg", "convert", "blueprint", "venue", "stage", "seating", "autocad", "dimensions", "dimension", "production drawing", "technical drawing"}
IMAGE_FILE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
CAD_FILE_EXTS = IMAGE_FILE_EXTS | {".pdf", ".dxf", ".dwg"}


def ucd_session_id(raw_sid: Optional[str]) -> str:
    sid = raw_sid or f"ucd-{uuid.uuid4()}"
    SESSIONS.setdefault(sid, {"phase": "UCD", "history": [], "pending": {}})
    return sid


def detect_ucd_intent(message: str, file_meta: Optional[UCDFileMeta] = None) -> str:
    text = (message or "").lower()
    ext = Path(file_meta.filename or "").suffix.lower() if file_meta else ""
    if ext in CAD_FILE_EXTS:
        return "cad_trace_file"
    if any(word in text for word in CAD_INTENT_WORDS):
        if any(word in text for word in {"trace", "convert", "upload", "image", "file", "pdf", "dwg", "dxf"}):
            return "cad_trace_file"
        return "cad_generate_layout"
    if any(word in text for word in {"concept", "idea", "creative route", "theme", "story", "launch idea"}):
        return "concept_strategy"
    if any(word in text for word in {"moodboard", "mood board", "materials", "ambience", "palette"}):
        return "moodboard_direction"
    if any(word in text for word in {"invite", "badge", "name batch", "name badge", "backdrop", "facade", "key visual", "2d", "graphic"}):
        return "graphics_2d_direction"
    if any(word in text for word in {"3d", "render", "structure", "sketchup", "spatial"}):
        return "render_3d_direction"
    if any(word in text for word in {"presentation", "pitch", "deck", "ppt", "proposal"}):
        return "presentation_direction"
    if any(word in text for word in {"deliverable", "deliverables", "budget", "brief", "requirement", "requirements"}):
        return "brief_orchestration"
    return "conversation"


def parse_dim_hint(text: str) -> Optional[Dict[str, Any]]:
    try:
        from cad_engine_pro import parse_dim_string

        dims = parse_dim_string(text or "", "m")
        if dims:
            return {"width_mm": dims[0], "depth_mm": dims[1], "unit": dims[2]}
    except Exception:
        pass
    return None


def ucd_missing_for_cad(intent: str, message: str, file_meta: Optional[UCDFileMeta]) -> List[str]:
    text = message or ""
    questions: List[str] = []
    has_dims = parse_dim_hint(text) is not None
    if intent == "cad_trace_file" and not file_meta:
        questions.append("Please upload the layout/image/PDF/DXF file you want me to trace into CAD.")
    if intent in {"cad_trace_file", "cad_generate_layout"} and not has_dims and not file_meta:
        questions.append("What is the approximate venue size, for example 40m x 30m or 120ft x 80ft?")
    if intent == "cad_generate_layout" and not re.search(r"\b(concert|conference|wedding|exhibition|award|launch|expo|summit|layout|generic)\b", text, re.I):
        questions.append("What type of event is this: conference, concert, exhibition, wedding, product launch, or generic?")
    if intent in {"cad_trace_file", "cad_generate_layout"} and not re.search(r"\b\d{2,5}\s*(pax|people|audience|seats|guests)\b", text, re.I):
        questions.append("How many people or seats should I plan for?")
    return questions[:3]


def ucd_cad_ui_contract(intent: str, has_file: bool) -> Dict[str, Any]:
    return {
        "mode": "cad_fullscreen",
        "panel": "cad",
        "show_chat": True,
        "show_progress": True,
        "preview": "live_svg",
        "requires_upload": intent == "cad_trace_file" and not has_file,
        "reason": "CAD/layout requests are single-job workflows and should open the CAD canvas immediately.",
        "progress_steps": [
            {"id": "receive", "label": "Reading brief or uploaded layout", "status": "pending"},
            {"id": "analyze", "label": "Detecting dimensions, zones, seating and production needs", "status": "pending"},
            {"id": "draft", "label": "Creating CAD geometry layer by layer", "status": "pending"},
            {"id": "annotate", "label": "Adding dimensions, title block, scale bar and symbols", "status": "pending"},
            {"id": "export", "label": "Exporting SVG preview, PDF and DXF when available", "status": "pending"},
        ],
    }


def ucd_human_message(intent: str, questions: List[str], has_file: bool) -> str:
    if intent.startswith("cad_") and questions:
        return "Yes, I’ll handle this as a CAD/layout job and open the CAD workspace full screen now. I only need these details to make it accurate: " + " ".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    if intent == "cad_trace_file" and has_file:
        return "I’ll open CAD full screen and send this file to the CAD agent for tracing. You will see the drawing build step by step."
    if intent == "cad_generate_layout":
        return "I’ll create a professional CAD layout from your request and show the CAD workspace full screen while it is generated."
    if intent == "concept_strategy":
        return "I’ll understand the brand, category, venue and brief first, then create multiple product-launch ideas with technology and global reference thinking."
    if intent == "moodboard_direction":
        return "I’ll build the moodboard around the selected concept with mood, materials, lighting, ambience, seating and stage look explained."
    if intent == "graphics_2d_direction":
        return "I’ll brief the 2D art director with exact outputs, copy, sizes, hierarchy and 3D/CAD handoff."
    if intent == "render_3d_direction":
        return "I’ll make CAD and dimensions guide 3D first, then render venue-aware scenes and return production needs to CAD."
    if intent == "presentation_direction":
        return "I’ll package the brief, concept, moodboard, 2D, 3D and production flow into a client-ready pitch presentation."
    return "I’ll understand what you need first, ask only necessary questions, then route the job to the correct specialist agent."


def ucd_response(req: UCDChatRequest) -> UCDChatResponse:
    sid = ucd_session_id(req.session_id)
    intent = detect_ucd_intent(req.message, req.file)
    questions = ucd_missing_for_cad(intent, req.message, req.file) if intent.startswith("cad_") else missing_brief_questions(req.message or "")
    has_file = req.file is not None
    ui: Dict[str, Any] = {}
    agent: Dict[str, Any] = {}
    next_actions: List[Dict[str, Any]] = []

    if intent.startswith("cad_"):
        ui = ucd_cad_ui_contract(intent, has_file)
        agent = {
            "target_agent": "CAD_AGENT",
            "intent": intent,
            "project_id": req.project_id or "demo-project",
            "title": req.title or "CAD Layout",
            "file": dump_model(req.file) if req.file else None,
            "preferred_endpoint": "/api/cad/pro/trace" if intent == "cad_trace_file" else "/api/cad/pro/generate",
            "instruction": "Open CAD fullscreen. Ask only missing CAD details. Generate layout, dimensions, production views, SVG/PDF/DXF.",
        }
        next_actions.append({"type": "open_cad_fullscreen"})
        if questions:
            next_actions.append({"type": "ask_user", "questions": questions})
        else:
            next_actions.append({"type": "send_to_agent", "agent": "CAD_AGENT", "endpoint": agent["preferred_endpoint"]})
    else:
        route = {
            "concept_strategy": ("CONCEPT_AGENT", "/projects/{project_id}/run"),
            "brief_orchestration": ("CONCEPT_AGENT", "/projects/{project_id}/run"),
            "conversation": ("CONCEPT_AGENT", "/projects/{project_id}/run"),
            "moodboard_direction": ("MOODBOARD_AGENT", "/api/moodboard/generate"),
            "graphics_2d_direction": ("GRAPHICS_2D_AGENT", "/ai/generate-2d"),
            "render_3d_direction": ("RENDER_3D_AGENT", "/projects/{project_id}/renders/generate-separated"),
            "presentation_direction": ("PRESENTATION_AGENT", "/projects/{project_id}/presentation/build"),
        }.get(intent, ("CONCEPT_AGENT", "/projects/{project_id}/run"))
        agent = {
            "target_agent": route[0],
            "supervised_by": "UCD_AGENT",
            "preferred_endpoint": route[1],
            "instruction": "Understand brand, category, venue, brief, technology opportunities and production needs before output.",
        }
        if questions and intent in {"concept_strategy", "brief_orchestration", "conversation"}:
            next_actions.append({"type": "ask_user", "questions": questions})
        next_actions.append({"type": "send_to_agent", "agent": route[0], "endpoint": route[1]})

    response = UCDChatResponse(
        session_id=sid,
        intent=intent,
        message=ucd_human_message(intent, questions, has_file),
        questions=questions,
        ui=ui,
        agent=agent,
        next_actions=next_actions,
    )
    SESSIONS[sid]["history"].append({"user": req.message, "intent": intent, "questions": questions, "ts": now_ts()})
    return response


def layout_job_response(message: str, project_id: str = "demo-project") -> Dict[str, Any]:
    questions = ucd_missing_for_cad("cad_generate_layout", message, None)
    return {
        "message": ucd_human_message("cad_generate_layout", questions, False),
        "intent": "cad_generate_layout",
        "ui": ucd_cad_ui_contract("cad_generate_layout", False),
        "agent": {"target_agent": "CAD_AGENT", "preferred_endpoint": "/api/cad/pro/generate", "project_id": project_id},
        "questions": questions,
        "next_actions": [{"type": "open_cad_fullscreen"}, {"type": "ask_user", "questions": questions}] if questions else [{"type": "open_cad_fullscreen"}, {"type": "send_to_agent", "agent": "CAD_AGENT", "endpoint": "/api/cad/pro/generate"}],
    }


def create_job(project_id: str, job_kind: str, agent_id: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    job = {
        "id": str(uuid.uuid4()),
        "project_id": project_id,
        "job_kind": job_kind,
        "agent_id": agent_id,
        "section": job_kind,
        "status": "queued",
        "progress": 0,
        "payload": payload or {},
        "result": None,
        "error": None,
        "created_at": now_ts(),
        "updated_at": now_ts(),
    }
    JOB_STORE[job["id"]] = job
    PROJECT_JOBS.setdefault(project_id, []).append(job)
    persist_job(job)
    return job


def update_job(job_id: str, status: Optional[str] = None, progress: Optional[int] = None, result: Any = None, error: Optional[str] = None) -> Dict[str, Any]:
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if status is not None:
        job["status"] = status
    if progress is not None:
        job["progress"] = max(0, min(100, int(progress)))
    if result is not None:
        job["result"] = result
    if error is not None:
        job["error"] = error
    job["updated_at"] = now_ts()
    persist_job(job)
    return job


def execute_job(job_id: str) -> None:
    job = JOB_STORE.get(job_id)
    if not job:
        return
    try:
        update_job(job_id, "running", 10)
        project_id = job["project_id"]
        kind = job["job_kind"]
        payload = job.get("payload") or {}
        result: Any
        if kind == "research":
            project = load_project(project_id) or {}
            ctx = brief_context(payload.get("brief") or project.get("brief") or "", None)
            ctx["brand"] = payload.get("brand") or project.get("brand") or ctx.get("brand")
            ctx["venue"] = payload.get("venue") or project.get("venue") or ctx.get("venue")
            result = run_research_pack(ctx, payload.get("max_results", 5), project_id)
        elif kind == "moodboard":
            result = {"assets": build_moodboard_assets(project_id, payload.get("concept_index", 0), payload.get("count", 6))}
        elif kind == "2d_graphics":
            result = {"assets": build_2d_assets(project_id, payload.get("concept_index", 0))}
        elif kind == "3d_renders":
            result = {"assets": build_3d_assets(project_id, payload.get("concept_index", 0))}
        elif kind == "departments":
            result = department_outputs(project_id, payload.get("concept_index", 0))
            result["pdf_assets"] = build_pdf_assets(project_id, payload.get("concept_index", 0))
        elif kind == "presentation":
            result = {"presentation": presentation_deck(project_id, payload.get("concept_index", 0))}
        else:
            result = {"message": f"Job kind {kind} accepted but no worker is configured yet."}
        update_job(job_id, "completed", 100, result=result)
    except Exception as exc:
        update_job(job_id, "failed", 100, error=str(exc))


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
def root():
    return {"ok": True, "service": APP_NAME, "version": API_VERSION}


@app.get("/health")
def health():
    return {"ok": True, "service": APP_NAME, "version": API_VERSION, "agents": len(AGENT_REGISTRY)}


@app.get("/agents")
def list_agents():
    return {"ok": True, "agents": sorted(AGENT_REGISTRY.values(), key=lambda a: (a.get("category", ""), a.get("name", ""))), "hourly_rates_inr": AGENT_HOURLY_RATES_INR}


@app.get("/studio/frontend-contract")
@app.post("/studio/frontend-contract")
def studio_frontend_contract():
    return {
        "ok": True,
        "service": APP_NAME,
        "api_version": API_VERSION,
        "features": {
            "ucd_orchestrator": True,
            "multi_concept_generation": True,
            "moodboard_generation": True,
            "graphics_2d_generation": True,
            "renders_3d_generation": True,
            "cad_generation": True,
            "cad_fullscreen_single_job": True,
            "cad_pdf_section": True,
            "presentation_generation": True,
            "account_credits": True,
            "brand_research_directives": True,
            "venue_research_directives": True,
            "technology_research_directives": True,
            "supabase_persistence": bool(get_supabase()),
            "job_queue": True,
            "research_agent": True,
        },
        "endpoints": {
            "agents": "/agents",
            "account_bootstrap": "/account/bootstrap",
            "account_balance": "/account/balance",
            "account_packages": "/account/packages",
            "project_create": "/projects",
            "jobs_create": "/jobs",
            "jobs_get": "/jobs/{job_id}",
            "research_run": "/research/run",
            "project_research": "/projects/{project_id}/research",
            "project_run": "/projects/{project_id}/run",
            "concept_select": "/projects/{project_id}/select-concept",
            "moodboard_generate": "/api/moodboard/generate",
            "moodboard_get": "/api/projects/{project_id}/moodboard",
            "graphics_2d": "/ai/generate-2d",
            "renders_3d": "/projects/{project_id}/renders/generate-separated",
            "departments": "/project/{project_id}/departments/build",
            "pdfs": "/projects/{project_id}/pdfs",
            "presentation": "/projects/{project_id}/presentation/build",
            "cad": "/api/cad/pro/generate",
            "ucd_chat": "/ucd/chat",
        },
        "ucd_rules": [
            "If user asks for a single job, route directly to that specialist instead of forcing full workflow.",
            "For layout/CAD requests, open CAD fullscreen immediately and ask missing CAD details inside that workflow.",
            "For concepts, understand brand, category, venue and technology opportunities before producing options.",
            "2D physical outputs should hand off to 3D for realistic renders and to CAD for fabrication dimensions.",
            "3D approved scenes should hand back to CAD for top/right/left/front production drawings and PDF/DXF export.",
        ],
        "agents": sorted(AGENT_REGISTRY.values(), key=lambda a: (a.get("category", ""), a.get("name", ""))),
        "hourly_rates_inr": AGENT_HOURLY_RATES_INR,
    }


@app.get("/briefcraft_backend_connector.js")
def briefcraft_backend_connector():
    path = Path("briefcraft_backend_connector.js")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frontend connector not found.")
    return FileResponse(path, media_type="application/javascript")


# Account endpoints
@app.get("/account/balance")
def account_balance(request: Request, x_user_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    return {"ok": True, **account_payload(account_user_id(request, x_user_id, authorization))}


@app.get("/account/bootstrap")
def account_bootstrap(request: Request, x_user_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    return {"ok": True, **account_payload(account_user_id(request, x_user_id, authorization))}


@app.get("/account/rates")
def account_rates():
    return {"ok": True, "currency": "INR", "hourly_rates_inr": AGENT_HOURLY_RATES_INR, "low_balance_threshold": LOW_BALANCE_THRESHOLD}


@app.get("/account/packages")
def account_packages():
    return {"ok": True, "currency": "INR", "packages": PACKAGE_PLANS, "payment_links_configured": False}


@app.get("/supabase/schema.sql")
def supabase_schema_sql():
    sql = """
create table if not exists public.bc_projects (
  id text primary key,
  project_id text unique,
  project_name text,
  title text,
  brief text,
  event_type text,
  brand text,
  venue text,
  style_direction text,
  status text,
  data jsonb not null default '{}'::jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists public.bc_assets (
  id text primary key,
  project_id text,
  section text,
  asset_type text,
  title text,
  description text,
  preview_url text,
  status text,
  data jsonb not null default '{}'::jsonb,
  created_at timestamptz default now()
);

create table if not exists public.bc_jobs (
  id text primary key,
  project_id text,
  job_kind text,
  agent_id text,
  section text,
  status text,
  progress int default 0,
  data jsonb not null default '{}'::jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists public.bc_research (
  id text primary key,
  project_id text,
  brand text,
  venue text,
  event_type text,
  query text,
  source text,
  data jsonb not null default '{}'::jsonb,
  created_at timestamptz default now()
);

create table if not exists public.bc_credit_ledger (
  id uuid primary key default gen_random_uuid(),
  user_key text not null,
  amount int not null,
  reason text,
  agent_id text,
  project_id text,
  endpoint text,
  request_id text,
  balance_after int,
  created_at timestamptz default now()
);

create table if not exists public.bc_credit_accounts (
  user_key text primary key,
  credit_balance int not null default 2500,
  updated_at timestamptz default now()
);

create or replace function public.bc_consume_credits(
  p_user_key text,
  p_amount int,
  p_reason text default null,
  p_agent_id text default null,
  p_project_id text default null,
  p_endpoint text default null,
  p_request_id text default null
) returns jsonb
language plpgsql
security definer
as $$
declare
  v_balance int;
begin
  insert into public.bc_credit_accounts(user_key, credit_balance)
  values (p_user_key, 2500)
  on conflict (user_key) do nothing;

  select credit_balance into v_balance
  from public.bc_credit_accounts
  where user_key = p_user_key
  for update;

  if coalesce(p_amount,0) > v_balance then
    raise exception 'Insufficient credits. Required %, balance %', p_amount, v_balance;
  end if;

  update public.bc_credit_accounts
  set credit_balance = credit_balance - coalesce(p_amount,0),
      updated_at = now()
  where user_key = p_user_key
  returning credit_balance into v_balance;

  insert into public.bc_credit_ledger(user_key, amount, reason, agent_id, project_id, endpoint, request_id, balance_after)
  values (p_user_key, coalesce(p_amount,0), p_reason, p_agent_id, p_project_id, p_endpoint, p_request_id, v_balance);

  return jsonb_build_object('credit_balance', v_balance, 'balance_after', v_balance);
end;
$$;
"""
    return {"ok": True, "sql": sql.strip()}


@app.post("/account/credits/consume")
def account_consume_credits(req: CreditConsumeRequest, request: Request, x_user_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    acct = consume_credits(account_user_id(request, x_user_id, authorization), req.amount, req.reason or "usage", req.agent_id, req.project_id, "/account/credits/consume")
    return {"ok": True, "account": acct, "balance": acct["credit_balance"], "unit": "tokens"}


@app.post("/account/checkout")
def account_checkout(req: CheckoutRequest, request: Request, x_user_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    user_id = account_user_id(request, x_user_id, authorization)
    pkg = next((p for p in PACKAGE_PLANS if p["id"] == req.package_id), None)
    if not pkg:
        raise HTTPException(status_code=404, detail="Package not found.")
    acct = grant_credits(user_id, int(pkg.get("credits") or 0), f"Demo activation: {pkg['name']}", package_id=req.package_id)
    acct["plan_id"] = pkg["id"]
    acct["plan_name"] = pkg["name"]
    acct["account_type"] = pkg["audience"]
    return {"ok": True, "checkout_id": str(uuid.uuid4()), "checkout_url": req.success_url or "/", "mode": "demo_grant", "package": pkg, "message": "Demo checkout activated. Connect payment links for production."}


@app.post("/account/agent")
def account_agent(req: AccountAgentRequest, request: Request, x_user_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    user_id = account_user_id(request, x_user_id, authorization)
    data = account_payload(user_id)
    low = data["low_balance"]
    message = f"Your current balance is {data['balance']} tokens. " + ("Your balance is low; please recharge or top up before heavy jobs." if low else "I will keep refreshing this while agents work.")
    return {"ok": True, "agent_id": "ACCOUNT_AGENT", "message": message, "action": "show_balance", **data}


# Project endpoints
@app.post("/projects")
def create_project(req: ProjectCreateRequest):
    brief = (req.brief or req.brief_text or "").strip()
    project_id = str(uuid.uuid4())
    now = now_ts()
    project = {
        "id": project_id,
        "project_id": project_id,
        "project_name": req.project_name or req.title or safe_title(brief, "New Creative Project"),
        "title": req.title or req.project_name or safe_title(brief, "New Creative Project"),
        "brief": brief,
        "event_type": req.event_type or detect_industry(brief),
        "brand": req.brand or extract_brand(brief),
        "venue": req.venue or extract_venue(brief),
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
    persist_project(project)
    return project


@app.get("/projects")
def list_projects():
    return {"ok": True, "projects": sorted(PROJECT_STORE.values(), key=lambda p: p.get("updated_at", 0), reverse=True)}


@app.get("/projects/{project_id}")
def get_project(project_id: str):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    return {"ok": True, "project": project, "assets": PROJECT_ASSETS.get(project_id, []), "jobs": PROJECT_JOBS.get(project_id, [])}


@app.post("/projects/{project_id}/run")
def run_project_pipeline(project_id: str, req: ProjectRunRequest, request: Request, x_user_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    brief = (req.brief or req.text or project.get("brief") or "").strip()
    if not brief:
        raise HTTPException(status_code=400, detail="Brief text is required.")
    ctx = brief_context(brief, req)
    research = run_research_pack(ctx, 5, project_id)
    structured = build_structured_brief(brief, ctx)
    structured["live_research"] = research
    concepts = generate_concepts(brief, ctx, 3)
    project.update({"brief": brief, "event_type": ctx["event_type"], "brand": ctx["brand"], "venue": ctx["venue"], "style_direction": ctx["style_direction"], "structured_brief": structured, "analysis": structured["executive_summary"], "concepts": concepts, "concept_options": concepts, "status": "concepts_ready", "updated_at": now_ts()})
    persist_project(project)
    job = {"id": str(uuid.uuid4()), "project_id": project_id, "job_kind": "concept", "agent_id": "CONCEPT_AGENT", "section": "concepts", "status": "completed", "progress": 100, "created_at": now_ts(), "updated_at": now_ts()}
    PROJECT_JOBS.setdefault(project_id, []).append(job)
    persist_job(job)
    acct = consume_credits(account_user_id(request, x_user_id, authorization), AGENT_REGISTRY["CONCEPT_AGENT"]["credit_cost"], "Concept generation", "CONCEPT_AGENT", project_id, f"/projects/{project_id}/run")
    return {"ok": True, "project_id": project_id, "analysis": structured["executive_summary"], "structured_brief": structured, "research": research, "concepts": concepts, "concept_options": concepts, "missing_questions": ctx["missing_questions"], "ucd_message": f"{ctx['user_name']}, I understood the brand and brief. Review missing questions, confirm deliverables and choose a concept route.", "account": acct, "hourly_rates_inr": AGENT_HOURLY_RATES_INR}


@app.get("/projects/{project_id}/concepts")
def get_project_concepts(project_id: str):
    return {"ok": True, "concepts": PROJECT_STORE.get(project_id, {}).get("concepts", [])}


@app.post("/projects/{project_id}/select-concept")
def select_project_concept(project_id: str, req: ConceptSelectRequest):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    concepts = project.get("concepts") or []
    if not concepts:
        raise HTTPException(status_code=400, detail="No concepts generated yet.")
    idx = max(0, min(len(concepts) - 1, int(req.concept_index)))
    project["selected_concept_index"] = idx
    project["selected_concept"] = concepts[idx]
    project["status"] = "concept_selected"
    project["updated_at"] = now_ts()
    persist_project(project)
    return {"ok": True, "project_id": project_id, "selected_concept_index": idx, "selected_concept": concepts[idx]}


@app.get("/projects/{project_id}/assets")
def list_project_assets(project_id: str, section: Optional[str] = None):
    assets = PROJECT_ASSETS.get(project_id, [])
    if section:
        assets = [a for a in assets if (a.get("section") or "").lower() == section.lower()]
    return {"ok": True, "assets": assets}


@app.get("/projects/{project_id}/jobs")
def list_project_jobs(project_id: str):
    return {"ok": True, "jobs": PROJECT_JOBS.get(project_id, [])}


@app.post("/jobs")
def create_background_job(req: JobCreateRequest, background_tasks: BackgroundTasks):
    if not load_project(req.project_id):
        raise HTTPException(status_code=404, detail="Project not found.")
    job = create_job(req.project_id, req.job_kind, req.agent_id, req.payload)
    if req.run_async:
        background_tasks.add_task(execute_job, job["id"])
    else:
        execute_job(job["id"])
        job = JOB_STORE[job["id"]]
    return {"ok": True, "job": job}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOB_STORE.get(job_id) or sb_table_select_one("bc_jobs", "id", job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"ok": True, "job": job}


@app.post("/jobs/{job_id}/run")
def run_job_now(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Job not found.")
    background_tasks.add_task(execute_job, job_id)
    return {"ok": True, "job": JOB_STORE[job_id]}


@app.get("/projects/{project_id}/pdfs")
def list_project_pdfs(project_id: str):
    assets = [a for a in PROJECT_ASSETS.get(project_id, []) if a.get("section") == "pdf"]
    return {"ok": True, "project_id": project_id, "assets": assets, "pdfs": assets}


@app.get("/projects/{project_id}/research")
def list_project_research(project_id: str):
    return {"ok": True, "project_id": project_id, "research": RESEARCH_STORE.get(project_id, [])}


@app.post("/research/run")
def research_run(req: ResearchRunRequest):
    brief = req.brief or req.query or ""
    ctx = brief_context(brief, None)
    if req.brand:
        ctx["brand"] = req.brand
    if req.venue:
        ctx["venue"] = req.venue
    if req.event_type:
        ctx["event_type"] = req.event_type
        ctx["industry"] = req.event_type
    pack = run_research_pack(ctx, req.max_results, req.project_id)
    return {"ok": True, "research": pack}


# Creative output endpoints
@app.get("/api/projects/{project_id}/moodboard")
def get_api_project_moodboard(project_id: str):
    assets = [a for a in PROJECT_ASSETS.get(project_id, []) if a.get("section") == "moodboard"]
    return {"ok": True, "project_id": project_id, "assets": assets, "moodboard": assets, "images": assets}


@app.post("/api/moodboard/generate")
def generate_moodboard(req: MoodboardGenerateRequest, request: Request, x_user_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    if not load_project(req.project_id):
        raise HTTPException(status_code=404, detail="Project not found.")
    acct = consume_credits(account_user_id(request, x_user_id, authorization), AGENT_REGISTRY["MOODBOARD_AGENT"]["credit_cost"], "Moodboard generation", "MOODBOARD_AGENT", req.project_id, "/api/moodboard/generate")
    assets = build_moodboard_assets(req.project_id, req.concept_index or 0, req.count or 6)
    job = {"id": str(uuid.uuid4()), "project_id": req.project_id, "job_kind": "moodboard", "agent_id": "MOODBOARD_AGENT", "section": "moodboard", "status": "completed", "progress": 100, "created_at": now_ts(), "updated_at": now_ts()}
    PROJECT_JOBS.setdefault(req.project_id, []).append(job)
    persist_job(job)
    return {"ok": True, "project_id": req.project_id, "assets": assets, "account": acct, "message": "Moodboard generated with mood, materials, lighting, ambience, seating and stage logic."}


@app.post("/ai/generate-2d")
def generate_2d(req: AssetGenerateRequest, request: Request, x_user_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    if not load_project(req.project_id):
        raise HTTPException(status_code=404, detail="Project not found.")
    acct = consume_credits(account_user_id(request, x_user_id, authorization), AGENT_REGISTRY["GRAPHICS_2D_AGENT"]["credit_cost"], "2D graphics generation", "GRAPHICS_2D_AGENT", req.project_id, "/ai/generate-2d")
    assets = build_2d_assets(req.project_id, req.concept_index or 0)
    job = {"id": str(uuid.uuid4()), "project_id": req.project_id, "job_kind": "2d_graphics", "agent_id": "GRAPHICS_2D_AGENT", "section": "2d_graphics", "status": "completed", "progress": 100, "created_at": now_ts(), "updated_at": now_ts()}
    PROJECT_JOBS.setdefault(req.project_id, []).append(job)
    persist_job(job)
    return {"ok": True, "project_id": req.project_id, "assets": assets, "account": acct}


@app.post("/projects/{project_id}/renders/generate-separated")
def generate_3d(project_id: str, request: Request, x_user_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    if not load_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found.")
    acct = consume_credits(account_user_id(request, x_user_id, authorization), AGENT_REGISTRY["RENDER_3D_AGENT"]["credit_cost"], "3D render generation", "RENDER_3D_AGENT", project_id, f"/projects/{project_id}/renders/generate-separated")
    idx = PROJECT_STORE[project_id].get("selected_concept_index") or 0
    assets = build_3d_assets(project_id, idx)
    job = {"id": str(uuid.uuid4()), "project_id": project_id, "job_kind": "3d_renders", "agent_id": "RENDER_3D_AGENT", "section": "renders", "status": "completed", "progress": 100, "created_at": now_ts(), "updated_at": now_ts()}
    PROJECT_JOBS.setdefault(project_id, []).append(job)
    persist_job(job)
    return {"ok": True, "project_id": project_id, "assets": assets, "account": acct}


@app.post("/project/{project_id}/departments/build")
def build_departments(project_id: str):
    if not load_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found.")
    idx = PROJECT_STORE[project_id].get("selected_concept_index") or 0
    outputs = department_outputs(project_id, idx)
    pdf_assets = build_pdf_assets(project_id, idx)
    PROJECT_STORE[project_id].update(outputs)
    PROJECT_STORE[project_id]["department_outputs"] = outputs
    PROJECT_STORE[project_id]["pdf_assets"] = pdf_assets
    PROJECT_STORE[project_id]["updated_at"] = now_ts()
    persist_project(PROJECT_STORE[project_id])
    return {"ok": True, "project_id": project_id, "pdf_assets": pdf_assets, **outputs}


@app.post("/projects/{project_id}/presentation/build")
def build_presentation(project_id: str, req: PresentationBuildRequest, request: Request, x_user_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    if not load_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found.")
    acct = consume_credits(account_user_id(request, x_user_id, authorization), AGENT_REGISTRY["PRESENTATION_AGENT"]["credit_cost"], "Presentation build", "PRESENTATION_AGENT", project_id, f"/projects/{project_id}/presentation/build")
    deck = presentation_deck(project_id, req.concept_index or PROJECT_STORE[project_id].get("selected_concept_index") or 0)
    PROJECT_STORE[project_id]["presentation"] = deck
    PROJECT_STORE[project_id]["updated_at"] = now_ts()
    persist_project(PROJECT_STORE[project_id])
    return {"ok": True, "project_id": project_id, "presentation": deck, "account": acct}


# Agent and UCD endpoints
@app.post("/agents/run")
def run_agent(req: AgentRunRequest, request: Request, x_user_id: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    user_id = account_user_id(request, x_user_id, authorization)
    agent_id = (req.agent_id or "").strip().upper()
    agent = AGENT_REGISTRY.get(agent_id)
    if not agent or not agent.get("enabled", True):
        raise HTTPException(status_code=404, detail="Agent not found or disabled.")
    cost = int(agent.get("credit_cost") or 0)
    if cost:
        consume_credits(user_id, cost, f"Agent run: {agent['name']}", agent_id, req.project_id, "/agents/run")
    if agent_id == "ACCOUNT_AGENT":
        data = account_payload(user_id)
        payload = {"message": f"Your balance is {data['balance']} tokens.", **data}
    elif agent_id == "UCD_AGENT":
        payload = dump_model(ucd_response(UCDChatRequest(session_id=req.session_id, message=req.message or "", project_id=req.project_id or "demo-project", context=req.context or {})))
    elif agent_id == "CAD_AGENT":
        payload = layout_job_response(req.message or "Create a CAD layout", req.project_id or "demo-project")
    elif agent_id == "CONCEPT_AGENT":
        ctx = brief_context(req.message or "Creative brief", None)
        payload = {"message": "Concept Agent will research brand/category/technology, then create multiple distinct routes.", "concepts": generate_concepts(req.message or "Creative brief", ctx, 3), "research_directives": ctx["research"], "endpoint": "/projects/{project_id}/run"}
    else:
        payload = {"message": f"{agent['name']} accepted the request.", "endpoint": agent["endpoint"], "context": req.context}
    acct = ensure_account(user_id)
    return {"ok": True, "agent": agent, "result": payload, "account": {"credit_balance": acct["credit_balance"], "plan_id": acct["plan_id"], "plan_name": acct["plan_name"], "unit": "tokens"}}


@app.post("/ucd/chat", response_model=UCDChatResponse)
def ucd_chat(req: UCDChatRequest):
    if not (req.message or "").strip():
        raise HTTPException(status_code=400, detail="Message is required.")
    return ucd_response(req)


@app.post("/ucd/cad/upload-intent", response_model=UCDChatResponse)
async def ucd_cad_upload_intent(file: UploadFile = File(...), message: str = Form("Trace this file in CAD"), session_id: Optional[str] = Form(None), project_id: str = Form("demo-project"), title: Optional[str] = Form(None)):
    data = await file.read()
    meta = UCDFileMeta(filename=file.filename, content_type=file.content_type, size_bytes=len(data))
    req = UCDChatRequest(session_id=session_id, message=message, project_id=project_id, title=title, file=meta)
    response = ucd_response(req)
    response.agent["upload_ready"] = True
    response.agent["upload_filename"] = file.filename
    response.agent["trace_endpoint"] = "/api/cad/pro/trace"
    return response
