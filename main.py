from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import UUID

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import OpenAI
from pydantic import BaseModel, Field
from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions


# =========================================================
# LOCAL .ENV LOADER
# =========================================================
def load_local_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()


# =========================================================
# CONFIG
# =========================================================
APP_NAME = "Creative Brief to Concept & Execution API"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*").strip() or "*"
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", "./exports")).resolve()
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads")).resolve()

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
if not SUPABASE_URL:
    raise RuntimeError("Missing SUPABASE_URL")
if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_SERVICE_ROLE_KEY")

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    options=ClientOptions(auto_refresh_token=False, persist_session=False, schema="public"),
)


# =========================================================
# APP
# =========================================================
app = FastAPI(title=APP_NAME, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)


# =========================================================
# PYDANTIC MODELS
# =========================================================
class ProjectCreate(BaseModel):
    project_name: str
    brand_name: Optional[str] = None
    industry: Optional[str] = None
    campaign_type: Optional[str] = None
    objective: Optional[str] = None


class ProjectUpdate(BaseModel):
    project_name: Optional[str] = None
    brand_name: Optional[str] = None
    industry: Optional[str] = None
    campaign_type: Optional[str] = None
    objective: Optional[str] = None
    status: Optional[str] = None


class BriefCreate(BaseModel):
    source_type: Literal["text", "pdf", "docx"] = "text"
    raw_text: Optional[str] = None


class BriefParseRequest(BaseModel):
    force_regenerate: bool = False


class BriefApproveRequest(BaseModel):
    cleaned_summary: Optional[str] = None
    target_audience: Optional[str] = None
    geography: Optional[str] = None
    languages: List[str] = Field(default_factory=list)
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    currency: str = "INR"
    success_metrics: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    approved_assumption_ids: List[str] = Field(default_factory=list)


class StrategyGenerateRequest(BaseModel):
    force_regenerate: bool = False


class ConceptsGenerateRequest(BaseModel):
    count: int = 3
    force_regenerate: bool = False


class SelectConceptRequest(BaseModel):
    selected: bool = True


class DeliverablePatch(BaseModel):
    id: str
    quantity: Optional[int] = None
    language_count: Optional[int] = None
    complexity: Optional[Literal["low", "medium", "high"]] = None
    turnaround: Optional[Literal["standard", "fast", "rush"]] = None
    notes: Optional[str] = None


class UpdateDeliverablesRequest(BaseModel):
    deliverables: List[DeliverablePatch]


class ExecutionGenerateRequest(BaseModel):
    route_id: Optional[str] = None
    force_regenerate: bool = False


class CostGenerateRequest(BaseModel):
    market: str = "India"
    vendor_tier: Literal["lean", "standard", "premium"] = "standard"
    contingency_percent: float = 10.0
    tax_percent: float = 18.0


class HandoffGenerateRequest(BaseModel):
    include_rfp: bool = True
    include_email_draft: bool = True
    force_regenerate: bool = False


# =========================================================
# JSON SCHEMAS FOR STRUCTURED OUTPUTS
# =========================================================
BRIEF_PARSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "cleaned_summary": {"type": "string"},
        "target_audience": {"type": "string"},
        "geography": {"type": "string"},
        "languages": {"type": "array", "items": {"type": "string"}},
        "budget_min": {"type": ["number", "null"]},
        "budget_max": {"type": ["number", "null"]},
        "currency": {"type": "string"},
        "timeline_notes": {"type": "string"},
        "success_metrics": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "assumptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "assumption_type": {"type": "string"},
                    "assumption_text": {"type": "string"},
                    "confidence_score": {"type": "number"},
                },
                "required": ["assumption_type", "assumption_text", "confidence_score"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "cleaned_summary",
        "target_audience",
        "geography",
        "languages",
        "budget_min",
        "budget_max",
        "currency",
        "timeline_notes",
        "success_metrics",
        "constraints",
        "missing_information",
        "assumptions",
    ],
    "additionalProperties": False,
}

STRATEGY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "business_challenge": {"type": "string"},
        "communication_objective": {"type": "string"},
        "audience_insight": {"type": "string"},
        "proposition": {"type": "string"},
        "tone_of_voice": {"type": "string"},
        "recommended_channels": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "business_challenge",
        "communication_objective",
        "audience_insight",
        "proposition",
        "tone_of_voice",
        "recommended_channels",
    ],
    "additionalProperties": False,
}

CONCEPTS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "routes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "route_name": {"type": "string"},
                    "route_summary": {"type": "string"},
                    "key_message": {"type": "string"},
                    "tagline_options": {"type": "array", "items": {"type": "string"}},
                    "visual_direction": {"type": "string"},
                    "why_it_works": {"type": "string"},
                    "complexity_level": {"type": "string", "enum": ["low", "medium", "high"]},
                    "cost_tendency": {"type": "string", "enum": ["lean", "recommended", "premium"]},
                },
                "required": [
                    "route_name",
                    "route_summary",
                    "key_message",
                    "tagline_options",
                    "visual_direction",
                    "why_it_works",
                    "complexity_level",
                    "cost_tendency",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["routes"],
    "additionalProperties": False,
}

EXECUTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "plan_summary": {"type": "string"},
        "campaign_phases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "phase_name": {"type": "string"},
                    "duration_days": {"type": "integer"},
                    "goal": {"type": "string"},
                },
                "required": ["phase_name", "duration_days", "goal"],
                "additionalProperties": False,
            },
        },
        "timeline_json": {"type": "object", "additionalProperties": True},
        "team_roles": {"type": "array", "items": {"type": "string"}},
        "dependencies": {"type": "array", "items": {"type": "string"}},
        "deliverables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "deliverable_type": {"type": "string"},
                    "deliverable_name": {"type": "string"},
                    "channel": {"type": "string"},
                    "format": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "language_count": {"type": "integer"},
                    "complexity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "turnaround": {"type": "string", "enum": ["standard", "fast", "rush"]},
                    "notes": {"type": "string"},
                },
                "required": [
                    "deliverable_type",
                    "deliverable_name",
                    "channel",
                    "format",
                    "quantity",
                    "language_count",
                    "complexity",
                    "turnaround",
                    "notes",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "plan_summary",
        "campaign_phases",
        "timeline_json",
        "team_roles",
        "dependencies",
        "deliverables",
    ],
    "additionalProperties": False,
}

HANDOFF_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "agency_brief_markdown": {"type": "string"},
        "sow_summary_markdown": {"type": "string"},
        "rfp_summary_markdown": {"type": "string"},
        "email_draft": {"type": "string"},
    },
    "required": [
        "agency_brief_markdown",
        "sow_summary_markdown",
        "rfp_summary_markdown",
        "email_draft",
    ],
    "additionalProperties": False,
}


# =========================================================
# HELPERS
# =========================================================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value).strip()
    return value


def unwrap_response_data(resp: Any) -> Any:
    if resp is None:
        return None
    if hasattr(resp, "data"):
        return resp.data
    if isinstance(resp, dict) and "data" in resp:
        return resp["data"]
    return resp


def first_row(data: Any) -> Optional[Dict[str, Any]]:
    payload = unwrap_response_data(data)
    if isinstance(payload, list):
        return payload[0] if payload else None
    if isinstance(payload, dict):
        return payload
    return None


def require_env() -> None:
    missing = []
    for key in ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]:
        if not os.getenv(key):
            missing.append(key)
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing env vars: {', '.join(missing)}")


def parse_json_content(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    return json.loads(raw)


def extract_message_content(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        chunks.append(text_value)
                    elif isinstance(text_value, dict) and isinstance(text_value.get("value"), str):
                        chunks.append(text_value["value"])
                elif isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                chunks.append(item.text)
        return "\n".join([c for c in chunks if c]).strip()
    return getattr(message, "content", "") or ""


def get_json_completion(
    *,
    system_prompt: str,
    user_payload: Dict[str, Any],
    schema_name: str,
    schema: Dict[str, Any],
    temperature: float = 0.4,
) -> Dict[str, Any]:
    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        },
    )
    message = completion.choices[0].message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise HTTPException(status_code=400, detail=f"Model refused request: {refusal}")
    content = extract_message_content(message)
    if not content:
        raise HTTPException(status_code=500, detail="Model returned empty content")
    return parse_json_content(content)


def db_insert(table: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    res = supabase.table(table).insert(payload).execute()
    row = first_row(res)
    if not row:
        raise HTTPException(status_code=500, detail=f"Insert failed for table {table}")
    return row


def db_update(table: str, filters: Dict[str, Any], payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    q = supabase.table(table).update(payload)
    for key, value in filters.items():
        q = q.eq(key, value)
    res = q.execute()
    data = unwrap_response_data(res) or []
    return data if isinstance(data, list) else [data]


def db_delete(table: str, filters: Dict[str, Any]) -> None:
    q = supabase.table(table).delete()
    for key, value in filters.items():
        q = q.eq(key, value)
    q.execute()


def db_select_one(table: str, filters: Dict[str, Any], columns: str = "*") -> Optional[Dict[str, Any]]:
    q = supabase.table(table).select(columns)
    for key, value in filters.items():
        q = q.eq(key, value)
    res = q.limit(1).execute()
    return first_row(res)


def db_select_many(
    table: str,
    filters: Optional[Dict[str, Any]] = None,
    columns: str = "*",
    order_by: Optional[str] = None,
) -> List[Dict[str, Any]]:
    q = supabase.table(table).select(columns)
    for key, value in (filters or {}).items():
        q = q.eq(key, value)
    if order_by:
        q = q.order(order_by)
    res = q.execute()
    data = unwrap_response_data(res) or []
    return data if isinstance(data, list) else [data]


def upsert_singleton_by_project(table: str, project_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    existing = db_select_one(table, {"project_id": project_id})
    if existing:
        updated = db_update(table, {"id": existing["id"]}, payload)
        return updated[0] if updated else {**existing, **payload}
    return db_insert(table, {"project_id": project_id, **payload})


def ensure_uuid(value: str, label: str) -> str:
    try:
        return str(UUID(value))
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid {label}")


def verify_token_and_get_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Dict[str, Any]:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = credentials.credentials

    user_id: Optional[str] = None
    email: Optional[str] = None

    try:
        claims_resp = supabase.auth.get_claims(token)
        claims = getattr(claims_resp, "claims", None)
        if claims is None and isinstance(claims_resp, dict):
            claims = claims_resp.get("claims") or claims_resp.get("data")
        if claims and isinstance(claims, dict):
            user_id = claims.get("sub")
            email = claims.get("email")
    except Exception:
        pass

    if not user_id:
        try:
            user_resp = supabase.auth.get_user(token)
            user = getattr(user_resp, "user", None)
            if user is None and isinstance(user_resp, dict):
                user = user_resp.get("user") or user_resp.get("data")
            if user is not None:
                if isinstance(user, dict):
                    user_id = user.get("id")
                    email = user.get("email")
                else:
                    user_id = getattr(user, "id", None)
                    email = getattr(user, "email", None)
        except Exception as exc:
            raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc

    if not user_id:
        raise HTTPException(status_code=401, detail="Could not resolve authenticated user")

    return {"id": user_id, "email": email, "access_token": token}


def ensure_project_owner(project_id: str, user_id: str) -> Dict[str, Any]:
    project_id = ensure_uuid(project_id, "project_id")
    project = db_select_one("projects", {"id": project_id, "user_id": user_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def get_project_bundle(project_id: str, user_id: str) -> Dict[str, Any]:
    project = ensure_project_owner(project_id, user_id)
    brief = db_select_one("creative_briefs", {"project_id": project_id})
    strategy = db_select_one("strategy_outputs", {"project_id": project_id})
    routes = db_select_many("concept_routes", {"project_id": project_id}, order_by="created_at")
    execution = db_select_one("execution_plans", {"project_id": project_id})
    selected_route = next((r for r in routes if r.get("is_selected")), None)
    deliverables = []
    if execution:
        deliverables = db_select_many("deliverables", {"execution_plan_id": execution["id"]}, order_by="created_at")
    return {
        "project": project,
        "brief": brief,
        "strategy": strategy,
        "routes": routes,
        "selected_route": selected_route,
        "execution": execution,
        "deliverables": deliverables,
    }


def log_generation(project_id: str, stage: str, input_snapshot: Dict[str, Any], output_snapshot: Dict[str, Any], status: str = "success", error_message: Optional[str] = None) -> None:
    try:
        db_insert(
            "generation_logs",
            {
                "project_id": project_id,
                "stage": stage,
                "input_snapshot": input_snapshot,
                "output_snapshot": output_snapshot,
                "status": status,
                "error_message": error_message,
            },
        )
    except Exception:
        # logging should never break the main flow
        pass


def try_extract_text_from_upload(filename: str, raw_bytes: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Install pypdf to parse PDF briefs") from exc
        reader = PdfReader(BytesIO(raw_bytes))
        return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()

    if suffix == ".docx":
        try:
            from docx import Document  # type: ignore
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Install python-docx to parse DOCX briefs") from exc
        doc = Document(BytesIO(raw_bytes))
        return "\n".join(p.text for p in doc.paragraphs).strip()

    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="ignore")


def save_text_export(project_id: str, export_type: str, content: str) -> str:
    project_dir = EXPORT_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{export_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    path = project_dir / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


def fetch_catalog_types() -> List[str]:
    rows = db_select_many("deliverable_catalog", order_by="deliverable_type")
    return [r["deliverable_type"] for r in rows if r.get("deliverable_type")]


def compute_line_total(
    base_cost: float,
    quantity: int,
    complexity: str,
    turnaround: str,
    language_count: int,
    complexity_map: Dict[str, Any],
    urgency_map: Dict[str, Any],
    extra_language_multiplier: float,
) -> Tuple[float, Dict[str, float]]:
    complexity_factor = float(complexity_map.get(complexity, 1.0))
    urgency_factor = float(urgency_map.get(turnaround, 1.0))
    language_factor = 1.0 + max(language_count - 1, 0) * float(extra_language_multiplier)
    line_total = base_cost * quantity * complexity_factor * urgency_factor * language_factor
    return round(line_total, 2), {
        "complexity": complexity_factor,
        "urgency": urgency_factor,
        "language": language_factor,
    }


def choose_cost_template(deliverable_type: str, market: str, vendor_tier: str) -> Optional[Dict[str, Any]]:
    exact = db_select_one(
        "cost_templates",
        {"deliverable_type": deliverable_type, "market": market, "vendor_tier": vendor_tier},
    )
    if exact:
        return exact
    fallback_market = db_select_one(
        "cost_templates",
        {"deliverable_type": deliverable_type, "vendor_tier": vendor_tier},
    )
    if fallback_market:
        return fallback_market
    return db_select_one("cost_templates", {"deliverable_type": deliverable_type})


# =========================================================
# PROMPT BUILDERS
# =========================================================
BRIEF_PARSER_PROMPT = """
You are a senior brand strategist and campaign planner.
Convert a raw creative brief into a structured planning brief.

Rules:
- Extract only clearly stated facts as facts.
- Put anything inferred into assumptions.
- Keep the language concise and professional.
- If the brief is incomplete, do not block. Produce a practical working brief.
- Normalize budget into numeric fields when present.
- Keep constraints tactical and reusable.
- Return valid JSON only.
""".strip()

STRATEGY_PROMPT = """
You are a senior strategic planner at a top creative agency.
Create the strategy foundation for campaign development.

Rules:
- Focus on communication strategy.
- Keep the proposition tight and memorable.
- Recommended channels must match objective, audience, and likely budget.
- Avoid generic marketing filler.
- Return valid JSON only.
""".strip()

CONCEPTS_PROMPT = """
You are an award-winning creative director and brand strategist.
Generate distinct campaign routes from the strategy.

Rules:
- Each route must be meaningfully different.
- Each route must be executable across real deliverables.
- Avoid generic slogans.
- Include clear business logic for why the route works.
- Tagline options must be short and usable.
- Return valid JSON only.
""".strip()

EXECUTION_PROMPT = """
You are an integrated campaign producer and creative operations planner.
Convert the selected concept into an execution-ready campaign scope.

Rules:
- Think in actual deliverables, timelines, dependencies, and team roles.
- Only include deliverables justified by the brief and strategy.
- Avoid bloated scope.
- Deliverable types must come from the provided allowed list when possible.
- Return valid JSON only.
""".strip()

HANDOFF_PROMPT = """
You are a senior account planner preparing an agency handoff pack.
Create a professional execution brief for external agencies.

Rules:
- Be structured, concise, and practical.
- Agencies must clearly understand objective, audience, concept, scope, timeline, and expectations.
- Keep the email draft ready for first outreach.
- Return valid JSON only.
""".strip()


# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "app": APP_NAME, "time": now_iso(), "model": OPENAI_MODEL}


@app.get("/health")
def health() -> Dict[str, Any]:
    require_env()
    return {"ok": True}


@app.post("/api/projects")
def create_project(payload: ProjectCreate, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    row = db_insert(
        "projects",
        {
            "user_id": user["id"],
            "project_name": payload.project_name,
            "brand_name": payload.brand_name,
            "industry": payload.industry,
            "campaign_type": payload.campaign_type,
            "objective": payload.objective,
            "status": "draft",
        },
    )
    return {"ok": True, "project": row}


@app.get("/api/projects")
def list_projects(user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    rows = db_select_many("projects", {"user_id": user["id"]}, order_by="created_at")
    rows.sort(key=lambda x: x.get("updated_at") or x.get("created_at") or "", reverse=True)
    return {"ok": True, "projects": rows}


@app.get("/api/projects/{project_id}")
def get_project(project_id: str, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    bundle = get_project_bundle(project_id, user["id"])
    return {"ok": True, **bundle}


@app.patch("/api/projects/{project_id}")
def update_project(project_id: str, payload: ProjectUpdate, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    project = ensure_project_owner(project_id, user["id"])
    update_payload = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not update_payload:
        return {"ok": True, "project": project}
    updated = db_update("projects", {"id": project_id}, update_payload)
    return {"ok": True, "project": updated[0] if updated else {**project, **update_payload}}


@app.post("/api/projects/{project_id}/brief")
def save_brief(project_id: str, payload: BriefCreate, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    ensure_project_owner(project_id, user["id"])
    existing = db_select_one("creative_briefs", {"project_id": project_id})
    brief_payload = {
        "source_type": payload.source_type,
        "raw_text": payload.raw_text,
        "approved": False,
    }
    if existing:
        updated = db_update("creative_briefs", {"id": existing["id"]}, brief_payload)
        row = updated[0] if updated else {**existing, **brief_payload}
    else:
        row = db_insert("creative_briefs", {"project_id": project_id, **brief_payload})
    db_update("projects", {"id": project_id}, {"status": "brief_saved"})
    return {"ok": True, "brief_id": row["id"]}


@app.post("/api/projects/{project_id}/brief/upload")
async def upload_brief_file(project_id: str, file: UploadFile = File(...), user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    ensure_project_owner(project_id, user["id"])
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    project_upload_dir = UPLOAD_DIR / project_id
    project_upload_dir.mkdir(parents=True, exist_ok=True)
    filename = file.filename or f"brief-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    saved_path = project_upload_dir / filename
    saved_path.write_bytes(raw)

    extracted_text = try_extract_text_from_upload(filename, raw)

    existing = db_select_one("creative_briefs", {"project_id": project_id})
    payload = {
        "source_type": Path(filename).suffix.lower().lstrip(".") or "text",
        "file_url": str(saved_path),
        "raw_text": extracted_text,
        "approved": False,
    }
    if existing:
        db_update("creative_briefs", {"id": existing["id"]}, payload)
    else:
        db_insert("creative_briefs", {"project_id": project_id, **payload})

    db_update("projects", {"id": project_id}, {"status": "brief_uploaded"})
    return {"ok": True, "file_url": str(saved_path), "extracted_chars": len(extracted_text)}


@app.post("/api/projects/{project_id}/brief/parse")
def parse_brief(project_id: str, payload: BriefParseRequest, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    ensure_project_owner(project_id, user["id"])
    brief = db_select_one("creative_briefs", {"project_id": project_id})
    if not brief or not clean_text(brief.get("raw_text")):
        raise HTTPException(status_code=400, detail="No raw brief text found")

    existing_assumptions = db_select_many("brief_assumptions", {"project_id": project_id}, order_by="created_at")
    if brief.get("cleaned_summary") and existing_assumptions and not payload.force_regenerate:
        return {
            "ok": True,
            "brief": {
                "cleaned_summary": brief.get("cleaned_summary"),
                "target_audience": brief.get("target_audience"),
                "geography": brief.get("geography"),
                "languages": brief.get("languages") or [],
                "budget_min": brief.get("budget_min"),
                "budget_max": brief.get("budget_max"),
                "currency": brief.get("currency") or "INR",
                "constraints": brief.get("constraints") or [],
            },
            "assumptions": existing_assumptions,
        }

    parsed = get_json_completion(
        system_prompt=BRIEF_PARSER_PROMPT,
        user_payload={
            "project_id": project_id,
            "raw_brief": brief.get("raw_text"),
            "existing_project_metadata": {
                "brand_name": ensure_project_owner(project_id, user["id"]).get("brand_name"),
                "industry": ensure_project_owner(project_id, user["id"]).get("industry"),
                "campaign_type": ensure_project_owner(project_id, user["id"]).get("campaign_type"),
                "objective": ensure_project_owner(project_id, user["id"]).get("objective"),
            },
        },
        schema_name="brief_parse_output",
        schema=BRIEF_PARSE_SCHEMA,
    )

    db_update(
        "creative_briefs",
        {"project_id": project_id},
        {
            "cleaned_summary": parsed["cleaned_summary"],
            "target_audience": parsed["target_audience"],
            "geography": parsed["geography"],
            "languages": parsed["languages"],
            "budget_min": parsed["budget_min"],
            "budget_max": parsed["budget_max"],
            "currency": parsed.get("currency") or "INR",
            "success_metrics": parsed.get("success_metrics", []),
            "constraints": parsed.get("constraints", []),
            "approved": False,
        },
    )

    db_delete("brief_assumptions", {"project_id": project_id})
    assumptions_saved = []
    for item in parsed.get("assumptions", []):
        assumptions_saved.append(
            db_insert(
                "brief_assumptions",
                {
                    "project_id": project_id,
                    "assumption_type": item["assumption_type"],
                    "assumption_text": item["assumption_text"],
                    "confidence_score": item["confidence_score"],
                    "user_approved": False,
                },
            )
        )

    db_update("projects", {"id": project_id}, {"status": "brief_parsed"})
    log_generation(project_id, "brief_parser", {"raw_brief": brief.get("raw_text")}, parsed)

    return {
        "ok": True,
        "brief": {
            "cleaned_summary": parsed["cleaned_summary"],
            "target_audience": parsed["target_audience"],
            "geography": parsed["geography"],
            "languages": parsed["languages"],
            "budget_min": parsed["budget_min"],
            "budget_max": parsed["budget_max"],
            "currency": parsed.get("currency") or "INR",
            "constraints": parsed.get("constraints", []),
        },
        "assumptions": assumptions_saved,
        "missing_information": parsed.get("missing_information", []),
        "timeline_notes": parsed.get("timeline_notes", ""),
    }


@app.patch("/api/projects/{project_id}/brief/approve")
def approve_brief(project_id: str, payload: BriefApproveRequest, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    ensure_project_owner(project_id, user["id"])
    brief = db_select_one("creative_briefs", {"project_id": project_id})
    if not brief:
        raise HTTPException(status_code=404, detail="Brief not found")

    db_update(
        "creative_briefs",
        {"project_id": project_id},
        {
            "cleaned_summary": payload.cleaned_summary if payload.cleaned_summary is not None else brief.get("cleaned_summary"),
            "target_audience": payload.target_audience if payload.target_audience is not None else brief.get("target_audience"),
            "geography": payload.geography if payload.geography is not None else brief.get("geography"),
            "languages": payload.languages or brief.get("languages") or [],
            "budget_min": payload.budget_min if payload.budget_min is not None else brief.get("budget_min"),
            "budget_max": payload.budget_max if payload.budget_max is not None else brief.get("budget_max"),
            "currency": payload.currency or brief.get("currency") or "INR",
            "success_metrics": payload.success_metrics or brief.get("success_metrics") or [],
            "constraints": payload.constraints or brief.get("constraints") or [],
            "approved": True,
        },
    )

    approved_ids = {ensure_uuid(x, "assumption_id") for x in payload.approved_assumption_ids}
    assumptions = db_select_many("brief_assumptions", {"project_id": project_id}, order_by="created_at")
    for assumption in assumptions:
        db_update(
            "brief_assumptions",
            {"id": assumption["id"]},
            {"user_approved": assumption["id"] in approved_ids},
        )

    db_update("projects", {"id": project_id}, {"status": "brief_approved"})
    return {"ok": True, "approved": True}


@app.post("/api/projects/{project_id}/strategy/generate")
def generate_strategy(project_id: str, payload: StrategyGenerateRequest, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    bundle = get_project_bundle(project_id, user["id"])
    brief = bundle["brief"]
    if not brief or not brief.get("approved"):
        raise HTTPException(status_code=400, detail="Approve the brief before generating strategy")

    if bundle["strategy"] and not payload.force_regenerate:
        return {"ok": True, "strategy": bundle["strategy"]}

    parsed = get_json_completion(
        system_prompt=STRATEGY_PROMPT,
        user_payload={
            "project": bundle["project"],
            "brief": brief,
            "approved_assumptions": db_select_many("brief_assumptions", {"project_id": project_id, "user_approved": True}),
        },
        schema_name="strategy_output",
        schema=STRATEGY_SCHEMA,
    )

    row = upsert_singleton_by_project("strategy_outputs", project_id, parsed)
    db_update("projects", {"id": project_id}, {"status": "strategy_ready"})
    log_generation(project_id, "strategy", {"brief": brief}, parsed)
    return {"ok": True, "strategy": row}


@app.get("/api/projects/{project_id}/strategy")
def get_strategy(project_id: str, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    ensure_project_owner(project_id, user["id"])
    strategy = db_select_one("strategy_outputs", {"project_id": project_id})
    return {"ok": True, "strategy": strategy}


@app.post("/api/projects/{project_id}/concepts/generate")
def generate_concepts(project_id: str, payload: ConceptsGenerateRequest, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    bundle = get_project_bundle(project_id, user["id"])
    if not bundle["brief"] or not bundle["strategy"]:
        raise HTTPException(status_code=400, detail="Brief and strategy are required before concept generation")

    existing_routes = bundle["routes"]
    if existing_routes and not payload.force_regenerate:
        return {"ok": True, "routes": existing_routes}

    if payload.force_regenerate:
        db_delete("concept_routes", {"project_id": project_id})

    parsed = get_json_completion(
        system_prompt=CONCEPTS_PROMPT,
        user_payload={
            "project": bundle["project"],
            "brief": bundle["brief"],
            "strategy": bundle["strategy"],
            "count": max(1, min(payload.count, 5)),
        },
        schema_name="concept_routes_output",
        schema=CONCEPTS_SCHEMA,
        temperature=0.7,
    )

    rows: List[Dict[str, Any]] = []
    for item in parsed.get("routes", []):
        rows.append(db_insert("concept_routes", {"project_id": project_id, **item}))

    db_update("projects", {"id": project_id}, {"status": "concepts_ready"})
    log_generation(project_id, "concepts", {"strategy": bundle["strategy"]}, parsed)
    return {"ok": True, "routes": rows}


@app.get("/api/projects/{project_id}/concepts")
def get_concepts(project_id: str, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    ensure_project_owner(project_id, user["id"])
    rows = db_select_many("concept_routes", {"project_id": project_id}, order_by="created_at")
    return {"ok": True, "routes": rows}


@app.patch("/api/projects/{project_id}/concepts/{route_id}/select")
def select_concept(project_id: str, route_id: str, payload: SelectConceptRequest, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    ensure_project_owner(project_id, user["id"])
    route_id = ensure_uuid(route_id, "route_id")
    route = db_select_one("concept_routes", {"id": route_id, "project_id": project_id})
    if not route:
        raise HTTPException(status_code=404, detail="Concept route not found")

    if payload.selected:
        db_update("concept_routes", {"project_id": project_id}, {"is_selected": False})
    db_update("concept_routes", {"id": route_id}, {"is_selected": payload.selected})
    db_update("projects", {"id": project_id}, {"status": "concept_selected" if payload.selected else "concepts_ready"})
    return {"ok": True}


@app.post("/api/projects/{project_id}/execution/generate")
def generate_execution(project_id: str, payload: ExecutionGenerateRequest, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    bundle = get_project_bundle(project_id, user["id"])
    if not bundle["brief"] or not bundle["strategy"]:
        raise HTTPException(status_code=400, detail="Brief and strategy are required")

    selected_route = bundle["selected_route"]
    if payload.route_id:
        selected_route = db_select_one("concept_routes", {"id": ensure_uuid(payload.route_id, "route_id"), "project_id": project_id})
    if not selected_route:
        raise HTTPException(status_code=400, detail="Select a concept route before execution planning")

    if bundle["execution"] and bundle["deliverables"] and not payload.force_regenerate:
        return {"ok": True, "execution_plan": bundle["execution"], "deliverables": bundle["deliverables"]}

    allowed_types = fetch_catalog_types()
    parsed = get_json_completion(
        system_prompt=EXECUTION_PROMPT,
        user_payload={
            "project": bundle["project"],
            "brief": bundle["brief"],
            "strategy": bundle["strategy"],
            "selected_route": selected_route,
            "allowed_deliverable_types": allowed_types,
        },
        schema_name="execution_output",
        schema=EXECUTION_SCHEMA,
        temperature=0.5,
    )

    plan_payload = {
        "concept_route_id": selected_route["id"],
        "plan_summary": parsed["plan_summary"],
        "campaign_phases": parsed["campaign_phases"],
        "timeline_json": parsed["timeline_json"],
        "team_roles": parsed["team_roles"],
        "dependencies": parsed["dependencies"],
    }
    plan_row = upsert_singleton_by_project("execution_plans", project_id, plan_payload)

    db_delete("deliverables", {"execution_plan_id": plan_row["id"]})
    deliverable_rows = []
    for item in parsed.get("deliverables", []):
        deliverable_rows.append(
            db_insert(
                "deliverables",
                {
                    "execution_plan_id": plan_row["id"],
                    "category": item["deliverable_type"],
                    **item,
                },
            )
        )

    db_update("projects", {"id": project_id}, {"status": "execution_ready"})
    log_generation(project_id, "execution", {"route": selected_route}, parsed)
    return {"ok": True, "execution_plan": plan_row, "deliverables": deliverable_rows}


@app.get("/api/projects/{project_id}/execution")
def get_execution(project_id: str, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    bundle = get_project_bundle(project_id, user["id"])
    return {"ok": True, "execution_plan": bundle["execution"], "deliverables": bundle["deliverables"]}


@app.patch("/api/projects/{project_id}/execution/deliverables")
def patch_deliverables(project_id: str, payload: UpdateDeliverablesRequest, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    bundle = get_project_bundle(project_id, user["id"])
    if not bundle["execution"]:
        raise HTTPException(status_code=404, detail="Execution plan not found")

    changed = []
    valid_ids = {d["id"] for d in bundle["deliverables"]}
    for item in payload.deliverables:
        if item.id not in valid_ids:
            raise HTTPException(status_code=404, detail=f"Deliverable not found: {item.id}")
        update_payload = {k: v for k, v in item.model_dump().items() if k != "id" and v is not None}
        rows = db_update("deliverables", {"id": item.id}, update_payload)
        if rows:
            changed.append(rows[0])

    db_update("projects", {"id": project_id}, {"status": "execution_edited"})
    return {"ok": True, "deliverables": changed}


@app.post("/api/projects/{project_id}/costs/generate")
def generate_costs(project_id: str, payload: CostGenerateRequest, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    bundle = get_project_bundle(project_id, user["id"])
    if not bundle["execution"] or not bundle["deliverables"]:
        raise HTTPException(status_code=400, detail="Execution plan with deliverables is required before costing")

    db_delete("cost_line_items", {"cost_estimate_id": "00000000-0000-0000-0000-000000000000"})  # harmless no-op safeguard

    estimate_map = {
        "lean": "lean",
        "recommended": "standard",
        "premium": "premium",
    }

    results = []
    for estimate_type, vendor_tier in estimate_map.items():
        existing_estimate = db_select_one("cost_estimates", {"project_id": project_id, "estimate_type": estimate_type})
        if existing_estimate:
            db_delete("cost_line_items", {"cost_estimate_id": existing_estimate["id"]})

        subtotal = 0.0
        assumptions: List[str] = []
        line_items: List[Dict[str, Any]] = []

        for deliverable in bundle["deliverables"]:
            template = choose_cost_template(deliverable["deliverable_type"], payload.market, vendor_tier)
            if not template:
                assumptions.append(f"Missing cost template for {deliverable['deliverable_type']}; skipped")
                continue

            complexity_map = template.get("complexity_multiplier") or {}
            urgency_map = template.get("urgency_multiplier") or {}
            line_total, mult_summary = compute_line_total(
                base_cost=float(template.get("base_cost") or 0),
                quantity=int(deliverable.get("quantity") or 1),
                complexity=deliverable.get("complexity") or "medium",
                turnaround=deliverable.get("turnaround") or "standard",
                language_count=int(deliverable.get("language_count") or 1),
                complexity_map=complexity_map,
                urgency_map=urgency_map,
                extra_language_multiplier=float(template.get("extra_language_multiplier") or 0),
            )
            subtotal += line_total
            line_items.append(
                {
                    "deliverable_id": deliverable.get("id"),
                    "deliverable_name": deliverable.get("deliverable_name"),
                    "template_used": template.get("deliverable_type"),
                    "unit_cost": float(template.get("base_cost") or 0),
                    "quantity": int(deliverable.get("quantity") or 1),
                    "multiplier_summary": mult_summary,
                    "line_total": round(line_total, 2),
                }
            )

        contingency = round(subtotal * (payload.contingency_percent / 100.0), 2)
        tax = round((subtotal + contingency) * (payload.tax_percent / 100.0), 2)
        total = round(subtotal + contingency + tax, 2)

        assumption_prefix = {
            "lean": "Freelancer-heavy or lean execution model",
            "recommended": "Agency-standard execution model",
            "premium": "Premium agency or production-heavy model",
        }
        assumptions = [
            assumption_prefix[estimate_type],
            f"Market assumed: {payload.market}",
            "Taxes and contingency included in total",
            *assumptions,
        ]

        estimate_payload = {
            "project_id": project_id,
            "estimate_type": estimate_type,
            "subtotal": round(subtotal, 2),
            "contingency": contingency,
            "tax": tax,
            "total": total,
            "assumptions": assumptions,
        }

        if existing_estimate:
            estimate_rows = db_update("cost_estimates", {"id": existing_estimate["id"]}, estimate_payload)
            estimate_row = estimate_rows[0] if estimate_rows else {**existing_estimate, **estimate_payload}
        else:
            estimate_row = db_insert("cost_estimates", estimate_payload)

        for item in line_items:
            db_insert(
                "cost_line_items",
                {
                    "cost_estimate_id": estimate_row["id"],
                    "deliverable_id": item["deliverable_id"],
                    "unit_cost": item["unit_cost"],
                    "quantity": item["quantity"],
                    "multiplier_summary": item["multiplier_summary"],
                    "line_total": item["line_total"],
                },
            )

        results.append({**estimate_row, "line_items": line_items})

    db_update("projects", {"id": project_id}, {"status": "costs_ready"})
    log_generation(project_id, "costing", {"deliverables": bundle["deliverables"]}, {"estimates": results})
    return {"ok": True, "estimates": results}


@app.get("/api/projects/{project_id}/costs")
def get_costs(project_id: str, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    ensure_project_owner(project_id, user["id"])
    rows = db_select_many("cost_estimates", {"project_id": project_id}, order_by="created_at")
    return {"ok": True, "estimates": rows}


@app.get("/api/projects/{project_id}/costs/line-items")
def get_cost_line_items(
    project_id: str,
    estimate_type: str = Query(..., pattern="^(lean|recommended|premium)$"),
    user: Dict[str, Any] = Depends(verify_token_and_get_user),
) -> Dict[str, Any]:
    ensure_project_owner(project_id, user["id"])
    estimate = db_select_one("cost_estimates", {"project_id": project_id, "estimate_type": estimate_type})
    if not estimate:
        raise HTTPException(status_code=404, detail="Cost estimate not found")
    items = db_select_many("cost_line_items", {"cost_estimate_id": estimate["id"]}, order_by="created_at")
    execution = db_select_one("execution_plans", {"project_id": project_id})
    deliverables = db_select_many("deliverables", {"execution_plan_id": execution["id"]}, order_by="created_at") if execution else []
    by_id = {d["id"]: d for d in deliverables}
    enriched = []
    for item in items:
        d = by_id.get(item.get("deliverable_id")) or {}
        enriched.append({
            **item,
            "deliverable_name": d.get("deliverable_name"),
            "deliverable_type": d.get("deliverable_type"),
            "channel": d.get("channel"),
        })
    return {"ok": True, "items": enriched}


@app.post("/api/projects/{project_id}/handoff/generate")
def generate_handoff(project_id: str, payload: HandoffGenerateRequest, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    bundle = get_project_bundle(project_id, user["id"])
    if not bundle["brief"] or not bundle["strategy"] or not bundle["selected_route"] or not bundle["execution"]:
        raise HTTPException(status_code=400, detail="Brief, strategy, selected concept, and execution plan are required")

    existing = db_select_one("agency_packs", {"project_id": project_id})
    if existing and existing.get("brief_doc_url") and not payload.force_regenerate:
        return {"ok": True, "agency_pack": existing}

    costs = db_select_many("cost_estimates", {"project_id": project_id}, order_by="created_at")
    parsed = get_json_completion(
        system_prompt=HANDOFF_PROMPT,
        user_payload={
            "project": bundle["project"],
            "brief": bundle["brief"],
            "strategy": bundle["strategy"],
            "selected_route": bundle["selected_route"],
            "execution_plan": bundle["execution"],
            "deliverables": bundle["deliverables"],
            "cost_summary": costs,
        },
        schema_name="handoff_output",
        schema=HANDOFF_SCHEMA,
        temperature=0.3,
    )

    brief_path = save_text_export(project_id, "agency-brief", parsed["agency_brief_markdown"])
    sow_path = save_text_export(project_id, "scope-of-work", parsed["sow_summary_markdown"])
    rfp_path = save_text_export(project_id, "rfp-summary", parsed["rfp_summary_markdown"])

    version = int(existing["version"]) + 1 if existing else 1
    agency_payload = {
        "brief_doc_url": brief_path,
        "sow_doc_url": sow_path,
        "rfp_doc_url": rfp_path,
        "email_draft": parsed["email_draft"],
        "version": version,
    }

    if existing:
        updated = db_update("agency_packs", {"id": existing["id"]}, agency_payload)
        row = updated[0] if updated else {**existing, **agency_payload}
    else:
        row = db_insert("agency_packs", {"project_id": project_id, **agency_payload})

    db_update("projects", {"id": project_id}, {"status": "handoff_ready"})
    log_generation(project_id, "handoff", {"bundle": bundle}, parsed)
    return {"ok": True, "agency_pack": row}


@app.get("/api/projects/{project_id}/handoff")
def get_handoff(project_id: str, user: Dict[str, Any] = Depends(verify_token_and_get_user)) -> Dict[str, Any]:
    ensure_project_owner(project_id, user["id"])
    row = db_select_one("agency_packs", {"project_id": project_id})
    return {"ok": True, "agency_pack": row}


# =========================================================
# DEV NOTES ROUTE
# =========================================================
@app.get("/api/dev/notes")
def dev_notes() -> Dict[str, Any]:
    return {
        "ok": True,
        "notes": [
            "Create the SQL tables first in Supabase before running this API.",
            "Set OPENAI_MODEL to a model that supports json_schema output, such as gpt-4o-mini.",
            "Send the Supabase access token from your frontend as Authorization: Bearer <token>.",
            "For PDF parsing install pypdf. For DOCX parsing install python-docx.",
            "This starter writes generated handoff documents as markdown files on local disk.",
        ],
    }


# =========================================================
# RUN
# =========================================================
# uvicorn main:app --reload
