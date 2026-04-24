"""
AICreative Studio — Production Backend v3.0
FastAPI + Anthropic Claude + Supabase
"""
import os
import json
import re
import asyncio
from datetime import datetime
from typing import Optional, AsyncGenerator

import anthropic
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
SUPABASE_URL    = os.getenv("SUPABASE_URL", "https://qkcvhkrhifudxmackwas.supabase.co")
SUPABASE_SVCKEY = os.getenv("SUPABASE_SERVICE_KEY", "")
MODEL           = "claude-sonnet-4-20250514"
MAX_TOKENS      = 2000

# ══════════════════════════════════════════════════
# CLIENTS
# ══════════════════════════════════════════════════
app = FastAPI(title="AICreative Studio API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

claude: Optional[anthropic.Anthropic] = (
    anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None
)

db: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_SVCKEY) if SUPABASE_SVCKEY else None
)

# ══════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════
class AuthBody(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

class ProjectCreate(BaseModel):
    brief: str
    style_chips: Optional[str] = ""

class AnalyzeBody(BaseModel):
    brief: str
    style_chips: Optional[str] = ""

class ConceptBody(BaseModel):
    brief: str
    style_chips: Optional[str] = ""

class ConceptStageBody(BaseModel):
    brief: str
    concept_id: str

class SelectConceptBody(BaseModel):
    concept_id: str

class EditBody(BaseModel):
    section: str
    feedback: str
    concept_id: Optional[str] = None
    current_data: Optional[dict] = None

class ChatBody(BaseModel):
    message: str
    project_id: Optional[str] = None
    context: Optional[dict] = {}

# ══════════════════════════════════════════════════
# AUTH DEPENDENCY
# ══════════════════════════════════════════════════
async def require_auth(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = authorization[7:]
    if not db:
        # Dev mode — return a dummy user
        class MockUser:
            id = "dev-user-id"
            email = "dev@example.com"
            user_metadata = {"name": "Dev User"}
        return MockUser()
    try:
        resp = db.auth.get_user(token)
        if not resp or not resp.user:
            raise HTTPException(401, "Invalid token")
        return resp.user
    except Exception as e:
        raise HTTPException(401, f"Auth error: {str(e)}")

# ══════════════════════════════════════════════════
# CLAUDE HELPERS
# ══════════════════════════════════════════════════
async def call_claude(system: str, user_msg: str, max_tok: int = MAX_TOKENS) -> str:
    if not claude:
        raise HTTPException(503, "Anthropic API key not configured")
    msg = claude.messages.create(
        model=MODEL,
        max_tokens=max_tok,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return msg.content[0].text

async def stream_claude_sse(system: str, user_msg: str) -> AsyncGenerator[str, None]:
    if not claude:
        yield 'data: {"error":"Anthropic API key not configured"}\n\n'
        return
    try:
        with claude.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for text in stream.text_stream:
                payload = json.dumps({"text": text})
                yield f"data: {payload}\n\n"
        yield 'data: {"done":true}\n\n'
    except Exception as e:
        yield f'data: {{"error":"{str(e)}"}}\n\n'

def parse_json_safe(raw: str):
    """Extract and parse JSON from LLM response robustly."""
    try:
        # Try code fence first
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            return json.loads(m.group(1).strip())
        # Try bare JSON object or array
        m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", raw)
        if m:
            return json.loads(m.group(1).strip())
        return json.loads(raw.strip())
    except Exception:
        return None

def db_insert(table: str, data: dict):
    if not db:
        return {}
    res = db.table(table).insert(data).execute()
    return res.data[0] if res.data else {}

def db_upsert(table: str, data: dict):
    if not db:
        return {}
    res = db.table(table).upsert(data).execute()
    return res.data[0] if res.data else {}

def db_update(table: str, data: dict, **filters):
    if not db:
        return
    q = db.table(table).update(data)
    for col, val in filters.items():
        q = q.eq(col, val)
    q.execute()

def db_select(table: str, columns: str = "*", order_col: str = "created_at",
              limit: int = 1, **filters):
    if not db:
        return []
    q = db.table(table).select(columns)
    for col, val in filters.items():
        q = q.eq(col, val)
    q = q.order(order_col, desc=True).limit(limit)
    return q.execute().data or []

# ══════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════
@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "model": MODEL,
        "anthropic": bool(claude),
        "supabase": bool(db),
        "timestamp": datetime.utcnow().isoformat(),
    }

# ══════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════
@app.post("/auth/signup")
async def signup(body: AuthBody):
    if not db:
        raise HTTPException(503, "Database not configured")
    try:
        name = body.name or body.email.split("@")[0]
        res = db.auth.sign_up({
            "email": body.email,
            "password": body.password,
            "options": {"data": {"name": name}},
        })
        if not res.user:
            raise HTTPException(400, "Signup failed — user already exists or invalid email")
        token = res.session.access_token if res.session else None
        return {
            "token": token,
            "user": {
                "id": res.user.id,
                "email": res.user.email,
                "name": name,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/auth/login")
async def login(body: AuthBody):
    if not db:
        raise HTTPException(503, "Database not configured")
    try:
        res = db.auth.sign_in_with_password({
            "email": body.email,
            "password": body.password,
        })
        if not res.user or not res.session:
            raise HTTPException(401, "Invalid email or password")
        name = res.user.user_metadata.get("name", res.user.email)
        return {
            "token": res.session.access_token,
            "user": {
                "id": res.user.id,
                "email": res.user.email,
                "name": name,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(401, str(e))

# ══════════════════════════════════════════════════
# PROJECTS
# ══════════════════════════════════════════════════
@app.get("/projects")
async def list_projects(user=Depends(require_auth)):
    if not db:
        return []
    res = (
        db.table("projects")
        .select("id,project_name,status,created_at,updated_at")
        .eq("user_id", user.id)
        .order("updated_at", desc=True)
        .limit(20)
        .execute()
    )
    return res.data or []

@app.post("/projects")
async def create_project(body: ProjectCreate, user=Depends(require_auth)):
    words = body.brief.strip().split()
    name = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
    row = db_insert("projects", {
        "user_id": user.id,
        "project_name": name,
        "status": "draft",
    })
    if not row:
        raise HTTPException(500, "Failed to create project")

    project_id = row["id"]
    # Store the raw brief
    db_insert("creative_briefs", {
        "project_id": project_id,
        "source_type": "text",
        "raw_text": body.brief,
    })
    return {"project_id": project_id, "name": name}

# ══════════════════════════════════════════════════
# STEP 1: BRIEF ANALYSIS (STREAMING)
# ══════════════════════════════════════════════════
@app.post("/projects/{project_id}/analyze")
async def analyze_brief(
    project_id: str,
    body: AnalyzeBody,
    user=Depends(require_auth),
):
    system = (
        "You are a world-class creative director and event designer with 20+ years of experience "
        "crafting spectacular brand experiences for Fortune 500 companies. "
        "Analyze event briefs with strategic depth, creative vision, and production knowledge. "
        "Write in clear, flowing prose paragraphs — no bullet lists. Be specific and insightful."
    )
    prompt = f"""Analyze this event brief in exactly 3 insightful paragraphs.

Brief: "{body.brief}"
Style preferences: {body.style_chips or 'Not specified'}

Paragraph 1 — Venue & Audience: Analyze the venue type, scale, audience profile, and psychology.
Paragraph 2 — Design Drivers & Mood: Identify the key design language, mood, materials, and aesthetic direction.
Paragraph 3 — Creative Opportunities: Highlight 2–3 specific creative opportunities and technical considerations that will make this event exceptional."""

    accumulated = []

    async def generate():
        async for chunk in stream_claude_sse(system, prompt):
            if '"text"' in chunk:
                try:
                    data = json.loads(chunk[6:])
                    accumulated.append(data.get("text", ""))
                except Exception:
                    pass
            yield chunk

        # Save to DB after streaming completes
        full_text = "".join(accumulated)
        if full_text:
            try:
                db_insert("brief_analysis", {
                    "project_id": project_id,
                    "analysis_text": full_text,
                })
                db_update("projects", {"status": "analyzed"}, id=project_id)
            except Exception as e:
                print(f"[DB] analysis save error: {e}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ══════════════════════════════════════════════════
# STEP 2: CONCEPTS
# ══════════════════════════════════════════════════
@app.post("/projects/{project_id}/concepts")
async def generate_concepts(
    project_id: str,
    body: ConceptBody,
    user=Depends(require_auth),
):
    # Pull latest analysis for enrichment
    analysis_rows = db_select("brief_analysis", "analysis_text", project_id=project_id)
    analysis_excerpt = (
        analysis_rows[0]["analysis_text"][:600] if analysis_rows else ""
    )

    raw = await call_claude(
        "Award-winning event creative director. Respond ONLY with a valid JSON array — no markdown fences, no preamble.",
        f"""Generate exactly 3 highly distinct creative event design concepts.

Brief: "{body.brief}"
Style preferences: {body.style_chips or 'Luxury / Premium'}
Strategic insight: {analysis_excerpt}

Each concept must be genuinely different in visual language, materials, and emotional impact.

Return ONLY a JSON array:
[{{
  "name": "2-4 word evocative concept title",
  "tagline": "One compelling sentence capturing the creative vision",
  "description": "Two sentences on visual language, key materials, and atmosphere",
  "style": "Short style tag (e.g. 'Neo-Futurist', 'Organic Luxury', 'Bold Heritage')",
  "emoji": "single relevant emoji",
  "bgColor": "dark hex e.g. #1a1230",
  "tagBg": "slightly lighter hex e.g. #2d1a5e",
  "tagColor": "complementary light hex e.g. #c4b5fd"
}}]""",
        max_tok=1500,
    )

    concepts = parse_json_safe(raw)
    if not concepts or not isinstance(concepts, list) or len(concepts) == 0:
        raise HTTPException(500, f"Failed to parse concepts from LLM. Raw: {raw[:200]}")

    # Delete old concepts for this project, insert fresh ones
    if db:
        db.table("studio_concepts").delete().eq("project_id", project_id).execute()

    saved = []
    for c in concepts[:3]:
        row = db_insert("studio_concepts", {
            "project_id": project_id,
            "name": c.get("name", "Concept"),
            "tagline": c.get("tagline", ""),
            "description": c.get("description", ""),
            "style": c.get("style", ""),
            "emoji": c.get("emoji", "✦"),
            "bg_color": c.get("bgColor", "#1a1230"),
            "tag_bg": c.get("tagBg", "#3b2a6e"),
            "tag_color": c.get("tagColor", "#a78bfa"),
            "is_selected": False,
        })
        saved.append({**c, "id": row.get("id", "")})

    db_update("projects", {"status": "concepts_ready"}, id=project_id)
    return {"concepts": saved}

# ══════════════════════════════════════════════════
# SELECT CONCEPT
# ══════════════════════════════════════════════════
@app.post("/projects/{project_id}/select-concept")
async def select_concept(
    project_id: str,
    body: SelectConceptBody,
    user=Depends(require_auth),
):
    if db:
        # Deselect all, then select chosen
        db.table("studio_concepts").update({"is_selected": False}).eq("project_id", project_id).execute()
        db.table("studio_concepts").update({"is_selected": True}).eq("id", body.concept_id).execute()
    db_update("projects", {"status": "concept_selected"}, id=project_id)
    return {"ok": True, "concept_id": body.concept_id}

# ══════════════════════════════════════════════════
# STEP 3: MOOD BOARD
# ══════════════════════════════════════════════════
@app.post("/projects/{project_id}/mood")
async def generate_mood(
    project_id: str,
    body: ConceptStageBody,
    user=Depends(require_auth),
):
    c_rows = db_select("studio_concepts", "*", id=body.concept_id)
    if not c_rows:
        raise HTTPException(404, "Concept not found")
    c = c_rows[0]

    raw = await call_claude(
        "Senior creative director and colour theorist. Respond ONLY with valid JSON — no markdown fences.",
        f"""Create a rich mood board specification for this event concept.

Concept: "{c['name']}" — {c.get('tagline', '')}
Visual description: {c.get('description', '')}
Brief: "{body.brief}"

Return ONLY this JSON (no other text):
{{
  "palette": [
    {{"hex":"#colorhex","name":"Color Name","role":"Primary"}},
    {{"hex":"#colorhex","name":"Color Name","role":"Accent 1"}},
    {{"hex":"#colorhex","name":"Color Name","role":"Accent 2"}},
    {{"hex":"#colorhex","name":"Color Name","role":"Neutral"}},
    {{"hex":"#colorhex","name":"Color Name","role":"Dark Base"}}
  ],
  "typography": {{
    "headline": "Headline font name (e.g. Playfair Display, Bebas Neue)",
    "body": "Body font name (e.g. DM Sans, Inter)"
  }},
  "materials": ["material 1","material 2","material 3","material 4","material 5"],
  "mood_words": ["evocative word 1","word 2","word 3","word 4","word 5","word 6"],
  "lighting": "Description of the lighting approach and atmosphere",
  "textures": ["texture/finish 1","texture 2","texture 3"]
}}

Use sophisticated, specific hex values that create a premium palette.""",
        max_tok=1200,
    )

    mood = parse_json_safe(raw)
    if not mood:
        raise HTTPException(500, f"Failed to parse mood board. Raw: {raw[:200]}")

    db_insert("mood_boards", {
        "project_id": project_id,
        "concept_id": body.concept_id,
        "palette": mood.get("palette", []),
        "typography": mood.get("typography", {}),
        "materials": mood.get("materials", []),
        "mood_words": mood.get("mood_words", []),
        "lighting": mood.get("lighting", ""),
        "textures": mood.get("textures", []),
    })

    return {"mood": mood}

# ══════════════════════════════════════════════════
# STEP 4: 2D GRAPHICS
# ══════════════════════════════════════════════════
@app.post("/projects/{project_id}/graphics")
async def generate_graphics(
    project_id: str,
    body: ConceptStageBody,
    user=Depends(require_auth),
):
    c_rows = db_select("studio_concepts", "*", id=body.concept_id)
    c = c_rows[0] if c_rows else {}
    mood_rows = db_select("mood_boards", "palette", concept_id=body.concept_id)
    palette = (mood_rows[0].get("palette", [])[:3]) if mood_rows else []

    raw = await call_claude(
        "Senior graphic designer for luxury events, brand experiences and exhibitions. Respond ONLY with valid JSON.",
        f"""Design 4 key 2D graphic elements for this event.

Brief: "{body.brief}"
Concept: "{c.get('name','')}" — {c.get('description','')}
Colour palette: {json.dumps(palette)}

Return ONLY this JSON (no other text):
{{
  "items": [
    {{
      "name": "Element name (e.g. Hero Stage Backdrop)",
      "description": "Purpose and display context",
      "sample": "Short sample text to render (max 5 words)",
      "font": "Syne",
      "fontSize": "18px",
      "textColor": "#hexcolor",
      "bgColor": "#dark-hexcolor",
      "letterSpacing": "0.06em",
      "textTransform": "uppercase"
    }}
  ],
  "notes": "One line creative direction note"
}}

Include all 4: 1) Hero Key Visual/Stage Backdrop, 2) Entrance Signage & Wayfinding, 3) Invitation/Save-the-Date Card, 4) Digital LED Screen Loop Title.""",
        max_tok=1200,
    )

    graphics = parse_json_safe(raw)
    if not graphics:
        raise HTTPException(500, f"Failed to parse graphics. Raw: {raw[:200]}")

    db_insert("graphics_outputs", {
        "project_id": project_id,
        "concept_id": body.concept_id,
        "items": graphics.get("items", []),
        "notes": graphics.get("notes", ""),
    })

    return {"graphics": graphics}

# ══════════════════════════════════════════════════
# STEP 5: CAD LAYOUT
# ══════════════════════════════════════════════════
@app.post("/projects/{project_id}/layout")
async def generate_layout(
    project_id: str,
    body: ConceptStageBody,
    user=Depends(require_auth),
):
    c_rows = db_select("studio_concepts", "*", id=body.concept_id)
    c = c_rows[0] if c_rows else {}

    raw = await call_claude(
        "Expert event spatial planner, CAD designer and flow architect. Respond ONLY with valid JSON.",
        f"""Design a detailed spatial floor plan layout for this event.

Brief: "{body.brief}"
Concept: "{c.get('name','')}" — {c.get('description','')}

Return ONLY this JSON (no other text):
{{
  "total_area": "X,XXX m²",
  "dimensions": "XX m × XX m",
  "capacity": "XXX guests",
  "zones": [
    {{
      "name": "Zone Name",
      "left": "10%",
      "top": "10%",
      "width": "35%",
      "height": "30%",
      "color": "#1a123088",
      "border": "#6b46c166"
    }}
  ],
  "notes": "2-3 sentences on spatial flow, zone rationale, and movement design"
}}

Include 7 zones: Main Stage/Performance Area, Guest Floor/Seating, VIP Lounge, Registration/Entry, F&B/Bar Station, Backstage/Production, Networking/Breakout. 
Use percentage positioning to fill a reasonable rectangular floor plan.""",
        max_tok=1500,
    )

    layout = parse_json_safe(raw)
    if not layout:
        raise HTTPException(500, f"Failed to parse layout. Raw: {raw[:200]}")

    db_insert("cad_layouts", {
        "project_id": project_id,
        "concept_id": body.concept_id,
        "total_area": layout.get("total_area", ""),
        "dimensions": layout.get("dimensions", ""),
        "capacity": layout.get("capacity", ""),
        "zones": layout.get("zones", []),
        "notes": layout.get("notes", ""),
    })

    return {"layout": layout}

# ══════════════════════════════════════════════════
# STEP 6: 3D RENDER PROMPTS
# ══════════════════════════════════════════════════
@app.post("/projects/{project_id}/render")
async def generate_render(
    project_id: str,
    body: ConceptStageBody,
    user=Depends(require_auth),
):
    c_rows = db_select("studio_concepts", "*", id=body.concept_id)
    c = c_rows[0] if c_rows else {}
    mood_rows = db_select("mood_boards", "palette,lighting", concept_id=body.concept_id)
    mood = mood_rows[0] if mood_rows else {}

    raw = await call_claude(
        "World-class architectural visualizer and cinematic art director for luxury brand experiences. Respond ONLY with valid JSON.",
        f"""Write 3 masterful cinematic 3D render prompts for this event space.

Brief: "{body.brief}"
Concept: "{c.get('name','')}" — {c.get('description','')}
Lighting direction: {mood.get('lighting','Dramatic atmospheric')}
Palette: {json.dumps(mood.get('palette',[])[:3])}

Return ONLY this JSON (no other text):
{{
  "prompts": [
    {{
      "view": "Hero Wide Angle",
      "prompt": "Full 80-100 word photorealistic cinematic prompt describing: specific materials and surfaces, lighting setup, atmosphere, camera angle, key design features, time of day, photography/render style. Make it vivid and specific."
    }},
    {{
      "view": "Intimate Detail Close-up",
      "prompt": "Full 80-100 word macro/detail prompt focusing on a hero design element — texture, finish, lighting interaction."
    }},
    {{
      "view": "Aerial Establishing Shot",
      "prompt": "Full 80-100 word aerial/drone prompt showing full spatial layout, all zones, surrounding context."
    }}
  ],
  "style_notes": "Concise note on render style, recommended settings and platform (Midjourney v6, DALL-E 3, etc.)",
  "negative_prompt": "Comma-separated negative prompt: quality issues, style conflicts to avoid"
}}""",
        max_tok=1800,
    )

    render = parse_json_safe(raw)
    if not render:
        raise HTTPException(500, f"Failed to parse render prompts. Raw: {raw[:200]}")

    db_insert("render_outputs", {
        "project_id": project_id,
        "concept_id": body.concept_id,
        "prompts": render.get("prompts", []),
        "style_notes": render.get("style_notes", ""),
        "negative_prompt": render.get("negative_prompt", ""),
    })

    return {"render": render}

# ══════════════════════════════════════════════════
# STEP 7: PRODUCTION SPECS
# ══════════════════════════════════════════════════
@app.post("/projects/{project_id}/specs")
async def generate_specs(
    project_id: str,
    body: ConceptStageBody,
    user=Depends(require_auth),
):
    c_rows = db_select("studio_concepts", "*", id=body.concept_id)
    c = c_rows[0] if c_rows else {}
    layout_rows = db_select("cad_layouts", "total_area,dimensions,capacity", concept_id=body.concept_id)
    layout = layout_rows[0] if layout_rows else {}

    raw = await call_claude(
        "Senior event production manager and technical director with 20+ years experience across large-format events. Respond ONLY with valid JSON.",
        f"""Generate comprehensive production specifications for this event.

Brief: "{body.brief}"
Concept: "{c.get('name','')}" — {c.get('description','')}
Venue: {layout.get('total_area','2,000 m²')} | Dimensions: {layout.get('dimensions','50m × 40m')} | Capacity: {layout.get('capacity','500 guests')}

Return ONLY this JSON (no other text):
{{
  "specs": [
    {{"label":"Venue Area","value":"X,XXX m²"}},
    {{"label":"Guest Capacity","value":"XXX pax (seated) / XXX (cocktail)"}},
    {{"label":"Main Stage Structure","value":"specific materials and dimensions"}},
    {{"label":"Rigging System","value":"type and load capacity"}},
    {{"label":"Lighting Rig","value":"fixture types and count"}},
    {{"label":"LED / AV System","value":"screen specs and configuration"}},
    {{"label":"Flooring Treatment","value":"materials and area"}},
    {{"label":"Signage & Branding","value":"number of pieces and key locations"}},
    {{"label":"Power Supply","value":"XXX kVA + backup"}},
    {{"label":"Load-in Duration","value":"XX hours across X days"}},
    {{"label":"Crew Required","value":"XX crew (breakdown by role)"}},
    {{"label":"Estimated Budget","value":"₹XX – ₹XX lakhs (INR)"}}
  ],
  "highlight": "One key production insight, risk flag, or recommendation for the client"
}}""",
        max_tok=1500,
    )

    specs = parse_json_safe(raw)
    if not specs:
        raise HTTPException(500, f"Failed to parse specs. Raw: {raw[:200]}")

    db_insert("production_specs", {
        "project_id": project_id,
        "concept_id": body.concept_id,
        "specs": specs.get("specs", []),
        "highlight": specs.get("highlight", ""),
    })

    db_update("projects", {"status": "complete"}, id=project_id)
    return {"specs": specs}

# ══════════════════════════════════════════════════
# EDIT / REVISE SECTION
# ══════════════════════════════════════════════════
@app.post("/projects/{project_id}/edit")
async def edit_section(
    project_id: str,
    body: EditBody,
    user=Depends(require_auth),
):
    section = body.section
    feedback = body.feedback
    concept_id = body.concept_id
    prev = body.current_data or {}

    if section == "analysis":
        raw = await call_claude(
            "Expert creative director. Revise the brief analysis based on the feedback. Write 3 insightful prose paragraphs — no bullet lists.",
            f"Previous analysis:\n\"{json.dumps(prev)}\"\n\nFeedback: \"{feedback}\"\n\nRevise the analysis accordingly. Return only the revised text.",
        )
        db_insert("brief_analysis", {"project_id": project_id, "analysis_text": raw})
        return {"section": "analysis", "result": raw}

    elif section == "concepts":
        raw = await call_claude(
            "Creative director. Return ONLY a valid JSON array of 3 concepts — no markdown.",
            f"Previous concepts: {json.dumps(prev)}\n\nFeedback: \"{feedback}\"\n\nRevise all 3 concepts. Same JSON structure.",
        )
        concepts = parse_json_safe(raw)
        if not concepts:
            raise HTTPException(500, "Failed to revise concepts")
        if db:
            db.table("studio_concepts").delete().eq("project_id", project_id).execute()
        saved = []
        for c in concepts[:3]:
            row = db_insert("studio_concepts", {
                "project_id": project_id,
                "name": c.get("name", ""),
                "tagline": c.get("tagline", ""),
                "description": c.get("description", ""),
                "style": c.get("style", ""),
                "emoji": c.get("emoji", "✦"),
                "bg_color": c.get("bgColor", "#1a1230"),
                "tag_bg": c.get("tagBg", "#3b2a6e"),
                "tag_color": c.get("tagColor", "#a78bfa"),
            })
            saved.append({**c, "id": row.get("id", "")})
        return {"section": "concepts", "result": saved}

    elif section == "mood" and concept_id:
        raw = await call_claude(
            "Creative director. Revise the mood board. Return ONLY valid JSON.",
            f"Previous mood: {json.dumps(prev)}\nFeedback: \"{feedback}\"\nRevise. Same JSON structure.",
        )
        mood = parse_json_safe(raw)
        if not mood:
            raise HTTPException(500, "Failed to revise mood")
        db_insert("mood_boards", {
            "project_id": project_id, "concept_id": concept_id,
            "palette": mood.get("palette", []), "typography": mood.get("typography", {}),
            "materials": mood.get("materials", []), "mood_words": mood.get("mood_words", []),
            "lighting": mood.get("lighting", ""), "textures": mood.get("textures", []),
        })
        return {"section": "mood", "result": mood}

    elif section == "graphics" and concept_id:
        raw = await call_claude(
            "Graphic designer. Revise the 2D graphics. Return ONLY valid JSON.",
            f"Previous: {json.dumps(prev)}\nFeedback: \"{feedback}\"\nRevise. Same JSON structure.",
        )
        graphics = parse_json_safe(raw)
        if not graphics:
            raise HTTPException(500, "Failed to revise graphics")
        db_insert("graphics_outputs", {
            "project_id": project_id, "concept_id": concept_id,
            "items": graphics.get("items", []), "notes": graphics.get("notes", ""),
        })
        return {"section": "graphics", "result": graphics}

    elif section == "layout" and concept_id:
        raw = await call_claude(
            "Spatial designer. Revise the CAD layout. Return ONLY valid JSON.",
            f"Previous: {json.dumps(prev)}\nFeedback: \"{feedback}\"\nRevise. Same JSON structure.",
        )
        layout = parse_json_safe(raw)
        if not layout:
            raise HTTPException(500, "Failed to revise layout")
        db_insert("cad_layouts", {
            "project_id": project_id, "concept_id": concept_id,
            "total_area": layout.get("total_area", ""),
            "dimensions": layout.get("dimensions", ""),
            "capacity": layout.get("capacity", ""),
            "zones": layout.get("zones", []), "notes": layout.get("notes", ""),
        })
        return {"section": "layout", "result": layout}

    elif section == "render" and concept_id:
        raw = await call_claude(
            "Architectural visualizer. Revise render prompts. Return ONLY valid JSON.",
            f"Previous: {json.dumps(prev)}\nFeedback: \"{feedback}\"\nRevise. Same JSON structure.",
        )
        render = parse_json_safe(raw)
        if not render:
            raise HTTPException(500, "Failed to revise render prompts")
        db_insert("render_outputs", {
            "project_id": project_id, "concept_id": concept_id,
            "prompts": render.get("prompts", []),
            "style_notes": render.get("style_notes", ""),
            "negative_prompt": render.get("negative_prompt", ""),
        })
        return {"section": "render", "result": render}

    elif section == "specs" and concept_id:
        raw = await call_claude(
            "Production manager. Revise specs. Return ONLY valid JSON.",
            f"Previous: {json.dumps(prev)}\nFeedback: \"{feedback}\"\nRevise. Same JSON structure.",
        )
        specs = parse_json_safe(raw)
        if not specs:
            raise HTTPException(500, "Failed to revise specs")
        db_insert("production_specs", {
            "project_id": project_id, "concept_id": concept_id,
            "specs": specs.get("specs", []), "highlight": specs.get("highlight", ""),
        })
        return {"section": "specs", "result": specs}

    raise HTTPException(400, f"Unknown section: {section}")

# ══════════════════════════════════════════════════
# AI CHAT (STREAMING)
# ══════════════════════════════════════════════════
@app.post("/chat")
async def chat(body: ChatBody, user=Depends(require_auth)):
    context_parts = [f"User: {user.email}"]

    if body.project_id and db:
        try:
            proj = db.table("projects").select("project_name,status").eq("id", body.project_id).single().execute()
            if proj.data:
                context_parts.append(f"Project: {proj.data['project_name']} (status: {proj.data['status']})")

            analysis = db_select("brief_analysis", "analysis_text", project_id=body.project_id)
            if analysis:
                context_parts.append(f"Brief Analysis: {analysis[0]['analysis_text'][:400]}…")

            concepts = db.table("studio_concepts").select("name,tagline,is_selected").eq("project_id", body.project_id).execute()
            if concepts.data:
                selected = next((c for c in concepts.data if c.get("is_selected")), None)
                names = [c["name"] for c in concepts.data]
                context_parts.append(f"Concepts: {', '.join(names)}. Selected: {selected['name'] if selected else 'none'}")

            mood = db_select("mood_boards", "mood_words,materials", project_id=body.project_id)
            if mood:
                context_parts.append(f"Mood words: {mood[0].get('mood_words', [])}. Materials: {mood[0].get('materials', [])}")

            layout = db_select("cad_layouts", "total_area,capacity", project_id=body.project_id)
            if layout:
                context_parts.append(f"Venue: {layout[0].get('total_area','?')} | {layout[0].get('capacity','?')}")
        except Exception as e:
            print(f"[Chat] Context fetch error: {e}")

    if body.context:
        context_parts.append(f"Additional: {json.dumps(body.context)}")

    context_str = "\n".join(context_parts)

    system = (
        "You are the AICreative Studio AI Assistant — an expert creative director, event designer, and production specialist. "
        "You help users refine their event creative packages. Be concise (under 120 words), specific, and immediately actionable. "
        "Reference project context when relevant. For section revisions, guide users to use the Feedback button on that output block."
    )
    user_msg = f"Project context:\n{context_str}\n\nUser message: \"{body.message}\""

    accumulated = []

    async def generate():
        async for chunk in stream_claude_sse(system, user_msg):
            if '"text"' in chunk:
                try:
                    data = json.loads(chunk[6:])
                    accumulated.append(data.get("text", ""))
                except Exception:
                    pass
            yield chunk

        # Persist chat messages
        full_response = "".join(accumulated)
        if body.project_id and db and full_response:
            try:
                db.table("chat_messages").insert([
                    {"project_id": body.project_id, "user_id": user.id, "role": "user", "content": body.message},
                    {"project_id": body.project_id, "user_id": user.id, "role": "assistant", "content": full_response},
                ]).execute()
            except Exception as e:
                print(f"[Chat] Persist error: {e}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
