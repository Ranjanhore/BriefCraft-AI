import html
import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# =========================================================
# CONFIG
# =========================================================

APP_NAME = "ExpoAI Concept-First Backend"
BASE_DIR = Path(__file__).resolve().parent

STORAGE_DIR = BASE_DIR / "storage"
DB_PATH = STORAGE_DIR / "expoai.db"
REFERENCE_DIR = STORAGE_DIR / "reference_images"
CONCEPT_DIR = STORAGE_DIR / "concepts"
OUTPUT_DIR = STORAGE_DIR / "outputs"

for p in [STORAGE_DIR, REFERENCE_DIR, CONCEPT_DIR, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# UTILS
# =========================================================

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def normalize_id(value: str) -> str:
    value = (value or "").strip()
    value = "".join(ch for ch in value if ch.isalnum() or ch == "-")
    return value

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any]:
    return dict(row) if row else {}

def json_load(value: Optional[str], fallback: Any = None) -> Any:
    if value in [None, ""]:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback

def safe_text(value: str) -> str:
    return html.escape(value or "")

def wrap_text(text: str, max_chars: int) -> list[str]:
    words = (text or "").split()
    if not words:
        return [""]
    lines = []
    current = ""
    for word in words:
        test = word if not current else f"{current} {word}"
        if len(test) <= max_chars:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:5]

# =========================================================
# DATABASE
# =========================================================

def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        project_type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'draft',
        selected_concept_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS briefs (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        brief_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS reference_images (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        label TEXT,
        notes TEXT,
        file_path TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS concepts (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        concept_no INTEGER NOT NULL,
        title TEXT NOT NULL,
        one_liner TEXT NOT NULL,
        summary TEXT NOT NULL,
        design_story TEXT NOT NULL,
        style_direction TEXT NOT NULL,
        layout_direction TEXT NOT NULL,
        materials_json TEXT NOT NULL,
        colors_json TEXT NOT NULL,
        hero_visual TEXT NOT NULL,
        board_path TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'draft',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS concept_reference_links (
        id TEXT PRIMARY KEY,
        concept_id TEXT NOT NULL,
        reference_image_id TEXT NOT NULL,
        caption TEXT,
        sort_order INTEGER NOT NULL DEFAULT 1
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS outputs (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        concept_id TEXT NOT NULL,
        output_type TEXT NOT NULL,
        title TEXT NOT NULL,
        file_path TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()

@app.on_event("startup")
def startup_event() -> None:
    init_db()

# =========================================================
# MODELS
# =========================================================

class ProjectCreate(BaseModel):
    name: str
    project_type: str = "exhibition_booth"  # exhibition_booth / event_stage / decor_zone / backdrop / experience_area

class BriefSubmit(BaseModel):
    brand_name: str
    event_name: str = ""
    width_mm: int = Field(..., gt=0)
    depth_mm: int = Field(..., gt=0)
    height_mm: int = Field(3500, gt=0)
    target_audience: str = ""
    goals: list[str] = []
    key_messages: list[str] = []
    must_have_zones: list[str] = []
    style_keywords: list[str] = []
    materials_preferences: list[str] = []
    budget_level: str = "mid"
    notes: str = ""

class ConceptUpdate(BaseModel):
    title: Optional[str] = None
    one_liner: Optional[str] = None
    summary: Optional[str] = None
    design_story: Optional[str] = None
    style_direction: Optional[str] = None
    layout_direction: Optional[str] = None
    materials: Optional[list[str]] = None
    colors: Optional[list[str]] = None
    hero_visual: Optional[str] = None

# =========================================================
# FETCH HELPERS
# =========================================================

def fetch_project(project_id: str) -> sqlite3.Row:
    project_id = normalize_id(project_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return row

def fetch_brief(project_id: str) -> sqlite3.Row:
    project = fetch_project(project_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM briefs WHERE project_id = ?", (project["id"],))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Brief not found")
    return row

def fetch_concept(concept_id: str) -> sqlite3.Row:
    concept_id = normalize_id(concept_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM concepts WHERE id = ?", (concept_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Concept not found")
    return row

def fetch_reference_image(reference_id: str) -> sqlite3.Row:
    reference_id = normalize_id(reference_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM reference_images WHERE id = ?", (reference_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Reference image not found")
    return row

def fetch_output(output_id: str) -> sqlite3.Row:
    output_id = normalize_id(output_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM outputs WHERE id = ?", (output_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Output not found")
    return row

# =========================================================
# PROJECT SNAPSHOT
# =========================================================

def get_project_full(project_id: str) -> dict[str, Any]:
    project = fetch_project(project_id)
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM briefs WHERE project_id = ?", (project["id"],))
    brief = cur.fetchone()

    cur.execute("SELECT * FROM reference_images WHERE project_id = ? ORDER BY created_at DESC", (project["id"],))
    refs = cur.fetchall()

    cur.execute("SELECT * FROM concepts WHERE project_id = ? ORDER BY concept_no ASC", (project["id"],))
    concepts = cur.fetchall()

    cur.execute("""
        SELECT * FROM outputs
        WHERE project_id = ?
        ORDER BY created_at DESC
    """, (project["id"],))
    outputs = cur.fetchall()

    conn.close()

    return {
        "project": dict(project),
        "brief": json_load(brief["brief_json"], {}) if brief else None,
        "reference_images": [dict(r) for r in refs],
        "concepts": [
            {
                **dict(c),
                "materials": json_load(c["materials_json"], []),
                "colors": json_load(c["colors_json"], []),
            }
            for c in concepts
        ],
        "outputs": [
            {
                **dict(o),
                "payload": json_load(o["payload_json"], {}),
            }
            for o in outputs
        ],
    }

# =========================================================
# CONCEPT LOGIC
# =========================================================

def get_concept_templates(project_type: str) -> list[dict[str, Any]]:
    mapping = {
        "exhibition_booth": [
            {
                "title": "Premium Tech Minimal",
                "one_liner": "A sleek booth with sharp brand focus and clean premium lines.",
                "design_story": "Minimal architecture, controlled lighting, sharp product storytelling, and a confident premium presence.",
                "style_direction": "Clean, futuristic, premium, balanced branding",
                "layout_direction": "Front reception, central interaction, side product display, hidden storage",
                "materials": ["White laminate", "Backlit acrylic", "Black metal", "Wood texture accents"],
                "colors": ["#0B1020", "#4F7CFF", "#FFFFFF", "#A7B8FF"],
                "hero_visual": "Strong fascia branding with LED-lit rear feature wall",
            },
            {
                "title": "Immersive Brand Experience",
                "one_liner": "A bold experiential booth with strong visitor engagement moments.",
                "design_story": "Layered spatial depth, immersive LED content, selfie moments, and a high-energy branded journey.",
                "style_direction": "Immersive, bold, dynamic, attention-grabbing",
                "layout_direction": "Open entry, central demo island, immersive rear screen, social interaction zone",
                "materials": ["Printed fabric", "LED mesh", "Acrylic fins", "Matte vinyl"],
                "colors": ["#101828", "#00C2FF", "#7A5AF8", "#FFFFFF"],
                "hero_visual": "A dramatic hero entry with layered brand graphics and immersive light lines",
            },
            {
                "title": "Elegant Corporate Luxury",
                "one_liner": "A refined booth for premium brand perception and executive meetings.",
                "design_story": "Warm materials, restrained branding, sophisticated lighting, and a mature premium hospitality feel.",
                "style_direction": "Warm luxury, high-end, corporate, refined",
                "layout_direction": "Formal reception, lounge-style meeting zone, premium brand backdrop, subtle product display",
                "materials": ["Wood veneer", "Warm fabric", "Champagne metal", "Soft backlit panels"],
                "colors": ["#1B1B1B", "#D6B98C", "#F5F0E8", "#5E4B3C"],
                "hero_visual": "A warm branded backdrop with elegant hospitality-forward styling",
            },
        ],
        "event_stage": [
            {
                "title": "Bold LED Keynote Stage",
                "one_liner": "A modern keynote stage designed for launch impact and high visibility.",
                "design_story": "Large LED story wall, confident stage wings, branded content hierarchy, and strong audience focus.",
                "style_direction": "Modern launch, cinematic, high-tech, bold",
                "layout_direction": "Wide center stage, rear LED, clean side wings, clear audience sightlines",
                "materials": ["LED wall", "Matte stage deck", "Printed side wings", "Truss lighting"],
                "colors": ["#0B1020", "#4F7CFF", "#FFFFFF", "#00D1FF"],
                "hero_visual": "A wide LED stage with crisp front-facing brand presence and dramatic lighting",
            },
            {
                "title": "Layered Premium Ceremony Stage",
                "one_liner": "A premium layered stage with elegant depth and ceremony presence.",
                "design_story": "Depth through layered scenic elements, premium lighting, elegant symmetry, and a polished event identity.",
                "style_direction": "Premium, ceremonial, balanced, elegant",
                "layout_direction": "Central stage platform, layered backdrop, side decorative elements, formal access stairs",
                "materials": ["Printed scenic flats", "Fabric layers", "Warm light accents", "Gloss signage"],
                "colors": ["#1E1E1E", "#D4AF7A", "#FFF7ED", "#7C5C36"],
                "hero_visual": "A grand central stage with elegant layered scenic composition",
            },
            {
                "title": "Immersive Launch Arena",
                "one_liner": "An energetic stage concept built for product launches and media moments.",
                "design_story": "High-energy geometry, graphic motion feel, strong brand transitions, and media-friendly backdrops.",
                "style_direction": "Energetic, immersive, launch-focused, vibrant",
                "layout_direction": "Center action zone, segmented LED wall, side branding towers, dramatic entry path",
                "materials": ["LED panels", "Printed towers", "Floor vinyl", "Accent lighting"],
                "colors": ["#111827", "#FF5D5D", "#FFD166", "#FFFFFF"],
                "hero_visual": "A powerful launch stage with segmented lighting and media impact",
            },
        ],
        "decor_zone": [
            {
                "title": "Photo-Op Statement Zone",
                "one_liner": "A high-impact decor corner designed for social content and guest engagement.",
                "design_story": "A signature decor moment with bold focal design, layered elements, and photogenic brand integration.",
                "style_direction": "Photogenic, decorative, bold focal point",
                "layout_direction": "Hero centerpiece, side decorative framing, guest standing zone, logo touchpoints",
                "materials": ["Fabric drape", "Printed scenic panels", "Artificial florals", "Light accents"],
                "colors": ["#0B1020", "#E9D5FF", "#FFFFFF", "#7C3AED"],
                "hero_visual": "A decorative branded moment built for photography and sharing",
            },
            {
                "title": "Luxury Lounge Decor",
                "one_liner": "A premium decor zone with lounge mood and high-end hospitality character.",
                "design_story": "Warm textures, premium decor accents, guest comfort, and sophisticated brand integration.",
                "style_direction": "Luxury, warm, lounge, premium",
                "layout_direction": "Soft seating zone, decorative rear wall, side accents, premium entry feel",
                "materials": ["Velvet fabric", "Metal accents", "Wood finishes", "Ambient lighting"],
                "colors": ["#201A17", "#C8A97E", "#F7F0E6", "#6A4B35"],
                "hero_visual": "A warm, elegant decor environment with premium lounge styling",
            },
            {
                "title": "Interactive Brand Corner",
                "one_liner": "A decor-driven engagement corner blending atmosphere with brand discovery.",
                "design_story": "Decor details plus interaction points, creating a memorable micro-experience inside the event.",
                "style_direction": "Interactive, decorative, stylish, branded",
                "layout_direction": "Branded centerpiece, interaction counter, decorative side build-ups, shareable moments",
                "materials": ["Acrylic accents", "Graphic panels", "Decor props", "Spot lighting"],
                "colors": ["#111827", "#14B8A6", "#FFFFFF", "#99F6E4"],
                "hero_visual": "A decorative engagement corner with a stylish branded focal point",
            },
        ],
        "backdrop": [
            {
                "title": "Minimal Media Wall",
                "one_liner": "A clean backdrop for media, PR, and corporate photography.",
                "design_story": "Simple but premium brand hierarchy, easy photography, and polished event presentation.",
                "style_direction": "Minimal, branded, corporate, clean",
                "layout_direction": "Straight feature wall, clear standing zone, centered brand composition",
                "materials": ["Printed flex", "Fabric wall", "Simple frame structure", "Soft front lighting"],
                "colors": ["#0B1020", "#FFFFFF", "#4F7CFF", "#CBD5E1"],
                "hero_visual": "A sharp brand wall with clean alignment and elegant spacing",
            },
            {
                "title": "Premium Step-and-Repeat",
                "one_liner": "A structured premium media backdrop with sponsor and brand balance.",
                "design_story": "An orderly but elegant media wall with strong co-branding logic and premium detailing.",
                "style_direction": "Premium sponsor-friendly, balanced, formal",
                "layout_direction": "Wide wall, sponsor grid, event title center, press photography zone",
                "materials": ["Printed fabric", "Matte finish board", "Simple support frame", "Warm light wash"],
                "colors": ["#111827", "#D6B98C", "#FFFFFF", "#6B7280"],
                "hero_visual": "A formal media wall with premium sponsor presentation",
            },
            {
                "title": "Layered Story Backdrop",
                "one_liner": "A scenic branded backdrop with more depth and storytelling.",
                "design_story": "Layered scenic forms, dimensional graphics, and richer visual storytelling than a flat media wall.",
                "style_direction": "Scenic, layered, stylish, event-rich",
                "layout_direction": "Main backdrop wall, dimensional side returns, hero message center",
                "materials": ["Scenic printed panels", "Foam letters", "Accent lights", "Layered boards"],
                "colors": ["#1F2937", "#F59E0B", "#FFFFFF", "#FDE68A"],
                "hero_visual": "A layered event wall with depth and stronger design personality",
            },
        ],
        "experience_area": [
            {
                "title": "Demo Journey Zone",
                "one_liner": "A structured experience area guiding visitors through product discovery.",
                "design_story": "Progressive journey design with clear touchpoints, brand immersion, and useful visitor flow.",
                "style_direction": "Smart, guided, immersive, structured",
                "layout_direction": "Entry touchpoint, demo path, content node, final engagement zone",
                "materials": ["Graphic walls", "Demo counters", "LED content", "Vinyl floor path"],
                "colors": ["#0B1020", "#4F7CFF", "#FFFFFF", "#22D3EE"],
                "hero_visual": "A guided product discovery zone with modern branded structure",
            },
            {
                "title": "Immersive Discovery Tunnel",
                "one_liner": "A memorable branded environment that surrounds visitors with story and content.",
                "design_story": "Layered structure, strong atmosphere, immersive brand storytelling, and media-ready visuals.",
                "style_direction": "Immersive, atmospheric, futuristic, memorable",
                "layout_direction": "Immersive entry, tunnel experience, central reveal moment, exit engagement zone",
                "materials": ["Printed scenic walls", "Lighting strips", "LED inserts", "Reflective details"],
                "colors": ["#111827", "#A855F7", "#FFFFFF", "#C4B5FD"],
                "hero_visual": "A dramatic branded journey with immersive lighting and depth",
            },
            {
                "title": "Warm Networking Experience",
                "one_liner": "An experience-led area blending brand, interaction, and comfortable guest flow.",
                "design_story": "A softer branded environment where conversation, product awareness, and visual quality coexist.",
                "style_direction": "Warm, premium, social, hospitality-driven",
                "layout_direction": "Welcome point, lounge seating, product touchpoint, subtle branded photo moment",
                "materials": ["Wood textures", "Soft fabric", "Graphic panels", "Warm ambient lighting"],
                "colors": ["#201A17", "#C08457", "#F8FAFC", "#7C2D12"],
                "hero_visual": "A warm branded lounge-experience zone with premium social energy",
            },
        ],
    }
    return mapping.get(project_type, mapping["exhibition_booth"])

def get_reference_rows(project_id: str) -> list[sqlite3.Row]:
    project = fetch_project(project_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM reference_images WHERE project_id = ? ORDER BY created_at DESC", (project["id"],))
    rows = cur.fetchall()
    conn.close()
    return rows

def build_concept_from_template(project: sqlite3.Row, brief: dict[str, Any], template: dict[str, Any], concept_no: int) -> dict[str, Any]:
    brand = brief.get("brand_name", "Brand")
    audience = brief.get("target_audience", "target audience")
    goals = ", ".join(brief.get("goals", [])[:2]) or "brand visibility"
    style_words = ", ".join(brief.get("style_keywords", [])[:3]) or template["style_direction"]
    must_have = ", ".join(brief.get("must_have_zones", [])[:4]) or template["layout_direction"]

    return {
        "title": template["title"],
        "one_liner": f"{template['one_liner']} Best suited for {audience}.",
        "summary": f"{brand} concept focused on {goals} with a {style_words.lower()} visual language.",
        "design_story": f"{template['design_story']} This concept is shaped for {brand} and tailored to the project brief.",
        "style_direction": f"{template['style_direction']} | Brief influence: {style_words}",
        "layout_direction": f"{template['layout_direction']} | Must-have zones: {must_have}",
        "materials": template["materials"],
        "colors": template["colors"],
        "hero_visual": template["hero_visual"],
        "concept_no": concept_no,
    }

def generate_concept_board_svg(project: sqlite3.Row, concept: dict[str, Any], refs: list[sqlite3.Row]) -> str:
    width = 1600
    height = 900
    colors = concept["colors"][:4] if concept["colors"] else ["#0B1020", "#4F7CFF", "#FFFFFF", "#CBD5E1"]
    ref_boxes = []
    y_start = 500
    x_positions = [70, 415, 760, 1105]

    ref_labels = []
    if refs:
        for r in refs[:4]:
            ref_labels.append(r["label"] or Path(r["file_path"]).name)
    else:
        ref_labels = [
            "Suggested reference: lighting mood",
            "Suggested reference: material finish",
            "Suggested reference: branding surface",
            "Suggested reference: hero angle",
        ]

    for i, label in enumerate(ref_labels):
        ref_boxes.append(f"""
        <rect x="{x_positions[i]}" y="{y_start}" width="280" height="220" rx="18" fill="#111827" stroke="#334155" />
        <text x="{x_positions[i] + 20}" y="{y_start + 40}" fill="#E5E7EB" font-size="24" font-family="Arial" font-weight="700">Reference {i+1}</text>
        <text x="{x_positions[i] + 20}" y="{y_start + 82}" fill="#CBD5E1" font-size="18" font-family="Arial">{safe_text(label)}</text>
        <rect x="{x_positions[i] + 20}" y="{y_start + 110}" width="240" height="85" rx="12" fill="#1F2937" stroke="#475569" stroke-dasharray="8 6"/>
        <text x="{x_positions[i] + 42}" y="{y_start + 158}" fill="#94A3B8" font-size="22" font-family="Arial">image slot</text>
        """)

    palette = "".join(
        [f'<rect x="{70 + idx*80}" y="390" width="60" height="60" rx="10" fill="{safe_text(c)}" stroke="#ffffff"/>' for idx, c in enumerate(colors)]
    )

    materials = wrap_text(", ".join(concept["materials"]), 42)
    materials_svg = "".join(
        [f'<text x="760" y="{170 + i*34}" fill="#E2E8F0" font-size="24" font-family="Arial">{safe_text(line)}</text>' for i, line in enumerate(materials)]
    )

    summary_lines = wrap_text(concept["summary"], 55)
    summary_svg = "".join(
        [f'<text x="70" y="{180 + i*34}" fill="#CBD5E1" font-size="24" font-family="Arial">{safe_text(line)}</text>' for i, line in enumerate(summary_lines)]
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
    <defs>
      <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
        <stop offset="0%" stop-color="{safe_text(colors[0])}"/>
        <stop offset="100%" stop-color="#111827"/>
      </linearGradient>
    </defs>
    <rect width="{width}" height="{height}" fill="url(#bg)"/>
    <rect x="50" y="50" width="1500" height="800" rx="28" fill="rgba(15,23,42,0.72)" stroke="rgba(255,255,255,0.18)"/>
    <text x="70" y="105" fill="#FFFFFF" font-size="46" font-family="Arial" font-weight="700">{safe_text(concept['title'])}</text>
    <text x="70" y="145" fill="#93C5FD" font-size="26" font-family="Arial">{safe_text(project['name'])} • Concept {concept['concept_no']}</text>
    {summary_svg}
    <text x="760" y="120" fill="#FFFFFF" font-size="30" font-family="Arial" font-weight="700">Materials</text>
    {materials_svg}
    <text x="70" y="360" fill="#FFFFFF" font-size="30" font-family="Arial" font-weight="700">Color palette</text>
    {palette}
    <text x="760" y="390" fill="#FFFFFF" font-size="30" font-family="Arial" font-weight="700">Hero visual</text>
    <text x="760" y="430" fill="#CBD5E1" font-size="24" font-family="Arial">{safe_text(concept['hero_visual'])}</text>
    <text x="70" y="470" fill="#FFFFFF" font-size="30" font-family="Arial" font-weight="700">Reference images</text>
    {''.join(ref_boxes)}
    </svg>"""
    return svg

def clear_project_concepts(project_id: str) -> None:
    project = fetch_project(project_id)
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id FROM concepts WHERE project_id = ?", (project["id"],))
    concept_rows = cur.fetchall()
    for c in concept_rows:
        cur.execute("DELETE FROM concept_reference_links WHERE concept_id = ?", (c["id"],))

    cur.execute("DELETE FROM concepts WHERE project_id = ?", (project["id"],))
    cur.execute("DELETE FROM outputs WHERE project_id = ?", (project["id"],))
    cur.execute("UPDATE projects SET selected_concept_id = NULL, status = ?, updated_at = ? WHERE id = ?", ("brief_submitted", now_iso(), project["id"]))
    conn.commit()
    conn.close()

def save_concept_board(project_id: str, concept_id: str, svg_content: str) -> str:
    folder = CONCEPT_DIR / project_id / concept_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "board.svg"
    path.write_text(svg_content, encoding="utf-8")
    return str(path)

# =========================================================
# OUTPUT GENERATION
# =========================================================

def get_default_zones(project_type: str) -> list[str]:
    mapping = {
        "exhibition_booth": ["reception", "display", "meeting", "storage"],
        "event_stage": ["main_stage", "led_wall", "side_wings", "podium", "control"],
        "decor_zone": ["photo_op", "decor_feature", "lounge", "branding"],
        "backdrop": ["backdrop_wall", "photo_lane", "sponsor_zone"],
        "experience_area": ["entry", "demo", "interaction", "lounge", "brand_moment"],
    }
    return mapping.get(project_type, ["reception", "display", "meeting", "storage"])

def build_layout_payload(project: sqlite3.Row, brief: dict[str, Any], concept: sqlite3.Row) -> dict[str, Any]:
    width = brief["width_mm"]
    depth = brief["depth_mm"]
    height = brief["height_mm"]
    zones = brief.get("must_have_zones") or get_default_zones(project["project_type"])

    layout_items = []
    margin = 200

    if project["project_type"] == "event_stage":
        layout_items = [
            {"name": "main_stage", "x": 200, "y": 200, "w": width - 400, "d": max(1200, depth // 2)},
            {"name": "led_wall", "x": 250, "y": 240, "w": width - 500, "d": 120},
            {"name": "side_wings", "x": 200, "y": max(1500, depth // 2 + 150), "w": width - 400, "d": max(700, depth // 3)},
        ]
    elif project["project_type"] == "decor_zone":
        layout_items = [
            {"name": "photo_op", "x": 200, "y": 200, "w": width - 400, "d": max(1000, depth // 2)},
            {"name": "lounge", "x": 300, "y": max(1400, depth // 2 + 100), "w": max(1500, width // 2), "d": max(800, depth // 3)},
            {"name": "branding", "x": width - max(1400, width // 3) - 250, "y": max(1400, depth // 2 + 100), "w": max(1400, width // 3), "d": max(800, depth // 3)},
        ]
    else:
        layout_items = [
            {"name": zones[0] if len(zones) > 0 else "reception", "x": margin, "y": margin, "w": min(1800, width // 3), "d": 700},
            {"name": zones[1] if len(zones) > 1 else "display", "x": margin, "y": max(1300, depth - 1200), "w": width - 2 * margin, "d": 900},
            {"name": zones[2] if len(zones) > 2 else "meeting", "x": max(2400, width // 2 - 900), "y": max(1100, depth // 3), "w": 1800, "d": 1600},
            {"name": zones[3] if len(zones) > 3 else "storage", "x": width - 1500 - margin, "y": margin, "w": 1500, "d": max(1200, depth // 3)},
        ]

    return {
        "project_type": project["project_type"],
        "width_mm": width,
        "depth_mm": depth,
        "height_mm": height,
        "zones": layout_items,
        "concept_title": concept["title"],
    }

def generate_layout_svg(project: sqlite3.Row, brief: dict[str, Any], concept: sqlite3.Row, payload: dict[str, Any]) -> str:
    width_px = 1400
    height_px = 900
    real_w = payload["width_mm"]
    real_d = payload["depth_mm"]
    sx = width_px / max(real_w, 1)
    sy = 650 / max(real_d, 1)

    rects = []
    for z in payload["zones"]:
        x = 80 + z["x"] * sx
        y = 140 + z["y"] * sy
        w = z["w"] * sx
        h = z["d"] * sy
        rects.append(f"""
        <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#1E293B" stroke="#60A5FA" stroke-width="3"/>
        <text x="{x + 18}" y="{y + 34}" fill="#E5E7EB" font-size="22" font-family="Arial">{safe_text(z['name'])}</text>
        """)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" viewBox="0 0 {width_px} {height_px}">
    <rect width="{width_px}" height="{height_px}" fill="#0B1020"/>
    <text x="70" y="70" fill="#FFFFFF" font-size="40" font-family="Arial" font-weight="700">Layout Plan</text>
    <text x="70" y="110" fill="#93C5FD" font-size="24" font-family="Arial">{safe_text(project['name'])} • {safe_text(concept['title'])}</text>
    <rect x="80" y="140" width="{real_w * sx}" height="{real_d * sy}" fill="none" stroke="#CBD5E1" stroke-width="4"/>
    {''.join(rects)}
    <text x="80" y="{830}" fill="#CBD5E1" font-size="20" font-family="Arial">Overall size: {real_w}mm × {real_d}mm × {brief['height_mm']}mm</text>
    </svg>"""

def generate_creative_svg(project: sqlite3.Row, brief: dict[str, Any], concept: sqlite3.Row) -> str:
    colors = json_load(concept["colors_json"], ["#0B1020", "#4F7CFF", "#FFFFFF"])
    messages = brief.get("key_messages", [])[:3]
    msg_svg = "".join(
        [f'<text x="90" y="{350 + i*42}" fill="#CBD5E1" font-size="28" font-family="Arial">• {safe_text(m)}</text>' for i, m in enumerate(messages)]
    ) or '<text x="90" y="350" fill="#CBD5E1" font-size="28" font-family="Arial">• Brand message space</text>'
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
    <rect width="1600" height="900" fill="{safe_text(colors[0])}"/>
    <circle cx="1340" cy="180" r="220" fill="{safe_text(colors[1])}" opacity="0.18"/>
    <rect x="70" y="70" width="1460" height="760" rx="28" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.18)"/>
    <text x="90" y="140" fill="#FFFFFF" font-size="54" font-family="Arial" font-weight="700">{safe_text(brief['brand_name'])}</text>
    <text x="90" y="200" fill="#93C5FD" font-size="34" font-family="Arial">{safe_text(concept['title'])}</text>
    <text x="90" y="280" fill="#FFFFFF" font-size="72" font-family="Arial" font-weight="700">{safe_text(project['name'])}</text>
    {msg_svg}
    <rect x="90" y="700" width="340" height="76" rx="16" fill="{safe_text(colors[1])}"/>
    <text x="130" y="748" fill="#FFFFFF" font-size="32" font-family="Arial" font-weight="700">Brand Call To Action</text>
    </svg>"""

def generate_preview_3d_svg(project: sqlite3.Row, brief: dict[str, Any], concept: sqlite3.Row) -> str:
    colors = json_load(concept["colors_json"], ["#0B1020", "#4F7CFF", "#FFFFFF"])
    title = concept["title"]

    if project["project_type"] == "event_stage":
        shape = f"""
        <polygon points="220,650 1200,650 1100,430 320,430" fill="#182234" stroke="{safe_text(colors[1])}" stroke-width="4"/>
        <rect x="420" y="220" width="520" height="180" rx="10" fill="{safe_text(colors[1])}" opacity="0.7"/>
        <rect x="280" y="280" width="120" height="220" fill="#1F2937"/>
        <rect x="960" y="280" width="120" height="220" fill="#1F2937"/>
        """
    elif project["project_type"] == "decor_zone":
        shape = f"""
        <ellipse cx="700" cy="650" rx="380" ry="90" fill="#111827"/>
        <rect x="360" y="260" width="680" height="300" rx="34" fill="#182234" stroke="{safe_text(colors[1])}" stroke-width="4"/>
        <circle cx="700" cy="360" r="110" fill="{safe_text(colors[1])}" opacity="0.35"/>
        """
    else:
        shape = f"""
        <polygon points="320,650 1020,650 1180,530 470,530" fill="#182234" stroke="{safe_text(colors[1])}" stroke-width="4"/>
        <polygon points="470,530 1180,530 1180,250 470,250" fill="#1E293B" stroke="#93C5FD" stroke-width="4"/>
        <polygon points="320,650 470,530 470,250 320,360" fill="#0F172A" stroke="#93C5FD" stroke-width="4"/>
        """

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">
    <rect width="1400" height="900" fill="#0B1020"/>
    <text x="70" y="90" fill="#FFFFFF" font-size="46" font-family="Arial" font-weight="700">3D Preview Direction</text>
    <text x="70" y="135" fill="#93C5FD" font-size="28" font-family="Arial">{safe_text(title)} • {safe_text(project['project_type'])}</text>
    {shape}
    <text x="70" y="820" fill="#CBD5E1" font-size="22" font-family="Arial">This version creates a preview direction sheet. Photoreal 3D rendering can be connected next.</text>
    </svg>"""

def generate_drawing_svg(project: sqlite3.Row, brief: dict[str, Any], concept: sqlite3.Row, layout_payload: dict[str, Any]) -> str:
    lines = []
    y = 170
    details = [
        f"Project: {project['name']}",
        f"Project Type: {project['project_type']}",
        f"Selected Concept: {concept['title']}",
        f"Overall Size: {brief['width_mm']} x {brief['depth_mm']} x {brief['height_mm']} mm",
        f"Budget Level: {brief.get('budget_level', '')}",
        f"Style: {concept['style_direction']}",
    ]
    for d in details:
        lines.append(f'<text x="80" y="{y}" fill="#E5E7EB" font-size="24" font-family="Arial">{safe_text(d)}</text>')
        y += 40

    zone_lines = []
    y2 = 450
    for z in layout_payload["zones"]:
        zone_lines.append(
            f'<text x="80" y="{y2}" fill="#CBD5E1" font-size="22" font-family="Arial">{safe_text(z["name"])}: {z["w"]} x {z["d"]} mm at ({z["x"]}, {z["y"]})</text>'
        )
        y2 += 34

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">
    <rect width="1400" height="900" fill="#F8FAFC"/>
    <rect x="40" y="40" width="1320" height="820" fill="white" stroke="#0F172A" stroke-width="3"/>
    <text x="80" y="100" fill="#111827" font-size="42" font-family="Arial" font-weight="700">Production Drawing Summary</text>
    {''.join(lines)}
    <text x="80" y="400" fill="#111827" font-size="30" font-family="Arial" font-weight="700">Zone Schedule</text>
    {''.join(zone_lines)}
    <text x="80" y="820" fill="#334155" font-size="20" font-family="Arial">Generated from selected concept for production planning reference.</text>
    </svg>"""

def save_output_file(project_id: str, concept_id: str, filename: str, content: str) -> str:
    folder = OUTPUT_DIR / project_id / concept_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_text(content, encoding="utf-8")
    return str(path)

def create_output_record(project_id: str, concept_id: str, output_type: str, title: str, file_path: str, payload: dict[str, Any]) -> str:
    output_id = str(uuid.uuid4())
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO outputs (id, project_id, concept_id, output_type, title, file_path, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (output_id, project_id, concept_id, output_type, title, file_path, json.dumps(payload), now_iso()),
    )
    conn.commit()
    conn.close()
    return output_id

# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def root():
    return {
        "message": "ExpoAI concept-first backend is running",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
def health():
    return {"status": "ok", "app": APP_NAME, "db_path": str(DB_PATH)}

@app.get("/v1/debug/projects/{project_id}")
def debug_project(project_id: str):
    pid = normalize_id(project_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return {
        "requested": project_id,
        "normalized": pid,
        "all_project_ids": [r["id"] for r in rows],
    }

@app.post("/v1/projects")
def create_project(payload: ProjectCreate):
    project_id = str(uuid.uuid4())
    now = now_iso()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO projects (id, name, project_type, status, selected_concept_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (project_id, payload.name, payload.project_type, "draft", None, now, now),
    )
    conn.commit()
    conn.close()

    return {
        "message": "Project created successfully",
        "project_id": project_id,
        "project_type": payload.project_type,
    }

@app.get("/v1/projects")
def list_projects():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows]}

@app.get("/v1/projects/{project_id}")
def get_project(project_id: str):
    return get_project_full(project_id)

@app.post("/v1/projects/{project_id}/brief")
def submit_brief(project_id: str, payload: BriefSubmit):
    project = fetch_project(project_id)
    brief_id = str(uuid.uuid4())
    now = now_iso()

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id FROM briefs WHERE project_id = ?", (project["id"],))
    existing = cur.fetchone()

    if existing:
        cur.execute(
            "UPDATE briefs SET brief_json = ?, updated_at = ? WHERE project_id = ?",
            (json.dumps(payload.model_dump()), now, project["id"]),
        )
    else:
        cur.execute(
            """
            INSERT INTO briefs (id, project_id, brief_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (brief_id, project["id"], json.dumps(payload.model_dump()), now, now),
        )

    cur.execute(
        "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
        ("brief_submitted", now, project["id"]),
    )
    conn.commit()
    conn.close()

    return {
        "message": "Brief submitted successfully",
        "project_id": project["id"],
        "brief": payload.model_dump(),
    }

@app.post("/v1/projects/{project_id}/reference-images")
async def upload_reference_image(
    project_id: str,
    file: UploadFile = File(...),
    label: str = Form("Reference Image"),
    notes: str = Form(""),
):
    project = fetch_project(project_id)

    folder = REFERENCE_DIR / project["id"]
    folder.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or ""
    filename = f"{uuid.uuid4()}{ext}"
    path = folder / filename

    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)

    ref_id = str(uuid.uuid4())

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO reference_images (id, project_id, label, notes, file_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (ref_id, project["id"], label, notes, str(path), now_iso()),
    )
    conn.commit()
    conn.close()

    return {
        "message": "Reference image uploaded successfully",
        "reference_image_id": ref_id,
        "project_id": project["id"],
        "label": label,
        "file_download_url": f"/v1/reference-images/{ref_id}/file",
    }

@app.get("/v1/projects/{project_id}/reference-images")
def list_reference_images(project_id: str):
    refs = get_reference_rows(project_id)
    return {
        "items": [
            {
                **dict(r),
                "file_download_url": f"/v1/reference-images/{r['id']}/file",
            }
            for r in refs
        ]
    }

@app.get("/v1/reference-images/{reference_id}/file")
def download_reference_image(reference_id: str):
    ref = fetch_reference_image(reference_id)
    if not os.path.exists(ref["file_path"]):
        raise HTTPException(status_code=404, detail="Reference file not found")
    return FileResponse(ref["file_path"], filename=Path(ref["file_path"]).name)

@app.post("/v1/projects/{project_id}/concepts/generate")
def generate_concepts(project_id: str):
    project = fetch_project(project_id)
    brief_row = fetch_brief(project["id"])
    brief = json_load(brief_row["brief_json"], {})
    refs = get_reference_rows(project["id"])

    clear_project_concepts(project["id"])
    templates = get_concept_templates(project["project_type"])

    created = []
    conn = get_conn()
    cur = conn.cursor()

    for idx, template in enumerate(templates, start=1):
        concept = build_concept_from_template(project, brief, template, idx)
        concept_id = str(uuid.uuid4())
        board_svg = generate_concept_board_svg(project, concept, refs)
        board_path = save_concept_board(project["id"], concept_id, board_svg)
        now = now_iso()

        cur.execute(
            """
            INSERT INTO concepts (
                id, project_id, concept_no, title, one_liner, summary, design_story,
                style_direction, layout_direction, materials_json, colors_json,
                hero_visual, board_path, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                concept_id,
                project["id"],
                idx,
                concept["title"],
                concept["one_liner"],
                concept["summary"],
                concept["design_story"],
                concept["style_direction"],
                concept["layout_direction"],
                json.dumps(concept["materials"]),
                json.dumps(concept["colors"]),
                concept["hero_visual"],
                board_path,
                "draft",
                now,
                now,
            ),
        )

        for sort_order, ref in enumerate(refs[:4], start=1):
            cur.execute(
                """
                INSERT INTO concept_reference_links (id, concept_id, reference_image_id, caption, sort_order)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    concept_id,
                    ref["id"],
                    ref["label"] or f"Reference {sort_order}",
                    sort_order,
                ),
            )

        created.append({
            "id": concept_id,
            "concept_no": idx,
            "title": concept["title"],
            "one_liner": concept["one_liner"],
            "summary": concept["summary"],
            "board_url": f"/v1/concepts/{concept_id}/board",
        })

    cur.execute(
        "UPDATE projects SET status = ?, updated_at = ?, selected_concept_id = NULL WHERE id = ?",
        ("concepts_generated", now_iso(), project["id"]),
    )
    conn.commit()
    conn.close()

    return {
        "message": "3 concepts generated successfully",
        "project_id": project["id"],
        "items": created,
    }

@app.get("/v1/projects/{project_id}/concepts")
def list_concepts(project_id: str):
    project = fetch_project(project_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM concepts WHERE project_id = ? ORDER BY concept_no ASC", (project["id"],))
    rows = cur.fetchall()

    result = []
    for c in rows:
        cur.execute("""
            SELECT l.*, r.label, r.notes, r.file_path
            FROM concept_reference_links l
            JOIN reference_images r ON r.id = l.reference_image_id
            WHERE l.concept_id = ?
            ORDER BY l.sort_order ASC
        """, (c["id"],))
        linked_refs = cur.fetchall()

        result.append({
            **dict(c),
            "materials": json_load(c["materials_json"], []),
            "colors": json_load(c["colors_json"], []),
            "board_url": f"/v1/concepts/{c['id']}/board",
            "reference_images": [
                {
                    "reference_image_id": r["reference_image_id"],
                    "label": r["label"],
                    "notes": r["notes"],
                    "file_download_url": f"/v1/reference-images/{r['reference_image_id']}/file",
                }
                for r in linked_refs
            ],
        })

    conn.close()
    return {"items": result}

@app.get("/v1/concepts/{concept_id}")
def get_concept(concept_id: str):
    concept = fetch_concept(concept_id)
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT l.*, r.label, r.notes
        FROM concept_reference_links l
        JOIN reference_images r ON r.id = l.reference_image_id
        WHERE l.concept_id = ?
        ORDER BY l.sort_order ASC
    """, (concept["id"],))
    refs = cur.fetchall()
    conn.close()

    return {
        **dict(concept),
        "materials": json_load(concept["materials_json"], []),
        "colors": json_load(concept["colors_json"], []),
        "board_url": f"/v1/concepts/{concept['id']}/board",
        "reference_images": [
            {
                "reference_image_id": r["reference_image_id"],
                "label": r["label"],
                "notes": r["notes"],
                "file_download_url": f"/v1/reference-images/{r['reference_image_id']}/file",
            }
            for r in refs
        ],
    }

@app.get("/v1/concepts/{concept_id}/board")
def get_concept_board(concept_id: str):
    concept = fetch_concept(concept_id)
    if not os.path.exists(concept["board_path"]):
        raise HTTPException(status_code=404, detail="Concept board file not found")
    return FileResponse(concept["board_path"], media_type="image/svg+xml", filename=Path(concept["board_path"]).name)

@app.patch("/v1/concepts/{concept_id}")
def update_concept(concept_id: str, payload: ConceptUpdate):
    concept = fetch_concept(concept_id)
    project = fetch_project(concept["project_id"])
    brief = json_load(fetch_brief(project["id"])["brief_json"], {})
    refs = get_reference_rows(project["id"])

    updated = {
        "title": payload.title if payload.title is not None else concept["title"],
        "one_liner": payload.one_liner if payload.one_liner is not None else concept["one_liner"],
        "summary": payload.summary if payload.summary is not None else concept["summary"],
        "design_story": payload.design_story if payload.design_story is not None else concept["design_story"],
        "style_direction": payload.style_direction if payload.style_direction is not None else concept["style_direction"],
        "layout_direction": payload.layout_direction if payload.layout_direction is not None else concept["layout_direction"],
        "materials": payload.materials if payload.materials is not None else json_load(concept["materials_json"], []),
        "colors": payload.colors if payload.colors is not None else json_load(concept["colors_json"], []),
        "hero_visual": payload.hero_visual if payload.hero_visual is not None else concept["hero_visual"],
        "concept_no": concept["concept_no"],
    }

    board_svg = generate_concept_board_svg(project, updated, refs)
    board_path = save_concept_board(project["id"], concept["id"], board_svg)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE concepts
        SET title = ?, one_liner = ?, summary = ?, design_story = ?, style_direction = ?,
            layout_direction = ?, materials_json = ?, colors_json = ?, hero_visual = ?,
            board_path = ?, status = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            updated["title"],
            updated["one_liner"],
            updated["summary"],
            updated["design_story"],
            updated["style_direction"],
            updated["layout_direction"],
            json.dumps(updated["materials"]),
            json.dumps(updated["colors"]),
            updated["hero_visual"],
            board_path,
            "edited",
            now_iso(),
            concept["id"],
        ),
    )
    conn.commit()
    conn.close()

    return {"message": "Concept updated successfully", "concept_id": concept["id"]}

@app.post("/v1/concepts/{concept_id}/select")
def select_concept(concept_id: str):
    concept = fetch_concept(concept_id)
    project = fetch_project(concept["project_id"])

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE concepts SET status = 'rejected', updated_at = ? WHERE project_id = ?", (now_iso(), project["id"]))
    cur.execute("UPDATE concepts SET status = 'selected', updated_at = ? WHERE id = ?", (now_iso(), concept["id"]))
    cur.execute(
        "UPDATE projects SET selected_concept_id = ?, status = ?, updated_at = ? WHERE id = ?",
        (concept["id"], "concept_selected", now_iso(), project["id"]),
    )
    conn.commit()
    conn.close()

    return {
        "message": "Concept selected successfully",
        "project_id": project["id"],
        "selected_concept_id": concept["id"],
        "selected_concept_title": concept["title"],
    }

@app.post("/v1/projects/{project_id}/generate-all-from-selected-concept")
def generate_all_from_selected_concept(project_id: str):
    project = fetch_project(project_id)
    brief = json_load(fetch_brief(project["id"])["brief_json"], {})
    selected_id = project["selected_concept_id"]

    if not selected_id:
        raise HTTPException(status_code=400, detail="No selected concept for this project")

    concept = fetch_concept(selected_id)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM outputs WHERE project_id = ?", (project["id"],))
    conn.commit()
    conn.close()

    layout_payload = build_layout_payload(project, brief, concept)

    layout_svg = generate_layout_svg(project, brief, concept, layout_payload)
    creative_svg = generate_creative_svg(project, brief, concept)
    preview_svg = generate_preview_3d_svg(project, brief, concept)
    drawing_svg = generate_drawing_svg(project, brief, concept, layout_payload)

    layout_path = save_output_file(project["id"], concept["id"], "layout_plan.svg", layout_svg)
    creative_path = save_output_file(project["id"], concept["id"], "creative_2d.svg", creative_svg)
    preview_path = save_output_file(project["id"], concept["id"], "preview_3d.svg", preview_svg)
    drawing_path = save_output_file(project["id"], concept["id"], "production_drawing.svg", drawing_svg)

    out1 = create_output_record(project["id"], concept["id"], "layout_plan", "Layout Plan", layout_path, layout_payload)
    out2 = create_output_record(project["id"], concept["id"], "creative_2d", "2D Creative", creative_path, {"concept_title": concept["title"]})
    out3 = create_output_record(project["id"], concept["id"], "preview_3d", "3D Preview Direction", preview_path, {"concept_title": concept["title"]})
    out4 = create_output_record(project["id"], concept["id"], "production_drawing", "Production Drawing Summary", drawing_path, layout_payload)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
        ("final_package_ready", now_iso(), project["id"]),
    )
    conn.commit()
    conn.close()

    return {
        "message": "Final outputs generated from selected concept",
        "project_id": project["id"],
        "selected_concept_id": concept["id"],
        "outputs": [
            {"output_id": out1, "type": "layout_plan", "file_url": f"/v1/outputs/{out1}/file"},
            {"output_id": out2, "type": "creative_2d", "file_url": f"/v1/outputs/{out2}/file"},
            {"output_id": out3, "type": "preview_3d", "file_url": f"/v1/outputs/{out3}/file"},
            {"output_id": out4, "type": "production_drawing", "file_url": f"/v1/outputs/{out4}/file"},
        ],
    }

@app.get("/v1/projects/{project_id}/outputs")
def list_project_outputs(project_id: str):
    project = fetch_project(project_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM outputs WHERE project_id = ? ORDER BY created_at DESC", (project["id"],))
    rows = cur.fetchall()
    conn.close()

    return {
        "items": [
            {
                **dict(r),
                "payload": json_load(r["payload_json"], {}),
                "file_url": f"/v1/outputs/{r['id']}/file",
            }
            for r in rows
        ]
    }

@app.get("/v1/outputs/{output_id}/file")
def download_output_file(output_id: str):
    output = fetch_output(output_id)
    if not os.path.exists(output["file_path"]):
        raise HTTPException(status_code=404, detail="Output file not found")

    suffix = Path(output["file_path"]).suffix.lower()
    media_type = "application/octet-stream"
    if suffix == ".svg":
        media_type = "image/svg+xml"
    elif suffix == ".json":
        media_type = "application/json"

    return FileResponse(output["file_path"], media_type=media_type, filename=Path(output["file_path"]).name)

# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
