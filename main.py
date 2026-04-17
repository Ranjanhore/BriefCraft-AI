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
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

# =========================================================
# BASIC CONFIG
# =========================================================

APP_NAME = "ExpoAI Single File Backend"
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "expoai.db"

STORAGE_DIR = BASE_DIR / "storage"
BRAND_ASSETS_DIR = STORAGE_DIR / "brand_assets"
ARTWORK_DIR = STORAGE_DIR / "artwork"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
BRAND_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
ARTWORK_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_NAME, version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DATABASE
# =========================================================

def now_iso() -> str:
    return datetime.utcnow().isoformat()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS brands (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        lock_mode TEXT NOT NULL DEFAULT 'controlled',
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        project_type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'draft',
        brand_id TEXT,
        booth_json TEXT NOT NULL,
        brief_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS brand_assets (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        brand_id TEXT NOT NULL,
        asset_type TEXT NOT NULL,
        file_url TEXT NOT NULL,
        meta_json TEXT,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS zones (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        zone_type TEXT NOT NULL,
        x_mm INTEGER NOT NULL,
        y_mm INTEGER NOT NULL,
        w_mm INTEGER NOT NULL,
        d_mm INTEGER NOT NULL,
        h_mm INTEGER,
        rules_json TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS panels (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        panel_code TEXT NOT NULL,
        panel_type TEXT NOT NULL,
        surface_ref TEXT NOT NULL,
        width_mm INTEGER NOT NULL,
        height_mm INTEGER NOT NULL,
        bleed_mm INTEGER NOT NULL DEFAULT 10,
        safe_margin_mm INTEGER NOT NULL DEFAULT 20,
        material TEXT,
        print_mode TEXT NOT NULL DEFAULT 'print',
        meta_json TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS panel_artworks (
        id TEXT PRIMARY KEY,
        panel_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        version_no INTEGER NOT NULL,
        headline TEXT,
        subheadline TEXT,
        cta TEXT,
        qr_url TEXT,
        style_hint TEXT,
        bg_color TEXT,
        accent_color TEXT,
        text_color TEXT,
        svg_path TEXT NOT NULL,
        json_path TEXT NOT NULL,
        preview_path TEXT NOT NULL,
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
# Pydantic MODELS
# =========================================================

class BoothInput(BaseModel):
    width_mm: int = Field(..., gt=0)
    depth_mm: int = Field(..., gt=0)
    height_mm: int = Field(..., gt=0)
    booth_type: str = "peninsula"
    open_sides: int = Field(3, ge=1, le=3)
    style: str = "premium modern tech"


class ProjectCreate(BaseModel):
    name: str
    project_type: str = "stall"
    booth: BoothInput
    brief: dict[str, Any] | None = None


class BrandCreate(BaseModel):
    name: str
    lock_mode: str = "controlled"
    project_id: Optional[str] = None


class LayoutGenerateRequest(BaseModel):
    zones: list[str] = ["reception", "meeting", "storage", "display"]
    screens: int = 1
    storage_required: bool = True


class ArtworkGenerateRequest(BaseModel):
    headline: str = "Build Smarter AI Workflows"
    subheadline: str = "Enterprise AI platform for modern teams"
    cta: str = "Scan to book a demo"
    qr_url: str = "https://example.com/demo"
    style_hint: str = "premium tech"
    bg_color: str = "#0B1020"
    accent_color: str = "#4F7CFF"
    text_color: str = "#FFFFFF"


# =========================================================
# HELPERS
# =========================================================

def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any]:
    return dict(row) if row else {}


def fetch_project(project_id: str) -> sqlite3.Row:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return row


def fetch_brand(brand_id: str) -> sqlite3.Row:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM brands WHERE id = ?", (brand_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Brand not found")
    return row


def fetch_panel(panel_id: str) -> sqlite3.Row:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM panels WHERE id = ?", (panel_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Panel not found")
    return row


def ensure_project_brand(project_id: str) -> str:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    project = cur.fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")

    if project["brand_id"]:
        brand_id = project["brand_id"]
        conn.close()
        return brand_id

    brand_id = str(uuid.uuid4())
    brand_name = f'{project["name"]} Brand'
    created_at = now_iso()

    cur.execute(
        "INSERT INTO brands (id, name, lock_mode, created_at) VALUES (?, ?, ?, ?)",
        (brand_id, brand_name, "controlled", created_at),
    )
    cur.execute(
        "UPDATE projects SET brand_id = ?, updated_at = ? WHERE id = ?",
        (brand_id, now_iso(), project_id),
    )
    conn.commit()
    conn.close()
    return brand_id


def clear_old_layout(project_id: str) -> None:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id FROM panels WHERE project_id = ?", (project_id,))
    panel_rows = cur.fetchall()
    panel_ids = [r["id"] for r in panel_rows]

    for panel_id in panel_ids:
        cur.execute("DELETE FROM panel_artworks WHERE panel_id = ?", (panel_id,))

    cur.execute("DELETE FROM zones WHERE project_id = ?", (project_id,))
    cur.execute("DELETE FROM panels WHERE project_id = ?", (project_id,))
    conn.commit()
    conn.close()


def generate_layout_data(booth: dict[str, Any], req: LayoutGenerateRequest) -> dict[str, Any]:
    width = booth["width_mm"]
    depth = booth["depth_mm"]
    height = booth["height_mm"]

    zones: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []

    margin = 200
    reception_w = min(1800, max(1200, width // 4))
    reception_d = 700

    if "reception" in req.zones:
        zones.append({
            "zone_type": "reception",
            "x_mm": margin,
            "y_mm": margin,
            "w_mm": reception_w,
            "d_mm": reception_d,
            "h_mm": 1100,
            "rules": {"facing": "front_open_side"},
        })

    if "meeting" in req.zones:
        zones.append({
            "zone_type": "meeting",
            "x_mm": max(margin, width // 2 - 900),
            "y_mm": max(1200, depth // 3),
            "w_mm": 1800,
            "d_mm": 1800,
            "h_mm": 0,
            "rules": {"chairs": 4},
        })

    if req.storage_required or "storage" in req.zones:
        storage_w = min(1600, max(1200, width // 4))
        storage_d = min(1800, max(1400, depth // 3))
        zones.append({
            "zone_type": "storage",
            "x_mm": max(margin, width - storage_w - margin),
            "y_mm": margin,
            "w_mm": storage_w,
            "d_mm": storage_d,
            "h_mm": height,
            "rules": {"closed": True},
        })

    if "display" in req.zones:
        zones.append({
            "zone_type": "display",
            "x_mm": margin,
            "y_mm": max(1500, depth - 1400),
            "w_mm": max(2000, width - 2 * margin),
            "d_mm": 900,
            "h_mm": 2400,
            "rules": {"screens": req.screens},
        })

    fascia_h = 500
    backwall_h = min(3000, max(1800, height - 200))
    backwall_w = max(2000, width - 60)
    sidewall_d = max(1500, depth - 60)

    panels.append({
        "panel_code": "FAS-01",
        "panel_type": "fascia_front",
        "surface_ref": "/Booth/Fascia/Front",
        "width_mm": width,
        "height_mm": fascia_h,
        "bleed_mm": 10,
        "safe_margin_mm": 20,
        "material": "vinyl",
        "print_mode": "print",
        "meta": {"position": "front"},
    })

    panels.append({
        "panel_code": "BKW-01",
        "panel_type": "backwall_center",
        "surface_ref": "/Booth/Backwall/Center",
        "width_mm": backwall_w,
        "height_mm": backwall_h,
        "bleed_mm": 10,
        "safe_margin_mm": 20,
        "material": "fabric",
        "print_mode": "print",
        "meta": {"position": "rear"},
    })

    if booth.get("open_sides", 3) <= 2:
        panels.append({
            "panel_code": "SDW-01",
            "panel_type": "sidewall_left",
            "surface_ref": "/Booth/Sidewall/Left",
            "width_mm": sidewall_d,
            "height_mm": backwall_h,
            "bleed_mm": 10,
            "safe_margin_mm": 20,
            "material": "fabric",
            "print_mode": "print",
            "meta": {"position": "left"},
        })

    return {"zones": zones, "panels": panels}


def save_layout(project_id: str, layout: dict[str, Any]) -> None:
    conn = get_conn()
    cur = conn.cursor()

    for zone in layout["zones"]:
        cur.execute(
            """
            INSERT INTO zones (
                id, project_id, zone_type, x_mm, y_mm, w_mm, d_mm, h_mm, rules_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                project_id,
                zone["zone_type"],
                zone["x_mm"],
                zone["y_mm"],
                zone["w_mm"],
                zone["d_mm"],
                zone["h_mm"],
                json.dumps(zone.get("rules", {})),
            ),
        )

    for panel in layout["panels"]:
        cur.execute(
            """
            INSERT INTO panels (
                id, project_id, panel_code, panel_type, surface_ref,
                width_mm, height_mm, bleed_mm, safe_margin_mm,
                material, print_mode, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                project_id,
                panel["panel_code"],
                panel["panel_type"],
                panel["surface_ref"],
                panel["width_mm"],
                panel["height_mm"],
                panel["bleed_mm"],
                panel["safe_margin_mm"],
                panel.get("material"),
                panel.get("print_mode", "print"),
                json.dumps(panel.get("meta", {})),
            ),
        )

    conn.commit()
    conn.close()


def get_project_full(project_id: str) -> dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    project = cur.fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")

    cur.execute("SELECT * FROM zones WHERE project_id = ?", (project_id,))
    zones = cur.fetchall()

    cur.execute("SELECT * FROM panels WHERE project_id = ?", (project_id,))
    panels = cur.fetchall()

    cur.execute("SELECT * FROM brand_assets WHERE project_id = ?", (project_id,))
    assets = cur.fetchall()

    cur.execute("""
        SELECT * FROM panel_artworks
        WHERE project_id = ?
        ORDER BY created_at DESC
    """, (project_id,))
    artworks = cur.fetchall()

    brand = None
    if project["brand_id"]:
        cur.execute("SELECT * FROM brands WHERE id = ?", (project["brand_id"],))
        brand = cur.fetchone()

    conn.close()

    return {
        "project": {
            "id": project["id"],
            "name": project["name"],
            "project_type": project["project_type"],
            "status": project["status"],
            "brand_id": project["brand_id"],
            "booth": json.loads(project["booth_json"]),
            "brief": json.loads(project["brief_json"]) if project["brief_json"] else None,
            "created_at": project["created_at"],
            "updated_at": project["updated_at"],
        },
        "brand": row_to_dict(brand) if brand else None,
        "brand_assets": [
            {
                **dict(a),
                "meta": json.loads(a["meta_json"]) if a["meta_json"] else None,
            }
            for a in assets
        ],
        "zones": [
            {
                **dict(z),
                "rules": json.loads(z["rules_json"]) if z["rules_json"] else {},
            }
            for z in zones
        ],
        "panels": [
            {
                **dict(p),
                "meta": json.loads(p["meta_json"]) if p["meta_json"] else {},
            }
            for p in panels
        ],
        "artworks": [
            {
                **dict(a),
                "payload": json.loads(a["payload_json"]) if a["payload_json"] else {},
            }
            for a in artworks
        ],
    }


def get_latest_brand_asset_for_project(project_id: str, asset_type: str = "logo") -> Optional[sqlite3.Row]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM brand_assets
        WHERE project_id = ? AND asset_type = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (project_id, asset_type),
    )
    row = cur.fetchone()
    conn.close()
    return row


def next_artwork_version(panel_id: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT COALESCE(MAX(version_no), 0) AS max_version FROM panel_artworks WHERE panel_id = ?",
        (panel_id,),
    )
    row = cur.fetchone()
    conn.close()
    return int(row["max_version"]) + 1


def wrap_text_for_svg(text: str, max_chars: int = 28) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = ""

    for word in words:
        test = word if not current else f"{current} {word}"
        if len(test) <= max_chars:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:4]


def image_href_from_file(path_str: str) -> Optional[str]:
    path = Path(path_str)
    if not path.exists():
        return None
    ext = path.suffix.lower()
    if ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"]:
        return path.resolve().as_uri()
    return None


def generate_svg_artwork(
    panel: sqlite3.Row,
    project: sqlite3.Row,
    payload: ArtworkGenerateRequest,
    logo_asset: Optional[sqlite3.Row] = None,
) -> str:
    panel_w = int(panel["width_mm"])
    panel_h = int(panel["height_mm"])
    safe = int(panel["safe_margin_mm"])
    bleed = int(panel["bleed_mm"])

    bg_color = payload.bg_color
    accent_color = payload.accent_color
    text_color = payload.text_color

    headline_lines = wrap_text_for_svg(payload.headline, max_chars=26 if panel_w < 3000 else 34)
    sub_lines = wrap_text_for_svg(payload.subheadline, max_chars=40 if panel_w < 3000 else 52)

    headline_svg = ""
    start_y = max(180, panel_h * 0.28)
    for i, line in enumerate(headline_lines):
        headline_svg += (
            f'<text x="{safe + 70}" y="{start_y + i * 100}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="78" font-weight="700" '
            f'fill="{html.escape(text_color)}">{html.escape(line)}</text>'
        )

    sub_svg = ""
    sub_start_y = start_y + len(headline_lines) * 100 + 40
    for i, line in enumerate(sub_lines):
        sub_svg += (
            f'<text x="{safe + 70}" y="{sub_start_y + i * 48}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="34" font-weight="400" '
            f'fill="{html.escape(text_color)}" opacity="0.92">{html.escape(line)}</text>'
        )

    cta_y = panel_h - safe - 120
    qr_box_size = 180

    logo_svg = ""
    logo_x = panel_w - safe - 320
    logo_y = safe + 40
    logo_href = None

    if logo_asset:
        logo_href = image_href_from_file(logo_asset["file_url"])

    if logo_href:
        logo_svg = (
            f'<image href="{html.escape(logo_href)}" '
            f'x="{logo_x}" y="{logo_y}" width="240" height="120" preserveAspectRatio="xMidYMid meet" />'
        )
    else:
        logo_svg = f"""
        <rect x="{logo_x}" y="{logo_y}" width="240" height="120" rx="14" fill="none"
              stroke="{html.escape(text_color)}" stroke-width="3" opacity="0.8"/>
        <text x="{logo_x + 28}" y="{logo_y + 72}"
              font-family="Arial, Helvetica, sans-serif" font-size="34" font-weight="700"
              fill="{html.escape(text_color)}">LOGO</text>
        """

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{panel_w + bleed * 2}" height="{panel_h + bleed * 2}" viewBox="0 0 {panel_w + bleed * 2} {panel_h + bleed * 2}">
  <defs>
    <linearGradient id="bgGrad" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="{html.escape(bg_color)}"/>
      <stop offset="100%" stop-color="#111A36"/>
    </linearGradient>
    <linearGradient id="accentGrad" x1="0" x2="1" y1="0" y2="0">
      <stop offset="0%" stop-color="{html.escape(accent_color)}"/>
      <stop offset="100%" stop-color="#7FA2FF"/>
    </linearGradient>
  </defs>

  <rect x="0" y="0" width="{panel_w + bleed * 2}" height="{panel_h + bleed * 2}" fill="#ffffff"/>
  <rect x="{bleed}" y="{bleed}" width="{panel_w}" height="{panel_h}" fill="url(#bgGrad)"/>

  <rect x="{bleed + safe}" y="{bleed + safe}" width="{panel_w - safe * 2}" height="{panel_h - safe * 2}"
        fill="none" stroke="rgba(255,255,255,0.22)" stroke-width="2" stroke-dasharray="10 8"/>

  <circle cx="{bleed + panel_w - 260}" cy="{bleed + 180}" r="220" fill="{html.escape(accent_color)}" opacity="0.13"/>
  <circle cx="{bleed + 180}" cy="{bleed + panel_h - 140}" r="160" fill="{html.escape(accent_color)}" opacity="0.12"/>

  <rect x="{bleed + safe + 70}" y="{bleed + 110}" width="180" height="12" rx="6" fill="url(#accentGrad)"/>

  {logo_svg}

  <g transform="translate({bleed}, {bleed})">
    {headline_svg}
    {sub_svg}
  </g>

  <g transform="translate({bleed}, {bleed})">
    <rect x="{safe + 70}" y="{cta_y - 56}" width="{max(320, min(620, len(payload.cta) * 18 + 80))}" height="74"
          rx="16" fill="{html.escape(accent_color)}"/>
    <text x="{safe + 105}" y="{cta_y - 8}" font-family="Arial, Helvetica, sans-serif" font-size="30"
          font-weight="700" fill="#ffffff">{html.escape(payload.cta)}</text>
  </g>

  <g transform="translate({bleed}, {bleed})">
    <rect x="{panel_w - safe - qr_box_size - 70}" y="{panel_h - safe - qr_box_size - 30}"
          width="{qr_box_size}" height="{qr_box_size}" rx="12" fill="#ffffff"/>
    <rect x="{panel_w - safe - qr_box_size - 50}" y="{panel_h - safe - qr_box_size - 10}"
          width="{qr_box_size - 40}" height="{qr_box_size - 40}" rx="8" fill="#f2f4f8" stroke="#d0d7e2"/>
    <text x="{panel_w - safe - qr_box_size - 5}" y="{panel_h - safe - 48}"
          font-family="Arial, Helvetica, sans-serif" font-size="20" text-anchor="end"
          fill="{html.escape(text_color)}">QR</text>
    <text x="{safe + 70}" y="{panel_h - safe - 22}"
          font-family="Arial, Helvetica, sans-serif" font-size="20"
          fill="{html.escape(text_color)}" opacity="0.7">{html.escape(project["name"])} • {html.escape(panel["panel_code"])}</text>
  </g>
</svg>
"""
    return svg


def save_artwork_files(
    project_id: str,
    panel_id: str,
    version_no: int,
    svg_content: str,
    payload_dict: dict[str, Any],
) -> tuple[str, str, str]:
    project_dir = ARTWORK_DIR / project_id / panel_id
    project_dir.mkdir(parents=True, exist_ok=True)

    svg_path = project_dir / f"v{version_no}.svg"
    json_path = project_dir / f"v{version_no}.json"
    preview_path = project_dir / f"v{version_no}_preview.svg"

    svg_path.write_text(svg_content, encoding="utf-8")
    json_path.write_text(json.dumps(payload_dict, indent=2), encoding="utf-8")
    preview_path.write_text(svg_content, encoding="utf-8")

    return str(svg_path), str(json_path), str(preview_path)


def latest_artwork_for_panel(panel_id: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM panel_artworks
        WHERE panel_id = ?
        ORDER BY version_no DESC
        LIMIT 1
        """,
        (panel_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row


def get_all_panels_for_project(project_id: str) -> list[sqlite3.Row]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM panels WHERE project_id = ? ORDER BY panel_code", (project_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def root():
    return {
        "message": "ExpoAI single-file backend is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "app": APP_NAME}


@app.post("/v1/projects")
def create_project(payload: ProjectCreate):
    project_id = str(uuid.uuid4())
    created_at = now_iso()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO projects (
            id, name, project_type, status, brand_id, booth_json,
            brief_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_id,
            payload.name,
            payload.project_type,
            "draft",
            None,
            json.dumps(payload.booth.model_dump()),
            json.dumps(payload.brief) if payload.brief else None,
            created_at,
            created_at,
        ),
    )
    conn.commit()
    conn.close()

    return {
        "message": "Project created successfully",
        "project_id": project_id,
        "name": payload.name,
    }


@app.get("/v1/projects")
def list_projects():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()

    items = []
    for row in rows:
        items.append({
            "id": row["id"],
            "name": row["name"],
            "project_type": row["project_type"],
            "status": row["status"],
            "brand_id": row["brand_id"],
            "booth": json.loads(row["booth_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    return {"items": items}


@app.get("/v1/projects/{project_id}")
def get_project(project_id: str):
    return get_project_full(project_id)


@app.post("/v1/brands")
def create_brand(payload: BrandCreate):
    brand_id = str(uuid.uuid4())
    created_at = now_iso()

    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO brands (id, name, lock_mode, created_at) VALUES (?, ?, ?, ?)",
        (brand_id, payload.name, payload.lock_mode, created_at),
    )

    if payload.project_id:
        cur.execute("SELECT id FROM projects WHERE id = ?", (payload.project_id,))
        project = cur.fetchone()
        if not project:
            conn.close()
            raise HTTPException(status_code=404, detail="Project not found")

        cur.execute(
            "UPDATE projects SET brand_id = ?, updated_at = ? WHERE id = ?",
            (brand_id, now_iso(), payload.project_id),
        )

    conn.commit()
    conn.close()

    return {
        "message": "Brand created successfully",
        "brand_id": brand_id,
        "name": payload.name,
        "project_id": payload.project_id,
    }


@app.get("/v1/projects/{project_id}/brand")
def get_project_brand(project_id: str):
    project = fetch_project(project_id)

    if not project["brand_id"]:
        return {"brand": None, "assets": []}

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM brands WHERE id = ?", (project["brand_id"],))
    brand = cur.fetchone()

    cur.execute(
        "SELECT * FROM brand_assets WHERE project_id = ? ORDER BY created_at DESC",
        (project_id,),
    )
    assets = cur.fetchall()
    conn.close()

    return {
        "brand": row_to_dict(brand),
        "assets": [
            {
                **dict(a),
                "meta": json.loads(a["meta_json"]) if a["meta_json"] else None,
            }
            for a in assets
        ],
    }


@app.post("/v1/projects/{project_id}/brand-assets")
async def upload_brand_asset(
    project_id: str,
    file: UploadFile = File(...),
    asset_type: str = Form("logo"),
    meta_json: str = Form("{}"),
):
    fetch_project(project_id)
    brand_id = ensure_project_brand(project_id)

    project_dir = BRAND_ASSETS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or ""
    safe_name = f"{uuid.uuid4()}{ext}"
    save_path = project_dir / safe_name

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    asset_id = str(uuid.uuid4())
    created_at = now_iso()

    try:
        parsed_meta = json.loads(meta_json) if meta_json else {}
    except json.JSONDecodeError:
        parsed_meta = {}

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO brand_assets (
            id, project_id, brand_id, asset_type, file_url, meta_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            asset_id,
            project_id,
            brand_id,
            asset_type,
            str(save_path),
            json.dumps(parsed_meta),
            created_at,
        ),
    )
    conn.commit()
    conn.close()

    return {
        "message": "Brand asset uploaded successfully",
        "asset_id": asset_id,
        "brand_id": brand_id,
        "asset_type": asset_type,
        "file_url": str(save_path),
    }


@app.post("/v1/projects/{project_id}/layout/generate")
def generate_layout(project_id: str, payload: LayoutGenerateRequest):
    project = fetch_project(project_id)
    booth = json.loads(project["booth_json"])

    clear_old_layout(project_id)
    layout = generate_layout_data(booth, payload)
    save_layout(project_id, layout)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
        ("layout_generated", now_iso(), project_id),
    )
    conn.commit()
    conn.close()

    return {
        "message": "Layout generated successfully",
        "project_id": project_id,
        "booth": booth,
        "zones": layout["zones"],
        "panels": layout["panels"],
    }


@app.get("/v1/projects/{project_id}/panels")
def list_project_panels(project_id: str):
    fetch_project(project_id)
    panels = get_all_panels_for_project(project_id)
    return {
        "items": [
            {
                **dict(p),
                "meta": json.loads(p["meta_json"]) if p["meta_json"] else {},
            }
            for p in panels
        ]
    }


@app.post("/v1/panels/{panel_id}/artwork/generate")
def generate_panel_artwork(panel_id: str, payload: ArtworkGenerateRequest):
    panel = fetch_panel(panel_id)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects WHERE id = ?", (panel["project_id"],))
    project = cur.fetchone()
    conn.close()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found for this panel")

    logo_asset = get_latest_brand_asset_for_project(panel["project_id"], "logo")

    version_no = next_artwork_version(panel_id)
    svg_content = generate_svg_artwork(panel, project, payload, logo_asset=logo_asset)

    payload_dict = payload.model_dump()
    payload_dict["panel_id"] = panel_id
    payload_dict["panel_code"] = panel["panel_code"]
    payload_dict["panel_type"] = panel["panel_type"]
    payload_dict["panel_size"] = {
        "width_mm": panel["width_mm"],
        "height_mm": panel["height_mm"],
        "bleed_mm": panel["bleed_mm"],
        "safe_margin_mm": panel["safe_margin_mm"],
    }

    svg_path, json_path, preview_path = save_artwork_files(
        project["id"], panel_id, version_no, svg_content, payload_dict
    )

    artwork_id = str(uuid.uuid4())

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO panel_artworks (
            id, panel_id, project_id, version_no,
            headline, subheadline, cta, qr_url, style_hint,
            bg_color, accent_color, text_color,
            svg_path, json_path, preview_path, payload_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            artwork_id,
            panel_id,
            project["id"],
            version_no,
            payload.headline,
            payload.subheadline,
            payload.cta,
            payload.qr_url,
            payload.style_hint,
            payload.bg_color,
            payload.accent_color,
            payload.text_color,
            svg_path,
            json_path,
            preview_path,
            json.dumps(payload_dict),
            now_iso(),
        ),
    )
    cur.execute(
        "UPDATE projects SET updated_at = ?, status = ? WHERE id = ?",
        (now_iso(), "artwork_generated", project["id"]),
    )
    conn.commit()
    conn.close()

    return {
        "message": "Artwork generated successfully",
        "artwork_id": artwork_id,
        "panel_id": panel_id,
        "panel_code": panel["panel_code"],
        "version_no": version_no,
        "svg_download_url": f"/v1/panels/{panel_id}/artwork/latest/svg",
        "json_download_url": f"/v1/panels/{panel_id}/artwork/latest/json",
        "preview_url": f"/v1/panels/{panel_id}/artwork/latest/preview",
    }


@app.get("/v1/panels/{panel_id}/artwork")
def list_panel_artworks(panel_id: str):
    fetch_panel(panel_id)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM panel_artworks
        WHERE panel_id = ?
        ORDER BY version_no DESC
        """,
        (panel_id,),
    )
    rows = cur.fetchall()
    conn.close()

    return {
        "items": [
            {
                **dict(r),
                "payload": json.loads(r["payload_json"]) if r["payload_json"] else {},
            }
            for r in rows
        ]
    }


@app.get("/v1/panels/{panel_id}/artwork/latest")
def get_latest_panel_artwork(panel_id: str):
    row = latest_artwork_for_panel(panel_id)
    if not row:
        raise HTTPException(status_code=404, detail="No artwork found for this panel")

    return {
        **dict(row),
        "payload": json.loads(row["payload_json"]) if row["payload_json"] else {},
    }


@app.get("/v1/panels/{panel_id}/artwork/latest/svg")
def download_latest_panel_svg(panel_id: str):
    row = latest_artwork_for_panel(panel_id)
    if not row:
        raise HTTPException(status_code=404, detail="No artwork found for this panel")

    svg_path = row["svg_path"]
    if not os.path.exists(svg_path):
        raise HTTPException(status_code=404, detail="SVG file not found")

    return FileResponse(
        svg_path,
        media_type="image/svg+xml",
        filename=Path(svg_path).name,
    )


@app.get("/v1/panels/{panel_id}/artwork/latest/preview")
def preview_latest_panel_svg(panel_id: str):
    row = latest_artwork_for_panel(panel_id)
    if not row:
        raise HTTPException(status_code=404, detail="No artwork found for this panel")

    preview_path = row["preview_path"]
    if not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="Preview file not found")

    return FileResponse(
        preview_path,
        media_type="image/svg+xml",
        filename=Path(preview_path).name,
    )


@app.get("/v1/panels/{panel_id}/artwork/latest/json")
def download_latest_panel_json(panel_id: str):
    row = latest_artwork_for_panel(panel_id)
    if not row:
        raise HTTPException(status_code=404, detail="No artwork found for this panel")

    json_path = row["json_path"]
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="JSON file not found")

    return FileResponse(
        json_path,
        media_type="application/json",
        filename=Path(json_path).name,
    )


@app.get("/v1/projects/{project_id}/artworks")
def list_project_artworks(project_id: str):
    fetch_project(project_id)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM panel_artworks
        WHERE project_id = ?
        ORDER BY created_at DESC
        """,
        (project_id,),
    )
    rows = cur.fetchall()
    conn.close()

    return {
        "items": [
            {
                **dict(r),
                "payload": json.loads(r["payload_json"]) if r["payload_json"] else {},
            }
            for r in rows
        ]
    }


# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
