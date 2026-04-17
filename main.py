import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================================================
# BASIC CONFIG
# =========================================================

APP_NAME = "ExpoAI Single File Backend"
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "expoai.db"
STORAGE_DIR = BASE_DIR / "storage"
BRAND_ASSETS_DIR = STORAGE_DIR / "brand_assets"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
BRAND_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_NAME, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later replace with your frontend URL
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
    booth_type: str = "peninsula"   # inline / corner / peninsula
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


# =========================================================
# HELPERS
# =========================================================

def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
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

    # Basic printable panels
    fascia_h = 500
    backwall_h = min(3000, height - 200)
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
    }


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


# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
