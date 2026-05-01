"""
cad_engine_pro.py - BriefCraft-AI Professional CAD Engine v5.0

Drop alongside main.py and register with:
    from cad_engine_pro import router as cad_pro_router
    app.include_router(cad_pro_router)
"""

from __future__ import annotations

import base64
import datetime
import io
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import ezdxf
    from ezdxf import colors as dxf_colors  # noqa: F401
    from ezdxf.enums import TextEntityAlignment  # noqa: F401
    from ezdxf.math import Vec3  # noqa: F401
    _EZDXF = True
except ImportError:
    ezdxf = None  # type: ignore
    _EZDXF = False

try:
    from PIL import Image as _PILImage  # noqa: F401
    _PIL = True
except ImportError:
    _PIL = False

try:
    import fitz as _fitz
    _PYMUPDF = True
except ImportError:
    _fitz = None  # type: ignore
    _PYMUPDF = False

try:
    import pdfplumber as _pdfplumber
    _PDFPLUMBER = True
except ImportError:
    _pdfplumber = None  # type: ignore
    _PDFPLUMBER = False

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile  # noqa: F401
from fastapi.responses import Response  # noqa: F401
from pydantic import BaseModel, Field  # noqa: F401

router = APIRouter(prefix="/api/cad/pro", tags=["CAD Pro"])


class CadUnit(str, Enum):
    MM = "mm"
    CM = "cm"
    M = "m"
    INCH = "inch"
    FEET = "feet"


_TO_MM = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "inch": 25.4,
    "in": 25.4,
    "inches": 25.4,
    "feet": 304.8,
    "ft": 304.8,
    "foot": 304.8,
    "'": 304.8,
    '"': 25.4,
}
_SUFFIX = {"mm": "mm", "cm": "cm", "m": "m", "inch": '"', "in": '"', "feet": "'", "ft": "'"}
_INSUNITS = {"inch": 1, "in": 1, "feet": 2, "ft": 2, "mm": 4, "cm": 5, "m": 6}


def _norm_unit(s: str) -> str:
    unit = str(s or "m").strip().lower()
    return {"meter": "m", "meters": "m", "metre": "m", "metres": "m"}.get(unit, unit)


def to_mm(value: float, unit: str) -> float:
    return float(value) * _TO_MM.get(_norm_unit(unit), 1000.0)


def from_mm(value_mm: float, unit: str) -> float:
    return float(value_mm) / _TO_MM.get(_norm_unit(unit), 1000.0)


def fmt_dim(value_mm: float, unit: str, prec: int = 2) -> str:
    u = _norm_unit(unit)
    return f"{from_mm(value_mm, u):.{prec}f}{_SUFFIX.get(u, u)}"


def parse_dim_string(text: str, default_unit: str = "m") -> Optional[Tuple[float, float, str]]:
    pattern = r"([\d.]+)\s*([a-zA-Z'\"]*)\s*[x×X*,]\s*([\d.]+)\s*([a-zA-Z'\"]*)"
    m = re.search(pattern, str(text or ""))
    if not m:
        return None
    try:
        w_u = m.group(2) or default_unit
        d_u = m.group(4) or default_unit
        return to_mm(float(m.group(1)), w_u), to_mm(float(m.group(3)), d_u), "mm"
    except Exception:
        return None


@dataclass
class LayerDef:
    name: str
    color: int
    lw: int = 25
    ltype: str = "CONTINUOUS"
    description: str = ""


LAYERS: Dict[str, LayerDef] = {
    "A-WALL": LayerDef("A-WALL", 7, 50, description="Walls"),
    "A-WALL-FULL": LayerDef("A-WALL-FULL", 7, 70, description="Full-height walls"),
    "A-FLOR-OTLN": LayerDef("A-FLOR-OTLN", 8, 35, description="Floor boundary"),
    "A-GRID": LayerDef("A-GRID", 9, 13, description="Reference grid"),
    "A-DIMS": LayerDef("A-DIMS", 2, 18, description="Dimensions"),
    "A-TEXT": LayerDef("A-TEXT", 7, 18, description="Annotations"),
    "A-TITLE": LayerDef("A-TITLE", 7, 35, description="Title block"),
    "A-HATCH": LayerDef("A-HATCH", 9, 13, description="Zone hatches"),
    "A-NORTH": LayerDef("A-NORTH", 7, 25, description="North arrow"),
    "A-CAMERA": LayerDef("A-CAMERA", 5, 18, description="Camera positions"),
    "A-SPEAKER": LayerDef("A-SPEAKER", 1, 18, description="Speaker positions"),
    "A-EXIT": LayerDef("A-EXIT", 1, 35, description="Emergency exits"),
    "E-POWER": LayerDef("E-POWER", 1, 25, "DASHED", "Power routes"),
    "E-SIGNAL": LayerDef("E-SIGNAL", 4, 18, "DASHED", "Signal routes"),
    "S-STAGE": LayerDef("S-STAGE", 1, 50, description="Stage"),
    "S-LED": LayerDef("S-LED", 5, 35, description="LED screen"),
    "S-AUDIENCE": LayerDef("S-AUDIENCE", 3, 25, description="Audience"),
    "S-VIP": LayerDef("S-VIP", 2, 25, description="VIP"),
    "S-FOH": LayerDef("S-FOH", 4, 25, description="FOH"),
    "S-BOH": LayerDef("S-BOH", 6, 25, description="BOH"),
    "S-CIRCULATION": LayerDef("S-CIRCULATION", 8, 18, description="Circulation"),
    "S-REGISTRATION": LayerDef("S-REGISTRATION", 3, 25, description="Registration"),
    "S-CATERING": LayerDef("S-CATERING", 2, 25, description="Catering"),
    "S-PARKING": LayerDef("S-PARKING", 9, 18, description="Parking"),
    "FURN-CHAIR": LayerDef("FURN-CHAIR", 8, 13, description="Chair symbols"),
    "FURN-TABLE": LayerDef("FURN-TABLE", 8, 13, description="Table symbols"),
}
ZONE_LAYER = {
    "stage": "S-STAGE",
    "led": "S-LED",
    "audience": "S-AUDIENCE",
    "vip": "S-VIP",
    "foh": "S-FOH",
    "boh": "S-BOH",
    "circulation": "S-CIRCULATION",
    "registration": "S-REGISTRATION",
    "catering": "S-CATERING",
    "boundary": "A-FLOR-OTLN",
    "wall": "A-WALL",
    "parking": "S-PARKING",
}
ZONE_HATCH = {
    "stage": ("ANSI31", 80),
    "led": ("ANSI37", 60),
    "audience": ("ANSI32", 120),
    "vip": ("ANSI33", 80),
    "foh": ("ANSI34", 80),
    "boh": ("ANSI35", 80),
    "circulation": ("DOTS", 200),
    "registration": ("ANSI36", 80),
}
ZONE_SVG_FILL = {
    "stage": "#3a1010",
    "led": "#0e1e38",
    "audience": "#0e2c14",
    "vip": "#302010",
    "foh": "#0a1a30",
    "boh": "#2a1040",
    "circulation": "#1a1a1a",
    "registration": "#0e2c14",
    "catering": "#1a2010",
    "boundary": "none",
    "parking": "#141414",
}
ZONE_SVG_STROKE = {
    "stage": "#ff6b6b",
    "led": "#58a6ff",
    "audience": "#76d275",
    "vip": "#ffd166",
    "foh": "#80bfff",
    "boh": "#d0a5ff",
    "circulation": "#888888",
    "registration": "#76d275",
    "catering": "#aedd88",
    "boundary": "#f6e7b1",
    "parking": "#666666",
}


@dataclass
class CadZone:
    id: str
    name: str
    zone_type: str
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    label: str = ""
    rotation: float = 0.0
    notes: str = ""

    @property
    def cx(self) -> float:
        return self.x_mm + self.w_mm / 2

    @property
    def cy(self) -> float:
        return self.y_mm + self.h_mm / 2

    @property
    def area_m2(self) -> float:
        return (self.w_mm * self.h_mm) / 1e6


@dataclass
class CadSymbol:
    id: str
    symbol_type: str
    x_mm: float
    y_mm: float
    rotation: float = 0.0
    scale: float = 1.0
    label: str = ""


@dataclass
class CadRoute:
    id: str
    route_type: str
    pts: List[Tuple[float, float]]
    label: str = ""


@dataclass
class CadLayout:
    id: str
    project_id: str
    title: str
    venue_w_mm: float
    venue_d_mm: float
    unit: str
    scale_ratio: int
    audience_count: int
    venue_type: str
    brief: str
    zones: List[CadZone] = field(default_factory=list)
    symbols: List[CadSymbol] = field(default_factory=list)
    routes: List[CadRoute] = field(default_factory=list)
    ai_analysis: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    notes: str = ""


VENUE_PROFILES = {
    "concert": {"stage_w_pct": 0.70, "stage_d_pct": 0.22},
    "conference": {"stage_w_pct": 0.40, "stage_d_pct": 0.20},
    "exhibition": {"stage_w_pct": 0.30, "stage_d_pct": 0.18},
    "wedding": {"stage_w_pct": 0.25, "stage_d_pct": 0.18},
    "award_show": {"stage_w_pct": 0.55, "stage_d_pct": 0.25},
    "product_launch": {"stage_w_pct": 0.45, "stage_d_pct": 0.22},
    "generic": {"stage_w_pct": 0.45, "stage_d_pct": 0.22},
}
MIN_AISLE_MM = 1500
MIN_STAGE_DEPTH_MM = 4000


def _detect_venue_type(brief: str, explicit: Optional[str] = None) -> str:
    if explicit and explicit.lower() in VENUE_PROFILES:
        return explicit.lower()
    t = (brief or "").lower()
    for kw, vtype in [
        ("concert", "concert"),
        ("festival", "concert"),
        ("exhibition", "exhibition"),
        ("expo", "exhibition"),
        ("wedding", "wedding"),
        ("award", "award_show"),
        ("launch", "product_launch"),
        ("conference", "conference"),
        ("summit", "conference"),
    ]:
        if kw in t:
            return vtype
    return "generic"


def _best_scale(w_mm: float, d_mm: float) -> int:
    for n in [20, 25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500]:
        if w_mm / n <= 760 and d_mm / n <= 520:
            return n
    return 500


def _chair_grid(zone: CadZone, audience: int) -> List[CadSymbol]:
    symbols: List[CadSymbol] = []
    usable_w = zone.w_mm - 2 * MIN_AISLE_MM
    usable_d = zone.h_mm - MIN_AISLE_MM
    if usable_w <= 0 or usable_d <= 0:
        return symbols
    cols = max(1, int(usable_w / 550))
    rows = max(1, int(usable_d / 900))
    for idx in range(min(audience, cols * rows)):
        r = idx // cols
        c = idx % cols
        x = zone.x_mm + MIN_AISLE_MM + 550 * c + 275 + (250 if r % 2 else 0)
        y = zone.y_mm + 750 + 900 * r + 450
        symbols.append(CadSymbol(f"chair_{r}_{c}_{uuid.uuid4().hex[:4]}", "chair", x, y))
    return symbols


def generate_venue_layout(
    w_mm: float,
    d_mm: float,
    venue_type: str,
    audience: int,
    brief: str,
    project_id: str,
    title: str,
    unit: str,
) -> CadLayout:
    vtype = _detect_venue_type(brief, venue_type)
    profile = VENUE_PROFILES.get(vtype, VENUE_PROFILES["generic"])
    w = max(float(w_mm), 15000)
    d = max(float(d_mm), 10000)
    zones: List[CadZone] = [CadZone("boundary_outer", "VENUE BOUNDARY", "boundary", 0, 0, w, d)]
    symbols: List[CadSymbol] = []
    routes: List[CadRoute] = []

    sw = w * profile["stage_w_pct"]
    sd = max(d * profile["stage_d_pct"], MIN_STAGE_DEPTH_MM)
    sx = (w - sw) / 2
    stage = CadZone("zone_stage", "STAGE", "stage", sx, d - sd, sw, sd, f"STAGE {fmt_dim(sw, unit)} x {fmt_dim(sd, unit)}")
    zones.append(stage)

    led_h = max(1200, sd * 0.18)
    led_w = sw * 0.78
    zones.append(CadZone("zone_led", "LED SCREEN / BACKDROP", "led", sx + (sw - led_w) / 2, stage.y_mm + sd - led_h, led_w, led_h, f"LED {fmt_dim(led_w, unit)} x {fmt_dim(led_h, unit)}"))

    aud_gap = max(1500, d * 0.04)
    aud_w = min(w * 0.62, w - 2 * MIN_AISLE_MM)
    aud_x = max((w - aud_w) / 2, MIN_AISLE_MM)
    aud_zone = CadZone("zone_audience", "AUDIENCE SEATING", "audience", aud_x, 0, aud_w, max(stage.y_mm - aud_gap, 3000), f"AUDIENCE {audience} pax")
    zones.append(aud_zone)
    symbols.extend(_chair_grid(aud_zone, audience))

    side_w = max(0, (w - aud_x - aud_w) * 0.85)
    if side_w > 2000:
        zones.append(CadZone("zone_vip", "VIP SEATING", "vip", aud_x + aud_w + (w - aud_x - aud_w - side_w) / 2, aud_zone.h_mm * 0.25, side_w, aud_zone.h_mm * 0.5, "VIP"))

    foh_w = max(3000, w * 0.09)
    foh_d = max(2000, d * 0.07)
    foh_x = aud_x + aud_w / 2 - foh_w / 2
    foh_y = aud_zone.h_mm * 0.65
    zones.append(CadZone("zone_foh", "FOH CONTROL", "foh", foh_x, foh_y, foh_w, foh_d, "FOH"))
    zones.append(CadZone("zone_boh", "BOH / GREEN ROOM", "boh", w - max(4000, w * 0.15), d - max(3000, d * 0.12), max(4000, w * 0.15), max(3000, d * 0.12), "BOH"))
    zones.append(CadZone("zone_reg", "REGISTRATION", "registration", w * 0.03, d * 0.02, max(5000, w * 0.18), max(2000, d * 0.08), "REGISTRATION"))

    symbols.append(CadSymbol("power_db_main", "power_db", w * 0.88, d * 0.80, label="MAIN DB"))
    for cx, cy, rot, label in [(aud_x + aud_w / 2, foh_y + foh_d + 800, 180, "CAM-1 CENTER"), (aud_x - 1200, aud_zone.cy, 210, "CAM-2 LEFT"), (aud_x + aud_w + 1200, aud_zone.cy, 150, "CAM-3 RIGHT")]:
        symbols.append(CadSymbol(f"camera_{_safe(label)}", "camera", cx, cy, rot, label=label))
    for ex, ey, label in [(0, 0, "EXIT 1"), (w, 0, "EXIT 2"), (0, d, "EXIT 3"), (w, d, "EXIT 4"), (w / 2, 0, "EXIT 5")]:
        symbols.append(CadSymbol(f"exit_{_safe(label)}", "exit", ex, ey, label=label))

    db_x, db_y = w * 0.88, d * 0.80
    routes.append(CadRoute("route_power_stage", "power", [(db_x, db_y), (stage.cx, stage.y_mm)], "PWR-STAGE"))
    routes.append(CadRoute("route_signal_foh_led", "signal", [(foh_x + foh_w, foh_y + foh_d / 2), (stage.cx, stage.y_mm)], "SIG-LED"))
    routes.append(CadRoute("route_audio_stage", "audio", [(foh_x + foh_w / 2, foh_y), (stage.cx, stage.y_mm + 200)], "AUDIO"))

    return CadLayout(str(uuid.uuid4()), project_id, title, w, d, unit, _best_scale(w, d), audience, vtype, brief, zones, symbols, routes, created_at=_iso_now())


def _define_blocks(doc: Any) -> None:
    if "CHAIR" not in doc.blocks:
        blk = doc.blocks.new(name="CHAIR", dxfattribs={"layer": "FURN-CHAIR"})
        blk.add_lwpolyline([(-225, -180), (225, -180), (225, 160), (-225, 160)], dxfattribs={"layer": "FURN-CHAIR", "closed": True})
        blk.add_line((-250, 220), (250, 220), dxfattribs={"layer": "FURN-CHAIR"})
    if "TABLE" not in doc.blocks:
        blk = doc.blocks.new(name="TABLE", dxfattribs={"layer": "FURN-TABLE"})
        blk.add_lwpolyline([(-600, -300), (600, -300), (600, 300), (-600, 300)], dxfattribs={"layer": "FURN-TABLE", "closed": True})
    if "CAMERA" not in doc.blocks:
        blk = doc.blocks.new(name="CAMERA", dxfattribs={"layer": "A-CAMERA"})
        blk.add_solid([(-240, -180), (180, 0), (-240, 180), (0, 0)], dxfattribs={"layer": "A-CAMERA"})
        blk.add_line((180, 0), (420, 0), dxfattribs={"layer": "A-CAMERA"})
    if "SPEAKER" not in doc.blocks:
        blk = doc.blocks.new(name="SPEAKER", dxfattribs={"layer": "A-SPEAKER"})
        blk.add_lwpolyline([(-180, -300), (180, -300), (180, 300), (-180, 300)], dxfattribs={"layer": "A-SPEAKER", "closed": True})
        blk.add_circle((0, 0), 120, dxfattribs={"layer": "A-SPEAKER"})
    if "POWER_DB" not in doc.blocks:
        blk = doc.blocks.new(name="POWER_DB", dxfattribs={"layer": "E-POWER"})
        blk.add_lwpolyline([(-350, -280), (350, -280), (350, 280), (-350, 280)], dxfattribs={"layer": "E-POWER", "closed": True})
    if "EXIT" not in doc.blocks:
        blk = doc.blocks.new(name="EXIT", dxfattribs={"layer": "A-EXIT"})
        blk.add_lwpolyline([(-180, -180), (180, -180), (0, 280)], dxfattribs={"layer": "A-EXIT", "closed": True})


def generate_dxf(layout: CadLayout) -> bytes:
    if not _EZDXF:
        raise RuntimeError("ezdxf not installed. Run: pip install ezdxf")
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = _INSUNITS.get(_norm_unit(layout.unit), 6)
    doc.header["$LUNITS"] = 2
    doc.header["$LUPREC"] = 3
    for layer in LAYERS.values():
        if layer.name not in doc.layers:
            doc.layers.new(layer.name, dxfattribs={"color": layer.color, "lineweight": layer.lw, "linetype": layer.ltype if layer.ltype in doc.linetypes else "Continuous"})
    _define_blocks(doc)
    msp = doc.modelspace()
    for zone in layout.zones:
        layer = ZONE_LAYER.get(zone.zone_type, "A-FLOR-OTLN")
        x, y, w, h = zone.x_mm, zone.y_mm, zone.w_mm, zone.h_mm
        msp.add_lwpolyline([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], dxfattribs={"layer": layer, "closed": True, "lineweight": LAYERS[layer].lw})
        if zone.zone_type in ZONE_HATCH:
            try:
                pattern, scale = ZONE_HATCH[zone.zone_type]
                hatch = msp.add_hatch(color=LAYERS[layer].color, dxfattribs={"layer": "A-HATCH", "transparency": 0.75})
                hatch.paths.add_polyline_path([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], is_closed=True)
                hatch.set_pattern_fill(pattern, scale=scale, angle=45)
            except Exception:
                pass
        msp.add_text(zone.label or zone.name, dxfattribs={"layer": "A-TEXT", "height": max(200, min(w, h) * 0.04), "insert": (zone.cx, zone.cy)})
    off = max(1500, layout.venue_w_mm * 0.05)
    _add_dxf_dim(msp, (0, 0), (layout.venue_w_mm, 0), (layout.venue_w_mm / 2, -off), 0)
    _add_dxf_dim(msp, (0, 0), (0, layout.venue_d_mm), (-off, layout.venue_d_mm / 2), 90)
    block_names = {"chair": "CHAIR", "table": "TABLE", "camera": "CAMERA", "speaker": "SPEAKER", "power_db": "POWER_DB", "exit": "EXIT"}
    for sym in layout.symbols:
        block = block_names.get(sym.symbol_type)
        if block:
            msp.add_blockref(block, insert=(sym.x_mm, sym.y_mm), dxfattribs={"rotation": sym.rotation, "xscale": sym.scale, "yscale": sym.scale})
            if sym.label and sym.symbol_type != "chair":
                msp.add_text(sym.label, dxfattribs={"layer": "A-TEXT", "height": 150, "insert": (sym.x_mm + 400, sym.y_mm + 150)})
    for route in layout.routes:
        if len(route.pts) >= 2:
            layer = {"power": "E-POWER", "signal": "E-SIGNAL", "audio": "E-SIGNAL", "data": "E-SIGNAL"}.get(route.route_type, "E-SIGNAL")
            msp.add_lwpolyline(route.pts, dxfattribs={"layer": layer, "lineweight": 18})
    _add_dxf_grid_and_title(doc, msp, layout)
    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode("utf-8")


def _add_dxf_dim(msp: Any, p1: Tuple[float, float], p2: Tuple[float, float], base: Tuple[float, float], angle: float) -> None:
    try:
        dim = msp.add_linear_dim(base=base, p1=p1, p2=p2, angle=angle, dimstyle="EZDXF", dxfattribs={"layer": "A-DIMS"})
        dim.render()
    except Exception:
        pass


def _add_dxf_grid_and_title(doc: Any, msp: Any, layout: CadLayout) -> None:
    off = max(1500, layout.venue_w_mm * 0.05)
    step = _round_to_nice(min(layout.venue_w_mm, layout.venue_d_mm) / 8)
    x = 0.0
    while x <= layout.venue_w_mm:
        msp.add_line((x, 0), (x, layout.venue_d_mm), dxfattribs={"layer": "A-GRID"})
        x += step
    y = 0.0
    while y <= layout.venue_d_mm:
        msp.add_line((0, y), (layout.venue_w_mm, y), dxfattribs={"layer": "A-GRID"})
        y += step
    na_x, na_y, r = layout.venue_w_mm + off * 1.2, layout.venue_d_mm - off * 2, off * 0.4
    msp.add_circle((na_x, na_y), r, dxfattribs={"layer": "A-NORTH"})
    msp.add_text("N", dxfattribs={"layer": "A-NORTH", "height": r * 0.5, "insert": (na_x - r * 0.15, na_y + r * 1.15)})
    tb_x, tb_w, tb_h = layout.venue_w_mm + off * 0.5, off * 4.5, layout.venue_d_mm
    msp.add_lwpolyline([(tb_x, 0), (tb_x + tb_w, 0), (tb_x + tb_w, tb_h), (tb_x, tb_h)], dxfattribs={"layer": "A-TITLE", "closed": True})
    rows = ["BRIEFCRAFT-AI", "PROFESSIONAL CAD v5.0", layout.title[:36], f"Venue: {fmt_dim(layout.venue_w_mm, 'm')} x {fmt_dim(layout.venue_d_mm, 'm')}", f"Audience: {layout.audience_count} pax", f"Scale 1:{layout.scale_ratio}", f"Unit: {layout.unit.upper()}"]
    for i, text in enumerate(rows):
        msp.add_text(text, dxfattribs={"layer": "A-TITLE", "height": 220 if i in {0, 2} else 160, "insert": (tb_x + 150, tb_h - 600 - i * 450)})


def generate_svg(layout: CadLayout, px_w: int = 1600) -> str:
    w_mm, d_mm = layout.venue_w_mm, layout.venue_d_mm
    margin, title_w = 80, 280
    px_h = int(px_w * (d_mm / max(w_mm, 1)) * 0.65 + margin * 2)
    draw_w, draw_h = px_w - margin * 2 - title_w, px_h - margin * 2
    sx, sy = draw_w / max(w_mm, 1), draw_h / max(d_mm, 1)

    def tx(x: float) -> float:
        return margin + x * sx

    def ty(y: float) -> float:
        return margin + (d_mm - y) * sy

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{px_w}" height="{px_h}" viewBox="0 0 {px_w} {px_h}" style="font-family:Arial,sans-serif;background:#070911">']
    lines.append('<defs><marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#d9d9d9"/></marker></defs>')
    lines.append(f'<rect width="{px_w}" height="{px_h}" fill="#070911"/>')
    step = _round_to_nice(min(w_mm, d_mm) / 8)
    x = 0.0
    while x <= w_mm:
        lines.append(f'<line x1="{tx(x):.1f}" y1="{ty(0):.1f}" x2="{tx(x):.1f}" y2="{ty(d_mm):.1f}" stroke="#1e2430"/>')
        x += step
    y = 0.0
    while y <= d_mm:
        lines.append(f'<line x1="{tx(0):.1f}" y1="{ty(y):.1f}" x2="{tx(w_mm):.1f}" y2="{ty(y):.1f}" stroke="#1e2430"/>')
        y += step
    for zone in layout.zones:
        fill = ZONE_SVG_FILL.get(zone.zone_type, "none")
        stroke = ZONE_SVG_STROKE.get(zone.zone_type, "#f6e7b1")
        lines.append(f'<rect x="{tx(zone.x_mm):.1f}" y="{ty(zone.y_mm + zone.h_mm):.1f}" width="{zone.w_mm * sx:.1f}" height="{zone.h_mm * sy:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{3 if zone.zone_type == "boundary" else 2}" rx="2"/>')
        lines.append(f'<text x="{tx(zone.cx):.1f}" y="{ty(zone.cy):.1f}" fill="{stroke}" font-size="12" text-anchor="middle" dominant-baseline="middle" font-weight="bold">{_svg_escape((zone.label or zone.name)[:44])}</text>')
    for sym in layout.symbols:
        spx, spy = tx(sym.x_mm), ty(sym.y_mm)
        if sym.symbol_type == "chair":
            s = max(3.0, 500 * sx * 0.4)
            lines.append(f'<rect x="{spx - s / 2:.1f}" y="{spy - s / 2:.1f}" width="{s:.1f}" height="{s:.1f}" rx="1" fill="#c8b870" stroke="#332c18"/>')
        elif sym.symbol_type == "camera":
            lines.append(f'<polygon points="{spx-10:.1f},{spy-8:.1f} {spx+10:.1f},{spy:.1f} {spx-10:.1f},{spy+8:.1f}" fill="#8fd3ff" transform="rotate({sym.rotation},{spx:.1f},{spy:.1f})"/>')
        elif sym.symbol_type == "power_db":
            lines.append(f'<rect x="{spx-9:.1f}" y="{spy-9:.1f}" width="18" height="18" fill="#ff5757"/>')
        elif sym.symbol_type == "exit":
            lines.append(f'<polygon points="{spx:.1f},{spy-10:.1f} {spx-8:.1f},{spy+7:.1f} {spx+8:.1f},{spy+7:.1f}" fill="#00ff88"/>')
    for route in layout.routes:
        pts = " ".join(f"{tx(px):.1f},{ty(py):.1f}" for px, py in route.pts)
        col = {"power": "#ff5757", "signal": "#58a6ff", "audio": "#ffd166", "data": "#76d275"}.get(route.route_type, "#888")
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-dasharray="9 6" marker-end="url(#arr)"/>')
    tb_x = px_w - title_w
    lines.append(f'<rect x="{tb_x}" y="0" width="{title_w}" height="{px_h}" fill="#0d1018" stroke="#b99a4d"/>')
    info = ["BriefCraft-AI CAD v5", "Professional Drawing", layout.title[:30], f"Project: {layout.project_id[:20]}", f"Venue: {fmt_dim(w_mm, 'm')} x {fmt_dim(d_mm, 'm')}", f"Audience: {layout.audience_count} pax", f"Type: {layout.venue_type.replace('_', ' ').title()}", f"Scale: 1:{layout.scale_ratio}", f"Unit: {layout.unit.upper()}", f"Date: {layout.created_at[:10]}"]
    for i, text in enumerate(info):
        weight = "bold" if i in {0, 2, 7} else "normal"
        lines.append(f'<text x="{tb_x+12}" y="{30+i*22}" fill="#f7f3dd" font-size="{16 if i == 0 else 11}" font-weight="{weight}">{_svg_escape(text)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def generate_pdf_from_layout(layout: CadLayout) -> bytes:
    try:
        from reportlab.lib.colors import HexColor, white
        from reportlab.lib.pagesizes import A1, landscape
        from reportlab.pdfgen import canvas as rl_canvas
    except ImportError:
        raise HTTPException(500, "reportlab required: pip install reportlab")
    buf = io.BytesIO()
    pw, ph = landscape(A1)
    margin, tb_w = 36, 220
    c = rl_canvas.Canvas(buf, pagesize=(pw, ph))
    c.setTitle(layout.title)
    sx = (pw - margin * 2 - tb_w) / max(layout.venue_w_mm, 1)
    sy = (ph - margin * 2) / max(layout.venue_d_mm, 1)

    def px(x: float) -> float:
        return margin + x * sx

    def py(y: float) -> float:
        return margin + y * sy

    c.setFillColor(HexColor("#070911"))
    c.rect(0, 0, pw, ph, fill=1, stroke=0)
    for zone in layout.zones:
        fill = ZONE_SVG_FILL.get(zone.zone_type, "#070911")
        stroke = ZONE_SVG_STROKE.get(zone.zone_type, "#f6e7b1")
        c.setFillColor(HexColor("#070911" if fill == "none" else fill))
        c.setStrokeColor(HexColor(stroke))
        c.rect(px(zone.x_mm), py(zone.y_mm), zone.w_mm * sx, zone.h_mm * sy, fill=1, stroke=1)
        c.setFillColor(HexColor(stroke))
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(px(zone.cx), py(zone.cy), (zone.label or zone.name)[:35])
    for sym in layout.symbols:
        if sym.symbol_type == "chair":
            s = max(2.0, 400 * sx * 0.35)
            c.setFillColor(HexColor("#c8b870"))
            c.rect(px(sym.x_mm) - s / 2, py(sym.y_mm) - s / 2, s, s, fill=1, stroke=0)
    tb_x = pw - tb_w
    c.setFillColor(HexColor("#0d1018"))
    c.rect(tb_x, 0, tb_w, ph, fill=1, stroke=0)
    c.setFillColor(white)
    y = ph - margin - 18
    for text, fs in [("BRIEFCRAFT-AI CAD v5.0", 14), ("Professional Drawing", 9), (layout.title[:28], 11), (f"Project: {layout.project_id[:20]}", 8), (f"Venue: {fmt_dim(layout.venue_w_mm, 'm')} x {fmt_dim(layout.venue_d_mm, 'm')}", 9), (f"Audience: {layout.audience_count} pax", 9), (f"Type: {layout.venue_type}", 9), (f"Scale: 1:{layout.scale_ratio}", 10), (f"Unit: {layout.unit.upper()}", 8), (f"Date: {layout.created_at[:10]}", 8)]:
        c.setFont("Helvetica-Bold" if fs >= 10 else "Helvetica", fs)
        c.drawString(tb_x + 10, y, text)
        y -= fs + 4
    c.showPage()
    c.save()
    return buf.getvalue()


_TRACE_SYSTEM = """
You are a professional CAD engineer and architectural drawing analyst.
Analyze the uploaded floor plan / venue image or PDF page.
Return ONLY JSON with venue_width, venue_depth, unit, scale_note, confidence,
venue_type, audience_estimate, zones, extracted_dimensions, structural_elements,
and analysis_notes. Use x_pct/y_pct/w_pct/h_pct fractions 0..1 from bottom-left.
"""


def analyze_image_with_vision(image_bytes: bytes, content_type: str, openai_client: Any, extra_context: str = "") -> Dict[str, Any]:
    if not openai_client:
        return _fallback_analysis()
    b64 = base64.b64encode(image_bytes).decode()
    try:
        resp = openai_client.chat.completions.create(
            model=os.getenv("CAD_VISION_MODEL", "gpt-4o"),
            max_tokens=2000,
            messages=[{"role": "user", "content": [{"type": "text", "text": _TRACE_SYSTEM + (f"\nContext: {extra_context}" if extra_context else "")}, {"type": "image_url", "image_url": {"url": f"data:{content_type or 'image/png'};base64,{b64}", "detail": "high"}}]}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        print("Vision analysis error:", repr(e))
        return _fallback_analysis()


def _fallback_analysis() -> Dict[str, Any]:
    return {"venue_width": None, "venue_depth": None, "unit": "m", "scale_note": None, "confidence": 0.1, "venue_type": "generic", "audience_estimate": None, "zones": [], "extracted_dimensions": [], "structural_elements": [], "analysis_notes": "Analysis failed; using defaults."}


def layout_from_vision_analysis(analysis: Dict[str, Any], project_id: str, title: str, brief: str, unit: str, fallback_w_mm: float = 40000, fallback_d_mm: float = 30000, audience: Optional[int] = None) -> CadLayout:
    detected_unit = analysis.get("unit") or unit or "m"
    w_mm = to_mm(float(analysis["venue_width"]), detected_unit) if analysis.get("venue_width") else fallback_w_mm
    d_mm = to_mm(float(analysis["venue_depth"]), detected_unit) if analysis.get("venue_depth") else fallback_d_mm
    aud = int(audience or analysis.get("audience_estimate") or 300)
    ai_zones = analysis.get("zones") or []
    if len(ai_zones) >= 2:
        layout = _layout_from_ai_zones(ai_zones, w_mm, d_mm, project_id, title, brief, unit, aud, analysis.get("venue_type") or "generic")
    else:
        layout = generate_venue_layout(w_mm, d_mm, analysis.get("venue_type") or "generic", aud, brief, project_id, title, unit)
    layout.ai_analysis = analysis
    return layout


def _layout_from_ai_zones(
    ai_zones: List[Dict[str, Any]],
    w_mm: float,
    d_mm: float,
    project_id: str,
    title: str,
    brief: str,
    unit: str,
    audience: int,
    venue_type: str,
) -> CadLayout:
    layout = CadLayout(
        id=str(uuid.uuid4()),
        project_id=project_id,
        title=title,
        venue_w_mm=w_mm,
        venue_d_mm=d_mm,
        unit=unit,
        scale_ratio=_best_scale(w_mm, d_mm),
        audience_count=audience,
        venue_type=venue_type,
        brief=brief,
        created_at=_iso_now(),
    )
    layout.zones.append(CadZone("boundary_outer", "VENUE BOUNDARY", "boundary", 0, 0, w_mm, d_mm))
    for i, az in enumerate(ai_zones):
        zone_type = az.get("type") or "circulation"
        x_pct = float(az.get("x_pct", 0) or 0)
        y_pct = float(az.get("y_pct", 0) or 0)
        w_pct = float(az.get("w_pct", 0.2) or 0.2)
        h_pct = float(az.get("h_pct", 0.2) or 0.2)
        zone = CadZone(
            id=f"zone_{i}_{uuid.uuid4().hex[:4]}",
            name=az.get("name") or zone_type.upper(),
            zone_type=zone_type,
            x_mm=x_pct * w_mm,
            y_mm=y_pct * d_mm,
            w_mm=max(w_pct * w_mm, 2000),
            h_mm=max(h_pct * d_mm, 1500),
            notes=az.get("notes") or "",
        )
        layout.zones.append(zone)
        if zone_type == "audience":
            layout.symbols.extend(_chair_grid(zone, audience))
    return layout


def import_from_image(data: bytes, content_type: str, openai_client: Any, project_id: str, title: str, brief: str, unit: str, fallback_w_mm: float, fallback_d_mm: float, audience: int) -> CadLayout:
    return layout_from_vision_analysis(analyze_image_with_vision(data, content_type, openai_client, brief), project_id, title, brief, unit, fallback_w_mm, fallback_d_mm, audience)


def import_from_pdf(data: bytes, openai_client: Any, project_id: str, title: str, brief: str, unit: str, fallback_w_mm: float, fallback_d_mm: float, audience: int) -> CadLayout:
    if _PYMUPDF:
        try:
            pdf = _fitz.open(stream=data, filetype="pdf")
            if pdf.page_count > 0:
                pix = pdf[0].get_pixmap(matrix=_fitz.Matrix(2.5, 2.5), colorspace=_fitz.csRGB, alpha=False)
                return import_from_image(pix.tobytes("png"), "image/png", openai_client, project_id, title, brief, unit, fallback_w_mm, fallback_d_mm, audience)
        except Exception as e:
            print("PyMuPDF render failed:", repr(e))
    if _PDFPLUMBER:
        try:
            with _pdfplumber.open(io.BytesIO(data)) as pdf:
                text = " ".join(page.extract_text() or "" for page in pdf.pages[:3])
            dims = parse_dim_string(text, unit)
            return generate_venue_layout(dims[0] if dims else fallback_w_mm, dims[1] if dims else fallback_d_mm, "generic", audience, brief, project_id, title, unit)
        except Exception as e:
            print("pdfplumber failed:", repr(e))
    return generate_venue_layout(fallback_w_mm, fallback_d_mm, "generic", audience, brief, project_id, title, unit)


def import_from_dxf(data: bytes, project_id: str, title: str, brief: str, unit: str, audience: int) -> CadLayout:
    if not _EZDXF:
        raise HTTPException(500, "ezdxf not installed")
    try:
        doc = ezdxf.read(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(422, f"Could not read DXF/DWG: {e}")
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for entity in doc.modelspace():
        pts: List[Tuple[float, float]] = []
        try:
            if entity.dxftype() == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in entity.get_points()]
            elif entity.dxftype() == "LINE":
                pts = [(entity.dxf.start.x, entity.dxf.start.y), (entity.dxf.end.x, entity.dxf.end.y)]
        except Exception:
            continue
        for px, py in pts:
            min_x, min_y, max_x, max_y = min(min_x, px), min(min_y, py), max(max_x, px), max(max_y, py)
    if min_x == float("inf"):
        w_mm, d_mm = 40000.0, 30000.0
    else:
        src_unit = {1: "inch", 2: "feet", 4: "mm", 5: "cm", 6: "m"}.get(doc.header.get("$INSUNITS", 6), "mm")
        w_mm, d_mm = to_mm(max_x - min_x, src_unit), to_mm(max_y - min_y, src_unit)
    layout = generate_venue_layout(w_mm, d_mm, "generic", audience, brief, project_id, title, unit)
    layout.ai_analysis = {"source": "dxf_import", "extents": {"min": [min_x, min_y], "max": [max_x, max_y]}}
    return layout


def _upload_to_supabase(data: bytes, path: str, content_type: str) -> str:
    import requests
    url_env = os.getenv("SUPABASE_URL", "").rstrip("/")
    key_env = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_SERVICE_KEY", "") or os.getenv("SUPABASE_KEY", "")
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "briefcraft-assets")
    if not url_env or not key_env:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")
    upload_url = f"{url_env}/storage/v1/object/{bucket}/{path}"
    headers = {"Authorization": f"Bearer {key_env}", "apikey": key_env, "Content-Type": content_type, "Cache-Control": "31536000", "x-upsert": "true"}
    res = requests.post(upload_url, headers=headers, data=data, timeout=120)
    if res.status_code not in (200, 201):
        raise RuntimeError(f"Supabase upload failed {res.status_code}: {res.text[:300]}")
    return f"{url_env}/storage/v1/object/public/{bucket}/{path}"


def _save_and_upload(data: bytes, project_id: str, filename: str, content_type: str, local_dir: Path) -> str:
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / filename
    local_path.write_bytes(data)
    try:
        return _upload_to_supabase(data, f"cad/pro/{_safe(project_id)}/{int(time.time())}-{uuid.uuid4().hex[:8]}-{filename}", content_type)
    except Exception as e:
        print("Supabase upload failed, returning local URL:", repr(e))
        host = os.getenv("RENDER_EXTERNAL_HOSTNAME", "")
        return f"https://{host}/media/cad_pro/{filename}" if host else f"/media/cad_pro/{filename}"


class CadProGenerateRequest(BaseModel):
    project_id: str = "demo-project"
    title: Optional[str] = "Venue Layout"
    brief: Optional[str] = ""
    venue_type: Optional[str] = None
    unit: Optional[str] = "m"
    venue_width: Optional[float] = None
    venue_depth: Optional[float] = None
    audience_count: Optional[int] = 300
    scale: Optional[int] = None
    notes: Optional[str] = ""
    include_dxf: bool = True
    include_svg: bool = True
    include_pdf: bool = True


def _get_openai() -> Any:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception:
        return None


def _get_cad_dir() -> Path:
    path = Path(os.getenv("CAD_PRO_DIR", "./media/cad_pro"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@router.get("/health")
def cad_pro_health() -> Dict[str, Any]:
    return {"ok": True, "version": "5.0", "ezdxf": _EZDXF, "pymupdf": _PYMUPDF, "pdfplumber": _PDFPLUMBER, "pillow": _PIL}


@router.post("/generate")
def cad_pro_generate(payload: CadProGenerateRequest) -> Dict[str, Any]:
    unit = _norm_unit(payload.unit or "m")
    dims = parse_dim_string(payload.brief or "", unit) if (not payload.venue_width or not payload.venue_depth) else None
    w_mm = to_mm(payload.venue_width, unit) if payload.venue_width else (dims[0] if dims else 40000)
    d_mm = to_mm(payload.venue_depth, unit) if payload.venue_depth else (dims[1] if dims else 30000)
    layout = generate_venue_layout(w_mm, d_mm, payload.venue_type or "generic", payload.audience_count or 300, payload.brief or "", payload.project_id, payload.title or "Venue Layout", unit)
    if payload.scale:
        layout.scale_ratio = payload.scale
    return _build_and_upload(layout, payload.include_dxf, payload.include_svg, payload.include_pdf)


@router.post("/trace")
async def cad_pro_trace(
    file: UploadFile = File(...),
    project_id: str = Form("demo-project"),
    title: str = Form("Traced Layout"),
    brief: str = Form(""),
    unit: str = Form("m"),
    venue_width: Optional[float] = Form(None),
    venue_depth: Optional[float] = Form(None),
    audience_count: int = Form(300),
    include_dxf: bool = Form(True),
    include_svg: bool = Form(True),
    include_pdf: bool = Form(True),
) -> Dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    ext = Path(file.filename or "upload").suffix.lower()
    content_type = file.content_type or ""
    u = _norm_unit(unit)
    fb_w = to_mm(venue_width, u) if venue_width else 40000.0
    fb_d = to_mm(venue_depth, u) if venue_depth else 30000.0
    client = _get_openai()
    if ext in {".dxf", ".dwg"}:
        layout = import_from_dxf(data, project_id, title, brief, u, audience_count)
    elif ext == ".pdf" or "pdf" in content_type:
        layout = import_from_pdf(data, client, project_id, title, brief, u, fb_w, fb_d, audience_count)
    elif ext in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"} or content_type.startswith("image/"):
        layout = import_from_image(data, content_type or "image/png", client, project_id, title, brief, u, fb_w, fb_d, audience_count)
    else:
        layout = import_from_image(data, "image/png", client, project_id, title, brief, u, fb_w, fb_d, audience_count)
    return _build_and_upload(layout, include_dxf, include_svg, include_pdf)


@router.post("/analyze-dimensions")
async def cad_pro_analyze(file: UploadFile = File(...), brief: str = Form(""), unit: str = Form("m")) -> Dict[str, Any]:
    data = await file.read()
    content_type = file.content_type or "image/png"
    ext = Path(file.filename or "upload").suffix.lower()
    if (ext == ".pdf" or "pdf" in content_type) and _PYMUPDF:
        try:
            doc = _fitz.open(stream=data, filetype="pdf")
            if doc.page_count > 0:
                pix = doc[0].get_pixmap(matrix=_fitz.Matrix(2, 2), alpha=False)
                data = pix.tobytes("png")
                content_type = "image/png"
        except Exception:
            pass
    analysis = analyze_image_with_vision(data, content_type, _get_openai(), brief)
    dims_summary = [f"{d.get('label')}: {d.get('value')} {d.get('unit')}" for d in (analysis.get("extracted_dimensions") or [])]
    if analysis.get("venue_width") and analysis.get("venue_depth"):
        dims_summary.insert(0, f"Overall: {analysis.get('venue_width')} x {analysis.get('venue_depth')} {analysis.get('unit', 'm')}")
    return {"ok": True, "analysis": analysis, "dimensions_summary": dims_summary, "confidence": analysis.get("confidence", 0), "detected_venue_type": analysis.get("venue_type"), "zones_detected": len(analysis.get("zones") or [])}


@router.get("/{project_id}/latest")
def cad_pro_latest(project_id: str) -> Dict[str, Any]:
    raise HTTPException(404, "Use POST /api/cad/pro/generate with project_id to create a layout first.")


def _build_and_upload(layout: CadLayout, include_dxf: bool, include_svg: bool, include_pdf: bool) -> Dict[str, Any]:
    cad_dir = _get_cad_dir()
    base = f"cad_pro_{_safe(layout.project_id)}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    urls: Dict[str, Any] = {}
    if include_svg:
        urls["svg_url"] = _save_and_upload(generate_svg(layout).encode("utf-8"), layout.project_id, f"{base}.svg", "image/svg+xml", cad_dir)
    if include_dxf and _EZDXF:
        urls["dxf_url"] = _save_and_upload(generate_dxf(layout), layout.project_id, f"{base}.dxf", "application/dxf", cad_dir)
    elif include_dxf:
        urls["dxf_warning"] = "ezdxf not installed; DXF skipped. Run: pip install ezdxf"
    if include_pdf:
        try:
            urls["pdf_url"] = _save_and_upload(generate_pdf_from_layout(layout), layout.project_id, f"{base}.pdf", "application/pdf", cad_dir)
        except Exception as e:
            urls["pdf_warning"] = str(e)
    return {
        "ok": True,
        "layout_id": layout.id,
        "project_id": layout.project_id,
        "title": layout.title,
        "venue": {"width_mm": layout.venue_w_mm, "depth_mm": layout.venue_d_mm, "width": fmt_dim(layout.venue_w_mm, layout.unit), "depth": fmt_dim(layout.venue_d_mm, layout.unit), "area_m2": round((layout.venue_w_mm * layout.venue_d_mm) / 1e6, 1)},
        "unit": layout.unit,
        "scale": f"1:{layout.scale_ratio}",
        "audience_count": layout.audience_count,
        "venue_type": layout.venue_type,
        "zones": [{"id": z.id, "name": z.name, "type": z.zone_type, "x_mm": z.x_mm, "y_mm": z.y_mm, "w_mm": z.w_mm, "h_mm": z.h_mm, "label": z.label, "area_m2": round(z.area_m2, 2), "notes": z.notes} for z in layout.zones],
        "symbol_count": len(layout.symbols),
        "route_count": len(layout.routes),
        "ai_analysis": layout.ai_analysis or {},
        **urls,
    }


def _safe(s: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", str(s).lower())[:40].strip("-")


def _iso_now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _round_to_nice(v: float) -> float:
    for n in [500, 1000, 2000, 2500, 5000, 10000]:
        if v <= n * 1.5:
            return float(n)
    return 10000.0


def _svg_escape(s: str) -> str:
    return str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
