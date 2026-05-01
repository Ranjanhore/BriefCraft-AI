# =============================================================================
# BriefCraftAI Brain  —  Production Backend
# Perfect Indian Teacher: Intro → Chapter Intro → Teaching → Quiz → Done
# =============================================================================
import os
import re
import time
import uuid
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
RENDER_DOMAIN = os.getenv("RENDER_EXTERNAL_HOSTNAME", "localhost:8000")
OPENAI_KEY    = os.getenv("OPENAI_API_KEY", "")
SB_URL        = os.getenv("SUPABASE_URL", "")
SB_KEY        = os.getenv("SUPABASE_SERVICE_KEY", "")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(CAD_PRO_DIR, exist_ok=True)

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

try:
    from cad_engine_pro import router as cad_pro_router
    app.include_router(cad_pro_router)
    print("[CAD] BriefCraft-AI Professional CAD router mounted at /api/cad/pro")
except Exception as _e:
    print(f"[WARN] CAD Pro router not mounted: {_e}")

# =============================================================================
# SESSION STORE
# =============================================================================
