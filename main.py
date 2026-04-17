from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.brands import router as brands_router
from app.api.health import router as health_router
from app.api.layouts import router as layouts_router
from app.api.projects import router as projects_router
from app.core.config import get_settings
from app.db.base import Base
from app.db.session import engine
from app.models.project import Brand, BrandAsset, BrandRule, Booth, Panel, Project, Zone  # noqa: F401

settings = get_settings()
Base.metadata.create_all(bind=engine)
Path(settings.local_storage_path).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ExpoAI API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(projects_router)
app.include_router(brands_router)
app.include_router(layouts_router)
