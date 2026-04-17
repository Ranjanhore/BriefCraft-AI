from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: List[str] | str = ["http://localhost:5173"]
    jwt_secret: str = "replace-me"

    openai_api_key: str = ""
    openai_model_text: str = "gpt-5.4-mini"
    openai_model_image: str = "gpt-image-1.5"

    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    database_url: str = "sqlite:///./expoai.db"

    redis_url: str = "redis://localhost:6379/0"

    storage_provider: str = "local"
    local_storage_path: str = "./storage"
    render_output_path: str = "./storage/renders"
    artwork_output_path: str = "./storage/artwork"
    drawing_output_path: str = "./storage/drawings"
    export_output_path: str = "./storage/exports"

    blender_binary_path: str = "/path/to/blender"
    freecad_binary_path: str = "/path/to/freecadcmd"
    usd_root_path: str = "./usd"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors_origins(cls, value: List[str] | str):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
