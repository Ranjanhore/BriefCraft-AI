import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

JsonType = JSONB().with_variant(Text(), "sqlite")


class Brand(Base):
    __tablename__ = "brands"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    lock_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="controlled")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    assets: Mapped[list["BrandAsset"]] = relationship(back_populates="brand", cascade="all, delete-orphan")
    rules: Mapped[list["BrandRule"]] = relationship(back_populates="brand", cascade="all, delete-orphan")
    projects: Mapped[list["Project"]] = relationship(back_populates="brand")


class BrandAsset(Base):
    __tablename__ = "brand_assets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("brands.id", ondelete="CASCADE"), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_url: Mapped[str] = mapped_column(String(512), nullable=False)
    meta: Mapped[str | None] = mapped_column(JsonType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    brand: Mapped["Brand"] = relationship(back_populates="assets")


class BrandRule(Base):
    __tablename__ = "brand_rules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("brands.id", ondelete="CASCADE"), nullable=False)
    colors: Mapped[str | None] = mapped_column(JsonType, nullable=True)
    fonts: Mapped[str | None] = mapped_column(JsonType, nullable=True)
    logo_rules: Mapped[str | None] = mapped_column(JsonType, nullable=True)
    text_rules: Mapped[str | None] = mapped_column(JsonType, nullable=True)
    export_rules: Mapped[str | None] = mapped_column(JsonType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    brand: Mapped["Brand"] = relationship(back_populates="rules")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    project_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    brief: Mapped[str | None] = mapped_column(JsonType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    brand: Mapped["Brand | None"] = relationship(back_populates="projects")
    booth: Mapped["Booth"] = relationship(back_populates="project", cascade="all, delete-orphan", uselist=False)
    zones: Mapped[list["Zone"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    panels
