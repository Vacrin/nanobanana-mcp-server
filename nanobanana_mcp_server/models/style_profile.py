"""Pydantic models for style profiles."""

from enum import Enum

from pydantic import BaseModel, Field


class ReferenceStrategy(str, Enum):
    """Strategy for selecting reference images from a profile."""

    FIRST_N = "first_n"
    RANDOM = "random"


class ResizeConfig(BaseModel):
    """Post-processing resize configuration."""

    enabled: bool = False
    width: int = Field(256, ge=1, le=8192)
    height: int = Field(256, ge=1, le=8192)
    resample: str = "nearest"


class PostProcessingConfig(BaseModel):
    """Post-processing configuration for generated images."""

    resize: ResizeConfig = Field(default_factory=ResizeConfig)
    format: str = "png"


class ReferenceImagesConfig(BaseModel):
    """Configuration for reference image selection."""

    strategy: ReferenceStrategy = ReferenceStrategy.FIRST_N
    max_images: int = Field(3, ge=1, le=3)


class ProfileDefaults(BaseModel):
    """Default generation parameters from a style profile."""

    model_tier: str | None = None
    aspect_ratio: str | None = None
    resolution: str | None = None
    n: int | None = None
    enable_grounding: bool | None = None
    thinking_level: str | None = None


class StyleProfile(BaseModel):
    """A complete style profile definition loaded from profile.json."""

    label: str = Field(
        ..., min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$"
    )
    version: str = "1.0"
    description: str = ""
    style_description: str = ""
    system_instruction: str | None = None
    negative_prompt: str | None = None
    defaults: ProfileDefaults = Field(default_factory=ProfileDefaults)
    reference_images: ReferenceImagesConfig = Field(
        default_factory=ReferenceImagesConfig
    )
    post_processing: PostProcessingConfig = Field(
        default_factory=PostProcessingConfig
    )
    tags: list[str] = Field(default_factory=list)


class ProfileSummary(BaseModel):
    """Lightweight profile summary for listings."""

    label: str
    description: str
    tags: list[str]
    reference_image_count: int
    has_post_processing: bool


class ValidationResult(BaseModel):
    """Result of profile validation."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
