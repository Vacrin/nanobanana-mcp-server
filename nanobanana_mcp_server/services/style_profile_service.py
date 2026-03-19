"""Style profile service for loading and managing reusable generation profiles."""

import json
import logging
import os
import random as random_module

from ..config.constants import (
    PROFILE_FILENAME,
    PROFILE_REFERENCES_DIR,
    SUPPORTED_REFERENCE_EXTENSIONS,
    SUPPORTED_RESAMPLE_METHODS,
)
from ..models.style_profile import (
    ProfileSummary,
    ReferenceStrategy,
    StyleProfile,
    ValidationResult,
)


class StyleProfileService:
    """Service for loading, validating, and managing style profiles on disk."""

    def __init__(self, profiles_dir: str):
        self.profiles_dir = profiles_dir
        self.logger = logging.getLogger(__name__)

    def list_profiles(self) -> list[ProfileSummary]:
        """List all available style profiles."""
        summaries = []

        if not os.path.isdir(self.profiles_dir):
            return summaries

        for entry in sorted(os.listdir(self.profiles_dir)):
            profile_dir = os.path.join(self.profiles_dir, entry)
            profile_file = os.path.join(profile_dir, PROFILE_FILENAME)

            if not os.path.isdir(profile_dir) or not os.path.isfile(profile_file):
                continue

            try:
                profile = self._load_json(profile_file, entry)
                ref_count = self._count_reference_images(profile_dir)
                summaries.append(
                    ProfileSummary(
                        label=profile.label,
                        description=profile.description,
                        tags=profile.tags,
                        reference_image_count=ref_count,
                        has_post_processing=profile.post_processing.resize.enabled,
                    )
                )
            except Exception as e:
                self.logger.warning(f"Skipping invalid profile '{entry}': {e}")

        return summaries

    def load_profile(self, label: str) -> StyleProfile:
        """Load and validate a profile by label."""
        profile_dir = os.path.join(self.profiles_dir, label)
        profile_file = os.path.join(profile_dir, PROFILE_FILENAME)

        if not os.path.isdir(profile_dir):
            raise FileNotFoundError(f"Profile directory not found: {profile_dir}")
        if not os.path.isfile(profile_file):
            raise FileNotFoundError(f"Profile file not found: {profile_file}")

        return self._load_json(profile_file, label)

    def validate_profile(self, label: str) -> ValidationResult:
        """Validate a profile for errors and warnings."""
        errors: list[str] = []
        warnings: list[str] = []

        profile_dir = os.path.join(self.profiles_dir, label)
        profile_file = os.path.join(profile_dir, PROFILE_FILENAME)

        # Check directory exists
        if not os.path.isdir(profile_dir):
            return ValidationResult(
                valid=False, errors=[f"Profile directory not found: {profile_dir}"]
            )

        # Check profile.json exists
        if not os.path.isfile(profile_file):
            return ValidationResult(
                valid=False, errors=[f"Profile file not found: {profile_file}"]
            )

        # Try to parse and validate
        try:
            profile = self._load_json(profile_file, label)
        except json.JSONDecodeError as e:
            return ValidationResult(valid=False, errors=[f"Invalid JSON: {e}"])
        except Exception as e:
            return ValidationResult(valid=False, errors=[f"Validation error: {e}"])

        # Validate resample method
        if profile.post_processing.resize.resample not in SUPPORTED_RESAMPLE_METHODS:
            errors.append(
                f"Unsupported resample method: '{profile.post_processing.resize.resample}'. "
                f"Must be one of: {', '.join(SUPPORTED_RESAMPLE_METHODS)}"
            )

        # Check reference images
        refs_dir = os.path.join(profile_dir, PROFILE_REFERENCES_DIR)
        if os.path.isdir(refs_dir):
            ref_files = self._list_reference_files(refs_dir)
            if not ref_files:
                warnings.append("References directory exists but contains no valid images")

            # Check for oversized references (>20MB)
            for ref_path in ref_files:
                size = os.path.getsize(ref_path)
                if size > 20 * 1024 * 1024:
                    warnings.append(
                        f"Reference image '{os.path.basename(ref_path)}' is "
                        f"{size / (1024 * 1024):.1f}MB (max recommended: 20MB)"
                    )
        else:
            warnings.append("No references directory found")

        # Check for empty style guidance
        if not profile.system_instruction and not profile.style_description:
            warnings.append(
                "No system_instruction or style_description set — "
                "profile provides only defaults and reference images"
            )

        return ValidationResult(
            valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def get_reference_image_paths(
        self, profile: StyleProfile, max_count: int = 3
    ) -> list[str]:
        """Select reference image paths from the profile's references directory."""
        refs_dir = os.path.join(self.profiles_dir, profile.label, PROFILE_REFERENCES_DIR)
        if not os.path.isdir(refs_dir):
            return []

        image_files = self._list_reference_files(refs_dir)
        if not image_files:
            return []

        limit = min(max_count, profile.reference_images.max_images, len(image_files))

        if profile.reference_images.strategy == ReferenceStrategy.RANDOM:
            return random_module.sample(image_files, limit)
        else:
            return image_files[:limit]

    def init_profile(self, label: str) -> str:
        """Scaffold a new profile directory with a template profile.json."""
        profile_dir = os.path.join(self.profiles_dir, label)

        if os.path.exists(profile_dir):
            raise FileExistsError(f"Profile already exists: {profile_dir}")

        # Validate label format
        StyleProfile(label=label)  # Will raise if invalid

        os.makedirs(profile_dir)
        os.makedirs(os.path.join(profile_dir, PROFILE_REFERENCES_DIR))

        template = {
            "label": label,
            "description": "",
            "style_description": "",
            "system_instruction": None,
            "negative_prompt": None,
            "defaults": {
                "model_tier": "nb2",
                "aspect_ratio": "1:1",
                "resolution": "high",
                "n": 1,
                "enable_grounding": False,
            },
            "reference_images": {"strategy": "first_n", "max_images": 3},
            "post_processing": {
                "resize": {
                    "enabled": False,
                    "width": 256,
                    "height": 256,
                    "resample": "nearest",
                },
                "format": "png",
            },
            "tags": [],
        }

        profile_file = os.path.join(profile_dir, PROFILE_FILENAME)
        with open(profile_file, "w") as f:
            json.dump(template, f, indent=2)

        self.logger.info(f"Created profile scaffold at {profile_dir}")
        return profile_dir

    def _load_json(self, profile_file: str, label: str) -> StyleProfile:
        """Load and parse a profile JSON file."""
        with open(profile_file) as f:
            data = json.load(f)

        # Ensure label matches directory name
        data["label"] = label

        return StyleProfile(**data)

    def _count_reference_images(self, profile_dir: str) -> int:
        """Count valid reference images in a profile directory."""
        refs_dir = os.path.join(profile_dir, PROFILE_REFERENCES_DIR)
        if not os.path.isdir(refs_dir):
            return 0
        return len(self._list_reference_files(refs_dir))

    def _list_reference_files(self, refs_dir: str) -> list[str]:
        """List valid image files in a references directory, sorted alphabetically."""
        return sorted(
            os.path.join(refs_dir, f)
            for f in os.listdir(refs_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_REFERENCE_EXTENSIONS
            and os.path.isfile(os.path.join(refs_dir, f))
        )
