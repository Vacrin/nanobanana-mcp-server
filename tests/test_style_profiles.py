"""Tests for the style profile system."""

import json
import os
import tempfile

from PIL import Image
from pydantic import ValidationError as PydanticValidationError
import pytest

from nanobanana_mcp_server.models.style_profile import (
    ProfileDefaults,
    ReferenceStrategy,
    StyleProfile,
)
from nanobanana_mcp_server.services.style_profile_service import StyleProfileService
from nanobanana_mcp_server.utils.image_utils import resize_image_exact

# --- Pydantic model tests ---


class TestStyleProfileModel:
    def test_minimal_profile(self):
        """Only label is required."""
        p = StyleProfile(label="test-profile")
        assert p.label == "test-profile"
        assert p.description == ""
        assert p.system_instruction is None
        assert p.post_processing.resize.enabled is False

    def test_full_profile(self):
        p = StyleProfile(
            label="pixel-forest-iso",
            description="Isometric pixel art forest",
            style_description="Pixel art, 45-degree perspective",
            system_instruction="You are a pixel art generator",
            negative_prompt="blurry, photorealistic",
            defaults=ProfileDefaults(
                model_tier="nb2", aspect_ratio="1:1", enable_grounding=False
            ),
            tags=["game-assets", "pixel-art"],
        )
        assert p.defaults.model_tier == "nb2"
        assert p.defaults.aspect_ratio == "1:1"
        assert len(p.tags) == 2

    def test_invalid_label_uppercase(self):
        with pytest.raises(PydanticValidationError):
            StyleProfile(label="Invalid-Label")

    def test_invalid_label_spaces(self):
        with pytest.raises(PydanticValidationError):
            StyleProfile(label="has spaces")

    def test_invalid_label_empty(self):
        with pytest.raises(PydanticValidationError):
            StyleProfile(label="")

    def test_valid_label_patterns(self):
        """Various valid label patterns."""
        for label in ["a", "test", "my-profile", "pixel_art_01", "0start"]:
            p = StyleProfile(label=label)
            assert p.label == label

    def test_resize_config_defaults(self):
        p = StyleProfile(label="test")
        assert p.post_processing.resize.width == 256
        assert p.post_processing.resize.height == 256
        assert p.post_processing.resize.resample == "nearest"

    def test_reference_strategy_enum(self):
        assert ReferenceStrategy.FIRST_N.value == "first_n"
        assert ReferenceStrategy.RANDOM.value == "random"


# --- StyleProfileService tests ---


def _create_test_profile(profiles_dir, label, profile_data=None, ref_images=0):
    """Helper to create a test profile on disk."""
    profile_dir = os.path.join(profiles_dir, label)
    os.makedirs(profile_dir)
    refs_dir = os.path.join(profile_dir, "references")
    os.makedirs(refs_dir)

    data = profile_data or {"label": label, "description": f"Test profile {label}"}
    with open(os.path.join(profile_dir, "profile.json"), "w") as f:
        json.dump(data, f)

    # Create small reference images
    for i in range(ref_images):
        img = Image.new("RGB", (64, 64), color=(i * 30, 100, 200))
        img.save(os.path.join(refs_dir, f"ref-{i:02d}.png"))

    return profile_dir


class TestStyleProfileService:
    def test_list_profiles_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = StyleProfileService(tmpdir)
            assert service.list_profiles() == []

    def test_list_profiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_profile(tmpdir, "profile-a", ref_images=2)
            _create_test_profile(tmpdir, "profile-b", ref_images=0)

            service = StyleProfileService(tmpdir)
            profiles = service.list_profiles()
            assert len(profiles) == 2
            labels = [p.label for p in profiles]
            assert "profile-a" in labels
            assert "profile-b" in labels

            a = next(p for p in profiles if p.label == "profile-a")
            assert a.reference_image_count == 2

    def test_load_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_profile(
                tmpdir,
                "my-style",
                {
                    "label": "my-style",
                    "description": "Test",
                    "system_instruction": "Be creative",
                    "negative_prompt": "ugly",
                    "defaults": {"model_tier": "nb2", "aspect_ratio": "1:1"},
                },
            )

            service = StyleProfileService(tmpdir)
            profile = service.load_profile("my-style")
            assert profile.label == "my-style"
            assert profile.system_instruction == "Be creative"
            assert profile.defaults.model_tier == "nb2"

    def test_load_profile_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = StyleProfileService(tmpdir)
            with pytest.raises(FileNotFoundError):
                service.load_profile("nonexistent")

    def test_load_profile_label_forced_from_dirname(self):
        """Label in JSON is overridden by directory name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_profile(
                tmpdir,
                "actual-name",
                {"label": "wrong-name", "description": "Test"},
            )

            service = StyleProfileService(tmpdir)
            profile = service.load_profile("actual-name")
            assert profile.label == "actual-name"

    def test_validate_profile_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_profile(
                tmpdir,
                "valid-profile",
                {
                    "label": "valid-profile",
                    "description": "Valid",
                    "system_instruction": "Test instruction",
                },
                ref_images=2,
            )

            service = StyleProfileService(tmpdir)
            result = service.validate_profile("valid-profile")
            assert result.valid is True
            assert len(result.errors) == 0

    def test_validate_profile_missing_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = StyleProfileService(tmpdir)
            result = service.validate_profile("missing")
            assert result.valid is False
            assert any("not found" in e for e in result.errors)

    def test_validate_profile_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = os.path.join(tmpdir, "bad-json")
            os.makedirs(profile_dir)
            with open(os.path.join(profile_dir, "profile.json"), "w") as f:
                f.write("not valid json{{{")

            service = StyleProfileService(tmpdir)
            result = service.validate_profile("bad-json")
            assert result.valid is False

    def test_validate_profile_warnings(self):
        """Profile with no refs and no style guidance should warn."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_profile(
                tmpdir,
                "sparse",
                {"label": "sparse", "description": "Sparse profile"},
                ref_images=0,
            )

            service = StyleProfileService(tmpdir)
            result = service.validate_profile("sparse")
            assert result.valid is True
            assert len(result.warnings) > 0

    def test_get_reference_image_paths_first_n(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_profile(tmpdir, "refs-test", ref_images=5)

            service = StyleProfileService(tmpdir)
            profile = service.load_profile("refs-test")
            paths = service.get_reference_image_paths(profile, max_count=3)
            assert len(paths) == 3
            # first_n should be sorted alphabetically
            basenames = [os.path.basename(p) for p in paths]
            assert basenames == sorted(basenames)

    def test_get_reference_image_paths_random(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_profile(
                tmpdir,
                "random-test",
                {
                    "label": "random-test",
                    "reference_images": {"strategy": "random", "max_images": 2},
                },
                ref_images=5,
            )

            service = StyleProfileService(tmpdir)
            profile = service.load_profile("random-test")
            paths = service.get_reference_image_paths(profile, max_count=3)
            # max_images=2 in profile, so should return at most 2
            assert len(paths) == 2

    def test_get_reference_image_paths_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_profile(tmpdir, "no-refs", ref_images=0)

            service = StyleProfileService(tmpdir)
            profile = service.load_profile("no-refs")
            paths = service.get_reference_image_paths(profile)
            assert paths == []

    def test_init_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = StyleProfileService(tmpdir)
            path = service.init_profile("new-profile")

            assert os.path.isdir(path)
            assert os.path.isfile(os.path.join(path, "profile.json"))
            assert os.path.isdir(os.path.join(path, "references"))

            # Verify the template is valid JSON and loadable
            profile = service.load_profile("new-profile")
            assert profile.label == "new-profile"

    def test_init_profile_already_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_profile(tmpdir, "existing")

            service = StyleProfileService(tmpdir)
            with pytest.raises(FileExistsError):
                service.init_profile("existing")

    def test_init_profile_invalid_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = StyleProfileService(tmpdir)
            with pytest.raises(PydanticValidationError):
                service.init_profile("INVALID LABEL")


# --- resize_image_exact tests ---


class TestResizeImageExact:
    def _make_image_bytes(self, width, height, fmt="PNG"):
        """Create test image bytes."""
        from io import BytesIO

        img = Image.new("RGB", (width, height), color=(255, 0, 0))
        buf = BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()

    def test_resize_basic(self):
        original = self._make_image_bytes(512, 512)
        resized = resize_image_exact(original, 256, 256)
        img = Image.open(__import__("io").BytesIO(resized))
        assert img.size == (256, 256)

    def test_resize_non_square(self):
        original = self._make_image_bytes(800, 600)
        resized = resize_image_exact(original, 128, 64)
        img = Image.open(__import__("io").BytesIO(resized))
        assert img.size == (128, 64)

    def test_resize_nearest(self):
        """Nearest resampling should work for pixel art."""
        original = self._make_image_bytes(32, 32)
        resized = resize_image_exact(original, 256, 256, resample="nearest")
        img = Image.open(__import__("io").BytesIO(resized))
        assert img.size == (256, 256)

    def test_resize_lanczos(self):
        original = self._make_image_bytes(512, 512)
        resized = resize_image_exact(original, 256, 256, resample="lanczos")
        img = Image.open(__import__("io").BytesIO(resized))
        assert img.size == (256, 256)

    def test_resize_output_format(self):
        original = self._make_image_bytes(64, 64)
        resized = resize_image_exact(original, 32, 32, output_format="JPEG")
        img = Image.open(__import__("io").BytesIO(resized))
        assert img.format == "JPEG"

    def test_resize_unknown_resample_falls_back(self):
        """Unknown resample method should fall back to nearest."""
        original = self._make_image_bytes(64, 64)
        resized = resize_image_exact(original, 32, 32, resample="unknown")
        img = Image.open(__import__("io").BytesIO(resized))
        assert img.size == (32, 32)
