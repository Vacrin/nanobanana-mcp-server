"""Style profile MCP resources."""

import logging
from typing import Any

from fastmcp import FastMCP


def register_style_profile_resources(server: FastMCP):
    """Register style profile resources with the FastMCP server."""

    @server.resource("nano-banana://style-profiles")
    def style_profiles_catalog() -> dict[str, Any]:
        """List all available style profiles."""
        logger = logging.getLogger(__name__)

        try:
            from ..services import get_style_profile_service

            service = get_style_profile_service()
            profiles = service.list_profiles()

            return {
                "total_profiles": len(profiles),
                "profiles_dir": service.profiles_dir,
                "profiles": {
                    p.label: {
                        "description": p.description,
                        "tags": p.tags,
                        "reference_image_count": p.reference_image_count,
                        "has_post_processing": p.has_post_processing,
                    }
                    for p in profiles
                },
            }

        except Exception as e:
            logger.error(f"Error listing style profiles: {e}")
            return {"error": "style_profiles_error", "message": str(e), "profiles": {}}

    @server.resource("nano-banana://style-profiles/{label}")
    def style_profile_detail(label: str) -> dict[str, Any]:
        """Get details of a specific style profile."""
        logger = logging.getLogger(__name__)

        try:
            from ..services import get_style_profile_service

            service = get_style_profile_service()
            profile = service.load_profile(label)
            ref_paths = service.get_reference_image_paths(profile)

            return {
                "profile": profile.model_dump(),
                "reference_image_paths": ref_paths,
                "reference_image_count": len(ref_paths),
            }

        except FileNotFoundError:
            logger.error(f"Style profile not found: {label}")
            return {
                "error": "profile_not_found",
                "message": f"Profile '{label}' not found",
                "label": label,
            }
        except Exception as e:
            logger.error(f"Error loading style profile '{label}': {e}")
            return {"error": "style_profile_error", "message": str(e), "label": label}
