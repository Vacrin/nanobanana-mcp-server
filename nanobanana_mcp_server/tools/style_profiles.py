"""Style profile management tool."""

import logging
from typing import Annotated

from fastmcp import Context, FastMCP
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent
from pydantic import Field

from ..core.exceptions import ValidationError


def register_style_profiles_tool(server: FastMCP):
    """Register the style_profiles tool with the FastMCP server."""

    @server.tool(
        annotations={
            "title": "Manage style profiles for consistent image generation",
            "readOnlyHint": True,
            "openWorldHint": False,
        }
    )
    def style_profiles(
        operation: Annotated[
            str,
            Field(
                description="Operation: 'list' (all profiles), 'get' (profile details), "
                "'validate' (check for errors), 'init' (scaffold new profile)"
            ),
        ],
        label: Annotated[
            str | None,
            Field(
                description="Profile label (required for get/validate/init). "
                "Lowercase alphanumeric with hyphens/underscores, e.g. 'pixel-forest-iso'.",
                max_length=64,
            ),
        ] = None,
        _ctx: Context | None = None,
    ) -> ToolResult:
        """
        Manage style profiles for consistent asset generation.

        Style profiles bundle system instructions, negative prompts, reference images,
        generation defaults, and post-processing settings into a reusable named profile.

        Use 'init' to scaffold a new profile, then edit profile.json and add reference
        images to the references/ folder. Use the style_profile parameter on generate_image
        to apply a profile.
        """
        logger = logging.getLogger(__name__)

        try:
            valid_operations = ["list", "get", "validate", "init"]
            if operation not in valid_operations:
                raise ValidationError(
                    f"Invalid operation. Must be one of: {', '.join(valid_operations)}"
                )

            if operation != "list" and not label:
                raise ValidationError(
                    f"Operation '{operation}' requires a label parameter"
                )

            service = _get_style_profile_service()

            if operation == "list":
                profiles = service.list_profiles()
                if not profiles:
                    summary = (
                        "No style profiles found.\n\n"
                        f"Profiles directory: {service.profiles_dir}\n"
                        "Use style_profiles(operation='init', label='my-style') "
                        "to create one."
                    )
                else:
                    lines = [f"Found {len(profiles)} style profile(s):\n"]
                    for p in profiles:
                        tags = ", ".join(p.tags) if p.tags else "none"
                        post = "resize enabled" if p.has_post_processing else "none"
                        lines.append(
                            f"  **{p.label}** — {p.description or '(no description)'}\n"
                            f"    Tags: {tags} | Refs: {p.reference_image_count} | "
                            f"Post-processing: {post}"
                        )
                    lines.append(f"\nProfiles directory: {service.profiles_dir}")
                    summary = "\n".join(lines)

                structured = {
                    "operation": "list",
                    "profiles_dir": service.profiles_dir,
                    "profiles": [p.model_dump() for p in profiles],
                    "count": len(profiles),
                }

            elif operation == "get":
                profile = service.load_profile(label)
                ref_paths = service.get_reference_image_paths(profile)

                lines = [
                    f"**{profile.label}** — {profile.description or '(no description)'}",
                    "",
                ]
                if profile.style_description:
                    lines.append(f"**Style:** {profile.style_description}")
                if profile.system_instruction:
                    lines.append(
                        f"**System instruction:** {profile.system_instruction[:200]}"
                        f"{'...' if len(profile.system_instruction or '') > 200 else ''}"
                    )
                if profile.negative_prompt:
                    lines.append(f"**Negative prompt:** {profile.negative_prompt}")

                lines.append(f"\n**Defaults:** {profile.defaults.model_dump(exclude_none=True)}")

                if profile.post_processing.resize.enabled:
                    r = profile.post_processing.resize
                    lines.append(
                        f"**Post-processing:** resize to {r.width}x{r.height} "
                        f"({r.resample}), format: {profile.post_processing.format}"
                    )

                if ref_paths:
                    lines.append(f"\n**Reference images ({len(ref_paths)}):**")
                    for rp in ref_paths:
                        lines.append(f"  - {rp}")
                else:
                    lines.append("\nNo reference images found.")

                if profile.tags:
                    lines.append(f"\n**Tags:** {', '.join(profile.tags)}")

                summary = "\n".join(lines)
                structured = {
                    "operation": "get",
                    "profile": profile.model_dump(),
                    "reference_image_paths": ref_paths,
                }

            elif operation == "validate":
                result = service.validate_profile(label)

                if result.valid:
                    lines = [f"Profile '{label}' is valid."]
                else:
                    lines = [f"Profile '{label}' has errors:"]
                    for err in result.errors:
                        lines.append(f"  - {err}")

                if result.warnings:
                    lines.append("\nWarnings:")
                    for warn in result.warnings:
                        lines.append(f"  - {warn}")

                summary = "\n".join(lines)
                structured = {
                    "operation": "validate",
                    "label": label,
                    "result": result.model_dump(),
                }

            else:  # init
                path = service.init_profile(label)
                summary = (
                    f"Created style profile scaffold at:\n  {path}\n\n"
                    "Next steps:\n"
                    "1. Edit profile.json with your style description, "
                    "system instruction, and defaults\n"
                    "2. Add reference images to the references/ folder\n"
                    "3. Use generate_image(prompt='...', style_profile='"
                    f"{label}') to generate with this profile"
                )
                structured = {
                    "operation": "init",
                    "label": label,
                    "path": path,
                }

            content = [TextContent(type="text", text=summary)]
            return ToolResult(content=content, structured_content=structured)

        except ValidationError as e:
            logger.error(f"Validation error in style_profiles: {e}")
            raise
        except FileNotFoundError as e:
            logger.error(f"Profile not found: {e}")
            raise ValidationError(str(e)) from e
        except FileExistsError as e:
            logger.error(f"Profile already exists: {e}")
            raise ValidationError(str(e)) from e
        except Exception as e:
            logger.error(f"Unexpected error in style_profiles: {e}")
            raise


def _get_style_profile_service():
    """Get the style profile service instance."""
    from ..services import get_style_profile_service

    return get_style_profile_service()
