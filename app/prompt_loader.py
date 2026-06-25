"""
Load AI system prompts from app/prompts/*.md at server startup.

Each prompt lives in its own markdown file for easy editing. Call load_prompts()
once during application startup before any Gemini calls run.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# ── Gemini image / quotation pipeline ─────────────────────────────────────────
PROMPT_SYSTEM_INSTRUCTION: str = ""
PROMPT_VERIFY_SYSTEM_INSTRUCTION: str = ""
EVAL_SYSTEM_INSTRUCTION: str = ""
QUOTATION_SYSTEM_INSTRUCTION: str = ""
QUOTATION_SYNTHESIS_INSTRUCTION: str = ""

# ── Agents ────────────────────────────────────────────────────────────────────
IMAGE_GEN_AGENT_SYSTEM: str = ""
DESIGNER_PROMPT_WRITER_SYSTEM: str = ""
CHAT_AGENT_SYSTEM: str = ""
LIVE_VOICE_AGENT_SYSTEM: str = ""

# ── Designer job templates (user-message bodies with optional placeholders) ───
DESIGNER_ANALYZE_PLAN: str = ""
DESIGNER_EVALUATE_GENERATION: str = ""
DESIGNER_FINAL_SUMMARY: str = ""
DESIGNER_COMPOSE_GENERATION_SUFFIX: str = ""
STUDIO_DIRECTIONS_AGENT_SYSTEM: str = ""

_PROMPT_FILES: dict[str, str] = {
    "PROMPT_SYSTEM_INSTRUCTION": "prompt-generator.md",
    "PROMPT_VERIFY_SYSTEM_INSTRUCTION": "prompt-verifier.md",
    "EVAL_SYSTEM_INSTRUCTION": "generation-evaluator.md",
    "QUOTATION_SYSTEM_INSTRUCTION": "quotation-analyzer.md",
    "QUOTATION_SYNTHESIS_INSTRUCTION": "quotation-synthesis.md",
    "IMAGE_GEN_AGENT_SYSTEM": "image-gen-agent.md",
    "DESIGNER_PROMPT_WRITER_SYSTEM": "designer-prompt-writer.md",
    "CHAT_AGENT_SYSTEM": "chat-agent.md",
    "LIVE_VOICE_AGENT_SYSTEM": "live-voice-agent.md",
    "DESIGNER_ANALYZE_PLAN": "designer-analyze-plan.md",
    "DESIGNER_EVALUATE_GENERATION": "designer-evaluate-generation.md",
    "DESIGNER_FINAL_SUMMARY": "designer-final-summary.md",
    "DESIGNER_COMPOSE_GENERATION_SUFFIX": "designer-compose-generation-suffix.md",
    "STUDIO_DIRECTIONS_AGENT_SYSTEM": "studio-directions-agent.md",
}

_loaded = False
_image_gen_agent_prompt_file = "image-gen-agent.md"
_prompt_update_lock = asyncio.Lock()


def _read_prompt_file(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8").strip()


def _apply_placeholders(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def load_prompts(
    *,
    image_gen_model: str | None = None,
    designer_pass_score: float | str | None = None,
) -> None:
    """Read all prompt markdown files into module-level variables."""
    global _loaded
    global PROMPT_SYSTEM_INSTRUCTION
    global PROMPT_VERIFY_SYSTEM_INSTRUCTION
    global EVAL_SYSTEM_INSTRUCTION
    global QUOTATION_SYSTEM_INSTRUCTION
    global QUOTATION_SYNTHESIS_INSTRUCTION
    global IMAGE_GEN_AGENT_SYSTEM
    global DESIGNER_PROMPT_WRITER_SYSTEM
    global CHAT_AGENT_SYSTEM
    global LIVE_VOICE_AGENT_SYSTEM
    global DESIGNER_ANALYZE_PLAN
    global DESIGNER_EVALUATE_GENERATION
    global DESIGNER_FINAL_SUMMARY
    global DESIGNER_COMPOSE_GENERATION_SUFFIX
    global STUDIO_DIRECTIONS_AGENT_SYSTEM

    image_gen_model = image_gen_model or os.getenv(
        "IMAGE_GEN_MODEL", "gemini-3-pro-image-preview"
    )
    designer_pass_score = (
        designer_pass_score
        if designer_pass_score is not None
        else os.getenv("DESIGNER_AGENT_PASS_SCORE", "7.5")
    )

    placeholders = {
        "IMAGE_GEN_MODEL": str(image_gen_model),
        "DESIGNER_AGENT_PASS_SCORE": str(designer_pass_score),
    }

    for attr, filename in _PROMPT_FILES.items():
        text = _read_prompt_file(filename)
        if "{{" in text:
            text = _apply_placeholders(text, placeholders)
        globals()[attr] = text

    _loaded = True


def ensure_prompts_loaded() -> None:
    if not _loaded:
        load_prompts()


def render_prompt_template(template: str, **values: str) -> str:
    """Replace {{NAME}} placeholders in a loaded template string."""
    return _apply_placeholders(template, values)


def _image_gen_model_name() -> str:
    return os.getenv("IMAGE_GEN_MODEL", "gemini-3-pro-image-preview")


def _prepare_image_gen_agent_system_for_storage(text: str) -> str:
    """Store model id as {{IMAGE_GEN_MODEL}} placeholder in the markdown file."""
    normalized = text.strip()
    model = _image_gen_model_name()
    if model and model in normalized:
        normalized = normalized.replace(model, "{{IMAGE_GEN_MODEL}}")
    return normalized


def _validate_image_gen_agent_system(text: str) -> None:
    if len(text) < 500:
        raise ValueError("Updated system prompt is too short.")
    if len(text) > 80_000:
        raise ValueError("Updated system prompt exceeds size limit.")
    upper = text.upper()
    for marker in ("PHASE 1", "PHASE 2", "PHASE 3"):
        if marker not in upper:
            raise ValueError(f"Updated system prompt missing {marker}.")


async def persist_image_gen_agent_system(updated_prompt: str) -> None:
    """
    Write image-gen-agent.md and hot-reload IMAGE_GEN_AGENT_SYSTEM in memory.
    Serialized so concurrent background learners do not corrupt the file.
    """
    prepared = _prepare_image_gen_agent_system_for_storage(updated_prompt)
    _validate_image_gen_agent_system(prepared)

    async with _prompt_update_lock:
        path = PROMPTS_DIR / _image_gen_agent_prompt_file
        path.write_text(prepared + "\n", encoding="utf-8")
        global IMAGE_GEN_AGENT_SYSTEM
        IMAGE_GEN_AGENT_SYSTEM = _apply_placeholders(
            prepared,
            {
                "IMAGE_GEN_MODEL": _image_gen_model_name(),
                "DESIGNER_AGENT_PASS_SCORE": os.getenv("DESIGNER_AGENT_PASS_SCORE", "7.5"),
            },
        )
        print(
            f"[PromptLoader] Updated {_image_gen_agent_prompt_file} "
            f"({len(IMAGE_GEN_AGENT_SYSTEM)} chars in memory)"
        )
