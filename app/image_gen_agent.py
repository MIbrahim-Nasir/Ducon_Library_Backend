"""
Unified prompt writer + post-generation evaluator for the autogenerate pipeline.

One Gemini session with shared memory:
  inputs → write prompt → Nano Banana → evaluate (same agent) → retry or return
  → (background) post-success system prompt improvement
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
from typing import Any, Optional

from PIL import Image
from google.genai.types import Content, GenerateContentConfig, Part, ThinkingConfig

from app.gemini import (
    IMAGE_GEN_MODEL,
    PROMPT_GEN_MODEL,
    _EVAL_CRITICAL_SECTIONS,
    _PROMPT_THINKING_LEVEL,
    _eval_section_passes,
    _parse_json_response,
    get_gemini_client,
    log_section_analysis,
)
from app import prompt_loader

_AUTO_LEARN_ENABLED = os.getenv("IMAGE_GEN_AGENT_AUTO_LEARN", "true").lower() not in (
    "0",
    "false",
    "no",
)


def _strip_prompt_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[\w]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    return text


def _pil_part(img: Image.Image) -> Part:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    return Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


class ImageGenAgent:
    """Single-session agent for prompt writing and post-generation QC."""

    def __init__(
        self,
        *,
        image1_name: str,
        image2_name: str | None,
        image2_is_user_space: bool,
    ) -> None:
        self._conversation: list[Content] = []
        self.last_prompt: str = ""
        self.image1_name = image1_name
        self.image2_name = image2_name
        self.image2_is_user_space = image2_is_user_space
        self.rejection_count: int = 0
        self.successful_round: int = 0

    def schedule_post_success_improvement(self) -> None:
        """
        Fire-and-forget: analyze session failures and maybe update image-gen-agent.md.
        Does not block the generation HTTP response.
        """
        if not _AUTO_LEARN_ENABLED:
            return
        if self.rejection_count == 0 and self.successful_round <= 1:
            print("[ImageGenAgent] Skipping prompt improvement — first-pass success")
            return
        asyncio.create_task(
            self._run_post_success_improvement(),
            name="image_gen_agent_post_success_improvement",
        )

    async def _run_post_success_improvement(self) -> None:
        try:
            result = await self.improve_system_prompt_after_success()
            if result.get("should_update"):
                print(
                    f"[ImageGenAgent] System prompt updated: {result.get('reason')}"
                )
            else:
                print(
                    f"[ImageGenAgent] System prompt unchanged: {result.get('reason')}"
                )
        except Exception as exc:
            print(f"[ImageGenAgent] Post-success improvement failed: {exc}")

    async def improve_system_prompt_after_success(self) -> dict[str, Any]:
        """
        Same-session meta turn after final approval.
        May persist a revised image-gen-agent system prompt to disk.
        """
        prompt_loader.ensure_prompts_loaded()
        learn_context = f"""
POST-SUCCESS IMPROVEMENT — generation finally APPROVED on round {self.successful_round}.

Session summary:
- Total QC rejections before success: {self.rejection_count}
- Final approved generation prompt:
\"\"\"
{self.last_prompt}
\"\"\"

Current standing system prompt (your instructions for all future sessions):
\"\"\"
{prompt_loader.IMAGE_GEN_AGENT_SYSTEM}
\"\"\"

Analyze this session's failures, retry prompts, and what worked in the final approval.
Decide whether the standing system prompt should be updated for future Nano Banana work.
Return POST-SUCCESS IMPROVEMENT JSON only.
""".strip()

        self._conversation.append(Content(role="user", parts=[Part(text=learn_context)]))

        client = get_gemini_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=PROMPT_GEN_MODEL,
            contents=self._conversation,
            config=GenerateContentConfig(
                system_instruction=prompt_loader.IMAGE_GEN_AGENT_SYSTEM,
                response_mime_type="application/json",
                thinking_config=ThinkingConfig(thinking_level=_PROMPT_THINKING_LEVEL),
            ),
        )

        data = _parse_json_response(response.text)
        self._conversation.append(
            Content(role="model", parts=[Part(text=response.text or "")])
        )

        should_update = bool(data.get("should_update"))
        updated = data.get("updated_system_prompt")
        print(
            f"[ImageGenAgent] post-success learn should_update={should_update} "
            f"reason={data.get('reason')}"
        )
        analysis = data.get("analysis") or {}
        if analysis.get("genuinely_new_learnings"):
            print(
                "[ImageGenAgent] new learnings: "
                f"{analysis.get('genuinely_new_learnings')}"
            )

        if should_update and isinstance(updated, str) and updated.strip():
            await prompt_loader.persist_image_gen_agent_system(updated.strip())
        elif should_update:
            print("[ImageGenAgent] should_update=true but no updated_system_prompt — skipped")

        return data

    async def generate_initial_prompt(
        self,
        *,
        image1: Image.Image,
        image2: Image.Image,
        image1_metadata: dict[str, Any] | None = None,
        image2_metadata: dict[str, Any] | None = None,
        user_hint: str | None = None,
    ) -> str:
        if self.image2_is_user_space:
            context = (
                f'Image 1 is a Ducon catalog image named "{self.image1_name}". '
                f"Image 2 is the user's outdoor space to transform. "
                f"Write the complete Nano Banana Pro generation prompt."
            )
        else:
            context = (
                f'Image 1 is a Ducon catalog image named "{self.image1_name}". '
                f'Image 2 is a Ducon catalog image named "{self.image2_name}". '
                f"Write the complete Nano Banana Pro generation prompt to combine them."
            )

        if image1_metadata:
            context += f"\n\nMetadata for Image 1:\n{json.dumps(image1_metadata, indent=2)}"
        if image2_metadata:
            context += f"\n\nMetadata for Image 2:\n{json.dumps(image2_metadata, indent=2)}"
        if user_hint:
            context += (
                f'\n\nUser direction (incorporate when consistent with images): "{user_hint}"'
            )

        context += (
            "\n\nReturn ONLY the full generation prompt text. Self-check all V1–V11 rules "
            "before responding."
        )

        parts: list[Part] = [_pil_part(image1), _pil_part(image2), Part(text=context)]
        self._conversation.append(Content(role="user", parts=parts))
        return await self._complete_prompt_turn()

    async def evaluate_output(
        self,
        *,
        generated: Image.Image,
        gen_round: int,
    ) -> tuple[bool, str | None, list[str]]:
        """Evaluate the latest generation; on rejection return a revised prompt."""
        eval_context = (
            f"Generation round {gen_round} — evaluate the LAST image (AI output from "
            f"{IMAGE_GEN_MODEL}) against your previous prompt and the input images from "
            f"this session.\n\n"
            f'Image 1 (Ducon reference): "{self.image1_name}". '
        )
        if self.image2_is_user_space:
            eval_context += "Image 2: user's outdoor space."
        else:
            eval_context += f'Image 2 (Ducon): "{self.image2_name}".'
        eval_context += (
            f'\n\nPrompt used:\n"""\n{self.last_prompt}\n"""\n\n'
            "For every section: fill section_analysis (aspect → reference observation → "
            "generated observation → evaluation → verdict) then section_results. "
            "section_analysis is REQUIRED for all sections on both approval and rejection. "
            "Return evaluation JSON only. If rejected, include revised_prompt with "
            "nano_banana_analysis-informed fixes."
        )

        parts = [_pil_part(generated), Part(text=eval_context)]
        self._conversation.append(Content(role="user", parts=parts))

        prompt_loader.ensure_prompts_loaded()
        client = get_gemini_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=PROMPT_GEN_MODEL,
            contents=self._conversation,
            config=GenerateContentConfig(
                system_instruction=prompt_loader.IMAGE_GEN_AGENT_SYSTEM,
                response_mime_type="application/json",
                thinking_config=ThinkingConfig(thinking_level=_PROMPT_THINKING_LEVEL),
            ),
        )

        data = _parse_json_response(response.text)
        self._conversation.append(
            Content(role="model", parts=[Part(text=response.text or "")])
        )

        issues: list[str] = data.get("issues") or []
        section_results = data.get("section_results") or {}
        failed_sections = [
            key
            for key in _EVAL_CRITICAL_SECTIONS
            if not _eval_section_passes(section_results.get(key))
        ]
        if failed_sections:
            issues = [
                *issues,
                f"Critical QC sections failed or missing: {', '.join(failed_sections)}",
            ]

        approved = data.get("verdict") == "approved" and not issues
        revised_prompt: str | None = None
        if approved:
            self.successful_round = gen_round
        else:
            self.rejection_count += 1
            revised = data.get("revised_prompt")
            if isinstance(revised, str) and revised.strip():
                revised_prompt = revised.strip()

        nb_analysis = data.get("nano_banana_analysis")
        print(
            f"[ImageGenAgent] round={gen_round} verdict={data.get('verdict')} "
            f"reason={data.get('reason')}"
        )
        log_section_analysis(data.get("section_analysis"), section_results)
        if nb_analysis:
            print(f"[ImageGenAgent] nano_banana_analysis: {nb_analysis}")
        if issues:
            print(f"[ImageGenAgent] issues: {issues}")
        if revised_prompt:
            self.last_prompt = revised_prompt
            print(f"[ImageGenAgent] revised_prompt (first 400 chars):\n{revised_prompt[:400]}\n")

        return approved, revised_prompt, issues

    async def _complete_prompt_turn(self) -> str:
        prompt_loader.ensure_prompts_loaded()
        client = get_gemini_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=PROMPT_GEN_MODEL,
            contents=self._conversation,
            config=GenerateContentConfig(
                system_instruction=prompt_loader.IMAGE_GEN_AGENT_SYSTEM,
                thinking_config=ThinkingConfig(thinking_level=_PROMPT_THINKING_LEVEL),
            ),
        )
        text = _strip_prompt_fences(response.text or "")
        if not text:
            raise RuntimeError("ImageGenAgent returned empty prompt.")

        self._conversation.append(Content(role="model", parts=[Part(text=text)]))
        self.last_prompt = text
        print(f"[ImageGenAgent] initial prompt (first 400 chars):\n{text[:400]}\n")
        return text
