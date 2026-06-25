"""
Multi-turn prompt generator session for designer jobs.

Designer jobs still use a separate prompt writer session plus stateless QC.
The autogenerate pipeline uses ImageGenAgent (app/image_gen_agent.py) for a
single unified prompt + evaluate session.
"""
from __future__ import annotations

import asyncio
import io
import json
import re
from typing import Any, Optional

from PIL import Image
from google.genai.types import Content, GenerateContentConfig, Part, ThinkingConfig

from app.gemini import get_gemini_client, PROMPT_GEN_MODEL, _PROMPT_THINKING_LEVEL
from app import prompt_loader


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


def _labels_context(labels: list[str]) -> str:
    return "\n".join(f"  Image {i}: {label}" for i, label in enumerate(labels, start=1))


class DesignerPromptSession:
    """Stateful prompt writer for one designer job (all attempts and inner retries)."""

    def __init__(self) -> None:
        self._conversation: list[Content] = []
        self.last_prompt: str = ""
        self.revision_count: int = 0

    async def generate_initial(
        self,
        *,
        images: list[Image.Image],
        labels: list[str],
        plan: dict[str, Any],
        design_hint: str,
        attempt_no: int = 1,
    ) -> str:
        """First prompt in the job (includes reference images in session history)."""
        preserve = ", ".join(plan.get("preserve") or []) or "important architecture and layout"
        success = ", ".join(plan.get("success_criteria") or [])
        plan_prompt = plan.get("generation_prompt") or design_hint or ""

        user_text = f"""
Designer job — write the generation prompt (attempt {attempt_no}).

Input images for the generator (same order as above):
{_labels_context(labels)}

Design plan:
{json.dumps(plan, ensure_ascii=False)}

Preserve: {preserve}
Success criteria: {success}

Planning-phase prompt suggestion (refine into a complete Nano Banana Pro prompt):
{plan_prompt}

User direction: {design_hint or "Choose a tasteful Ducon concept."}

Write the complete generation prompt now. Use the same rigorous structure that
made the earlier Ducon single-image generations reliable: visual observation,
camera lock, fixed-structure preservation, zone mapping, direct visual reference
for complex patterns, no creative licence, and photorealistic close.
""".strip()

        parts: list[Part] = [_pil_part(img) for img in images]
        parts.append(Part(text=user_text))
        self._conversation.append(Content(role="user", parts=parts))
        return await self._complete_turn()

    async def revise_from_qc(
        self,
        evaluation: dict[str, Any],
        *,
        context_label: str = "Post-generation QC",
    ) -> str:
        """
        Continue the session after a stateless QC review failed.
        `evaluation` should carry issues, improvements, section_results, score, etc.
        """
        feedback = {
            "context": context_label,
            "your_previous_prompt": self.last_prompt,
            "score": evaluation.get("score"),
            "passed": evaluation.get("passed"),
            "issues": evaluation.get("issues") or [],
            "improvements": evaluation.get("improvements"),
            "reference_match_issues": evaluation.get("reference_match_issues") or [],
            "client_space_preservation_issues": evaluation.get(
                "client_space_preservation_issues"
            ) or [],
            "hallucinations": evaluation.get("hallucinations") or [],
            "section_results": evaluation.get("section_results"),
            "strengths": evaluation.get("strengths") or [],
        }
        # Inner-loop evaluator uses verdict/issues without improvements text
        if evaluation.get("verdict") == "rejected" and not feedback["improvements"]:
            feedback["improvements"] = "; ".join(feedback["issues"]) or "Address all QC issues."

        user_text = f"""
{context_label} reviewed the image produced with your previous prompt and rejected it.

Stateless QC feedback (treat as ground truth for what failed):
{json.dumps(feedback, ensure_ascii=False)}

Revise your prompt to fix every failed criterion. Keep what worked. Return the full new prompt.
""".strip()

        self._conversation.append(Content(role="user", parts=[Part(text=user_text)]))
        return await self._complete_turn()

    def record_external_revision(self, prompt: str, *, source: str, issues: Optional[list[str]] = None) -> None:
        """
        Keep session memory aligned when a stateless verifier rewrites the prompt.
        The next revision turn should remember the exact prompt that was actually
        sent to image generation, not only the prompt this session originally wrote.
        """
        prompt = _strip_prompt_fences(prompt)
        if not prompt:
            return

        note = {
            "source": source,
            "issues": issues or [],
            "current_prompt": prompt,
        }
        self._conversation.append(Content(
            role="user",
            parts=[Part(text=(
                "A stateless quality gate revised the prompt before image generation. "
                "Treat this as the current prompt state for future retries:\n"
                f"{json.dumps(note, ensure_ascii=False)}"
            ))],
        ))
        self._conversation.append(Content(role="model", parts=[Part(text=prompt)]))
        self.last_prompt = prompt

    async def _complete_turn(self) -> str:
        prompt_loader.ensure_prompts_loaded()
        client = get_gemini_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=PROMPT_GEN_MODEL,
            contents=self._conversation,
            config=GenerateContentConfig(
                system_instruction=prompt_loader.DESIGNER_PROMPT_WRITER_SYSTEM,
                thinking_config=ThinkingConfig(thinking_level=_PROMPT_THINKING_LEVEL),
            ),
        )
        text = _strip_prompt_fences(response.text or "")
        if not text:
            raise RuntimeError("Prompt generator returned empty text.")

        self._conversation.append(Content(role="model", parts=[Part(text=text)]))
        self.last_prompt = text
        self.revision_count += 1
        return text
