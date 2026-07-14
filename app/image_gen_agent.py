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
import logging
import os
import re
from typing import Any, Optional

from PIL import Image
from google.genai.types import Content, GenerateContentConfig, Part, ThinkingConfig

from app.gemini import (
    image_gen_model,
    prompt_gen_model,
    _PROMPT_THINKING_LEVEL,
    _parse_json_response,
    extract_input_quality,
    finalize_evaluation,
    get_gemini_client,
    log_section_analysis,
)
from app import prompt_loader

logger = logging.getLogger(__name__)
from app import llm_provider
from app.admin.settings_store import cfg

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.benchmark.types import StepMetrics

# Auto-learn rewrites image-gen-agent.md after successful (post-retry) runs.
# It is OFF by default: an unconstrained "improve your own system prompt" loop
# previously summarised the detailed evaluation rubric down to placeholders
# (e.g. "[Evaluation sections A-E remain as defined previously]"), which broke
# QC and caused chronic false rejections. Only enable it deliberately.
_AUTO_LEARN_ENABLED = str(cfg("IMAGE_GEN_AGENT_AUTO_LEARN", "false")).lower() in (
    "1",
    "true",
    "yes",
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
        labels: list[str] | None = None,
        user_space_index: int | None = None,
        design_direction_index: int | None = None,
        product_indices: list[int] | None = None,
        prompt_model: Optional[str] = None,
        prompt_thinking: Optional[str] = None,
        max_eval_rounds: Optional[int] = None,
    ) -> None:
        self._conversation: list[Content] = []
        # Parallel transcript used when USE_CLAUDE is enabled (Anthropic messages).
        self._claude_messages: list[dict] = []
        self.last_prompt: str = ""
        self.image1_name = image1_name
        self.image2_name = image2_name
        self.image2_is_user_space = image2_is_user_space
        self.rejection_count: int = 0
        self.successful_round: int = 0
        # Captured once on the first evaluation; the input photo does not change
        # between retries, so we keep the first assessment to surface to the user.
        self.input_quality: dict | None = None

        # Per-call overrides (benchmark). None ⇒ fall back to live cfg().
        self._prompt_model = prompt_model
        self._prompt_thinking = prompt_thinking
        self._max_eval_rounds = max_eval_rounds

        if labels:
            self.labels = labels
            self.user_space_index = user_space_index if user_space_index is not None else 0
            self.design_direction_index = (
                design_direction_index if design_direction_index is not None else 1
            )
            self.product_indices = product_indices or []
        else:
            self.labels = [image1_name, image2_name or "user space"]
            self.user_space_index = 1 if image2_is_user_space else len(self.labels) - 1
            self.design_direction_index = 0
            self.product_indices = []

    def _effective_prompt_model(self) -> str:
        return self._prompt_model or prompt_gen_model()

    def _effective_prompt_thinking(self) -> str:
        return self._prompt_thinking or _PROMPT_THINKING_LEVEL

    def _effective_max_eval_rounds(self) -> int:
        if self._max_eval_rounds is not None:
            return int(self._max_eval_rounds)
        from app.gemini import gen_eval_max_rounds
        return gen_eval_max_rounds()

    def _role_line(self, index: int) -> str:
        label = self.labels[index]
        img_num = index + 1
        if index == self.user_space_index:
            return f'Image {img_num}: user\'s outdoor space — "{label}"'
        if index == self.design_direction_index:
            return f'Image {img_num}: Ducon design direction — "{label}"'
        if index in self.product_indices:
            return f'Image {img_num}: Ducon product reference — "{label}"'
        return f'Image {img_num}: Ducon catalog reference — "{label}"'

    def _build_prompt_context(
        self,
        *,
        user_hint: str | None = None,
        image1_metadata: dict[str, Any] | None = None,
        image2_metadata: dict[str, Any] | None = None,
    ) -> str:
        if len(self.labels) > 2 or self.product_indices:
            role_block = "\n".join(
                f"- {self._role_line(i)}" for i in range(len(self.labels))
            )
            context = (
                "Multi-image studio generation. Input roles:\n"
                f"{role_block}\n\n"
                "Write the complete Nano Banana Pro generation prompt. Apply the design "
                "direction to eligible zones in the user's space. Integrate EACH product "
                "reference image as a distinct Ducon product with placement from its own image number."
            )
        elif self.image2_is_user_space:
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
            "\n\nReturn ONLY the full generation prompt text. Self-check all V1–V16 rules "
            "before responding."
        )
        return context

    # ── Provider-agnostic transcript helpers ────────────────────────────────
    async def _append_user_turn(self, pil_images: list[Image.Image], text: str) -> None:
        # pil_image_block / _pil_part do PIL JPEG encode (CPU-bound) — offload
        # per batch so the event loop keeps handling concurrent SSE streams.
        if llm_provider.use_claude():
            if pil_images:
                blocks = await asyncio.to_thread(
                    lambda: [llm_provider.pil_image_block(img) for img in pil_images]
                )
            else:
                blocks = []
            blocks.append(llm_provider.text_block(text))
            self._claude_messages.append({"role": "user", "content": blocks})
        else:
            if pil_images:
                parts_body = await asyncio.to_thread(
                    lambda: [_pil_part(img) for img in pil_images]
                )
            else:
                parts_body = []
            parts: list[Part] = parts_body + [Part(text=text)]
            self._conversation.append(Content(role="user", parts=parts))

    def _append_model_turn(self, text: str) -> None:
        if llm_provider.use_claude():
            self._claude_messages.append(
                {"role": "assistant", "content": [llm_provider.text_block(text)]}
            )
        else:
            self._conversation.append(Content(role="model", parts=[Part(text=text)]))

    async def _claude_complete(self, *, as_json: bool) -> str:
        """Run one Claude turn over the parallel transcript; returns raw text."""
        msg = await llm_provider.acomplete_message(
            system=prompt_loader.IMAGE_GEN_AGENT_SYSTEM,
            messages=self._claude_messages,
            thinking=True,
        )
        return llm_provider.extract_text(msg)

    def schedule_post_success_improvement(self) -> None:
        """
        Fire-and-forget: analyze session failures and maybe update image-gen-agent.md.
        Does not block the generation HTTP response.
        """
        if not _AUTO_LEARN_ENABLED:
            return
        if self.rejection_count == 0 and self.successful_round <= 1:
            logger.info("[ImageGenAgent] Skipping prompt improvement — first-pass success")
            return
        asyncio.create_task(
            self._run_post_success_improvement(),
            name="image_gen_agent_post_success_improvement",
        )

    async def _run_post_success_improvement(self) -> None:
        try:
            result = await self.improve_system_prompt_after_success()
            if result.get("should_update"):
                logger.info(
                    f"[ImageGenAgent] System prompt updated: {result.get('reason')}"
                )
            else:
                logger.info(
                    f"[ImageGenAgent] System prompt unchanged: {result.get('reason')}"
                )
        except Exception as exc:
            logger.warning("[ImageGenAgent] Post-success improvement failed: %s", exc)

    async def improve_system_prompt_after_success(
        self,
        *,
        metrics: "Optional[StepMetrics]" = None,
    ) -> dict[str, Any]:
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

        await self._append_user_turn([], learn_context)

        if llm_provider.use_claude():
            raw_text = await self._claude_complete(as_json=True)
            data = _parse_json_response(raw_text)
            self._append_model_turn(raw_text)
        else:
            client = get_gemini_client()
            _model = self._effective_prompt_model()
            _think = self._effective_prompt_thinking()
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=_model,
                contents=self._conversation,
                config=GenerateContentConfig(
                    system_instruction=prompt_loader.IMAGE_GEN_AGENT_SYSTEM,
                    response_mime_type="application/json",
                    thinking_config=ThinkingConfig(thinking_level=_think),
                ),
            )
            if metrics is not None:
                from app.gemini import _record_metrics
                _record_metrics(response, agent="bench_prompt", model=_model, metrics=metrics)
            data = _parse_json_response(response.text)
            self._conversation.append(
                Content(role="model", parts=[Part(text=response.text or "")])
            )

        should_update = bool(data.get("should_update"))
        updated = data.get("updated_system_prompt")
        logger.info(
            f"[ImageGenAgent] post-success learn should_update={should_update} "
            f"reason={data.get('reason')}"
        )
        analysis = data.get("analysis") or {}
        if analysis.get("genuinely_new_learnings"):
            logger.info(
                "[ImageGenAgent] new learnings: "
                f"{analysis.get('genuinely_new_learnings')}"
            )

        if should_update and isinstance(updated, str) and updated.strip():
            await prompt_loader.persist_image_gen_agent_system(updated.strip())
        elif should_update:
            logger.warning("[ImageGenAgent] should_update=true but no updated_system_prompt — skipped")

        return data

    async def generate_initial_prompt(
        self,
        *,
        image1: Image.Image,
        image2: Image.Image | None = None,
        images: list[Image.Image] | None = None,
        image1_metadata: dict[str, Any] | None = None,
        image2_metadata: dict[str, Any] | None = None,
        user_hint: str | None = None,
        metrics: "Optional[StepMetrics]" = None,
    ) -> str:
        pil_images = images if images else [image1, image2]  # type: ignore[list-item]
        context = self._build_prompt_context(
            user_hint=user_hint,
            image1_metadata=image1_metadata,
            image2_metadata=image2_metadata,
        )
        await self._append_user_turn(list(pil_images), context)
        return await self._complete_prompt_turn(metrics=metrics)

    async def evaluate_output(
        self,
        *,
        generated: Image.Image,
        gen_round: int,
        metrics: "Optional[StepMetrics]" = None,
    ) -> tuple[bool, list[str]]:
        """
        Evaluate the latest generation against the input images and the prompt
        that was used.  Returns (approved, issues).

        On rejection the conversation history is updated with the evaluation
        analysis so the subsequent generate_retry_prompt() call can synthesise
        a better prompt using the full session context.
        """
        eval_context = (
            f"Generation round {gen_round} — evaluate the LAST image (AI output from "
            f"{image_gen_model()}) against your previous prompt and the input images from "
            f"this session.\n\n"
            "Input image roles:\n"
            + "\n".join(f"- {self._role_line(i)}" for i in range(len(self.labels)))
            + f'\n\nPrompt used:\n"""\n{self.last_prompt}\n"""\n\n'
            "Evaluate against ALL input images. For B4_product_integration check EACH "
            "product reference separately (N/A only if no product images). "
            "For every section: fill section_analysis then section_results. "
            "If ANY section_results value is fail, verdict MUST be rejected. "
            "ALSO assess the user's space photo for BOTH capture quality AND "
            "suitability for the selected design direction/products (e.g. wrong "
            "space type, no eligible area, scene unsuited) and return the "
            "input_quality object (this never affects the verdict). "
            "Return evaluation JSON only."
        )

        prompt_loader.ensure_prompts_loaded()
        await self._append_user_turn([generated], eval_context)

        if llm_provider.use_claude():
            raw_text = await self._claude_complete(as_json=True)
            data = _parse_json_response(raw_text)
            self._append_model_turn(raw_text)
        else:
            client = get_gemini_client()
            _model = self._effective_prompt_model()
            _think = self._effective_prompt_thinking()
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=_model,
                contents=self._conversation,
                config=GenerateContentConfig(
                    system_instruction=prompt_loader.IMAGE_GEN_AGENT_SYSTEM,
                    response_mime_type="application/json",
                    thinking_config=ThinkingConfig(thinking_level=_think),
                ),
            )
            if metrics is not None:
                from app.gemini import _record_metrics
                _record_metrics(response, agent="bench_eval", model=_model, metrics=metrics)
            data = _parse_json_response(response.text)
            self._conversation.append(
                Content(role="model", parts=[Part(text=response.text or "")])
            )

        approved, issues, _inline_revised = finalize_evaluation(data)

        # Capture the input-photo quality assessment once (it does not change
        # across retries). Surfaced to the user as a non-blocking notice.
        if self.input_quality is None:
            iq = extract_input_quality(data)
            if iq is not None:
                self.input_quality = iq
                if iq.get("ok"):
                    logger.info("[ImageGenAgent] input_quality: ok (input photo looks fine)")
                else:
                    logger.info(
                        f"[ImageGenAgent] input_quality FLAGGED ({iq.get('severity')}): "
                        f"{iq.get('issues')}"
                    )
            else:
                logger.info("[ImageGenAgent] input_quality: model returned no assessment")

        if approved:
            self.successful_round = gen_round
        else:
            self.rejection_count += 1

        nb_analysis = data.get("nano_banana_analysis")
        logger.info(
            f"[ImageGenAgent] round={gen_round} verdict={data.get('verdict')} "
            f"reason={data.get('reason')}"
        )
        log_section_analysis(data.get("section_analysis"), data.get("section_results") or {})
        if nb_analysis:
            logger.info(f"[ImageGenAgent] nano_banana_analysis: {nb_analysis}")
        if issues:
            logger.info(f"[ImageGenAgent] issues: {issues}")

        return approved, issues

    async def generate_retry_prompt(
        self,
        *,
        gen_round: int,
        issues: list[str],
        metrics: "Optional[StepMetrics]" = None,
    ) -> str:
        """
        After an evaluation rejection, run a dedicated prompt-writing turn.

        The conversation already contains the evaluation analysis from
        evaluate_output(); this turn instructs the agent to synthesise that
        analysis into a fresh, improved generation prompt.  Returns the new
        prompt string and updates self.last_prompt.
        """
        issue_lines = "\n".join(f"- {i}" for i in issues) if issues else "- (see evaluation above)"
        retry_context = (
            f"Round {gen_round} was rejected.\n"
            f"Issues identified:\n{issue_lines}\n\n"
            "Using the evaluation analysis above, write an improved Nano Banana Pro "
            "generation prompt that fully addresses every identified issue. "
            "Apply all V1–V16 rules. Return ONLY the full generation prompt text."
        )
        await self._append_user_turn([], retry_context)
        prompt = await self._complete_prompt_turn(metrics=metrics)
        logger.info(f"[ImageGenAgent] retry prompt (round {gen_round}, first 400 chars):\n{prompt[:400]}\n")
        return prompt

    async def _complete_prompt_turn(
        self,
        *,
        metrics: "Optional[StepMetrics]" = None,
    ) -> str:
        prompt_loader.ensure_prompts_loaded()
        if llm_provider.use_claude():
            text = _strip_prompt_fences(await self._claude_complete(as_json=False))
        else:
            client = get_gemini_client()
            _model = self._effective_prompt_model()
            _think = self._effective_prompt_thinking()
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=_model,
                contents=self._conversation,
                config=GenerateContentConfig(
                    system_instruction=prompt_loader.IMAGE_GEN_AGENT_SYSTEM,
                    thinking_config=ThinkingConfig(thinking_level=_think),
                ),
            )
            if metrics is not None:
                from app.gemini import _record_metrics
                _record_metrics(response, agent="bench_prompt", model=_model, metrics=metrics)
            text = _strip_prompt_fences(response.text or "")
        if not text:
            raise RuntimeError("ImageGenAgent returned empty prompt.")

        self._append_model_turn(text)
        self.last_prompt = text
        logger.info(f"[ImageGenAgent] initial prompt (first 400 chars):\n{text[:400]}\n")
        return text
