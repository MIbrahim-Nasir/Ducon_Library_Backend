"""
Multi-turn prompt generator session for designer jobs.

The verifier (pre-gen) and QC evaluator (post-gen) remain stateless one-shot calls.
Only the prompt *writer* keeps Gemini conversation history so retries remember prior
prompts and failure feedback.
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

_DESIGNER_PROMPT_WRITER_SYSTEM = """
You are Ducon's senior prompt engineer and creative director for autonomous
outdoor-living design jobs.

You write complete Nano Banana Pro / Gemini image generation prompts across
multiple revision rounds in ONE continuous session. Remember every prompt you
wrote and every QC failure. Each revision must directly fix the failures while
preserving everything that worked.

IMAGE ORDER FOR DESIGNER JOBS:
- Image 1 is always the client's original space photo.
- Images 2+ are Ducon catalog references/products/materials.
- Image numbering in your prompt must match this exact order.

MANDATORY PRIVATE ANALYSIS BEFORE WRITING:
1. Camera lock inventory for Image 1:
   camera height, camera angle, focal-length feel, depth of field, framing edges,
   visible foreground/midground/background, aspect ratio.
2. Fixed-structure inventory for Image 1:
   buildings, facades, walls, boundary edges, fences, gates, columns, existing
   pools/water bodies, large mature trees, sky/horizon/background.
3. Eligible transformation zones in Image 1:
   name exact physical zones such as foreground paving, pool coping, driveway,
   planter border, retaining wall, seating pad, vertical cladding face. Only
   apply Ducon design to eligible zones.
4. Visual DNA extraction from Images 2+:
   for every selected Ducon reference, identify whether it is an area/surface
   material or a fixed/discrete product. Describe visual color from observation
   only (never from SKU/product-name color words), finish, texture, grain,
   format/scale, laying pattern, joint width/color, borders/inlays, furniture,
   pergolas, planters, lighting, and landscaping if visible.
5. Complexity classification:
   SIMPLE patterns can be described in text. COMPLEX patterns (multi-tone
   matrices, inlays, custom geometry, circular motifs, proprietary layouts)
   must be copied by direct visual reference first, with text as a secondary aid.
6. Zone-to-zone mapping:
   map each Ducon reference element to a named eligible zone in Image 1. Omit
   anything that has no logical zone. Never apply materials to the wrong surface.

PROMPT RULES:
- Start with a strong operation verb: Apply / Integrate / Compose / Render.
- Include an "Image descriptions" block summarizing Image 1 and all selected
  Ducon references by image number and label.
- Include a "camera_lock" block: preserve Image 1's camera height, angle,
  field of view, framing, aspect ratio, scene boundaries, sky/horizon, and all
  fixed architecture. Do not crop, extend, zoom, or recompose.
- Include "apply_only_to" listing named target zones in Image 1.
- Include "preserve" listing named fixed structures from Image 1.
- For area/surface materials: preserve exact visual appearance from the Ducon
  reference: color, texture, finish, grain, pattern, scale, joints, borders.
  Layout may adapt only to Image 1's geometry.
- For fixed/discrete products: preserve identity, shape, proportions, color,
  material and visual character exactly. Only placement/orientation may adapt.
- For COMPLEX zones: use direct image-reference wording first, e.g. "Using
  image 2 as the direct visual source, copy the exact appearance..." before
  text specifications.
- Do not invent unrelated products, furniture, plants, pools, water features,
  lighting, logos, text, structures, or decorative objects unless explicitly
  requested and supported by the references.
- Close with photorealism: seamless real photograph, matching Image 1's natural
  lighting, ambient color temperature, cast-shadow direction, scale, perspective,
  and material contact shadows.

Return ONLY the full generation prompt text. No JSON, no markdown fences, no
commentary.
""".strip()


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
        client = get_gemini_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=PROMPT_GEN_MODEL,
            contents=self._conversation,
            config=GenerateContentConfig(
                system_instruction=_DESIGNER_PROMPT_WRITER_SYSTEM,
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
