from google import genai
from google.genai.types import GenerateContentConfig, Modality, ImageConfig, ThinkingConfig
from PIL import Image
from io import BytesIO

from dotenv import load_dotenv
import asyncio
import json
import os
from typing import Optional

from app import prompt_loader

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ── Image generation model ────────────────────────────────────────────────────
# Set IMAGE_GEN_MODEL in .env to switch between:
#   gemini-3-pro-image-preview   → Nano Banana Pro  (default, no thinking)
#   gemini-3.1-flash-image-preview → Nano Banana 2  (thinking model)
IMAGE_GEN_MODEL = os.getenv("IMAGE_GEN_MODEL", "gemini-3-pro-image-preview")

# ── Quotation analysis model ───────────────────────────────────────────────────
# Uses Gemini 3.1 Pro for deep multi-image analysis.
# Set QUOTATION_MODEL in .env to override.
QUOTATION_MODEL = os.getenv("QUOTATION_MODEL", "gemini-3.1-pro-preview")

# Thinking depth for the quotation analyser.  High is recommended — accurate
# area measurements require careful multi-step spatial reasoning.
# Options: Minimal | Low | Medium | High
_QUOTATION_THINKING_LEVEL = os.getenv("QUOTATION_THINKING_LEVEL", "High")

# Controls reasoning depth for the prompt generator (Flash-Lite).
# Set PROMPT_THINKING_LEVEL in .env to "Minimal" or "High" (default: "High").
_PROMPT_THINKING_LEVEL = os.getenv("PROMPT_THINKING_LEVEL", "High")

# Maximum verification rounds in the prompt-verify loop.
# After this many rounds the pipeline proceeds regardless.
PROMPT_VERIFY_MAX_ROUNDS: int = int(os.getenv("PROMPT_VERIFY_MAX_ROUNDS", "2"))

# Maximum generation+evaluation rounds in the post-gen loop.
GEN_EVAL_MAX_ROUNDS: int = int(os.getenv("GEN_EVAL_MAX_ROUNDS", "3"))

# Nano Banana 2 only — controls reasoning depth before image generation.
# Set IMAGE_THINKING_LEVEL in .env to "Minimal" or "High" (default: "High").
# Has no effect when IMAGE_GEN_MODEL is Nano Banana Pro.
_IMAGE_THINKING_LEVEL = os.getenv("IMAGE_THINKING_LEVEL", "High")

_NANO_BANANA_2 = "gemini-3.1-flash-image-preview"

# Gemini 3.1 Flash-Lite — fast multimodal prompt generation (text output only)
PROMPT_GEN_MODEL = "gemini-3-flash-preview"

_client = None


def get_gemini_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


def _parse_json_response(text: str | None) -> dict:
    """
    Parse Gemini JSON output.

    Models sometimes append extra text after a valid JSON object even when
    response_mime_type is application/json. raw_decode keeps the first object.
    """
    if not text or not str(text).strip():
        raise ValueError("Empty JSON response from model")

    raw = str(text).strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from model: {raw[:200]}") from exc

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")

    trailing = raw[end:].strip()
    if trailing:
        print(f"[Gemini JSON] ignored trailing content after JSON ({len(trailing)} chars)")

    return obj


# System prompts live in app/prompts/ and are loaded via prompt_loader.

_EVAL_CRITICAL_SECTIONS = (
    "A1_pov",
    "A2_structures",
    "A3_scene",
    "B1_area_products",
    "B2_fixed_products",
    "B3_zones",
    "C1_no_extra",
    "C2_no_missing",
    "E1_site_geometry",
    "E2_circulation",
    "E3_placement_logic",
    "E4_surface_orientation",
    "E5_user_experience",
)

_EVAL_SECTION_ORDER = (
    "A1_pov",
    "A2_structures",
    "A3_scene",
    "B1_area_products",
    "B2_fixed_products",
    "B3_zones",
    "C1_no_extra",
    "C2_no_missing",
    "D1_photorealism",
    "D2_lighting",
    "E1_site_geometry",
    "E2_circulation",
    "E3_placement_logic",
    "E4_surface_orientation",
    "E5_user_experience",
)


def _eval_section_passes(value: object) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"pass", "na", "n/a", "not_applicable", "not applicable"}


def log_section_analysis(
    section_analysis: object,
    section_results: object,
    *,
    prefix: str = "[ImageGenAgent]",
) -> None:
    """Print structured per-section observations and evaluations."""
    if not isinstance(section_analysis, dict) or not section_analysis:
        print(f"{prefix} section_analysis: (missing — model did not return structured analysis)")
        return

    results = section_results if isinstance(section_results, dict) else {}
    print(f"{prefix} --- section analysis ---")
    for key in _EVAL_SECTION_ORDER:
        entry = section_analysis.get(key)
        verdict = results.get(key, "?")
        if not isinstance(entry, dict):
            if verdict != "?":
                print(f"{prefix} {key} [{verdict}]: (no structured analysis)")
            continue

        verdict = entry.get("verdict") or verdict
        aspect = (entry.get("aspect") or "").strip()
        ref_obs = (entry.get("reference_observation") or "").strip()
        gen_obs = (entry.get("generated_observation") or "").strip()
        evaluation = (entry.get("evaluation") or "").strip()

        print(f"{prefix} {key} [{verdict}]")
        if aspect:
            print(f"{prefix}   aspect: {aspect}")
        if ref_obs:
            print(f"{prefix}   reference: {ref_obs}")
        if gen_obs:
            print(f"{prefix}   generated: {gen_obs}")
        if evaluation:
            print(f"{prefix}   evaluation: {evaluation}")
    print(f"{prefix} --- end section analysis ---")


def evaluate_generation(
    image1: Image.Image,
    image2: Image.Image,
    generated: Image.Image,
    prompt_used: str,
    image1_name: str,
    image2_name: str | None = None,
) -> tuple[bool, str | None, list[str]]:
    """
    Evaluates the generated image against the Ducon reference(s) and the prompt.

    Returns:
        (approved, revised_prompt, issues)
        approved       — True if the generation passes all quality criteria.
        revised_prompt — A corrected prompt if rejected, None if approved.
        issues         — List of specific issues found (empty if approved).
    """
    prompt_loader.ensure_prompts_loaded()
    client = get_gemini_client()

    context = (
        f'Image 1 is the Ducon reference: "{image1_name}". '
        f'Image 2 is {"the Ducon catalogue image: " + image2_name if image2_name else "the user\'s outdoor space"}. '
        f'The last image is the AI-generated result produced using this prompt:\n\n"{prompt_used}"\n\n'
        "For every section: fill section_analysis (aspect → reference observation → "
        "generated observation → evaluation → verdict) then section_results. "
        "section_analysis is required on approval and rejection."
    )

    response = client.models.generate_content(
        model=PROMPT_GEN_MODEL,
        contents=[image1, image2, generated, context],
        config=GenerateContentConfig(
            system_instruction=prompt_loader.EVAL_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            thinking_config=ThinkingConfig(thinking_level=_PROMPT_THINKING_LEVEL),
        ),
    )

    data = _parse_json_response(response.text)
    issues: list[str] = data.get("issues") or []
    section_results = data.get("section_results") or {}
    failed_sections = [
        key for key in _EVAL_CRITICAL_SECTIONS
        if not _eval_section_passes(section_results.get(key))
    ]
    if failed_sections:
        issues = [
            *issues,
            f"Critical QC sections failed or missing: {', '.join(failed_sections)}",
        ]

    approved = data.get("verdict") == "approved" and not issues
    revised_prompt = data.get("revised_prompt") if not approved else None

    print(f"[Evaluator] verdict={data.get('verdict')}  reason={data.get('reason')}")
    log_section_analysis(
        data.get("section_analysis"),
        section_results,
        prefix="[Evaluator]",
    )
    if issues:
        print(f"[Evaluator] issues: {issues}")
    if revised_prompt:
        print(f"[Evaluator] revised_prompt (first 400 chars):\n{revised_prompt[:400]}\n")

    return approved, revised_prompt, issues


def verify_prompt(
    images: list[Image.Image],
    labels: list[str],
    prompt: str,
) -> tuple[bool, list[str], str | None]:
    """
    Pre-generation prompt verifier.  Checks the prompt against the input images
    to ensure it correctly references them, specifies Ducon elements precisely,
    includes a camera lock, preservation list, and no-hallucination constraint.

    Args:
        images  — Ordered list of PIL Images that will be sent to the generator.
        labels  — Corresponding human-readable labels (e.g. "Ducon reference",
                  "user space", "Ducon pergola").
        prompt  — The generation prompt to verify.

    Returns:
        (passed, issues, improved_prompt)
        passed          — True if all rules are satisfied.
        issues          — List of rule failures (empty when passed=True).
        improved_prompt — A corrected prompt if passed=False, else None.
    """
    prompt_loader.ensure_prompts_loaded()
    client = get_gemini_client()

    # Build context string that tells the verifier what each image is
    image_context_parts = []
    for i, label in enumerate(labels, start=1):
        image_context_parts.append(f"Image {i}: {label}")
    image_context = "\n".join(image_context_parts)
    context = (
        f"Images provided to the generation model (in this order):\n{image_context}\n\n"
        f"Prompt to verify:\n{prompt}"
    )

    contents = [*images, context]

    response = client.models.generate_content(
        model=PROMPT_GEN_MODEL,
        contents=contents,
        config=GenerateContentConfig(
            system_instruction=prompt_loader.PROMPT_VERIFY_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            thinking_config=ThinkingConfig(thinking_level=_PROMPT_THINKING_LEVEL),
        ),
    )

    data = _parse_json_response(response.text)
    passed: bool = bool(data.get("passed", False))
    issues: list[str] = data.get("issues") or []
    improved_prompt: str | None = data.get("improved_prompt") if not passed else None

    print(f"[PromptVerifier] passed={passed}  issues={len(issues)}")
    if issues:
        print(f"[PromptVerifier] issues: {issues}")
    if improved_prompt:
        print(f"[PromptVerifier] improved_prompt (first 400 chars):\n{improved_prompt[:400]}\n")

    return passed, issues, improved_prompt


def generate_prompt(
    image1: Image.Image,
    image1_name: str,
    image2: Image.Image,
    image2_name: str | None = None,
    image1_metadata: dict | None = None,
    image2_metadata: dict | None = None,
    user_hint: str | None = None,
) -> str:
    """
    Uses gemini-3.1-flash-lite-preview to analyse the two images and generate
    an optimal Nano Banana Pro prompt.

    Args:
        image1:          The primary Ducon catalog image (PIL Image).
        image1_name:     Display name of the Ducon image from the DB.
        image2:          The second image — user upload or another Ducon image.
        image2_name:     Name of the second Ducon image, or None if it's a user upload.
        image1_metadata: Full metadata entry for image1 from metadata.json, if available.
        image2_metadata: Full metadata entry for image2 from metadata.json, if it's a Ducon image.
        user_hint:       Optional prompt supplied by the user/frontend.

    Returns:
        A detailed prompt string ready to send to Nano Banana Pro.
    """
    prompt_loader.ensure_prompts_loaded()
    client = get_gemini_client()

    if image2_name:
        context = (
            f'Image 1 is a Ducon catalog image named "{image1_name}". '
            f'Image 2 is also a Ducon catalog image named "{image2_name}". '
            f"Analyse both and generate an optimal Nano Banana Pro prompt to combine or integrate them."
        )
    else:
        context = (
            f'Image 1 is a Ducon catalog image named "{image1_name}". '
            f"Image 2 is a photo of the user's own outdoor space that they want to transform. "
            f"Analyse both and generate an optimal Nano Banana Pro prompt to apply Ducon's design into the user's space."
        )

    if image1_metadata:
        context += (
            f"\n\nDetailed metadata for Image 1:\n"
            f"{json.dumps(image1_metadata, indent=2)}"
        )

    if image2_metadata:
        context += (
            f"\n\nDetailed metadata for Image 2:\n"
            f"{json.dumps(image2_metadata, indent=2)}"
        )

    if user_hint:
        context += (
            f'\n\nThe user has also provided the following direction: "{user_hint}". '
            f"Incorporate this intent into the generated prompt where it is consistent with the images and Ducon's style."
        )

    response = client.models.generate_content(
        model=PROMPT_GEN_MODEL,
        contents=[image1, image2, context],
        config=GenerateContentConfig(
            system_instruction=prompt_loader.PROMPT_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            thinking_config=ThinkingConfig(thinking_level=_PROMPT_THINKING_LEVEL),
        ),
    )

    data = _parse_json_response(response.text)
    prompt = data.get("image_generation_prompt", "")

    print(f"[Prompt Generator] operation_type: {data.get('operation_type', 'unknown')}")
    print(f"[Prompt Generator] generated prompt:\n{prompt}\n")

    return prompt


def combine_images(
    filename: str,
    image1: Image.Image,
    image2: Image.Image,
    prompt: str,
    subfolder: str = None,
):
    """
    Sends both images and the generated prompt to the configured image generation
    model (Nano Banana Pro or Nano Banana 2).

    Images are passed in order:
      image1 = "the first reference image"
      image2 = "the second reference image"

    When Nano Banana 2 is active, ThinkingConfig is applied and thoughts are
    captured (see commented block below). Thinking level is controlled by the
    THINKING_LEVEL env variable ("Minimal" or "High").
    """
    client = get_gemini_client()

    is_nano_banana_2 = IMAGE_GEN_MODEL == _NANO_BANANA_2
    print(f"[ImageGen] model={IMAGE_GEN_MODEL!r}  is_nano_banana_2={is_nano_banana_2}  image_thinking_level={_IMAGE_THINKING_LEVEL!r}")

    config = GenerateContentConfig(
        response_modalities=[Modality.TEXT, Modality.IMAGE],
        thinking_config=ThinkingConfig(
            thinking_level=_IMAGE_THINKING_LEVEL,
            include_thoughts=True,
        ) if is_nano_banana_2 else None,
    )

    response = client.models.generate_content(
        model=IMAGE_GEN_MODEL,
        contents=[image1, image2, prompt],
        config=config,
    )

    # ── Uncomment to inspect Nano Banana 2 reasoning thoughts ────────────────
    if is_nano_banana_2:
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", False) and part.text:
                print(f"[NB2 Thought]\n{part.text}\n")
    # ─────────────────────────────────────────────────────────────────────────

    return save_image(filename, response=response, subfolder=subfolder)


def save_image(filename: str, response, subfolder: str = None):
    for part in response.candidates[0].content.parts:
        if part.text:
            print(part.text)
        elif part.inline_data:
            image = Image.open(BytesIO(part.inline_data.data))
            output_dir = os.path.join("outputs", subfolder) if subfolder else "outputs"
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename)
            image.save(path)
            return filename


def generate_image(prompt: str = None):
    """Standalone text-to-image generation for testing."""
    if not prompt:
        print("no prompt given")
        return
    client = get_gemini_client()
    response = client.models.generate_content(
        model=IMAGE_GEN_MODEL,
        contents=prompt,
        config=GenerateContentConfig(
            response_modalities=[Modality.TEXT, Modality.IMAGE],
        ),
    )
    save_image("test.png", response=response)


def list_available_models():
    client = get_gemini_client()
    for model in client.models.list():
        if "image" in model.name.lower():
            print(model.name)


async def _single_quotation_call(
    client,
    images: list,
    context: str,
    pass_label: str,
) -> dict:
    """One independent analysis pass. Runs fully async against the Gemini API."""
    prompt_loader.ensure_prompts_loaded()
    response = await client.aio.models.generate_content(
        model=QUOTATION_MODEL,
        contents=[*images, context],
        config=GenerateContentConfig(
            system_instruction=prompt_loader.QUOTATION_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            thinking_config=ThinkingConfig(thinking_level=_QUOTATION_THINKING_LEVEL),
        ),
    )
    try:
        result = _parse_json_response(response.text)
        print(
            f"[Quotation {pass_label}] "
            f"area_items={len(result.get('area_measurements', []))}  "
            f"fixed_items={len(result.get('fixed_items', []))}"
        )
        return result
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"Pass {pass_label} returned non-JSON: {response.text[:200]}"
        ) from exc


def _build_context(
    ducon_metadata: Optional[dict],
    reference_measurements: Optional[str],
) -> str:
    """Build the context string for one analysis pass."""
    parts = [
        "Image 1: Ducon catalog reference image.",
        "Image 2: User's outdoor space (before).",
        "Image 3: AI-generated visualisation with the Ducon design applied (after).",
    ]
    if ducon_metadata:
        parts.append(
            f"\nDucon catalog metadata for Image 1:\n{json.dumps(ducon_metadata, indent=2)}"
        )
    if reference_measurements:
        parts.append(
            f"\nReference measurements provided by the user:\n{reference_measurements}\n"
            "Use these exact dimensions as your primary calibration source — they take "
            "priority over any visual perspective estimates. Raise confidence levels "
            "accordingly for zones that can be anchored to these measurements."
        )
    return "\n".join(parts)


async def analyze_quotation(
    ducon_image: Image.Image,
    user_image: Image.Image,
    generation_image: Image.Image,
    ducon_metadata: Optional[dict] = None,
    reference_measurements: Optional[str] = None,
) -> dict:
    """
    Ensemble quotation analysis: runs 4 independent Gemini passes concurrently,
    then feeds all results back to the model for a final reconciled answer.

    Pass strategy:
    • If reference_measurements provided: 2 passes WITH reference (A, B) +
      2 passes WITHOUT (C, D).  The synthesis is told which is which so it can
      weight calibrated vs. visual-only estimates appropriately.
    • If no reference provided: all 4 passes run without reference.

    Args:
        ducon_image:             PIL Image — Ducon catalog reference (Image 1).
        user_image:              PIL Image — user's space before design (Image 2).
        generation_image:        PIL Image — AI-generated result (Image 3).
        ducon_metadata:          Optional catalog metadata for product identification.
        reference_measurements:  Optional user-provided real-world dimensions.

    Returns:
        Reconciled dict with keys: area_measurements, fixed_items, summary, caveats.

    Raises:
        ValueError: if all passes fail or the synthesis returns unparseable JSON.
        Exception:  propagates Gemini API errors to the caller.
    """
    client = get_gemini_client()
    images = [ducon_image, user_image, generation_image]

    has_ref = bool(reference_measurements)
    ctx_with_ref    = _build_context(ducon_metadata, reference_measurements)
    ctx_without_ref = _build_context(ducon_metadata, None)

    if has_ref:
        # A & B use reference measurements; C & D rely on visual anchors only.
        pass_configs = [
            ("A", ctx_with_ref,    "WITH user reference measurements"),
            ("B", ctx_with_ref,    "WITH user reference measurements"),
            ("C", ctx_without_ref, "WITHOUT reference measurements (visual anchors only)"),
            ("D", ctx_without_ref, "WITHOUT reference measurements (visual anchors only)"),
        ]
        print("[Quotation] launching 4 concurrent passes (2 with reference, 2 without)…")
    else:
        # No reference supplied — all 4 passes use visual anchors only.
        pass_configs = [
            ("A", ctx_without_ref, "WITHOUT reference measurements (visual anchors only)"),
            ("B", ctx_without_ref, "WITHOUT reference measurements (visual anchors only)"),
            ("C", ctx_without_ref, "WITHOUT reference measurements (visual anchors only)"),
            ("D", ctx_without_ref, "WITHOUT reference measurements (visual anchors only)"),
        ]
        print("[Quotation] launching 4 concurrent passes (no reference provided)…")

    # ── Step 1: four concurrent independent passes ────────────────────────────
    raw_results = await asyncio.gather(
        *[
            _single_quotation_call(client, images, ctx, label)
            for label, ctx, _ in pass_configs
        ],
        return_exceptions=True,
    )

    # Pair each result with its label and description; filter failed passes.
    labelled: list[tuple[str, str, dict]] = []  # (label, description, result)
    for (label, _, description), result in zip(pass_configs, raw_results):
        if isinstance(result, dict):
            labelled.append((label, description, result))
        else:
            print(f"[Quotation {label}] failed — {result}")

    if not labelled:
        raise ValueError("All 4 quotation analysis passes failed.")
    if len(labelled) == 1:
        print("[Quotation] only 1 pass succeeded — returning without synthesis")
        return labelled[0][2]

    # ── Step 2: synthesis pass ────────────────────────────────────────────────
    synthesis_parts = [
        "The following are independent analyses of the same three images.",
        f"{len(labelled)} of 4 passes succeeded.",
        "",
    ]
    if has_ref:
        synthesis_parts.append(
            "NOTE: Passes A and B were given exact user-provided reference measurements "
            "as their primary calibration source — their area figures should be treated "
            "as more accurate than the visual-only estimates from C and D. "
            "Weight A/B estimates more heavily when reconciling area figures, but use "
            "C/D as a cross-check for consistency.\n"
        )

    for label, description, result in labelled:
        synthesis_parts.append(
            f"Analysis {label} ({description}):\n{json.dumps(result, indent=2)}\n"
        )
    synthesis_context = "\n".join(synthesis_parts)

    print(f"[Quotation] running synthesis pass over {len(labelled)} results…")
    prompt_loader.ensure_prompts_loaded()
    synthesis_response = await client.aio.models.generate_content(
        model=QUOTATION_MODEL,
        contents=[*images, synthesis_context],
        config=GenerateContentConfig(
            system_instruction=prompt_loader.QUOTATION_SYNTHESIS_INSTRUCTION,
            response_mime_type="application/json",
            thinking_config=ThinkingConfig(thinking_level=_QUOTATION_THINKING_LEVEL),
        ),
    )

    try:
        final = _parse_json_response(synthesis_response.text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"Synthesis pass returned non-JSON: {synthesis_response.text[:200]}"
        ) from exc

    print(
        f"[Quotation] synthesis complete — "
        f"area_items={len(final.get('area_measurements', []))}  "
        f"fixed_items={len(final.get('fixed_items', []))}"
    )
    return final


if __name__ == "__main__":
    generate_image("generate the last image again")
