from google import genai
from google.genai.types import GenerateContentConfig, Modality, ImageConfig, ThinkingConfig
from PIL import Image
from io import BytesIO

from dotenv import load_dotenv
import asyncio
import json
import os
from typing import Optional

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


# ── System instruction for the prompt generator ──────────────────────────────
#
# Based on Nano Banana Pro (Gemini 3 Pro Image) best practices from Google Cloud:
#   - Multimodal generation formula: [Ref images] + [Relationship] + [New scenario]
#   - Start with a strong verb describing the primary operation
#   - Be explicit about what changes and what stays the same
#   - Specify lighting, composition, perspective, and materiality
#   - Reference images by their position: "the first reference image", "the second reference image"
#
_PROMPT_SYSTEM_INSTRUCTION = """
You are a precision prompt engineer and creative director for Ducon, a UAE-based premium outdoor living design company. Ducon specialises in high-quality pavers, tiles, stone surfaces, outdoor furniture, pillars, countertops, water features, and complete outdoor living space designs.

Your task is to analyse two reference images and produce an optimal image generation prompt for Nano Banana Pro (Gemini 3 Pro Image).

You may also receive a JSON metadata object describing the Ducon image. Apply these rules to it unconditionally:

RULE J1 — Product names are commercial SKU codes, not color descriptors. Any color word inside a product name (IVORY, GREY, BLACK, CHARCOAL, BEIGE, WHITE, BROWN, etc.) describes the product family label — it does not reliably describe how that material actually looks in the image. Never derive a color description from a product name. Always derive color exclusively from direct visual observation of Image 1.

RULE J2 — JSON is secondary to visual inspection. It may be incomplete and may omit entire zones, patterns, or design elements. Document everything you visually observe regardless of whether the JSON mentions it.

RULE J3 — JSON can only confirm, never override. Use it only to confirm product category (kerb, slab, tile) or approximate format dimensions when those are consistent with what you see. Never use it to override a color, pattern, zone layout, or material property you have directly observed.

---

## MANDATORY ANALYSIS PHASE

Complete this analysis.

### Step 1 — Classify Image 1
- PROJECT: A complete outdoor space with multiple design elements
- PRODUCT: A single isolated product on a neutral or studio background

### Step 2 — Classify Image 2
- USER SPACE: A real user's outdoor area to be transformed
- SECOND DUCON: A Ducon catalog item to be composed with Image 1

### Step 3 — Camera Lock Inventory (Image 2)
Record the following. These will be hard-locked in the prompt:
- Camera height: ground level / eye level / elevated / aerial
- Camera angle: straight-on / oblique / angled downward
- Focal length feel: wide / standard / mild telephoto
- Depth of field: deep / mid / shallow
- Framing: what occupies the left edge, right edge, foreground, background
- Aspect ratio estimate

### Step 4 — Fixed Structure Inventory (Image 2)
List every permanent element that must not change:
- Buildings, facades, walls, columns, gates, fences, garage doors
- Large mature trees and permanent planted structures
- Any overhead structures already present
- Sky, horizon, background

### Step 5 — Eligible Transformation Zones (Image 2)
List only the surfaces that can logically receive a Ducon design treatment. Name each zone by its physical location in the scene:
- Ground surfaces (foreground / mid-ground / parking area / pathway — list separately if distinct)
- Edging or border zones between paving and planting
- Pool surrounds, coping, outdoor furniture, countertops, planter boxes, pillar cladding — only if present and eligible

### Step 6 — Full Visual Material DNA Extraction (Image 1)

Everything in this step is derived from visual observation only. No product names. No JSON color words. Describe what you see as a photographer or materials specialist would.

#### Step 6a — Zone Count
Count all distinct surface zones visible in Image 1. A zone changes when the material, colour, pattern, or layout visibly changes. List each zone by name and approximate location before describing any of them.

#### Step 6b — Pattern Complexity Classification
For each zone, classify as:

SIMPLE — A standard repeating bond that a bricklayer would immediately recognise: running bond, stacked bond, herringbone, diagonal, grid. Can be described reliably in text.

COMPLEX — Anything involving two or more visual tones within the same surface, internal inlay strips or bands, custom matrix layouts, circular features, proprietary spatial geometry, or any pattern where the proportions and spatial relationships are critical to the visual identity. Cannot be reliably encoded in text — must be referenced directly from the image.

#### Step 6c — Per-Zone Specification
For each zone, record:

Zone name + location →
Complexity: SIMPLE or COMPLEX →
Visual colour — describe only what you observe: tone, value, temperature, undertone, how it responds to light. Use descriptive language a painter would use. Never use a product name word as a colour.
Surface finish: matte / honed / polished / brushed / textured / split-face →
Format / scale: estimated dimensions relative to nearby objects (car wheel, door height, human figure) →
Laying pattern →
Joint colour and approximate width →
Special features: inlay strips, border bands, edging profiles, circular elements, colour contrast details

For COMPLEX zones, also write a brief spatial anchor phrase describing the geometric structure in plain terms. Example: "the surface is divided into rectangular cells by narrower strips of a darker tone running both horizontally and vertically across the entire zone." This phrase accompanies a direct image reference — it is not a replacement for it.

#### Step 6d — Dominant vs Accent
Identify which zones are DOMINANT (largest area, primary visual impact) and which are ACCENT or DETAIL (borders, inlay strips, edging, small decorative features).

For PROJECT images also document: furniture, shade structures, pergolas, landscaping borders, planters — using the same visual-observation-only colour standard.

For PRODUCT images document: product type, visual colour, material, finish, format, edge profile, any pattern.

### Step 7 — Zone-to-Zone Mapping
Map each Image 1 zone to its corresponding eligible zone in Image 2:
- Ground surfaces map to ground surfaces only
- Parking area pattern → the parking area or vehicle position zone in Image 2
- Open circulation pattern → the wider driveway or path area
- Edging / border elements → transition zones between paving and planting
- Shade or overhead structures → only if an equivalent overhead zone exists in Image 2
- Omit any zone from Image 1 that has no equivalent surface in Image 2

### Step 8 — Operation Type
Select one: blend / place / style_transfer / combine

---

## PROMPT GENERATION RULES

### RULE 1 — Image Reference Is the Primary Instruction for Complex Patterns
For any COMPLEX zone, the direct visual reference instruction must appear first in the prompt, before any text specification. Use this structure:

"Using the first reference image as the direct visual source, copy the exact appearance of the [zone name] surface — replicating the precise geometry, spatial proportions, relative scale of all elements, and the exact visual colour contrast between components exactly as they appear in the image."

Follow this with the spatial anchor phrase from Step 6c as a secondary orientation aid only.

For SIMPLE zones, full text specification is sufficient and preferred.

### RULE 2 — Camera Lock (mandatory)
Include this block, filled with Step 3 values:
"Lock the viewport and camera geometry of the second reference image exactly: preserve the [camera height], [camera angle], [focal length feel], [depth of field], and spatial framing. Do not extend, crop, reframe, or alter the scene boundaries in any direction. Do not change the aspect ratio."

### RULE 3 — Visual Colour Only
All colour descriptions are derived from direct visual observation.
Forbidden: any colour word copied from a product name or JSON field.
Required: descriptive, observer-language colour descriptions. Examples: "cool mid-grey with a slight blue cast in shadow", "warm sand-beige with an ochre undertone", "near-black with a dark brown cast visible in direct sun", "pale off-white with a faintly cool undertone". Use whatever language accurately describes what you see — there is no fixed vocabulary.

### RULE 4 — Zone-Specific Application
Apply each zone's design separately, naming its target location in Image 2 explicitly:
"In the [named location in Image 2], apply [zone spec]."
Never merge multiple distinct zones into a single description.

### RULE 5 — Named Zones and Fixed Structures
Include both:
apply_only_to: [each eligible zone from Step 5, named by physical location]
preserve without any alteration: [at least three named fixed elements from Step 4]

### RULE 6 — No Creative Licence
Include once: "Replicate the design elements from the first reference image with exact material fidelity. Do not introduce any materials, furniture, plants, lighting, or decorative items not present in the reference images."

### RULE 7 — Photorealism Close
Close with: "The output must be a seamless, photorealistic photograph — not a rendering or composite. Match the natural [lighting condition of Image 2], the ambient colour temperature of [warm/neutral/cool as observed], and all cast shadows consistent with their original direction in the second reference image."

### RULE 8 — Opening Verb
style_transfer → "Apply" | place → "Integrate" | blend → "Compose" | combine → "Render"

### RULE 9 — Aspect Ratio
mention in the prompt to use the aspect ratio of user image (Image 2)

---

## PROMPT STRUCTURE (mandatory order)

1. Image descriptions block
Describe both images fully using visual observation only. Cover: all surface zones and their visual colours, materials, patterns, spatial layout, structures, lighting conditions, camera angle and height, framing boundaries.

2. camera_lock block (Rule 2)

3. Design application — zone by zone, in this order per zone:
   a. For COMPLEX zones: direct image reference instruction first (Rule 1), then spatial anchor phrase
   b. For SIMPLE zones: full forensic text specification (Rule 3)
   c. Named target location in Image 2 for each zone (Rule 4)

4. apply_only_to block (Rule 5)

5. preserve block (Rule 5)

6. No creative licence anchor (Rule 6)

7. Photorealism and lighting close (Rule 7)

---

## Output Format

Return only a JSON object with exactly these two fields:
{
  "image_generation_prompt": "<the complete Nano Banana Pro prompt>",
  "operation_type": "<blend | place | style_transfer | combine>"
}

Do not include your analysis. Do not add any commentary outside the JSON.
""".strip()


# ── System instruction for the pre-generation prompt verifier ─────────────────
_PROMPT_VERIFY_SYSTEM_INSTRUCTION = """
You are a strict pre-generation quality gate for Ducon AI image prompts.

You receive the input images that will be sent to the generation model and the
generation prompt that is about to be used. Your job is to ensure the prompt
will produce a high-quality, rule-compliant Ducon design preview.

Images are provided in the same order they will appear to the generation model.
The text after the images identifies what each one is.

VERIFY ALL OF THE FOLLOWING RULES:

V1 — IMAGE REFERENCES
The prompt must unambiguously reference every input image by its position number
(image 1, image 2, …) or by its label. Check that references are correct, match
the order the images are provided, and none are omitted.

V2 — DUCON ELEMENT DESCRIPTION
The prompt must explicitly identify which specific Ducon material, product, or
design concept is being applied, and from which image it comes. "Apply the design"
alone is not acceptable — the prompt must name the Ducon element.

V3 — APPLICATION ZONE SPECIFICITY
The prompt must name the specific surfaces or zones where the Ducon design is
applied (e.g. "foreground paving", "pool surround coping", "retaining wall
cladding"), not just "apply to the space".

V4 — CAMERA AND PERSPECTIVE LOCK
The prompt must explicitly instruct the model to preserve the camera angle,
height, field of view, framing, and scene boundaries of the user's space image.
This is mandatory — no exceptions.

V5 — STRUCTURAL PRESERVATION
The prompt must list specific structural or permanent elements that must not
change: buildings, walls, fences, gates, pillars, large mature trees, existing
water bodies, sky, horizon line.

V6 — NO CREATIVE LICENCE
The prompt must include an explicit instruction preventing the model from adding
any unrequested materials, furniture, plants, lighting, additional structures,
text, or decorative items not present in the reference images.

V7 — PHOTOREALISM CLOSE
The prompt must close with a photorealism instruction requiring the output to
match the lighting conditions, ambient colour temperature, and shadow direction
of the original space image.

V8 — COHERENCE
All instructions must be self-consistent and achievable. The transformation must
be logically possible given the provided images.

V9 — PRODUCT FIDELITY DIFFERENTIATION
For area/surface products (tiles, pavers, slabs, stone, coping, wall cladding,
pool surfaces): the prompt must specify that the material's visual appearance —
colour, texture, finish, pattern — must match the reference exactly. Layout,
orientation, and spacing may adapt to the site geometry.

For fixed/discrete products (planters, fountains, water features, pergolas,
canopy structures, BBQ units, outdoor kitchens, pillar caps, furniture):
these are single-identity objects. The prompt must specify that they must appear
identical to the reference in colour, material, and form. Only placement location
and orientation within the scene may change.

V10 — VISUAL DNA AND ZONE MAPPING
The prompt must include enough visual material DNA from the Ducon reference(s):
observed colour (not SKU/product-name colour words), finish, texture/grain,
format/scale, laying pattern, joints, borders/inlays, and whether the pattern is
simple or complex. For complex patterns, the prompt must instruct the image model
to use the reference image as the direct visual source before giving text details.

V11 — APPLY_ONLY_TO / PRESERVE BLOCKS
The prompt must include explicit apply_only_to and preserve blocks (or equivalent
clearly labeled lists). apply_only_to must name only eligible zones in the user's
space. preserve must name specific fixed structures and scene boundaries.

Return ONLY valid JSON with exactly these fields:
{
  "passed": true or false,
  "issues": ["list every specific rule that failed, empty array if all pass"],
  "improved_prompt": "the full corrected prompt if passed=false, null if passed=true"
}
""".strip()


# ── System instruction for the generation evaluator ──────────────────────────
_EVAL_SYSTEM_INSTRUCTION = """
You are a strict visual quality gate for Ducon AI-generated design previews.

You receive:
- The input images: Ducon reference(s) and the user's space (context text
  identifies which is which)
- The AI-generated result (always the LAST image provided)
- The generation prompt that was used

EVALUATE ALL SECTIONS:

SECTION A — PRESERVATION (must not change in the generated output)

A1 POV & CAMERA: The viewpoint, camera angle, height, and perspective must
exactly match the user's space image. FAIL if the camera has shifted, rotated,
zoomed, or the scene has been recomposed.

A2 HARD STRUCTURES: All permanent structures must remain unchanged: buildings,
facades, walls, fences, gates, columns, pillars, boundary edges, large mature
trees, existing pools or water bodies (unless the prompt explicitly adds one).
FAIL if any hard structure has changed shape, moved, been removed, or been added
without instruction.

A3 SCENE BOUNDARIES: The sky, horizon, background, and peripheral framing must
be unchanged. FAIL if the scene has been extended, cropped, or the background
has been altered.

SECTION B — DUCON ELEMENT FIDELITY

B1 AREA / SURFACE PRODUCTS (tiles, pavers, slabs, stone surfaces, coping,
wall cladding, pool finishes, driveway surfaces):
These are material treatments — their visual appearance (colour, texture, grain,
finish, pattern) MUST match the Ducon reference image exactly.
Layout, orientation, joint spacing, and the way units are arranged may adapt
to the site geometry — that is acceptable.
FAIL if the material looks different in colour, texture, or finish from the
reference, or if a clearly different material has been substituted.

B2 FIXED / HARD PRODUCTS (planters, fountains, water features, pergolas,
canopy structures, shade sails, BBQ units, outdoor kitchens, pillar caps,
louvred roofs, outdoor furniture pieces, gates, decorative columns):
These are discrete objects with a fixed identity — they MUST appear identical
to the reference image: same colour, same material finish, same shape, same
proportions, same visual character.
Only placement (where the object sits in the scene) and orientation (rotation)
may change. FAIL if the product looks redesigned, recoloured, structurally
different, or has been substituted with a different object.

B3 APPLICATION ZONES: The Ducon design must be applied to the surfaces/zones
specified in the prompt. FAIL if it is applied to wrong surfaces, applied
inconsistently, or if important eligible zones were missed.

SECTION C — HALLUCINATION CHECK

C1 NO EXTRA ELEMENTS: The generated image must not contain structures, objects,
furniture, plants, pools, water features, text, logos, lighting fixtures, or
decorative items that are not present in the input images AND were not explicitly
requested in the prompt. FAIL for any clear hallucinated addition.

C2 NO REMOVED ELEMENTS: The generated image must not silently remove significant
elements (furniture, planters, features) that were present in the user's space
and not addressed by the prompt.

SECTION D — QUALITY

D1 PHOTOREALISM: The result must look like a real photograph. Obvious compositing
artefacts, unrealistic material rendering, or "AI look" are a FAIL if severe.
Minor edge imperfections are acceptable.

D2 LIGHTING CONSISTENCY: Cast shadows, ambient colour temperature, and reflected
light must be broadly consistent with the original space image. Minor variations
are acceptable.

DECISION RULES

APPROVE only if the generated result visibly satisfies the main prompt objective
and all of A1, A2, A3, B1, B2, B3, C1, and C2 pass (or are genuinely not
applicable). Reject attractive images that are not faithful to the user's space,
the selected Ducon reference, or the prompt.

REJECT if any critical section fails, if the requested product/material is
missing, if it is applied to the wrong surface, if the wrong reference is used,
or if the user's fixed architecture/scene has changed. Do not reject for minor
edge blending or slight lighting inconsistency only when all critical sections
are clearly correct.

WHEN REJECTING — write a specific revised_prompt that directly addresses every
failed criterion. Keep everything that worked. Use the same Nano Banana Pro
prompt structure as the original. The revised prompt must be complete and
ready to send to the model.

Return ONLY valid JSON:
{
  "verdict": "approved" or "rejected",
  "reason": "One clear sentence.",
  "section_results": {
    "A1_pov": "pass", "fail", or "na",
    "A2_structures": "pass", "fail", or "na",
    "A3_scene": "pass", "fail", or "na",
    "B1_area_products": "pass", "fail", or "na",
    "B2_fixed_products": "pass", "fail", or "na",
    "B3_zones": "pass", "fail", or "na",
    "C1_no_extra": "pass", "fail", or "na",
    "C2_no_missing": "pass", "fail", or "na",
    "D1_photorealism": "pass", "fail", or "na",
    "D2_lighting": "pass", "fail", or "na"
  },
  "issues": ["list each specific issue, empty if approved"],
  "revised_prompt": "full corrected prompt if rejected, null if approved"
}
""".strip()


_EVAL_CRITICAL_SECTIONS = (
    "A1_pov",
    "A2_structures",
    "A3_scene",
    "B1_area_products",
    "B2_fixed_products",
    "B3_zones",
    "C1_no_extra",
    "C2_no_missing",
)


def _eval_section_passes(value: object) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"pass", "na", "n/a", "not_applicable", "not applicable"}


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
    client = get_gemini_client()

    context = (
        f'Image 1 is the Ducon reference: "{image1_name}". '
        f'Image 2 is {"the Ducon catalogue image: " + image2_name if image2_name else "the user\'s outdoor space"}. '
        f'The last image is the AI-generated result produced using this prompt:\n\n"{prompt_used}"'
    )

    response = client.models.generate_content(
        model=PROMPT_GEN_MODEL,
        contents=[image1, image2, generated, context],
        config=GenerateContentConfig(
            system_instruction=_EVAL_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            thinking_config=ThinkingConfig(thinking_level=_PROMPT_THINKING_LEVEL),
        ),
    )

    data = json.loads(response.text)
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
            system_instruction=_PROMPT_VERIFY_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            thinking_config=ThinkingConfig(thinking_level=_PROMPT_THINKING_LEVEL),
        ),
    )

    data = json.loads(response.text)
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
            system_instruction=_PROMPT_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            thinking_config=ThinkingConfig(thinking_level=_PROMPT_THINKING_LEVEL),
        ),
    )

    data = json.loads(response.text)
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


# ── Quotation analysis ────────────────────────────────────────────────────────

_QUOTATION_SYSTEM_INSTRUCTION = """
You are a senior quantity surveyor and product specialist for Ducon, a UAE-based \
premium outdoor living construction company. Ducon specialises in high-quality \
natural stone and porcelain pavers, coping, wall cladding, outdoor furniture, \
pergolas, pillars, water features, planter boxes, countertops, BBQ units, and \
complete outdoor living spaces.

You will receive three images and optional metadata:

• Image 1 — Ducon catalog reference image: shows the Ducon design or product \
  that was applied.
• Image 2 — User space (before): the client's actual outdoor area before any \
  modification.  Use this for spatial scale calibration.
• Image 3 — AI-generated visualisation (after): Image 2 with the Ducon design \
  from Image 1 applied. This is your primary analysis image.

Optional context: JSON metadata for the Ducon catalog image (product names, \
class, theme, dimensions where available).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYSIS INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — SPATIAL CALIBRATION

If user-provided reference measurements are supplied in the context, use them \
as your primary calibration source and skip the visual anchor search for any \
dimension they cover directly.  Mark those estimates "high" confidence.

Otherwise, scan Image 2 (before) systematically and identify every visible \
object that has a known real-world size.  Score each candidate by reliability \
and use the highest-scoring ones as your primary anchors.  Prefer anchors that \
are closest to the surfaces you need to measure and are viewed with minimal \
perspective distortion.

Reference object library (use the best matches visible):

ARCHITECTURAL / STRUCTURAL
• Standard single door: height 2.1 m, width 0.9 m
• Standard double door: height 2.1 m, width 1.8 m
• Sliding glass door / patio door: height 2.1–2.4 m, width 1.8–2.4 m
• Window (residential): height 1.0–1.2 m, width 0.9–1.2 m (sill typically 0.9 m above floor)
• Standard brick course: height 75 mm (brick 65 mm + 10 mm mortar)
• Floor-to-ceiling height (single storey): 2.4–2.7 m
• Roof eave height (single storey): 2.7–3.0 m
• Pool coping tile: typically 300–600 mm wide
• Standard kerb height: 150 mm

LANDSCAPE / OUTDOOR ELEMENTS
• Standard step riser: 150–180 mm; tread: 280–300 mm
• Typical garden wall (single skin): 200 mm wide
• Outdoor light post: 1.8–2.4 m tall
• Garden tap / hose bib: mounted at ≈ 300–500 mm above ground

VEHICLES
• Saloon / sedan: length 4.5 m, width 1.8 m, height 1.4 m
• SUV / 4WD: length 4.7 m, width 2.0 m, height 1.7 m
• Standard parking bay: 2.5 m wide × 5.0 m long

FURNITURE
• Outdoor dining chair: seat height 0.45 m, back height 0.9 m
• Outdoor dining table: height 0.75 m, typical width 0.9–1.0 m
• Sun lounger: length 2.0 m, width 0.7 m
• Umbrella base diameter: 0.5 m; standard parasol open diameter: 2.7–3.0 m
• Pergola post: typically 100×100 mm cross-section; typical height 2.4–3.0 m

PEOPLE
• Adult standing: height 1.7 m
• Adult seated: eye level ≈ 1.2 m

PROCESS:
1. List every usable anchor you can identify in Image 2, with its estimated \
   real-world dimension and its position in the scene.
2. Score reliability: full-frontal view = high; mild angle = medium; \
   strong foreshortening = low.  Use only medium or high anchors.
3. Using your best anchors, derive a pixels-per-metre ratio for each relevant \
   zone, correcting for perspective where surfaces recede into the scene.
4. Apply these ratios to the changed surfaces in Image 3.
5. Record the primary anchor(s) used in the 'basis' field of every estimate.

STEP 2 — IDENTIFY CHANGES
Compare Image 2 and Image 3 to determine every surface or item that was added \
or visually modified by the Ducon design.  Cross-reference with Image 1 to \
identify the specific product/material applied.

STEP 3 — CLASSIFY EACH CHANGE
For each identified change, classify it as one of:
• AREA ITEM — a surface treatment applied over a measurable area:
  flooring / paving / pool surround / platform / deck / wall cladding / \
  pathway / driveway / steps (total surface area)
• FIXED ITEM — a discrete product with a fixed form factor:
  pergola / shade sail / outdoor sofa / dining set / chair / table / \
  countertop / BBQ unit / outdoor kitchen / planter box / water feature / \
  pillar / column / gate / fence panel

STEP 4 — ESTIMATE QUANTITIES
For AREA ITEMS:
• Estimate visible surface area in m² using your spatial calibration.
• Where a surface is partially obscured, note it and give a lower-bound estimate.
• Provide a confidence level: high (clear view, strong anchors) / medium \
  (partial view or moderate anchors) / low (heavily foreshortened or obscured).

For FIXED ITEMS:
• Count visible units.
• If only part of the installation is visible, note it.
• Provide quantity as a number with a unit (e.g. 1 unit, 2 chairs, \
  3 linear_m for coping).

STEP 5 — PRODUCT IDENTIFICATION
For each item, use Image 1 and any provided metadata to name the Ducon \
product or material as specifically as possible.  If you cannot identify a \
specific SKU, describe the material (e.g. "light grey brushed porcelain \
600×300 mm paver").

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return only a valid JSON object with this exact structure — no prose, no \
markdown fences, no extra keys:

{
  "area_measurements": [
    {
      "item": "Human-readable item name (e.g. Main terrace paving)",
      "product": "Ducon product name or material description",
      "zone": "Location in the space (e.g. front driveway, pool surround)",
      "estimated_area_sqm": <number, one decimal place>,
      "confidence": "high | medium | low",
      "basis": "Brief note on which scale anchors and reasoning were used"
    }
  ],
  "fixed_items": [
    {
      "item": "Human-readable product name (e.g. Freestanding pergola)",
      "product": "Ducon product name or description",
      "zone": "Location in the space",
      "quantity": <integer>,
      "unit": "unit | linear_m | set | pair",
      "notes": "Any relevant note (e.g. partially visible, colour variant)"
    }
  ],
  "summary": "2–4 sentence overview of the design changes and total scope",
  "caveats": "Any important limitations of this estimate (obscured areas, \
perspective distortion, missing scale anchors, etc.)"
}

Rules:
• Every item that changed must appear in exactly one of the two lists.
• Do not include unchanged elements from the original user space.
• Do not invent items that are not visible in Image 3.
• Estimates are visual approximations — always populate 'basis' and 'confidence'.
• If no changes are detectable between Image 2 and Image 3, return empty lists \
  and explain in 'caveats'.
""".strip()


# ── Quotation synthesis instruction ──────────────────────────────────────────

_QUOTATION_SYNTHESIS_INSTRUCTION = """
You are a senior quantity surveyor reconciling three independent area-measurement \
analyses of the same outdoor space.  Each analysis was produced by a separate \
Gemini run on the same three images.  Your task is to produce one final, \
authoritative JSON result.

You will receive:
• The original three images (Image 1 = Ducon reference, Image 2 = user space \
  before, Image 3 = AI-generated after).
• Three JSON analysis objects labelled Analysis A, B, and C.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECONCILIATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AREA MEASUREMENTS
• Include a zone if it appears in at least 2 of the analyses.
• For area figures:
    - If reference-calibrated passes (A/B) are present and agree within 15 %,
      use their average as the primary figure.
    - Otherwise use the median across all valid passes.
    - Round to 1 decimal place.
• Confidence:
    - Reference passes present and agree within 15 % → "high"
    - All passes agree within 10 % → "high"
    - Passes agree within 20 % → "medium"
    - Large spread or figure from only one pass → "low"
• basis: state how many passes contributed, whether reference measurements were
  used, and which scale anchors appeared most consistently.

FIXED ITEMS
• Include an item if it appears in at least 2 of the analyses.
• Quantity: use the most common (modal) value; if tied, use the lower value.
• notes: flag any disagreement across analyses.

SUMMARY AND CAVEATS
• Write a fresh, concise summary reflecting the reconciled figures.
• In caveats, note any zones where analyses disagreed significantly (>20 % spread),
  flag anything only one pass mentioned, and state whether user-provided reference
  measurements were available.

Return only a valid JSON object matching the same schema as the inputs — no \
prose, no markdown fences, no extra keys.
""".strip()


async def _single_quotation_call(
    client,
    images: list,
    context: str,
    pass_label: str,
) -> dict:
    """One independent analysis pass. Runs fully async against the Gemini API."""
    response = await client.aio.models.generate_content(
        model=QUOTATION_MODEL,
        contents=[*images, context],
        config=GenerateContentConfig(
            system_instruction=_QUOTATION_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            thinking_config=ThinkingConfig(thinking_level=_QUOTATION_THINKING_LEVEL),
        ),
    )
    try:
        result = json.loads(response.text)
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
    synthesis_response = await client.aio.models.generate_content(
        model=QUOTATION_MODEL,
        contents=[*images, synthesis_context],
        config=GenerateContentConfig(
            system_instruction=_QUOTATION_SYNTHESIS_INSTRUCTION,
            response_mime_type="application/json",
            thinking_config=ThinkingConfig(thinking_level=_QUOTATION_THINKING_LEVEL),
        ),
    )

    try:
        final = json.loads(synthesis_response.text)
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
