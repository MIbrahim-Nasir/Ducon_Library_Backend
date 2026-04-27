from google import genai
from google.genai.types import GenerateContentConfig, Modality, ImageConfig, ThinkingConfig
from PIL import Image
from io import BytesIO

from dotenv import load_dotenv
import json
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ── Image generation model ────────────────────────────────────────────────────
# Set IMAGE_GEN_MODEL in .env to switch between:
#   gemini-3-pro-image-preview   → Nano Banana Pro  (default, no thinking)
#   gemini-3.1-flash-image-preview → Nano Banana 2  (thinking model)
IMAGE_GEN_MODEL = os.getenv("IMAGE_GEN_MODEL", "gemini-3-pro-image-preview")

# Controls reasoning depth for the prompt generator (Flash-Lite).
# Set PROMPT_THINKING_LEVEL in .env to "Minimal" or "High" (default: "High").
_PROMPT_THINKING_LEVEL = os.getenv("PROMPT_THINKING_LEVEL", "High")

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


# ── System instruction for the generation evaluator ──────────────────────────
_EVAL_SYSTEM_INSTRUCTION = """
You are a quality control reviewer for Ducon, a UAE-based outdoor living design company.

You will receive three images and a generation prompt:
- Image 1: The Ducon catalogue reference (product or project design)
- Image 2: The user's outdoor space or a second Ducon catalogue image
- Image 3: The AI-generated visualisation result (what Nano Banana Pro produced)
- A text description of the prompt used to produce Image 3

Your task is to evaluate whether Image 3 correctly achieves the intended transformation.

## Evaluation criteria

1. Ducon design elements present — Are the materials, surfaces, or products from Image 1 visibly and correctly applied in Image 3? (Most important criterion)
2. User space preserved — Does Image 3 preserve the fixed structures of Image 2: walls, buildings, fences, permanent landscaping? These must not change.
3. Perspective and framing intact — Does Image 3 match the camera angle, height, and scene boundaries of Image 2? The viewpoint must not shift.
4. Photorealism — Does Image 3 look like a real photograph, not a rough composite or obvious AI artifact?

## Decision rules

Approve if: The Ducon design elements are clearly visible and applied, the space is recognisable, and the result is broadly photorealistic.
Reject if: The Ducon elements are absent or barely visible, OR the user's space is unrecognisably altered, OR the output is clearly a failed generation (blank, nonsensical, or completely wrong composition).

Do NOT reject for: Minor lighting inconsistencies, slightly imperfect blending at edges, or artistic choices that are reasonable given the inputs. If in doubt, approve.

## When rejecting, write a revised prompt

The revised prompt must be a corrected, specific version of the original prompt that directly addresses the identified problem.
Keep what worked. Fix only what failed.
Use the same Nano Banana Pro prompt structure as the original.

## Output format

Return only a JSON object:
{
  "verdict": "approved" or "rejected",
  "reason": "One clear sentence explaining the verdict",
  "revised_prompt": "The full corrected prompt if rejected, null if approved"
}
""".strip()


def evaluate_generation(
    image1: Image.Image,
    image2: Image.Image,
    generated: Image.Image,
    prompt_used: str,
    image1_name: str,
    image2_name: str | None = None,
) -> tuple[bool, str | None]:
    """
    Sends the generated image back to Flash-Lite alongside the original reference
    images and the prompt that was used, and asks it to evaluate the result.

    Returns:
        (approved, revised_prompt)
        approved       — True if the generation is acceptable.
        revised_prompt — A corrected prompt if rejected, None if approved.
    """
    client = get_gemini_client()

    context = (
        f'Image 1 is the Ducon reference: "{image1_name}". '
        f'Image 2 is {"the Ducon catalogue image: " + image2_name if image2_name else "the user\'s outdoor space"}. '
        f'Image 3 is the AI-generated visualisation produced using this prompt:\n\n"{prompt_used}"'
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
    approved = data.get("verdict", "approved") == "approved"
    revised_prompt = data.get("revised_prompt") if not approved else None

    print(f"[Evaluator] verdict={data.get('verdict')}  reason={data.get('reason')}")
    if revised_prompt:
        print(f"[Evaluator] revised_prompt:\n{revised_prompt}\n")

    return approved, revised_prompt


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


if __name__ == "__main__":
    generate_image("generate the last image again")
