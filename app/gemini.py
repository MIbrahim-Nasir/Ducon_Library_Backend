from google import genai
from google.genai.types import GenerateContentConfig, Modality
from PIL import Image
from io import BytesIO

from dotenv import load_dotenv
import json
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Nano Banana Pro — image generation / combining
IMAGE_GEN_MODEL = "gemini-3-pro-image-preview"

# Gemini 3.1 Flash-Lite — fast multimodal prompt generation (text output only)
PROMPT_GEN_MODEL = "gemini-3.1-flash-lite-preview"

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
You are a creative director for Ducon, an Australian outdoor living design company. Ducon specialises in high-quality pavers, tiles, stone surfaces, outdoor furniture, pillars, countertops, water features, and complete outdoor living space designs.

Your task is to analyse two reference images and write an optimal image generation prompt for Nano Banana Pro (Gemini 3 Pro Image), Google's premium image generation model.

## About the two images you will receive

**Image 1** is always a Ducon catalog image. It will be one of:
- **PROJECT image**: A complete outdoor living space design showing flooring/paving, outdoor furniture, planters, lighting, water features, pool surrounds, alfresco areas, etc.
- **PRODUCT image**: A standalone studio shot of a single product — a paver, tile, pillar, countertop, stone slab, or similar item against a plain or minimal background.

**Image 2** is either:
- **USER SPACE**: A photo of the user's real outdoor area — their backyard, garden, alfresco, patio, or home exterior — that they want transformed.
- **SECOND DUCON image**: Another catalog item or project that is being combined with Image 1.

## Rules for generating the prompt

### If Image 1 is a PROJECT image:
- Extract ONLY Ducon's outdoor design elements: flooring patterns, paving materials, outdoor furniture style, planters, lighting fixtures, water features, alfresco structures.
- Exclude fixed architectural elements that are not Ducon's work: buildings, walls, fences, cars, neighbouring structures.
- Apply these design elements to Image 2 in a photorealistic, natural way that respects the existing space layout.

### If Image 1 is a PRODUCT image:
- Identify the specific product and its material, colour, texture, and finish.
- Place this product into Image 2's scene in the most logical and aesthetically consistent location.
- Maintain the product's correct scale relative to surrounding elements.

### For Image 2 (any type):
- Preserve all fixed structures: buildings, walls, fences, existing permanent landscaping.
- Match the lighting direction, time of day, and atmospheric conditions of Image 2.
- Match the perspective and camera angle of Image 2.
- The final result must look photorealistic — not a composite or collage.

## Nano Banana Pro prompting formula to follow

Use this structure for the generated prompt:
[Strong opening verb] + [Image 1 reference — what to take from it] + [Image 2 reference — what the scene is] + [Specific transformation instruction] + [What to preserve] + [Photorealism/Composition notes] + [Lighting and atmosphere]

Always refer to images as "the first reference image" (Image 1) and "the second reference image" (Image 2) — Nano Banana Pro will receive them in this exact order.

Start the prompt with a strong verb: "Apply", "Place", "Transform", "Blend", "Integrate", "Render".

Be specific about materials, textures, finishes, colours, and spatial relationships.
Use positive framing — describe what you want, not what you don't want.

## Output format

Return a JSON object with exactly these two fields:
{
  "image_generation_prompt": "<the complete, detailed Nano Banana Pro prompt>",
  "operation_type": "<one of: blend, place, style_transfer, combine>"
}
""".strip()


def generate_prompt(
    image1: Image.Image,
    image1_name: str,
    image2: Image.Image,
    image2_name: str | None = None,
) -> str:
    """
    Uses gemini-3.1-flash-lite-preview to analyse the two images and generate
    an optimal Nano Banana Pro prompt.

    Args:
        image1:      The primary Ducon catalog image (PIL Image).
        image1_name: Display name of the Ducon image from the DB.
        image2:      The second image — user upload or another Ducon image.
        image2_name: Name of the second Ducon image, or None if it's a user upload.

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

    response = client.models.generate_content(
        model=PROMPT_GEN_MODEL,
        contents=[image1, image2, context],
        config=GenerateContentConfig(
            system_instruction=_PROMPT_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
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
    Sends both images and the generated prompt to Nano Banana Pro for image generation.
    Images are passed in order: image1 = "the first reference image", image2 = "the second reference image".
    """
    client = get_gemini_client()

    response = client.models.generate_content(
        model=IMAGE_GEN_MODEL,
        contents=[image1, image2, prompt],
        config=GenerateContentConfig(
            response_modalities=[Modality.TEXT, Modality.IMAGE],
        ),
    )

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
