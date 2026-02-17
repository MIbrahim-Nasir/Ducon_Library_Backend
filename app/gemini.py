from google import genai
from google.genai.types import GenerateContentConfig, Modality
from PIL import Image
from io import BytesIO

from dotenv import load_dotenv
import os

load_dotenv()

# GLOBALS
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")

MODEL_ID = "gemini-3-pro-image-preview"

_client = None


def get_gemini_client():
    global _client
    if _client is None:
        _client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    return _client

# image1 = Image.open("output_folder/generation.png")
# image2 = Image.open("output_folder/4x60+slate-kh T2. Bump.jpg")

def generate_image(image1, image2, prompt=None):
    client = get_gemini_client()
    if prompt is None or prompt == "":
        print("no prompt given")
        return
    else:
        response = client.models.generate_content(
        model=MODEL_ID,
        contents=(image1, image2, prompt),
        config=GenerateContentConfig(
                response_modalities=[Modality.TEXT, Modality.IMAGE],
            ),
        )

        save_image(response=response)

def combine_images(filename, image1: Image.Image, image2: Image.Image, prompt=None, ):
    client = get_gemini_client()
    if prompt is None or prompt == "":
        print("no prompt given")
        return
    else:
        response = client.models.generate_content(
        model=MODEL_ID,
        contents=(image1, image2, prompt),
        config=GenerateContentConfig(
                response_modalities=[Modality.TEXT, Modality.IMAGE],
            ),
        )

        image_path = save_image(filename, response=response, )
        return image_path

def save_image(filename, response):
    
    for part in response.candidates[0].content.parts:
        if part.text:
            print(part.text)
        elif part.inline_data:
            image = Image.open(BytesIO((part.inline_data.data)))
            # Ensure the output directory exists
            output_dir = "output_folder"
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename)
            image.save(path)
            return path

def list_available_models():
    client = get_gemini_client()
    models = client.models.list()
    for model in models:
        if "image" in model.name.lower():
            print(model.name)


if __name__ == "__main__":
    generate_image("add image 2 on the left half of image 1")