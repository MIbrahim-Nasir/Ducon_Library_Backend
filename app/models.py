from app.ml import TextEmbeddingModel, MultimodalEmbeddingModel
import asyncio

text_model = None
image_model = None

text_lock = asyncio.Lock()
image_lock = asyncio.Lock()


async def preload_models(app):
    await get_text_model(app)
    await get_image_model(app)


async def get_text_model(app=None):
    global text_model
    
    if text_model is None:
        async with text_lock:
            if text_model is None:
                text_model = TextEmbeddingModel()
                if app:
                    app.state.text_model = text_model

    return text_model

async def get_image_model(app=None):
    global image_model

    if image_model is None:
        async with image_lock:
            if image_model is None:
                image_model = MultimodalEmbeddingModel()
                if app:
                    app.state.image_model = image_model

    return image_model
