from app.ml import (
    TextEmbeddingModel,
    MultimodalEmbeddingModel,
    get_active_embedding_profile,
    get_collection_names,
)
import asyncio

text_model = None
image_model = None
preload_task = None

text_lock = asyncio.Lock()
image_lock = asyncio.Lock()


async def preload_models(app):
    global text_model, image_model

    if text_model is None or image_model is None:
        async with text_lock:
            async with image_lock:
                if text_model is None or image_model is None:
                    loaded_text_model, loaded_image_model = await asyncio.gather(
                        asyncio.to_thread(TextEmbeddingModel),
                        asyncio.to_thread(MultimodalEmbeddingModel),
                    )
                    text_model = loaded_text_model
                    image_model = loaded_image_model

    if app:
        app.state.text_model = text_model
        app.state.image_model = image_model
        app.state.embedding_profile = get_active_embedding_profile()
        app.state.embedding_collections = get_collection_names()


def start_preload_models(app):
    """Kick off model loading in background without blocking server startup."""
    global preload_task
    if preload_task is None or preload_task.done():
        preload_task = asyncio.create_task(preload_models(app))


async def get_text_model(app=None):
    global text_model
    
    if text_model is None:
        if preload_task is not None and not preload_task.done():
            await preload_task
        async with text_lock:
            if text_model is None:
                text_model = await asyncio.to_thread(TextEmbeddingModel)
                if app:
                    app.state.text_model = text_model

    return text_model

async def get_image_model(app=None):
    global image_model

    if image_model is None:
        if preload_task is not None and not preload_task.done():
            await preload_task
        async with image_lock:
            if image_model is None:
                image_model = await asyncio.to_thread(MultimodalEmbeddingModel)
                if app:
                    app.state.image_model = image_model

    return image_model
