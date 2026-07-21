from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from contextlib import asynccontextmanager
from . import chromadb
from app.ml import GeminiEmbeddingModel
import asyncio
import io
import json
import logging
import os
import httpx
from PIL import Image
import pillow_heif
from fastapi.middleware.cors import CORSMiddleware

# Register HEIC/HEIF support so Pillow can open iPhone images
pillow_heif.register_heif_opener()
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from typing import Optional
from . import gemini
from . import storage
from app.image_gen_agent import ImageGenAgent
from pathlib import Path
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from dotenv import load_dotenv

load_dotenv()

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from app.prompt_loader import load_prompts
from app.studio_directions_agent import curate_studio_directions, stream_studio_directions_events

logger = logging.getLogger(__name__)

load_prompts()

from app.routers.auth import router as auth_router
from app.routers.bookmarks import router as bookmarks_router
from app.routers.generations import router as generations_router
from app.routers.images import router as images_router
from app.guest_identity import build_guest_request_identity
from app.routers.guest import (
    router as guest_router,
    get_or_create_guest_session,
    increment_guest_count,
    log_guest_consent,
    verify_turnstile,
)
from app.routers.voice import router as voice_router
from app.routers.quotation import router as quotation_router
from app.routers.chat import router as chat_router
from app.routers.multi_image_gen import router as multi_image_gen_router
from app.routers.designer_jobs import router as designer_jobs_router
from app.routers.contact import router as contact_router
from app.routers.meta import router as meta_router
from app.db.database import get_db
from app.db.models import Generation, GuestGeneration, Image as DBImage
from app.auth import get_current_user, get_optional_user
from app.rate_limiter import require_rate_limit
from app.sse import SSE_HEADERS
from app.error_logger import log_error, log_warning
from datetime import datetime, timedelta, timezone


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: loading prompts")
    load_prompts()
    logger.info("Startup: opening ChromaDB collection")
    app.state.collection = chromadb.get_db_collection()
    logger.info("Startup: initializing embedding model")
    app.state.embedding_model = GeminiEmbeddingModel()
    # ── Admin / metrics subsystem startup ────────────────────────────────
    from app.db.database import Base, async_session_maker, engine
    import app.db.models  # noqa: F401 — register all tables on Base.metadata
    from app.admin.settings_store import get_settings_store
    from app.admin.usage_recorder import get_usage_recorder
    # Auto-create any admin/metrics tables missing in the DB (idempotent).
    logger.info("Startup: connecting to Postgres and ensuring schema")
    try:
        async with engine.begin() as conn:
            # gunicorn -w 3: every worker runs this concurrently. Serialize the
            # DDL behind a Postgres advisory lock so CREATE TABLE IF NOT EXISTS
            # cannot race (pg_type duplicate-key crash-loops on cold starts).
            from sqlalchemy import text as _sql_text
            await conn.execute(_sql_text("SELECT pg_advisory_xact_lock(788214001)"))
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        logger.error(
            "Startup failed: could not connect to Postgres or create schema "
            "(check DATABASE_URL and that Postgres is running): %s",
            exc,
        )
        raise RuntimeError(
            "Database startup failed — verify DATABASE_URL and that Postgres is reachable"
        ) from exc
    logger.info("Startup: loading admin settings from database")
    try:
        async with async_session_maker() as _db:
            await get_settings_store().load_all(_db)
    except Exception as exc:
        logger.warning(
            "Startup: admin settings load failed — continuing with env/default values: %s",
            exc,
        )
    logger.info("Startup: starting usage recorder")
    get_usage_recorder().start()
    from app.error_logger import get_error_logger
    logger.info("Startup: starting error logger")
    get_error_logger().start()

    async def _rate_limit_cleanup_loop():
        from app.rate_limiter import cleanup_expired_windows, cleanup_old_keys
        while True:
            await asyncio.sleep(300)
            try:
                removed = cleanup_old_keys(3600)
                if removed:
                    logger.debug("rate_limiter cleaned %s stale keys", removed)
                await cleanup_expired_windows(3600)
            except Exception:
                logger.exception("rate_limiter cleanup failed")

    cleanup_task = asyncio.create_task(_rate_limit_cleanup_loop(), name="rate-limit-cleanup")
    logger.info("Startup complete")
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        # Cancel in-flight generation jobs so their CancelledError handlers can
        # persist a terminal status — otherwise a worker restart strands rows
        # in status='running' and cross-worker SSE pollers loop forever.
        try:
            from app.generation_jobs import JOBS as _GEN_JOBS
            in_flight = [j.task for j in _GEN_JOBS.values() if j.task and not j.task.done()]
            for t in in_flight:
                t.cancel()
            if in_flight:
                logger.info("Shutdown: cancelling %d in-flight generation job(s)", len(in_flight))
                await asyncio.wait(in_flight, timeout=10)
        except Exception:
            logger.exception("Shutdown: generation job drain failed")
        logger.info("Shutdown: stopping background writers")
        from app.error_logger import get_error_logger
        await get_error_logger().stop()
        await get_usage_recorder().stop()
        try:
            from app.observability.langfuse_client import shutdown as langfuse_shutdown
            langfuse_shutdown()
        except Exception:
            logger.debug("Langfuse shutdown skipped", exc_info=True)
        logger.info("Shutdown: disposing database connection pool")
        await engine.dispose()
        logger.info("Shutdown complete")

app = FastAPI(lifespan=lifespan)

app.include_router(auth_router)
app.include_router(bookmarks_router)
app.include_router(generations_router)
app.include_router(images_router)
app.include_router(guest_router)
app.include_router(voice_router)
app.include_router(quotation_router)
app.include_router(chat_router)
app.include_router(multi_image_gen_router)
app.include_router(designer_jobs_router)
app.include_router(contact_router)
app.include_router(meta_router)
from app.routers.admin import router as admin_router
app.include_router(admin_router)

from app.config import get_cors_origins, IS_PRODUCTION
from app.middleware.cache_control import HtmlNoCacheMiddleware
from app.middleware.request_id import RequestIdMiddleware
from app.routers.generation_jobs import router as generation_jobs_router

app.include_router(generation_jobs_router)

# ── Dev-only benchmark dashboard router (mounted only when explicitly enabled
#    AND not in production). Fail-closed in prod.
if os.getenv("DEV_DASHBOARD_ENABLED", "").lower() in ("1", "true", "yes") and not IS_PRODUCTION:
    from app.routers.dev_benchmark import router as dev_benchmark_router
    app.include_router(dev_benchmark_router)

# Serve public library images statically
os.makedirs("data/images", exist_ok=True)
app.mount("/public/images", StaticFiles(directory="data/images"), name="public_images")

# Request-ID outermost among app middlewares so contextvar covers route handlers
# and error logging. Starlette runs add_middleware in reverse registration order.
app.add_middleware(RequestIdMiddleware)
app.add_middleware(HtmlNoCacheMiddleware)

_cors_origins = get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)



# @app.get("/")
# def read_root(request: Request):
#     col = request.app.state.collection

#     return {"item_count": request.app.state.collection.count() }

def _mime_type(filename: str) -> str:
    from app.image_utils import mime_type
    return mime_type(filename)


# Upper bound on catalog search results to prevent scraping the whole index.
_SEARCH_MAX_LIMIT = 50
# Cap uploaded search/curation images to protect memory and the embedding model.
_MAX_SEARCH_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB


@app.post("/search")
async def search(
    request: Request,
    query: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    limit: int = 5,
):
    # Public (guest-accessible) but expensive: rate limit per IP so it can't be
    # used to hammer the embedding model or scrape the catalog.
    await require_rate_limit(request, max_requests=30, window_seconds=60, key_prefix="search")

    limit = max(1, min(limit, _SEARCH_MAX_LIMIT))
    model: GeminiEmbeddingModel = app.state.embedding_model
    collection = app.state.collection

    # Embedding calls are blocking SDK calls — run them off the event loop so
    # they don't stall other requests.
    if query and file:
        image_bytes = await file.read()
        _guard_image_size(image_bytes)
        embedding = await asyncio.to_thread(
            model.get_multimodal_embedding, query, image_bytes, _mime_type(file.filename)
        )
    elif query:
        embedding = await asyncio.to_thread(model.get_text_embedding, query)
    elif file:
        image_bytes = await file.read()
        _guard_image_size(image_bytes)
        embedding = await asyncio.to_thread(
            model.get_image_embedding, image_bytes, _mime_type(file.filename)
        )
    else:
        return {"results": []}

    results = await asyncio.to_thread(chromadb.retrieve, collection, embedding, limit)
    return {"results": results["ids"]}


def _guard_image_size(image_bytes: bytes) -> None:
    if len(image_bytes) > _MAX_SEARCH_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Maximum size is {_MAX_SEARCH_IMAGE_BYTES // (1024 * 1024)} MB.",
        )


@app.post("/studio/directions")
async def studio_directions(
    request: Request,
    space: str = Form(...),
    style: str = Form(...),
    file: UploadFile = File(...),
    stream: str = Form("true"),
    current_user=Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Step 4 curator: AI search agent picks 9 catalog directions for the studio wizard.
    """
    # Public (guest-accessible) but very expensive (embeddings + LLM curation).
    await require_rate_limit(request, max_requests=15, window_seconds=60, key_prefix="studio_directions")

    try:
        space_data = json.loads(space)
        style_data = json.loads(style)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Invalid space/style JSON.") from exc

    if not isinstance(space_data, dict) or not isinstance(style_data, dict):
        raise HTTPException(status_code=422, detail="space and style must be JSON objects.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="User space photo is required.")
    _guard_image_size(image_bytes)

    user_mime = _mime_type(file.filename or "space.jpg")
    collection = request.app.state.collection
    embedding_model: GeminiEmbeddingModel = app.state.embedding_model

    user_id = int(current_user.id) if current_user else None
    guest_session_id = None
    if user_id is None:
        from app.guest_session_token import resolve_guest_session_id
        guest_session_id = resolve_guest_session_id(request)

    use_stream = str(stream).strip().lower() in ("true", "1", "yes", "on")
    if use_stream:
        return StreamingResponse(
            stream_studio_directions_events(
                collection=collection,
                embedding_model=embedding_model,
                user_image_bytes=image_bytes,
                user_mime=user_mime,
                space=space_data,
                style=style_data,
                user_id=user_id,
                guest_session_id=guest_session_id,
            ),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    try:
        return await curate_studio_directions(
            db=db,
            collection=collection,
            embedding_model=embedding_model,
            user_image_bytes=image_bytes,
            user_mime=user_mime,
            space=space_data,
            style=style_data,
            user_id=user_id,
            guest_session_id=guest_session_id,
        )
    except Exception as exc:
        logger.exception("[StudioDirections] endpoint error")
        raise HTTPException(
            status_code=500,
            detail="Studio direction curation failed.",
        ) from exc
