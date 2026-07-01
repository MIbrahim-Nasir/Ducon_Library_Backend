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

from app.prompt_loader import load_prompts
from app.studio_directions_agent import curate_studio_directions, stream_studio_directions_events

logger = logging.getLogger(__name__)

load_prompts()

from app.routers.auth import router as auth_router
from app.routers.bookmarks import router as bookmarks_router
from app.routers.generations import router as generations_router
from app.routers.images import router as images_router
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
from app.db.database import get_db
from app.db.models import Generation, GuestGeneration, Image as DBImage
from app.auth import get_current_user, get_optional_user
from app.rate_limiter import require_rate_limit
from app.sse import SSE_HEADERS
from app.image_utils import (
    load_ducon_image as _load_ducon_image,
    normalize_user_image as _normalize_user_image,
    get_image_metadata as _get_image_metadata,
    load_image_info as _load_image_info,
)
from datetime import datetime, timedelta, timezone


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_prompts()
    app.state.collection = chromadb.get_db_collection()
    app.state.embedding_model = GeminiEmbeddingModel()
    yield

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

# Serve public library images statically
app.mount("/public/images", StaticFiles(directory="data/images"), name="public_images")

from app.config import get_cors_origins

_cors_origins = get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    require_rate_limit(request, max_requests=30, window_seconds=60, key_prefix="search")

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

    results = chromadb.retrieve(collection, embedding, limit)
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
    db: AsyncSession = Depends(get_db),
):
    """
    Step 4 curator: AI search agent picks 9 catalog directions for the studio wizard.
    """
    # Public (guest-accessible) but very expensive (embeddings + LLM curation).
    require_rate_limit(request, max_requests=15, window_seconds=60, key_prefix="studio_directions")

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
        )
    except Exception as exc:
        logger.exception("[StudioDirections] endpoint error")
        raise HTTPException(
            status_code=500,
            detail="Studio direction curation failed.",
        ) from exc


_VALID_ASPECT_RATIOS = {"1:1", "4:3", "3:4", "16:9", "9:16"}


def _infer_aspect_ratio(width: int, height: int) -> str | None:
    """Snap raw pixel dimensions to the nearest supported generation ratio.

    Used as a server-side fallback when the client does not send an explicit
    aspect_ratio, so the output matches the user's space photo instead of
    whatever ratio Nano Banana picks on its own.
    """
    if not width or not height:
        return None
    target = width / height
    candidates = {"1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "16:9": 16 / 9, "9:16": 9 / 16}
    return min(candidates, key=lambda r: abs(candidates[r] - target))


@app.post("/autogenerate-images")
@app.post("/autogenerate-images/")
async def auto_generate_images(
    request: Request,
    image_id: Optional[int] = Form(None),
    ducon_reference: Optional[UploadFile] = File(None),
    ducon_reference_name: Optional[str] = Form(None),
    user_image: Optional[UploadFile] = File(None),
    second_image_id: Optional[int] = Form(None),
    prompt: str = Form(None),
    enhance_prompt: bool = Form(True),
    use_image_info: bool = Form(True),
    aspect_ratio: Optional[str] = Form(None),
    cf_turnstile_token: Optional[str] = Form(None),
    current_user=Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    # ── Auth gate ─────────────────────────────────────────────────────────────
    # Accept authenticated users (Bearer token) OR guests (X-Guest-Session-Id).
    is_guest = current_user is None
    if is_guest:
        raw_session_id = request.headers.get("x-guest-session-id")
        if not raw_session_id:
            raise HTTPException(
                status_code=400,
                detail="Authentication required. Provide a Bearer token or X-Guest-Session-Id header.",
            )
        try:
            uuid.UUID(raw_session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid X-Guest-Session-Id. Must be a valid UUID.")

        client_ip = request.client.host if request.client else "unknown"

        if not await verify_turnstile(cf_turnstile_token or "", client_ip):
            raise HTTPException(
                status_code=403,
                detail="Bot verification failed. Please try again.",
            )

        guest_session = await get_or_create_guest_session(db, raw_session_id, client_ip)

    # ── Image validation ──────────────────────────────────────────────────────
    if image_id is None and ducon_reference is None:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'image_id' (catalog id) or 'ducon_reference' (design image upload).",
        )
    if image_id is not None and ducon_reference is not None:
        raise HTTPException(
            status_code=422,
            detail="Provide only one of 'image_id' or 'ducon_reference', not both.",
        )
    if user_image is None and second_image_id is None:
        raise HTTPException(status_code=422, detail="Provide either 'user_image' (file upload) or 'second_image_id' (Ducon image id).")
    if user_image is not None and second_image_id is not None:
        raise HTTPException(status_code=422, detail="Provide only one of 'user_image' or 'second_image_id', not both.")
    if aspect_ratio and aspect_ratio not in _VALID_ASPECT_RATIOS:
        raise HTTPException(status_code=422, detail=f"aspect_ratio must be one of: {_VALID_ASPECT_RATIOS}.")

    # ── Resolve primary Ducon image ───────────────────────────────────────────
    if image_id is not None:
        result = await db.execute(select(DBImage).where(DBImage.id == image_id))
        ducon_db_image = result.scalar_one_or_none()
        if not ducon_db_image:
            raise HTTPException(status_code=404, detail="Image not found")
        ducon_image = await _load_ducon_image(ducon_db_image)
        ducon_image_name = ducon_db_image.name or ducon_db_image.filename
    else:
        ducon_db_image = None
        ref_data = await ducon_reference.read()
        ducon_image = _normalize_user_image(ref_data)
        ducon_image_name = (ducon_reference_name or "").strip() or "Ducon Design"

    # ── Resolve second image ──────────────────────────────────────────────────
    second_image_name: str | None = None
    second_db_image = None
    if user_image is not None:
        image_data = await user_image.read()
        user_img = _normalize_user_image(image_data)
    else:
        result2 = await db.execute(select(DBImage).where(DBImage.id == second_image_id))
        second_db_image = result2.scalar_one_or_none()
        if not second_db_image:
            raise HTTPException(status_code=404, detail=f"Second image with id {second_image_id} not found")
        user_img = await _load_ducon_image(second_db_image)
        second_image_name = second_db_image.name or second_db_image.filename

    # Lock the output ratio to the user's space photo: prefer the client-sent
    # value, else infer it from the second (user-space) image dimensions.
    active_aspect_ratio = aspect_ratio or _infer_aspect_ratio(*user_img.size)

    async def run_generation() -> dict:
        # ── Step 1: Build Nano Banana prompt ─────────────────────────────────
        # enhance_prompt=True (default): unified ImageGenAgent writes prompt + evaluates retries.
        # enhance_prompt=False: skip agent, use user prompt directly.
        active_prompt = prompt
        image_gen_agent: ImageGenAgent | None = None
        if enhance_prompt:
            image_gen_agent = ImageGenAgent(
                image1_name=ducon_image_name,
                image2_name=second_image_name,
                image2_is_user_space=second_image_name is None,
            )
            active_prompt = await image_gen_agent.generate_initial_prompt(
                image1=ducon_image,
                image2=user_img,
                image1_metadata=_get_image_metadata(ducon_db_image.filename) if (ducon_db_image and use_image_info) else None,
                image2_metadata=_get_image_metadata(second_db_image.filename) if (second_db_image and use_image_info) else None,
                user_hint=prompt or None,
            )
        else:
            if not active_prompt:
                raise HTTPException(
                    status_code=422,
                    detail="A prompt is required when prompt enhancement is off.",
                )

        # ── Step 2: Generation + post-evaluation loop ─────────────────────────
        ducon_stem = ducon_db_image.filename if ducon_db_image else "reference"
        generation_filename = f"{ducon_stem}_{uuid.uuid4()}.png"

        if is_guest:
            subfolder = f"guests/{raw_session_id}"
        else:
            subfolder = str(current_user.id)

        local_output = Path("outputs") / subfolder / generation_filename
        max_gen_rounds = gemini.GEN_EVAL_MAX_ROUNDS if enhance_prompt else 1
        generation_approved = False
        last_eval_issues: list[str] = []

        for gen_round in range(max_gen_rounds):
            # gemini.combine_images is blocking. Run it off the event loop so
            # SSE keepalives can continue while Cloudflare Tunnel is waiting.
            await asyncio.to_thread(
                gemini.combine_images,
                generation_filename,
                ducon_image,
                user_img,
                prompt=active_prompt,
                subfolder=subfolder,
                aspect_ratio=active_aspect_ratio,
            )

            if not enhance_prompt:
                break

            try:
                generated_img = Image.open(local_output)
                approved, issues = await image_gen_agent.evaluate_output(
                    generated=generated_img,
                    gen_round=gen_round + 1,
                )
                if approved:
                    generation_approved = True
                    logger.info("[ImageGenAgent] Approved on generation round %d", gen_round + 1)
                    image_gen_agent.schedule_post_success_improvement()
                    break
                last_eval_issues = list(issues or [])
                if gen_round + 1 < max_gen_rounds:
                    logger.info(
                        "[ImageGenAgent] Rejected on round %d (%d issue(s)) — requesting improved prompt",
                        gen_round + 1, len(issues),
                    )
                    active_prompt = await image_gen_agent.generate_retry_prompt(
                        gen_round=gen_round + 1,
                        issues=issues,
                    )
                else:
                    logger.info(
                        "[ImageGenAgent] Rejected on round %d — max rounds reached, keeping last output",
                        gen_round + 1,
                    )
                    break
            except Exception as e:
                # Evaluation failure must not block the response — proceed with current image
                logger.warning("[ImageGenAgent] Skipped on generation round %d due to error: %s", gen_round + 1, e)
                break

        # ── Step 3: Persist & respond ─────────────────────────────────────────
        if is_guest:
            stored_key = storage.save_guest_generation(raw_session_id, generation_filename)
            expires_at = datetime.now(timezone.utc) + timedelta(hours=48)

            db_generation = GuestGeneration(
                guest_session_id=raw_session_id,
                generation_name=generation_filename,
                url=stored_key,
                ducon_image_id=ducon_db_image.id if ducon_db_image else None,
                expires_at=expires_at,
            )
            db.add(db_generation)
            await increment_guest_count(db, guest_session)
            await log_guest_consent(db, raw_session_id, client_ip)
            await db.flush()
            await db.commit()

            signed_url = storage.get_guest_generation_url(db_generation.id, stored_key)
            guest_result = {
                "id": db_generation.id,
                "generation_name": generation_filename,
                "signed_url": signed_url,
                "ducon_image_id": ducon_db_image.id if ducon_db_image else None,
                "expires_at": expires_at.isoformat(),
            }
            if image_gen_agent is not None and image_gen_agent.input_quality \
                    and not image_gen_agent.input_quality.get("ok"):
                guest_result["input_quality"] = image_gen_agent.input_quality
            if enhance_prompt and not generation_approved and last_eval_issues:
                gen_notice = gemini.build_quality_notice(
                    last_eval_issues,
                    default_message=(
                        "We saved the closest result, but some quality checks were not fully met. "
                        "You can continue in chat to refine or try another direction."
                    ),
                    title_prefix="Visualization quality",
                )
                if gen_notice:
                    guest_result["generation_warnings"] = gen_notice
            return guest_result

        stored_key = storage.save_generation(current_user.id, generation_filename)

        db_generation = Generation(
            user_id=current_user.id,
            generation_name=generation_filename,
            url=stored_key,
            ducon_image_id=ducon_db_image.id if ducon_db_image else None,
        )
        db.add(db_generation)
        await db.flush()
        await db.commit()

        signed_url = storage.get_generation_url(db_generation.id, stored_key)
        user_result = {
            "id": db_generation.id,
            "generation_name": generation_filename,
            "url": stored_key,
            "signed_url": signed_url,
            "ducon_image_id": ducon_db_image.id if ducon_db_image else None,
        }
        if image_gen_agent is not None and image_gen_agent.input_quality \
                and not image_gen_agent.input_quality.get("ok"):
            user_result["input_quality"] = image_gen_agent.input_quality
        if enhance_prompt and not generation_approved and last_eval_issues:
            gen_notice = gemini.build_quality_notice(
                last_eval_issues,
                default_message=(
                    "We saved the closest result, but some quality checks were not fully met. "
                    "You can continue in chat to refine or try another direction."
                ),
                title_prefix="Visualization quality",
            )
            if gen_notice:
                user_result["generation_warnings"] = gen_notice
        return user_result

    async def stream_autogeneration():
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

        async def worker():
            try:
                result = await run_generation()
                await queue.put(("done", result))
            except HTTPException as exc:
                await queue.put(("error", exc.detail or "Generation failed."))
            except Exception:
                logger.exception("[AutoGenerate] stream error")
                await queue.put(("error", "Generation failed. Please try again."))

        task = asyncio.create_task(worker())
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generation started'})}\n\n"
            while True:
                try:
                    event_type, payload = await asyncio.wait_for(queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                if event_type == "done":
                    yield f"data: {json.dumps({'type': 'done', **payload})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n"
                break
        finally:
            await task

    return StreamingResponse(
        stream_autogeneration(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def main():
    print("Hello from ducon-library-backend!")


if __name__ == "__main__":
    main()
