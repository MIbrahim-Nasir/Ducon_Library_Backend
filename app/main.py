from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from contextlib import asynccontextmanager
from . import chromadb
from app.ml import GeminiEmbeddingModel
import io
import os
import httpx
from PIL import Image
import pillow_heif
from fastapi.middleware.cors import CORSMiddleware

# Register HEIC/HEIF support so Pillow can open iPhone images
pillow_heif.register_heif_opener()
from fastapi.staticfiles import StaticFiles
from typing import Optional
from . import gemini
from . import storage
from pathlib import Path
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from dotenv import load_dotenv

load_dotenv()

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
from app.image_utils import (
    load_ducon_image as _load_ducon_image,
    normalize_user_image as _normalize_user_image,
    get_image_metadata as _get_image_metadata,
    load_image_info as _load_image_info,
)
from datetime import datetime, timedelta, timezone


@asynccontextmanager
async def lifespan(app: FastAPI):
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"], # Allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)



# @app.get("/")
# def read_root(request: Request):
#     col = request.app.state.collection

#     return {"item_count": request.app.state.collection.count() }

def _mime_type(filename: str) -> str:
    from app.image_utils import mime_type
    return mime_type(filename)


@app.post("/search")
async def search(
    query: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    limit: int = 5,
):
    model: GeminiEmbeddingModel = app.state.embedding_model
    collection = app.state.collection

    if query and file:
        image_bytes = await file.read()
        embedding = model.get_multimodal_embedding(query, image_bytes, _mime_type(file.filename))
    elif query:
        embedding = model.get_text_embedding(query)
    elif file:
        image_bytes = await file.read()
        embedding = model.get_image_embedding(image_bytes, _mime_type(file.filename))
    else:
        return {"results": []}

    results = chromadb.retrieve(collection, embedding, limit)
    return {"results": results["ids"]}

@app.post("/autogenerate-images")
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

    # ── Step 1: Build Nano Banana prompt ─────────────────────────────────────
    # enhance_prompt=True (default): the prompt generator runs; user prompt is a hint.
    # enhance_prompt=False: skip generation/verification, use user prompt directly.
    # These are used in both the pre-gen verify and the re-verify after rejection.
    verify_images: list = []
    verify_labels: list = []
    if enhance_prompt:
        prompt = gemini.generate_prompt(
            image1=ducon_image,
            image1_name=ducon_image_name,
            image1_metadata=_get_image_metadata(ducon_db_image.filename) if (ducon_db_image and use_image_info) else None,
            image2=user_img,
            image2_name=second_image_name,
            image2_metadata=_get_image_metadata(second_db_image.filename) if (second_db_image and use_image_info) else None,
            user_hint=prompt or None,
        )

        # ── Step 1b: Pre-generation prompt verification loop ──────────────────
        # The verifier checks image references, camera lock, preservation rules,
        # product-fidelity rules, and coherence before sending to the image model.
        # Each failed round feeds the improved prompt back into the verifier.
        verify_images = [ducon_image, user_img]
        verify_labels = [
            f'Ducon reference: "{ducon_image_name}"',
            second_image_name and f'Second Ducon image: "{second_image_name}"' or "User's outdoor space",
        ]
        for vround in range(gemini.PROMPT_VERIFY_MAX_ROUNDS):
            try:
                passed, issues, improved_prompt = gemini.verify_prompt(
                    images=verify_images,
                    labels=verify_labels,
                    prompt=prompt,
                )
                if passed:
                    print(f"[PromptVerifier] Passed on round {vround + 1}")
                    break
                if improved_prompt:
                    print(f"[PromptVerifier] Round {vround + 1} failed ({len(issues)} issue(s)), using improved prompt")
                    prompt = improved_prompt
                else:
                    print(f"[PromptVerifier] Round {vround + 1} failed but no improvement available, proceeding")
                    break
            except Exception as e:
                print(f"[PromptVerifier] Skipped on round {vround + 1} due to error: {e}")
                break
    else:
        if not prompt:
            raise HTTPException(
                status_code=422,
                detail="A prompt is required when enhance_prompt is disabled.",
            )

    # ── Step 2: Generation + post-evaluation loop ─────────────────────────────
    # On the first pass: generate with the (verified) prompt.
    # On rejection: regenerate up to GEN_EVAL_MAX_ROUNDS times using the revised
    # prompt from the evaluator. All attempts write to the same local file so only
    # the final approved (or best-effort last) image is uploaded to storage.
    ducon_stem = ducon_db_image.filename if ducon_db_image else "reference"
    generation_filename = f"{ducon_stem}_{uuid.uuid4()}.png"

    if is_guest:
        subfolder = f"guests/{raw_session_id}"
    else:
        subfolder = str(current_user.id)

    local_output = Path("outputs") / subfolder / generation_filename
    max_gen_rounds = gemini.GEN_EVAL_MAX_ROUNDS if enhance_prompt else 1

    for gen_round in range(max_gen_rounds):
        gemini.combine_images(
            generation_filename,
            ducon_image,
            user_img,
            prompt=prompt,
            subfolder=subfolder,
        )

        if not enhance_prompt:
            break  # no evaluation when prompt enhancement is disabled

        try:
            generated_img = Image.open(local_output)
            approved, revised_prompt, issues = gemini.evaluate_generation(
                image1=ducon_image,
                image2=user_img,
                generated=generated_img,
                prompt_used=prompt,
                image1_name=ducon_image_name,
                image2_name=second_image_name,
            )
            if approved:
                print(f"[Evaluator] Approved on generation round {gen_round + 1}")
                break
            if revised_prompt and gen_round + 1 < max_gen_rounds:
                print(
                    f"[Evaluator] Rejected on round {gen_round + 1} "
                    f"({len(issues)} issue(s)) — regenerating with revised prompt"
                )
                # Run the revised prompt back through the pre-gen verifier so it
                # also passes structural rules before the next generation attempt.
                for vround in range(gemini.PROMPT_VERIFY_MAX_ROUNDS):
                    try:
                        v_passed, _, v_improved = gemini.verify_prompt(
                            images=verify_images,
                            labels=verify_labels,
                            prompt=revised_prompt,
                        )
                        if v_passed or not v_improved:
                            break
                        revised_prompt = v_improved
                    except Exception as ve:
                        print(f"[PromptVerifier] Skipped on re-verify round {vround + 1}: {ve}")
                        break
                prompt = revised_prompt
            else:
                print(
                    f"[Evaluator] Rejected on round {gen_round + 1} — "
                    f"no revised prompt available or max rounds reached, keeping last output"
                )
                break
        except Exception as e:
            # Evaluation failure must not block the response — proceed with current image
            print(f"[Evaluator] Skipped on generation round {gen_round + 1} due to error: {e}")
            break

    # ── Step 3: Persist & respond ─────────────────────────────────────────────
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
        return {
            "id": db_generation.id,
            "generation_name": generation_filename,
            "signed_url": signed_url,
            "ducon_image_id": ducon_db_image.id if ducon_db_image else None,
            "expires_at": expires_at.isoformat(),
        }
    else:
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
        return {
            "id": db_generation.id,
            "generation_name": generation_filename,
            "url": stored_key,
            "signed_url": signed_url,
            "ducon_image_id": ducon_db_image.id if ducon_db_image else None,
        }


def main():
    print("Hello from ducon-library-backend!")


if __name__ == "__main__":
    main()
