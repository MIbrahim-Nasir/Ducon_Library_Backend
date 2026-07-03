"""Contact form submissions — designer consultations and customer service."""

from __future__ import annotations

import base64
import json
import logging
import os
from datetime import date, timedelta
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import storage
from app.auth import get_current_user
from app.db.database import get_db
from app.db.models import Bookmark, Generation, Image, User
from app.email_service import send_contact_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/contact", tags=["contact"])

DESIGNER_CONTACT_EMAIL = os.getenv("DESIGNER_CONTACT_EMAIL", "")
CUSTOMER_SERVICE_EMAIL = os.getenv("CUSTOMER_SERVICE_EMAIL", "")

VALID_METHODS = {"email", "whatsapp", "phone"}
VALID_TIMES = {"morning", "afternoon"}


def _valid_slot_dates() -> set[str]:
    """Next 7 calendar days excluding Sundays."""
    out: set[str] = set()
    d = date.today()
    while len(out) < 7:
        d += timedelta(days=1)
        if d.weekday() == 6:  # Sunday
            continue
        out.add(d.isoformat())
    return out


def _parse_methods(raw: str) -> list[str]:
    try:
        methods = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Invalid contact_methods JSON") from exc
    if not isinstance(methods, list) or not methods:
        raise HTTPException(status_code=422, detail="Select at least one contact method")
    normalized = []
    for m in methods:
        if not isinstance(m, str):
            raise HTTPException(status_code=422, detail="Invalid contact method")
        key = m.strip().lower()
        if key not in VALID_METHODS:
            raise HTTPException(status_code=422, detail=f"Unknown contact method: {m}")
        if key not in normalized:
            normalized.append(key)
    return normalized


def _parse_id_list(raw: str, field: str) -> list[int]:
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid {field} JSON") from exc
    if not isinstance(data, list):
        raise HTTPException(status_code=422, detail=f"{field} must be a JSON array")
    out: list[int] = []
    for item in data:
        try:
            out.append(int(item))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid id in {field}") from exc
    return out


def _validate_schedule(
    methods: list[str],
    preferred_date: Optional[str],
    preferred_time: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    allowed = _valid_slot_dates()
    if preferred_date and preferred_date not in allowed:
        raise HTTPException(status_code=422, detail="Preferred date is not available")

    if "phone" in methods:
        if not preferred_date:
            raise HTTPException(status_code=422, detail="Preferred date is required for phone calls")
        if preferred_time not in VALID_TIMES:
            raise HTTPException(status_code=422, detail="Preferred time must be morning or afternoon")
    elif preferred_time and preferred_time not in VALID_TIMES:
        raise HTTPException(status_code=422, detail="Invalid preferred time")

    return preferred_date, preferred_time


def _read_generation_attachment(stored_key: str) -> Optional[tuple[bytes, str]]:
    raw = storage.read_generation_bytes(stored_key)
    if not raw:
        return None
    name = stored_key.rsplit("/", 1)[-1] if "/" in stored_key else "generation.png"
    return raw, name


async def _resolve_generations(
    db: AsyncSession,
    user_id: int,
    generation_ids: list[int],
) -> list[dict]:
    if not generation_ids:
        return []
    result = await db.execute(
        select(Generation).where(
            Generation.user_id == user_id,
            Generation.id.in_(generation_ids),
        )
    )
    gens = result.scalars().all()
    found_ids = {g.id for g in gens}
    missing = set(generation_ids) - found_ids
    if missing:
        raise HTTPException(status_code=404, detail=f"Generation(s) not found: {sorted(missing)}")

    rows = []
    for g in gens:
        rows.append({
            "id": g.id,
            "name": g.generation_name,
            "stored_key": g.url,
        })
    return rows


async def _resolve_bookmarks(
    db: AsyncSession,
    user_id: int,
    image_ids: list[int],
) -> list[dict]:
    if not image_ids:
        return []
    result = await db.execute(
        select(Bookmark, Image)
        .join(Image, Bookmark.image_id == Image.id)
        .where(
            Bookmark.user_id == user_id,
            Bookmark.image_id.in_(image_ids),
        )
    )
    rows = []
    found_ids: set[int] = set()
    for bookmark, image in result.all():
        found_ids.add(image.id)
        rows.append({
            "bookmark_id": bookmark.id,
            "image_id": image.id,
            "name": image.name or image.filename,
        })
    missing = set(image_ids) - found_ids
    if missing:
        raise HTTPException(status_code=404, detail=f"Bookmark image(s) not found: {sorted(missing)}")
    return rows


async def _collect_upload_attachments(
    uploads: list[UploadFile],
    prefix: str,
) -> list[tuple[str, bytes, str]]:
    attachments: list[tuple[str, bytes, str]] = []
    for idx, upload in enumerate(uploads):
        if not upload.filename:
            continue
        data = await upload.read()
        if not data:
            continue
        ctype = upload.content_type or "application/octet-stream"
        safe_name = upload.filename.replace("/", "_").replace("\\", "_")
        attachments.append((f"{prefix}_{idx + 1}_{safe_name}", data, ctype))
    return attachments


def _format_methods(methods: list[str]) -> str:
    labels = {"email": "Email", "whatsapp": "WhatsApp", "phone": "Phone call"}
    return ", ".join(labels.get(m, m) for m in methods)


def _user_block(user: User) -> str:
    phone = user.phone_number or "—"
    return (
        f"Name: {user.name}\n"
        f"Email: {user.email}\n"
        f"Phone: {phone}\n"
        f"User ID: {user.id}\n"
    )


@router.post("/designer")
async def submit_designer_contact(
    contact_methods: Annotated[str, Form()],
    preferred_date: Annotated[Optional[str], Form()] = None,
    preferred_time: Annotated[Optional[str], Form()] = None,
    notes: Annotated[Optional[str], Form()] = None,
    generation_ids: Annotated[str, Form()] = "[]",
    bookmark_image_ids: Annotated[str, Form()] = "[]",
    space_images: Annotated[list[UploadFile], File()] = [],
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not DESIGNER_CONTACT_EMAIL:
        raise HTTPException(status_code=503, detail="Designer contact email is not configured")

    methods = _parse_methods(contact_methods)
    preferred_date, preferred_time = _validate_schedule(methods, preferred_date, preferred_time)
    if not preferred_date:
        raise HTTPException(status_code=422, detail="Preferred day is required")

    gen_ids = _parse_id_list(generation_ids, "generation_ids")
    bookmark_ids = _parse_id_list(bookmark_image_ids, "bookmark_image_ids")

    generations = await _resolve_generations(db, current_user.id, gen_ids)
    bookmarks = await _resolve_bookmarks(db, current_user.id, bookmark_ids)

    file_attachments = await _collect_upload_attachments(space_images, "space")

    email_attachments: list[dict] = []
    for filename, data, _ctype in file_attachments:
        email_attachments.append({
            "filename": filename,
            "content": base64.b64encode(data).decode("ascii"),
        })

    gen_lines = []
    for g in generations:
        gen_lines.append(f"- Generation #{g['id']}: {g['name']}")
        payload = _read_generation_attachment(g["stored_key"])
        if payload:
            raw, fname = payload
            email_attachments.append({
                "filename": f"generation_{g['id']}_{fname}",
                "content": base64.b64encode(raw).decode("ascii"),
            })

    bookmark_lines = [
        f"- Catalog image #{b['image_id']}: {b['name']}"
        for b in bookmarks
    ]

    schedule = f"Preferred day: {preferred_date}"
    if preferred_time:
        schedule += f" ({preferred_time.capitalize()})"

    text_body = (
        "New designer consultation request\n\n"
        f"{_user_block(current_user)}"
        f"Contact via: {_format_methods(methods)}\n"
        f"{schedule}\n\n"
        f"Notes:\n{notes.strip() if notes else '—'}\n\n"
        f"Selected AI generations ({len(generations)}):\n"
        f"{chr(10).join(gen_lines) if gen_lines else '—'}\n\n"
        f"Selected bookmarked designs ({len(bookmarks)}):\n"
        f"{chr(10).join(bookmark_lines) if bookmark_lines else '—'}\n\n"
        f"Uploaded space images: {len(file_attachments)}\n"
    )

    html_body = f"""\
<p><strong>New designer consultation request</strong></p>
<pre style="font-family: monospace; white-space: pre-wrap;">{_user_block(current_user)}</pre>
<p><strong>Contact via:</strong> {_format_methods(methods)}</p>
<p><strong>{schedule}</strong></p>
<p><strong>Notes:</strong><br>{notes.strip() if notes else '—'}</p>
<h4>Selected AI generations ({len(generations)})</h4>
<ul>{''.join(f'<li>{line[2:]}</li>' for line in gen_lines) or '<li>—</li>'}</ul>
<h4>Selected bookmarked designs ({len(bookmarks)})</h4>
<ul>{''.join(f'<li>{line[2:]}</li>' for line in bookmark_lines) or '<li>—</li>'}</ul>
<p><strong>Uploaded space images:</strong> {len(file_attachments)}</p>
"""

    send_contact_email(
        to=DESIGNER_CONTACT_EMAIL,
        subject=f"Designer consultation — {current_user.name}",
        text=text_body,
        html=html_body,
        attachments=email_attachments or None,
    )

    return {"ok": True, "message": "Designer consultation request sent"}


@router.post("/customer-service")
async def submit_customer_service_contact(
    message: Annotated[str, Form()],
    contact_methods: Annotated[str, Form()],
    preferred_date: Annotated[Optional[str], Form()] = None,
    preferred_time: Annotated[Optional[str], Form()] = None,
    current_user: User = Depends(get_current_user),
):
    if not CUSTOMER_SERVICE_EMAIL:
        raise HTTPException(status_code=503, detail="Customer service email is not configured")

    body = message.strip()
    if not body:
        raise HTTPException(status_code=422, detail="Message is required")

    methods = _parse_methods(contact_methods)
    preferred_date, preferred_time = _validate_schedule(methods, preferred_date, preferred_time)
    if not preferred_date:
        raise HTTPException(status_code=422, detail="Preferred day is required")

    schedule = f"Preferred day: {preferred_date}"
    if preferred_time:
        schedule += f" ({preferred_time.capitalize()})"

    text_body = (
        "New customer service request\n\n"
        f"{_user_block(current_user)}"
        f"Contact via: {_format_methods(methods)}\n"
        f"{schedule}\n\n"
        f"Issue / message:\n{body}\n"
    )

    html_body = f"""\
<p><strong>New customer service request</strong></p>
<pre style="font-family: monospace; white-space: pre-wrap;">{_user_block(current_user)}</pre>
<p><strong>Contact via:</strong> {_format_methods(methods)}</p>
<p><strong>{schedule}</strong></p>
<p><strong>Issue / message:</strong><br>{body.replace(chr(10), '<br>')}</p>
"""

    send_contact_email(
        to=CUSTOMER_SERVICE_EMAIL,
        subject=f"Customer service — {current_user.name}",
        text=text_body,
        html=html_body,
    )

    return {"ok": True, "message": "Customer service request sent"}
