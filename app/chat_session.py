"""
Per-user chat session store — Gemini Interactions API interaction_id + compact
transcript for voice↔chat continuity.

Backed by Postgres (``chat_sessions``) so state is shared across gunicorn
workers. Public async APIs preserve the former in-memory contract.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.database import async_session_maker
from app.db.models import ChatSessionRow, GuestSession

_MAX_TRANSCRIPT_TURNS = 40


def _trim_transcript(turns: list[dict]) -> list[dict]:
    if len(turns) <= _MAX_TRANSCRIPT_TURNS:
        return turns
    return turns[-_MAX_TRANSCRIPT_TURNS:]


def _to_live_contents(raw: list[dict]) -> list[dict]:
    seeded: list[dict] = []
    for turn in raw or []:
        text = (turn.get("text") or "").strip()
        role = turn.get("role")
        if not text or role not in ("user", "model"):
            continue
        seeded.append({"role": role, "parts": [{"text": text}]})
    return seeded[-_MAX_TRANSCRIPT_TURNS:]


async def _ensure_guest_session_exists(db, guest_session_id: str) -> None:
    """Ensure ``guest_sessions`` has a row before ``chat_sessions`` FK insert.

    Chat routers create guest rows on the request session (often flush-only until
    end-of-stream). ``chat_session`` opens a separate connection, so an
    uncommitted parent is invisible and Postgres raises
    ``chat_sessions_guest_session_id_fkey``. Upsert here so mid-stream saves
    never hard-fail; concurrent creators are handled via IntegrityError.
    """
    existing = (
        await db.execute(
            select(GuestSession.id).where(GuestSession.session_id == guest_session_id)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return

    db.add(
        GuestSession(
            session_id=guest_session_id,
            generation_count=0,
            chat_turn_count=0,
            voice_turn_count=0,
        )
    )
    try:
        await db.flush()
    except IntegrityError:
        # Concurrent first-touch (e.g. request db commit vs this ensure).
        await db.rollback()
        existing = (
            await db.execute(
                select(GuestSession.id).where(
                    GuestSession.session_id == guest_session_id
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            raise


async def _get_or_create(
    db,
    *,
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
) -> ChatSessionRow:
    now = datetime.now(timezone.utc)
    if user_id is None and not guest_session_id:
        raise ValueError("user_id or guest_session_id required")

    if user_id is None and guest_session_id:
        await _ensure_guest_session_exists(db, guest_session_id)

    where = (
        ChatSessionRow.user_id == int(user_id)
        if user_id is not None
        else ChatSessionRow.guest_session_id == guest_session_id
    )
    row = (await db.execute(select(ChatSessionRow).where(where))).scalar_one_or_none()
    if row is not None:
        return row

    # First turn for this identity. Two concurrent first turns (e.g. chat +
    # voice) both reach here; the unique partial index makes the second INSERT
    # raise — recover by re-selecting the winner instead of 500ing the turn.
    row = ChatSessionRow(
        user_id=int(user_id) if user_id is not None else None,
        guest_session_id=guest_session_id if user_id is None else None,
        transcript=[],
        updated_at=now,
    )
    db.add(row)
    try:
        await db.flush()
        return row
    except IntegrityError:
        await db.rollback()
        # Rollback drops an ensure we just flushed; re-ensure then re-select.
        if user_id is None and guest_session_id:
            await _ensure_guest_session_exists(db, guest_session_id)
        row = (await db.execute(select(ChatSessionRow).where(where))).scalar_one_or_none()
        if row is None:
            raise
        return row


async def get_interaction_id(user_id: int) -> Optional[str]:
    async with async_session_maker() as db:
        row = (
            await db.execute(select(ChatSessionRow).where(ChatSessionRow.user_id == int(user_id)))
        ).scalar_one_or_none()
        return row.interaction_id if row else None


async def get_guest_interaction_id(guest_session_id: str) -> Optional[str]:
    async with async_session_maker() as db:
        row = (
            await db.execute(
                select(ChatSessionRow).where(ChatSessionRow.guest_session_id == guest_session_id)
            )
        ).scalar_one_or_none()
        return row.interaction_id if row else None


async def set_interaction_id(user_id: int, interaction_id: Optional[str]) -> None:
    async with async_session_maker() as db:
        row = await _get_or_create(db, user_id=user_id)
        row.interaction_id = interaction_id or None
        row.updated_at = datetime.now(timezone.utc)
        await db.commit()


async def set_guest_interaction_id(guest_session_id: str, interaction_id: Optional[str]) -> None:
    async with async_session_maker() as db:
        row = await _get_or_create(db, guest_session_id=guest_session_id)
        row.interaction_id = interaction_id or None
        row.updated_at = datetime.now(timezone.utc)
        await db.commit()


async def append_turn(user_id: int, user_text: str, model_text: str) -> None:
    user_clean = (user_text or "").strip()
    model_clean = (model_text or "").strip()
    if not user_clean and not model_clean:
        return
    async with async_session_maker() as db:
        row = await _get_or_create(db, user_id=user_id)
        turns = list(row.transcript or [])
        if user_clean:
            turns.append({"role": "user", "text": user_clean})
        if model_clean:
            turns.append({"role": "model", "text": model_clean})
        row.transcript = _trim_transcript(turns)
        row.updated_at = datetime.now(timezone.utc)
        await db.commit()


async def append_guest_turn(guest_session_id: str, user_text: str, model_text: str) -> None:
    user_clean = (user_text or "").strip()
    model_clean = (model_text or "").strip()
    if not user_clean and not model_clean:
        return
    async with async_session_maker() as db:
        row = await _get_or_create(db, guest_session_id=guest_session_id)
        turns = list(row.transcript or [])
        if user_clean:
            turns.append({"role": "user", "text": user_clean})
        if model_clean:
            turns.append({"role": "model", "text": model_clean})
        row.transcript = _trim_transcript(turns)
        row.updated_at = datetime.now(timezone.utc)
        await db.commit()


async def get_voice_seed_turns(user_id: int) -> list[dict]:
    async with async_session_maker() as db:
        row = (
            await db.execute(select(ChatSessionRow).where(ChatSessionRow.user_id == int(user_id)))
        ).scalar_one_or_none()
        return _to_live_contents(list(row.transcript or []) if row else [])


async def get_guest_voice_seed_turns(guest_session_id: str) -> list[dict]:
    async with async_session_maker() as db:
        row = (
            await db.execute(
                select(ChatSessionRow).where(ChatSessionRow.guest_session_id == guest_session_id)
            )
        ).scalar_one_or_none()
        return _to_live_contents(list(row.transcript or []) if row else [])


async def clear_session(user_id: int) -> None:
    async with async_session_maker() as db:
        row = (
            await db.execute(select(ChatSessionRow).where(ChatSessionRow.user_id == int(user_id)))
        ).scalar_one_or_none()
        if row:
            await db.delete(row)
            await db.commit()


async def clear_guest_session(guest_session_id: str) -> None:
    async with async_session_maker() as db:
        row = (
            await db.execute(
                select(ChatSessionRow).where(ChatSessionRow.guest_session_id == guest_session_id)
            )
        ).scalar_one_or_none()
        if row:
            await db.delete(row)
            await db.commit()
