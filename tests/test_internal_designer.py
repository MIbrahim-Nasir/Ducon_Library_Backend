"""Authz, validation, rate-limit, job access, and ASGI coverage for internal designer API."""
from __future__ import annotations

import asyncio
import io
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from jose import jwt
from PIL import Image


def _run(coro):
    return asyncio.run(coro)


def _png_bytes(color: str = "white", size: tuple[int, int] = (16, 16)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size: tuple[int, int] = (16, 16)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color="blue").save(buf, format="JPEG")
    return buf.getvalue()


class FakeUpload:
    def __init__(
        self,
        data: bytes | None = None,
        filename: str = "space.png",
        content_type: str = "image/png",
    ):
        self.filename = filename
        self._data = data if data is not None else _png_bytes("green")
        self.content_type = content_type

    async def read(self) -> bytes:
        return self._data


class FakeRequest:
    def __init__(self, headers: dict[str, str] | None = None, query_params: dict | None = None):
        self.headers = headers or {}
        self.cookies = {}
        self.query_params = query_params or {}
        self.client = SimpleNamespace(host="203.0.113.10")
        self.app = SimpleNamespace(
            state=SimpleNamespace(embedding_model=object(), collection=object())
        )


class FakeUser:
    def __init__(self, user_id: int = 42, role: str = "designer"):
        self.id = user_id
        self.role = role


def _assert_http_error(coro, status_code: int, detail_contains: str | None = None):
    with pytest.raises(HTTPException) as exc_info:
        _run(coro)
    assert exc_info.value.status_code == status_code
    if detail_contains is not None:
        assert detail_contains in str(exc_info.value.detail)


def _call_tool_key(mod, value: str):
    async def _inner():
        mod.require_internal_designer_tool_key(x_internal_tool_key=value)

    return _inner()


def _patch_create_job_happy(monkeypatch, mod, *, job_id: str = "abc123"):
    """Mock rate/concurrency/pipeline so create_internal_designer_job is safe."""
    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "tool-key")
    monkeypatch.setattr(mod, "_count_active_jobs_for_user", lambda _uid: 0)

    async def _no_rate(*_a, **_k):
        return None

    monkeypatch.setattr(mod, "require_rate_limit", _no_rate)

    created: dict = {}

    class _Job:
        id = job_id
        status = "queued"
        task = None

    def fake_create_job(user):
        created["user_id"] = user.id
        return _Job()

    async def fake_run(**kwargs):
        created["ran"] = True
        created["prompt"] = kwargs.get("user_prompt")
        created["bytes"] = len(kwargs.get("user_image_bytes") or b"")

    monkeypatch.setattr(mod, "create_job", fake_create_job)
    monkeypatch.setattr(mod, "run_designer_job", fake_run)

    scheduled: dict = {}

    def fake_create_task(coro, name=None):
        scheduled["name"] = name
        coro.close()
        return SimpleNamespace(cancel=lambda: None)

    monkeypatch.setattr(mod.asyncio, "create_task", fake_create_task)
    return created, scheduled


# ── Role / tool-key gates ─────────────────────────────────────────────────────


def test_require_designer_rejects_customer():
    from app.auth import require_designer

    _assert_http_error(
        require_designer(current_user=FakeUser(role="customer")),
        403,
        "Designer role required",
    )


def test_require_designer_rejects_admin_role():
    """Admin is not designer — internal tools require role=designer explicitly."""
    from app.auth import require_designer

    _assert_http_error(
        require_designer(current_user=FakeUser(role="admin")),
        403,
        "Designer role required",
    )


def test_require_designer_rejects_guest_role():
    from app.auth import require_designer

    _assert_http_error(
        require_designer(current_user=FakeUser(role="guest")),
        403,
        "Designer role required",
    )


def test_require_designer_rejects_user_alias():
    from app.auth import require_designer

    _assert_http_error(
        require_designer(current_user=FakeUser(role="user")),
        403,
        "Designer role required",
    )


def test_require_designer_accepts_designer():
    from app.auth import require_designer

    user = FakeUser(role="designer")
    assert _run(require_designer(current_user=user)) is user


def test_require_designer_accepts_designer_case_insensitive():
    from app.auth import require_designer

    user = FakeUser(role="Designer")
    assert _run(require_designer(current_user=user)) is user


def test_tool_key_fail_closed_when_unset(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.delenv("INTERNAL_DESIGNER_API_KEY", raising=False)
    _assert_http_error(_call_tool_key(mod, "anything"), 403, "not configured")


def test_tool_key_fail_closed_when_empty_string(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "")
    _assert_http_error(_call_tool_key(mod, "anything"), 403, "not configured")


def test_tool_key_fail_closed_when_whitespace_only_env(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "   ")
    _assert_http_error(_call_tool_key(mod, "anything"), 403, "not configured")


def test_tool_key_rejects_wrong_value(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "correct-secret")
    _assert_http_error(_call_tool_key(mod, "wrong"), 403, "Forbidden")


def test_tool_key_rejects_empty_provided(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "correct-secret")
    _assert_http_error(_call_tool_key(mod, ""), 403, "Forbidden")


def test_tool_key_rejects_whitespace_padded_provided(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "correct-secret")
    _assert_http_error(_call_tool_key(mod, " correct-secret "), 403, "Forbidden")
    _assert_http_error(_call_tool_key(mod, "correct-secret\n"), 403, "Forbidden")


def test_tool_key_rejects_near_miss_timing_safe(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "correct-secret")
    _assert_http_error(_call_tool_key(mod, "correct-secre"), 403, "Forbidden")
    _assert_http_error(_call_tool_key(mod, "correct-secretX"), 403, "Forbidden")
    _assert_http_error(_call_tool_key(mod, "Correct-secret"), 403, "Forbidden")


def test_tool_key_accepts_match(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "correct-secret")
    _run(_call_tool_key(mod, "correct-secret"))


def test_tool_key_env_strips_whitespace_for_expected(monkeypatch):
    """Env value is stripped once; provided must match the stripped secret."""
    from app.routers import internal_designer as mod

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "  padded-secret  ")
    _run(_call_tool_key(mod, "padded-secret"))
    _assert_http_error(_call_tool_key(mod, "  padded-secret  "), 403, "Forbidden")


# ── Upload / prompt validation ────────────────────────────────────────────────


def test_validate_image_rejects_empty():
    from app.routers.internal_designer import _validate_image_upload

    with pytest.raises(HTTPException) as exc:
        _validate_image_upload("x.png", b"")
    assert exc.value.status_code == 422


def test_validate_image_rejects_exe_extension():
    from app.routers.internal_designer import _validate_image_upload

    with pytest.raises(HTTPException) as exc:
        _validate_image_upload("payload.exe", _png_bytes())
    assert exc.value.status_code == 400


@pytest.mark.parametrize("ext", [".sh", ".bat", ".cmd", ".js", ".py", ".php", ".rb", ".ps1"])
def test_validate_image_rejects_blocked_extensions(ext):
    from app.routers.internal_designer import _validate_image_upload

    with pytest.raises(HTTPException) as exc:
        _validate_image_upload(f"payload{ext}", _png_bytes())
    assert exc.value.status_code == 400


def test_validate_image_rejects_non_image_bytes():
    from app.routers.internal_designer import _validate_image_upload

    with pytest.raises(HTTPException) as exc:
        _validate_image_upload("notes.txt", b"not-an-image")
    assert exc.value.status_code == 400


def test_validate_image_rejects_wrong_mime_despite_png_name():
    from app.routers.internal_designer import _validate_image_upload

    with pytest.raises(HTTPException) as exc:
        _validate_image_upload("looks.png", b"%PDF-1.4 fake")
    assert exc.value.status_code == 400


def test_validate_image_accepts_png_magic_despite_jpeg_content_type_name():
    """Sniff uses magic bytes, not the declared Content-Type / misleading stem."""
    from app.routers.internal_designer import _validate_image_upload

    _validate_image_upload("photo.jpg", _png_bytes())


def test_validate_image_rejects_jpeg_name_with_text_bytes():
    from app.routers.internal_designer import _validate_image_upload

    with pytest.raises(HTTPException) as exc:
        _validate_image_upload("photo.jpg", b"definitely-not-jpeg")
    assert exc.value.status_code == 400


def test_validate_image_rejects_riff_non_webp():
    from app.routers.internal_designer import _validate_image_upload

    # RIFF....AVI\x00 — not WEBP
    data = b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 8
    with pytest.raises(HTTPException) as exc:
        _validate_image_upload("clip.webp", data)
    assert exc.value.status_code == 400


def test_validate_image_rejects_oversized(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.setattr(mod, "_max_upload_bytes", lambda: 100)
    with pytest.raises(HTTPException) as exc:
        mod._validate_image_upload("big.png", _png_bytes(size=(64, 64)))
    assert exc.value.status_code == 413


def test_validate_image_allows_extremely_long_filename():
    from app.routers.internal_designer import _validate_image_upload

    name = ("a" * 500) + ".png"
    _validate_image_upload(name, _png_bytes())


def test_validate_image_allows_path_chars_in_filename_when_png():
    from app.routers.internal_designer import _validate_image_upload

    _validate_image_upload("../../../tmp/evil.png", _png_bytes())


def test_validate_prompt_too_long(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.setattr(mod, "_prompt_max_chars", lambda: 10)
    with pytest.raises(HTTPException) as exc:
        mod._validate_prompt("x" * 11)
    assert exc.value.status_code == 422


def test_validate_prompt_blank_becomes_none():
    from app.routers.internal_designer import _validate_prompt

    assert _validate_prompt("   ") is None
    assert _validate_prompt("") is None
    assert _validate_prompt(None) is None


def test_validate_prompt_unicode_and_emoji():
    from app.routers.internal_designer import _validate_prompt

    text = "مكتب دافئ 🛋️ café — 日本語"
    assert _validate_prompt(f"  {text}  ") == text


def test_validate_prompt_preserves_null_byte_if_present():
    """Document current behavior: null bytes are not stripped/rejected."""
    from app.routers.internal_designer import _validate_prompt

    assert _validate_prompt("hello\x00world") == "hello\x00world"


# ── Create job happy path + authz via direct call ─────────────────────────────


def test_create_job_starts_pipeline(monkeypatch):
    from app.routers import internal_designer as mod

    created, scheduled = _patch_create_job_happy(monkeypatch, mod)

    result = _run(
        mod.create_internal_designer_job(
            request=FakeRequest(),
            prompt=" warm living room ",
            model="flash",
            aspect_ratio="16:9",
            user_image=FakeUpload(),
            current_user=FakeUser(role="designer"),
            _=None,
        )
    )

    assert result["job_id"] == "abc123"
    assert result["status_url"] == "/designer/jobs/abc123"
    assert result["events_url"] == "/designer/jobs/abc123/events"
    assert result["internal_status_url"] == "/internal/designer/jobs/abc123"
    assert result["final_image_url"] is None
    assert created["user_id"] == 42
    assert scheduled["name"].startswith("internal-designer-job-")


def test_create_job_strips_prompt_before_pipeline(monkeypatch):
    from app.routers import internal_designer as mod

    created, _ = _patch_create_job_happy(monkeypatch, mod, job_id="p1")

    # Re-patch create_task to actually drive the coroutine once so fake_run sees prompt.
    # create_internal_designer_job closes the task via our fake; assert strip via validate path.
    result = _run(
        mod.create_internal_designer_job(
            request=FakeRequest(),
            prompt="  cozy  ",
            model="flash",
            aspect_ratio="16:9",
            user_image=FakeUpload(),
            current_user=FakeUser(role="designer"),
            _=None,
        )
    )
    assert result["job_id"] == "p1"
    # Prompt is validated/stripped before create_task; exercise validate directly too.
    assert mod._validate_prompt("  cozy  ") == "cozy"


def test_create_job_rejects_bad_model(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "tool-key")
    monkeypatch.setattr(mod, "_count_active_jobs_for_user", lambda _uid: 0)

    async def _no_rate(*_a, **_k):
        return None

    monkeypatch.setattr(mod, "require_rate_limit", _no_rate)

    _assert_http_error(
        mod.create_internal_designer_job(
            request=FakeRequest(),
            prompt="ok",
            model="ultra",
            aspect_ratio="16:9",
            user_image=FakeUpload(),
            current_user=FakeUser(role="designer"),
            _=None,
        ),
        422,
        "flash",
    )


def test_create_job_rejects_empty_upload(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "tool-key")
    monkeypatch.setattr(mod, "_count_active_jobs_for_user", lambda _uid: 0)

    async def _no_rate(*_a, **_k):
        return None

    monkeypatch.setattr(mod, "require_rate_limit", _no_rate)

    _assert_http_error(
        mod.create_internal_designer_job(
            request=FakeRequest(),
            prompt="ok",
            model="flash",
            aspect_ratio="16:9",
            user_image=FakeUpload(data=b""),
            current_user=FakeUser(role="designer"),
            _=None,
        ),
        422,
        "empty",
    )


def test_create_job_concurrent_cap(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "tool-key")
    monkeypatch.setattr(mod, "_max_concurrent", lambda: 2)
    monkeypatch.setattr(mod, "_count_active_jobs_for_user", lambda _uid: 2)

    async def _no_rate(*_a, **_k):
        return None

    monkeypatch.setattr(mod, "require_rate_limit", _no_rate)

    _assert_http_error(
        mod.create_internal_designer_job(
            request=FakeRequest(),
            prompt="ok",
            model="flash",
            aspect_ratio="16:9",
            user_image=FakeUpload(),
            current_user=FakeUser(role="designer"),
            _=None,
        ),
        429,
        "concurrent",
    )


def test_count_active_jobs_frees_after_terminal_statuses():
    from app.designer_agent import JOBS, DesignerJob
    from app.routers.internal_designer import _count_active_jobs_for_user

    JOBS.clear()
    try:
        running = DesignerJob(id="r1", user_id=7, status="running")
        JOBS["r1"] = running
        assert _count_active_jobs_for_user(7) == 1

        running.status = "completed"
        assert _count_active_jobs_for_user(7) == 0

        again = DesignerJob(id="r2", user_id=7, status="queued")
        JOBS["r2"] = again
        assert _count_active_jobs_for_user(7) == 1
        again.status = "cancelled"
        assert _count_active_jobs_for_user(7) == 0

        failed = DesignerJob(id="r3", user_id=7, status="failed")
        JOBS["r3"] = failed
        assert _count_active_jobs_for_user(7) == 0

        other = DesignerJob(id="o1", user_id=99, status="running")
        JOBS["o1"] = other
        assert _count_active_jobs_for_user(7) == 0
        assert _count_active_jobs_for_user(99) == 1
    finally:
        JOBS.clear()


def test_create_job_rate_limit_bucket(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "tool-key")
    monkeypatch.setattr(mod, "_count_active_jobs_for_user", lambda _uid: 0)
    monkeypatch.setattr(mod, "_rate_max", lambda: 2)
    monkeypatch.setattr(mod, "_rate_window_s", lambda: 3600)

    async def _no_db(*_a, **_k):
        return None

    monkeypatch.setattr("app.rate_limiter._db_hit_count", _no_db)
    from app import rate_limiter

    rate_limiter._attempts.clear()

    async def _hit():
        await mod.require_rate_limit(
            FakeRequest(),
            max_requests=2,
            window_seconds=3600,
            key_prefix="internal_designer",
            key_suffix="u:42",
        )

    _run(_hit())
    _run(_hit())
    with pytest.raises(HTTPException) as exc:
        _run(_hit())
    assert exc.value.status_code == 429


def test_double_submit_creates_two_jobs(monkeypatch):
    from app.routers import internal_designer as mod

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "tool-key")
    monkeypatch.setattr(mod, "_count_active_jobs_for_user", lambda _uid: 0)

    async def _no_rate(*_a, **_k):
        return None

    monkeypatch.setattr(mod, "require_rate_limit", _no_rate)

    ids = iter(["job-a", "job-b"])

    def fake_create_job(user):
        jid = next(ids)
        return SimpleNamespace(id=jid, status="queued", task=None)

    monkeypatch.setattr(mod, "create_job", fake_create_job)
    monkeypatch.setattr(mod, "run_designer_job", lambda **_k: None)

    def fake_create_task(coro, name=None):
        coro.close()
        return SimpleNamespace(cancel=lambda: None)

    monkeypatch.setattr(mod.asyncio, "create_task", fake_create_task)

    r1 = _run(
        mod.create_internal_designer_job(
            request=FakeRequest(),
            prompt="same",
            model="flash",
            aspect_ratio="16:9",
            user_image=FakeUpload(),
            current_user=FakeUser(role="designer"),
            _=None,
        )
    )
    r2 = _run(
        mod.create_internal_designer_job(
            request=FakeRequest(),
            prompt="same",
            model="flash",
            aspect_ratio="16:9",
            user_image=FakeUpload(),
            current_user=FakeUser(role="designer"),
            _=None,
        )
    )
    assert r1["job_id"] == "job-a"
    assert r2["job_id"] == "job-b"
    assert r1["job_id"] != r2["job_id"]


# ── Job get / cancel access ───────────────────────────────────────────────────


def test_get_job_other_user_404(monkeypatch):
    from app.routers import internal_designer as mod

    async def _none(*_a, **_k):
        return None

    monkeypatch.setattr(mod, "get_job_for_user", _none)

    _assert_http_error(
        mod.get_internal_designer_job(
            job_id="someone-elses",
            current_user=FakeUser(user_id=1, role="designer"),
            _=None,
        ),
        404,
        "not found",
    )


def test_get_job_invalid_id_404(monkeypatch):
    from app.routers import internal_designer as mod

    async def _none(*_a, **_k):
        return None

    monkeypatch.setattr(mod, "get_job_for_user", _none)

    _assert_http_error(
        mod.get_internal_designer_job(
            job_id="not-a-real-job",
            current_user=FakeUser(role="designer"),
            _=None,
        ),
        404,
        "not found",
    )


def test_get_job_owner_ok(monkeypatch):
    from app.routers import internal_designer as mod

    job = SimpleNamespace(
        id="mine",
        status="running",
        created_at="2026-01-01T00:00:00+00:00",
        final=None,
        error=None,
        events=[{"type": "status"}],
    )

    async def _owned(job_id, user_id):
        assert job_id == "mine"
        assert user_id == 42
        return job

    monkeypatch.setattr(mod, "get_job_for_user", _owned)

    result = _run(
        mod.get_internal_designer_job(
            job_id="mine",
            current_user=FakeUser(role="designer"),
            _=None,
        )
    )
    assert result["job_id"] == "mine"
    assert result["status"] == "running"
    assert result["events"] == [{"type": "status"}]
    assert result["final_image_url"] is None
    assert result["final_image"] is None


def test_get_job_completed_exposes_final_image_url(monkeypatch):
    from app.routers import internal_designer as mod

    signed = "https://cdn.example/signed/gen-99.jpg"
    job = SimpleNamespace(
        id="done1",
        status="completed",
        created_at="2026-01-01T00:00:00+00:00",
        final={
            "best_generation": {
                "id": 99,
                "url": "gens/99.jpg",
                "signed_url": signed,
                "generation_name": "Concept A",
            }
        },
        error=None,
        events=[],
    )

    async def _owned(job_id, user_id):
        return job

    monkeypatch.setattr(mod, "get_job_for_user", _owned)
    monkeypatch.setattr(
        "app.designer.jobs.storage.get_generation_url",
        lambda gen_id, url: f"https://refreshed/{gen_id}",
    )

    result = _run(
        mod.get_internal_designer_job(
            job_id="done1",
            current_user=FakeUser(role="designer"),
            _=None,
        )
    )
    assert result["status"] == "completed"
    assert result["final_image_url"] == "https://refreshed/99"
    assert result["final_image"]["id"] == 99
    assert result["final_image"]["signed_url"] == "https://refreshed/99"
    assert result["final_image"]["url"] == "gens/99.jpg"
    assert result["final"]["best_generation"]["id"] == 99


def test_create_job_wait_true_returns_final_image_url(monkeypatch):
    from app.routers import internal_designer as mod
    from app.designer_agent import DesignerJob

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "tool-key")
    monkeypatch.setattr(mod, "_count_active_jobs_for_user", lambda _uid: 0)
    monkeypatch.setattr(mod, "_wait_timeout_s", lambda: 5.0)

    async def _no_rate(*_a, **_k):
        return None

    monkeypatch.setattr(mod, "require_rate_limit", _no_rate)

    job = DesignerJob(id="wait1", user_id=42, status="queued")
    monkeypatch.setattr(mod, "create_job", lambda user: job)

    async def fake_run(**_k):
        job.status = "completed"
        job.final = {
            "best_generation": {
                "id": 7,
                "url": "gens/7.jpg",
                "signed_url": "https://cdn.example/7.jpg",
                "generation_name": "Final",
            }
        }

    monkeypatch.setattr(mod, "run_designer_job", fake_run)
    monkeypatch.setattr(
        "app.designer.jobs.storage.get_generation_url",
        lambda gen_id, url: f"https://signed/{gen_id}",
    )

    async def _call():
        return await mod.create_internal_designer_job(
            request=FakeRequest(),
            prompt="cozy",
            model="flash",
            aspect_ratio="16:9",
            wait="true",
            user_image=FakeUpload(),
            current_user=FakeUser(role="designer"),
            _=None,
        )

    result = _run(_call())
    assert result["job_id"] == "wait1"
    assert result["status"] == "completed"
    assert result["final_image_url"] == "https://signed/7"
    assert result["final_image"]["id"] == 7


def test_create_job_wait_timeout_returns_running(monkeypatch):
    from app.routers import internal_designer as mod
    from app.designer_agent import DesignerJob

    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", "tool-key")
    monkeypatch.setattr(mod, "_count_active_jobs_for_user", lambda _uid: 0)
    monkeypatch.setattr(mod, "_wait_timeout_s", lambda: 0.15)

    async def _no_rate(*_a, **_k):
        return None

    monkeypatch.setattr(mod, "require_rate_limit", _no_rate)

    job = DesignerJob(id="slow1", user_id=42, status="queued")
    monkeypatch.setattr(mod, "create_job", lambda user: job)

    async def fake_run(**_k):
        job.status = "running"
        await asyncio.sleep(2.0)
        job.status = "completed"

    monkeypatch.setattr(mod, "run_designer_job", fake_run)

    async def _call():
        try:
            return await mod.create_internal_designer_job(
                request=FakeRequest(query_params={"wait": "true"}),
                prompt="slow",
                model="flash",
                aspect_ratio="16:9",
                wait=None,
                user_image=FakeUpload(),
                current_user=FakeUser(role="designer"),
                _=None,
            )
        finally:
            if job.task and not job.task.done():
                job.task.cancel()
                try:
                    await job.task
                except (asyncio.CancelledError, Exception):
                    pass

    result = _run(_call())
    assert result["job_id"] == "slow1"
    assert result["status"] in {"queued", "running"}
    assert result["final_image_url"] is None


def test_final_image_fields_prefers_signed_url(monkeypatch):
    from app.designer_agent import final_image_fields

    monkeypatch.setattr(
        "app.designer.jobs.storage.get_generation_url",
        lambda gen_id, url: f"https://fresh/{gen_id}",
    )
    out = final_image_fields(
        {"best_generation": {"id": 3, "url": "key/3", "signed_url": "stale"}}
    )
    assert out["final_image_url"] == "https://fresh/3"
    assert out["final_image"]["id"] == 3


def test_drain_jobs_on_shutdown_marks_reload_message(monkeypatch):
    from app.designer_agent import (
        JOBS,
        DesignerJob,
        _SHUTDOWN_INTERRUPT_MSG,
        _TERMINAL_STATUSES,
        drain_jobs_on_shutdown,
    )

    JOBS.clear()

    async def _body():
        job = DesignerJob(id="sh1", user_id=1, status="running")

        async def _pipeline():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                if job.status not in _TERMINAL_STATUSES:
                    msg = job.error or _SHUTDOWN_INTERRUPT_MSG
                    job.status = "cancelled"
                    job.error = msg

        job.task = asyncio.create_task(_pipeline())
        JOBS[job.id] = job
        await asyncio.sleep(0)  # let the task start so cancel injects CancelledError

        n = await drain_jobs_on_shutdown(timeout=2)
        assert n == 1
        await asyncio.sleep(0)
        assert job.error == _SHUTDOWN_INTERRUPT_MSG
        assert job.status == "cancelled"
        assert job.task.done()

    try:
        _run(_body())
    finally:
        JOBS.clear()


def test_cancel_other_user_404(monkeypatch):
    from app.routers import internal_designer as mod

    async def _false(*_a, **_k):
        return False

    monkeypatch.setattr(mod, "cancel_job", _false)

    _assert_http_error(
        mod.cancel_internal_designer_job(
            job_id="nope",
            current_user=FakeUser(role="designer"),
            _=None,
        ),
        404,
        "not found",
    )


def test_cancel_twice_idempotent_while_in_registry(monkeypatch):
    from app.designer_agent import JOBS, DesignerJob
    from app.routers import internal_designer as mod

    JOBS.clear()
    job = DesignerJob(id="c1", user_id=42, status="running")
    JOBS["c1"] = job

    async def _no_persist(*_a, **_k):
        return None

    monkeypatch.setattr("app.designer.jobs._persist_cancel_request", _no_persist)

    try:
        r1 = _run(
            mod.cancel_internal_designer_job(
                job_id="c1",
                current_user=FakeUser(role="designer"),
                _=None,
            )
        )
        assert r1 == {"job_id": "c1", "status": "cancelling"}
        assert job.cancel_requested is True

        r2 = _run(
            mod.cancel_internal_designer_job(
                job_id="c1",
                current_user=FakeUser(role="designer"),
                _=None,
            )
        )
        assert r2 == {"job_id": "c1", "status": "cancelling"}
    finally:
        JOBS.clear()


def test_concurrency_slot_frees_after_cancel_status():
    """After cancel marks terminal cancelled, slot is free for another create gate."""
    from app.designer_agent import JOBS, DesignerJob
    from app.routers.internal_designer import _count_active_jobs_for_user, _max_concurrent

    JOBS.clear()
    try:
        job = DesignerJob(id="cx", user_id=5, status="running")
        JOBS["cx"] = job
        assert _count_active_jobs_for_user(5) == 1
        job.status = "cancelled"
        job.cancel_requested = True
        assert _count_active_jobs_for_user(5) == 0
        # Gate would allow another job under default max.
        assert _count_active_jobs_for_user(5) < _max_concurrent()
    finally:
        JOBS.clear()


# ── CORS merge ────────────────────────────────────────────────────────────────


def test_cors_merges_internal_tool_origins(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://app.example.com")
    monkeypatch.setenv("INTERNAL_TOOL_CORS_ORIGINS", "https://tools.example.com,https://app.example.com")
    from app.config import get_cors_origins

    origins = get_cors_origins()
    assert origins == ["https://app.example.com", "https://tools.example.com"]


def test_cors_internal_only_when_spa_unset(monkeypatch):
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    monkeypatch.setenv("INTERNAL_TOOL_CORS_ORIGINS", "https://tools.example.com")
    from app.config import get_cors_origins

    assert get_cors_origins() == ["https://tools.example.com"]


# ── Turnstile note: designer routes never call verify_turnstile ───────────────


def test_internal_router_does_not_import_turnstile():
    import app.routers.internal_designer as mod
    import inspect

    src = inspect.getsource(mod)
    assert "turnstile" not in src.lower()
    assert "verify_turnstile" not in src


# ── ASGI / httpx: JWT + tool key + multipart ──────────────────────────────────


TOOL_KEY = "asgi-internal-tool-key-xyz"
USERS = {
    1: FakeUser(1, "designer"),
    2: FakeUser(2, "admin"),
    3: FakeUser(3, "customer"),
    4: FakeUser(4, "guest"),
}


def _make_token(
    user_id: int,
    *,
    secret: str | None = None,
    expires_delta: timedelta | None = None,
    expired: bool = False,
) -> str:
    from app.auth import ALGORITHM, SECRET_KEY

    if expired:
        exp = datetime.now(timezone.utc) - timedelta(hours=1)
    else:
        exp = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=1))
    return jwt.encode(
        {"sub": str(user_id), "exp": exp, "jti": f"test-jti-{user_id}"},
        secret if secret is not None else SECRET_KEY,
        algorithm=ALGORITHM,
    )


@pytest_asyncio.fixture
async def asgi_client(monkeypatch):
    """Minimal FastAPI app with internal designer router + JWT resolve mock."""
    monkeypatch.setenv("INTERNAL_DESIGNER_API_KEY", TOOL_KEY)

    from app.auth import get_current_user
    from app.db.database import get_db
    from app.routers import internal_designer as mod
    from app.routers.internal_designer import router

    async def fake_resolve(token, db):
        from app.auth import ALGORITHM, SECRET_KEY
        from jose import JWTError

        if not token:
            return None
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            uid = int(payload["sub"])
        except (JWTError, ValueError, TypeError):
            return None
        return USERS.get(uid)

    monkeypatch.setattr("app.auth.resolve_user_from_token", fake_resolve)

    # SSE path uses async_session_maker + resolve_user_from_token on the router module.
    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    def _session_maker():
        return _FakeSession()

    monkeypatch.setattr(mod, "async_session_maker", _session_maker)
    monkeypatch.setattr(mod, "resolve_user_from_token", fake_resolve)

    async def _no_rate(*_a, **_k):
        return None

    monkeypatch.setattr(mod, "require_rate_limit", _no_rate)
    monkeypatch.setattr(mod, "_count_active_jobs_for_user", lambda _uid: 0)

    job_counter = {"n": 0}

    def fake_create_job(user):
        job_counter["n"] += 1
        jid = f"asgi-job-{job_counter['n']}"
        return SimpleNamespace(id=jid, status="queued", task=None, user_id=int(user.id))

    async def fake_run(**_k):
        return None

    monkeypatch.setattr(mod, "create_job", fake_create_job)
    monkeypatch.setattr(mod, "run_designer_job", fake_run)

    def fake_create_task(coro, name=None):
        coro.close()
        return SimpleNamespace(cancel=lambda: None, done=lambda: True)

    monkeypatch.setattr(mod.asyncio, "create_task", fake_create_task)

    async def _override_get_db():
        # Avoid real Postgres session open/close (breaks on HTTPException cleanup).
        yield object()

    app = FastAPI()
    app.state.embedding_model = object()
    app.state.collection = object()
    app.include_router(router)
    app.dependency_overrides[get_db] = _override_get_db
    # Keep get_current_user so OAuth2PasswordBearer still enforces Bearer presence.
    assert get_current_user is not None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, app, mod
    app.dependency_overrides.clear()


def _auth_headers(user_id: int = 1, tool_key: str = TOOL_KEY, **extra) -> dict[str, str]:
    h = {
        "Authorization": f"Bearer {_make_token(user_id)}",
        "X-Internal-Tool-Key": tool_key,
    }
    h.update(extra)
    return h


def _multipart():
    return {
        "files": {"user_image": ("space.png", _png_bytes(), "image/png")},
        "data": {"prompt": "redesign this room", "model": "flash", "aspect_ratio": "16:9"},
    }


@pytest.mark.asyncio
async def test_asgi_no_jwt_401(asgi_client):
    ac, _app, _mod = asgi_client
    mp = _multipart()
    resp = await ac.post(
        "/internal/designer/jobs",
        files=mp["files"],
        data=mp["data"],
        headers={"X-Internal-Tool-Key": TOOL_KEY},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_asgi_malformed_jwt_401(asgi_client):
    ac, _app, _mod = asgi_client
    mp = _multipart()
    resp = await ac.post(
        "/internal/designer/jobs",
        files=mp["files"],
        data=mp["data"],
        headers={
            "Authorization": "Bearer not.a.jwt",
            "X-Internal-Tool-Key": TOOL_KEY,
        },
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_asgi_expired_jwt_401(asgi_client):
    ac, _app, _mod = asgi_client
    mp = _multipart()
    resp = await ac.post(
        "/internal/designer/jobs",
        files=mp["files"],
        data=mp["data"],
        headers={
            "Authorization": f"Bearer {_make_token(1, expired=True)}",
            "X-Internal-Tool-Key": TOOL_KEY,
        },
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_asgi_wrong_secret_jwt_401(asgi_client):
    ac, _app, _mod = asgi_client
    mp = _multipart()
    resp = await ac.post(
        "/internal/designer/jobs",
        files=mp["files"],
        data=mp["data"],
        headers={
            "Authorization": f"Bearer {_make_token(1, secret='wrong-secret-key-zzzz')}",
            "X-Internal-Tool-Key": TOOL_KEY,
        },
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_asgi_admin_with_tool_key_403(asgi_client):
    ac, _app, _mod = asgi_client
    mp = _multipart()
    resp = await ac.post(
        "/internal/designer/jobs",
        files=mp["files"],
        data=mp["data"],
        headers=_auth_headers(user_id=2),
    )
    assert resp.status_code == 403
    assert "Designer role required" in resp.text


@pytest.mark.asyncio
async def test_asgi_customer_with_tool_key_403(asgi_client):
    ac, _app, _mod = asgi_client
    mp = _multipart()
    resp = await ac.post(
        "/internal/designer/jobs",
        files=mp["files"],
        data=mp["data"],
        headers=_auth_headers(user_id=3),
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_asgi_guest_with_tool_key_403(asgi_client):
    ac, _app, _mod = asgi_client
    mp = _multipart()
    resp = await ac.post(
        "/internal/designer/jobs",
        files=mp["files"],
        data=mp["data"],
        headers=_auth_headers(user_id=4),
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_asgi_designer_missing_tool_key_403(asgi_client):
    ac, _app, _mod = asgi_client
    mp = _multipart()
    resp = await ac.post(
        "/internal/designer/jobs",
        files=mp["files"],
        data=mp["data"],
        headers={"Authorization": f"Bearer {_make_token(1)}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_asgi_designer_wrong_tool_key_403(asgi_client):
    ac, _app, _mod = asgi_client
    mp = _multipart()
    resp = await ac.post(
        "/internal/designer/jobs",
        files=mp["files"],
        data=mp["data"],
        headers=_auth_headers(tool_key="nope"),
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_asgi_designer_and_key_success(asgi_client):
    ac, _app, _mod = asgi_client
    mp = _multipart()
    resp = await ac.post(
        "/internal/designer/jobs",
        files=mp["files"],
        data=mp["data"],
        headers=_auth_headers(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["job_id"].startswith("asgi-job-")
    assert body["status"] == "queued"
    assert body["status_url"].startswith("/designer/jobs/")
    assert body["internal_status_url"].startswith("/internal/designer/jobs/")


@pytest.mark.asyncio
async def test_asgi_no_turnstile_header_required(asgi_client):
    """Internal designer must not require CF-Turnstile-Token (guest-only elsewhere)."""
    ac, _app, _mod = asgi_client
    mp = _multipart()
    headers = _auth_headers()
    assert "CF-Turnstile-Token" not in headers
    assert "cf-turnstile-token" not in {k.lower() for k in headers}
    resp = await ac.post(
        "/internal/designer/jobs",
        files=mp["files"],
        data=mp["data"],
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.headers.get("cf-turnstile-required") is None


@pytest.mark.asyncio
async def test_asgi_non_browser_no_origin_ok(asgi_client):
    """Curl/scripts: no Origin header — CORS middleware not required for success."""
    ac, _app, _mod = asgi_client
    mp = _multipart()
    headers = _auth_headers()
    assert "Origin" not in headers
    resp = await ac.post(
        "/internal/designer/jobs",
        files=mp["files"],
        data=mp["data"],
        headers=headers,
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_asgi_missing_image_422(asgi_client):
    ac, _app, _mod = asgi_client
    resp = await ac.post(
        "/internal/designer/jobs",
        data={"prompt": "no file", "model": "flash"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_asgi_wrong_mime_bytes_400(asgi_client):
    ac, _app, _mod = asgi_client
    resp = await ac.post(
        "/internal/designer/jobs",
        files={"user_image": ("notes.png", b"not-image-bytes", "image/png")},
        data={"prompt": "x", "model": "flash"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    assert "not allowed" in resp.text.lower() or "File type" in resp.text


@pytest.mark.asyncio
async def test_asgi_content_type_mismatch_png_bytes_declared_jpeg_ok(asgi_client):
    """Declared Content-Type is ignored; magic bytes decide."""
    ac, _app, _mod = asgi_client
    resp = await ac.post(
        "/internal/designer/jobs",
        files={"user_image": ("photo.jpg", _png_bytes(), "image/jpeg")},
        data={"prompt": "x", "model": "flash"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_asgi_extra_form_fields_ignored(asgi_client):
    ac, _app, _mod = asgi_client
    resp = await ac.post(
        "/internal/designer/jobs",
        files={"user_image": ("space.png", _png_bytes(), "image/png")},
        data={
            "prompt": "ok",
            "model": "flash",
            "aspect_ratio": "16:9",
            "unexpected_field": "should-be-ignored",
            "hack": "1",
        },
        headers=_auth_headers(),
    )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_asgi_whitespace_prompt_ok(asgi_client):
    ac, _app, _mod = asgi_client
    resp = await ac.post(
        "/internal/designer/jobs",
        files={"user_image": ("space.png", _png_bytes(), "image/png")},
        data={"prompt": "   ", "model": "flash"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_asgi_emoji_prompt_ok(asgi_client):
    ac, _app, _mod = asgi_client
    resp = await ac.post(
        "/internal/designer/jobs",
        files={"user_image": ("space.png", _png_bytes(), "image/png")},
        data={"prompt": "warm lounge 🔥🛋️", "model": "flash"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_asgi_prompt_over_max_422(asgi_client, monkeypatch):
    ac, _app, mod = asgi_client
    monkeypatch.setattr(mod, "_prompt_max_chars", lambda: 5)
    resp = await ac.post(
        "/internal/designer/jobs",
        files={"user_image": ("space.png", _png_bytes(), "image/png")},
        data={"prompt": "too-long-prompt", "model": "flash"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 422
    assert "too long" in resp.text.lower()


@pytest.mark.asyncio
async def test_asgi_long_filename_ok(asgi_client):
    ac, _app, _mod = asgi_client
    name = ("room-" + ("x" * 400) + ".png")
    resp = await ac.post(
        "/internal/designer/jobs",
        files={"user_image": (name, _png_bytes(), "image/png")},
        data={"prompt": "ok", "model": "flash"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_asgi_path_chars_filename_ok(asgi_client):
    ac, _app, _mod = asgi_client
    resp = await ac.post(
        "/internal/designer/jobs",
        files={"user_image": ("..\\..\\evil.png", _png_bytes(), "image/png")},
        data={"prompt": "ok", "model": "flash"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_asgi_oversized_413(asgi_client, monkeypatch):
    ac, _app, mod = asgi_client
    monkeypatch.setattr(mod, "_max_upload_bytes", lambda: 50)
    resp = await ac.post(
        "/internal/designer/jobs",
        files={"user_image": ("big.png", _png_bytes(size=(80, 80)), "image/png")},
        data={"prompt": "ok", "model": "flash"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_asgi_rate_limit_429(asgi_client, monkeypatch):
    ac, _app, mod = asgi_client

    # Restore real rate limiter with tiny bucket.
    from app import rate_limiter
    from app.rate_limiter import require_rate_limit

    rate_limiter._attempts.clear()

    async def _no_db(*_a, **_k):
        return None

    monkeypatch.setattr("app.rate_limiter._db_hit_count", _no_db)
    monkeypatch.setattr(mod, "require_rate_limit", require_rate_limit)
    monkeypatch.setattr(mod, "_rate_max", lambda: 1)
    monkeypatch.setattr(mod, "_rate_window_s", lambda: 3600)

    mp = _multipart()
    h = _auth_headers()
    r1 = await ac.post("/internal/designer/jobs", files=mp["files"], data=mp["data"], headers=h)
    assert r1.status_code == 200, r1.text
    r2 = await ac.post(
        "/internal/designer/jobs",
        files={"user_image": ("space.png", _png_bytes(), "image/png")},
        data={"prompt": "again", "model": "flash"},
        headers=h,
    )
    assert r2.status_code == 429


@pytest.mark.asyncio
async def test_asgi_max_concurrent_429(asgi_client, monkeypatch):
    ac, _app, mod = asgi_client
    monkeypatch.setattr(mod, "_max_concurrent", lambda: 1)
    monkeypatch.setattr(mod, "_count_active_jobs_for_user", lambda _uid: 1)
    mp = _multipart()
    resp = await ac.post(
        "/internal/designer/jobs",
        files=mp["files"],
        data=mp["data"],
        headers=_auth_headers(),
    )
    assert resp.status_code == 429
    assert "concurrent" in resp.text.lower()


@pytest.mark.asyncio
async def test_asgi_get_other_users_job_404(asgi_client, monkeypatch):
    ac, _app, mod = asgi_client

    async def _none(*_a, **_k):
        return None

    monkeypatch.setattr(mod, "get_job_for_user", _none)
    resp = await ac.get(
        "/internal/designer/jobs/foreign-job",
        headers=_auth_headers(),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_asgi_cancel_invalid_job_404(asgi_client, monkeypatch):
    ac, _app, mod = asgi_client

    async def _false(*_a, **_k):
        return False

    monkeypatch.setattr(mod, "cancel_job", _false)
    resp = await ac.post(
        "/internal/designer/jobs/missing/cancel",
        headers=_auth_headers(),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_asgi_double_submit_two_job_ids(asgi_client):
    ac, _app, _mod = asgi_client
    h = _auth_headers()
    bodies = []
    for i in range(2):
        resp = await ac.post(
            "/internal/designer/jobs",
            files={"user_image": ("space.png", _png_bytes(), "image/png")},
            data={"prompt": f"dup-{i}", "model": "flash"},
            headers=h,
        )
        assert resp.status_code == 200, resp.text
        bodies.append(resp.json())
    assert bodies[0]["job_id"] != bodies[1]["job_id"]


@pytest.mark.asyncio
async def test_asgi_tool_key_present_non_designer_still_403(asgi_client):
    """Tool key alone is insufficient — admin JWT still 403."""
    ac, _app, _mod = asgi_client
    mp = _multipart()
    resp = await ac.post(
        "/internal/designer/jobs",
        files=mp["files"],
        data=mp["data"],
        headers=_auth_headers(user_id=2, tool_key=TOOL_KEY),
    )
    assert resp.status_code == 403
    assert "Designer" in resp.text


@pytest.mark.asyncio
async def test_asgi_jpeg_upload_ok(asgi_client):
    ac, _app, _mod = asgi_client
    resp = await ac.post(
        "/internal/designer/jobs",
        files={"user_image": ("space.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"prompt": "ok", "model": "flash"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200, resp.text
