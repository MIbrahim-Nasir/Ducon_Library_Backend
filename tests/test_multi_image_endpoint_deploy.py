import asyncio
import io
import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image


def _png_bytes(color: str = "white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (12, 12), color=color).save(buf, format="PNG")
    return buf.getvalue()


class FakeUpload:
    filename = "space.png"

    async def read(self) -> bytes:
        return _png_bytes("green")


class FakeRequest:
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}
        self.cookies = {}
        self.client = SimpleNamespace(host="203.0.113.10")


class FakeDb:
    def __init__(self):
        self.events: list[str] = []

    async def commit(self):
        self.events.append("commit")


class FakeUser:
    id = 42


def _run(coro):
    return asyncio.run(coro)


def _assert_http_error(coro, status_code: int, detail_contains: str):
    with pytest.raises(HTTPException) as exc_info:
        _run(coro)
    assert exc_info.value.status_code == status_code
    assert detail_contains in str(exc_info.value.detail)


def test_guest_request_requires_auth_or_guest_session():
    from app.routers.multi_image_gen import create_multi_image_generation

    _assert_http_error(
        create_multi_image_generation(
            request=FakeRequest(),
            prompt="Generate a design.",
            images_meta=json.dumps([{"label": "space", "source": "file:0"}]),
            image_specs=None,
            model="pro",
            aspect_ratio=None,
            files=[FakeUpload()],
            cf_turnstile_token=None,
            current_user=None,
            db=FakeDb(),
        ),
        401,
        "Authentication required",
    )


def test_guest_request_rejects_invalid_guest_uuid():
    from app.routers.multi_image_gen import create_multi_image_generation

    _assert_http_error(
        create_multi_image_generation(
            request=FakeRequest({"x-guest-session-id": "not-a-uuid"}),
            prompt="Generate a design.",
            images_meta=json.dumps([{"label": "space", "source": "file:0"}]),
            image_specs=None,
            model="pro",
            aspect_ratio=None,
            files=[FakeUpload()],
            cf_turnstile_token=None,
            current_user=None,
            db=FakeDb(),
        ),
        400,
        "valid UUID",
    )


@pytest.mark.parametrize(
    ("kwargs", "detail"),
    [
        ({"images_meta": None}, "images_meta is required"),
        ({"images_meta": "{bad json"}, "images_meta is not valid JSON"),
        ({"images_meta": json.dumps([])}, "non-empty JSON array"),
        (
            {"images_meta": json.dumps([{"label": f"img-{i}", "source": i} for i in range(11)])},
            "Too many images",
        ),
        (
            {"images_meta": json.dumps([{"label": "space", "source": "file:1"}]), "files": [FakeUpload()]},
            "references files[1]",
        ),
        (
            {"images_meta": json.dumps([{"label": "space", "source": "file:abc"}]), "files": [FakeUpload()]},
            "numeric index",
        ),
        (
            {"images_meta": json.dumps([{"label": "", "source": "file:0"}]), "files": [FakeUpload()]},
            "label is required",
        ),
    ],
)
def test_logged_in_validation_examples(kwargs, detail):
    from app.routers.multi_image_gen import create_multi_image_generation

    call_kwargs = {
        "request": FakeRequest(),
        "prompt": "Generate a design.",
        "image_specs": None,
        "model": "pro",
        "aspect_ratio": None,
        "files": [],
        "cf_turnstile_token": None,
        "current_user": FakeUser(),
        "db": FakeDb(),
    }
    call_kwargs.update(kwargs)

    _assert_http_error(create_multi_image_generation(**call_kwargs), 422, detail)


def test_logged_in_request_builds_stream_without_guest_commit(monkeypatch):
    from app.routers import multi_image_gen as module

    captured = {}

    def fake_stream_generation(**kwargs):
        captured.update(kwargs)

        async def gen():
            yield 'data: {"type": "done", "id": 1}\n\n'

        return gen()

    monkeypatch.setattr(module, "_stream_generation", fake_stream_generation)

    db = FakeDb()
    response = _run(
        module.create_multi_image_generation(
            request=FakeRequest(),
            prompt="Generate a design.",
            images_meta=json.dumps([{"label": "space", "source": "file:0"}]),
            image_specs=None,
            model="pro",
            aspect_ratio=None,
            files=[FakeUpload()],
            cf_turnstile_token=None,
            current_user=FakeUser(),
            db=db,
        )
    )

    assert isinstance(response, StreamingResponse)
    assert db.events == []
    assert captured["user_id"] == FakeUser.id
    assert captured["guest_session_id"] is None
    assert captured["descriptors"][0].type == "file"
    assert captured["descriptors"][0].pil_image is not None


def test_guest_request_commits_session_before_sse_worker(monkeypatch):
    from app.routers import multi_image_gen as module

    guest_session_id = str(uuid.uuid4())
    db = FakeDb()
    request_guest = SimpleNamespace(session_id=guest_session_id, source="request")
    captured = {}

    async def fake_verify_turnstile(token: str, ip: str) -> bool:
        return True

    async def fake_get_or_create_guest_session(db_arg, session_id, *, identity, usage_kind=None):
        assert db_arg is db
        assert session_id == guest_session_id
        assert identity.fingerprint_hash is None
        db_arg.events.append("guest-created")
        return request_guest

    def fake_stream_generation(**kwargs):
        captured.update(kwargs)
        assert db.events == ["guest-created", "commit"]

        async def gen():
            yield 'data: {"type": "done", "id": 1}\n\n'

        return gen()

    from app.routers import guest as guest_module
    monkeypatch.setattr(guest_module, "verify_turnstile", fake_verify_turnstile)
    monkeypatch.setattr(guest_module, "get_or_create_guest_session", fake_get_or_create_guest_session)
    monkeypatch.setattr(module, "_stream_generation", fake_stream_generation)

    response = _run(
        module.create_multi_image_generation(
            request=FakeRequest({"x-guest-session-id": guest_session_id}),
            prompt="Generate a design.",
            images_meta=json.dumps([{"label": "space", "source": "file:0"}]),
            image_specs=None,
            model="pro",
            aspect_ratio=None,
            files=[FakeUpload()],
            cf_turnstile_token=None,
            current_user=None,
            db=db,
        )
    )

    assert isinstance(response, StreamingResponse)
    assert captured["guest_session_id"] == guest_session_id
    assert captured["guest_session"] is request_guest


def test_guest_request_uses_canonical_session_id_after_fingerprint_remap(monkeypatch):
    """Header UUID may differ from fingerprint-matched row; SSE must use canonical id."""
    from app.routers import multi_image_gen as module

    header_uuid = str(uuid.uuid4())
    canonical_uuid = str(uuid.uuid4())
    assert header_uuid != canonical_uuid

    db = FakeDb()
    canonical_guest = SimpleNamespace(session_id=canonical_uuid, source="fingerprint")
    captured = {}

    async def fake_verify_turnstile(token: str, ip: str) -> bool:
        return True

    async def fake_get_or_create_guest_session(db_arg, session_id, *, identity, usage_kind=None):
        assert session_id == header_uuid
        db_arg.events.append("guest-created")
        return canonical_guest

    def fake_stream_generation(**kwargs):
        captured.update(kwargs)

        async def gen():
            yield 'data: {"type": "done", "id": 1}\n\n'

        return gen()

    from app.routers import guest as guest_module
    monkeypatch.setattr(guest_module, "verify_turnstile", fake_verify_turnstile)
    monkeypatch.setattr(guest_module, "get_or_create_guest_session", fake_get_or_create_guest_session)
    monkeypatch.setattr(module, "_stream_generation", fake_stream_generation)

    response = _run(
        module.create_multi_image_generation(
            request=FakeRequest({
                "x-guest-session-id": header_uuid,
                "x-guest-fingerprint": "a" * 64,
            }),
            prompt="Remove the fencing.",
            images_meta=json.dumps([{"label": "space", "source": "file:0"}]),
            image_specs=None,
            model="pro",
            aspect_ratio=None,
            files=[FakeUpload()],
            cf_turnstile_token="token",
            current_user=None,
            db=db,
        )
    )

    assert isinstance(response, StreamingResponse)
    assert captured["guest_session_id"] == canonical_uuid
    assert captured["guest_session"] is canonical_guest
    assert db.events == ["guest-created", "commit"]


def test_guest_sse_worker_reloads_guest_session_before_saving(monkeypatch):
    from app.routers import multi_image_gen as module

    guest_session_id = str(uuid.uuid4())
    worker_db = SimpleNamespace(committed=False, executed=False)
    reloaded_guest = SimpleNamespace(session_id=guest_session_id, source="worker")
    captured = {}

    class FakeResult:
        def scalar_one_or_none(self):
            return reloaded_guest

    class FakeWorkerSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def execute(self, statement):
            worker_db.executed = True
            return FakeResult()

        async def commit(self):
            worker_db.committed = True

    def fake_session_maker():
        return FakeWorkerSession()

    async def fake_generate_multi_image(**kwargs):
        captured.update(kwargs)
        return {
            "id": 7,
            "generation_name": "multi.png",
            "url": "guest/key",
            "signed_url": "https://signed.example/multi.png",
            "model_used": "flash",
            "images_used": ["space (image 1)"],
        }

    monkeypatch.setattr(module, "async_session_maker", fake_session_maker)
    monkeypatch.setattr(module, "generate_multi_image", fake_generate_multi_image)

    async def collect():
        return [
            chunk
            async for chunk in module._stream_generation(
                descriptors=[],
                prompt="Generate a design.",
                model="flash",
                aspect_ratio=None,
                user_id=None,
                guest_session_id=guest_session_id,
                guest_session=SimpleNamespace(session_id=guest_session_id, source="request"),
                db=object(),
            )
        ]

    chunks = _run(collect())

    assert worker_db.executed is True
    assert worker_db.committed is True
    assert captured["guest_session"] is reloaded_guest
    assert captured["guest_session_id"] == guest_session_id
    assert captured["user_id"] is None
    assert any('"type": "done"' in chunk for chunk in chunks)
