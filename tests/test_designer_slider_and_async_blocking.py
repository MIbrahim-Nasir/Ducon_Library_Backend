"""Regression tests for the Ducon Designer slider "before" image and for
async/non-blocking server paths.

Two concerns are guarded here:

1. SLIDER "BEFORE" IMAGE (Issue 1)
   The designer job's comparison slider must show the user's *original* uploaded
   space photo as the "before" image, not a Ducon catalogue reference. The fix
   persists the user's normalized input image via dedicated storage helpers (no
   watermark), exposes it through `GET /designer/jobs/{job_id}/input-image`, and
   surfaces it in the job's `final` payload as `user_image` plus an early
   `input_image` SSE event. These tests pin the storage contract and the
   payload/event contract so the user image is never silently dropped again.

2. ASYNC / NON-BLOCKING REGRESSION (Issue 2)
   Synchronous, event-loop-blocking calls (PIL decode/encode via
   `normalize_user_image` / `Image.open`, sync `httpx.get`/`httpx.post`,
   `time.sleep`, blocking `requests.*`, and the sync `client.models.generate_content`)
   must not run directly inside `async def` paths — they must be offloaded with
   `asyncio.to_thread(...)` or replaced by the async `client.aio.*` API. The AST
   walker below flags any such call that is lexically *directly* inside an
   `async def` body (i.e. not inside a nested sync function/lambda, which only
   runs when that callable is invoked — and those callables are themselves
   offloaded). This is a targeted guard for the highest-impact blocking calls,
   not an exhaustive CPU-work detector.
"""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import pytest
from PIL import Image


# ─── helpers ──────────────────────────────────────────────────────────────

def _png_bytes(color: str = "white", size: int = 12) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color=color).save(buf, format="PNG")
    return buf.getvalue()


# ─── Issue 1: designer input image storage ─────────────────────────────────

def test_designer_input_key_is_stable_and_namespaced():
    from app import storage

    key = storage.designer_input_key(42, "job-abc")
    assert key == "designer_inputs/42/job-abc.png"
    # user_id is coerced to int so a stray float/str can't escape the prefix.
    assert storage.designer_input_key("7", "job-abc") == "designer_inputs/7/job-abc.png"


def test_designer_input_image_storage_roundtrip_local(monkeypatch, tmp_path):
    """Local mode: save → url → serve path → exists forms a complete round-trip
    that the slider "before" endpoint relies on. No watermark is applied."""
    from app import storage

    monkeypatch.setattr(storage, "CLOUD_STORAGE", False)
    monkeypatch.setattr(storage, "_OUTPUTS_DIR", tmp_path)

    uid, job_id = 42, "job-abc"
    raw = _png_bytes("blue")

    key = storage.save_designer_input(uid, job_id, raw)
    assert key == storage.designer_input_key(uid, job_id)

    # Local serving URL points at the auth-gated FastAPI endpoint (not R2).
    url = storage.get_designer_input_url(job_id, key)
    assert url == f"/designer/jobs/{job_id}/input-image"

    # The on-disk path resolves under the outputs dir and actually exists.
    path = storage.serve_designer_input_path(key)
    assert path.parent == (tmp_path / "designer_inputs" / str(uid)).resolve()
    assert path.read_bytes() == raw
    assert storage.designer_input_exists(key) is True

    # The persisted bytes are the exact normalized input — no watermark added.
    reloaded = Image.open(io.BytesIO(path.read_bytes()))
    assert reloaded.size == (12, 12)


def test_designer_input_url_local_contains_job_id(monkeypatch, tmp_path):
    """The frontend fetches the "before" image via this URL; it must carry the
    job_id so the auth-gated endpoint can locate the row."""
    from app import storage

    monkeypatch.setattr(storage, "CLOUD_STORAGE", False)
    monkeypatch.setattr(storage, "_OUTPUTS_DIR", tmp_path)
    key = storage.save_designer_input(5, "job-xyz", _png_bytes("red"))
    url = storage.get_designer_input_url("job-xyz", key)
    assert "job-xyz" in url and url.startswith("/designer/jobs/")


def test_designer_input_rejects_path_traversal_key(monkeypatch, tmp_path):
    """A malformed stored key must never resolve outside the outputs dir."""
    from app import storage

    monkeypatch.setattr(storage, "CLOUD_STORAGE", False)
    monkeypatch.setattr(storage, "_OUTPUTS_DIR", tmp_path)
    with pytest.raises(ValueError):
        storage.serve_designer_input_path("designer_inputs/../../etc/passwd")
    assert storage.designer_input_exists("bogous/not/a/key") is False


# ─── Issue 1: designer final payload + event contract ──────────────────────

DESIGNER_AGENT = Path(__file__).resolve().parent.parent / "app" / "designer_agent.py"


def _designer_ast() -> ast.Module:
    return ast.parse(DESIGNER_AGENT.read_text(encoding="utf-8"))


def test_designer_final_payload_includes_user_image_before_slot():
    """The `job.final` dict must carry a `user_image` entry (the slider "before")
    so the frontend can place the user's original photo first, ahead of the
    generated designs and Ducon references."""
    tree = _designer_ast()

    found_user_image = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        tgt = node.targets[0] if node.targets else None
        if not (isinstance(tgt, ast.Attribute) and tgt.attr == "final"
                and isinstance(tgt.value, ast.Name) and tgt.value.id == "job"):
            continue
        if isinstance(node.value, ast.Dict):
            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
            found_user_image = found_user_image or ("user_image" in keys)

    assert found_user_image, (
        "job.final payload no longer includes a 'user_image' key — the slider "
        "'before' image (user's original space photo) would be dropped."
    )


def test_designer_emits_input_image_event_with_label():
    """An early `input_image` SSE event must be emitted carrying the user's
    space photo URL labelled 'Your space', so the frontend can show the
    'before' image before generation completes."""
    src = DESIGNER_AGENT.read_text(encoding="utf-8")
    tree = ast.parse(src)

    emitted = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not (isinstance(call.func, ast.Name) and call.func.id == "emit"):
            continue
        # emit(job, "input_image", user_image={...}, message=...)
        if any(isinstance(a, ast.Constant) and a.value == "input_image" for a in call.args):
            # Confirm the user_image kwarg carries the "Your space" label.
            for kw in call.keywords:
                if kw.arg == "user_image" and isinstance(kw.value, ast.Dict):
                    for k, v in zip(kw.value.keys, kw.value.values):
                        if (isinstance(k, ast.Constant) and k.value == "label"
                                and isinstance(v, ast.Constant) and v.value == "Your space"):
                            emitted = True
    assert emitted, (
        "Designer agent must emit `input_image` with user_image.label == 'Your space' "
        "so the slider 'before' is the user's original photo."
    )


def test_designer_input_image_endpoint_exists():
    """The auth-gated serving endpoint for the persisted input image must exist
    on the designer jobs router (used by aiService.getDesignerInputBlobUrl)."""
    router_src = (DESIGNER_AGENT.parent / "routers" / "designer_jobs.py").read_text(encoding="utf-8")
    assert '"/{job_id}/input-image"' in router_src, (
        "GET /designer/jobs/{job_id}/input-image endpoint is missing — the "
        "frontend cannot fetch the user's original space photo for the slider."
    )


# ─── Issue 2: async / non-blocking AST regression ──────────────────────────

# Files whose async paths were audited/fixed. image_utils.load_ducon_image uses
# a lazy Image.open (header read only) by design — its callers force the decode
# inside to_thread — so it is intentionally excluded to avoid false positives.
_ASYNC_FILES = [
    "app/designer_agent.py",
    "app/tool_generate_image.py",
    "app/studio_directions_agent.py",
    "app/image_gen_agent.py",
    "app/prompt_generator_session.py",
    "app/chat_agent.py",
    "app/gemini.py",
    "app/main.py",
    "app/routers/images.py",
    "app/routers/multi_image_gen.py",
    "app/routers/designer_jobs.py",
    "app/otp_service.py",
    "app/routers/contact.py",
]

_BLOCKING_NAMES = {"normalize_user_image", "_normalize_user_image"}
# attribute-style blocking calls: (trail of attribute attrs from a Name root)
_BLOCKING_ATTR_TRAILS = {
    ("Image", "open"),
    ("time", "sleep"),
    ("httpx", "get"),
    ("httpx", "post"),
    ("requests", "get"),
    ("requests", "post"),
    ("client", "models", "generate_content"),  # sync SDK; client.aio.* is allowed
}


def _attr_trail(func: ast.expr) -> tuple[str, ...] | None:
    """Collapse an attribute/Name chain into a tuple of names, e.g.
    client.models.generate_content -> ('client', 'models', 'generate_content')."""
    parts: list[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return tuple(reversed(parts))
    return None


class _BlockingFinder(ast.NodeVisitor):
    """Flags blocking Calls that run *directly* on the event loop.

    A call is "directly on the loop" when it is lexically inside an `async def`
    body and NOT inside a nested sync `def`/`lambda` (those bodies only execute
    when the nested callable is invoked, and those callables are themselves
    offloaded via `asyncio.to_thread`). Calls that appear as bare references
    (e.g. `asyncio.to_thread(normalize_user_image, data)`) are Name nodes, not
    Call nodes, so they are naturally not flagged.
    """

    def __init__(self) -> None:
        self.violations: list[tuple[str, int]] = []
        # Stack of "is this frame's body executed on the loop?". Top-level = off.
        self._on_loop_stack: list[bool] = []

    @property
    def _directly_on_loop(self) -> bool:
        return bool(self._on_loop_stack) and self._on_loop_stack[-1]

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._on_loop_stack.append(True)
        self.generic_visit(node)
        self._on_loop_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # A nested *sync* function's body runs only when it is called, not when
        # the enclosing async def runs — so it is not directly on the loop.
        self._on_loop_stack.append(False)
        self.generic_visit(node)
        self._on_loop_stack.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # A lambda body runs only when the lambda is called (typically offloaded
        # via asyncio.to_thread(lambda: ...)).
        self._on_loop_stack.append(False)
        self.generic_visit(node)
        self._on_loop_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        name = self._blocking_name(node)
        if name and self._directly_on_loop:
            self.violations.append((name, node.lineno))
        self.generic_visit(node)

    @staticmethod
    def _blocking_name(node: ast.Call) -> str | None:
        f = node.func
        if isinstance(f, ast.Name) and f.id in _BLOCKING_NAMES:
            return f.id
        trail = _attr_trail(f)
        if trail and trail in _BLOCKING_ATTR_TRAILS:
            return ".".join(trail)
        return None


def test_no_blocking_calls_directly_in_async_paths():
    """No audited async path may run a known blocking call on the event loop."""
    backend_root = Path(__file__).resolve().parent.parent
    all_violations: list[tuple[str, str, int]] = []

    for rel in _ASYNC_FILES:
        path = backend_root / rel
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        finder = _BlockingFinder()
        finder.visit(tree)
        for name, lineno in finder.violations:
            all_violations.append((rel, name, lineno))

    assert not all_violations, (
        "Blocking calls found directly inside async paths (must be wrapped in "
        "asyncio.to_thread or replaced with client.aio.*):\n" +
        "\n".join(f"  {rel}:{lineno} -> {name}" for rel, name, lineno in all_violations)
    )


def test_gemini_vision_calls_reachable_from_async_are_offloaded():
    """gemini._vision_json / evaluate_generation / verify_prompt / combine_images
    are sync and contain client.models.generate_content. They must never be
    awaited directly from an async path — only via asyncio.to_thread wrappers
    (_call_verify / _call_eval / _call_eval_multi) or async client.aio.* ."""
    backend_root = Path(__file__).resolve().parent.parent
    tool_src = (backend_root / "app" / "tool_generate_image.py").read_text(encoding="utf-8")
    main_src = (backend_root / "app" / "main.py").read_text(encoding="utf-8")

    # The sync wrappers must be invoked through asyncio.to_thread, not awaited.
    for wrapper in ("_call_verify", "_call_eval_multi", "_call_eval"):
        assert f"asyncio.to_thread(\n                    {wrapper}" in tool_src or \
               f"asyncio.to_thread(\n                {wrapper}" in tool_src or \
               f"to_thread(\n                    {wrapper}" in tool_src or \
               f"to_thread(\n                {wrapper}" in tool_src, \
               f"{wrapper} must be called via asyncio.to_thread, not awaited directly"

    # gemini.combine_images left main.py with the /autogenerate-images removal;
    # guard against a blocking call sneaking back onto the event loop.
    assert "gemini.combine_images" not in main_src, \
        "main.py must not call sync gemini.combine_images on the event loop"


def test_email_and_metadata_httpx_offloaded():
    """Resend email (sync httpx.post) and remote metadata (sync httpx.get) must
    be offloaded so async routes don't stall the event loop waiting on the API."""
    backend_root = Path(__file__).resolve().parent.parent
    otp_src = (backend_root / "app" / "otp_service.py").read_text(encoding="utf-8")
    contact_src = (backend_root / "app" / "routers" / "contact.py").read_text(encoding="utf-8")
    images_src = (backend_root / "app" / "routers" / "images.py").read_text(encoding="utf-8")

    assert "asyncio.to_thread(send_otp_email" in otp_src, \
        "send_otp_email (sync httpx.post) must be offloaded in otp_service.py"
    assert contact_src.count("asyncio.to_thread(\n        send_contact_email") == 2, \
        "Both send_contact_email calls in contact.py must be offloaded"
    assert "asyncio.to_thread(_load_metadata)" in images_src, \
        "_load_metadata (sync httpx.get) must be offloaded in images.py"


def test_chat_generation_card_css_constrains_image_overflow():
    """In-chat generation previews must not expand the designer modal horizontally."""
    css_path = Path(__file__).resolve().parents[2] / "Ducon_Library" / "src" / "index.css"
    if not css_path.is_file():
        pytest.skip(f"Frontend CSS not found at {css_path}")

    css = css_path.read_text(encoding="utf-8")
    for rule in (
        ".ck-gen-output img",
        "object-fit: contain",
        "position: absolute",
        ".dc-messages",
        "overflow-x: hidden",
        ".ck-msg-ai .ck-msg-body",
        "max-width: calc(100% - 36px)",
    ):
        assert rule in css, f"Missing chat image overflow guard: {rule!r}"
