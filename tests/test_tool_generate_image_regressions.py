import asyncio
import io

from PIL import Image


def _png_bytes(color: str = "white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_generate_multi_image_passes_keyword_only_thread_args(monkeypatch):
    """Regression: _run_generation_sync has keyword-only image_thinking/metrics."""
    from app import tool_generate_image as module

    captured = {}

    async def fake_to_thread(func, *args, **kwargs):
        # The resolve loop and dry-run final decode are now also offloaded via
        # asyncio.to_thread (lambdas) — return a PIL image for those so the
        # pipeline's img.size / pil_images introspection succeeds. Only the
        # generation call (_run_generation_sync) returns raw PNG bytes.
        if func is module._run_generation_sync:
            captured["func"] = func
            captured["args"] = args
            captured["kwargs"] = kwargs
            return _png_bytes()
        return Image.new("RGB", (8, 8), color="blue")

    monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)

    image = Image.new("RGB", (8, 8), color="blue")
    descriptor = module.ImageDescriptor(
        label="user space",
        type="file",
        pil_image=image,
    )

    result = asyncio.run(
        module.generate_multi_image(
            user_id=123,
            prompt="Place the pergola in the garden.",
            descriptors=[descriptor],
            model="flash",
            image_model="test-image-model",
            image_thinking="Low",
            enable_verify=False,
            dry_run=True,
        )
    )

    assert result["image_bytes"]
    assert captured["func"] is module._run_generation_sync
    assert captured["args"] == (
        "test-image-model",
        captured["args"][1],
        ["user space"],
        "Place the pergola in the garden.",
        "auto",
        123,
    )
    assert captured["kwargs"]["image_thinking"] == "Low"
    assert "metrics" in captured["kwargs"]


def test_classify_image_roles_designer_order():
    from app.tool_generate_image import classify_image_roles

    roles = classify_image_roles([
        "client space photo",
        "Modern Ducon pergola",
        "Outdoor seating product",
    ])

    assert roles == {
        "user_space_index": 0,
        "design_direction_index": 1,
        "product_indices": [2],
    }


def test_classify_image_roles_legacy_order():
    from app.tool_generate_image import classify_image_roles

    roles = classify_image_roles([
        "Ducon design direction",
        "User space photo",
        "Outdoor seating product",
    ])

    assert roles == {
        "user_space_index": 1,
        "design_direction_index": 0,
        "product_indices": [2],
    }


def test_classify_image_roles_does_not_treat_design_reference_room_photo_as_user_space():
    from app.tool_generate_image import classify_image_roles

    roles = classify_image_roles([
        "Ducon design reference room photo",
        "courtyard upload",
        "Outdoor seating product",
    ])

    assert roles == {
        "user_space_index": 1,
        "design_direction_index": 0,
        "product_indices": [2],
    }


def test_classify_image_roles_voice_shorthand_labels():
    """Voice agents often use short labels like 'user terrace' / 'Ducon pergola'."""
    from app.tool_generate_image import classify_image_roles

    roles = classify_image_roles([
        "user terrace",
        "Ducon pergola",
        "Fountain product",
    ])

    assert roles == {
        "user_space_index": 0,
        "design_direction_index": 1,
        "product_indices": [2],
    }


def test_generate_multi_image_enable_verify_uses_image_gen_agent(monkeypatch):
    """Chat/voice path with 2+ labeled images should construct ImageGenAgent."""
    from app import tool_generate_image as module

    constructed = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

        async def generate_initial_prompt(self, **kwargs):
            return "enhanced studio-style prompt"

        async def evaluate_output(self, **kwargs):
            return True, []

        def schedule_post_success_improvement(self):
            pass

        async def generate_retry_prompt(self, **kwargs):
            return "retry prompt"

    monkeypatch.setattr(module, "ImageGenAgent", FakeAgent)

    async def fake_to_thread(func, *args, **kwargs):
        if func is module._run_generation_sync:
            return _png_bytes()
        return Image.new("RGB", (8, 8), color="blue")

    monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)

    descriptors = [
        module.ImageDescriptor(
            label="User space photo",
            type="file",
            pil_image=Image.new("RGB", (8, 8), color="green"),
        ),
        module.ImageDescriptor(
            label="Ducon design direction",
            type="file",
            pil_image=Image.new("RGB", (8, 8), color="red"),
        ),
    ]

    result = asyncio.run(
        module.generate_multi_image(
            user_id=1,
            prompt="Apply the design to the space.",
            descriptors=descriptors,
            model="flash",
            image_model="test-image-model",
            enable_verify=True,
            dry_run=True,
            max_eval_rounds=1,
        )
    )

    assert constructed.get("image2_is_user_space") is True
    assert constructed.get("labels") == ["User space photo", "Ducon design direction"]
    assert result["approved"] is True
    assert result.get("image_bytes")


def test_run_generation_sync_empty_candidates_raises_clear_error(monkeypatch):
    """Safety-blocked / empty Gemini responses must not IndexError."""
    from types import SimpleNamespace

    from app import tool_generate_image as module

    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(candidates=[], usage_metadata=None)

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(module, "get_gemini_client", lambda: FakeClient())
    monkeypatch.setattr(
        "app.admin.usage_helpers.record_from_response",
        lambda *a, **k: None,
    )

    try:
        module._run_generation_sync(
            "test-model",
            [Image.new("RGB", (8, 8), color="blue")],
            ["user space"],
            "Remove the fencing.",
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "no candidates" in str(exc).lower()


def test_resolve_descriptor_guest_generation_scoped_to_session(monkeypatch):
    """Guests must load gen:N from guest_generations for their session only."""
    from types import SimpleNamespace

    from app import tool_generate_image as module

    guest_row = SimpleNamespace(
        id=55,
        guest_session_id="guest-aaa",
        url="guests/guest-aaa/out.png",
    )
    executed = []

    class FakeResult:
        def __init__(self, row):
            self._row = row

        def scalar_one_or_none(self):
            return self._row

    class FakeDb:
        async def execute(self, statement):
            executed.append(statement)
            return FakeResult(guest_row)

    async def fake_load(row, *, label):
        assert row is guest_row
        assert label == "previous"
        return Image.new("RGB", (4, 4), color="yellow")

    monkeypatch.setattr(module, "_load_generation_image", fake_load)

    img = asyncio.run(
        module._resolve_descriptor(
            module.ImageDescriptor(label="previous", type="generation_id", source="55"),
            FakeDb(),
            guest_session_id="guest-aaa",
        )
    )
    assert img.size == (4, 4)
    assert len(executed) == 1


def test_resolve_descriptor_guest_cannot_read_user_generation_table(monkeypatch):
    """Guest gen:N must not fall through to the authenticated generations table."""
    from types import SimpleNamespace

    from app import tool_generate_image as module

    class FakeResult:
        def scalar_one_or_none(self):
            return None

    class FakeDb:
        async def execute(self, statement):
            # Only guest_generations query should run; returning None → not found.
            return FakeResult()

    called_user_load = {"v": False}

    async def fake_load(row, *, label):
        called_user_load["v"] = True
        return Image.new("RGB", (4, 4))

    monkeypatch.setattr(module, "_load_generation_image", fake_load)

    try:
        asyncio.run(
            module._resolve_descriptor(
                module.ImageDescriptor(label="previous", type="generation_id", source="99"),
                FakeDb(),
                guest_session_id="guest-bbb",
            )
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "generation 99 not found" in str(exc)
    assert called_user_load["v"] is False
