"""Generation image serving and stored-key path resolution."""

from __future__ import annotations

import pytest


def test_guest_stored_key_resolves_under_outputs_guests(monkeypatch, tmp_path):
    from app import storage

    monkeypatch.setattr(storage, "CLOUD_STORAGE", False)
    monkeypatch.setattr(storage, "_OUTPUTS_DIR", tmp_path)

    session_id = "sess-abc"
    filename = "gen.png"
    guest_dir = tmp_path / "guests" / session_id
    guest_dir.mkdir(parents=True)
    (guest_dir / filename).write_bytes(b"png")

    key = f"guests/{session_id}/{filename}"
    path = storage.serve_local_path(key)
    assert path == (tmp_path / "guests" / session_id / filename).resolve()
    assert path.exists()


def test_user_stored_key_resolves_under_outputs_user_id(monkeypatch, tmp_path):
    from app import storage

    monkeypatch.setattr(storage, "CLOUD_STORAGE", False)
    monkeypatch.setattr(storage, "_OUTPUTS_DIR", tmp_path)

    user_id = "7"
    filename = "gen.png"
    user_dir = tmp_path / user_id
    user_dir.mkdir(parents=True)
    (user_dir / filename).write_bytes(b"png")

    key = f"generations/{user_id}/{filename}"
    path = storage.serve_local_path(key)
    assert path == (tmp_path / user_id / filename).resolve()
    assert path.exists()


def test_should_proxy_generation_images_only_in_cloud_proxy_mode(monkeypatch):
    from app import storage

    monkeypatch.setattr(storage, "CLOUD_STORAGE", True)
    monkeypatch.setattr(storage, "GENERATION_IMAGE_SERVE_MODE", "redirect")
    assert storage.should_proxy_generation_images() is False

    monkeypatch.setattr(storage, "GENERATION_IMAGE_SERVE_MODE", "proxy")
    assert storage.should_proxy_generation_images() is True

    monkeypatch.setattr(storage, "CLOUD_STORAGE", False)
    monkeypatch.setattr(storage, "GENERATION_IMAGE_SERVE_MODE", "proxy")
    assert storage.should_proxy_generation_images() is False


def test_read_generation_bytes_roundtrip_local(monkeypatch, tmp_path):
    from app import storage

    monkeypatch.setattr(storage, "CLOUD_STORAGE", False)
    monkeypatch.setattr(storage, "_OUTPUTS_DIR", tmp_path)

    key = "generations/3/out.png"
    local = tmp_path / "3"
    local.mkdir(parents=True)
    (local / "out.png").write_bytes(b"image-bytes")

    assert storage.read_generation_bytes(key) == b"image-bytes"
    assert storage.read_generation_bytes("generations/3/missing.png") is None
