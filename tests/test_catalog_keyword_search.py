"""Keyword search uses filename→meta dict + SQL-side filter narrowing."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_image_info_by_filename_is_o1(monkeypatch):
    from app import image_utils

    monkeypatch.setattr(
        image_utils,
        "_IMAGE_INFO_CACHE",
        [{"filename": "a.jpg", "class": "Pergola"}, {"filename": "b.jpg", "class": "Fence"}],
    )
    monkeypatch.setattr(
        image_utils,
        "_IMAGE_INFO_BY_FILENAME",
        {"a.jpg": {"filename": "a.jpg", "class": "Pergola"}, "b.jpg": {"filename": "b.jpg", "class": "Fence"}},
    )
    monkeypatch.setattr(image_utils, "_IMAGE_INFO_CACHE_AT", 1e18)

    assert image_utils.get_image_metadata("a.jpg")["class"] == "Pergola"
    assert image_utils.get_image_metadata("missing.jpg") is None


def test_filenames_matching_filters_narrows():
    from app.catalog_keyword_search import _filenames_matching_filters

    meta = {
        "pergola.jpg": {"class": "Pergola", "level": 5, "tags": ["wood"]},
        "fence.jpg": {"class": "Fence", "level": 1, "tags": ["metal"]},
    }
    allowed = _filenames_matching_filters(
        meta,
        level="Designs",
        class_="Pergola",
        category=None,
        tag_list=[],
        tag_logic_norm="OR",
    )
    assert allowed == {"pergola.jpg"}


def test_keyword_search_uses_filename_in_sql(monkeypatch):
    from app import catalog_keyword_search as cks

    meta = {
        "pergola.jpg": {"class": "Pergola", "level": 5, "tags": ["wood"], "name": "Wood Pergola"},
        "fence.jpg": {"class": "Fence", "level": 1, "tags": ["metal"], "name": "Metal Fence"},
    }
    monkeypatch.setattr(cks, "image_info_by_filename", lambda: meta)

    captured = {}

    class _Result:
        def scalars(self):
            return self

        def all(self):
            return [
                SimpleNamespace(id=1, name="Wood Pergola", filename="pergola.jpg", url="/u/1"),
            ]

    async def fake_execute(stmt):
        # SQLAlchemy 2 select has whereclause when .where() was applied
        captured["has_where"] = stmt.whereclause is not None
        return _Result()

    db = AsyncMock()
    db.execute = fake_execute

    out = asyncio.run(
        cks.keyword_search_catalog(db, query="wood", level="Designs", class_="Pergola")
    )
    assert captured["has_where"] is True
    assert out["hit_count"] == 1
    assert out["hits"][0]["filename"] == "pergola.jpg"


def test_serialize_hit_removed():
    import app.catalog_keyword_search as cks

    assert not hasattr(cks, "_serialize_hit")
