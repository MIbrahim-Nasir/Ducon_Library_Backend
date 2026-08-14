"""Admin settings PUT must never write NULL updated_at (prod 500 on model toggle)."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.admin.settings_store import SettingsStore
from app.db.models import AppSetting


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class _FakeDb:
    def __init__(self, row=None):
        self.row = row
        self.added = []
        self.committed = False
        self.refreshed = []

    async def execute(self, _stmt):
        return _FakeResult(self.row)

    def add(self, row):
        self.added.append(row)

    async def commit(self):
        if self.row is not None and self.row.updated_at is None:
            raise AssertionError(
                "commit would SET updated_at=NULL (NotNullViolationError)"
            )
        for row in self.added:
            if getattr(row, "updated_at", "missing") is None:
                raise AssertionError("insert left updated_at=None")
        self.committed = True

    async def refresh(self, row):
        self.refreshed.append(row)


@pytest.mark.asyncio
async def test_put_existing_row_stamps_updated_at_not_null():
    existing = AppSetting(
        namespace="ai_models",
        key="USE_CLAUDE",
        value="false",
        value_type="bool",
        is_secret=False,
        description="test",
        updated_by=1,
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    existing.id = 9
    db = _FakeDb(existing)
    store = SettingsStore()

    row = await store.put("ai_models", "USE_CLAUDE", True, admin_user_id=10, db=db)

    assert db.committed is True
    assert row.updated_at is not None
    assert isinstance(row.updated_at, datetime)
    assert row.updated_at.tzinfo is not None
    assert row.updated_at > datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert row.updated_by == 10
    assert store.cfg("USE_CLAUDE") is True


@pytest.mark.asyncio
async def test_put_new_row_does_not_force_null_updated_at():
    db = _FakeDb(row=None)
    store = SettingsStore()

    row = await store.put("ai_models", "USE_CLAUDE", False, admin_user_id=3, db=db)

    assert db.committed is True
    assert db.added == [row]
    assert getattr(row, "updated_at", SimpleNamespace()) is not None
    assert row.key == "USE_CLAUDE"
    assert store.cfg("USE_CLAUDE") is False
