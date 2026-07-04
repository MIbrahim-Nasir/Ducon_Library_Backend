"""Hot-reloaded runtime configuration.

Precedence for every key: app_settings DB row > os.getenv(key) > spec.default.

Catalog defaults (``settings_catalog.py``) should match ``env_template.txt`` so
the admin UI shows the same initial values as a fresh ``.env``. Once you save
from admin, the DB row wins until you change or delete it.

Agents read config via the sync ``cfg()`` helper, which reads an in-memory
cache that is updated atomically on admin PUT. This means:
  - No await on the hot path (just a dict lookup).
  - Changes take effect immediately for new requests, no restart.
  - Existing .env values apply when no admin override exists in DB.

Secrets are intentionally NOT cached or exposed here — they remain only in
os.environ and are read by the modules that need them directly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.settings_catalog import (
    SettingSpec,
    all_keys,
    cast_value,
    decode_value,
    encode_value,
    get_namespace,
    get_spec,
)
from app.db.models import AppSetting

logger = logging.getLogger(__name__)


class SettingsStore:
    def __init__(self) -> None:
        # "namespace.key" -> native value (typed)
        self._cache: dict[str, Any] = {}
        # key -> raw DB row (so we can answer "is this from DB or env?")
        self._db_keys: set[str] = set()
        self._lock = asyncio.Lock()
        self._loaded = False

    # ── loading ────────────────────────────────────────────────────────────

    async def load_all(self, db: AsyncSession) -> None:
        """Initial load from DB. Called once at startup."""
        async with self._lock:
            result = await db.execute(select(AppSetting))
            rows = result.scalars().all()
            self._cache.clear()
            self._db_keys.clear()
            for row in rows:
                spec = get_spec(row.key)
                if spec is None:
                    # Unknown key (e.g. removed from catalog) — skip but remember
                    self._db_keys.add(row.key)
                    continue
                try:
                    val = decode_value(spec, row.value)
                except ValueError as e:
                    logger.warning("app_settings %s has invalid value: %s", row.key, e)
                    continue
                self._cache[f"{row.namespace}.{row.key}"] = val
                self._db_keys.add(row.key)
            # Seed env-fallback values for non-secret tunables so cfg() is fast
            for key in all_keys():
                spec = get_spec(key)
                if spec is None or spec.is_secret or not spec.use_env_fallback:
                    continue
                cache_key = f"{get_namespace_for_key(key)}.{key}"
                if cache_key not in self._cache:
                    env_val = os.getenv(key)
                    if env_val is not None:
                        try:
                            self._cache[cache_key] = cast_value(spec, env_val)
                        except ValueError:
                            pass
            self._loaded = True

    # ── reads ──────────────────────────────────────────────────────────────

    def cfg(self, key: str, default: Optional[Any] = None) -> Any:
        """Sync read from in-memory cache. Falls back to env, then to ``default``.

        This is the function agents call instead of os.getenv(key, default).
        """
        spec = get_spec(key)
        cache_key = f"{get_namespace_for_key(key)}.{key}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        # env fallback (only for non-secret, env-fallback-enabled keys)
        if spec and not spec.is_secret and spec.use_env_fallback:
            env_val = os.getenv(key)
            if env_val is not None and env_val != "":
                try:
                    return cast_value(spec, env_val)
                except ValueError:
                    pass
        if default is not None:
            return default
        if spec is not None:
            return spec.default
        return None

    # ── writes ─────────────────────────────────────────────────────────────

    async def put(
        self,
        namespace: str,
        key: str,
        value: Any,
        admin_user_id: int,
        db: AsyncSession,
    ) -> AppSetting:
        """Upsert a setting and refresh the in-memory cache. Commits the session."""
        spec = get_spec(key)
        if spec is None:
            raise ValueError(f"Unknown setting key: {key}")
        if not spec.editable:
            raise PermissionError(f"{key} is not editable")
        if get_namespace(namespace) is None or get_namespace(namespace).name != namespace:
            # verify the key actually belongs to this namespace
            ns_obj = _find_namespace_for_key(key)
            if ns_obj is None or ns_obj.name != namespace:
                raise ValueError(f"Key {key} does not belong to namespace {namespace}")

        coerced = cast_value(spec, value)
        encoded = encode_value(spec, coerced)

        result = await db.execute(
            select(AppSetting).where(
                AppSetting.namespace == namespace, AppSetting.key == key
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = AppSetting(
                namespace=namespace,
                key=key,
                value=encoded,
                value_type=spec.value_type,
                is_secret=spec.is_secret,
                description=spec.description,
                updated_by=admin_user_id,
            )
            db.add(row)
        else:
            row.value = encoded
            row.value_type = spec.value_type
            row.updated_by = admin_user_id
            row.updated_at = None  # let server_default re-apply on commit
        await db.commit()
        await db.refresh(row)

        # refresh cache atomically
        async with self._lock:
            self._cache[f"{namespace}.{key}"] = coerced
            self._db_keys.add(key)
        return row

    async def invalidate(self, namespace: str, key: str) -> None:
        async with self._lock:
            self._cache.pop(f"{namespace}.{key}", None)
            self._db_keys.discard(key)

    # ── introspection ──────────────────────────────────────────────────────

    def snapshot(self) -> list[dict[str, Any]]:
        """Return all known settings with current effective value + source.

        Secrets are returned with ``value=None`` and ``masked=True``; the actual
        value is only retrievable via the explicit reveal endpoint (re-auth).
        """
        from app.admin.settings_catalog import ALL_NAMESPACES
        out: list[dict[str, Any]] = []
        for ns in ALL_NAMESPACES:
            for spec in ns.settings:
                cache_key = f"{ns.name}.{spec.key}"
                in_db = spec.key in self._db_keys
                effective = self._cache.get(cache_key)
                if effective is None and not spec.is_secret and spec.use_env_fallback:
                    env_val = os.getenv(spec.key)
                    if env_val is not None and env_val != "":
                        try:
                            effective = cast_value(spec, env_val)
                        except ValueError:
                            effective = None
                if effective is None:
                    effective = spec.default
                source = "db" if in_db else (
                    "env" if not spec.is_secret and spec.use_env_fallback and os.getenv(spec.key) not in (None, "")
                    else "default"
                )
                out.append({
                    "namespace": ns.name,
                    "namespace_label": ns.label,
                    "key": spec.key,
                    "label": spec.label,
                    "value_type": spec.value_type,
                    "value": None if spec.is_secret else effective,
                    "masked": spec.is_secret,
                    "is_set": bool(os.getenv(spec.key)) if spec.is_secret else (in_db or os.getenv(spec.key) is not None),
                    "source": source,
                    "description": spec.description,
                    "choices": spec.choices,
                    "min": spec.min,
                    "max": spec.max,
                    "editable": spec.editable,
                    "is_secret": spec.is_secret,
                })
        return out

    def secret_is_set(self, key: str) -> bool:
        """Whether a secret env var is present (without exposing its value)."""
        return bool(os.getenv(key))

    def secret_value(self, key: str) -> Optional[str]:
        """Return the raw secret value. Caller MUST have verified admin re-auth."""
        spec = get_spec(key)
        if spec is None or not spec.is_secret:
            return None
        return os.getenv(key)


# ── helpers ───────────────────────────────────────────────────────────────────

def get_namespace_for_key(key: str) -> str:
    spec = get_spec(key)
    if spec is None:
        return ""
    from app.admin.settings_catalog import ALL_NAMESPACES
    for ns in ALL_NAMESPACES:
        for s in ns.settings:
            if s.key == key:
                return ns.name
    return ""


def _find_namespace_for_key(key: str):
    from app.admin.settings_catalog import ALL_NAMESPACES
    for ns in ALL_NAMESPACES:
        for s in ns.settings:
            if s.key == key:
                return ns
    return None


# ── module-level singleton ────────────────────────────────────────────────────

_store: Optional[SettingsStore] = None


def get_settings_store() -> SettingsStore:
    global _store
    if _store is None:
        _store = SettingsStore()
    return _store


def cfg(key: str, default: Optional[Any] = None) -> Any:
    """Convenience: read config from the singleton store.

    Drop-in replacement for ``os.getenv(key, default)`` in agent hot paths.
    Returns the typed value (DB > env > default).
    """
    return get_settings_store().cfg(key, default)


def cfg_bool(key: str, default: bool = False) -> bool:
    """Read a boolean setting. Handles typed bools from SettingsStore."""
    val = cfg(key, default)
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def cfg_str(key: str, default: str = "") -> str:
    """Read a string setting. Never raises on non-string typed values."""
    val = cfg(key, default)
    if val is None:
        return default
    return str(val)


def cfg_int(key: str, default: int = 0) -> int:
    """Read an integer setting."""
    val = cfg(key, default)
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    return int(str(val).strip())


def cfg_float(key: str, default: float = 0.0) -> float:
    """Read a float setting."""
    val = cfg(key, default)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return float(str(val).strip())
