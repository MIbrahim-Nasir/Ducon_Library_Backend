"""Admin panel, runtime config, and metrics subsystem.

Public surface:
    settings_store  — hot-reloaded tunable config (DB > env > default)
    usage_recorder  — non-blocking AI usage/cost tracking (asyncio queue + batch insert)
    pricing         — per-model token pricing for cost calculation
    admin_auth      — require_admin / require_admin_or_analytics (JWT role + admin session)
    settings_catalog — declarative registry of admin-controllable keys
"""
from app.admin import settings_catalog  # noqa: F401
