"""Declarative registry of every admin-controllable setting.

Single source of truth for:
  - the SettingsStore (defaults + types + validation)
  - the /admin/settings API (what's exposed, what's editable, what's secret)
  - the frontend config editors (renders forms from this metadata)

Keep ``SettingSpec.default`` values in sync with ``env_template.txt`` so the
admin UI initial state matches a fresh deployment .env.

Only the seven tunable namespaces are editable. Secrets are read-only and masked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional


@dataclass(frozen=True)
class SettingSpec:
    key: str                       # env var name (also the canonical key)
    label: str
    value_type: str                # string|int|float|bool|choice
    default: Any                   # fallback if neither DB nor env has a value
    description: str = ""
    choices: Optional[list[str]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    is_secret: bool = False
    editable: bool = True          # False for secrets (read-only / masked)
    # env fallback: read os.getenv(key) when no DB row. Disabled for secrets
    # since we never want to expose env-stored secret values through the API.
    use_env_fallback: bool = True


@dataclass(frozen=True)
class Namespace:
    name: str
    label: str
    description: str
    settings: list[SettingSpec] = field(default_factory=list)


# ── 1. AI models ──────────────────────────────────────────────────────────────

AI_MODELS = Namespace(
    name="ai_models",
    label="AI Agents & Models",
    description="Model IDs used by each agent. Changes apply to new requests immediately.",
    settings=[
        SettingSpec("USE_CLAUDE", "Use Claude instead of Gemini (text agents)", "bool", "false",
                    "Routes chat / image-gen / designer agents through Anthropic Claude."),
        SettingSpec("CHAT_MODEL", "Chat agent model", "string", "gemini-3.5-flash"),
        SettingSpec("CLAUDE_MODEL", "Claude model", "string", "claude-sonnet-4-6"),
        SettingSpec("LIVE_MODEL", "Live voice model", "string", "gemini-3.1-flash-live-preview"),
        SettingSpec("DESIGNER_AGENT_MODEL", "Designer agent model", "string", "gemini-3.5-flash"),
        SettingSpec("STUDIO_DIRECTIONS_MODEL", "Studio directions model", "string", "gemini-3-flash-preview"),
        SettingSpec("IMAGE_GEN_MODEL", "Image gen model (studio / autogen)", "string", "gemini-3-pro-image-preview"),
        SettingSpec("MULTI_IMAGE_PRO_MODEL", "Multi-image pro model", "string", "gemini-3-pro-image-preview"),
        SettingSpec("MULTI_IMAGE_FLASH_MODEL", "Multi-image flash model", "string", "gemini-3.1-flash-image-preview"),
        SettingSpec("DESIGNER_AGENT_IMAGE_MODEL", "Designer image model tier", "choice", "flash",
                    choices=["flash", "pro"]),
        SettingSpec("QUOTATION_MODEL", "Quotation model", "string", "gemini-3.1-pro-preview"),
    ],
)


# ── 2. Thinking modes ────────────────────────────────────────────────────────

_THINKING_CHOICES = ["minimal", "low", "medium", "High", "max"]

THINKING = Namespace(
    name="thinking",
    label="Thinking Modes",
    description="Reasoning budget per agent. Higher = more thoughtful but slower and costlier.",
    settings=[
        SettingSpec("CHAT_THINKING_LEVEL", "Chat thinking level", "choice", "low", choices=_THINKING_CHOICES),
        SettingSpec("CLAUDE_THINKING", "Claude thinking mode", "choice", "adaptive",
                    choices=["disabled", "adaptive", "high"]),
        SettingSpec("LIVE_THINKING_LEVEL", "Live voice thinking level", "choice", "minimal", choices=_THINKING_CHOICES),
        SettingSpec("STUDIO_DIRECTIONS_THINKING_LEVEL", "Studio directions thinking level", "choice", "High", choices=_THINKING_CHOICES),
        SettingSpec("QUOTATION_THINKING_LEVEL", "Quotation thinking level", "choice", "High", choices=_THINKING_CHOICES),
        SettingSpec("PROMPT_THINKING_LEVEL", "Prompt writer / verifier thinking level", "choice", "High", choices=_THINKING_CHOICES),
        SettingSpec("IMAGE_THINKING_LEVEL", "Image gen thinking level (Nano Banana 2)", "choice", "High", choices=_THINKING_CHOICES),
        SettingSpec("MULTI_IMAGE_THINKING_LEVEL", "Multi-image thinking level", "choice", "High", choices=_THINKING_CHOICES),
    ],
)


# ── 3. Max limits ─────────────────────────────────────────────────────────────

LIMITS = Namespace(
    name="limits",
    label="Max Limits",
    description="Agent loop bounds, truncation thresholds, and output caps.",
    settings=[
        SettingSpec("DESIGNER_AGENT_MAX_STEPS", "Designer agent max steps", "int", 12, min=1, max=50),
        SettingSpec("DESIGNER_AGENT_MAX_GENERATIONS", "Designer max candidate generations", "int", 3, min=1, max=12),
        SettingSpec("GEN_EVAL_MAX_ROUNDS", "Generation eval retries", "int", 3, min=0, max=10),
        SettingSpec("PROMPT_VERIFY_MAX_ROUNDS", "Prompt verify retries", "int", 2, min=0, max=10),
        SettingSpec("STUDIO_DIRECTIONS_MAX_TURNS", "Studio directions max turns", "int", 8, min=1, max=30),
        SettingSpec("CHAT_TOOL_RESULT_MAX_CHARS", "Chat tool result max chars", "int", 12000, min=1000, max=100000),
        SettingSpec("CHAT_AISEARCH_MAX_ITEMS", "Chat AI search max items", "int", 8, min=1, max=50),
        SettingSpec("MAX_UPLOAD_SIZE_MB", "Max upload size (MB)", "int", 50, min=1, max=500),
        SettingSpec("CLAUDE_CHAT_MAX_MESSAGES", "Claude chat history trim", "int", 60, min=4, max=500),
        SettingSpec("CLAUDE_MAX_TOKENS", "Claude max output tokens", "int", 16000, min=1024, max=200000),
    ],
)


# ── 4. Metadata paths ─────────────────────────────────────────────────────────

PATHS = Namespace(
    name="paths",
    label="Metadata Paths",
    description="Filesystem paths for catalog metadata and image info JSON.",
    settings=[
        SettingSpec("METADATA_PATH", "Catalog metadata JSON path", "string", "data/images/metadata.json"),
        SettingSpec("METADATA_CACHE_TTL", "Metadata cache TTL (seconds)", "int", 86400, min=0, max=604800),
        SettingSpec("IMAGE_INFO_JSON_PATH", "Image info JSON path (empty = disabled)", "string", ""),
    ],
)


# ── 5. Guest limits ───────────────────────────────────────────────────────────

GUEST = Namespace(
    name="guest",
    label="Guest Limits",
    description="Per-session and per-IP usage caps for unauthenticated guests.",
    settings=[
        SettingSpec("GUEST_GEN_LIMIT", "Guest generation limit / session", "int", 3, min=0, max=100),
        SettingSpec("GUEST_CHAT_LIMIT", "Guest chat turn limit / session", "int", 10, min=0, max=200),
        SettingSpec("GUEST_VOICE_LIMIT", "Guest voice turn limit / session", "int", 5, min=0, max=100),
        SettingSpec("GUEST_IP_TOTAL_LIMIT", "Per-IP total limit", "int", 15, min=0, max=500),
    ],
)


# ── 6. Voice ──────────────────────────────────────────────────────────────────

VOICE = Namespace(
    name="voice",
    label="Voice & Live",
    description="Voice persona and Live API tuning (VAD, timeouts, compression).",
    settings=[
        SettingSpec("LIVE_VOICE", "Live voice persona", "choice", "Iapetus",
                    choices=["Iapetus", "Aoede", "Charon", "Fenrir", "Kore", "Puck", "Leda", "Orus", "Zephyr"]),
        SettingSpec("LIVE_VAD_SILENCE_MS", "VAD silence (ms)", "int", 800, min=100, max=5000),
        SettingSpec("LIVE_VAD_PREFIX_MS", "VAD prefix padding (ms)", "int", 100, min=0, max=1000),
        SettingSpec("LIVE_TOOL_CALL_TIMEOUT", "Live tool call timeout (s)", "float", 30.0, min=1, max=120),
        SettingSpec("LIVE_SEND_WAIT_TIMEOUT", "Live send-wait timeout (s)", "float", 8.0, min=1, max=60),
        SettingSpec("LIVE_COMPRESSION_TRIGGER", "Live context compression trigger (tokens)", "int", 100000, min=10000, max=500000),
        SettingSpec("LIVE_COMPRESSION_TARGET", "Live context compression target (tokens)", "int", 32000, min=5000, max=200000),
    ],
)


# ── 7. Watermark ──────────────────────────────────────────────────────────────

WATERMARK = Namespace(
    name="watermark",
    label="Watermark",
    description="Post-processing watermark applied to every generated image (PIL, not AI). "
                "Applied once before storage upload — covers chat, studio, designer, and guest flows.",
    settings=[
        SettingSpec("WATERMARK_ENABLED", "Enable watermark", "bool", "true",
                    "When false, generated images are saved without any watermark. "
                    "Also controllable via ENABLE_WATERMARK in .env (takes precedence when set)."),
        SettingSpec("WATERMARK_OPACITY", "Diagonal watermark opacity (0-1)", "float", 0.4,
                    "Translucency of the large diagonal logo/text mark. 0.35-0.5 is the sweet spot.",
                    min=0.0, max=1.0),
        SettingSpec("WATERMARK_LOGO_PATH", "Logo PNG path (empty = DUCON text fallback)", "string",
                    "app/static/ducon_logo_white.png",
                    "Single-logo override (backward-compat). When set, used for both diagonal and "
                    "badge marks. If the file is missing or unreadable, the watermark falls back to "
                    "the adaptive white/black logos, then to bold 'DUCON' text."),
        SettingSpec("WATERMARK_LOGO_WHITE_PATH", "White logo PNG (used on dark images)", "string",
                    "app/static/ducon_logo_white.png",
                    "White Ducon logo used when the image is dark (mean luminance < 128). "
                    "Adaptive: selected automatically based on image brightness."),
        SettingSpec("WATERMARK_LOGO_BLACK_PATH", "Black logo PNG (used on light images)", "string",
                    "app/static/ducon_logo_black.png",
                    "Black Ducon logo used when the image is light (mean luminance >= 128). "
                    "Adaptive: selected automatically based on image brightness."),
        SettingSpec("WATERMARK_CORNER_PATH", "Corner watermark image path", "string",
                    "app/static/corner watermark.jpeg",
                    "Precomposed top-left badge (logo + contact info). JPEG or PNG."),
    ],
)


# ── 8. Debug ──────────────────────────────────────────────────────────────────

DEBUG = Namespace(
    name="debug",
    label="Debug",
    description="Runtime debug toggles. Disable in production.",
    settings=[
        SettingSpec("LIVE_DEBUG", "Verbose debug logging", "bool", "false",
                    "Enables _dbg() prints across chat / live / designer / multi-image."),
        SettingSpec("CHAT_STREAM", "Stream chat responses", "bool", "true",
                    "Set false to disable SSE streaming for debugging."),
    ],
)


# ── Secrets (read-only, masked, never editable) ───────────────────────────────

SECRETS = Namespace(
    name="secrets",
    label="Secrets (read-only)",
    description="API keys and signing secrets. Masked by default; reveal requires re-auth. Never editable.",
    settings=[
        SettingSpec("GOOGLE_API_KEY", "Google API key", "string", "", is_secret=True, editable=False, use_env_fallback=False),
        SettingSpec("ANTHROPIC_API_KEY", "Anthropic API key", "string", "", is_secret=True, editable=False, use_env_fallback=False),
        SettingSpec("JWT_SECRET_KEY", "JWT signing secret", "string", "", is_secret=True, editable=False, use_env_fallback=False),
        SettingSpec("GOOGLE_CLIENT_ID", "Google OAuth client ID", "string", "", is_secret=True, editable=False, use_env_fallback=False),
        SettingSpec("URL_SIGNING_SECRET", "URL signing secret", "string", "", is_secret=True, editable=False, use_env_fallback=False),
        SettingSpec("TURNSTILE_SECRET_KEY", "Turnstile secret key", "string", "", is_secret=True, editable=False, use_env_fallback=False),
        SettingSpec("GUEST_CLEANUP_SECRET", "Guest cleanup secret", "string", "", is_secret=True, editable=False, use_env_fallback=False),
        SettingSpec("R2_ACCESS_KEY_ID", "R2 access key ID", "string", "", is_secret=True, editable=False, use_env_fallback=False),
        SettingSpec("R2_SECRET_ACCESS_KEY", "R2 secret access key", "string", "", is_secret=True, editable=False, use_env_fallback=False),
        SettingSpec("DATABASE_URL", "Database URL", "string", "", is_secret=True, editable=False, use_env_fallback=False),
        SettingSpec("ADMIN_PASSWORD_HASH", "Admin password hash", "string", "", is_secret=True, editable=False, use_env_fallback=False),
    ],
)


ALL_NAMESPACES: list[Namespace] = [AI_MODELS, THINKING, LIMITS, PATHS, GUEST, VOICE, WATERMARK, DEBUG, SECRETS]
EDITABLE_NAMESPACES: list[Namespace] = [AI_MODELS, THINKING, LIMITS, PATHS, GUEST, VOICE, WATERMARK, DEBUG]

_SPEC_INDEX: dict[str, SettingSpec] = {}
_NS_INDEX: dict[str, Namespace] = {}


def _build_indexes() -> None:
    for ns in ALL_NAMESPACES:
        _NS_INDEX[ns.name] = ns
        for s in ns.settings:
            _SPEC_INDEX[s.key] = s


_build_indexes()


def get_spec(key: str) -> Optional[SettingSpec]:
    return _SPEC_INDEX.get(key)


def get_namespace(name: str) -> Optional[Namespace]:
    return _NS_INDEX.get(name)


def all_keys() -> Iterable[str]:
    return _SPEC_INDEX.keys()


def cast_value(spec: SettingSpec, raw: Any) -> Any:
    """Coerce a raw value to the spec's type with bounds checking. Raises ValueError."""
    if spec.value_type == "bool":
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
        raise ValueError(f"{spec.key}: expected boolean, got {raw!r}")

    if spec.value_type == "int":
        if raw is None or raw == "":
            raise ValueError(f"{spec.key}: expected integer, got {raw!r}")
        try:
            v = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{spec.key}: expected integer, got {raw!r}")
    elif spec.value_type == "float":
        if raw is None or raw == "":
            raise ValueError(f"{spec.key}: expected number, got {raw!r}")
        try:
            v = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{spec.key}: expected number, got {raw!r}")
    elif spec.value_type in ("string", "choice"):
        if raw is None:
            raise ValueError(f"{spec.key}: expected string, got None")
        v = str(raw)
        if spec.value_type == "choice" and spec.choices and v not in spec.choices:
            raise ValueError(f"{spec.key}: {v!r} not in {spec.choices}")
        return v
    else:
        return raw

    if spec.min is not None and v < spec.min:
        raise ValueError(f"{spec.key}: {v} below min {spec.min}")
    if spec.max is not None and v > spec.max:
        raise ValueError(f"{spec.key}: {v} above max {spec.max}")
    return v


def encode_value(spec: SettingSpec, value: Any) -> str:
    """JSON-encode a value for DB storage."""
    import json
    if spec.value_type == "bool":
        return json.dumps(bool(value))
    if spec.value_type in ("int", "float"):
        return json.dumps(value)
    return json.dumps(str(value))


def decode_value(spec: SettingSpec, raw: str) -> Any:
    """Decode a JSON-encoded DB value back to its native type."""
    import json
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = raw
    return cast_value(spec, parsed)
