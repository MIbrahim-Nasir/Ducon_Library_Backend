"""Designer agent session / context configuration for the dev benchmark."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional


# Context policies
# - auto: router-aware defaults (OpenRouter plugin, Claude server compaction, client trim)
# - provider_native: only provider middleware; minimal client trimming
# - compact_tools: shrink tool JSON + strip old images + trim message count
# - trim: drop oldest turns (keep leading user task + recent tail)
# - summarize: LLM summary of middle history when over budget
# - none: no client-side compaction (may fail when context is full)
CONTEXT_POLICIES = (
    "auto",
    "provider_native",
    "compact_tools",
    "trim",
    "summarize",
    "none",
)

SUMMARIZER_MODES = ("default", "custom", "disabled")

CLAUDE_COMPACTION_MODES = ("auto", "enabled", "disabled")

# Routers supported by the dev benchmark (see provider_registry.ROUTERS).
ROUTER_GEMINI = "gemini_native"
ROUTER_CLAUDE = "claude_native"
ROUTER_OPENROUTER = "openrouter"
ALL_ROUTERS = (ROUTER_GEMINI, ROUTER_CLAUDE, ROUTER_OPENROUTER)

# Upper bounds for dashboard inputs (parse_session_config still clamps on submit).
# Sourced from provider docs (2026-07): Anthropic compact_20260112 (min trigger 50k,
# default 150k), Gemini 1M input / 65,536 max output, OpenRouter context-compression
# plugin (boolean enabled per request; middle-out trim when over model window).
SESSION_LIMITS = {
    "max_tokens": {
        "min": 256,
        "max": 128_000,
        "default": 8192,
    },
    "eval_max_tokens": {
        "min": 256,
        "max": 65_536,
        "default": 4096,
    },
    "max_messages": {"min": 4, "max": 500, "default": 60},
    "max_tool_result_chars": {"min": 1000, "max": 500_000, "default": 12_000},
    "retain_recent_image_turns": {"min": 0, "max": 50, "default": 2},
    "context_token_budget": {
        "min": 8_192,
        "max": 1_048_576,
        "default": 128_000,
    },
    "context_trigger_ratio": {"min": 0.1, "max": 0.99, "default": 0.75, "step": 0.05},
    "claude_compaction_trigger_tokens": {
        "min": 50_000,
        "max": 1_000_000,
        "default": 150_000,
    },
    "max_eval_rounds": {"min": 1, "max": 20, "default": 3},
    "max_prompt_verify_rounds": {"min": 0, "max": 10, "default": 2},
}

CONTEXT_POLICY_META: dict[str, dict[str, Any]] = {
    "auto": {
        "label": "auto",
        "description": (
            "Router-aware defaults: OpenRouter compression when enabled, Claude server "
            "compaction when enabled, plus client trim/compact when estimated tokens exceed "
            "the trigger threshold."
        ),
        "routers": list(ALL_ROUTERS),
    },
    "provider_native": {
        "label": "provider_native",
        "description": (
            "Prefer provider middleware only (OpenRouter compression plugin, Claude compaction). "
            "Minimal client trimming unless the budget is exceeded."
        ),
        "routers": list(ALL_ROUTERS),
    },
    "compact_tools": {
        "label": "compact_tools",
        "description": (
            "Shrink tool JSON payloads, strip old tool images, and trim message count when "
            "over budget. Works on all routers; primary fallback for Gemini (no server middleware)."
        ),
        "routers": list(ALL_ROUTERS),
    },
    "trim": {
        "label": "trim",
        "description": (
            "Drop oldest turns (keep opening user task + recent tail) when over budget. "
            "May summarize middle history if summarizer is enabled."
        ),
        "routers": list(ALL_ROUTERS),
    },
    "summarize": {
        "label": "summarize",
        "description": (
            "When over budget, call the designer model to summarize middle history into a "
            "compact block. Especially important on Gemini native where there is no server "
            "compaction plugin."
        ),
        "routers": list(ALL_ROUTERS),
    },
    "none": {
        "label": "none",
        "description": (
            "No client-side compaction. Context may overflow on long runs; use only for short "
            "debug sessions."
        ),
        "routers": list(ALL_ROUTERS),
    },
}

DEFAULT_SUMMARIZER_INSTRUCTIONS = """\
Summarize this Ducon designer agent conversation so work can continue without the full history.

Preserve:
- Client brief and space constraints
- Catalog search queries and selected catalog ids
- Generation indices, prompts used, eval approved/rejected verdicts
- Current plan, open defects, and what still needs doing

Omit:
- Verbose tool JSON, repeated image descriptions, and intermediate reasoning.
Write dense bullet points only.\
"""


@dataclass(frozen=True)
class DesignerSessionConfig:
    """Per-job session limits and context-management policy."""

    max_tokens: int = 8192
    eval_max_tokens: int = 4096
    max_messages: int = 60
    max_tool_result_chars: int = 12_000
    retain_recent_image_turns: int = 2
    context_token_budget: int = 128_000
    context_trigger_ratio: float = 0.75
    context_policy: str = "auto"
    openrouter_context_compression: bool = True
    claude_compaction: str = "auto"
    claude_compaction_trigger_tokens: int = 150_000
    claude_compaction_instructions: Optional[str] = None
    summarizer_mode: str = "default"
    summarizer_instructions: Optional[str] = None
    max_eval_rounds: Optional[int] = None
    max_prompt_verify_rounds: Optional[int] = None

    def effective_summarizer_instructions(self) -> Optional[str]:
        if self.summarizer_mode == "disabled":
            return None
        if self.summarizer_mode == "custom" and (self.summarizer_instructions or "").strip():
            return self.summarizer_instructions.strip()
        if self.summarizer_mode == "default":
            return DEFAULT_SUMMARIZER_INSTRUCTIONS
        return None

    def claude_compaction_enabled(self, *, router: str) -> bool:
        mode = (self.claude_compaction or "auto").strip().lower()
        if mode == "disabled":
            return False
        if mode == "enabled":
            return True
        # auto: enable on native Claude router
        return router == "claude_native"

    def openrouter_compression_enabled(self) -> bool:
        if self.context_policy == "none":
            return False
        if self.context_policy == "provider_native" or self.context_policy == "auto":
            return bool(self.openrouter_context_compression)
        return bool(self.openrouter_context_compression)

    def client_compaction_enabled(self) -> bool:
        return self.context_policy not in {"none", "provider_native"}

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_session_config(raw: Optional[dict[str, Any]] = None) -> DesignerSessionConfig:
    """Build config from job input_meta / form fields with safe defaults."""
    data = raw or {}
    policy = str(data.get("context_policy") or "auto").strip().lower()
    if policy not in CONTEXT_POLICIES:
        policy = "auto"
    summarizer_mode = str(data.get("summarizer_mode") or "default").strip().lower()
    if summarizer_mode not in SUMMARIZER_MODES:
        summarizer_mode = "default"
    claude_compaction = str(data.get("claude_compaction") or "auto").strip().lower()
    if claude_compaction not in CLAUDE_COMPACTION_MODES:
        claude_compaction = "auto"

    def _int(key: str, default: int) -> int:
        val = _optional_int(data.get(key))
        return default if val is None else max(0, val)

    or_compress = data.get("openrouter_context_compression")
    if isinstance(or_compress, str):
        or_compress = or_compress.strip().lower() not in {"0", "false", "no", "off"}

    return DesignerSessionConfig(
        max_tokens=max(256, _int("max_tokens", 8192)),
        eval_max_tokens=max(256, _int("eval_max_tokens", 4096)),
        max_messages=max(4, _int("max_messages", 60)),
        max_tool_result_chars=max(1000, _int("max_tool_result_chars", 12_000)),
        retain_recent_image_turns=max(0, _int("retain_recent_image_turns", 2)),
        context_token_budget=max(8_192, _int("context_token_budget", 128_000)),
        context_trigger_ratio=min(0.99, max(0.1, _optional_float(data.get("context_trigger_ratio"), 0.75))),
        context_policy=policy,
        openrouter_context_compression=bool(or_compress if or_compress is not None else True),
        claude_compaction=claude_compaction,
        claude_compaction_trigger_tokens=max(
            50_000, _int("claude_compaction_trigger_tokens", 150_000)
        ),
        claude_compaction_instructions=(str(data["claude_compaction_instructions"]).strip() or None)
        if data.get("claude_compaction_instructions") else None,
        summarizer_mode=summarizer_mode,
        summarizer_instructions=(str(data["summarizer_instructions"]).strip() or None)
        if data.get("summarizer_instructions") else None,
        max_eval_rounds=_optional_int(data.get("max_eval_rounds")),
        max_prompt_verify_rounds=_optional_int(data.get("max_prompt_verify_rounds")),
    )


def session_config_schema() -> dict[str, Any]:
    """Machine-readable session/context field metadata for the dev dashboard."""
    limits = SESSION_LIMITS
    fields: list[dict[str, Any]] = [
        {
            "key": "max_tokens",
            "label": "Max output tokens (agent turn)",
            "description": (
                "Cap on tokens the designer model may emit per agent turn (tool calls + text). "
                "Gemini native max output is 65,536; Claude Sonnet 4.6 up to 64,000 (Fable/Mythos "
                "up to 128,000). Extended thinking may require raising this above the thinking budget."
            ),
            "type": "integer",
            **limits["max_tokens"],
            "routers": list(ALL_ROUTERS),
        },
        {
            "key": "eval_max_tokens",
            "label": "Eval max tokens",
            "description": (
                "Max tokens for the generation evaluator LLM calls inside the image pipeline "
                "(approve/reject verdict JSON). Gemini models cap at 65,536 output tokens."
            ),
            "type": "integer",
            **limits["eval_max_tokens"],
            "routers": list(ALL_ROUTERS),
        },
        {
            "key": "context_policy",
            "label": "Context policy",
            "description": "How the agent shrinks in-memory message history before each model call.",
            "type": "enum",
            "options": [
                {"value": k, "label": v["label"], "description": v["description"]}
                for k, v in CONTEXT_POLICY_META.items()
            ],
            "default": "auto",
            "routers": list(ALL_ROUTERS),
        },
        {
            "key": "context_token_budget",
            "label": "Context token budget",
            "description": (
                "Estimated input token ceiling for the in-memory session. Client compaction runs "
                "when estimate ≥ budget × trigger ratio. Gemini 2.5/3 models support up to "
                "1,048,576 input tokens; Claude Sonnet 4.6/Opus 4.6 support up to 1M on the API."
            ),
            "type": "integer",
            **limits["context_token_budget"],
            "routers": list(ALL_ROUTERS),
            "hide_when": {"context_policy": ["none"]},
        },
        {
            "key": "context_trigger_ratio",
            "label": "Compact trigger ratio",
            "description": (
                "Fraction of context token budget (0.1–0.99) at which client compaction starts. "
                "Example: 0.75 on 128k budget triggers at ~96k estimated tokens."
            ),
            "type": "number",
            **limits["context_trigger_ratio"],
            "routers": list(ALL_ROUTERS),
            "hide_when": {"context_policy": ["none"]},
        },
        {
            "key": "max_messages",
            "label": "Max messages in history",
            "description": (
                "After compaction, keep the opening user task plus at most this many recent "
                "messages (assistant + tool pairs)."
            ),
            "type": "integer",
            **limits["max_messages"],
            "routers": list(ALL_ROUTERS),
            "hide_when": {"context_policy": ["none", "provider_native"]},
        },
        {
            "key": "max_tool_result_chars",
            "label": "Max tool result chars",
            "description": (
                "Truncate or summarize large tool JSON (catalog hits, generation metadata) "
                "before it is stored in messages[] for the next turn."
            ),
            "type": "integer",
            **limits["max_tool_result_chars"],
            "routers": list(ALL_ROUTERS),
            "hide_when": {"context_policy": ["none", "provider_native"]},
        },
        {
            "key": "retain_recent_image_turns",
            "label": "Retain images (recent tool turns)",
            "description": (
                "How many recent tool turns keep PIL image attachments in context. Older tool "
                "images are stripped to text/JSON only to save tokens."
            ),
            "type": "integer",
            **limits["retain_recent_image_turns"],
            "routers": list(ALL_ROUTERS),
            "hide_when": {"context_policy": ["none"]},
        },
        {
            "key": "summarizer_mode",
            "label": "Summarizer mode",
            "description": (
                "When policy is summarize (or auto on Gemini over budget), replace middle history "
                "with an LLM summary. disabled skips summarization even if policy would allow it."
            ),
            "type": "enum",
            "options": [
                {
                    "value": "default",
                    "label": "default",
                    "description": "Use built-in Ducon designer summary instructions.",
                },
                {
                    "value": "custom",
                    "label": "custom",
                    "description": "Use the custom summarizer instructions textarea below.",
                },
                {
                    "value": "disabled",
                    "label": "disabled",
                    "description": "Never call the summarizer LLM.",
                },
            ],
            "default": "default",
            "routers": list(ALL_ROUTERS),
            "hide_when": {"context_policy": ["none", "provider_native"]},
            "show_when": {"context_policy": ["auto", "summarize", "trim", "compact_tools"]},
        },
        {
            "key": "summarizer_instructions",
            "label": "Custom summarizer instructions",
            "description": "Prompt for the summarizer LLM when summarizer mode is custom.",
            "type": "text",
            "routers": list(ALL_ROUTERS),
            "show_when": {"summarizer_mode": ["custom"]},
        },
        {
            "key": "openrouter_context_compression",
            "label": "OpenRouter context-compression plugin",
            "description": (
                "OpenRouter-only: middle-out message trimming via the context-compression plugin "
                "when the conversation exceeds the model window (enabled: true/false per request; "
                "no extra token knobs). Ignored on Gemini/Claude native routers."
            ),
            "type": "boolean",
            "default": True,
            "routers": [ROUTER_OPENROUTER],
            "hide_when": {"context_policy": ["none"]},
        },
        {
            "key": "claude_compaction",
            "label": "Claude compaction",
            "description": (
                "Claude native only: server-side compact_20260112 beta (header compact-2026-01-12). "
                "auto enables on claude_native; enabled/disabled override. Supported on Claude "
                "Sonnet 4.6 / Opus 4.6 class models."
            ),
            "type": "enum",
            "options": [
                {"value": "auto", "label": "auto", "description": "Enable on Claude native router only."},
                {"value": "enabled", "label": "enabled", "description": "Force server compaction on."},
                {"value": "disabled", "label": "disabled", "description": "Never use Claude compaction beta."},
            ],
            "default": "auto",
            "routers": [ROUTER_CLAUDE],
        },
        {
            "key": "claude_compaction_trigger_tokens",
            "label": "Claude compaction trigger (tokens)",
            "description": (
                "Input token count that triggers Claude server compaction (compact_20260112 beta). "
                "API minimum 50,000; default 150,000. Use up to 1M on 1M-context Claude models."
            ),
            "type": "integer",
            **limits["claude_compaction_trigger_tokens"],
            "routers": [ROUTER_CLAUDE],
            "hide_when": {"claude_compaction": ["disabled"]},
        },
        {
            "key": "claude_compaction_instructions",
            "label": "Claude compaction instructions",
            "description": (
                "Optional override for Anthropic's compaction prompt. Leave empty to use the "
                "provider default."
            ),
            "type": "text",
            "routers": [ROUTER_CLAUDE],
            "hide_when": {"claude_compaction": ["disabled"]},
        },
        {
            "key": "max_eval_rounds",
            "label": "Pipeline max eval rounds",
            "description": (
                "Max generate→eval→retry loops inside generate_multi_image_pipeline (not the "
                "outer agent turn budget)."
            ),
            "type": "integer",
            **limits["max_eval_rounds"],
            "routers": list(ALL_ROUTERS),
            "scope": "pipeline",
        },
        {
            "key": "max_prompt_verify_rounds",
            "label": "Pipeline prompt verify rounds",
            "description": (
                "How many ImageGenAgent prompt verification passes run before image generation "
                "inside the pipeline tool."
            ),
            "type": "integer",
            **limits["max_prompt_verify_rounds"],
            "routers": list(ALL_ROUTERS),
            "scope": "pipeline",
        },
    ]

    applicability: dict[str, list[str]] = {}
    for field in fields:
        key = field["key"]
        applicability[key] = list(field.get("routers") or ALL_ROUTERS)

    return {
        "routers": list(ALL_ROUTERS),
        "limits": limits,
        "context_policies": list(CONTEXT_POLICIES),
        "context_policy_meta": CONTEXT_POLICY_META,
        "summarizer_modes": list(SUMMARIZER_MODES),
        "claude_compaction_modes": list(CLAUDE_COMPACTION_MODES),
        "default_summarizer_instructions": DEFAULT_SUMMARIZER_INSTRUCTIONS,
        "fields": fields,
        "applicability": applicability,
        "notes": [
            "Each designer job builds an in-memory messages[] list (not a persistent DB session).",
            "Every agent turn appends assistant + tool messages; images are re-sent on later turns.",
            "OpenRouter: context-compression plugin trims middle messages when over the model window.",
            "Claude native: compact_20260112 server-side compaction (beta) when enabled.",
            "Gemini generate_content: no server middleware — client trim/compact/summarize only.",
        ],
    }


def session_config_defaults() -> dict[str, Any]:
    """Dashboard metadata defaults."""
    schema = session_config_schema()
    base = DesignerSessionConfig().to_dict()
    base["context_policies"] = schema["context_policies"]
    base["context_policy_meta"] = schema["context_policy_meta"]
    base["summarizer_modes"] = schema["summarizer_modes"]
    base["claude_compaction_modes"] = schema["claude_compaction_modes"]
    base["default_summarizer_instructions"] = schema["default_summarizer_instructions"]
    base["limits"] = schema["limits"]
    base["schema"] = schema
    base["notes"] = schema["notes"]
    return base
