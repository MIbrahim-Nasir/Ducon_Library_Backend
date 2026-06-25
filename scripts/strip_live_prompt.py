"""Strip inline live voice system prompt after moving to app/prompts/live-voice-agent.md."""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "app" / "live_session.py"
source = path.read_text(encoding="utf-8")

start = "# ── Default system instruction ────────────────────────────────────────────────"
end = "# ── Debug printing ─────────────────────────────────────────────────────────────"

insert = '''# ── Default system instruction ────────────────────────────────────────────────
from app import prompt_loader


def get_live_system_instruction() -> str:
    override = os.getenv("LIVE_SYSTEM_INSTRUCTION")
    if override:
        return override.strip()
    prompt_loader.ensure_prompts_loaded()
    return prompt_loader.LIVE_VOICE_AGENT_SYSTEM


'''

i = source.index(start)
j = source.index(end, i)
source = source[:i] + insert + source[j:]

# Update LiveSession default
source = source.replace(
    "self.system_instruction = system_instruction or _DEFAULT_SYSTEM_INSTRUCTION",
    "self.system_instruction = system_instruction or get_live_system_instruction()",
)

path.write_text(source, encoding="utf-8")
print("updated live_session.py")
