"""Strip inline system prompts from gemini.py after moving them to app/prompts/."""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "app" / "gemini.py"
source = path.read_text(encoding="utf-8")

markers = [
    (
        "# ── System instruction for the prompt generator ──────────────────────────────",
        "# ── System instruction for the pre-generation prompt verifier ─────────────────",
        "# System prompts live in app/prompts/ and are loaded via prompt_loader.\n\n",
    ),
    (
        "# ── System instruction for the pre-generation prompt verifier ─────────────────",
        "# ── System instruction for the generation evaluator ──────────────────────────",
        "",
    ),
    (
        "# ── System instruction for the generation evaluator ──────────────────────────",
        "_EVAL_CRITICAL_SECTIONS = (",
        "",
    ),
    (
        "# ── Quotation analysis ────────────────────────────────────────────────────────",
        "# ── Quotation synthesis instruction ──────────────────────────────────────────",
        "",
    ),
    (
        "# ── Quotation synthesis instruction ──────────────────────────────────────────",
        "async def _single_quotation_call(",
        "",
    ),
]

for start, end, replacement in markers:
    i = source.index(start)
    j = source.index(end, i)
    source = source[:i] + replacement + source[j:]

# Restore load_dotenv() if missing
if "load_dotenv()" not in source:
    source = source.replace(
        "from app import prompt_loader\n",
        "from app import prompt_loader\n\nload_dotenv()\n",
    )

path.write_text(source, encoding="utf-8")
print("updated gemini.py")
