"""One-off helper: extract inline triple-quoted prompts from app modules into app/prompts/."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
OUT = APP / "prompts"


def _string_from_node(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "strip" and len(node.args) == 0:
            return _string_from_node(node.func.value)
    return None


def _string_assignments(path: Path) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        print(f"skip AST for {path}: {exc}")
        return {}
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        text = _string_from_node(node.value)
        if text is not None:
            found[target.id] = text
        elif isinstance(node.value, ast.JoinedStr):
            start = node.value.lineno - 1
            end = getattr(node.value, "end_lineno", node.value.lineno)
            lines = source.splitlines()[start:end]
            block = "\n".join(lines)
            m = re.search(r'=\s*f"""(.*?)"""\.strip\(\)', block, re.DOTALL)
            if m:
                body = re.sub(r"\{IMAGE_GEN_MODEL\}", "{{IMAGE_GEN_MODEL}}", m.group(1))
                found[target.id] = body
    return found


def _write(name: str, content: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    gemini = _string_assignments(APP / "gemini.py")
    mapping = {
        "_PROMPT_SYSTEM_INSTRUCTION": "prompt-generator.md",
        "_PROMPT_VERIFY_SYSTEM_INSTRUCTION": "prompt-verifier.md",
        "_EVAL_SYSTEM_INSTRUCTION": "generation-evaluator.md",
        "_QUOTATION_SYSTEM_INSTRUCTION": "quotation-analyzer.md",
        "_QUOTATION_SYNTHESIS_INSTRUCTION": "quotation-synthesis.md",
    }
    for var, filename in mapping.items():
        if var in gemini:
            _write(filename, gemini[var])

    session = _string_assignments(APP / "prompt_generator_session.py")
    if "_DESIGNER_PROMPT_WRITER_SYSTEM" in session:
        _write("designer-prompt-writer.md", session["_DESIGNER_PROMPT_WRITER_SYSTEM"])

    chat = _string_assignments(APP / "chat_agent.py")
    if "CHAT_SYSTEM_INSTRUCTION" in chat:
        # file still has `or """` pattern — read manually below if missing
        pass

    live_source = (APP / "live_session.py").read_text(encoding="utf-8")
    live_match = re.search(
        r'_DEFAULT_SYSTEM_INSTRUCTION = os\.getenv\("LIVE_SYSTEM_INSTRUCTION"\) or \\\s*\n\s*"""(.*?)"""',
        live_source,
        re.DOTALL,
    )
    if live_match:
        _write("live-voice-agent.md", live_match.group(1))

    image_agent_source = (APP / "image_gen_agent.py").read_text(encoding="utf-8")
    image_match = re.search(
        r'_IMAGE_GEN_AGENT_SYSTEM = f"""(.*?)"""\.strip\(\)',
        image_agent_source,
        re.DOTALL,
    )
    if image_match:
        body = re.sub(r"\{IMAGE_GEN_MODEL\}", "{{IMAGE_GEN_MODEL}}", image_match.group(1))
        body = body.replace("{{", "{{").replace("}}", "}}")
        # fix doubled braces from JSON example in prompt
        _write("image-gen-agent.md", body)

    chat_source = (APP / "chat_agent.py").read_text(encoding="utf-8")
    chat_match = re.search(
        r'CHAT_SYSTEM_INSTRUCTION = os\.getenv\("CHAT_SYSTEM_INSTRUCTION"\) or """(.*?)"""\.strip\(\)',
        chat_source,
        re.DOTALL,
    )
    if chat_match:
        _write("chat-agent.md", chat_match.group(1))

    designer_source = (APP / "designer_agent.py").read_text(encoding="utf-8")

    analyze_match = re.search(
        r'async def _analyze_and_plan.*?prompt = f"""(.*?)User request/suggestions:',
        designer_source,
        re.DOTALL,
    )
    if analyze_match:
        _write("designer-analyze-plan.md", analyze_match.group(1).rstrip())

    eval_match = re.search(
        r'eval_prompt = f"""(.*?)Plan: \{json\.dumps\(plan',
        designer_source,
        re.DOTALL,
    )
    if eval_match:
        body = eval_match.group(1).rstrip()
        body = re.sub(
            r"score >= \{DESIGNER_AGENT_PASS_SCORE\}",
            "score >= {{DESIGNER_AGENT_PASS_SCORE}}",
            body,
        )
        _write("designer-evaluate-generation.md", body)

    summary_match = re.search(
        r'async def _summarize_final.*?prompt = f"""(.*?)Plan: \{json\.dumps\(plan',
        designer_source,
        re.DOTALL,
    )
    if summary_match:
        _write("designer-final-summary.md", summary_match.group(1).rstrip())

    compose_match = re.search(
        r'prompt = f"""\n\{base_prompt\}\n\n(.*?)"""',
        designer_source,
        re.DOTALL,
    )
    if compose_match:
        suffix = compose_match.group(1)
        suffix = re.sub(
            r"\{', '\.join\(plan\.get\('preserve'\).*?\)\}",
            "{{PRESERVE_LIST}}",
            suffix,
        )
        suffix = re.sub(
            r"\{', '\.join\(plan\.get\('success_criteria'\).*?\)\}",
            "{{SUCCESS_CRITERIA_LIST}}",
            suffix,
        )
        suffix = re.sub(
            r'\nImprove this attempt based on reviewer feedback: \{previous_feedback\}\n',
            "\n{{RETRY_FEEDBACK_BLOCK}}",
            suffix,
        )
        _write("designer-compose-generation-suffix.md", suffix.strip())


if __name__ == "__main__":
    main()
