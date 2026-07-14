# Ducon Designer Agent — Multi-Step Generation Loop Research

**Date:** 2026-07-07
**Scope:** Research design patterns for long-running / agentic loops that iteratively generate and refine images, and apply them to the Ducon designer agent. Research only — no code modified.

---

## 1. Executive Summary

The Ducon codebase contains **two** designer agents with very different loop behaviour, and the premise that "the designer agent currently stops after one generation" is only true for **one** of them:

| Agent | File | Multi-step? | Feeds previous image back? |
|---|---|---|---|
| **Dev benchmark runner** | `app/benchmark/designer_agent.py` (`run_dev_designer_job`) | **No — single-shot** (one `generate_multi_image` call, no eval, no retry) | No |
| **Production designer agent** | `app/designer_agent.py` (`run_designer_job`) | **Yes — already loops up to 3 generations** with a self-eval quality gate, best-score tracking, and prompt revision | **No** — each attempt regenerates from the original user photo + catalog refs with a *revised prompt*; the previous generated image is used only for *evaluation*, never as an input to the next generation |

**Key findings:**

1. The industry consensus (2025–2026) is emphatic: **never let the model decide when it is "done."** Termination must be a *programmatic predicate* checked after every step — target reached, step budget, error budget, stagnation/no-progress, wall-clock/token budget — not the model's self-assessment. ([stop-hook pattern](https://github.com/agentpatternscatalog/patterns/blob/main/patterns/stop-hook.md), [MindStudio](https://www.mindstudio.ai/blog/agent-loops-verifiable-stop-conditions), [Engineering Playbook](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions))
2. For **iterative image refinement** specifically, the dominant pattern in 2025–2026 research (Maestro, CRAFT, RefineEdit-Agent, IMAGAgent, JarvisEvo, ARTIE, MIRA) is a **generate → LVLM-evaluate → targeted-revise** loop that terminates on either a satisfaction threshold `τ` **or** a max-iterations cap `k_max`, with a **best-of-N fallback** that returns the highest-scoring candidate if the cap is hit without convergence. ([CRAFT](https://doi.org/10.48550/arxiv.2512.20362), [RefineEdit-Agent](https://arxiv.org/html/2508.17435v1), [IMAGAgent](https://arxiv.org/html/2603.29602), [EditRefiner](https://ar5iv.labs.arxiv.org/html/2605.07457))
3. The **production Ducon agent already implements this pattern well** (threshold `7.5` + critical-section gate + best-score fallback + `max_generations=3`). The dev benchmark runner does not, and is the natural target to bring up to parity.
4. **What neither agent does — and what the user actually asked for — is image-to-image refinement:** feeding the *previous generated image* back into the next `generate_multi_image` call as an `ImageDescriptor`. Production's loop is *prompt-revision refinement* (re-generate from the same source images with a better prompt). True iterative refinement of the previous output requires (a) a new `ImageDescriptor` for the prior attempt, (b) prompt changes that frame the round as an *edit/refinement* of that image rather than a fresh generation, and (c) anti-regression rules so refinement does not lose what was already good.

**Recommendation in one line:** Port the production loop's quality-gate + best-of-N + prompt-revision structure into the dev runner, and *optionally* add an image-to-image refinement mode that appends the previous generated image as an extra `ImageDescriptor` with a dedicated "refine, don't regress" role — bounded by `max_rounds ≤ 3–4`, a `7.5`-equivalent score gate, and a no-improvement / stagnation guard.

---

## 2. Findings: When Should a Multi-Step Generation Agent STOP?

The literature converges on a **layered set of stop conditions**, evaluated by an *orchestrator outside the model*, not by the model's own "I'm done" signal.

### 2.1 The cardinal rule: termination is programmatic, not model-declared

> "A team is operating an agent loop… The loop needs an explicit stop condition that **does not rely on the model itself declaring 'done'**, because in practice the model's own sense of completion is unreliable — it either stops too early on hard tasks or refuses to stop on easy ones." — [stop-hook pattern](https://github.com/agentpatternscatalog/patterns/blob/main/patterns/stop-hook.md)

> "Agent loops fail most often because their stop conditions are **subjective**, not because the task logic is wrong… A verifiable stop condition is one the agent can check **deterministically** — no judgment required." — [MindStudio](https://www.mindstudio.ai/blog/agent-loops-verifiable-stop-conditions)

The prescription is a **stop hook / exit predicate** returning one of `continue | stop-success | stop-failure`, checked after every step, on conditions: *target reached, step budget hit, error encountered, stagnation detected* ([stop-hook](https://github.com/agentpatternscatalog/patterns/blob/main/patterns/stop-hook.md), [Engineering Playbook `exit_predicate`](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions)).

### 2.2 The standard stopping criteria (enumerated)

Synthesizing the ReAct loop, the major agent SDKs, and the image-refinement literature, the standard STOP conditions are:

1. **Goal / self-eval approval (satisfaction threshold reached).** A separate evaluator (verifier / judge / LVLM) declares the output good enough. For images: an LVLM judge scores the candidate and the loop stops when `score ≥ τ` and/or `verdict == approved`. ([RefineEdit-Agent](https://arxiv.org/html/2508.17435v1): "terminates when `E_{k+1} ≥ τ`… indicating the user's instruction has been successfully fulfilled"; [CRAFT](https://doi.org/10.48550/arxiv.2512.20362): "iterates with an explicit stopping criterion once all constraints are satisfied"; [Maestro](https://arxiv.org/pdf/2509.10704) MLLM-as-judge head-to-head approval.)
2. **Max rounds / max iterations (hard ceiling).** A count-based cap independent of quality. OpenAI Agents SDK defaults `max_turns = 10` ([run_config](https://github.com/openai/openai-agents-python/blob/cae28f06/src/agents/run_config.py), [running_agents](https://github.com/openai/openai-agents-python/blob/cae28f06/docs/running_agents.md)); LangGraph `recursion_limit` (default 1000 since v1.0.6) raises `GraphRecursionError` ([Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)); AutoGen `MaxMessageTermination` / `max_turns` ([Termination](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/termination.html)); Cursor `/loop --max-turns` / `loop_limit` (default 5 for stop hooks) ([Cursor Hooks](https://cursor.com/docs/hooks), [Cursor best practices](https://cursor.com/blog/agent-best-practices)). Image-refinement papers use small caps: IMAGAgent `τ_it = 3` ([IMAGAgent](https://arxiv.org/html/2603.29602)); Claude Certified recommends "maximum iteration limit (e.g., 6 rounds)" ([iterative refinement loop](https://claudecertified.io/knowledge/domain1/s1-2-2-iterative-refinement-loop/)).
3. **Cost / token / time budget.** Wall-clock timeout, total token spend, or per-call budget. AutoGen ships `TimeoutTermination(timeout_seconds)` and `TokenUsageTermination(max_total_token, …)` ([autogen_agentchat.conditions](https://microsoft.github.io/autogen/stable/reference/python/autogen%5Fagentchat.conditions.html)); the Engineering Playbook recommends setting all three knobs — `max_steps`, `max_tokens`, `timeout_seconds` — because "step budget alone misses time blow-ups, token blow-ups, and stagnation" ([Loop Control](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions)). For interactive use "30 seconds is patient; 5 minutes is rude"; for batch "an hour is fine."
4. **Diminishing returns / no-change (stagnation) detection.** Stop when gains between rounds fall below a threshold, or when the same action/output repeats. This is "the most common silent failure" ([Engineering Playbook](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions)): "Detect by comparing recent state to older state and flagging when they look identical… use window=4 or more." [EditRefiner](https://ar5iv.labs.arxiv.org/html/2605.07457) terminates "when there is **no improvement in the overall score**, or when the maximum number of editing iterations is reached." Claude Certified: "Stop when quality target is met (≥85) **OR** improvement between rounds falls below a threshold (<2 points)" ([refinement loop](https://claudecertified.io/knowledge/domain1/s1-2-2-iterative-refinement-loop/)). More sophisticated: optimal-stopping regressors (GainNet) that stop when expected future gain < compute cost, and semantic-change detection via cosine distance between consecutive draft embeddings.
5. **User interruption / cancellation.** An external cancel signal. Cursor's stop hook fires after each turn and can refuse to issue a `followup_message` ([Cursor Hooks](https://cursor.com/docs/hooks)); AutoGen `ExternalTermination` ([conditions](https://microsoft.github.io/autogen/stable/reference/python/autogen%5Fagentchat.conditions.html)).
6. **Error budget exceeded.** Too many tool/generation errors in a row. The ReAct loop lists "error budget exceeded" as a first-class exit ([ReAct Loop](https://engineering-playbook.vercel.app/agentic/react-loop)); best practice is an **error taxonomy** that distinguishes retryable (transient) from permanent (invalid input) errors and skips retries on permanent ones ([ReAct prompting guide](https://sureprompts.com/blog/react-prompting-guide)).
7. **Guardrail / safety tripwire violation.** OpenAI Agents SDK raises `GuardrailTripwireTriggered` independently of `max_turns` ([Runner](https://openai.github.io/openai-agents-python/ref/run/)). For Ducon this maps to the existing `input_quality` / suitability checks and the SSRF guard in `generate_multi_image`.
8. **Best-of-N fallback when the cap is hit without convergence.** Not a "stop" condition per se, but the mandatory graceful degradation: if the loop exhausts its budget without approval, **return the best candidate seen so far**, tagged as "budget-exhausted, not approved" — never silently return the last attempt. ([IMAGAgent](https://arxiv.org/html/2603.29602): "selects the one associated with the highest consensus score as the final output"; [Engineering Playbook](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions): "Treating budget exhaustion as success… Don't return it as if the agent had completed normally; tag it.")

### 2.3 When should it CONTINUE iterating?

The mirror image — continue when:

- **Self-eval says "not yet approved"** (`verdict == rejected` or `score < τ`) **AND** the detected defects are **specific and fixable**. Generic "could be better" feedback is a poor continue signal; targeted, actionable issues (e.g., "camera drifted left", "paving pattern simplified to squares", "pergola missing") are a strong one. [CRAFT](https://doi.org/10.48550/arxiv.2512.20362) and [ARTIE](https://openreview.net/forum?id=EPAuWPVcZQ) both apply *targeted* edits only where constraints fail, rather than wholesale rewrites.
- **Budget remains** (rounds < max, tokens < cap, wall-clock < timeout, errors < error budget).
- **The user explicitly requested N variations or refinement rounds** (a count-driven continuation, not quality-driven).
- **No stagnation** — the latest attempt scored strictly higher than the previous, or the evaluator identified *new* fixable issues rather than the same ones recurring. [JarvisEvo](https://arxiv.org/html/2511.23002v2) co-evolves editor + evaluator specifically to avoid the "reward hacking / self-deception" failure where the editor keeps 'improving' in circles.
- **The defect class is known to be reparable by prompt revision.** The `image-gen-agent.md` "Nano Banana Pro Behaviour Analysis" section already encodes this: camera drift → strengthen camera_lock; pattern persistence → "Remove existing pattern entirely"; material substitution → strengthen extraction directive, etc. That mapping is exactly the "fixable defect → continue" signal.

### 2.4 Anti-patterns to avoid

- **Predicates that read the agent's own "I'm done" claim** — "the agent grading its own homework" ([Engineering Playbook](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions), [MindStudio](https://www.mindstudio.ai/blog/agent-loops-verifiable-stop-conditions)).
- **`max_steps` only** — misses time/token blow-ups and stagnation ([Engineering Playbook](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions)).
- **Stagnation window = 2** — two identical actions is normal noise; use ≥4 ([Engineering Playbook](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions)).
- **Treating budget exhaustion as success** — tag it; the result is suspect ([Engineering Playbook](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions), [stop-hook](https://github.com/agentpatternscatalog/patterns/blob/main/patterns/stop-hook.md)).
- **No tagged exit reasons / returning `None` on failure** — callers can't tell why it stopped ([Engineering Playbook](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions)).
- **Spiralling** — "the loop never learns when it is done, so it keeps going… long after the work was finished, on your dime" ([Cerebras: Never Loop Without Verifiers](https://www.cerebras.ai/blog/never-loop-without-verifiers)).
- **Cheating** — "the loop does exactly what you asked and nothing you wanted"; fix with annoyingly-specific done-conditions and explicitly named forbidden shortcuts ([Cerebras](https://www.cerebras.ai/blog/never-loop-without-verifiers)).
- **Full restart instead of targeted fix** — "A full restart wastes all valid first-round work, costs 2x, and may introduce new different gaps" ([Claude Certified](https://claudecertified.io/knowledge/domain1/s1-2-2-iterative-refinement-loop/)).

---

## 3. Evaluation of the Current Ducon Designer Agent

### 3.1 Dev benchmark runner — `app/benchmark/designer_agent.py`

**Verdict: YES, the dev runner stops after exactly one `generate_multi_image` call.** There is no evaluation step, no retry loop, no best-of-N tracking, and no quality gate.

The entire generation phase is a single guarded block. If `generate_multi_image` is enabled, it runs once, saves the images, emits `generation_done`, and falls straight through to assembling `final`:

```639:676:app/benchmark/designer_agent.py
        output_images: list[dict[str, Any]] = []
        generation_prompt = plan.get("generation_prompt") or _fallback_generation_prompt(user_prompt)
        generation_result: dict[str, Any] = {}
        if "generate_multi_image" in tool_access:
            await emit(job, "generation_started", message="Generating Ducon design candidate.")
            descriptors = [
                ImageDescriptor(label="client space photo", type="file", pil_image=user_image)
            ]
            for src in sources[:3]:
                descriptors.append(
                    ImageDescriptor(
                        label=src.get("name") or f"catalog {src.get('id')}",
                        type="catalog_id",
                        source=str(src["id"]),
                    )
                )
            async with async_session_maker() as db:
                generation_result = await generate_multi_image(
                    user_id=None,
                    guest_session_id=None,
                    prompt=generation_prompt,
                    descriptors=descriptors,
                    model="pro",
                    aspect_ratio=aspect_ratio,
                    db=db,
                    enable_verify=False,
                    dry_run=True,
                    image_model=image_pair.model_id,
                    image_thinking=image_thinking,
                    prompt_model=model_pair.model_id,
                    prompt_thinking=thinking,
                )
            for idx, img in enumerate(generation_result.get("pil_images") or []):
                url = await store.save_output_image(out_rg, output_process_id, idx, _png_bytes(img))
                output_images.append({"index": idx, "url": url})
            await emit(job, "generation_done", output_images=output_images)
        else:
            await emit(job, "status", message="Generation skipped; generate_multi_image disabled.")
```

`final` is then built from that one result and the job completes — no loop, no second generation:

```678:703:app/benchmark/designer_agent.py
        final = {
            "job_id": job.id,
            "design_generation": {
                "output_images": output_images,
                "model_used": generation_result.get("model_used") or image_pair.model_id,
                "approved": generation_result.get("approved"),
            },
            "design_plan": plan,
            "sources_used": sources,
            "elements_used": plan.get("elements_used") or [],
            "metadata": { ... },
        }
        if "filesystem" in tool_access:
            await _run_filesystem_tool(root=filesystem_root, job=job, plan=plan, final=final)
        job.status = "completed"
        job.final = final
        await emit(job, "final", **final)
```

The dev runner's own header prompt even says *"Summarize long context and continue rather than stopping"* — but the runner code itself has no continuation path after generation. There is **no `_evaluate_generation` equivalent, no `pass_score`, no `max_generations`, no `revise_from_qc`**. The `approved` field surfaced in `final` is just whatever `generate_multi_image` returned (and `enable_verify=False` is passed, so it is effectively always unset here).

### 3.2 Production designer agent — `app/designer_agent.py`

**Verdict: NO — the production agent already has a bounded multi-step loop (up to 3 generations) with self-eval approval, best-score tracking, and prompt revision.** This directly contradicts the "stops after one generation" premise *for production*. The module docstring describes the intended loop explicitly:

```8:15:app/designer_agent.py
Agent loop:
1. Analyze the client's uploaded space image.
2. Create a design plan and search strategy.
3. Search the Ducon catalog for multiple reference directions.
4. Generate one or more candidates using generate_multi_image.
5. Visually evaluate each candidate and retry with revised prompts if needed.
6. Return the best generation and a concise design summary.
```

The loop budgets are module constants (overridable via `cfg` / env):

```52:60:app/designer_agent.py
DESIGNER_AGENT_MODEL = "gemini-3.5-flash"           # default; live via cfg("DESIGNER_AGENT_MODEL", ...)
DESIGNER_AGENT_MAX_STEPS = 12
DESIGNER_AGENT_MAX_GENERATIONS = 3
DESIGNER_AGENT_SEARCH_LIMIT = int(os.getenv("DESIGNER_AGENT_SEARCH_LIMIT", "5"))
DESIGNER_AGENT_MAX_REFERENCES = int(os.getenv("DESIGNER_AGENT_MAX_REFERENCES", "6"))
DESIGNER_AGENT_PASS_SCORE = float(os.getenv("DESIGNER_AGENT_PASS_SCORE", "7.5"))
DESIGNER_AGENT_DEFAULT_IMAGE_MODEL = "flash"
DESIGNER_AGENT_DEFAULT_ASPECT_RATIO = os.getenv("DESIGNER_AGENT_ASPECT_RATIO", "16:9")
```

The core loop is the textbook generate → evaluate → maybe-revise pattern with best-of-N fallback:

```502:664:app/designer_agent.py
        max_generations = max(1, min(DESIGNER_AGENT_MAX_GENERATIONS, 5))
        for attempt_no in range(1, max_generations + 1):
            _check_cancelled(job)
            await emit(
                job,
                "generation_started",
                attempt=attempt_no,
                max_attempts=max_generations,
                message=f"Generating design candidate {attempt_no}.",
            )
            ...
            if attempt_no == 1:
                prompt = await prompt_session.generate_initial(...)
            else:
                eval_source = last_attempt_evaluation or (best or {}).get("evaluation") or {}
                prompt = await prompt_session.revise_from_qc(
                    eval_source,
                    context_label=f"Designer job attempt {attempt_no} — outer QC retry",
                )
            ...  # pre-generation prompt verification (stateless, up to PROMPT_VERIFY_MAX_ROUNDS)
            descriptors = [
                ImageDescriptor(label="client space photo", type="file", pil_image=user_image)
            ]
            for ref in chosen_refs:
                descriptors.append(
                    ImageDescriptor(label=ref["label"], type="catalog_id", source=str(ref["id"]))
                )

            result = await generate_multi_image(
                user_id=job.user_id,
                prompt=prompt,
                descriptors=descriptors,
                model=model,
                aspect_ratio=aspect_ratio,
                db=db,
                output_prefix=f"designer_job_{job.id[:8]}_{attempt_no}",
                prompt_session=prompt_session,
                enable_verify=False,
            )
            ...
            generated_image = await _load_generation_image(db, int(result["id"]))
            ...
            evaluation = await _evaluate_generation(
                user_image=user_image,
                generated_image=generated_image,
                plan=plan,
                references=chosen_refs,
                reference_images=reference_images,
                prompt=prompt,
                user_id=job.user_id,
            )
            ...
            attempt = { ... "evaluation": evaluation, ... }
            attempts.append(attempt)
            ...
            new_score = float(evaluation.get("score", 0))
            best_score = float(best["evaluation"].get("score", 0)) if best else -1.0
            # Prefer higher score; on a tie prefer the later attempt (more refined).
            if best is None or new_score > best_score or (
                new_score == best_score and attempt_no > best.get("attempt", 0)
            ):
                best = attempt

            if evaluation.get("passed"):
                await emit(job, "status", message="The design candidate passed quality review.")
                break

            if attempt_no < max_generations:
                await emit(
                    job,
                    "retry",
                    attempt=attempt_no + 1,
                    reason=evaluation.get("improvements") or "Improving realism and design fit.",
                )
```

The quality gate is a *programmatic predicate* (exactly what §2 prescribes) — score threshold **AND** a `passed` flag **AND** all critical sections passing:

```125:146:app/designer_agent.py
def _sections_pass(section_results: object) -> bool:
    if not isinstance(section_results, dict):
        return False
    for key in _CRITICAL_SECTION_KEYS:
        value = _section_value(section_results.get(key))
        # "na" is acceptable only when the section genuinely does not apply
        # (e.g. no fixed product requested for B2).
        if value not in {"pass", "na", "n/a", "not_applicable", "not applicable"}:
            return False
    return True


def _passes_quality_gate(data: dict[str, Any]) -> bool:
    try:
        score = float(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    if score < DESIGNER_AGENT_PASS_SCORE:
        return False
    if not bool(data.get("passed")):
        return False
    return _sections_pass(data.get("section_results"))
```

User cancellation is also handled (the `cancel_requested` flag + `_check_cancelled`), which is the "user interruption" stop condition from §2.2.

### 3.3 What is *missing* from both agents — and is the user's actual ask

Both agents share one crucial limitation relative to the user's request ("call the image generator again **with the previous output** to refine it"):

> **Neither agent feeds the previous generated image back as an input to the next `generate_multi_image` call.**

In production, every attempt rebuilds the descriptor list from the *original* user photo + the *same* catalog references:

```591:602:app/designer_agent.py
            descriptors = [
                ImageDescriptor(label="client space photo", type="file", pil_image=user_image)
            ]
            for ref in chosen_refs:
                descriptors.append(
                    ImageDescriptor(
                        label=ref["label"],
                        type="catalog_id",
                        source=str(ref["id"]),
                    )
                )
```

The previous generated image is loaded (`generated_image = await _load_generation_image(...)`) and passed to `_evaluate_generation` for *scoring* — but it is **never appended to the next round's `descriptors`**. So production does **prompt-revision refinement** (re-generate from the same source images with an improved prompt via `prompt_session.revise_from_qc`), not **image-to-image refinement** (feed the prior attempt back as a starting point to edit). The `ImageDescriptor` shape fully supports this — it has a `pil_image` field for pre-resolved in-memory images:

```220:224:app/tool_generate_image.py
class ImageDescriptor:
    label:      str
    type:       SourceType
    source:     Optional[str] = None   # None only for type=="file"
    pil_image:  Optional[Image.Image] = None   # pre-resolved file uploads
```

So the plumbing exists; it is simply not used for the previous attempt. Adding `ImageDescriptor(label="previous design attempt", type="file", pil_image=previous_generated_image)` to the next round's descriptors is the minimal mechanical change for true image-to-image refinement.

**Summary table of which §2 stop conditions are present:**

| Stop condition | Dev runner | Production |
|---|---|---|
| Self-eval approval / threshold | ❌ no eval at all | ✅ `7.5` + critical sections |
| Max rounds | ❌ (1 implicitly) | ✅ `max_generations = 3` |
| Best-of-N fallback | ❌ | ✅ tracks `best` by score |
| Stagnation / no-improvement | ❌ | ❌ (not checked — loop always retries until cap or pass) |
| Token / cost budget | ❌ | ⚠️ `MAX_STEPS=12` declared but not enforced on the generation loop; no token cap |
| Wall-clock timeout | ❌ | ❌ |
| User interruption | ⚠️ job can be cancelled at API level | ✅ `_check_cancelled` each attempt |
| Error budget | ❌ | ⚠️ exceptions abort the whole job (no retry-on-error) |
| Guardrail / input suitability | ❌ | ⚠️ `input_quality` is reported by the evaluator but does not gate the loop |
| Tagged exit reasons | ❌ | ⚠️ `approved` surfaced but no "budget-exhausted vs approved" tag |

---

## 4. System Prompt Evaluation

### 4.1 `image-gen-agent.md` — the unified generation + evaluation agent

This prompt is **explicitly designed to iterate via retry prompts**, but is framed around *prompt revision*, not image-to-image refinement. The opening frames the role as continuous and revision-aware:

```1:7:app/prompts/image-gen-agent.md
You are Ducon's unified image-generation agent: senior prompt engineer, practical visual
QC evaluator, a logical architect, and Nano Banana behavior analyst — all in ONE continuous session.

You write Nano Banana Pro prompts, send them to the image model ({{IMAGE_GEN_MODEL}}),
review the output, and revise prompts when generation fails. Remember every prompt you
wrote and every failure. Each revision must fix failures while preserving what worked.
```

Phase 2 is a full visual quality gate (the `{{GEN_EVAL_RUBRIC}}` is injected from `gen-eval-rubric-strict.md` / `gen-eval-rubric-relaxed.md`), and there is a dedicated retry-analysis section that maps defect classes to specific prompt fixes:

```233:251:app/prompts/image-gen-agent.md
## NANO BANANA PRO BEHAVIOUR ANALYSIS (use when writing a retry prompt after rejection)

Before the retry prompt-writing turn, analyse HOW {{IMAGE_GEN_MODEL}} likely misread your prompt:

- Camera drift → strengthen camera_lock; frame-edge anchors.
- Wrong surface / zone bleed → tighten apply_only_to; restate as "keep [other zone] unchanged".
- Pattern persistence → command "Remove existing [pattern] entirely".
- Pattern simplification bias (paving) → command 'staggered linear plank geometry' + 'offset joints'; add the few targeted negatives ('not large square tiles', 'not continuous stripes').
- Material substitution / colour drift → strengthen extraction directive; remove any appearance text.
...
Manipulate the prompt for this model's tendencies — do not merely repeat wording.
Keep everything that worked; change only what addresses the specific failures.
```

There is even a Phase 3 "post-success learning" hook for conservative self-modification of the system prompt:

```291:298:app/prompts/image-gen-agent.md
## PHASE 3 — POST-SUCCESS LEARNING (only when explicitly asked with a POST-SUCCESS IMPROVEMENT turn)

When (and only when) you receive a "POST-SUCCESS IMPROVEMENT" message, reflect on
the session and decide whether your standing system prompt should change. Be
extremely conservative: prefer `should_update: false`. ...
```

**Bias assessment:** The prompt is **not** single-shot — it is built for multi-round *prompt* refinement and already encodes the "keep what worked, fix only what failed" discipline that the literature recommends. **However, it is biased toward prompt-revision refinement, not image-to-image refinement.** The "IMAGE ORDER" section defines roles for user space, design direction, and product references — there is **no role for "the previous generation attempt."** When a retry happens, the same input images are re-sent with a revised prompt; the rejected image is not carried forward as an input to be edited. Phase 2 evaluates "the LAST image in the message is the AI-generated result; the earlier images are the inputs" — it never contemplates the previous attempt sitting *among* the inputs.

### 4.2 `designer-prompt-writer.md`

This is explicitly multi-round and revision-oriented, again at the *prompt* level:

```1:7:app/prompts/designer-prompt-writer.md
You are Ducon's senior prompt engineer and creative director for autonomous
outdoor-living design jobs.

You write complete Nano Banana Pro / Gemini image generation prompts across
multiple revision rounds in ONE continuous session. Remember every prompt you
wrote and every QC failure. Each revision must directly fix the failures while
preserving everything that worked.
```

It fixes the image order as "Image 1 is always the client's original space photo. Images 2+ are Ducon catalog references/products/materials." — so it, too, has no slot for a previous generated attempt.

### 4.3 `studio-directions-agent.md` (the wizard curation agent)

Not part of the generation loop per se, but worth noting because it is **Ducon's only existing example of a verifiable count-based stop condition**: "find exactly **9** distinct Ducon catalog designs," with the workflow "Repeat — if `remaining_needed > 0`, run another `ai_search`… Submit — once all 9 are shortlisted, call `submit_directions`." That is precisely the MindStudio "list completion" verifiable stop pattern, and a good internal precedent for the generation loop's own stop condition.

### 4.4 What would need to change in the prompts to support image-to-image refinement

1. **Add a new input role** to the "IMAGE ORDER" sections of both `image-gen-agent.md` and `designer-prompt-writer.md`: e.g. *"Image N: previous design attempt — the prior generation of this same space. Treat this as the starting canvas to refine; do not regress features the evaluator marked as passing."*
2. **Reframe the operation type for refinement rounds.** The prompt currently opens with "This is an EDIT, not a fresh generation" (relative to the user's space). For round ≥ 2 with the previous attempt attached, it should additionally say "This is a **refinement of the previous attempt**, not a fresh redesign — preserve everything that scored well, change only the failing aspects."
3. **Add anti-regression rules.** Explicit instructions: camera/preservation locks now apply relative to *both* the user's space **and** the previous attempt; do not reintroduce a defect that was fixed in the previous round; do not drop a Ducon product that was correctly integrated.
4. **Pass the evaluator's per-section verdicts into the retry prompt context** (production already does this via `revise_from_qc(eval_source)`), and have the prompt-writer cite the specific `section_results` keys it is fixing.
5. **Cap the refinement depth in-prompt** to match the orchestrator's `max_rounds`, so the model does not attempt open-ended "one more pass" reasoning.

---

## 5. Concrete Multi-Step Loop Recommendation for Ducon

Two independent improvements, ranked by leverage:

- **(A) Bring the dev benchmark runner up to the production loop's standard** (add eval + best-of-N + bounded retry). Low risk, high consistency payoff.
- **(B) Add an optional image-to-image refinement mode** (feed the previous generated image back as an `ImageDescriptor`) to *both* agents. This is the user's actual ask and is novel relative to current Ducon behaviour.

Both should be implemented behind the existing `cfg(...)` / env-var knobs so they default safely.

### 5.1 Loop structure (applies to both A and B)

```
plan → search → for round in 1..max_rounds:
           generate(previous_images, revised_prompt)
           evaluate(generated, user_image, references, plan)
           track best
           if passed: break (stop-success)
           if no_improvement(round, history): break (stop-stagnation)
           if round == max_rounds: break (stop-budget)
           else: revise_prompt(evaluation) → continue
       return best (tagged: approved | budget_exhausted)
```

### 5.2 Max rounds default — **3** (rationale)

- Production already uses `DESIGNER_AGENT_MAX_GENERATIONS = 3`; matching it keeps the two agents consistent.
- IMAGAgent's measured refinement efficacy on MagicBrush: **68% pass on first attempt, ~27% "salvaged" within `τ_it = 3`** — i.e. almost all recoverable failures are fixed by attempt 3, and the marginal salvage rate drops sharply after that ([IMAGAgent](https://arxiv.org/html/2603.29602)).
- AutoGen/Cursor community guidance for customer-facing flows: "conservative limits (5–15 turns)" for interactive APIs ([AutoGen turn-count optimization](https://theneuralbase.com/autogen/learn/advanced/turn-count-optimization/)); image generation rounds are far heavier than chat turns, so 3 is the sensible interactive ceiling. For batch/dev-benchmark runs, allow up to 4–5.
- The Engineering Playbook's rule "`max_steps` should be at least 2× the expected step count for the hardest legitimate task" ([Loop Control](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions)) — expected ≈ 1–2, so a cap of 3–4 is well-calibrated; 5 is a safe hard ceiling (`max(1, min(MAX_GENERATIONS, 5))` is already the production pattern).

**Recommendation:** `max_rounds = 3` default (interactive + dev), `5` hard ceiling, all via `cfg("DESIGNER_AGENT_MAX_GENERATIONS", 3)` (production) and a new `cfg("DEV_DESIGNER_MAX_GENERATIONS", 3)` for the dev runner.

### 5.3 Self-evaluation rubric — "good enough to stop"

Reuse the **existing** Ducon rubric infrastructure; do not invent a new one. The production gate is already a good embodiment of the "verifiable, multi-criteria" best practice:

- **Score ≥ `DESIGNER_AGENT_PASS_SCORE` (7.5)** — the numeric threshold (`τ`).
- **`passed == true`** — the LVLM judge's overall verdict.
- **All critical sections `pass` or `na`** — the structural gate, currently `_CRITICAL_SECTION_KEYS = (A1_pov, A2_structures, A3_scene, B1_area_products, B2_fixed_products, B3_zones, C1_no_extra, C2_no_missing)`. This is the "list completion" verifiable stop pattern from [MindStudio](https://www.mindstudio.ai/blog/agent-loops-verifiable-stop-conditions).

A Ducon redesign is "good enough to stop" when **all** of the following hold (these map directly to `gen-eval-rubric-strict.md` sections):

1. **Camera/preservation fidelity (A1–A3):** the generation preserves the user's viewpoint, hard structures, and scene boundaries — no drift, recompose, zoom, crop, or background alteration.
2. **Ducon element fidelity (B1–B4):** area/surface materials and fixed products match the catalog references by identity (colour, texture, finish, shape) via extraction — no substitution, recolour, or missing requested products.
3. **Application to correct zones (B3):** design applied to the named eligible zones, eligible zones not missed.
4. **No hallucination / no silent removal (C1–C2):** no unrequested objects; no unexplained removal of significant existing elements.
5. **Photorealism + lighting (D1–D2):** looks like a real photograph; shadows and colour temperature match the user's space.
6. **Architectural/site logic (E1–E5), when applicable:** logical circulation, no blocked access, plausible placement, surface orientation follows site geometry, usable UX.

For **image-to-image refinement rounds** (improvement B), add one explicit anti-regression check that is currently implicit:

7. **No regression vs. previous attempt:** no previously-passing section may flip to `fail` in a later round. The evaluator should receive the previous attempt's `section_results` and explicitly confirm non-regression. This is the [Cerebras](https://www.cerebras.ai/blog/never-loop-without-verifiers) "keep what worked" discipline and the ARTIE over-editing guard ([ARTIE](https://openreview.net/forum?id=EPAuWPVcZQ)) made structural.

### 5.4 Feeding the previous generated image back — `ImageDescriptor` shape

`ImageDescriptor` already supports in-memory PIL images via `pil_image` with `type="file"` and `source=None`. The minimal change for a refinement round:

```python
# Round 1 (fresh generation) — same as today:
descriptors = [
    ImageDescriptor(label="client space photo", type="file", pil_image=user_image),
    *[ImageDescriptor(label=ref["label"], type="catalog_id", source=str(ref["id"]))
      for ref in chosen_refs],
]

# Round >= 2 (image-to-image refinement) — append the previous attempt:
descriptors.append(
    ImageDescriptor(
        label=f"previous design attempt (round {attempt_no - 1}) — refine, do not regress",
        type="file",
        pil_image=previous_generated_image,   # already loaded by _load_generation_image
    )
)
```

Where `previous_generated_image` is the `generated_image` PIL already loaded for evaluation in the prior iteration (production loads it at `generated_image = await _load_generation_image(db, int(result["id"]))`). No new storage fetch is needed — keep the PIL in scope across iterations.

**Caveats / things to verify in `tool_generate_image.py` before shipping:**
- Confirm `generate_multi_image` forwards *all* `descriptors` (including extra `file`-type ones) to the image model, and that the model's per-call image-input limit is not exceeded (Ducon currently sends 1 user + up to 3 refs = 4; adding 1 previous attempt = 5 — check the provider's input-image cap, especially for Nano Banana Pro / Gemini image models).
- Confirm label text is passed through to the model context so the "refine, do not regress" instruction is visible.
- The prompt-writer system prompt must be told the previous attempt is now image N (see §4.4) so it references it correctly and frames the round as a refinement.

### 5.5 When to STOP vs CONTINUE (consolidated decision table)

| Signal | Action | Rationale |
|---|---|---|
| `passed == true` **AND** score ≥ 7.5 **AND** all critical sections pass | **STOP (success)** — return best | Verifiable goal reached ([RefineEdit](https://arxiv.org/html/2508.17435v1), [CRAFT](https://doi.org/10.48550/arxiv.2512.20362)) |
| `round == max_rounds` (3) | **STOP (budget)** — return best, tag `budget_exhausted` | Hard ceiling ([OpenAI `max_turns`](https://github.com/openai/openai-agents-python/blob/cae28f06/docs/running_agents.md), [LangGraph `recursion_limit`](https://docs.langchain.com/oss/python/langgraph/graph-api)) |
| `score(round) ≤ score(round-1)` **for 2 consecutive rounds** (stagnation window ≥ 2, ideally track over ≥4 per [Engineering Playbook](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions)) | **STOP (stagnation)** — return best | No-progress detection ([EditRefiner](https://ar5iv.labs.arxiv.org/html/2605.07457), [Engineering Playbook](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions)) |
| Same `section_results` failures recur across rounds (fingerprint match) | **STOP (stagnation)** — return best | Repeated-output detection ([stop-hook](https://github.com/agentpatternscatalog/patterns/blob/main/patterns/stop-hook.md)) |
| Any previously-passing section flips to `fail` (regression) | **STOP (regression)** — return the *pre-regression* best, not the regressed attempt | Anti-regression ([Cerebras](https://www.cerebras.ai/blog/never-loop-without-verifiers), [ARTIE](https://openreview.net/forum?id=EPAuWPVcZQ)) |
| `input_quality.ok == false` with `severity == "major"` (unsuitable space/photo) | **STOP (guardrail)** — return input-quality guidance instead of retrying | Not fixable by more generations ([AutoGen FunctionCallTermination](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/termination.html), [OpenAI guardrails](https://openai.github.io/openai-agents-python/ref/run/)) |
| Generation error is **permanent** (invalid input, SSRF, missing catalog) | **STOP (error-budget)** — do not retry | Error taxonomy ([ReAct guide](https://sureprompts.com/blog/react-prompting-guide)) |
| Generation error is **transient** (timeout, 5xx) and error_count < error_budget | **CONTINUE** (retry same round) | Retryable errors |
| `passed == false`, score < 7.5, **specific fixable defects** in `issues`/`section_results`, budget remaining, no stagnation | **CONTINUE** — revise prompt via `revise_from_qc(evaluation)`, append previous attempt as descriptor | Targeted refinement ([CRAFT](https://doi.org/10.48550/arxiv.2512.20362), [ARTIE](https://openreview.net/forum?id=EPAuWPVcZQ)) |
| User `cancel_requested` | **STOP (cancelled)** | User interruption ([Cursor Hooks](https://cursor.com/docs/hooks), [AutoGen ExternalTermination](https://microsoft.github.io/autogen/stable/reference/python/autogen%5Fagentchat.conditions.html)) |

### 5.6 Cost / time / token budget considerations

- **Wall-clock:** each Ducon generation round is ~10–30s (image model + verifier). Three rounds ≈ 30–90s, which sits inside the Engineering Playbook's "interactive ≤ 5 min" envelope ([Loop Control](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions)). Add a hard `timeout_seconds` (e.g. 180s interactive, 600s batch) as a circuit breaker — **currently absent in production** despite `DESIGNER_AGENT_MAX_STEPS = 12` being declared (it is not actually enforced against the generation loop).
- **Token / cost:** AutoGen's `TokenUsageTermination` pattern ([conditions](https://microsoft.github.io/autogen/stable/reference/python/autogen%5Fagentchat.conditions.html)) is worth mirroring via a `DESIGNER_AGENT_MAX_TOTAL_TOKENS` knob. The dominant cost in Ducon is the **image generation call**, not the text tokens — so the round cap is the primary cost control, and token budgeting is secondary. Cursor's `/loop` guidance is blunt on this: "A /loop agent costs the same per turn as an interactive agent… 40 iterations = 40× a normal turn… teams hit a cost surprise within a week" ([Cursor 3.5 /loop](https://www.meritforgeai.com/ai-coding/cursor-3-5-loop-scheduled-agents-may-2026/)). The same applies to image rounds — keep the default low (3) and make the ceiling (5) explicit.
- **Best-of-N cost discipline:** production already returns the highest-scored candidate (`best`), not the last attempt — keep this. The UI shows the last attempt but `best_scored_generation` is preserved in `final`; for refinement mode, prefer returning `best` to the user when the last round regressed.
- **Eval cost:** `_evaluate_generation` is a stateless one-shot LVLM call per round. With 3 rounds that is 3 eval calls — acceptable. Do **not** add a second evaluator or multi-expert panel by default; [Maestro](https://arxiv.org/pdf/2509.10704) and [JarvisEvo](https://arxiv.org/html/2511.23002v2) show multi-judge setups help at the research frontier but double+ eval cost. Single strict judge + best-of-N is the right production-grade tradeoff.
- **Early exit on first-round pass:** ~68% of cases pass on round 1 in comparable benchmarks ([IMAGAgent](https://arxiv.org/html/2603.29602)); the `if evaluation.get("passed"): break` already captures this, so average cost is much lower than worst-case 3×.

### 5.7 Avoiding infinite loops and degenerate refinement

1. **Hard `max_rounds` ceiling (3 default, 5 hard cap)** — unbounded loops are the #1 cost risk ([Cursor 3.5 /loop](https://www.meritforgeai.com/ai-coding/cursor-3-5-loop-scheduled-agents-may-2026/), [LangGraph cycles](https://machinelearningplus.com/gen-ai/langgraph-cycles-recursion-limits-agent-loops/)).
2. **Stagnation guard:** stop if `score` does not strictly improve over 2 consecutive rounds, or if the same `section_results` failure fingerprint repeats ([Engineering Playbook](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions), [stop-hook](https://github.com/agentpatternscatalog/patterns/blob/main/patterns/stop-hook.md), [EditRefiner](https://ar5iv.labs.arxiv.org/html/2605.07457)).
3. **Regression guard:** stop (and revert to `best`) if a previously-passing section flips to `fail` in a refinement round — prevents the "refinement makes it worse" degenerate path ([Cerebras](https://www.cerebras.ai/blog/never-loop-without-verifiers), [ARTIE over-editing](https://openreview.net/forum?id=EPAuWPVcZQ)).
4. **Targeted, not wholesale, prompt revision:** `revise_from_qc` must change only the failing aspects; full rewrites discard good work and "may introduce new different gaps" ([Claude Certified](https://claudecertified.io/knowledge/domain1/s1-2-2-iterative-refinement-loop/), [CRAFT](https://doi.org/10.48550/arxiv.2512.20362)).
5. **Error taxonomy:** distinguish transient vs permanent generation errors; never retry permanent ones ([ReAct guide](https://sureprompts.com/blog/react-prompting-guide)).
6. **Tagged exit reasons:** return `stop_reason ∈ {approved, budget_exhausted, stagnation, regression, guardrail, cancelled, error}` in `final`, not just a boolean. "Returning `None` on failure forces the caller to guess" ([Engineering Playbook](https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions)).
7. **External cancellation:** keep `_check_cancelled(job)` at the top of every round (production already does this).
8. **Evaluator/generator decoupling:** the evaluator must be a separate stateless call (production's `_evaluate_generation` is already separate from `generate_multi_image`) — this is the MindStudio "separate the do step from the check step" rule ([MindStudio](https://www.mindstudio.ai/blog/agent-loops-verifiable-stop-conditions)) and avoids the "agent grading its own homework" anti-pattern.
9. **Lock the goal before the loop:** the `plan` (with `success_criteria`) is computed once before the loop and not redefined mid-run — this is the O'Reilly/Cursor "write down the done condition before the agent starts" rule ([O'Reilly Long-Running Agents](https://www.oreilly.com/radar/long-running-agents/)).

---

## 6. References

### Agent loop stop conditions (general)
1. Agent Patterns Catalog — **Stop Hook pattern**: https://github.com/agentpatternscatalog/patterns/blob/main/patterns/stop-hook.md
2. MindStudio — **How to Design Agent Loops with Verifiable Stop Conditions**: https://www.mindstudio.ai/blog/agent-loops-verifiable-stop-conditions
3. Engineering Playbook — **Loop Control and Exit Conditions**: https://engineering-playbook.vercel.app/agentic/loop-control-and-exit-conditions
4. Engineering Playbook — **The ReAct Loop**: https://engineering-playbook.vercel.app/agentic/react-loop
5. Cerebras — **Never Loop Without Verifiers**: https://www.cerebras.ai/blog/never-loop-without-verifiers
6. O'Reilly Radar — **Long-Running Agents**: https://www.oreilly.com/radar/long-running-agents/
7. Claude Certified — **The Iterative Refinement Loop**: https://claudecertified.io/knowledge/domain1/s1-2-2-iterative-refinement-loop/
8. SurePrompts — **ReAct Prompting Guide (2026)**: https://sureprompts.com/blog/react-prompting-guide

### Agent SDKs / frameworks
9. OpenAI Agents SDK — **Running Agents / the agent loop** (max_turns, MaxTurnsExceeded): https://github.com/openai/openai-agents-python/blob/cae28f06/docs/running_agents.md
10. OpenAI Agents SDK — **Run config** (`DEFAULT_MAX_TURNS = 10`): https://github.com/openai/openai-agents-python/blob/cae28f06/src/agents/run_config.py
11. OpenAI Agents SDK — **Runner reference** (guardrails + max_turns exceptions): https://openai.github.io/openai-agents-python/ref/run/
12. LangChain/LangGraph — **Graph API overview** (recursion_limit, GraphRecursionError, RemainingSteps): https://docs.langchain.com/oss/python/langgraph/graph-api
13. LangChain/LangGraph — **Use the graph API** (loops + termination): https://docs.langchain.com/oss/python/langgraph/use-graph-api
14. machinelearningplus — **LangGraph Cycles & Recursion Limits**: https://machinelearningplus.com/gen-ai/langgraph-cycles-recursion-limits-agent-loops/
15. AutoGen — **Termination** (MaxMessageTermination, TokenUsageTermination, TimeoutTermination, ExternalTermination, AND/OR): https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/termination.html
16. AutoGen — **`autogen_agentchat.conditions` reference**: https://microsoft.github.io/autogen/stable/reference/python/autogen%5Fagentchat.conditions.html
17. The Neural Base — **AutoGen max_turns vs custom termination** (circuit breaker pattern): https://theneuralbase.com/autogen/learn/intermediate/max-turns-vs-custom-termination/
18. The Neural Base — **AutoGen turn count optimization**: https://theneuralbase.com/autogen/learn/advanced/turn-count-optimization/
19. Cursor Docs — **Hooks** (stop hook, followup_message, loop_limit): https://cursor.com/docs/hooks
20. Cursor — **Best practices for coding with agents** (long-running loop example, MAX_ITERATIONS=5): https://cursor.com/blog/agent-best-practices
21. MeritForgeAI — **Cursor 3.5 /loop** (--max-turns, --max-runtime, cost warnings): https://www.meritforgeai.com/ai-coding/cursor-3-5-loop-scheduled-agents-may-2026/
22. Liran Tal — **Cursor Agent Hooks: Lint and Build Checks After Each Turn**: https://lirantal.com/blog/cursor-stop-hook-lint-build-verification
23. Vikas Tiwari — **Cursor hooks.json: the stop event and its stdin JSON schema**: https://tiwarivikas.in/writing/cursor-hooks-stop-event-stdin-json-schema

### Iterative / agentic image generation & refinement
24. Maestro — **Self-Improving Text-to-Image Generation via Agent Orchestration** (2025): https://arxiv.org/pdf/2509.10704
25. CRAFT — **Continuous Reasoning and Agentic Feedback Tuning for Multimodal T2I** (explicit stopping criterion, targeted edits): https://doi.org/10.48550/arxiv.2512.20362
26. RefineEdit-Agent — **An LLM-LVLM Driven Agent for Iterative and Fine-Grained Image Editing** (terminate on `E ≥ τ` OR `k_max`): https://arxiv.org/html/2508.17435v1
27. IMAGAgent — **Orchestrating Multi-Turn Image Editing via Constraint-Aware Planning and Reflection** (dual-threshold, `τ_it=3`, best-of-N fallback): https://arxiv.org/html/2603.29602
28. EditRefiner — **A Human-Aligned Agentic Framework for Image Editing Refinement** (stop on no-score-improvement or max iterations): https://ar5iv.labs.arxiv.org/html/2605.07457
29. JarvisEvo — **Self-Evolving Photo Editing Agent with Synergistic Editor-Evaluator Optimization** (anti reward-hacking dual loop): https://arxiv.org/html/2511.23002v2
30. ARTIE — **A Plug-and-Play Agentic Framework for Text Guided Image Editing** (perception-reasoning-action loop, over-editing guard): https://openreview.net/forum?id=EPAuWPVcZQ
31. MIRA — **Multimodal Iterative Reasoning Agent for Image Editing** (state + closed-loop re-evaluation against constant goal): https://arxiv.org/html/2511.21087v1

### Internal Ducon files reviewed (no external URL)
- `app/designer_agent.py` — production long-running designer loop (already multi-step).
- `app/benchmark/designer_agent.py` — dev benchmark runner (single-shot).
- `app/tool_generate_image.py` — `ImageDescriptor` dataclass and `generate_multi_image`.
- `app/gemini.py` — `PROMPT_VERIFY_MAX_ROUNDS`, `GEN_EVAL_MAX_ROUNDS`, strictness config.
- `app/prompt_loader.py` — prompt registry (`IMAGE_GEN_AGENT_SYSTEM`, `DESIGNER_PROMPT_WRITER_SYSTEM`, `STUDIO_DIRECTIONS_AGENT_SYSTEM`, `DESIGNER_EVALUATE_GENERATION`, `GEN_EVAL_RUBRIC`).
- `app/prompts/image-gen-agent.md`, `app/prompts/designer-prompt-writer.md`, `app/prompts/studio-directions-agent.md`, `app/prompts/generation-evaluator.md`, `app/prompts/gen-eval-rubric-strict.md`.
