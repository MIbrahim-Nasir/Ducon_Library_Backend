"""Designer agent tool registry and toolbox."""
from __future__ import annotations

import asyncio
import io
import json
from typing import Any, Optional

from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import prompt_loader, storage
from app.catalog_filter_context import get_keyword_filter_context
from app.db.models import Generation
from app.designer import budgets
from app.designer.catalog import ai_search, keyword_search, load_catalog_images_by_id
from app.designer.jobs import DesignerJob, check_cancelled, emit
from app.gemini import get_gemini_client
from app.ml import GeminiEmbeddingModel
from app.tool_generate_image import ImageDescriptor, generate_multi_image
from app import llm_provider
from google.genai.types import GenerateContentConfig


TOOL_NAMES = (
    "submit_plan",
    "ai_search",
    "keyword_search",
    "get_image",
    "generate_image",
    "evaluate_design",
    "finish",
)


def tool_declarations() -> list[dict[str, Any]]:
    """JSON-schema style tool declarations for Gemini / Claude function calling."""
    filter_ctx = get_keyword_filter_context()
    return [
        {
            "name": "submit_plan",
            "description": (
                "Record or refine your design plan. Call early after analyzing the space photo, "
                "and call again whenever search results or evaluation change your direction. "
                "Each call replaces the previous plan and emits it to the UI."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "space_analysis": {"type": "string"},
                    "design_direction": {"type": "string"},
                    "preserve": {"type": "array", "items": {"type": "string"}},
                    "opportunities": {"type": "array", "items": {"type": "string"}},
                    "generation_prompt_seed": {
                        "type": "string",
                        "description": "Generation prompt seed (roles + camera lock); update on refine.",
                    },
                    "success_criteria": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["space_analysis", "design_direction"],
            },
        },
        {
            "name": "ai_search",
            "description": (
                "Semantic catalog search (Chroma embeddings). Best for mood, style, layout "
                "inspiration, vague briefs, and natural-language materials/scenes. "
                "For concrete product types (pergola, fountain, paver, kitchen, planter…), "
                "also call keyword_search in the same workflow — do not rely on ai_search alone. "
                "Browse freely within the search budget; choose the best hits after inspecting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "description": (
                            f"Max hits (default {budgets.DEFAULT_SEARCH_LIMIT}, "
                            f"max {budgets.SEARCH_LIMIT_HARD_MAX})."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "keyword_search",
            "description": (
                "Exact keyword + metadata filter search on the Ducon catalog. "
                "Use for: (1) modular product tiles (level=Products), "
                "(2) exact product/design names, SKUs, or material codes, "
                "(3) filtered lookups by level / class / category / tags. "
                "Pair with ai_search for concrete product types — keyword_search finds "
                "named products and category tiles that semantic search may miss. "
                f"{filter_ctx}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Keyword text matched against name, filename, project, tags, and "
                            "description. Use exact or partial names (e.g. 'pergola', "
                            "'Outdoor Kitchen', 'SLAB IVORY', 'fountain')."
                        ),
                    },
                    "level": {
                        "type": "string",
                        "enum": ["All", "Designs", "Products", "Areas"],
                        "description": (
                            "Catalog tab. 'Products' = modular tiles; 'Designs' = full projects; "
                            "'Areas' = area entries; 'All' = cross-tab."
                        ),
                    },
                    "class": {
                        "type": "string",
                        "description": (
                            "Exact Ducon class metadata (see filter reference). "
                            "Omit or All to skip."
                        ),
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Product/area type within Products/Areas "
                            "(e.g. Pergola, Fountain, Kitchen, Planter box) — matches catalog type."
                        ),
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags — only when a specific tag is known; avoid guessing.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            f"Max hits (default {budgets.DEFAULT_SEARCH_LIMIT}, "
                            f"max {budgets.SEARCH_LIMIT_HARD_MAX})."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_image",
            "description": (
                "Load catalog images into context so you can SEE them before selecting as references. "
                f"Up to {budgets.DEFAULT_MAX_IMAGES_PER_GET} ids per call. Inspect several candidates "
                "before committing refs for generate_image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["image_ids"],
            },
        },
        {
            "name": "generate_image",
            "description": (
                "Generate a Ducon redesign via generate_multi_image. "
                "image_refs: 'user_photo' first, then up to "
                f"{budgets.DEFAULT_MAX_REFERENCES} 'catalog:<id>' refs and/or 'generation:<n>' "
                "to refine. You choose which catalog ids after browsing — prefer the best fit, "
                "not a tiny fixed shortlist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Full generation prompt."},
                    "image_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            f"'user_photo', then up to {budgets.DEFAULT_MAX_REFERENCES} "
                            "'catalog:<id>', and/or 'generation:<n>'."
                        ),
                    },
                    "aspect_ratio": {"type": "string"},
                },
                "required": ["prompt", "image_refs"],
            },
        },
        {
            "name": "evaluate_design",
            "description": (
                "Independent visual QC of a generated candidate vs the client photo. "
                "Returns score/passed/issues. Use before refining or finishing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "generation_index": {
                        "type": "integer",
                        "description": "1-based index from generate_image.",
                    },
                },
                "required": ["generation_index"],
            },
        },
        {
            "name": "finish",
            "description": (
                "End the job when you have a client-ready design (or cannot proceed). "
                "Select the best generation and summarize."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "selected_generation": {
                        "type": "integer",
                        "description": "1-based generation index to present.",
                    },
                },
                "required": ["summary"],
            },
        },
    ]


_CRITICAL_SECTION_KEYS = (
    "A1_pov",
    "A2_structures",
    "A3_scene",
    "B1_area_products",
    "B2_fixed_products",
    "B3_zones",
    "C1_no_extra",
    "C2_no_missing",
    "F1_mark_followthrough",
    "F2_mark_cleanup",
)


def _section_value(value: object) -> str:
    return str(value or "").strip().lower()


def _sections_pass(section_results: object) -> bool:
    if not isinstance(section_results, dict):
        return False
    for key in _CRITICAL_SECTION_KEYS:
        value = _section_value(section_results.get(key))
        if value not in {"pass", "na", "n/a", "not_applicable", "not applicable"}:
            return False
    return True


def passes_quality_gate(data: dict[str, Any]) -> bool:
    try:
        score = float(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    if score < budgets.pass_score():
        return False
    if not bool(data.get("passed")):
        return False
    return _sections_pass(data.get("section_results"))


def parse_selected_generation(raw: Any, generations: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Resolve finish.selected_generation to a generation dict (safety: last if invalid)."""
    if not generations:
        return None
    idx: Optional[int] = None
    if raw is not None:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            idx = None
    if idx is not None:
        match = next((g for g in generations if g.get("index") == idx), None)
        if match is not None:
            return match
    return generations[-1]


class DesignerToolbox:
    """Executes tool calls and tracks agent state for one designer job."""

    def __init__(
        self,
        *,
        job: DesignerJob,
        db: AsyncSession,
        user_image: Image.Image,
        embedding_model: GeminiEmbeddingModel,
        collection,
        image_model: str,
        aspect_ratio: Optional[str],
    ) -> None:
        self.job = job
        self.db = db
        self.user_image = user_image
        self.embedding_model = embedding_model
        self.collection = collection
        self.image_model = image_model
        self.aspect_ratio = (aspect_ratio or budgets.default_aspect_ratio()).strip() or budgets.default_aspect_ratio()

        self.max_generations = budgets.max_generations()
        self.max_searches = budgets.max_searches()
        self.search_limit = budgets.search_limit()
        self.max_references = budgets.max_references()

        self.searches_used = 0
        self.plan: Optional[dict[str, Any]] = None
        self.sources: dict[int, dict[str, Any]] = {}
        self.catalog_images: dict[int, Image.Image] = {}
        self.generations: list[dict[str, Any]] = []
        self.eval_history: list[dict[str, Any]] = []
        self.finish_args: Optional[dict[str, Any]] = None
        self.pruned: list[dict[str, Any]] = []

    def declarations(self) -> list[dict[str, Any]]:
        return tool_declarations()

    def _budget_note(self) -> dict[str, Any]:
        return {
            "searches_used": self.searches_used,
            "searches_remaining": max(0, self.max_searches - self.searches_used),
            "generations_used": len(self.generations),
            "generations_remaining": max(0, self.max_generations - len(self.generations)),
        }

    async def execute(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[Image.Image], Optional[str]]:
        try:
            return await self._execute_inner(name, args)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}, [], None

    async def _execute_inner(
        self, name: str, args: dict[str, Any]
    ) -> tuple[dict[str, Any], list[Image.Image], Optional[str]]:
        check_cancelled(self.job)

        if name == "submit_plan":
            plan = {
                "space_analysis": str(args.get("space_analysis") or ""),
                "design_direction": str(args.get("design_direction") or ""),
                "preserve": list(args.get("preserve") or []),
                "opportunities": list(args.get("opportunities") or []),
                "generation_prompt": str(args.get("generation_prompt_seed") or ""),
                "success_criteria": list(args.get("success_criteria") or []),
            }
            self.plan = plan
            await emit(self.job, "plan", plan=plan)
            return {"ok": True, "plan": plan}, [], None

        if name == "ai_search":
            if self.searches_used >= self.max_searches:
                return {"error": "search budget exhausted", **self._budget_note()}, [], None
            query = str(args.get("query") or "").strip()
            if not query:
                return {"error": "query is required"}, [], None
            limit = min(
                max(int(args.get("limit") or self.search_limit), 1),
                budgets.SEARCH_LIMIT_HARD_MAX,
            )
            self.searches_used += 1
            await emit(self.job, "search_started", query=query, search_type="ai")
            hits = await ai_search(
                db=self.db,
                embedding_model=self.embedding_model,
                collection=self.collection,
                query=query,
                limit=limit,
            )
            for h in hits:
                self.sources[int(h["id"])] = h
            await emit(
                self.job,
                "search_done",
                query=query,
                search_type="ai",
                raw_ids=[h["id"] for h in hits],
                hit_count=len(hits),
            )
            return {
                "query": query,
                "hits": hits,
                "hint": (
                    "Inspect promising ids with get_image. For concrete product types, "
                    "also call keyword_search (level=Products / category) — use both tools."
                ),
                **self._budget_note(),
            }, [], None

        if name == "keyword_search":
            if self.searches_used >= self.max_searches:
                return {"error": "search budget exhausted", **self._budget_note()}, [], None
            query = str(args.get("query") or "").strip()
            if not query:
                return {"error": "query is required"}, [], None
            tags = args.get("tags")
            self.searches_used += 1
            await emit(self.job, "search_started", query=f"[keyword] {query}", search_type="keyword")
            hits = await keyword_search(
                db=self.db,
                query=query,
                level=(str(args["level"]) if args.get("level") else None),
                class_=(str(args["class"]) if args.get("class") else None),
                category=(str(args["category"]) if args.get("category") else None),
                tags=[str(t) for t in tags] if isinstance(tags, list) else None,
                limit=min(
                    max(int(args.get("limit") or self.search_limit), 1),
                    budgets.SEARCH_LIMIT_HARD_MAX,
                ),
            )
            for h in hits:
                self.sources[int(h["id"])] = h
            await emit(
                self.job,
                "search_done",
                query=f"[keyword] {query}",
                search_type="keyword",
                raw_ids=[h["id"] for h in hits],
                hit_count=len(hits),
            )
            return {
                "query": query,
                "hits": hits,
                "hint": (
                    "Inspect promising ids with get_image. For concrete product types, "
                    "pair with ai_search if you have not already."
                ),
                **self._budget_note(),
            }, [], None

        if name == "get_image":
            raw_ids = args.get("image_ids") or []
            if not isinstance(raw_ids, list) or not raw_ids:
                return {"error": "image_ids (non-empty array) is required"}, [], None
            ids = [int(i) for i in raw_ids][: budgets.DEFAULT_MAX_IMAGES_PER_GET]
            infos, images = await load_catalog_images_by_id(self.db, ids)
            for info, img in zip([i for i in infos if i.get("loaded")], images):
                self.catalog_images[int(info["id"])] = img
            loaded = [i for i in infos if i.get("loaded")]
            await emit(
                self.job,
                "get_image",
                source_ids=ids,
                loaded_count=len(loaded),
            )
            caption = (
                "Attached catalog reference images: "
                + ", ".join(f"id {i['id']} ({i.get('name') or 'catalog'})" for i in loaded)
                if loaded
                else None
            )
            return (
                {
                    "images": infos,
                    "note": "Loaded images are attached — use catalog:<id> in generate_image image_refs.",
                },
                images,
                caption,
            )

        if name == "generate_image":
            return await self._run_generation(args)

        if name == "evaluate_design":
            return await self._run_evaluate(args)

        if name == "finish":
            self.finish_args = {
                "summary": str(args.get("summary") or ""),
                "selected_generation": args.get("selected_generation"),
            }
            selected = parse_selected_generation(args.get("selected_generation"), self.generations)
            refs = list((selected or {}).get("references") or [])
            if refs:
                await emit(
                    self.job,
                    "reference_board",
                    selected=refs,
                    removed=self.pruned[:20],
                    removed_count=len(self.pruned),
                    message=f"Selected {len(refs)} catalog references for the final design.",
                )
            return {"ok": True, "note": "Job will end after this turn."}, [], None

        return {"error": f"unknown tool: {name}"}, [], None

    def _resolve_refs(self, refs: list[Any]) -> tuple[list[ImageDescriptor], list[dict[str, Any]], list[str]]:
        descriptors: list[ImageDescriptor] = []
        chosen_refs: list[dict[str, Any]] = []
        errors: list[str] = []
        for raw in refs:
            ref = str(raw or "").strip()
            low = ref.lower()
            if low in {"user_photo", "user photo", "client space photo", "user_image"}:
                descriptors.append(
                    ImageDescriptor(label="client space photo", type="file", pil_image=self.user_image)
                )
            elif low.startswith("catalog:"):
                try:
                    cid = int(low.split(":", 1)[1])
                except ValueError:
                    errors.append(f"invalid catalog ref: {ref}")
                    continue
                src = self.sources.get(cid) or {"id": cid, "name": f"catalog {cid}", "label": f"catalog {cid}"}
                chosen_refs.append(src)
                descriptors.append(
                    ImageDescriptor(
                        label=str(src.get("label") or src.get("name") or f"catalog {cid}"),
                        type="catalog_id",
                        source=str(cid),
                    )
                )
            elif low.startswith("generation:"):
                try:
                    gidx = int(low.split(":", 1)[1])
                except ValueError:
                    errors.append(f"invalid generation ref: {ref}")
                    continue
                gen = next((g for g in self.generations if g["index"] == gidx), None)
                if gen is None or gen.get("pil") is None:
                    errors.append(f"unknown generation index: {gidx}")
                    continue
                descriptors.append(
                    ImageDescriptor(
                        label="previous design candidate",
                        type="file",
                        pil_image=gen["pil"],
                    )
                )
            elif low.isdigit():
                cid = int(low)
                src = self.sources.get(cid) or {"id": cid, "name": f"catalog {cid}", "label": f"catalog {cid}"}
                chosen_refs.append(src)
                descriptors.append(
                    ImageDescriptor(label=str(src.get("label") or f"catalog {cid}"), type="catalog_id", source=low)
                )
            else:
                errors.append(f"unknown image ref: {ref}")

        # Safety cap on catalog refs (heuristics only as caps — model chooses which).
        catalog_desc = [d for d in descriptors if d.type == "catalog_id"]
        if len(catalog_desc) > self.max_references:
            keep = set(id(d) for d in catalog_desc[: self.max_references])
            trimmed = [d for d in descriptors if d.type != "catalog_id" or id(d) in keep]
            dropped = [d for d in catalog_desc[self.max_references :]]
            for d in dropped:
                self.pruned.append({"id": d.source, "reason": "max_references safety cap"})
            descriptors = trimmed
            chosen_refs = chosen_refs[: self.max_references]

        if not any(d.label == "client space photo" for d in descriptors):
            descriptors.insert(
                0, ImageDescriptor(label="client space photo", type="file", pil_image=self.user_image)
            )
        return descriptors, chosen_refs, errors

    async def _run_generation(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], list[Image.Image], Optional[str]]:
        if len(self.generations) >= self.max_generations:
            return {
                "error": "generation budget exhausted",
                **self._budget_note(),
                "hint": "Call finish with your best generation.",
            }, [], None
        prompt = str(args.get("prompt") or "").strip()
        refs = args.get("image_refs") or ["user_photo"]
        if not prompt:
            return {"error": "prompt is required"}, [], None
        descriptors, chosen_refs, ref_errors = self._resolve_refs(refs if isinstance(refs, list) else [refs])
        if ref_errors and len([d for d in descriptors if d.type == "catalog_id" or d.label == "previous design candidate"]) == 0:
            # Allow user_photo-only only if agent explicitly asked; still proceed if only warnings.
            if len(descriptors) <= 1 and any("unknown" in e or "invalid" in e for e in ref_errors):
                return {
                    "error": "could not resolve image_refs",
                    "details": ref_errors,
                    "hint": "Use user_photo, catalog:<id> from search hits, or generation:<n>.",
                }, [], None

        aspect = str(args.get("aspect_ratio") or "").strip() or self.aspect_ratio
        gen_index = len(self.generations) + 1
        await emit(
            self.job,
            "generation_started",
            attempt=gen_index,
            max_attempts=self.max_generations,
            round=gen_index,
            message=f"Generating design candidate {gen_index}.",
        )
        if chosen_refs:
            await emit(
                self.job,
                "reference_board",
                selected=chosen_refs,
                removed=self.pruned[:20],
                removed_count=len(self.pruned),
                message=f"Using {len(chosen_refs)} catalog references for attempt {gen_index}.",
            )

        result = await generate_multi_image(
            user_id=self.job.user_id,
            prompt=prompt,
            descriptors=descriptors,
            model=self.image_model,
            aspect_ratio=aspect,
            db=self.db,
            output_prefix=f"designer_job_{self.job.id[:8]}_{gen_index}",
            enable_verify=True,
        )
        pil_images: list[Image.Image] = []
        try:
            gen_id = int(result["id"])
            pil = await self._load_generation_image(gen_id)
            pil_images = [pil]
        except Exception:
            raw_pils = result.get("pil_images") or []
            pil_images = list(raw_pils[:1])

        output_images = []
        if result.get("signed_url") or result.get("url"):
            output_images.append(
                {
                    "id": result.get("id"),
                    "url": result.get("signed_url") or result.get("url"),
                    "round": gen_index,
                }
            )

        entry = {
            "index": gen_index,
            "attempt": gen_index,
            "generation": dict(result),
            "pil": pil_images[0] if pil_images else None,
            "prompt": str(result.get("final_prompt") or prompt),
            "references": chosen_refs,
            "evaluation": None,
        }
        self.generations.append(entry)

        await emit(
            self.job,
            "generation_done",
            attempt=gen_index,
            round=gen_index,
            generation=result,
            output_images=output_images,
        )
        payload: dict[str, Any] = {
            "generation_index": gen_index,
            "generation_id": result.get("id"),
            "output_images": output_images,
            "resolved_image_refs": refs,
            **self._budget_note(),
            "note": (
                f"Candidate generation_index={gen_index} attached. "
                "Call evaluate_design or refine with generation:<n> / finish."
            ),
        }
        if ref_errors:
            payload["image_ref_warnings"] = ref_errors
        caption = (
            f"Attached: generated design candidate generation_index={gen_index}."
            if pil_images
            else None
        )
        return payload, pil_images, caption

    async def _load_generation_image(self, generation_id: int) -> Image.Image:
        row = (
            await self.db.execute(select(Generation).where(Generation.id == generation_id))
        ).scalar_one_or_none()
        if not row:
            raise RuntimeError(f"Generation {generation_id} not found after creation.")
        if storage.CLOUD_STORAGE:
            import httpx

            url = storage.get_generation_url(row.id, row.url)
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to fetch generated image ({resp.status_code}).")
            content = resp.content
            return await asyncio.to_thread(lambda: Image.open(io.BytesIO(content)).convert("RGB"))
        local = storage.serve_local_path(row.url)
        if not local.exists():
            raise RuntimeError(f"Generated image file is missing: {local}")
        return await asyncio.to_thread(lambda: Image.open(local).convert("RGB"))

    async def _run_evaluate(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], list[Image.Image], Optional[str]]:
        if not self.generations:
            return {"error": "no generations to evaluate yet"}, [], None
        gidx = int(args.get("generation_index") or 0)
        gen = next((g for g in self.generations if g["index"] == gidx), None)
        if gen is None:
            gen = self.generations[-1]
            gidx = gen["index"]
        if gen.get("pil") is None:
            return {"error": f"generation {gidx} has no image to evaluate"}, [], None

        prompt_loader.ensure_prompts_loaded()
        plan = self.plan or {"design_direction": "Ducon redesign"}
        references = gen.get("references") or []
        eval_prompt = (
            f"{prompt_loader.DESIGNER_EVALUATE_GENERATION}\n\n"
            f"Plan: {json.dumps(plan, ensure_ascii=False)}\n"
            f"References: {json.dumps(references, ensure_ascii=False)}\n"
            f"Prompt used: {gen.get('prompt') or ''}"
        )
        user_image = self.user_image
        generated = gen["pil"]
        ref_images = [self.catalog_images[int(r["id"])] for r in references if int(r.get("id") or 0) in self.catalog_images]

        if llm_provider.use_claude():
            blocks = [
                llm_provider.pil_image_block(user_image),
                llm_provider.pil_image_block(generated),
            ]
            blocks += [llm_provider.pil_image_block(img) for img in ref_images]
            blocks.append(llm_provider.text_block(eval_prompt))
            text = await llm_provider.agenerate_text("", blocks) or "{}"
        else:
            client = get_gemini_client()
            response = await client.aio.models.generate_content(
                model=budgets.designer_model(),
                contents=[user_image, generated, *ref_images, eval_prompt],
                config=GenerateContentConfig(response_mime_type="application/json"),
            )
            text = response.text or "{}"
            try:
                from app.admin.usage_helpers import record_from_response
                record_from_response(
                    response,
                    agent="designer",
                    model=budgets.designer_model(),
                    user_id=self.job.user_id,
                )
            except Exception:
                pass

        try:
            data = llm_provider.parse_json_text(text) if llm_provider.use_claude() else json.loads(text)
        except (json.JSONDecodeError, ValueError):
            data = {
                "score": 6,
                "passed": False,
                "strengths": [],
                "issues": [text[:500]],
                "improvements": "",
            }
        if not isinstance(data, dict):
            data = {"score": 6, "passed": False, "issues": ["invalid eval payload"]}
        data["passed"] = passes_quality_gate(data)
        gen["evaluation"] = data
        entry = {"round": gidx, "attempt": gidx, **data}
        self.eval_history.append(entry)
        await emit(self.job, "evaluation", attempt=gidx, evaluation=data)
        await emit(
            self.job,
            "eval",
            round=gidx,
            approved=bool(data.get("passed")),
            reasons=data.get("strengths") or data.get("reasons") or [],
            defects=data.get("issues") or data.get("defects") or [],
        )
        if not data.get("passed"):
            await emit(
                self.job,
                "retry",
                attempt=gidx + 1,
                reason=data.get("improvements") or "Improving realism and design fit.",
            )
        return entry, [], None
