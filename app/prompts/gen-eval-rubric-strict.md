**Be strict and thorough.** Reject previews that are not faithful to the user's space, the
Ducon references, and the generation prompt. Minor cosmetic tolerance is limited — when in
doubt between `pass` and `fail`, choose `fail`.

### TWO OUTCOME TIERS (no `accepted` section tier)

Use these tiers for the overall result (via `quality_tier` + `verdict`):

| Tier | Meaning | When to use |
|------|---------|-------------|
| **pass** | Strong result — objective met, all sections `pass` or `na` | Every evaluated section is `pass` or `na` |
| **rejected** | Must regenerate | Any section is `fail`, any critical section is missing, or the result is materially wrong |

**Do NOT use section verdict `accepted`.** Use only `pass`, `fail`, or `na`.

**Mark section `fail` (→ overall `rejected`) for:**
- Any camera/viewpoint shift, recompose, zoom, or framing change vs the user's space
- Any change, move, removal, or unrequested addition of permanent structures
- Scene boundary extension, crop, or background alteration
- Material colour, texture, finish, or pattern that does not match the Ducon reference
- Product redesigned, recoloured, substituted, missing, or unrecognisable
- Design applied to wrong zones or important eligible zones missed
- Any unrequested structure, object, or major hallucination
- Silent removal of significant existing elements
- Severe compositing artefacts or clearly fake rendering
- Lighting or shadow direction that contradicts the user's space
- Illogical architecture, blocked access, or implausible placement
- User colour marks (red/blue/green/yellow/purple/white) ignored, applied in the wrong place, or still visible in the output

### MANDATORY PROCESS — EVERY CHECK (do not skip steps)

For **each** section below, work through these steps **in order** and record them in
`section_analysis` before setting `section_results`:

1. **Aspect** — one sentence: what this check evaluates and why it matters.
2. **Reference observation** — describe what you see in the relevant **input** image(s)
   for this aspect only. Be concrete: positions, edges, angles, frame boundaries, named structures.
3. **Generated observation** — describe the **same aspect** in the generated image (last image)
   with the same level of detail.
4. **Evaluation** — direct comparison or identity check. Any meaningful deviation is `fail`.
5. **Verdict** — `pass`, `fail`, or `na` only. Must match `section_results`.

Write observations in complete sentences. If N/A, still state the aspect and why.

EVALUATE ALL SECTIONS:

SECTION A — PRESERVATION (compare user's space → generation; must not change)

A1 POV & CAMERA: Viewpoint, height, angle, and framing must match the user's space image.
- `fail` if the camera has shifted, recomposed, zoomed, panned, or reframed in any noticeable way.

A2 HARD STRUCTURES: Permanent buildings, walls, fences, gates, trees, pools, and vehicles.
- `fail` if any changed shape, moved, disappeared, or appeared without instruction.

A3 SCENE BOUNDARIES: Sky, horizon, background, and peripheral framing.
- `fail` if the scene was extended, cropped, or the background altered.

SECTION B — DUCON ELEMENT FIDELITY

B1 AREA/SURFACE PRODUCTS: Material colour, texture, grain, finish, and pattern must match
the Ducon reference. Layout may adapt to site geometry.
- `fail` if material looks different from the reference or was substituted.

B2 FIXED/HARD PRODUCTS: Discrete products must match reference identity (colour, material, shape).
- `fail` if redesigned, recoloured, substituted, or missing.

B3 APPLICATION ZONES: Design applied to correct surfaces/zones per the prompt.
- `fail` if applied to wrong surfaces, inconsistently, or eligible zones were missed.

B4 PRODUCT INTEGRATION (multi-image — N/A if no separate product images):
- `fail` if any requested product is missing, generic, substituted, or unrecognisable.

SECTION C — HALLUCINATION

C1 NO EXTRA ELEMENTS: No unrequested structures, objects, furniture, plants, pools, text, or logos.
  Also fail if the generation copied extra parts from the Ducon reference that the user
  did not ask for (full-scene transplant beyond the requested change).
- `fail` for any clear hallucinated addition.

C2 NO REMOVED ELEMENTS: Significant existing elements must not be silently removed.
- `fail` for unexplained removal of major features.

SECTION D — QUALITY

D1 PHOTOREALISM: Must look like a real photograph.
- `fail` on severe compositing artefacts or unnatural rendering.

D2 LIGHTING CONSISTENCY: Cast shadows and ambient colour temperature must match the user's space.
- `fail` on noticeable lighting or shadow mismatch.

SECTION E — ARCHITECTURAL LOGIC & SITE ADAPTATION

Judge logical/architectural correctness. Mark N/A only for flat material swaps.
- `fail` on illogical routes, blocked access, implausible placement, or axis paste that ignores site geometry.

SECTION F — USER MARKS / COLOUR ANNOTATIONS

When the **user space** input has hand-drawn or overlaid marks in **red, blue, green,
yellow, purple, or white** (strokes, circles, arrows, boxes, scribbles, notes), treat
them as spatial instructions. If no such marks exist, set F1 and F2 to `na`.

F1 MARK FOLLOW-THROUGH: The generation must execute the user's stated intent **at the
marked location(s)**.
- Read the user's text/voice request together with each mark colour and position.
- `fail` if a required change is missing, applied to the wrong spot, or contradicts
  the mark + instruction (wrong product/material/action at that mark).
- `pass` only when each mark's requested action is visibly done in the correct place.

F2 MARK CLEANUP: The final image must be a clean photoreal photo with **no leftover
annotation ink**.
- `fail` if any coloured mark, stroke, arrow, circle, box, scribble, highlighter, or
  handwritten note from the input still appears (even faintly) on the generation.
- `pass` when all marks are fully removed and underlying surfaces look natural.

DECISION RULES

**Section consistency:** Each `section_analysis[*].verdict` MUST match `section_results` for that key.

**Overall verdict mapping:**
- If ANY `section_results` value is `"fail"` or missing for a critical section → `verdict` MUST be `"rejected"`, `quality_tier` = null.
- If ALL sections are `pass` or `na` → `verdict` = `"approved"`, `quality_tier` = `"pass"`.

**PASS only if** the prompt's main requested transformation is visibly fulfilled AND every
critical section (A1–A3, B1–B4, C1–C2, E1–E5 when applicable, F1–F2 when marks present) is `pass` or genuinely `na`.

**DO reject** when the candidate used the wrong reference, wrong zone, omitted requested Ducon
elements, changed permanent architecture, added unrelated objects, looks materially different
from the references, ignored user marks, applied mark intents in the wrong place, or left
annotation drawings visible — even if the image is otherwise attractive.
