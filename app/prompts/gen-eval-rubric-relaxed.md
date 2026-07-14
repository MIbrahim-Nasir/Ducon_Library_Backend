**Be fair, not punitive.** Reject only genuinely bad previews. Many real-world generations have
minor imperfections that are still useful to the user. Reserve `"fail"` and `"rejected"` for
serious problems — not cosmetic nitpicks.

### THREE OUTCOME TIERS

Use these tiers for the overall result (via `quality_tier` + `verdict`):

| Tier | Meaning | When to use |
|------|---------|-------------|
| **pass** | Strong result — objective met, no meaningful issues | All critical sections are `pass` or `na` |
| **accepted** | Good enough to deliver — minor imperfections only | Main objective met; some sections may be `accepted` but **none** are `fail` |
| **rejected** | Bad enough to regenerate | Any critical section is `fail`, or the result is fundamentally wrong |

**Minor issues → mark section `accepted`, NOT `fail`:**
- Slight camera/viewpoint shift (small pan, minor zoom, gentle reframing)
- Slightly extended, cropped, or shifted scene edges
- Minor lighting or shadow inconsistency
- Small edge blending artefacts
- Product shown from a different angle / repositioned / rescaled to fit the
  space — as long as it is clearly THE SAME Ducon product

**Serious issues → mark section `fail` → overall `rejected`:**
- Completely wrong output vs the request
- Major hallucinations (structures, pools, products not requested)
- Missing requested Ducon product or material entirely
- **Ducon product identity changed** — these are REAL products the client will
  buy. The generated product must match the reference's design: frame profile,
  slat/louver pattern, colour/finish, proportions, and distinctive details.
  A generic look-alike, a different model, or altered materials/finish is a
  `fail` even if it looks attractive. Only viewpoint, placement, and scale may
  adapt to the user's space.
- Buildings, walls, or major structures altered, moved, or removed
- Extremely changed viewpoint — different scene composition
- Implausible architecture or blocked access paths
- User colour marks ignored, applied in the wrong place, or still visible in the output

### MANDATORY PROCESS — EVERY CHECK (do not skip steps)

For **each** section below, work through these steps **in order** and record them in
`section_analysis` before setting `section_results`:

1. **Aspect** — one sentence: what this check evaluates and why it matters.
2. **Reference observation** — describe what you see in the relevant **input** image(s)
   for this aspect only. Be concrete: positions, edges, angles, what is at frame
   boundaries, what cuts off, left/right/foreground/background content, named structures.
3. **Generated observation** — describe the **same aspect** in the generated image (last image)
   with the same level of detail.
4. **Evaluation** — compare or judge:
   - **Preservation checks (A, C2, D2)** — direct comparison: does the generated image
     match the user's space image for this aspect? Note specific differences; distinguish
     minor drift (`accepted`) from major change (`fail`).
   - **Ducon fidelity (B1, B2)** — compare Ducon reference to the applied result in
     the generation; layout/orientation adaptation is OK — judge material/product identity.
   - **Application zones (B3)** — compare prompt intent + eligible zones to what was applied.
   - **Hallucination (C1)** — inventory extras not in inputs or prompt.
   - **Photorealism (D1)** — describe rendering quality in the generation.
   - **Architectural / UX (E)** — describe site context from user's space, describe adapted
     layout in the generation, judge **logical/architectural correctness only** (not layout
     match to the original photo where design was intentionally changed).
5. **Verdict** — `pass`, `accepted`, `fail`, or `na`. Must match `section_results`.

Write observations in complete sentences. If N/A, still state the aspect and why.

EVALUATE ALL SECTIONS:

SECTION A — PRESERVATION (compare user's space → generation)

A1 POV & CAMERA: Aspect = viewpoint, height, angle, framing, frame edges. Describe the
user's space image POV in detail. Describe the same in the generation.
- `pass` — viewpoint matches closely.
- `accepted` — slight shift, minor zoom/pan, or small reframing; scene still clearly the same space.
- `fail` — obvious recompose, rotation, major zoom, or completely different viewpoint.

A2 HARD STRUCTURES: Aspect = permanent elements. List major structures in the user's space;
describe the same in the generation.
- `fail` only on significant change, move, remove, or unrequested major add.

A3 SCENE BOUNDARIES: Aspect = sky, horizon, background, framing.
- `accepted` — slight extend/crop/shift at edges.
- `fail` — major background alteration or scene extension that changes the space.

SECTION B — DUCON ELEMENT FIDELITY

B1 AREA/SURFACE PRODUCTS: Material identity on applied surfaces. Layout adaptation OK.
- `accepted` — material broadly correct with minor drift.
- `fail` — wrong material, clear substitution, or design not applied.

B2 FIXED/HARD PRODUCTS: Discrete product identity.
- `accepted` — product recognisable with minor imperfections.
- `fail` — redesigned, substituted, or missing.

B3 APPLICATION ZONES: Correct surfaces treated.
- `fail` on clearly wrong or completely missed zones.

B4 PRODUCT INTEGRATION (multi-image — N/A if no separate product images): For EACH product
reference image, check it appears correctly. Mark N/A only when no product images were provided.
- `fail` if any requested product is missing, generic/substituted, or unrecognisable.

SECTION C — HALLUCINATION

C1 NO EXTRA ELEMENTS: Unrequested additions vs inputs + prompt.
- `accepted` — tiny incidental detail.
- `fail` — clear unrequested structures, objects, major extras, or extra parts
  copied from the Ducon reference beyond what the user asked for.

C2 NO REMOVED ELEMENTS: Silent removals of significant elements.
- `fail` for unexplained removal of major structures/features.

SECTION D — QUALITY

D1 PHOTOREALISM: Rendering believability.
- `accepted` — minor compositing imperfections.
- `fail` — severe artefacts or clearly fake rendering.

D2 LIGHTING CONSISTENCY: Light and shadows vs user's space.
- `accepted` — minor inconsistency.
- `fail` — major contradictory lighting.

SECTION E — ARCHITECTURAL LOGIC & SITE ADAPTATION (logic-only)

Do not fail adapted layout for differing from the original photo — judge logical correctness.
Mark N/A only for flat material swaps.

E1–E5: Same tier logic — `fail` only for serious logical/placement/UX problems.

SECTION F — USER MARKS / COLOUR ANNOTATIONS

When the user space has marks in **red, blue, green, yellow, purple, or white**, treat
them as spatial instructions bound to the user's request. If no marks exist → F1/F2 = `na`.

F1 MARK FOLLOW-THROUGH: Requested action must appear at the marked location(s).
- `fail` — mark intent missing, wrong place, or clearly contradicted.
- `accepted` — intent mostly met with slight positional offset that still reads as the marked zone.
- `pass` — correct action at the correct marked spot.

F2 MARK CLEANUP: No leftover coloured annotations in the final photoreal image.
- `fail` — any visible leftover mark, stroke, arrow, circle, box, scribble, or note.
- `accepted` — almost clean with a barely perceptible residual speck that does not read as a drawing.
- `pass` — fully clean; marks completely removed and surfaces look natural.

DECISION RULES

**Section consistency:** Each `section_analysis[*].verdict` MUST match `section_results` for that key.

**Overall verdict mapping:**
- If ANY `section_results` value is `"fail"` → `verdict` MUST be `"rejected"`, `quality_tier` = null.
- If NO sections are `"fail"` and all critical objectives are met:
  - All sections are `pass` or `na` → `verdict` = `"approved"`, `quality_tier` = `"pass"`.
  - One or more sections are `accepted` (but none `fail`) → `verdict` = `"approved"`, `quality_tier` = `"accepted"`.

**Do NOT reject** when the main prompt objective is visibly fulfilled and only minor
imperfections remain (slight POV drift, minor edge shift, small lighting differences).

**DO reject** when the result is materially wrong: missing products/materials, major
hallucinations, altered architecture, completely wrong scene, unrecognisable Ducon elements,
ignored user marks, mark intent applied in the wrong place, or leftover annotation drawings.
