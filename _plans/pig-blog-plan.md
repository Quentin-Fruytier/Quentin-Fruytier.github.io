# PIG blog post: plan (2026-10-16)

Lives in this website repo (`_plans/` is not built by Jekyll). Target: a post on quentin-fruytier.github.io (Jekyll, `_posts/`, Blog tab), intuition-first, GIF-heavy,
at the level of the DLM project pages surveyed below. Status (2026-10-16 evening): sections 2-4 approved; hero/decode GIFs, budget walk, D-hat heat-map and the static figures are rendered in `images/pig/`; first full draft in `_drafts/pig-parallel-inconsistency-guidance.md`; blog index reworked into cards. Not committed.

## 1. What the reference pages actually do

| page | tooling | length | media | intuition device |
|---|---|---|---|---|
| DUO (s-sahoo.com/duo) | Nerfies "Academic Project Page" static HTML | ~350 words | 1 schematic + 3 side-by-side generation GIFs (DUO / MDLM / GPT-2) + YouTube tutorial link | one-line claim, epigraph, 3 "Key Innovations" bullets; no equations |
| MDLM (s-sahoo.com/mdlm) | same template | ~1,100 words | 1 static + 1 unmasking GIF | abstract, then dense math; tables |
| BD3-LM (m-arriola.com/bd3lms) | same template | ~1,200 words | 3 GIFs (AR / diffusion / block diffusion) in a check-mark comparison panel | "pros/cons" panels with a GIF each; math in the middle |
| Fast-dLLM (nvlabs.github.io/Fast-dLLM) | static site | ~800 words | hero video; side-by-side GIF (one token/step vs confidence-parallel) with a colour legend = generation step | 2 bullets "core design details", bolded speedup numbers |
| Dream 7B (hkunlp.github.io/blog/2025/dream) | Jekyll (al-folio) blog | ~1,800 words | 9 static figures + 5 GIFs (completion, infilling, L2R / random / mixed order) | contents list, "Why diffusion", no equations |
| DAPD | GitHub README only (no project page, no media) | ~700 words | none | teaser = unmasking trajectories, DAPD vs Fast-dLLM |

Take-aways. (i) Every page that lands has an animated unmasking canvas, usually side by side with a baseline,
tokens coloured by the round they were committed in. (ii) The strongest pages (Dream, Fast-dLLM) say the
idea in one or two bullets before any figure, and keep equations out of the main flow. (iii) Links row at the
top (paper / code / models), BibTeX at the bottom. (iv) 800-1,800 words; DUO shows that shorter is fine when
the GIFs do the work. Our post should sit at ~1,500 words, Dream-style structure, Fast-dLLM-style comparison GIF.

## 2. Narrative (the intuition arc)

1. **Hook: "dog meows".** One forward pass predicts every masked slot from the same canvas, so slots committed
   together cannot see each other. Individually plausible, jointly wrong. Sequential decoding fixes it but is
   the slow oracle. (Intro paragraph 2 of the paper, kept verbatim in spirit.)
2. **The question we ask is different.** Not "are these tokens confident / independent", but "would the model
   write the same tokens if it decoded them one at a time, in this order?" Name it: sequential-parallel
   inconsistency. Show the target-vs-parallel transition diagram (eq. 1) as a picture, not an equation.
3. **Why it is predictable without replaying.** Committing j changes the entropy at i: D_ij, signed. The
   whole matrix is one matrix product on top of the forward pass the model already makes (2M params, <0.1% of
   the MACs). Show the real D-hat heat-map over a canvas with the tokens printed on the axes.
4. **Turning D into a decision: the regret budget.** Walk candidates by decreasing confidence; keep adding while
   the predicted number of mismatches stays under phi (PIG.INT). Animated bar filling up to phi, each added
   token raising it by its calibrated mismatch probability. One knob, one direction: larger phi = faster.
5. **PIG.EXT in one sentence.** Same walk, but the budget is the regret you could avoid by waiting one more
   pass (theta). Show that it commits wider when the canvas is "flat" and narrower at the hard spots
   (curvature figure F06 reused).
6. **See it decode.** The hero GIF: Fast-dLLM (nu) vs PIG.INT (phi) on the same GSM8K / HumanEval prompt,
   same forward count, cell colour = round committed. Then the frontier (F22).
7. **Numbers that matter** (3 bullets max): up to 1.7x TPF at matched accuracy on LLaDA, 1.4x on Nemotron,
   +28 points at the fastest confidence settings; transfers to MMLU without refitting.
8. **Try it** (the only code block besides BibTeX): the 4-line `AutoModel.from_pretrained(..., trust_remote_code=True)`
   snippet from the README, plus `pip install` line. Links to the two Hub repos.
9. **Limits and what's next.** Adapter is per backbone; tables fitted on GSM8K-train + OpenCoder; greedy
   commits only; ms-per-round overhead stated honestly.
10. BibTeX.

## 3. Media list (what to build, how, from what)

All animations rendered on genai1 from real decodes through the release library (`pig.load` + a recording
rule like `hub/parity.py::Recorder`, extended to keep the committed set S and the token strings per round).
Rendering: matplotlib frames -> `ffmpeg` (palettegen) -> GIF, plus an MP4 fallback for the hero. Sizes kept
under ~3 MB each (site images dir is 0.4-0.5 MB per post today; GIFs will dominate).

| # | asset | kind | source | notes |
|---|---|---|---|---|
| A1 | hero: Fast-dLLM vs PIG.INT side-by-side canvas, LLaDA block 32, GSM8K prompt | GIF + MP4 | recorder, 2 decodes, same prompt | colour = round index (Fast-dLLM legend style); counters "forward passes / tokens" under each panel |
| A2 | second LLaDA examples: HumanEval/0 (identical outputs) and GSM8K #2 (same answer, wording differs from token 37) | GIF | recordings | rendered: decode-llada-humaneval0.gif, decode-llada-gsm8k2.gif; Nemotron variant not rendered (output lengths differ across lanes, no side-by-side) |
| A3 | "dog meows" toy | static SVG/PNG | hand-drawn | 2 slots, 4 combos, parallel vs sequential |
| A4 | D-hat heat-map for one round, tokens on axes, sign shown by colour | PNG | recorder (rnd.D, rnd.cand) | from the same decode as A1 |
| A5 | budget bar animation: confidence walk filling to phi | GIF | recorder (rnd.conf, table lookup) | ~8-12 candidates, shows the stop |
| A6 | PIG.EXT width vs curvature over the decode | PNG | PAPER_READY F06 (re-rendered at web width) | reuse |
| A7 | frontier, both models, 4 benchmarks | PNG | PAPER_READY F22 MAIN_FIXEDBLOCK | reuse, web-sized |
| A8 | the arXiv's own method figures: Fig. 1 teaser, PIG.INT walk, PIG.EXT walk, FDE adapter | PNG | TikZ sources `figures/fig-*.tex` of the paper repo, rendered with tectonic from a scratch copy (preamble minus `\pdfoutput=1`), 300 dpi, autocropped | done 2026-10-16. Quentin's call: keep the arXiv PIG.INT / PIG.EXT walks, but use PIG_OVERVIEW (fuller than the arXiv Fig. 1) and M3_PIPELINE (the arXiv adapter figure was shortened for the page limit) |
| A9 | calibration: predicted vs realised mismatch | PNG | PAPER_READY F25 | optional, appendix-ish |

Static figures come straight from the paper's PAPER_READY folder (already the source of truth; re-export at
~1600 px wide). Only A1/A2/A4/A5 need new GPU work: ~4 decodes, minutes, not hours.

## 4. Site mechanics

- `_posts/2026-10-xx-pig-parallel-inconsistency-guidance.md`, front matter `title / date / summary / repo / math: true`
  (same as the unrolled-CFM post). Images under `images/pig/`. Markdown `![]()` for PNG/GIF; raw `<video autoplay loop muted playsinline>` for the MP4 hero (post layout passes raw HTML through).
- Links row under the title (arXiv / Code / Models) as a one-line paragraph; the layout already prints `Code` from `repo`.
- Math: at most two display equations (D_ij definition; the budget rule), MathJax already wired.
- The website index already lists the paper; add the post to the Blog list (automatic) and optionally a
  "blog" link from the PIG entry on the home page.

## 5. Work plan (sequential, each step one approval)

Done: 1-4 (decodes were NOT re-run on the GPU: every animation is reconstructed from the parity recordings of the release check, `/ssd1/qdf76/pig_release_check/tf5/llada_branch_matrix.pt`, by re-running the selection rules on the recorded per-round state; renderer scripts live in the agent scratchpad and should be copied into `_plans/` if we want them versioned).

Original steps:

1. Approve this outline (sections 2-3). Trim/extend the media list.
2. Record the decodes for A1/A2/A4/A5 on genai1 (release library, ~10 min GPU). Needs a go.
3. Render GIFs/PNGs into `images/pig/`, review stills here before the write-up.
4. Draft the post (~1,500 words) in `_posts/`, local Jekyll build, screenshot for review.
5. Publish: push website; add the post link to the GitHub README "Paper" line and the HF cards if wanted.
