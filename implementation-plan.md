# VisiPilot — Implementation Plan

Status legend: `[ ]` not started / in progress · `[x]` implemented **and verified** (evidence required — a test run, a measurement, a benchmark output; "written but not run" stays `[ ]`) · `~~struck~~` removed from scope, with a one-line reason.

This plan is sequenced so each phase produces something measurable before the next begins. No phase 2+ item should be marked done while its phase's exit criteria are unmet.

---

## Phase 0 — Environment & technology verification (complete)

**Goal:** know, with evidence, what actually runs on this machine before choosing anything for the pipeline.

- [x] Git repository state inspected (branch `main`, clean except one stray empty file `Visipilot`, remote `origin` → `github.com/JeffryDaniel20/Visipilot`, up to date).
- [x] Hardware inventory collected: Windows 11 (build 26200), Intel Core Ultra 7 255HX (20 threads), 16 GB RAM, RTX 5060 Laptop GPU (8 GB VRAM, Blackwell sm_120), driver 595.79 / CUDA 13.2 via `nvidia-smi`.
- [x] Software inventory collected: Python 3.12.10 and 3.14.6 present (no `uv`, no `conda`), Node 24.18.0 / npm 11.16.0 (no `pnpm`), Git 2.55.0, Docker 29.6.1, WSL2 (Ubuntu default), Chrome 152, Edge 153, no Firefox, no system CUDA toolkit/cuDNN/TensorRT installed, no ML Python packages installed yet, no Tesseract binary on PATH.
- [x] **GPU smoke test**: PyTorch 2.11.0+cu128 installed in an isolated venv; `torch.cuda.is_available()` → `True`; `torch.cuda.get_device_name(0)` → `NVIDIA GeForce RTX 5060 Laptop GPU`; `torch.cuda.get_device_capability(0)` → `(12, 0)` (sm_120/Blackwell, confirmed); a 2000×2000 matmul on-device completed successfully (`torch.version.cuda` → `12.8`). Note: the wheel download stalled twice via `pip`'s downloader on this sandboxed connection near ~77% (resolved with `curl --retry -C -`), and the install itself failed under both a long temp path (Windows `MAX_PATH`) and under `C:\` (blocked by an Application Control / WDAC policy on that path) before succeeding from a venv under `D:\` — neither issue relates to Blackwell/CUDA compatibility; both are environment-specific and worth remembering for the real install.
- [x] Technology research completed for all 11 evaluation areas (screenshot capture, browser automation, UI detection, OCR, visual grounding, inference runtime, agent reasoning, coordinate mapping, verification, evaluation datasets, frontend) with current versions/licenses cited — see the accompanying research summaries; final selections recorded in Phase A below.

**Exit criteria:** GPU smoke test passes (or fails with a documented, specific incompatibility and a fallback decision), and a primary + fallback technology choice exists for every area in this plan.

---

## Phase A — Core reliability (smallest viable vertical slice) — complete, exit criteria met

**Goal:** one instruction, one controlled page, fully pixels-first, end to end, measured.

### A.1 Controlled test environment
- [x] Local static test page server (`visipilot/testserver.py`, stdlib `http.server`, no external network dependency) hosting `testpages/search_basic.html`: search box + "Search" button + 3 distractor elements (newsletter input, subscribe button, unrelated text paragraph). Verified via `tests/test_capture_integration.py::test_dom_ground_truth_finds_all_page_elements` (all 5 elements resolve to a non-zero bbox).
- [x] Fixed browser launch config: `visipilot/capture/screenshot.py::launch_page` sets explicit viewport (default 1280×800) and `device_scale_factor` via the Playwright browser context (never OS default). Zoom is fixed at 100% for Phase A by design (real zoom control is Phase B scope — see A.2 note below). Verified: `test_screenshot_capture_metadata`, `test_screenshot_dimensions_scale_with_dpr[1.0/1.5/2.0]`.
- [x] DOM-derived ground truth extractor (`visipilot/eval/dom_ground_truth.py`) — eval-only, lives under `visipilot/eval/` whose package docstring states it must never be imported by runtime perception/grounding code. Verified against the real test page for all 5 elements.

### A.2 Perception (pixels only)
- [x] Screenshot capture module (`visipilot/capture/screenshot.py::capture_screenshot`): wraps Playwright's `page.screenshot()` in a typed `Screenshot` record carrying image bytes (written to disk, content-hashed filename) + explicit `ScreenshotMeta` (viewport, DPR, zoom, scroll, full_page flag). PNG width/height read directly from the IHDR chunk (no Pillow dependency added just for that). Verified via 8 passing tests in `tests/test_capture_integration.py`, including real DPR 1.0/1.5/2.0 captures against a live Chrome instance.
- [x] UI element detector integrated: `visipilot/perception/detector.py::UIDetector` wraps **OWLv2** (`google/owlv2-base-patch16-ensemble`, Apache 2.0) via `transformers`, zero-shot open-vocabulary detection with a small `ElementType`-mapped query set. Chosen over OmniParser's `icon_detect` specifically to avoid its AGPL-3.0 license by default (Instructions.md #2/#10) — no training required, PyTorch-native, reuses the Phase 0-verified cu128 CUDA stack. Verified: `tests/test_perception_integration.py::test_detector_produces_elements` passes against a real screenshot; measured 1.02–1.11 s inference, ~780 MB peak VRAM standalone. **Honest accuracy note**: at the default query set and threshold (0.1), OWLv2 labeled every detected box `text_input` on this flat synthetic test page — it never selected the `"a button"` query even for the actual Search button, and confidence scores were mediocre (0.11–0.52) throughout. This is real, measured zero-shot behavior (OWLv2 is trained on natural images, not UI screenshots, exactly as flagged as a risk in the Phase 0 report), not a code defect — confirmed by inspecting the model's own `text_labels` output during debugging. Type classification accuracy is therefore a documented open problem for Phase B, not something to paper over with cherry-picked prompts.
- [x] OCR engine integrated: `visipilot/perception/ocr.py::OCREngine` wraps **EasyOCR** (Apache 2.0), chosen over PaddleOCR specifically to stay PyTorch-native and reuse the already-verified CUDA stack rather than adding PaddlePaddle as a second, unverified-for-Blackwell ML framework (Instructions.md #10 decision rule: lower risk, verified compatibility). Verified: `test_ocr_finds_search_button_text` passes; on the real test-page screenshot it correctly read all 11 real text regions with high confidence (e.g. "Search" conf 1.000, "Subscribe" conf 0.917, "No results yet" conf 0.970) — OCR is clearly the more reliable of the two perception signals on this page. Measured: 1.55 s inference, ~1.1 GB peak VRAM standalone (991 ms / 1108 MB in the combined run below).
- [x] Fusion step merging detector + OCR output: `visipilot/perception/fusion.py::fuse()`, pure function (no model dependency), **6/6 unit tests pass** (`tests/test_fusion.py`) plus `test_fusion_pipeline_end_to_end` against real model output. On the real screenshot it correctly associated the OCR-read "Search" text with the overlapping detector box, "Subscribe" with its button box, etc. — the combined `SemanticUIState` a target-selection stage would consume is now real, not synthetic.

**Combined resource measurement** (detector + OCR loaded in the same process, matching the real pipeline's modular-monolith shape): **peak VRAM 1708.5 MB** — comfortably within the ≤6 GB budget with ~4.3 GB of headroom remaining for a future grounding VLM/reasoning LLM. Model load time (one-time, first construction): OWLv2 ~14.9 s, EasyOCR ~36.9 s (both download-then-load on a cold cache; subsequent loads are disk-cache reads only, not separately timed this session).

### A.3 Semantic UI representation
- [x] `UIElement` / `SemanticUIState` Pydantic schema implemented in `visipilot/types.py` with all specified fields (`id`, `type`, `bbox`, coordinate-space tag via `BBox.space: CoordinateSpace`, `text`, `confidence`, `interactable`, `semantic_role`, `relations`, `source`). Verified: 5 passing tests in `tests/test_types.py` (field validation, JSON round-trip, defaults).
- [x] Relation inference (nearby / label-of / contained-in): `visipilot/semantic/relations.py::infer_relations()`, pure function. **7/7 unit tests pass** (`tests/test_relations.py`) covering `contained_in`, `label_of` (created only for non-overlapping nearby pairs, not overlapping or far-apart ones), symmetric `nearby`, and no-duplicate-relation-when-a-more-specific-one-applies. Run against the real fused test-page elements: `contained_in` and `nearby` both fire meaningfully (e.g. the newsletter input/Subscribe button/distractor paragraph are all `contained_in` a large outer detector box, and pairwise `nearby`); `label_of` never fires on this specific page because fusion already absorbed every OCR text box directly into its overlapping detector box, leaving no standalone non-overlapping TEXT element for the pattern to match — a real, honestly-reported observation, not a design flaw (see A.2's fusion note for why).
- [x] Semantic role inference (not originally itemized as its own checklist line, but required by the architecture's "Semantic UI Representation Builder... semantic_role inference" — added as `visipilot/semantic/semantic_role.py::infer_semantic_role()`, a small keyword+type heuristic per Instructions.md's "simplest technically sound implementation" rule, not a trained classifier. **8/8 unit tests pass** (`tests/test_semantic_role.py`). On the real page it correctly assigned `search-input`, `search-button`/`search-related`, `submit-button`, and `labeled-input` roles to the relevant elements.
- [x] `visipilot/semantic/builder.py::build_semantic_state()` ties fusion + semantic-role inference + relation inference together into the final `SemanticUIState`, matching the README's "Semantic UI Representation Builder" module boundary. Verified via the A.4 integration tests below (it's the direct input to target selection).

### A.4 Target selection & grounding
- [x] Instruction parser/matcher: `visipilot/target_selection/instruction_parser.py::parse_instruction()` (rule-based clause splitting into FIND/TYPE/CLICK steps — chosen over an LLM parser per Instructions.md #3's pipeline-over-VLM default bias; there's no evidence yet that rule-based parsing is insufficient for Phase A's fixed instruction shapes) + `visipilot/target_selection/matcher.py::match_target()` (text + `label_of`-relation + structural-word scoring, resolving a target phrase against a `SemanticUIState`). **27 new unit tests pass** across `tests/test_instruction_parser.py` (7), `tests/test_matcher.py` (6), `tests/test_relations.py` (7), `tests/test_semantic_role.py` (8, counted under A.3 above). Also **4/4 real integration tests pass** (`tests/test_target_selection_integration.py`) running the full chain — real screenshot → real OWLv2 detector → real EasyOCR → `build_semantic_state()` → `parse_instruction()` → `match_target()` — against the controlled test page, with results scored against real DOM ground truth (eval-only, per the pixels-first rule: used only to check the matcher's output, never to produce it).
  - **A real bug was found and fixed during this integration testing, not hidden**: the test page's own distractor paragraph text literally contains the words "search box" (it describes the page's purpose in its own copy), and without a length-based penalty, naive token-overlap scoring ranked that paragraph *above* the real search input for the phrase "search box". Fixed with a general, non-page-specific heuristic in `matcher.py::_length_penalty()`: text longer than ~6 words is very unlikely to be a UI control's own label, regardless of token overlap. A regression test (`test_long_paragraph_incidentally_containing_phrase_words_loses_to_short_real_match`) locks this in.
  - **A second, honest finding — partially resolved in A.5**: for the phrase "search box" / "Search", the real search input's fused text ("Search:" — EasyOCR misread the input's "..." placeholder ellipsis as a colon) and the real Search button's text ("Search") score **exactly the same** — both tokenize to `{"search"}` once punctuation is stripped, so the matcher cannot distinguish "the input near the word Search" from "the button labeled Search" by text alone. This is a genuine limitation of the current signals (OWLv2's type field is unreliable, per A.2; OCR punctuation is not a robust signal), not a bug — and it's the *correct* system behavior per Instructions.md #7 (ambiguous candidates must be surfaced, never silently guessed): `match_target()` returns both as top candidates rather than picking one. **Update (A.5 milestone)**: an aspect-ratio-based geometric signal (`matcher.py::_aspect_ratio_bonus`, added alongside the action executor — real text inputs are much wider than tall, buttons are not) now correctly resolves the "search box" phrase (which contains a structural word) in favor of the real input. The bare "Search" phrase (the CLICK step's target, with no structural word to key off) remains genuinely tied — see A.5 for the real, honest end-to-end consequence of that.
- [x] Deterministic coordinate mapping module (`visipilot/grounding/coordinates.py`): screenshot px → viewport CSS px → page CSS px, accounting for DPR, zoom, and scroll offset. **20/20 unit tests pass** (`tests/test_coordinates.py`) covering DPR 1.0/1.25/1.5/2.0, zoom 1.0/1.25/1.5, scroll offset (viewport-clipped and full-page cases), combined DPR+zoom+scroll, round-trips, and error handling. **Additionally verified against a real browser**: `test_coordinate_mapping_matches_real_dom_ground_truth[1.0/1.5/2.0]` in `tests/test_capture_integration.py` confirms the mapping model matches live Playwright/Chrome DOM ground truth within 0.5px at three DPR values — this is real evidence the zoom/DPR modeling assumption documented in `ScreenshotMeta` holds for the DPR case; the zoom-factor part of the model is still unverified against a live browser (no zoom control wired up yet — see A.1 note) and remains an open item for Phase B.
- [x] Grounding: `bbox_to_viewport_click_point()` maps a `SCREENSHOT_PX` bbox to a viewport CSS click point; unit-tested (`test_bbox_to_viewport_click_point_center`, `test_bbox_wrong_space_raises`). Now exercisable end to end since target selection produces real bboxes — `test_target_selection_integration.py` uses `screenshot_point_to_viewport_css()` (the same underlying transform) to score matcher candidates against DOM ground truth, though a dedicated "grounding takes a matcher candidate and clicks it" wiring is still A.5's job, not yet built.

**Target selection latency**: `build_semantic_state()` ~0.46 ms, `parse_instruction()` ~0.04 ms, `match_target()` ~0.11 ms/call — all pure Python, negligible next to the detector/OCR inference cost measured in A.2. Confirms these stages won't meaningfully affect the ≤5 s/instruction Phase A budget.

### A.5 Action & verification
- [x] Action executor: `visipilot/action/executor.py` (`click_element`, `type_into_element`, `resolve_single_candidate`) + `visipilot/action/runner.py` (`run_steps`, sequencing FIND→TYPE→CLICK through the matcher and executor for a parsed instruction). Trusted input events only (`page.mouse.click`, `page.keyboard.type` — never a JS-injected click or a direct value assignment). `resolve_single_candidate` enforces Instructions.md #7: it raises `NoConfidentTargetError` below a minimum confidence (0.3) and `AmbiguousTargetError` when the top candidates are tied within an epsilon (0.05) — the runner stops rather than guessing in either case, and a single pass with no retries is a deliberate scope choice (bounded by construction; real retry/re-perception policy belongs to A.6/Phase C, not half-built here ahead of that design work). **10/10 unit tests pass** (`tests/test_executor.py`, mocked Playwright page) + **5/5 unit tests pass** (`tests/test_runner.py`, including a test that a genuinely tied CLICK step halts without clicking). **4/4 real integration tests pass** (`tests/test_action_integration.py`): a real trusted click on the real "Subscribe" button lands inside its true DOM bbox; a real FIND+TYPE resolves and types "Python" into the real search input (the aspect-ratio fix from A.4 is what makes this resolve unambiguously); and — reported honestly, not hidden — running the **exact vertical-slice instruction** for real against the real page succeeds on FIND and TYPE but halts at CLICK with `FAILED_AMBIGUOUS`, reproducing the tied "Search:" vs "Search" ambiguity documented in A.4. This is the correct, designed behavior, not a bug: the system refuses to guess between two same-text elements rather than risk clicking the wrong one.
- [x] Visual verification: `visipilot/action/verification.py::verify_text_present()` — re-runs OCR on a fresh screenshot and checks for expected text, pixels-first, no DOM access. **4/4 unit tests pass** (`tests/test_verification.py`, stub OCR) + verified for real in `test_action_integration.py::test_find_and_type_search_box_then_verify_pixels_first` (typed "Python" is visibly OCR-readable in the input after typing) and `test_pixels_first_and_dom_verification_agree_after_a_real_search` (post-search "Results for: Python" message correctly detected).
- [x] Evaluation-mode-only DOM check: `visipilot/eval/dom_verification.py::dom_text_present()` — isolated in `visipilot/eval/`, whose package docstring already states it's never imported by runtime code; this file adds no exception. Cross-checked directly against the pixels-first verifier on the same real state change in `test_pixels_first_and_dom_verification_agree_after_a_real_search` — both agree (both `True`) after a DOM-triggered search (the DOM interaction there is test scaffolding to set up the state change, exactly like `dom_ground_truth.py`'s existing tests, never part of the runtime verification path).

**A.4's tie was partially closed as part of the A.5 milestone**: the FIND step's "search box" vs "Search" ambiguity was resolved with an aspect-ratio-based geometric signal (real UI text inputs are much wider than tall, buttons are not). The CLICK step's bare-"Search" ambiguity remained open at that point, since neither structural words nor aspect ratio apply to a phrase with no structural word.

**Update (A.6 milestone) — the CLICK-step ambiguity is now resolved too, with evidence, not a hack.** Before implementing anything, the ambiguity was analyzed empirically: what general visual signal, if any, could distinguish "Search:" (input) from "Search" (button) when text alone can't? The real screenshot was sampled directly (this project's own real pixel data, not assumed) at four real elements' bboxes:

| Element | Real fill (avg RGB) | Saturation | Distance from white |
|---|---|---|---|
| search input | (255,255,255) white | 0.000 | ~0 |
| newsletter input | (251,251,251) near-white | 0.000 | ~4 |
| Search button | (87,125,255) blue | 0.659 | ~293 |
| Subscribe button | (188,188,189) gray | 0.004 | ~116 |

A first hypothesis ("high color saturation = button") was tried and **rejected**: the page's own gray Subscribe button has saturation 0.004, statistically indistinguishable from the white inputs' 0.000 — it would not have generalized. **Distance-from-pure-white**, tested second, cleanly separates all four real elements regardless of whether the button happens to be brand-colored or a neutral gray: real text-entry controls are overwhelmingly styled white/near-white (the default browser style, kept for legibility of typed text); real buttons — colored or neutral — are reliably some visible distance from that. This matches a genuinely common, near-universal web design convention, not something tuned to this one page — though it is *not* universal, and that limitation is stated up front (see below).

Implemented as `visipilot/target_selection/visual_signals.py::fill_distance_from_white()` (samples the real screenshot image directly — allowed by Instructions.md #2's actual pixels-first rule, which permits target selection to use screenshot pixels and only forbids DOM/AX access; the README's stricter "never over raw pixels" paraphrase of this boundary was inaccurate and has been corrected) and wired into `matcher.py::_fill_bonus()`, **gated off whenever a structural word is present** in the phrase specifically so it cannot re-introduce the tie `_aspect_ratio_bonus` already fixed for "search box"-style phrases (verified by hand-computing both cases before wiring it in, then confirmed by test). **5/5 new unit tests pass** (`tests/test_visual_signals.py`, synthetic images with exact real-world colors) + **4 new matcher tests pass** (`tests/test_matcher.py`): the bare-"Search" tie now resolves to the real button; the structural-word case is unaffected; and — the critical safety check — **`test_genuine_tie_between_two_identically_styled_buttons_is_preserved` confirms two truly indistinguishable candidates still score an exact tie**, so `resolve_single_candidate()` still correctly raises `AmbiguousTargetError` rather than guessing. The no-blind-clicking rule is unchanged; only the set of cases that are genuinely ambiguous got smaller, backed by real measurement.

**Stated limitation, not fixed here**: this breaks for "ghost"/outline buttons with a white or transparent fill, and for dark-mode pages where inputs may not be styled white. Flagged as a Phase B item, to be measured against real dark-mode/varied-styling test pages once they exist rather than guessed at now.

**Real-world consequence**: running the exact vertical-slice instruction against the real page now succeeds end to end — FIND, TYPE, and CLICK all resolve and execute correctly (see A.6 below for the traced, verified run). `tests/test_action_integration.py::test_full_vertical_slice_instruction_succeeds_end_to_end` replaces the earlier test that only proved the safe-halt behavior.

**Resource measurement** (full chain: screenshot → OWLv2 detect → EasyOCR read → build_semantic_state → parse_instruction → FIND+TYPE via the real action executor → screenshot → pixels-first verify): **2789.7 ms total**, well under the ≤5 s/instruction Phase A budget; find+type action latency **15.4 ms**; verification (OCR re-run) **483.7 ms**; **peak VRAM 1708.6 MB** — identical to A.2's detector+OCR baseline, confirming the action/verification stages add no additional VRAM (verification reuses the already-loaded OCR model, and Playwright's mouse/keyboard APIs are CPU-only).

### A.6 Tracing & logging
- [x] Trace record implemented and written to disk for every run: `visipilot/tracing/pipeline.py::run_instruction()` orchestrates the entire chain (screenshot → detection → OCR → semantic build → parse → act per step → optional verification) behind one function — the first point in the project where every stage built so far is called together as a single pipeline, rather than only exercised independently in tests — and writes a `TraceRecord` (`visipilot/types.py`) to `out/traces/<run_id>.json`. See A.7 below for how the schema was adapted from the original sketch. **4/4 real integration tests pass** (`tests/test_tracing.py`): a successful run (Subscribe click) produces a complete record with all expected timing keys; the trace round-trips through JSON back into a `TraceRecord`; the **full vertical-slice instruction runs end to end with real verification** (`failure_reason is None`, all three action records `SUCCESS`, `verification.passed is True`); and a genuinely-failing instruction ("Click NonexistentThingXYZ.") still produces a valid trace with a populated `failure_reason`, proving the recorder captures failure as data rather than needing a try/except at the call site.
- [x] Structured logging across all stages: every stage in `run_instruction()` logs `run=<id> stage=<name> status=<ok|fail|start> timing_ms=<...>` (plus element/step counts and failure reasons where relevant) via `logging.getLogger("visipilot.pipeline")` — a named logger, not a `basicConfig()` call, so a host application/test can attach its own handler rather than have library code dictate logging config. Verified directly: `test_successful_run_produces_complete_trace_record` uses `caplog` to assert a log line exists for every stage (screenshot, detection, ocr, build_semantic_state, parse_instruction, action).

### A.7 Trace record schema (as implemented)
Adapted from the original sketch to the types actually built in this project — one record per whole-instruction run (matching what `run_steps()` already returns), and OCR output reuses `list[UIElement]` (`source=OCR`) rather than a separate `OCRResult` type, so there's no second schema to keep in sync:
```python
class TraceRecord(BaseModel):
    run_id: str
    instruction: str
    screenshot_hash: str
    detected_elements: list[UIElement]
    ocr_elements: list[UIElement]
    fused_elements: list[UIElement]
    action_records: list[ActionRecord]
    verification: VerificationResult | None = None
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)
    runtime_info: RuntimeInfo
    failure_reason: str | None = None
```

**Resource measurement for a full traced run** (screenshot → detect → OCR → build state → parse → FIND+TYPE+CLICK → screenshot → verify, the complete vertical-slice instruction, single real run): **2525.6 ms total wall time**, well under the ≤5 s budget; **peak VRAM 1708.6 MB**, unchanged from A.2's baseline. Per-stage breakdown: screenshot 296.6 ms, detection 942.6 ms, OCR 744.8 ms, build_semantic_state 0.5 ms, parse_instruction 0.03 ms, action (all 3 steps) 25.1 ms, verification 514.7 ms.

**Exit criteria (vertical slice pass — see full criteria below): MET.** The harness now exists (`visipilot/eval/vertical_slice.py` — see below) and was run for real at the required 20-run scale: **20/20 (100%) success, 100% click accuracy (40/40 checks — both the TYPE step's focus-click into the search input and the CLICK step's click into the button, every run), max latency 2.58 s, mean 1.60 s, peak VRAM 1708.6 MB, peak RAM 1932.0 MB** — every criterion passes with real margin, not a marginal squeak. Full per-run detail is in `out/traces/*.json` (one `TraceRecord` per run, gitignored) and the aggregate report `out/vertical_slice_report.json` (also gitignored; regenerate with the command below). See `TESTING.md` §3 for the evidence-log entry.

### A.8 Vertical-slice evaluation harness
- [x] `visipilot/eval/vertical_slice.py` — a CLI (`python -m visipilot.eval.vertical_slice`) that loops `visipilot.tracing.pipeline.run_instruction()` N times against a live page, reusing one browser and one loaded detector/OCR pair across the whole batch (fresh `page.goto()` per run to reset in-page JS state — a normal navigation, not a DOM manipulation) and scores each run against DOM ground truth (`visipilot/eval/dom_ground_truth.py` — eval-only, exactly the module the pixels-first rule already carves out for this purpose; the pipeline it calls remains pixels-first throughout). Aggregates success rate, click accuracy (checked for *every* action with a grounded click point — both the TYPE step's focus-click and the CLICK step's click, not just the headline one), max/mean latency, peak VRAM (`torch.cuda.max_memory_allocated()`, reset after model load so the number matches every prior "loaded + running" measurement in this project), and peak RAM (`psutil`, BSD-3-Clause, added as a new dependency specifically for this — no Windows-native equivalent to Linux's `resource` module existed already). Writes a JSON report and prints a pass/fail line per criterion, never silently passing an unmeasurable case (e.g. zero checkable click actions is treated as a **fail**, not skipped).
- [x] **16 unit tests pass** (`tests/test_vertical_slice_harness.py`) covering the pure pass/fail logic in isolation with synthetic run data: each criterion's pass and fail boundary (including exact-threshold edge cases: 18/20 success rate and VRAM exactly at the ceiling both correctly pass), the "no checkable clicks ≠ pass" rule, and empty-input safety. **1 real integration test passes** (`test_run_batch_real_small_scale`, 2 runs) proving the actual wiring — server, browser reuse, DOM scoring, VRAM/RAM measurement, JSON-serializable output — works end to end, without needing a slow 20-run cycle inside the regular test suite.
- [x] **The real 20-run evaluation was executed** via the exact documented command (see below) — not simulated, not estimated. Result: **PASS**, with the numbers quoted above. Criteria were not loosened or reinterpreted to obtain this result; they're the same thresholds `implementation-plan.md` has carried since Phase 0 planning.

**Design note on what this harness does *not* yet do**: it measures a single fixed instruction against a single fixed page, exactly matching Phase A's intentionally narrow scope ("page variants at this phase: 1"). The `ground_truth_selectors`/`click_target_for_action` mappings are hardcoded to this one page's DOM structure — Phase B's multi-page suite will need a per-page mapping, not a copy-pasted hardcoded one, and that's a real piece of follow-up work, not an oversight.

---

## Phase B — Robustness

**Goal:** the slice generalizes beyond one page layout.

### B.1 Controlled page suite and evaluation infrastructure
- [x] `visipilot/eval/page_suite.py` — runs `visipilot.eval.vertical_slice.run_batch()` across multiple pages, reusing one loaded detector/OCR pair for the whole suite (a real, justified refactor of `run_batch()`/`run_instruction()` to accept pre-loaded models and an optional `full_page` capture flag — see below — both backward compatible, all Phase A tests still pass unchanged). **18/18 tests pass** across `tests/test_page_suite.py` (3, including 1 real end-to-end run of the whole suite at N=1) and the updated `tests/test_vertical_slice_harness.py`/`tests/test_tracing.py` (confirming the generalization didn't regress the Phase A gate).
- [~] Expand controlled test pages to ≥10 layout variants — **5 new pages built (6 total with the Phase A baseline), not 10.** Deliberately scoped down per this milestone's explicit instruction to "prefer meaningful test diversity over many superficial variants" — each new page stresses a genuinely different, real condition (see B.2 below) rather than being a color/font reskin of the same layout. Reaching 10 by adding more superficial variants would not have added evidence; adding more *meaningfully* different pages (a real SPA-style dynamically-rendered page, a partially-occluded/overlapping-elements layout) remains open Phase B work, not abandoned scope.

### B.2 Real evaluation results, 6 pages × 5 runs each
Run via `python -m visipilot.eval.page_suite --runs-per-page 5 --report out/page_suite_report.json` (report gitignored, reproducible). **Every failure below was investigated before any code change, per this milestone's explicit rule** — three were found to be real, generalizable bugs (fixed, with regression tests); the rest are genuine, currently-unsolved limitations, documented honestly rather than worked around.

| Page | Stresses | Success | Click acc. | Real finding |
|---|---|---|---|---|
| `search_basic.html` | Phase A baseline | 5/5 | 100% | Regression check — unchanged. |
| `search_bootstrap.html` | Different CSS framework conventions (pill input, `#0d6efd` blue button, different font stack) | 5/5 (after fix) | 100% | **Found and fixed a real verification bug** (see B.3) — perception, matching, and action all worked correctly on the first run; only the pixels-first verification step was wrong. |
| `search_small_icons.html` | Sub-20px (16×16) icon-only button, no text label | 0/5 | 100%* | **Resolved in the next milestone** (see B.5): the confidently-wrong click is now a safe `failed_ambiguous` refusal, consistently across 5/5 runs. Originally: OWLv2 detects the tiny button precisely, but with no OCR text `match_target("Search")` only matched the input, and the system clicked the wrong element with full confidence, caught only by downstream verification. *Click accuracy 100% because only the TYPE step's focus-click was checkable — CLICK never executed, same as dark_mode/scroll below. |
| `search_dark_mode.html` | Dark background, dark (not white) input fill | 0/5 | 100%* | **Confirms the documented A.6 limitation, empirically, for real**: the fill-color-distance-from-white signal doesn't help in dark mode (input and button both score 1.900 — the same tied pattern seen before that signal existed), so the CLICK step correctly falls back to refusing rather than guessing. *Click accuracy is 100% because only the TYPE step's focus-click was checkable — CLICK never executed. This is the safety mechanism working exactly as designed, not a defect. |
| `search_duplicate.html` | Two structurally/visually identical search box+button pairs | 0/5 | n/a | **Correct, designed behavior, not a failure of the system**: `match_target` correctly finds 3 tied candidates for "the search box" and refuses to guess. A 0% task-success rate on a deliberately ambiguous page is the *right* outcome. |
| `search_scroll.html` | Search box ~2200px below an 800px viewport (`full_page=True` capture) | 0/5 | 100%* | **Correct, designed safety behavior**: perception (with `full_page=True`, a new, tested capability — see B.3) successfully detects and reads the off-screen content; grounding correctly computes a click point far outside the current viewport; `is_within_viewport()` correctly refuses to act on it (`failed_out_of_viewport`). Scrolling the viewport to reach a resolved target is real, separate, not-yet-built functionality (bringing an off-screen target into view), not a bug in what exists. *Same caveat as dark_mode — only the checkable action was correct. |

**A test-page bug was also found and fixed during this investigation, not just pipeline bugs**: `search_scroll.html` was originally built with an incorrect assumption about its own content height (estimated ~1900px of filler; the real rendered height was 849px, leaving the target only 27px past the fold — not a meaningful test at all). Verified empirically via `get_element_bbox()` before and after the fix; the corrected page (explicit 340px-tall filler blocks) puts the target at y≈2263px, genuinely below an 800px viewport.

### B.3 Real bugs found and fixed (all with regression tests, all motivated by this investigation, none by "make a page pass")
- [x] **Fusion first-match bug** (`visipilot/perception/fusion.py::fuse()`): when an OCR text box overlapped *multiple* detector boxes above the absorption threshold, the OLD code assigned it to whichever detector box came first in iteration order — not the best match. Found on `search_small_icons.html`: OWLv2 produced both a precise box tightly matching the real input and a spurious, oversized box spanning an unrelated toolbar plus the search row; both fully contained the "Search" OCR text (overlap ratio 1.0 each — overlap ratio alone can't distinguish a tight fit from a loose one), and the spurious box won purely by list position. **Fixed in two steps, both evidence-driven**: (1) assign each OCR box to the detector box with the *highest* overlap ratio, not the first one reaching threshold; (2) on an exact ratio tie, prefer the *smaller-area* (tighter) detector box. **2 new regression tests pass** (`test_fuse_ocr_text_goes_to_best_overlap_not_first_match`, `test_fuse_full_containment_tie_prefers_smaller_detector_box`), all 6 pre-existing fusion tests still pass unchanged.
- [x] **Verification text-fragmentation bug** (`visipilot/action/verification.py::verify_text_present()`): on `search_bootstrap.html`, the DOM confirmed `#results` correctly contained "Results for: Python" and OCR correctly read every word of it — but EasyOCR split it into two separate regions ("Results for:" and "Python") instead of one, and the old verification only checked individual OCR regions, so it reported failure on a run that actually succeeded completely. **Fixed** by grouping OCR regions into visual lines (by overlapping vertical center — a plain top/bottom sort can reverse same-line fragments with different detected text heights, which is exactly what happened here: centers matched at ~349px, y-tops differed at 341 vs 335) and checking each line's joined text, never the whole page joined into one string (an earlier version of the fix did that and was caught by its own test producing a false positive from unrelated, distant page text — see the regression test for it). **3 new regression tests pass**, all 4 pre-existing verification tests still pass unchanged.
- [x] **`full_page` capture support** (`visipilot/tracing/pipeline.py::run_instruction()`): added an optional `full_page: bool = False` parameter (default preserves all existing behavior unchanged), threaded through to `capture_screenshot()`. Not a bug fix — new, tested capability needed to make the below-the-fold page a meaningful test at all (a viewport-only capture would never see the target, which would just test "the system can't see what it can't see," not scroll-awareness).

### B.4 Not yet done this milestone (honestly scoped, not abandoned)
- [ ] Zoom/DPR variation sweep — DPR is already parameterized in `launch_page()` (Phase A verified DPR 1.0/1.5/2.0 for coordinate mapping) but was not re-run across the new page suite this milestone; real browser *zoom* (as opposed to DPR) has no control wired up at all yet (Phase A's fixed-100%-zoom decision still holds).
- [ ] Dynamic pages / stale-screenshot detection / re-perception cycle — genuinely separate, real engineering (detecting staleness, re-capturing, re-running perception) that deserves its own focused milestone, not something to bolt onto this one's already-large scope.
- [ ] Partially-occluded elements, additional real framework diversity (e.g. a genuinely client-rendered/SPA-style page) — not built this milestone; candidates for the next page-suite expansion.

### B.5 Investigating and closing the "confidently wrong click" (icon-button) finding

**Investigated before any code changed, per this milestone's explicit rule.** Re-ran `build_semantic_state()` against the real `search_small_icons.html` screenshot and inspected the actual computed relations: the real button (`det-10`, bbox almost exactly matching DOM ground truth) already had a `nearby` relation to the real input (`det-9`, which absorbed the misread "Search..." placeholder) — option (b) from the earlier note, using the existing `nearby` relation, was directly testable. The input had **8 other `nearby` neighbors** too (toolbar icons, a distractor paragraph — 39 `nearby` pairs total on this one page), so a naive "any nearby textless interactable" rule would have been noisy; the button was reliably the *closest* one by edge gap.

**Two false starts, both caught by testing against real pages before trusting the fix — not assumed correct:**
1. First version picked the closest neighbor by **bbox-center distance**. On the real page, a large spurious detector box that heavily *overlapped* the input had a center closer to the input's center than the true (small, non-overlapping) button — purely because the overlapping box's own size distorted its center position. Fixed by excluding neighbors that overlap the matched element (reusing the same "spurious duplicate detection" insight fusion.py's B.3 fix already established) and ranking survivors by **edge-to-edge gap** instead of center distance.
2. Even after that fix, re-running the **full 6-page suite** (not just the one page being fixed) found a **new regression**: `search_bootstrap.html`, which had no icon-button ambiguity at all, started failing. A large (375×96), non-overlapping, textless detector box happened to sit within the `nearby` distance threshold of the real Search button there and was wrongly treated as an "adjacent icon," redirecting the click to it — a wrong click that **did not exist before this feature**. Real icon-only controls are compact; fixed by also requiring both dimensions of a candidate neighbor to be ≤48px (`_NEIGHBOR_MAX_DIMENSION_PX` in `visipilot/target_selection/matcher.py`).

**Design of the fix, once validated**: rather than guess which of two candidates (a text-matched element and its closest small, non-overlapping, textless interactable neighbor) the user means — a genuine, currently-unresolvable ambiguity given OWLv2's unreliable type field — `match_target()` now **ties their scores**, so `resolve_single_candidate()`'s existing, already-tested `AmbiguousTargetError` path fires instead of confidently picking the (arbitrary) higher scorer. This is deliberately *not* an attempt to correctly identify the right target; it's a conversion of "confidently wrong" into "safely refuses," exactly matching this milestone's explicit instruction that a safe refusal is preferable to a confident wrong action. Gated off for phrases containing a structural word ("box"/"field"/"input"), so it cannot re-introduce the tie the A.5/A.6 aspect-ratio/fill-color fixes already removed for those phrases — verified directly (`match_target("the search box")` still resolves uniquely) before and after each iteration of the fix.

**4 new regression tests pass** (`tests/test_matcher.py`): the real tie-creation case, the structural-phrase gate, a neighbor-with-text exclusion (a labeled button beside the input must not be redirected to), and the exact overlapping-spurious-box false start reproduced and locked in. **19/19 matcher tests total pass**, all pre-existing tests unchanged.

**Real-world result, re-verified with the full 6-page suite at 5 runs/page (not just the one page being fixed)**:

| Page | Before B.5 | After B.5 |
|---|---|---|
| `search_basic.html` | 5/5 | 5/5 (unchanged) |
| `search_bootstrap.html` | 5/5 | 5/5 (unchanged — after catching and fixing the false-start regression above) |
| `search_small_icons.html` | 0/5, **confidently wrong click** every run | 0/5, **`failed_ambiguous` safe refusal** every run (`det-9(1.600)` vs `det-10(1.600)`, consistent across all 5 runs) |
| `search_dark_mode.html` | 0/5, safe refusal | 0/5, safe refusal (unchanged) |
| `search_duplicate.html` | 0/5, safe refusal | 0/5, safe refusal (unchanged) |
| `search_scroll.html` | 0/5, safe refusal | 0/5, safe refusal (unchanged) |

Task-success rate for the icon-button page is numerically unchanged (0/5 either way) — **the fix was never expected to make it succeed**, since the system genuinely cannot tell which of the two elements is meant with current signals. What changed is the *kind* of failure: a silent, undetected wrong action became a correctly-flagged refusal, which is the safety property this investigation was asked to establish, not a benchmark score to chase.

**Still open, honestly**: this doesn't make the icon button *findable* — a task that genuinely requires clicking it (with no other candidate) will still correctly refuse rather than succeed. Closing that needs a real, independent visual signal for "this is a button" (OWLv2's type field is already documented as unreliable) or a Phase C clarification path, neither in scope here.

**Exit criteria:** click/grounding accuracy and task success rate measured (not assumed) across all variants above, with results recorded in `TESTING.md`. **Met for the 6 pages actually built**, including the B.5 re-evaluation — every number is a real, repeated measurement, and every failure (five to date, across two milestones) was investigated to a specific, understood root cause. Not fully met for the broader Phase B checklist (zoom sweep, dynamic pages, occlusion) — see B.4.

**Resource measurement across the suite**: peak VRAM per page ranged **1708–3813 MB**; peak RAM **2382 MB** suite-wide — both comfortably within the ≤6 GB / ≤12 GB budgets, but the range itself is a real finding: `full_page=True` capture on the scroll page (a ~2500px-tall image vs. an 800px viewport) **more than doubled peak VRAM** (3812.6 MB vs. ~1709 MB on every other page) and OCR latency (~1.8–2.2 s vs. ~0.3–0.5 s) — a real, measured cost of full-page capture that matters for any future work on scroll-aware perception, not something to assume is free.

---

## Phase C — Ambiguity and multi-step tasks

**Goal:** the system knows what it doesn't know, and can chain actions.

- [ ] Multiple plausible candidates: ranking + confidence threshold; below threshold, return candidates instead of acting.
- [ ] Clarification path: a defined (even if simple, e.g. CLI prompt) mechanism for surfacing "which one did you mean?" rather than guessing.
- [ ] Multi-step instructions ("search for X, then open the second result") — state tracking across steps, bounded retries per step, bounded total steps per task.
- [ ] Ambiguous/underspecified instructions — documented behavior (ask vs. best-effort with low-confidence flag).

**Exit criteria:** a defined set of ambiguous-case test scenarios, each with a documented expected behavior and a passing test confirming the system doesn't blindly act.

---

## Phase D — Efficiency

**Goal:** same reliability, less resource cost.

- [ ] Latency profiling per stage; identify the dominant cost (expected: detector or grounding VLM inference).
- [ ] Quantization pass on whichever model(s) are the VRAM/latency bottleneck (INT8/INT4/AWQ as applicable per Phase 0's runtime research).
- [ ] ONNX export + ONNX Runtime (CUDA EP) evaluated as a drop-in replacement for PyTorch eager mode on the detector/OCR stages, if it reduces latency/VRAM without accuracy loss.
- [ ] TensorRT evaluated as a further optimization only if ONNX Runtime's gains are insufficient and the added engine-build complexity is justified.
- [ ] Final peak VRAM/RAM numbers recorded and compared against the Phase A baseline.

**Exit criteria:** documented before/after latency and VRAM numbers for each optimization applied, with net improvement quantified.

---

## Phase E — Benchmarking

**Goal:** credible, reproducible numbers against real baselines.

- [ ] Controlled benchmark suite (the Phase B page set) run end-to-end with full metrics recorded.
- [ ] Public dataset evaluation where licensing permits: ScreenSpot / ScreenSpot-v2 (Apache 2.0 — confirmed usable), Mind2Web (CC BY 4.0 — usable with attribution); ScreenSpot-Pro's exact license to be confirmed from its repo before publishing derived benchmark numbers.
- [ ] DOM/AX baseline implemented and run over the same task set for direct comparison (baseline is eval-only, per the pixels-first rule).
- [ ] Comparison against at least one alternative perception configuration (e.g. detector+OCR pipeline vs. a single grounding VLM) on the same task set.
- [ ] Results written up with methodology, hardware, exact versions, and raw numbers — reproducible by re-running one documented command.

**Exit criteria:** a benchmark report existing as a versioned artifact in the repo, not just a claim in a chat message.

---

## Vertical-slice pass criteria (Phase A)

These numbers are intentionally modest and explained, not arbitrary:

- **Minimum successful runs:** 20 consecutive runs of the exact instruction on the base test page, ≥18/20 (90%) succeed end-to-end (correct element found, correct action taken, verification passes). 20 runs is enough to distinguish "reliably works" from "got lucky once" without requiring a large dataset before any pipeline code exists; 90% leaves room for legitimate transient flakiness (e.g. a slow page load) while still requiring the core loop to be dependable.
- **Page variants at this phase:** 1 (the base page) for the pass gate itself; Phase B is where variant count scales up. Testing only 1 page here keeps Phase A focused on proving the mechanism works before proving it generalizes.
- **Click/grounding accuracy:** click point must land inside the target element's true bounding box (from DOM ground truth, eval-only) in ≥95% of successful runs — a click just outside a 40×40px button is a real failure mode this must catch.
- **Task success rate:** ≥90% (see above), defined as: correct element clicked, "Python" typed correctly, Search clicked, and verification confirms the expected state change.
- **Maximum acceptable latency:** ≤5 seconds per full instruction (screenshot → perception → grounding → action → verification) on this hardware. Chosen as a generous but real ceiling for an interactive local agent — tightened in Phase D once a baseline exists; not tightened prematurely before real numbers exist.
- **Peak VRAM:** ≤6 GB, per the project's hard constraint — measured via `nvidia-smi` or `torch.cuda.max_memory_allocated()` during the run, not estimated.
- **Peak RAM:** ≤12 GB for the Python process (leaving headroom on a 16 GB machine already running a browser and OS) — a soft ceiling to be tightened if real numbers come in much lower.
- **Failure/retry limits:** maximum 2 retries per action step, maximum 1 full re-perception cycle per step on verification failure; a task that still hasn't succeeded after that returns a structured failure, never loops indefinitely.

## Exact vertical-slice evaluation command

**Implemented and run for real (A.8).** The page path below is corrected from the original placeholder — the real controlled test page lives at `testpages/search_basic.html` at the repo root, not `tests/pages/`.

```
python -m visipilot.eval.vertical_slice --page testpages/search_basic.html --instruction "Find the search box, type Python, and click Search." --runs 20 --report out/vertical_slice_report.json
```

**Actual output from the real run (2026-09-18):**
```
=== Vertical-slice evaluation: 20 runs ===
Success rate: 20/20 (100.0%) — need >= 90%: PASS
Click accuracy: 100.0% (20 checks) — need >= 95%: PASS
Max latency: 2.58s (mean 1.60s) — need <= 5.0s: PASS
Peak VRAM: 1708.6 MB — need <= 6144 MB: PASS
Peak RAM: 1932.0 MB — need <= 12288 MB: PASS

OVERALL: PASS
Report written to out\vertical_slice_report.json
```

The harness prints a pass/fail summary against every criterion above and writes the full metrics to `out/vertical_slice_report.json` (gitignored, along with the per-run traces in `out/traces/`) so this result is reproducible by re-running the exact command, not something to take on faith. No retry logic exists yet (see A.5), so every one of the 20 runs is a genuine single-attempt success — the 100% success rate is not inflated by retries.
