# VisiPilot — Implementation Plan

Status legend: `[ ]` not started / in progress · `[x]` implemented **and verified** (evidence required — a test run, a measurement, a benchmark output; "written but not run" stays `[ ]`) · `~~struck~~` removed from scope, with a one-line reason.

This plan is sequenced so each phase produces something measurable before the next begins. No phase 2+ item should be marked done while its phase's exit criteria are unmet.

---

## Phase 0 — Environment & technology verification (current phase)

**Goal:** know, with evidence, what actually runs on this machine before choosing anything for the pipeline.

- [x] Git repository state inspected (branch `main`, clean except one stray empty file `Visipilot`, remote `origin` → `github.com/JeffryDaniel20/Visipilot`, up to date).
- [x] Hardware inventory collected: Windows 11 (build 26200), Intel Core Ultra 7 255HX (20 threads), 16 GB RAM, RTX 5060 Laptop GPU (8 GB VRAM, Blackwell sm_120), driver 595.79 / CUDA 13.2 via `nvidia-smi`.
- [x] Software inventory collected: Python 3.12.10 and 3.14.6 present (no `uv`, no `conda`), Node 24.18.0 / npm 11.16.0 (no `pnpm`), Git 2.55.0, Docker 29.6.1, WSL2 (Ubuntu default), Chrome 152, Edge 153, no Firefox, no system CUDA toolkit/cuDNN/TensorRT installed, no ML Python packages installed yet, no Tesseract binary on PATH.
- [x] **GPU smoke test**: PyTorch 2.11.0+cu128 installed in an isolated venv; `torch.cuda.is_available()` → `True`; `torch.cuda.get_device_name(0)` → `NVIDIA GeForce RTX 5060 Laptop GPU`; `torch.cuda.get_device_capability(0)` → `(12, 0)` (sm_120/Blackwell, confirmed); a 2000×2000 matmul on-device completed successfully (`torch.version.cuda` → `12.8`). Note: the wheel download stalled twice via `pip`'s downloader on this sandboxed connection near ~77% (resolved with `curl --retry -C -`), and the install itself failed under both a long temp path (Windows `MAX_PATH`) and under `C:\` (blocked by an Application Control / WDAC policy on that path) before succeeding from a venv under `D:\` — neither issue relates to Blackwell/CUDA compatibility; both are environment-specific and worth remembering for the real install.
- [x] Technology research completed for all 11 evaluation areas (screenshot capture, browser automation, UI detection, OCR, visual grounding, inference runtime, agent reasoning, coordinate mapping, verification, evaluation datasets, frontend) with current versions/licenses cited — see the accompanying research summaries; final selections recorded in Phase A below.

**Exit criteria:** GPU smoke test passes (or fails with a documented, specific incompatibility and a fallback decision), and a primary + fallback technology choice exists for every area in this plan.

---

## Phase A — Core reliability (smallest viable vertical slice)

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
  - **A second, unresolved honest finding**: for the phrase "search box" / "Search", the real search input's fused text ("Search:" — EasyOCR misread the input's "..." placeholder ellipsis as a colon) and the real Search button's text ("Search") score **exactly the same** — both tokenize to `{"search"}` once punctuation is stripped, so the matcher cannot currently distinguish "the input near the word Search" from "the button labeled Search" by text alone. This is a genuine limitation of the current signals (OWLv2's type field is unreliable, per A.2; OCR punctuation is not a robust signal), not a bug — and it's the *correct* system behavior per Instructions.md #7 (ambiguous candidates must be surfaced, never silently guessed): `match_target()` returns both as top candidates rather than picking one. `tests/test_target_selection_integration.py` asserts the real element is *among* top candidates for both phrases, not uniquely ranked first, precisely because forcing an artificial disambiguation would be worse than an honest tie. Documented here as an open item for Phase B/C (candidate: element size/aspect-ratio as an additional signal, or a real clarification-request path once Phase C exists).
- [x] Deterministic coordinate mapping module (`visipilot/grounding/coordinates.py`): screenshot px → viewport CSS px → page CSS px, accounting for DPR, zoom, and scroll offset. **20/20 unit tests pass** (`tests/test_coordinates.py`) covering DPR 1.0/1.25/1.5/2.0, zoom 1.0/1.25/1.5, scroll offset (viewport-clipped and full-page cases), combined DPR+zoom+scroll, round-trips, and error handling. **Additionally verified against a real browser**: `test_coordinate_mapping_matches_real_dom_ground_truth[1.0/1.5/2.0]` in `tests/test_capture_integration.py` confirms the mapping model matches live Playwright/Chrome DOM ground truth within 0.5px at three DPR values — this is real evidence the zoom/DPR modeling assumption documented in `ScreenshotMeta` holds for the DPR case; the zoom-factor part of the model is still unverified against a live browser (no zoom control wired up yet — see A.1 note) and remains an open item for Phase B.
- [x] Grounding: `bbox_to_viewport_click_point()` maps a `SCREENSHOT_PX` bbox to a viewport CSS click point; unit-tested (`test_bbox_to_viewport_click_point_center`, `test_bbox_wrong_space_raises`). Now exercisable end to end since target selection produces real bboxes — `test_target_selection_integration.py` uses `screenshot_point_to_viewport_css()` (the same underlying transform) to score matcher candidates against DOM ground truth, though a dedicated "grounding takes a matcher candidate and clicks it" wiring is still A.5's job, not yet built.

**Target selection latency**: `build_semantic_state()` ~0.46 ms, `parse_instruction()` ~0.04 ms, `match_target()` ~0.11 ms/call — all pure Python, negligible next to the detector/OCR inference cost measured in A.2. Confirms these stages won't meaningfully affect the ≤5 s/instruction Phase A budget.

### A.5 Action & verification
- [ ] Action executor: click search box, type "Python", click Search — using Playwright's trusted input APIs.
- [ ] Visual verification: post-action screenshot diff / OCR check confirming expected state change (e.g. results text appears, URL/title changes) — pixels-first.
- [ ] Evaluation-mode-only DOM check confirming the same outcome, clearly labeled and isolated from the runtime path, used only to score the run.

### A.6 Tracing & logging
- [ ] Per-step trace record implemented (schema per `Instructions.md`/this plan's Phase A.7) and written to disk for every run.
- [ ] Structured logging across all stages (stage name, timing, pass/fail, failure reason if any).

### A.7 Trace record schema (to implement)
```python
class TraceStep(BaseModel):
    run_id: str
    step_index: int
    screenshot_ref: str          # path or content hash
    screenshot_hash: str
    detected_elements: list[UIElement]
    ocr_results: list[OCRResult]
    candidate_targets: list[UIElement]
    selected_target: UIElement | None
    grounding_result: GroundingResult | None
    action: ActionRecord | None
    verification_outcome: VerificationResult
    stage_timings_ms: dict[str, float]
    model_versions: dict[str, str]
    runtime_info: RuntimeInfo      # torch/onnxruntime version, device, dtype
    failure_reason: str | None
```

**Exit criteria (vertical slice pass — see full criteria below):** the fixed instruction succeeds on the controlled page across the minimum run count, within the latency/VRAM/RAM ceilings defined below, using only pixel-derived perception.

---

## Phase B — Robustness

**Goal:** the slice generalizes beyond one page layout.

- [ ] Expand controlled test pages to ≥10 layout variants (different fonts/colors/positions/frameworks — plain HTML, a Bootstrap-styled page, a React-rendered page, a page with overlapping/near-duplicate elements).
- [ ] Small-element handling: detector/OCR evaluated specifically on sub-20px icon buttons and dense toolbars.
- [ ] Scrolling: target below the fold — grounding must account for scroll offset correctly (extends A.4's coordinate tests).
- [ ] Zoom/DPR variation sweep: 100%/125%/150% browser zoom × DPR 1.0/1.25/1.5/2.0, measuring click accuracy degradation if any.
- [ ] Dynamic pages: elements that appear/move after a delay or animation; stale-screenshot detection and re-perception cycle implemented and tested.
- [ ] Difficult screenshots: low-contrast themes, dark mode, partially occluded elements — documented failure modes, not necessarily solved.

**Exit criteria:** click/grounding accuracy and task success rate measured (not assumed) across all variants above, with results recorded in `TESTING.md`.

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

*(To be created in Phase A; placeholder until the harness exists — this plan will be updated with the real command once `A.1`–`A.6` are implemented, not before.)*

```
python -m visipilot.eval.vertical_slice --page tests/pages/search_basic.html --instruction "Find the search box, type Python, and click Search." --runs 20 --report out/vertical_slice_report.json
```

The harness must (once implemented) print a pass/fail summary against every criterion above and write the full trace + metrics to the report file, so a `[x]` on any Phase A item can be checked against real evidence.
