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
- [ ] Local static test page server (simple HTTP server, no external network dependency) hosting an initial page: search box + "Search" button + ≥3 distractor elements (a decoy input, a decoy button, unrelated text).
- [ ] Fixed browser launch config: fixed viewport (e.g. 1280×800), fixed `deviceScaleFactor` (e.g. 1.0), fixed zoom (100%) — all explicit, not OS-default.
- [ ] DOM-derived ground truth extractor for this page (eval-only module, isolated from runtime perception code per the pixels-first rule).

### A.2 Perception (pixels only)
- [ ] Screenshot capture module: Playwright `page.screenshot()` wrapped in a typed `Screenshot` record carrying image bytes + explicit DPR/viewport/zoom metadata.
- [ ] UI element detector integrated (primary choice from Phase 0 technology evaluation), producing raw bounding boxes + type/confidence.
- [ ] OCR engine integrated, producing text + bounding boxes.
- [ ] Fusion step merging detector + OCR output into deduplicated `UIElement` list.

### A.3 Semantic UI representation
- [ ] `UIElement` / `SemanticUIState` Pydantic schema implemented per the fields specified in `README.md` (`id`, `type`, `bbox`, coordinate-space tag, `text`, `confidence`, `interactable`, `semantic_role`, `relations`, `source`).
- [ ] Relation inference (nearby / label-of / contained-in) implemented with unit tests on synthetic layouts.

### A.4 Target selection & grounding
- [ ] Instruction parser/matcher: resolves "find the search box" / "click Search" against `SemanticUIState` using text + role + relation signals (not raw pixels, not the DOM).
- [ ] Deterministic coordinate mapping module (screenshot px → CSS px → viewport → page, accounting for DPR/zoom/scroll) with a dedicated unit test suite covering at least: DPR=1, DPR=1.25, DPR=1.5, zoom=100%, zoom=125%, non-zero scroll offset.
- [ ] Grounding: selected `UIElement` bbox → click-point in viewport coordinates Playwright can act on.

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
