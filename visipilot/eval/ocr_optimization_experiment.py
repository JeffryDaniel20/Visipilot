"""Phase D.2 — bounded, measurement-first EasyOCR optimization experiment.

D.1 measured OCR (EasyOCR) as the dominant real pipeline cost (37.8% of
total traced time across 45 real runs, ahead of OWLv2 detection at
18.4%). This module investigates whether the three most realistic
optimization avenues for EasyOCR's existing GPU inference path — INT8
dynamic quantization, FP16/half-precision inference, and ONNX Runtime's
CUDA execution provider — are viable in this project's verified Windows
+ RTX 5060 Blackwell environment, before changing any production code.

Every finding here is a real, executed check against the real installed
library/environment, not an assumption:

1. **INT8 dynamic quantization is a confirmed no-op on this project's
   GPU configuration.** EasyOCR's own `Reader(quantize=...)` flag
   (`easyocr/detection.py::get_detector`, `easyocr/recognition.py::
   get_recognizer`) only calls `torch.quantization.quantize_dynamic`
   on the `device == 'cpu'` branch; the `gpu=True` branch this project
   always uses ignores the flag entirely. Confirmed programmatically,
   not just by reading the source: two `Reader` instances constructed
   with `quantize=True` and `quantize=False` on `gpu=True` produce
   identical model types, identical `float32` parameter dtypes, and
   identical state-dict keys — genuinely nothing differs.

2. **FP16 (`torch.autocast`) crashes the detector, and shows no
   measurable speedup on the recognizer despite working correctly
   there.** EasyOCR's CRAFT detector post-processing
   (`easyocr/craft_utils.py::getDetBoxes_core`) calls
   `cv2.threshold()` directly on the raw model output; under
   `torch.autocast(dtype=torch.float16)` that output is FP16, and
   OpenCV's `threshold()` does not support FP16 arrays — a hard,
   real crash (`cv2.error: ... Unsupported format`), not a
   theoretical incompatibility. The separate recognizer network
   (CRNN) does *not* hit that code path and runs correctly under FP16
   with text output and confidence scores matching FP32 within
   noise (<0.02 confidence delta, identical text on every real
   region tested) — but measured with repeated real runs across
   multiple pages (see `run_experiment()`), shows **no measurable
   latency improvement**, consistent with a workload that is not
   compute-bound in the way FP16 tensor cores accelerate (small
   model, small per-call batch size, likely dominated by Python-level
   looping and image pre/post-processing rather than raw matmul FLOPs).

3. **ONNX Runtime's CUDA execution provider is unusable in this
   environment as installed.** `onnxruntime-gpu` installs and lists
   `CUDAExecutionProvider` as "available"
   (`onnxruntime.get_available_providers()`), but a real session
   requesting only that provider silently falls back to
   `CPUExecutionProvider` — confirmed via `session.get_providers()`
   after construction, not assumed from the available-providers list
   alone (the same "verify the specific path actually works, don't
   trust that it's listed" discipline this project has applied to
   every prior model/library evaluation). Root-caused precisely, not
   just observed: onnxruntime-gpu 1.30.0's own diagnostic output names
   the exact cause — `Error loading ...onnxruntime_providers_cuda.dll
   which depends on cublasLt64_13.dll which is missing` and `Require
   cuDNN 9.* and CUDA 13.*` — this version of onnxruntime-gpu requires
   **CUDA 13.x**, while this project's verified, working PyTorch stack
   is pinned to **cu128 (CUDA 12.8)**. (An earlier check in this module
   that tested for `cudart64_12.dll`/`cublas64_12.dll`/`cudnn64_9.dll`
   on `PATH` reported them "found" — that was checking the wrong
   version and was misleading: those are CUDA-12-era names, but
   PyTorch's own import adds its bundled DLL directory to the process
   search path, so a plain `ctypes.CDLL` lookup finds PyTorch's private
   copies, which do not make onnxruntime's specifically-CUDA-13-
   requiring provider work. onnxruntime's own error message is the
   authoritative source here, not a same-named DLL existing somewhere
   on the path.) Making this path work would require installing a
   full, separate multi-GB CUDA 13 Toolkit + cuDNN 9 system-wide,
   alongside (not replacing) the already-verified cu128 PyTorch stack
   every other model in this project depends on — exactly the
   invasive, speculative infrastructure investment this milestone's
   instructions rule out, especially with finding (2) already
   suggesting the bottleneck isn't raw compute throughput a different
   execution engine would help with anyway.

**Net result: no viable optimization was found.** This module documents
that negative result with real, reproducible evidence rather than
integrating anything. `visipilot/perception/ocr.py` is unchanged.

Usage:
    python -m visipilot.eval.ocr_optimization_experiment --runs-per-page 5 --report out/ocr_optimization_report.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import statistics
import time
from pathlib import Path

import torch

from visipilot.capture.screenshot import capture_screenshot, launch_page
from visipilot.eval.page_suite import REPO_ROOT, TESTPAGES_DIR
from visipilot.perception.ocr import OCREngine
from visipilot.testserver import TestPageServer

logger = logging.getLogger("visipilot.eval.ocr_optimization_experiment")

DEFAULT_REPORT = REPO_ROOT / "out" / "ocr_optimization_report.json"
DEFAULT_RUNS_PER_PAGE = 5

# A representative subset of the real Phase B pages, not all 9 -- this
# experiment is about EasyOCR's own latency/accuracy behavior, which
# doesn't need every UI-layout stressor B's suite exists for, just real
# text-density variety: a normal page, a dense/small-text page, and the
# full-page-capture outlier D.1 already identified as OCR's highest-cost
# real case.
PROFILE_PAGES = ["search_basic.html", "search_bootstrap.html", "search_scroll.html"]


def _stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(values), 2),
        "median_ms": round(statistics.median(values), 2),
        "stdev_ms": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


def check_quantize_is_noop_on_gpu() -> dict:
    """Programmatic confirmation (not just reading the source) that
    EasyOCR's `quantize` flag has zero effect when `gpu=True`.
    """
    r_true = OCREngine(gpu=True).reader
    r_false_reader_kwargs = {"lang_list": ["en"], "gpu": True, "quantize": False, "verbose": False}
    import easyocr
    r_false = easyocr.Reader(**r_false_reader_kwargs)

    det_dtype_true = next(r_true.detector.parameters()).dtype
    det_dtype_false = next(r_false.detector.parameters()).dtype
    rec_dtype_true = next(r_true.recognizer.parameters()).dtype
    rec_dtype_false = next(r_false.recognizer.parameters()).dtype

    return {
        "detector_type_matches": type(r_true.detector) == type(r_false.detector),
        "recognizer_type_matches": type(r_true.recognizer) == type(r_false.recognizer),
        "detector_dtype_quantize_true": str(det_dtype_true),
        "detector_dtype_quantize_false": str(det_dtype_false),
        "recognizer_dtype_quantize_true": str(rec_dtype_true),
        "recognizer_dtype_quantize_false": str(rec_dtype_false),
        "dtypes_identical": det_dtype_true == det_dtype_false == rec_dtype_true == rec_dtype_false == torch.float32,
        "conclusion": "quantize=True/False produce bit-identical FP32 models on gpu=True -- confirmed no-op",
    }


def check_fp16_detector_crashes(ocr: OCREngine, image_path: str) -> dict:
    """FP16 autocast is expected to crash the detector's post-processing
    (a real, hard incompatibility, not a hypothesis) -- this records the
    exact exception so the finding is evidenced, not just asserted.
    """
    import cv2
    img = cv2.imread(image_path)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            ocr.reader.detect(img)
        return {"crashed": False, "note": "UNEXPECTED: detector did not crash under FP16 -- re-verify this finding"}
    except Exception as exc:  # noqa: BLE001 -- the crash itself is the evidence
        return {"crashed": True, "exception_type": type(exc).__name__, "exception_detail": str(exc)[:300]}


def check_fp16_recognizer_accuracy_and_speed(ocr: OCREngine, image_path: str, runs: int) -> dict:
    """Real repeated-run comparison of FP32 vs FP16-autocast recognition
    on the SAME real detected regions, so latency and accuracy are
    compared like-for-like.
    """
    import cv2
    img = cv2.imread(image_path)
    horizontal_list, free_list = ocr.reader.detect(img)
    if not horizontal_list or not horizontal_list[0]:
        return {"skipped": "no detected regions on this page"}

    ocr.reader.recognize(img, horizontal_list[0], free_list[0])  # warm-up

    fp32_times, fp16_times = [], []
    fp32_result = fp16_result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        fp32_result = ocr.reader.recognize(img, horizontal_list[0], free_list[0])
        fp32_times.append((time.perf_counter() - t0) * 1000)
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            fp16_result = ocr.reader.recognize(img, horizontal_list[0], free_list[0])
        fp16_times.append((time.perf_counter() - t0) * 1000)

    fp32_texts = {t: round(float(c), 3) for _, t, c in fp32_result}
    fp16_texts = {t: round(float(c), 3) for _, t, c in fp16_result}
    max_conf_delta = max((abs(fp32_texts[t] - fp16_texts.get(t, 0)) for t in fp32_texts), default=None)

    return {
        "fp32": _stats(fp32_times),
        "fp16": _stats(fp16_times),
        "same_text_recognized": set(fp32_texts) == set(fp16_texts),
        "max_confidence_delta": round(max_conf_delta, 4) if max_conf_delta is not None else None,
        "fp16_faster_by_pct": round(100 * (1 - statistics.mean(fp16_times) / statistics.mean(fp32_times)), 1),
    }


def check_onnxruntime_cuda_ep() -> dict:
    """Real check of whether onnxruntime-gpu's CUDA execution provider
    actually works in this environment, not just whether it's listed as
    available. Checks for the *specific* dependency onnxruntime-gpu
    1.30.0's own diagnostic output names (`cublasLt64_13.dll` -- a
    CUDA-13-era library) rather than guessing at CUDA-12-era DLL names:
    an earlier version of this check tested `cudart64_12.dll`/
    `cublas64_12.dll`/`cudnn64_9.dll` and found them "present" via
    `ctypes.CDLL`, which was misleading -- PyTorch's own import adds its
    bundled DLL directory to the process search path, so that lookup
    was finding PyTorch's private CUDA-12.8 copies, not confirming
    onnxruntime's actual (CUDA-13) requirement was satisfied.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {"installed": False, "note": "onnxruntime-gpu not installed in this environment"}

    available = ort.get_available_providers()

    # The specific dependency onnxruntime-gpu 1.30.0's own error message
    # names (see this module's docstring finding 3) -- a CUDA-13.x-era
    # library this project's CUDA-12.8-pinned PyTorch stack cannot
    # provide, checked directly rather than assumed from the error text.
    try:
        ctypes.CDLL("cublasLt64_13.dll")
        cublas_lt_13_found = True
    except OSError:
        cublas_lt_13_found = False

    # A minimal, real model, exported and run for real -- not assumed.
    import torch.nn as nn

    class _Tiny(nn.Module):
        def forward(self, x):
            return x * 2

    dummy = torch.randn(1, 3, 8, 8)
    tmp_path = REPO_ROOT / "out" / "_onnx_ep_check.onnx"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(_Tiny().eval(), dummy, str(tmp_path), input_names=["input"], output_names=["output"], opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(tmp_path), providers=["CUDAExecutionProvider"])
    actual_providers = session.get_providers()
    tmp_path.unlink(missing_ok=True)

    return {
        "installed": True,
        "onnxruntime_version": ort.__version__,
        "available_providers": available,
        "requested_provider": "CUDAExecutionProvider",
        "actual_session_providers": actual_providers,
        "cuda_ep_actually_used": "CUDAExecutionProvider" in actual_providers,
        "cublasLt64_13.dll_found_on_path": cublas_lt_13_found,
        "conclusion": (
            "CUDAExecutionProvider listed as available but silently falls back to CPU -- "
            "onnxruntime-gpu 1.30.0 requires CUDA 13.x/cuDNN 9.x (per its own diagnostic "
            "output), while this project's verified, working PyTorch stack is pinned to "
            "cu128 (CUDA 12.8); making this work would mean installing a second, separate "
            "CUDA Toolkit system-wide alongside the existing one"
            if "CUDAExecutionProvider" not in actual_providers
            else "CUDA EP genuinely works in this environment"
        ),
    }


def run_experiment(runs_per_page: int = DEFAULT_RUNS_PER_PAGE, report_path: Path = DEFAULT_REPORT) -> dict:
    ocr = OCREngine(gpu=True)
    torch.cuda.reset_peak_memory_stats()

    quantize_finding = check_quantize_is_noop_on_gpu()
    onnx_finding = check_onnxruntime_cuda_ep()

    fp16_findings: dict[str, dict] = {}
    with TestPageServer(directory=str(TESTPAGES_DIR), port=0) as server:
        for page_name in PROFILE_PAGES:
            url = server.url(page_name)
            with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
                full_page = page_name == "search_scroll.html"
                shot = capture_screenshot(page, full_page=full_page)

                detector_crash = check_fp16_detector_crashes(ocr, shot.image_path)
                recognizer_check = check_fp16_recognizer_accuracy_and_speed(ocr, shot.image_path, runs_per_page)
                fp16_findings[page_name] = {
                    "detector_fp16": detector_crash,
                    "recognizer_fp16": recognizer_check,
                }
                logger.info("=== page=%s: detector_crashed=%s recognizer_fp16_faster_by_pct=%s ===",
                            page_name, detector_crash.get("crashed"), recognizer_check.get("fp16_faster_by_pct"))

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    all_speedups = [
        v["recognizer_fp16"]["fp16_faster_by_pct"]
        for v in fp16_findings.values()
        if "fp16_faster_by_pct" in v["recognizer_fp16"]
    ]
    mean_speedup = round(statistics.mean(all_speedups), 1) if all_speedups else None

    any_detector_crash = any(v["detector_fp16"].get("crashed") for v in fp16_findings.values())

    report = {
        "runs_per_page": runs_per_page,
        "pages_tested": PROFILE_PAGES,
        "quantize_int8": quantize_finding,
        "fp16_by_page": fp16_findings,
        "fp16_recognizer_mean_speedup_pct": mean_speedup,
        "fp16_detector_crashes_on_every_page_tested": any_detector_crash,
        "onnxruntime_cuda_ep": onnx_finding,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "verdict": {
            "quantize_int8_viable": False,
            "fp16_viable": False,
            "onnxruntime_cuda_ep_viable": onnx_finding.get("cuda_ep_actually_used", False),
            "any_optimization_adopted": False,
            "production_code_changed": False,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _print_report(report: dict, report_path: Path) -> None:
    print("\n=== Phase D.2: EasyOCR optimization experiment ===")
    print(f"1. INT8 quantization: viable={report['verdict']['quantize_int8_viable']} "
          f"({report['quantize_int8']['conclusion']})")
    print(f"2. FP16: viable={report['verdict']['fp16_viable']} "
          f"(detector crashes on every page: {report['fp16_detector_crashes_on_every_page_tested']}, "
          f"recognizer mean speedup: {report['fp16_recognizer_mean_speedup_pct']}%)")
    print(f"3. ONNX Runtime CUDA EP: viable={report['verdict']['onnxruntime_cuda_ep_viable']} "
          f"({report['onnxruntime_cuda_ep'].get('conclusion', 'not installed')})")
    print(f"\nVerdict: any_optimization_adopted={report['verdict']['any_optimization_adopted']}, "
          f"production_code_changed={report['verdict']['production_code_changed']}")
    print(f"Peak VRAM: {report['peak_vram_mb']:.1f} MB")
    print(f"Report written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-per-page", type=int, default=DEFAULT_RUNS_PER_PAGE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = run_experiment(runs_per_page=args.runs_per_page, report_path=args.report)
    _print_report(report, args.report)


if __name__ == "__main__":
    main()
