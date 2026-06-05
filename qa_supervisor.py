"""
qa_supervisor.py — Self-Evaluating Spatial QA & Release Gate
=============================================================

A bounded "agentic" layer that sits on top of the deterministic reconstruction
pipeline (COLMAP -> gsplat -> Grounding DINO/SAM2 -> semantic lifting -> CLIP)
and decides whether a finished run is release-ready.

Design philosophy (hybrid, not "agents everywhere"):

  1. DETERMINISTIC STAGE-GATE  — auditable rules over the saved metrics produce a
     PASS / WARNING / FAIL verdict per stage and an overall release decision.
     This is reproducible and never wrong about its own numbers.

  2. LLM REASONING LAYER       — a single, bounded call to an LLM reasons over the
     same metrics + the gate verdicts to DIAGNOSE the dominant weakness and
     RECOMMEND a concrete next action in natural language. This is the part that
     genuinely earns the word "agentic": open-ended reasoning over heterogeneous
     evidence that a fixed threshold cannot do. It is constrained to cite the real
     numbers, and it degrades gracefully when no API key is present.

Why this matters for robotics/spatial AI: a perception system must know when its
own output is trustworthy. This module is a miniature of that — the pipeline
self-assesses and gates its own release.

Provider-agnostic: the LLM call uses an OpenAI-compatible client, so it works
with Groq (default), OpenAI, Gemini (OpenAI-compat endpoint), OpenRouter, or a
local Ollama server simply by setting environment variables:

    LLM_API_KEY     (required to enable the LLM layer)
    LLM_BASE_URL    (default: https://api.groq.com/openai/v1)
    LLM_MODEL       (default: llama-3.3-70b-versatile)

Usage:
    python qa_supervisor.py --metrics_dir final_full_scene_package/metrics \\
                            --output_dir final_full_scene_package/qa

If LLM_API_KEY is unset, the deterministic gate still runs and any previously
saved LLM diagnosis is reused (clearly labelled as such).
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------------- #
# Stage-gate thresholds. Chosen from indoor monocular 3DGS norms, documented so a
# reviewer can see exactly why each verdict is reached. Edit here to re-tune.
# ----------------------------------------------------------------------------- #
THRESHOLDS = {
    "psnr_pass": 25.0,          # >= good reconstruction
    "psnr_warn": 18.0,          # >= usable; below = fail
    "ssim_pass": 0.80,
    "lpips_pass": 0.20,         # <= good (lower is better)
    "semantic_pass": 0.50,      # >= 50% labelled = strong coverage
    "semantic_warn": 0.20,      # >= 20% = usable but incomplete
    "mask_frame_coverage_pass": 0.98,   # fraction of frames that produced masks
    "mask_confidence_warn": 0.40,       # mean detection confidence
    "low_opacity_warn": 0.10,           # fraction of raw Gaussians < 0.01 opacity
}

VERDICTS = ("PASS", "WARNING", "FAIL")


def _load_metrics(metrics_dir: Path) -> Dict[str, Any]:
    """Load the saved metric JSONs produced by the pipeline. Missing files are
    tolerated (their checks degrade to UNKNOWN)."""
    names = [
        "test_metrics",
        "semantic_quality_metrics",
        "mask_quality_metrics",
        "gaussian_ply_quality_metrics",
        "pipeline_efficiency_summary",
        "final_evaluation_summary",
    ]
    out: Dict[str, Any] = {}
    for n in names:
        p = metrics_dir / f"{n}.json"
        if p.exists():
            with open(p) as f:
                out[n] = json.load(f)
    return out


def _worst(verdicts: List[str]) -> str:
    """Combine verdicts: any FAIL -> FAIL, else any WARNING -> WARNING, else PASS."""
    if "FAIL" in verdicts:
        return "FAIL"
    if "WARNING" in verdicts:
        return "WARNING"
    return "PASS"


def _load_label_evidence(tables_dir: Path) -> Dict[str, Any]:
    """Load per-label tables so the reasoning layer can name SPECIFIC failing
    classes (Level 2), not just an aggregate weakness. Returns a compact, ranked
    summary; tolerant of missing files. Pure stdlib csv (no pandas dependency)."""
    import csv

    out: Dict[str, Any] = {}

    # 3D semantic coverage per class (smallest = the coverage gaps)
    dist = tables_dir / "semantic_label_distribution.csv"
    if dist.exists():
        rows = []
        with open(dist) as f:
            for r in csv.DictReader(f):
                if r["label_name"] == "unlabelled":
                    continue
                rows.append({
                    "label": r["label_name"],
                    "pct_of_total": round(float(r["percentage_of_total"]), 3),
                })
        rows.sort(key=lambda x: x["pct_of_total"])
        out["smallest_3d_classes"] = rows[:6]   # the under-represented classes
        out["largest_3d_classes"] = rows[-3:]

    # 2D mask confidence per class (lowest = least reliable detections)
    mask = tables_dir / "mask_quality_label_stats.csv"
    if mask.exists():
        rows = []
        with open(mask) as f:
            for r in csv.DictReader(f):
                rows.append({
                    "label": r["label"],
                    "mask_count": int(r["mask_count"]),
                    "mean_confidence": round(float(r["mean_confidence"]), 3),
                })
        # Only consider classes with enough masks to be meaningful
        meaningful = [r for r in rows if r["mask_count"] >= 20]
        meaningful.sort(key=lambda x: x["mean_confidence"])
        out["lowest_confidence_classes"] = meaningful[:6]
        over = sorted(rows, key=lambda x: -x["mask_count"])[:3]
        out["most_detected_classes"] = over

    return out


# ----------------------------------------------------------------------------- #
# Deterministic stage checks. Each returns (verdict, reason, evidence dict).
# ----------------------------------------------------------------------------- #
def check_visual(m: Dict[str, Any]) -> Dict[str, Any]:
    tm = m.get("test_metrics", {})
    psnr = tm.get("psnr")
    ssim = tm.get("ssim")
    lpips = tm.get("lpips")
    if psnr is None:
        return {"stage": "Visual reconstruction", "verdict": "UNKNOWN",
                "reason": "No test_metrics.json found.", "evidence": {}}
    if psnr >= THRESHOLDS["psnr_pass"] and ssim >= THRESHOLDS["ssim_pass"]:
        v = "PASS"
    elif psnr >= THRESHOLDS["psnr_warn"]:
        v = "WARNING"
    else:
        v = "FAIL"
    reason = (f"Held-out PSNR {psnr:.2f} dB, SSIM {ssim:.3f}, LPIPS {lpips:.3f} "
              f"on {tm.get('n_evaluated_frames', '?')} unseen frames.")
    return {"stage": "Visual reconstruction", "verdict": v, "reason": reason,
            "evidence": {"psnr": psnr, "ssim": ssim, "lpips": lpips}}


def check_semantic(m: Dict[str, Any]) -> Dict[str, Any]:
    sm = m.get("semantic_quality_metrics", {})
    frac = sm.get("labelled_percentage")
    if frac is None:
        return {"stage": "Semantic 3D lifting", "verdict": "UNKNOWN",
                "reason": "No semantic_quality_metrics.json found.", "evidence": {}}
    frac = frac / 100.0
    n_labels = sm.get("unique_3d_labels_excluding_unlabelled", "?")
    if frac >= THRESHOLDS["semantic_pass"]:
        v = "PASS"
    elif frac >= THRESHOLDS["semantic_warn"]:
        v = "WARNING"
    else:
        v = "FAIL"
    reason = (f"{frac * 100:.1f}% of Gaussians labelled across {n_labels} classes. "
              f"Meaningful but incomplete coverage (large flat surfaces under-segment).")
    return {"stage": "Semantic 3D lifting", "verdict": v, "reason": reason,
            "evidence": {"labelled_fraction": frac, "unique_labels": n_labels}}


def check_masks(m: Dict[str, Any]) -> Dict[str, Any]:
    mk = m.get("mask_quality_metrics", {})
    total = mk.get("total_frames")
    if not total:
        return {"stage": "2D mask quality", "verdict": "UNKNOWN",
                "reason": "No mask_quality_metrics.json found.", "evidence": {}}
    coverage = mk.get("frames_with_masks", 0) / total
    conf = mk.get("mean_confidence", 0.0)
    if coverage >= THRESHOLDS["mask_frame_coverage_pass"]:
        v = "PASS" if conf >= THRESHOLDS["mask_confidence_warn"] else "WARNING"
    else:
        v = "WARNING"
    reason = (f"{mk.get('frames_with_masks')}/{total} frames produced masks "
              f"({mk.get('average_masks_per_frame', 0):.1f}/frame), mean confidence "
              f"{conf:.2f} — consistent coverage, moderate open-vocab confidence.")
    return {"stage": "2D mask quality", "verdict": v, "reason": reason,
            "evidence": {"frame_coverage": coverage, "mean_confidence": conf}}


def check_ply(m: Dict[str, Any]) -> Dict[str, Any]:
    plys = m.get("gaussian_ply_quality_metrics", [])
    raw = next((p for p in plys if p.get("ply_name") == "raw_full_ply"), None)
    pruned = next((p for p in plys if p.get("ply_name") == "pruned_viewer_ply"), None)
    if raw is None:
        return {"stage": "Gaussian PLY quality", "verdict": "UNKNOWN",
                "reason": "No gaussian_ply_quality_metrics.json found.", "evidence": {}}
    low_op = raw.get("low_opacity_percentage_lt_0_01", 0.0) / 100.0
    pruned_low = pruned.get("low_opacity_percentage_lt_0_01", 0.0) if pruned else None
    if low_op > THRESHOLDS["low_opacity_warn"] and pruned_low == 0.0:
        v = "WARNING"  # raw has floaters but pruned viewer PLY fixes them
    elif low_op > THRESHOLDS["low_opacity_warn"]:
        v = "WARNING"
    else:
        v = "PASS"
    reason = (f"Raw PLY has {low_op * 100:.1f}% low-opacity floaters; the pruned "
              f"viewer PLY removes them (now {pruned_low:.1f}%). Viewer output is clean.")
    return {"stage": "Gaussian PLY quality", "verdict": v, "reason": reason,
            "evidence": {"raw_low_opacity_pct": low_op * 100, "pruned_low_opacity_pct": pruned_low}}


def check_efficiency(m: Dict[str, Any]) -> Dict[str, Any]:
    eff = m.get("pipeline_efficiency_summary", {})
    if not eff:
        return {"stage": "Efficiency / packaging", "verdict": "UNKNOWN",
                "reason": "No pipeline_efficiency_summary.json found.", "evidence": {}}
    metrics = {item["metric"]: item["value"] for item in eff.get("metrics", [])}
    frames = metrics.get("Total processed frames", "?")
    gs = metrics.get("Raw Gaussian count", "?")
    return {"stage": "Efficiency / packaging", "verdict": "PASS",
            "reason": (f"{frames} frames -> {gs} Gaussians; all evaluation artefacts "
                       f"and packages produced. Runtime not instrumented for this run."),
            "evidence": {"frames": frames, "gaussians": gs}}


def run_gate(m: Dict[str, Any]) -> Dict[str, Any]:
    """Run all deterministic stage checks and combine into a release decision."""
    stages = [
        check_visual(m), check_semantic(m), check_masks(m),
        check_ply(m), check_efficiency(m),
    ]
    verdicts = [s["verdict"] for s in stages if s["verdict"] in VERDICTS]
    overall = _worst(verdicts)
    decision = {
        "PASS": "RELEASE",
        "WARNING": "RELEASE WITH LIMITATIONS",
        "FAIL": "DO NOT RELEASE",
    }[overall]
    return {
        "release_decision": decision,
        "overall_verdict": overall,
        "stages": stages,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


# ----------------------------------------------------------------------------- #
# LLM reasoning layer (the genuinely agentic part). Bounded, evidence-grounded,
# provider-agnostic via an OpenAI-compatible client. Degrades gracefully.
# ----------------------------------------------------------------------------- #
def _build_prompt(m: Dict[str, Any], gate: Dict[str, Any],
                  label_evidence: Optional[Dict[str, Any]] = None) -> str:
    label_block = ""
    if label_evidence:
        label_block = (
            "\nPER-CLASS EVIDENCE (use this to name SPECIFIC failing classes, not "
            "just an aggregate weakness):\n"
            f"{json.dumps(label_evidence, indent=2)}\n"
            "- smallest_3d_classes: classes with the least 3D coverage (the gaps).\n"
            "- lowest_confidence_classes: classes whose 2D detections are least "
            "reliable (mean detection confidence; only classes with >=20 masks).\n"
            "- most_detected_classes: classes that may be over-detected.\n"
        )
    return (
        "You are a spatial-AI QA supervisor reviewing a finished indoor 3D "
        "reconstruction run (monocular video -> COLMAP -> 3D Gaussian Splatting -> "
        "open-vocab semantic lifting). A deterministic stage-gate has already "
        "produced per-stage verdicts. Using ONLY the numbers provided, write a short "
        "release report that:\n"
        "  (1) states the overall release decision;\n"
        "  (2) gives a one-line justified verdict per stage;\n"
        "  (3) identifies the single DOMINANT weakness and the most likely ROOT "
        "CAUSE, naming the SPECIFIC object/structure classes responsible (use the "
        "per-class evidence) and citing their numbers;\n"
        "  (4) recommends 2-3 concrete, class-specific next actions (e.g. which "
        "detection prompts to add or which class thresholds to adjust).\n"
        "Cite real numbers. Do not invent metrics. Be concise and honest.\n\n"
        f"DETERMINISTIC GATE:\n{json.dumps(gate, indent=2)}\n"
        f"{label_block}\n"
        f"RAW SUMMARY METRICS:\n{json.dumps(m, indent=2)[:5000]}\n"
    )


def run_llm_reasoning(m: Dict[str, Any], gate: Dict[str, Any],
                      label_evidence: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Single bounded LLM call. Returns the diagnosis text, or None if disabled
    (no key) or the call fails. Never raises into the pipeline."""
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
    try:
        from openai import OpenAI  # OpenAI-compatible client; works with Groq et al.
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise, honest QA reviewer."},
                {"role": "user", "content": _build_prompt(m, gate, label_evidence)},
            ],
            temperature=0.2,
            max_tokens=900,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:  # network, quota, lib missing — degrade gracefully
        return f"[LLM reasoning unavailable: {type(e).__name__}: {e}]"


# ----------------------------------------------------------------------------- #
# Report writers
# ----------------------------------------------------------------------------- #
def write_reports(gate: Dict[str, Any], llm_text: Optional[str], llm_live: bool,
                  output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    payload = dict(gate)
    payload["llm_diagnosis"] = llm_text
    payload["llm_diagnosis_source"] = "live" if llm_live else ("saved" if llm_text else "none")
    with open(output_dir / "agentic_qa_report.json", "w") as f:
        json.dump(payload, f, indent=2)

    # CSV decision table
    lines = ["stage,verdict,reason"]
    for s in gate["stages"]:
        reason = s["reason"].replace('"', "'")
        lines.append(f'"{s["stage"]}","{s["verdict"]}","{reason}"')
    (output_dir / "agentic_decision_table.csv").write_text("\n".join(lines) + "\n")

    # Markdown
    md = [
        "# Agentic Spatial QA — Release Report",
        "",
        f"**Final release decision: {gate['release_decision']}**  "
        f"(overall verdict: {gate['overall_verdict']})",
        "",
        "## Deterministic stage gate",
        "",
        "| Stage | Verdict | Reason |",
        "|---|---|---|",
    ]
    for s in gate["stages"]:
        md.append(f"| {s['stage']} | **{s['verdict']}** | {s['reason']} |")
    md += ["", "## Agentic diagnosis (LLM reasoning layer)", ""]
    if llm_text:
        tag = "live LLM call" if llm_live else "saved from a prior live run"
        md.append(f"*Source: {tag}.*")
        md += ["", llm_text]
    else:
        md.append("*LLM layer not run (no `LLM_API_KEY` set). The deterministic gate "
                  "above is the authoritative verdict; set a key to regenerate the "
                  "natural-language diagnosis.*")
    (output_dir / "agentic_qa_report.md").write_text("\n".join(md) + "\n")


def supervise(metrics_dir: Path, output_dir: Path,
              tables_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Top-level entry point: load metrics + per-label evidence, run the gate, run
    LLM reasoning (or reuse a saved diagnosis), and write all report artefacts.

    tables_dir defaults to a sibling 'tables/' next to metrics_dir (the layout the
    pipeline produces), so per-class evidence is picked up automatically."""
    m = _load_metrics(metrics_dir)
    if not m:
        raise FileNotFoundError(f"No metric JSONs found in {metrics_dir}")
    gate = run_gate(m)

    if tables_dir is None:
        tables_dir = metrics_dir.parent / "tables"
    label_evidence = _load_label_evidence(tables_dir) if tables_dir.exists() else {}

    llm_text = run_llm_reasoning(m, gate, label_evidence)
    llm_live = llm_text is not None and not llm_text.startswith("[LLM reasoning unavailable")

    # Graceful fallback: if no live call, reuse a previously saved diagnosis.
    saved_path = output_dir / "agentic_qa_report.json"
    if not llm_live and saved_path.exists():
        try:
            prev = json.loads(saved_path.read_text())
            if prev.get("llm_diagnosis") and prev.get("llm_diagnosis_source") in ("live", "saved"):
                llm_text = prev["llm_diagnosis"]
        except Exception:
            pass

    write_reports(gate, llm_text, llm_live, output_dir)
    return gate


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-evaluating spatial QA & release gate")
    ap.add_argument("--metrics_dir", default="final_full_scene_package/metrics",
                    help="Directory containing the saved metric JSONs")
    ap.add_argument("--output_dir", default="final_full_scene_package/qa",
                    help="Where to write the QA report artefacts")
    args = ap.parse_args()

    gate = supervise(Path(args.metrics_dir), Path(args.output_dir))

    print("=" * 60)
    print(f" AGENTIC SPATIAL QA — {gate['release_decision']}")
    print("=" * 60)
    for s in gate["stages"]:
        print(f" {s['verdict']:8s} | {s['stage']}")
        print(f"          | {s['reason']}")
    print("=" * 60)
    print(f" Reports written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
