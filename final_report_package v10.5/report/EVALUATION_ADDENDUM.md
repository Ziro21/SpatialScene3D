# Evaluation Addendum

This addendum complements the auto-generated `final_evaluation_summary.md`. That
summary is produced programmatically from the numeric metrics and therefore covers
visual reconstruction, Gaussian counts, 3D semantic coverage, 2D mask quality, and
storage. It does **not** describe three further parts of the pipeline that are
central to the submission. This file documents them for reviewer clarity. All
figures below are from the same v10.5 final run.

---

## 1. Open-vocabulary semantic search (CLIP)

Beyond per-Gaussian semantic labels, the pipeline produces **CLIP ViT-L/14
embeddings** so the reconstructed scene supports open-vocabulary text queries
("show me the sofa", "where is the door"), not just a fixed label set.

- For each detected class, image crops are encoded with CLIP and averaged into a
  single normalised embedding.
- The embeddings are written to `embeddings.npz` alongside their label IDs, so a
  text query can be matched (cosine similarity) against class embeddings and the
  result highlighted on the 3D Gaussians in the viewer.
- In the interactive viewer this drives the **Text Query** render mode: a typed
  phrase produces a CLIP heatmap over the scene.

This moves the system from "reconstruction + fixed labels" toward **queryable
scene understanding**, which is the more useful capability for a spatial-AI /
robotics context.

A worked demonstration with real query results and a heatmap visualisation is in
[`CLIP_QUERY_DEMO.md`](CLIP_QUERY_DEMO.md) (e.g. querying `"sofa"` lights up the
sofa region).

---

## 2. Packaging and artefact accessibility

The run is exported as two self-describing evidence bundles plus a manifest:

- **`final_report_package v10.5/`** — lightweight evidence: metrics JSON, the
  per-class distribution and 2D→3D comparison tables, ground-truth-vs-render
  comparison images, the train/test split, and the agentic QA report.
- **`final_full_scene_package v10.5/`** — the same evidence plus the scene PLYs
  and the semantic label map.

Large 3D artefacts are distributed according to their size. The release-artefact
filenames are listed explicitly so they can be located directly:

| Artefact (filename) | Size | Location |
|---|---:|---|
| `splat_raw_pruned_for_viewer.ply` (pruned viewer PLY) | ~90 MB | In-repo via Git LFS |
| `splat_semantic.ply` (labelled scene) | ~117 MB | GitHub Release `v10.5` |
| `splat_final.ply` (raw reconstruction) | ~115 MB | GitHub Release `v10.5` |
| `final_full_scene_package_v10.5.zip` (full evidence bundle) | ~84 MB | GitHub Release `v10.5` |
| `embeddings.npz` (CLIP embeddings) | <1 MB | In-repo |
| `masks.zip` (2D masks archive) | ~2.5 MB | In-repo |

The GitHub Release `v10.5` contains `splat_final.ply`, `splat_semantic.ply`, and
`final_full_scene_package_v10.5.zip`:
https://github.com/Ziro21/SpatialScene3D/releases/tag/v10.5

---

## 3. Agentic Spatial QA & release gate

A bounded **agentic layer** (`qa_supervisor.py`, Section 10 of the notebook)
reasons over this run's metrics and decides whether it is release-ready. It has a
deterministic stage-gate (auditable PASS/WARNING/FAIL rules) plus an LLM reasoning
tier that diagnoses the dominant weakness and names the specific under-covered
classes.

**Verdict for this run: `RELEASE WITH LIMITATIONS`.**

| Stage | Verdict |
|---|---|
| Visual reconstruction | PASS |
| Semantic 3D lifting | WARNING (37.4% coverage) |
| 2D mask quality | PASS |
| Gaussian PLY quality | WARNING (raw-PLY floaters; pruned PLY is clean) |
| Efficiency / packaging | PASS |

The full report (deterministic verdicts + the live LLM diagnosis and recommended
next actions) is saved under `qa/` in each package
(`agentic_qa_report.md` / `.json` / `agentic_decision_table.csv`).

Why this matters: a perception system must know *when its own output is
trustworthy*. This layer is a compact, auditable demonstration of that
self-assessment step — the pipeline gates its own release rather than assuming
every run is good.
