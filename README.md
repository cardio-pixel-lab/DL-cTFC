# DLcTFC: Deep-Learning-based corrected TIMI Frame Count

Automated quantification of **corrected TIMI Frame Count (CTFC)** from coronary
angiography (CAG) sequences and their per-frame deep-learning segmentation masks.

The frame-counting algorithm follows Gibson et al.,
[*Circulation* 1996;93:879–888](https://www.ahajournals.org/doi/10.1161/01.CIR.93.5.879):
CTFC = (number of frames from proximal landmark filling to distal landmark
filling) × 2 (15 → 30 fps standardization), with the LAD length correction ÷ 1.7.

## Method overview

For each vessel branch (LCA is split into LAD / LCX by mask pixel value,
RCA is a single vessel), the pipeline works on cleaned per-frame binary masks:

- **FO (first opacification) map** — per-pixel frame index of first non-zero
  mask within `[0, Fmax]`, requiring 2 consecutive frames (hysteresis) to
  reject single-frame flicker.
- **S (seed)** — largest connected component of the earliest-filled
  `seed_frac` (0.02) pixels; anchors the ostium spatially.
- **E (extent)** — vessel extent taken from the mask at `Fmax`
  (`extent_merge`: `big` = largest component + components ≥ 10% of its area,
  `largest`, or `all`).
- **P / D (proximal / distal landmarks)** — pixels within the nearest
  `proximal_frac` (0.10) / farthest `distal_frac` (0.10) of geodesic distance
  from S over E (plateaus are merged whole; disconnected components are joined
  by jump edges).
- **F1** — `f1.percentile` (0.15) quantile of FO over P ("first full entry" of
  the proximal segment); **F2** — `f2.percentile` (0.10) quantile of FO over D
  ("distal arrival").
- **CTFC = (F2 − F1) × 2.0**; **÷ 1.7 for LAD**. Floating-point value is
  reported (plus rounded companion columns).
- Validation guards and notes: `f2 < f1` → invalid (retrograde flow / artifact);
  `f2 == f1` kept with note; single-phase filling flagged; hard guards on frame
  count, Fmax mask size and CTFC upper bound.

Per-frame regional filling curves (`fill_p/fill_d/fill_E`) are exported for
visualization and manual QC only — they do not participate in F1/F2 detection.

## Segmentation module (`seg/`)

Per-frame vessel masks consumed by the CTFC pipeline are produced by a 2-stage
medformer pipeline (code vendored under [`seg/`](seg/), upstream
CBIM-Medical-Image-Segmentation, Apache-2.0):

1. binary vessel segmentation (`config/cag2/medformer_v6_2d.yaml`, dataset
   `cag_seg_v6`) → `segmented/*_pred.png`, used as priors;
2. 3-class segmentation (`config/cag3/medformer_2d.yaml`, dataset
   `cag_seg_3class`, 2-channel input = image + prior) →
   `segmented3c2/*_pred.png` (0/128/255 = background/LCX/LAD) for LCA sequences.

Training / inference commands: see [seg/README.md](seg/README.md).
Seg evaluation vs. manual GT: `seg/eval_seg_dice.py`.

## Classification module (`cls/`)

Per-sequence LCA/RCA routing (MedViT, vendored upstream code, see
[LICENSE](cls/LICENSE)): per-frame binary classification aggregated to a
`cls_summary.csv` by `cls/make_cls_summary.py` — the CSV consumed by
`ctfc/main.py --cls-summary-csv` to split LCA into LAD/LCX and keep RCA as a
single vessel. See [cls/README.md](cls/README.md).

## Repository layout

| Path | Purpose |
|---|---|
| `ctfc/main.py` | Batch CLI pipeline (dataset walk → per-branch CTFC → reports) |
| `ctfc/ctfc_analyzer.py` | Core algorithm: mask cleaning, FO, geodesic landmarks, F1/F2, CTFC |
| `ctfc/visualizer.py` | Per-branch visualization panels |
| `ctfc/qc_report.py` | QC contact-sheet HTML + manual review workflow (`qc/qc_report.html`, `qc_queue.csv`) |
| `ctfc/qc_diag.py` | Input diagnostics: source vs. predicted masks side-by-side, 4 strata |
| `ctfc/qc_compare.py` | Overlay comparisons of QC frames |
| `ctfc/bland_altman.py` | Gold-standard validation: bias ± 1.96 SD, LoA, RMSE, Pearson r, plots |
| `ctfc/boundary_report.py` | Measurability/censoring analysis (per-vessel segmentation failure rates) |
| `ctfc/cohort_clean.py` | Cohort freeze / cleaning against the QC ledger |
| `ctfc/rederive_f1.py`, `ctfc/scan_bad_frames.py`, `ctfc/seq_cycle_diag.py`, `ctfc/fo_*.py` | Algorithm-audit and diagnostic utilities used during method development |
| `ctfc/test_ctfc.py` | Unit self-test on synthetic data |
| `ctfc/test_qc_report.js` | Node assertions for the QC HTML pages |
| `seg/` | Coronary segmentation network (medformer): training, prior-based 3-class inference, eval |
| `cls/` | LCA/RCA sequence classification (MedViT): training + queue-level `cls_summary.csv` |
| `paper_scripts/` | Manuscript figure pipelines (Fig 4–7, cohort assembly, clinical statistics) |
| `data_specs/` | Column templates for the manual gold-standard inputs (TIMI grade, manual cTFC, echo, Likert) |

## Working data (kept out of this repository)

This repository is **code-only**. All run outputs, cohort tables and internal
docs stay in the local working directory (`utils/tfc`, patient-derived — never
commit): `results_full_v2/` (full-cohort run with per-branch `detailed/` NPZ,
the substrate for parameter-free F1/F2 re-derivation), `results_f1q/`
(**frozen 1963-row cohort** + `ctfc_final.csv` + `边界说明.md` censoring
report + QC review pages — the paper's truth source), `results_diag/`,
`results_segeval/`, `f1_rule_check/`, `paper_results/`, and the `计划*.md` /
`产物说明.md` / `方法描述*.md` working docs. `paper_scripts/` entry paths point
there; derived artifacts such as `results_segeval/pairs.csv` are regenerated
by `seg/eval_seg_dice.py`.

## Installation

Two independent stacks:

```bash
pip install -r requirements.txt          # ctfc/ + paper_scripts/ (Python ≥ 3.9)
pip install -r seg/requirements.txt      # seg/ (PyTorch training/inference)
pip install -r cls/requirements.txt      # cls/ (PyTorch + timm; may share env with seg/)
```

The QC HTML page test additionally requires [Node.js](https://nodejs.org/).

## Input data layout

```
<root_dir>/
├── <case_id>/
│   ├── <sequence_id>/
│   │   ├── segmented3c2/        # per-frame masks (or fallback: segmented/)
│   │   │   └── *_pred.png       # 0/128/255 = background/LCX/LAD (RCA: 255)
│   │   └── *_src.png            # source grayscale frames
```

`--segmented-dirname "segmented3c2,segmented"` picks the first candidate that
exists per sequence. LCA/RCA assignment comes from a classification CSV
(`--cls-summary-csv` with `case_id, seq_name, vessel`); **pass the complete
file** — missing entries default to RCA.

## Usage

Batch CTFC computation:

```bash
python ctfc/main.py -r <root_dir> -o <output_dir> \
    --segmented-dirname "segmented3c2,segmented" \
    --cls-summary-csv <root_dir>/.cls_tmp/cls_summary.csv \
    --qc
```

Configuration defaults live in [ctfc/config.yaml](ctfc/config.yaml) (picked up
automatically; override with `-c`).

Outputs: `ctfc_results.json` (full precision), `ctfc_results.csv` /
`ctfc_results_valid.csv`, per-branch `detailed/` curves + masks,
`frames/` F1/F2 source-frame overlays, `visualizations/`,
`qc/qc_report.html` (manual review page, exports `qc_results.csv`),
`ctfc_report.html` summary, `logs/`.

Input diagnostics (are the masks usable at all?):

```bash
python ctfc/qc_diag.py --results <output_dir>/ctfc_results.csv \
    --root-dir <root_dir> --detailed <output_dir>/detailed \
    --out results_diag --per-class 8
```

Gold-standard validation (ground-truth CSV:
`case_id, sequence_id, ctfc_manual [, branch]`):

```bash
python ctfc/bland_altman.py --gt gt.csv --results ctfc_results.json \
    --outdir ./ba --prefix ctfc
```

Unit tests:

```bash
python ctfc/test_ctfc.py                   # 7 synthetic-data cases
node ctfc/test_qc_report.js <output_dir>/qc/qc_report.html
```

Key configuration lives in [ctfc/config.yaml](ctfc/config.yaml)
(`landmarks.seed_frac/proximal_frac/distal_frac`, `f1/f2.percentile`,
`calculation.*`, `validation.*`, `qc.*`).

## Paper figure pipeline

`paper_scripts/` reproduces the manuscript figures from the cohort outputs and
manual/expert inputs (column templates in `data_specs/`):

- `assemble_cohort.py` → merged cohort table
- `make_fig4.py`, `seg_hd95.py` → segmentation qualitative + Dice/HD95
- `make_fig5_gallery.py`, `make_fo_example.py` → CTFC
  curve galleries and FO example
- `run_clinical.py`, `make_fig6.py` → expert gold-standard agreement (Bland–Altman)
- `run_clinical.py`, `make_fig7.py` → clinical correlation (E/A, e′/a′)

Internal output paths point inside this repository; source CAG data
(`DATA_ROOT`, e.g. `/home/share/clr/share/data/AI_TMPFC`) is
environment-specific and not redistributed.

## Known limitations

1. Landmark regions are defined by pixel-area fractions of geodesic distance,
   not centerline length.
2. `seed_frac`, `proximal_frac`, `distal_frac`, `f1.percentile`, `f2.percentile`
   are preset values pending gold-standard calibration.
3. Incomplete contrast opacification (truncated Fmax) cannot be detected
   per-sequence automatically; it is flagged via notes and manual QC.
4. When the vessel classification CSV is missing or incomplete, sequences
   default to RCA and LCA branches may be misclassified.
5. The frame-rate correction assumes 15 fps acquisition.
6. Per-frame masks are predicted independently and may flicker over time; FO
   hysteresis mitigates but does not eliminate this, so the manual QC pass
   (`qc_report.html`) is a required step.
7. CTFC is reported only for the measurable subset; sequences whose distal
   segmentation fails are censored explicitly (see `boundary_report.py`)
   rather than imputed.

## Data policy

This repository contains **no patient data**. `data_specs/` holds empty column
templates for expert-provided inputs only.

## Reference

- Gibson CM, Cannon CP, Daley WL, et al. TIMI frame count: a quantitative
  method of assessing coronary artery flow. *Circulation*. 1996;93(5):879–888.
