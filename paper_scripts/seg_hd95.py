#!/usr/bin/env python3
"""Test-split segmentation metrics: Dice/IoU/Precision/Recall + HD95.

Reuses results_segeval acceptance (pairs.csv, status=ok) and recomputes
metrics from raw masks so HD95 is available (segeval did not store it).
GT: cag_seg_v6/annotations/{split}/{stem}.png (0/1). Pred: pairs.csv pred_path.
"""
import json, os, sys
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

GT_ROOT = '/home/share/clr/share/data/cag_seg_v6'
PAIRS = '/home/share/clr/share/work/DLcTFC/results_segeval/pairs.csv'
OUTDIR = '/home/share/clr/share/work/DLcTFC/paper_results/01_segmentation'
FOV_MM = 350.0          # assumed 512 px FOV (no pixel spacing in source DICOM)
MM_PER_PX = FOV_MM / 512.0

def boundary(mask: np.ndarray) -> np.ndarray:
    er = ndimage.binary_erosion(mask)
    return mask & ~er

def hd95(gt: np.ndarray, pred: np.ndarray) -> float:
    bg, bp = gt > 0, pred > 0
    if bg.sum() == 0 or bp.sum() == 0:
        return float('nan')
    sb, sp = boundary(bg), boundary(bp)
    if sb.sum() == 0 or sp.sum() == 0:
        return float('nan')
    dt_bg = ndimage.distance_transform_edt(~bg)
    dt_bp = ndimage.distance_transform_edt(~bp)
    d1 = dt_bg[sp]   # pred boundary -> gt
    d2 = dt_bp[sb]   # gt boundary -> pred
    return float(np.percentile(np.concatenate([d1, d2]), 95))

def metrics(gt, pred):
    inter = (gt & pred).sum()
    union = (gt | pred).sum()
    dice = 2 * inter / (gt.sum() + pred.sum()) if (gt.sum() + pred.sum()) else float('nan')
    iou = inter / union if union else float('nan')
    prec = inter / pred.sum() if pred.sum() else float('nan')
    rec = inter / gt.sum() if gt.sum() else float('nan')
    return dice, iou, prec, rec, hd95(gt, pred)

def main():
    pairs = pd.read_csv(PAIRS, encoding='utf-8-sig')
    rows = []
    for split in ['training', 'validation', 'test']:
        sub = pairs[(pairs['split'] == split) & (pairs['status'] == 'ok')]
        for _, r in sub.iterrows():
            gt_path = os.path.join(GT_ROOT, 'annotations', split, f"{r['stem']}.png")
            if not os.path.isfile(gt_path):
                continue
            gt = np.array(Image.open(gt_path)) > 0
            pred = np.array(Image.open(r['pred_path'])) > 0
            d, i, p, rc, h = metrics(gt, pred)
            rows.append(dict(split=split, stem=r['stem'], dice=d, iou=i,
                             precision=p, recall=rc, hd95_px=h,
                             hd95_mm=h * MM_PER_PX, gt_px=int(gt.sum()), pred_px=int(pred.sum())))
    df = pd.DataFrame(rows)
    os.makedirs(OUTDIR, exist_ok=True)
    df.to_csv(os.path.join(OUTDIR, 'seg_metrics_all_splits.csv'), index=False)

    summary = {}
    for split in ['training', 'validation', 'test']:
        s = df[df['split'] == split]
        if len(s) == 0:
            continue
        summary[split] = {
            'n_frames': int(len(s)),
            'dice_mean': round(float(s['dice'].mean()), 4),
            'dice_std': round(float(s['dice'].std()), 4),
            'sensitivity_mean': round(float(s['recall'].mean()), 4),
            'precision_mean': round(float(s['precision'].mean()), 4),
            'iou_mean': round(float(s['iou'].mean()), 4),
            'hd95_px_mean': round(float(s['hd95_px'].mean()), 4),
            'hd95_px_median': round(float(s['hd95_px'].median()), 4),
            'hd95_mm_mean': round(float(s['hd95_mm'].mean()), 4),
            'hd95_mm_median': round(float(s['hd95_mm'].median()), 4),
            'fov_assumption_mm': FOV_MM,
        }
    with open(os.path.join(OUTDIR, 'seg_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))

if __name__ == '__main__':
    main()
