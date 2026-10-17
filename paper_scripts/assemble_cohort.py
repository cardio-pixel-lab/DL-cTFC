#!/usr/bin/env python3
"""Assemble the per-branch auto-CTFC table for the three cohorts into one
normalized file that paper_scripts/run_clinical.py consumes.

Output columns (run_clinical.py contract):
  case_id, sequence_id, branch, cohort, ctfc_auto, valid, f1, f2, fmax,
  n_frames, fmax_area, note
"""
import argparse, os
import pandas as pd

BASE = '/home/share/clr/share/work/utils/tfc/paper_results/02_ctfc/full_rerun'
QUEUES = [
    ('CAD', f'{BASE}/CAD-ID1-ID64/ctfc_results.csv'),
    ('CMVD', f'{BASE}/CMVD-ID1-ID50/ctfc_results.csv'),
    ('Others', f'{BASE}/Others-ID1-ID27/ctfc_results.csv'),
]
COLMAP = {
    '病例ID': 'case_id', '序列ID': 'sequence_id', '分支类型': 'branch',
    '总帧数': 'n_frames', 'Fmax': 'fmax', '最大非零像素数': 'fmax_area',
    'F1': 'f1', 'F2': 'f2', 'CTFC(浮点)': 'ctfc_auto', '有效性': 'valid_raw',
    '备注': 'note',
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='/home/share/clr/share/work/utils/tfc/paper_results/02_ctfc')
    args = ap.parse_args()
    frames = []
    for cohort, path in QUEUES:
        if not os.path.isfile(path):
            print(f'MISSING {path} — skipped')
            continue
        df = pd.read_csv(path, encoding='utf-8-sig')
        df = df.rename(columns=COLMAP)
        df['cohort'] = cohort
        df['valid'] = df['valid_raw'].eq('有效')
        keep = [c for c in ['case_id', 'sequence_id', 'branch', 'cohort', 'ctfc_auto',
                            'valid', 'f1', 'f2', 'fmax', 'n_frames', 'fmax_area', 'note'] if c in df]
        frames.append(df[keep])
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(args.outdir, exist_ok=True)
    dest = os.path.join(args.outdir, 'ctfc_cohort.csv')
    out.to_csv(dest, index=False, encoding='utf-8-sig')
    summ = (out.groupby(['cohort', 'branch'])
              .agg(rows=('case_id', 'size'), valid=('valid', 'sum'),
                   ctfc_median=('ctfc_auto', 'median'))
              .reset_index())
    summ.to_csv(os.path.join(args.outdir, 'ctfc_cohort_summary.csv'), index=False, encoding='utf-8-sig')
    print(summ.to_string(index=False))
    print(f'saved {dest} ({len(out)} rows)')

if __name__ == '__main__':
    main()
