#!/usr/bin/env python3
"""Fig 4: representative coronary segmentation examples (3 LCA + 1 RCA).
Panels per case: (a) raw, (b) GT mask, (c) pred overlay (yellow LAD, orange LCX, magenta FP, blue FN).
LCA cases come from segeval-accepted test pairs (pred_path authoritative).
RCA: no accepted rcag GT pairs -> pred-only row (GT panel marked TBD).
"""
import os, glob, re
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GT_ROOT = '/home/share/clr/share/data/cag_seg_v6'
PAIRS = '/home/share/clr/share/work/utils/tfc/results_segeval/pairs.csv'
METRICS = '/home/share/clr/share/work/utils/tfc/paper_results/01_segmentation/seg_metrics_all_splits.csv'
CMVD_CTFC = '/home/share/clr/share/work/utils/tfc/paper_results/02_ctfc/full_rerun/CMVD-ID1-ID50/ctfc_results.csv'
OUT = '/home/share/clr/share/work/utils/tfc/paper_results/01_segmentation/fig4'

def load(p): return np.array(Image.open(p))

def overlay(base, gt_bin, pred):
    b = base.astype(float)
    def blend(mask_arr, color):
        out = b.copy(); out[mask_arr] = out[mask_arr]*0.35 + np.array(color)*0.65; return out
    pred_bin = pred > 0
    if set(np.unique(pred)) <= {0, 128, 255} and 128 in np.unique(pred):
        p = b.copy()
        lad = pred == 255; lcxx = pred == 128
        p[lad] = p[lad]*0.3 + np.array([255,255,0])*0.7
        p[lcxx] = p[lcxx]*0.3 + np.array([255,165,0])*0.7
    else:
        p = blend(pred_bin, (0,255,0))
    fp = pred_bin & ~gt_bin; fn = gt_bin & ~pred_bin
    p[fp] = p[fp]*0.3 + np.array([255,0,255])*0.7
    p[fn] = p[fn]*0.3 + np.array([0,128,255])*0.7
    return p.astype(np.uint8)

def make_row(tag, title, src_path, gt_path, pred_path_3c, pred_path_1c):
    src = Image.open(src_path).convert('RGB')
    base = np.asarray(src)
    if gt_path and os.path.isfile(gt_path):
        gt = load(gt_path) > 0
    else:
        gt = None
    pred3 = load(pred_path_3c) if (pred_path_3c and os.path.isfile(pred_path_3c)) else None
    pred1 = load(pred_path_1c) if (pred_path_1c and os.path.isfile(pred_path_1c)) else None
    pred = pred3 if pred3 is not None else pred1
    if gt is None:
        panel_gt = None
    else:
        w = gt[..., None].astype(float) * 0.65
        panel_gt = (base.astype(float) * (1 - w) + np.array([255, 255, 255]) * w).astype(np.uint8)
    if gt is None:
        gt_bin = pred > 0
    else:
        gt_bin = gt
    panel_pred = overlay(base, gt_bin, pred) if pred is not None else None
    return dict(tag=tag, title=title, raw=base, gt=panel_gt, pred=panel_pred)

def main():
    os.makedirs(OUT, exist_ok=True)
    met = pd.read_csv(METRICS); pairs = pd.read_csv(PAIRS, encoding='utf-8-sig')
    test = met[(met['split']=='test')].merge(
        pairs[(pairs['split']=='test') & (pairs['status']=='ok')][['stem','pred_path']], on='stem')
    test['case'] = test['stem'].str.extract(r'-c(\d+)-')[0]
    test['seq'] = test['stem'].str.extract(r'-s(\d+)-')[0].astype(int)
    pool = test[test['dice'] > 0.92]
    rows = []
    # 3 LCA rows: one sequence per patient (c56/c57/c58), peak-opacification frame per sequence
    for case in sorted(pool['case'].unique(), key=int):
        pc = pool[pool['case'] == case]
        best_seq = pc.groupby('seq')['gt_px'].max().idxmax()
        r = pc[pc['seq'] == best_seq].sort_values('gt_px', ascending=False).iloc[0]
        pred_path = r['pred_path']
        src_path = pred_path.replace('_pred.png', '_src.png')
        d3c = os.path.join(os.path.dirname(pred_path).replace('/segmented/', '/segmented3c2/'),
                           os.path.basename(pred_path))
        if not (os.path.isfile(src_path) and os.path.isfile(pred_path)): continue
        rows.append(make_row(chr(ord('A')+len(rows)), f'LCA (case {case})', src_path,
                             os.path.join(GT_ROOT,'annotations','test',f"{r['stem']}.png"), d3c, pred_path))
        if len(rows) == 3: break
    # RCA row from CMVD run (slow flow cohort); GT-style naming cag2-c{case}-s{seq}-f{frame}
    # T<500 excludes the T=512/1024 striped bad frames; area band = clean vessel-scale opacification
    ct = pd.read_csv(CMVD_CTFC, encoding='utf-8-sig')
    rc = ct[(ct['分支类型']=='RCA') & (ct['有效性']=='有效') & (ct['总帧数']<500)
            & (ct['最大非零像素数']>=2000) & (ct['最大非零像素数']<=40000)].sort_values('最大非零像素数', ascending=False)
    rca_row = None
    for _, c in rc.iterrows():
        seq_name = str(c['序列ID']).rsplit('_',1)[0]
        case_dir = str(c['病例ID'])
        fmax = int(c['Fmax'])
        cnum = case_dir.split('-')[0]
        snum = int(str(seq_name).split('IM')[-1])
        cands = []
        for root in ['CMVD-ID1-ID50', 'CMVD']:
            cands += glob.glob(f'/home/share/clr/share/data/AI_TMPFC/{root}/{case_dir}/{seq_name}/segmented/cag2-c{cnum}-s{snum}-f{fmax}_pred.png')
            cands += glob.glob(f'/home/share/clr/share/data/AI_TMPFC/{root}/{case_dir}/{seq_name}/segmented3c2/*{snum}*f{fmax:02d}_pred.png')
            cands += glob.glob(f'/home/share/clr/share/data/AI_TMPFC/{root}/{case_dir}/{seq_name}/segmented/*f{fmax:02d}_pred.png')
        if cands:
            src_path = cands[0].replace('_pred.png','_src.png')
            if os.path.isfile(src_path):
                rca_row = make_row('D','RCA (CMVD)', src_path, None, None, cands[0]); break
    if rca_row: rows.append(rca_row)

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(11.5, 2.9*n))
    for i, row in enumerate(rows):
        for j, key in enumerate(['raw','gt','pred']):
            ax = axes[i][j]
            im = row[key]
            if im is None:
                ax.text(0.5, 0.5, 'no GT available\n(rotational angiography batch)', ha='center',
                        va='center', fontsize=9); ax.axis('off')
            else:
                ax.imshow(im); ax.axis('off')
    for j, lab in enumerate(['(a) raw angiogram','(b) ground truth',
                             '(c) prediction: yellow LAD, orange LCX, green RCA; magenta FP, blue FN']):
        axes[0][j].set_title(lab, fontsize=10)
    fig.suptitle('Figure 4. Representative examples of the coronary segmentation results (test set)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, 'fig4.png'), dpi=300)
    print('saved', os.path.join(OUT, 'fig4.png'), [ (r['tag'], r['title']) for r in rows ])

if __name__ == '__main__':
    main()
