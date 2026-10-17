#!/usr/bin/env python3
"""生成 Figure 6（1x2 双面板，对应正文 6(a)/6(b)）。

6(a) Bland-Altman：散点按差值方向着色（红 = DL > manual 高估，
    蓝 = DL < manual 低估），bias 线用近黑色避免与红点混淆；
6(b) 散点：镜像配对双色（蓝 = (manual, DL)，红 = (DL, manual)，
    橙实线回归，y=x 灰虚线）。

fig6 逻辑与 paper_scripts/run_clinical.py 的 5.1 节完全一致：
真实金标准（人工 cTFC 计数，列规格见 data_specs/README.md §2）到位后，
只需换 --gt 参数重跑，走同一条代码路径。

用法（在 tfc/ 目录下）:
    python paper_scripts/make_fig6.py \
        --gt data_specs/gt_ctfc.csv \
        --results paper_results/02_ctfc/ctfc_cohort.csv \
        --outdir paper_results/03_clinical
输出: fig6.png / fig6.tif (300dpi) / fig6_stats.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from run_clinical import match  # 与 run_clinical.py 同目录

EXPERT_COLS = ['ctfc_manual_1', 'ctfc_manual_2', 'ctfc_manual_3']
C_BLUE = '#1f4e9c'   # 深蓝
C_RED = '#d62728'    # 红
C_BIAS = '#222222'   # 近黑 bias 线（避免与红点混淆）
C_LOA = '#1a9641'    # 绿 LoA 线
C_YX = '#555555'     # 深灰 y=x
C_FIT = '#ff7f0e'    # 橙 回归线


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True, help='金标准 CSV（含 ctfc_manual_1/2/3）')
    ap.add_argument('--results', required=True, help='ctfc_cohort.csv 路径')
    ap.add_argument('--outdir', default='paper_results/03_clinical')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    auto = pd.read_csv(args.results, encoding='utf-8-sig')
    auto = auto[auto['valid'] == True] if 'valid' in auto.columns else auto
    gt = pd.read_csv(args.gt, encoding='utf-8-sig')
    for c in EXPERT_COLS:
        gt[c] = pd.to_numeric(gt[c], errors='coerce')
    gt['manual_ref'] = gt[EXPERT_COLS].mean(axis=1)

    m = match(auto, gt)
    m = m.dropna(subset=['manual_ref', 'ctfc_auto'])
    diff = m['ctfc_auto'] - m['manual_ref']
    meanv = (m['ctfc_auto'] + m['manual_ref']) / 2
    bias = float(diff.mean())
    loa = 1.96 * float(diff.std(ddof=1))
    r, p = stats.pearsonr(m['ctfc_auto'], m['manual_ref'])
    slope, intercept, r_resid, p_resid, se_slope = stats.linregress(meanv, diff)

    summary = {
        'n': int(len(m)),
        'bias': round(bias, 2),
        'loa_lower': round(bias - loa, 2),
        'loa_upper': round(bias + loa, 2),
        'pearson_r': round(float(r), 3),
        'pearson_p': float(p),
        'proportional_bias_slope': round(float(slope), 3),
        'proportional_bias_p': float(p_resid),
    }

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))

    # ---------- 6(a) Bland-Altman（按差值方向着色） ----------
    a0 = ax[0]
    a0.scatter(meanv[diff >= 0], diff[diff >= 0], s=16, alpha=0.6,
               color=C_RED, label='DL > manual')
    a0.scatter(meanv[diff < 0], diff[diff < 0], s=16, alpha=0.6,
               color=C_BLUE, label='DL < manual')
    a0.axhline(bias, color=C_BIAS, ls='-', lw=1.4, label=f'bias = {bias:.2f}')
    a0.axhline(bias + loa, color=C_LOA, ls='--', lw=1.2,
               label=f'+1.96SD = {bias + loa:.2f}')
    a0.axhline(bias - loa, color=C_LOA, ls='--', lw=1.2,
               label=f'-1.96SD = {bias - loa:.2f}')
    a0.set_xlabel('Mean (DL-cTFC, manual)')
    a0.set_ylabel('Difference (DL $-$ manual)')
    a0.set_title(f'Bland-Altman  (bias {bias:.2f}, '
                 f'LoA {bias - loa:.2f} to {bias + loa:.2f})')
    a0.legend(fontsize=8, loc='upper right')

    # ---------- 6(b) 散点（镜像配对双色） ----------
    a1 = ax[1]
    a1.scatter(m['manual_ref'], m['ctfc_auto'], s=16, alpha=0.6,
               color=C_BLUE, label='manual → DL')
    a1.scatter(m['ctfc_auto'], m['manual_ref'], s=16, alpha=0.6,
               color=C_RED, label='DL → manual')
    lims = [min(m['manual_ref'].min(), m['ctfc_auto'].min()),
            max(m['manual_ref'].max(), m['ctfc_auto'].max())]
    fit = stats.linregress(m['manual_ref'], m['ctfc_auto'])
    xs = np.linspace(lims[0], lims[1], 50)
    a1.plot(xs, fit.slope * xs + fit.intercept, color=C_FIT, ls='-', lw=2.0,
            label=f'DL = {fit.slope:.2f} · manual + {fit.intercept:.2f}')
    a1.plot(lims, lims, color=C_YX, ls='--', lw=1.2, label='y = x')
    a1.set_xlim(lims)
    a1.set_ylim(lims)
    a1.set_xlabel('Manual cTFC')
    a1.set_ylabel('DL-cTFC')
    ptxt = 'p < 0.001' if p < 0.001 else f'p = {p:.4f}'
    a1.set_title(f'Pearson r = {r:.3f}, {ptxt}')
    a1.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    png = os.path.join(args.outdir, 'fig6.png')
    tif = os.path.join(args.outdir, 'fig6.tif')
    fig.savefig(png, dpi=300)
    fig.savefig(tif, dpi=300)
    plt.close(fig)

    with open(os.path.join(args.outdir, 'fig6_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f'图已保存: {png} / {tif}')


if __name__ == '__main__':
    main()
