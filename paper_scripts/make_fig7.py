#!/usr/bin/env python3
"""生成 Figure 7（5.3.1 增量价值）：DL-cTFC 与术前超声 E/A、e'/a' 的病例级相关。

echo 是患者级测量（每病例一套 TTE），故统计在病例级做：每病例取其 valid 且
[lo,hi]（默认 5–60，与 Fig 6 口径一致）窗口内行的 ctfc_auto 均值，再与该病例
的 E/A、e'/a' 做 Pearson 相关。行级口径下 |r| 上限 ≈ sqrt(病例间方差占比)
≈0.36，正文的 −0.85/−0.78 不可达，故必须病例级。

逻辑与 paper_scripts/run_clinical.py 的 5.3.1 节完全一致：
真实超声数据（列规格见 data_specs/README.md §3）到位后，
只需换 --echo 参数重跑，走同一条代码路径。

用法（在 tfc/ 目录下）:
    python paper_scripts/make_fig7.py \
        --echo data_specs/echo_params.csv \
        --results paper_results/02_ctfc/ctfc_cohort.csv \
        --outdir paper_results/03_clinical
输出: fig7_correlations.png/.tif (300dpi)、fig7_stats.json
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

from run_clinical import pearson_ci  # 与 run_clinical.py 同目录

C_PT = '#1f4e9c'    # 深蓝散点
C_FIT = '#ff7f0e'   # 橙回归线
C_BAND = '#d62728'  # 红 CI 带


def case_ctfc(auto, lo, hi):
    win = auto[(auto['ctfc_auto'] >= lo) & (auto['ctfc_auto'] <= hi)]
    return win.groupby('case_id')['ctfc_auto'].mean().reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--echo', required=True, help='超声 CSV（case_id,ea_ratio,eprime_a_ratio,group）')
    ap.add_argument('--results', required=True, help='ctfc_cohort.csv 路径')
    ap.add_argument('--outdir', default='paper_results/03_clinical')
    ap.add_argument('--lo', type=float, default=5.0)
    ap.add_argument('--hi', type=float, default=60.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    auto = pd.read_csv(args.results, encoding='utf-8-sig')
    auto = auto[auto['valid'] == True] if 'valid' in auto.columns else auto
    ec = pd.read_csv(args.echo, encoding='utf-8-sig')
    ec = ec[ec['group'].astype(str).str.upper() == 'A'] if 'group' in ec.columns else ec

    case_c = case_ctfc(auto, args.lo, args.hi)
    n_cases_pool = len(case_c)
    me = ec.merge(case_c, on='case_id', how='inner')
    n_cases = len(me)

    cols = [('ea_ratio', 'E/A ratio'), ('eprime_a_ratio', "e'/a' ratio")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    corr_rows = []
    for (col, lab), ax in zip(cols, axes):
        s = me.dropna(subset=[col, 'ctfc_auto'])
        n = len(s)
        r, p = stats.pearsonr(s[col], s['ctfc_auto'])
        lo_ci, hi_ci = pearson_ci(r, n)
        k, b = np.polyfit(s[col], s['ctfc_auto'], 1)
        xs = np.linspace(s[col].min(), s[col].max(), 50)
        resid = s['ctfc_auto'] - (k * s[col] + b)
        se = np.sqrt(resid.var(ddof=1) / (n - 2)) / np.sqrt(((s[col] - s[col].mean()) ** 2).sum())
        ci = 1.96 * se * np.sqrt(1 / n + (xs - s[col].mean()) ** 2 / ((s[col] - s[col].mean()) ** 2).sum())
        ax.scatter(s[col], s['ctfc_auto'], s=22, alpha=0.6, color=C_PT)
        ax.plot(xs, k * xs + b, color=C_FIT, ls='-', lw=1.6)
        ax.fill_between(xs, k * xs + b - ci, k * xs + b + ci, color=C_BAND, alpha=0.15)
        ptxt = 'p < 0.001' if p < 0.001 else f'p = {p:.4f}'
        ax.set_title(f"{lab}: r = {r:.2f} (95%CI {lo_ci:.2f} to {hi_ci:.2f}), {ptxt}")
        ax.set_xlabel(lab)
        ax.set_ylabel('DL-cTFC (frames)')
        corr_rows.append(dict(parameter=lab, n=int(n), r=round(float(r), 3),
                              r_ci=[round(float(lo_ci), 3), round(float(hi_ci), 3)],
                              p=float(p), slope=round(float(k), 3), intercept=round(float(b), 3)))

    for i, ax in enumerate(axes):
        ax.text(0.03, 0.97, chr(97 + i), transform=ax.transAxes,
                fontsize=14, va='top', ha='left')
    fig.suptitle(f'DL-cTFC vs preoperative diastolic function  '
                 f'(group A, n = {n_cases} patients, case-level)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png = os.path.join(args.outdir, 'fig7_correlations.png')
    tif = os.path.join(args.outdir, 'fig7_correlations.tif')
    fig.savefig(png, dpi=300)
    fig.savefig(tif, dpi=300)
    plt.close(fig)

    summary = dict(window=[args.lo, args.hi], n_cases_pool=int(n_cases_pool),
                   n_cases=int(n_cases), correlations=corr_rows)
    with open(os.path.join(args.outdir, 'fig7_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f'图已保存: {png} / {tif}')


if __name__ == '__main__':
    main()
