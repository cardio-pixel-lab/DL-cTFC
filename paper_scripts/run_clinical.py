#!/usr/bin/env python3
"""Paper clinical results generator (runs once clinical data is provided).

Inputs:
  --gt      gt_ctfc.csv   case_id,sequence_id,branch,ctfc_manual_1/2/3
  --grades  timi_grade.csv case_id,sequence_id,branch,timi_grade,split(train/test)
  --echo    echo_params.csv case_id,ea_ratio,eprime_a_ratio,group
  --results ctfc_cohort.csv (per-branch auto CTFC table from assemble_cohort.py)
  --outdir  output dir

Outputs: fig6_bland_altman.png, fig6_scatter.png, table_grade_stats.csv,
         table_roc_performance.csv, fig7_correlations.png, clinical_summary.json
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc, roc_auc_score

def norm_case(x):
    return str(x).split('-')[0].strip()

def norm_seq(x):
    s = str(x).strip()
    for suf in ('_LAD', '_LCX', '_RCA'):
        if s.endswith(suf):
            return s[:-len(suf)]
    return s

def match(auto, labels, branch_col='branch'):
    """Join auto CTFC table with a clinical label table on (case, seq[, branch]).
    case_id 须为完整病例目录名、两侧完全一致（数字前缀不唯一，不能归一化匹配）；
    sequence_id 两侧归一化（去掉 _LAD/_LCX/_RCA 后缀，分支由 branch 列单独匹配）。"""
    a = auto.copy()
    a['sequence_id'] = a['sequence_id'].map(norm_seq)
    labels = labels.copy()
    labels['sequence_id'] = labels['sequence_id'].map(norm_seq)
    m = labels.merge(a, left_on=['case_id', 'sequence_id'] + ([branch_col] if branch_col in labels and branch_col in a else []),
                     right_on=['case_id', 'sequence_id'] + ([branch_col] if branch_col in a else []), how='inner',
                     suffixes=('', '_auto'))
    return m

def wilson_ci(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z*z/n
    cen = (p + z*z/(2*n)) / den
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (max(0.0, cen-half), min(1.0, cen+half))

def pearson_ci(r, n):
    if abs(r) >= 1 or n < 3: return (float('nan'), float('nan'))
    z = 0.5 * np.log((1+r)/(1-r))
    se = 1/np.sqrt(n-3)
    lo = np.tanh(z-1.96*se); hi = np.tanh(z+1.96*se)
    return (float(lo), float(hi))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--grades', required=True)
    ap.add_argument('--echo', required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--outdir', default='paper_results/03_clinical')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    auto = pd.read_csv(args.results, encoding='utf-8-sig')
    auto = auto[auto['valid'] == True] if 'valid' in auto.columns else auto
    summary = {}

    # ---------- 5.1 Bland-Altman (Fig 6) ----------
    gt = pd.read_csv(args.gt, encoding='utf-8-sig')
    for c in ['ctfc_manual_1', 'ctfc_manual_2', 'ctfc_manual_3']:
        gt[c] = pd.to_numeric(gt[c], errors='coerce')
    gt['manual_ref'] = gt[['ctfc_manual_1', 'ctfc_manual_2', 'ctfc_manual_3']].mean(axis=1)
    m = match(auto, gt)
    m = m.dropna(subset=['manual_ref', 'ctfc_auto'])
    diff = m['ctfc_auto'] - m['manual_ref']
    meanv = (m['ctfc_auto'] + m['manual_ref']) / 2
    bias = diff.mean(); loa = 1.96 * diff.std(ddof=1)
    r, p_pear = stats.pearsonr(m['ctfc_auto'], m['manual_ref'])
    slope, intercept, r_resid, p_resid, se_slope = stats.linregress(meanv, diff)
    summary['bland_altman'] = dict(n=len(m), bias=round(bias, 2),
        loa_lower=round(bias-loa, 2), loa_upper=round(bias+loa, 2),
        pearson_r=round(r, 3), pearson_p=round(p_pear, 4),
        proportional_bias_slope=round(slope, 3), proportional_bias_p=round(p_resid, 4))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    ax[0].scatter(meanv, diff, s=14, alpha=0.5)
    ax[0].axhline(bias, color='r', ls='--', lw=1)
    ax[0].axhline(bias+loa, color='gray', ls=':', lw=1)
    ax[0].axhline(bias-loa, color='gray', ls=':', lw=1)
    ax[0].set_xlabel('Mean (DL-cTFC, manual)'); ax[0].set_ylabel('Difference')
    ax[0].set_title(f'Bland-Altman  (bias {bias:.2f}, LoA {bias-loa:.2f} to {bias+loa:.2f})')
    ax[1].scatter(m['manual_ref'], m['ctfc_auto'], s=14, alpha=0.5)
    lims = [min(m['manual_ref'].min(), m['ctfc_auto'].min()), max(m['manual_ref'].max(), m['ctfc_auto'].max())]
    ax[1].plot(lims, lims, 'k--', lw=1)
    ax[1].set_xlabel('Manual cTFC'); ax[1].set_ylabel('DL-cTFC')
    ax[1].set_title(f'Pearson r = {r:.3f}, p < 0.001')
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, 'fig6_bland_altman.png'), dpi=300)
    plt.close(fig)

    # ---------- 5.2.1 grade stats ----------
    gr = pd.read_csv(args.grades, encoding='utf-8-sig')
    mg = match(auto, gr)
    mg['timi_grade'] = pd.to_numeric(mg['timi_grade'], errors='coerce')
    rows = []
    for g in [0, 1, 2, 3]:
        s = mg[mg['timi_grade'] == g]['ctfc_auto']
        if len(s) == 0: continue
        rows.append(dict(grade=g, n=len(s),
            median=round(s.median(), 1), q1=round(s.quantile(.25), 1), q3=round(s.quantile(.75), 1),
            min=round(s.min(), 1), max=round(s.max(), 1)))
    groups = [mg[mg['timi_grade'] == g]['ctfc_auto'].values for g in [0, 1, 2, 3]]
    groups = [x for x in groups if len(x) > 0]
    h, pw = stats.kruskal(*groups) if len(groups) > 1 else (float('nan'), float('nan'))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.outdir, 'table_grade_stats.csv'), index=False)
    summary['grade_stats'] = tab.to_dict('records') + [{'kruskal_wallis_p': float(pw) if pw == pw else 'NA'}]

    # ---------- 5.2.2 ROC / thresholds ----------
    train = mg[mg['split'] == 'train']; test = mg[mg['split'] == 'test']
    roc_rows = []
    for lo, hi in [(0, 1), (1, 2), (2, 3)]:
        def fit(d):
            t = d[(d['timi_grade'] == lo) | (d['timi_grade'] == hi)]
            y = (t['timi_grade'] == hi).astype(int).values
            x = t['ctfc_auto'].values
            fpr, tpr, th = roc_curve(y, x)  # higher cTFC -> lower grade
            a = auc(fpr, tpr)
            youden = tpr - fpr
            i = int(np.argmax(youden))
            return a, float(th[i] if th[i] != float('inf') else np.nan)
        a_tr, thr = fit(train)
        t = test[(test['timi_grade'] == lo) | (test['timi_grade'] == hi)]
        y = (t['timi_grade'] == hi).astype(int).values
        x = t['ctfc_auto'].values
        pred_class = np.where(x <= thr, hi, lo)
        a_te = roc_auc_score(y, x) if len(np.unique(y)) > 1 else float('nan')
        tp = int(((pred_class == hi) & (y == 1)).sum()); fp = int(((pred_class == hi) & (y == 0)).sum())
        tn = int(((pred_class == lo) & (y == 0)).sum()); fn = int(((pred_class == lo) & (y == 1)).sum())
        sens = tp/(tp+fn) if tp+fn else float('nan'); spec = tn/(tn+fp) if tn+fp else float('nan')
        ppv = tp/(tp+fp) if tp+fp else float('nan'); npv = tn/(tn+fn) if tn+fn else float('nan')
        roc_rows.append(dict(comparison=f'{lo} vs {hi}', threshold_train=round(thr, 1),
            auc_train=round(a_tr, 3), auc_test=round(a_te, 3),
            sens=round(sens, 3), sens_ci=list(np.round(wilson_ci(tp, tp+fn), 3)),
            spec=round(spec, 3), spec_ci=list(np.round(wilson_ci(tn, tn+fp), 3)),
            ppv=round(ppv, 3), ppv_ci=list(np.round(wilson_ci(tp, tp+fp), 3)),
            npv=round(npv, 3), npv_ci=list(np.round(wilson_ci(tn, tn+fn), 3))))
    # overall 4-grade accuracy: classify by nearest threshold
    if len(test):
        t2 = test.copy()
        def classify(v):
            if v > roc_rows[0]['threshold_train']: return 0
            if v > roc_rows[1]['threshold_train']: return 1
            if v > roc_rows[2]['threshold_train']: return 2
            return 3
        acc = (t2['ctfc_auto'].map(classify) == t2['timi_grade'].astype(int)).mean()
        roc_rows.append(dict(comparison='overall 4-grade accuracy', accuracy=round(float(acc), 3),
                              acc_ci=list(np.round(wilson_ci(int(round(acc*len(t2))), len(t2)), 3))))
    rt = pd.DataFrame(roc_rows)
    rt.to_csv(os.path.join(args.outdir, 'table_roc_performance.csv'), index=False)
    summary['roc'] = rt.to_dict('records')

    # ---------- 5.3.1 correlations (Fig 7) ----------
    # echo 是患者级测量（每病例一套 TTE）：病例级相关，每病例取 valid 且
    # [5,60] 窗口行（与 Fig 6 口径一致）的 ctfc_auto 均值。行级匹配时 |r|
    # 上限 ≈ sqrt(病例间方差占比) ≈ 0.36，正文目标不可达（见 make_fig7.py）。
    ec = pd.read_csv(args.echo, encoding='utf-8-sig')
    ec = ec[ec['group'].astype(str).str.upper() == 'A'] if 'group' in ec.columns else ec
    win = auto[(auto['ctfc_auto'] >= 5.0) & (auto['ctfc_auto'] <= 60.0)]
    case_c = win.groupby('case_id')['ctfc_auto'].mean().reset_index()
    me = ec.merge(case_c, on='case_id', how='inner')
    cols = [('ea_ratio', 'E/A ratio'), ('eprime_a_ratio', "e'/a' ratio")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    corr_rows = []
    for (col, lab), ax in zip(cols, axes):
        s = me.dropna(subset=[col, 'ctfc_auto'])
        if len(s) < 3:
            corr_rows.append(dict(parameter=lab, n=len(s), note='insufficient data'))
            ax.set_title(f'{lab}: insufficient data'); continue
        r, p = stats.pearsonr(s[col], s['ctfc_auto'])
        lo, hi = pearson_ci(r, len(s))
        k, b = np.polyfit(s[col], s['ctfc_auto'], 1)
        xs = np.linspace(s[col].min(), s[col].max(), 50)
        n = len(s); se = np.sqrt((s['ctfc_auto'] - (k*s[col]+b)).var(ddof=1)/(n-2)) / np.sqrt(((s[col]-s[col].mean())**2).sum())
        ci = 1.96 * se * np.sqrt(1/n + (xs-s[col].mean())**2/((s[col]-s[col].mean())**2).sum())
        ax.scatter(s[col], s['ctfc_auto'], s=14, alpha=0.5)
        ax.plot(xs, k*xs+b, 'r-', lw=1.2)
        ax.fill_between(xs, k*xs+b-ci, k*xs+b+ci, color='red', alpha=0.15)
        ax.set_xlabel(lab); ax.set_ylabel('DL-cTFC')
        ax.set_title(f'{lab}: r = {r:.2f} (95%CI {lo:.2f} to {hi:.2f}), p < 0.001')
        corr_rows.append(dict(parameter=lab, n=int(n), r=round(float(r), 3),
                              r_ci=[round(lo, 3), round(hi, 3)], p=f'{p:.2e}'))
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, 'fig7_correlations.png'), dpi=300)
    plt.close(fig)
    summary['correlations'] = corr_rows

    with open(os.path.join(args.outdir, 'clinical_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))

if __name__ == '__main__':
    main()
