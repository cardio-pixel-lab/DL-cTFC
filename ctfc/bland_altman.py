#!/usr/bin/env python3
"""
CTFC 金标准验证（可插拔）：自动 vs 手工 CTFC 的 Bland-Altman + 散点图

金标准 CSV 列（utf-8 / utf-8-sig）:
    case_id, sequence_id, ctfc_manual [, branch]
    - branch 可缺省（单分支/RCA 场景）；取值 LAD / LCX / RCA
    - case_id/sequence_id 允许目录名或纯 ID（按前导数字 / IM 编号归一化匹配）

用法:
    python bland_altman.py --gt gt.csv --results ctfc_results.json \\
        --outdir ./ba --prefix ctfc
输出:
    {prefix}_bland_altman.png / .tif (600dpi), {prefix}_scatter.png / .tif,
    {prefix}_stats.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ctfc_analyzer import CTFCDatasetProcessor


def load_ground_truth(path: str) -> pd.DataFrame:
    """读取金标准 CSV，返回 [case_id, sequence_id, ctfc_manual, branch]（branch 可为空）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"金标准文件不存在: {path}")
    df = pd.read_csv(path, encoding='utf-8-sig')
    headers = {str(h).strip().lower() for h in df.columns}
    required = {'case_id', 'sequence_id', 'ctfc_manual'}
    missing = required - headers
    if missing:
        raise ValueError(f"金标准 CSV 缺少列: {sorted(missing)}（现有列: {list(df.columns)}）")

    out = pd.DataFrame({
        'case_id': df['case_id'].astype(str).str.strip(),
        'sequence_id': df['sequence_id'].astype(str).str.strip(),
        'ctfc_manual': pd.to_numeric(df['ctfc_manual'], errors='coerce'),
    })
    branch_col = next((c for c in ('branch', 'branch_type')
                       if str(c).strip().lower() in headers), None)
    if branch_col:
        out['branch'] = df[branch_col].astype(str).str.strip().str.upper()
    else:
        out['branch'] = ''
    n_bad = int(out['ctfc_manual'].isna().sum())
    if n_bad:
        print(f"警告: {n_bad} 行 ctfc_manual 无法解析为数字，已剔除")
        out = out.dropna(subset=['ctfc_manual'])
    return out.reset_index(drop=True)


def build_auto_lookup(results: Dict) -> Dict[Tuple[str, int], List[Tuple[str, float]]]:
    """results(dict) -> {(norm_case, seq_num): [(branch_type, ctfc), ...]}"""
    lookup: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}
    for case_id, sequences in results.items():
        norm_case = CTFCDatasetProcessor._norm_case_id(case_id)
        for seq_id, a in sequences.items():
            seq_num = CTFCDatasetProcessor._extract_seq_id(seq_id)
            if seq_num is None or a.ctfc is None:
                continue
            lookup.setdefault((norm_case, seq_num), []).append((a.branch_type, float(a.ctfc)))
    return lookup


def match_pairs(gt: pd.DataFrame,
                lookup: Dict[Tuple[str, int], List[Tuple[str, float]]]
                ) -> Tuple[List[Tuple], List[str]]:
    """逐行匹配金标准与自动结果。返回 (pairs, unmatched)。
    pairs 元素: (case_id, seq_id, branch, auto, manual)"""
    pairs: List[Tuple] = []
    unmatched: List[str] = []
    for _, row in gt.iterrows():
        norm_case = CTFCDatasetProcessor._norm_case_id(row['case_id'])
        seq_num = CTFCDatasetProcessor._extract_seq_id(row['sequence_id'])
        if seq_num is None:
            unmatched.append(f"{row['case_id']}/{row['sequence_id']}: 无法解析序列号")
            continue
        entries = lookup.get((norm_case, seq_num))
        if not entries:
            unmatched.append(f"{row['case_id']}/{row['sequence_id']}: 自动结果中无此序列(或 ctfc 缺失)")
            continue

        branch = row['branch']
        if branch:
            hit = [e for e in entries if e[0] == branch]
            if not hit:
                unmatched.append(f"{row['case_id']}/{row['sequence_id']} [{branch}]: "
                                 f"自动分支为 {[e[0] for e in entries]}，无匹配")
                continue
            auto = hit[0][1]
        else:
            rca = [e for e in entries if e[0] == 'RCA']
            if len(rca) == 1:
                auto, branch = rca[0][1], 'RCA'
            elif len(entries) == 1:
                branch, auto = entries[0]
            else:
                unmatched.append(
                    f"{row['case_id']}/{row['sequence_id']}: 自动为多分支 "
                    f"{[e[0] for e in entries]}，金标准未指定 branch，无法匹配")
                continue

        pairs.append((row['case_id'], row['sequence_id'], branch, auto, float(row['ctfc_manual'])))
    return pairs, unmatched


def compute_stats(auto: np.ndarray, manual: np.ndarray) -> Dict:
    diff = auto - manual
    mean_diff = auto.mean()
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    ci95 = 1.96 * sd / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    r = float(np.corrcoef(auto, manual)[0, 1]) if len(auto) > 1 and auto.std() > 0 and manual.std() > 0 else float('nan')
    return {
        'n': int(len(auto)),
        'bias': bias,
        'sd': sd,
        'ci95': float(ci95),
        'limits_of_agreement': [float(bias - 1.96 * sd), float(bias + 1.96 * sd)],
        'rmse': rmse,
        'pearson_r': r,
        'auto_mean': float(mean_diff),
        'manual_mean': float(manual.mean()),
        'auto_sd': float(auto.std(ddof=1)) if len(auto) > 1 else 0.0,
        'manual_sd': float(manual.std(ddof=1)) if len(manual) > 1 else 0.0,
    }


def save_fig(fig, out_dir: str, name: str) -> None:
    png = os.path.join(out_dir, name + '.png')
    tif = os.path.join(out_dir, name + '.tif')
    fig.savefig(png, dpi=150)
    fig.savefig(tif, dpi=600)
    plt.close(fig)
    print(f"图已保存: {png} / {tif}")


def plot_bland_altman(auto: np.ndarray, manual: np.ndarray, stats: Dict,
                      out_dir: str, prefix: str) -> None:
    diff = auto - manual
    mean_vals = (auto + manual) / 2.0
    lo, hi = stats['limits_of_agreement']

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(mean_vals, diff, s=28, alpha=0.8, color='tab:blue', edgecolors='none')
    ax.axhline(stats['bias'], color='black', lw=1.4,
               label=f"bias = {stats['bias']:.2f}")
    ax.axhline(lo, color='red', ls='--', lw=1.2, label=f"-1.96SD = {lo:.2f}")
    ax.axhline(hi, color='red', ls='--', lw=1.2, label=f"+1.96SD = {hi:.2f}")
    ax.set_xlabel('Mean (auto + manual) / 2')
    ax.set_ylabel('Difference (auto − manual)')
    ax.set_title(f"Bland-Altman (n={stats['n']}, CI95 bias ± {stats['ci95']:.2f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir, f"{prefix}_bland_altman")


def plot_scatter(auto: np.ndarray, manual: np.ndarray, stats: Dict,
                 out_dir: str, prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(manual, auto, s=28, alpha=0.8, color='tab:blue', edgecolors='none')
    lims = [min(manual.min(), auto.min()) - 1, max(manual.max(), auto.max()) + 1]
    ax.plot(lims, lims, color='gray', lw=1.2, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Manual CTFC')
    ax.set_ylabel('Auto CTFC')
    ax.set_title(f"Scatter (Pearson r = {stats['pearson_r']:.3f}, n={stats['n']})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir, f"{prefix}_scatter")


def main():
    parser = argparse.ArgumentParser(description='CTFC 金标准 Bland-Altman 验证')
    parser.add_argument('--gt', required=True, type=str, help='金标准 CSV 路径')
    parser.add_argument('--results', required=True, type=str,
                        help='自动结果 JSON（main.py 生成的 ctfc_results.json）')
    parser.add_argument('--outdir', default='./bland_altman', type=str, help='输出目录')
    parser.add_argument('--prefix', default='ctfc', type=str, help='输出文件名前缀')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.results, 'r', encoding='utf-8') as f:
        results = json.load(f)

    gt = load_ground_truth(args.gt)
    lookup = build_auto_lookup(results)
    pairs, unmatched = match_pairs(gt, lookup)

    print(f"金标准行数: {len(gt)}, 匹配成功: {len(pairs)}, 未匹配: {len(unmatched)}")
    for u in unmatched:
        print(f"  [未匹配] {u}")

    if len(pairs) < 3:
        print("错误: 有效匹配对 < 3，无法做 Bland-Altman 分析")
        sys.exit(1)

    auto = np.array([p[3] for p in pairs])
    manual = np.array([p[4] for p in pairs])
    stats = compute_stats(auto, manual)

    print("\n" + "=" * 60)
    print(f"n = {stats['n']}")
    print(f"bias (auto − manual) = {stats['bias']:.2f} ± {stats['sd']:.2f} SD")
    print(f"bias CI95 = [{stats['bias'] - stats['ci95']:.2f}, {stats['bias'] + stats['ci95']:.2f}]")
    print(f"LOA (±1.96SD) = [{stats['limits_of_agreement'][0]:.2f}, "
          f"{stats['limits_of_agreement'][1]:.2f}]")
    print(f"RMSE = {stats['rmse']:.2f}")
    print(f"Pearson r = {stats['pearson_r']:.3f}")
    print(f"auto: mean={stats['auto_mean']:.2f} ± {stats['auto_sd']:.2f}")
    print(f"manual: mean={stats['manual_mean']:.2f} ± {stats['manual_sd']:.2f}")
    print("=" * 60)

    plot_bland_altman(auto, manual, stats, args.outdir, args.prefix)
    plot_scatter(auto, manual, stats, args.outdir, args.prefix)

    stats_file = os.path.join(args.outdir, f"{args.prefix}_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({'stats': stats,
                   'pairs': [{'case_id': p[0], 'sequence_id': p[1], 'branch': p[2],
                              'auto': p[3], 'manual': p[4]} for p in pairs],
                   'unmatched': unmatched}, f, ensure_ascii=False, indent=2)
    print(f"统计已保存: {stats_file}")


if __name__ == '__main__':
    main()
