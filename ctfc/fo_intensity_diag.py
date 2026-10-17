#!/usr/bin/env python3
"""判定实验：E = M[Fmax] 里 `FO == Fmax` 的像素，早帧到底有没有造影？

背景：`_pred.png` 对真值的空间 Dice ≈0.92（见 `eval_seg_dice.py`），但时序抖动。
F1/F2 偏晚/被钉在 Fmax 的嫌疑是：E 里大比例像素直到 Fmax 才被分割出来，其 FO 必等于
Fmax（sentinel）。这些像素分两种可能，直接决定下一步：

- **早帧本来就没有造影** → FO 是对的，问题只在「E 用单帧定义」，改 E 的像素集合（方向 B）即可。
- **早帧其实已有造影、只是没被分割出来** → 早帧召回不足，改 E 治不了，得换到底物（方向 C）。

做法（不依赖分割掩码，纯灰度）：对序列每一帧算背景 `bg_f = median(I_f)`，暗度
`dark = bg_f - I_f`，再按该帧 99 分位暗度归一化得 `dn ∈ [0,1]`（≈1 为最暗的血管芯）。
然后按掩码给的 `fo`/`E` 把像素分三组，看各组「平均归一化暗度」随帧的曲线：

- 组1 `E & FO<Fmax`：掩码认为早到——曲线应早早抬高（阳性对照）
- 组2 `E & FO==Fmax`：**疑点**——若早帧曲线贴着背景，就是「本来没有」；若明显高于背景，就是「有但没分出来」
- 组3 `非E`：背景对照

再算一个模型无关的“强度到达帧” `FO_int`（`dn>τ` 连续 2 帧的首帧），与 `fo` 对比。

只读数据；结果写 `--out`（默认 `results_diag/fo_intensity`）。

用法:
  python fo_intensity_diag.py                      # 自动挑疑点/对照各若干
  python fo_intensity_diag.py --items 1-Dengxiao__1-IM000093_LAD ...
"""

import argparse
import csv
import glob
import json
import os
import re
from typing import Dict, List, Optional

import cv2
import numpy as np

ROOTS = [
    '/home/share/clr/share/data/AI_TMPFC/CAD-ID1-ID64',
    '/home/share/clr/share/data/AI_TMPFC/CMVD',
    '/home/share/clr/share/data/AI_TMPFC/CMVD-ID1-ID50',
    '/home/share/clr/share/data/AI_TMPFC/Others-ID1-ID27',
]
TAU = 0.30          # 归一化暗度阈值（判定“这像素已经有造影”）
CONSEC = 2          # 连续帧数（与掩码 FO 的迟滞一致）


def load_frames(case_id: str, sequence_id: str) -> Optional[List[np.ndarray]]:
    """按 f 号排序读出该序列全部 `_src.png`。"""
    for root in ROOTS:
        for sub in ('segmented3c2', 'segmented'):
            d = os.path.join(root, case_id, sequence_id, sub)
            if not os.path.isdir(d):
                continue
            files = []
            for fn in os.listdir(d):
                m = re.match(r'^.*-f(\d+)_src\.png$', fn)
                if m:
                    files.append((int(m.group(1)), os.path.join(d, fn)))
            if not files:
                continue
            files.sort()
            imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for _, p in files]
            if any(i is None for i in imgs):
                continue
            return imgs
    return None


def darkness_stack(imgs: List[np.ndarray]) -> np.ndarray:
    """(T,H,W) 归一化暗度，1.0 ≈ 该帧最暗的血管芯，0 ≈ 背景。"""
    I = np.stack(imgs).astype(np.float32)
    bg = np.median(I, axis=(1, 2))                     # (T,)
    dark = bg[:, None, None] - I
    scale = np.percentile(dark, 99, axis=(1, 2))       # (T,)
    scale = np.maximum(scale, 1.0)
    return dark / scale[:, None, None]


def first_arrival(dn: np.ndarray, tau: float = TAU, consec: int = CONSEC) -> np.ndarray:
    """模型无关的强度到达帧：dn>tau 连续 consec 帧的首帧。"""
    T = dn.shape[0]
    hit = dn > tau
    if consec > 1:
        hit = np.logical_and.reduce([hit[i:T - consec + 1 + i] for i in range(consec)])
        first = np.full(dn.shape[1:], -1, np.int16)
        idx = hit.argmax(axis=0)
        ok = hit.any(axis=0)
        first[ok] = idx[ok]
        return first
    first = np.full(dn.shape[1:], -1, np.int16)
    idx = hit.argmax(axis=0)
    ok = hit.any(axis=0)
    first[ok] = idx[ok]
    return first


def analyze(name: str, ddir: str, logger=print) -> Optional[Dict]:
    j = json.load(open(os.path.join(ddir, name + '.json')))
    z = np.load(os.path.join(ddir, name + '.npz'))
    E = z['fmax_mask'].astype(bool)
    fo = z['fo'].astype(np.int32)
    fmax = int(j['fmax'])
    imgs = load_frames(j['case_id'], j['sequence_id'])
    if imgs is None:
        logger(f'  [跳过] {name}: 找不到 _src 帧')
        return None
    if imgs[0].shape != E.shape:
        logger(f'  [跳过] {name}: src {imgs[0].shape} 与掩码 {E.shape} 不一致')
        return None

    T = len(imgs)
    dn = darkness_stack(imgs)
    g_early = E & (fo < fmax)
    g_susp = E & (fo == fmax)
    g_bg = ~E

    fo_int = first_arrival(dn)

    def curve(mask):
        if not mask.any():
            return None
        return dn[:, mask].mean(axis=1)     # (T,)

    c_e, c_s, c_b = (curve(g) for g in (g_early, g_susp, g_bg))
    early = slice(0, max(1, fmax))          # Fmax 之前（含 Fmax）

    def at(curve_arr, sl):
        return float(curve_arr[sl].mean()) if curve_arr is not None else float('nan')

    # 疑点组早帧暗度 vs 背景组早帧暗度：差多少倍标准差
    res = {
        'name': name,
        'case_id': j['case_id'],
        'sequence_id': j['sequence_id'],
        'branch_type': j['branch_type'],
        'n_frames': int(j['n_frames']), 'T_imgs': T,
        'fmax': fmax, 'f1': j['f1'], 'f2': j['f2'],
        'n_E': int(E.sum()),
        'frac_fo_eq_fmax': round(float((fo[E] == fmax).mean()), 4) if E.any() else None,
        'n_early': int(g_early.sum()), 'n_susp': int(g_susp.sum()),
        'dn_before_fmax': {
            'early': round(at(c_e, early), 4),
            'susp': round(at(c_s, early), 4),
            'background': round(at(c_b, early), 4),
        },
        'dn_after_fmax': {
            'early': round(at(c_e, slice(fmax + 1, T)), 4),
            'susp': round(at(c_s, slice(fmax + 1, T)), 4),
            'background': round(at(c_b, slice(fmax + 1, T)), 4),
        },
    }
    # 疑点组里“强度上早就到了”的比例
    if g_susp.any():
        fi = fo_int[g_susp]
        res['susp_fo_int_median'] = int(np.median(fi[fi >= 0])) if (fi >= 0).any() else -1
        res['susp_frac_arrived_intensity'] = round(float((fi >= 0).mean()), 4)
        res['susp_frac_int_le_fmax_minus3'] = round(float((fi <= fmax - 3).mean()), 4)
    if g_bg.any():
        res['bg_frac_arrived_intensity'] = round(float((fo_int[g_bg] >= 0).mean()), 4)

    res['_curves'] = {
        'early': None if c_e is None else [round(float(x), 4) for x in c_e],
        'susp': None if c_s is None else [round(float(x), 4) for x in c_s],
        'background': None if c_b is None else [round(float(x), 4) for x in c_b],
    }
    return res


def plot(res: Dict, out_png: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    cs = res['_curves']
    T = len(cs['background'])
    x = np.arange(T)
    for key, color, label in [('early', '#2b8a3e', f"E & FO<Fmax (n={res['n_early']})"),
                              ('susp', '#e03131', f"E & FO==Fmax 疑点 (n={res['n_susp']})"),
                              ('background', '#868e96', f"非 E 背景 (n=?)")]:
        y = cs.get(key)
        if y:
            ax.plot(x, y, color=color, lw=1.8, label=label)
    ax.axvline(res['fmax'], color='#1971c2', ls='--', lw=1.2,
               label=f"Fmax={res['fmax']}")
    for f1f2, c in [(res['f1'], '#f59f00'), (res['f2'], '#e8590c')]:
        if f1f2 is not None and f1f2 >= 0:
            ax.axvline(f1f2, color=c, ls=':', lw=1.2)
    ax.set_xlabel('frame index')
    ax.set_ylabel('mean normalized darkness')
    ax.set_title(f"{res['name']}  (F1={res['f1']} F2={res['f2']} Fmax={res['fmax']}, "
                 f"FO==Fmax 占 E 的 {res['frac_fo_eq_fmax']:.0%})")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def pick_items(csv_path: str, ddir: str, top: int, ctrl: int, logger=print):
    """从已有结果里挑：FO==Fmax 占比最高的若干个 = 疑点；最低且有效的若干个 = 对照。"""
    rows = []
    for name in os.listdir(ddir):
        if not name.endswith('.npz'):
            continue
        stem = name[:-4]
        if not stem.startswith('1-Dengxiao'):
            continue
        z = np.load(os.path.join(ddir, name))
        E = z['fmax_mask'].astype(bool)
        fo = z['fo']
        if not E.any():
            continue
        j = json.load(open(os.path.join(ddir, stem + '.json')))
        rows.append((float((fo[E] == j['fmax']).mean()), stem))
    rows.sort(key=lambda t: -t[0])
    return [s for _, s in rows[:top]], [s for _, s in rows[-ctrl:]]


def main():
    ap = argparse.ArgumentParser(description='判定：E 内 FO==Fmax 的像素早帧有没有造影')
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--detailed', default=os.path.join(here, 'results_full_v2', 'detailed'))
    ap.add_argument('--out', default=os.path.join(here, 'results_diag', 'fo_intensity'))
    ap.add_argument('--items', nargs='*', default=None)
    ap.add_argument('--top', type=int, default=3, help='自动挑 FO==Fmax 占比最高的几条')
    ap.add_argument('--ctrl', type=int, default=2, help='自动挑占比最低的几条做对照')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = lambda m: print(m, flush=True)

    if args.items:
        names, ctrl_names = list(args.items), []
    else:
        names, ctrl_names = pick_items('', args.detailed, args.top, args.ctrl, log)
        log(f'自动挑选：疑点 {names}；对照 {ctrl_names}')

    results = []
    for n in names + ctrl_names:
        log(f'分析 {n} ...')
        r = analyze(n, args.detailed, log)
        if r:
            results.append(r)
            png = os.path.join(args.out, n + '.png')
            plot(r, png)
            log(f"  E={r['n_E']} FO==Fmax={r['frac_fo_eq_fmax']:.0%} | "
                f"Fmax 前平均暗度 疑点={r['dn_before_fmax']['susp']:.3f} "
                f"早到组={r['dn_before_fmax']['early']:.3f} 背景={r['dn_before_fmax']['background']:.3f}"
                + (f" | 疑点组强度到达帧中位={r.get('susp_fo_int_median')}"
                   f" 其中 {r.get('susp_frac_int_le_fmax_minus3', 0):.0%} 早于 Fmax-3"
                   if 'susp_fo_int_median' in r else ''))
            log(f'  曲线图 {png}')

    out_json = os.path.join(args.out, 'summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f'\n汇总: {out_json}')


if __name__ == '__main__':
    main()
