#!/usr/bin/env python3
"""POC：把逐像素「到达帧」从掩码 FO 换成 `_src` 灰度强度到达帧，重算 F1/F2 与 CTFC。

背景（已确立）
--------------
现算法从逐帧硬分割掩码 `_pred.png` 得到逐像素首显影帧 FO，再取
F1 = quantile(FO[P], 0.80)、F2 = quantile(FO[D], 0.10)，CTFC=(F2-F1)*2（LAD /1.7）。
掩码逐帧独立、时序抖动大，导致大量近端像素的 FO 被钳到哨兵值 Fmax，
把 F1 抬到 Fmax（实测 F1==Fmax 占 32%、F1>F2 占 40%）。
双对照实验已证明：这些 FO==Fmax 的近端像素早帧其实有造影，只是掩码没分割出来。

本脚本只做一件事：把到达帧换成**灰度强度到达帧** `fo_int`，看
(1) F1 能否脱离 Fmax；(2) F1<F2 是否改善；(3) 有效 CTFC 中位是否回到 ~20 量级。

到达帧算法（逐像素先归一化阈值与幅度，避免长序列/弱对比误触发）
--------------------------------------------------------------
- 每帧背景 `bg_t = median(I_t)`；暗度 `dark(t) = bg_t - I_t`（越暗＝造影越多）。
- 只在 `extent`（血管区）内算，别处忽略。
- 逐像素：基线 `b = median_t dark`；幅度 `A = percentile_t(dark, 99) - b`；
  要求 `A >= min_amp`（默认 3 灰度级）否则该像素记「无到达」（fo_int=-1）。
- 到达判据：`dark(t) > b + max(k*A, abs_floor)`（k=0.5, abs_floor=2.0）
  且**连续 consec 帧**（默认 2，与现有 FO 的 2 帧迟滞一致）的首帧。
- 得到 `fo_int`（512x512，未到达为 -1）。

重算
----
- `F1_new = quantile(fo_int[P & valid], 0.80)`；`F2_new = quantile(fo_int[D & valid], 0.10)`；
  有效像素 <5 记「不足」(None)。
- F2 稳健变体：q10/q20/q30/median/`median - max(2, 0.5*(median-q10))`（rob）。
- `CTFC_new = (F2_new - F1_new) * 2.0`（LAD 再 /1.7）。

样本（自动挑 20-30 条）：n_frames∈[30,150]、不在坏图名单、P/D≥30px，
分 A(F1==Fmax ~12) / B(有效且 F1<F2 ~8) / C(F2==Fmax 或 F2==0 ~6)，覆盖 LAD/LCX/RCA。

输出（均写在本脚本目录下）
- results_diag/fo_poc/compare.csv               逐条 旧 vs 新 对照
- results_diag/fo_poc/summary.json              分组汇总（旧 vs 新 + F2 变体择优）
- results_diag/fo_poc/look/<stem>.png           逐条人工核对图（5 帧 + 到达曲线）

只读原始数据；不修改任何现有文件。
"""

import argparse
import csv
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

ROOTS = [
    '/home/share/clr/share/data/AI_TMPFC/CAD-ID1-ID64',
    '/home/share/clr/share/data/AI_TMPFC/CMVD',
    '/home/share/clr/share/data/AI_TMPFC/CMVD-ID1-ID50',
    '/home/share/clr/share/data/AI_TMPFC/Others-ID1-ID27',
]
BRANCH_SUF = re.compile(r'_(LAD|LCX|RCA|LCA)$')

# 到达帧默认参数
MIN_AMP = 3.0        # 灰度幅度下限（< 视为无到达）
K_FRAC = 0.5         # 阈值 = b + k*A
ABS_FLOOR = 2.0      # 阈值绝对下限
CONSEC = 2           # 连续命中帧数（与 FO 迟滞一致）

FRAME_RATE_CORRECTION = 2.0
LAD_CORRECTION = 1.7


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
def load_frames(case_id: str, sequence_id: str) -> Tuple[Optional[List[np.ndarray]], Optional[str], Optional[str]]:
    """按解析出的 f 号（0-based）排序读出 `_src.png`。

    sequence_id 可能带 `_LAD/_LCX/RCA/LCA` 后缀而磁盘目录名没有，需剥掉重试。
    返回 (imgs, root, dir)；失败 (None, None, None)。
    """
    if cv2 is None:
        return None, None, None
    for seq in dict.fromkeys([sequence_id, BRANCH_SUF.sub('', sequence_id)]):
        for root in ROOTS:
            for sub in ('segmented3c2', 'segmented'):
                d = os.path.join(root, case_id, seq, sub)
                if not os.path.isdir(d):
                    continue
                pairs = []
                for fn in os.listdir(d):
                    m = re.match(r'^.*-f(\d+)_src\.png$', fn)
                    if m:
                        pairs.append((int(m.group(1)), os.path.join(d, fn)))
                if not pairs:
                    continue
                pairs.sort(key=lambda t: t[0])          # 按 f 号排序，不是字符串
                imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for _, p in pairs]
                if any(i is None for i in imgs):
                    continue
                return imgs, root, d
    return None, None, None


def load_bad_set(bad_csv: str) -> set:
    """坏图序列集合 {(case_id, sequence_id 去分支后缀)}（verdict == 'bad'）。"""
    bad = set()
    if not bad_csv or not os.path.isfile(bad_csv):
        return bad
    with open(bad_csv, newline='', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if (r.get('verdict') or '').strip().lower() == 'bad':
                cid = (r.get('case_id') or '').strip()
                sid = BRANCH_SUF.sub('', (r.get('sequence_id') or '').strip())
                bad.add((cid, sid))
    return bad


def _to_int(v) -> Optional[int]:
    try:
        if v is None or str(v).strip() == '':
            return None
        return int(round(float(v)))
    except Exception:
        return None


def load_rows(results_csv: str, bad: set) -> List[Dict]:
    """从 ctfc_results.csv 读旧值并做初筛（30<=T<=150、非坏图、P/D>=30px）。"""
    rows: List[Dict] = []
    if not os.path.isfile(results_csv):
        return rows
    with open(results_csv, newline='', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            case = (r.get('病例ID') or '').strip()
            seq = (r.get('序列ID') or '').strip()
            branch = (r.get('分支类型') or '').strip() or (BRANCH_SUF.search(seq).group(1) if BRANCH_SUF.search(seq) else 'RCA')
            T = _to_int(r.get('总帧数'))
            fmax = _to_int(r.get('Fmax'))
            f1 = _to_int(r.get('F1'))
            f2 = _to_int(r.get('F2'))
            nP = _to_int(r.get('近端区像素'))
            nD = _to_int(r.get('远端区像素'))
            if None in (T, fmax, f1, f2, nP, nD):
                continue
            if not (30 <= T <= 150):
                continue
            if nP < 30 or nD < 30:
                continue
            if (case, BRANCH_SUF.sub('', seq)) in bad:
                continue
            rows.append(dict(case=case, seq=seq, branch=branch, T=T, fmax=fmax,
                             f1=f1, f2=f2, nP=nP, nD=nD))
    return rows


def assign_group(r: Dict) -> Optional[str]:
    if r['f1'] == r['fmax']:
        return 'A'                       # 当前 F1 被钉在 Fmax（最该被救）
    if r['f2'] == r['fmax'] or r['f2'] == 0:
        return 'C'                       # 当前 F2 哨兵/零
    if r['f1'] < r['f2'] and r['f1'] != r['fmax']:
        return 'B'                       # 阳性对照
    return None


def pick_group(rows: List[Dict], grp: str, n: int, seed: int = 7) -> List[Dict]:
    """按分支轮转抽样，尽量覆盖 LAD/LCX/RCA。"""
    pool = [r for r in rows if r['grp'] == grp]
    by: Dict[str, List[Dict]] = {}
    for r in pool:
        by.setdefault(r['branch'], []).append(r)
    rnd = random.Random(seed)
    for b in by:
        rnd.shuffle(by[b])
    order = ['LAD', 'LCX', 'RCA', 'LCA'] + [b for b in sorted(by) if b not in ('LAD', 'LCX', 'RCA', 'LCA')]
    out: List[Dict] = []
    while len(out) < n:
        progressed = False
        for b in order:
            if b in by and by[b]:
                out.append(by[b].pop())
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    return out


# --------------------------------------------------------------------------
# 到达帧 / 分位
# --------------------------------------------------------------------------
def intensity_arrival(imgs: List[np.ndarray], extent: np.ndarray,
                      min_amp: float = MIN_AMP, k: float = K_FRAC,
                      abs_floor: float = ABS_FLOOR, consec: int = CONSEC) -> np.ndarray:
    """灰度强度到达帧 fo_int（512x512 int16，未到达 -1）。只算 extent 内像素。"""
    H, W = extent.shape
    ys, xs = np.where(extent)
    n = ys.size
    fo_int = np.full((H, W), -1, np.int16)
    if n == 0 or len(imgs) == 0:
        return fo_int
    T = len(imgs)
    dark = np.empty((T, n), np.float32)
    for t, im in enumerate(imgs):
        f = im.astype(np.float32)
        bg = float(np.median(f))
        dark[t] = bg - f[ys, xs]
    b = np.median(dark, axis=0)
    A = np.percentile(dark, 99, axis=0) - b
    thr = b + np.maximum(k * A, abs_floor)
    hit = dark > thr[None, :]
    hit[:, A < min_amp] = False                    # 弱对比像素 = 无到达
    if T >= consec:
        h = hit[0:T - consec + 1].copy()
        for j in range(1, consec):
            h &= hit[j:T - consec + 1 + j]
        any_hit = h.any(axis=0)
        idx = h.argmax(axis=0)
        vals = np.where(any_hit, idx, -1)
        fo_int[ys, xs] = vals.astype(np.int16)
    return fo_int


def plateau_quantile(values: np.ndarray, frac: float) -> Optional[int]:
    """高原感知分位（与 ctfc_analyzer._fo_quantile 一致）：最小的 v 使累计占比 >= frac。"""
    v = np.asarray(values).ravel()
    if v.size == 0:
        return None
    uniq, cnts = np.unique(v, return_counts=True)
    cum = np.cumsum(cnts)
    idx = int(np.searchsorted(cum, frac * cum[-1], side='left'))
    if idx >= len(uniq):
        idx = len(uniq) - 1
    return int(uniq[idx])


def f2_robust(dv: np.ndarray) -> Optional[int]:
    """稳健 F2 变体：D 上取中位数，再减一个稳健下界（median-q10 的一半，至少 2 帧）。"""
    if dv.size == 0:
        return None
    med = plateau_quantile(dv, 0.50)
    q10 = plateau_quantile(dv, 0.10)
    if med is None or q10 is None:
        return None
    return int(max(0, med - max(2, 0.5 * (med - q10))))


def ctfc_of(f1: Optional[int], f2: Optional[float], branch: str) -> Optional[float]:
    if f1 is None or f2 is None:
        return None
    raw = (f2 - f1) * FRAME_RATE_CORRECTION
    return float(raw / LAD_CORRECTION) if branch == 'LAD' else float(raw)


# --------------------------------------------------------------------------
# 单条分析 + 画图
# --------------------------------------------------------------------------
def analyze(stem: str, ddir: str, log=print) -> Optional[Dict]:
    jp = os.path.join(ddir, stem + '.json')
    zp = os.path.join(ddir, stem + '.npz')
    if not (os.path.isfile(jp) and os.path.isfile(zp)):
        log(f'  [跳过] {stem}: 缺 json/npz')
        return None
    j = json.load(open(jp, encoding='utf-8'))
    z = np.load(zp)
    E = z['fmax_mask'] > 0            # bool / uint8 0-1 均适用
    extent = z['extent'] > 0
    P = z['P'] > 0
    D = z['D'] > 0

    fmax = int(j['fmax'])
    f1_old = _to_int(j.get('f1'))
    f2_old = _to_int(j.get('f2'))
    branch = j.get('branch_type') or 'RCA'

    imgs, root, dpath = load_frames(j['case_id'], j['sequence_id'])
    if imgs is None:
        log(f'  [跳过] {stem}: 找不到 _src 帧')
        return None
    if imgs[0].shape != E.shape:
        log(f'  [跳过] {stem}: src {imgs[0].shape} vs 掩码 {E.shape}')
        return None
    T = len(imgs)

    fo_int = intensity_arrival(imgs, extent)
    Pv = P & (fo_int >= 0)
    Dv = D & (fo_int >= 0)
    nPv, nDv = int(Pv.sum()), int(Dv.sum())

    f1_new = plateau_quantile(fo_int[Pv], 0.80) if nPv >= 5 else None
    f2_q10 = plateau_quantile(fo_int[Dv], 0.10) if nDv >= 5 else None
    f2_q20 = plateau_quantile(fo_int[Dv], 0.20) if nDv >= 5 else None
    f2_q30 = plateau_quantile(fo_int[Dv], 0.30) if nDv >= 5 else None
    f2_med = plateau_quantile(fo_int[Dv], 0.50) if nDv >= 5 else None
    f2_rob = f2_robust(fo_int[Dv]) if nDv >= 5 else None

    res = dict(
        stem=stem, case_id=j['case_id'], sequence_id=j['sequence_id'], branch_type=branch,
        T=T, fmax=fmax,
        f1_old=f1_old, f2_old=f2_old, ctfc_old=ctfc_of(f1_old, f2_old, branch),
        f1_new=f1_new, f2_new=f2_q10, ctfc_new=ctfc_of(f1_new, f2_q10, branch),
        f2_new_q20=f2_q20, f2_new_q30=f2_q30, f2_new_med=f2_med, f2_new_rob=f2_rob,
        ctfc_new_q20=ctfc_of(f1_new, f2_q20, branch),
        ctfc_new_q30=ctfc_of(f1_new, f2_q30, branch),
        ctfc_new_rob=ctfc_of(f1_new, f2_rob, branch),
        nP=int(P.sum()), nD=int(D.sum()), nP_valid=nPv, nD_valid=nDv,
        f1_old_pinned=int(f1_old == fmax) if f1_old is not None else 0,
        f2_old_pinned=int(f2_old == fmax) if f2_old is not None else 0,
        f2_old_zero=int(f2_old == 0) if f2_old is not None else 0,
        f1_old_gt_f2_old=int(f1_old is not None and f2_old is not None and f1_old > f2_old),
        f1_new_gt_f2_new=int(f1_new is not None and f2_q10 is not None and f1_new > f2_q10),
        f1_escaped_fmax=int(f1_new is not None and f1_new != fmax),
        f1_new_insufficient=int(f1_new is None),
        f2_new_insufficient=int(f2_q10 is None),
        src_dir=dpath,
    )
    # 留画图所需
    res['_imgs'] = imgs
    res['_E'] = E
    res['_extent'] = extent
    res['_P'] = P
    res['_D'] = D
    res['_fo_int'] = fo_int
    res['_Pv'] = Pv
    res['_Dv'] = Dv
    return res


def plot_look(res: Dict, out_png: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    imgs = res['_imgs']
    T = len(imgs)
    E, P, D, Pv, Dv, fo_int = res['_E'], res['_P'], res['_D'], res['_Pv'], res['_Dv'], res['_fo_int']
    fmax = res['fmax']

    def clamp(v):
        if v is None:
            return None
        return int(max(0, min(T - 1, v)))

    frames = [
        ('F1_old', res['f1_old']),
        ('F1_new', res['f1_new']),
        ('F2_old', res['f2_old']),
        ('F2_new', res['f2_new']),
        ('Fmax', fmax),
    ]

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.22, wspace=0.12)

    # 叠加层：P 黄 / D 红（半透明），E 轮廓后续用等值线
    ov = np.zeros(E.shape + (4,), np.float32)
    ov[P] = (1.0, 0.85, 0.0, 0.45)
    ov[D] = (1.0, 0.0, 0.0, 0.45)

    for k, (lab, val) in enumerate(frames):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        iv = clamp(val)
        if iv is None:
            ax.text(0.5, 0.5, f'{lab}: 不足(None)', ha='center', va='center',
                    fontsize=13, color='crimson', transform=ax.transAxes)
            ax.axis('off')
            continue
        ax.imshow(imgs[iv], cmap='gray')
        ax.imshow(ov, interpolation='nearest')
        if E.any():
            ax.contour(E.astype(np.uint8), levels=[0.5], colors=['cyan'], linewidths=0.8)
        ax.set_title(f'{lab} = {val}  (frame {iv}/{T - 1})', fontsize=10)
        ax.axis('off')

    # 第 6 格：到达曲线（P/D 的 fo_int 累计到达比例） + 新旧 F 竖线
    ax = fig.add_subplot(gs[1, 2])
    t = np.arange(T)
    for lab, mask, color in [('P', Pv, '#f59f00'), ('D', Dv, '#e03131')]:
        if mask.any():
            fv = fo_int[mask]
            cdf = np.array([(fv <= x).mean() for x in t])
            ax.plot(t, cdf, color=color, lw=1.8, label=f'{lab} valid n={int(mask.sum())}')
    for val, color, ls, lab in [
        (res['f1_old'], '#f59f00', '--', 'F1_old'),
        (res['f1_new'], '#f59f00', '-', 'F1_new'),
        (res['f2_old'], '#e03131', '--', 'F2_old'),
        (res['f2_new'], '#e03131', '-', 'F2_new'),
    ]:
        if val is not None and 0 <= val <= T - 1:
            ax.axvline(val, color=color, ls=ls, lw=1.2, alpha=0.9, label=f'{lab}={val}')
    ax.axvline(fmax, color='#1971c2', ls=':', lw=1.2, label=f'Fmax={fmax}')
    ax.set_xlabel('frame index')
    ax.set_ylabel('cumulative arrival fraction (fo_int)')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title('intensity arrival curve', fontsize=10)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"{res['stem']}  [{res['branch_type']}]  T={T} Fmax={fmax}  |  "
        f"F1 {res['f1_old']}->{res['f1_new']}  F2 {res['f2_old']}->{res['f2_new']}  "
        f"(P/D valid {res['nP_valid']}/{res['nD_valid']})",
        fontsize=11,
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------
def _frac_bool(vals: List[Optional[bool]]) -> Optional[float]:
    """None（无定义/不足）不参与分母，返回非 None 中 True 的占比。"""
    v = [bool(x) for x in vals if x is not None]
    return round(float(np.mean(v)), 4) if v else None


def _median(vals: List[Optional[float]]) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return round(float(np.median(v)), 3) if v else None


def summarize(results: List[Dict], group_of: Dict[str, str]) -> Dict:
    variants = [('q10', 'f2_new', 'ctfc_new'), ('q20', 'f2_new_q20', 'ctfc_new_q20'),
                ('q30', 'f2_new_q30', 'ctfc_new_q30'), ('rob', 'f2_new_rob', 'ctfc_new_rob')]

    def block(rs: List[Dict]) -> Dict:
        old_f1 = [r['f1_old'] for r in rs]
        old_f2 = [r['f2_old'] for r in rs]
        fm = [r['fmax'] for r in rs]
        out = {
            'n': len(rs),
            'old': {
                'frac_F1_eq_Fmax': _frac_bool([a == b if a is not None else None
                                               for a, b in zip(old_f1, fm)]),
                'frac_F1_gt_F2': _frac_bool([a > b if (a is not None and b is not None) else None
                                             for a, b in zip(old_f1, old_f2)]),
                'frac_F2_eq_Fmax': _frac_bool([b == c if b is not None else None
                                               for b, c in zip(old_f2, fm)]),
                'frac_F2_eq_0': _frac_bool([b == 0 if b is not None else None for b in old_f2]),
                'median_ctfc_valid': _median([r['ctfc_old'] for r in rs
                                              if r['ctfc_old'] is not None and r['f1_old'] is not None
                                              and r['f2_old'] is not None and r['f2_old'] >= r['f1_old']]),
            },
            'new': {
                'frac_F1_eq_Fmax': _frac_bool([r['f1_new'] == r['fmax'] if r['f1_new'] is not None else None for r in rs]),
                'frac_F1_gt_F2': _frac_bool([r['f1_new'] > r['f2_new']
                                             if (r['f1_new'] is not None and r['f2_new'] is not None) else None
                                             for r in rs]),
                'frac_F2_eq_Fmax': _frac_bool([r['f2_new'] == r['fmax'] if r['f2_new'] is not None else None for r in rs]),
                'frac_F2_eq_0': _frac_bool([r['f2_new'] == 0 if r['f2_new'] is not None else None for r in rs]),
                'frac_F1_insufficient': _frac_bool([r['f1_new'] is None for r in rs]) if rs else None,
                'frac_F1_escaped_Fmax': _frac_bool([bool(r['f1_escaped_fmax']) for r in rs]) if rs else None,
                'median_ctfc_valid': _median([r['ctfc_new'] for r in rs
                                              if r['ctfc_new'] is not None and r['f1_new'] is not None
                                              and r['f2_new'] is not None and r['f2_new'] >= r['f1_new']]),
            },
            'f2_variants': {},
        }
        for name, f2key, ckey in variants:
            out['f2_variants'][name] = {
                'frac_F2_eq_Fmax': _frac_bool([r[f2key] == r['fmax'] if r[f2key] is not None else None for r in rs]),
                'frac_F2_eq_0': _frac_bool([r[f2key] == 0 if r[f2key] is not None else None for r in rs]),
                'frac_F1_gt_F2': _frac_bool([r['f1_new'] > r[f2key]
                                             if (r['f1_new'] is not None and r[f2key] is not None) else None
                                             for r in rs]),
                'median_ctfc_valid': _median([r[ckey] for r in rs
                                              if r[ckey] is not None and r['f1_new'] is not None
                                              and r[f2key] is not None and r[f2key] >= r['f1_new']]),
            }
        return out

    groups: Dict[str, List[Dict]] = {}
    for r in results:
        groups.setdefault(group_of.get(r['stem'], '?'), []).append(r)

    summary = {'overall': block(results)}
    for g, rs in sorted(groups.items()):
        summary[f'group_{g}'] = block(rs)

    # 全局择优：最小化 F2 哨兵/零占比，其次最大化有效数
    best = None
    for name, _, _ in variants:
        bs = summary['overall']['f2_variants'][name]
        score = (bs['frac_F2_eq_Fmax'] or 0) + (bs['frac_F2_eq_0'] or 0)
        if best is None or score < best[1]:
            best = (name, score)
    summary['best_f2_variant'] = best[0] if best else None
    return summary


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='POC: 灰度强度到达帧替代掩码 FO 重算 F1/F2')
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--detailed', default=os.path.join(here, 'results_full_v2', 'detailed'))
    ap.add_argument('--bad-csv', default=os.path.join(here, 'results_diag', 'bad_frames', 'all_sequences.csv'))
    ap.add_argument('--results-csv', default=os.path.join(here, 'results_full_v2', 'ctfc_results.csv'))
    ap.add_argument('--out', default=os.path.join(here, 'results_diag', 'fo_poc'))
    ap.add_argument('--items', nargs='*', default=None, help='手动指定 stem；给定时忽略自动挑选')
    ap.add_argument('--nA', type=int, default=12)
    ap.add_argument('--nB', type=int, default=8)
    ap.add_argument('--nC', type=int, default=6)
    ap.add_argument('--seed', type=int, default=7)
    # 到达帧参数（便于敏感性）
    ap.add_argument('--min-amp', type=float, default=MIN_AMP)
    ap.add_argument('--k-frac', type=float, default=K_FRAC)
    ap.add_argument('--abs-floor', type=float, default=ABS_FLOOR)
    args = ap.parse_args()

    def log(m):
        print(m, flush=True)

    if cv2 is None:
        log('错误：缺少 cv2（请用 /root/miniconda3/envs/MedSegBMark/bin/python 运行）。')
        return 2

    out = os.path.abspath(args.out)
    look_dir = os.path.join(out, 'look')
    os.makedirs(look_dir, exist_ok=True)

    bad = load_bad_set(args.bad_csv)
    log(f'坏图名单: {len(bad)} 条')

    if args.items:
        chosen = [dict(case=s.split('__')[0], seq=s.split('__', 1)[1], branch='',
                       stem=s) for s in args.items]
        group_of = {c['stem']: 'manual' for c in chosen}
        rows = []
    else:
        rows = load_rows(args.results_csv, bad)
        for r in rows:
            r['grp'] = assign_group(r)
        n_elig = sum(1 for r in rows if r['grp'])
        log(f'候选（T∈[30,150]、非坏图、P/D≥30px）: {len(rows)} 条，其中可分组 {n_elig} 条')
        picks = (pick_group(rows, 'A', args.nA, args.seed)
                 + pick_group(rows, 'B', args.nB, args.seed)
                 + pick_group(rows, 'C', args.nC, args.seed))
        chosen = []
        for r in picks:
            stem = f"{r['case']}__{r['seq']}"
            chosen.append(dict(case=r['case'], seq=r['seq'], branch=r['branch'], stem=stem))
        group_of = {f"{r['case']}__{r['seq']}": r['grp'] for r in picks}
        log('自动挑选:')
        for g in ('A', 'B', 'C'):
            gs = [c['stem'] for c in chosen if group_of.get(c['stem']) == g]
            log(f"  {g} ({len(gs)}): {', '.join(gs)}")

    results: List[Dict] = []
    for c in chosen:
        log(f'分析 {c["stem"]} ...')
        try:
            res = analyze(c['stem'], args.detailed, log)
        except Exception as e:  # noqa: BLE001
            log(f'  [错误] {c["stem"]}: {e}')
            continue
        if res is None:
            continue
        res['group'] = group_of.get(c['stem'], '?')
        results.append(res)
        png = os.path.join(look_dir, res['stem'] + '.png')
        try:
            plot_look(res, png)
        except Exception as e:  # noqa: BLE001
            log(f'  [画图失败] {res["stem"]}: {e}')
        log(f"  T={res['T']} Fmax={res['fmax']} | F1 {res['f1_old']}->{res['f1_new']} "
            f"F2 {res['f2_old']}->{res['f2_new']} | P/D valid {res['nP_valid']}/{res['nD_valid']} "
            f"| CTFC {res['ctfc_old']}->{res['ctfc_new']}")
        # 释放大数组
        for k in ('_imgs', '_E', '_extent', '_P', '_D', '_fo_int', '_Pv', '_Dv'):
            res.pop(k, None)

    # compare.csv
    cmp_path = os.path.join(out, 'compare.csv')
    fields = ['stem', 'branch_type', 'T', 'fmax', 'f1_old', 'f2_old', 'ctfc_old',
              'f1_new', 'f2_new', 'ctfc_new',
              'f2_new_q20', 'f2_new_q30', 'f2_new_med', 'f2_new_rob',
              'ctfc_new_q20', 'ctfc_new_q30', 'ctfc_new_rob',
              'nP', 'nD', 'nP_valid', 'nD_valid',
              'f1_escaped_fmax', 'f1_new_insufficient', 'f2_new_insufficient',
              'f1_old_pinned', 'f2_old_pinned', 'f2_old_zero',
              'f1_old_gt_f2_old', 'f1_new_gt_f2_new', 'group', 'src_dir']
    with open(cmp_path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in results:
            w.writerow(r)
    log(f'\n对照表: {cmp_path}')

    summary = summarize(results, group_of)
    sum_path = os.path.join(out, 'summary.json')
    with open(sum_path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    log(f'汇总: {sum_path}')

    # 终端结论
    log('\n================ 结论 ================')
    for key in ['overall', 'group_A', 'group_B', 'group_C']:
        if key in summary:
            b = summary[key]
            if b['n'] == 0:
                continue
            o, nw = b['old'], b['new']
            log(f"[{key}] n={b['n']}")
            log(f"  F1==Fmax  旧={o['frac_F1_eq_Fmax']}  新={nw['frac_F1_eq_Fmax']}")
            log(f"  F1>F2     旧={o['frac_F1_gt_F2']}  新={nw['frac_F1_gt_F2']}")
            log(f"  F2==Fmax  旧={o['frac_F2_eq_Fmax']}  新={nw['frac_F2_eq_Fmax']}")
            log(f"  F2==0     旧={o['frac_F2_eq_0']}  新={nw['frac_F2_eq_0']}")
            log(f"  有效CTFC中位 旧={o['median_ctfc_valid']}  新={nw['median_ctfc_valid']}")
            log(f"  新F1不足比例={nw['frac_F1_insufficient']}  新F1脱离Fmax={nw['frac_F1_escaped_Fmax']}")
    log(f"最优 F2 变体（F2哨兵/零最少）: {summary.get('best_f2_variant')}")
    insuff = [r['stem'] for r in results if r['f1_new'] is None]
    if insuff:
        log(f"新 F1 无效(不足) 的 stem: {insuff}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
