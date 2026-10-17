#!/usr/bin/env python3
"""判定：E=M[Fmax] 里 FO==Fmax 的像素，早帧到底有没有造影？

背景：`_pred.png` 空间 Dice≈0.92，但逐帧像素成员资格抖动（面积翻转率≈0.43）。
F1/F2 是 fo（逐像素首显帧）在 P（近端）/D（远端）上的极端分位数
（F1=q80(P), F2=q10(D)），P/D 各仅约 100 像素，故被少数异常 fo 主导，
也容易被 fo==Fmax（哨兵值）钉住。

本脚本不问「分割准不准」，只问一件事：**FO==Fmax 的那批像素，在 Fmax 之前
的原始图像里到底有没有造影剂**。若没有 → FO 是对的，改 E/P/D 即可（方向 B）；
若明明有 → 早帧召回不足，必须换强度底物（方向 C）。

做法（组级、幅度自归一，避免逐像素阈值的不稳）：
  1. 每组像素算逐帧平均灰度曲线 g_f；
  2. Δ = g_f0 − min_f g_f（该组自己的对比度幅度）；
  3. t50 = 该组灰度相对 f0 下降达到 0.5Δ 的首帧（该组自己的「半显影时刻」）；
  4. 用 seed 区（最早显影核心，肯定有造影）的 Δ 做基准，输出各组的
     Δ/Δ_seed 与 t50。

读法：`P_sent`（P 里 FO==Fmax 的像素）若 Δ/Δ_seed 明显 >0 且 t50 远小于 Fmax，
说明近端哨兵像素早帧就有造影、只是没被分割 → 方向 C。
若 Δ/Δ_seed≈0 且 t50≈Fmax，说明确实没有造影 → 方向 B。

只读数据；结果写 --out（默认 results_diag/sentinel_diag）。
"""
import argparse, glob, json, os, random, re
import numpy as np
import cv2

ROOTS = ['/home/share/clr/share/data/AI_TMPFC/CAD-ID1-ID64',
         '/home/share/clr/share/data/AI_TMPFC/CMVD',
         '/home/share/clr/share/data/AI_TMPFC/CMVD-ID1-ID50',
         '/home/share/clr/share/data/AI_TMPFC/Others-ID1-ID27']
BRANCH_SUF = re.compile(r'_(LAD|LCX|RCA|LCA)$')


def load_frames(case, seq):
    """sequence_id 带 _LAD/_LCX 后缀时，磁盘目录名其实没有这个后缀。"""
    for s in dict.fromkeys([seq, BRANCH_SUF.sub('', seq)]):
        for root in ROOTS:
            for sub in ('segmented3c2', 'segmented'):
                d = os.path.join(root, case, s, sub)
                if not os.path.isdir(d):
                    continue
                fs = sorted((int(m.group(1)), os.path.join(d, fn))
                            for fn in os.listdir(d)
                            for m in [re.match(r'^.*-f(\d+)_src\.png$', fn)] if m)
                if fs:
                    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for _, p in fs]
                    if all(i is not None for i in imgs):
                        return imgs, d
    return None, None


def profile(imgs, mask):
    return np.array([im.mean() for im in (im[mask] for im in imgs)], np.float32)


def t50(g):
    """相对首帧下降达到自身幅度 50% 的首帧；幅度过小返回 -1。"""
    delta = float(g[0] - g.min())
    if delta <= 1.0:
        return -1
    below = np.where(g <= g[0] - 0.5 * delta)[0]
    return int(below[0]) if below.size else -1


def measure(stem, ddir):
    j = json.load(open(os.path.join(ddir, stem + '.json')))
    z = np.load(os.path.join(ddir, stem + '.npz'))
    E = z['fmax_mask'].astype(bool)
    fo = z['fo'].astype(np.int32)
    seed = z['seed'].astype(bool); P = z['P'].astype(bool); Dm = z['D'].astype(bool)
    fmax = int(j['fmax'])
    if not (E.any() and seed.sum() >= 15 and P.sum() >= 30 and Dm.sum() >= 30):
        return None
    imgs, d = load_frames(j['case_id'], j['sequence_id'])
    if imgs is None or imgs[0].shape != E.shape:
        return None
    T = len(imgs)
    groups = {'seed': seed, 'E': E, 'bg': ~E,
              'P_all': P, 'P_sent': P & (fo == fmax), 'P_early': P & (fo < fmax),
              'D_all': Dm, 'D_sent': Dm & (fo == fmax), 'D_early': Dm & (fo < fmax)}
    groups = {k: v for k, v in groups.items() if v.sum() >= 15}
    if 'seed' not in groups:
        return None
    g = {k: profile(imgs, v) for k, v in groups.items()}
    d_seed = float(g['seed'][0] - g['seed'].min())
    if d_seed <= 1.0:
        return None
    out = {'stem': stem, 'dir': os.path.basename(os.path.dirname(d)), 'T': T, 'fmax': fmax,
           'f1': j['f1'], 'f2': j['f2'], 'seed_delta': round(d_seed, 1),
           'seed_t50': t50(g['seed']),
           'n': {k: int(v.sum()) for k, v in groups.items()},
           'sent_frac': {'P': round(float((fo[P] == fmax).mean()), 3),
                         'D': round(float((fo[Dm] == fmax).mean()), 3),
                         'E': round(float((fo[E] == fmax).mean()), 3)}}
    for k in groups:
        delta = float(g[k][0] - g[k].min())
        out[k] = {'delta': round(delta, 1), 'amp': round(delta / d_seed, 2),
                  't50': t50(g[k]),
                  'g0': round(float(g[k][0]), 1), 'gmin': round(float(g[k].min()), 1),
                  'g_f0_minus_fmax': round(float(g[k][0] - g[k][min(fmax, T-1)]), 1)}
    out['_curves'] = {k: [round(float(x), 1) for x in v] for k, v in g.items()}
    return out


def pick(ddir, n_lo, n_mid, n_hi, seed=3):
    cands = []
    for f in sorted(glob.glob(os.path.join(ddir, '*.json'))):
        stem = os.path.basename(f)[:-5]
        try:
            z = np.load(os.path.join(ddir, stem + '.npz'))
        except Exception:
            continue
        E = z['fmax_mask'].astype(bool); fo = z['fo']
        if E.any():
            cands.append((float((fo[E] == json.load(open(f))['fmax']).mean()), stem))
    random.seed(seed)
    out = []
    for lo, hi, n in [(0.0, 0.20, n_lo), (0.30, 0.50, n_mid), (0.60, 1.01, n_hi)]:
        pool = [c for c in cands if lo <= c[0] < hi]
        out += random.sample(pool, min(n, len(pool)))
    return out


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--detailed', default=os.path.join(here, 'results_full_v2', 'detailed'))
    ap.add_argument('--out', default=os.path.join(here, 'results_diag', 'sentinel_diag'))
    ap.add_argument('--items', nargs='*')
    ap.add_argument('--n-lo', type=int, default=15)
    ap.add_argument('--n-mid', type=int, default=15)
    ap.add_argument('--n-hi', type=int, default=15)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    items = [(None, s) for s in args.items] if args.items else pick(args.detailed, args.n_lo, args.n_mid, args.n_hi)
    print(f'样本 {len(items)} 例', flush=True)
    res = []
    for frac, stem in items:
        r = measure(stem, args.detailed)
        if r is None:
            print(f'  [跳过] {stem}', flush=True); continue
        if frac is not None:
            r['picked_frac'] = round(frac, 3)
        res.append(r)
        def s(k): return r.get(k, {})
        print(f"  {stem:46s} T={r['T']:4d} Fmax={r['fmax']:4d} F1={r['f1']:4d} F2={r['f2']:4d} "
              f"P_sent={r['sent_frac']['P']:.2f} D_sent={r['sent_frac']['D']:.2f} | "
              f"seed Δ/Δs=1.00 t50={r['seed_t50']:3d} | "
              f"P_sent amp={s('P_sent').get('amp')} t50={s('P_sent').get('t50')} | "
              f"bg amp={s('bg').get('amp')} t50={s('bg').get('t50')}", flush=True)
    if res:
        def col(k, f, pred=lambda r: True):
            return np.array([r[k][f] for r in res if k in r and f in r[k] and pred(r) and r[k][f] >= 0], float)
        def amp(k, pred=lambda r: True):
            return np.array([r[k]['amp'] for r in res if k in r and pred(r)], float)
        print(f"\n可用 {len(res)} 例")
        for k in ('seed', 'P_early', 'P_sent', 'D_sent', 'bg'):
            if any(k in r for r in res):
                a = amp(k)
                print(f"  {k:8s} n={a.size:3d} 幅度Δ/Δseed 均={a.mean():5.2f} 中位={np.median(a):5.2f}   "
                      f"t50 均={col(k,'t50').mean() if col(k,'t50').size else float('nan'):6.1f} "
                      f"（样例 Fmax 中位={np.median([r['fmax'] for r in res if k in r]):.0f}）")
        ps = [r for r in res if 'P_sent' in r]
        if ps:
            early = [r for r in ps if r['P_sent']['t50'] >= 0 and r['P_sent']['t50'] <= r['fmax'] - 3]
            amp_ok = [r for r in ps if r['P_sent']['amp'] > 0.5]
            print(f"  P_sent: t50<=Fmax-3 的例占 {len(early)}/{len(ps)}"
                  f"；P_sent 幅度>0.5*seed 的例占 {len(amp_ok)}/{len(ps)}")
    json.dump(res, open(os.path.join(args.out, 'summary.json'), 'w'), ensure_ascii=False, indent=1)
    print('→', os.path.join(args.out, 'summary.json'))


if __name__ == '__main__':
    main()
