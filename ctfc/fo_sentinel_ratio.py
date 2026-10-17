#!/usr/bin/env python3
"""判定：E=M[Fmax] 里 FO==Fmax 的像素，早帧到底有没有造影（自归一化版）。

对每一帧 f 定义相对对比度（以「最早显影的 seed 区」为该帧的对比度基准）：

    ratio_f = (bg_f - sus_f) / (bg_f - seed_f)

其中 bg_f = 该帧非 E 像素平均灰度，seed_f = seed 区（最早显影核心）平均灰度，
sus_f = E 且 FO==Fmax 的像素平均灰度。分子/分母同帧相除，自动抵消曝光漂移、
造影剂绝对强度、帧间亮度变化；造影剂变淡时两者一起变小，比值仍可比。

读法：ratio≈0 → 该组像素此刻还没有造影；ratio≈1 → 此刻造影强度和 seed 一样。
阳性对照 E&FO<Fmax 应该在早帧就接近 1；若疑点组早帧也接近 1，说明
「有造影但没被分割出来」（方向 C）；若疑点组早帧贴 0、直到 Fmax 才跳起，
说明「本来就没有造影」（方向 B）。

只读数据；结果写 --out（默认 results_diag/sentinel_ratio）。
"""
import argparse, glob, json, os, re, random
import numpy as np, cv2

D_DEFAULT = 'results_full_v2/detailed'
ROOTS = ['/home/share/clr/share/data/AI_TMPFC/CAD-ID1-ID64',
         '/home/share/clr/share/data/AI_TMPFC/CMVD',
         '/home/share/clr/share/data/AI_TMPFC/CMVD-ID1-ID50',
         '/home/share/clr/share/data/AI_TMPFC/Others-ID1-ID27']


def load_frames(case, seq):
    for root in ROOTS:
        for sub in ('segmented3c2', 'segmented'):
            d = os.path.join(root, case, seq, sub)
            if not os.path.isdir(d):
                continue
            fs = sorted((int(m.group(1)), os.path.join(d, fn))
                        for fn in os.listdir(d)
                        for m in [re.match(r'^.*-f(\d+)_src\.png$', fn)] if m)
            if fs:
                imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for _, p in fs]
                if any(i is None for i in imgs):
                    continue
                return imgs, sub
    return None, None


def measure(stem, ddir):
    j = json.load(open(os.path.join(ddir, stem + '.json')))
    z = np.load(os.path.join(ddir, stem + '.npz'))
    E = z['fmax_mask'].astype(bool); fo = z['fo'].astype(np.int32)
    seed = z['seed'].astype(bool); fmax = int(j['fmax'])
    if not (E.any() and seed.sum() >= 20 and E.sum() >= 100):
        return None
    imgs, sub = load_frames(j['case_id'], j['sequence_id'])
    if imgs is None or imgs[0].shape != E.shape:
        return None
    T = len(imgs)
    ge = E & (fo < fmax); gs = E & (fo == fmax); gb = ~E
    if ge.sum() < 100 or gs.sum() < 100:
        return None
    I = np.stack(imgs).astype(np.float32)
    bg = I[:, gb].mean(1); sd = I[:, seed].mean(1)
    ratio = lambda m: (bg - I[:, m].mean(1)) / np.maximum(bg - sd, 1e-3)
    r_c, r_s = ratio(ge), ratio(gs)
    pre = slice(0, max(1, fmax))
    out = {'stem': stem, 'sub': sub, 'T': T, 'fmax': fmax, 'f1': j['f1'], 'f2': j['f2'],
           'frac_sentinel': round(float((fo[E] == fmax).mean()), 3),
           'P_sentinel': round(float((fo[z['P'].astype(bool)] == fmax).mean()), 3) if z['P'].any() else None,
           'D_sentinel': round(float((fo[z['D'].astype(bool)] == fmax).mean()), 3) if z['D'].any() else None,
           'f1_pinned': int(j['f1'] == fmax), 'f2_pinned': int(j['f2'] == fmax),
           'seed_fo_med': int(np.median(fo[seed])),
           'ctl_ratio_early': round(float(r_c[:5].mean()), 3) if T >= 5 else None,
           'sus_ratio_early': round(float(r_s[:5].mean()), 3) if T >= 5 else None,
           'ctl_ratio_pre': round(float(r_c[pre].mean()), 3),
           'sus_ratio_pre': round(float(r_s[pre].mean()), 3),
           'sus_ratio_at_fmax': round(float(r_s[fmax]), 3) if fmax < T else None,
           'sus_ratio_peak_pre': round(float(r_s[pre].max()), 3),
           'frac_sus_ratio_gt_05': round(float((r_s[pre] > 0.5).mean()), 3),
           '_curves': {'ctl': [round(float(x), 3) for x in r_c],
                       'sus': [round(float(x), 3) for x in r_s]}}
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
    lo = [c for c in cands if c[0] < 0.20]
    mid = [c for c in cands if 0.30 < c[0] < 0.50]
    hi = [c for c in cands if c[0] > 0.60]
    return (random.sample(lo, min(n_lo, len(lo))) + random.sample(mid, min(n_mid, len(mid)))
            + random.sample(hi, min(n_hi, len(hi))))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--detailed', default=os.path.join(here, D_DEFAULT))
    ap.add_argument('--out', default=os.path.join(here, 'results_diag', 'sentinel_ratio'))
    ap.add_argument('--items', nargs='*')
    ap.add_argument('--n-lo', type=int, default=10)
    ap.add_argument('--n-mid', type=int, default=10)
    ap.add_argument('--n-hi', type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.items:
        items = [(None, s) for s in args.items]
    else:
        items = pick(args.detailed, args.n_lo, args.n_mid, args.n_hi)
        print(f'抽样 {len(items)} 例', flush=True)
    res = []
    for frac, stem in items:
        r = measure(stem, args.detailed)
        if r is None:
            print(f'  [跳过] {stem}', flush=True); continue
        if frac is not None:
            r['picked_frac'] = round(frac, 3)
        res.append(r)
        print(f"  {stem:44s} T={r['T']:4d} Fmax={r['fmax']:4d} F1={r['f1']:4d}{'*' if r['f1_pinned'] else ' '} "
              f"F2={r['f2']:4d}{'*' if r['f2_pinned'] else ' '} sent={r['frac_sentinel']:.0%} "
              f"P_sent={r['P_sentinel']} D_sent={r['D_sentinel']} | "
              f"早帧ratio 阳性={r['ctl_ratio_early']} 疑点={r['sus_ratio_early']}", flush=True)
    if res:
        c = np.array([r['ctl_ratio_early'] for r in res if r['ctl_ratio_early'] is not None])
        s = np.array([r['sus_ratio_early'] for r in res if r['sus_ratio_early'] is not None])
        sp = np.array([r['sus_ratio_pre'] for r in res])
        nm = np.array([r['frac_sus_ratio_gt_05'] for r in res])
        print(f"\n可用 {len(res)} 例")
        print(f"阳性组 E&FO<Fmax 早5帧 ratio 均={c.mean():.2f} 中位={np.median(c):.2f}")
        print(f"疑点组 E&FO==Fmax 早5帧 ratio 均={s.mean():.2f} 中位={np.median(s):.2f}")
        print(f"疑点组 Fmax 前 ratio 均={sp.mean():.2f}")
        print(f"疑点组「Fmax 前 ratio>0.5 的帧占比」均={nm.mean():.2f}")
    json.dump(res, open(os.path.join(args.out, 'summary.json'), 'w'), ensure_ascii=False, indent=1)
    print('→', os.path.join(args.out, 'summary.json'))


if __name__ == '__main__':
    main()
