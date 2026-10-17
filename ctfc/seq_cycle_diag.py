#!/usr/bin/env python3
"""看序列的「造影时间过程」形状：掩码面积 + 灰度造影量随帧的曲线。

目的：区分三种可能
  A. 完整单循环（注射→达峰→消退）：曲线单峰、末端回落到≈0
  B. 多次注射拼接：曲线多峰
  C. 真实造影只占开头一小段、后面是空转/静止：峰位很早，末端面积不归零

指标（只读）:
  area(t) = _pred 非零像素数
  cmass(t) = mean( max(ref - I_t, 0) ), ref = 全序列灰度的 90 分位  —— 造影量（越暗越有造影）
打印 24 段采样后的曲线，并给出 峰位/末端比/峰数。
"""
import json, os, re
import cv2, numpy as np

ROOTS = ['/home/share/clr/share/data/AI_TMPFC/CAD-ID1-ID64',
         '/home/share/clr/share/data/AI_TMPFC/CMVD',
         '/home/share/clr/share/data/AI_TMPFC/CMVD-ID1-ID50',
         '/home/share/clr/share/data/AI_TMPFC/Others-ID1-ID27']
BRANCH_SUF = re.compile(r'_(LAD|LCX|RCA|LCA)$')


def locate(stem):
    if '__' not in stem:
        return None
    case, seq = stem.split('__', 1)
    seq_dir = BRANCH_SUF.sub('', seq)
    for root in ROOTS:
        for sub in ('segmented3c2', 'segmented'):
            d = os.path.join(root, case, seq_dir, sub)
            if os.path.isdir(d):
                return d
    return None


def load_series(ddir):
    def idx(fn):
        m = re.search(r'-f(\d+)_', fn)
        return int(m.group(1)) if m else -1
    preds = sorted([(idx(f), f) for f in os.listdir(ddir) if f.endswith('_pred.png')])
    srcs = sorted([(idx(f), f) for f in os.listdir(ddir) if f.endswith('_src.png')])
    area = np.array([int((cv2.imread(os.path.join(ddir, f), cv2.IMREAD_GRAYSCALE) > 0).sum())
                     for _, f in preds], np.float32)
    I = np.stack([cv2.imread(os.path.join(ddir, f), cv2.IMREAD_GRAYSCALE) for _, f in srcs]).astype(np.float32)
    ref = float(np.percentile(I, 90))
    cmass = np.maximum(ref - I, 0).mean(axis=(1, 2))
    return area, cmass


def spark(y, bins=32):
    n = len(y)
    if n == 0:
        return ''
    e = np.linspace(0, n, bins + 1).astype(int)
    s = np.array([y[max(e[i], 0):max(e[i + 1], 1)].mean() for i in range(bins)])
    lo, hi = s.min(), s.max()
    chars = ' .:-=+*#%@'
    if hi - lo < 1e-9:
        return chars[0] * bins
    return ''.join(chars[int((v - lo) / (hi - lo) * 9)] for v in s)


def npk(y, k=3):
    """粗数峰数：平滑后局部极大，且显著高于邻域。"""
    if len(y) < 5:
        return 0
    w = max(3, len(y) // 60)
    sm = np.convolve(y, np.ones(w) / w, 'same')
    peaks = [i for i in range(1, len(sm) - 1) if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1]]
    if not peaks:
        return 0
    thr = sm.max() * 0.35
    peaks = [i for i in peaks if sm[i] >= thr]
    # 合并邻近
    merged = []
    for i in peaks:
        if merged and i - merged[-1] < len(y) // 20:
            continue
        merged.append(i)
    return len(merged)


def report(r):
    ddir = locate(r['stem'])
    if not ddir:
        print(f"  [跳过] {r['stem']}"); return None
    area, cmass = load_series(ddir)
    T = len(area)
    a_pos = int(np.argmax(area)) / max(1, T - 1)
    c_pos = int(np.argmax(cmass)) / max(1, T - 1)
    rec = {'stem': r['stem'], 'T': r['T'], 'len_area': T, 'len_src': len(cmass),
           'fmax': r['fmax'], 'F1': r['f1'], 'F2': r['f2'],
           'area_pos': round(a_pos, 3), 'area_tail/peak': round(float(area[-1]) / max(float(area.max()), 1), 3),
           'cmass_pos': round(c_pos, 3), 'cmass_peaks': npk(cmass), 'area_peaks': npk(area),
           'area_spark': spark(area), 'cmass_spark': spark(cmass)}
    print(f"  {r['stem']}  T={r['T']} fmax={r['fmax']} F1={r['f1']} F2={r['f2']}")
    print(f"    面积 峰位{a_pos:.2f} 末端/峰={rec['area_tail/peak']:.2f} 峰数={rec['area_peaks']} | {rec['area_spark']}")
    print(f"    造影量 峰位{c_pos:.2f} 峰数={rec['cmass_peaks']} | {rec['cmass_spark']}")
    return rec


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rows = json.load(open(os.path.join(here, 'results_diag', 'f1_f2_consistency.json')))
    def uniq(group, n):
        seen, out = set(), []
        for r in group:
            k = r['stem'].split('__')[1].split('_')[0]
            if k in seen:
                continue
            seen.add(k); out.append(r)
            if len(out) >= n:
                break
        return out
    long_ = uniq([r for r in rows if r['T'] == 1024], 4)
    mid = uniq([r for r in rows if 30 <= r['T'] <= 60], 4)
    out = {'long_1024': [], 'short_30_60': []}
    for tag, g in (('long_1024', long_), ('short_30_60', mid)):
        print(f'\n===== {tag} =====')
        for r in g:
            rec = report(r)
            if rec:
                out[tag].append(rec)
    os.makedirs(os.path.join(here, 'results_diag', 'cycle'), exist_ok=True)
    p = os.path.join(here, 'results_diag', 'cycle', 'summary.json')
    json.dump(out, open(p, 'w'), ensure_ascii=False, indent=2)
    print(f'\n汇总: {p}')


if __name__ == '__main__':
    main()
