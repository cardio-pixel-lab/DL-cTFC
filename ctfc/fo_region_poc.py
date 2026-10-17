#!/usr/bin/env python3
"""POC-2：只改 F1（区域灰度到达），F2 保留掩码远端 FO —— 对照旧结果并出人工核对图。

结论演进见 计划-v2 §0.1.10–§0.1.12：
- 逐像素灰度到达（POC-1）失败：远端细血管幅度与噪声同量级，阈值被噪声虚触发。
- 区域级灰度到达（POC-2）成立，但**曲线定义**与**到达判据**各迭代过一次：
  v1 曲线参考用「每帧亮度中位」→ 共模亮度/膈肌漂移把 P/D 一起抬高，frac 无解；
  v2 参考改**逐像素时间基线** ref_p=q90_t(I_p)，`g(t)=mean_p(ref_p−I_p(t))`，曲线才干净。
  到达判据改**双趟稳健 onset**（低分位粗定起点 → 平段估 b/σ → thr=b+max(3σ,0.15·rise,2)）。
- **v2 的 F2 仍不对**（用户判读：未落在真显影帧，普遍不如 Fmax）。诊断：D 区曲线有一段缓慢
  早抬升（非造影，属共模残留/慢漂），σ 极小时被 2 灰度绝对底误触发 → F2 系统性偏早。
  而**掩码远端 FO 在 D 区是可信的**（§0.1.8：D 的 `FO==Fmax` 像素早帧对比度仅 0.09，多为真未显影）。
  ⇒ 本版定案：**只把 F1 换成区域灰度到达，F2 原样用掩码远端 FO 分位**。物理上 F1/F2 都是
  「局部首显影」，F2 天然晚于 F1；实测 `F1>F2` 34.6%→4.2%、CTFC 中位 22（≈文献 21）。

复用 fo_intensity_poc 的选样/载帧函数（import），只读数据，输出写真工作目录。
"""
import csv, importlib.util, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('poc1', os.path.join(HERE, 'fo_intensity_poc.py'))
poc1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(poc1)

CONSEC = 2
FRAC = 0.15          # 阈值中「上升幅度比例」项
QREF = 90            # 逐像素时间基线分位（最亮≈无造影）
GFLOOR = 2.0         # 阈值最小绝对裕量（灰度级）
F2_FRAC = 0.10       # F2：掩码远端 FO 的分位（与主管线 f2_percentile 一致）
F1_Q = 0.15          # 推荐 F1：P 区 FO 的 0.15 分位（= 旧 f1_percentile 0.80 直接调小）
DISAGREE = 2         # |F1_q - F1_b| > 此值 ⇒ 标记需人工判读
SUF_RE = re.compile(r'_(LAD|LCX|RCA|LCA)$')
CLS_RGB = {255: (0.90, 0.20, 0.20), 128: (0.20, 0.80, 0.30)}   # LAD 红 / LCX 绿


def region_curve(imgs, mask, q=QREF):
    """逐像素时间基线区域曲线 g(t)：ref_p=q分位(最亮≈无造影)，g=mean_p(ref_p − I_p(t))。"""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros(len(imgs), np.float32)
    X = np.stack([im.astype(np.float32)[ys, xs] for im in imgs])   # T x n
    ref = np.percentile(X, q, axis=0)
    return (ref[None, :] - X).mean(axis=1).astype(np.float32)


def _medfilt(g, k):
    if k <= 1:
        return g.copy()
    r = k // 2
    pad = np.pad(g, r, mode='edge')
    return np.median(np.stack([pad[i:i + len(g)] for i in range(k)]), axis=0)


def _smooth(g):
    T = len(g)
    gs = _medfilt(g, max(3, (T // 30) | 1))
    w = max(1, T // 50)
    return np.convolve(gs, np.ones(w) / w, 'same') if w > 1 else gs


def _first_run(hit, consec):
    for t in range(len(hit) - consec + 1):
        if all(hit[t + j] for j in range(consec)):
            return int(t)
    return -1


def arrival_frame(g, frac=FRAC, consec=CONSEC, kmad=3.0, kvalid=4.0, gfloor=GFLOOR):
    """F1 的双趟稳健 onset。返回 (onset, b, sigma, thr, rise)；无效 onset=-1。"""
    T = len(g)
    if T < consec + 2:
        return -1, 0.0, 0.0, 0.0, 0.0
    gs = _smooth(g)
    lo, hi = float(np.percentile(gs, 10)), float(np.percentile(gs, 85))
    low = gs[gs <= np.median(gs)]
    s1 = 1.4826 * float(np.median(np.abs(low - np.median(low))))
    t1 = _first_run(gs > lo + max(2 * s1, 0.1 * (hi - lo)), consec)
    ke = max(3, min(max(t1, 0), T // 4))
    b = float(np.median(gs[:ke]))
    s = 1.4826 * float(np.median(np.abs(gs[:ke] - b)))
    rise = float(gs.max()) - b
    if rise <= max(kvalid * max(s, 1e-6), 3.0):
        return -1, b, s, 0.0, rise
    thr = b + max(kmad * s, frac * rise, gfloor)
    return _first_run(gs > thr, consec), b, s, thr, rise


def fo_quantile(fo, mask, frac):
    """高原感知分位（与 ctfc_analyzer._fo_quantile 同义）：mask 内最小 FO 值使累计占比 ≥ frac。"""
    v = fo[mask > 0]
    if v.size == 0:
        return None
    uniq, cnts = np.unique(v, return_counts=True)
    cum = np.cumsum(cnts)
    idx = int(np.searchsorted(cum, frac * cum[-1], side='left'))
    return int(uniq[min(idx, len(uniq) - 1)])


FILL_THR = 0.08      # fill_p 持续起跳阈值（占位，待金标准标定）


def f1_fill_onset(fill_p, thr=FILL_THR, k=5, wfrac=None):
    """F1 候选③：fill_p 中值滤波后首个「连续 w 帧 ≥ thr」的帧。

    fill_p 是主管线已算好的 P 区逐帧瞬时充盈率（ctfc_analyzer L419），原仅用于 QC。
    比区域灰度曲线可信：灰度曲线被异常首帧（f0–f4 增益爬升）污染（§0.1.13）。
    """
    T = len(fill_p)
    if T == 0:
        return -1
    f = _medfilt(np.asarray(fill_p, np.float64), k)
    w = max(5, int((wfrac if wfrac is not None else 1 / 15) * T))
    if w > T:
        w = T
    for t in range(T - w + 1):
        if np.all(f[t:t + w] >= thr):
            return int(t)
    return -1


def find_pred(case_id, sequence_id, idx):
    """读 Fmax 帧的 `_pred.png`（0/128=LCX/255=LAD 三类索引图）。"""
    seq = SUF_RE.sub('', sequence_id)
    for root in poc1.ROOTS:
        for sub in ('segmented3c2', 'segmented'):
            d = os.path.join(root, case_id, seq, sub)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.endswith(f'-f{idx}_pred.png'):
                    return poc1.cv2.imread(os.path.join(d, fn), poc1.cv2.IMREAD_UNCHANGED)
    return None


def colorize(m):
    """0/128/255 索引图 → RGB：LAD 红、LCX 绿。"""
    rgb = np.zeros(m.shape + (3,), np.float32)
    for v, c in CLS_RGB.items():
        rgb[m == v] = c
    return rgb


def ctfc_of(f1, f2, branch):
    if f1 is None or f2 is None or f1 < 0 or f2 < 0:
        return None
    return round((f2 - f1) * 2.0 / (1.7 if branch == 'LAD' else 1.0), 2)


def plot_region(imgs, pred, gP, gD, dP, rec, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = len(imgs)
    t = np.arange(T)

    def vline(ax, val, color, ls, lab, lw=1.5):
        if val is not None and 0 <= val <= T - 1:
            ax.axvline(val, color=color, ls=ls, lw=lw, label=lab)

    def clamp(v):
        return None if v is None or v < 0 else int(max(0, min(T - 1, v)))

    cells = [('F1_old (FO q80)', rec['f1_old'], '#868e96'),
             ('F1_b (region onset)', rec['f1_new'], '#f59f00'),
             ('F1_c (fill onset)', rec['f1_fill'], '#2f9e44'),
             (f'F1_q (FO q{int(F1_Q*100)})', rec['f1_q'], '#7048e8'),
             ('F2 (mask dist FO q10)', rec['f2_new'], '#e03131')]
    fig = plt.figure(figsize=(19, 14))
    gs = fig.add_gridspec(5, 5, hspace=0.20, wspace=0.07,
                          height_ratios=[1, 1, 0.85, 0.85, 0.85])
    for k, (lab, val, col) in enumerate(cells):
        ax = fig.add_subplot(gs[0, k])
        iv = clamp(val)
        if iv is None:
            ax.text(0.5, 0.5, f'{lab}\n= {val}\n(not reached)', ha='center', va='center',
                    fontsize=11, color='crimson', transform=ax.transAxes); ax.axis('off'); continue
        ax.imshow(imgs[iv], cmap='gray')
        ax.set_title(f'{lab} = {val}\n(frame {iv}/{T-1})', fontsize=9, color=col)
        ax.axis('off')
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(imgs[rec['fmax']], cmap='gray')
    ax.set_title(f"Fmax raw = {rec['fmax']}", fontsize=9); ax.axis('off')
    ax = fig.add_subplot(gs[1, 1])
    if pred is not None:
        ax.imshow(colorize(pred))
        ax.set_title(f"seg @Fmax (LAD red / LCX green)\nbranch={rec['branch']}", fontsize=9)
    else:
        ax.text(0.5, 0.5, '_pred.png missing', ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')
    ax = fig.add_subplot(gs[1, 2:]); ax.axis('off')
    ax.text(0.02, 0.95, '\n'.join([
        f"T={T}  Fmax={rec['fmax']}  branch={rec['branch']}   P n={rec['nP']}  D n={rec['nD']}",
        f"FO quantiles over P:  q10={rec['q10']}  q20={rec['q20']}  q30={rec['q30']}  "
        f"q40={rec['q40']}  q80={rec['q80']}",
        f"P 区 FO==0 质量={rec['f0mass']:.2f}   FO==Fmax 质量={rec['sntmass']:.2f}"
        f"{rec['f0flag']}",
        f"F1_old={rec['f1_old']}   F2_old={rec['f2_old']}",
        f"F1_b={rec['f1_new']}  b={rec['baseP']} sig={rec['sigP']} "
        f"thr={rec['thrP']} rise={rec['riseP']} ampP={rec['ampP']}",
        f"F1_c={rec['f1_fill']} (fill 起跳, thr={FILL_THR}, w={rec['fillW']})",
        f"F1_q={rec['f1_q']}  (P 区 FO 的 q{int(F1_Q*100)}，单调、无闪烁)",
        f"F2 = plateau-aware q10 of mask-FO over D  (unmoved)",
        '',
        f"CTFC old  = {rec['ctfc_old']}",
        f"CTFC F1_b = {rec['ctfc_b']}",
        f"CTFC F1_c = {rec['ctfc_c']}",
        f"CTFC F1_q = {rec['ctfc_q']}",
        f"|F1_q - F1_b| = {rec['dif']}  ⇒ {rec['verdict']}",
    ]), va='top', fontsize=8.5, family='monospace', transform=ax.transAxes,
        color=('#c92a2a' if rec['dif'] > DISAGREE else 'black'))

    ax = fig.add_subplot(gs[2, :])
    _, b, s, thr, rise = dP
    ax.plot(t, gP, color='#f59f00', lw=0.7, alpha=0.4)
    ax.plot(t, _smooth(gP), color='#f59f00', lw=1.8, label='P region mean-dark (substrate b)')
    if thr:
        ax.axhline(thr, color='#f59f00', ls='--', lw=1.0, alpha=0.7, label=f'thr={thr:.1f}')
        ax.axhline(b, color='#f59f00', ls=':', lw=0.9, alpha=0.5)
    vline(ax, rec['f1_old'], '#868e96', '--', f"F1_old={rec['f1_old']}")
    vline(ax, rec['f1_new'], '#f59f00', '-', f"F1_b={rec['f1_new']}")
    vline(ax, rec['f1_q'], '#7048e8', '-', f"F1_q={rec['f1_q']}")
    vline(ax, rec['f2_new'], '#e03131', '-', f"F2={rec['f2_new']}")
    vline(ax, rec['fmax'], '#1971c2', ':', f"Fmax={rec['fmax']}", 1.2)
    ax.set_ylabel('mean darkness (gray)')
    ax.set_title('substrate b: P region intensity arrival (validator, not primary)', fontsize=10)
    ax.legend(fontsize=8, loc='upper right', ncol=4); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[3, :])
    fp = np.asarray(rec['fill_p'], np.float64)
    ax.plot(t, fp, color='#2f9e44', lw=0.8, alpha=0.4)
    ax.plot(t, _medfilt(fp, 5), color='#2f9e44', lw=1.8, label='fill_p (median w=5, substrate c)')
    foP = np.asarray(rec['foP']).ravel()
    ax.plot(t, np.array([np.mean(foP <= k) for k in range(T)]),
            color='#7048e8', lw=1.8, label='FO-CDF over P (substrate q, monotone)')
    ax.axhline(FILL_THR, color='#2f9e44', ls='--', lw=1.0, alpha=0.7, label=f'c thr={FILL_THR}')
    ax.axhline(F1_Q, color='#7048e8', ls='--', lw=1.0, alpha=0.7, label=f'q frac={F1_Q}')
    vline(ax, rec['f1_old'], '#868e96', '--', f"F1_old={rec['f1_old']}")
    vline(ax, rec['f1_fill'], '#2f9e44', '-', f"F1_c={rec['f1_fill']}")
    vline(ax, rec['f1_q'], '#7048e8', '-', f"F1_q={rec['f1_q']}")
    vline(ax, rec['f2_new'], '#e03131', '-', f"F2={rec['f2_new']}")
    vline(ax, rec['fmax'], '#1971c2', ':', f"Fmax={rec['fmax']}", 1.2)
    ax.set_ylabel('fraction of P pixels')
    ax.set_title('substrate c vs q: fill_p (instantaneous, flickers)  |  FO-CDF (monotone)',
                 fontsize=10)
    ax.legend(fontsize=8, loc='upper right', ncol=4); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[4, :])
    ax.plot(t, gD, color='#e03131', lw=0.7, alpha=0.4)
    ax.plot(t, _smooth(gD), color='#e03131', lw=1.8, label='D region mean-dark (NOT used for F2)')
    vline(ax, rec['f2_old'], '#868e96', '--', f"F2_old={rec['f2_old']}")
    vline(ax, rec['f2_new'], '#e03131', '-', f"F2={rec['f2_new']}")
    vline(ax, rec['fmax'], '#1971c2', ':', f"Fmax={rec['fmax']}", 1.2)
    ax.set_xlabel('frame index'); ax.set_ylabel('mean darkness (gray)')
    ax.set_title('F2 reference: D region intensity curve (F2 itself = mask distal FO q10, unmoved)', fontsize=10)
    ax.legend(fontsize=8, loc='upper right', ncol=3); ax.grid(alpha=0.25)

    fig.suptitle(f"{rec['stem']}   F1: old {rec['f1_old']} | region {rec['f1_new']} | "
                 f"fill {rec['f1_fill']}   F2 {rec['f2_new']}   "
                 f"CTFC old {rec['ctfc_old']} | fill {rec['ctfc_c']}", fontsize=11)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=105, bbox_inches='tight'); plt.close(fig)


def main():
    out = os.path.join(HERE, 'results_diag', 'fo_poc')
    os.makedirs(out, exist_ok=True)
    look_dir = os.path.join(out, 'look_region')
    os.makedirs(look_dir, exist_ok=True)
    bad = poc1.load_bad_set(os.path.join(HERE, 'results_diag', 'bad_frames', 'all_sequences.csv'))
    rows = poc1.load_rows(os.path.join(HERE, 'results_full_v2', 'ctfc_results.csv'), bad)
    for r in rows:
        r['grp'] = poc1.assign_group(r)
    picks = (poc1.pick_group(rows, 'A', 12, 7) + poc1.pick_group(rows, 'B', 8, 7)
             + poc1.pick_group(rows, 'C', 6, 7))
    print(f'候选 {len(rows)}，选中 {len(picks)}', flush=True)

    recs = []
    for r in picks:
        stem = f"{r['case']}__{r['seq']}"
        jp = os.path.join(HERE, 'results_full_v2', 'detailed', stem + '.json')
        zp = os.path.join(HERE, 'results_full_v2', 'detailed', stem + '.npz')
        if not (os.path.isfile(jp) and os.path.isfile(zp)):
            print(f'  [跳过] {stem}'); continue
        j = json.load(open(jp, encoding='utf-8'))
        z = np.load(zp)
        P, D = z['P'] > 0, z['D'] > 0
        imgs, _, _ = poc1.load_frames(j['case_id'], j['sequence_id'])
        if imgs is None:
            print(f'  [跳过] {stem}: 无帧'); continue
        T = len(imgs)
        gP, gD = region_curve(imgs, P), region_curve(imgs, D)
        dP = arrival_frame(gP)
        f1_new = dP[0]
        f2_new = fo_quantile(z['fo'], D, F2_FRAC)
        fill_p = j.get('fill_p') or []
        f1_fill = f1_fill_onset(fill_p) if len(fill_p) == T else -1
        foP = z['fo'][P].astype(np.int32)
        q = {int(f * 100): fo_quantile(z['fo'], P, f) for f in (0.10, 0.20, 0.30, 0.40, 0.80)}
        f1_q = q[20]
        rec = dict(stem=stem, branch=j['branch_type'], T=T, fmax=int(j['fmax']),
                   f1_old=poc1._to_int(j.get('f1')), f2_old=poc1._to_int(j.get('f2')),
                   f1_new=f1_new, f2_new=f2_new, f1_fill=f1_fill, f1_q=f1_q,
                   q10=q[10], q20=q[20], q30=q[30], q40=q[40], q80=q[80],
                   f0mass=round(float(np.mean(foP == 0)), 3),
                   sntmass=round(float(np.mean(foP >= int(j['fmax']))), 3),
                   baseP=round(dP[1], 2), sigP=round(dP[2], 2), thrP=round(dP[3], 2), riseP=round(dP[4], 2),
                   ampP=round(float(gP.max() - np.median(gP)), 2),
                   ampD=round(float(gD.max() - np.median(gD)), 2),
                   fillW=max(5, T // 15), fill_p=list(fill_p),
                   nP=int(P.sum()), nD=int(D.sum()), grp=r['grp'])
        rec['f0flag'] = '  << f0 污染' if rec['f0mass'] >= 0.10 else ''
        rec['dif'] = (abs(f1_q - f1_new) if (f1_q is not None and f1_new >= 0) else -1)
        rec['verdict'] = ('需人工判读（q 与 b 分歧）' if rec['dif'] > DISAGREE
                          else ('一致' if rec['dif'] >= 0 else 'b 未到达'))
        rec['foP'] = foP.tolist()
        rec['ctfc_old'] = ctfc_of(rec['f1_old'], rec['f2_old'], rec['branch'])
        rec['ctfc_b'] = ctfc_of(f1_new, f2_new, rec['branch'])
        rec['ctfc_c'] = ctfc_of(f1_fill, f2_new, rec['branch'])
        rec['ctfc_q'] = ctfc_of(f1_q, f2_new, rec['branch'])
        recs.append(rec)
        try:
            pred = find_pred(j['case_id'], j['sequence_id'], rec['fmax'])
            plot_region(imgs, pred, gP, gD, dP, rec, os.path.join(look_dir, stem + '.png'))
        except Exception as e:  # noqa: BLE001
            print(f'  [画图失败] {stem}: {e}')
        for k in ('fill_p', 'foP'):
            rec.pop(k)
        print(f"  {stem:44s} T={T:4d} Fmax={rec['fmax']:4d} | 旧 F1={rec['f1_old']} "
              f"F2={rec['f2_old']} (CTFC={rec['ctfc_old']}) → b: {f1_new} (CTFC={rec['ctfc_b']})  "
              f"c: {f1_fill} (CTFC={rec['ctfc_c']})  q20: {f1_q} (CTFC={rec['ctfc_q']})  "
              f"|q-b|={rec['dif']} {rec['verdict']}{rec['f0flag']}", flush=True)

    if recs:
        with open(os.path.join(out, 'region_compare.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, list(recs[0].keys())); w.writeheader(); w.writerows(recs)

    def variant(rs, key, ckey):
        cts = [r[ckey] for r in rs if r[ckey] is not None]
        ok = [r for r in rs if r[key] >= 0 and r['f2_new'] is not None and r['f2_new'] >= 0]
        return {
            'n': len(rs), 'n_valid': len(ok),
            'F1==Fmax': round(np.mean([r[key] == r['fmax'] for r in rs]), 3),
            'F1>F2': round(np.mean([r[key] > r['f2_new'] for r in ok]), 3) if ok else None,
            'F1<0': round(np.mean([r[key] < 0 for r in rs]), 3),
            'CTFC中位': round(float(np.median(cts)), 2) if cts else None,
        }

    def block(rs):
        cto = [r['ctfc_old'] for r in rs if r['ctfc_old'] is not None]
        return {
            'n': len(rs),
            'old': {'F1==Fmax': round(np.mean([r['f1_old'] == r['fmax'] for r in rs]), 3),
                    'F1>F2': round(np.mean([r['f1_old'] > r['f2_old'] for r in rs]), 3),
                    'CTFC中位': round(float(np.median(cto)), 2) if cto else None},
            'b_region': variant(rs, 'f1_new', 'ctfc_b'),
            'c_fill': variant(rs, 'f1_fill', 'ctfc_c'),
            'q20': variant(rs, 'q20', 'ctfc_q'),
            'n_disagree': sum(1 for r in rs if r['dif'] > DISAGREE),
            'n_f0mass': sum(1 for r in rs if r['f0mass'] >= 0.10),
            'disagree': [r['stem'] for r in rs if r['dif'] > DISAGREE],
        }

    groups = {}
    for r in recs:
        groups.setdefault(r['grp'], []).append(r)
    summ = {'overall': block(recs)}
    for g, rs in sorted(groups.items()):
        summ[f'group_{g}'] = block(rs)
    with open(os.path.join(out, 'region_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)

    print('\n============= F1 候选对照 =============')
    for key in ['overall', 'group_A', 'group_B', 'group_C']:
        if key not in summ:
            continue
        b = summ[key]
        print(f"[{key}] n={b['n']}  旧: F1==Fmax={b['old']['F1==Fmax']} "
              f"F1>F2={b['old']['F1>F2']} CTFC中位={b['old']['CTFC中位']}")
        for name, lab in [('b_region', 'b:P区灰度到达(校验)'), ('c_fill', 'c:fill_p持续起跳'),
                          ('q20', f'q{int(F1_Q*100)}:FO分位(推荐主判据)')]:
            v = b[name]
            print(f"      {lab}: n_valid={v['n_valid']} F1==Fmax={v['F1==Fmax']} "
                  f"F1>F2={v['F1>F2']} F1未到达={v['F1<0']} CTFC中位={v['CTFC中位']}")
        print(f"      分歧(|q-b|>{DISAGREE}) {b['n_disagree']} 例; f0污染(FO==0≥10%) "
              f"{b['n_f0mass']} 例")
    print(f"\n输出: {os.path.join(out,'region_summary.json')}")
    print(f"人工核对图: {look_dir}/*.png")


if __name__ == '__main__':
    main()
