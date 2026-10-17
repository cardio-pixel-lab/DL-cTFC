#!/usr/bin/env python3
"""全量核验队列序列的原始帧是否为坏图（横向条纹），生成问题数据列表。

背景见 计划-v2 §0.1.9：T=1024/T=512 组的 `_src.png` 是纯横向条纹（每行近似常数灰度）。
判据（模型无关）：stripe = std(每行均值) / mean(每行标准差)。
  - 真造影图：行内有结构 → 行内标准差大 → 比值 ≈0.5–1
  - 横向条纹：行内近似常数 → 行内标准差≈0 → 比值爆表（实测 1e4 量级）
另附 frame_std（整帧标准差）作空白/近乎无信息帧的旁证。

只读队列；输出全部写 --out（默认 results_diag/bad_frames）。
"""
import argparse, csv, glob, json, os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

ROOTS = ['/home/share/clr/share/data/AI_TMPFC/CAD-ID1-ID64',
         '/home/share/clr/share/data/AI_TMPFC/CMVD',
         '/home/share/clr/share/data/AI_TMPFC/CMVD-ID1-ID50',
         '/home/share/clr/share/data/AI_TMPFC/Others-ID1-ID27']
STRIPE_THRESH = 3.0      # 判为条纹的阈值（真图≈0.6，条纹图 1e4，间距极大）
BLANK_STD = 2.0          # 整帧标准差低于此值视为空白帧
SUBDIRS = ('segmented3c2', 'segmented')


def list_sequences(root):
    """返回 [(case_id, seq_id, ddir)]，ddir 为含 _src.png 的目录。"""
    out = []
    for case in sorted(os.listdir(root)):
        cdir = os.path.join(root, case)
        if not os.path.isdir(cdir):
            continue
        for seq in sorted(os.listdir(cdir)):
            sdir = os.path.join(cdir, seq)
            if not os.path.isdir(sdir):
                continue
            for sub in SUBDIRS:
                d = os.path.join(sdir, sub)
                if os.path.isdir(d) and any(f.endswith('_src.png') for f in os.listdir(d)):
                    out.append((case, seq, d))
                    break
    return out


def stripe(im):
    rowm = im.mean(axis=1); rowsd = im.std(axis=1)
    return float(rowm.std() / (rowsd.mean() + 1e-3))


def check(job):
    case, seq, ddir = job
    srcs = []
    for fn in os.listdir(ddir):
        if fn.endswith('_src.png'):
            try:
                srcs.append((int(fn.rsplit('-f', 1)[1].split('_')[0]), fn))
            except (ValueError, IndexError):
                srcs.append((-1, fn))
    srcs.sort()
    n_src = len(srcs)
    n_pred = sum(1 for f in os.listdir(ddir) if f.endswith('_pred.png'))
    picks = [0, n_src // 2, n_src - 1] if n_src else []
    scores, stds = [], []
    for i in picks:
        im = cv2.imread(os.path.join(ddir, srcs[i][1]), cv2.IMREAD_GRAYSCALE)
        if im is None:
            scores.append(float('nan')); stds.append(float('nan')); continue
        f = im.astype(np.float32)
        scores.append(stripe(f)); stds.append(float(f.std()))
    smax = max([s for s in scores if s == s] or [0.0])
    sstd = min([s for s in stds if s == s] or [255.0])
    verdict = 'ok'
    reason = ''
    if smax > STRIPE_THRESH:
        verdict, reason = 'bad', 'horizontal-stripe artifact'
    elif sstd < BLANK_STD:
        verdict, reason = 'bad', 'flat/blank frames'
    return {
        'root': os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ddir)))),
        'case_id': case, 'sequence_id': seq, 'subdir': os.path.basename(ddir),
        'n_frames': n_src, 'n_src': n_src, 'n_pred': n_pred,
        'stripe_first': round(scores[0], 3) if scores else None,
        'stripe_mid': round(scores[1], 3) if len(scores) > 1 else None,
        'stripe_last': round(scores[2], 3) if len(scores) > 2 else None,
        'stripe_max': round(smax, 3),
        'frame_std_mid': round(stds[1], 2) if len(stds) > 1 else None,
        'verdict': verdict, 'reason': reason, 'path': ddir,
    }


def main():
    global STRIPE_THRESH
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--out', default=os.path.join(here, 'results_diag', 'bad_frames'))
    ap.add_argument('--roots', nargs='*', default=ROOTS)
    ap.add_argument('--thresh', type=float, default=STRIPE_THRESH)
    args = ap.parse_args()
    STRIPE_THRESH = args.thresh
    os.makedirs(args.out, exist_ok=True)

    jobs = []
    for root in args.roots:
        js = list_sequences(root)
        print(f'{root}: {len(js)} 个序列目录', flush=True)
        jobs += js
    print(f'合计待核验序列 {len(jobs)}', flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(check, j): j for j in jobs}
        for k, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            if k % 1000 == 0:
                print(f'  已核验 {k}/{len(jobs)}', flush=True)
    rows.sort(key=lambda r: (r['root'], r['case_id'], r['sequence_id']))

    fields = list(rows[0].keys())
    with open(os.path.join(args.out, 'all_sequences.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)

    bad = [r for r in rows if r['verdict'] == 'bad']
    with open(os.path.join(args.out, 'bad_sequences.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(bad)

    # 关联 CTFC 分支行
    csv_path = os.path.join(here, 'results_full_v2', 'ctfc_results.csv')
    linked = []
    bad_keys = {(r['case_id'], r['sequence_id']) for r in bad}
    if os.path.exists(csv_path):
        with open(csv_path, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                seq = row['序列ID']
                base = seq
                for suf in ('_LAD', '_LCX', '_RCA', '_LCA'):
                    if base.endswith(suf):
                        base = base[:-len(suf)]; break
                if (row['病例ID'], base) in bad_keys:
                    linked.append({'病例ID': row['病例ID'], '序列ID': seq,
                                   '分支类型': row['分支类型'], '总帧数': row['总帧数'],
                                   'F1': row['F1'], 'F2': row['F2'],
                                   'CTFC(浮点)': row['CTFC(浮点)'], '有效性': row['有效性']})
    if linked:
        with open(os.path.join(args.out, 'bad_ctfc_rows.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, list(linked[0].keys())); w.writeheader(); w.writerows(linked)

    # summary
    from collections import Counter
    by_T = Counter((r['n_frames'] // 100) * 100 for r in bad)
    summ = {
        'n_scanned': len(rows), 'n_bad': len(bad),
        'bad_frac': round(len(bad) / max(len(rows), 1), 4),
        'thresh': args.thresh,
        'bad_by_root': dict(Counter(r['root'] for r in bad)),
        'bad_by_Tbucket': {str(k): v for k, v in sorted(by_T.items())},
        'bad_T_values': dict(Counter(r['n_frames'] for r in bad).most_common(20)),
        'n_bad_ctfc_rows': len(linked),
        'bad_ctfc_valid_rows': sum(1 for r in linked if r['有效性'] == '有效'),
    }
    json.dump(summ, open(os.path.join(args.out, 'summary.json'), 'w'), ensure_ascii=False, indent=2)
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    print(f'\n输出目录: {args.out}')


if __name__ == '__main__':
    main()
