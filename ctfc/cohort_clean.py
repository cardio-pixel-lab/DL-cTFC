#!/usr/bin/env python3
"""固化分析队列口径：坏图排除清单 + 全队列内容级去重 + 干净队列定义（一次性）。

两件事以前是散的：
  1. 坏图（290 条 T∈{1024,512} 的横向条纹图）只存在于 results_diag/bad_frames/，各脚本靠
     读那个路径间接排除；41 条坏图序列连 ctfc_results.csv 的行都没有。
  2. 存在**逐字节相同的重复序列**（已确认 36-Niejianbin__36-IM000391 ≡ __36-IM000450：
     f1/f150 的 _src/_pred md5 全同、npz md5 全同）⇒ 不 dedup 会在队列层双计。

本脚本把两者落成自包含清单，并给出唯一的口径文件：
  下游一律读 cohort/cohort_clean.csv，筛 `纳入==1 且 dup_keep==1`。

去重口径：对每条序列算「按文件名排序的 *_src.png **内容 md5 串联**」的摘要（**不含文件名**，
否则同族序列因文件名带各自序列号而永不相等）。完全相同才算重复。代表（dup_keep=1）只在
**纳入行**之间选，优先有 npz、再次序列号最小；与「纳入」正交，坏图组间的偶然碰撞不影响入选。
摘要缓存进 cohort/_seq_digest.json，重跑秒级。

输出： results_f1q/cohort/{cohort_clean.csv, exclusions_badframes.csv, duplicates.csv,
                        summary.json, _seq_digest.json}
用法： cohort_clean.py [--procs N] [--rehash]
"""
import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'results_f1q', 'f1q_rederived.csv')
BAD = os.path.join(HERE, 'results_diag', 'bad_frames', 'all_sequences.csv')
DET = os.path.join(HERE, 'results_full_v2', 'detailed')
OUT = os.path.join(HERE, 'results_f1q', 'cohort')
CACHE = os.path.join(OUT, '_seq_digest.json')
BRANCH_SUF = re.compile(r'_(LAD|LCX|RCA|LCA)$')

FIELDS = ['病例ID', '序列ID', '分支类型', '总帧数', 'Fmax', '坏图', '缺npz',
          '纳入', '排除理由', 'dup_group', 'dup_kept', 'dup_keep', 'n_dup']


def seq_digest(path):
    """序列内容摘要：按文件名排序的全部 *_src.png 内容 md5 串联（不含文件名）。"""
    try:
        fns = sorted(f for f in os.listdir(path) if f.endswith('_src.png'))
    except OSError:
        return None, 0
    if not fns:
        return None, 0
    h = hashlib.md5()
    for fn in fns:
        try:
            with open(os.path.join(path, fn), 'rb') as fh:
                h.update(hashlib.md5(fh.read()).digest())
        except OSError:
            return None, 0
    return h.hexdigest(), len(fns)


def _job(t):
    return t[0], seq_digest(t[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=max(1, (os.cpu_count() or 4) // 4))
    ap.add_argument('--rehash', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    # ---- 坏图清单（序列级字符串键） ----
    bad = {}
    if os.path.exists(BAD):
        for r in csv.DictReader(open(BAD, encoding='utf-8-sig')):
            if (r.get('verdict') or 'ok') != 'ok':
                bad[f"{r['case_id']}__{r['sequence_id']}"] = r
    print(f'坏图序列 {len(bad)} 条（T 分布 {dict(Counter(int(r["n_frames"]) for r in bad.values()))}）')
    with open(os.path.join(OUT, 'exclusions_badframes.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['case_id', 'sequence_id', 'n_frames', 'stripe_max',
                                         'frame_std_mid', 'verdict', 'reason', 'path'],
                           extrasaction='ignore')
        w.writeheader()
        for k in sorted(bad):
            w.writerow(bad[k])

    # ---- 序列目录 → 内容摘要（带缓存） ----
    seqs = {}
    if os.path.exists(BAD):
        for r in csv.DictReader(open(BAD, encoding='utf-8-sig')):
            seqs[f"{r['case_id']}__{r['sequence_id']}"] = r['path']
    dig = {}
    if os.path.exists(CACHE) and not args.rehash:
        dig = json.load(open(CACHE))
        dig = {k: v for k, v in dig.items() if k in seqs}
        print(f'读缓存摘要 {len(dig)}/{len(seqs)}')
    todo = [k for k in seqs if k not in dig]
    if todo:
        print(f'去重扫描 {len(todo)} 个序列目录 …')
        import multiprocessing as mp
        with mp.Pool(args.procs) as pool:
            for k, (d, n) in pool.imap_unordered(_job, [(k, seqs[k]) for k in todo],
                                                 chunksize=8):
                dig[k] = d
        json.dump(dig, open(CACHE, 'w'))
    print(f'有摘要 {sum(1 for v in dig.values() if v)}/{len(seqs)}')
    groups = defaultdict(list)
    for k, d in dig.items():
        if d:
            groups[d].append(k)
    all_dups = {d: sorted(v) for d, v in groups.items() if len(v) > 1}
    print(f'全量重复组 {len(all_dups)}，涉及序列 {sum(len(v) for v in all_dups.values())} 条')

    # ---- 逐行判定 ----
    rows = list(csv.DictReader(open(CSV, encoding='utf-8-sig')))
    npz_ks = set()
    for r in rows:
        r['_s'] = BRANCH_SUF.sub('', r['序列ID'])
        if os.path.exists(os.path.join(DET, f"{r['病例ID']}__{r['序列ID']}.npz")):
            npz_ks.add(f"{r['病例ID']}__{r['_s']}")
    for r in rows:
        r['_k'] = f"{r['病例ID']}__{r['_s']}"
        r['_inc'] = 0 if (r['_k'] in bad or r.get('缺npz') == '是') else 1
        if r['_k'] in bad:
            r['_why'] = 'bad_frame'
        elif r.get('缺npz') == '是':
            note = (r.get('备注_old') or '').strip()
            r['_why'] = 'no_npz' + (f':{note}' if note else '')
        else:
            r['_why'] = ''

    # 只在「该序列有纳入行」的成员里选重复代表
    inc_by_k = defaultdict(int)
    for r in rows:
        inc_by_k[r['_k']] += r['_inc']
    canon = {}
    for d, v in all_dups.items():
        live = [m for m in v if inc_by_k.get(m)]
        if len(live) > 1:
            canon[d] = sorted(live, key=lambda x: (x not in npz_ks, x))
    print(f'重复组中有 >1 个纳入序列的：{len(canon)} 组')

    out, dup_rows = [], 0
    for r in rows:
        d = dig.get(r['_k'])
        grp = all_dups.get(d, []) if d else []
        c = canon.get(d) if d else None
        if c:
            kept = c[0]
            r['_kept'], r['_keep'] = kept, 1 if r['_k'] == kept else 0
            dup_rows += 1 if r['_inc'] else 0
        else:
            r['_kept'], r['_keep'] = r['_k'], 1
        out.append({'病例ID': r['病例ID'], '序列ID': r['序列ID'],
                    '分支类型': r.get('分支类型', ''), '总帧数': r.get('总帧数', ''),
                    'Fmax': r.get('Fmax', ''), '坏图': r.get('坏图', ''),
                    '缺npz': r.get('缺npz', ''), '纳入': r['_inc'], '排除理由': r['_why'],
                    'dup_group': r['_kept'] if len(grp) > 1 else '',
                    'dup_kept': ';'.join(c) if c else '', 'dup_keep': r['_keep'],
                    'n_dup': len(grp) if grp else 1})
    with open(os.path.join(OUT, 'cohort_clean.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(out)

    with open(os.path.join(OUT, 'duplicates.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['dup_group', '成员数', '纳入成员数', '成员', '代表', '全是坏图'])
        for d, v in sorted(all_dups.items(), key=lambda t: -len(t[1])):
            live = [m for m in v if inc_by_k.get(m)]
            w.writerow([v[0], len(v), len(live), ';'.join(v),
                        (canon.get(d) or [v[0]])[0],
                        '是' if all(m in bad for m in v) else ''])

    summ = {
        'n_rows': len(rows), 'n_seq_dirs': len(seqs), 'n_digest': sum(1 for v in dig.values() if v),
        'n_dup_groups_all': len(all_dups),
        'n_dup_groups_among_included': len(canon),
        'n_dup_seqs_all': sum(len(v) for v in all_dups.values()),
        'n_dup_seqs_among_included': sum(len(c) for c in canon.values()),
        'n_dup_rows': dup_rows,
        'n_dup_rows_dropped': sum(1 for r in out if r['纳入'] and r['dup_keep'] == 0),
        'n_excluded_bad': sum(1 for r in out if r['排除理由'] == 'bad_frame'),
        'n_excluded_no_npz': sum(1 for r in out if r['排除理由'].startswith('no_npz')),
        'n_included': sum(r['纳入'] for r in out),
        'n_analysis': sum(1 for r in out if r['纳入'] and r['dup_keep']),
        'excluded_reason_top': dict(Counter(
            r['排除理由'] for r in out if r['排除理由']).most_common(8)),
        'secs': round(time.time() - t0, 1),
    }
    json.dump(summ, open(os.path.join(OUT, 'summary.json'), 'w'), ensure_ascii=False, indent=2)
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    print('→', OUT)


if __name__ == '__main__':
    main()
