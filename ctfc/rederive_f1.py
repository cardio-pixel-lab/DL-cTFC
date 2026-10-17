#!/usr/bin/env python3
"""从已有 detailed/*.npz 重导出 F1（改 f1_percentile 无需重跑 2 小时的全队列）。

原理：`fo`（逐像素首充帧）与 `P`（近端区）都不依赖 `f1_percentile`，且两者都存进了 npz。
故 `F1_new = _fo_quantile(fo, P, frac_new)` 与「改 config 重跑」逐位等价。
先自检：frac=0.80 时必须逐行复现 CSV 里的 F1，否则中止。

只读 results_full_v2；输出全部写 --out（默认 results_f1q）。
"""
import argparse, csv, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOTS_DEFAULT = '/home/share/clr/share/data/AI_TMPFC/CAD-ID1-ID64'   # 仅用于记录来源，不读数据
BRANCH_SUF = re.compile(r'_(LAD|LCX|RCA|LCA)$')

# F1 依赖的备注（改 F1 后需重算）；其余备注保留
F1_DEP_NOTES = (r'^0帧已充盈', r'^同帧到达', r'^F2<F1', r'^CTFC过大',
                r'^未能检测到F1', r'^未能检测到F2')


def fo_quantile(fo, mask, frac):
    v = fo[mask > 0]
    if v.size == 0:
        return None
    uniq, cnts = np.unique(v, return_counts=True)
    cum = np.cumsum(cnts)
    idx = int(np.searchsorted(cum, frac * cum[-1], side='left'))
    return int(uniq[min(idx, len(uniq) - 1)])


def ctfc_of(f1, f2, branch, fr=2.0, lad=1.7):
    if f1 is None or f2 is None:
        return None, None
    if f2 < f1:
        return None, None
    raw = (f2 - f1) * fr
    return (raw / lad if branch == 'LAD' else raw), raw


def strip_f1_notes(notes):
    """去掉与 F1 相关的旧备注（CSV 中以 '; ' 分隔）。"""
    keep = []
    for n in (notes or '').split('; '):
        s = n.strip()
        if s and not any(re.search(p, s) for p in F1_DEP_NOTES):
            keep.append(s)
    return keep


def pipeline_eval(f1, f2, ctfc, max_ctfc, base_notes):
    """复刻 CTFCSequenceProcessor 的软 notes + CTFCCalculator.calculate。

    有效性只看 calculator 的 calc_ok（F1/F2 缺失、F2<F1、CTFC 过大），
    区域/软 notes 仅作备注，不置无效 —— 与 ctfc_analyzer.py:611-628 一致。
    返回 (notes, is_valid)。
    """
    notes = list(base_notes)
    if f1 == 0:
        notes.append('0帧已充盈, CTFC可能偏低')
    if f1 is not None and f2 is not None and f2 == f1:
        notes.append('同帧到达(快流)')
    if f1 is None or f2 is None:
        if f1 is None:
            notes.append('未能检测到F1')
        if f2 is None:
            notes.append('未能检测到F2')
        return notes, False
    if f2 < f1:
        notes.append(f'F2<F1 ({f2}<{f1}): 逆流/伪影，无效')
        return notes, False
    if ctfc is not None and ctfc > max_ctfc:
        notes.append(f'CTFC过大: {ctfc:.2f} > {max_ctfc}')
        return notes, False
    return notes, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(HERE, 'results_full_v2', 'ctfc_results.csv'))
    ap.add_argument('--detailed', default=os.path.join(HERE, 'results_full_v2', 'detailed'))
    ap.add_argument('--bad-csv', default=os.path.join(HERE, 'results_diag', 'bad_frames',
                                                     'all_sequences.csv'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results_f1q'))
    ap.add_argument('--qs', default='0.10,0.15,0.20,0.25,0.30,0.80')
    ap.add_argument('--max-ctfc', type=float, default=200.0)
    args = ap.parse_args()
    QS = [float(x) for x in args.qs.split(',')]
    os.makedirs(args.out, exist_ok=True)

    bad = set()
    if os.path.exists(args.bad_csv):
        with open(args.bad_csv, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if (r.get('verdict') or 'ok') != 'ok':
                    bad.add((r['case_id'], r['sequence_id']))
        print(f'坏图清单: {len(bad)} 条 (verdict!=ok)')

    rows = []
    with open(args.csv, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f'CSV 行数: {len(rows)}')

    def _int(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    out_rows, miss, mismatch, valid_mismatch = [], 0, [], []
    for r in rows:
        case, seq = r['病例ID'], r['序列ID']
        stem = f'{case}__{seq}'
        zp = os.path.join(args.detailed, stem + '.npz')
        branch = (r['分支类型'] or '').strip() or (
            BRANCH_SUF.search(seq).group(1) if BRANCH_SUF.search(seq) else 'RCA')
        f2 = _int(r['F2'])
        f1_old = _int(r['F1'])
        fmax = _int(r['Fmax'])
        T = _int(r['总帧数'])
        rec = {'病例ID': case, '序列ID': seq, '分支类型': branch, '总帧数': T, 'Fmax': fmax,
               'F1_old': f1_old, 'F2': f2, 'CTFC_old': r['CTFC(浮点)'],
               '有效性_old': r['有效性'], '备注_old': r['备注'],
               '坏图': '是' if (case, BRANCH_SUF.sub('', seq)) in bad else ''}
        if not os.path.exists(zp):
            miss += 1
            rec['缺npz'] = '是'
            out_rows.append(rec)
            continue
        z = np.load(zp)
        P = z['P'] > 0
        fo = z['fo']
        f1s = {q: fo_quantile(fo, P, q) for q in QS}
        if f1s[0.80] != f1_old:
            mismatch.append((stem, f1_old, f1s[0.80]))
        base_notes = strip_f1_notes(r['备注'])
        for q in QS:
            f1 = f1s[q]
            c, _ = ctfc_of(f1, f2, branch)
            notes, ok = pipeline_eval(f1, f2, c, args.max_ctfc, base_notes)
            tag = f'q{int(q*100)}'
            rec[f'F1_{tag}'] = f1
            rec[f'CTFC_{tag}'] = None if c is None else round(c, 2)
            rec[f'有效性_{tag}'] = '有效' if ok else '无效'
            rec[f'备注_{tag}'] = '; '.join(notes)
            if q == 0.80 and ok != (r['有效性'] == '有效'):
                valid_mismatch.append((stem, r['有效性'], f1, f2, c))
        # 分歧标记：q20 与 q80（旧主判据）差 >2
        if f1s[0.20] is not None and f1_old is not None:
            rec['q20_异旧'] = '是' if abs(f1s[0.20] - f1_old) > 2 else ''
        out_rows.append(rec)

    print(f'缺 detailed npz: {miss}')
    if mismatch:
        print(f'!! 自检失败：q0.80 与 CSV 的 F1 不符 {len(mismatch)} 条，例：{mismatch[:5]}')
    else:
        print('自检通过：q0.80 逐行复现 CSV 的 F1 ✓')
    if valid_mismatch:
        print(f'!! 自检失败：q0.80 有效性 与 CSV 不符 {len(valid_mismatch)} 条，例：{valid_mismatch[:5]}')
    else:
        print('自检通过：q0.80 逐行复现 CSV 的有效性 ✓')

    fields = list(out_rows[0].keys())
    for r in out_rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(os.path.join(args.out, 'f1q_rederived.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(out_rows)

    # ---- 汇总 ----
    def stats(sel, tag):
        f1k, ck, vk = f'F1_{tag}', f'CTFC_{tag}', f'有效性_{tag}'
        have = [r for r in sel if r.get('缺npz') != '是']
        f1v = [r[f1k] for r in have if r.get(f1k) is not None]
        cts = [r[ck] for r in have if r.get(ck) is not None and r.get(vk) == '有效']
        allvalid = [r for r in have if r.get(vk) == '有效']
        return {
            'n_rows': len(sel), 'n_npz': len(have),
            'F1==Fmax': round(np.mean([r[f1k] == r['Fmax'] for r in have
                                       if r.get(f1k) is not None]), 4) if f1v else None,
            'F1未到达': round(np.mean([r.get(f1k) is None for r in have]), 4),
            'F2<F1': round(np.mean([(r.get(f1k) is not None and r.get('F2') is not None
                                     and r['F2'] < r[f1k]) for r in have]), 4) if f1v else None,
            '有效数': len(allvalid), '有效占比': round(len(allvalid) / max(len(have), 1), 4),
            'CTFC中位': round(float(np.median(cts)), 2) if cts else None,
            'CTFC均值': round(float(np.mean(cts)), 2) if cts else None,
            'CTFC_P25': round(float(np.percentile(cts, 25)), 2) if cts else None,
            'CTFC_P75': round(float(np.percentile(cts, 75)), 2) if cts else None,
            'CTFC==0占比': round(float(np.mean([c == 0 for c in cts])), 4) if cts else None,
        }

    clean = [r for r in out_rows if r.get('坏图') != '是']
    summ = {
        'csv': args.csv, 'n_total': len(out_rows), 'n_bad_frames': sum(
            1 for r in out_rows if r.get('坏图') == '是'),
        'self_check_q080_mismatch': len(mismatch),
        'self_check_q080_valid_mismatch': len(valid_mismatch),
        'all': {}, 'clean(排除坏图)': {},
    }
    old_cts = [float(r['CTFC_old']) for r in out_rows
               if r.get('有效性_old') == '有效' and r.get('CTFC_old')]
    summ['old_基线'] = {
        'F1==Fmax': round(np.mean([r.get('F1_old') == r['Fmax'] for r in out_rows
                                   if r.get('F1_old') is not None]), 4),
        '有效数': sum(1 for r in out_rows if r.get('有效性_old') == '有效'),
        'CTFC中位': round(float(np.median(old_cts)), 2) if old_cts else None,
        'CTFC均值': round(float(np.mean(old_cts)), 2) if old_cts else None,
        'CTFC==0占比': round(float(np.mean([c == 0 for c in old_cts])), 4) if old_cts else None,
    }
    for q in QS:
        tag = f'q{int(q*100)}'
        summ['all'][tag] = stats(out_rows, tag)
        summ['clean(排除坏图)'][tag] = stats(clean, tag)
    json.dump(summ, open(os.path.join(args.out, 'f1q_summary.json'), 'w'), ensure_ascii=False, indent=2)
    print(json.dumps(summ['old_基线'], ensure_ascii=False, indent=2))
    for k in ('all', 'clean(排除坏图)'):
        print(f'\n--- {k} ---')
        for q in QS:
            s = summ[k][f'q{int(q*100)}']
            print(f"  q{int(q*100):<3} n={s['n_npz']:5d} F1==Fmax={s['F1==Fmax']} "
                  f"F1未到达={s['F1未到达']} F2<F1={s['F2<F1']} 有效={s['有效数']}"
                  f"({s['有效占比']}) CTFC中位={s['CTFC中位']} 均值={s['CTFC均值']} "
                  f"==0占{s['CTFC==0占比']}")
    print(f'\n输出: {args.out}/f1q_rederived.csv, f1q_summary.json')


if __name__ == '__main__':
    main()
