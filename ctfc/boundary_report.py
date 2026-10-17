#!/usr/bin/env python3
"""「承认边界」产物：把 5195 分支落成一条 status + 分层可测率 + 缺失随机性检查。

背景：全队列 5195 分支里只有 2273 到得了 FO 检测，再扣坏图/重复后 **1963** 才参与统计；
其中还有 F2==Fmax 20.7%。用户 2026-09-27 拍板「承认边界」：CTFC 定义为**可测子集**上的
指标，其余显式标记，不救。本脚本落地三件事：

  1. 分支级 status 四分类（口径由用户定）：
       measured      正常，CTFC 有效
       censored      F2==Fmax ⇒ CTFC 只有**下界**（片内远端未达）
       invalid       F2<F1（逆流）或 F2==F1（同帧到达）⇒ 生理不可能，剔除
       unmeasurable  没到检测就死了（空掩码/帧数过少/Fmax面积过小/坏图/重复）⇒ 无信息
  2. 分层可测率：by 分支类型、by unmeasurable 子因、by 帧数档。
  3. **缺失随机性检查**（决定 B 能否写进论文）：可测率若与分支/帧数系统相关，
     则 measured 子集有偏、CTFC 分布不可外推。给出分档可测率与 CTFC 分布。

分母口径同时给两种：
  全尝试 (5195) —— 诚实上限口径，含分割彻底失败者
  到检测 (2273) —— 只看掩码非空且帧数够的分支

输入： results_full_v2/ctfc_results.csv、results_f1q/f1q_rederived.csv、
       results_f1q/cohort/cohort_clean.csv
输出： results_f1q/cohort/ctfc_final.csv、results_f1q/cohort/boundary_report.json
用法： boundary_report.py
"""
import collections
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results_full_v2', 'ctfc_results.csv')
F1Q = os.path.join(HERE, 'results_f1q', 'f1q_rederived.csv')
COH = os.path.join(HERE, 'results_f1q', 'cohort', 'cohort_clean.csv')
OUT = os.path.join(HERE, 'results_f1q', 'cohort')

DIV = {'LAD': 1.7}                      # 其余为 1.0（Gibson 1996）


def num(s):
    s = (s or '').strip()
    try:
        return float(s)
    except ValueError:
        return None


def reason_class(txt):
    """排除理由自由文本 → 归一到子因集合（一行可含多个）。"""
    t = txt or ''
    out = []
    if '空分支掩码' in t or '无128像素' in t:
        out.append('空掩码')
    if '帧数过少' in t:
        out.append('帧数过少')
    if 'Fmax像素数过小' in t:
        out.append('Fmax面积过小')
    if '污染' in t:
        out.append('分支污染')
    return out


def main():
    coh = {}
    for r in csv.DictReader(open(COH, encoding='utf-8-sig')):
        coh[(r['病例ID'], r['序列ID'], r['分支类型'])] = r
    f1q = {}
    for r in csv.DictReader(open(F1Q, encoding='utf-8-sig')):
        f1q[(r['病例ID'], r['序列ID'], r['分支类型'])] = r

    rows = []
    for r in csv.DictReader(open(RES, encoding='utf-8-sig')):
        k = (r['病例ID'], r['序列ID'], r['分支类型'])
        c = coh.get(k, {})
        q = f1q.get(k, {})
        rec = {
            '病例ID': r['病例ID'], '序列ID': r['序列ID'], '分支类型': r['分支类型'],
            '组别': 'CAD',                       # 本批 5195 = 64 CAD 病例；CMVD/Others 未跑
            '总帧数': int(num(r['总帧数']) or 0),
            'Fmax': int(num(r['Fmax']) or -1),
            'Fmax面积': int(num(r['最大非零像素数']) or 0),
            '纳入': c.get('纳入', ''),
            'dup_keep': c.get('dup_keep', ''),
            '坏图': q.get('坏图', c.get('坏图', '')),
            '缺npz': q.get('缺npz', c.get('缺npz', '')),
            '排除理由': c.get('排除理由', ''),
        }
        div = DIV.get(r['分支类型'], 1.0)
        f1 = num(q.get('F1_q15'))
        f2 = num(q.get('F2'))
        rec['F1'] = None if f1 is None else int(f1)
        rec['F2'] = None if f2 is None else int(f2)
        rec['CTFC'] = num(q.get('CTFC_q15'))
        reasons = reason_class(rec['排除理由'])
        rec['子因'] = ';'.join(reasons)

        # ---- status 判定（优先级顺序即口径）----
        if rec['dup_keep'] == '0':
            rec['status'] = 'excluded_dup'
        elif rec['坏图'] == '是' or rec['纳入'] == '0' and not rec['缺npz']:
            rec['status'] = 'excluded_badframe'
        elif rec['缺npz'] == '是':
            rec['status'] = 'unmeasurable'
        elif rec['F2'] is None or rec['F1'] is None:
            rec['status'] = 'unmeasurable'
        elif rec['F2'] < rec['F1']:
            rec['status'] = 'invalid_retrograde'
        elif rec['F2'] == rec['F1']:
            rec['status'] = 'invalid_same'
        elif rec['F2'] >= rec['Fmax']:
            rec['status'] = 'censored'
        else:
            rec['status'] = 'measured'

        rec['CTFC_lb'] = None
        if rec['status'] == 'censored':
            rec['CTFC_lb'] = rec['CTFC']          # F2==Fmax ⇒ CTFC 是下界
        rec['F1_pinned'] = int(rec['F1'] is not None and rec['F1'] >= rec['Fmax'])
        rows.append(rec)

    fields = list(rows[0].keys())
    with open(os.path.join(OUT, 'ctfc_final.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    # ---------------- 分层统计 ----------------
    def pct(n, m):
        return round(100.0 * n / max(m, 1), 1)

    N_ALL = len(rows)
    reach = [r for r in rows if r['缺npz'] != '是']          # 有 npz（含坏图）
    usable = [r for r in rows if r['纳入'] == '1']           # 坏图已剔、含重复
    analysis = [r for r in rows if r['纳入'] == '1' and r['dup_keep'] == '1']

    def status_counts(rs):
        return dict(collections.Counter(r['status'] for r in rs))

    def cstats(vals):
        if not vals:
            return None
        a = np.array(vals, float)
        return {'n': len(a), 'median': round(float(np.median(a)), 2),
                'p10': round(float(np.percentile(a, 10)), 2),
                'p90': round(float(np.percentile(a, 90)), 2),
                'iqr': [round(float(np.percentile(a, 25)), 2),
                        round(float(np.percentile(a, 75)), 2)]}

    rep = {}
    rep['全尝试_5195'] = {'n': N_ALL, **status_counts(rows)}
    rep['有npz_2273'] = {'n': len(reach), **status_counts(reach)}
    rep['输入可用_1993'] = {'n': len(usable), **status_counts(usable)}
    rep['分析队列_1963'] = {'n': len(analysis), **status_counts(analysis)}
    rep['口径'] = {
        '坏图有npz数': sum(1 for r in reach if r['status'] == 'excluded_badframe'),
        'invalid_same里F1也钉在Fmax': sum(1 for r in analysis
                                    if r['status'] == 'invalid_same' and r['F1_pinned']),
        'F1钉在Fmax(分析队列)': sum(1 for r in analysis if r['F1_pinned']),
    }

    # 按分支类型（分析队列口径 + 全尝试口径的可测率）
    by_branch = {}
    for br in ('LAD', 'LCX', 'RCA'):
        rs = [r for r in rows if r['分支类型'] == br]
        ru = [r for r in usable if r['分支类型'] == br]
        ra = [r for r in analysis if r['分支类型'] == br]
        sc = status_counts(ra)
        sc.update({'n_全尝试': len(rs),
                   '可测率_全尝试%': pct(sum(1 for r in rs if r['status'] == 'measured'), len(rs)),
                   '可测率_输入可用%': pct(sum(1 for r in ru if r['status'] == 'measured'), len(ru)),
                   'censored%': pct(sc.get('censored', 0), len(ra)),
                   'invalid%': pct(sc.get('invalid_retrograde', 0) + sc.get('invalid_same', 0), len(ra)),
                   'CTFC_measured': cstats([r['CTFC'] for r in ra if r['status'] == 'measured'])})
        by_branch[br] = sc
    rep['按分支类型'] = by_branch

    # unmeasurable 子因（全尝试口径，按分支类型）
    sub = {}
    for br in ('LAD', 'LCX', 'RCA'):
        rs = [r for r in rows if r['分支类型'] == br and r['status'] == 'unmeasurable']
        c = collections.Counter()
        for r in rs:
            for s in (r['子因'].split(';') if r['子因'] else ['(未归类)']):
                c[s] += 1
        sub[br] = {'n': len(rs), **dict(c)}
    rep['unmeasurable子因'] = sub

    # ---- 缺失随机性检查 ----
    # (a) 帧数档 × 可测率（分析基础＝到检测行，看帧数是否影响走到 measured）
    tb = [0, 10, 20, 30, 45, 60, 90, 10 ** 9]
    frame_tab = []
    for i in range(len(tb) - 1):
        lo, hi = tb[i], tb[i + 1]
        rs = [r for r in rows if lo <= r['总帧数'] < hi]      # 全尝试口径
        rr = [r for r in rs if r['缺npz'] != '是']             # 有 npz
        rb = [r for r in rr if r['status'] != 'excluded_badframe']   # 输入可用
        if rs:
            frame_tab.append({'帧数∈': f'[{lo},{min(hi,10**8)})', 'n_全尝试': len(rs),
                              'n_可用': len(rb),
                              'unmeasurable%': pct(sum(1 for r in rs if r['status'] == 'unmeasurable'), len(rs)),
                              'censored%': pct(sum(1 for r in rb if r['status'] == 'censored'), len(rb)),
                              'measured%': pct(sum(1 for r in rb if r['status'] == 'measured'), len(rb))})
    rep['帧数档'] = frame_tab

    # (b) Fmax 面积档 × status（只看到检测行；面积小=分割弱）
    ab = [0, 800, 1500, 3000, 6000, 10 ** 9]
    area_tab = []
    for i in range(len(ab) - 1):
        lo, hi = ab[i], ab[i + 1]
        rs = [r for r in reach if lo <= r['Fmax面积'] < hi]
        if rs:
            area_tab.append({'Fmax面积∈': f'[{lo},{min(hi,10**8)})', 'n': len(rs),
                             'censored%': pct(sum(1 for r in rs if r['status'] == 'censored'), len(rs)),
                             'measured%': pct(sum(1 for r in rs if r['status'] == 'measured'), len(rs)),
                             'invalid%': pct(sum(1 for r in rs if r['status'].startswith('invalid')), len(rs))})
    rep['Fmax面积档'] = area_tab

    rep['CTFC_measured_总体'] = cstats([r['CTFC'] for r in analysis if r['status'] == 'measured'])
    rep['CTFC_censored下界_总体'] = cstats([r['CTFC_lb'] for r in analysis if r['status'] == 'censored'])

    # ---- 算法视角：把「非诊断性采集」从算法失责里摘出去 ----
    # T<10 是真实短采集（源图仅 3–9 帧），CTFC 按定义不可测，属**队列级**排除；
    # 剩下的（有足够帧、有造影机会）还测不出，才是**算法级**的账。
    short = [r for r in rows if r['总帧数'] < 10]
    alg_den = [r for r in rows if r['总帧数'] >= 10 and r['status'] != 'excluded_badframe']
    alg_ok = [r for r in alg_den if r['status'] in ('measured', 'censored', 'invalid_retrograde',
                                                    'invalid_same')]
    rep['算法视角'] = {
        '非诊断短采集_排除': len(short),
        '算法分母': len(alg_den),
        'measured': sum(1 for r in alg_den if r['status'] == 'measured'),
        'censored': sum(1 for r in alg_den if r['status'] == 'censored'),
        'invalid': sum(1 for r in alg_den if r['status'].startswith('invalid')),
        '分割失败': sum(1 for r in alg_den if r['status'] == 'unmeasurable'),
        '测出F1F2%': pct(len(alg_ok), len(alg_den)),
        '可测(measured)%': pct(sum(1 for r in alg_den if r['status'] == 'measured'), len(alg_den)),
        '面积过小行_Fmax面积中位': (float(np.median([r['Fmax面积'] for r in alg_den
                                              if r['status'] == 'unmeasurable' and
                                              'Fmax面积过小' in r['子因']])) if any(
            r['status'] == 'unmeasurable' and 'Fmax面积过小' in r['子因'] for r in alg_den) else None),
    }
    # 每分支的算法可测率
    rep['算法视角_按分支'] = {}
    for br in ('LAD', 'LCX', 'RCA'):
        d = [r for r in alg_den if r['分支类型'] == br]
        rep['算法视角_按分支'][br] = {
            'n': len(d),
            'measured%': pct(sum(1 for r in d if r['status'] == 'measured'), len(d)),
            'censored%': pct(sum(1 for r in d if r['status'] == 'censored'), len(d)),
            '分割失败%': pct(sum(1 for r in d if r['status'] == 'unmeasurable'), len(d))}
    json.dump(rep, open(os.path.join(OUT, 'boundary_report.json'), 'w'),
              ensure_ascii=False, indent=2)

    # ---------------- 打印 ----------------
    print(f"\n  队列：{N_ALL} 分支 / 64 CAD 病例（CMVD、Others 尚未跑管线）")
    for k in ('全尝试_5195', '有npz_2273', '输入可用_1993', '分析队列_1963'):
        d = rep[k]
        print(f'  {k:14s} n={d["n"]:5d}  ' + '  '.join(
            f'{s}={d.get(s,0)}' for s in ('measured', 'censored', 'invalid_retrograde',
                                          'invalid_same', 'unmeasurable', 'excluded_badframe',
                                          'excluded_dup') if d.get(s)))
    print(f"  口径：坏图有 npz {rep['口径']['坏图有npz数']}；invalid_same 里 F1 也钉 Fmax "
          f"{rep['口径']['invalid_same里F1也钉在Fmax']}；分析队列 F1 钉 Fmax "
          f"{rep['口径']['F1钉在Fmax(分析队列)']}")
    print('\n  分支    全尝试  可测率(全尝试/输入可用)  censored%  invalid%    CTFC中位[IQR]')
    for br, d in rep['按分支类型'].items():
        ct = d['CTFC_measured']
        s = f"{ct['median']} [{ct['iqr'][0]}-{ct['iqr'][1]}]" if ct else '—'
        print(f"  {br:6s} {d['n_全尝试']:7d}   {d['可测率_全尝试%']:6.1f}% / {d['可测率_输入可用%']:6.1f}%"
              f"    {d['censored%']:6.1f}    {d['invalid%']:6.1f}    {s}")
    print('\n  unmeasurable 子因（按分支）')
    for br, d in rep['unmeasurable子因'].items():
        parts = '  '.join(f'{k}={v}' for k, v in d.items() if k != 'n')
        print(f'  {br:6s} n={d["n"]:5d}  {parts}')
    print('\n  帧数档          n_全尝试 n_可用  unmeasurable%(全)  censored%(可用)  measured%(可用)')
    for d in rep['帧数档']:
        print(f"  {d['帧数∈']:14s} {d['n_全尝试']:8d} {d['n_可用']:6d}   {d['unmeasurable%']:14.1f}  "
              f"{d['censored%']:14.1f}  {d['measured%']:14.1f}")
    print('\n  Fmax面积档（到检测行）     n   censored%  measured%  invalid%')
    for d in rep['Fmax面积档']:
        print(f"  {d['Fmax面积∈']:14s} {d['n']:6d}   {d['censored%']:9.1f}  "
              f"{d['measured%']:9.1f}  {d['invalid%']:8.1f}")
    a = rep['算法视角']
    print(f"\n  算法视角（摘掉非诊断短采集 T<10）")
    print(f"    排非诊断短采集 {a['非诊断短采集_排除']}；算法分母 {a['算法分母']}"
          f"  ⇒ 测出 F1/F2 {a['测出F1F2%']}%，可测(measured) {a['可测(measured)%']}%")
    print(f"    censored {a['censored']}  invalid {a['invalid']}  分割失败 {a['分割失败']}"
          f"（这些行 Fmax 面积中位 {a['面积过小行_Fmax面积中位']} px）")
    for br, d in rep['算法视角_按分支'].items():
        print(f"    {br:4s} n={d['n']:5d}  measured {d['measured%']:5.1f}%  "
              f"censored {d['censored%']:5.1f}%  分割失败 {d['分割失败%']:5.1f}%")
    print(f"\n→ {OUT}/ctfc_final.csv  boundary_report.json")


if __name__ == '__main__':
    sys.exit(main())
