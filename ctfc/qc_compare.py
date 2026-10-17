#!/usr/bin/env python3
"""对比式 QC 页面：同一序列并排展示 old-F1 / 新分位 F1(可多档) / F2 / Fmax 帧。

用途：L1（f1_percentile 0.80 → 低分位）落地后，人工翻页核对每个序列的 F1 是否落在
「真首显影帧」。帧图沿用主管线口径（分支掩码绿 / P 青 / D 红 / S 黄轮廓）。

帧图来源：优先复用 `<old-frames>/<safe>_f<fid>.png`（主管线 --qc 已渲染的逐帧叠加图），
缺失的（典型是新分位 F1 对应的帧）用 render_qc_frame 现渲染，全部硬链/复制到 `--out/frames`，
以保证 HTML 只有一个相对目录。

只读 results_full_v2；全部输出写 --out（默认 results_f1q/qc）。
"""
import argparse
import csv
import html
import os
import re
import shutil

import cv2
import numpy as np

from ctfc_analyzer import MaskProcessor
from qc_report import (_collect_stems, _extract_branch_mask, render_qc_frame)

HERE = os.path.dirname(os.path.abspath(__file__))
BRANCH_SUF = re.compile(r'_(LAD|LCX|RCA|LCA)$')
LEGEND = 'green=branch  cyan=P(proximal)  red=D(distal)  yellow=S(seed/ostium)'
LEGEND_STRETCH = 'cyan=P(proximal)  red=D(distal)  yellow=S(seed)  (outline only, contrast stretched)'

CAP_H = 22


def _safe(case, seq):
    return re.sub(r'[^0-9A-Za-z_.-]', '_', f'{case}__{seq}')


def _outline(bgr, sub, color, thick=1):
    """只画轮廓、不填充 —— 保留亮度信息（判读显影必需）。"""
    if sub is None or not sub.any():
        return bgr
    cnts, _ = cv2.findContours((sub > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, cnts, -1, color, thick)
    return bgr


def render_stretch(src, P, D, S, caption, lo, hi, crop, scale=1.0):
    """裁剪到血管 bbox + 百分位拉伸 + 仅轮廓叠加（P 青 / D 红 / S 黄）。"""
    y0, y1, x0, x1 = crop
    g = src[y0:y1, x0:x1].astype(np.float32)
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    bgr = cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cy0, cx0 = y0, x0
    _outline(bgr, P[cy0:y1, cx0:x1] if P is not None else None, (255, 255, 0))
    _outline(bgr, D[cy0:y1, cx0:x1] if D is not None else None, (0, 0, 255))
    _outline(bgr, S[cy0:y1, cx0:x1] if S is not None else None, (0, 255, 255))
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    bgr = cv2.copyMakeBorder(bgr, CAP_H, 0, 0, 0, cv2.BORDER_CONSTANT, value=(38, 38, 38))
    cv2.putText(bgr, caption, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def _bbox(mask, margin, shape):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0, shape[0], 0, shape[1]
    h, w = shape[:2]
    return (max(0, ys.min() - margin), min(h, ys.max() + 1 + margin),
            max(0, xs.min() - margin), min(w, xs.max() + 1 + margin))


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _load_params(path):
    import yaml
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    m = cfg.get('mask_processing', {})
    return {'border_width': int(m.get('border_width', 30)),
            'min_component_area': int(m.get('min_component_area', 50)),
            'mask_suffix': str(cfg.get('dataset', {}).get('mask_suffix', '_pred.png'))}


def _link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _rows_html(rec, tiles, frame_rel):
    cells = []
    for label, fid in tiles:
        if fid is None:
            cells.append(f'<figure class="miss"><div class="ph">{label}<br>—</div></figure>')
            continue
        cells.append(
            f'<figure><img src="{frame_rel}/{rec["safe"]}_f{fid}.png" loading="lazy" '
            f'decoding="async" alt="{label} f={fid}"><figcaption>{label} (f={fid})</figcaption></figure>')
    flags = rec['flags']
    d = rec
    return f"""
<section class="card" data-flags="{','.join(flags)}">
  <div class="head">
    <span class="t">{html.escape(d['case'])}/{html.escape(d['seq'])} <b>[{d['branch']}]</b></span>
    <span class="pill">{d['valid_new']}</span>
    <span class="kv">F1: old=<b>{d['f1_old']}</b> → q{int(d['f1q']*100)}=<b class="hl">{d['f1_new']}</b> · </span>
    <span class="kv">F2=<b>{d['f2']}</b> · Fmax=<b>{d['fmax']}</b> · n={d['n']} · ΔF1=<b>{d['df1']}</b></span>
    <span class="kv">CTFC: old=<b>{d['ctfc_old']}</b> → new=<b class="hl">{d['ctfc_new']}</b></span>
    <span class="notes">{html.escape(d['notes'])}</span>
  </div>
  <div class="strip">{''.join(cells)}</div>
</section>"""


CSS = """
body{font:13px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#12141a;color:#e6e8ee}
header{position:sticky;top:0;background:#1a1d26;padding:10px 16px;border-bottom:1px solid #2b3040;z-index:9}
h1{font-size:15px;margin:0 0 6px}
.filters button{background:#232838;color:#cfd4e2;border:1px solid #333a4d;border-radius:14px;padding:3px 11px;
  margin:2px 4px 2px 0;cursor:pointer;font-size:12px}
.filters button.on{background:#3b5bdb;color:#fff;border-color:#3b5bdb}
.card{background:#181b23;border:1px solid #262c3a;border-radius:8px;margin:10px 14px;padding:8px 10px}
.head{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center;margin-bottom:6px}
.t{font-size:13px;color:#fff}
.pill{font-size:11px;padding:1px 8px;border-radius:10px;background:#2b3142}
.pill.ok{background:#2b8a3e}.pill.bad{background:#c92a2a}
.kv{font-size:12px;color:#aab2c5}
.hl{color:#ffd43b}
.notes{font-size:11px;color:#7d8597}
.strip{display:flex;gap:6px;flex-wrap:wrap}
figure{margin:0;flex:0 0 auto}
figure img{width:236px;height:auto;display:block;border:1px solid #2b3040;border-radius:4px;background:#000;cursor:zoom-in}
figure img.big{width:auto;max-width:min(94vw,1100px);cursor:zoom-out}
figcaption{font-size:11px;color:#9aa3b6;text-align:center;padding-top:2px}
figure.miss .ph{width:236px;height:236px;display:flex;align-items:center;justify-content:center;
  border:1px dashed #333a4d;border-radius:4px;color:#5c6478;font-size:11px;text-align:center}
"""

JS = """
const state={f:new Set()};
function apply(){
  document.querySelectorAll('.card').forEach(c=>{
    const fl=(c.dataset.flags||'').split(',').filter(Boolean);
    const ok=[...state.f].every(x=>fl.includes(x));
    c.style.display = ok?'':'none';
  });
  const vis=document.querySelectorAll('.card:not([style*="none"])').length;
  document.getElementById('cnt').textContent='显示 '+vis+' / '+document.querySelectorAll('.card').length;
}
document.querySelectorAll('.filters button').forEach(b=>{
  b.onclick=()=>{const f=b.dataset.f;
    if(f==='__all'){state.f.clear();document.querySelectorAll('.filters button').forEach(x=>x.classList.remove('on'));apply();return;}
    state.f.has(f)?state.f.delete(f):state.f.add(f);
    b.classList.toggle('on');apply();};
});
document.querySelectorAll('figure img').forEach(im=>{
  im.onclick=()=>im.classList.toggle('big');
});
apply();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(HERE, 'results_f1q', 'f1q_rederived.csv'))
    ap.add_argument('--detailed', default=os.path.join(HERE, 'results_full_v2', 'detailed'))
    ap.add_argument('--old-frames', default=os.path.join(HERE, 'results_full_v2', 'qc', 'frames'))
    ap.add_argument('--params', default=os.path.join(HERE, 'results_full_v2', 'config.used.yaml'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results_f1q', 'qc'))
    ap.add_argument('--f1q', type=float, default=0.15, help='主推分位')
    ap.add_argument('--also', default='0.20', help='并排展示的其它分位（逗号分隔，可空）')
    ap.add_argument('--style', choices=('overlay', 'stretch'), default='overlay',
                    help='overlay=掩码填充叠加（复用主管线帧）；stretch=裁剪+拉伸+仅轮廓（判读显影用）')
    ap.add_argument('--margin', type=int, default=20, help='stretch 风格裁剪外扩像素')
    ap.add_argument('--scale', type=float, default=1.0, help='stretch 风格放大倍数')
    ap.add_argument('--limit', type=int, default=0, help='>0 时只处理前 N 条（调试）')
    ap.add_argument('--include-bad', action='store_true', help='包含条纹坏图序列')
    args = ap.parse_args()
    q = args.f1q
    tag_q = f'q{int(round(q * 100))}'
    also = [float(x) for x in args.also.split(',') if x.strip()]
    params = _load_params(args.params)
    suffix = params['mask_suffix']

    import importlib.util
    spec = importlib.util.spec_from_file_location('poc1', os.path.join(HERE, 'fo_intensity_poc.py'))
    poc1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(poc1)

    frame_dir = os.path.join(args.out, 'frames')
    os.makedirs(frame_dir, exist_ok=True)

    rows = list(csv.DictReader(open(args.csv, encoding='utf-8-sig')))
    cards, n_new, miss_npz, miss_frames = [], 0, 0, 0
    for r in rows:
        if r.get('坏图') == '是' and not args.include_bad:
            continue
        case, seq = r['病例ID'], r['序列ID']
        stem = f'{case}__{seq}'
        zp = os.path.join(args.detailed, stem + '.npz')
        if not os.path.exists(zp):
            miss_npz += 1
            continue
        f1_old = _int(r['F1_old'])
        f1_new = _int(r[f'F1_{tag_q}'])
        f2 = _int(r['F2'])
        fmax = _int(r['Fmax'])
        n = _int(r['总帧数'])
        safe = _safe(case, seq)

        tiles = [('F1_old', f1_old), (f'F1_{tag_q}', f1_new)]
        for a in also:
            tiles.append((f'F1_q{int(round(a * 100))}', _int(r[f'F1_q{int(round(a * 100))}'])))
        tiles += [('F2', f2), ('Fmax', fmax)]

        # 目标帧图画布
        want = sorted({f for _, f in tiles if f is not None and 0 <= f < (n or 0)})
        z = np.load(zp)
        S, P, D = z['seed'], z['P'], z['D']
        branch_type = (r['分支类型'] or '').strip() or (
            BRANCH_SUF.search(seq).group(1) if BRANCH_SUF.search(seq) else 'RCA')
        jp = os.path.join(args.detailed, stem + '.json')
        try:
            import json as _json
            fill_p = _json.load(open(jp, encoding='utf-8')).get('fill_p')
        except (OSError, ValueError):
            fill_p = None

        if args.style == 'stretch':
            # 裁剪到 extent bbox + 百分位拉伸 + 仅轮廓：为「判读显影」优化，不复用旧帧
            imgs, root, segdir = poc1.load_frames(case, seq)
            if imgs is None:
                miss_frames += 1
                continue
            crop = _bbox(z['extent'], args.margin, imgs[0].shape)
            y0, y1, x0, x1 = crop
            stack = np.stack([im[y0:y1, x0:x1] for im in imgs]).astype(np.float32)
            lo, hi = float(np.percentile(stack, 2)), float(np.percentile(stack, 99.5))
            for fid in want:
                if fid >= len(imgs):
                    continue
                cap = (f'{case}/{seq} [{branch_type}] f={fid}  '
                       f'fill_p={fill_p[fid]:.2f}' if fill_p and fid < len(fill_p)
                       else f'{case}/{seq} [{branch_type}] f={fid}')
                out = render_stretch(imgs[fid], P, D, S, cap, lo, hi, crop, args.scale)
                cv2.imwrite(os.path.join(frame_dir, f'{safe}_f{fid}.png'), out)
                n_new += 1
        else:
            # overlay 风格：优先复用主管线已渲染帧，缺失的现渲染
            need_render = [f for f in want if not os.path.exists(os.path.join(args.old_frames, f'{safe}_f{f}.png'))]
            if need_render:
                imgs, root, segdir = poc1.load_frames(case, seq)
                if imgs is None:
                    miss_frames += 1
                else:
                    heads = _collect_stems(segdir, suffix)
                    for fid in need_render:
                        head = heads.get(fid)
                        if head is None or fid >= len(imgs):
                            continue
                        src = imgs[fid]
                        raw = MaskProcessor.load_mask(os.path.join(segdir, head + suffix))
                        if raw is None:
                            continue
                        bm = _extract_branch_mask(raw, branch_type, params, src.shape[:2])
                        cap = f'{case}/{seq} [{branch_type}] new F1 f={fid}'
                        out = render_qc_frame(src, bm, S, P, D, cap, LEGEND)
                        cv2.imwrite(os.path.join(frame_dir, f'{safe}_f{fid}.png'), out)
                        n_new += 1
            for f in want:
                old = os.path.join(args.old_frames, f'{safe}_f{f}.png')
                if os.path.exists(old):
                    _link_or_copy(old, os.path.join(frame_dir, f'{safe}_f{f}.png'))

        flags = []
        if f1_old is not None and f1_new is not None and abs(f1_new - f1_old) > 2:
            flags.append('df1')
        if f1_new == 0:
            flags.append('f1_zero')
        if f1_new is not None and fmax is not None and f1_new == fmax:
            flags.append('f1_fmax')
        if f2 is not None and fmax is not None and f2 == fmax:
            flags.append('f2_fmax')
        if r[f'有效性_{tag_q}'] != r['有效性_old']:
            flags.append('valid_chg')

        cards.append(_rows_html({
            'safe': safe, 'case': case, 'seq': seq, 'branch': r['分支类型'],
            'f1_old': f1_old if f1_old is not None else '—',
            'f1_new': f1_new if f1_new is not None else '—',
            'f1q': q, 'f2': f2 if f2 is not None else '—',
            'fmax': fmax, 'n': n,
            'df1': (f1_new - f1_old) if (f1_old is not None and f1_new is not None) else '—',
            'ctfc_old': r['CTFC_old'] or '—', 'ctfc_new': r[f'CTFC_{tag_q}'] or '—',
            'valid_new': r[f'有效性_{tag_q}'],
            'notes': r[f'备注_{tag_q}'] or r['备注_old'],
            'flags': flags,
        }, tiles, 'frames'))
        if args.limit and len(cards) >= args.limit:
            break

    filts = [('__all', '全部'), ('df1', '|ΔF1|>2'), ('f1_zero', f'F1_{tag_q}==0'),
             ('f1_fmax', f'F1_{tag_q}==Fmax'), ('f2_fmax', 'F2==Fmax'), ('valid_chg', '有效性变化')]
    btns = ''.join(f'<button data-f="{f}">{l}</button>' for f, l in filts)
    doc = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>CTFC F1 对比 QC (old q80 vs q{int(round(q*100))}, {args.style})</title><style>{CSS}</style></head><body>
<header><h1>F1 对比 QC：old(q80) vs 新分位 q{int(round(q*100))}
{' + q' + '/q'.join(str(int(round(a*100))) for a in also) if also else ''} <span class="kv">[{args.style}]</span></h1>
<div class="filters">{btns}</div>
<div class="kv" id="cnt" style="margin-top:6px">渲染帧 {n_new} 张 · 缺 npz {miss_npz} · 缺源帧 {miss_frames}</div>
</header>
{''.join(cards)}
<div class="kv" style="padding:12px">legend: {LEGEND_STRETCH if args.style == 'stretch' else LEGEND}</div>
<script>{JS}</script></body></html>"""
    idx = os.path.join(args.out, 'qc_compare.html')
    with open(idx, 'w', encoding='utf-8') as f:
        f.write(doc)
    print(f'卡片 {len(cards)} · 新渲染帧 {n_new} · 缺npz {miss_npz} · 缺源帧 {miss_frames}')
    print(f'页面: {idx}')


if __name__ == '__main__':
    main()
