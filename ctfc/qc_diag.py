#!/usr/bin/env python3
"""诊断用并排核对页面：`_src` 与 `_pred` 逐帧对照（小样本）。

与 `qc_report.py` 的区别：那边是「算法结果核对」（叠 S/P/D 看 F1/F2 合不合理），
这边是「输入可信度诊断」——同一帧把源图和分割掩码并排摆出来，判定
**掩码本身能不能用来做帧计数**：

  1. `_pred` 是否跟得上 `_src` 里的造影剂（分割质量）；
  2. 128/LCX 与 255/LAD 是否在帧间来回跳（类别翻转，会毁掉 FO）；
  3. 「空分支掩码」的序列，`_src` 里到底有没有造影。

按 4 类各抽 N 条（干净有效 / F2==F1 / F2<F1 / 空分支掩码），
每分支铺 7 帧（F1±1 / F2±1 / Fmax；无 F1/F2 的改用全程均匀取样），
每帧两格：左 src（全分支统一拉伸）| 右 pred（128=绿 / 255=红，烧录各计数）。
有效行另加一行 Fmax 的 S/P/D 叠加图，便于把区域和掩码对上。

用法:
  python qc_diag.py --results results_full_v2/ctfc_results.csv \\
      --root-dir /home/share/clr/share/data/AI_TMPFC/CAD-ID1-ID64 \\
      --detailed results_full_v2/detailed \\
      --out results_diag --per-class 8
"""

import argparse
import csv
import html
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import qc_report as qr
from ctfc_analyzer import MaskProcessor

# 类别：(flag, 显示名, 诊断提示)
CLASSES: List[Tuple[str, str, str]] = [
    ('clean', '干净有效', '阴性对照：src 应能看到造影由近端向远端推进，pred 应同步跟上'),
    ('f2eqf1', 'F2==F1', 'F1/F2 塌缩到同一帧：看 pred 是否只在最后一帧才出现大片分支'),
    ('f2ltf1', 'F2<F1', '远端被判定早于近端：看 pred 是否帧间跳变/类别翻转'),
    ('emptymask', '空分支掩码', '守卫拦截：看 src 里到底有没有造影（无造影 vs 分割全失败）'),
]
CLASS_LABEL = {c[0]: c[1] for c in CLASSES}
CLASS_HINT = {c[0]: c[2] for c in CLASSES}

# pred 类别配色（BGR）
PRED_COLORS = {128: (0, 200, 0), 255: (0, 0, 255)}   # 128=LCX 绿 / 255=LAD 红
PRED_COLOR_OTHER = (0, 255, 255)                      # 其他非零 黄

CAPTION_H = qr.CAPTION_H
CELL = 226


# ------------------------------------------------------------------ CSV 读取

def _col(row: Dict[str, str], key: str) -> str:
    for h in row:
        if h is not None and h.strip() == key:
            return row[h]
    return ''


def _int(v: str) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _float(v: str) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def classify(row: Dict[str, str]) -> Optional[str]:
    """把一行归入 4 类之一；不属于任何类返回 None。"""
    notes = _col(row, '备注') or ''
    valid = _col(row, '有效性') == '有效'
    f1, f2 = _int(_col(row, 'F1')), _int(_col(row, 'F2'))
    ctfc = _float(_col(row, 'CTFC(浮点)'))
    if not valid and '空分支掩码' in notes:
        return 'emptymask'
    if not valid and 'F2<F1' in notes:
        return 'f2ltf1'
    if valid and f1 is not None and f2 is not None and f1 == f2:
        return 'f2eqf1'
    if valid and ctfc is not None and ctfc > 0 and '退化' not in notes:
        return 'clean'
    return None


def sample_spread(rows: List[Tuple[int, Dict[str, str]]], n: int) -> List[Tuple[int, Dict[str, str]]]:
    """按序均匀取样，同一病例最多取 1 条；不够则放宽补足。"""
    picked, used_case = [], set()
    if len(rows) <= n:
        return rows
    idxs = sorted({round(i * (len(rows) - 1) / (n - 1)) for i in range(n)})
    for i in idxs:
        case = _col(rows[i][1], '病例ID')
        if case in used_case:
            continue
        used_case.add(case)
        picked.append(rows[i])
    if len(picked) < n:
        have = {i for i, _ in picked}
        for i, r in rows:
            if len(picked) >= n:
                break
            if i not in have:
                picked.append((i, r))
    return picked


# ------------------------------------------------------------------ 路径解析

def resolve_segmented_dir(root_dir: str, case_id: str, seq_id: str,
                          branch_type: str, candidates: List[str]) -> Optional[str]:
    """复刻 CTFCDatasetProcessor.process_dataset 的目录解析。

    LCA 拆支后 results 里的 sequence_id 是 `{seq_dir}_LAD` / `_LCX`，
    但磁盘上的目录名只有 `{seq_dir}`。
    """
    base = seq_id
    if branch_type in ('LAD', 'LCX') and seq_id.endswith('_' + branch_type):
        base = seq_id[: -len(branch_type) - 1]
    seq_path = os.path.join(root_dir, case_id, base)
    if not os.path.isdir(seq_path):
        return None
    for dname in candidates:
        cand = os.path.join(seq_path, dname)
        if os.path.isdir(cand):
            return cand
    return None


FrameGroup = Tuple[str, List[Tuple[str, Optional[int]]]]


def frame_specs(f1: Optional[int], f2: Optional[int], fmax: Optional[int],
                n_frames: Optional[int]) -> List[FrameGroup]:
    """帧布局：有 F1/F2 用 7 帧连拍（F1±1 / F2±1 / Fmax），否则全程均匀取样。

    返回 [(组标题, [(帧标签, 帧号), ...]), ...]。
    """
    n = n_frames or 0
    if f1 is not None and f2 is not None and n > 0:
        rows = qr.qc_frame_rows(f1, f2, fmax if fmax is not None else n - 1, n)
        if f1 == f2:
            return [('F1==F2 附近', rows[0]), ('Fmax', rows[2])]
        return [('F1 附近', rows[0]), ('F2 附近', rows[1]), ('Fmax', rows[2])]

    def ok(v):
        return None if v is None or v < 0 or v >= n else int(v)

    picks: List[Tuple[str, Optional[int]]] = []
    for label, fid in (('f=0', 0), ('25%', n // 4), ('50%', n // 2),
                       ('75%', (3 * n) // 4), ('Fmax', fmax)):
        fid = ok(fid)
        if fid is not None and all(fid != f for _, f in picks):
            picks.append((label, fid))
    if not picks:
        picks = [('f=0', 0)] if n > 0 else [('Fmax', fmax)]
    return [('全程取样', picks)]


# ------------------------------------------------------------------ 渲染

def render_src(gray: np.ndarray, caption: str) -> np.ndarray:
    """原样显示源图灰度（不做归一化：帧间亮度本身就是要看的信息）。"""
    return _with_caption(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), caption)


def render_pred(raw: np.ndarray, caption: str) -> np.ndarray:
    """把 128/255 渲染成不同颜色，便于肉眼抓类别翻转。"""
    img = np.zeros((raw.shape[0], raw.shape[1], 3), np.uint8)
    for val, col in PRED_COLORS.items():
        img[raw == val] = col
    rest = (raw > 0) & ~np.isin(raw, list(PRED_COLORS))
    img[rest] = PRED_COLOR_OTHER
    return _with_caption(img, caption)


def _with_caption(img: np.ndarray, caption: str) -> np.ndarray:
    bar = img[0:CAPTION_H].copy()
    img[0:CAPTION_H] = (0.25 * bar).astype(np.uint8)
    qr._put_text(img, caption, (8, 21), 0.5)
    return img


# ------------------------------------------------------------------ HTML

def _extra_css() -> str:
    return """
  .cell.label-src img { border-color: #3a6fa8; }
  .cell.label-pred img { border-color: #8a8f99; }
  .cell.label-ov img { border-color: #9a6fbf; }
  .hint { font-size: 12px; color: #c8a45a; flex-basis: 100%; }
  .pair { display: flex; gap: 8px; }
  .grouptitle { font-size: 12px; font-weight: 600; color: #8fb4f0; width: 100%;
                border-bottom: 1px solid #2e323c; padding: 2px 0 3px; margin-top: 4px; }
  .cls-pill { background: #2a3550; color: #9fc0f5; font-size: 12px; padding: 2px 9px;
              border-radius: 10px; font-weight: 600; }
  .predlegend { font-size: 12px; color: #b9c0cb; }
  .predlegend i { display: inline-block; width: 11px; height: 11px; border-radius: 3px;
                  margin: 0 4px 0 10px; vertical-align: -1px; }
  .sw-lad { background: #ff0000; }
  .sw-lcx { background: #00c800; }
"""


def _select_opts() -> str:
    btns = ''.join(f'<button data-filter="{f}">{lbl}</button>' for f, lbl, _ in CLASSES)
    return f"""
    <button data-filter="all" class="active">全部</button>
    {btns}
    <button data-filter="invalid">无效</button>
    <button data-filter="pending">待审核</button>
    <input id="q" placeholder="搜索 病例/序列 ...">
    <span class="spacer"></span>
    <button id="export">导出审核结果 CSV</button>
    """


def _render_card(case_id: str, seq_id: str, row: Dict[str, str], cls: str,
                 specs, frame_dir_rel: str, safe: str, saved: Dict[str, set]) -> str:
    branch = _col(row, '分支类型')
    valid = _col(row, '有效性') == '有效'
    f1, f2 = _col(row, 'F1'), _col(row, 'F2')
    ctfc = _col(row, 'CTFC(浮点)')
    notes = _col(row, '备注') or ''
    rid = html.escape(f"{case_id}/{seq_id}")
    flags = [cls] + ([] if valid else ['invalid'])

    def cell(kind: str, label: str, fid: Optional[int]) -> str:
        cls_ = 'cell label-' + kind
        if fid is None or fid not in saved.get(kind, set()):
            return (f'<figure class="{cls_} missing"><div class="ph">{label}<br>'
                    f'{"帧号越界" if fid is None else "无图"}</div></figure>')
        cap = f"{rid} [{branch}] {kind} {label} f={fid}"
        src = f"{frame_dir_rel}/{safe}_f{fid}_{kind}.png"
        return (f'<figure class="{cls_}"><img src="{src}" data-cap="{html.escape(cap)}" '
                f'alt="{html.escape(cap)}" loading="lazy" decoding="async">'
                f'<figcaption>{label} (f={fid})</figcaption></figure>')

    rows_html = []
    for group_title, frames in specs:
        rows_html.append(f'<div class="row"><span class="grouptitle">{group_title}</span></div>')
        for label, fid in frames:
            rows_html.append('<div class="row pair">' +
                             cell('src', label, fid) + cell('pred', label, fid) + '</div>')
    if saved.get('ov'):
        fmax = _int(_col(row, 'Fmax'))
        if fmax is not None and fmax in saved['ov']:
            rows_html.append('<div class="row"><span class="predlegend" style="width:100%">'
                             f'Fmax 叠加（绿=分支 / 青=P / 红=D / 黄=S）</span></div>')
            rows_html.append('<div class="row pair">' + cell('ov', 'Fmax', fmax) + '</div>')

    ctfc_txt = ctfc if ctfc not in ('', None) else '—'
    status = ('<span class="pill ok">有效</span>' if valid
              else '<span class="pill bad">无效</span>')
    return f"""
<section class="card" data-id="{rid}" data-case="{html.escape(case_id)}"
         data-seq="{html.escape(seq_id)}" data-branch="{html.escape(branch)}"
         data-f1="{html.escape(f1)}" data-f2="{html.escape(f2)}" data-ctfc="{ctfc_txt}"
         data-valid="{'有效' if valid else '无效'}" data-rank="0"
         data-ranklabel="{CLASS_LABEL[cls]}" data-flags="{','.join(flags)}">
  <div class="card-head">
    <span class="title">{rid} [{branch}]</span>
    {status}
    <span class="cls-pill">{CLASS_LABEL[cls]}</span>
    <span class="kv">F1=<b>{f1 or '—'}</b> · F2=<b>{f2 or '—'}</b> · CTFC=<b>{ctfc_txt}</b>
      · Fmax=<b>{_col(row, 'Fmax') or '—'}</b> · n=<b>{_col(row, '总帧数') or '—'}</b>
      · Fmax像素=<b>{_col(row, '最大非零像素数') or '—'}</b></span>
    <span class="notes">{html.escape(notes)}</span>
    <span class="hint">诊断：{CLASS_HINT[cls]}</span>
  </div>
  <div class="sheet">{''.join(rows_html)}</div>
  <div class="verdict">
    <label><input type="radio" name="v-{rid}" value="掩码可用">掩码可用</label>
    <label><input type="radio" name="v-{rid}" value="掩码不可用">掩码不可用</label>
    <label><input type="radio" name="v-{rid}" value="拿不准">拿不准</label>
    <textarea placeholder="复核理由 / 存疑说明（自动缓存到本地，可导出 CSV）"></textarea>
  </div>
</section>"""


# ------------------------------------------------------------------ 入口

def build_report(results_csv: str, root_dir: str, out_dir: str, per_class: int = 8,
                 segmented_candidates: Optional[List[str]] = None,
                 detailed_dir: Optional[str] = None, mask_suffix: str = '_pred.png',
                 logger: logging.Logger = None) -> Dict:
    logger = logger or logging.getLogger('CTFC.DIAG')
    candidates = segmented_candidates or ['segmented3c2', 'segmented']

    with open(results_csv, 'r', encoding='utf-8-sig', newline='') as f:
        all_rows = list(csv.DictReader(f))
    logger.info(f"读入 {len(all_rows)} 行: {results_csv}")

    buckets: Dict[str, List[Tuple[int, Dict[str, str]]]] = {c[0]: [] for c in CLASSES}
    for i, row in enumerate(all_rows):
        cls = classify(row)
        if cls:
            buckets[cls].append((i, row))
    for flag, label, _ in CLASSES:
        logger.info(f"  类别 {label}: 共 {len(buckets[flag])} 行")
    pool = [(flag, row) for flag, _, _ in CLASSES
            for _, row in sample_spread(buckets[flag], per_class)]

    frame_dir = os.path.join(out_dir, 'frames')
    os.makedirs(frame_dir, exist_ok=True)

    cards, n_img, n_skip, n_missing = [], 0, 0, 0
    for cls, row in pool:
        case_id, seq_id = _col(row, '病例ID'), _col(row, '序列ID')
        branch = _col(row, '分支类型')
        safe = re.sub(r'[^0-9A-Za-z_.-]', '_', f"{case_id}__{seq_id}")
        specs = frame_specs(_int(_col(row, 'F1')), _int(_col(row, 'F2')),
                            _int(_col(row, 'Fmax')), _int(_col(row, '总帧数')))
        wanted = [fid for _, frames in specs for _, fid in frames if fid is not None]

        seg_dir = resolve_segmented_dir(root_dir, case_id, seq_id, branch, candidates)
        saved = {'src': set(), 'pred': set(), 'ov': set()}
        if seg_dir is None:
            logger.warning(f"{case_id}/{seq_id}: 未找到分割目录")
            n_skip += 1
            cards.append(_render_card(case_id, seq_id, row, cls, specs, 'frames',
                                      safe, saved))
            continue

        stems = qr._collect_stems(seg_dir, mask_suffix)
        missing_fids = [f for f in wanted if f not in stems]
        if missing_fids:
            logger.warning(f"{case_id}/{seq_id}: 磁盘上缺帧 {missing_fids}")
            n_missing += len(missing_fids)
        srcs, raws = {}, {}
        for fid in wanted:
            stem = stems.get(fid)
            if stem is None:
                continue
            s = cv2.imread(os.path.join(seg_dir, stem + '_src.png'))
            r = MaskProcessor.load_mask(os.path.join(seg_dir, stem + mask_suffix))
            if s is None or r is None:
                continue
            if r.shape != s.shape[:2]:
                r = cv2.resize(r, (s.shape[1], s.shape[0]), interpolation=cv2.INTER_NEAREST)
            srcs[fid] = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY) if s.ndim == 3 else s
            raws[fid] = r

        if not srcs:
            logger.warning(f"{case_id}/{seq_id}: 帧文件缺失")
            n_skip += 1
            cards.append(_render_card(case_id, seq_id, row, cls, specs, 'frames',
                                      safe, saved))
            continue

        for fid in sorted(srcs):
            gray, raw = srcs[fid], raws[fid]
            n_lad, n_lcx = int((raw == 255).sum()), int((raw == 128).sum())
            n_oth = int(((raw > 0) & (raw != 128) & (raw != 255)).sum())
            cap_s = f"{branch} src f={fid}"
            cap_p = (f"{branch} pred f={fid}  255/LAD={n_lad}px  128/LCX={n_lcx}px"
                     + (f" 其他={n_oth}px" if n_oth else ""))
            for kind, img in (('src', render_src(gray, cap_s)),
                              ('pred', render_pred(raw, cap_p))):
                cv2.imwrite(os.path.join(frame_dir, f"{safe}_f{fid}_{kind}.png"), img,
                            [cv2.IMWRITE_PNG_COMPRESSION, 9])
                saved[kind].add(fid)
                n_img += 1

        fmax = _int(_col(row, 'Fmax'))
        if detailed_dir and fmax is not None and fmax in srcs:
            npz_path = os.path.join(detailed_dir, safe + '.npz')
            if os.path.isfile(npz_path):
                z = np.load(npz_path)
                ov = qr.render_qc_frame(
                    cv2.cvtColor(srcs[fmax], cv2.COLOR_GRAY2BGR),
                    (raws[fmax] == {'LAD': 255, 'LCX': 128}.get(branch, 255)).astype(np.uint8)
                    if branch in ('LAD', 'LCX') else (raws[fmax] > 0).astype(np.uint8),
                    z['seed'], z['P'], z['D'],
                    f"{branch} Fmax={fmax}  S/P/D 叠加")
                cv2.imwrite(os.path.join(frame_dir, f"{safe}_f{fmax}_ov.png"), ov,
                            [cv2.IMWRITE_PNG_COMPRESSION, 9])
                saved['ov'].add(fmax)
                n_img += 1

        cards.append(_render_card(case_id, seq_id, row, cls, specs, 'frames', safe, saved))

    counts = {CLASS_LABEL[c[0]]: sum(1 for f, _ in pool if f == c[0]) for c in CLASSES}
    legend = ('<span class="predlegend">pred 配色：'
              '<i class="sw-lad"></i>255=LAD<i class="sw-lcx"></i>128=LCX</span>'
              '<span class="predlegend">src 原样灰度（未做归一化，便于比较帧间亮度）</span>')
    doc = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>CTFC 诊断：src / pred 并排核对</title>
<style>{qr._css()}{_extra_css()}</style>
</head>
<body>
<header>
  <h1>CTFC 诊断：_src 与 _pred 并排核对</h1>
  <div class="meta" id="counts"></div>
  <div class="meta">共 {len(cards)} 条（{ ' · '.join(f'{k} {v}' for k, v in counts.items()) }）
    · 每帧左 src / 右 pred · 点击放大（原尺寸 ↔ 2×）· j/k 跳转 · Esc 关闭</div>
  <div class="legend">{legend}</div>
  <div class="toolbar">{_select_opts()}</div>
</header>
<main>
{''.join(cards)}
</main>
<div class="modal" id="modal" hidden>
  <div class="modal-bar">
    <span id="modal-cap"></span>
    <span class="spacer"></span>
    <button id="zoom-btn">2×</button>
    <button id="close-btn">关闭 (Esc)</button>
  </div>
  <div class="modal-body"><img id="modal-img" alt=""></div>
</div>
<script>{qr._js('ctfc_diag_' + os.path.basename(os.path.abspath(out_dir)))}</script>
</body>
</html>
"""
    html_path = os.path.join(out_dir, 'qc_diag.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(doc)
    logger.info(f"诊断页面: {html_path}（{len(cards)} 分支，{n_img} 图）")
    if n_skip:
        logger.warning(f"{n_skip} 分支无图（目录/帧缺失）")
    if n_missing:
        logger.warning(f"共 {n_missing} 个取帧号在磁盘上不存在（页面显示为占位）")
    return {'n_branches': len(cards), 'n_images': n_img, 'html': html_path}


def main():
    ap = argparse.ArgumentParser(description='CTFC 诊断：src/pred 并排核对页面')
    ap.add_argument('--results', required=True, help='ctfc_results.csv 路径')
    ap.add_argument('--root-dir', '-r', required=True, help='队列根目录')
    ap.add_argument('--out', '-o', required=True, help='输出目录')
    ap.add_argument('--detailed', default=None, help='detailed/ 目录（生成 S/P/D 叠加行）')
    ap.add_argument('--per-class', type=int, default=8, help='每类抽样条数 (默认 8)')
    ap.add_argument('--segmented-dirname', default='segmented3c2,segmented',
                    help='分割子目录候选，逗号分隔')
    ap.add_argument('--mask-suffix', default='_pred.png', help='掩码文件后缀')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s', stream=sys.stdout)
    logger = logging.getLogger('CTFC.DIAG')
    os.makedirs(args.out, exist_ok=True)
    cands = [x for x in re.split(r'[,\s]+', args.segmented_dirname) if x] or ['segmented']
    stats = build_report(args.results, args.root_dir, args.out,
                         per_class=args.per_class, segmented_candidates=cands,
                         detailed_dir=args.detailed, mask_suffix=args.mask_suffix,
                         logger=logger)
    print(f"\n诊断页面: {stats['html']}  ({stats['n_branches']} 分支 / {stats['n_images']} 图)")


if __name__ == '__main__':
    main()
