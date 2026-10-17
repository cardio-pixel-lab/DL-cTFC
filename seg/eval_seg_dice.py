#!/usr/bin/env python3
"""用 cag_seg_v6 的真值掩码评测队列里 `_pred.png` 的分割质量（Dice/IoU/P/R）。

只读数据；所有结果写到本工具工作目录（默认 `tfc/results_segeval`）。

配对为什么不能靠文件名
----------------------
GT 的命名是 `[r]cag{批次}-c{case}-[s{seq}-]f{frame}`，队列 `_pred.png` 也是同一套字面量，
但 **case 号在两棵树里并不一致**（实测 GT `cag2-c56-s269-f18` 与队列同名字面量其实
不是同一例；真身是别人）。因此不能按文件名对上，必须靠 **图像内容**：

1. 序列号 `seq` + 帧号 `frame` 在整棵队列里基本唯一，先用它把候选缩到个位数；
2. 逐个候选读同目录的 `<stem>_src.png`，与 GT `images/<split>/<stem>.png` 求相关系数；
3. 取相关性最高的候选，并要求它 ≥ `--cc-threshold`（GT 是重编码过的，不是逐字节相同，
   实测真配对 cc 0.85–0.98，误配 cc 0.06–0.5，间隔很清楚）。

还有一条必须注意的 **帧号偏移**：GT 是 1-based、队列是 0-based，`cag_seg_v6` 里没有 f0，
所以 GT 第 f 帧对应队列的 f-1 帧（`--frame-offset -1`，可用参数改）。不做这个偏移，
Dice 会被压到 0.47 左右（相邻 DSA 帧高度相似，相关系数分不出来）；做了偏移后
600/600 抽样的最优偏移都落在 -1，Dice 均值 0.915。脚本会记录实际用到的队列帧号
（`cohort_frame` 列）以便复核。

`rcag*`（**旋转造影**批次，不是右冠）在队列里没有任何 `_pred.png`，无法评测，会单独计数。

用法:
  python eval_seg_dice.py                 # 建索引 + 全量评测（含抽查页）
  python eval_seg_dice.py --rebuild-index # 强制重建候选索引
"""

import argparse
import base64
import csv
import json
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_GT_ROOT = '/home/share/clr/share/data/cag_seg_v6'
DEFAULT_ROOTS = [
    '/home/share/clr/share/data/AI_TMPFC/CAD-ID1-ID64',
    '/home/share/clr/share/data/AI_TMPFC/CMVD',
    '/home/share/clr/share/data/AI_TMPFC/CMVD-ID1-ID50',
    '/home/share/clr/share/data/AI_TMPFC/Others-ID1-ID27',
    '/home/share/clr/share/data/AI_TMPFC/cag_cls_src',
]
SPLITS = ['training', 'validation', 'test']
PRED_SUFFIX = '_pred.png'
SRC_SUFFIX = '_src.png'

# LCA GT：cag2-c56-s269-f18 ；旋转造影 GT：rcag1-c9-f66（无序列号段）
RE_LCA = re.compile(r'^(r?)cag\d+-c(\d+)-s(\d+)-f(\d+)$')
RE_RCAG = re.compile(r'^(r?)cag\d+-c(\d+)-f(\d+)$')

# 候选文件名里带 segmented3c2 的优先级更高（CTFC 实际消费的就是它）
def _rank(path: str) -> Tuple[int, int]:
    in3c = 0 if f'{os.sep}segmented3c2{os.sep}' in path else 1
    return (in3c, path.count(os.sep))


# ------------------------------------------------------------------ 索引

def build_index(roots: List[str]) -> Dict:
    """返回 {'by_sf': {(seq,frame): [pred...]}, 'by_cf': {(tag,case,frame): [...]},
             'exact': {stem: [pred...]}}（同一文件只登记一次）。"""
    by_sf: Dict[Tuple[int, int], List[str]] = {}
    by_cf: Dict[Tuple[str, int, int], List[str]] = {}
    exact: Dict[str, List[str]] = {}
    seen = set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith(PRED_SUFFIX):
                    continue
                path = os.path.join(dirpath, fn)
                if path in seen:
                    continue
                seen.add(path)
                stem = fn[:len(fn) - len(PRED_SUFFIX)]
                exact.setdefault(stem, []).append(path)
                m = RE_LCA.match(stem)
                if m:
                    by_sf.setdefault((int(m.group(3)), int(m.group(4))), []).append(path)
                    continue
                m = RE_RCAG.match(stem)
                if m:
                    by_cf.setdefault((m.group(1), int(m.group(2)), int(m.group(3))), []).append(path)
    return {'by_sf': by_sf, 'by_cf': by_cf, 'exact': exact}


def _save_index(idx: Dict, path: str) -> None:
    ser = {k: v for k, v in idx.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(ser, f)


def _load_index(path: str) -> Dict:
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    return {
        'by_sf': {tuple(int(x) for x in k.split(',')): v for k, v in raw['by_sf'].items()},
        'by_cf': {(k.split('|')[0], int(k.split('|')[1]), int(k.split('|')[2])): v
                  for k, v in raw['by_cf'].items()},
        'exact': raw['exact'],
    }


# ------------------------------------------------------------------ 指标

def dice_iou(gt_bin: np.ndarray, pr_bin: np.ndarray) -> Tuple[float, float, float, float]:
    """返回 (dice, iou, precision, recall)。GT 空且预测空视为 1.0。"""
    g = gt_bin.astype(bool)
    p = pr_bin.astype(bool)
    inter = int(np.logical_and(g, p).sum())
    gs, ps = int(g.sum()), int(p.sum())
    union = gs + ps - inter
    dice = 1.0 if (gs + ps) == 0 else 2.0 * inter / (gs + ps)
    iou = 1.0 if union == 0 else inter / union
    prec = 1.0 if ps == 0 else inter / ps
    rec = 1.0 if gs == 0 else inter / gs
    return dice, iou, prec, rec


def n_components(bin_img: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(bin_img.astype(np.uint8), connectivity=8)
    return int(n) - 1


def corr(gray_a: np.ndarray, gray_b: np.ndarray) -> Optional[float]:
    """两个灰度图的相关性（尺寸不同则把 b 缩到 a）。任一恒定则返回 0。"""
    if gray_a is None or gray_b is None:
        return None
    if gray_b.shape != gray_a.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


# ------------------------------------------------------------------ 配对

def _src_of(pred_path: str) -> str:
    return pred_path[:-len(PRED_SUFFIX)] + SRC_SUFFIX


def candidates_for(stem: str, idx: Dict, frame_offset: int) -> Tuple[List[str], Optional[int]]:
    """返回 (候选 pred 路径, 实际使用的队列帧号)。

    GT 帧号是 1-based、队列是 0-based，故默认要用 GT 帧号 + frame_offset（-1）。
    GT 帧 f 对应队列帧不存在时（如 f+offset < 0）退回 f 本身。
    """
    m = RE_LCA.match(stem)
    if m:
        seq, fr = int(m.group(3)), int(m.group(4))
        for cand_fr in ([fr + frame_offset, fr] if frame_offset else [fr]):
            if cand_fr < 0:
                continue
            got = idx['by_sf'].get((seq, cand_fr))
            if got:
                return list(got), cand_fr
        return [], None
    m = RE_RCAG.match(stem)
    if m:
        key = (m.group(1), int(m.group(2)), int(m.group(3)))
        return list(idx['by_cf'].get(key, [])), int(m.group(3))
    return [], None


def match_one(stem: str, gt_img: np.ndarray, idx: Dict, frame_offset: int) -> Dict:
    """按内容选候选。返回 {status, pred_path, cc, cc2, n_cand, n_src, cohort_frame}。"""
    cands, cohort_frame = candidates_for(stem, idx, frame_offset)
    if not cands:
        return {'status': 'no_candidate', 'pred_path': None, 'cc': None, 'cc2': None,
                'n_cand': 0, 'n_src': 0, 'cohort_frame': cohort_frame}

    # 同一 _src 只算一次（segmented / segmented3c2 常并存，src 相同）
    by_src: Dict[str, str] = {}
    for p in cands:
        s = _src_of(p)
        if s not in by_src or _rank(p) < _rank(by_src[s]):
            by_src[s] = p

    scored: List[Tuple[float, str]] = []
    n_src = 0
    for s, p in by_src.items():
        if not os.path.exists(s):
            continue
        n_src += 1
        c = corr(gt_img, cv2.imread(s, cv2.IMREAD_GRAYSCALE))
        if c is not None:
            scored.append((c, p))
    if not scored:
        return {'status': 'no_src', 'pred_path': None, 'cc': None, 'cc2': None,
                'n_cand': len(cands), 'n_src': 0, 'cohort_frame': cohort_frame}
    scored.sort(key=lambda t: -t[0])
    return {'status': 'scored', 'pred_path': scored[0][1], 'cc': round(scored[0][0], 4),
            'cc2': round(scored[1][0], 4) if len(scored) > 1 else None,
            'n_cand': len(cands), 'n_src': n_src, 'cohort_frame': cohort_frame}


# ------------------------------------------------------------------ 抽查页

def _b64_png(img: np.ndarray, max_side: int = 256, color: bool = False) -> str:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf.tobytes()).decode('ascii') if ok else ''


def _overlay(gt_bin: np.ndarray, pr_bin: np.ndarray) -> np.ndarray:
    """GT 绿、预测红、重叠黄（三通道 BGR）。"""
    out = np.zeros((*gt_bin.shape, 3), np.uint8)
    out[gt_bin & ~pr_bin] = (0, 200, 0)
    out[pr_bin & ~gt_bin] = (0, 0, 255)
    out[gt_bin & pr_bin] = (0, 220, 220)
    return out


def write_review_sheet(rows: List[Dict], gt_root: str, out_path: str, top: int = 40) -> str:
    """挑 cc 最低的若干「已接受」配对，铺 GT/src/pred/叠加 四联图，供人工抽查。"""
    picked = [r for r in rows if r.get('status') == 'ok']
    picked.sort(key=lambda r: r['cc'])
    picked = picked[:top]
    parts = ["""<!doctype html><meta charset="utf-8"><title>Dice 配对抽查</title>
<style>body{background:#111;color:#eee;font:13px/1.5 system-ui,sans-serif;margin:16px}
.row{display:flex;gap:6px;align-items:center;margin:10px 0;padding:8px;background:#1c1c1c;border-radius:6px}
.cell{text-align:center}.cap{font-size:11px;color:#aaa}
img{display:block;image-rendering:pixelated;border:1px solid #333}
.meta{min-width:260px;font-size:12px}
.bad{color:#ff6b6b}</style>"""]
    parts.append(f'<h2>配对抽查：cc 最低的 {len(picked)} 条（已接受配对）</h2>')
    for r in picked:
        gt_lab = cv2.imread(os.path.join(gt_root, 'labels', r['split'], r['stem'] + '.png'),
                            cv2.IMREAD_GRAYSCALE)
        src = cv2.imread(_src_of(r['pred_path']), cv2.IMREAD_GRAYSCALE)
        pr = cv2.imread(r['pred_path'], cv2.IMREAD_GRAYSCALE)
        if gt_lab is None or pr is None:
            continue
        if pr.shape != gt_lab.shape:
            pr = cv2.resize(pr, (gt_lab.shape[1], gt_lab.shape[0]), interpolation=cv2.INTER_NEAREST)
        gb, pb = gt_lab > 0, pr > 0
        ov = _overlay(gb, pb)
        parts.append('<div class="row">')
        parts.append('<div class="meta">'
                     f"<b>{r['stem']}</b><br>split={r['split']}<br>"
                     f"cc=<b>{r['cc']}</b>" +
                     (f" (次高 {r['cc2']})" if r.get('cc2') is not None else '') +
                     f"<br>dice=<b>{r['dice']}</b> iou={r['iou']}<br>"
                     f"gt_px={r['gt_px']} pred_px={r['pred_px']}<br>"
                     f"<span class='cap'>{r['pred_path'].replace(os.sep, '/').split('/AI_TMPFC/')[-1]}</span>"
                     '</div>')
        for cap, im, col in [('src 队列原图', src, False), ('GT 标签', gt_lab, False),
                             ('pred 预测', pr, False), ('GT绿/预测红/重叠黄', ov, True)]:
            if im is None:
                continue
            parts.append(f'<div class="cell"><div class="cap">{cap}</div>'
                         f'<img src="data:image/png;base64,{_b64_png(im, color=col)}"></div>')
        parts.append('</div>')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    return out_path


# ------------------------------------------------------------------ 主流程

def run(gt_root: str, roots: List[str], out_dir: str, splits: List[str],
        cc_threshold: float, frame_offset: int, rebuild: bool, review_top: int,
        logger=print) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    idx_path = os.path.join(out_dir, 'pred_index.json')
    if rebuild or not os.path.exists(idx_path):
        logger(f'扫描 {len(roots)} 个根目录建立候选索引 ...')
        idx = build_index(roots)
        with open(idx_path, 'w', encoding='utf-8') as f:
            json.dump({'by_sf': {f'{k[0]},{k[1]}': v for k, v in idx['by_sf'].items()},
                       'by_cf': {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in idx['by_cf'].items()},
                       'exact': idx['exact']}, f)
        logger(f'  索引: seq+frame 键 {len(idx["by_sf"])} 个、rcag 键 {len(idx["by_cf"])} 个、'
               f'同名主干 {len(idx["exact"])} 个 -> {idx_path}')
    else:
        idx = _load_index(idx_path)
        logger(f'复用索引 {idx_path}（seq+frame 键 {len(idx["by_sf"])}）')

    rows: List[Dict] = []
    for split in splits:
        lab_dir = os.path.join(gt_root, 'labels', split)
        img_dir = os.path.join(gt_root, 'images', split)
        if not os.path.isdir(lab_dir):
            logger(f'跳过 {split}: 无 {lab_dir}')
            continue
        stems = sorted(fn[:-4] for fn in os.listdir(lab_dir) if fn.endswith('.png'))
        logger(f'[{split}] {len(stems)} 帧真值')
        n_acc = 0
        for stem in stems:
            # 原图（images）只用于内容配对；掩码（labels）才用于 Dice
            gt = cv2.imread(os.path.join(img_dir, stem + '.png'), cv2.IMREAD_GRAYSCALE)
            gt_lab = cv2.imread(os.path.join(lab_dir, stem + '.png'), cv2.IMREAD_GRAYSCALE)
            if gt is None or gt_lab is None:
                rows.append({'split': split, 'stem': stem, 'status': 'gt_read_fail'})
                continue
            mg = match_one(stem, gt, idx, frame_offset)
            rec = {'split': split, 'stem': stem, **mg}
            if mg['status'] != 'scored' or mg['cc'] < cc_threshold:
                rec['status'] = ('rcag_unsupported' if mg['status'] == 'no_candidate' and RE_RCAG.match(stem)
                                 else mg['status'] if mg['status'] != 'scored' else 'low_cc')
                rows.append(rec)
                continue
            pr = cv2.imread(mg['pred_path'], cv2.IMREAD_GRAYSCALE)
            if pr is None:
                rec['status'] = 'pred_read_fail'
                rows.append(rec)
                continue
            gt_bin = gt_lab > 0
            if pr.shape != gt_lab.shape:
                pr = cv2.resize(pr, (gt_lab.shape[1], gt_lab.shape[0]), interpolation=cv2.INTER_NEAREST)
            pr_bin = pr > 0
            d, iou, prec, rec_ = dice_iou(gt_bin, pr_bin)
            rec.update({
                'status': 'ok',
                'gt_px': int(gt_bin.sum()),
                'pred_px': int(pr_bin.sum()),
                'n_gt_cc': n_components(gt_bin),
                'n_pred_cc': n_components(pr_bin),
                'pred_lad_px': int((pr == 255).sum()),
                'pred_lcx_px': int((pr == 128).sum()),
                'dice': round(d, 4), 'iou': round(iou, 4),
                'precision': round(prec, 4), 'recall': round(rec_, 4),
            })
            rows.append(rec)
            n_acc += 1
        logger(f'  接受 {n_acc} 帧')

    ok = [r for r in rows if r.get('status') == 'ok']
    status_ct: Dict[str, int] = {}
    for r in rows:
        status_ct[r['status']] = status_ct.get(r['status'], 0) + 1

    summary = {
        'gt_root': gt_root,
        'search_roots': roots,
        'cc_threshold': cc_threshold,
        'frame_offset': frame_offset,
        'n_gt_total': len(rows),
        'n_accepted': len(ok),
        'status_counts': dict(sorted(status_ct.items(), key=lambda kv: -kv[1])),
        'cc_hist': {},
        'by_split': {},
        'by_split_all': {},
        'worst': [],
    }
    ccs = [r['cc'] for r in ok]
    if ccs:
        edges = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 1.01]
        hist = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            hist[f'{lo:.2f}-{hi:.2f}'] = sum(1 for c in ccs if lo <= c < hi)
        summary['cc_hist'] = hist
        for sp in splits:
            sub = [r for r in ok if r['split'] == sp]
            allsub = [r for r in rows if r['split'] == sp]
            summary['by_split_all'][sp] = {
                'n_gt': len(allsub),
                'n_accepted': len(sub),
                'accept_rate': round(len(sub) / len(allsub), 4) if allsub else None,
            }
            if not sub:
                continue
            ds = np.array([r['dice'] for r in sub])
            summary['by_split'][sp] = {
                'n': len(sub),
                'cc_mean': round(float(np.mean([r['cc'] for r in sub])), 4),
                'dice_mean': round(float(ds.mean()), 4),
                'dice_median': round(float(np.median(ds)), 4),
                'dice_p10': round(float(np.percentile(ds, 10)), 4),
                'dice_p25': round(float(np.percentile(ds, 25)), 4),
                'dice_min': round(float(ds.min()), 4),
                'dice_lt_0.5': int((ds < 0.5).sum()),
                'iou_mean': round(float(np.mean([r['iou'] for r in sub])), 4),
                'precision_mean': round(float(np.mean([r['precision'] for r in sub])), 4),
                'recall_mean': round(float(np.mean([r['recall'] for r in sub])), 4),
            }
        summary['worst'] = sorted(ok, key=lambda r: r['dice'])[:15]

    fields = ['split', 'stem', 'status', 'cohort_frame', 'cc', 'cc2', 'n_cand', 'n_src',
              'gt_px', 'pred_px', 'n_gt_cc', 'n_pred_cc', 'pred_lad_px', 'pred_lcx_px',
              'dice', 'iou', 'precision', 'recall', 'pred_path']
    csv_path = os.path.join(out_dir, 'per_frame_dice.csv')
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    pairs_path = os.path.join(out_dir, 'pairs.csv')
    with open(pairs_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'stem', 'cohort_frame', 'cc', 'cc2', 'n_src',
                                          'status', 'pred_path'], extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    sheet = write_review_sheet(rows, gt_root, os.path.join(out_dir, 'review_sheet.html'), top=review_top)

    logger(f'\nCSV : {csv_path}\n配对: {pairs_path}\n抽查: {sheet}')
    logger(json.dumps({k: v for k, v in summary.items() if k != 'worst'},
                      ensure_ascii=False, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description='用 cag_seg_v6 真值评测队列 _pred 分割质量（内容配对）')
    ap.add_argument('--gt-root', default=DEFAULT_GT_ROOT)
    ap.add_argument('--search-roots', nargs='*', default=DEFAULT_ROOTS)
    ap.add_argument('--out', '-o',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'results_segeval'))
    ap.add_argument('--splits', default=','.join(SPLITS))
    ap.add_argument('--frame-offset', type=int, default=-1,
                    help='队列帧号 = GT 帧号 + 本值。GT 是 1-based、队列 0-based，故默认 -1'
                         '（600/600 抽样在 -1 取到 argmax，Dice 0.47→0.92）')
    ap.add_argument('--cc-threshold', type=float, default=0.80,
                    help='接受配对所需的最低相关系数（GT 为重编码图，真配对实测 0.85+）')
    ap.add_argument('--review-top', type=int, default=40, help='抽查页铺多少条 cc 最低的已接受配对')
    ap.add_argument('--rebuild-index', action='store_true')
    args = ap.parse_args()
    splits = [s for s in args.splits.split(',') if s]
    run(args.gt_root, args.search_roots, args.out, splits, args.cc_threshold,
        args.frame_offset, args.rebuild_index, args.review_top,
        logger=lambda m: print(m, flush=True))


if __name__ == '__main__':
    main()
