#!/usr/bin/env python3
"""CTFC QC 连拍图 + 人工逐序列核对页面。

每个分支一张「7 帧连拍」缩略索引：
    行1 = [F1-1, F1, F1+1]   行2 = [F2-1, F2, F2+1]   行3 = [Fmax]
每帧叠加：分支掩码(绿) / 近端区 P(青) / 远端区 D(红) / 种子区 S(黄轮廓)，
并烧录帧号与 fill_p/fill_d 数值。汇总为可翻页 HTML：点击帧放大至原尺寸/2×，
每分支可勾选「通过/拒绝/存疑」+ 复核理由，导出 qc_results.csv。
"""

import csv
import html
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ctfc_analyzer import MaskProcessor

# 叠加色（BGR）
COL_BRANCH = (0, 200, 0)      # 绿
COL_P = (255, 255, 0)         # 青
COL_D = (0, 0, 255)           # 红
COL_S = (0, 255, 255)         # 黄（种子区轮廓）

CAPTION_H = 30


# ---------------------------------------------------------------------- 图像处理

def _fill_outline(img: np.ndarray, region: Optional[np.ndarray], color,
                  alpha: float = 0.28) -> np.ndarray:
    """半透明填充 region 非零处 + 1px 轮廓。region 为空则原样返回。"""
    if region is None or int(region.sum()) == 0:
        return img
    m = (region > 0).astype(np.uint8)
    out = img.copy()
    blended = alpha * img.astype(np.float32) + (1 - alpha) * np.array(color, dtype=np.float32)
    out[m > 0] = blended[m > 0].astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 1)
    return out


def _outline_only(img: np.ndarray, region: Optional[np.ndarray], color,
                  thickness: int = 2) -> np.ndarray:
    """仅画轮廓（不填充），用于在不遮挡填充区的前提下标出种子区。"""
    if region is None or int(region.sum()) == 0:
        return img
    m = (region > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)
    return img


def _put_text(img: np.ndarray, text: str, org, scale: float) -> None:
    for thick, color in ((4, (0, 0, 0)), (2, (255, 255, 255))):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render_qc_frame(src: np.ndarray, branch: np.ndarray,
                    S: Optional[np.ndarray], P: Optional[np.ndarray], D: Optional[np.ndarray],
                    caption: str, legend: str = '') -> np.ndarray:
    """渲染单帧：branch(绿填充) → P(青填充) → D(红填充) → S(黄轮廓) + 顶部标题条。"""
    img = src.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = _fill_outline(img, branch, COL_BRANCH, 0.22)
    img = _fill_outline(img, P, COL_P, 0.28)
    img = _fill_outline(img, D, COL_D, 0.28)
    img = _outline_only(img, S, COL_S, 2)

    h, w = img.shape[:2]
    bar = img[0:CAPTION_H].copy()
    img[0:CAPTION_H] = (0.25 * bar).astype(np.uint8)
    _put_text(img, caption, (8, 21), 0.58)
    if legend:
        _put_text(img, legend, (8, h - 8), 0.5)
    return img


def _extract_branch_mask(raw: np.ndarray, branch_type: str, params: Dict,
                         src_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """从 pred 掩码中取出该分支的清理后二值掩码（与算法口径一致）。"""
    if raw.shape != src_shape:
        raw = cv2.resize(raw, (src_shape[1], src_shape[0]), interpolation=cv2.INTER_NEAREST)
    pix_val = {'LAD': 255, 'LCX': 128}.get(branch_type)   # RCA 为 None（全非零）
    branch = (raw == pix_val) if pix_val is not None else (raw > 0)
    cleaned = MaskProcessor.clean_mask(branch.astype(np.uint8),
                                       params['border_width'],
                                       params['min_component_area'])
    return (cleaned > 0).astype(np.uint8)


def _collect_stems(segmented_dir: str, suffix: str) -> Dict[int, str]:
    """帧号 -> 文件主干（..._pred/_src 之前的部分）。"""
    stems: Dict[int, str] = {}
    try:
        names = os.listdir(segmented_dir)
    except OSError:
        return stems
    for fn in names:
        if not fn.endswith(suffix):
            continue
        head = fn[:len(fn) - len(suffix)]
        m = re.search(r'f(\d+)$', head)
        if m:
            stems[int(m.group(1))] = head
    return stems


# ---------------------------------------------------------------------- 帧选取

def qc_frame_rows(f1: Optional[int], f2: Optional[int], fmax: int,
                  n_frames: int) -> List[List[Tuple[str, Optional[int]]]]:
    """构造 3 行连拍布局：[F1±1] / [F2±1] / [Fmax]，帧号越界置 None。"""
    def clamp(v):
        if v is None or v < 0 or v >= n_frames:
            return None
        return int(v)

    def triple(tag, fid):
        if fid is None:
            return [(f"{tag}-1", None), (tag, None), (f"{tag}+1", None)]
        return [(f"{tag}-1", clamp(fid - 1)), (tag, clamp(fid)), (f"{tag}+1", clamp(fid + 1))]

    return [triple('F1', f1), triple('F2', f2), [('Fmax', clamp(fmax))]]


# ---------------------------------------------------------------------- 优先级排序

def qc_priority(a) -> int:
    """待审优先级：1 无效 > 2 退化单相 > 3 F2==F1 > 4 干净有效（阴性对照）。"""
    if not a.is_valid:
        return 1
    if any('退化' in n for n in a.notes):
        return 2
    if a.f1 is not None and a.f2 is not None and a.f1 == a.f2:
        return 3
    return 4


PRIORITY_LABEL = {1: '无效', 2: '退化单相', 3: 'F2==F1', 4: '干净有效'}


def _select_rows(rows: List[Tuple[str, str, object]], sample_frac: float
                 ) -> List[Tuple[str, str, object]]:
    """无效/退化/F2==F1 全量；干净有效按 sample_frac 确定性抽样。"""
    rows = sorted(rows, key=lambda r: (qc_priority(r[2]), r[0], r[1]))
    if sample_frac >= 1.0:
        return rows
    out, clean_idx = [], 0
    step = max(int(round(1.0 / max(sample_frac, 1e-6))), 1)
    for r in rows:
        if qc_priority(r[2]) == 4:
            clean_idx += 1
            if (clean_idx - 1) % step != 0:
                continue
        out.append(r)
    return out


# ---------------------------------------------------------------------- HTML

def _css() -> str:
    return """
  * { box-sizing: border-box; }
  body { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0; color: #e8e8e8;
         background: #16181d; }
  header { position: sticky; top: 0; z-index: 20; background: #1e2128; padding: 12px 20px;
           border-bottom: 1px solid #333; }
  header h1 { font-size: 18px; margin: 0 0 6px; }
  .meta { color: #9aa0aa; font-size: 13px; margin-bottom: 8px; }
  .toolbar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .toolbar button { background: #2b2f3a; color: #ddd; border: 1px solid #444; border-radius: 6px;
                    padding: 5px 12px; cursor: pointer; font-size: 13px; }
  .toolbar button.active { background: #2f6fd0; border-color: #2f6fd0; color: #fff; }
  .toolbar button:hover { border-color: #666; }
  .toolbar input { background: #12141a; color: #ddd; border: 1px solid #444; border-radius: 6px;
                   padding: 5px 10px; font-size: 13px; width: 220px; }
  .toolbar .spacer { flex: 1; }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px; color: #b9c0cb;
            margin: 6px 0 8px; align-items: center; }
  .legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px;
              margin-right: 5px; vertical-align: -2px; }
  .legend .sw-branch { background: #00c800; }
  .legend .sw-p { background: #00ffff; }
  .legend .sw-d { background: #ff0000; }
  .legend .sw-s { background: #ffff00; }
  .legend .bd-f1 { border: 2px solid #2ecc71; padding: 0 6px; border-radius: 4px; }
  .legend .bd-f2 { border: 2px solid #e74c3c; padding: 0 6px; border-radius: 4px; }
  .legend .bd-fmax { border: 2px solid #8a8f99; padding: 0 6px; border-radius: 4px; }
  main { padding: 16px 20px 80px; }
  .card { background: #1c1f26; border: 1px solid #2e323c; border-radius: 10px;
          padding: 12px 14px; margin-bottom: 18px; }
  .card.reviewed { border-color: #3a5f3a; }
  .card.v-拒绝 { border-color: #7a3434; }
  .card.v-存疑 { border-color: #7a6a2a; }
  .card-head { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
  .title { font-weight: 600; font-size: 15px; }
  .pill { font-size: 12px; padding: 2px 9px; border-radius: 10px; font-weight: 600; }
  .pill.ok { background: #1d3d24; color: #6ede8a; }
  .pill.bad { background: #4a1f1f; color: #ff8b8b; }
  .pill.rank1 { background: #4a1f1f; color: #ff8b8b; }
  .pill.rank2 { background: #3d3520; color: #e8ce6a; }
  .pill.rank3 { background: #2a3550; color: #8fb4f0; }
  .pill.rank4 { background: #24302a; color: #9fd7b5; }
  .kv { font-size: 13px; color: #b9c0cb; }
  .kv b { color: #fff; font-weight: 600; }
  .notes { font-size: 12px; color: #8d94a0; flex-basis: 100%; }
  .sheet { display: flex; flex-direction: column; gap: 8px; }
  .sheet .row { display: flex; gap: 8px; }
  .cell { width: 226px; }
  .cell img { width: 226px; height: 226px; object-fit: contain; background: #0c0d10;
              border: 2px solid #3a3f4a; border-radius: 6px; cursor: zoom-in; display: block; }
  .cell.label-F1 img { border-color: #2ecc71; }
  .cell.label-F2 img { border-color: #e74c3c; }
  .cell.label-Fmax img { border-color: #8a8f99; }
  .cell figcaption { font-size: 12px; color: #9aa0aa; text-align: center; padding-top: 3px; }
  .cell.missing div.ph { width: 226px; height: 226px; border: 2px dashed #3a3f4a; border-radius: 6px;
                         display: flex; align-items: center; justify-content: center;
                         color: #666; font-size: 12px; background: #14161a; }
  .verdict { display: flex; gap: 14px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
  .verdict label { font-size: 13px; color: #cfd5de; cursor: pointer; }
  .verdict input[type=radio] { vertical-align: middle; margin-right: 4px; }
  .verdict textarea { flex: 1; min-width: 260px; background: #12141a; color: #ddd;
                      border: 1px solid #444; border-radius: 6px; padding: 6px 8px;
                      font-size: 13px; font-family: inherit; resize: vertical; min-height: 34px; }
  .navbtns { display: flex; gap: 8px; margin-top: 10px; }
  .navbtns button { background: #2b2f3a; color: #ddd; border: 1px solid #444; border-radius: 6px;
                    padding: 4px 12px; cursor: pointer; font-size: 12px; }
  .modal { position: fixed; inset: 0; background: rgba(0,0,0,.88); z-index: 50;
           display: flex; flex-direction: column; }
  .modal[hidden] { display: none; }
  .modal-bar { display: flex; gap: 12px; align-items: center; padding: 10px 16px;
               background: #1e2128; border-bottom: 1px solid #333; font-size: 13px; }
  .modal-bar .spacer { flex: 1; }
  .modal-bar button { background: #2b2f3a; color: #ddd; border: 1px solid #444;
                      border-radius: 6px; padding: 5px 12px; cursor: pointer; font-size: 13px; }
  .modal-body { flex: 1; overflow: auto; display: flex; justify-content: center;
                align-items: flex-start; padding: 14px; }
  #modal-img { width: 512px; height: auto; image-rendering: auto; }
  #modal-img.zoom { width: 1024px; image-rendering: pixelated; }
  .empty { color: #888; padding: 40px; text-align: center; }
"""


def _js(store_key: str) -> str:
    return """
const STORE_KEY = %r;
let store = {};
try { store = JSON.parse(localStorage.getItem(STORE_KEY) || '{}') || {}; } catch (e) { store = {}; }

const cards = Array.from(document.querySelectorAll('.card'));
const modal = document.getElementById('modal');
const modalImg = document.getElementById('modal-img');
const modalCap = document.getElementById('modal-cap');
const zoomBtn = document.getElementById('zoom-btn');

function saveStore() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(store)); } catch (e) {}
}

function applyStore() {
  cards.forEach(card => {
    const id = card.dataset.id;
    const rec = store[id] || {};
    card.querySelectorAll('input[type=radio]').forEach(r => { r.checked = (r.value === rec.verdict); });
    const ta = card.querySelector('textarea');
    if (ta) ta.value = rec.note || '';
    card.classList.toggle('reviewed', !!rec.verdict);
    card.classList.remove('v-拒绝', 'v-存疑', 'v-通过');
    if (rec.verdict) card.classList.add('v-' + rec.verdict);
    card.dataset.verdict = rec.verdict || '';
  });
}

cards.forEach(card => {
  const id = card.dataset.id;
  card.querySelectorAll('input[type=radio]').forEach(r => {
    r.addEventListener('change', () => {
      store[id] = store[id] || {};
      store[id].verdict = r.value;
      saveStore(); applyStore(); updateCounts();
    });
  });
  const ta = card.querySelector('textarea');
  if (ta) {
    let timer = null;
    ta.addEventListener('input', () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        store[id] = store[id] || {};
        store[id].note = ta.value;
        saveStore();
      }, 300);
    });
  }
});

// 点击缩略图放大
document.querySelectorAll('.cell img').forEach(img => {
  img.addEventListener('click', () => {
    modalCap.textContent = img.dataset.cap || '';
    modalImg.src = img.src;
    modalImg.classList.remove('zoom');
    zoomBtn.textContent = '2×';
    modal.hidden = false;
  });
});
zoomBtn.addEventListener('click', () => {
  modalImg.classList.toggle('zoom');
  zoomBtn.textContent = modalImg.classList.contains('zoom') ? '1×' : '2×';
});
function closeModal() { modal.hidden = true; modalImg.src = ''; }
document.getElementById('close-btn').addEventListener('click', closeModal);
modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
  if (modal.hidden && (e.key === 'j' || e.key === 'k')) {
    const vis = visibleCards();
    const idx = vis.findIndex(c => c.getBoundingClientRect().top > 80);
    const target = e.key === 'j' ? (idx < 0 ? vis.length - 1 : idx)
                                 : (idx <= 0 ? 0 : idx - 2);
    if (vis[target]) vis[target].scrollIntoView({block: 'start', behavior: 'smooth'});
  }
});

// 筛选
let filter = 'all';
function visibleCards() { return cards.filter(c => !c.hidden); }
function applyFilter() {
  cards.forEach(c => {
    let show = true;
    if (filter === 'all') show = true;
    else if (filter === 'pending') show = !c.dataset.verdict;
    else show = (c.dataset.flags || '').split(',').includes(filter);
    c.hidden = !show;
  });
  updateCounts();
}
document.querySelectorAll('.toolbar button[data-filter]').forEach(btn => {
  btn.addEventListener('click', () => {
    filter = btn.dataset.filter;
    document.querySelectorAll('.toolbar button[data-filter]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyFilter();
  });
});
document.getElementById('q').addEventListener('input', e => {
  const q = e.target.value.trim().toLowerCase();
  cards.forEach(c => {
    const hit = !q || c.dataset.id.toLowerCase().includes(q);
    if (!hit) c.hidden = true;
  });
  if (q) updateCounts();
  else applyFilter();
});

function updateCounts() {
  const total = cards.length;
  const reviewed = cards.filter(c => c.dataset.verdict).length;
  document.getElementById('counts').textContent =
    `共 ${total} 条 · 已审核 ${reviewed} · 待审核 ${total - reviewed}`;
}

// 导出审核结果
document.getElementById('export').addEventListener('click', () => {
  const cols = ['病例ID','序列ID','分支类型','F1','F2','CTFC','有效性','优先级','审核结论','复核理由'];
  const lines = [cols.join(',')];
  cards.forEach(c => {
    const rec = store[c.dataset.id] || {};
    // 直接读 DOM 现值：备注写入有 300ms 防抖，若输入后立刻导出会丢失
    const checked = c.querySelector('input[type=radio]:checked');
    const verdict = checked ? checked.value : (rec.verdict || '');
    const ta = c.querySelector('textarea');
    const note = (ta ? ta.value : (rec.note || '')).replace(/[\\r\\n,]+/g, ' ');
    const row = [
      c.dataset.case, c.dataset.seq, c.dataset.branch, c.dataset.f1, c.dataset.f2,
      c.dataset.ctfc, c.dataset.valid, c.dataset.ranklabel,
      verdict, note
    ].map(v => {
      const s = String(v == null ? '' : v);
      return /[",\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    });
    lines.push(row.join(','));
  });
  const blob = new Blob(['\\ufeff' + lines.join('\\n')], {type: 'text/csv;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'qc_results.csv';
  a.click();
  URL.revokeObjectURL(a.href);
});

applyStore();
applyFilter();
""" % (store_key,)


def _select_opts() -> str:
    return """
    <button data-filter="all" class="active">全部</button>
    <button data-filter="invalid">无效</button>
    <button data-filter="degenerate">退化单相</button>
    <button data-filter="f2eqf1">F2==F1</button>
    <button data-filter="pending">待审核</button>
    <input id="q" placeholder="搜索 病例/序列 ...">
    <span class="spacer"></span>
    <button id="export">导出审核结果 CSV</button>
    """


def _render_card(case_id: str, seq_id: str, a, specs, frame_dir_rel: str,
                 safe: str, saved_fids: set) -> str:
    flags = []
    if not a.is_valid:
        flags.append('invalid')
    if any('退化' in n for n in a.notes):
        flags.append('degenerate')
    if a.f1 is not None and a.f2 is not None and a.f1 == a.f2:
        flags.append('f2eqf1')
    rank = qc_priority(a)

    status = '<span class="pill ok">有效</span>' if a.is_valid else '<span class="pill bad">无效</span>'
    ctfc_txt = f"{a.ctfc:.2f}" if a.ctfc is not None else '—'
    rid = html.escape(f"{case_id}/{seq_id}")

    def cell(label: str, fid: Optional[int], fill_p, fill_d) -> str:
        cls = 'cell label-' + label.split('-')[0].split('+')[0]
        if fid is None or fid not in saved_fids:
            reason = '无对应源帧' if fid is not None else '帧号越界'
            return (f'<figure class="{cls} missing"><div class="ph">{label}<br>{reason}</div></figure>')
        fp = f" fill_p={fill_p[fid]:.2f}" if fill_p is not None and fid < len(fill_p) else ''
        fd = f" fill_d={fill_d[fid]:.2f}" if fill_d is not None and fid < len(fill_d) else ''
        cap = f"{rid} [{a.branch_type}]  {label}  f={fid}{fp}{fd}"
        src = f"{frame_dir_rel}/{safe}_f{fid}.png"
        return (f'<figure class="{cls}"><img src="{src}" data-cap="{html.escape(cap)}" '
                f'alt="{html.escape(cap)}" loading="lazy" decoding="async">'
                f'<figcaption>{label} (f={fid})</figcaption></figure>')

    fill_p = a.detail.get('fill_p') if a.detail else None
    fill_d = a.detail.get('fill_d') if a.detail else None

    rows_html = []
    for ri, row in enumerate(specs):
        if ri == 0 and a.f1 is None:
            rows_html.append('<div class="row"><div class="empty" style="width:100%">'
                             'F1 未检出（无效行，仅核对空间标志区）</div></div>')
            continue
        if ri == 1 and a.f2 is None:
            rows_html.append('<div class="row"><div class="empty" style="width:100%">'
                             'F2 未检出（无效行，仅核对空间标志区）</div></div>')
            continue
        rows_html.append('<div class="row">' +
                         ''.join(cell(lb, fid, fill_p, fill_d) for lb, fid in row) +
                         '</div>')

    notes = '; '.join(a.notes)
    return f"""
<section class="card" data-id="{rid}" data-case="{html.escape(case_id)}" data-seq="{html.escape(seq_id)}"
         data-branch="{a.branch_type}" data-f1="{a.f1 if a.f1 is not None else ''}"
         data-f2="{a.f2 if a.f2 is not None else ''}" data-ctfc="{ctfc_txt}"
         data-valid="{'有效' if a.is_valid else '无效'}" data-rank="{rank}"
         data-ranklabel="{PRIORITY_LABEL[rank]}" data-flags="{','.join(flags)}">
  <div class="card-head">
    <span class="title">{rid} [{a.branch_type}]</span>
    {status}
    <span class="pill rank{rank}">{PRIORITY_LABEL[rank]}</span>
    <span class="kv">F1=<b>{a.f1 if a.f1 is not None else '—'}</b> · F2=<b>{a.f2 if a.f2 is not None else '—'}</b>
      · CTFC=<b>{ctfc_txt}</b> · Fmax=<b>{a.max_frame_id}</b> · n={a.n_frames}</span>
    <span class="kv">S={a.seed_px}px · P={a.proximal_px}px · D={a.distal_px}px</span>
    <span class="notes">{html.escape(notes)}</span>
  </div>
  <div class="sheet">{''.join(rows_html)}</div>
  <div class="verdict">
    <label><input type="radio" name="v-{rid}" value="通过">通过</label>
    <label><input type="radio" name="v-{rid}" value="存疑">存疑</label>
    <label><input type="radio" name="v-{rid}" value="拒绝">拒绝</label>
    <textarea placeholder="复核理由 / 存疑说明（自动缓存到本地，可导出 CSV）"></textarea>
  </div>
  <div class="navbtns">
    <button onclick="this.closest('.card').scrollIntoView({{behavior:'smooth'}})">回到本卡</button>
  </div>
</section>"""


# ---------------------------------------------------------------------- 入口

def build_qc_report(results: Dict, output_dir: str, params: Dict,
                    logger: logging.Logger = None,
                    sample_frac: float = 1.0) -> Dict:
    """生成 QC 连拍图帧 + HTML 核对页面 + 待审队列 CSV。

    results: {case_id: {seq_id: CTFCSequenceAnalysis}}
    返回统计 dict {n_branches, n_frames, html, queue_csv}
    """
    logger = logger or logging.getLogger('CTFC.QC')
    qc_dir = os.path.join(output_dir, 'qc')
    frame_dir = os.path.join(qc_dir, 'frames')
    os.makedirs(frame_dir, exist_ok=True)

    rows = [(c, s, results[c][s]) for c in sorted(results) for s in sorted(results[c])]
    rows = _select_rows(rows, sample_frac)

    suffix = params.get('mask_suffix', '_pred.png')
    legend = 'green=branch  cyan=P(proximal)  red=D(distal)  yellow=S(seed/ostium)'
    n_frames_saved = 0
    n_skipped = 0
    cards_html = []
    queue_rows = []

    for case_id, seq_id, a in rows:
        safe = re.sub(r'[^0-9A-Za-z_.-]', '_', f"{case_id}__{seq_id}")
        specs = qc_frame_rows(a.f1, a.f2, a.max_frame_id, a.n_frames)

        wanted = []
        for row in specs:
            for label, fid in row:
                if fid is not None and fid not in wanted:
                    wanted.append(fid)

        saved_fids = set()
        if a.segmented_dir:
            stems = _collect_stems(a.segmented_dir, suffix)
            S = a.detail.get('seed') if a.detail else None
            P = a.detail.get('P') if a.detail else None
            D = a.detail.get('D') if a.detail else None
            for fid in wanted:
                stem = stems.get(fid)
                if stem is None:
                    continue
                src = cv2.imread(os.path.join(a.segmented_dir, stem + '_src.png'))
                if src is None:
                    continue
                raw = MaskProcessor.load_mask(os.path.join(a.segmented_dir, stem + suffix))
                if raw is None:
                    continue
                branch = _extract_branch_mask(raw, a.branch_type, params, src.shape[:2])
                fp = a.detail.get('fill_p') if a.detail else None
                fd = a.detail.get('fill_d') if a.detail else None
                extra = ''
                if fp is not None and fid < len(fp):
                    extra += f"  fill_p={fp[fid]:.2f}"
                if fd is not None and fid < len(fd):
                    extra += f"  fill_d={fd[fid]:.2f}"
                label = next((lb for row in specs for lb, f in row if f == fid), str(fid))
                img = render_qc_frame(src, branch, S, P, D,
                                      f"{a.branch_type} {label} f={fid}{extra}", legend)
                cv2.imwrite(os.path.join(frame_dir, f"{safe}_f{fid}.png"), img,
                            [cv2.IMWRITE_PNG_COMPRESSION, 9])
                saved_fids.add(fid)
                n_frames_saved += 1
        else:
            n_skipped += 1

        cards_html.append(_render_card(case_id, seq_id, a, specs, 'frames',
                                       safe, saved_fids))
        queue_rows.append({
            'case_id': case_id,
            'sequence_id': seq_id,
            'branch_type': a.branch_type,
            'F1': a.f1 if a.f1 is not None else '',
            'F2': a.f2 if a.f2 is not None else '',
            'CTFC': round(a.ctfc, 2) if a.ctfc is not None else '',
            '有效性': '有效' if a.is_valid else '无效',
            '优先级': PRIORITY_LABEL[qc_priority(a)],
            'notes': '; '.join(a.notes),
            '审核结论': '',
            '复核理由': '',
        })

    n_invalid = sum(1 for _, _, a in rows if not a.is_valid)
    n_valid = len(rows) - n_invalid
    doc = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>CTFC QC 逐序列核对</title>
<style>{_css()}</style>
</head>
<body>
<header>
  <h1>CTFC QC 逐序列核对</h1>
  <div class="meta" id="counts"></div>
  <div class="meta">共 {len(rows)} 条（有效 {n_valid} / 无效 {n_invalid}）·
    点击帧放大（原尺寸 ↔ 2×）· 键盘 j/k 跳转上/下一条 · Esc 关闭大图</div>
  <div class="legend">
    <span><i class="sw-branch"></i>绿 = 分支掩码</span>
    <span><i class="sw-p"></i>青 = 近端区 P（离开口最近 10%）</span>
    <span><i class="sw-d"></i>红 = 远端区 D（离开口最远 10%）</span>
    <span><i class="sw-s"></i>黄 = 种子区 S（开口锚点，最早 2%）</span>
    <span class="bd-f1">绿框 = F1 帧</span>
    <span class="bd-f2">红框 = F2 帧</span>
    <span class="bd-fmax">灰框 = Fmax 帧</span>
  </div>
  <div class="toolbar">{_select_opts()}</div>
</header>
<main>
{''.join(cards_html)}
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
<script>{_js('ctfc_qc_' + os.path.basename(os.path.abspath(output_dir)))}</script>
</body>
</html>
"""
    html_path = os.path.join(qc_dir, 'qc_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(doc)

    queue_path = os.path.join(qc_dir, 'qc_queue.csv')
    fieldnames = ['case_id', 'sequence_id', 'branch_type', 'F1', 'F2', 'CTFC',
                  '有效性', '优先级', 'notes', '审核结论', '复核理由']
    with open(queue_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(queue_rows)

    logger.info(f"QC 页面: {html_path}（{len(rows)} 分支，{n_frames_saved} 帧）")
    logger.info(f"QC 队列: {queue_path}")
    if n_skipped:
        logger.warning(f"QC: {n_skipped} 分支无 segmented_dir，未生成图像")
    return {'n_branches': len(rows), 'n_frames': n_frames_saved,
            'html': html_path, 'queue_csv': queue_path}
