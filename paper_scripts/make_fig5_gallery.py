#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_fig5_gallery.py — 生成 SCI 论文 Fig 5 候选曲线画廊。

对三个队列 full_rerun 的 ctfc_results.csv 中所有 有效性==有效 且 F1/F2 非空 的行，
找到对应的 detailed JSON（含 fill_p / fill_d 充盈率曲线），每张画一幅小图
（FR_P / FR_D 曲线 + F1 / F2 / Fmax 竖线），并生成可筛选的静态 index.html 与
gallery_summary.json。

用法：
    python paper_scripts/make_fig5_gallery.py                 # 跑全部三队列
    python paper_scripts/make_fig5_gallery.py --queues Others-ID1-ID27
    python paper_scripts/make_fig5_gallery.py --limit 5       # 每队列最多 5 行（调试）
    python paper_scripts/make_fig5_gallery.py --force         # 重画已存在的 PNG

可中断续跑：已存在的 PNG 默认跳过（--force 强制重画）。
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_QUEUES = ["CAD-ID1-ID64", "CMVD-ID1-ID50", "Others-ID1-ID27"]
BRANCH_ORDER = {"LAD": 0, "LCX": 1, "RCA": 2}

TFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASE = os.path.join(TFC_ROOT, "paper_results", "02_ctfc", "full_rerun")
DEFAULT_OUT = os.path.join(TFC_ROOT, "paper_results", "02_ctfc", "fig5_gallery")


def sanitize(name):
    """文件名安全化：只保留 字母/数字/-/_/. """
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("_")


def seq_stem(seq, branch):
    """归一化序列 stem：CSV 序列ID 若已含 _<branch> 后缀则不重复拼接。"""
    seq = str(seq)
    suf = "_" + str(branch)
    if seq.endswith(suf):
        return seq
    return seq + suf


def strip_branch(seq, branch):
    """去掉序列ID尾部的 _<branch>（若有）。"""
    seq = str(seq)
    suf = "_" + str(branch)
    return seq[: -len(suf)] if seq.endswith(suf) else seq


def build_json_index(queue_dir):
    """
    返回 dict: (case_id, sequence_id, branch_type) -> json_path
    解析 detailed/ 下所有 .json 的头部字段（只缓存路径，不保留曲线）。
    """
    det = os.path.join(queue_dir, "detailed")
    index = {}
    if not os.path.isdir(det):
        return index
    for fn in sorted(os.listdir(det)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(det, fn)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            key = (str(d.get("case_id")), str(d.get("sequence_id")),
                   str(d.get("branch_type")))
            index.setdefault(key, path)
        except Exception:
            continue
    return index


def find_json(row, filename_index, parsed_index):
    """
    按 case/seq/branch 找 detailed JSON。
    1) 直接文件名匹配：<case>__<seq>.json 或 <case>__<seq>_<branch>.json
    2) 兜底：解析索引 (case_id, sequence_id, branch_type)
    返回 path 或 None。
    """
    case, seq, branch = (str(row["病例ID"]), str(row["序列ID"]), str(row["分支类型"]))
    for stem in (f"{case}__{seq}", f"{case}__{seq}_{branch}",
                 f"{case}__{strip_branch(seq, branch)}"):
        p = filename_index.get(stem)
        if p:
            return p
    for key in ((case, seq, branch), (case, strip_branch(seq, branch), branch)):
        p = parsed_index.get(key)
        if p:
            return p
    return None


def compute_tags(row):
    """备注含 退化/单相 → degenerate；F2==F1 → F2==F1；F2==Fmax → censored。"""
    tags = []
    note = str(row.get("备注") or "")
    if "退化" in note or "单相" in note:
        tags.append("degenerate")
    f1, f2, fmax = row.get("F1"), row.get("F2"), row.get("Fmax")
    if pd.notna(f2) and pd.notna(f1) and float(f2) == float(f1):
        tags.append("F2==F1")
    if pd.notna(f2) and pd.notna(fmax) and float(f2) == float(fmax):
        tags.append("censored")
    return tags


def plot_curve(out_png, queue, case, seq, branch, f1, f2, fmax, ctfc, tags, js, dpi):
    fill_p = np.asarray(js["fill_p"], dtype=float)
    fill_d = np.asarray(js["fill_d"], dtype=float)
    n = len(fill_p)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
    ax.plot(x, fill_p, color="cyan", lw=1.2, label="FR_P")
    ax.plot(x, fill_d, color="red", lw=1.2, label="FR_D")

    def vline(v, color, ls, label):
        if v is None or pd.isna(v):
            return
        iv = int(round(float(v)))
        if 0 <= iv < n:
            ax.axvline(iv, color=color, linestyle=ls, lw=1.0, alpha=0.9)
            ax.text(iv + 0.35, 0.04, label, color=color, fontsize=7.5,
                    rotation=90, va="bottom", ha="left")

    vline(f1, "green", "solid", f"F1={int(round(float(f1)))}")
    vline(f2, "red", "solid", f"F2={int(round(float(f2)))}")
    vline(fmax, "gray", "dotted", "Fmax")

    ax.set_xlim(0, n - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("frame index", fontsize=8)
    ax.set_ylabel("filling rate", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.6)

    title = (f"{queue} | {case} / {seq} | {branch} | "
             f"F1={int(round(float(f1)))} F2={int(round(float(f2)))} "
             f"CTFC={float(ctfc):.2f}")
    if tags:
        title += "  " + " ".join(f"[{t}]" for t in tags)
    ax.set_title(title, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def build_html(out_html, items, n_missing, n_written, n_skipped):
    """生成纯静态单文件 index.html（内嵌 CSS/JS，无外网依赖）。"""
    data_json = json.dumps(items, ensure_ascii=False)
    html = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CTFC Fig 5 候选曲线画廊</title>
<style>
  body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:16px;background:#fafafa;color:#222}
  h2{margin:0 0 10px}
  .bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
  .bar label{font-size:13px}
  select,input[type=search]{padding:5px 8px;font-size:13px;border:1px solid #bbb;border-radius:4px}
  #count{color:#666;font-size:13px}
  .hint{color:#888;font-size:12px;margin-bottom:10px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}
  .card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px;cursor:zoom-in}
  .card:hover{box-shadow:0 2px 10px rgba(0,0,0,.25)}
  .card img{width:100%;border:1px solid #eee;border-radius:4px;display:block}
  .meta{font-size:12px;color:#333;margin-top:6px;line-height:1.6}
  .tag{display:inline-block;border-radius:3px;padding:0 5px;margin-right:4px;font-size:11px;background:#eee}
  .tag.degenerate{background:#fdd}
  .tag.censored{background:#d9efe9}
  .tag.f2f1{background:#fdf3c9}
  #modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:10;
         justify-content:center;align-items:center;cursor:zoom-out}
  #modal img{max-width:96vw;max-height:96vh;box-shadow:0 0 30px rgba(0,0,0,.6)}
</style>
</head>
<body>
<h2>CTFC Fig 5 候选曲线画廊</h2>
<div class="bar">
  <label>队列
    <select id="q"><option value="all">全部</option></select></label>
  <label>分支
    <select id="b"><option value="all">全部</option>
      <option>LAD</option><option>LCX</option><option>RCA</option></select></label>
  <label>标签
    <select id="t"><option value="all">全部</option>
      <option value="clean">干净</option>
      <option value="degenerate">degenerate</option>
      <option value="censored">censored</option>
      <option value="F2==F1">F2==F1</option></select></label>
  <input id="s" type="search" placeholder="搜索 病例 / 序列 / 分支 …">
  <span id="count"></span>
</div>
<div class="hint">共 __N__ 条（出图 __W__ 张，缺 JSON 跳过 __M__ 张，续跑跳过 __K__ 张）。排序：队列 → 分支 → 干净优先 → 病例 → 序列。点卡片放大，Esc 或点空白关闭。</div>
<div class="grid" id="grid"></div>
<div id="modal"><img id="modalimg" alt="enlarged plot"></div>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const grid = $('grid'), modal = $('modal'), mimg = $('modalimg');

// 队列下拉选项（按 DATA 出现顺序）
(function(){
  const seen = new Set(), sel = $('q');
  DATA.forEach(d => { if(!seen.has(d.queue)){ seen.add(d.queue);
    const o = document.createElement('option'); o.value = d.queue; o.textContent = d.queue;
    sel.appendChild(o); }});
})();

const TAGCLASS = {degenerate:'degenerate', censored:'censored', 'F2==F1':'f2f1'};
function tagHtml(tags){
  if(!tags.length) return '<span class="tag">干净</span>';
  return tags.map(t => '<span class="tag ' + (TAGCLASS[t] || '') + '">' + t + '</span>').join('');
}

// 渲染（DATA 已按目标顺序排好，过滤只切换显隐，保持顺序）
DATA.forEach((d, i) => {
  const c = document.createElement('div');
  c.className = 'card'; c.dataset.i = i;
  const img = document.createElement('img');
  img.loading = 'lazy'; img.src = d.f;
  img.alt = d.queue + ' ' + d.case + ' ' + d.seq + ' ' + d.branch;
  const m = document.createElement('div'); m.className = 'meta';
  m.innerHTML = d.queue + ' / ' + d.case + ' / ' + d.seq + ' / ' + d.branch +
    ' / F1=' + d.f1 + ' F2=' + d.f2 + ' CTFC=' + d.ctfc + '<br>' + tagHtml(d.tags);
  c.appendChild(img); c.appendChild(m); grid.appendChild(c);
});

function searchText(d){
  return [d.queue, d.case, d.seq, d.branch, d.tags.join(' '),
          'f1=' + d.f1, 'f2=' + d.f2, 'ctfc=' + d.ctfc].join(' ');
}
function apply(){
  const q = $('q').value, b = $('b').value, t = $('t').value;
  const s = $('s').value.trim().toLowerCase();
  let n = 0;
  const cards = grid.children;
  for(let i = 0; i < cards.length; i++){
    const d = DATA[i]; let ok = true;
    if(q !== 'all' && d.queue !== q) ok = false;
    if(ok && b !== 'all' && d.branch !== b) ok = false;
    if(ok && t !== 'all'){
      if(t === 'clean') ok = d.tags.length === 0;
      else ok = d.tags.indexOf(t) >= 0;
    }
    if(ok && s) ok = searchText(d).toLowerCase().indexOf(s) >= 0;
    cards[i].style.display = ok ? '' : 'none';
    if(ok) n++;
  }
  $('count').textContent = '显示 ' + n + ' / ' + DATA.length;
}
['q','b','t'].forEach(id => $(id).addEventListener('change', apply));
$('s').addEventListener('input', apply);

grid.addEventListener('click', e => {
  const card = e.target.closest('.card'); if(!card) return;
  mimg.src = DATA[card.dataset.i].f;
  modal.style.display = 'flex';
});
modal.addEventListener('click', () => { modal.style.display = 'none'; });
document.addEventListener('keydown', e => { if(e.key === 'Escape') modal.style.display = 'none'; });
apply();
</script>
</body>
</html>
"""
    html = html.replace("__DATA__", data_json)
    html = html.replace("__N__", str(len(items)))
    html = html.replace("__W__", str(n_written))
    html = html.replace("__M__", str(n_missing))
    html = html.replace("__K__", str(n_skipped))
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)


def main():
    ap = argparse.ArgumentParser(description="CTFC Fig 5 候选曲线画廊生成")
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="full_rerun 根目录（默认 %(default)s）")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="gallery 输出目录（默认 %(default)s）")
    ap.add_argument("--queues", default=",".join(DEFAULT_QUEUES),
                    help="逗号分隔队列名（默认全部三队列）")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--force", action="store_true", help="重画已存在的 PNG")
    ap.add_argument("--limit", type=int, default=0,
                    help="每队列最多处理的行数（0=不限，调试用）")
    args = ap.parse_args()

    queues = [q.strip() for q in args.queues.split(",") if q.strip()]
    png_dir = os.path.join(args.out, "png")
    os.makedirs(png_dir, exist_ok=True)
    plt.ioff()

    items = []          # HTML 卡片数据（已排序前）
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base": args.base,
        "out_dir": args.out,
        "total_rows": 0,
        "valid_rows": 0,
        "pngs_written": 0,
        "pngs_skipped_existing": 0,
        "missing_json": 0,
        "json_mismatch": 0,
        "bad_json": 0,
        "valid_by_queue_branch": {},
        "by_queue": {},
    }

    for qi, queue in enumerate(queues):
        qdir = os.path.join(args.base, queue)
        csv_path = os.path.join(qdir, "ctfc_results.csv")
        if not os.path.isfile(csv_path):
            print(f"[WARN] {csv_path} 不存在，跳过队列 {queue}", file=sys.stderr)
            continue

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        total = len(df)
        vdf = df[(df["有效性"] == "有效") & df["F1"].notna() & df["F2"].notna()].copy()
        if args.limit > 0:
            vdf = vdf.head(args.limit)

        # 文件名索引：stem(去掉 .json) -> path，仅 .json
        det = os.path.join(qdir, "detailed")
        filename_index = {}
        if os.path.isdir(det):
            for fn in os.listdir(det):
                if fn.endswith(".json"):
                    filename_index[fn[:-5]] = os.path.join(det, fn)

        # 解析索引（兜底用）
        parsed_index = build_json_index(qdir)

        qstat = {
            "total_rows": total,
            "valid_rows": len(vdf),
            "by_branch": {},
            "missing_json": 0,
            "json_mismatch": 0,
            "bad_json": 0,
            "pngs_written": 0,
            "pngs_skipped_existing": 0,
        }
        by_branch = {}

        for idx, (_, row) in enumerate(vdf.iterrows()):
            case = str(row["病例ID"])
            seq = str(row["序列ID"])
            branch = str(row["分支类型"])
            f1, f2 = float(row["F1"]), float(row["F2"])
            fmax = row.get("Fmax")
            ctfc = float(row["CTFC(浮点)"])
            by_branch[branch] = by_branch.get(branch, 0) + 1

            stem = seq_stem(seq, branch)
            out_png = os.path.join(
                png_dir, f"{sanitize(queue)}__{sanitize(case)}__{sanitize(stem)}.png")

            if os.path.isfile(out_png) and not args.force:
                # 续跑：PNG 已存在，仍需收集卡片元数据
                tags = compute_tags(row)
                items.append({
                    "f": f"png/{os.path.basename(out_png)}",
                    "queue": queue, "case": case, "seq": seq, "branch": branch,
                    "f1": int(round(f1)), "f2": int(round(f2)),
                    "ctfc": round(ctfc, 2), "tags": tags,
                })
                qstat["pngs_skipped_existing"] += 1
                summary["pngs_skipped_existing"] += 1
                continue

            jpath = find_json(row, filename_index, parsed_index)
            if jpath is None:
                qstat["missing_json"] += 1
                summary["missing_json"] += 1
                continue

            try:
                with open(jpath, "r", encoding="utf-8") as fh:
                    js = json.load(fh)
                # 内容一致性校验，防止文件名误配
                if (str(js.get("case_id")) != case
                        or str(js.get("branch_type")) != branch
                        or str(js.get("sequence_id")) not in
                           (seq, strip_branch(seq, branch))):
                    qstat["json_mismatch"] += 1
                    summary["json_mismatch"] += 1
                    continue
                fp, fd = js.get("fill_p"), js.get("fill_d")
                if not (isinstance(fp, list) and isinstance(fd, list)
                        and len(fp) == len(fd) and len(fp) > 0):
                    qstat["bad_json"] += 1
                    summary["bad_json"] += 1
                    continue
            except Exception as e:
                print(f"[WARN] 读取 JSON 失败 {jpath}: {e}", file=sys.stderr)
                qstat["bad_json"] += 1
                summary["bad_json"] += 1
                continue

            tags = compute_tags(row)
            plot_curve(out_png, queue, case, seq, branch, f1, f2, fmax,
                       ctfc, tags, js, args.dpi)
            qstat["pngs_written"] += 1
            summary["pngs_written"] += 1
            items.append({
                "f": f"png/{os.path.basename(out_png)}",
                "queue": queue, "case": case, "seq": seq, "branch": branch,
                "f1": int(round(f1)), "f2": int(round(f2)),
                "ctfc": round(ctfc, 2), "tags": tags,
            })
            if (idx + 1) % 100 == 0:
                print(f"[{queue}] {idx + 1}/{len(vdf)} 行已处理 "
                      f"(缺JSON {qstat['missing_json']}, 出图 {qstat['pngs_written']})")

        qstat["by_branch"] = by_branch
        summary["by_queue"][queue] = qstat
        summary["valid_by_queue_branch"][queue] = by_branch
        summary["total_rows"] += total
        summary["valid_rows"] += len(vdf)
        print(f"[{queue}] 完成：总行 {total}，有效 {len(vdf)}，"
              f"分支分布 {by_branch}，缺JSON {qstat['missing_json']}，"
              f"新出图 {qstat['pngs_written']}，续跑跳过 {qstat['pngs_skipped_existing']}")

    # 排序：队列（按 --queues 顺序）→ 分支 LAD/LCX/RCA → 干净优先 → 病例 → 序列
    qorder = {q: i for i, q in enumerate(queues)}
    items.sort(key=lambda d: (
        qorder.get(d["queue"], 99),
        BRANCH_ORDER.get(d["branch"], 9),
        0 if not d["tags"] else 1,
        d["case"], d["seq"],
    ))

    out_html = os.path.join(args.out, "index.html")
    build_html(out_html, items, summary["missing_json"],
               summary["pngs_written"], summary["pngs_skipped_existing"])
    summary["html"] = out_html
    summary["png_dir"] = png_dir
    summary["n_cards"] = len(items)

    out_sum = os.path.join(args.out, "gallery_summary.json")
    with open(out_sum, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n===== 汇总 =====")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("by_queue", "valid_by_queue_branch")},
                     ensure_ascii=False, indent=2))
    for q, s in summary["by_queue"].items():
        print(f"  {q}: 总 {s['total_rows']} / 有效 {s['valid_rows']} "
              f"分支 {s['by_branch']} 缺JSON {s['missing_json']} "
              f"出图 {s['pngs_written']} 续跑跳过 {s['pngs_skipped_existing']}")
    print(f"HTML: {out_html}")
    print(f"Summary: {out_sum}")


if __name__ == "__main__":
    main()
