#!/usr/bin/env python3
"""Batch-generate the two NEW DL-cTFC figure types for the CAD cohort.

For every CAD row whose detailed data all exist (JSON fill curves + NPZ FO
arrays + raw src frame at Fmax) render:

  (1) curve  -- single-panel DL-cTFC filling-rate curve (the right half of
                fig5_compare, redesigned): extent / proximal P / distal D
                curves + F1 / F2 / Fmax markers. Title = branch + figure name
                (no case name). Legend top-left, values box bottom-right.
  (2) fo     -- two-panel first-opacification (FO) spatial map (tighter):
                (a) raw Fmax angiogram, (b) FO viridis map + P/D contours +
                seed. Title = branch + figure name (no case name).

Outputs (paper-quality, default 300 dpi):
  02_ctfc/fig5_curve_gallery/{png/, index.html, gallery_summary.json}
  02_ctfc/fo_map_gallery/    {png/, index.html, gallery_summary.json}

Interrupt-safe: existing PNGs skipped unless --force. One progress line per
--every rows.

Usage:
  python make_cad_galleries.py                      # full CAD, both figures
  python make_cad_galleries.py --only curve         # only the curve figure
  python make_cad_galleries.py --only fo            # only the FO figure
  python make_cad_galleries.py --limit 3            # debug: 3 rows/queue
  python make_cad_galleries.py --force              # redraw existing PNGs
"""
import argparse, json, os, re, sys, time
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

TFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = "/home/share/clr/share/data/AI_TMPFC"
BASE = os.path.join(TFC_ROOT, "paper_results", "02_ctfc", "full_rerun")

BRANCH_ORDER = {"LAD": 0, "LCX": 1, "RCA": 2}
CMAP = cm.viridis


def sanitize(name):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("_")


def seq_stem(seq, branch):
    """CSV 序列ID 若已含 _<branch> 后缀则不重复拼接。"""
    seq = str(seq)
    suf = "_" + str(branch)
    return seq if seq.endswith(suf) else seq + suf


def strip_branch(seq, branch):
    seq = str(seq)
    suf = "_" + str(branch)
    return seq[: -len(suf)] if seq.endswith(suf) else seq


EXCLUSIONS_CSV = os.path.join(TFC_ROOT, "results_f1q", "cohort", "exclusions_badframes.csv")


def load_bad_frames(queue):
    """官方条纹坏图排除清单（results_f1q/cohort），返回 (queue, case, base_seq) 集合。"""
    if not os.path.isfile(EXCLUSIONS_CSV):
        return set()
    ex = pd.read_csv(EXCLUSIONS_CSV, encoding="utf-8-sig")
    out = set()
    for _, r in ex.iterrows():
        m = re.search(r"AI_TMPFC/([^/]+)/", str(r.get("path", "")))
        # 清单里 sequence_id 无分支后缀，直接作 base seq 用
        out.add((m.group(1) if m else "", sanitize(r["case_id"]),
                 sanitize(r["sequence_id"])))
    return {k for k in out if k[0] == queue}


def seq_num(seq):
    return int(str(seq).split("IM")[-1].split("_")[0])


def case_num(case):
    return str(case).split("-")[0]


def contour(mask):
    out = cv2.findContours((mask > 0).astype(np.uint8),
                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = out[1] if len(out) == 3 else out[0]
    return [c.reshape(-1, 2) for c in cnts if len(c) >= 8]


def compute_tags(row, is_bad_frame):
    tags = []
    if is_bad_frame:
        tags.append("bad-frame")
    if str(row.get("有效性") or "").strip() != "有效":
        tags.append("invalid")
    note = str(row.get("备注") or "")
    if "退化" in note or "单相" in note:
        tags.append("degenerate")
    f1, f2, fmax = row.get("F1"), row.get("F2"), row.get("Fmax")
    if pd.notna(f2) and pd.notna(f1) and float(f2) == float(f1):
        tags.append("F2==F1")
    if pd.notna(f2) and pd.notna(fmax) and float(f2) == float(fmax):
        tags.append("censored")
    return tags


# ---------------------------------------------------------------------------
# Figure 1: DL-cTFC filling-rate curve (right-half only, redesigned)
# ---------------------------------------------------------------------------
def plot_curve(out_png, branch, js, f1, f2, ctfc, dpi):
    fe = np.asarray(js["fill_e"], dtype=float)
    fp = np.asarray(js["fill_p"], dtype=float)
    fd = np.asarray(js["fill_d"], dtype=float)
    n = int(js.get("n_frames", len(fe)))
    fmax = int(js.get("fmax", 0))
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(x, fe, color="navy", lw=1.6, label="extent (whole branch)")
    ax.plot(x, fp, color="deepskyblue", lw=1.1, alpha=0.9, label="proximal region P")
    ax.plot(x, fd, color="salmon", lw=1.1, alpha=0.95, label="distal region D")

    def mark(val, color, marker, label, dx, dy, fs=9):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return
        iv = int(round(float(val)))
        if 0 <= iv < n:
            ax.plot(iv, fe[iv], marker, color=color, ms=9, zorder=6)
            ax.axvline(iv, color=color, ls="--", alpha=0.7)
            ax.annotate(label, (iv, fe[iv]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=fs, color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec=color, alpha=0.85))

    mark(f1, "orange", "o", f"F1={int(round(float(f1)))}", -2, 10)
    mark(f2, "magenta", "s", f"F2={int(round(float(f2)))}", 6, -4)
    mark(fmax, "gray", "o", f"Fmax={fmax}", -4, 10)

    # legend top-left (empty region for filling curves)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.75,
              bbox_to_anchor=(0.0, 1.0))

    # values box bottom-right (empty in late frames: curves sit at top)
    vals = [f"F1={int(round(float(f1)))}" if pd.notna(f1) else "F1=–",
            f"F2={int(round(float(f2)))}" if pd.notna(f2) else "F2=–",
            f"Fmax={fmax}"]
    if ctfc is not None and pd.notna(ctfc):
        vals.append(f"cTFC={float(ctfc):.1f} fr")
    ax.text(0.985, 0.04, "  ".join(vals), transform=ax.transAxes,
            fontsize=8.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec="gray", alpha=0.9))

    ax.set_title(f"{branch} — DL-cTFC filling-rate curve", fontsize=11)
    ax.set_xlabel("Frame", fontsize=9)
    ax.set_ylabel("Filling rate (fraction of peak opacification)", fontsize=9)
    ax.set_xlim(-1.5, n + 1.5)
    ax.set_ylim(-0.03, 1.12)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: FO first-opacification spatial map (tighter, renamed title)
# ---------------------------------------------------------------------------
def plot_fo(out_png, branch, z, raw, f1, f2, dpi, margin=12):
    fo, ext, P, D, seed = z["fo"], z["extent"], z["P"], z["D"], z["seed"]
    fmax = int(np.quantile(fo[ext > 0], 1.0))

    ys, xs = np.where(ext > 0)
    if len(ys) == 0:
        return False
    r0, r1 = max(0, ys.min() - margin), min(raw.shape[0], ys.max() + margin + 1)
    c0, c1 = max(0, xs.min() - margin), min(raw.shape[1], xs.max() + margin + 1)
    sl = (slice(r0, r1), slice(c0, c1))
    raw, ext, P, D, seed, fo = (a[sl] for a in (raw, ext, P, D, seed, fo))
    H, W = raw.shape

    # size the figure to the actual crop so the content fills the canvas
    aspect = float(np.clip(W / H, 0.5, 1.5))
    ph = 3.8                       # panel image height (in)
    pw = ph * aspect               # panel image width (in)
    fig_w = 2 * pw + 1.05          # + colorbar column + generous (a)-(b) gap
    fig_h = ph + 0.74              # + suptitle + axis titles

    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1, 1, 0.085], "wspace": 0.11})

    axa.imshow(raw, cmap="gray", vmin=0, vmax=1)
    axa.set_title(f"(a) Raw angiogram (Fmax={fmax})", fontsize=9.5)
    axa.axis("off")

    axb.imshow(raw, cmap="gray", vmin=0, vmax=1, alpha=0.22)
    fo_disp = np.where(ext > 0, fo, np.nan)
    im = axb.imshow(fo_disp, cmap=CMAP, norm=Normalize(vmin=0, vmax=fmax),
                    interpolation="nearest")
    for cnt in contour(P):
        axb.plot(np.append(cnt[:, 0], cnt[0, 0]), np.append(cnt[:, 1], cnt[0, 1]),
                 color="cyan", lw=1.8)
    for cnt in contour(D):
        axb.plot(np.append(cnt[:, 0], cnt[0, 0]), np.append(cnt[:, 1], cnt[0, 1]),
                 color="deeppink", lw=1.8)
    # background mask (not extent / P / D / seed) for label placement
    bg = (ext == 0) & (P == 0) & (D == 0) & (seed == 0)

    def bg_x_in_band(y0f, y1f):
        band = bg[int(y0f * H):int(y1f * H)]
        return float(np.argmax(band.sum(axis=0))) / W   # most-blank column

    f1s = int(f1) if f1 is not None and not (isinstance(f1, float) and np.isnan(f1)) else "–"
    f2s = int(f2) if f2 is not None and not (isinstance(f2, float) and np.isnan(f2)) else "–"
    py, px = np.where(P > 0); dy, dx = np.where(D > 0)
    # P label -> blank spot in the TOP band; D label -> blank spot in the BOTTOM band
    if len(py):
        pc = (px.mean(), py.mean())
        axb.annotate(f"P (proximal)\nF1=q15(P-FO)={f1s}", xy=pc,
                     xytext=(W * bg_x_in_band(0.0, 0.24), H * 0.11), fontsize=7.5,
                     color="cyan", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="cyan", lw=1.0),
                     bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="cyan",
                               alpha=0.92))
    if len(dy):
        dc = (dx.mean(), dy.mean())
        axb.annotate(f"D (distal)\nF2=q10(D-FO)={f2s}", xy=dc,
                     xytext=(W * bg_x_in_band(0.76, 1.0), H * 0.89), fontsize=7.5,
                     color="deeppink", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="deeppink", lw=1.0),
                     bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="deeppink",
                               alpha=0.92))
    axb.set_title("(b) FO map + P/D regions", fontsize=9.5)
    axb.axis("off")

    cb = fig.colorbar(im, cax=axc)
    cb.set_label("FO (frame 0→Fmax)", fontsize=7.5)
    axc.tick_params(labelsize=7)

    fig.suptitle(f"{branch} — first-opacification (FO) spatial map", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
def src_frame_path(queue, case, seq, branch, fmax):
    cnum, snum = case_num(case), seq_num(seq)
    sub = "segmented3c2" if branch in ("LAD", "LCX") else "segmented"
    base_seq = str(seq).split("_")[0]
    return (f"{DATA_ROOT}/{queue}/{case}/{base_seq}/{sub}/"
            f"cag2-c{cnum}-s{snum}-f{fmax}_src.png")


def build_html(out_html, title, items, n_missing, n_written, n_skipped):
    data_json = json.dumps(items, ensure_ascii=False)
    html = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:16px;background:#fafafa;color:#222}
  h2{margin:0 0 10px}
  .bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
  .bar label{font-size:13px}
  select,input[type=search]{padding:5px 8px;font-size:13px;border:1px solid #bbb;border-radius:4px}
  #count{color:#666;font-size:13px}
  .hint{color:#888;font-size:12px;margin-bottom:10px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}
  .card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px;cursor:zoom-in}
  .card:hover{box-shadow:0 2px 10px rgba(0,0,0,.25)}
  .card img{width:100%;border:1px solid #eee;border-radius:4px;display:block}
  .meta{font-size:12px;color:#333;margin-top:6px;line-height:1.6}
  .tag{display:inline-block;border-radius:3px;padding:0 5px;margin-right:4px;font-size:11px;background:#eee}
  .tag.degenerate{background:#fdd}.tag.censored{background:#d9efe9}.tag.f2f1{background:#fdf3c9}
.tag.badframe{background:#f8d7da}.tag.invalid{background:#fff3cd}
  #modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:10;
         justify-content:center;align-items:center;cursor:zoom-out}
  #modal img{max-width:96vw;max-height:96vh;box-shadow:0 0 30px rgba(0,0,0,.6)}
</style></head><body>
<h2>__TITLE__</h2>
<div class="bar">
  <label>分支
    <select id="b"><option value="all">全部</option>
      <option>LAD</option><option>LCX</option><option>RCA</option></select></label>
  <label>标签
    <select id="t"><option value="all">全部</option>
      <option value="clean">干净</option>
      <option value="bad-frame">bad-frame 条纹坏图</option>
      <option value="invalid">invalid 无效</option>
      <option value="degenerate">degenerate</option>
      <option value="censored">censored</option>
      <option value="F2==F1">F2==F1</option></select></label>
  <input id="s" type="search" placeholder="搜索 病例 / 序列 / 分支 …">
  <span id="count"></span>
</div>
<div class="hint">共 __N__ 条（出图 __W__ 张，缺数据跳过 __M__ 张，续跑跳过 __K__ 张）。
排序：分支 LAD/LCX/RCA → 干净优先、bad-frame/invalid 垫底 → 病例 → 序列。点卡片放大，Esc 或点空白关闭。
bad-frame=条纹坏图（排除清单），invalid=有效性≠有效，选论文图请避开这两类。</div>
<div class="grid" id="grid"></div>
<div id="modal"><img id="modalimg" alt="enlarged"></div>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const grid = $('grid'), modal = $('modal'), mimg = $('modalimg');
const TAGCLASS = {degenerate:'degenerate', censored:'censored', 'F2==F1':'f2f1',
                  'bad-frame':'badframe', invalid:'invalid'};
function tagHtml(tags){
  if(!tags.length) return '<span class="tag">干净</span>';
  return tags.map(t => '<span class="tag ' + (TAGCLASS[t]||'') + '">' + t + '</span>').join('');
}
DATA.forEach((d, i) => {
  const c = document.createElement('div'); c.className='card'; c.dataset.i=i;
  const img = document.createElement('img'); img.loading='lazy'; img.src=d.f;
  img.alt = d.case + ' ' + d.seq + ' ' + d.branch;
  const m = document.createElement('div'); m.className='meta';
  m.innerHTML = d.case + ' / ' + d.seq + ' / ' + d.branch +
    ' / F1=' + d.f1 + ' F2=' + d.f2 + ' CTFC=' + (d.ctfc==null?'–':d.ctfc) + '<br>' + tagHtml(d.tags);
  c.appendChild(img); c.appendChild(m); grid.appendChild(c);
});
function searchText(d){
  return [d.case, d.seq, d.branch, d.tags.join(' '),
          'f1='+d.f1, 'f2='+d.f2, 'ctfc='+d.ctfc].join(' ');
}
function apply(){
  const b=$('b').value, t=$('t').value, s=$('s').value.trim().toLowerCase();
  let n=0; const cards=grid.children;
  for(let i=0;i<cards.length;i++){
    const d=DATA[i]; let ok=true;
    if(b!=='all' && d.branch!==b) ok=false;
    if(ok && t!=='all'){ if(t==='clean') ok=d.tags.length===0; else ok=d.tags.indexOf(t)>=0; }
    if(ok && s) ok=searchText(d).toLowerCase().indexOf(s)>=0;
    cards[i].style.display = ok?'':'none'; if(ok) n++;
  }
  $('count').textContent='显示 '+n+' / '+DATA.length;
}
['b','t'].forEach(id=>$(id).addEventListener('change',apply));
$('s').addEventListener('input',apply);
grid.addEventListener('click',e=>{const card=e.target.closest('.card');if(!card)return;
  mimg.src=DATA[card.dataset.i].f; modal.style.display='flex';});
modal.addEventListener('click',()=>{modal.style.display='none';});
document.addEventListener('keydown',e=>{if(e.key==='Escape')modal.style.display='none';});
apply();
</script></body></html>
"""
    html = html.replace("__TITLE__", title).replace("__DATA__", data_json)
    html = html.replace("__N__", str(len(items))).replace("__W__", str(n_written))
    html = html.replace("__M__", str(n_missing)).replace("__K__", str(n_skipped))
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="CAD-ID1-ID64")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=os.path.join(TFC_ROOT, "paper_results", "02_ctfc"))
    ap.add_argument("--only", choices=["both", "curve", "fo"], default="both")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--every", type=int, default=100, help="progress line interval")
    args = ap.parse_args()

    qdir = os.path.join(args.base, args.queue)
    csv_path = os.path.join(qdir, "ctfc_results.csv")
    det = os.path.join(qdir, "detailed")
    if not os.path.isfile(csv_path):
        print(f"FATAL: {csv_path} missing", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if args.limit > 0:
        df = df.head(args.limit)

    curve_dir = os.path.join(args.out, "fig5_curve_gallery", "png")
    fo_dir = os.path.join(args.out, "fo_map_gallery", "png")
    os.makedirs(curve_dir, exist_ok=True)
    os.makedirs(fo_dir, exist_ok=True)
    plt.ioff()
    bad_frames = load_bad_frames(args.queue)

    curve_items, fo_items = [], []
    stat = {"rows": len(df), "no_data": 0, "curve_written": 0, "curve_skip": 0,
            "curve_missing": 0, "fo_written": 0, "fo_skip": 0, "fo_missing": 0,
            "err": 0}
    t0 = time.time()

    for idx, (_, row) in enumerate(df.iterrows()):
        case, seq, branch = str(row["病例ID"]), str(row["序列ID"]), str(row["分支类型"])
        # CSV 序列ID already matches the detailed filename for all branches
        # (LAD/LCX carry the _<branch> suffix; RCA has none) — use it raw.
        stem = seq
        fname = f"{sanitize(args.queue)}__{sanitize(case)}__{sanitize(stem)}.png"
        jpath = os.path.join(det, f"{case}__{stem}.json")
        npzpath = os.path.join(det, f"{case}__{stem}.npz")
        f1 = row.get("F1"); f2 = row.get("F2"); fmax_c = row.get("Fmax")
        ctfc = row.get("CTFC(浮点)")
        bkey = (args.queue, sanitize(case), sanitize(strip_branch(seq, branch)))
        tags = compute_tags(row, is_bad_frame=bkey in bad_frames)
        meta = {"f": f"png/{fname}", "case": case, "seq": seq, "branch": branch,
                "f1": int(round(float(f1))) if pd.notna(f1) else None,
                "f2": int(round(float(f2))) if pd.notna(f2) else None,
                "ctfc": round(float(ctfc), 2) if pd.notna(ctfc) else None,
                "tags": tags}
        want_curve = args.only in ("both", "curve")
        want_fo = args.only in ("both", "fo")

        try:
            # ---- curve (needs JSON) ----
            if want_curve:
                cp = os.path.join(curve_dir, fname)
                if os.path.isfile(cp) and not args.force:
                    curve_items.append(meta); stat["curve_skip"] += 1
                else:
                    if not os.path.isfile(jpath):
                        stat["curve_missing"] += 1
                    else:
                        js = json.load(open(jpath, encoding="utf-8"))
                        if all(isinstance(js.get(k), list) and len(js[k]) for k in ("fill_e", "fill_p", "fill_d")):
                            plot_curve(cp, branch, js, f1, f2, ctfc, args.dpi)
                            curve_items.append(meta); stat["curve_written"] += 1
                        else:
                            stat["curve_missing"] += 1
            # ---- fo (needs NPZ + src) ----
            if want_fo:
                fp = os.path.join(fo_dir, fname)
                if os.path.isfile(fp) and not args.force:
                    fo_items.append(meta); stat["fo_skip"] += 1
                else:
                    if not (os.path.isfile(npzpath) and pd.notna(fmax_c)):
                        stat["fo_missing"] += 1
                    else:
                        z = np.load(npzpath)
                        sp = src_frame_path(args.queue, case, seq, branch, int(fmax_c))
                        if not os.path.isfile(sp):
                            stat["fo_missing"] += 1
                        else:
                            raw = np.array(Image.open(sp).convert("L")).astype(float)
                            lo, hi = np.percentile(raw, [1, 99])
                            raw = np.clip((raw - lo) / (hi - lo), 0, 1)
                            if plot_fo(fp, branch, z, raw, f1, f2, args.dpi):
                                fo_items.append(meta); stat["fo_written"] += 1
                            else:
                                stat["fo_missing"] += 1
        except Exception as e:
            stat["err"] += 1
            print(f"[ERR] {case} {stem}: {e}", file=sys.stderr)

        if (idx + 1) % args.every == 0:
            el = time.time() - t0
            print(f"[{args.queue}] {idx+1}/{len(df)}  曲线 新{stat['curve_written']}/跳{stat['curve_skip']}  "
                  f"FO 新{stat['fo_written']}/跳{stat['fo_skip']}  err{stat['err']}  "
                  f"{el:.0f}s", flush=True)

    # sort: branch -> clean first, soft tags middle, bad-frame/invalid last -> case -> seq
    def sortk(d):
        if "bad-frame" in d["tags"] or "invalid" in d["tags"]:
            sev = 2
        elif d["tags"]:
            sev = 1
        else:
            sev = 0
        return (BRANCH_ORDER.get(d["branch"], 9), sev, d["case"], d["seq"])
    curve_items.sort(key=sortk)
    fo_items.sort(key=sortk)

    summary = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "queue": args.queue, "dpi": args.dpi, "only": args.only,
               "pngs_curve": stat["curve_written"], "pngs_curve_skip": stat["curve_skip"],
               "pngs_curve_missing": stat["curve_missing"],
               "pngs_fo": stat["fo_written"], "pngs_fo_skip": stat["fo_skip"],
               "pngs_fo_missing": stat["fo_missing"], "errors": stat["err"],
               "n_curve_cards": len(curve_items), "n_fo_cards": len(fo_items),
               "by_tag": {t: sum(1 for d in curve_items if t in d["tags"])
                          for t in ("bad-frame", "invalid", "degenerate", "censored", "F2==F1")},
               "clean_cards": sum(1 for d in curve_items if not d["tags"]),
               "elapsed_s": round(time.time() - t0, 1)}

    def write_gallery(name, title, items, missing, written, skipped):
        build_html(os.path.join(args.out, name, "index.html"), title, items,
                   missing, written, skipped)
        with open(os.path.join(args.out, name, "gallery_summary.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({**summary, "by_branch": {b: sum(1 for d in items if d["branch"] == b)
                                                for b in ("LAD", "LCX", "RCA")}},
                      fh, ensure_ascii=False, indent=2)

    if want_curve:
        write_gallery("fig5_curve_gallery", "DL-cTFC 充盈率曲线画廊 (CAD)",
                      curve_items, stat["curve_missing"], stat["curve_written"], stat["curve_skip"])
    if want_fo:
        write_gallery("fo_map_gallery", "FO 首充帧空间图画廊 (CAD)",
                      fo_items, stat["fo_missing"], stat["fo_written"], stat["fo_skip"])

    print("\n===== 完成 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
