#!/usr/bin/env python3
"""队列级 LCA/RCA 分类汇总：逐序列调用 CustomDataset/test.py，聚合成 cls_summary.csv。

对 <queue_root>/<case_id>/<seq_id>/ 下的每个序列目录：
  1. 以该序列为 --test-dir 运行 CustomDataset/test.py（--save-pred 产出 predictions.csv，
     逐帧 pred/prob）；
  2. 聚合：若有 prob_<class> 列则按类取帧均值取 argmax；否则按帧多数票（均值概率打破平局）；
  3. 按 class_map（默认 0:LCA,1:RCA）映射 vessel。

输出 CSV 列：case_id, seq_name, vessel, pred_idx, pred_name, score, n_images
—— 即 ctfc/main.py --cls-summary-csv 所需格式（case_id/seq_name/vessel 三列必需）。

用法：
  python cls/make_cls_summary.py --queue-root <队列根目录> \
      --checkpoint <medvit_cls_exp>/<run>/checkpoint_best.pth \
      --output-csv <queue_root>/.cls_tmp/cls_summary.csv
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_PY = Path(__file__).resolve().parent / "CustomDataset" / "test.py"


def parse_class_map(s):
    """'0:LCA,1:RCA' -> {0: 'LCA', 1: 'RCA'}"""
    out = {}
    for kv in str(s).split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, _, v = kv.partition(":")
        out[int(k)] = v.strip()
    return out


def aggregate_rows(rows):
    """predictions.csv 行 -> (pred_idx, score, n_images, prob 列类名)。

    与推断脚本约定一致：优先 prob_<class> 列逐类取帧均值；否则多数票 + 均值概率。
    """
    if not rows:
        raise RuntimeError("predictions.csv 为空")
    fieldnames = rows[0].keys()
    prob_cols = [c for c in fieldnames if c.startswith("prob_")]
    if prob_cols:
        mean_probs = []
        for col in prob_cols:
            vals = [float(r[col]) for r in rows if r.get(col, "") != ""]
            mean_probs.append(sum(vals) / len(vals) if vals else 0.0)
        pred_idx = max(range(len(mean_probs)), key=lambda i: mean_probs[i])
        return pred_idx, float(mean_probs[pred_idx]), len(rows), \
            [c[len("prob_"):] for c in prob_cols]

    counter = Counter(int(r["pred"]) for r in rows)

    def cls_score(k):
        probs = [float(r.get("prob", "0") or 0) for r in rows if int(r["pred"]) == k]
        return (counter[k], sum(probs) / len(probs) if probs else 0.0)

    pred_idx = max(counter.keys(), key=cls_score)
    probs = [float(r.get("prob", "0") or 0) for r in rows if int(r["pred"]) == pred_idx]
    return pred_idx, (sum(probs) / len(probs) if probs else 0.0), len(rows), []


def classify_sequence(seq_dir, args, class_names):
    """跑一个序列的逐帧分类，返回 (pred_idx, score, n_images)。"""
    seq_dir = Path(seq_dir)
    with tempfile.TemporaryDirectory(prefix="cls_out_") as tmp:
        cmd = [
            sys.executable, str(TEST_PY),
            "--model", args.model,
            "--checkpoint", str(args.checkpoint),
            "--data-path", str(seq_dir.parent),
            "--test-dir", seq_dir.name,
            "--output-dir", tmp,
            "--nb_classes", str(args.nb_classes),
            "--no-labels",
            "--save-pred",
            "--device", args.device,
        ]
        if class_names and all(i < len(class_names) for i in range(args.nb_classes)):
            cmd += ["--class-names", ",".join(class_names)]
        subprocess.run(cmd, check=True, cwd=str(TEST_PY.parent))

        pred_csv = Path(tmp) / "predictions.csv"
        if not pred_csv.is_file():
            raise RuntimeError(f"predictions.csv 未生成: {seq_dir}")
        with open(pred_csv, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        return aggregate_rows(rows)[:3]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue-root", required=True, help="队列根目录（<case>/<seq>/…_src.png）")
    ap.add_argument("--checkpoint", required=True, help="分类 checkpoint（.pth）")
    ap.add_argument("--model", default="MedViT_small")
    ap.add_argument("--nb-classes", dest="nb_classes", type=int, default=2)
    ap.add_argument("--class-map", default="0:LCA,1:RCA", help="类索引->vessel，如 0:LCA,1:RCA")
    ap.add_argument("--output-csv", default=None, help="默认 <queue_root>/cls_summary.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    class_map = parse_class_map(args.class_map)
    class_names = [class_map.get(i, str(i)) for i in range(args.nb_classes)]
    queue_root = Path(args.queue_root)
    out_csv = Path(args.output_csv) if args.output_csv else queue_root / "cls_summary.csv"

    fields = ["case_id", "seq_name", "vessel", "pred_idx", "pred_name", "score", "n_images"]
    n_seq = 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for case_dir in sorted(p for p in queue_root.iterdir() if p.is_dir()):
            for seq_dir in sorted(p for p in case_dir.iterdir() if p.is_dir()):
                if not any(seq_dir.glob("*.png")):
                    continue
                pred_idx, score, n_images = classify_sequence(seq_dir, args, class_names)
                row = {
                    "case_id": case_dir.name,
                    "seq_name": seq_dir.name,
                    "vessel": class_map.get(pred_idx, str(pred_idx)),
                    "pred_idx": pred_idx,
                    "pred_name": class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx),
                    "score": f"{score:.6f}",
                    "n_images": n_images,
                }
                w.writerow(row)
                n_seq += 1
                print(f"[CLS] {case_dir.name}/{seq_dir.name}: "
                      f"{row['vessel']} score={score:.4f} n={n_images}", flush=True)
    print(f"共 {n_seq} 个序列 -> {out_csv}")


if __name__ == "__main__":
    main()
