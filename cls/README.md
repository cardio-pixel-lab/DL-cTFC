# cls/ — LCA/RCA 序列分类（MedViT）

逐帧 CAG 图像二分类（LCA / RCA），按序列聚合后产出 `cls_summary.csv`，
供 `ctfc/main.py --cls-summary-csv` 做 LCA 拆支（LAD/LCX）与 RCA 单支的路由。
缺该 CSV 或条目不全时，ctfc 管线会把序列默认按 RCA 处理（见根 README 已知局限）。

代码vendored自开源 [MedViT](https://github.com/Cuthbert-Cai/MedViT)（见
[LICENSE](LICENSE) 与 [upstream_README.md](upstream_README.md)）：
顶层 `MedViT.py`/`utils.py` 为模型，`CustomDataset/` 为本项目的训练/推理适配层
（timm 风格 `main.py`/`test.py`，`train.sh` 为分布式训练入口）。

## 环境

```bash
pip install -r cls/requirements.txt   # torch + timm 栈（可与 seg/ 共用环境）
```

## 数据（不入库）

ImageFolder 帧级分类集（如 `train/{lca,rca}/*.png`、`test/...`）。

## 训练

```bash
cd cls/CustomDataset
bash train.sh <GPU数> # 透传 main.py 其余参数；输出到 ../medvit_cls_exp/<时间戳>/
```

## 推理：队列 → cls_summary.csv

```bash
python cls/make_cls_summary.py \
    --queue-root <队列根目录> \
    --checkpoint <medvit_cls_exp>/<run>/checkpoint_best.pth \
    --output-csv <队列根目录>/.cls_tmp/cls_summary.csv
```

内部对每个 `<case>/<seq>` 序列调用 `CustomDataset/test.py`（`--no-labels
--save-pred`）得逐帧 `predictions.csv`，再按序列聚合：有 `prob_<class>` 列时
逐类取帧均值取 argmax，否则多数票（均值概率打破平局）；`--class-map`
（默认 `0:LCA,1:RCA`）映射为 vessel。聚合逻辑已含单元自检
（`aggregate_rows`，合成 CSV 两分支 + 类映射解析）。

单序列手工推理：

```bash
python cls/CustomDataset/test.py --model MedViT_small \
    --checkpoint <ckpt> --data-path <case目录> --test-dir <序列目录名> \
    --output-dir <临时目录> --nb_classes 2 --no-labels --save-pred
```

## 参考 checkpoint

队列结果所用模型：`medvit_cls_exp` 下 0311_162826 运行的
`checkpoint_best.pth`（权重不入库，`*.pth` 已 gitignore）；该次训练日志存档于
[experiments/cls_0311_162826_train.log](experiments/cls_0311_162826_train.log)。
