# seg/ — 冠脉分割网络（medformer, 2D）

逐帧 CAG 血管分割模块，为 `ctfc/` 管线提供 `*_pred.png` 掩码
（LCA 序列：0/128/255 = 背景/LCX/LAD；RCA 序列：0/255）。

代码基于开源框架 CBIM-Medical-Image-Segmentation（Apache-2.0，见
[LICENSE](LICENSE) 与 [upstream_README.md](upstream_README.md)），本模块只保留
冠脉分割用到的子集（`model/`、`training/`、`metric/`、入口脚本与 `config/`）。

## 两级分割链路（队列掩码的实际来源）

1. **二值血管分割**（`config/cag2/medformer_v6_2d.yaml`，数据 `cag_seg_v6`，
   0/255）→ 产出各序列 `segmented/` 二值掩码，作为下一级的先验输入；
2. **3 类分割**（`config/cag3/medformer_2d.yaml`，数据 `cag_seg_3class`，
   `classes: 3`、`in_chan: 2` 即图像+先验双通道）→ LCA 序列产出
   `segmented3c2/` 三类掩码（128=LCX / 255=LAD）。

## 环境

独立于 `ctfc/` 的 PyTorch 栈：

```bash
pip install -r seg/requirements.txt
```

## 训练

```bash
cd seg
# 3 类（队列 LCA 掩码模型）
python train_3cv2.py --dataset cag3 --model medformer --dimension 2d \
    --unique_name cag3_medformer --gpu 0
# 二值（先验模型）
python train.py --dataset cag2 --model medformer --dimension 2d \
    --unique_name cag2_medformer --gpu 0
```

配置按 `--dataset/--model/--dimension` 自动解析为
`config/<dataset>/<model>_<dimension>.yaml`（即 `config/cag3/medformer_2d.yaml`）；
checkpoint 与日志写 `exp/`、`log/`（均不入库）。

## 推理（生成序列掩码）

对单个序列目录（含 `*_src.png` 与先验 `segmented/`）：

```bash
cd seg
python test_3cv2.py --model medformer --dimension 2d --dataset cag3 \
    --unique_name dltfc_infer --gpu 0 \
    --load <exp_dir>/fold_0_best.pth --save \
    --save_path <seq_dir>/segmented3c2 \
    --test_root <seq_dir> \
    --test_without_calc --test_without_log \
    --prior_pattern "{data_root}/segmented/{stem}_pred{ext}"
```

`--prior_pattern` 中的 `{data_root}`/`{stem}`/`{ext}` 由脚本展开为每帧先验掩码路径。

## 评测

- `../seg` 侧逐帧 Dice（对人工 GT）：见根 README 的 `eval_seg_dice.py`
  （已随模块移至本目录）。
- 论文 Fig 4 / HD95：`../paper_scripts/make_fig4.py`、`seg_hd95.py`。
