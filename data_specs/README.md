# Expert-provided input specs

以下四类数据需由心内专家提供（论文 Results 中依赖人工判读的部分）。
按模板填入真实数据后，在仓库根目录运行：

```bash
python paper_scripts/run_clinical.py \
    --gt data_specs/gt_ctfc_template.csv \
    --grades data_specs/timi_grade_template.csv \
    --echo data_specs/echo_params_template.csv \
    --results paper_results/02_ctfc/ctfc_cohort.csv \
    --outdir paper_results/03_clinical
```

## 1. gt_ctfc_template.csv — 手工 cTFC（Bland-Altman / Fig 6 / 5.1）
| 列 | 说明 |
|---|---|
| case_id | 病例目录名，如 `1-patient01`（或纯数字前缀，匹配器会归一化） |
| sequence_id | 序列目录名，如 `1-IM000093` |
| branch | LAD / LCX / RCA（可省略=单支） |
| ctfc_manual_1/2/3 | 三位心内专家各自手工计数的 cTFC（30fps 校正后，LAD 已除 1.7）；参考值取三人均值 |

要求：test 队列（196 例设计）；专家对 DL-cTFC 结果设盲。

## 2. timi_grade_template.csv — TIMI 血流分级（5.2 分级表 / ROC / 阈值）
| 列 | 说明 |
|---|---|
| timi_grade | 0/1/2/3，三位专家共识 |
| split | train（用于 Youden 阈值）/ test（用于报告性能） |

## 3. echo_params_template.csv — 超声参数（5.3.1 增量价值 / Fig 7）
| 列 | 说明 |
|---|---|
| case_id | 病例目录名，如 `1-patient01`。**每病例一行**（病例级，无 sequence_id——echo 是患者级测量；run_clinical / make_fig7 按 case_id 病例级匹配，病例均值 cTFC 取 valid 且 [5,60] 窗口行） |
| ea_ratio | 二尖瓣 E/A（术前超聲） |
| eprime_a_ratio | 组织多普勒 e'/a' |
| group | A/B（论文仅在 group A test 病例做相关）。映射约定：A=CAD、B=Others；CMVD 单标 `C`，不进 A/B 统计 |

> ⚠️ 投稿要求：Fig 6 / Fig 7 的专家金标准与超声参数必须是**真实测量数据**，
> 不得使用模拟占位值。

## 4. likert_scores_template.csv — 分割质量 Likert 评分（定性）
case_id × territory(LAD/LCX/RCA) × reader(1/2/3)，score 1-5。
需 3 位心内专家对 test 集人工评分，无法由代码生成。
