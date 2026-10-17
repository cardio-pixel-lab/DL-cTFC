#!/usr/bin/env python3
"""
CTFC 可视化：逐序列 1×3 面板
  1) 区域充盈曲线（fill_p / fill_d / fill_e）+ F1/F2/Fmax 竖线 + 阈值横线
  2) Fmax 掩码上叠加近端区 P（蓝）/ 远端区 D（红）
  3) 逐像素首充帧 FO 热图（viridis，仅 E 内）
"""

import logging
import os
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


class CTFCVisualizer:
    def __init__(self, output_dir: str, logger: logging.Logger = None):
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger('CTFC.Visualizer')
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_sequence(self, case_id: str, sequence_id: str, analysis) -> bool:
        """绘制单个序列/分支。analysis.detail 缺失时跳过并返回 False。"""
        d = analysis.detail
        if d is None:
            self.logger.debug(f"跳过可视化（无 detail）: {case_id}/{sequence_id}")
            return False

        n = d['n_frames']
        frames = np.arange(n)
        fmax = analysis.max_frame_id
        is_degenerate = '单相充盈(退化)' in (analysis.notes or [])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # ---------------- 面板 1: 充盈曲线
        ax = axes[0]
        ax.plot(frames, d['fill_p'], color='tab:blue', lw=1.4, label='近端区 fill_p')
        ax.plot(frames, d['fill_d'], color='tab:red', lw=1.4, label='远端区 fill_d')
        ax.plot(frames, d['fill_e'], color='gray', lw=1.0, ls='--', label='整范围 fill_E')
        if analysis.f1 is not None:
            ax.axvline(analysis.f1, color='green', ls='-', lw=1.6,
                       label=f"F1={analysis.f1}")
        if analysis.f2 is not None:
            ax.axvline(analysis.f2, color='crimson', ls='-', lw=1.6,
                       label=f"F2={analysis.f2}")
        ax.axvline(fmax, color='black', ls=':', lw=1.2, label=f"Fmax={fmax}")
        ax.set_xlim(0, max(n - 1, 1))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('frame')
        ax.set_ylabel('fill ratio')
        title = f"{case_id} / {sequence_id} [{analysis.branch_type}]"
        if is_degenerate:
            title += '  (退化单相)'
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)

        # ---------------- 面板 2: Fmax 掩码 + P/D
        ax = axes[1]
        rgb = np.zeros((*d['fmax_mask'].shape, 3), dtype=np.float64)
        rgb[...] = (0.25, 0.25, 0.25)
        rgb[d['P'] > 0] = (0.0, 0.0, 1.0)
        rgb[d['D'] > 0] = (1.0, 0.0, 0.0)
        ax.imshow(rgb)
        ax.set_title(f"Fmax mask + P (blue, {analysis.proximal_px}px) / "
                     f"D (red, {analysis.distal_px}px)", fontsize=10)
        ax.axis('off')

        # ---------------- 面板 3: FO 热图
        ax = axes[2]
        fo_disp = np.where(d['extent'] > 0, d['fo'].astype(np.float64), np.nan)
        cmap = cm.viridis.copy()
        cmap.set_bad('white')
        im = ax.imshow(fo_disp, cmap=cmap, vmin=0, vmax=max(fmax, 1))
        ax.set_title(f"First-opacification frame FO (0..Fmax={fmax})", fontsize=10)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        safe = f"{case_id}__{sequence_id}_ctfc.png".replace('/', '_')
        out_path = os.path.join(self.output_dir, safe)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self.logger.debug(f"可视化已保存: {out_path}")
        return True
