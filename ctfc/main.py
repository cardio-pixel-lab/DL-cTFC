#!/usr/bin/env python3
"""
CTFC (corrected TIMI Frame Count) 自动分析系统 — CLI 入口

F1 = 近端"首全进入"帧，F2 = 空间远端到达帧（Gibson 1996, Circulation 93:879）。
自包含实现，核心算法见 ctfc_analyzer.py，方法概要见 README.md。
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import yaml

from ctfc_analyzer import (
    CTFCCalculator,
    CTFCDatasetProcessor,
    CTFCSequenceAnalysis,
    CTFCSequenceProcessor,
    CTFCDetector,
    FirstOpac,
    FrameAnalyzer,
    MaskProcessor,
    print_results,
    save_results,
)


def setup_logging(output_dir: str, log_level: str = 'INFO') -> logging.Logger:
    """设置日志系统（文件 DEBUG + 控制台 INFO，带时间戳日志文件）"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'ctfc_analysis_{timestamp}.log')

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    logger = logging.getLogger('CTFC')
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

    logger.info(f"日志文件: {log_file}")
    logger.info(f"日志级别: {log_level}")
    logger.info("=" * 60)

    return logger


class CTFCAnalysisConfig:
    """配置管理器（扁平化参数表，键与 ctfc_analyzer.py 读取的键一一对应）"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        if config_path is None:
            # 未显式指定时，优先用脚本同目录自带的 config.yaml
            bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
            if os.path.exists(bundled):
                config_path = bundled
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}

    def _get(self, section: str) -> Dict:
        v = self.config.get(section)
        return v if isinstance(v, dict) else {}

    def get_params(self) -> Dict:
        mp = self._get('mask_processing')
        fo = self._get('first_opac')
        lm = self._get('landmarks')
        dg = self._get('degenerate')
        f1 = self._get('f1')
        f2 = self._get('f2')
        cal = self._get('calculation')
        val = self._get('validation')
        ds = self._get('dataset')
        return {
            'mask_suffix': str(ds.get('mask_suffix', '_pred.png')),
            'border_width': int(mp.get('border_width', 30)),
            'min_component_area': int(mp.get('min_component_area', 50)),
            'hysteresis_frames': int(fo.get('hysteresis_frames', 2)),
            'seed_frac': float(lm.get('seed_frac', 0.02)),
            'proximal_frac': float(lm.get('proximal_frac', 0.10)),
            'distal_frac': float(lm.get('distal_frac', 0.10)),
            'max_region_frac': float(lm.get('max_region_frac', 0.40)),
            'min_region_pixels': int(lm.get('min_region_pixels', 30)),
            'extent_merge': str(lm.get('extent_merge', 'big')),
            'extent_min_comp_frac': float(lm.get('extent_min_comp_frac', 0.10)),
            'degenerate_span': int(dg.get('degenerate_span', 1)),
            'f1_percentile': float(f1.get('percentile', 0.15)),
            'f2_percentile': float(f2.get('percentile', 0.10)),
            'frame_rate_correction': float(cal.get('frame_rate_correction', 2.0)),
            'lad_correction': float(cal.get('lad_correction', 1.7)),
            'lad_correction_enabled': bool(cal.get('lad_correction_enabled', True)),
            'min_frames': int(val.get('min_frames', 10)),
            'min_fmax_pixels': int(val.get('min_fmax_pixels', 800)),
            'max_ctfc': float(val.get('max_ctfc', 200)),
        }

    def get_output_params(self) -> Dict:
        out = self._get('output')
        qc = self._get('qc')
        return {
            'format': out.get('format', 'csv'),
            'save_detailed_frames': bool(out.get('save_detailed_frames', True)),
            'generate_plots': bool(out.get('generate_plots', True)),
            'save_f1f2_frames': bool(out.get('save_f1f2_frames', True)),
            'save_qc_report': bool(qc.get('enabled', False)),
            'qc_sample_frac': float(qc.get('sample_frac', 1.0)),
        }

    def get_dataset_filter_params(self) -> Dict:
        flt = self._get('dataset_filter')
        return {
            'enabled': bool(flt.get('enabled', False)),
            'include_case_sequences': flt.get('include_case_sequences', []),
        }


class CTFCAnalyzer:
    """CTFC 分析主类（装配各组件 + 可视化 + 导出）"""

    def __init__(self, config: CTFCAnalysisConfig, output_dir: str, logger: logging.Logger = None):
        self.config = config
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger('CTFC')
        os.makedirs(output_dir, exist_ok=True)

        self.params = config.get_params()
        self.output_params = config.get_output_params()

        self.logger.info("初始化CTFC分析器...")
        self.logger.debug(f"输出目录: {output_dir}")
        self.logger.info(
            f"掩码处理: border_width={self.params['border_width']}, "
            f"min_component_area={self.params['min_component_area']}"
        )

        self.mask_processor = MaskProcessor(logger=self.logger)
        self.frame_analyzer = FrameAnalyzer(logger=self.logger)
        self.first_opac = FirstOpac(
            hysteresis_frames=self.params['hysteresis_frames'], logger=self.logger
        )
        self.detector = CTFCDetector(params=self.params, logger=self.logger)
        self.calculator = CTFCCalculator(params=self.params, logger=self.logger)

        self.sequence_processor = CTFCSequenceProcessor(
            mask_processor=self.mask_processor,
            frame_analyzer=self.frame_analyzer,
            first_opac=self.first_opac,
            detector=self.detector,
            calculator=self.calculator,
            params=self.params,
            logger=self.logger,
        )
        self.dataset_processor = CTFCDatasetProcessor(
            sequence_processor=self.sequence_processor, logger=self.logger
        )

        self.visualizer = None  # 惰性创建（避免无 matplotlib 环境导入失败）

    def analyze_dataset(self, root_dir: str, segmented_dirname='segmented',
                        cls_summary_csv=None,
                        include_case_sequences_override: Optional[List[Dict[str, Any]]] = None
                        ) -> Dict[str, Dict[str, CTFCSequenceAnalysis]]:
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"数据集目录不存在: {root_dir}")

        dataset_filter_params = self.config.get_dataset_filter_params()
        include_case_sequences = (
            include_case_sequences_override
            if include_case_sequences_override is not None
            else (
                dataset_filter_params['include_case_sequences']
                if dataset_filter_params['enabled'] else None
            )
        )

        self.logger.info("=" * 60)
        self.logger.info(f"开始分析数据集: {root_dir}")
        self.logger.info(f"分割目录名: {segmented_dirname}")
        self.logger.info(f"cls_summary: {cls_summary_csv}")
        if include_case_sequences is None:
            self.logger.info("筛选模式: 关闭（处理全部病例/序列）")
        else:
            seq_count = sum(len(item.get('sequence_ids', [])) for item in include_case_sequences)
            self.logger.info(f"筛选模式: 启用（病例数={len(include_case_sequences)}, 序列数={seq_count}）")
        self.logger.info("=" * 60)

        return self.dataset_processor.process_dataset(
            root_dir,
            segmented_dirname=segmented_dirname,
            cls_summary_csv=cls_summary_csv,
            include_case_sequences=include_case_sequences,
        )

    # ------------------------------------------------------------------ 可视化

    def generate_visualizations(self, results: Dict) -> None:
        if not self.output_params['generate_plots']:
            self.logger.info("generate_plots=False，跳过可视化")
            return
        try:
            from visualizer import CTFCVisualizer
        except ImportError as e:
            self.logger.warning(f"可视化模块不可用（{e}），跳过")
            return

        if self.visualizer is None:
            self.visualizer = CTFCVisualizer(
                output_dir=os.path.join(self.output_dir, 'visualizations'),
                logger=self.logger,
            )
        n_plotted = 0
        for case_id in sorted(results.keys()):
            for seq_id in sorted(results[case_id].keys()):
                analysis = results[case_id][seq_id]
                try:
                    if self.visualizer.plot_sequence(case_id, seq_id, analysis):
                        n_plotted += 1
                except Exception as e:
                    self.logger.warning(f"可视化失败 {case_id}/{seq_id}: {e}")
        self.logger.info(f"可视化完成: {n_plotted} 张")

    # ------------------------------------------------------------------ 导出

    def export_results(self, results: Dict) -> None:
        json_file = os.path.join(self.output_dir, 'ctfc_results.json')
        save_results(results, json_file)

        self._export_csv(results, os.path.join(self.output_dir, 'ctfc_results.csv'),
                         valid_only=False)
        self._export_csv(results, os.path.join(self.output_dir, 'ctfc_results_valid.csv'),
                         valid_only=True)

        if self.output_params['save_detailed_frames']:
            self._save_detailed(results)
        else:
            self.logger.info("save_detailed_frames=False，跳过明细保存")

        if self.output_params['save_f1f2_frames']:
            self._save_f1f2_frames(results)
        else:
            self.logger.info("save_f1f2_frames=False，跳过 F1/F2 帧保存")

        if self.output_params['save_qc_report']:
            self._generate_qc_report(results)
        else:
            self.logger.info("save_qc_report=False，跳过 QC 连拍图")

        self._generate_html_report(results)

    @staticmethod
    def _iter_rows(results: Dict, valid_only: bool = False):
        for case_id in sorted(results.keys()):
            for seq_id in sorted(results[case_id].keys()):
                a = results[case_id][seq_id]
                if valid_only and not a.is_valid:
                    continue
                yield case_id, seq_id, a

    def _export_csv(self, results: Dict, csv_file: str, valid_only: bool = False) -> None:
        rows = []
        for case_id, seq_id, a in self._iter_rows(results, valid_only):
            rows.append({
                '病例ID': case_id,
                '序列ID': seq_id,
                '分支类型': a.branch_type,
                '总帧数': a.n_frames,
                'Fmax': a.max_frame_id,
                '最大非零像素数': a.max_nonzero_count,
                't_p': a.t_p,
                't_d': a.t_d,
                '近端区像素': a.proximal_px,
                '远端区像素': a.distal_px,
                'F1': a.f1,
                'F2': a.f2,
                'CTFC(浮点)': round(a.ctfc, 2) if a.ctfc is not None else '',
                'CTFC(原始)': round(a.ctfc_raw, 2) if a.ctfc_raw is not None else '',
                'CTFC(取整)': a.ctfc_rounded if a.ctfc_rounded is not None else '',
                '有效性': '有效' if a.is_valid else '无效',
                '备注': '; '.join(a.notes),
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"CSV 已保存: {csv_file} ({len(df)} 行)")

    def _save_detailed(self, results: Dict) -> None:
        detail_dir = os.path.join(self.output_dir, 'detailed')
        os.makedirs(detail_dir, exist_ok=True)
        n_saved = 0
        for case_id, seq_id, a in self._iter_rows(results):
            if not a.detail:
                continue
            d = a.detail
            safe = re.sub(r'[^0-9A-Za-z_.-]', '_', f"{case_id}__{seq_id}")
            json_path = os.path.join(detail_dir, safe + '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'case_id': case_id,
                    'sequence_id': seq_id,
                    'branch_type': a.branch_type,
                    'n_frames': d['n_frames'],
                    'fmax': a.max_frame_id,
                    'f1': a.f1,
                    'f2': a.f2,
                    't_p': a.t_p,
                    't_d': a.t_d,
                    'f1_percentile': d['f1_percentile'],
                    'f2_percentile': d['f2_percentile'],
                    'fill_p': d['fill_p'].tolist(),
                    'fill_d': d['fill_d'].tolist(),
                    'fill_e': d['fill_e'].tolist(),
                }, f, ensure_ascii=False)
            npz_path = os.path.join(detail_dir, safe + '.npz')
            np.savez(
                npz_path,
                fmax_mask=d['fmax_mask'],
                extent=d['extent'],
                seed=d['seed'],
                P=d['P'],
                D=d['D'],
                fo=d['fo'],
            )
            n_saved += 1
        self.logger.info(f"明细已保存: {n_saved} 序列 -> {detail_dir}")

    def _save_f1f2_frames(self, results: Dict) -> None:
        """保存 F1/F2 对应帧的源帧 PNG（分支掩码 + P/D 区叠加），便于人工查验。"""
        frame_dir = os.path.join(self.output_dir, 'frames')
        os.makedirs(frame_dir, exist_ok=True)
        n_saved = 0
        for case_id, seq_id, a in self._iter_rows(results):
            if a.f1 is None or a.f2 is None or not a.segmented_dir:
                continue
            try:
                n_saved += save_branch_f1f2_frames(a, frame_dir, self.params, self.logger)
            except Exception as e:
                self.logger.warning(f"F1/F2 帧保存失败 {case_id}/{seq_id}: {e}")
        self.logger.info(f"F1/F2 帧已保存: {n_saved} 张 -> {frame_dir}")

    def _generate_qc_report(self, results: Dict) -> None:
        try:
            from qc_report import build_qc_report
        except ImportError as e:
            self.logger.warning(f"QC 模块不可用（{e}），跳过")
            return
        try:
            build_qc_report(results, self.output_dir, self.params, self.logger,
                            sample_frac=self.output_params['qc_sample_frac'])
        except Exception as e:
            self.logger.warning(f"QC 生成失败: {e}", exc_info=True)

    def _generate_html_report(self, results: Dict) -> None:
        all_rows = list(self._iter_rows(results))
        valid_rows = [r for r in all_rows if r[2].is_valid]
        n_cases = len(results)
        n_total = len(all_rows)
        n_valid = len(valid_rows)
        n_invalid = n_total - n_valid

        valid_ctfc = [a.ctfc for _, _, a in valid_rows if a.ctfc is not None]
        overall_mean = float(np.mean(valid_ctfc)) if valid_ctfc else None
        overall_sd = float(np.std(valid_ctfc)) if valid_ctfc else None

        branch_stats = {}
        for btype in ('LAD', 'LCX', 'RCA'):
            vals = [a.ctfc for _, _, a in valid_rows
                    if a.branch_type == btype and a.ctfc is not None]
            branch_stats[btype] = (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (None, None, 0)

        def fmt(v, nd=2):
            return f"{v:.{nd}f}" if v is not None else '-'

        def stat_cell(t):
            m, s, n = t
            return f"{fmt(m)} ± {fmt(s)} (n={n})" if m is not None else f"n=0"

        rows_html = []
        for case_id, seq_id, a in all_rows:
            ctfc_txt = f"{a.ctfc:.2f}" if a.ctfc is not None else '-'
            status = '<span class="ok">有效</span>' if a.is_valid else '<span class="bad">无效</span>'
            rows_html.append(
                "<tr>"
                f"<td>{case_id}</td><td>{seq_id}</td><td>{a.branch_type}</td>"
                f"<td>{a.n_frames}</td><td>{a.max_frame_id}</td>"
                f"<td>{a.f1 if a.f1 is not None else '-'}</td>"
                f"<td>{a.f2 if a.f2 is not None else '-'}</td>"
                f"<td>{ctfc_txt}</td>"
                f"<td>{a.ctfc_rounded if a.ctfc_rounded is not None else '-'}</td>"
                f"<td>{status}</td>"
                f"<td class='notes'>{a.notes and '; '.join(a.notes) or ''}</td>"
                "</tr>"
            )

        html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>CTFC 分析报告</title>
<style>
  body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 24px; color: #222; }}
  h1 {{ font-size: 22px; }}
  .cards {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 16px 0; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 18px; min-width: 140px; }}
  .card .num {{ font-size: 24px; font-weight: 600; }}
  .card .lbl {{ color: #666; font-size: 13px; }}
  .ok {{ color: #1a7f37; font-weight: 600; }}
  .bad {{ color: #c0392b; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
  th {{ background: #f5f5f5; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .notes {{ max-width: 420px; color: #555; }}
</style>
</head>
<body>
<h1>CTFC 分析报告</h1>
<p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<div class="cards">
  <div class="card"><div class="num">{n_cases}</div><div class="lbl">病例数</div></div>
  <div class="card"><div class="num">{n_total}</div><div class="lbl">分支总数</div></div>
  <div class="card"><div class="num ok">{n_valid}</div><div class="lbl">有效</div></div>
  <div class="card"><div class="num bad">{n_invalid}</div><div class="lbl">无效</div></div>
  <div class="card"><div class="num">{fmt(overall_mean)} ± {fmt(overall_sd)}</div><div class="lbl">CTFC 均值±SD（有效）</div></div>
  <div class="card"><div class="num">{stat_cell(branch_stats['LAD'])}</div><div class="lbl">LAD（已÷1.7）</div></div>
  <div class="card"><div class="num">{stat_cell(branch_stats['LCX'])}</div><div class="lbl">LCX</div></div>
  <div class="card"><div class="num">{stat_cell(branch_stats['RCA'])}</div><div class="lbl">RCA</div></div>
</div>
<table>
<thead>
<tr><th>病例ID</th><th>序列ID</th><th>分支</th><th>总帧数</th><th>Fmax</th>
<th>F1</th><th>F2</th><th>CTFC</th><th>CTFC(取整)</th><th>有效性</th><th>备注</th></tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</body>
</html>
"""
        html_file = os.path.join(self.output_dir, 'ctfc_report.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"HTML 报告已保存: {html_file}")


# ---------------------------------------------------------------------- 工具函数

def _overlay_region(img: np.ndarray, region: Optional[np.ndarray], color, alpha: float = 0.5) -> np.ndarray:
    """在 BGR 图像上叠加半透明填充 + 1px 轮廓（region 非零处）。"""
    if region is None or int(region.sum()) == 0:
        return img
    m = (region > 0).astype(np.uint8)
    out = img.copy()
    blended = (alpha * img.astype(np.float32) + (1 - alpha) * np.array(color, dtype=np.float32))
    out[m > 0] = blended[m > 0].astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 1)
    return out


def save_branch_f1f2_frames(a: CTFCSequenceAnalysis, out_dir: str,
                            params: Dict, logger: logging.Logger) -> int:
    """保存该分支 F1/F2 帧的源帧（_src.png）叠加图。

    叠加: 分支掩码(绿) / 近端区 P(青) / 远端区 D(红)，左上角标注帧号与区域充盈比。
    返回实际保存的张数（0–2）。
    """
    if a.f1 is None or a.f2 is None or not a.segmented_dir:
        return 0
    suffix = params.get('mask_suffix', '_pred.png')

    # 帧号 -> 文件主干（..._pred 之前的部分；_src 同源）
    stems = {}
    for fn in os.listdir(a.segmented_dir):
        if fn.endswith(suffix):
            m = re.search(r'f(\d+)$', fn[:len(fn) - len(suffix)])
            if m:
                stems[int(m.group(1))] = fn[:len(fn) - len(suffix)]
    if not stems:
        logger.warning(f"{a.case_id}/{a.sequence_id}: 分割目录中无 {suffix} 文件")
        return 0

    pix_val = {'LAD': 255, 'LCX': 128}.get(a.branch_type)  # RCA 为 None（全非零）
    P = a.detail.get('P') if a.detail else None
    D = a.detail.get('D') if a.detail else None
    fill_p = a.detail.get('fill_p') if a.detail else None
    fill_d = a.detail.get('fill_d') if a.detail else None

    safe = re.sub(r'[^0-9A-Za-z_.-]', '_', f"{a.case_id}__{a.sequence_id}")
    n_saved = 0
    for tag, fid in (('F1', a.f1), ('F2', a.f2)):
        stem = stems.get(fid)
        if stem is None:
            logger.debug(f"{a.case_id}/{a.sequence_id}: {tag} 帧 {fid} 无对应源文件，跳过")
            continue
        src = cv2.imread(os.path.join(a.segmented_dir, stem + '_src.png'))
        raw = MaskProcessor.load_mask(os.path.join(a.segmented_dir, stem + suffix))
        if src is None or raw is None:
            continue
        if raw.shape != src.shape[:2]:
            raw = cv2.resize(raw, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_NEAREST)
        branch = (raw == pix_val) if pix_val else (raw > 0)
        branch = (MaskProcessor.clean_mask(branch.astype(np.uint8),
                                           params['border_width'],
                                           params['min_component_area']) > 0)

        img = _overlay_region(src, branch, (0, 220, 0))
        img = _overlay_region(img, P, (255, 255, 0))
        img = _overlay_region(img, D, (0, 0, 255))

        txt = f"{a.branch_type} {tag} frame={fid}"
        if fill_p is not None and fid < len(fill_p) and fill_d is not None and fid < len(fill_d):
            txt += f"  fill_p={fill_p[fid]:.2f}  fill_d={fill_d[fid]:.2f}"
        for thick, color in ((4, (0, 0, 0)), (2, (255, 255, 255))):
            cv2.putText(img, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thick, cv2.LINE_AA)
        legend = "green=branch  cyan=P(proximal)  red=D(distal)"
        y0 = img.shape[0] - 10
        for thick, color in ((4, (0, 0, 0)), (2, (255, 255, 255))):
            cv2.putText(img, legend, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thick, cv2.LINE_AA)

        cv2.imwrite(os.path.join(out_dir, f"{safe}_{tag}.png"), img)
        n_saved += 1
    return n_saved


def parse_segmented_dirnames(v):
    """支持 "a,b" / "a b" / ["a", "b"] 输入。"""
    if v is None:
        return ["segmented"]
    if isinstance(v, (list, tuple)):
        parts = []
        for x in v:
            parts.extend(re.split(r"[,\s]+", str(x).strip()))
        out = [p for p in parts if p]
        return out or ["segmented"]
    out = [p for p in re.split(r"[,\s]+", str(v).strip()) if p]
    return out or ["segmented"]


def _normalize_include_case_sequences(raw_items: Any) -> List[Dict[str, Any]]:
    """将外部清单标准化为 include_case_sequences 结构。"""
    if not isinstance(raw_items, list):
        raise ValueError("清单格式错误: include_case_sequences 必须是列表")

    merged: Dict[str, set] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        case_id = str(item.get('case_id', '')).strip()
        if not case_id:
            continue
        seq_ids = item.get('sequence_ids', [])
        if isinstance(seq_ids, str):
            seq_ids = [x.strip() for x in re.split(r"[,\s]+", seq_ids) if x.strip()]
        elif isinstance(seq_ids, (list, tuple)):
            seq_ids = [str(x).strip() for x in seq_ids if str(x).strip()]
        else:
            seq_ids = []
        merged.setdefault(case_id, set()).update(seq_ids)

    return [
        {'case_id': case_id, 'sequence_ids': sorted(list(seq_set))}
        for case_id, seq_set in sorted(merged.items())
    ]


def load_filter_file(filter_file: str) -> List[Dict[str, Any]]:
    """加载外部筛选清单（CSV 需 case_id+sequence_id 列；YAML 顶层 dict 或 list）。"""
    path = filter_file
    if not os.path.isfile(path):
        raise FileNotFoundError(f"筛选清单不存在: {filter_file}")

    ext = os.path.splitext(path)[1].lower()
    if ext in ('.yaml', '.yml'):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            raw_items = data.get('include_case_sequences', [])
        elif isinstance(data, list):
            raw_items = data
        else:
            raise ValueError("YAML清单格式错误: 顶层必须是字典或列表")
        return _normalize_include_case_sequences(raw_items)

    if ext == '.csv':
        raw_items = []
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV清单格式错误: 缺少表头")
            headers = {h.strip() for h in reader.fieldnames if h}
            if 'case_id' not in headers or 'sequence_id' not in headers:
                raise ValueError("CSV清单格式错误: 必须包含 case_id 和 sequence_id 两列")
            for row in reader:
                case_id = str(row.get('case_id', '')).strip()
                sequence_id = str(row.get('sequence_id', '')).strip()
                if not case_id or not sequence_id:
                    continue
                raw_items.append({'case_id': case_id, 'sequence_ids': [sequence_id]})
        return _normalize_include_case_sequences(raw_items)

    raise ValueError("仅支持 .csv / .yaml / .yml 清单文件")


def main():
    parser = argparse.ArgumentParser(
        description='CTFC (corrected TIMI Frame Count) 自动分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py -r /path/to/cohort -o ./results \\
      --segmented-dirname "segmented3c2,segmented" \\
      --cls-summary-csv /path/to/cls_summary.csv
  python main.py -r /path/to/cohort -o ./results -c config.yaml --filter-file filters.csv
  python main.py -r /path/to/cohort -o ./results --no-visualize
        """
    )
    parser.add_argument('--root-dir', '-r', required=True, type=str,
                        help='数据集（队列）根目录路径')
    parser.add_argument('--output-dir', '-o', default='./ctfc_results', type=str,
                        help='输出目录 (默认: ./ctfc_results)')
    parser.add_argument('--config', '-c', type=str,
                        help='配置文件路径 (可选)')
    parser.add_argument('--segmented-dirname', type=str, default='segmented3c2,segmented',
                        help='分割掩码子目录名候选，逗号分隔 (默认: segmented3c2,segmented)')
    parser.add_argument('--cls-summary-csv', type=str, default=None,
                        help='分类结果CSV路径（含 LCA/RCA 列，缺失时默认按 RCA 处理）')
    parser.add_argument('--filter-file', type=str, default=None,
                        help='外部筛选清单路径（csv/yaml），优先级高于config中的dataset_filter')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志级别 (默认: INFO)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细输出模式 (等同于 --log-level DEBUG)')
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument('--visualize', dest='visualize', action='store_true', default=None,
                           help='强制生成可视化图表')
    viz_group.add_argument('--no-visualize', dest='visualize', action='store_false',
                           help='强制跳过可视化图表')
    qc_group = parser.add_mutually_exclusive_group()
    qc_group.add_argument('--qc', dest='qc', action='store_true', default=None,
                          help='强制生成 QC 连拍图与人工核对页面')
    qc_group.add_argument('--no-qc', dest='qc', action='store_false',
                          help='强制跳过 QC 连拍图与人工核对页面')

    args = parser.parse_args()
    args.segmented_dirname = parse_segmented_dirnames(args.segmented_dirname)

    if not os.path.exists(args.root_dir):
        print(f"错误: 数据集目录不存在: {args.root_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    log_level = 'DEBUG' if args.verbose else args.log_level
    logger = setup_logging(args.output_dir, log_level)

    logger.info("CTFC分析系统启动")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"命令行参数: {vars(args)}")

    try:
        config = CTFCAnalysisConfig(config_path=args.config)

        include_case_sequences_override = None
        if args.filter_file:
            logger.info(f"加载外部筛选清单: {args.filter_file}")
            include_case_sequences_override = load_filter_file(args.filter_file)
            total_seq = sum(len(item.get('sequence_ids', [])) for item in include_case_sequences_override)
            logger.info(
                f"外部筛选清单加载成功: 病例数={len(include_case_sequences_override)}, "
                f"序列数={total_seq}"
            )

        analyzer = CTFCAnalyzer(config=config, output_dir=args.output_dir, logger=logger)
        if args.qc is not None:
            analyzer.output_params['save_qc_report'] = args.qc

        results = analyzer.analyze_dataset(
            args.root_dir,
            segmented_dirname=args.segmented_dirname,
            cls_summary_csv=args.cls_summary_csv,
            include_case_sequences_override=include_case_sequences_override,
        )

        print_results(results)

        do_viz = (
            analyzer.output_params['generate_plots']
            if args.visualize is None else args.visualize
        )
        if do_viz:
            analyzer.generate_visualizations(results)
        else:
            logger.info("跳过可视化")

        analyzer.export_results(results)

        logger.info("=" * 60)
        logger.info("分析任务完成!")
        logger.info(f"结果已保存到: {args.output_dir}")
        logger.info("=" * 60)

        print(f"\n✅ 分析完成! 结果已保存在: {args.output_dir}")
        print(f"📄 日志文件: {os.path.join(args.output_dir, 'logs')}/")

    except Exception as e:
        logger.error(f"分析失败: {str(e)}", exc_info=True)
        print(f"\n❌ 错误: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
