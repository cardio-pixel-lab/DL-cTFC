"""
CTFC (corrected TIMI Frame Count) 自动化分析系统

参考 Gibson et al., Circulation 1996;93:879，逐分支算法（详见 方法描述.md / config.yaml）：
1. 逐像素首充帧 FO（窗口 [0, Fmax]，2 帧迟滞去单帧闪烁；残留 sentinel 钳到 Fmax）
2. 种子区 S = 最早充 seed_frac(0.02) 像素的最大连通分量（空间 ostium 锚点）
3. 标志区（均以 S 为源，算 E 内测地距离，距离高原整段并入）：
   P = 离 S 测地距离最近的 proximal_frac(0.10) 像素；D = 最远的 distal_frac(0.10)
4. 逐帧区域充盈 fill_p/fill_d/fill_E（仅用于可视化/QC，不参与 F1/F2 检测）
5. F1 = P 区 FO 的 f1_percentile(0.80) 分位（近端基本充满）；
   F2 = D 区 FO 的 f2_percentile(0.10) 分位（远端开始到达）。单调、鲁棒、无兜底
6. 退化单相充盈（t_d - t_p <= degenerate_span）：仅记 note，F1/F2 仍走上条 FO 分位
7. CTFC = (F2-F1)×frame_rate_correction；LAD ÷ lad_correction；浮点 + 取整报告
"""

import os
import re
import csv
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FrameAnalysis:
    """单帧分析结果"""
    frame_id: int
    nonzero_count: int
    percent_of_max: float


@dataclass
class CTFCSequenceAnalysis:
    """序列/分支 CTFC 分析结果"""
    case_id: str
    sequence_id: str
    branch_type: str = 'RCA'          # LAD / LCX / RCA
    n_frames: int = 0
    max_frame_id: int = -1            # Fmax（全序列全局最大充盈帧，含再循环）
    max_nonzero_count: int = 0
    f1: Optional[int] = None
    f2: Optional[int] = None
    t_p: Optional[int] = None
    t_d: Optional[int] = None
    seed_px: int = 0
    proximal_px: int = 0
    distal_px: int = 0
    ctfc: Optional[float] = None          # 全精度浮点（LAD 已 ÷1.7）
    ctfc_raw: Optional[float] = None      # 校正前（仅 ×frame_rate_correction）
    ctfc_rounded: Optional[int] = None    # 取整伴随列
    is_valid: bool = True
    notes: List[str] = field(default_factory=list)
    # 该序列实际使用的分割目录（segmented3c2/ 或 segmented/），用于回读 _src.png 源帧
    segmented_dir: Optional[str] = None
    # 中间产物（可视化/明细保存用，不进入 JSON/CSV 导出）
    detail: Optional[dict] = None


class MaskProcessor:
    """掩码处理和清理"""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger('CTFC.MaskProcessor')

    @staticmethod
    def load_mask(mask_path: str, target_dtype=np.uint8, logger: logging.Logger = None) -> Optional[np.ndarray]:
        """加载掩码文件；失败返回 None"""
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                if logger:
                    logger.warning(f"无法加载掩码文件: {mask_path}")
                return None
            return mask.astype(target_dtype)
        except Exception as e:
            if logger:
                logger.error(f"加载掩码文件失败 {mask_path}: {e}")
            return None

    @staticmethod
    def remove_edge_artifacts(mask: np.ndarray, border_width: int = 15) -> np.ndarray:
        """去除图像边框造成的错误，将边框部分置零"""
        mask_cleaned = mask.copy()
        h, w = mask.shape
        mask_cleaned[:border_width, :] = 0
        mask_cleaned[-border_width:, :] = 0
        mask_cleaned[:, :border_width] = 0
        mask_cleaned[:, -border_width:] = 0
        return mask_cleaned

    @staticmethod
    def remove_small_components(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
        """去除面积小于阈值的连通分量"""
        _, binary = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)
        num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))
        output = np.zeros_like(mask)
        for label_id in range(1, num_labels):  # 跳过背景(0)
            component_mask = (labels == label_id)
            if np.sum(component_mask) >= min_area:
                output[component_mask] = mask[component_mask]
        return output

    @staticmethod
    def clean_mask(mask: np.ndarray, border_width: int = 15,
                   min_component_area: int = 20) -> np.ndarray:
        """完整清理流程：去边框 + 去小连通分量"""
        mask = MaskProcessor.remove_edge_artifacts(mask, border_width)
        mask = MaskProcessor.remove_small_components(mask, min_component_area)
        return mask


class FrameAnalyzer:
    """帧级别分析"""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger('CTFC.FrameAnalyzer')

    @staticmethod
    def count_nonzero_pixels(mask: np.ndarray) -> int:
        return np.count_nonzero(mask)

    def analyze_all_frames(self, frame_masks: Dict[int, np.ndarray]) -> Tuple[Dict[int, FrameAnalysis], int, int]:
        """分析所有帧，返回 (帧分析字典, 最大充盈帧ID, 最大非零像素数)"""
        self.logger.debug(f"开始分析 {len(frame_masks)} 帧")
        analyses = {}
        max_count = 0
        max_frame_id = -1

        for frame_id, mask in frame_masks.items():
            nonzero_count = FrameAnalyzer.count_nonzero_pixels(mask)
            analyses[frame_id] = FrameAnalysis(
                frame_id=frame_id,
                nonzero_count=nonzero_count,
                percent_of_max=0.0
            )
            if nonzero_count > max_count:
                max_count = nonzero_count
                max_frame_id = frame_id

        if max_count > 0:
            for analysis in analyses.values():
                analysis.percent_of_max = 100.0 * analysis.nonzero_count / max_count

        self.logger.info(f"帧分析完成: 最大充盈帧={max_frame_id}, 最大像素数={max_count}")
        return analyses, max_frame_id, max_count


def largest_connected_component(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """返回 (最大连通分量掩码 uint8, 被丢弃的像素数)"""
    m8 = (mask > 0).astype(np.uint8)
    total = int(m8.sum())
    if total == 0:
        return m8, 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n <= 1:
        return m8, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    out = (labels == best).astype(np.uint8)
    return out, total - int(areas[best - 1])


def select_extent(mask: np.ndarray, mode: str = 'largest', min_comp_frac: float = 0.10
                  ) -> Tuple[np.ndarray, int, int]:
    """从 Fmax 掩码计算范围 E。

    mode:
      largest — 仅保留最大连通分量（原行为）
      big     — 最大分量 + 面积 ≥ min_comp_frac × 最大分量面积 的分量
                （覆盖近端段被洗脱/分割间隙截断的情况；全队列实测显示真血管段
                 与噪声分量的间距分布高度重叠，纯距离阈值不可用，面积比可分）
      all     — 合并所有分量（全掩码）
    返回 (extent uint8, 合并的分量数, 排除的像素数)。
    """
    m8 = (mask > 0).astype(np.uint8)
    total = int(m8.sum())
    if total == 0:
        return m8, 0, 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n <= 1:
        return m8, 0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    best_area = int(areas[best - 1])
    if mode == 'all':
        return m8, n - 1, 0
    if mode == 'big' and min_comp_frac > 0:
        keep = [i for i in range(1, n)
                if i == best or int(stats[i, cv2.CC_STAT_AREA]) >= min_comp_frac * best_area]
        out = np.isin(labels, keep).astype(np.uint8)
        return out, len(keep) - 1, total - int(out.sum())
    out = (labels == best).astype(np.uint8)
    return out, 0, total - best_area


def geodesic_distance_in_extent(extent: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """限定在 extent 内的多源 8 连通 BFS 测地距离（种子距离为 0，未达成为 -1）

    extent 可为多分量（extent_merge=big/all 合并后血管仍被间隙断开）：分量间以
    "跳跃边"连接（代价=两分量像素间最小欧氏距离），Dijkstra 求每个分量的最短路链，
    再在各分量内 BFS（入口=该分量上距前一分量最近的一排像素）；单分量时等价于
    直接 BFS。
    注意：分量内距离必须用 BFS（欧氏 distanceTransform 会抄近道）。
    """
    h, w = extent.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    if int((seed > 0).sum()) == 0:
        return dist
    n, labels = cv2.connectedComponents((extent > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return dist
    comps = list(range(1, n))
    comp_bool = {c: (labels == c) for c in comps}
    seed_comps = [int(c) for c in np.unique(labels[seed > 0]) if c >= 1]
    # dt[b]: 每个像素到分量 b 最近像素的欧氏距离（在 ~b 上算，零像素处为 0）
    dt = {c: cv2.distanceTransform((~comp_bool[c]).astype(np.uint8), cv2.DIST_L2, 5)
          for c in comps}

    def min_dist(a: int, b: int) -> float:
        return float(dt[b][comp_bool[a]].min())

    g = {c: (0.0 if c in seed_comps else np.inf) for c in comps}
    pred: Dict[int, int] = {}
    done = set()
    for _ in range(len(comps)):
        c = min((c for c in comps if c not in done), key=lambda c: g[c])
        if g[c] == np.inf:
            break
        done.add(c)
        for c2 in comps:
            if c2 in done or c2 == c:
                continue
            wgt = min_dist(c, c2)
            if g[c] + wgt < g[c2]:
                g[c2] = g[c] + wgt
                pred[c2] = c

    for c in comps:
        if g[c] == np.inf:
            continue
        if c in seed_comps:
            entry = (seed > 0) & comp_bool[c]
        else:
            dt_p = dt[pred[c]]
            dmin = float(dt_p[comp_bool[c]].min())
            entry = comp_bool[c] & (dt_p <= dmin + 1e-6)
        cm = comp_bool[c]
        d_c = np.full((h, w), -1, dtype=np.int32)
        d_c[entry] = 0
        q = deque(zip(*np.nonzero(entry)))
        while q:
            y, x = q.popleft()
            nd = d_c[y, x] + 1
            for dy in (-1, 0, 1):
                y0 = y + dy
                if y0 < 0 or y0 >= h:
                    continue
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    x0 = x + dx
                    if x0 < 0 or x0 >= w:
                        continue
                    if cm[y0, x0] and d_c[y0, x0] == -1:
                        d_c[y0, x0] = nd
                        q.append((y0, x0))
        dist[cm] = int(round(g[c])) + d_c[cm]
    return dist


class FirstOpac:
    """Step 1: 逐像素首充帧 FO（窗口 [0, Fmax]，对再循环鲁棒）"""

    def __init__(self, hysteresis_frames: int = 0, logger: logging.Logger = None):
        self.hysteresis_frames = max(0, int(hysteresis_frames))
        self.logger = logger or logging.getLogger('CTFC.FirstOpac')

    def compute(self, seq: List[np.ndarray], fmax: int, extent: np.ndarray) -> np.ndarray:
        """seq: 按帧号 0.. 排序的掩码列表；extent: E（uint8，Fmax 掩码范围，合并策略见 select_extent）

        返回 int16 FO 图：每个 E 像素的首充帧；未置位像素（sentinel）在循环后钳到 Fmax。
        """
        h, w = extent.shape
        sentinel = fmax + 1
        fo = np.full((h, w), sentinel, dtype=np.int16)
        unassigned = extent > 0
        rem = int(unassigned.sum())
        n = len(seq)
        hyst = self.hysteresis_frames

        for f in range(fmax + 1):
            if rem == 0:
                break
            ok = seq[f] != 0
            if hyst >= 2:
                if f + hyst - 1 < n:
                    for g in range(1, hyst):
                        ok = ok & (seq[f + g] != 0)
                else:
                    continue  # 序列末尾无法验证 hysteresis
            new = unassigned & ok
            if not new.any():
                continue
            fo[new] = f
            unassigned &= ~new
            rem -= int(new.sum())
            if rem <= 0:
                break

        # 残留 sentinel（Fmax 单帧闪现未过 hysteresis）钳到 Fmax，保证 FO[E] ≤ Fmax
        if rem > 0:
            fo[unassigned] = fmax

        return fo


class CTFCDetector:
    """Steps 2-5: 标志区（P 时序近端 / D 空间远端）+ 逐帧充盈 + F1/F2 检测"""

    def __init__(self, params: Dict, logger: logging.Logger = None):
        self.params = params
        self.logger = logger or logging.getLogger('CTFC.CTFCDetector')

    @staticmethod
    def _fo_quantile(fo: np.ndarray, mask: np.ndarray, frac: float) -> Optional[int]:
        """高原感知分位：mask 内最小的 FO 值 v 使累计像素占比 ≥ frac（高原整段并入）"""
        v = fo[mask > 0]
        if v.size == 0:
            return None
        uniq, cnts = np.unique(v, return_counts=True)  # 升序
        cum = np.cumsum(cnts)
        idx = int(np.searchsorted(cum, frac * cum[-1], side='left'))
        if idx >= len(uniq):
            idx = len(uniq) - 1
        return int(uniq[idx])

    @staticmethod
    def _proximal_cut(dist: np.ndarray, extent: np.ndarray, frac: float) -> int:
        """近端距离切点 d_p*：使 #{dist ≤ d_p*} ≥ frac·n 的最小距离值（高原整段并入）。

        仅统计可达像素（dist ≥ 0）；测地距离未达成的像素（-1）不参与。
        """
        dv = dist[(extent > 0) & (dist >= 0)]
        if dv.size == 0:
            return 0
        uniq, cnts = np.unique(dv, return_counts=True)  # 升序
        target = frac * dv.size
        cum = 0.0
        for val, c in zip(uniq, cnts):
            cum += c
            if cum >= target:
                return int(val)
        return int(uniq[-1])

    @staticmethod
    def _distal_cut(dist: np.ndarray, extent: np.ndarray, frac: float) -> int:
        """远端距离切点 d*：使 #{dist ≥ d*} ≥ frac·n 的最小距离值（高原整段并入）。

        仅统计可达像素（dist ≥ 0）。
        """
        dv = dist[(extent > 0) & (dist >= 0)]
        if dv.size == 0:
            return 0
        uniq, cnts = np.unique(dv, return_counts=True)  # 升序
        target = frac * dv.size
        cum = 0.0
        for val, c in zip(uniq[::-1], cnts[::-1]):
            cum += c
            if cum >= target:
                return int(val)
        return int(uniq[0])

    def detect(self, seq: List[np.ndarray], fmax: int, extent: np.ndarray,
               fo: np.ndarray) -> Dict:
        p = self.params
        notes: List[str] = []
        n_ext = int(extent.sum())

        # Step 2a: 种子 S = 最早充盈 seed_frac 像素的最大连通分量（ostium 锚点，空间）
        q_seed = self._fo_quantile(fo, extent, p['seed_frac'])
        S = ((extent > 0) & (fo <= q_seed)).astype(np.uint8)
        S, _ = largest_connected_component(S)
        s_px = int(S.sum())
        if s_px == 0:  # 防御（extent 非空时 seed_frac 分位恒非空，理论不触发）
            S = (extent > 0).astype(np.uint8)
            s_px = int(S.sum())
            notes.append('种子区为空，回退为 E')

        # Step 2b: 空间标志区（近端 P / 远端 D，均以 S 为种子做测地距离）
        dist = geodesic_distance_in_extent(extent, S)
        d_p_star = self._proximal_cut(dist, extent, p['proximal_frac'])
        d_d_star = self._distal_cut(dist, extent, p['distal_frac'])
        P = ((extent > 0) & (dist >= 0) & (dist <= d_p_star)).astype(np.uint8)
        D = ((extent > 0) & (dist >= 0) & (dist >= d_d_star)).astype(np.uint8)
        p_px = int(P.sum())
        d_px = int(D.sum())

        # t_p / t_d（FO 分位，仅用于退化 note）
        t_p = self._fo_quantile(fo, extent, p['proximal_frac'])
        t_d = self._fo_quantile(fo, extent, 1.0 - p['distal_frac'])
        degenerate = (t_p is not None and t_d is not None) and (t_d - t_p) <= p['degenerate_span']

        if n_ext > 0:
            if p_px / n_ext > p['max_region_frac']:
                notes.append(f"近端区占比 {p_px / n_ext:.2f} > {p['max_region_frac']}")
            if p_px < p['min_region_pixels']:
                notes.append(f"近端区过小: {p_px} < {p['min_region_pixels']}")
            if d_px / n_ext > p['max_region_frac']:
                notes.append(f"远端区占比 {d_px / n_ext:.2f} > {p['max_region_frac']}")
            if d_px < p['min_region_pixels']:
                notes.append(f"远端区过小: {d_px} < {p['min_region_pixels']}")

        # Step 3: 逐帧区域充盈（保留用于可视化/QC，不再用于 F1/F2 检测）
        fill_p = np.array([np.count_nonzero(m & P) for m in seq], dtype=np.float64) / max(p_px, 1)
        fill_d = np.array([np.count_nonzero(m & D) for m in seq], dtype=np.float64) / max(d_px, 1)
        fill_e = np.array([np.count_nonzero(m & extent) for m in seq], dtype=np.float64) / max(n_ext, 1)

        # Step 4: F1 = P 区 FO 高分位（近端基本充满）；F2 = D 区 FO 低分位（远端开始到达）。
        # 单调、鲁棒、无兜底；退化单相充盈不再需要单独路径。
        f1 = self._fo_quantile(fo, P, p['f1_percentile'])
        f2 = self._fo_quantile(fo, D, p['f2_percentile'])

        self.logger.debug(
            f"检测: seed={s_px}px, t_p={t_p}, t_d={t_d}, degenerate={degenerate}, "
            f"P={p_px}px, D={d_px}px, F1={f1}, F2={f2}"
        )

        return {
            't_p': t_p, 't_d': t_d, 'degenerate': degenerate,
            'seed': S, 'P': P, 'D': D,
            'seed_px': s_px, 'proximal_px': p_px, 'distal_px': d_px,
            'fill_p': fill_p, 'fill_d': fill_d, 'fill_e': fill_e,
            'f1': f1, 'f2': f2,
            'notes': notes,
        }


class CTFCCalculator:
    """Step 7 + 校验: CTFC 计算与硬无效判定"""

    def __init__(self, params: Dict, logger: logging.Logger = None):
        self.params = params
        self.logger = logger or logging.getLogger('CTFC.CTFCCalculator')

    def calculate(self, f1: Optional[int], f2: Optional[int], branch_type: str,
                  p: Dict) -> Tuple[Optional[float], Optional[float], Optional[int], bool, List[str]]:
        """返回 (ctfc, ctfc_raw, ctfc_rounded, is_valid, notes)"""
        notes: List[str] = []
        if f1 is None or f2 is None:
            if f1 is None:
                notes.append('未能检测到F1')
            if f2 is None:
                notes.append('未能检测到F2')
            return None, None, None, False, notes

        if f2 < f1:
            return None, None, None, False, [f'F2<F1 ({f2}<{f1}): 逆流/伪影，无效']

        ctfc_raw = (f2 - f1) * p['frame_rate_correction']
        if branch_type == 'LAD' and p['lad_correction_enabled']:
            ctfc = ctfc_raw / p['lad_correction']
        else:
            ctfc = ctfc_raw

        if ctfc > p['max_ctfc']:
            notes.append(f"CTFC过大: {ctfc:.2f} > {p['max_ctfc']}")

        self.logger.info(
            f"CTFC计算: branch={branch_type}, F1={f1}, F2={f2}, "
            f"raw={ctfc_raw:.2f}, ctfc={ctfc:.2f}"
        )
        return ctfc, ctfc_raw, int(round(ctfc)), len(notes) == 0, notes


class CTFCSequenceProcessor:
    """序列级别处理"""

    def __init__(self, mask_processor: MaskProcessor = None,
                 frame_analyzer: FrameAnalyzer = None,
                 first_opac: FirstOpac = None,
                 detector: CTFCDetector = None,
                 calculator: CTFCCalculator = None,
                 params: Optional[Dict] = None,
                 logger: logging.Logger = None):
        self.logger = logger or logging.getLogger('CTFC.SequenceProcessor')
        self.mask_processor = mask_processor or MaskProcessor(logger=self.logger)
        self.frame_analyzer = frame_analyzer or FrameAnalyzer(logger=self.logger)
        self.first_opac = first_opac
        self.detector = detector
        self.calculator = calculator
        self.params = params or {}

    def _load_frame_masks_from_dir(self, segmented_dir: str) -> Dict[int, np.ndarray]:
        """仅从目录读取原始掩码（不做清理）。mask 后缀可配置，默认 _pred.png。"""
        frame_masks: Dict[int, np.ndarray] = {}
        if not os.path.isdir(segmented_dir):
            return frame_masks

        suffix = self.params.get('mask_suffix', '_pred.png')
        for mask_file in sorted(os.listdir(segmented_dir)):
            if not mask_file.endswith(suffix):
                continue
            try:
                basename = mask_file[:len(mask_file) - len(suffix)]
                parts = basename.split('-')
                frame_part = parts[-1]  # f{frame_id}
                if not frame_part.startswith('f'):
                    continue
                frame_id = int(frame_part[1:])
                mask = self.mask_processor.load_mask(os.path.join(segmented_dir, mask_file))
                if mask is not None:
                    frame_masks[frame_id] = mask
            except Exception as e:
                self.logger.warning(f"错误解析文件 {mask_file}: {e}")
        return frame_masks

    def process_sequence(self, case_id: str, sequence_id: str, segmented_dir: str,
                         branch_type: str = 'RCA',
                         extra_notes: Optional[List[str]] = None) -> CTFCSequenceAnalysis:
        raw = self._load_frame_masks_from_dir(segmented_dir)
        a = self.process_sequence_from_masks(case_id, sequence_id, raw,
                                             branch_type=branch_type,
                                             extra_notes=extra_notes)
        a.segmented_dir = segmented_dir
        return a

    def process_sequence_from_masks(self, case_id: str, sequence_id: str,
                                    frame_masks: Dict[int, np.ndarray],
                                    branch_type: str = 'RCA',
                                    extra_notes: Optional[List[str]] = None) -> CTFCSequenceAnalysis:
        p = self.params
        notes = list(extra_notes or [])
        a = CTFCSequenceAnalysis(case_id=case_id, sequence_id=sequence_id, branch_type=branch_type)

        # 硬守卫：无掩码
        if not frame_masks:
            a.is_valid = False
            a.notes = notes + ['无掩码']
            return a

        # 清理所有帧（一致性是 F1/F2 存在性保证的前提）
        cleaned = {
            fid: MaskProcessor.clean_mask(m, p['border_width'], p['min_component_area'])
            for fid, m in frame_masks.items()
        }

        frame_analyses, fmax, max_count = self.frame_analyzer.analyze_all_frames(cleaned)
        a.n_frames = len(cleaned)
        a.max_frame_id = fmax
        a.max_nonzero_count = max_count

        # 硬守卫：空分支 / 帧数 / Fmax 像素数
        if max_count == 0:
            a.is_valid = False
            a.notes = notes + ['空分支掩码']
            return a
        if a.n_frames < p['min_frames']:
            a.is_valid = False
            a.notes = notes + [f"帧数过少: {a.n_frames} < {p['min_frames']}"]
            return a
        if max_count < p['min_fmax_pixels']:
            a.is_valid = False
            a.notes = notes + [f"Fmax像素数过小: {max_count} < {p['min_fmax_pixels']}"]
            return a

        # 帧序列（0 基连续；缺帧以零掩码填补并记 note）
        max_fid = max(cleaned.keys())
        if min(cleaned.keys()) != 0:
            notes.append(f"帧号未从0开始 (min={min(cleaned.keys())})")
        missing = [f for f in range(max_fid + 1) if f not in cleaned]
        if missing:
            notes.append(f"缺失帧(零填补): {len(missing)} 帧")
        zero = np.zeros_like(next(iter(cleaned.values())))
        seq = [cleaned.get(f, zero) for f in range(max_fid + 1)]
        n = len(seq)

        # E = M[Fmax] 的范围（合并策略由 extent_merge 决定，见 select_extent）
        extent, n_merged, n_dropped = select_extent(
            cleaned[fmax] > 0, p.get('extent_merge', 'big'), p.get('extent_min_comp_frac', 0.10))
        if n_merged > 0 or n_dropped > 0:
            if p.get('extent_merge') == 'all':
                notes.append(f"多分量范围(合并所有分量, {n_merged} 个)")
            elif n_merged > 0:
                notes.append(f"多分量范围(合并 {n_merged} 个大分量"
                             f"(面积≥{p.get('extent_min_comp_frac', 0.10):g}×最大), "
                             f"排除 {n_dropped} px)")
            else:
                notes.append(f"多分量范围(仅取最大连通分量, 丢弃 {n_dropped} px)")
        if int(extent.sum()) == 0:
            a.is_valid = False
            a.notes = notes + ['空范围']
            return a

        # Step 1: FO
        if self.first_opac is None or self.detector is None or self.calculator is None:
            raise RuntimeError("CTFCSequenceProcessor 缺少 first_opac/detector/calculator 组件")
        fo = self.first_opac.compute(seq, fmax, extent)

        # Steps 2-5: 标志区 + 充盈 + F1/F2
        det = self.detector.detect(seq, fmax, extent, fo)
        a.t_p, a.t_d = det['t_p'], det['t_d']
        a.seed_px, a.proximal_px, a.distal_px = det['seed_px'], det['proximal_px'], det['distal_px']
        a.f1, a.f2 = det['f1'], det['f2']
        notes += det['notes']

        # 软 notes
        if det['degenerate']:
            notes.append('单相充盈(退化)')
        if a.f1 == 0:
            notes.append('0帧已充盈, CTFC可能偏低')
        if fmax == n - 1:
            notes.append('Fmax为末帧(充盈可能截断)')
        if a.f2 is not None and a.f2 == fmax:
            notes.append('F2==Fmax(待核)')
        if a.f1 is not None and a.f2 is not None and a.f2 == a.f1:
            notes.append('同帧到达(快流)')

        # Step 7: CTFC + 硬校验
        ctfc, ctfc_raw, ctfc_rounded, calc_ok, calc_notes = self.calculator.calculate(
            a.f1, a.f2, branch_type, p)
        a.ctfc, a.ctfc_raw, a.ctfc_rounded = ctfc, ctfc_raw, ctfc_rounded
        notes += calc_notes
        a.is_valid = calc_ok
        a.notes = notes

        # 中间产物（可视化/明细保存用）
        a.detail = {
            'fill_p': det['fill_p'],
            'fill_d': det['fill_d'],
            'fill_e': det['fill_e'],
            'fmax_mask': (cleaned[fmax] > 0).astype(np.uint8),
            'extent': extent,
            'seed': det['seed'],
            'P': det['P'],
            'D': det['D'],
            'fo': fo,
            'f1_percentile': p['f1_percentile'],
            'f2_percentile': p['f2_percentile'],
            'n_frames': n,
        }

        status = '有效' if a.is_valid else '无效'
        self.logger.info(
            f"[{branch_type}] {case_id}/{sequence_id}: N={n}, Fmax={fmax}, "
            f"F1={a.f1}, F2={a.f2}, CTFC={ctfc if ctfc is None else round(ctfc, 2)}, {status}"
        )
        return a


class CTFCDatasetProcessor:
    """数据集级别处理（遍历、LCA 拆支 + 128 守卫、血管推断、白名单）"""

    def __init__(self, sequence_processor: CTFCSequenceProcessor = None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger('CTFC.DatasetProcessor')
        self.sequence_processor = sequence_processor

    @staticmethod
    def _norm_case_id(case_text: str) -> str:
        # "10" / "10-wanyanquan" -> "10"
        m = re.match(r"^\s*(\d+)", str(case_text))
        return m.group(1) if m else str(case_text).strip()

    @staticmethod
    def _extract_seq_id(seq_name: str):
        # "10-IM000098" -> 98
        s = str(seq_name)
        m = re.search(r"IM0*(\d+)", s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"(?:^|[-_])(s|seq)0*(\d+)", s, flags=re.IGNORECASE)
        if m:
            return int(m.group(2))
        nums = re.findall(r"(\d+)", s)
        return int(nums[-1]) if nums else None

    def _load_vessel_map(self, cls_summary_csv: str) -> Dict:
        """读取 cls_summary CSV（case_id, seq_name, vessel），key=(case_id, seq_id)"""
        vessel_map = {}
        if not cls_summary_csv or (not os.path.isfile(cls_summary_csv)):
            self.logger.warning(f"cls_summary_csv 不存在或未提供: {cls_summary_csv}")
            return vessel_map

        with open(cls_summary_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vessel = str(row.get("vessel", "")).strip().upper()
                case_id = self._norm_case_id(row.get("case_id", ""))
                seq_name = row.get("seq_name", "")
                seq_id = self._extract_seq_id(seq_name)
                if vessel in ("LCA", "RCA") and case_id and seq_id is not None:
                    vessel_map[(case_id, seq_id)] = vessel

        self.logger.info(f"_load_vessel_map: {len(vessel_map)} 条")
        return vessel_map

    def _infer_vessel(self, vessel_map, case_id_from_dir: str, sequence_id_from_dir: str) -> str:
        case_id = self._norm_case_id(case_id_from_dir)
        seq_id = self._extract_seq_id(sequence_id_from_dir)
        if seq_id is None:
            self.logger.warning(f"无法解析 seq_id，默认 RCA: {case_id_from_dir}/{sequence_id_from_dir}")
            return "RCA"
        vessel = vessel_map.get((case_id, seq_id))
        return vessel if vessel in ("LCA", "RCA") else "RCA"

    def _process_lca_two_branches(self, case_id: str, sequence_id: str,
                                  segmented_dir: str, selected_sequence_ids=None) -> Dict:
        """LCA 拆两支：LCX=像素 128，LAD=像素 255（内存处理）。

        128 守卫：拆支前算 n128_total=Σnz(m==128)；若为 0（解析到 fallback segmented/
        的合并 0/255 掩码）→ LCX 分支判无效 + note；LAD 分支保留值 + note。
        """
        out = {}
        p = self.sequence_processor.params
        raw_masks = self.sequence_processor._load_frame_masks_from_dir(segmented_dir)
        if not raw_masks:
            self.logger.warning(f"[LCA] 未找到有效掩码: {case_id}/{sequence_id} @ {segmented_dir}")
            return out

        n128_total = sum(int(np.count_nonzero(m == 128)) for m in raw_masks.values())
        merged_suspect = (n128_total == 0)
        if merged_suspect:
            self.logger.warning(
                f"[LCA] {case_id}/{sequence_id}: 无 128 像素（疑似合并 0/255 掩码, "
                f"segmented/ fallback?），LCX 判无效、LAD 记污染 note"
            )

        for branch_name, pix_val in [("LCX", 128), ("LAD", 255)]:
            branch_sequence_id = f"{sequence_id}_{branch_name}"
            if selected_sequence_ids is not None and branch_sequence_id not in selected_sequence_ids:
                self.logger.debug(f"[LCA] 跳过未选分支: {case_id}/{branch_sequence_id}")
                continue

            extra_notes = []
            if merged_suspect:
                if branch_name == "LCX":
                    extra_notes.append('无128像素(合并0/255掩码?)')
                else:
                    extra_notes.append('LAD范围被合并LCX像素污染')

            branch_masks = {
                fid: ((m == pix_val).astype(np.uint8) * 255)
                for fid, m in raw_masks.items()
            }
            nonzero_total = sum(int(np.count_nonzero(m)) for m in branch_masks.values())
            self.logger.info(
                f"[LCA-{branch_name}] {case_id}/{sequence_id}: "
                f"frames={len(branch_masks)}, nonzero_total={nonzero_total}, pix_val={pix_val}"
            )

            analysis = self.sequence_processor.process_sequence_from_masks(
                case_id=case_id,
                sequence_id=branch_sequence_id,
                frame_masks=branch_masks,
                branch_type=branch_name,
                extra_notes=extra_notes,
            )
            if merged_suspect and branch_name == "LCX":
                analysis.is_valid = False
            analysis.segmented_dir = segmented_dir
            out[branch_sequence_id] = analysis

        return out

    def process_dataset(self, root_dir: str, segmented_dirname='segmented',
                        cls_summary_csv=None, include_case_sequences=None) -> Dict:
        """遍历数据集。segmented_dirname 支持逗号分隔候选（按序取第一个存在的目录）。"""
        if isinstance(segmented_dirname, str):
            segmented_candidates = [x for x in re.split(r"[,\s]+", segmented_dirname) if x]
        else:
            segmented_candidates = [str(x).strip() for x in segmented_dirname if str(x).strip()]
        if not segmented_candidates:
            segmented_candidates = ['segmented']
        self.logger.info(f"候选分割目录: {segmented_candidates}")

        selected_map = None
        if include_case_sequences:
            selected_map = {}
            for item in include_case_sequences:
                case_id = str(item.get('case_id', '')).strip()
                seq_ids = item.get('sequence_ids', [])
                if not case_id:
                    continue
                if isinstance(seq_ids, (list, tuple)):
                    selected_map[case_id] = {str(x).strip() for x in seq_ids if str(x).strip()}
                else:
                    seq_text = str(seq_ids).strip()
                    selected_map[case_id] = {seq_text} if seq_text else set()
            self.logger.info(f"启用病例/序列白名单: {len(selected_map)} 个病例")

        vessel_map = self._load_vessel_map(cls_summary_csv)
        results = {}

        for case_dir in sorted(os.listdir(root_dir)):
            case_path = os.path.join(root_dir, case_dir)
            if not os.path.isdir(case_path):
                continue
            if selected_map is not None and case_dir not in selected_map:
                continue

            case_id = case_dir
            results.setdefault(case_id, {})
            selected_seq_ids = selected_map.get(case_id, set()) if selected_map is not None else None

            for seq_dir in sorted(os.listdir(case_path)):
                seq_path = os.path.join(case_path, seq_dir)
                if not os.path.isdir(seq_path):
                    continue

                sequence_id = seq_dir

                if selected_seq_ids is not None:
                    # 支持：基础序列ID / *_RCA / *_LAD / *_LCX
                    has_base = sequence_id in selected_seq_ids
                    has_rca = f"{sequence_id}_RCA" in selected_seq_ids
                    has_lad = f"{sequence_id}_LAD" in selected_seq_ids
                    has_lcx = f"{sequence_id}_LCX" in selected_seq_ids
                    if not (has_base or has_rca or has_lad or has_lcx):
                        continue

                # 按候选序查找分割目录
                segmented_dir = None
                for dname in segmented_candidates:
                    cand = os.path.join(seq_path, dname)
                    if os.path.isdir(cand):
                        segmented_dir = cand
                        break
                if segmented_dir is None:
                    self.logger.debug(f"跳过，无分割目录: {case_id}/{sequence_id}")
                    continue

                vessel = self._infer_vessel(vessel_map, case_id, sequence_id)

                try:
                    if vessel == "LCA":
                        selected_branches = None
                        if selected_seq_ids is not None and sequence_id not in selected_seq_ids:
                            selected_branches = (
                                {f"{sequence_id}_LAD", f"{sequence_id}_LCX"} & selected_seq_ids
                            )
                            if not selected_branches:
                                continue

                        self.logger.info(
                            f"[LCA] split branches: {case_id}/{sequence_id} -> LCX(128), LAD(255)"
                        )
                        branch_results = self._process_lca_two_branches(
                            case_id, sequence_id, segmented_dir,
                            selected_sequence_ids=selected_branches,
                        )
                        results[case_id].update(branch_results)
                    else:
                        if selected_seq_ids is not None:
                            allow_rca = (
                                sequence_id in selected_seq_ids
                                or f"{sequence_id}_RCA" in selected_seq_ids
                            )
                            if not allow_rca:
                                continue

                        self.logger.info(f"[RCA] single branch: {case_id}/{sequence_id}")
                        analysis = self.sequence_processor.process_sequence(
                            case_id, sequence_id, segmented_dir, branch_type='RCA'
                        )
                        results[case_id][sequence_id] = analysis
                except Exception as e:
                    self.logger.exception(f"[ERROR] {case_id}/{sequence_id}: {e}")

        return results


def print_results(results: Dict[str, Dict[str, CTFCSequenceAnalysis]]):
    """打印结果"""
    print("\n" + "=" * 80)
    print("CTFC 分析结果")
    print("=" * 80)

    for case_id in sorted(results.keys()):
        print(f"\n病例 {case_id}:")
        print("-" * 80)

        for sequence_id in sorted(results[case_id].keys()):
            analysis = results[case_id][sequence_id]
            status = "✓ 有效" if analysis.is_valid else "✗ 无效"
            print(f"  {sequence_id} [{analysis.branch_type}]: {status}")
            print(f"    总帧数: {analysis.n_frames}, Fmax: {analysis.max_frame_id} "
                  f"(像素数: {analysis.max_nonzero_count})")
            print(f"    F1: {analysis.f1}, F2: {analysis.f2}, "
                  f"t_p: {analysis.t_p}, t_d: {analysis.t_d}")
            if analysis.ctfc is not None:
                print(f"    CTFC: {analysis.ctfc:.2f} (raw={analysis.ctfc_raw:.2f}, "
                      f"取整={analysis.ctfc_rounded})")
            else:
                print(f"    CTFC: 未计算")
            if analysis.notes:
                print(f"    备注: {'; '.join(analysis.notes)}")


def save_results(results: Dict[str, Dict[str, CTFCSequenceAnalysis]], output_file: str):
    """保存结果到 JSON（不含 detail 中间产物）"""
    output_data = {}
    for case_id, sequences in results.items():
        output_data[case_id] = {}
        for seq_id, analysis in sequences.items():
            output_data[case_id][seq_id] = {
                'branch_type': analysis.branch_type,
                'n_frames': analysis.n_frames,
                'max_frame_id': analysis.max_frame_id,
                'max_nonzero_count': analysis.max_nonzero_count,
                't_p': analysis.t_p,
                't_d': analysis.t_d,
                'proximal_px': analysis.proximal_px,
                'distal_px': analysis.distal_px,
                'f1': analysis.f1,
                'f2': analysis.f2,
                'ctfc': analysis.ctfc,
                'ctfc_raw': analysis.ctfc_raw,
                'ctfc_rounded': analysis.ctfc_rounded,
                'is_valid': analysis.is_valid,
                'notes': analysis.notes,
            }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")
