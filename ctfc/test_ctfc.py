#!/usr/bin/env python3
"""
CTFC 单元自检（合成数据）

运行:
    /root/miniconda3/envs/mamba/bin/python tfc/test_ctfc.py

用例:
  1. 正常增长管（RCA / LAD 校正）：F1/F2/Fmax 精确值 + CTFC 公式
  2. 瞬时全充满（退化单相）：退化 note + 同帧到达 note + CTFC=0 且有效
  3. 合并 LCA 掩码（0/255 无 128）：128 守卫 → LCX 无效、LAD 保留 + 污染 note
  4. 多分量 E：big 模式下小游离块（<10%）被排除（回归）
  5. 多分量 E 合并策略：断管 big/largest/all 三模式 + 断连 E 的 D 落远端
"""

import logging
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from ctfc_analyzer import (
    CTFCCalculator,
    CTFCDatasetProcessor,
    CTFCSequenceProcessor,
    CTFCDetector,
    FirstOpac,
    FrameAnalyzer,
    MaskProcessor,
)

LOG = logging.getLogger('CTFC.Test')
logging.basicConfig(level=logging.WARNING)

DEFAULT_PARAMS = {
    'mask_suffix': '_pred.png',
    'border_width': 30, 'min_component_area': 50,
    'hysteresis_frames': 2,
    'seed_frac': 0.02, 'proximal_frac': 0.10, 'distal_frac': 0.10,
    'max_region_frac': 0.40, 'min_region_pixels': 30,
    'extent_merge': 'big', 'extent_min_comp_frac': 0.10,
    'degenerate_span': 1,
    'f1_percentile': 0.80, 'f2_percentile': 0.10,
    'frame_rate_correction': 2.0, 'lad_correction': 1.7,
    'lad_correction_enabled': True,
    'min_frames': 10, 'min_fmax_pixels': 800, 'max_ctfc': 200,
}

# 几何: 512x512, 管 = rows 200..215 (16 行) x cols 50..449 (400 列)
# 帧 f 填充 min(20f, 400) 列（自 col 50 起），帧 0 空
H, W = 512, 512
R0, R1 = 200, 216
C0 = 50
NCOLS = 400


def growing_masks(n_frames: int):
    masks = {}
    for f in range(n_frames):
        m = np.zeros((H, W), dtype=np.uint8)
        ncols = min(20 * f, NCOLS)
        if ncols > 0:
            m[R0:R1, C0:C0 + ncols] = 255
        masks[f] = m
    return masks


def make_processor(params=None):
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    mp = MaskProcessor(logger=LOG)
    fa = FrameAnalyzer(logger=LOG)
    fo = FirstOpac(hysteresis_frames=p['hysteresis_frames'], logger=LOG)
    det = CTFCDetector(params=p, logger=LOG)
    calc = CTFCCalculator(params=p, logger=LOG)
    return CTFCSequenceProcessor(mask_processor=mp, frame_analyzer=fa, first_opac=fo,
                                 detector=det, calculator=calc, params=p, logger=LOG)


# ---------------------------------------------------------------- 用例 1

def test_growing_tube_rca():
    proc = make_processor()
    a = proc.process_sequence_from_masks('C1', 'IM000001', growing_masks(21), branch_type='RCA')

    assert a.is_valid, f"应有效: {a.notes}"
    assert a.max_frame_id == 20, f"Fmax={a.max_frame_id} != 20"
    assert a.f1 == 2, f"F1={a.f1} != 2"
    assert a.f2 == 19, f"F2={a.f2} != 19"
    assert a.f2 <= a.max_frame_id
    assert a.ctfc == 34.0, f"CTFC={a.ctfc} != 34.0"
    assert a.ctfc_raw == 34.0
    assert a.ctfc_rounded == 34
    assert a.branch_type == 'RCA'
    assert abs(a.ctfc - (a.f2 - a.f1) * 2.0) < 1e-9, "RCA 不应有 LAD 校正"
    assert a.detail['fill_p'][a.max_frame_id] == 1.0, "fill_p[Fmax] 必须为 1（存在性保证）"
    assert not any('退化' in n for n in a.notes), f"不应退化: {a.notes}"
    print("PASS test_growing_tube_rca  (F1=2, F2=19, Fmax=20, CTFC=34.0)")


def test_growing_tube_lad():
    proc = make_processor()
    a = proc.process_sequence_from_masks('C1', 'IM000001', growing_masks(21), branch_type='LAD')

    assert a.is_valid, f"应有效: {a.notes}"
    assert a.f1 == 2 and a.f2 == 19
    assert a.ctfc_raw == 34.0
    assert abs(a.ctfc - 34.0 / 1.7) < 1e-9, f"LAD 应 ÷1.7: {a.ctfc}"
    assert a.ctfc_rounded == 20, f"round(20.0)=20: {a.ctfc_rounded}"
    print("PASS test_growing_tube_lad  (CTFC=34/1.7=20.0, 取整=20)")


# ---------------------------------------------------------------- 用例 2

def test_degenerate_instant_fill():
    full = np.zeros((H, W), dtype=np.uint8)
    full[R0:R1, C0:C0 + NCOLS] = 255
    masks = {f: full.copy() for f in range(21)}  # 每帧全充满
    proc = make_processor()
    a = proc.process_sequence_from_masks('C2', 'IM000002', masks, branch_type='RCA')

    assert a.is_valid, f"快流值应保留: {a.notes}"
    assert any('单相充盈(退化)' in n for n in a.notes), f"应有退化 note: {a.notes}"
    assert any('同帧到达(快流)' in n for n in a.notes), f"应有同帧到达 note: {a.notes}"
    assert a.f1 == 0 and a.f2 == 0, f"F1={a.f1}, F2={a.f2} != 0"
    assert a.ctfc == 0.0 and a.ctfc_rounded == 0
    print("PASS test_degenerate_instant_fill  (F1=F2=0, CTFC=0.0, 有效+note)")


# ---------------------------------------------------------------- 用例 3

def test_merged_lca_128_guard():
    with tempfile.TemporaryDirectory() as tmp:
        for f in range(16):
            m = growing_masks(16)[f]  # 0/255，无 128
            cv2.imwrite(os.path.join(tmp, f"cag2-c1-s1-f{f}_pred.png"), m)

        proc = make_processor()
        ds = CTFCDatasetProcessor(sequence_processor=proc, logger=LOG)
        out = ds._process_lca_two_branches('1', '1-IM000001', tmp)

    assert '1-IM000001_LCX' in out and '1-IM000001_LAD' in out, f"分支缺失: {list(out)}"
    lcxa, lada = out['1-IM000001_LCX'], out['1-IM000001_LAD']

    assert not lcxa.is_valid, "LCX 应无效"
    assert any('无128像素' in n for n in lcxa.notes), f"LCX 应有 128 note: {lcxa.notes}"

    assert lada.is_valid, f"LAD 应保留值: {lada.notes}"
    assert any('污染' in n for n in lada.notes), f"LAD 应有污染 note: {lada.notes}"
    assert lada.f1 == 2 and lada.f2 == 14, f"LAD F1={lada.f1} F2={lada.f2} != 2/14"
    assert abs(lada.ctfc - 24.0 / 1.7) < 1e-9, f"LAD CTFC: {lada.ctfc}"
    print("PASS test_merged_lca_128_guard  (LCX 无效+128 note; LAD 保留+污染 note)")


# ---------------------------------------------------------------- 用例 4

def test_multi_component_extent():
    masks = growing_masks(21)
    blob = np.zeros((H, W), dtype=np.uint8)
    blob[60:80, 380:400] = 255  # 游离块（仅末帧出现）
    masks[20] = cv2.bitwise_or(masks[20], blob)

    proc = make_processor()
    a = proc.process_sequence_from_masks('C3', 'IM000003', masks, branch_type='RCA')

    assert a.is_valid, f"应有效: {a.notes}"
    assert any('多分量范围' in n for n in a.notes), f"应有多分量 note: {a.notes}"
    assert a.max_frame_id == 20
    assert a.max_nonzero_count == 6800, f"max_count={a.max_nonzero_count} != 6800"
    assert a.detail['extent'][60:80, 380:400].sum() == 0, "E 应取最大连通分量（管）"
    assert a.detail['D'][60:80, 380:400].sum() == 0, "D 不应含游离块"
    assert a.f1 == 2 and a.f2 == 19, f"F1={a.f1} F2={a.f2} != 2/19"
    print("PASS test_multi_component_extent  (E=最大CC, 游离块被排除, F1/F2 不变)")


# ---------------------------------------------------------------- 用例 5

def test_extent_merge_modes():
    # 断管: 段 A cols 50..259 (3360px, 帧1 即充满=近端段), 4px 间隙,
    # 段 B cols 264..449 (2976px, 帧2 起每帧 +12 col 向远端生长)
    def broken_masks(n_frames: int):
        masks = {}
        for f in range(n_frames):
            m = np.zeros((H, W), dtype=np.uint8)
            if f >= 1:
                m[R0:R1, 50:260] = 255
            if f >= 2:
                ncols = min(12 * (f - 1), 186)
                if ncols > 0:
                    m[R0:R1, 264:264 + ncols] = 255
            masks[f] = m
        return masks

    masks = broken_masks(21)

    # big（默认）: B 面积比 0.89 ≥ 0.10 → 两段并入 E
    a = make_processor().process_sequence_from_masks('C5', 'IM000007', masks, branch_type='RCA')
    assert a.is_valid, f"应有效: {a.notes}"
    assert a.detail['extent'].sum() == 3360 + 2976, f"E 应含两段: {a.detail['extent'].sum()}"
    assert any('多分量范围' in n and '合并' in n for n in a.notes), f"应有合并 note: {a.notes}"
    assert a.f1 == 1, f"F1={a.f1} != 1（近端段 A 帧1 全充）"
    dmap = a.detail['D']
    assert dmap[:, :264].sum() == 0, "D 不应含近端段 A / 间隙"
    assert dmap[R0:R1, 400:].sum() > 0, "D 应落在 B 远端（断连 E 测地距离跨分量）"
    assert a.f2 is not None and a.f2 <= a.max_frame_id, f"F2={a.f2}"
    assert abs(a.ctfc - (a.f2 - a.f1) * 2.0) < 1e-9, f"CTFC={a.ctfc}"
    print(f"PASS test_extent_merge_modes  (big: E={a.detail['extent'].sum()}px, "
          f"F1={a.f1}, F2={a.f2}, CTFC={a.ctfc})")

    # largest: 仅保留最大分量 A
    a2 = make_processor(params={'extent_merge': 'largest'}).process_sequence_from_masks(
        'C5', 'IM000008', masks, branch_type='RCA')
    assert a2.detail['extent'].sum() == 3360, f"largest 应仅留 A: {a2.detail['extent'].sum()}"

    # all: 连小游离块（320px, 比 0.095 < 0.10）也并入
    blob = np.zeros((H, W), dtype=np.uint8)
    blob[60:80, 380:396] = 255
    masks_all = dict(masks)
    masks_all[17] = cv2.bitwise_or(masks[17], blob)
    a3 = make_processor(params={'extent_merge': 'all'}).process_sequence_from_masks(
        'C5', 'IM000009', masks_all, branch_type='RCA')
    assert a3.detail['extent'].sum() == 3360 + 2976 + 320, f"all 应含游离块: {a3.detail['extent'].sum()}"
    a4 = make_processor().process_sequence_from_masks('C5', 'IM000010', masks_all, branch_type='RCA')
    assert a4.detail['extent'].sum() == 3360 + 2976, "big 应排除小游离块(<10%)"
    print("PASS test_extent_merge_modes  (largest/all/big 三模式 extent 边界正确)")


# ---------------------------------------------------------------- 用例 6（守卫）

def test_hard_guards():
    proc = make_processor()
    a = proc.process_sequence_from_masks('C4', 'IM000004', {}, branch_type='RCA')
    assert not a.is_valid and any('无掩码' in n for n in a.notes)

    a = proc.process_sequence_from_masks('C4', 'IM000005', growing_masks(5), branch_type='RCA')
    assert not a.is_valid and any('帧数过少' in n for n in a.notes)

    small = {f: np.zeros((H, W), dtype=np.uint8) for f in range(12)}
    for f in range(1, 12):
        small[f][200:225, 200:225] = 255  # 625 px: 过小分量保留(≥50) 但 < 800
    a = proc.process_sequence_from_masks('C4', 'IM000006', small, branch_type='RCA')
    assert not a.is_valid and any('Fmax像素数过小' in n for n in a.notes)
    print("PASS test_hard_guards  (无掩码/帧数过少/Fmax像素过小 均判无效)")


def main():
    tests = [
        test_growing_tube_rca,
        test_growing_tube_lad,
        test_degenerate_instant_fill,
        test_merged_lca_128_guard,
        test_multi_component_extent,
        test_extent_merge_modes,
        test_hard_guards,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print("=" * 60)
    if failed:
        print(f"{failed}/{len(tests)} 个用例失败")
        sys.exit(1)
    print(f"全部 {len(tests)} 个用例通过 ✅")


if __name__ == '__main__':
    main()
