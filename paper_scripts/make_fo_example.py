#!/usr/bin/env python3
"""Example figure: the per-pixel first-opacification (FO) spatial map.

Shows, for one sequence, the spatial basis of the DL-cTFC landmarks:
  (a) raw angiogram at the peak-opacification (Fmax) frame
  (b) FO map -- each territory pixel colored by the frame in which it first
      opacified (window [0, Fmax], 2-frame hysteresis). The extent E is the
      colored region (its boundary = the Fmax territory, merged per
      extent_merge). Proximal region P (cyan) and distal region D (salmon)
      are the geodesic bands from which F1 = q15(P-FO) and F2 = q10(D-FO)
      are read; the seed (white) is the early-opacification ostium anchor.

All arrays come from the frozen-q15 rerun NPZ (reproduces F1/F2 exactly).
Output: PNG + TIF.
"""
import argparse, os
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

NEW_BASE = '/home/share/clr/share/work/DLcTFC/paper_results/02_ctfc/full_rerun'
DATA_ROOT = '/home/share/clr/share/data/AI_TMPFC'
OUT = '/home/share/clr/share/work/DLcTFC/paper_results/02_ctfc/fo_example'


def contour(mask):
    """Return list of (N,2) contour point arrays for overlay."""
    out = cv2.findContours((mask > 0).astype(np.uint8),
                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = out[1] if len(out) == 3 else out[0]
    return [c.reshape(-1, 2) for c in cnts if len(c) >= 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue-dir', default=f'{DATA_ROOT}/CMVD-ID1-ID50')
    ap.add_argument('--case', default='2-Huangxuexi')
    ap.add_argument('--seq', default='2-IM000012')
    ap.add_argument('--branch', default='LAD')
    ap.add_argument('--margin', type=int, default=24)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    qdir = os.path.basename(args.queue_dir)
    npz = os.path.join(NEW_BASE, qdir, 'detailed',
                       f'{args.case}__{args.seq}_{args.branch}.npz')
    z = np.load(npz)
    fo, ext, P, D, seed = z['fo'], z['extent'], z['P'], z['D'], z['seed']
    fmax = int(np.quantile(fo[ext > 0], 1.0))  # == Fmax (sentinel clamped)

    # raw Fmax frame
    cnum = args.case.split('-')[0]
    snum = int(args.seq.split('IM')[-1])
    src_path = (f'{args.queue_dir}/{args.case}/{args.seq}/segmented3c2/'
                f'cag2-c{cnum}-s{snum}-f{fmax}_src.png')
    raw = np.array(Image.open(src_path).convert('L')).astype(float)
    lo, hi = np.percentile(raw, [1, 99])
    raw = np.clip((raw - lo) / (hi - lo), 0, 1)

    # crop to extent bbox + margin
    ys, xs = np.where(ext > 0)
    r0, r1 = max(0, ys.min() - args.margin), min(raw.shape[0], ys.max() + args.margin)
    c0, c1 = max(0, xs.min() - args.margin), min(raw.shape[1], xs.max() + args.margin)
    sl = (slice(r0, r1), slice(c0, c1))
    raw, ext, P, D, seed, fo = (a[sl] for a in (raw, ext, P, D, seed, fo))

    cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=fmax)
    fo_disp = np.where(ext > 0, fo, np.nan)  # sentinel outside extent hidden

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9),
                             gridspec_kw={'width_ratios': [1, 1.35]})
    H, W = raw.shape

    # (a) raw Fmax
    ax = axes[0]
    ax.imshow(raw, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'(a) Raw angiogram\n(Fmax = frame {fmax})', fontsize=11)
    ax.axis('off')

    # (b) FO map + P/D/seed  (extent = the colored region's boundary)
    ax = axes[1]
    ax.imshow(raw, cmap='gray', vmin=0, vmax=1, alpha=0.25)
    im = ax.imshow(fo_disp, cmap=cmap, norm=norm, interpolation='nearest')
    for cnt in contour(P):
        ax.plot(np.append(cnt[:, 0], cnt[0, 0]), np.append(cnt[:, 1], cnt[0, 1]),
                color='cyan', lw=2.2)
    for cnt in contour(D):
        ax.plot(np.append(cnt[:, 0], cnt[0, 0]), np.append(cnt[:, 1], cnt[0, 1]),
                color='deeppink', lw=2.2)
    sy, sx = np.where(seed > 0)
    ax.scatter(sx, sy, s=6, c='white', edgecolors='k', linewidths=0.3, zorder=6, label='seed (ostium)')

    f1 = int(np.quantile(fo[P > 0], 0.15))
    f2 = int(np.quantile(fo[D > 0], 0.10))
    # labels in the right (empty) area, arrows to region centroids
    py, px = np.where(P > 0); dy, dx = np.where(D > 0)
    pc, dc = (px.mean(), py.mean()), (dx.mean(), dy.mean())
    ax.annotate(f'P region (proximal)\nF1 = q15(P-FO) = {f1}',
                xy=pc, xytext=(W * 0.58, H * 0.16), fontsize=9,
                color='cyan', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='cyan', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.25', fc='black', alpha=0.6))
    ax.annotate(f'D region (distal)\nF2 = q10(D-FO) = {f2}',
                xy=dc, xytext=(W * 0.58, H * 0.82), fontsize=9,
                color='deeppink', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='deeppink', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.25', fc='black', alpha=0.6))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.10)
    cb.set_label('FO (first-opac frame, 0 to Fmax)', fontsize=9)
    ax.set_title(f'(b) First-opacification map + P/D regions', fontsize=11)
    ax.axis('off')

    fig.suptitle(f'{args.case} / {args.seq} / {args.branch}  --  FO spatial map '
                 f'(color = frame of first opacification; viridis: dark=early, bright=late)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, f'fo_map_{args.case.split("-")[0]}_{args.seq.split("-")[1]}_{args.branch}')
    fig.savefig(base + '.png', dpi=300)
    fig.savefig(base + '.tif', dpi=300)
    print('saved', base + '.png / .tif')
    print(f'Fmax={fmax}  F1={f1}  F2={f2}  extent={int(ext.sum())}px  P={int(P.sum())}px  D={int(D.sum())}px  seed={int(seed.sum())}px')


if __name__ == '__main__':
    main()
