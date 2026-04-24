from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import color, exposure, filters, io, measure, morphology, segmentation, util


def segment_image(path: Path) -> tuple[str, pd.DataFrame, np.ndarray]:
    image = io.imread(path)
    if image.ndim == 3:
        image = color.rgb2gray(image)
    image = util.img_as_float(image)
    image = exposure.equalize_adapthist(image)
    threshold = filters.threshold_otsu(image)
    mask = image > threshold
    mask = morphology.remove_small_objects(mask, min_size=30)
    mask = morphology.remove_small_holes(mask, area_threshold=30)
    labels = measure.label(mask)
    rows = []
    for region in measure.regionprops(labels):
        minr, minc, maxr, maxc = region.bbox
        rows.append({
            'image': path.name,
            'label': region.label,
            'bbox_min_row': minr,
            'bbox_min_col': minc,
            'bbox_max_row': maxr,
            'bbox_max_col': maxc,
            'area_px': region.area,
            'major_axis_length_px': region.major_axis_length,
            'minor_axis_length_px': region.minor_axis_length,
            'centroid_row': region.centroid[0],
            'centroid_col': region.centroid[1],
        })
    return path.name, pd.DataFrame(rows), segmentation.mark_boundaries(image, labels)


def _worker(path_str: str):
    return segment_image(Path(path_str))


def run(folder: Path, mode: str, workers: int) -> tuple[pd.DataFrame, float]:
    image_paths = [p for p in sorted(folder.glob('*')) if p.suffix.lower() in {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}]
    start = time.perf_counter()
    if mode == 'serial':
        outputs = [segment_image(path) for path in image_paths]
    else:
        with Pool(processes=workers) as pool:
            outputs = pool.map(_worker, [str(p) for p in image_paths])
    elapsed = time.perf_counter() - start

    rows = []
    out_dir = folder.parents[2] / 'results'
    overlay_dir = out_dir / f'overlays_{mode}'
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for name, df, overlay in outputs:
        rows.append(df)
        io.imsave(overlay_dir / f'{Path(name).stem}_overlay.png', util.img_as_ubyte(overlay))
    full = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return full, elapsed


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['image', 'detected_cells', 'avg_width_px', 'avg_length_px', 'std_width_px', 'std_length_px'])
    summary = df.groupby('image').agg(
        detected_cells=('label', 'count'),
        avg_width_px=('minor_axis_length_px', 'mean'),
        avg_length_px=('major_axis_length_px', 'mean'),
        std_width_px=('minor_axis_length_px', 'std'),
        std_length_px=('major_axis_length_px', 'std'),
    ).reset_index()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--mode', choices=['serial', 'parallel'], default='serial')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    folder = Path(args.input)
    measurements, elapsed = run(folder, args.mode, args.workers)
    summary = summarize(measurements)

    out_dir = folder.parents[2] / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    measurements.to_csv(out_dir / f'measurements_{args.mode}.csv', index=False)
    summary.to_csv(out_dir / f'summary_{args.mode}.csv', index=False)
    pd.DataFrame([{'mode': args.mode, 'workers': args.workers, 'time_s': elapsed, 'images': summary.shape[0]}]).to_csv(
        out_dir / f'timing_{args.mode}.csv', index=False
    )
    print(summary)
    print({'mode': args.mode, 'time_s': elapsed})


if __name__ == '__main__':
    main()
