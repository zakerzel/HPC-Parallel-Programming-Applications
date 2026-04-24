import argparse
from pathlib import Path

import pandas as pd
from skimage import io


def inspect_folder(folder: Path) -> pd.DataFrame:
    records = []
    for path in sorted(folder.glob('*')):
        if path.suffix.lower() not in {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}:
            continue
        img = io.imread(path)
        records.append({
            'file': path.name,
            'height': img.shape[0],
            'width': img.shape[1],
            'channels': 1 if img.ndim == 2 else img.shape[2],
            'dtype': str(img.dtype),
            'format': path.suffix.lower(),
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()
    folder = Path(args.input)
    df = inspect_folder(folder)
    out = folder.parents[2] / 'results' / 'dataset_inspection.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.head())
    if not df.empty:
        print('Unique sizes:', df[['height', 'width']].drop_duplicates().to_dict(orient='records'))


if __name__ == '__main__':
    main()
