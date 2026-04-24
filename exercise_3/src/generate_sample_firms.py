from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({
        'latitude': 18.0 + rng.random(n) * 0.5,
        'longitude': -90.0 + rng.random(n) * 0.5,
        'frp': rng.uniform(5, 60, n),
        'brightness': rng.uniform(300, 400, n),
        'acq_date': ['2026-03-01'] * n,
    })
    out = Path(__file__).resolve().parents[1] / 'data' / 'firms_sample.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(out)


if __name__ == '__main__':
    main()
