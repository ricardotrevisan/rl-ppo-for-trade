Utilities

- Multi‑seed runner (`run_multi_seed.py`)
  - Purpose: run adaptation.py across multiple seeds, parse printed metrics, and write per‑seed and summary CSVs under `data/metrics/`.
  - Typical use:
    - 5 seeds (fast): `uv run data/util/run_multi_seed.py --num-seeds 5 --num-envs 8`
    - Explicit seeds: `uv run data/util/run_multi_seed.py --seeds 0 1 2 3 4 --num-envs 8`
    - Pass through training/eval windows and reward/trading knobs as needed (same flags as adaptation.py).
  - Output:
    - `data/metrics/multi_seed_<stamp>.csv` (per seed)
    - `data/metrics/multi_seed_<stamp>_summary.csv` (mean/std/median/p5/p95)

- Data cleaner (`clear_data.py`)
  - Purpose: delete `.csv` and `.png` files under the `data/` directory (recursive) with safety features.
  - Safe usage:
    - Dry run: `uv run data/util/clear_data.py -n`
    - Confirm delete: `uv run data/util/clear_data.py -y`
    - Custom dir/exts: `uv run data/util/clear_data.py --data-dir data -e .csv .png .jpg -y`
  - Notes: refuses to operate outside the repo; previews files before deletion unless `-y` is used.

