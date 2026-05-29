#!/usr/bin/env python3
"""
calibrate_position.py — Sleep-position calibration & heuristic evaluation.

Workflow:
  1. Lie still in each position for ~30s, in the order given by --sequence
     (default: back, right, left — 3-position. Add 'stomach' to the sequence
     if you also calibrated stomach).
  2. Stop recording.
  3. Run this script on the resulting CSV.

The script:
  * finds each "still window" (low movement, low gyro) longer than MIN_STILL_S
  * pairs them in order with the position labels
  * averages the gravity vector over each window -> position centroid
  * runs heuristics on each centroid pretending we don't have labels
  * prints a confusion matrix (heuristic vs ground truth)
  * writes calibration.json with the centroids so analyze_night.py can use them

Usage:
  python calibrate_position.py /path/to/calibration.csv
  python calibrate_position.py /path/to/calibration.csv --sequence back,right,stomach,left
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


# ---------------------- Tunables -------------------------------------------

STILL_GYRO_DPS = 5.0   # gyro magnitude below this = still
STILL_ACC_G = 0.03     # bandpassed accel envelope below this = still
MIN_STILL_S = 5.0      # require at least this many seconds of stillness
                       # (lowered from 15 — 5s is enough for a reliable gravity centroid
                       # and lets you do a quick re-calibration if the app crashed)
MAX_STILL_S = 60.0     # cap each window to keep early/late drift out
GAP_S = 2.0            # require this much non-still between windows

LOWPASS_HZ = 0.5       # for gravity vector extraction
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 10.0
ENV_WINDOW_S = 0.20


# ---------------------- Loading --------------------------------------------

def load_csv(path: Path, sensor: str = "auto") -> pd.DataFrame:
    """Load a recording. If dual-sensor (ankle + chest columns), uses the
    sensor specified by `sensor` ('ankle', 'chest', or 'auto' which prefers
    chest when present). Renames columns to the legacy single-sensor names."""
    df = pd.read_csv(path)
    df["t"] = pd.to_datetime(df["timestamp_iso"], utc=True)
    df = df.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)

    has_dual = "ankle_acc_x_g" in df.columns and "chest_acc_x_g" in df.columns
    if has_dual:
        if sensor == "auto":
            sensor = "chest"  # prefer chest for body-position calibration
        prefix = sensor
        for k in ["acc_x_g", "acc_y_g", "acc_z_g",
                  "gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]:
            df[k] = pd.to_numeric(df[f"{prefix}_{k}"], errors="coerce")
        df = df.dropna(subset=["acc_x_g"]).reset_index(drop=True)
        print(f"  using {sensor} channel for calibration")
    return df


def estimate_fs(df: pd.DataFrame) -> float:
    return 1.0 / df["t"].diff().dt.total_seconds().dropna().median()


# ---------------------- Stillness detection --------------------------------

def find_still_windows(df: pd.DataFrame, fs: float) -> list[tuple[int, int]]:
    """Return (start_i, end_i) pairs for windows where the ankle is still
    for at least MIN_STILL_S seconds."""
    # 1) Bandpassed accel envelope (movement)
    a = np.sqrt(df.acc_x_g**2 + df.acc_y_g**2 + df.acc_z_g**2) - 1.0
    sos = signal.butter(4, [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
                        btype="bandpass", fs=fs, output="sos")
    bp = signal.sosfiltfilt(sos, a)
    win = max(1, int(ENV_WINDOW_S * fs))
    env = pd.Series(np.abs(bp)).rolling(win, min_periods=1, center=True).mean().to_numpy()

    # 2) Gyro magnitude
    gyro_mag = np.sqrt(df.gyro_x_dps**2 + df.gyro_y_dps**2 + df.gyro_z_dps**2)

    # Smooth gyro magnitude over 1 second to ignore brief jitters
    gw = max(1, int(1.0 * fs))
    gyro_sm = pd.Series(gyro_mag).rolling(gw, min_periods=1, center=True).mean().to_numpy()

    still = (env < STILL_ACC_G) & (gyro_sm < STILL_GYRO_DPS)

    # Find runs of stillness >= MIN_STILL_S
    min_n = int(MIN_STILL_S * fs)
    max_n = int(MAX_STILL_S * fs)
    gap_n = int(GAP_S * fs)
    windows = []
    i = 0
    while i < len(still):
        if still[i]:
            j = i
            while j < len(still) and still[j]:
                j += 1
            run_len = j - i
            if run_len >= min_n:
                # Trim to MAX_STILL_S, biased to the middle of the run
                if run_len > max_n:
                    s = i + (run_len - max_n) // 2
                    e = s + max_n
                else:
                    s, e = i, j
                # Enforce minimum non-still gap to previous window
                if not windows or s - windows[-1][1] >= gap_n:
                    windows.append((s, e))
            i = j
        else:
            i += 1
    return windows


# ---------------------- Gravity vector -------------------------------------

def gravity_vectors(df: pd.DataFrame, fs: float) -> np.ndarray:
    sos = signal.butter(4, LOWPASS_HZ, btype="lowpass", fs=fs, output="sos")
    gx = signal.sosfiltfilt(sos, df.acc_x_g.to_numpy())
    gy = signal.sosfiltfilt(sos, df.acc_y_g.to_numpy())
    gz = signal.sosfiltfilt(sos, df.acc_z_g.to_numpy())
    norm = np.sqrt(gx**2 + gy**2 + gz**2)
    norm = np.where(norm < 1e-6, 1e-6, norm)
    return np.column_stack([gx / norm, gy / norm, gz / norm])


def centroid(g: np.ndarray, s: int, e: int) -> np.ndarray:
    v = g[s:e].mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


# ---------------------- Heuristic classifier -------------------------------

def classify_heuristic(unit_g: np.ndarray) -> str:
    """Classify a unit gravity vector into back/stomach/right/left.

    Mounting-agnostic version: we don't know which axis is which on the
    ankle, so we look at signature *patterns* of the components:
      - Back (supine): foot tends to fall outward, sole points up-and-out
        => one component is strongly +1 or -1 (foot vertical-ish)
      - Stomach (prone): foot points down, sole down
        => same axis as back but opposite sign
      - Side: foot is roughly horizontal => no single component dominates;
        the gravity vector is split between two axes

    Without calibration we can't reliably distinguish back from stomach
    or left from right. So this baseline only outputs:
      - 'vertical_foot'  (back OR stomach)
      - 'horizontal_foot' (some side)
    The eval will then test whether even this 2-way split is reliable.
    """
    abs_max = np.max(np.abs(unit_g))
    if abs_max > 0.85:
        return "vertical_foot"
    return "horizontal_foot"


def classify_calibrated(unit_g: np.ndarray, centroids: dict[str, np.ndarray]) -> str:
    """Calibrated nearest-centroid classifier (cosine similarity)."""
    best_label, best_sim = None, -2.0
    for label, c in centroids.items():
        sim = float(np.dot(unit_g, c))
        if sim > best_sim:
            best_sim, best_label = sim, label
    return best_label


# ---------------------- Main -----------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--sequence", default="back,right,left",
                    help="comma-separated position labels in the order you lay in them. "
                         "Default is 3-position (back, right, left); add 'stomach' if you "
                         "calibrated all 4.")
    ap.add_argument("--out", type=Path,
                    help="path to write calibration.json (default: alongside CSV)")
    ap.add_argument("--placement", default="",
                    help="free-text note describing where the sensor is (e.g. "
                         "'right sock, medial side, behind ankle near Achilles')")
    ap.add_argument("--sensor", default="auto", choices=["auto", "ankle", "chest"],
                    help="which sensor to calibrate (dual-sensor recordings only)")
    args = ap.parse_args()

    expected = [s.strip() for s in args.sequence.split(",")]

    print(f"loading {args.csv}…")
    df = load_csv(args.csv, sensor=args.sensor)
    fs = estimate_fs(df)
    print(f"  {len(df):,} rows, fs = {fs:.2f} Hz, "
          f"span = {(df.t.iloc[-1] - df.t.iloc[0]).total_seconds():.0f} s")

    print("finding still windows…")
    windows = find_still_windows(df, fs)
    print(f"  found {len(windows)} still windows >= {MIN_STILL_S}s")
    for i, (s, e) in enumerate(windows):
        t0 = df.t.iloc[s].tz_convert("America/Los_Angeles").strftime("%H:%M:%S")
        t1 = df.t.iloc[e-1].tz_convert("America/Los_Angeles").strftime("%H:%M:%S")
        dur = (df.t.iloc[e-1] - df.t.iloc[s]).total_seconds()
        print(f"    {i+1}: {t0} → {t1}  ({dur:.1f}s)")

    if len(windows) < len(expected):
        print(f"\nERROR: expected {len(expected)} windows but only found {len(windows)}.")
        print("Either the recording was too short, you moved too much during the still phase,")
        print("or stillness thresholds need tuning. Try --sequence with fewer positions.")
        return

    if len(windows) > len(expected):
        print(f"\nWARN: found {len(windows)} windows but sequence only has {len(expected)}.")
        print(f"Using the first {len(expected)} windows.")
        windows = windows[: len(expected)]

    print("\ncomputing gravity vectors…")
    g = gravity_vectors(df, fs)

    centroids: dict[str, np.ndarray] = {}
    print(f"\n{'pos':<10} {'gx':>7} {'gy':>7} {'gz':>7}   heuristic")
    print("-" * 50)
    rows = []
    for label, (s, e) in zip(expected, windows):
        c = centroid(g, s, e)
        centroids[label] = c
        h = classify_heuristic(c)
        print(f"{label:<10} {c[0]:+.3f}  {c[1]:+.3f}  {c[2]:+.3f}   {h}")
        rows.append({"truth": label, "heuristic": h,
                     "gx": float(c[0]), "gy": float(c[1]), "gz": float(c[2])})

    # Heuristic eval — collapse truth labels to vertical_foot / horizontal_foot
    print("\nheuristic accuracy (vertical_foot vs horizontal_foot):")
    truth_collapsed = {
        "back": "vertical_foot", "stomach": "vertical_foot",
        "right": "horizontal_foot", "left": "horizontal_foot",
    }
    correct = 0
    for r in rows:
        expected_h = truth_collapsed.get(r["truth"], "?")
        ok = r["heuristic"] == expected_h
        correct += int(ok)
        mark = "✓" if ok else "✗"
        print(f"  {mark} {r['truth']:<8} expected={expected_h:<16} got={r['heuristic']}")
    print(f"  overall: {correct}/{len(rows)} correct")

    # Calibrated eval — sanity check that nearest-centroid recovers the labels
    print("\ncalibrated 4-way classifier (sanity check, leave-one-out):")
    correct = 0
    for r in rows:
        label = r["truth"]
        loo_centroids = {k: v for k, v in centroids.items() if k != label}
        if not loo_centroids:
            continue
        unit_g = np.array([r["gx"], r["gy"], r["gz"]])
        pred = classify_calibrated(unit_g, loo_centroids)
        ok = (pred == label)
        # leave-one-out can't recover a unique label, so just show predictions
        print(f"  {r['truth']:<8} loo-pred={pred}")

    # Save centroids + metadata for analyze_night.py to use later
    out_path = args.out or args.csv.with_suffix(".calibration.json")
    out_obj = {
        "centroids": {label: c.tolist() for label, c in centroids.items()},
        "metadata": {
            "source_csv": str(args.csv),
            "calibrated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "sequence": expected,
            "placement": args.placement,
            "sensor": args.sensor,
            "fs_hz": float(fs),
        },
    }
    out_path.write_text(json.dumps(out_obj, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
