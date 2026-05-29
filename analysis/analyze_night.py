#!/usr/bin/env python3
"""
analyze_night.py — PLM detection + Oura sleep-stage join for one night.

Usage:
  # Option A: environment variable
  export OURA_PAT="your_personal_access_token"
  python analyze_night.py /path/to/twitch_YYYYMMDD_HHMMSS.csv

  # Option B: .env file (preferred — keeps token out of shell config)
  echo 'OURA_PAT=your_personal_access_token' >> ~/twitch/analysis/.env
  python analyze_night.py /path/to/twitch_YYYYMMDD_HHMMSS.csv

Outputs (alongside the input CSV):
  <stem>_events.csv          — one row per detected movement event, classified
  <stem>_overview.png        — full-night plot: hypnogram on top, accel envelope below, events overlaid
  <stem>_summary.txt         — counts by classification, by sleep stage

Detection pipeline:
  1. Read raw 30 Hz accel CSV, drop duplicate-timestamp rows.
  2. Compute |a| - 1g (gravity-removed magnitude).
  3. Bandpass 0.5–10 Hz (Butterworth, zero-phase).
  4. Envelope: rectify, 200 ms moving average.
  5. Threshold with hysteresis: T_on=0.05g, T_off=0.02g.
  6. Duration gate: keep events 0.3–10 s. Anything 10–30 s flagged as "rollover".
     Anything >30 s flagged as "out_of_bed".
  7. Series detection: PLMs in clinical scoring are ≥4 events with 5–90 s
     inter-event gaps. Tag those as PLM_in_series; isolated short events
     are isolated_movement.
  8. Pull Oura sleep_phase_5_min and tag each event with its sleep stage.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal
import requests

# ---------------------- .env loader ----------------------------------------
# Reads KEY=VALUE pairs from ~/twitch/analysis/.env (if present) and sets
# them as environment variables, without overwriting values already in the
# environment.  This lets you store OURA_PAT there instead of in .zshrc.
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k = _k.strip()
        _v = _v.strip().strip('"').strip("'")
        if _k and _k not in os.environ:
            os.environ[_k] = _v


# ---------------------- Tunables -------------------------------------------

T_ON = 0.05         # g, envelope must exceed to start an event
T_OFF = 0.02        # g, envelope must fall below to end an event
EVENT_MIN_S = 0.3   # min duration of a kept event (clinical PLM is 0.5+, we relax for v0)
EVENT_MAX_S = 10.0  # max duration of a kept event (clinical PLM ceiling)
ROLLOVER_MIN_S = 1.5     # >=1.5 s after merging = rollover candidate
ROLLOVER_MAX_S = 30.0    # 1.5-30 s = rollover; >30 s = out_of_bed
SERIES_MIN_GAP_S = 5.0   # PLM series: events must be >=5 s apart
SERIES_MAX_GAP_S = 90.0  # PLM series: events must be <=90 s apart
SERIES_MIN_COUNT = 4     # PLM series: at least 4 events
PLM_PEAK_MAX_G = 0.5     # PLMs are small twitches; bigger spikes are gross movements
MERGE_GAP_S = 2.0        # merge adjacent events within this gap (kills fragmentation)

# Tilt-based rollover detection (uses gravity vector — orientation, not acceleration)
TILT_LOWPASS_HZ = 0.5    # low-pass cutoff to extract gravity from raw accel
TILT_REF_WINDOW_S = 30.0 # rolling window for reference orientation
TILT_MIN_DEG = 20.0      # min sustained tilt change to call a rollover (catches back->side shifts)
TILT_HOLD_S = 2.0        # tilt must persist this long (a true rollover stays put)

# Kick detection: high-peak short event with no sustained tilt -> probably a leg kick
KICK_MIN_PEAK_G = 0.5    # peak must be at least this big
KICK_MAX_DURATION_S = 5.0  # kicks are brief
KICK_MAX_TILT_DEG = 15.0   # if tilt is bigger than this, it's a rollover instead

ENVELOPE_WINDOW_S = 0.20
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 10.0


# ---------------------- Loading --------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV. Supports three formats:
      1. Legacy single-sensor (acc_x_g etc.)
      2. Ankle + chest dual-sensor (ankle_* + chest_*)
      3. Dual-ankle (ankleR_* + ankleL_*)  -- current

    In all cases, the "primary" sensor (used for PLM detection) is aliased
    to legacy column names so existing code works unchanged. For dual-ankle,
    the right ankle is primary. For ankle+chest, the ankle is primary.
    """
    print(f"loading {path}…")
    df = pd.read_csv(path)
    df["t"] = pd.to_datetime(df["timestamp_iso"], utc=True)
    df = df.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)

    has_dual_ankle = "ankleR_acc_x_g" in df.columns and "ankleL_acc_x_g" in df.columns
    has_ankle_chest = "ankle_acc_x_g" in df.columns and "chest_acc_x_g" in df.columns
    has_hr = "hr_bpm" in df.columns

    if has_dual_ankle:
        # Right ankle drives the legacy pipeline (PLM detection, tilt, position)
        for k in ["acc_x_g", "acc_y_g", "acc_z_g",
                  "gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]:
            df[k] = pd.to_numeric(df[f"ankleR_{k}"], errors="coerce")
        for k in ["ankleL_acc_x_g", "ankleL_acc_y_g", "ankleL_acc_z_g",
                  "ankleL_gyro_x_dps", "ankleL_gyro_y_dps", "ankleL_gyro_z_dps",
                  "ankleR_acc_x_g", "ankleR_acc_y_g", "ankleR_acc_z_g",
                  "ankleR_gyro_x_dps", "ankleR_gyro_y_dps", "ankleR_gyro_z_dps"]:
            df[k] = pd.to_numeric(df[k], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["acc_x_g", "acc_y_g", "acc_z_g"]).reset_index(drop=True)
        if len(df) < before:
            print(f"  dropped {before - len(df)} rows missing right-ankle data")
    elif has_ankle_chest:
        # Legacy ankle+chest layout: ankle drives the primary pipeline
        df["acc_x_g"] = pd.to_numeric(df["ankle_acc_x_g"], errors="coerce")
        df["acc_y_g"] = pd.to_numeric(df["ankle_acc_y_g"], errors="coerce")
        df["acc_z_g"] = pd.to_numeric(df["ankle_acc_z_g"], errors="coerce")
        df["gyro_x_dps"] = pd.to_numeric(df["ankle_gyro_x_dps"], errors="coerce")
        df["gyro_y_dps"] = pd.to_numeric(df["ankle_gyro_y_dps"], errors="coerce")
        df["gyro_z_dps"] = pd.to_numeric(df["ankle_gyro_z_dps"], errors="coerce")
        for k in ["chest_acc_x_g", "chest_acc_y_g", "chest_acc_z_g",
                  "chest_gyro_x_dps", "chest_gyro_y_dps", "chest_gyro_z_dps"]:
            df[k] = pd.to_numeric(df[k], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["acc_x_g", "acc_y_g", "acc_z_g"]).reset_index(drop=True)
        if len(df) < before:
            print(f"  dropped {before - len(df)} rows missing ankle data")
    if has_hr:
        df["hr_bpm"] = pd.to_numeric(df["hr_bpm"], errors="coerce")
        df["rr_interval_ms"] = pd.to_numeric(df["rr_interval_ms"], errors="coerce")

    flags = []
    if has_dual_ankle: flags.append("dual-ankle")
    elif has_ankle_chest: flags.append("ankle+chest")
    elif "acc_x_g" in df.columns: flags.append("single-sensor")
    if has_hr and df["hr_bpm"].notna().any(): flags.append("HRM")
    print(f"  {len(df):,} rows after dedup, "
          f"{(df.t.iloc[-1] - df.t.iloc[0]).total_seconds()/3600:.2f} h span"
          + f"  [{', '.join(flags) if flags else 'unknown'}]")
    return df


def estimate_fs(df: pd.DataFrame) -> float:
    """Empirical sample rate (Hz)."""
    dt = df["t"].diff().dt.total_seconds().dropna()
    return 1.0 / dt.median()


# ---------------------- Detection ------------------------------------------

def compute_envelope(df: pd.DataFrame, fs: float) -> np.ndarray:
    a = np.sqrt(df.acc_x_g ** 2 + df.acc_y_g ** 2 + df.acc_z_g ** 2) - 1.0
    sos = signal.butter(4, [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
                        btype="bandpass", fs=fs, output="sos")
    bp = signal.sosfiltfilt(sos, a)
    rectified = np.abs(bp)
    win = max(1, int(ENVELOPE_WINDOW_S * fs))
    envelope = pd.Series(rectified).rolling(win, min_periods=1, center=True).mean().to_numpy()
    return envelope


def find_events(envelope: np.ndarray, t: pd.Series) -> pd.DataFrame:
    """Hysteresis threshold → events with start/end/peak."""
    events = []
    in_event = False
    start_i = 0
    peak = 0.0
    for i, v in enumerate(envelope):
        if not in_event:
            if v > T_ON:
                in_event = True
                start_i = i
                peak = v
        else:
            peak = max(peak, v)
            if v < T_OFF:
                in_event = False
                events.append((start_i, i, peak))
    if in_event:
        events.append((start_i, len(envelope) - 1, peak))

    rows = []
    for s_i, e_i, pk in events:
        start = t.iloc[s_i]
        end = t.iloc[e_i]
        dur = (end - start).total_seconds()
        rows.append({"start": start, "end": end, "duration_s": dur, "peak_g": pk})
    return pd.DataFrame(rows)


def find_tilt_changes(df: pd.DataFrame, fs: float, return_series: bool = False,
                      accel_cols: tuple[str, str, str] = ("acc_x_g", "acc_y_g", "acc_z_g")):
    """Detect rollovers from orientation change, not acceleration magnitude.

    Strategy: low-pass the accel signal to extract the gravity vector, which
    points 'down' relative to the sensor. When the body rotates, the gravity
    vector in sensor frame rotates the opposite way. Compare the current
    gravity direction to a rolling reference (the orientation a few seconds
    ago) — sustained changes >TILT_MIN_DEG are rollovers.

    `accel_cols` selects which 3 accel columns to use. Defaults to ankle.
    For dual-sensor recordings, pass chest columns to compute body-position
    tilt instead of foot tilt.
    """
    sos = signal.butter(4, TILT_LOWPASS_HZ, btype="lowpass", fs=fs, output="sos")
    cx, cy, cz = accel_cols
    gx = signal.sosfiltfilt(sos, df[cx].to_numpy())
    gy = signal.sosfiltfilt(sos, df[cy].to_numpy())
    gz = signal.sosfiltfilt(sos, df[cz].to_numpy())

    # Normalize to unit gravity vector
    norm = np.sqrt(gx**2 + gy**2 + gz**2)
    norm = np.where(norm < 1e-6, 1e-6, norm)
    ux, uy, uz = gx / norm, gy / norm, gz / norm

    # Reference: orientation TILT_REF_WINDOW_S seconds ago (rolling mean of
    # the unit vector, then renormalize). A real rollover settles into a new
    # orientation, so the angle between current and "recent past" stays large.
    win = max(1, int(TILT_REF_WINDOW_S * fs))
    rx = pd.Series(ux).rolling(win, min_periods=1).mean().to_numpy()
    ry = pd.Series(uy).rolling(win, min_periods=1).mean().to_numpy()
    rz = pd.Series(uz).rolling(win, min_periods=1).mean().to_numpy()
    rn = np.sqrt(rx**2 + ry**2 + rz**2)
    rn = np.where(rn < 1e-6, 1e-6, rn)
    rx, ry, rz = rx / rn, ry / rn, rz / rn

    # Angle between current and reference unit vectors (degrees)
    dot = np.clip(ux * rx + uy * ry + uz * rz, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))

    # Find spans where angle exceeds threshold for at least TILT_HOLD_S
    above = angle_deg > TILT_MIN_DEG
    hold_n = max(1, int(TILT_HOLD_S * fs))
    events = []
    in_evt = False
    start_i = 0
    peak = 0.0
    t = df["t"]
    for i, hi in enumerate(above):
        if not in_evt and hi:
            in_evt = True
            start_i = i
            peak = angle_deg[i]
        elif in_evt:
            peak = max(peak, angle_deg[i])
            if not hi:
                if i - start_i >= hold_n:
                    events.append((start_i, i, peak))
                in_evt = False
    if in_evt and len(above) - start_i >= hold_n:
        events.append((start_i, len(above) - 1, peak))

    rows = []
    for s_i, e_i, pk in events:
        rows.append({
            "start": t.iloc[s_i],
            "end": t.iloc[e_i],
            "duration_s": (t.iloc[e_i] - t.iloc[s_i]).total_seconds(),
            "peak_deg": pk,
        })
    out = pd.DataFrame(rows)
    if return_series:
        unit_g = np.column_stack([ux, uy, uz])
        return out, angle_deg, unit_g
    return out


def apply_tilt_rollovers(events: pd.DataFrame, tilt_events: pd.DataFrame) -> pd.DataFrame:
    """Reclassify events as rollover if they overlap a tilt event, and add
    tilt events that don't overlap anything as new rollover rows."""
    if tilt_events.empty:
        events["tilt_deg"] = 0.0
        return events
    events = events.copy()
    events["tilt_deg"] = 0.0

    matched_tilt = set()
    for ei, ev in events.iterrows():
        for ti, tilt in tilt_events.iterrows():
            # Overlap: tilt starts before event ends, tilt ends after event starts
            if tilt["start"] <= ev["end"] and tilt["end"] >= ev["start"]:
                events.at[ei, "tilt_deg"] = max(events.at[ei, "tilt_deg"], tilt["peak_deg"])
                matched_tilt.add(ti)

    # Add un-matched tilt events as new rollover rows
    new_rows = []
    for ti, tilt in tilt_events.iterrows():
        if ti in matched_tilt:
            continue
        new_rows.append({
            "start": tilt["start"], "end": tilt["end"],
            "duration_s": tilt["duration_s"],
            "peak_g": 0.0,                # came from tilt detector, not accel
            "tilt_deg": tilt["peak_deg"],
        })
    if new_rows:
        events = pd.concat([events, pd.DataFrame(new_rows)], ignore_index=True)
        events = events.sort_values("start").reset_index(drop=True)
    return events


# ---------------------- Position classifier --------------------------------

def load_calibration(path: Path) -> dict[str, np.ndarray]:
    """Load centroids from a calibration.json file.
    Supports both the old format ({label: [gx,gy,gz], ...}) and the new
    nested format with metadata."""
    obj = json.loads(path.read_text())
    if "centroids" in obj:
        raw = obj["centroids"]
    else:
        raw = obj
    return {label: np.array(vec) for label, vec in raw.items()}


def classify_positions(unit_g: np.ndarray, t: pd.Series, fs: float,
                       centroids: dict[str, np.ndarray],
                       envelope: np.ndarray | None = None,
                       window_s: float = 30.0,
                       upright_thresh_g: float = 0.04,
                       upright_frac: float = 0.40,
                       sim_floor: float = 0.80) -> pd.DataFrame:
    """For each ~window_s second window, classify the average gravity
    vector as nearest centroid (cosine similarity), with refinements:
      - if the window has sustained motion (envelope > upright_thresh_g
        for >= upright_frac of samples) -> 'upright' (walking/standing)
      - if best cosine similarity < sim_floor -> 'unknown'
      - otherwise nearest centroid label
    """
    win = max(1, int(window_s * fs))
    n = len(unit_g)
    rows = []
    labels = list(centroids.keys())
    cs = np.stack([centroids[l] for l in labels])  # (k, 3)
    for i in range(0, n, win):
        j = min(i + win, n)
        # Upright detection (A): sustained motion over the window
        is_upright = False
        if envelope is not None:
            seg = envelope[i:j]
            if len(seg) > 0:
                frac_active = float((seg > upright_thresh_g).mean())
                if frac_active >= upright_frac:
                    is_upright = True

        v = unit_g[i:j].mean(axis=0)
        nrm = np.linalg.norm(v)
        if nrm < 1e-6:
            label = "unknown"
            sim = 0.0
        else:
            v = v / nrm
            sims = cs @ v
            best = int(np.argmax(sims))
            sim = float(sims[best])
            if is_upright:
                label = "upright"
            elif sim < sim_floor:
                label = "unknown"
            else:
                label = labels[best]
        rows.append({"start": t.iloc[i], "end": t.iloc[j-1],
                     "label": label, "similarity": sim})
    return pd.DataFrame(rows)


def compute_breathing_rate(df: pd.DataFrame, fs: float,
                           accel_cols: tuple[str, str, str] = ("chest_acc_x_g", "chest_acc_y_g", "chest_acc_z_g"),
                           window_s: float = 60.0,
                           hop_s: float = 30.0,
                           low_hz: float = 0.1,
                           high_hz: float = 0.6) -> pd.DataFrame:
    """Estimate breaths-per-minute from chest accelerometer in sliding windows.

    Method: bandpass filter the magnitude of the chest accel signal to the
    normal breathing range (6–36 BPM = 0.1–0.6 Hz), then for each window
    find the dominant frequency via FFT. Returns a DataFrame with one row
    per window: {start, end, bpm, signal_strength}.

    `signal_strength` is the FFT peak height divided by the median bin
    height — high values mean a clear breathing signal, low values mean
    noisy / no breathing detected (could be apnea, or sensor not on chest).
    """
    cx, cy, cz = accel_cols
    if not all(c in df.columns for c in (cx, cy, cz)):
        return pd.DataFrame()

    # Use chest accel magnitude (sqrt(x²+y²+z²)) as the breathing-modulated signal
    a = np.sqrt(df[cx]**2 + df[cy]**2 + df[cz]**2)
    a = a.ffill().bfill().to_numpy()
    if len(a) < int(window_s * fs):
        return pd.DataFrame()

    # Bandpass to breathing range
    sos = signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    bp = signal.sosfiltfilt(sos, a)

    win_n = int(window_s * fs)
    hop_n = int(hop_s * fs)
    rows = []
    t = df["t"]
    for i in range(0, len(bp) - win_n + 1, hop_n):
        seg = bp[i:i + win_n]
        # FFT to find dominant frequency
        spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)
        # Restrict to breathing range
        valid = (freqs >= low_hz) & (freqs <= high_hz)
        if not valid.any():
            continue
        sp_v = spectrum[valid]
        fq_v = freqs[valid]
        peak_idx = int(np.argmax(sp_v))
        peak_freq = float(fq_v[peak_idx])
        peak_strength = float(sp_v[peak_idx])
        median_strength = float(np.median(sp_v)) + 1e-9
        rows.append({
            "start": t.iloc[i],
            "end": t.iloc[i + win_n - 1],
            "bpm": peak_freq * 60.0,
            "signal_strength": peak_strength / median_strength,
        })
    return pd.DataFrame(rows)


def load_hr_series(df: pd.DataFrame) -> pd.DataFrame:
    """Extract a clean HR time series from the recording. The HRM updates
    once per heartbeat (~1 Hz), but our 30 Hz CSV polling repeats the same
    HR value many times. We dedupe by keeping rows where HR changes and
    where the timestamp differs from the previous row meaningfully."""
    if "hr_bpm" not in df.columns:
        return pd.DataFrame()
    hr = df[["t", "hr_bpm"]].dropna().copy()
    hr = hr[hr["hr_bpm"] > 0]
    if hr.empty:
        return pd.DataFrame()
    # Keep only rows where HR changed from the previous row (BLE updates
    # are change-driven so this gives us per-beat resolution)
    hr["changed"] = hr["hr_bpm"].diff().ne(0)
    hr = hr[hr["changed"] | (hr.index == hr.index[0])].drop(columns=["changed"])
    return hr.reset_index(drop=True)


def detect_hr_arousals(hr: pd.DataFrame,
                       spike_bpm: float = 10.0,
                       spike_window_s: float = 30.0,
                       refractory_s: float = 60.0) -> pd.DataFrame:
    """Find HR spike events suggestive of arousal. An arousal is defined
    as an HR rise of >= spike_bpm over a window <= spike_window_s.
    Returns DataFrame with one row per arousal: {start, end, baseline,
    peak, delta_bpm}.

    With refractory_s, we suppress repeat detections from the same
    physiological arousal (HR doesn't drop back to baseline instantly).
    """
    if hr.empty or len(hr) < 5:
        return pd.DataFrame()
    hr = hr.sort_values("t").reset_index(drop=True)
    times = hr["t"].to_numpy()
    bpm = hr["hr_bpm"].to_numpy()
    n = len(hr)

    rows = []
    last_arousal_end_s = -np.inf
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    times_s = (hr["t"] - epoch).dt.total_seconds().to_numpy()

    for i in range(n):
        if times_s[i] - last_arousal_end_s < refractory_s:
            continue
        # Look at the window backward to estimate baseline (median of HR
        # in the 60s before this sample, excluding the last 5s which may
        # already be rising)
        window_lo = times_s[i] - 60.0
        window_hi = times_s[i] - 5.0
        baseline_mask = (times_s >= window_lo) & (times_s <= window_hi)
        if baseline_mask.sum() < 3:
            continue
        baseline = float(np.median(bpm[baseline_mask]))
        # Look forward up to spike_window_s for the peak
        peak_window_hi = times_s[i] + spike_window_s
        peak_mask = (times_s >= times_s[i]) & (times_s <= peak_window_hi)
        if peak_mask.sum() < 2:
            continue
        peak = float(np.max(bpm[peak_mask]))
        delta = peak - baseline
        if delta >= spike_bpm:
            peak_idx = int(np.argmax(bpm[peak_mask])) + np.where(peak_mask)[0][0]
            rows.append({
                "start": hr["t"].iloc[i],
                "end": hr["t"].iloc[peak_idx],
                "baseline": baseline,
                "peak": peak,
                "delta_bpm": delta,
            })
            last_arousal_end_s = times_s[peak_idx]
    return pd.DataFrame(rows)


def link_arousals_to_plms(events: pd.DataFrame, arousals: pd.DataFrame,
                          window_s: float = 30.0) -> pd.DataFrame:
    """For each PLM event, find the most recent HR arousal that started
    within `window_s` seconds before the PLM. This tests the apnea→arousal
    →PLM hypothesis: arousal precedes PLM by 0–30 seconds."""
    if events.empty or arousals.empty:
        events = events.copy()
        events["preceded_by_arousal_s"] = np.nan
        return events
    events = events.copy()
    events["preceded_by_arousal_s"] = np.nan
    # Work in UTC seconds-since-epoch as plain floats to avoid numpy
    # timezone-handling complications.
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    arousal_s = (pd.to_datetime(arousals["start"], utc=True) - epoch).dt.total_seconds().to_numpy()
    event_s = (pd.to_datetime(events["start"], utc=True) - epoch).dt.total_seconds().to_numpy()
    for i in range(len(events)):
        gaps = event_s[i] - arousal_s
        valid = gaps[(gaps >= 0) & (gaps <= window_s)]
        if len(valid) > 0:
            events.iloc[i, events.columns.get_loc("preceded_by_arousal_s")] = float(valid.min())
    return events


def overlay_awake_to_unknown(positions: pd.DataFrame,
                             sleep_session: dict) -> pd.DataFrame:
    """B: when Oura says we're awake, downgrade the position from one of the
    4 lying categories to 'awake_unknown' (preserve 'upright' if already set)."""
    if not sleep_session or positions.empty:
        return positions
    positions = positions.copy()
    spm = sleep_session.get("sleep_phase_5_min") or ""
    bs = pd.to_datetime(sleep_session["bedtime_start"]).tz_convert("UTC")
    for idx, row in positions.iterrows():
        t = row["start"].tz_convert("UTC")
        epoch = int((t - bs).total_seconds() // 300)
        if 0 <= epoch < len(spm):
            stage_char = spm[epoch]
            if stage_char == "4" and row["label"] in ("back", "left", "right", "stomach"):
                positions.at[idx, "label"] = "awake_unknown"
    return positions


def merge_close_events(events: pd.DataFrame, gap_s: float = MERGE_GAP_S) -> pd.DataFrame:
    """Merge events whose end-to-next-start gap is < gap_s.
    Real rollovers / walking around fragment into many short sub-events because
    the envelope dips below T_off briefly between limb motions. Glue them back
    together so the duration classifier sees the true movement length."""
    if events.empty:
        return events
    has_tilt = "tilt_deg" in events.columns
    events = events.sort_values("start").reset_index(drop=True)
    merged = []
    cur_start = events.loc[0, "start"]
    cur_end = events.loc[0, "end"]
    cur_peak = events.loc[0, "peak_g"]
    cur_tilt = events.loc[0, "tilt_deg"] if has_tilt else 0.0
    for i in range(1, len(events)):
        gap = (events.loc[i, "start"] - cur_end).total_seconds()
        if gap < gap_s:
            cur_end = max(cur_end, events.loc[i, "end"])
            cur_peak = max(cur_peak, events.loc[i, "peak_g"])
            if has_tilt:
                cur_tilt = max(cur_tilt, events.loc[i, "tilt_deg"])
        else:
            row = {"start": cur_start, "end": cur_end,
                   "duration_s": (cur_end - cur_start).total_seconds(),
                   "peak_g": cur_peak}
            if has_tilt:
                row["tilt_deg"] = cur_tilt
            merged.append(row)
            cur_start = events.loc[i, "start"]
            cur_end = events.loc[i, "end"]
            cur_peak = events.loc[i, "peak_g"]
            cur_tilt = events.loc[i, "tilt_deg"] if has_tilt else 0.0
    row = {"start": cur_start, "end": cur_end,
           "duration_s": (cur_end - cur_start).total_seconds(),
           "peak_g": cur_peak}
    if has_tilt:
        row["tilt_deg"] = cur_tilt
    merged.append(row)
    return pd.DataFrame(merged)


def classify_by_duration(events: pd.DataFrame) -> pd.DataFrame:
    def label(d):
        if d < EVENT_MIN_S: return "noise"
        if d <= EVENT_MAX_S: return "movement"            # candidate PLM
        if d <= ROLLOVER_MAX_S: return "rollover"
        return "out_of_bed"
    events = events.copy()
    events["class"] = events["duration_s"].apply(label)

    has_tilt = "tilt_deg" in events.columns
    tilt = events["tilt_deg"] if has_tilt else pd.Series(0.0, index=events.index)

    # 1) Tilt-based: any event with sustained orientation change is a rollover.
    tilted = tilt >= TILT_MIN_DEG
    events.loc[tilted & (events["class"] != "out_of_bed"), "class"] = "rollover"

    # 2) Kick detection: high-peak, brief, *no* significant tilt -> leg kick.
    #    Must run before the generic peak->rollover rule so kicks aren't mislabeled.
    is_kick = (
        (events["peak_g"] > KICK_MIN_PEAK_G)
        & (events["duration_s"] <= KICK_MAX_DURATION_S)
        & (tilt < KICK_MAX_TILT_DEG)
        & (events["class"].isin(["movement", "noise"]))
    )
    events.loc[is_kick, "class"] = "kick"

    # 3) Remaining big-peak long movements -> rollover.
    big = (events["class"] == "movement") & (events["peak_g"] > PLM_PEAK_MAX_G)
    events.loc[big & (events["duration_s"] >= ROLLOVER_MIN_S), "class"] = "rollover"
    return events


def tag_series(events: pd.DataFrame) -> pd.DataFrame:
    """Group `movement`-class events into PLM series.
    Clinical scoring rules: >=4 events with 5-90s gaps, *during sleep only*.
    Events tagged `awake` (or out_of_window/unknown) cannot be PLMs."""
    events = events.copy()
    events["episode_id"] = -1
    events["final_class"] = events["class"]

    # Only consider candidate movements during actual sleep
    sleep_stages_ok = {"deep", "light", "rem"}
    if "sleep_stage" in events.columns:
        eligible_mask = (events["class"] == "movement") & events["sleep_stage"].isin(sleep_stages_ok)
    else:
        eligible_mask = (events["class"] == "movement")

    movements = events[eligible_mask].sort_values("start").reset_index()
    if movements.empty:
        # Anything still labeled "movement" but not eligible -> isolated_movement
        events.loc[events["class"] == "movement", "final_class"] = "isolated_movement"
        return events

    gap = movements["start"].diff().dt.total_seconds().fillna(SERIES_MAX_GAP_S + 1)
    breaks = (gap < SERIES_MIN_GAP_S) | (gap > SERIES_MAX_GAP_S)
    group_id = breaks.cumsum()
    movements["episode_id"] = group_id

    counts = movements.groupby("episode_id").size()
    real_episodes = counts[counts >= SERIES_MIN_COUNT].index

    for ep_id, sub in movements.groupby("episode_id"):
        if ep_id in real_episodes:
            for orig_idx in sub["index"]:
                events.at[orig_idx, "episode_id"] = ep_id
                events.at[orig_idx, "final_class"] = "PLM_in_series"
        else:
            for orig_idx in sub["index"]:
                events.at[orig_idx, "final_class"] = "isolated_movement"

    # Movements that were ineligible (awake, etc.) -> isolated_movement
    still_movement = events["final_class"] == "movement"
    events.loc[still_movement, "final_class"] = "isolated_movement"
    return events


# ---------------------- Oura -----------------------------------------------

OURA_BASE = "https://api.ouraring.com/v2/usercollection"
STAGE_NAMES = {"1": "deep", "2": "light", "3": "rem", "4": "awake"}


def fetch_oura_sleep(pat: str, day_start: datetime, day_end: datetime):
    """Find the long_sleep session that overlaps our recording."""
    url = f"{OURA_BASE}/sleep"
    params = {"start_date": day_start.strftime("%Y-%m-%d"),
              "end_date": (day_end + timedelta(days=1)).strftime("%Y-%m-%d")}
    r = requests.get(url, headers={"Authorization": f"Bearer {pat}"}, params=params, timeout=15)
    r.raise_for_status()
    sessions = r.json().get("data", [])
    # Pick the long_sleep session whose window most overlaps ours
    best = None
    best_overlap = timedelta(0)
    for s in sessions:
        if s.get("type") != "long_sleep":
            continue
        bs = pd.to_datetime(s["bedtime_start"]).tz_convert("UTC")
        be = pd.to_datetime(s["bedtime_end"]).tz_convert("UTC")
        overlap = max(timedelta(0), min(be, day_end) - max(bs, day_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best = s
    return best


def stage_for_timestamp(t: pd.Timestamp, sleep_session: dict) -> str:
    if not sleep_session:
        return "unknown"
    bs = pd.to_datetime(sleep_session["bedtime_start"]).tz_convert("UTC")
    be = pd.to_datetime(sleep_session["bedtime_end"]).tz_convert("UTC")
    if t < bs or t > be:
        return "out_of_window"
    spm = sleep_session.get("sleep_phase_5_min") or ""
    epoch = int((t - bs).total_seconds() // 300)
    if epoch >= len(spm):
        return "out_of_window"
    return STAGE_NAMES.get(spm[epoch], "unknown")


# ---------------------- Output --------------------------------------------

def load_night_log_row(log_path: Path, recording_start: pd.Timestamp) -> dict | None:
    """Look up the night_log.csv row matching the recording's local date.
    Recording started in evening -> use that calendar date as the night key.
    Accepts ISO (2026-04-29), US (4/29/26), and US-long (4/29/2026) date formats."""
    if not log_path.exists():
        return None
    try:
        log = pd.read_csv(log_path, dtype=str).fillna("")
    except Exception:
        return None
    target_dt = recording_start.tz_convert("America/Los_Angeles")
    candidates = {
        target_dt.strftime("%Y-%m-%d"),
        target_dt.strftime("%-m/%-d/%y"),
        target_dt.strftime("%-m/%-d/%Y"),
    }
    match = log[log["date"].isin(candidates)]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def write_summary(events: pd.DataFrame, sleep_session: dict, out_path: Path,
                  positions: pd.DataFrame | None = None,
                  night_log_row: dict | None = None,
                  left_events: pd.DataFrame | None = None) -> str:
    lines = []
    lines.append(f"Twitch Recorder analysis — {len(events)} raw events detected")
    lines.append("")
    if night_log_row:
        lines.append("Night log:")
        for k in ["date", "gabapentin_mg", "bedtime", "caffeine_cutoff", "alcohol",
                 "exercise", "sex", "stress_1to5", "last_meal", "notes"]:
            v = night_log_row.get(k, "")
            if v:
                lines.append(f"  {k:<18} {v}")
        lines.append("")
    lines.append("By classification:")
    for cls, n in events["final_class"].value_counts().items():
        lines.append(f"  {cls:<22}  {n}")
    lines.append("")
    plm_events = events[events["final_class"] == "PLM_in_series"]
    if not plm_events.empty:
        n_episodes = plm_events["episode_id"].nunique()
        lines.append(f"PLM episodes (≥4 events, 5–90s gaps): {n_episodes}")
        lines.append(f"  events per episode: "
                     f"min={plm_events.groupby('episode_id').size().min()}, "
                     f"max={plm_events.groupby('episode_id').size().max()}")
    lines.append("")
    if "sleep_stage" in events.columns:
        lines.append("Events by sleep stage:")
        ct = pd.crosstab(events["final_class"], events["sleep_stage"])
        lines.append(ct.to_string())
    if "position" in events.columns:
        lines.append("")
        lines.append("Events by sleep position:")
        ct = pd.crosstab(events["final_class"], events["position"])
        lines.append(ct.to_string())
        # Time spent in each position
        if positions is not None and not positions.empty:
            lines.append("")
            lines.append("Time in each position (entire recording):")
            dur_min = positions.groupby("label").size() * 0.5  # 30s windows -> minutes
            for label, m in dur_min.sort_values(ascending=False).items():
                lines.append(f"  {label:<10}  {m:.1f} min")
    # Per-leg breakdown if dual-ankle data
    if "leg" in events.columns:
        lines.append("")
        lines.append("Per-leg event counts:")
        n_right  = (events["leg"] == "right").sum()
        n_left   = (events["leg"] == "left").sum()
        n_bilat  = events.get("bilateral", pd.Series(dtype=bool)).sum() if "bilateral" in events.columns else 0
        n_uni_r  = n_right - n_bilat
        lines.append(f"  Right leg events (incl. bilateral): {n_right}")
        lines.append(f"  Left  leg events (unilateral only): {n_left}")
        lines.append(f"  Bilateral (both legs within 1s):    {n_bilat}")
        lines.append(f"  Unilateral right only:              {n_uni_r}")
        total = n_right + n_left
        if total > 0:
            # Asymmetry: counts bilateral as right, so compare right-only vs left-only
            asym = (n_uni_r - n_left) / total
            lines.append(f"  Asymmetry index (R_uni - L_uni) / total: {asym:+.2f}")
        plm_right = events[(events["final_class"] == "PLM_in_series") & (events["leg"] == "right")]
        plm_left  = events[(events["final_class"] == "PLM_in_series") & (events["leg"] == "left")]
        if len(plm_right) + len(plm_left) > 0:
            lines.append(f"  PLMs — right: {len(plm_right)}  left: {len(plm_left)}")
    if sleep_session:
        lines.append("")
        lines.append(f"Oura window: {sleep_session['bedtime_start']} → {sleep_session['bedtime_end']}")
    text = "\n".join(lines)
    out_path.write_text(text)
    return text


def plot_overview(df: pd.DataFrame, envelope: np.ndarray, events: pd.DataFrame,
                  sleep_session: dict, out_path: Path):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 3]})
    fig.suptitle(out_path.stem.replace("_overview", ""))

    # Hypnogram band
    if sleep_session:
        bs = pd.to_datetime(sleep_session["bedtime_start"]).tz_convert("UTC")
        spm = sleep_session.get("sleep_phase_5_min") or ""
        for i, ch in enumerate(spm):
            color = {"1": "#08306b", "2": "#6baed6", "3": "#9e9ac8", "4": "#fee08b"}.get(ch, "#cccccc")
            t0 = bs + timedelta(minutes=5 * i)
            t1 = bs + timedelta(minutes=5 * (i + 1))
            ax0.axvspan(t0, t1, color=color, alpha=0.9)
    ax0.set_yticks([])
    ax0.set_title("Oura sleep stages (deep=dark blue, light=blue, REM=purple, awake=yellow)")

    # Envelope + events
    times = df["t"].dt.tz_convert("UTC")
    ax1.plot(times, envelope, lw=0.4, color="#333333")
    ax1.axhline(T_ON, color="red", ls="--", lw=0.5, label=f"T_on={T_ON}g")
    color_map = {
        "PLM_in_series": "red",
        "isolated_movement": "#888888",
        "kick": "#2ca02c",
        "rollover": "orange",
        "out_of_bed": "purple",
        "noise": None,
    }
    for cls, color in color_map.items():
        if color is None:
            continue
        sub = events[events["final_class"] == cls]
        if sub.empty:
            continue
        ax1.scatter(sub["start"].dt.tz_convert("UTC"), sub["peak_g"],
                    color=color, s=10, label=f"{cls} ({len(sub)})", zorder=3)
    ax1.set_ylabel("envelope (g)")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=None))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_annotated(df: pd.DataFrame, envelope: np.ndarray, tilt_series: np.ndarray,
                   events: pd.DataFrame, sleep_session: dict, out_path: Path,
                   positions: pd.DataFrame | None = None,
                   night_log_row: dict | None = None,
                   hr_series: pd.DataFrame | None = None,
                   arousals: pd.DataFrame | None = None,
                   breathing: pd.DataFrame | None = None):
    """Annotated timeline focused on the actual sleep window:
       1. Hypnogram (sleep stages)
       2. Sleep position (if calibration provided)
       3. Accel envelope (g) with PLM and kick markers
       4. Tilt angle (deg) with rollover markers
       5. Event timeline strip showing every event by class
    """
    LOCAL_TZ = "America/Los_Angeles"

    # X-axis range: zoom to Oura window if we have one, else full data span
    if sleep_session:
        t_lo = pd.to_datetime(sleep_session["bedtime_start"]).tz_convert(LOCAL_TZ)
        t_hi = pd.to_datetime(sleep_session["bedtime_end"]).tz_convert(LOCAL_TZ)
        # Pad slightly so events near the edges are visible
        t_lo = t_lo - timedelta(minutes=10)
        t_hi = t_hi + timedelta(minutes=10)
    else:
        t_lo = df["t"].iloc[0].tz_convert(LOCAL_TZ)
        t_hi = df["t"].iloc[-1].tz_convert(LOCAL_TZ)

    has_positions = positions is not None and not positions.empty
    has_hr = hr_series is not None and not hr_series.empty
    has_breathing = breathing is not None and not breathing.empty

    # Build the panel list dynamically based on what data we have
    panels = ["hypnogram"]
    if has_positions: panels.append("position")
    panels.append("envelope")
    panels.append("tilt")
    if has_hr: panels.append("hr")
    if has_breathing: panels.append("breathing")
    panels.append("events")

    height_map = {
        "hypnogram": 0.7, "position": 0.7,
        "envelope": 3.0, "tilt": 2.0,
        "hr": 1.5, "breathing": 1.0,
        "events": 1.5,
    }
    heights = [height_map[p] for p in panels]
    fig_height = max(8, sum(heights) * 0.9)

    fig, axes = plt.subplots(len(panels), 1, figsize=(18, fig_height), sharex=True,
                             gridspec_kw={"height_ratios": heights})
    if len(panels) == 1: axes = [axes]
    panel_axes = dict(zip(panels, axes))
    ax_h = panel_axes["hypnogram"]
    ax_p = panel_axes.get("position")
    ax_e = panel_axes["envelope"]
    ax_t = panel_axes["tilt"]
    ax_hr = panel_axes.get("hr")
    ax_br = panel_axes.get("breathing")
    ax_s = panel_axes["events"]
    title = out_path.stem.replace("_annotated", "") + " — annotated night"
    if night_log_row:
        bits = []
        for label, key in [("gabapentin", "gabapentin_mg"), ("bedtime", "bedtime"),
                           ("alcohol", "alcohol"), ("exercise", "exercise"),
                           ("sex", "sex"), ("stress", "stress_1to5"),
                           ("caffeine cutoff", "caffeine_cutoff"),
                           ("last meal", "last_meal"), ("notes", "notes")]:
            v = night_log_row.get(key, "")
            if v and v != "no":
                bits.append(f"{label}: {v}")
        subtitle = "  •  ".join(bits)
        fig.suptitle(f"{title}\n{subtitle}", fontsize=12)
    else:
        fig.suptitle(title, fontsize=14)

    # --- Panel 1: Hypnogram ----------------------------------------------
    if sleep_session:
        bs = pd.to_datetime(sleep_session["bedtime_start"]).tz_convert(LOCAL_TZ)
        spm = sleep_session.get("sleep_phase_5_min") or ""
        stage_color = {"1": "#08306b", "2": "#6baed6", "3": "#9e9ac8", "4": "#fee08b"}
        for i, ch in enumerate(spm):
            color = stage_color.get(ch, "#cccccc")
            t0 = bs + timedelta(minutes=5 * i)
            t1 = bs + timedelta(minutes=5 * (i + 1))
            ax_h.axvspan(t0, t1, color=color, alpha=0.95)
    ax_h.set_yticks([])
    ax_h.set_title("Sleep stages  (deep = dark blue,  light = light blue,  REM = purple,  awake = yellow)",
                   loc="left", fontsize=10)

    # --- Panel 1b: Sleep position ---------------------------------------
    if has_positions:
        position_color = {
            "back":    "#1f77b4",   # blue
            "right":   "#2ca02c",   # green
            "stomach": "#d62728",   # red
            "left":    "#9467bd",   # purple
            "upright": "#fee08b",   # yellow (matches awake)
            "awake_unknown": "#fdae6b",  # orange
            "unknown": "#cccccc",
        }
        for _, row in positions.iterrows():
            t0 = row["start"].tz_convert(LOCAL_TZ)
            t1 = row["end"].tz_convert(LOCAL_TZ)
            color = position_color.get(row["label"], "#cccccc")
            ax_p.axvspan(t0, t1, color=color, alpha=0.85)
        ax_p.set_yticks([])
        ax_p.set_title("Sleep position  (back=blue, right=green, stomach=red, left=purple, "
                       "upright=yellow, awake=orange, unknown=gray)",
                       loc="left", fontsize=10)

    # --- Panel 2: Accel envelope -----------------------------------------
    times = df["t"].dt.tz_convert(LOCAL_TZ)
    ax_e.plot(times, envelope, lw=0.4, color="#444444")
    ax_e.axhline(T_ON, color="red", ls="--", lw=0.5, alpha=0.7)
    ax_e.set_ylabel("movement strength (g)")
    ax_e.set_ylim(0, 0.6)  # cap so PLM-scale events are visible
    ax_e.text(0.005, 0.92, "(clipped at 0.6g — bigger spikes shown as ↑ markers)",
              transform=ax_e.transAxes, fontsize=8, color="#666")

    # PLM markers on envelope panel — split by leg if available
    plm = events[events["final_class"] == "PLM_in_series"]
    if not plm.empty:
        if "leg" in plm.columns:
            plm_r = plm[plm["leg"] == "right"]
            plm_l = plm[plm["leg"] == "left"]
            plm_b = plm[plm.get("bilateral", pd.Series(False, index=plm.index))]
            if not plm_r.empty:
                ax_e.scatter(plm_r["start"].dt.tz_convert(LOCAL_TZ),
                             plm_r["peak_g"].clip(upper=0.58),
                             color="red", s=40, marker="o", zorder=4,
                             label=f"PLM right ({len(plm_r)})", edgecolors="white", linewidths=0.5)
            if not plm_l.empty:
                ax_e.scatter(plm_l["start"].dt.tz_convert(LOCAL_TZ),
                             plm_l["peak_g"].clip(upper=0.58),
                             color="#e377c2", s=40, marker="o", zorder=4,
                             label=f"PLM left ({len(plm_l)})", edgecolors="white", linewidths=0.5)
            if not plm_b.empty:
                ax_e.scatter(plm_b["start"].dt.tz_convert(LOCAL_TZ),
                             plm_b["peak_g"].clip(upper=0.58),
                             color="purple", s=50, marker="*", zorder=5,
                             label=f"PLM bilateral ({len(plm_b)})", edgecolors="white", linewidths=0.5)
        else:
            ax_e.scatter(plm["start"].dt.tz_convert(LOCAL_TZ),
                         plm["peak_g"].clip(upper=0.58),
                         color="red", s=40, marker="o", zorder=4,
                         label=f"PLM ({len(plm)})", edgecolors="white", linewidths=0.5)

    # Kicks on envelope panel
    kicks = events[events["final_class"] == "kick"]
    if not kicks.empty:
        ax_e.scatter(kicks["start"].dt.tz_convert(LOCAL_TZ),
                     [0.55] * len(kicks),
                     color="#2ca02c", s=80, marker="^", zorder=4,
                     label=f"kick ({len(kicks)})", edgecolors="white", linewidths=0.5)

    # Off-scale arrow markers for events with peak_g > 0.6
    big = events[(events["peak_g"] > 0.6)
                 & (events["final_class"].isin(["rollover", "kick", "isolated_movement", "out_of_bed"]))]
    if not big.empty:
        ax_e.scatter(big["start"].dt.tz_convert(LOCAL_TZ),
                     [0.58] * len(big),
                     color="#999999", s=15, marker=(3, 0, 0), zorder=3, alpha=0.6)

    ax_e.legend(loc="upper right", fontsize=9)
    ax_e.grid(axis="y", alpha=0.2)

    # --- Panel 3: Tilt angle ----------------------------------------------
    ax_t.plot(times, tilt_series, lw=0.5, color="#1f4e79")
    ax_t.axhline(TILT_MIN_DEG, color="orange", ls="--", lw=0.5, alpha=0.7)
    ax_t.set_ylabel("tilt vs recent (deg)")
    ax_t.set_ylim(0, 180)
    ax_t.set_yticks([0, 30, 60, 90, 120, 150, 180])

    rolls = events[events["final_class"] == "rollover"]
    if not rolls.empty:
        ax_t.scatter(rolls["start"].dt.tz_convert(LOCAL_TZ),
                     rolls["tilt_deg"],
                     color="orange", s=40, marker="o", zorder=4,
                     label=f"rollover ({len(rolls)})", edgecolors="white", linewidths=0.5)
    ax_t.legend(loc="upper right", fontsize=9)
    ax_t.grid(axis="y", alpha=0.2)

    # --- Panel: Heart rate + arousals ------------------------------------
    if ax_hr is not None and has_hr:
        hr_t = hr_series["t"].dt.tz_convert(LOCAL_TZ)
        ax_hr.plot(hr_t, hr_series["hr_bpm"], lw=0.5, color="#c0392b")
        ax_hr.set_ylabel("heart rate (bpm)")
        ax_hr.grid(axis="y", alpha=0.2)
        if arousals is not None and not arousals.empty:
            ax_hr.scatter(arousals["start"].dt.tz_convert(LOCAL_TZ),
                          arousals["peak"],
                          color="#e67e22", s=30, marker="^", zorder=4,
                          label=f"arousal ({len(arousals)})",
                          edgecolors="white", linewidths=0.5)
            ax_hr.legend(loc="upper right", fontsize=9)

    # --- Panel: Breathing rate ------------------------------------------
    if ax_br is not None and has_breathing:
        # Only plot windows with strong breathing signal (above noise)
        good = breathing[breathing["signal_strength"] > 2.0]
        if not good.empty:
            br_t = good["start"].dt.tz_convert(LOCAL_TZ)
            ax_br.plot(br_t, good["bpm"], lw=0.7, color="#2980b9", marker=".", markersize=3)
        ax_br.set_ylabel("breathing (bpm)")
        ax_br.set_ylim(6, 30)
        ax_br.grid(axis="y", alpha=0.2)

    # --- Panel: Event timeline strip -------------------------------------
    # Each class gets its own y-row. PLMs split into right / left / bilateral.
    has_legs = "leg" in events.columns
    sleep_only = {"deep", "light", "rem"}

    if has_legs:
        rows = [
            ("PLM_right",         "red",       "PLM right"),
            ("PLM_left",          "#e377c2",   "PLM left"),
            ("PLM_bilateral",     "purple",    "PLM bilateral"),
            ("kick",              "#2ca02c",   "kick"),
            ("rollover",          "orange",    "rollover"),
            ("isolated_movement", "#888888",   "isolated"),
            ("out_of_bed",        "#7f7f7f",   "out of bed"),
        ]
        # Build pseudo-class subsets for the PLM rows
        plm_all = events[events["final_class"] == "PLM_in_series"]
        bilat_mask = plm_all.get("bilateral", pd.Series(False, index=plm_all.index)).astype(bool)
        subsets = {
            "PLM_right":     plm_all[(plm_all["leg"] == "right") & ~bilat_mask],
            "PLM_left":      plm_all[(plm_all["leg"] == "left")  & ~bilat_mask],
            "PLM_bilateral": plm_all[bilat_mask],
            "kick":          events[events["final_class"] == "kick"],
            "rollover":      events[events["final_class"] == "rollover"],
            "isolated_movement": events[events["final_class"] == "isolated_movement"],
            "out_of_bed":    events[events["final_class"] == "out_of_bed"],
        }

        def row_label_leg(key, label):
            sub = subsets[key]
            n = len(sub)
            if key.startswith("PLM"):
                n_ep = sub["episode_id"].nunique() if not sub.empty and "episode_id" in sub.columns else "?"
                return f"{label}  ({n_ep} ep, {n} events)"
            if "sleep_stage" in events.columns:
                n_asleep = sub["sleep_stage"].isin(sleep_only).sum() if not sub.empty else 0
                return f"{label}  (n={n}, asleep={n_asleep})"
            return f"{label}  (n={n})"

        for y, (key, color, label) in enumerate(rows):
            sub = subsets[key]
            if sub.empty:
                continue
            x = sub["start"].dt.tz_convert(LOCAL_TZ)
            ax_s.scatter(x, [y] * len(sub), color=color, s=30, marker="|", linewidths=2)
        ax_s.set_yticks(range(len(rows)))
        ax_s.set_yticklabels([row_label_leg(k, l) for k, _, l in rows], fontsize=9)
        ax_s.set_ylim(-0.5, len(rows) - 0.5)
    else:
        rows = [
            ("PLM_in_series",      "red",       "PLM"),
            ("kick",               "#2ca02c",   "kick"),
            ("rollover",           "orange",    "rollover"),
            ("isolated_movement",  "#888888",   "isolated"),
            ("out_of_bed",         "purple",    "out of bed"),
        ]

        def row_label(cls, label):
            sub = events[events.final_class == cls]
            n_total = len(sub)
            if cls == "PLM_in_series":
                n_episodes = sub["episode_id"].nunique() if not sub.empty else 0
                return f"{label}  ({n_episodes} episodes, {n_total} events)"
            if "sleep_stage" in events.columns:
                n_asleep = ((events.final_class == cls)
                            & events.sleep_stage.isin(sleep_only)).sum()
                return f"{label}  (n={n_total}, asleep={n_asleep})"
            return f"{label}  (n={n_total})"

        for y, (cls, color, label) in enumerate(rows):
            sub = events[events["final_class"] == cls]
            if sub.empty:
                continue
            x = sub["start"].dt.tz_convert(LOCAL_TZ)
            ax_s.scatter(x, [y] * len(sub), color=color, s=30, marker="|", linewidths=2)
        ax_s.set_yticks(range(len(rows)))
        ax_s.set_yticklabels([row_label(cls, label) for cls, _, label in rows], fontsize=9)
        ax_s.set_ylim(-0.5, len(rows) - 0.5)

    ax_s.invert_yaxis()
    ax_s.grid(axis="y", alpha=0.2)
    ax_s.set_xlabel("local time")

    # Format x-axis as local-time HH:MM
    locator = mdates.HourLocator(interval=1)
    ax_s.xaxis.set_major_locator(locator)
    ax_s.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=times.iloc[0].tzinfo))
    ax_s.set_xlim(t_lo, t_hi)
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------- Main -----------------------------------------------

def main(csv_path: Path, calibration_path: Path | None = None):
    pat = os.environ.get("OURA_PAT")
    if not pat:
        print("WARN: OURA_PAT env var not set — Oura join will be skipped", file=sys.stderr)

    df = load_csv(csv_path)
    fs = estimate_fs(df)
    print(f"  empirical fs = {fs:.2f} Hz")

    print("computing envelope…")
    envelope = compute_envelope(df, fs)

    print("finding events…")
    raw_events = find_events(envelope, df["t"])
    print(f"  {len(raw_events)} raw events")

    print("finding tilt-based rollovers (ankle gravity vector)…")
    tilt_events, tilt_series, unit_g = find_tilt_changes(df, fs, return_series=True)
    print(f"  {len(tilt_events)} sustained tilt changes (>={TILT_MIN_DEG} deg, >={TILT_HOLD_S}s)")

    raw_events = apply_tilt_rollovers(raw_events, tilt_events)

    print(f"merging events within {MERGE_GAP_S}s gap…")
    events = merge_close_events(raw_events, MERGE_GAP_S)
    print(f"  {len(events)} merged events")

    # Detect dual-sensor data
    has_chest = "chest_acc_x_g" in df.columns
    has_left_ankle = "ankleL_acc_x_g" in df.columns
    if has_chest:
        print("dual-sensor data detected — using chest sensor for body position")
        _, _, chest_unit_g = find_tilt_changes(
            df, fs, return_series=True,
            accel_cols=("chest_acc_x_g", "chest_acc_y_g", "chest_acc_z_g"))
    else:
        chest_unit_g = None

    # Dual-ankle: also detect movements on the LEFT ankle independently
    left_events = pd.DataFrame()
    if has_left_ankle:
        print("dual-ankle data detected — also detecting left-leg events")
        # Build a temporary frame with the left ankle aliased as the primary
        df_left = df.copy()
        df_left["acc_x_g"] = df["ankleL_acc_x_g"]
        df_left["acc_y_g"] = df["ankleL_acc_y_g"]
        df_left["acc_z_g"] = df["ankleL_acc_z_g"]
        df_left = df_left.dropna(subset=["acc_x_g"]).reset_index(drop=True)
        if len(df_left) > int(fs * 60):  # need at least a minute of data
            left_envelope = compute_envelope(df_left, fs)
            left_raw = find_events(left_envelope, df_left["t"])
            left_tilt_events, _, _ = find_tilt_changes(
                df_left, fs, return_series=True,
                accel_cols=("ankleL_acc_x_g", "ankleL_acc_y_g", "ankleL_acc_z_g"))
            left_raw = apply_tilt_rollovers(left_raw, left_tilt_events)
            left_events = merge_close_events(left_raw, MERGE_GAP_S)
            print(f"  {len(left_events)} merged left-leg events")
        else:
            print("  not enough left-ankle data; skipping")

    # Auto-detect calibration alongside CSV if not supplied
    if calibration_path is None:
        candidate = csv_path.with_suffix(".calibration.json")
        if candidate.exists():
            calibration_path = candidate
            print(f"auto-detected calibration: {candidate.name}")

    positions = None
    if calibration_path:
        print(f"loading calibration from {calibration_path.name}…")
        centroids = load_calibration(calibration_path)
        print(f"  {list(centroids.keys())}")
        # Use chest if available, else ankle
        pos_unit_g = chest_unit_g if chest_unit_g is not None else unit_g
        source = "chest" if chest_unit_g is not None else "ankle"
        print(f"classifying body position per 30-second window (source: {source})…")
        positions = classify_positions(pos_unit_g, df["t"], fs, centroids,
                                       envelope=envelope, window_s=30.0)
        print("  position counts (pre-Oura overlay):",
              positions["label"].value_counts().to_dict())

    # Compute breathing rate from chest sensor if available
    breathing = None
    if has_chest:
        print("computing breathing rate from chest accelerometer…")
        breathing = compute_breathing_rate(df, fs)
        if not breathing.empty:
            valid = breathing[breathing["signal_strength"] > 2.0]
            if not valid.empty:
                print(f"  median {valid['bpm'].median():.1f} BPM "
                      f"(IQR {valid['bpm'].quantile(0.25):.1f}–{valid['bpm'].quantile(0.75):.1f})")

    # Heart-rate processing if HRM was used
    hr_series = pd.DataFrame()
    arousals = pd.DataFrame()
    if "hr_bpm" in df.columns:
        print("loading HR data…")
        hr_series = load_hr_series(df)
        if not hr_series.empty:
            print(f"  {len(hr_series):,} HR samples, "
                  f"median {hr_series['hr_bpm'].median():.0f} bpm "
                  f"(range {hr_series['hr_bpm'].min():.0f}–{hr_series['hr_bpm'].max():.0f})")
            print("detecting HR arousals (>=10 bpm rise in <=30s)…")
            arousals = detect_hr_arousals(hr_series)
            print(f"  {len(arousals)} HR arousals detected")

    sleep_session = None
    if pat:
        print("fetching Oura sleep stages…")
        day_start = df["t"].iloc[0].to_pydatetime()
        day_end = df["t"].iloc[-1].to_pydatetime()
        sleep_session = fetch_oura_sleep(pat, day_start, day_end)
        if sleep_session:
            print(f"  matched session {sleep_session['id']}")
            events["sleep_stage"] = events["start"].apply(
                lambda t: stage_for_timestamp(t.tz_convert("UTC"), sleep_session))
        else:
            print("  no matching Oura session found")

    if positions is not None and sleep_session:
        positions = overlay_awake_to_unknown(positions, sleep_session)
        print("  position counts (post-Oura overlay):",
              positions["label"].value_counts().to_dict())

    if positions is not None:
        # Tag each event with its position window
        def position_at(t):
            mask = (positions["start"] <= t) & (positions["end"] >= t)
            sub = positions[mask]
            return sub["label"].iloc[0] if not sub.empty else "unknown"
        events["position"] = events["start"].apply(position_at)

    events = classify_by_duration(events)
    events = tag_series(events)
    events["leg"] = "right"

    # Dual-ankle: score the left leg independently and merge into events.
    #
    # The right-only pipeline misses nights where the left leg is dominant
    # (e.g. 5/12: 79 left events vs 25 right). Fix: classify + tag_series
    # on left_events too, then union with right events, marking each row's
    # leg. Bilateral events (both legs within 1s) are deduplicated — we keep
    # the right-leg row and mark it bilateral rather than double-counting.
    BILATERAL_WINDOW_S = 1.0
    if not left_events.empty:
        # Apply the same sleep-stage and position tags to left events
        if sleep_session is not None:
            left_events["sleep_stage"] = left_events["start"].apply(
                lambda t: stage_for_timestamp(t.tz_convert("UTC"), sleep_session))
        if positions is not None:
            left_events["position"] = left_events["start"].apply(position_at)

        left_events = classify_by_duration(left_events)
        left_events = tag_series(left_events)
        left_events["leg"] = "left"
        left_events["bilateral"] = False

        # Find bilateral pairs: right event has a left event within 1s
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        right_s = (events["start"] - epoch).dt.total_seconds().to_numpy()
        left_s  = (left_events["start"] - epoch).dt.total_seconds().to_numpy()

        bilateral_left_indices = set()
        events["bilateral"] = False
        for i, rt in enumerate(right_s):
            close = np.where(np.abs(left_s - rt) < BILATERAL_WINDOW_S)[0]
            if close.size > 0:
                events.iloc[i, events.columns.get_loc("bilateral")] = True
                bilateral_left_indices.update(close.tolist())

        # Keep only unilateral left events (not already represented by a right event)
        left_only = left_events[~left_events.index.isin(bilateral_left_indices)].copy()

        n_bilat   = events["bilateral"].sum()
        n_left_only = len(left_only)
        print(f"  bilateral events (right + left within {BILATERAL_WINDOW_S}s): "
              f"{n_bilat} of {len(events)} right ({100*n_bilat/max(1,len(events)):.0f}%)")
        print(f"  unilateral left-only events added to output: {n_left_only}")

        # Union: right events + unilateral-left events, sorted by time
        events = pd.concat([events, left_only], ignore_index=True).sort_values("start").reset_index(drop=True)
    else:
        events["bilateral"] = False

    # Link PLMs and other events to preceding HR arousals
    if not arousals.empty:
        events = link_arousals_to_plms(events, arousals, window_s=30.0)
        plm_events = events[events["final_class"] == "PLM_in_series"]
        if not plm_events.empty:
            n_with_arousal = plm_events["preceded_by_arousal_s"].notna().sum()
            pct = 100.0 * n_with_arousal / len(plm_events)
            print(f"  {n_with_arousal}/{len(plm_events)} PLMs ({pct:.0f}%) preceded by HR arousal within 30s")

    stem = csv_path.stem
    out_dir = csv_path.parent
    events_path = out_dir / f"{stem}_events.csv"
    summary_path = out_dir / f"{stem}_summary.txt"
    plot_path = out_dir / f"{stem}_overview.png"
    annotated_path = out_dir / f"{stem}_annotated.png"

    # Look up night log entry (if any)
    log_path = Path(__file__).parent / "night_log.csv"
    night_log_row = load_night_log_row(log_path, df["t"].iloc[0])
    if night_log_row:
        print(f"matched night_log entry for {night_log_row.get('date', '')}")

    events.to_csv(events_path, index=False)
    text = write_summary(events, sleep_session, summary_path, positions=positions,
                         night_log_row=night_log_row, left_events=left_events)
    plot_overview(df, envelope, events, sleep_session, plot_path)
    plot_annotated(df, envelope, tilt_series, events, sleep_session, annotated_path,
                   positions=positions, night_log_row=night_log_row,
                   hr_series=hr_series, arousals=arousals, breathing=breathing)

    print()
    print(text)
    print()
    print(f"wrote {events_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {annotated_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--calibration", type=Path, default=None,
                    help="path to calibration.json (auto-detected from CSV name if omitted)")
    args = ap.parse_args()
    main(args.csv.expanduser(),
         args.calibration.expanduser() if args.calibration else None)
