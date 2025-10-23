import json
import pandas as pd
import numpy as np

def compute_angle_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the signed difference between target_angle and hand_fb_angle within [-180, 180].
    """
    df['angle_diff'] = ((df['target_angle'] - df['hand_fb_angle'] + 180) % 360) - 180
    return df


def detect_and_replace_outliers(
    df: pd.DataFrame,
    low: float = -60.0,
    high: float = 60.0,
    neighbor_count: int = 4
) -> float:
    """
    Identify values where angle_diff is outside [low, high], compute percentage,
    and replace each outlier with the mean of its `neighbor_count` neighbors.
    Returns percentage of outliers.
    """
    mask = (df['angle_diff'] < low) | (df['angle_diff'] > high) | (df['rt'] < 10)
    percent = mask.mean() * 100
    outlier_indices = df.index[mask]

    for i in outlier_indices:
        start = max(0, i - neighbor_count)
        end = min(len(df), i + neighbor_count + 1)
        neighbors = (
            df['angle_diff'].iloc[start:i].tolist() +
            df['angle_diff'].iloc[i+1:end].tolist()
        )
        if neighbors:
            df.at[i, 'angle_diff'] = np.mean(neighbors)
    return percent


def _extract_angle_from_trajectory(
    movement_traj_str: str,
    mt_value: float,
    mt_thresh: float,
    fallback_angle: float
) -> float:
    """
    Given a JSON string representing a list of points with keys 'x', 'y', 't',
    parse it (possibly nested JSON strings), append an artificial point at t=mt_thresh,
    and find the point whose timestamp is closest to mt_thresh. If that closest
    timestamp is the artificial one, returns fallback_angle. Otherwise computes
    the screen-relative angle of the chosen point.
    """
    result = json.loads(movement_traj_str)
    max_iter = 5
    iter_count = 0
    while isinstance(result, str) and iter_count < max_iter:
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            break
        iter_count += 1

    if not isinstance(result, list):
        raise ValueError(
            f"Decoding did not result in a list. Got: {result!r} of type {type(result)}"
        )

    if len(result) == 0:
        return 150.0

    xs = np.array([pt["x"] for pt in result], dtype=float)
    ys = np.array([pt["y"] for pt in result], dtype=float)
    ts = np.array([pt["t"] for pt in result], dtype=float)

    # Append artificial timestamp at mt_thresh
    ts = np.append(ts, mt_value)

    # Find index of timestamp closest to mt_thresh
    idx = int(np.argmin(np.abs(ts - mt_thresh)))

    # If the artificial point is the closest, use the fallback
    if idx == len(ts) - 1:
        return fallback_angle

    # Otherwise compute angle from the chosen real point
    x = xs[idx]
    y = ys[idx]
    # t = ts[idx]
    # print(x, y, "time:", t)

    center_x, center_y = 0.5, 0.5
    dx = x - center_x
    dy = center_y - y  # invert y for screen coordinates
    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 360.0

    return angle


def _compute_for_invalid(row: pd.Series, mt_thresh: float = 400.0) -> float:
    """
    Helper to compute hand_fb_angle for rows where mt >= mt_val.
    Uses the movement_trajectory JSON and falls back to original hand_fb_angle.
    """
    movement_str = row["movement_trajectory"]
    mt_val = row["mt"]
    fallback = float(row["hand_fb_angle"])
    return _extract_angle_from_trajectory(movement_str, mt_val, mt_thresh, fallback)


def compute_angle_diff_remove_invalid_mt(
    df: pd.DataFrame,
    mt_thresh: float = 400.0
) -> pd.DataFrame:
    """
    For entries with mt < mt_thresh, uses the existing hand_fb_angle.
    For entries with mt >= mt_thresh, parses the trajectory to find the point
    closest to mt_thresh (or falls back to hand_fb_angle if that artificial
    time is closest), then recomputes angle_diff.
    """
    df = df.copy()

    mask_valid = df["mt"].astype(float) < mt_thresh

    valid_angles = np.zeros(len(df), dtype=float)
    valid_angles[mask_valid] = df.loc[mask_valid, "hand_fb_angle"].astype(float).values

    mask_invalid = ~mask_valid
    valid_angles[mask_invalid] = (
        df.loc[mask_invalid]
          .apply(lambda row: _compute_for_invalid(row, mt_thresh=mt_thresh), axis=1)
          .astype(float)
          .values
    )

    df["valid_hand_fb_angle"] = valid_angles
    df['angle_diff'] = ((df['target_angle'] - df["valid_hand_fb_angle"] + 180) % 360) - 180

    return df

def compute_angle_diff_average_invalid_mt(
    df: pd.DataFrame,
    mt_threshold: float = 400.0,
    neighbor_count: int = 4
) -> pd.DataFrame:
    df = df.copy()
    diffs = df['angle_diff'].to_numpy().copy()
    total = len(diffs)

    invalid_idxs = np.where(df['mt'].astype(float) >= mt_threshold)[0]
    replaced = len(invalid_idxs)
    fallback = 0

    for i in invalid_idxs:
        start = max(0, i - neighbor_count)
        end   = min(total, i + neighbor_count + 1)
        neigh = [j for j in range(start, end) if j != i]
        # neigh = [j for j in range(start, i)]

        valid = [diffs[j] for j in neigh if df.loc[j, 'mt'] < mt_threshold]

        if len(valid) >= 2:
            diffs[i] = np.mean(valid)
        else:
            prev_idx = next((j for j in range(i-1, -1, -1)
                             if df.loc[j, 'mt'] < mt_threshold), None)
            if prev_idx is not None:
                diffs[i] = diffs[prev_idx]
            else:
                next_idx = next((j for j in range(i+1, total)
                                 if df.loc[j, 'mt'] < mt_threshold), None)
                diffs[i] = diffs[next_idx] if next_idx is not None else np.nan
            fallback += 1

    df['angle_diff'] = diffs

    print(f"[INFO] Smoothed {replaced/total*100:.2f}% of mt ≥ {mt_threshold} entries")
    print(f"[INFO] {fallback/total*100:.2f}% fell back to nearest valid diff")

    return df

def detect_and_replace_outliers_hampel(
    df: pd.DataFrame,
    *,
    window_size: int = 9,          # odd window; 9 is a solid default for short spikes
    n_sigmas: float = 3.0,         # Hampel cutoff (k)
    mad_scale: float = 1.4826,     # Gaussian-consistent MAD scale
    col: str = "angle_diff",       # column to clean
    rt_col: str = "rt",            # keep: trials with rt < 10 are flagged
    replace_with: str = "median"   # "median" (standard Hampel) or "noop" to only flag
) -> float:
    """
    Robust outlier detection and replacement via a Hampel filter (no range guard).

    Model:
        For a series x_t = df[col], with odd window w = 2h + 1:
            m_t  = median{x_{t-h}, …, x_{t+h}}
            MAD_t = median{|x_{t+i} - m_t| : i in [-h, h]}
            sigma_hat_t = mad_scale * MAD_t
            Outlier if |x_t - m_t| > n_sigmas * sigma_hat_t

    Replacement:
        If flagged, replace x_t with m_t (window median) when replace_with == "median".

    Parameter:
        window_size (odd; default 9): 
            make it roughly longer than a typical spike but not so long that it steamrolls real dynamics. 
            5–9 for short impulsive artifacts; 11–21 if spikes span several samples.
        n_sigmas (default 3.0): 
            lower means more aggressive (2.5), higher is more conservative (3.5–4.5) if your signal has legitimate fast swings.
        mad_scale (1.4826): 
            keep this unless you have non-Gaussian baseline noise and want a different scale.
        rt_col rule: 
            still flags rt < 10. If you want to disable that too, 
            set rt_col to a name that doesn’t exist or add a boolean switch; I can wire that in.

    Returns:
        Percentage of entries replaced (100 * flagged / N).
    """
    # Defensive copy and type
    s = df[col].astype(float).to_numpy().copy()
    n = len(s)
    if n == 0:
        return 0.0

    # Flag by rt (preserves prior behavior)
    flag_rt = df[rt_col].astype(float).to_numpy() < 10

    # Hampel: ensure odd window
    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2

    med = np.empty(n)
    avg = np.empty(n)
    mad = np.empty(n)

    for t in range(n):
        start = max(0, t - half)
        end = min(n, t + half + 1)
        window = s[start:end]
        m = np.median(window)
        a = np.mean(window)
        med[t] = m
        avg[t] = a
        mad[t] = np.median(np.abs(window - m))

    sigma_hat = mad_scale * mad
    sigma_hat_safe = np.where(sigma_hat == 0.0, np.finfo(float).eps, sigma_hat)

    flag_hampel = np.abs(s - med) > (n_sigmas * sigma_hat_safe)

    flagged = flag_rt | flag_hampel
    num_flagged = int(flagged.sum())


    if replace_with == "median":
        s[flagged] = med[flagged]
    elif replace_with == "mean":
        s[flagged] = avg[flagged]
    else:
        raise ValueError(f"Unknown replace_with mode: {replace_with!r}")
    
    df[col] = s
    return 100.0 * num_flagged / max(1, n)

def adjust_angle_diff_baseline(
    df: pd.DataFrame,
    baseline_start_idx: int,   # kept for backward compatibility (used only in legacy fallback)
    baseline_end_idx: int,     # kept for backward compatibility (used only in legacy fallback)
    *,
    pre_window: int = 100,
    rotation_col: str = "rotation",
    angle_col: str = "angle_diff",
    baseline_out_col: str = "baseline_mean",
) -> pd.DataFrame:
    df = df.copy()

    sel_start: int
    sel_end: int

    if rotation_col in df.columns:
        rot_vals = df[rotation_col].to_numpy()
        # robust to types; treat anything exactly != 0 as perturbation
        nonzero_idx = np.flatnonzero(rot_vals != 0)

        if len(nonzero_idx) > 0:
            first_nz = int(nonzero_idx[0])
            sel_start = max(0, first_nz - pre_window)
            sel_end = first_nz
        else:
            sel_start = 0
            sel_end = min(pre_window, len(df))
    else:
        # Legacy behavior if rotation column is unavailable
        sel_start = int(baseline_start_idx)
        sel_end = int(baseline_end_idx)

    sel_start = 0
    print(f"use the range at [{sel_start}:{sel_end}]")

    if sel_start >= sel_end or len(df) == 0:
        # Degenerate case: nothing to compute; also ensure column exists (NaN)
        df[baseline_out_col] = np.nan
        return df

    # Compute baseline mean and shift the signal
    baseline_value = df[angle_col].iloc[sel_start:sel_end].mean()
    df[angle_col] = df[angle_col] - baseline_value

    # Persist the baseline mean so downstream code can inspect/plot it
    df[baseline_out_col] = baseline_value

    return df