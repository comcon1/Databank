"""Analysis module for FormFactor curve."""

import numpy as np
import scipy.signal


def get_mins_from_ffdata(ffdata: np.ndarray) -> float:
    """Find the positions of minimums in form factor data."""
    sg_window_q = 0.03  # Savitsky-Golay window (in Q)
    delta_q = ffdata[1, 0] - ffdata[0, 0]  # Q step in FF data
    sg_window_n = int(np.ceil(sg_window_q / delta_q))  # S-G window (in num frames)
    try:
        filtered = scipy.signal.savgol_filter(ffdata[:, 1], sg_window_n, 2)
    except ValueError as e:
        msg = f"Problems running Savitsky-Golay on data with d={delta_q} using window {sg_window_n}."
        raise ValueError(msg) from e

    min_q_distance = 0.01  # Min distance btw peaks (in Q)
    mqd_n = int(np.ceil(min_q_distance / delta_q))  # same in num frames
    peak_ind = scipy.signal.find_peaks(-filtered, distance=mqd_n)
    min_peak_q = 0.1

    return [ffdata[i, 0] for i in peak_ind[0] if ffdata[i, 0] > min_peak_q]
