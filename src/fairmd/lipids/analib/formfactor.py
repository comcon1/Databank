"""Analysis module for FormFactor curve."""

import numpy as np
import scipy.interpolate
import scipy.signal


def get_mins_from_ffdata(ffdata: np.ndarray) -> list[float]:
    """Find the positions of minimums in form factor data."""
    sg_window_q = 0.05  # Savitsky-Golay window (in Q)
    delta_q = ffdata[1, 0] - ffdata[0, 0]  # Q step in FF data
    sg_window_n = int(np.ceil(sg_window_q / delta_q))  # S-G window (in num frames)
    try:
        filtered = scipy.signal.savgol_filter(ffdata[:, 1], sg_window_n, 2)
    except ValueError as e:
        msg = f"Problems running Savitsky-Golay on data with d={delta_q} using window {sg_window_n}."
        raise ValueError(msg) from e

    min_q_distance = 0.01  # Min distance btw peaks (in Q)
    mqd_n = int(np.ceil(min_q_distance / delta_q))  # same in num frames
    peak_prominence = (filtered.max() - filtered.min()) * 0.02
    peak_ind = scipy.signal.find_peaks(-filtered, distance=mqd_n, prominence=peak_prominence)
    min_peak_q = 0.1

    return [ffdata[i, 0] for i in peak_ind[0] if ffdata[i, 0] > min_peak_q]


def calc_ff_scaling_distance(ffd_exp: np.ndarray, ffd_sim: np.ndarray) -> tuple[float, float]:
    """
    Calculate scaling factor and Chi2-distance for exp SAXS values.

    Scaling as defined by Kučerka et al. 2008b, doi:10.1529/biophysj.107.122465
    Quality as defined by Kučerka et al. 2010, doi:10.1007/s00232-010-9254-5

    :param ffd_sim: Simulation FF data (float 2D list)
    :param ffd_exp: Experiment FF data (float 2D list)

    :return: [scaling coeffitient, Chi2-distance] (floats)
    """
    min_q = max(ffd_sim[:, 0].min(), ffd_exp[:, 0].min())
    max_q = min(ffd_sim[:, 0].max(), ffd_exp[:, 0].max())

    val_interpolator = scipy.interpolate.interp1d(ffd_exp[:, 0], ffd_exp[:, 1])
    err_interpolator = scipy.interpolate.interp1d(ffd_exp[:, 0], ffd_exp[:, 2])
    min_i = ffd_sim[:, 0].searchsorted(min_q)
    max_i = ffd_sim[:, 0].searchsorted(max_q)
    exp_vals = val_interpolator(ffd_sim[min_i:max_i, 0])
    exp_errs = err_interpolator(ffd_sim[min_i:max_i, 0])
    md_vals = ffd_sim[min_i:max_i, 1]

    sum1 = (np.abs(md_vals * exp_vals) / exp_errs**2).sum()
    sum2 = (exp_vals**2 / exp_errs**2).sum()

    scf = sum1 / sum2

    sum1 = (np.abs(md_vals) - scf * np.abs(exp_vals)) ** 2 / (scf * exp_errs) ** 2
    chi = np.sqrt(sum1.sum()) / np.sqrt(max_i - min_i - 1)

    return [scf, chi]


def calc_minpos_with_error(ffdata: np.ndarray, backup_const_error: float = 0.1) -> (float, float):
    """Estimate error of minimum position in form factor data."""
    m1pos = get_mins_from_ffdata(ffdata)[0]
    # find max x val where ypts < 0 in the vicinity x0+-maxerr
    maxXerr = 0.03
    idxPlusErr = ffdata[:, 0].searchsorted(m1pos + maxXerr)
    idxMinusErr = ffdata[:, 0].searchsorted(m1pos - maxXerr)
    popt, pcov = scipy.optimize.curve_fit(
        lambda x, a, b, c: a * x**2 + b * x + c,
        ffdata[idxMinusErr:idxPlusErr, 0],
        ffdata[idxMinusErr:idxPlusErr, 1],
        sigma=ffdata[idxMinusErr:idxPlusErr, 2] if ffdata.shape[1] > 2 else backup_const_error,
        absolute_sigma=True,
        p0=[1, -2 * m1pos, 0],
    )
    a, b, _c = popt
    min_x = -b / 2 / a
    delta_minx = (
        (-1 / 2 / a) ** 2 * pcov[0, 0]
        + (b / 2 / a**2) ** 2 * pcov[1, 1]
        + 2 * (-1 / 2 / a) * (b / 2 / a**2) * pcov[0, 1]
    )
    return min_x, np.sqrt(delta_minx)
