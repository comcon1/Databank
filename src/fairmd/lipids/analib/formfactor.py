"""Analysis module for FormFactor curve."""

import scipy.signal


def get_mins_from_ffdata(ffdata):
    """Find the position of first minimum in form factor data."""
    FFtmp = []
    for i in ffdata:
        FFtmp.append(-i[1])

    try:
        w = scipy.signal.savgol_filter(FFtmp, 31, 1)
    except ValueError as e:
        print("FFtmp:")
        print(FFtmp)
        raise e

    minX = []

    peak_ind = scipy.signal.find_peaks(w)

    for i in peak_ind[0]:
        if ffdata[i][0] > 0.1:
            minX.append(ffdata[i][0])

    print(minX)
    return minX
