from pre_processing import Preprocess


def fit_reference(
    timeseries,
    raw_reference,
    raw_signal,
    fr,
    cutoff,
    drop=200,
    window_size=11,
    r_squared_threshold=0.7,
    pos_coef = False,
    detrend_last=False,
    smoothing_method="tma",
    baseline_method="lpf",
    fit_method="l"
):
    """
    Input:
        fr: sampling frequency
    Using preprocessing method in Preprocess object, fit reference channel to signal channel,
    and produce (z scored) reference, signal, and fitted reference

    Returns:
        fitted_reference: np.ndarray
    """
    data = Preprocess(timeseries, raw_signal, raw_reference, pos_coef, fr, cutoff, drop, window_size, r_squared_threshold)
    data.pipeline(smoothing_method, baseline_method, fit_method, detrend_last)
    return data.fitted_ref
