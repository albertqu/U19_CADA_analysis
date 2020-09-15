# System

# Data
import numpy as np
import pandas as pd
from scipy import interpolate
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Utils
from utils import *


#######################################################
###################### Analysis #######################
#######################################################
def time_aligned_from_files():
    pass


def align_activities_with_event(sigs, times, event_times, time_window, discrete=True, align_last=False):
    """ Takes signals (... x T), time warp is there is more than one event time type (mean frames)
    :param sigs: (... x T),
    :param times: (... x T) if discrete else float for frame rate (Hz)
    :param event_times: np.array (... (x e) x K), e as number of different events
    :param time_window: np.ndarray or [pre, post, (si, if discrete==False)], if discrete, int, else,
    double in ms
    :return: aligned: np.array (... x K x W) / list, W is variable depending on
    aligned_times
    TODO: allow the functionality of align with the last event if needed
    TODO: subjected to change as the trial numbers for different sessions are largely variable
    TODO: Generalize to arbitrary events with uniform time window
    """
    if isinstance(time_window, tuple):
        time_window = calculate_best_time_window(times, event_times, time_window)
    dt = 2 * (time_window[1] - time_window[0])
    k = event_times.shape[-1]
    if discrete:
        result = [None] * k
        pass
    else:
        if len(sigs.shape) != 1:
            return np.concatenate([align_activities_with_event(sigs[i], times[i], event_times[i],
                    time_window, discrete)[np.newaxis, ...] for i in range(sigs.shape[0])], axis=0)
        else:
            # ASSUMING time_window + 2dt is shorter than time horizon
            assert (time_window[-1] - time_window[0] + 2 * dt) < (np.max(times)-np.min(times)), \
                "Choice of time window too big for the time horizon!"
            # TODO: better method by using matrix reshaping in same order w. parallel interpolation
            result = np.full((k, len(time_window)), np.nan)
            for ik in range(k):
                if len(event_times.shape) > len(sigs.shape):
                    evnt = event_times[:, ik]
                else:
                    evnt = event_times[np.newaxis, ik]
                align = evnt[-1] if align_last else evnt[0]

                rois = (times >= align + time_window[0] - dt) & (times <= align + time_window[-1] + dt)
                # print(dt, time_window[0], time_window[-1], evnt[0], (times[rois] - evnt[0])[[0, -1]])
                if (times[0] > align + time_window[0] - dt) & (times[-1] < align + time_window[-1] + dt):
                    print(f'WARNING: trial {ik}, time traces gets cutoff at edges by time_window, '
                          f'results might not be as expected (e.g. nans)')
                # TODO: looking into the interpolation for better performance
                # change to extrapolate
                # TODO: compare np.nan vs "extrapolate"
                result[ik] = interpolate.interp1d(times[rois] - align, sigs[rois],
                                                  fill_value="extrapolate")(time_window)
            # TODO: proof read again
            return result


def calculate_best_time_window(times, event_times, twindow_tuple, align_last):
    # TODO: add time warp, dynamic method
    # si should be shorter than sampling interval in times
    pre, post, si = twindow_tuple
    if len(event_times.shape) > len(times.shape):
        evts = event_times[:, 0]
        start, end = evts[0], evts[-1]
        if align_last:
            return np.arange(post, start - end - pre - si, -si)[::-1]
        else:
            return np.arange(- pre, end - start + post + si, si)
    else:
        return np.arange(pre, post+1, si)


#######################################################
################### Visualization #####################
#######################################################
def peristimulus_time_trial_average_plot(sigs, times, tags, extra_event_times=None, ax=None):
    """
    TODO: enable feeding error bars
    Take in list of signal groups plot hued line plots in ax.
    :param sigs: np.ndarray or list of disparate signals (np.ndarray); if sigs are organized in 2D list,
        then treat each inner list as one group and plot with
    :param times: in matching dimension as sigs, with time windows; if just np.ndarray, time windows is
        treated as uniform for all sigs
    :param tags: list with same length as len(sigs)
    :param extra_event_times: list of tuples (name, t) extra events
    :return:
    """
    # TODO: enable specific color schemes
    if ax is None:
        ax = plt.gca()
    if isinstance(sigs, np.ndarray):
        sigs = [sigs]
    evnt, xlb, ylbl, lgs = tags
    nolegend = False
    if lgs is None:
        nolegend = True
        lgs = ['sig']
    for i, isig in enumerate(sigs):
        ilg = lgs[i]
        if isinstance(isig, list):
            # TODO: special treatment of group of inhomogeneous signals
            pass
        else:
            itime = times if isinstance(times, np.ndarray) else times[i]
            isig = isig.reshape((-1, isig.shape[-1]), order='C')
            meansig = np.nanmean(isig, axis=0)
            # TODO: better method for calculating stderr with nans
            stderr = np.nanstd(isig, axis=0) / np.sqrt(isig.shape[0])
            ax.plot(itime, meansig, label=ilg)
            ax.fill_between(itime, meansig-stderr, meansig+stderr, alpha=0.2)
    ax.axvline(0, ls='--')
    if extra_event_times is None:
        extra_event_times = []
    for ename, eevnt in extra_event_times:
        ax.axvline(eevnt, ls='--')
    ax.set_xlabel(xlb)
    ax.set_ylabel(ylbl)
    if not nolegend:
        ax.legend(fontsize="xx-small")
    return ax


def peristimulus_time_trial_heatmap_plot(sigs, times, trials, tags, extra_event_times=None, trial_marks=None,
                                         sort=True, ax=None):
    """ Takes in signals, time windows
    :param sigs: np.ndarray
    :param trials: trial numbers
    :param tags: list with same length as len(sigs)
    :param extra_event_times: list of extra events
    :return:
    TODO:generalize to multiple events and sort heatmap with event times
    """
    # https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html
    # TODO: enable specific color schemes
    if ax is None:
        ax = plt.gca()
    ttle, xlb, ylbl = tags

    assert len(sigs.shape) >= 2, "multiple trial information is required"
    sigs = sigs.reshape((-1, )+sigs.shape[-2:], order='C')
    sigs = np.nanmean(sigs, axis=0)
    # TODO: more careful play of colorbar
    # TODO: heatmap align time stamps to the grids
    # Sorting signal with respect to extra_event_times
    assert len(np.where(times == 0)[0]) == 1, "0ms should be unique"
    zero = np.where(times == 0)[0][0]
    if extra_event_times is None:
        extra_event_times = []
    if extra_event_times and isinstance(extra_event_times[0][1], np.ndarray):
        ename0, eevnt0 = extra_event_times[0]
        assert len(eevnt0) == sigs.shape[0]
        # Sort signal according to extra event times
        sort_args = np.argsort(eevnt0)
        sigs = sigs[sort_args]
        trials = trials[sort_args]
        for j in range(len(extra_event_times)):
            enamej, eevntj = extra_event_times[j]
            extra_event_times[j] = (enamej, eevntj[sort_args])

    ticks = np.zeros(len(extra_event_times)+3)
    tlabels = np.zeros(len(extra_event_times)+3)
    ticks[:3] = [0, zero, len(times)-1]
    tlabels[:3] = [times[0], 0, times[-1]]
    tt = 3
    for ename, eevnt in extra_event_times:
        # i+3
        eind = np.where(times == eevnt)[0][0]
        ticks[tt] = eind
        tlabels[tt] = eevnt
        tt += 1


    sns.heatmap(sigs, cmap='coolwarm', ax=ax)
    ax.axvline(zero, ls='--')
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabels)
    if xlb:
        ax.set_xlabel(xlb)
    if ylbl:
        ax.set_ylabel(ylbl)
    return ax


def peristimulus_multiple_file_multiple_events(mats, event_types):
    """load list of events and align all files in mats and average them"""
    pass
