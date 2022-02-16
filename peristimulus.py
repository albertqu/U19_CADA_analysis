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
from behaviors import *


#######################################################
###################### Analysis #######################
#######################################################
def time_aligned_from_files():
    pass


def align_activities_with_event(sigs, times, event_times, time_window, discrete=False, align_last=False, fr=None):
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
    if times is None:
        times = np.arange(sigs.shape[-1])
    assert np.all(~np.isnan(event_times)), "event_times must not contain nans"
    if isinstance(time_window, tuple):
        time_window = calculate_best_time_window(times, event_times, time_window)
    dt = 2 * (time_window[1] - time_window[0])
    k = event_times.shape[-1]
    if discrete:
        assert isinstance(sigs, np.ndarray), 'currently only support discrete on ndarray'
        assert len(times) == sigs.shape[-1]
        assert fr is not None
        min_time, max_time = np.min(event_times), np.max(event_times)
        t0, tf = time_window[0], time_window[-1]
        t0_new, tf_new = int(t0 * fr), int(tf * fr)
        twindow = np.arange(t0_new, tf_new+1)
        assert ((min_time + t0_new) >= 0) & ((max_time + tf_new) >= np.max(times))
        result = np.concatenate([sigs[..., np.newaxis, twindow+et] for et in event_times], axis=len(sigs)-1)
        return result
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
                #print(times[0], align+time_window[0]-dt)
                rois = (times >= align + time_window[0] - dt) & (times <= align + time_window[-1] + dt)
                # print(dt, time_window[0], time_window[-1], evnt[0], (times[rois] - evnt[0])[[0, -1]])
                if (times[0] > align + time_window[0] - dt) & (times[-1] < align + time_window[-1] + dt):
                    logging.warning(f'WARNING: trial {ik}, time traces gets cutoff at edges by time_window, '
                          f'results might not be as expected (e.g. nans)')
                # TODO: looking into the interpolation for better performance
                # change to extrapolate
                # TODO: compare np.nan vs "extrapolate"
                dtt = np.mean(np.diff(times)) # originally made the mistake assuming time_window having same sampling rate as sig_time
                if np.sum(rois) < int(np.floor((time_window[-1] - time_window[0]) / dtt)):
                    logging.info(f"skipping {ik}th entry in event_times")
                else:
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
def peristimulus_time_trial_average_plot(sigs, times, tags, extra_event_times=None, ylim=None, ax=None,
                                         method='ci95'):
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
    if len(tags) == 4:
        evnt, xlb, ylbl, lgs = tags
        colors = sns.color_palette("gist_ncar_r", n_colors=len(sigs))
    else:
        evnt, xlb, ylbl, lgs, colors = tags
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
            ax.plot(itime, meansig, color=colors[i], label=ilg)
            if method == 'stderr':
                lower, upper = meansig-stderr, meansig+stderr
            elif method[:2] == 'ci':
                alpha = 1 - float(method[2:]) / 100
                meanlamb = lambda xs: np.mean(xs, axis=0)
                lower, upper = get_bootstrap_CI(bootstrap_ests(isig, meanlamb, B=10000), alpha=alpha)
            else:
                raise NotImplementedError(f"Unknown method {method}")
            ax.fill_between(itime, lower, upper, color=colors[i], alpha=0.2)
    ax.axvline(0, ls='--')
    if extra_event_times is None:
        extra_event_times = []
    for ename, eevnt in extra_event_times:
        ax.axvline(eevnt, ls='--')
    ax.set_xlabel(xlb)
    ax.set_ylabel(ylbl)
    if ylim is not None:
        ax.set_ylim(ylim)
    if not nolegend:
        ax.legend(fontsize="xx-small")
    return ax


def get_bootstrap_samples(X_sample):
    "TODO: change by default do axis 0"
    # X_sample is P_hat
    N = X_sample.shape[0]
    sample_index = np.random.choice(N, N)
    return X_sample[sample_index]


def bootstrap_ests(X_sample, est, B=10000):
    theta_n = est(X_sample)
    if isinstance(theta_n, np.number):
        K = 1
    else:
        K = len(theta_n)
    theta_new = np.empty((B, K))
    for i in range(B):
        isamp = get_bootstrap_samples(X_sample)
        theta_new[i] = est(isamp)
    return theta_new


def get_bootstrap_CI(bs_samples, alpha=0.05):
    beta_lb = np.percentile(bs_samples, (alpha / 2) * 100, axis=0)
    beta_ub = np.percentile(bs_samples, (1 - alpha / 2)*100, axis=0)
    return beta_lb, beta_ub


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



def behavior_aligned_FP_plots(folder, plots, behaviors, choices, options, zscore=True,
                              base_method='robust', denoise=True):
    # TODO: think more carefully about multiple behavior alignment
    """ Core function for plotting peristimulus FP signals grouped by stimulus modalities.
    :param folder: str
            root folder storing the FP data, e.g. "ProbSwitch_FP_data"
    :param plots: str
            root folder storing all the analysis plots
    :param behaviors: list
            list of behaviors of interests, currently the function works best with only one stimulus
            but soon addition behavior would be added (water marks for instance)
    :param choices: dict
            standard CHOICE DICT defined in the documents
    :param base_method: so far FP method is lost in hdf5, incorporate this
    :return:
    """
    sigs = options['sigs']
    tags = ['DA', 'Ca']
    row, rows, col, cols = options['row'], options['rows'], options['col'], options['cols']
    hue, hues, plot_type = options['hue'], options['hues'], options['plot_type']
    if 'ylim' in options:
        # HAS TO BE 2D list
        ylims = options['ylim']
    else:
        ylims = None

    if isinstance(behaviors, str):
        behaviors = [behaviors]
    if choices is None:
        choices = {g: get_prob_switch_all_sessions(folder, g) for g in ('D1', 'A2A')}
    meas = ('zscore_' if zscore else '') + 'dF/F'
    denoise_arg = '_denoise' if denoise else ''
    effect_arg = "_".join([e for e in [row, col, hue] if e])
    behavior_arg = "_".join(behaviors)

    # hue: ITI, row,  col: laterality
    for group in ['D1', 'A2A']:
        neur_type = group if group == 'D1' else 'D2'
        sessions = choices[group]
        time_window = np.arange(-2000, 2001, 50)
        for animal in sessions:
            for session in sessions[animal]:
                print(animal, session)
                files = encode_to_filename(folder, animal, session)
                matfile, green, red, fp = files['processed'], files['green'], files['red'], files['FP']
                # Load FP
                if fp is not None:
                    with h5py.File(fp, 'r') as fp_hdf5:
                        fp_sigs = [access_mat_with_path(fp_hdf5, f'{tags[i]}/dff/{base_method}')
                                   for i in range(len(tags))]
                        fp_times = [access_mat_with_path(fp_hdf5, f'{tags[i]}/time') for i in
                                    range(len(tags))]
                else:
                    print(f"Warning {animal} {session} does not have photometry processed!")
                    fp_times, fp_sigs, iso_times, iso_sigs = get_sources_from_csvs([green, red],
                                                                                tags=('DA', 'Ca'), show=False)

                    fp_sigs = [raw_fluor_to_dff(fp_times[i], fp_sigs[i], iso_times[i], iso_sigs[i], base_method,
                                               zscore=False) for i in range(len(fp_sigs))]

                if denoise:
                    L = len(fp_times)
                    new_times, new_sigs = [None] * L, [None] * L
                    for i in range(L):
                        new_sigs[i], new_times[i] = denoise_quasi_uniform(fp_sigs[i], fp_times[i])
                    fp_sigs, fp_times = new_sigs, new_times
                if zscore:
                    fp_sigs = [(fp_sigs[i] - np.mean(fp_sigs[i])) / np.std(fp_sigs[i], ddof=1)
                               for i in range(len(fp_sigs))]

                # TODO: for now just do plots for one session
                mat = h5py.File(matfile, 'r')
                # Get aligned signals to behaviors
                behavior_times = np.vstack([get_behavior_times(mat, beh) for beh in behaviors])
                nonan_sel = ~np.any(np.isnan(behavior_times), axis=0)
                behavior_times_nonan = behavior_times[:, nonan_sel]
                # TODO: ADD caps for multiple behavior time latencies
                aligned = [align_activities_with_event(fp_sigs[i], fp_times[i], behavior_times_nonan,
                                                       time_window, False) for i in range(len(fp_sigs))]

                # get trial features
                def opt2selgroups(opt):
                    # add in different data
                    if opt is None:
                        return {'all': None}
                    if opt == 'FP':
                        return {'DA': 0, 'Ca': 1}
                    return get_trial_features(mat, opt)

                rsel_groups, csel_groups, hsel_groups = opt2selgroups(row), opt2selgroups(col), \
                                                        opt2selgroups(hue)
                mat.close()

                zfolder = "zscore" if zscore else "dff"
                # TODO: come up with unifying code for fname
                subfolder = os.path.join(plots, "behavior_aligned", plot_type, effect_arg, behavior_arg,
                                         f"{base_method}_{zfolder}{denoise_arg}")
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                for k, fsig in enumerate(sigs):
                    N_trials = np.arange(len(nonan_sel))[nonan_sel]
                    session_left = len(N_trials)
                    justsig = (row == 'FP') or (col == 'FP') or (hue == 'FP')
                    if fsig == 'all' or (len(sigs) == 1 and justsig):
                        k_aligned = aligned
                    else:
                        k_aligned = aligned[opt2selgroups('FP')[fsig]]

                    if plot_type == 'trial_raw':
                        all_ns = set()
                        for i in range(len(rows)):
                            # TODO: add extra event times if needed
                            for j in range(len(cols)):
                                # TODO: only handle the case when row is the signal variable
                                rsels, csels = rsel_groups[rows[i]], csel_groups[cols[j]]
                                # TODO: find out reason for incomplete sampling
                                if rsels is None:
                                    rsels = np.full_like(nonan_sel, 1)
                                if csels is None:
                                    csels = np.full_like(nonan_sel, 1)
                                if isinstance(csels, np.ndarray):
                                    sels = csels[nonan_sel]
                                else:
                                    raise NotImplementedError("Only row can be different signals")
                                if isinstance(rsels, np.ndarray):
                                    sels = sels & rsels[nonan_sel]
                                    ijk_aligned = k_aligned
                                else:
                                    ijk_aligned = k_aligned[rsels]

                                for h in hues:
                                    hsels = hsel_groups[h]
                                    if hsels is not None:
                                        sels = hsels[nonan_sel] & sels

                                    in_sigs = ijk_aligned[sels]
                                    for l in range(in_sigs.shape[0]):
                                        extra_times = zip(behaviors[1:], np.diff(behavior_times[:, l]))
                                        fig, ax = plt.subplots(nrows=1, ncols=1)
                                        ax = peristimulus_time_trial_average_plot(in_sigs[l],
                                                                                  time_window,
                                                                                  (behaviors[0], "time(ms)", "",
                                                                                  [h]), extra_times, ax=ax)
                                        ax.set_ylabel(rows[i] + f' ({meas})')
                                        ax.set_title(cols[j])
                                        lnum = N_trials[sels][l]
                                        all_ns.add(lnum)
                                        session_left -= 1
                                        print(f'trial {lnum}, sessions left: {session_left}')
                                        fig.suptitle(f"{effect_arg} effects on {neur_type} {behaviors} phase {fsig}")
                                        sf=f"{neur_type}_{fsig}_{behavior_arg}_{rows[i]}_{cols[j]}_{h}_t{lnum}"
                                        subfolderK = os.path.join(subfolder, f"{animal}_{session}")
                                        if not os.path.exists(subfolderK):
                                            os.makedirs(subfolderK)
                                        fname = os.path.join(subfolderK, sf)
                                        fig.savefig(fname + '.png')
                                        plt.close(fig)
                            print('done!!!', len(all_ns), aligned[0].shape[0])
                    else:
                        if ylims is None:
                            sharey_opt = 'row' if row == 'FP' else 'col'  # TODO: make it more generalized
                        else:
                            sharey_opt = False
                        fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True,
                                                 sharey=sharey_opt, figsize=(20, 10))
                        if len(rows) == 1 and len(cols) == 1:
                            axes = np.array([[axes]])
                        if len(rows) == 1:
                            axes = axes.reshape((1, -1))
                        elif len(cols) == 1:
                            axes = axes.reshape((-1, 1))
                        for i in range(len(axes)):
                            # TODO: add extra event times if needed
                            axes[i][0].set_ytitle = rows[i] + f' ({meas})'
                            for j in range(len(axes[i])):
                                # TODO: only handle the case when row is the signal variable
                                rsels, csels = rsel_groups[rows[i]], csel_groups[cols[j]]
                                if rsels is None:
                                    rsels = np.full_like(nonan_sel, 1)
                                if csels is None:
                                    csels = np.full_like(nonan_sel, 1)
                                if isinstance(csels, np.ndarray):
                                    sels = csels[nonan_sel]
                                else:
                                    raise NotImplementedError("Only row can be different signals")
                                if isinstance(rsels, np.ndarray):
                                    sels = sels & rsels[nonan_sel]
                                    ijk_aligned = k_aligned
                                else:
                                    ijk_aligned = k_aligned[rsels]
                                if hue:
                                    in_sigs = [ijk_aligned[hsel_groups[h][nonan_sel]& sels] for h in hues]
                                else:
                                    in_sigs = [ijk_aligned[sels]]

                                ax = peristimulus_time_trial_average_plot(in_sigs, time_window,
                                                                          (behaviors, "time(ms)", "", hues),
                                                                          ax=axes[i][j])
                                Ns = [str(isig.shape[0]) for isig in in_sigs]
                                opt = "(N: " + ",".join(Ns) + ")"
                                if i == 0:
                                    # TODO: include stats significance and trial N
                                    axes[i][j].set_title(cols[j]+opt)
                                else:
                                    axes[i][j].set_title(opt, fontsize='x-small')
                                if ylims is not None:
                                    axes[i][j].set_ylim(ylims[i][j])
                            axes[i][0].set_ylabel(rows[i] + f' ({meas})')
                        plt.subplots_adjust(hspace=0.3)
                        fig.suptitle(f"{effect_arg} effects on {neur_type} {behaviors} phase {fsig}")
                        fname = os.path.join(subfolder,
                                     f"{neur_type}_{fsig}_{behavior_arg}_{effect_arg}_{animal}_{session}")
                        #plt.tight_layout()
                        fig.savefig(fname + '.png')
                        fig.savefig(fname + '.eps')
                        plt.close(fig)
