# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Dat
import numpy as np
import pandas as pd


class NBVisualizer:

    def __init__(self):
        pass


"""###############################################
##################### NBMat ######################
###############################################"""


def trial_av_vline_timedots(data=None, event=None, sort_order=None, id_cols=None, y_pos=None, ylim0=None,
                            time_func=None, **kwargs):
    # assume ypos has no duplicates
    ax = plt.gca()
    if ylim0 is None:
        ylim0 = ax.get_ylim()
    if time_func:
        plt.axvline(time_func(0), c='k', ls='--')
    else:
        plt.axvline(0, c='k', ls='--')
    nbmat = kwargs['nbmat']
    events = nbmat.behavior_events
    ev_colors = sns.color_palette("hls", len(events))
    event_cmap = {events[i]: ev_colors[i] for i in range(len(events))}
    evt_cont_map = {'outcome': ['center_in', 'center_out', 'outcome', 'first_side_out'],
                    'center_out': ['first_side_out{t-1}', 'center_in', 'center_out', 'outcome'],
                    'first_side_out': ['outcome', 'first_side_out', 'center_in{t+1}']}
    # first drop duplicates due to lagging operations
    if id_cols is None:
        id_cols = ['animal', 'session', 'trial']
    dots_df = data[id_cols + evt_cont_map[event]].drop_duplicates(id_cols)

    event_zeros = dots_df[event].values
    dots_df = dots_df.copy()
    for evdots in evt_cont_map[event]:
        dots_df[evdots] = dots_df[evdots].values - event_zeros
    # default ascending
    if sort_order:
        dots_df = dots_df.sort_values(sort_order)
    xmin, xmax = ax.get_xlim()
    if y_pos is None:
        ymin, ymax = ylim0
        oneThird = (ymax - ymin) / 3
        margin = (ymax - ymin) * 0.05
        start_dot = margin + ymax
        end_dot = start_dot + oneThird
        plt.gca().set_ylim(top=ymax + 2 * margin + oneThird)
        total_trials = len(dots_df)
        mradius = oneThird / (3 * total_trials)
        y_pos = end_dot - np.arange(total_trials) * 3 * mradius  # ascending order
    else:
        mradius = 0.15

    for evdots in evt_cont_map[event]:
        # not exact when doing heatmap
        dot_times = dots_df[evdots].values
        if time_func is not None:
            dot_times = np.apply_along_axis(time_func, 0, dot_times)
        sels = (dot_times >= xmin) & (dot_times <= xmax)
        ax.scatter(dot_times[sels], y_pos[sels], color=event_cmap[evdots.split('{')[0]],
                   s=radius2marker_size(mradius))
    return ax


def radius2marker_size(r):
    return np.pi * (plt.gca().transData.transform([r, 0])[0] - plt.gca().transData.transform([0, 0])[0]) ** 2


def df_long_heatmap(data=None, event=None, sort_cols=None, id_cols=None, **kwargs):
    # if 'df_form' in kwargs:
    #     df_form = kwargs['df_form']
    # else:
    #     df_form = 'long'
    # df[f'{event}_neur_time'] = f'{event}_neur' + df[f'{event}_neur_time'].astype('str')
    # df.pivot_table(index=np.setdiff1d(df.columns, twovar), columns=f'{event}_neur_time', values=f'{event}_neur_ZdFF')
    # add dendogram functions
    nbmat = kwargs['nbmat']
    # nbmat.nb_cols[event]
    if id_cols is None:
        id_cols = ['animal', 'session', 'roi', 'trial']

    heat_cols = [f'{event}_neur_time', f'{event}_neur_ZdFF']
    if sort_cols is None:
        # TODO: incorporate session_trial sorting
        sort_cols = ['trial']
    else:
        if 'trial' not in sort_cols:
            sort_cols.append('trial')
    heat_df = data[id_cols + heat_cols + sort_cols].drop_duplicates(id_cols)
    heat_df
    pass


def df_wide_heatmap(data=None, event=None, sort_cols=None, id_cols=None, nbmat=None, **kwargs):
    ax = plt.gca()
    data_original = data
    data = data.reset_index(drop=True)
    if id_cols is None:
        id_cols = ['animal', 'session', 'trial']
    ids = np.add.reduce([data[idc].astype('str') for idc in id_cols])
    assert len(ids) == len(np.unique(ids)), 'suspect not wide form'
    assert nbmat is not None, 'must specify nb_mat'
    heat_cols = nbmat.nb_cols[event + '_neur']
    assert np.all(np.isin(heat_cols, data.columns)), 'nbcols does not contain all the columns?'

    if sort_cols is None:
        # TODO: incorporate session_trial sorting
        sort_cols = ['trial']
    else:
        # add dendogram functions
        # if 'dend' in sort_cols:
        #     data = data.copy()
        #     data['dend'] = 0
        if 'trial' not in sort_cols:
            sort_cols.append('trial')
        for scol in sort_cols:
            if scol in nbmat.behavior_events:
                data[scol] = data[scol].values - data[event].values
    heat_df = data[id_cols + heat_cols + sort_cols[:-1]].drop_duplicates(id_cols)
    # if dend do data['dend'] = dendcluster
    heat_df = heat_df.sort_values(sort_cols)
    # ypos option
    sns.heatmap(heat_df[heat_cols].values, ax=ax, yticklabels=False, cmap='Greys_r', **kwargs)
    # else:
    #     sns.heatmap(heat_df[heat_cols].values, ax=ax, yticklabels=False, **kwargs)
    # heatmap start from top_left corner
    ax.set_yticks([0, len(heat_df) - 1])
    ax.set_yticklabels([1, len(heat_df)])
    times = np.sort(np.core.defchararray.replace(heat_cols,
                                                 event + '_neur|', '', count=None).astype(np.float))
    zero = np.where(times == 0)[0][0]
    times[zero] = 0
    ticks = [0, zero, len(times) - 1]
    tlabels = [times[0], 0, times[-1]]
    ax.axvline(zero, c='k', ls='--')
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabels)
    # ax.set_ylim(0, len(heat_df))
    # ind2time = lambda times: np.arange(len(times)) * (times[-1] - times[0]) / (len(times)-1) + times[0]
    time2ind = lambda t, tmin, tmax, tlen: tlen * (t - tmin) / (tmax - tmin)
    t2i_final = lambda t: time2ind(t, times[0], times[-1], len(times) - 1)
    # remove dendogram
    # TODO: if dend in remove it
    # TODO: ypos to plot on top
    ax = trial_av_vline_timedots(data=data_original, event=event,
                                 sort_order=sort_cols, id_cols=id_cols,
                                 time_func=t2i_final, nbmat=nbmat,
                                 y_pos=np.arange(len(heat_df)), **kwargs)
    return ax


def nb_df_reorder_column(nb_df, column, orders):
    return pd.concat([nb_df[nb_df[column] == order] for order in orders], axis=0)
