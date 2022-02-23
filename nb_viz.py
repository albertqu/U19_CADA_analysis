# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Utils
from utils import df_select_kwargs

class NBVisualizer:

    def __init__(self):
        pass

class RR_NBViz(NBVisualizer):

    def __init__(self):
        super().__init__()


def rr_psychometric(nb_df, animal, session):
    """ Takes
    """
    special_arg = ''
    # Preparation of nb_df for data analysis
    reg_df = nb_df[['animal', 'session', 'trial', 'tone_onset', 'T_Entry', 'choice', 'restaurant', 'tone_prob',
                    'accept']].reset_index(drop=True)
    reg_df['hall_time'] = reg_df['T_Entry'] - reg_df['tone_onset']
    reg_df = df_select_kwargs(reg_df, hall_time=lambda s: (s >= 0))
    reg_df = reg_df[reg_df['hall_time'] <= np.percentile(reg_df['hall_time'], 95)]

    # Data modeling
    X = pd.concat([pd.get_dummies(reg_df['restaurant'].map({i: f'R{i}' for i in range(1, 5)})),
                   reg_df['tone_prob']], axis=1)
    y = reg_df['accept'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RAND_STATE)
    # clf = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train, y_train)
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    y_pred = clf.fit(X, y).predict_proba(X)

    # Convert modeling result to interpretable dataframe
    import itertools
    restaurants = [f'R{i}' for i in range(1, 5)]
    X_base = X.drop_duplicates().reset_index(drop=True)
    clf_psy = clf.fit(X, y)
    y_pred_base = clf_psy.predict_proba(X_base)
    psy_df = X_base.copy()
    psy_df['response'] = y_pred_base[:, 1]
    psy_df['restaurant'] = X_base.iloc[:, :4].idxmax(axis=1)
    logits = X_base.values @ clf_psy.coef_.T + clf_psy.intercept_
    psy_df['logit'] = logits
    Xpsy_df = X.copy()
    Xpsy_df['logit'] = Xpsy_df.values @ clf_psy.coef_.T + clf_psy.intercept_
    Xpsy_df['restaurant'] = X.iloc[:, :4].idxmax(axis=1)
    Xpsy_df['accept'] = y
    plot_df = Xpsy_df.groupby('logit', as_index=False).agg({'accept': 'mean'})
    plot_df['err'] = Xpsy_df.groupby('logit', as_index=False).agg({'accept': lambda xs: np.std(xs) / np.sqrt(len(xs))})[
        'accept']
    # Plotting Psychometric curve
    tone_palette = sns.color_palette('coolwarm', n_colors=4)
    r_markers = ['o', 's', '+', 'x']
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    sns.set_context('talk')
    ax.errorbar(plot_df['logit'], plot_df['accept'], yerr=plot_df['err'], color='k', zorder=-1, ls='none')
    psydf_xy = psy_df[['logit', 'response']].sort_values('logit')
    ax.plot(psydf_xy['logit'], psydf_xy['response'], color='brown', zorder=0)
    for i in range(4):
        r = f'R{i + 1}'
        for j, tone in enumerate([0, 20, 80, 100]):
            r_sel = psy_df['restaurant'] == r
            tone_sel = psy_df['tone_prob'] == tone
            if i == 0:
                if j == 0:
                    lab = f'{r}_{tone}'
                else:
                    lab = tone
                ax.scatter(psy_df.loc[r_sel & tone_sel, 'logit'], psy_df.loc[r_sel & tone_sel, 'response'],
                           marker=r_markers[i], color=tone_palette[j], label=lab, zorder=1)
            else:
                if j == 0:
                    ax.scatter(psy_df.loc[r_sel & tone_sel, 'logit'], psy_df.loc[r_sel & tone_sel, 'response'],
                               marker=r_markers[i], color=tone_palette[j], label=r, zorder=1)
                ax.scatter(psy_df.loc[r_sel & tone_sel, 'logit'], psy_df.loc[r_sel & tone_sel, 'response'],
                           marker=r_markers[i], color=tone_palette[j], zorder=1)
    plt.legend(loc=4)
    plt.ylim([0, 1])
    sns.despine()
    animal = reg_df['animal'].unique()[0]
    plt.suptitle(f'RR behavior {animal} {session}{special_arg}')
    ax.set_xlabel('action logit')
    ax.set_ylabel('Accept%')


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
