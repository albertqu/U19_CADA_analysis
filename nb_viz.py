# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Utils
from utils import df_select_kwargs, RAND_STATE

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import logging
from matplotlib.colors import Normalize
import copy

class PlotlyFig:
    # TODO: add automatic color palette

    def __init__(
        self, rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1, **kwargs
    ):
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=shared_xaxes,
            vertical_spacing=vertical_spacing,
            **kwargs,
        )
        self.fig = fig

    def plot(
        self, x, y, name="", color="blue", mode="lines+markers", row=1, col=1, **kwargs
    ):
        self.fig.add_trace(
            go.Scatter(x=x, y=y, mode=mode, name=name, line=dict(color=color)),
            row=row,
            col=col,
            **kwargs
        )

    def show(self):
        self.fig.show()


class NBVisualizer:
    def __init__(self, expr):
        self.expr = expr
        pass


class RR_NBViz(NBVisualizer):
    def __init__(self, expr):
        super().__init__(expr)
        self.expr = expr

    def psychometric(self, nb_df, behavior_var="accept"):
        # setting up data, identify useful features
        reg_df = nb_df[
            [
                "animal",
                "session",
                "trial",
                "tone_onset",
                "T_Entry",
                "choice",
                "restaurant",
                "tone_prob",
                "accept",
                "stimulation_on",
                "stimulation_off",
            ]
        ].reset_index(drop=True)
        reg_df["hall_time"] = reg_df["T_Entry"] - reg_df["tone_onset"]
        reg_df["decision_time"] = reg_df["choice"] - reg_df["tone_onset"]
        # TODO: classify trials into baseline, postStim, noSTIM, STIM, (restDay),
        reg_df = df_select_kwargs(reg_df, hall_time=lambda s: (s >= 0)).reset_index(
            drop=True
        )
        reg_df = reg_df[
            reg_df["decision_time"] <= np.percentile(reg_df["decision_time"], 95)
        ].reset_index(drop=True)
        reg_df["restaurant"] = reg_df["restaurant"].map(
            {i: f"R{i}" for i in range(1, 5)}
        )

        # compute action value TODO using baseline trials
        endog_map = self.expr.nbm.fit_action_value_function(
            reg_df[reg_df["stimulation_on"].isnull()].reset_index(drop=True)
        )
        reg_df = self.expr.nbm.add_action_value_feature(reg_df, endog_map)

        # STEP 3: set up plotting to plot logit against behavioral variables
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        sns.set_context("talk")
        sns.regplot(
            x="action_logit",
            y="accept",
            data=reg_df[reg_df["stimulation_on"].isnull()],
            x_estimator=np.mean,
            logistic=True,
            n_boot=500,
            scatter_kws={"zorder": 0},
            color="k",
            marker="",
            ax=ax,
        )
        sns.regplot(
            x="action_logit",
            y="accept",
            data=reg_df[~reg_df["stimulation_on"].isnull()],
            x_estimator=np.mean,
            logistic=True,
            n_boot=500,
            scatter_kws={"zorder": 0},
            color="g",
            ax=ax,
        )

        slice_df = reg_df.loc[
            reg_df["stimulation_on"].isnull(),
            ["restaurant", "tone_prob", "action_logit", behavior_var],
        ].reset_index(drop=True)
        plot_df = slice_df.groupby(
            ["restaurant", "tone_prob", "action_logit"], as_index=False
        ).agg({behavior_var: "mean"})
        sns.scatterplot(
            x="action_logit",
            y=behavior_var,
            data=plot_df,
            style="restaurant",
            hue="tone_prob",
            palette="coolwarm",
            ax=ax,
            s=120,
            linewidth=0,
            zorder=10,
        )
        ax.set_ylim((0, 1.1))
        return fig


def rr_psychometric(reg_df, special_arg, suptitle=None):
    special_arg = ""
    X = pd.concat(
        [
            pd.get_dummies(reg_df["restaurant"].map({i: f"R{i}" for i in range(1, 5)})),
            reg_df["tone_prob"],
        ],
        axis=1,
    )
    y = reg_df["accept"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RAND_STATE
    )
    # clf = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train, y_train)
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    y_pred = clf.fit(X, y).predict_proba(X)

    # Convert modeling result to interpretable dataframe
    restaurants = [f"R{i}" for i in range(1, 5)]
    X_base = X.drop_duplicates().reset_index(drop=True)
    clf_psy = clf.fit(X, y)
    y_pred_base = clf_psy.predict_proba(X_base)
    psy_df = X_base.copy()
    psy_df["response"] = y_pred_base[:, 1]
    psy_df["restaurant"] = X_base.iloc[:, :4].idxmax(axis=1)
    logits = X_base.values @ clf_psy.coef_.T + clf_psy.intercept_
    psy_df["logit"] = logits
    Xpsy_df = X.copy()
    Xpsy_df["logit"] = Xpsy_df.values @ clf_psy.coef_.T + clf_psy.intercept_
    Xpsy_df["restaurant"] = X.iloc[:, :4].idxmax(axis=1)
    Xpsy_df["accept"] = y
    plot_df = Xpsy_df.groupby("logit", as_index=False).agg({"accept": "mean"})
    plot_df["err"] = Xpsy_df.groupby("logit", as_index=False).agg(
        {"accept": lambda xs: np.std(xs) / np.sqrt(len(xs))}
    )["accept"]
    # Plotting Psychometric curve
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    sns.set_context("talk")
    ax.errorbar(
        plot_df["logit"],
        plot_df["accept"],
        yerr=plot_df["err"],
        color="k",
        zorder=-1,
        ls="none",
    )
    psydf_xy = psy_df[["logit", "response"]].sort_values("logit")
    ax.plot(psydf_xy["logit"], psydf_xy["response"], color="brown", zorder=0)
    tone_palette = sns.color_palette("coolwarm", n_colors=4)
    r_markers = ["o", "s", "+", "x"]
    for i in range(4):
        r = f"R{i+1}"
        for j, tone in enumerate([0, 20, 80, 100]):
            r_sel = psy_df["restaurant"] == r
            tone_sel = psy_df["tone_prob"] == tone
            if i == 0:
                if j == 0:
                    lab = f"{r}_{tone}"
                else:
                    lab = tone
                ax.scatter(
                    psy_df.loc[r_sel & tone_sel, "logit"],
                    psy_df.loc[r_sel & tone_sel, "response"],
                    marker=r_markers[i],
                    color=tone_palette[j],
                    label=lab,
                    zorder=1,
                )
            else:
                if j == 0:
                    ax.scatter(
                        psy_df.loc[r_sel & tone_sel, "logit"],
                        psy_df.loc[r_sel & tone_sel, "response"],
                        marker=r_markers[i],
                        color=tone_palette[j],
                        label=r,
                        zorder=1,
                    )
                ax.scatter(
                    psy_df.loc[r_sel & tone_sel, "logit"],
                    psy_df.loc[r_sel & tone_sel, "response"],
                    marker=r_markers[i],
                    color=tone_palette[j],
                    zorder=1,
                )
    plt.legend(loc=4)
    plt.ylim([0, 1])
    sns.despine()
    animal = reg_df["animal"].unique()[0]
    if suptitle is not None:
        plt.suptitle(suptitle)
    ax.set_xlabel("action logit")
    ax.set_ylabel("Accept%")


"""###############################################
##################### NBMat ######################
###############################################"""


def get_sample_size_facegrid(
    data=None, row=None, col=None, hue=None, style=None, **kwargs
):
    def sample_size_recursive(data, categories, pre_arg=""):
        if categories:
            category = categories[0]
            assert category in data.columns, f"DATA must contain category {category}"
            for ctg in np.unique(data[category]):
                ctg_arg = f"category={ctg}"
                sample_size_recursive(
                    data[data[category] == ctg],
                    categories[1:],
                    pre_arg + ", " + ctg_arg,
                )
        else:
            sub_df = data[["animal", "session", "trial"]].drop_duplicates()
            n_animal = len(sub_df["animal"].unique())
            n_trial = len(sub_df)
            n_session = len(np.unique(sub_df["animal"] + sub_df["session"]))
            print(pre_arg + f": A:{n_animal}, S: {n_session}, T: {n_trial}")

    prearg = ""
    if col is not None:
        prearg = f"{col}={data[col].unique()[0]}"
    if row is not None:
        prearg = prearg + f", {row}={data[row].unique()[0]}"
    sample_size_recursive(data, [c for c in [hue, style] if c is not None], prearg)


def trial_av_vline_timedots(
    data=None,
    event=None,
    sort_order=None,
    id_cols=None,
    y_pos=None,
    ylim0=None,
    time_func=None,
    peri_event_map=None,
    palette=None,
    mradius=None,
    **kwargs,
):
    # assume ypos has no duplicates
    ax = plt.gca()
    if ylim0 is None:
        ylim0 = ax.get_ylim()
    if time_func:
        plt.axvline(time_func(0), c="k", ls="--")
    else:
        plt.axvline(0, c="k", ls="--")
    nbmat = kwargs["nbmat"]
    events = nbmat.behavior_events
    if isinstance(palette, str):
        ev_colors = sns.color_palette(palette, len(events))
    elif palette is not None:
        ev_colors = palette
    else:
        ev_colors = sns.color_palette("hls", len(events))
    event_cmap = {events[i]: ev_colors[i] for i in range(len(events))}
    if peri_event_map is not None:
        evt_cont_map = peri_event_map
    else:
        evt_cont_map = {
            "outcome": [
                "center_in",
                "center_out",
                "outcome",
                "first_side_out",
                "center_in{t+1}",
            ],
            "center_out": ["first_side_out{t-1}", "center_in", "center_out", "outcome"],
            "first_side_out": ["outcome", "first_side_out", "center_in{t+1}"],
        }
    # first drop duplicates due to lagging operations
    if id_cols is None:
        id_cols = ["animal", "session", "trial"]
    dots_df = data[id_cols + evt_cont_map[event]].drop_duplicates(id_cols)

    event_zeros = dots_df[event].values
    dots_df = dots_df.copy()
    for evdots in evt_cont_map[event]:
        dots_df[evdots] = dots_df[evdots].values - event_zeros
    # default ascending
    if sort_order:
        # assuming sort_order are strings
        data_order_cols = [so for so in sort_order if so not in dots_df.columns]
        if data_order_cols:
            dots_df[data_order_cols] = data[data_order_cols]
        dots_df = dots_df.sort_values(sort_order)
        if data_order_cols:
            dots_df.drop(columns=data_order_cols, inplace=True)
    xmin, xmax = ax.get_xlim()
    if y_pos is None:
        ymin, ymax = ylim0
        oneThird = (ymax - ymin) / 3
        margin = (ymax - ymin) * 0.05
        start_dot = margin + ymax
        end_dot = start_dot + oneThird
        plt.gca().set_ylim(top=ymax + 2 * margin + oneThird)
        total_trials = len(dots_df)
        if mradius is None:
            mradius = oneThird / (3 * total_trials)
        y_pos = end_dot - np.arange(total_trials) * 3 * mradius  # ascending order
    else:
        if mradius is None:
            mradius = 0.15

    event_labeled = {ev.split("{")[0]: False for ev in evt_cont_map[event]}

    for evdots in evt_cont_map[event]:
        # not exact when doing heatmap
        dot_times = dots_df[evdots].values
        if time_func is not None:
            dot_times = np.apply_along_axis(time_func, 0, dot_times)
        sels = (dot_times >= xmin) & (dot_times <= xmax)
        color_event = evdots.split("{")[0]
        if event_labeled[color_event]:
            ax.scatter(
                dot_times[sels],
                y_pos[sels],
                color=event_cmap[color_event],
                s=radius2marker_size(mradius),
            )
        else:
            ax.scatter(
                dot_times[sels],
                y_pos[sels],
                color=event_cmap[color_event],
                s=radius2marker_size(mradius),
                label=color_event,
            )

    return ax


def df_wide_heatmap(
    data=None,
    event=None,
    sort_cols=None,
    id_cols=None,
    nbmat=None,
    cmap=None,
    peri_event_map=None,
    peri_event_palette=None,
    # arbitrary alignment time
    **kwargs,
):
    if cmap is None:
        cmap_opt = "Greys_r"
    else:
        cmap_opt = cmap
    ax = plt.gca()
    data_original = data
    data = data.reset_index(drop=True)
    if id_cols is None:
        id_cols = ["animal", "session", "trial"]
    ids = np.add.reduce([data[idc].astype("str") for idc in id_cols])
    assert len(ids) == len(np.unique(ids)), "suspect not wide form"
    assert nbmat is not None, "must specify nb_mat"
    heat_cols = nbmat.nb_cols[event + "_neur"]
    assert np.all(
        np.isin(heat_cols, data.columns)
    ), "nbcols does not contain all the columns?"

    if sort_cols is None:
        # TODO: incorporate session_trial sorting
        sort_cols = ["trial"]
    else:
        # add dendogram functions
        # if 'dend' in sort_cols:
        #     data = data.copy()
        #     data['dend'] = 0
        if "trial" not in sort_cols:
            sort_cols.append("trial")
        for scol in sort_cols:
            if scol in nbmat.behavior_events:
                data[scol] = data[scol].values - data[event].values
    heat_df = data[id_cols + heat_cols + sort_cols[:-1]].drop_duplicates(id_cols)
    # if dend do data['dend'] = dendcluster
    heat_df = heat_df.sort_values(sort_cols)
    # ypos option
    sns.heatmap(
        heat_df[heat_cols].values, ax=ax, yticklabels=False, cmap=cmap_opt, **kwargs
    )
    # else:
    #     sns.heatmap(heat_df[heat_cols].values, ax=ax, yticklabels=False, **kwargs)
    # heatmap start from top_left corner
    ax.set_yticks([0, len(heat_df) - 1])
    ax.set_yticklabels([1, len(heat_df)])
    times = np.sort(
        np.core.defchararray.replace(
            heat_cols, event + "_neur|", "", count=None
        ).astype(float)
    )
    zero = np.where(times == 0)[0][0]
    times[zero] = 0
    ticks = [0, zero, len(times) - 1]
    tlabels = [times[0], 0, times[-1]]
    ax.axvline(zero, c="k", ls="--")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabels)
    # ax.set_ylim(0, len(heat_df))
    # ind2time = lambda times: np.arange(len(times)) * (times[-1] - times[0]) / (len(times)-1) + times[0]
    time2ind = lambda t, tmin, tmax, tlen: tlen * (t - tmin) / (tmax - tmin)
    t2i_final = lambda t: time2ind(t, times[0], times[-1], len(times) - 1)
    # remove dendogram
    # TODO: if dend in remove it
    # TODO: ypos to plot on top
    ax = trial_av_vline_timedots(
        data=data_original,
        event=event,
        sort_order=sort_cols,
        id_cols=id_cols,
        time_func=t2i_final,
        nbmat=nbmat,
        y_pos=np.arange(len(heat_df)),
        peri_event_map=peri_event_map,
        palette=peri_event_palette,
        **kwargs,
    )
    return ax


def radius2marker_size(r):
    return (
        np.pi
        * (
            plt.gca().transData.transform([r, 0])[0]
            - plt.gca().transData.transform([0, 0])[0]
        )
        ** 2
    )


def df_long_heatmap(data=None, event=None, sort_cols=None, id_cols=None, **kwargs):
    # if 'df_form' in kwargs:
    #     df_form = kwargs['df_form']
    # else:
    #     df_form = 'long'
    # df[f'{event}_neur_time'] = f'{event}_neur' + df[f'{event}_neur_time'].astype('str')
    # df.able(index=np.setdiff1d(df.columns, twovar), columns=f'{event}_neur_time', values=f'{event}_neur_ZdFF')
    # add dendogram functions
    nbmat = kwargs["nbmat"]
    # nbmat.nb_cols[event]
    if id_cols is None:
        id_cols = ["animal", "session", "roi", "trial"]

    heat_cols = [f"{event}_neur_time", f"{event}_neur_ZdFF"]
    if sort_cols is None:
        # TODO: incorporate session_trial sorting
        sort_cols = ["trial"]
    else:
        if "trial" not in sort_cols:
            sort_cols.append("trial")
    heat_df = data[id_cols + heat_cols + sort_cols].drop_duplicates(id_cols)
    heat_df
    pass


def nb_df_reorder_column(nb_df, column, orders):
    return pd.concat([nb_df[nb_df[column] == order] for order in orders], axis=0)


def plot_nb_df_rr(nb_df, data_cols, idvars):
    nb_df_plot = nb_df[data_cols + idvars]
    nb_df_neur = pd.melt(
        nb_df_plot,
        id_vars=idvars,
        value_vars=data_cols,
        var_name="hemi_event_time",
        value_name="dZF",
    )
    nb_df_neur[["hemi", "event_time"]] = nb_df_neur["hemi_event_time"].str.split(
        "--", expand=True
    )
    nb_df_neur[["event", "neur_time"]] = nb_df_neur["event_time"].str.split(
        "|", expand=True
    )
    nb_df_neur["neur_time"] = nb_df_neur["neur_time"].astype(np.float)
    sns.relplot(
        data=nb_df_neur[nb_df_neur["event"] == "T_Entry"],
        col="hemi",
        x="neur_time",
        y="dZF",
        row="animal",
        kind="line",
        hue="decision",
    )


def plot_correlation(dataset: pd.DataFrame, dendo=True) -> None:
    corrs = dataset.corr(method="pearson")
    if np.any(corrs.isnull().values.ravel()):
        logging.warning("null entries in correlation matrix, check data")
    corrmap = sns.clustermap(
        corrs.fillna(0),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Correlation"},
        figsize=(12, 12),
    )
    corrmap.ax_row_dendrogram.set_visible(dendo)
    corrmap.ax_col_dendrogram.set_visible(dendo)
    return corrmap


def df_barplot_w_ebar(
    df,
    x,
    y,
    ebar,
    ebar_u=None,
    hue=None,
    x_order=None,
    hue_order=None,
    width=0.8,
    ax=None,
    palette=None,
):
    if x_order is None:
        xs = df[x].unique()
    else:
        xs = x_order

    ebar_args = [ebar] if ebar_u is None else [ebar, ebar_u]
    if hue is not None:
        if hue_order is None:
            hues = df[hue].unique()
        else:
            hues = hue_order
        k_hue = len(hues)
        uw = width / k_hue
        ys = [[0] * len(xs) for _ in range(k_hue)]
        if ebar_u is None:
            ebars = [[0] * len(xs) for _ in range(k_hue)]
        else:
            ebars = [np.zeros((2, len(xs))) for _ in range(k_hue)]
        for i in range(k_hue):
            for j in range(len(xs)):
                vs = df.loc[
                    (df[x] == xs[j]) & (df[hue] == hues[i]), [y] + ebar_args
                ].values.ravel()
                ys[i][j] = vs[0]
                if ebar_u is None:
                    ebars[i][j] = vs[1]
                else:
                    ebars[i][:, j] = vs[1:]
    else:
        uw = width
        k_hue = 1
        ys = [[0] * len(xs)]
        if ebar_u is None:
            ebars = [[0] * len(xs)]
        else:
            ebars = [np.zeros((2, len(xs)))]
        for j in range(len(xs)):
            vs = df.loc[df[x] == xs[j], [y] + ebar_args].values.ravel()
            assert len(vs) <= 3, "duplicate values for x, hue pairs"
            ys[0][j] = vs[0]
            if ebar_u is None:
                ebars[0][j] = vs[1]
            else:
                ebars[0][:, j] = vs[1:]

    # Position of bars on x-axis
    ind = np.arange(len(xs))

    if ax is None:
        # Figure size
        plt.figure(figsize=(10, 5))
        ax = plt.gca()

    if palette is None:
        palette = sns.color_palette(n_colors=k_hue)
    # Plotting
    for i in range(k_hue):
        ax.bar(ind + i * uw, ys[i], width=uw, yerr=ebars[i], color=palette[i])
    if hue is not None:
        ax.legend(hues, loc="best")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xticks(ind + width / 2, xs)
    return ax


def get_sample_size_facegrid(
    data=None, row=None, col=None, hue=None, style=None, **kwargs
):

    def sample_size_recursive(data, categories, pre_arg=""):
        if categories:
            category = categories[0]
            assert category in data.columns, f"DATA must contain category {category}"
            for ctg in np.unique(data[category]):
                ctg_arg = f"category={ctg}"
                sample_size_recursive(
                    data[data[category] == ctg],
                    categories[1:],
                    pre_arg + ", " + ctg_arg,
                )
        else:
            sub_df = data[["animal", "session", "trial"]].drop_duplicates()
            n_animal = len(sub_df["animal"].unique())
            n_trial = len(sub_df)
            n_session = len(np.unique(sub_df["animal"] + sub_df["session"]))
            print(pre_arg + f": A:{n_animal}, S: {n_session}, T: {n_trial}")

    prearg = ""
    if col is not None:
        prearg = f"{col}={data[col].unique()[0]}"
    if row is not None:
        prearg = prearg + f", {row}={data[row].unique()[0]}"
    sample_size_recursive(data, [c for c in [hue, style] if c is not None], prearg)


def neural_dynamics_plot(
    nb_df,
    pse,
    event,
    t_start=0,
    t_end=1,
    hue=None,
    cropcol=None,
    vmin=None,
    vmax=None,
    ax=None,
    palette="coolwarm",
    **kwargs,
):
    """
    # TODO: adding offset column or z column
    # not labeling after events for now
    t_start, t_end (inclusive): range describing start end ending time
    nb_df: dataframe with neural data
    pse: PS_Expr object]
    event: event for aligning to
    hue: column in nb_df used for hue,
    cropcol: column in nb_df used for cropping timestamps of the signal
    """
    neur_cols = pse.nbm.nb_cols[pse.nbm.default_ev_neur(event)]
    col_sels = [
        c
        for c in neur_cols
        if (pse.nbm.align_time_in(c, t_start, t_end, include_upper=True))
    ]
    xts = np.array([float(c.split("|")[1]) for c in col_sels])
    total_cols = copy.deepcopy(col_sels)
    if hue is not None:
        total_cols.append(hue)
    if cropcol is not None:
        total_cols.append(cropcol)
    data_df = nb_df.dropna(subset=total_cols).reset_index(drop=True)
    data = data_df[col_sels].values
    if hue is not None:
        hue_vals = data_df[hue].values
        vmin = np.min(hue_vals) if vmin is None else vmin
        vmax = np.max(hue_vals) if vmax is None else vmax
        norm_f = Normalize(vmin=vmin, vmax=vmax)
        cmap = sns.color_palette(palette, as_cmap=True)
        plt.colorbar(plt.cm.ScalarMappable(norm=norm_f, cmap=cmap), ax=ax, label=hue)

    if cropcol is None:
        crop_thres = np.full(len(data), np.max(xts))
    else:
        crop_thres = data_df[cropcol].values
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()

    d = {"linewidth": 0.5}
    d.update(kwargs)
    for i in range(len(data)):
        upper = crop_thres[i]
        ix = xts[xts <= upper]
        zs = data[i]
        if hue is not None:
            hue_val = hue_vals[i]
            ax.plot(ix, zs[xts <= upper], ls="-", color=cmap(norm_f(hue_val)), **d)
        else:
            ax.plot(ix, zs[xts <= upper], ls="-", **d)
    return ax


def plot_neural_trial_average(
    nb_df,
    expr,
    event,
    row=None,
    col=None,
    hue=None,
    style=None,
    xlabel=None,
    ylabel=None,
    debase=False,
    base_event=None,
    base_ts=(0, 0),
    t_range=None,
    verbose=True,
    aux_cols=None,
    df_sel_func=None,
    **kwargs,
):
    """
    Function to plot trial average, using seaborn as underlying mechanism to automate figure multiplexing.
    expr: neurobehavior experiment object used to organize data frames
    event: str
        behavior event to align
    row, col, hue, style: str, denoting columns used for sub-dataframe selections
        detail regarding multiplexing behavior refer to [`seaborn.relplot`](https://seaborn.pydata.org/generated/seaborn.relplot.html)
    xlabel, ylabel: str
        customary x/y label
    debase: bool
        whether to remove baseline data
    base_event: str
        baseline event to use as a baseline
    base_ts: tuple
        organized as (b_start, b_end), denoting starting time, end time to be used for baselining purpose, respectively.
    t_range: tuple
        organized as (t_start, t_end), denoting starting time, end time for neural event.
    verbose: True
        if True, print out sample size information
    **kwargs:
        optional keyword arguments for relplot
    """
    expr.nbm.nb_cols, expr.nbm.nb_lag_cols = expr.nbm.parse_nb_cols(nb_df)
    value_cols = list(
        set([p for p in [row, col, hue, style] if (p is not None)] + expr.nbm.uniq_cols)
    )
    ev_cols = expr.nbm.nb_cols[f"{event}_neur"]
    if t_range is not None:
        value_cols = value_cols + [
            c
            for c in ev_cols
            if expr.nbm.align_time_in(c, t_range[0], t_range[1], True)
        ]
    else:
        value_cols = value_cols + ev_cols
    if (base_event is not None) and (base_event != event):
        value_cols = value_cols + [
            c
            for c in expr.nbm.nb_cols[f"{base_event}_neur"]
            if expr.nbm.align_time_in(c, base_ts[0], base_ts[1], True)
        ]
    if aux_cols is not None:
        value_cols = value_cols + aux_cols
    nb_df = nb_df[value_cols].dropna().reset_index(drop=True)
    expr.nbm.nb_cols, expr.nbm.nb_lag_cols = expr.nbm.parse_nb_cols(nb_df)
    if debase:
        expr.nbm.debase_gradient(nb_df, event, base_event, base_ts[0], base_ts[1])
    plot_df = expr.nbm.lag_wide_df(
        nb_df, {f"{event}_neur": {"long": True}}
    ).reset_index(drop=True)
    xcol = f"{event}_neur_time"
    if df_sel_func is not None:
        plot_df = df_sel_func(plot_df).reset_index(drop=True)

    g = sns.relplot(
        data=plot_df,
        x=xcol,
        y=f"{event}_neur_ZdFF",
        row=row,
        col=col,
        hue=hue,
        style=style,
        kind="line",
        **kwargs,
    )
    if ylabel is not None:
        g.set_ylabels(ylabel)
    else:
        g.set_ylabels("Z(DA)")
    if xlabel is not None:
        g.set_xlabels(xlabel)
    else:
        g.set_xlabels(f"Time since {event} revealed (s)")
    g.map_dataframe(lambda data, **kwargs: plt.gca().axvline(0, c="gray", ls="--"))
    if verbose:
        g.map_dataframe(
            get_sample_size_facegrid, row=row, col=col, hue=hue, style=style
        )
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    sns.despine()
    return g
