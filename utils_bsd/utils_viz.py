import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import seaborn as sns
from matplotlib.colors import Normalize
import copy


def compare_all_model_trajectories(
    data,
    models=None,
    trajs=None,
    figsize=None,
    xmin=100,
    xmax=200,
    hue_order=None,
    title=None,
    palette=None,
):
    all_sims = []
    if trajs is None:
        for m in models:
            if m == "LR":
                data_sim = models[m].fitsim(data.drop(columns="choice_p").copy())
            else:
                model = models[m]
                d = model.fit_marginal(data)
                model.fit(d, method="L-BFGS-B")
                data_sim = model.sim(d, model.fitted_params).copy()
                data_sim["choice_p"] = model.get_proba(data_sim, model.fitted_params)
            data_sim["model"] = m
            all_sims.append(
                data_sim[["Target", "Trial", "Decision", "choice_p", "model"]]
            )
        trajs = pd.concat(all_sims, axis=0)
    # palettes = sns.color_palette('Set2', len(models))
    pad = 0.1
    xs = data["Trial"]
    rs = data["Reward"]

    if figsize is None:
        figsize = (25, 4)
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.plot(xs, data["Target"], "k-")
    ax.scatter(
        data.loc[rs == 1, "Trial"],
        data.loc[rs == 1, "Decision"],
        color="g",
        s=figsize[0],
    )
    ax.scatter(
        data.loc[rs == 0, "Trial"],
        data.loc[rs == 0, "Decision"],
        color="r",
        s=figsize[0],
    )
    ax.plot(xs, data["Decision"], color="gray", ls="--", label="data")
    if hue_order is not None:
        if palette is None:
            cpalette = sns.color_palette("tab10", n_colors=len(hue_order) + 1)[1:]
        elif isinstance(palette, str):
            cpalette = sns.color_palette(palette, n_colors=len(hue_order) + 1)[1:]
        else:
            cpalette = palette
    else:
        cpalette = palette
    sns.lineplot(
        data=trajs,
        x="Trial",
        y="choice_p",
        hue="model",
        palette=cpalette,
        hue_order=hue_order,
    )
    ax.set_xlim((xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin)))
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    if title is not None:
        ax.set_title(title)
    sns.despine()
    return trajs, fig


def plot_2yaxis_offset(
    df, x, z, ax, zmin=None, zmax=None, uspace=0.4, lspace=0.3, offset=0.5, **kwargs
):
    # offset is in y-axis in relative ratio scale, should be between 0 and 1
    yscale = 1 + offset
    if zmin is None:
        zmin = df[z].min()
    if zmax is None:
        zmax = df[z].max()
    # Create a second y-axis for z with transformed coordinates
    ax2 = ax.twinx()
    d = {"color": "b"}
    d.update(kwargs)

    zsel = (df[z] >= zmin) & (df[z] <= zmax)
    ax2.plot(df[x], df[z], label=z, **d)

    # Set transformation for y-axis ticks on the second y-axis
    ax2.yaxis.set_major_locator(
        FixedLocator(np.linspace(zmin, zmax, num=3, endpoint=1))
    )
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.1f}"))
    ax2.set_ylabel(z)

    ymin, ymax = ax.get_ylim()
    y_rg = ymax - ymin
    zrg = (zmax - zmin) * yscale / (1 - uspace - lspace)
    if offset == 0:
        uz = uspace
    else:
        uz = uspace + 1
    zmin1, zmax1 = zmin - zrg * lspace / yscale, zmax + zrg * uz / yscale

    # Set y-axis limits for both y and z
    ax.set_ylim(ymin - y_rg * (yscale - 1), ymax)
    ax2.set_ylim(zmin1, zmax1)  # z limits

    # Show a legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        lines + lines2, labels + labels2, bbox_to_anchor=(1.05, 0.95), loc="upper left"
    )
    return ax2


def neural_ridge_plot(
    data, xts, yts, crop_thres=None, ax=None, zscale=2, zoffset=0, **kwargs
):
    """
    data: float[N x T] matrix with each row representing a time series
    xts: float[T], length T array of timestamps for x axis (columns) in data
    yts: float[T], length T array of timestamps for y axis (rows) in data
    crop_thres: float[T], threshold for cropping the neural traces
    ax: matplotlib ax object
    zscale: determines the space between each neural trace in original units in data (normalize by 2 std)
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
    if crop_thres is None:
        crop_thres = np.full_like(xts, np.max(xts))

    dscale = np.std(data)
    print(dscale)
    xtmax = np.max(xts)

    d = {"linewidth": 1}
    d.update(kwargs)

    dyt = np.min(np.diff(yts))
    for i in range(len(data)):
        upper = crop_thres[i]
        ix = xts
        zs = (data[i] - zoffset) / (zscale * dscale)
        ax.plot(ix, yts[i] - zs, ls="-", **d)
        if upper <= xtmax:
            mrange = yts[i] - zs
            ax.plot([upper, upper], [np.min(mrange), np.max(mrange)])
    ax.invert_yaxis()
    return ax


def plot_task_trajectory(
    data,
    xmin=100,
    xmax=200,
    pad=0.1,
    figsize=(25, 4),
    ax=None,
    return_ax=False,
    **kwargs,
):
    if figsize is None:
        figsize = (25, 4)
        assert ax is not None
        return_ax = True
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    xs = data["Trial"]
    rs = data["Reward"]
    ax.plot(xs, data["Target"], "k-", **kwargs)
    ax.scatter(
        data.loc[rs == 1, "Trial"],
        data.loc[rs == 1, "Decision"],
        color="g",
        label="rewarded",
        s=figsize[0],
        **kwargs,
    )
    ax.scatter(
        data.loc[rs == 0, "Trial"],
        data.loc[rs == 0, "Decision"],
        color="r",
        label="unrewarded",
        s=figsize[0],
        **kwargs,
    )
    ax.plot(xs, data["Decision"], color="gray", ls="--", label="data", **kwargs)
    ax.set_ylabel("Decision")
    ax.set_xlabel("Trial")
    ax.set_xlim((xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin)))
    ax.set_ylim(-0.1, 1.1)
    if return_ax:
        return ax
    return fig, ax


def plot_coord_flip(x, y, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    (curve,) = ax.plot(x, y, **kwargs)
    newx = curve.get_xdata()
    newy = curve.get_ydata()
    curve.set_ydata(newx)
    curve.set_xdata(newy)
    xlim0 = ax.get_xlim()
    ylim0 = ax.get_ylim()
    ax.set_xlim(ylim0)
    ax.set_ylim(xlim0)
    ax.invert_yaxis()
    # ax.clear()
    # curve2, _ = ax.plot(x, y, c='r')
    # curve2.set_xdata(newy)
    # curve2.set_ydata(newx)
    return ax


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
    """ Utility for plotting neural dynamics
    # TODO: adding offset column or z column
    # not labeling after events for now
    t_start, t_end (inclusive): range describing start end ending time
    nb_df: dataframe with neural data
    pse: PS_Expr object]
    event: event for aligning to
    hue: column in nb_df used for hue,
    cropcol: column in nb_df used for cropping timestamps of the signal, RELATIVE TO EVENT
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
    pse,
    event,
    hue=None,
    style=None,
    xlabel=None,
    ylabel=None,
    debase=False,
    **kwargs,
):
    if debase:
        nb_df = nb_df.copy()
        pse.nbm.debase_gradient(nb_df, event)
    plot_df = pse.nbm.lag_wide_df(nb_df, {f"{event}_neur": {"long": True}}).reset_index(
        drop=True
    )
    g = sns.relplot(
        data=plot_df,
        x=f"{event}_neur_time",
        y=f"{event}_neur_ZdFF",
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
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    sns.despine()
    return g
