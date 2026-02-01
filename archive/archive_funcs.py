

def plot_neural_trial_average_archive(
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
    warp_col=None,
    plot_df_callback=None,
    return_plot_df=False,
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
    if aux_cols is None:
        aux_cols = []
    if warp_col is not None:
        aux_cols = aux_cols + [warp_col]
    value_cols = value_cols + aux_cols
    nb_df = nb_df[value_cols].dropna().reset_index(drop=True)
    expr.nbm.nb_cols, expr.nbm.nb_lag_cols = expr.nbm.parse_nb_cols(nb_df)
    if debase:
        expr.nbm.debase_gradient(nb_df, event, base_event, base_ts[0], base_ts[1])
    plot_df = expr.nbm.lag_wide_df(
        nb_df, {f"{event}_neur": {"long": True}}
    ).reset_index(drop=True)
    xcol = f"{event}_neur_time"
    ycol = f"{event}_neur_ZdFF"
    if warp_col is not None:
        # plot_df = plot_df[plot_df[xcol]<=plot_df[warp_col]].reset_index(drop=True)
        rel_time = plot_df[xcol].values / plot_df[warp_col].values
        plot_df[f'rel_{xcol}'] = rel_time
        xcol = f'rel_{xcol}'
        id_cols = expr.nbm.uniq_cols
        # min_data = plot_df.groupby(id_cols).agg({xcol: 'min'}).max().values[0]
        plot_df_warped = plot_df[id_cols + [xcol, ycol]].groupby(id_cols,
                                                             as_index=True).apply(normalize_and_interp,
                                                                                   xcol, ycol, 0, 0.95, include_groups=False).reset_index(level=-1, 
                                                                                                                    drop=True).reset_index()
        key_cols = list(set([c for c in [row, col, hue, style] if c is not None]+aux_cols + id_cols))
        plot_df_warped = plot_df_warped.merge(plot_df[key_cols].drop_duplicates(), on=id_cols, how='left')
        plot_df = plot_df_warped
    if df_sel_func is not None:
        plot_df = df_sel_func(plot_df).reset_index(drop=True)
    if plot_df_callback is not None:
        plot_df_callback(plot_df)
    g = sns.relplot(
        data=plot_df,
        x=xcol,
        y=ycol,
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
    if return_plot_df:
        return g, plot_df
    return g