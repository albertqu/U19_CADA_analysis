import seaborn as sns
from nb_viz import get_sample_size_facegrid
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def fig_subplots_best_dim(n, width=5, height= 3):
    col = int(np.ceil(np.sqrt(n)))
    row = int(np.ceil(n / col))
    fig, axes = plt.subplots(row, col, figsize=(col * width, row * height))
    return fig, axes

def visualize_model_posthoc_loss(model, X, y, aux_df, aux_cols, endog_vars, 
                                 loss_type='clf'):
    """
    Analyze posthoc prediction loss w.r.t. auxiliary (excluded) features to understand what features need to be 
    further included to improve prediction

    Parameters
    ----------  
    model : sklearn model
    X : pd.DataFrame
    y : pd.Series    
    aux_df : pd.DataFrame
    aux_cols : list
        list of columns in aux_df needed for analysis
    endog_vars : list
        list of columns in X needed for analysis
    loss_type: str
        'clf', 'reg_abs', 'reg_residual'
    """
    posthoc_df = pd.concat([X, y], axis=1)
    posthoc_df[aux_cols] = aux_df.iloc[X.index][aux_cols]
    y_vec = y.values.ravel()
    model.fit(X, y_vec)
    if loss_type == 'clf':
        y_proba = model.predict_proba(X)[:, 1]
        loss = -y_vec * np.log(y_proba) - (1 - y_vec) * np.log(1 - y_proba)
    elif loss_type == 'reg_abs':
        y_pred = model.predict(X)
        loss = np.abs(y_vec - y_pred)
    elif loss_type == 'reg_resid':
        y_pred = model.predict(X)
        loss = y_vec - y_pred
    posthoc_df['loss'] = loss
    fig, axes = fig_subplots_best_dim(len(aux_cols), width=5, height= 3)
    axes = axes.ravel()
    for i, col in enumerate(aux_cols+endog_vars):
        sns.scatterplot(ax=axes[i], data=posthoc_df, x=col, y='loss', s=20, alpha=0.5, color='gray')
        sns.regplot(ax=axes[i], data=posthoc_df, x=col, y='loss', scatter=False, lowess=True, line_kws={'color':'k', 'linewidth':2})
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('loss')
    for j in range(i+1, len(aux_cols)):
        axes[j].set_visible(False)
    sns.despine()
    fig.subplots_adjust(hspace=0.6)
    return fig


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
    nb_df = nb_df[value_cols].dropna().reset_index(drop=True)
    expr.nbm.nb_cols, expr.nbm.nb_lag_cols = expr.nbm.parse_nb_cols(nb_df)
    if debase:
        expr.nbm.debase_gradient(nb_df, event, base_event, base_ts[0], base_ts[1])
    plot_df = expr.nbm.lag_wide_df(
        nb_df, {f"{event}_neur": {"long": True}}
    ).reset_index(drop=True)
    xcol = f"{event}_neur_time"

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
