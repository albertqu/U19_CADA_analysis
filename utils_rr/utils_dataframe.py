import logging
import numpy as np
import pandas as pd
from typing import List
from statsmodels.tsa.tsatools import lagmat
from typing import List

def df_col_is_str(df, c):
    return df[c].dtype == object and isinstance(df.iloc[0][c], str)

def get_lagged_cols(df, cols, maxlag, trim='forward'):
    """
    Gen I function for lagging columns, it is built with the assumption to compute past lag.
    """
    lagcols = [f'{c}_lag{i}' for i in range(1, maxlag+1) for c in cols]
    lagdf = pd.DataFrame(lagmat(df[cols].values, maxlag=maxlag, trim=trim, original="ex"), columns=lagcols)
    lagdf.iloc[:maxlag, :] = np.nan
    return lagdf

def df_lagmat(df:pd.DataFrame, cols:List[str], back:int = 0, forw:int=0) -> pd.DataFrame:
    """ 
    This function takes in a dataframe and returns a dataframe with each column in cols
    lagged by forw and back. This function keeps NaN behaviors consistent with 
    input, and fills in boundary rows in lagged columns with NaNs.

    Objective:
    1. we wish to perform forward and back lags for multiple columns in a single lagmat call
    2. we wish to fill in boundary rows with NaNs
    3. we wish to retain the freedom to leave in 

    Fun fact: this task is surprisingly hard for LLMs to generate in a general and 
    efficient manner. (sonnet-3.7, deepseek-R1)

    TODO: implement df_lagmap for a more general lagging function, that can be used to assign 
    arbitrary lag to different columns

    Parameters:
    -----------
    df: dataframe
    cols: list of columns to lag
    forw: number of forward lags
    back: number of backward lags    
    Returns:
    --------
    lagged_df: dataframe with lagged columns
    """
    if forw == 0 and back == 0:
        return None
    lagged_data = []
    lagcols = []
    ncol = len(cols)
    if forw > 0:
        lm_forw = lagmat(df[cols].values, maxlag=forw, trim='backward', original="in")[:, :-len(cols)]
        for i in range(-forw, 0):
            lm_forw[i:, (forw+i) * ncol: (forw+i+1) * ncol] = np.nan
        lagged_data.append(lm_forw)
        lagcols.extend([f"{c}__f{-i}" for i in range(-forw, 0) for c in cols])
    
    if back > 0:
        lm_back = lagmat(df[cols].values, maxlag=back, trim='forward', original="ex")
        for i in range(1, back+1):
            lm_back[:i, (i-1) * ncol: i * ncol] = np.nan
        lagged_data.append(lm_back)
        lagcols.extend([f"{c}__b{i}" for i in range(1, back+1) for c in cols])
    return pd.DataFrame(np.concatenate(lagged_data, axis=1), columns=lagcols)

def df_select_kwargs(df, return_index=False, **kwargs):
    """R style select for pd.DataFrame
    Alternatively can use query method but very ugly syntax
    def query_method(pse, proj, qarg=""):
        self=pse
        proj_sel = self.meta['animal'].str.startswith(proj)
        full_qarg = f"animal.str.startswith('{proj}').values"
        if qarg:
            full_qarg = full_qarg + ' & ' + qarg
        print(full_qarg)
        return self.meta.query(full_qarg)
    query_method(pse, 'BSD', f"(animal_ID=='{animal}') & (session=='{session}')")
    """
    for kw in kwargs:
        if kw in df.columns:
            karg = kwargs[kw]
            if not hasattr(karg, "__call__"):
                kwargs[kw] = (lambda a: (lambda s: s == a))(karg)
        else:
            logging.warning(f"keyword argument key {kw} is not in dataframe!")
    # add function capacity
    df_sel = np.logical_and.reduce(
        [kwargs[kw](df[kw]) for kw in kwargs if kw in df.columns]
    )
    if return_index:
        return df_sel
    else:
        return df[df_sel]


def df_melt_lagged_features(df, feat, id_vars, value_vars=None):
    if value_vars is None:
        value_vars = [c for c in df.columns if (feat in c)]
    df = pd.melt(df, id_vars, value_vars, f"{feat}_arg", value_name=f"{feat}_value")
    df[f"{feat}_lag"] = (
        df[f"{feat}_arg"]
        .str.replace(feat, "")
        .apply(lambda x: x[1:-1].replace("t", ""))
    )
    df.loc[df[f"{feat}_lag"] == "", f"{feat}_lag"] = 0
    df[f"{feat}_lag"] = df[f"{feat}_lag"].astype(np.int)
    df.drop(columns=f"{feat}_arg", inplace=True)
    return df

def pds_is_valid(pds):
    if pd.api.types.is_string_dtype(pds):
        return ~(pds.isnull() | (pds == ""))
    else:
        return ~pds.isnull()

def pds_neq(x, y):
    return pds_is_valid(x) & pds_is_valid(y) & (x != y)

def df_groupwise_shift(df, col, lag):
    if lag > 0:
        arg = f't-{lag}'
    elif lag < 0:
        arg = f't+{-lag}'
    new_col = col+'{'+arg+'}'
    df[new_col] = df[col].shift(lag)
    if lag > 0:
        df.loc[df['trial'] == 1, new_col] = np.nan
    elif lag < 0:
        df.loc[df['lastTrial']==True, new_col] = np.nan
    return df

####################################################
########## Graphics and Visualization ##############
####################################################
import seaborn as sns
def df_lag_corr_plots(analysis_df, exog, endog, ax):
    """ 
    This function plots the correlation between endog and exog for different past lags.
    When exog=endog, this performs autocorrelation (acf); otherwise, cross-correlation.
    
    Parameters:
    -----------
    analysis_df: pd.DataFrame
        DataFrame with columns for exog and endog at different lags.
    exog: str
        Name of the exogenous variable, or y
    endog: str
        Name of the endogenous variable, or x
    ax: matplotlib axis
    """
    cols = [c for c in analysis_df.columns if f'{endog}__b' in c]
    lags = sorted([int(c.replace(f'{endog}__b', '')) for c in cols])
    analysis_df = analysis_df[cols + [exog]].dropna()
    corrs = [np.corrcoef(analysis_df[exog], analysis_df[f'{endog}__b{lag}'])[0, 1] for lag in lags]
    ax.stem(lags, corrs)
    return ax

def df_partial_interaction_heatmap(analysis_df, exog, endogs, ax):
    """ 
    This function plots the partial interaction effect between two endogenous variables
    with respect to the `exog` variable. This assumes the endogs are catgorical variable.
    
    Parameters:
    -----------
    analysis_df: pd.DataFrame
        DataFrame with columns for exog and endog
    exog: str
        Name of the exogenous variable, or y
    endogs: list of str
        Names of the endogenous variables, or x
    """
    vs = analysis_df[endogs + [exog]].groupby(endogs).agg('mean').reset_index()
    vs = vs.sort_values(by=endogs)
    plot_data = vs.pivot(index=endogs[0], columns=endogs[1], values=exog)
    sns.heatmap(plot_data, annot=True, cmap='coolwarm', ax=ax)
    return ax