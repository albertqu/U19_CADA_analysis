import statsmodels.api as sm
from scipy import stats
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
from pingouin import pairwise_tests
from statannotations.Annotator import Annotator
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoCV
from utils_bsd.configs import RAND_STATE
import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro


def calculate_trial2sw(df):
    # trial2sw first trial to switch to new port
    # trial2asymp first switch to stick
    # TODO: make it better by considering consecutive stays as well instead of just consecutive corrects
    switch_sel = (df["correct"] == 1) & (df["Switch"] == 1)
    # first switch
    swtrials = df.loc[switch_sel, "blockTrial"].values
    if len(swtrials) == 0:
        # print(df['blockNum'].unique())
        trial2sw = np.nan
    else:
        trial2sw = swtrials[0]
    stay2sel = (df["correct"].shift(-1) == 1) & (df["correct"].shift(-2) == 1)
    asymptrials = df.loc[stay2sel & switch_sel, "blockTrial"].values
    if len(asymptrials) == 0:
        trial2asymp = np.nan
    else:
        trial2asymp = asymptrials[0]
    # 'Subject': [df['Subject'].iat[0]],'session_num': [df['session_num'].iat[0]],'blockNum': [df['blockNum'].iat[0]],
    return pd.DataFrame({"trial2sw": [trial2sw], "trial2asymp": [trial2asymp]})


def bs_mean_CI(data, conf=0.95):
    # give boostrap interval for mean estimate
    if np.any(np.isnan(data)):
        data = data[~np.isnan(data)]
    res = stats.bootstrap(
        (data,), np.mean, confidence_level=conf, random_state=RAND_STATE
    )
    lb, ub = res.confidence_interval
    return (lb, ub)


def nonparam_anova(df, x, y):
    xvals = df[x].unique()
    s_ij, p_ij = stats.kruskal(*[df.loc[df[x] == xval, y].values for xval in xvals])
    return s_ij, p_ij


def nonparam_pair_test(df, dv, between, a, b=None, alternative="two-sided"):
    """Non-parametric paired test using df
    between: variable group to compare against between variance axis
    """
    if b is None:
        return stats.wilcoxon(
            df.loc[df[between] == a, dv].values, alternative=alternative
        )
    else:
        x = df.loc[df[between] == a, dv].values
        y = df.loc[df[between] == b, dv].values
        return stats.wilcoxon(x, y, alternative=alternative)


def nonparam_ind_test(df, dv, between, a, b=None, alternative="two-sided"):
    """Non-parametric paired test using df
    between: variable group to compare against between variance axis
    """
    if b is None:
        return stats.mannwhitneyu(
            df.loc[df[between] == a, dv].values, alternative=alternative
        )
    else:
        x = df.loc[df[between] == a, dv].values
        y = df.loc[df[between] == b, dv].values
        return stats.mannwhitneyu(x, y, alternative=alternative)


"""
Procedure testing object to perform nonparametric anova, pairwise posthoc, 
and then for each pairwise difference compute effect size
"""


def calculate_cohenD_CI(df, dv, between, a, b, paired=True):
    """Calculate the effect size a-b using Cohen's d"""
    df = df[[between, dv]].dropna()
    x = df.loc[df[between] == a, dv].values
    y = df.loc[df[between] == b, dv].values
    es = pg.compute_effsize(x, y, paired=paired, eftype="cohen")
    ci = pg.compute_esci(stat=es, nx=len(x), ny=len(y), paired=paired, eftype="cohen")
    return es, ci


def proportion_test(df, dv, between, a, b, alternative="two-sided"):
    # assume not paired, a is the comparison for alternative
    df = df[[between, dv]].dropna()
    ctab = pd.crosstab(df[dv], df[between])
    ctab = ctab.loc[[1, 0], [a, b]].values
    return stats.fisher_exact(ctab, alternative=alternative)


def extract_OLS_coef(reg_results, X):
    coef_df = pd.concat(
        [reg_results.params, reg_results.pvalues, reg_results.conf_int()], axis=1
    ).reset_index()
    coef_df.columns = ["feature", "weight", "pval", "C_lb", "C_ub"]

    # calculate VIF:
    if isinstance(X, pd.DataFrame):
        X = X.values
    vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    coef_df["VIF"] = vifs
    return coef_df


def ols_testing(df, y, x_vars, norm_x=False):
    """Performs OLS regression on df from x_vars to y
    norm_x: boolean to normalize x
    """
    # add mixed regression TODO
    X = df[x_vars]
    X = sm.add_constant(X)
    if norm_x:
        for col in X.columns:
            if col != "const":
                X[col] = scipy.stats.zscore(X[col])
            if np.any(X[col].isnull()):
                print(col)
    model = sm.OLS(df[y], X)
    reg_results = model.fit()
    coef_df = pd.concat(
        [reg_results.params, reg_results.pvalues, reg_results.conf_int()], axis=1
    ).reset_index()
    coef_df.columns = ["feature", "weight", "pval", "C_lb", "C_ub"]

    # calculate VIF:
    vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    coef_df["VIF"] = vifs
    collinears = coef_df.loc[coef_df["VIF"] > 10, "feature"].values
    if len(collinears) > 0:
        print("VIF inflated for", collinears)
    return coef_df, reg_results


def pairplot_stats(
    df, x_vars, y_vars, cat_cols, scatter_kws={}, box_kws={}, show_p=True, **kwargs
):
    # assume all y_vars are numeric
    df = df.dropna(subset=x_vars + y_vars).reset_index(drop=True)
    fig, axes = plt.subplots(
        nrows=len(y_vars), ncols=len(x_vars), figsize=(4 * len(x_vars), 4 * len(y_vars))
    )
    coef_dfs = []
    for j in range(axes.shape[0]):
        y = y_vars[j]
        coef_df, reg_results = ols_testing(df, y, x_vars, norm_x=True)
        coef_dfs.append(coef_df)
        for i in range(axes.shape[1]):
            x, y = x_vars[i], y_vars[j]
            p_reg = coef_df.loc[coef_df["feature"] == x, "pval"].values[0]
            if x in cat_cols:
                xvals = df[x].unique()
                s_ij, p_ij = stats.kruskal(
                    *[df.loc[df[x] == xval, y].values for xval in xvals]
                )
                sns.boxplot(x=df[x], y=df[y], ax=axes[j][i], **box_kws, **kwargs)
                if show_p:
                    axes[j][i].text(
                        0.05,
                        1.1,
                        f"KW p: {p_ij:.4f}\nReg p: {p_reg:.4f}",
                        transform=axes[j][i].transAxes,
                    )
            else:
                r_ij, p_ij = stats.mstats.pearsonr(x=df[x].values, y=df[y].values)
                sns.regplot(
                    x=df[x],
                    y=df[y],
                    ax=axes[j][i],
                    scatter=True,
                    scatter_kws=scatter_kws,
                    **kwargs,
                )
                if show_p:
                    axes[j][i].text(
                        0.05,
                        1.1,
                        f"R: {r_ij:.4f} (p={p_ij:.4f})\nReg p: {p_reg:.4f}",
                        transform=axes[j][i].transAxes,
                    )
    plt.tight_layout()
    sns.despine()
    all_coef_df = pd.concat(coef_dfs, axis=0).reset_index(drop=True)
    # fig.subplots_adjust(wspace=0.3, hspace=0.3)
    return fig, all_coef_df


def boxplot_stats(df, x, y, ax=None, **kwargs):
    # perform nonparametric test and then posthoc and mark in boxplot accordingly
    # example: ax, (p_kruskal, posthoc_result) = boxplot_stats(df, 'x', 'y', ax=plt.gca())
    if ax is None:
        ax = plt.gca()
    order = df[x].unique()
    s, p = stats.kruskal(*[df.loc[df[x] == xval, y].values for xval in order])
    posthoc_result = pairwise_tests(
        data=df, parametric=False, dv=y, between=x, padjust="holm"
    )
    if len(posthoc_result) == 1:
        posthoc_result["p-corr"] = posthoc_result["p-unc"]
    sns.boxplot(data=df, x=x, y=y, ax=ax, order=order, **kwargs)
    pairs = [
        posthoc_result.loc[i, ["A", "B"]].values
        for i in range(len(posthoc_result))
        if posthoc_result.loc[i, "p-corr"] < 0.05
    ]
    if pairs:
        annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
        annotator.configure(
            test="Mann-Whitney",
            comparisons_correction="holm",
            text_format="star",
            loc="outside",
        )
        annotator.apply_and_annotate()
    sns.despine()
    return ax, (p, posthoc_result)


#############################################
################ R stats ####################
#############################################


def rlmer(lmer_df, formula):
    def pandas2R(df):
        """Local conversion of pandas dataframe to R dataframe as recommended by rpy2"""
        with localconverter(ro.default_converter + pandas2ri.converter):
            data = ro.conversion.py2rpy(df)
        return data

    def R2pandas(rdf):
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(rdf)
        return df

    rbase = importr("base")
    utils = importr("utils")
    lme4 = importr("lme4")
    lmerTest = importr("lmerTest")
    broom = importr("broom.mixed")
    pd.DataFrame.iteritems = pd.DataFrame.items
    rdf = pandas2R(lmer_df)
    fm1 = lmerTest.lmer(formula, data=rdf)
    rstring = """
        function(model){
        summary(model)
        tidy(model)
        }
    """
    # rstring = (
    #     """
    #     function(model){
    #     out.coef <- data.frame(unclass(summary(model))$coefficients)
    #     out <- out.coef
    #     list(out,rownames(out))
    #     }
    # """
    # )
    estimates_func = ro.r(rstring)
    out_summary = estimates_func(fm1)
    df = R2pandas(out_summary)
    return df
