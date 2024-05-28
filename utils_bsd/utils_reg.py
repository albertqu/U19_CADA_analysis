import copy
import patsy
from sklearn.model_selection._search import ParameterGrid
from sklearn.metrics._scorer import check_scoring
from preprocessing import CV_df_Preprocessor, PSLR_Preprocessor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LassoCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from statsmodels.tsa.tsatools import lagmat
from utils_bsd.configs import RAND_STATE
import statsmodels.api as sm
from os.path import join as oj


#####################################################
################ Behavior Modeling ##################
#####################################################


def df2Xy(rdf, nlag=6):
    rdf["Decision"] = rdf["action"].map({"left": 0, "right": 1}).astype(float)
    rdf["C"] = 2 * rdf["Decision"] - 1
    features = ["C", "R"]
    lagfeats = list(
        np.concatenate(
            [[feat + f"_{i}back" for feat in features] for i in range(1, nlag + 1)]
        )
    )

    lagdf = pd.DataFrame(
        lagmat(rdf[features].values, maxlag=nlag, trim="forward", original="ex"),
        columns=lagfeats,
    )
    col_keys = ["C"] + [f"C_{i}back" for i in range(1, nlag + 1)]
    lagdf = pd.concat([rdf, lagdf], axis=1)
    lagdf = lagdf[
        (lagdf["trial"] > nlag)
        & np.logical_and.reduce([(~lagdf[c].isnull()) for c in col_keys])
    ].reset_index(drop=True)
    interactions = [f"C_{i}back:R_{i}back" for i in range(1, nlag + 1)]
    formula = "Decision ~ " + "+".join(lagfeats + interactions)
    y, X = patsy.dmatrices(formula, data=lagdf, return_type="dataframe")
    id_df = lagdf[["animal", "session", "trial"]]
    return X, y, id_df


def fit_action_value_function(df, nlag=4):
    # endogs

    rdf = (
        df[["animal", "session", "trial", "rewarded", "action"]]
        .rename(columns={"rewarded": "R"})
        .reset_index(drop=True)
    )
    X, y, _ = df2Xy(rdf, nlag=nlag)

    # Use held out dataset to evaluate score
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RAND_STATE
    )
    # clf = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train, y_train)
    clf = LogisticRegression().fit(X_train, y_train)
    cv_score = clf.score(X_test, y_test)
    # use full dataset to calculate action logits
    clf_psy = clf.fit(X, y)

    def func(X):
        logits = X @ clf_psy.coef_.T + clf_psy.intercept_
        return logits

    return {"score": cv_score, "func": func, "name": "action_logit"}


def add_action_value_feature(df, endog_map, nlag=6):
    rdf = (
        df[["animal", "session", "trial", "rewarded", "action"]]
        .rename(columns={"rewarded": "R"})
        .reset_index(drop=True)
    )
    X, _, id_df = df2Xy(rdf, nlag=nlag)
    id_df[endog_map["name"]] = endog_map["func"](X)
    return df.merge(id_df, how="left", on=["animal", "session", "trial"])


def decode_from_regfeature(feature):
    get_lag = lambda s: int(s.split("_")[1][:-4])
    if ":" in feature:
        ftype = "C:R"
        lag = get_lag(feature.split(":")[0])
    elif feature == "Intercept":
        ftype = feature
        lag = 0
    else:
        ftype = feature.split("_")[0]
        lag = get_lag(feature)
    return ftype, lag


def fit_LR_model_add_logit(data, nlag=4):

    # prep = PSLR_Preprocessor(4)
    # mdl = LogisticRegressionCV()
    # X, y = prep.fit_transform(data)
    # mdl.fit(X, y.values.ravel())
    # params_sk = pd.DataFrame(mdl.coef_)

    # binomial_model = sm.GLM(y, X, family=sm.families.Binomial())
    # binomial_results = binomial_model.fit()
    # params = pd.DataFrame({0: binomial_results.params}).transpose()
    # params
    prep = PSLR_Preprocessor(nlag)
    mdl = LogisticRegressionCV()
    X, y = prep.fit_transform(data)
    mdl.fit(X.values, y.values.ravel())
    # mdl.coef_.shape
    coefs = pd.DataFrame(mdl.coef_, columns=prep.x_cols)
    # multiply coefs with features
    data.loc[prep.idx, "action_logit"] = (
        X.values @ coefs.T.values
    ).ravel() + mdl.intercept_
    coefs["Intercept"] = mdl.intercept_
    coef_df = coefs.T.reset_index()
    coef_df.columns = ["feature", "weight"]
    coef_df["feature_d"] = coef_df["feature"].apply(decode_from_regfeature)
    first_elem = lambda v: v[0]
    second_elem = lambda v: v[1]
    coef_df["feature_type"] = coef_df["feature_d"].apply(first_elem)
    coef_df["lag"] = coef_df["feature_d"].apply(second_elem)
    coef_df.drop(columns=["feature_d"], inplace=True)
    return coef_df


def fit_LR_model(data, nlag=4, i2=False, alpha=0.01):
    # assume data comes from one animal
    # formula = "accept ~ restaurant:stimType + tone_prob:stimType + stimType + restaurant + tone_prob"
    # TODO: cross validation! (train_inds, test_inds)
    # TODO: transform function to turn data into logit
    rdf = (
        data[["ID", "Session", "Trial", "Decision", "Reward"]]
        .rename(columns={"Reward": "R"})
        .reset_index(drop=True)
    )
    # feature engineering
    # input: rdf with C, R columns
    rdf["C"] = 2 * rdf["Decision"] - 1
    features = ["C", "R"]
    lagfeats = list(
        np.concatenate(
            [[feat + f"_{i}back" for feat in features] for i in range(1, nlag + 1)]
        )
    )

    lagdf = pd.DataFrame(
        lagmat(rdf[features].values, maxlag=nlag, trim="forward", original="ex"),
        columns=lagfeats,
    )
    col_keys = ["C"] + [f"C_{i}back" for i in range(1, nlag + 1)]
    lagdf = pd.concat([rdf, lagdf], axis=1)
    lagdf = lagdf[
        (lagdf["Trial"] > nlag)
        & np.logical_and.reduce([(lagdf[c] != -3) for c in col_keys])
    ].reset_index(drop=True)
    interactions = [f"C_{i}back:R_{i}back" for i in range(1, nlag + 1)]
    lagdf["R_1back_neg"] = 1 - lagdf["R_1back"]
    if i2:
        interactions = ["C_1back:R_1back"] + [
            f"C_{i}back:R_{i}back:R_1back" for i in range(2, nlag + 1)
        ]
        interactions2 = [f"C_{i}back:R_{i}back:R_1back_neg" for i in range(2, nlag + 1)]
    else:
        interactions2 = []
    formula = "Decision ~ " + "+".join(lagfeats + interactions + interactions2)
    binomial_model = sm.GLM.from_formula(
        formula, data=lagdf, family=sm.families.Binomial()
    )
    binomial_results = binomial_model.fit()
    # print(binomial_results.summary())
    coef_df = binomial_results.params.reset_index()
    coef_df.columns = ["feature", "weight"]
    coef_df["pval"] = binomial_results.pvalues.values
    coef_df.loc[coef_df["pval"] > alpha, "weight"] = 0
    coef_df["feature_d"] = coef_df["feature"].apply(decode_from_regfeature)
    first_elem = lambda v: v[0]
    second_elem = lambda v: v[1]
    coef_df["feature_type"] = coef_df["feature_d"].apply(first_elem)
    if i2:
        sel = (~coef_df["feature"].str.contains("_neg")) & (
            coef_df["feature"].str.count(r":") == 2
        )
        coef_df.loc[sel, "feature_type"] = "C:R_b"
    coef_df["lag"] = coef_df["feature_d"].apply(second_elem)
    coef_df.drop(columns=["feature_d"], inplace=True)
    coef_df["feature_name"] = coef_df["feature_type"].map(
        {
            "C": "Choice",
            "R": "Reward",
            "C:R_b": "R1-blocked",
            "C:R": "Rew. choice",
            "Intecept": "Bias",
        }
    )
    return coef_df


def get_modelsim_data(model_str, cache_folder, gen_arg):
    reg_df = pd.read_csv(
        oj(cache_folder, f"genrec_{gen_arg}_{model_str}_L-BFGS-B_gendata.csv")
    )
    reg_df.drop(columns=[c for c in reg_df.columns if "Unnamed" in c], inplace=True)
    return reg_df


def df_preprocess_cv_fit(data, model, preprocessor, cv, param_grid, metrics=None):
    """Function splits data into train test folds, preprocess data, and then tune the hyperparameter
    data: pd.DataFrame with data to fit
    y: column variable in data used for model fitting
    model: sklearn style model
    preprocessor: CV_df_Preprocessor instance
    cv: sklearn style cv splitter class, e.g.
        from sklearn.model_selection import StratifiedKFold, KFold
        skf = StratifiedKFold(n_splits=3); skf.split(X, y)
    param_grid: dictionary of {'model__*': ..., 'prep__*': ...} for gridsearch hyperparam tune
    """
    all_res = []
    pgrid = ParameterGrid(param_grid)

    for i, (train_inds, test_inds) in enumerate(cv.split(data)):
        for pdict in pgrid:
            # use itertools to parse preprocessor arg
            # use iter
            model_p = {}
            prep_p = {}
            for k, v in pdict.items():
                step, arg = k.split("__")
                if step == "model":
                    model_p[arg] = v
                else:
                    prep_p[arg] = v

            prep = preprocessor(**prep_p)
            X, y = prep.fit_transform(data)
            d_train_inds, d_test_inds = prep.index_train_test(train_inds, test_inds)
            mdl = model(**model_p)
            # raveling y to satisfy sklearn model validation requirements
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.loc[d_train_inds].values, X.loc[d_test_inds].values
                y_ravel = y.values.ravel()
                y_train, y_test = y_ravel[d_train_inds], y_ravel[d_test_inds]
            else:
                X_train, X_test = X[d_train_inds], X[d_test_inds]
                if len(y.shape) != 1:
                    y = y.ravel()
                y_train, y_test = y[d_train_inds], y[d_test_inds]
            mdl.fit(X_train, y_train)
            # metrics
            m_res = {}
            if metrics is None:
                m_res["accuracy"] = mdl.score(X_test, y_test)
            else:
                for m in metrics:
                    scorer = check_scoring(mdl, m)
                    m_res[m] = scorer(mdl, X_test, y_test)
            res_d = copy.deepcopy(model_p)
            res_d.update(prep_p)
            res_d.update(m_res)
            res_d["iter"] = i
            all_res.append(res_d)
            # pd.DataFrame(mdl.coef_, columns=prep.x_cols)
    result_df = pd.DataFrame(all_res)
    return result_df


def reference_lag_transform(df, ref_t, lag, inplace=False):
    """
    Encode past trial actions with respect to actions at trial r (ref_t) ($c_{r}$).
    $a_{t-i} = 1 \mathcal{1} \{c_{t-i} = c_r\}$
    reference r (t-r) C_{t-l}, lag l: ref[r]_L[l]
    ref[r]_L[l] = 1 if C_{t-l} == C_{t-r}
    """
    inplace = False
    if not inplace:
        df = df.copy()
    lcols = [f"L{i}" for i in range(1, lag + 1)]
    rcols = [f"R_{i}back" for i in range(1, lag + 1)]
    df[lcols] = 0
    df[rcols] = 0
    df[lcols] = lagmat(df["Decision"], maxlag=lag, trim="forward", original="ex")
    df[rcols] = lagmat(df["Reward"], maxlag=lag, trim="forward", original="ex")
    ref_col = "Decision" if ref_t == 0 else f"L{ref_t}"
    for i in range(1, lag + 1):
        arg = f"L{ref_t}_{i}back"  # f'ref{ref_t}_L{i}'
        df[arg] = (df[f"L{i}"] == df[ref_col]).astype(float)
        df.loc[df["Trial"] <= i, arg] = np.nan
    return df


def dropna_Xy(X, y, index=False):
    sel = ~(np.any(np.isnan(X), axis=1) & np.isnan(y))
    if index:
        return np.where(sel)[0]
    else:
        return np.array(X.values[sel]), y[sel]


def cv_with_covars(reg_df, y, covars, interactions, train_index, test_index):
    formula = f"{y} ~ " + "+".join(covars + interactions)
    # Issue/insight: interactive rew effect cannot compete with the biasing effect of the same side switching effect.
    y_train, X_train = patsy.dmatrices(
        formula, data=reg_df.loc[train_index], return_type="dataframe", NA_action="drop"
    )
    y_test, X_test = patsy.dmatrices(
        formula, data=reg_df.loc[test_index], return_type="dataframe", NA_action="drop"
    )
    reg = LassoCV(cv=5, random_state=0).fit(
        X_train.drop(columns=["Intercept"]), y_train.values.ravel()
    )
    cv_score = reg.score(X_test.drop(columns=["Intercept"]), y_test.values.ravel())
    return cv_score


def cv_kfold_covars(reg_df, y, covars, interactions, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    scores = [0] * 5
    for i, (train_index, test_index) in enumerate(kf.split(reg_df)):
        scores[i] = cv_with_covars(
            reg_df, y, covars, interactions, train_index, test_index
        )
    return scores


def beh_regmodel_preprocess(data, lag=4):
    rdf = (
        data[["ID", "Session", "Trial", "Decision", "Reward"]]
        .rename(columns={"Reward": "R"})
        .reset_index(drop=True)
    )
    # feature engineering
    # input: rdf with C, R columns
    rdf["C"] = 2 * rdf["Decision"] - 1
    features = ["C", "R"]
    lagfeats = list(
        np.concatenate(
            [[feat + f"_{i}back" for feat in features] for i in range(1, lag + 1)]
        )
    )

    lagdf = pd.DataFrame(
        lagmat(rdf[features].values, maxlag=lag, trim="forward", original="ex"),
        columns=lagfeats,
    )
    col_keys = ["C"] + [f"C_{i}back" for i in range(1, lag + 1)]
    lagdf = pd.concat([rdf, lagdf], axis=1)
    lagdf = lagdf[
        (lagdf["Trial"] > lag)
        & np.logical_and.reduce([(lagdf[c] != -3) for c in col_keys])
    ].reset_index(drop=True)
    lagdf["R_1back_neg"] = 1 - lagdf["R_1back"]
    return lagdf
