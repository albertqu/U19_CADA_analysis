import numpy as np
import pandas as pd
from scipy.special import expit
from statsmodels.tsa.tsatools import lagmat
from sklearn.linear_model import LogisticRegressionCV
import patsy

from cogmodels.base import CogModel


class LR(CogModel):
    def __init__(self, nlag=4):
        super().__init__()
        self.nlag = nlag
        self.clf = None

    def __str__(self):
        return "LR"

    def df2Xy(self, rdf):
        nlag = self.nlag
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
        formula = "Decision ~ " + "+".join(lagfeats + interactions)
        y, X = patsy.dmatrices(formula, data=lagdf, return_type="dataframe")
        id_df = lagdf[["ID", "Session", "Trial"]]
        return X, y, id_df

    def fitsim(self, data, *args, **kwargs):
        # assume data comes from one animal

        rdf = (
            data[["ID", "Session", "Trial", "Decision", "Reward"]]
            .rename(columns={"Reward": "R"})
            .reset_index(drop=True)
        )
        X, y, id_df = self.df2Xy(rdf)
        # clf = LogisticRegressionCV().fit(X, y)
        # self.clf = clf
        # id_df['choice_p'] = clf.predict_proba(X)[:, 1]
        # sklearn and GLM produces similar results
        binomial_model = sm.GLM(y, X, family=sm.families.Binomial())
        binomial_results = binomial_model.fit()
        params = pd.DataFrame({0: binomial_results.params}).transpose()
        params["ID"] = data["ID"].unique()[0]
        self.fitted_params = params
        self.summary = {"aic": binomial_results.aic, "bic": binomial_results.bic_llf}
        id_df["choice_p"] = binomial_model.predict(binomial_results.params, X)
        data_sim = data.merge(id_df, how="left", on=["ID", "Session", "Trial"])
        return data_sim
