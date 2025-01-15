import numpy as np
import pandas as pd
from scipy.special import expit
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from cogmodels.utils import Probswitch_2ABT, add_switch
import scipy
import patsy

from cogmodels.base import CogModel, CogParam


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


class RFLR(CogModel):
    """Recursive Formulation LR from Beron et al. 2022
    https://doi.org/10.1073/pnas.2113961119
    """

    def __init__(self):
        super().__init__()
        self.fixed_params = {"phi0": 0}
        self.param_dict = {
            "beta": CogParam(scipy.stats.expon(1), lb=0),
            "alpha": CogParam(scipy.stats.norm(0, 1)),
            "tau": CogParam(scipy.stats.expon(1), lb=1e-4),
        }
        self.fitted_params = None
        self.latent_names = ["phi"]  # ['qdiff', 'rpe']

    def __str__(self):
        return "RFLR"

    def latents_init(self, N):
        phis = np.zeros(N, dtype=float)
        return phis

    def id_param_init(self, params, id):
        varp_list = ["beta", "alpha", "tau"]
        beta, alpha, tau = params.loc[params["ID"] == id, varp_list].values.ravel()
        return {"alpha": alpha, "beta": beta, "tau": tau, "phi0": 0}

    def get_proba(self, data, params=None):
        if "loglik" in data.columns:
            return np.exp(data["loglik"].values)
        else:
            params_in = (
                self.fitted_params[["ID", "alpha"]] if params is None else params
            )
            new_data = data.merge(params_in, how="left", on="ID")
            return expit(new_data["phi"] + new_data["m"] * new_data["alpha"])

    def sim(self, data, params, *args, **kwargs):
        """
        Simulates the model for single subject that matches the data

        Input:
            data: pd.DataFrame
            ... params listed in class description
            params: pd.DataFrame
            ... containing parameters of interest

        Returns:
            data with new columns filled with latents listed in class description
        """
        N = data.shape[0]
        # qdiff, rpe, b_arr, w_arr = self.latents_init(N)

        c = data["Decision"]
        # TODO: handle miss decisions
        sess = data["Session"]
        subj = data["Subject"]
        id_i = data["ID"]
        r = data["Reward"]

        params_d = self.id_param_init(params, id_i.iat[0])
        phis = self.latents_init(N)
        # contains beta, tau, alpha
        phi = params_d["phi0"]
        c_bar = 2 * c - 1
        m1back = 0
        margs = np.zeros(N)

        # when ID changes: certain parameter changes
        # np.seterr(all='raise')
        for n in range(N):
            # initializing latents
            if (n > 0) and (id_i.iat[n] != id_i.iat[n - 1]):
                params_d = self.id_param_init(params, id_i.iat[n])

            if (n == 0) or (subj.iat[n] != subj.iat[n - 1]):
                phi = params_d["phi0"]
            elif sess.iat[n] != sess.iat[n - 1]:
                phi = params_d["phi0"]

            margs[n] = m1back
            phis[n] = phi

            ## Model update
            decay = np.exp(-1 / params_d["tau"])
            if c.iat[n] == -1:
                # handling miss trials
                phi = decay * phi
                m1back = 0
            else:
                update = c_bar.iat[n] * r.iat[n] * params_d["beta"]
                phi = decay * phi + update
                m1back = c_bar.iat[n]

        # qdiff[n], rpe[n], b, w, b_arr[n], w_arr[n]
        data["phi"] = phis
        data["m"] = margs
        return data

    def score(self, data):
        # Cross validate held out data
        pass

    def select_action(self, phi, m_1back, params):
        choice_p = expit(phi + m_1back * params["alpha"])
        return int(np.random.random() <= choice_p)

    def generate(self, params, *args, **kwargs):
        # n_trial, n_session, serialN=1,
        """
        Simulates the model for single subject/ID that matches the data

        for each variable parameter: randomly generate parameter i that range, n_ids
        params <- full sets of params for all ids

        Input:
            data: pd.DataFrame
            ... params listed in class description
            params: pd.DataFrame
            ID, vars, n_trial, n_session
            ... containing parameters of interest

        Returns:
            data with new columns filled with latents listed in class description
        """
        uniq_id = params["ID"].values[0]
        n_trial = params["n_trial"].values[0]
        n_session = params["n_session"].values[0]
        N = n_trial * n_session
        phis = self.latents_init(N)
        c = np.zeros(N, dtype=int)
        r = np.zeros(N, dtype=int)

        data = {k: np.zeros(N, dtype=int) for k in ["Target", "Decision", "Reward"]}
        data["Session"] = np.repeat(
            [f"{i:02d}" for i in range(1, n_session + 1)], n_trial
        )
        data["Trial"] = np.tile([i + 1 for i in range(n_trial)], n_session)
        data = pd.DataFrame(data)
        data["Subject"] = uniq_id
        data["ID"] = uniq_id
        sess = data["Session"]

        task_params = {"blockLen_range": (8, 15), "condition": "75-0"}
        task_params.update(kwargs)
        probs = task_params["condition"].split("-")
        data["Condition"] = task_params["condition"]
        p_cor, p_inc = [float(p) / 100 for p in probs]
        task_params.update({"p_cor": p_cor, "p_inc": p_inc})
        task = Probswitch_2ABT(**task_params)

        # fix_cols = ['ID', 'Subject', 'Session', 'Trial', 'blockTrial', 'blockLength', 'Target', 'Condition']
        # ID, Subject, Session, Trial, blockTrial, blockLength, Target, Decision, Switch, Reward, Condition
        params_d = self.id_param_init(params, uniq_id)
        phi = params_d["phi0"]
        m_1back = 0

        # when ID changes: certain parameter changes
        # np.seterr(all='raise')
        for n in range(N):
            if (n > 0) and (sess.iat[n] != sess.iat[n - 1]):
                phi = params_d["phi0"]
                m_1back = 0
                task.initialize()

            c_t = self.select_action(phi, m_1back, params_d)
            c_bar_t = 2 * c_t - 1
            m_1back = c_bar_t

            data.loc[n, "Target"] = task.target
            data.loc[n, "blockTrial"] = task.btrial
            data.loc[n, "blockNum"] = task.blockN
            r_t = task.getOutcome(c_t)
            phis[n] = phi

            ## Model update
            decay = np.exp(-1 / params_d["tau"])

            update = c_bar_t * r_t * params_d["beta"]
            phi = decay * phi + update
            c[n], r[n] = c_t, r_t

        data["Decision"] = c
        data["Reward"] = r
        v = data.groupby(["ID", "Session", "blockNum"], as_index=False).apply(len)
        v.columns = list(v.columns[:3]) + ["blockLength"]
        data = data.merge(v, how="left", on=["ID", "Session", "blockNum"])
        data = add_switch(data)
        return data
