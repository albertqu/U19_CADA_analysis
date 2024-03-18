import numpy as np
import pandas as pd
from scipy.special import expit
import scipy
from numbers import Number
from abc import abstractmethod
from cogmodels._legacy.cogmodels_utils import Probswitch_2ABT, add_switch
from statsmodels.tsa.tsatools import lagmat
from sklearn.linear_model import LogisticRegressionCV
import patsy

import statsmodels.api as sm

__author__ = "Albert Qü"
__version__ = "0.0.1"
__email__ = "albert_qu@berkeley.edu"
__status__ = "Dev"


class CogModel:
    """
    Base model class for cognitive models for decision making, with the objective of
    generalizing to various task structure, given a preset data format.

    data: pd.DataFrame
        .columns: Subject, Decision, Reward, Target, *State* (applies to tasks with
        distinct sensory states)
    """

    data_cols = [
        "ID",
        "Subject",
        "Session",
        "Trial",
        "blockTrial",
        "blockNum",
        "blockLength",
        "Target",
        "Decision",
        "Switch",
        "Reward",
        "Condition",
    ]

    # defines experiment columns that have little to no dependency to subject behavior
    expr_cols = [
        "ID",
        "Subject",
        "Session",
        "Trial",
        "blockTrial",
        "blockNum",
        "blockLength",
        "Target",
        "Reward",
        "Condition",
    ]

    def __init__(self):
        self.fixed_params = {}
        self.param_dict = {}
        self.fitted_params = None
        self.latent_names = []  # ['qdiff', 'rpe']
        self.summary = {}
        pass

    def sim(self, data, params, *args, **kwargs):
        pass

    def fit(self, data, *args, **kwargs):
        pass

    def score(self, data):
        # Cross validate held out data
        pass

    def emulate(self, *args, **kwargs):
        pass


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


class CogModel2ABT_BWQ(CogModel):
    """
    Base class for 2ABT cognitive models
    input data must have the following columns:
    ID, Subject, Session, Trial, blockTrial, blockLength, Target, Decision, Switch, Reward, Condition
    """

    def __init__(self):
        super().__init__()
        self.k_action = 2
        self.fixed_params = {
            "predict_thres": 0.5,
            "CF": False,
        }  # whether or not to calculate counterfactual rpe
        self.marg_name = "stay"
        # used fixed for hyperparam_tuning

    def latents_init(self, N):
        qdiff = np.zeros(N)
        if self.fixed_params["CF"]:
            rpe = np.zeros((N, 2))
        else:
            rpe = np.zeros((N, 1))
        b_arr = np.zeros(N)
        w_arr = np.zeros((N, 2))
        return qdiff, rpe, b_arr, w_arr

    @abstractmethod
    def id_param_init(self, params, id):
        """abstract method for iniatiating ID parameter
        Returns: Dict
        """
        pass

    @abstractmethod
    def calc_q(self, b, w):
        """abstract method for calcualting action value
        Returns: np.array of q values
        """
        pass

    @abstractmethod
    def update_b(self, b, w, c_t, r_t, params_i):
        """abstract method for updating model latent b
        Returns: updated b, scalar or np.array
        """
        pass

    @abstractmethod
    def update_w(self, b, w, c_t, rpe_t, params_i):
        """abstract method for updating model latent w
        Returns: updated w, scalar or np.array
        """
        pass

    @abstractmethod
    def assign_latents(self, data, qdiff, rpe, b_arr, w_arr):
        """abstract method for saving simulated latents,
        Returns: data appended with columns storing different latents
        """
        pass

    def get_proba(self, data, params=None):
        if "loglik" in data.columns:
            return np.exp(data["loglik"].values)
        else:
            params_in = (
                self.fitted_params[["ID", "beta", "st"]] if params is None else params
            )
            new_data = data.merge(params_in, how="left", on="ID")
            return expit(
                new_data["qdiff"] * new_data["beta"] + new_data["stay"] * new_data["st"]
            )

    def select_action(self, qdiff, m_1back, params):
        stay = 1
        if m_1back == 0:
            stay = -1
        elif m_1back == np.nan:
            stay = 0
        choice_p = expit(qdiff * params["beta"] + stay * params["st"])
        return int(np.random.random() <= choice_p)

    def marginal_init(self):
        return np.nan

    def update_marginal(self, c_t, m_1back):
        return c_t

    def predict(self, data):
        return (self.get_proba(data) >= self.fixed_params["predict_thres"]).astype(
            float
        )

    def fit_marginal(self, data):
        """Fits fake marginal with only past trial information, useful for BI and RL model"""
        # test marginal stay
        c = data["Decision"]
        c_lag = c.shift(1)
        data["stay"] = -1
        data.loc[c_lag == 1, "stay"] = 1
        data.loc[data["Trial"] == 1, "stay"] = 0
        data.loc[(c == -1) | (c_lag == -1), "stay"] = 0
        return data

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
        qdiff, rpe, b_arr, w_arr = self.latents_init(N)

        c = data["Decision"]
        # TODO: handle miss decisions
        sess = data["Session"]
        subj = data["Subject"]
        id_i = data["ID"]
        r = data["Reward"]

        params_d = self.id_param_init(params, id_i.iat[0])
        b0, w0, gam = params_d["b0"], params_d["w0"], params_d["gam"]
        b, w = b0, w0

        # when ID changes: certain parameter changes
        # np.seterr(all='raise')
        for n in range(N):
            # initializing latents
            if id_i.iat[n] != id_i.iat[n - 1]:
                params_d = self.id_param_init(params, id_i.iat[n])

            if (n == 0) or (subj.iat[n] != subj.iat[n - 1]):
                b = b0
                w = np.copy(w0)
            elif sess.iat[n] != sess.iat[n - 1]:
                b = b0
                w = w0 * gam + (1 - gam) * w
            ## Action value calculation
            qs = self.calc_q(b, w).ravel()
            # compute value difference
            qdiff[n] = qs[1] - qs[0]
            # print(n, qs)
            ## Model update
            if c.iat[n] == -1:
                # handling miss trials
                rpe[n, :] = np.nan
                w_arr[n, :] = w
                b_arr[n] = b
                # w, b remains the same
                # TODO: plot by adjusting parameter and see if behavior change.
            else:
                rpe_c = r.iat[n] - qs[c.iat[n]]
                if self.fixed_params["CF"]:
                    rpe_cf = (1 - r.iat[n]) - qs[1 - c.iat[n]]
                    rpe_t = np.array([rpe_c, rpe_cf])
                else:
                    rpe_t = rpe_c
                rpe[n, :] = rpe_t
                # w, b reflects information prior to reward
                w_arr[n, :] = w
                b_arr[n] = b
                w = self.update_w(b, w, c.iat[n], rpe_t, params_d)
                # Updating b!
                b = self.update_b(b, w, c.iat[n], r.iat[n], params_d)

        data = self.assign_latents(data, qdiff, rpe, b_arr, w_arr)
        # qdiff[n], rpe[n], b, w, b_arr[n], w_arr[n]
        return data

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
        qdiff, rpe, b_arr, w_arr = self.latents_init(N)
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
        b0, w0, gam = params_d["b0"], params_d["w0"], params_d["gam"]

        b, w = b0, w0
        m_1back = self.marginal_init()

        # when ID changes: certain parameter changes
        # np.seterr(all='raise')
        for n in range(N):
            if (n != 0) and (sess.iat[n] != sess.iat[n - 1]):
                b = b0
                w = w0 * gam + (1 - gam) * w
                m_1back = self.marginal_init()
                task.initialize()
            ## Action value calculation
            qs = self.calc_q(b, w).ravel()
            # compute value difference
            qdiff[n] = qs[1] - qs[0]

            c_t = self.select_action(qdiff[n], m_1back, params_d)
            m_1back = self.update_marginal(c_t, m_1back)

            data.loc[n, "Target"] = task.target
            data.loc[n, "blockTrial"] = task.btrial
            data.loc[n, "blockNum"] = task.blockN
            r_t = task.getOutcome(c_t)

            ## Model update
            rpe_c = r_t - qs[c_t]

            if self.fixed_params["CF"]:
                rpe_cf = (1 - r_t) - qs[1 - c_t]
                rpe_t = np.array([rpe_c, rpe_cf])
            else:
                rpe_t = rpe_c
            c[n], r[n] = c_t, r_t
            rpe[n, :] = rpe_t
            # w, b reflects information prior to reward
            w_arr[n, :] = w
            b_arr[n] = b
            w = self.update_w(b, w, c[n], rpe_t, params_d)
            # Updating b!
            b = self.update_b(b, w, c[n], r[n], params_d)

        data["Decision"] = c
        data["Reward"] = r
        v = data.groupby(["ID", "Session", "blockNum"], as_index=False).apply(len)
        v.columns = list(v.columns[:3]) + ["blockLength"]
        data = data.merge(v, how="left", on=["ID", "Session", "blockNum"])
        data = add_switch(data)
        return data

    def emulate(self, data, params, *args, **kwargs):
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
        qdiff, rpe, b_arr, w_arr = self.latents_init(N)
        c = np.zeros(N, dtype=int)
        r = np.zeros(N, dtype=int)

        # TODO: handle miss decisions
        sess = data["Session"]
        subj = data["Subject"]
        id_i = data["ID"]
        targets = data["Target"]
        c_data = data["Decision"]
        r_data = data["Reward"]

        probs = data["Condition"].unique()[0].split("-")
        p_cor, p_inc = [float(p) / 100 for p in probs]
        fix_cols = [
            "ID",
            "Subject",
            "Session",
            "Trial",
            "blockTrial",
            "blockLength",
            "Target",
            "Condition",
        ]
        emu_data = data[fix_cols].reset_index(drop=True)
        params_d = self.id_param_init(params, id_i.iat[0])
        b0, w0, gam = params_d["b0"], params_d["w0"], params_d["gam"]

        b, w = b0, w0
        m_1back = self.marginal_init()

        # when ID changes: certain parameter changes
        # np.seterr(all='raise')
        for n in range(N):
            # initializing latents
            if (n == 0) or (id_i.iat[n] != id_i.iat[n - 1]):
                params_d = self.id_param_init(params, id_i.iat[n])
                # generalize recency

            if (n == 0) or (subj.iat[n] != subj.iat[n - 1]):
                b = b0
                w = np.copy(w0)
                m_1back = self.marginal_init()
            elif sess.iat[n] != sess.iat[n - 1]:
                b = b0
                w = w0 * gam + (1 - gam) * w
                m_1back = self.marginal_init()
            ## Action value calculation
            qs = self.calc_q(b, w).ravel()
            # compute value difference
            qdiff[n] = qs[1] - qs[0]

            c_t = self.select_action(qdiff[n], m_1back, params_d)
            m_1back = self.update_marginal(c_t, m_1back)

            if c_t == c_data.iat[n]:
                r_t = r_data.iat[n]
            else:
                dice = np.random.random()
                if targets.iat[n] == c_t:
                    r_t = int(dice <= p_cor)
                else:
                    r_t = int(dice <= p_inc)  # check this block
            ## Model update
            rpe_c = r_t - qs[c_t]

            if self.fixed_params["CF"]:
                rpe_cf = (1 - r_t) - qs[1 - c_t]
                rpe_t = np.array([rpe_c, rpe_cf])
            else:
                rpe_t = rpe_c
            c[n], r[n] = c_t, r_t
            rpe[n, :] = rpe_t
            # w, b reflects information prior to reward
            w_arr[n, :] = w
            b_arr[n] = b
            w = self.update_w(b, w, c[n], rpe_t, params_d)
            # Updating b!
            b = self.update_b(b, w, c[n], r[n], params_d)
        emu_data["Decision"] = c
        emu_data["Reward"] = r
        emu_data = add_switch(emu_data)
        emu_data = self.assign_latents(emu_data, qdiff, rpe, b_arr, w_arr)
        # qdiff[n], rpe[n], b, w, b_arr[n], w_arr[n]
        return emu_data

    def create_params(self, data):
        uniq_ids = data["ID"].unique()
        n_id = len(uniq_ids)
        new_params = {}
        for p in self.param_dict:
            self.param_dict[p].n_id = n_id
            new_params[p] = self.param_dict[p].eval()
        new_params["ID"] = uniq_ids
        return pd.DataFrame(new_params)

    def fit(self, data, *args, **kwargs):
        # Fit model to single subjects and get cross-validation results
        #
        # INPUTS:
        #   data - dataframe with all relevant data
        #
        # OUTPUTS:
        #   bic - subjects x model BIC values
        #   loglik - subjects x model log likelihood values from cross-validation
        #   param - model parameters for each subject for model 1 (qdiff + logodds)
        #   data - dataframe with 'qdiff' (Q-value difference),'rpe' (reward prediction error)
        #          and 'logodds' (log choice probability) columns added
        # contrastd softmax anne's with qdiff implementation

        # data <- fit_marginal
        # params <- create_params
        # params2x, x2params
        # x0 = params2x
        # nll(x, *data, *params): x2params(x), sim, get_prob, negsum
        # results = scipy.optimize.minimize(nll, x0)
        # self.fitted_params = x2params(results)

        data = self.fit_marginal(data)
        params_df = self.create_params(data)
        id_list = params_df["ID"]
        param_names = [c for c in params_df if c != "ID"]
        params2x = lambda params_df: params_df.drop(columns="ID").values.ravel(
            order="C"
        )
        x2params = (
            lambda x: pd.DataFrame(
                x.reshape((-1, len(self.param_dict)), order="C"),
                columns=param_names,
                index=id_list,
            )
            .reset_index()
            .rename({"index": "ID"})
        )
        x0 = params2x(params_df)

        def nll(x):
            params = x2params(x)
            data_sim = self.sim(data, params, *args, **kwargs)
            # balanced weights
            # vcs = data_sim['Decision'].value_counts()
            # total1s, total0s = 0, 0
            # if 1 in vcs.index:
            #     total1s = vcs[1]
            # if 0 in vcs.index:
            #     total0s = vcs[0]
            # data_sim.loc[data_sim['Decision'] == 1, 'class_weight'] = total1s / (total1s + total0s)
            # data_sim.loc[data_sim['Decision'] == 0, 'class_weight'] = total0s / (total1s + total0s)
            # weight by class weights
            c_valid_sel = data_sim["Decision"] != -1
            p_s = self.get_proba(data_sim, params)[c_valid_sel].values

            c_vs = data_sim.loc[c_valid_sel, "Decision"].values
            epsilon = 1e-15  # 0.001
            p_s = np.minimum(np.maximum(epsilon, p_s), 1 - epsilon)
            # try out class_weights
            return -(c_vs @ np.log(p_s) + (1 - c_vs) @ np.log(1 - p_s))

        params_bounds = [self.param_dict[pn].bounds for pn in param_names] * len(
            id_list
        )
        method = "L-BFGS-B"
        if "method" in kwargs:
            method = kwargs["method"]

        res = scipy.optimize.minimize(
            nll, x0, method=method, bounds=params_bounds, tol=1e-6
        )
        if not res.success:
            print("failed", res.message)
        # TODO: if result gives failure, you throw an error
        # constrain optimize
        self.fitted_params = x2params(res.x)
        negloglik = res.fun
        bic = len(res.x) * np.log(len(data)) + 2 * negloglik
        aic = 2 * len(res.x) + 2 * negloglik

        data_sim_opt = self.sim(data, self.fitted_params, *args, **kwargs)
        data["choice_p"] = self.get_proba(data_sim_opt)

        self.summary = {
            "bic": bic,
            "aic": aic,
            "latents": data[self.latent_names + ["choice_p"]].reset_index(drop=True),
        }
        return self


class RL_4p(CogModel2ABT_BWQ):
    def __init__(self):
        super().__init__()
        self.fixed_params.update({"b0": 1, "q_init": 0, "gam": 1, "CF": False})
        # used fixed for hyperparam_tuning
        self.param_dict = {
            "beta": CogParam(scipy.stats.expon(1), lb=0),
            "a_pos": CogParam(scipy.stats.uniform(), lb=0, ub=1),
            "a_neg": CogParam(scipy.stats.uniform(), lb=0, ub=1),
            "st": CogParam(scipy.stats.gamma(2, scale=0.2), lb=0),
        }
        self.latent_names = ["qdiff", "rpe", "q0", "q1"]

    def __str__(self):
        return "RL4p"

    def id_param_init(self, params, id):
        varp_list = ["a_pos", "a_neg", "beta", "st"]
        fixp_list = ["b0", "q_init", "gam"]
        a_pos, a_neg, beta, st = params.loc[
            params["ID"] == id, varp_list
        ].values.ravel()
        b0, q0, gam = [self.fixed_params[fp] for fp in fixp_list]
        w0 = np.array([q0] * 2)
        return {
            "b0": b0,
            "w0": w0,
            "gam": gam,
            "a_pos": a_pos,
            "a_neg": a_neg,
            "beta": beta,
            "st": st,
        }

    def calc_q(self, b, w):
        return w

    def update_b(self, b, w, c_t, r_t, params_i):
        return b

    def update_w(self, b, w, c_t, rpe_t, params_i):
        if self.fixed_params["CF"]:
            d_rpe = rpe_t
            if c_t == 1:
                d_rpe = rpe_t[::-1]
        else:
            if c_t == 1:
                d_rpe = np.array([0, rpe_t])
            else:
                d_rpe = np.array([rpe_t, 0])
        if "alpha" in params_i:
            alphas = params_i["alpha"]
        elif ("a_pos" in params_i) and ("a_neg" in params_i):
            alphas = params_i["a_pos"] * (d_rpe >= 0) + params_i["a_neg"] * (d_rpe < 0)
        else:
            raise ValueError("NO alpha?")
        return w + alphas * d_rpe

    def assign_latents(self, data, qdiff, rpe, b_arr, w_arr):
        data["qdiff"] = qdiff
        if self.fixed_params["CF"]:
            data[["rpe", "rpe_cf"]] = rpe
        else:
            data["rpe"] = rpe
        data[["q0", "q1"]] = w_arr
        return data


class RLCF(RL_4p):
    """Model class for counter-factual Q-learning, described in Eckstein et al. 2022
    https://doi.org/10.1016/j.dcn.2022.101106
    Here we implement a model where choice stays does not alter Q value but rather
    affects choice selection.
    # TODO: same model without stickiness
    """

    def __init__(self):
        super().__init__()
        self.fixed_params.update({"b0": 1, "q_init": 0, "gam": 1, "CF": True})
        self.latent_names = ["qdiff", "rpe", "rpe_cf", "q0", "q1"]  # ['qdiff', 'rpe']

    def __str__(self):
        return "RLCF"


class RL_st(RL_4p):
    """Model class for Q-learning with stickiness"""

    def __init__(self):
        super().__init__()
        self.fixed_params.update({"b0": 1, "q_init": 0, "gam": 1, "CF": False})
        # used fixed for hyperparam_tuning
        self.param_dict = {
            "beta": CogParam(scipy.stats.expon(1), lb=0),
            "alpha": CogParam(scipy.stats.uniform(), lb=0, ub=1),
            "st": CogParam(scipy.stats.gamma(2, scale=0.2), lb=0),
        }
        self.latent_names = ["qdiff", "rpe", "q0", "q1"]  # ['qdiff', 'rpe']

    def __str__(self):
        return "RLst"

    def id_param_init(self, params, id):
        varp_list = ["alpha", "beta", "st"]
        fixp_list = ["b0", "q_init", "gam"]
        alpha, beta, st = params.loc[params["ID"] == id, varp_list].values.ravel()
        b0, q0, gam = [self.fixed_params[fp] for fp in fixp_list]
        w0 = np.array([q0] * 2)
        return {"b0": b0, "w0": w0, "gam": gam, "alpha": alpha, "beta": beta, "st": st}


class RL(RL_4p):
    """Model class for counter-factual Q-learning, described in Eckstein et al. 2022
    https://doi.org/10.1016/j.dcn.2022.101106
    Here we implement a model where choice stays does not alter Q value but rather
    affects choice selection.
    # TODO: same model without stickiness
    """

    def __init__(self):
        super().__init__()
        self.fixed_params.update({"b0": 1, "q_init": 0, "gam": 1, "CF": False})
        # used fixed for hyperparam_tuning
        self.param_dict = {
            "beta": CogParam(scipy.stats.expon(1), lb=0),
            "alpha": CogParam(scipy.stats.uniform(), lb=0, ub=1),
        }
        self.latent_names = ["qdiff", "rpe", "q0", "q1"]  # ['qdiff', 'rpe']

    def __str__(self):
        return "RL"

    def id_param_init(self, params, id):
        varp_list = ["alpha", "beta"]
        fixp_list = ["b0", "q_init", "gam"]
        alpha, beta = params.loc[params["ID"] == id, varp_list].values.ravel()
        b0, q0, gam = [self.fixed_params[fp] for fp in fixp_list]
        w0 = np.array([q0] * 2)
        return {"b0": b0, "w0": w0, "gam": gam, "alpha": alpha, "beta": beta}

    def get_proba(self, data, params=None):
        if "loglik" in data.columns:
            return np.exp(data["loglik"].values)
        else:
            params_in = self.fitted_params[["ID", "beta"]] if params is None else params
            new_data = data.merge(params_in, how="left", on="ID")
            return expit(new_data["qdiff"] * new_data["beta"])

    def select_action(self, qdiff, m_1back, params):
        stay = 1
        if m_1back == 0:
            stay = -1
        elif m_1back == np.nan:
            stay = 0
        choice_p = expit(qdiff * params["beta"])
        return int(np.random.random() <= choice_p)


class BayesianModel(CogModel2ABT_BWQ):
    def __init__(self):
        super().__init__()

    @staticmethod
    def bayesian_step(b, ps, c_t, r_t, sw):
        # performs bayesian step to update b
        def projected_rew(r_i, c_i, z):
            if c_i == 1:
                P = ps[::-1]
            else:
                P = ps
            if r_i == 0:
                P = 1 - P
            return P[int(z)]

        p1, p0 = projected_rew(r_t, c_t, 1), projected_rew(r_t, c_t, 0)
        eps = 1e-15
        b = ((1 - sw) * b * p1 + sw * (1 - b) * p0) / (
            b * p1 + (1 - b) * p0
        )  # catch when b goes to 1
        b = np.maximum(np.minimum(b, 1 - eps), eps)
        return b


class BIModel(BayesianModel):
    """Model class for Bayesian inference model, described in Eckstein et al. 2022,
    adapted to BWQ framework: https://doi.org/10.1016/j.dcn.2022.101106
    Here we implement a model where choice stays does not alter Q value but rather
    affects choice selection.
    """

    def __init__(self):
        super().__init__()
        self.fixed_params.update({"b0": 0.5, "gam": 1, "p_eps": 1e-4, "w0": 0.5})
        # used fixed for hyperparam_tuning
        self.param_dict = {
            "beta": CogParam(scipy.stats.expon(1), lb=0),  # v9 constraining
            "st": CogParam(scipy.stats.gamma(2, scale=0.2), lb=0),
            "p_rew": CogParam(scipy.stats.beta(90, 30), lb=0, ub=1),  # 0.75
            "sw": CogParam(scipy.stats.uniform(loc=0, scale=0.05), lb=0.001),
        }  # v10 0.05
        self.latent_names = ["qdiff", "b", "rpe"]

    def __str__(self):
        return "BI"

    def latents_init(self, N):
        qdiff = np.zeros(N)
        if self.fixed_params["CF"]:
            rpe = np.zeros((N, 2))
        else:
            rpe = np.zeros((N, 1))
        b_arr = np.zeros(N)
        w_arr = np.zeros((N, 1))
        return qdiff, rpe, b_arr, w_arr

    def id_param_init(self, params, id):
        varp_list = ["p_rew", "sw", "beta", "st"]
        fixp_list = ["b0", "w0", "p_eps", "gam"]
        p_rew, sw, beta, st = params.loc[params["ID"] == id, varp_list].values.ravel()
        b0, w0, p_eps, gam = [self.fixed_params[fp] for fp in fixp_list]
        return {
            "p_rew": p_rew,
            "p_eps": p_eps,
            "sw": sw,
            "gam": gam,
            "w0": w0,
            "b0": b0,
            "beta": beta,
            "st": st,
        }

    def calc_q(self, b, w):
        return np.array([(1 - b) / 2, b / 2])

    def update_b(self, b, w, c_t, r_t, params_i):
        sw = params_i["sw"]
        ps = np.array([params_i["p_rew"], params_i["p_eps"]])
        return self.bayesian_step(b, ps, c_t, r_t, sw)

    def update_w(self, b, w, c_t, rpe_t, params_i):
        return w

    def assign_latents(self, data, qdiff, rpe, b_arr, w_arr):
        data["qdiff"] = qdiff
        data["rpe"] = rpe
        data["b"] = b_arr
        return data

    def get_proba(self, data, params=None):
        if "loglik" in data.columns:
            return np.exp(data["loglik"].values)
        else:
            params_in = (
                self.fitted_params[["ID", "beta", "st"]] if params is None else params
            )
            new_data = data.merge(params_in, how="left", on="ID")
            return expit(
                new_data["qdiff"] * new_data["beta"] + new_data["stay"] * new_data["st"]
            )
        # v10 mix beta and st


class BI_log(BIModel):
    """Model class for Bayesian inference model, described in Eckstein et al. 2022,
    adapted to BWQ framework: https://doi.org/10.1016/j.dcn.2022.101106
    Here we implement a model where choice stays does not alter Q value but rather
    affects choice selection.
    TODO: implements a log version of BI model
    """

    def __init__(self):
        super().__init__()
        self.fixed_params.update(
            {"b0": 0.5, "gam": 1, "p_eps": 1e-4, "w0": 1, "p_rew": 0.75}
        )
        # used fixed for hyperparam_tuning
        self.param_dict = {
            "beta": CogParam(scipy.stats.expon(1), lb=0),  # v9 constraining
            "st": CogParam(scipy.stats.gamma(2, scale=0.2), lb=0),
            "sw": CogParam(scipy.stats.uniform(loc=0, scale=0.05), lb=0.001),
        }  # v10 0.05

    def __str__(self):
        return "BIlog"

    def calc_q(self, b, w):
        return np.array([np.log(1 - b), np.log(b)])

    def id_param_init(self, params, id):
        varp_list = ["sw", "beta", "st"]
        fixp_list = ["b0", "w0", "p_rew", "p_eps", "gam"]
        sw, beta, st = params.loc[params["ID"] == id, varp_list].values.ravel()
        b0, w0, p_rew, p_eps, gam = [self.fixed_params[fp] for fp in fixp_list]
        return {
            "p_rew": p_rew,
            "p_eps": p_eps,
            "sw": sw,
            "gam": gam,
            "w0": w0,
            "b0": b0,
            "beta": beta,
            "st": st,
        }


class BIModel_fixp(BIModel):
    """Model class for Bayesian inference model, described in Eckstein et al. 2022,
    adapted to BWQ framework: https://doi.org/10.1016/j.dcn.2022.101106
    Here we implement a model where choice stays does not alter Q value but rather
    affects choice selection.
    """

    def __init__(self):
        super().__init__()
        self.fixed_params.update(
            {"b0": 0.5, "gam": 1, "p_eps": 1e-4, "w0": 0.5, "p_rew": 0.75}
        )
        # used fixed for hyperparam_tuning
        self.param_dict = {
            "beta": CogParam(scipy.stats.expon(1), lb=0),
            "st": CogParam(scipy.stats.gamma(2, scale=0.2), lb=0),
            # "p_rew": CogParam(scipy.stats.beta(90, 30), lb=0, ub=1), # 0.75
            "sw": CogParam(scipy.stats.uniform(loc=0, scale=0.05), lb=0.001, ub=1),
        }  # 0.05

    def __str__(self):
        return "BIfp"

    def id_param_init(self, params, id):
        varp_list = ["sw", "beta", "st"]
        fixp_list = ["b0", "w0", "p_rew", "p_eps", "gam"]
        sw, beta, st = params.loc[params["ID"] == id, varp_list].values.ravel()
        b0, w0, p_rew, p_eps, gam = [self.fixed_params[fp] for fp in fixp_list]
        return {
            "p_rew": p_rew,
            "p_eps": p_eps,
            "sw": sw,
            "gam": gam,
            "w0": w0,
            "b0": b0,
            "beta": beta,
            "st": st,
        }


class PCModel(BayesianModel):
    """

    Debate: single subject fit, with one paramter sets, and multiple PC models
    OR:
    one fit for all subjects, and multiple parameter sets -> bayesian mixed effects model

    Potentially use ID to identify different snippets of data


    Implements base policy compression model that seeks to maximize rewards subject
    to cognitive resource constraint encoded as an upper bound to mutual information
    between marginal policy state/belief dependent policy.

    TODO: implement interface that returns form of b, Q according to task structure

    @cite: Lai, Gershman 2021, https://doi.org/10.1016/bs.plm.2021.02.004.
    @credit: inspired by pcmodel.py written by Sam Gershman

    params:
        .beta: policy compression parameter
        .p_rew: reward prob in correct choice
        .p_eps: reward prob in wrong choice
        .st: stickiness to previous choice
        .sw: block switch probability
        .gam: decay rate simulating forgetting across days, $w_{t+1} = \gamma w_0 + (1-\gamma) w_{t}$

    latents:
        .b: belief vector, propagating and updated via bayesian inference
        .w: weights for belief vector
        .rpe: reward prediction error
        .Q: action values

    input data must have the following columns:
    ID, Subject, Session, Trial, blockTrial, blockLength, Target, Decision, Switch, Reward, Condition

    """

    def __init__(self):
        super().__init__()
        self.fixed_params.update({"b0": 0.5, "K_marginal": 50, "a_marg": 0.2})
        # used fixed for hyperparam_tuning
        self.param_dict = {
            "alpha": CogParam(scipy.stats.uniform(), lb=0, ub=1),
            "beta": CogParam(scipy.stats.expon(1), lb=0),
            "p_rew_init": CogParam(scipy.stats.beta(90, 30), lb=0, ub=1),
            # "p_rew_init": CogParam(0.75, lb=0, ub=1),
            "p_eps_init": CogParam(scipy.stats.beta(1, 30), lb=0, ub=1),
            # "p_eps_init": CogParam(0, lb=0, ub=1),
            "st": CogParam(scipy.stats.gamma(2, scale=0.2), lb=0),
            "sw": CogParam(scipy.stats.uniform(loc=0, scale=0.05), lb=0.001, ub=1),
            "gam": CogParam(scipy.stats.uniform(), lb=0, ub=1),
        }
        self.fitted_params = None
        self.latent_names = ["qdiff", "rpe", "p_rew", "p_eps", "b"]
        self.marg_name = "logodds"
        self.summary = {}

    def __str__(self):
        return "PC"

    def id_param_init(self, params, id):
        varp_list = ["p_rew_init", "p_eps_init", "gam", "sw", "alpha", "beta", "st"]
        fixp_list = ["b0"]
        p_rew_init, p_eps_init, gam, sw, alpha, beta, st = params.loc[
            params["ID"] == id, varp_list
        ].values.ravel()
        b0 = self.fixed_params["b0"]
        w0 = np.array([p_rew_init, p_eps_init])
        return {
            "b0": b0,
            "w0": w0,
            "gam": gam,
            "sw": sw,
            "alpha": alpha,
            "beta": beta,
            "st": st,
        }

    def assign_latents(self, data, qdiff, rpe, b_arr, w_arr):
        data["qdiff"] = qdiff
        data["rpe"] = rpe
        data["b"] = b_arr
        data[["p_rew", "p_eps"]] = w_arr
        return data

    def calc_q(self, b, w):
        """
        Function to calculate action values given b, belief states, and w, the weights
        b = probability believing in state 1
        """
        f_b = np.array([1 - b, b]).reshape((1, 2))
        W = np.diag([w[0] - w[1]] * len(w)) + w[1]
        return f_b @ W

    def calc_qdiff(self, b, w):
        """
        Function to calculate q value differences given b, w
        """
        return (2 * b - 1) * (w[0] - w[1])

    def update_w(self, b, w, c_t, rpe_t, params_i):
        # update w according to reward prediction error, uncomment later
        f_b = np.array([1 - b, b])
        # change the update rule
        # w[c.iat[n]] = w[c.iat[n]] + alpha * rpe[n] * f_b
        dw = np.array([1 - b, b])  # alternatively p2 = b+c-2bc
        if c_t == 1:
            dw = np.array([b, 1 - b])
        w += params_i["alpha"] * rpe_t * dw
        w = np.minimum(np.maximum(w, 0), 1)
        return w

    def update_b(self, b, w, c_t, r_t, params_i):
        sw = params_i["sw"]
        return self.bayesian_step(b, w, c_t, r_t, sw)

    def fit_marginal(self, data):
        # Fit marginal choice probability
        #
        # INPUTS:
        #   data - dataframe containing all the relevant data
        #   K - number of different exponential weighting values
        #
        # OUTPUTS:
        #   data with new column 'logodds' (choice log odds)
        K = self.fixed_params["K_marginal"]
        alpha = np.linspace(0.001, 0.3, num=K)
        N = data.shape[0]
        m = np.zeros((N, K)) + 0.5
        c = data["Decision"]
        sess = data["Session"]

        for n in range(N):
            if (n > 0) and (sess.iat[n] == sess.iat[n - 1]):
                if c.iat[n - 1] == -1:
                    # Handling miss decisions
                    m[n,] = m[n - 1,]
                else:
                    m[n,] = (1 - alpha) * m[n - 1,] + alpha * c.iat[n - 1]
        eps = 0.001
        m = np.minimum(np.maximum(eps, m), 1 - eps)
        c_vec = c[c != -1].values  # handling missing decisions
        m_mat = m[c != -1]
        # np.seterr(all='raise')
        L = np.dot(c_vec, np.log(m_mat)) + np.dot((1 - c_vec), np.log(1 - m_mat))
        m = m[:, np.argmax(L)]
        data["logodds"] = np.log(m) - np.log(1 - m)
        data["marg"] = m
        alpha_opt = alpha[np.argmax(L)]
        print("alpha =", alpha_opt)
        self.fixed_params["a_marg"] = alpha_opt
        return data

    def select_action(self, qdiff, m_1back, params):
        logodd = np.log(m_1back) - np.log(1 - m_1back)
        choice_p = expit(qdiff * params["beta"] + logodd * params["st"])
        return int(np.random.random() <= choice_p)

    def marginal_init(self):
        return 0.5

    def update_marginal(self, c_t, m_1back):
        a_marg = self.fixed_params["a_marg"]
        return (1 - a_marg) * m_1back + a_marg * c_t

    def get_proba(self, data, params=None):
        if "loglik" in data.columns:
            return np.exp(data["loglik"].values)
        else:
            params_in = (
                self.fitted_params[["ID", "beta", "st"]] if params is None else params
            )
            new_data = data.merge(params_in, how="left", on="ID")
            return expit(
                new_data["qdiff"] * new_data["beta"]
                + new_data["logodds"] * new_data["st"]
            )

    # pseudo R-squared: http://courses.atlas.illinois.edu/fall2016/STAT/STAT200/RProgramming/LogisticRegression.html#:~:text=Null%20Model,-The%20simplest%20model&text=The%20fitted%20equation%20is%20ln,e%E2%88%923.36833)%3D0.0333.


class PCModel_fixpswgam(PCModel):
    """PCModel class with fixed p, sw, gam
    # STILL updates w
    """

    def __init__(self):
        super().__init__()
        self.fixed_params.update(
            {
                "b0": 0.5,
                "K_marginal": 50,
                "p_rew_init": 0.75,
                "p_eps_init": 0,
                "sw": 0.02,
                "gam": 1,
                "alpha": 0,
                "a_marg": 0.2,
            }
        )
        # used fixed for hyperparam_tuning
        self.param_dict = {
            "beta": CogParam(scipy.stats.expon(1), lb=0),
            "st": CogParam(scipy.stats.gamma(2, scale=0.2), lb=0),
        }

    def __str__(self):
        return "PCf"

    def id_param_init(self, params, id):
        varp_list = ["beta", "st"]
        fixp_list = ["b0", "p_rew_init", "p_eps_init", "gam", "sw", "alpha"]
        beta, st = params.loc[params["ID"] == id, varp_list].values.ravel()
        b0, p_rew_init, p_eps_init, gam, sw, alpha = [
            self.fixed_params[fp] for fp in fixp_list
        ]
        w0 = np.array([p_rew_init, p_eps_init])
        return {
            "b0": b0,
            "w0": w0,
            "gam": gam,
            "sw": sw,
            "alpha": alpha,
            "beta": beta,
            "st": st,
        }

    def update_w(self, b, w, c_t, rpe_t, params_i):
        # No update!
        # v8: update
        # v7: no update
        # <= v8: gam = 0
        return w


class CogParam:
    def __init__(self, value=None, fixed_init=False, n_id=1, lb=None, ub=None):
        self.value = value
        self.fixed_init = fixed_init
        self.bounds = (lb, ub)
        self.n_id = n_id

    def eval(self):
        if isinstance(self.value, Number):
            return np.full(self.n_id, self.value)
        elif hasattr(self.value, "rvs"):
            return self.value.rvs(size=self.n_id)
        else:
            return ValueError()
