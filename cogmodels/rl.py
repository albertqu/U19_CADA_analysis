import numpy as np
from scipy.special import expit
import scipy

from cogmodels.base import CogModel2ABT_BWQ, CogParam


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
    """
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
            if "beta" in self.fixed_params:  # change emulate function
                params_in = (
                    self.fitted_params[["ID", "st"]].copy()
                    if params is None
                    else params
                )
                params_in["beta"] = self.fixed_params["beta"]
            else:
                params_in = (
                    self.fitted_params[["ID", "beta", "st"]]
                    if params is None
                    else params
                )
            new_data = data.merge(params_in, how="left", on="ID")
            return expit(new_data["qdiff"] * new_data["beta"])

    def select_action(self, qdiff, m_1back, params):
        choice_p = expit(qdiff * params["beta"])
        return int(np.random.random() <= choice_p)


class RL_Forgetting(RL_4p):
    """
    Forgetting only on unchosen option ~ stickiness
    [Katahira et al. 2015](https://doi.org/10.1016/j.cub.2021.12.006)
    """

    def __init__(self):
        super().__init__()
        self.fixed_params.update({"st": 0})
        self.param_dict = {
            "beta": CogParam(scipy.stats.expon(1), lb=0),
            "a_pos": CogParam(scipy.stats.uniform(), lb=0, ub=1),
            "a_neg": CogParam(scipy.stats.uniform(), lb=0, ub=1),
            "zeta": CogParam(
                scipy.stats.uniform(), lb=0, ub=1
            ),  # forgetting of unchosen
        }

    def __str__(self):
        return "RLFQ"

    def get_proba(self, data, params=None):
        if "loglik" in data.columns:
            return np.exp(data["loglik"].values)
        else:
            params_in = self.fitted_params[["ID", "beta"]] if params is None else params
            new_data = data.merge(params_in, how="left", on="ID")
            return expit(new_data["qdiff"] * new_data["beta"])

    def select_action(self, qdiff, m_1back, params):
        choice_p = expit(qdiff * params["beta"])
        return int(np.random.random() <= choice_p)

    def id_param_init(self, params, id):
        varp_list = ["a_pos", "a_neg", "beta", "zeta"]
        fixp_list = ["b0", "q_init", "gam", "st"]
        var_dict = params.loc[params["ID"] == id, varp_list].iloc[0].to_dict()
        w0_arr = [self.fixed_params["q_init"]] * 2
        w0 = np.array(w0_arr)
        d = {"w0": w0}
        d.update({fp: self.fixed_params[fp] for fp in fixp_list})
        d.update(var_dict)
        return d

    def update_w(self, b, w, c_t, rpe_t, params_i):
        w = super().update_w(b, w, c_t, rpe_t, params_i)
        w[1 - c_t] = w[1 - c_t] * params_i["zeta"]
        return w


class RL_Forgetting3p(RL_4p):
    """
    Forgetting only on unchosen option ~ stickiness
    [Katahira et al. 2015](https://doi.org/10.1016/j.cub.2021.12.006)
    """

    def __init__(self):
        super().__init__()
        self.fixed_params.update({"st": 0})
        self.param_dict = {
            "beta": CogParam(scipy.stats.expon(1), lb=0),
            "a_pos": CogParam(scipy.stats.uniform(), lb=0, ub=1),
            "a_neg": CogParam(scipy.stats.uniform(), lb=0, ub=1),
        }

    def __str__(self):
        return "RLFQ3p"

    def get_proba(self, data, params=None):
        if "loglik" in data.columns:
            return np.exp(data["loglik"].values)
        else:
            params_in = self.fitted_params[["ID", "beta"]] if params is None else params
            new_data = data.merge(params_in, how="left", on="ID")
            return expit(new_data["qdiff"] * new_data["beta"])

    def select_action(self, qdiff, m_1back, params):
        choice_p = expit(qdiff * params["beta"])
        return int(np.random.random() <= choice_p)

    def id_param_init(self, params, id):
        varp_list = ["a_pos", "a_neg", "beta"]
        fixp_list = ["b0", "q_init", "gam", "st"]
        var_dict = params.loc[params["ID"] == id, varp_list].iloc[0].to_dict()
        w0_arr = [self.fixed_params["q_init"]] * 2
        w0 = np.array(w0_arr)
        d = {"w0": w0}
        d.update({fp: self.fixed_params[fp] for fp in fixp_list})
        d.update(var_dict)
        return d

    def update_w(self, b, w, c_t, rpe_t, params_i):
        w = super().update_w(b, w, c_t, rpe_t, params_i)
        w[1 - c_t] = w[1 - c_t] * (1 - params_i["a_pos"] / 2 - params_i["a_neg"] / 2)
        return w


class RL_FQST(RL_4p):
    """
    Forgetting on all options + stickiness, use forgetting bilaterally
    [Beron and Linderman 2022](https://www.pnas.org/doi/abs/10.1073/pnas.2113961119)
    relative equivalence to RFLR
    """

    def __init__(self):
        super().__init__()
        self.param_dict.update(
            {
                "zeta": CogParam(
                    scipy.stats.uniform(), lb=0, ub=1
                ),  # forgetting of unchosen
            }
        )

    def __str__(self):
        return "RLFQST"

    def id_param_init(self, params, id):
        varp_list = ["a_pos", "a_neg", "beta", "st", "zeta"]
        fixp_list = ["b0", "q_init", "gam"]
        var_dict = params.loc[params["ID"] == id, varp_list].iloc[0].to_dict()
        w0_arr = [self.fixed_params["q_init"]] * 2
        w0 = np.array(w0_arr)
        d = {"w0": w0}
        d.update({fp: self.fixed_params[fp] for fp in fixp_list})
        d.update(var_dict)
        return d

    def update_w(self, b, w, c_t, rpe_t, params_i):
        w = super().update_w(b, w, c_t, rpe_t, params_i)
        w = w * params_i["zeta"]
        return w
