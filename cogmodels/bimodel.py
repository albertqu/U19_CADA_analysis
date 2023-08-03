import numpy as np
from scipy.special import expit
import scipy

from cogmodels.bayesian_base import BayesianModel
from cogmodels.base import CogParam


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


class BRL_fp(BIModel):
    """Model class for Bayesian inference model RL model, based on Babayan et al., with stickiness"""

    def __init__(self):
        super().__init__()
        self.fixed_params.update({"b0": 0.5, "gam": 1, "p_eps": 1e-4, "p_rew": 0.75})
        # used fixed for hyperparam_tuning
        self.param_dict = {
            "beta": CogParam(scipy.stats.expon(1), lb=0),
            "st": CogParam(scipy.stats.gamma(2, scale=0.2), lb=0),
            "sw": CogParam(scipy.stats.uniform(loc=0, scale=0.05), lb=0.001, ub=1),
        }  # 0.05

    def __str__(self):
        return "BRLfp"

    def latents_init(self, N):
        qdiff = np.zeros(N)
        rpe = np.zeros((N, 1))
        b_arr = np.zeros(N)
        w_arr = np.zeros((N, 2))
        return qdiff, rpe, b_arr, w_arr

    def id_param_init(self, params, id):
        varp_list = ["sw", "beta", "st"]
        fixp_list = ["b0", "p_rew", "p_eps", "gam"]
        sw, beta, st = params.loc[params["ID"] == id, varp_list].values.ravel()
        b0, p_rew, p_eps, gam = [self.fixed_params[fp] for fp in fixp_list]
        w0 = np.array([p_rew, p_eps])
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
        f_b = np.array([1 - b, b]).reshape((1, 2))
        W = np.array([[w[0], w[1]], [w[1], w[0]]])
        # W0 = np.diag([w[0] - w[1]] * len(w)) + w[1]
        return f_b @ W


class BRL_pr(BIModel):
    """Model class for Bayesian inference model RL model, based on Babayan et al., with stickiness"""

    def __init__(self):
        super().__init__()
        self.fixed_params.update({"b0": 0.5, "gam": 1, "p_eps": 1e-4, "wr0": 0.75})
        # used fixed for hyperparam_tuning
        self.param_dict = {
            "beta": CogParam(scipy.stats.expon(1), lb=0),
            "st": CogParam(scipy.stats.gamma(2, scale=0.2), lb=0),
            "sw": CogParam(scipy.stats.uniform(loc=0, scale=0.05), lb=0.001, ub=1),
            "p_rew": CogParam(scipy.stats.beta(90, 30), lb=0, ub=1),
        }  # 0.05

    def __str__(self):
        return "BRLpr"

    def latents_init(self, N):
        qdiff = np.zeros(N)
        rpe = np.zeros((N, 1))
        b_arr = np.zeros(N)
        w_arr = np.zeros((N, 2))
        return qdiff, rpe, b_arr, w_arr

    def id_param_init(self, params, id):
        varp_list = ["sw", "beta", "st", "p_rew"]
        fixp_list = ["b0", "p_eps", "gam", "wr0"]
        sw, beta, st, p_rew = params.loc[params["ID"] == id, varp_list].values.ravel()
        b0, p_eps, gam, wr0 = [self.fixed_params[fp] for fp in fixp_list]
        w0 = np.array([wr0, p_eps])
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
        f_b = np.array([1 - b, b]).reshape((1, 2))
        W = np.array([[w[0], w[1]], [w[1], w[0]]])
        # W0 = np.diag([w[0] - w[1]] * len(w)) + w[1]
        return f_b @ W


class BRL_wr(BIModel):
    """Model class for Bayesian inference model RL model, based on Babayan et al., with stickiness"""

    def __init__(self):
        super().__init__()
        self.fixed_params.update({"b0": 0.5, "gam": 1, "p_eps": 1e-4, "p_rew": 0.75})
        # used fixed for hyperparam_tuning
        self.param_dict = {
            "beta": CogParam(scipy.stats.expon(1), lb=0),
            "st": CogParam(scipy.stats.gamma(2, scale=0.2), lb=0),
            "sw": CogParam(scipy.stats.uniform(loc=0, scale=0.05), lb=0.001, ub=1),
            "wr0": CogParam(scipy.stats.beta(90, 30), lb=0, ub=1),
        }  # 0.05

    def __str__(self):
        return "BRLwr"

    def latents_init(self, N):
        qdiff = np.zeros(N)
        rpe = np.zeros((N, 1))
        b_arr = np.zeros(N)
        w_arr = np.zeros((N, 2))
        return qdiff, rpe, b_arr, w_arr

    def id_param_init(self, params, id):
        varp_list = ["sw", "beta", "st", "wr0"]
        fixp_list = ["b0", "p_rew", "p_eps", "gam"]
        sw, beta, st, wr0 = params.loc[params["ID"] == id, varp_list].values.ravel()
        b0, p_rew, p_eps, gam = [self.fixed_params[fp] for fp in fixp_list]
        w0 = np.array([wr0, p_eps])
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
        f_b = np.array([1 - b, b]).reshape((1, 2))
        W = np.array([[w[0], w[1]], [w[1], w[0]]])
        # W0 = np.diag([w[0] - w[1]] * len(w)) + w[1]
        return f_b @ W

    def update_w(self, b, w, c_t, rpe_t, params_i):
        # update w according to reward prediction error, uncomment later
        # change the update rule
        # w[c.iat[n]] = w[c.iat[n]] + alpha * rpe[n] * f_b
        dw = 1 - b  # alternatively p2 = b+c-2bc
        w_new = np.copy(w)
        if c_t == 1:
            dw = b
        elif c_t == -1:
            return w
        w_new[0] += params_i["alpha"] * rpe_t * dw
        return w_new
