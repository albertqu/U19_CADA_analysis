import numpy as np
from scipy.special import expit
import scipy

from cogmodels.base import CogParam
from cogmodels.rl import RL_4p


class RL_Grossman(RL_4p):
    """
    Adapted from Grossman et al., 2021

    expected uncertainty reduces learning rate
    unexpected uncertainty increases learning rate

    latent: w: [q0, q1, omega, nu, a_neg_t]

    """

    def __init__(self):
        super().__init__()
        self.fixed_params.update({"b0": 1, "q_init": 0, "gam": 1})
        # used fixed for hyperparam_tuning
        self.param_dict.update(
            {
                "zeta": CogParam(scipy.stats.uniform(), lb=0, ub=1),
                # forgetting parameter
                "alpha_nu": CogParam(scipy.stats.uniform(), lb=0, ub=1),
                # adaptive parameter for expected uncertainty
                "psi": CogParam(scipy.stats.uniform(), lb=0, ub=1),
                # adaptive parameter for alpha_neg
            }
        )
        self.latent_names = [
            "qdiff",
            "rpe",
            "q0",
            "q1",
            "nu",  # sporatic / unexpected uncertainty
            "omega",  # expected uncertainty
            "a_neg_t",  # adaptive negative learning rate
        ]

    def __str__(self):
        return "RL_meta"

    def latents_init(self, N):
        qdiff, rpe, b_arr, _ = super().latents_init(N)
        w_arr = np.zeros((N, 5))
        return qdiff, rpe, b_arr, w_arr

    def id_param_init(self, params, id):
        varp_list = ["a_pos", "a_neg", "beta", "st", "zeta", "alpha_nu", "psi"]
        fixp_list = ["b0", "q_init", "gam"]
        var_dict = params.loc[params["ID"] == id, varp_list].iloc[0].to_dict()
        w0_arr = [self.fixed_params["q_init"]] * 2 + [0, 0, var_dict["a_neg"]]
        w0 = np.array(w0_arr)
        d = {"w0": w0}
        d.update({fp: self.fixed_params[fp] for fp in fixp_list})
        d.update(var_dict)
        return d

    def calc_q(self, b, w):
        return w[:2]

    def update_b(self, b, w, c_t, r_t, params_i):
        return b

    def update_w(self, b, w, c_t, rpe_t, params_i):
        if c_t == 1:
            d_rpe = np.array([0, rpe_t])
        else:
            d_rpe = np.array([rpe_t, 0])

        if "alpha" in params_i:
            alphas = params_i["alpha"]
        elif "a_pos" in params_i:
            alphas = params_i["a_pos"] * (d_rpe >= 0) + w[-1] * (d_rpe < 0)
        else:
            raise ValueError("NO alpha?")

        # update qs
        w[:2] = w[:2] + alphas * d_rpe
        w[1 - c_t] = w[1 - c_t] * params_i["zeta"]  # forgetting decay

        # calculate uncertainty latents
        omega = w[2]
        nu_t = abs(rpe_t) - omega
        omega = omega + params_i["alpha_nu"] * nu_t
        w[2:4] = [omega, nu_t]

        # adapt learning rate
        if rpe_t < 0:
            alpha_neg_t = w[-1]
            a_neg_hat = nu_t + params_i["a_neg"]
            alpha_neg_t = alpha_neg_t + params_i["psi"] * (a_neg_hat - alpha_neg_t)
            alpha_neg_t = max(0, alpha_neg_t)
            w[-1] = alpha_neg_t
        return w

    def assign_latents(self, data, qdiff, rpe, b_arr, w_arr):
        data["qdiff"] = qdiff
        if self.fixed_params["CF"]:
            data[["rpe", "rpe_cf"]] = rpe
        else:
            data["rpe"] = rpe
        data[["q0", "q1", "omega", "nu", "a_neg_t"]] = w_arr
        return data


class RL_Grossman_nof(RL_Grossman):
    """
    Adapted from Grossman et al., 2021

    expected uncertainty reduces learning rate
    unexpected uncertainty increases learning rate

    no forgetting parameter due to forgetting and stickiness redundancy
    latent: w: [q0, q1, omega, nu, a_neg_t]

    """

    def __init__(self):
        super().__init__()
        # used fixed for hyperparam_tuning
        del self.param_dict["zeta"]

    def __str__(self):
        return "RL_meta_nof"

    def id_param_init(self, params, id):
        varp_list = ["a_pos", "a_neg", "beta", "st", "alpha_nu", "psi"]
        fixp_list = ["b0", "q_init", "gam"]
        var_dict = params.loc[params["ID"] == id, varp_list].iloc[0].to_dict()
        w0_arr = [self.fixed_params["q_init"]] * 2 + [0, 0, var_dict["a_neg"]]
        w0 = np.array(w0_arr)
        d = {"w0": w0}
        d.update({fp: self.fixed_params[fp] for fp in fixp_list})
        d.update(var_dict)
        return d

    def update_w(self, b, w, c_t, rpe_t, params_i):
        if c_t == 1:
            d_rpe = np.array([0, rpe_t])
        else:
            d_rpe = np.array([rpe_t, 0])

        if "alpha" in params_i:
            alphas = params_i["alpha"]
        elif "a_pos" in params_i:
            alphas = params_i["a_pos"] * (d_rpe >= 0) + w[-1] * (d_rpe < 0)
        else:
            raise ValueError("NO alpha?")

        # update qs
        w[:2] = w[:2] + alphas * d_rpe

        # calculate uncertainty latents
        omega = w[2]
        nu_t = abs(rpe_t) - omega
        omega = omega + params_i["alpha_nu"] * nu_t
        w[2:4] = [omega, nu_t]

        # adapt learning rate
        if rpe_t < 0:
            alpha_neg_t = w[-1]
            a_neg_hat = nu_t + params_i["a_neg"]
            alpha_neg_t = alpha_neg_t + params_i["psi"] * (a_neg_hat - alpha_neg_t)
            alpha_neg_t = max(0, alpha_neg_t)
            w[-1] = alpha_neg_t
        return w


class RL_Grossman_nost(RL_Grossman):
    """
    Adapted from Grossman et al., 2021

    expected uncertainty reduces learning rate
    unexpected uncertainty increases learning rate

    latent: w: [q0, q1, omega, nu, a_neg_t]

    """

    def __init__(self):
        super().__init__()
        del self.param_dict["st"]

    def __str__(self):
        return "RL_meta_nost"

    def id_param_init(self, params, id):
        varp_list = ["a_pos", "a_neg", "beta", "zeta", "alpha_nu", "psi"]
        fixp_list = ["b0", "q_init", "gam"]
        var_dict = params.loc[params["ID"] == id, varp_list].iloc[0].to_dict()
        w0_arr = [self.fixed_params["q_init"]] * 2 + [0, 0, var_dict["a_neg"]]
        w0 = np.array(w0_arr)
        d = {"w0": w0}
        d.update({fp: self.fixed_params[fp] for fp in fixp_list})
        d.update(var_dict)
        return d

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
