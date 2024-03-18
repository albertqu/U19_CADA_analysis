import numpy as np
from scipy.special import expit
import scipy
from cogmodels.bayesian_base import BayesianModel
from cogmodels.base import CogParam


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
        fixp_list = ["b0", "a_marg"]
        p_rew_init, p_eps_init, gam, sw, alpha, beta, st = params.loc[
            params["ID"] == id, varp_list
        ].values.ravel()
        b0, a_marg = [self.fixed_params[fp] for fp in fixp_list]
        w0 = np.array([p_rew_init, p_eps_init])
        return {
            "b0": b0,
            "w0": w0,
            "gam": gam,
            "sw": sw,
            "alpha": alpha,
            "a_marg": a_marg,
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
        if self.fixed_params["sim_marg"]:
            return data
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
        data["m"] = m
        alpha_opt = alpha[np.argmax(L)]
        print("alpha =", alpha_opt)
        self.fixed_params["a_marg"] = alpha_opt
        return data

    def get_proba(self, data, params=None):
        if "loglik" in data.columns:
            return np.exp(data["loglik"].values)
        else:
            if "beta" in self.fixed_params:  # change emulate function
                params_in = (
                    self.fitted_params[["ID", "st"]].copy() if params is None else params
                )
                params_in["beta"] = self.fixed_params["beta"]
            else:
                params_in = (
                    self.fitted_params[["ID", "beta", "st"]]
                    if params is None
                    else params
                )

            new_data = data.merge(params_in, how="left", on="ID")
            ms = new_data["m"].values
            logodd = np.log(ms) - np.log(1 - ms)
            return expit(new_data["qdiff"] * new_data["beta"] + logodd * new_data["st"])

    def select_action(self, qdiff, m_1back, params):
        logodd = np.log(m_1back) - np.log(1 - m_1back)
        choice_p = expit(qdiff * params["beta"] + logodd * params["st"])
        return int(np.random.random() <= choice_p)

    def marginal_init(self):
        return 0.5

    def update_m(self, c_t, m, params):
        a_marg = self.fixed_params["a_marg"]
        if c_t == -1:
            return m
        m = (1 - a_marg) * m + a_marg * c_t
        eps = 1e-4
        m = np.minimum(np.maximum(eps, m), 1 - eps)
        return m

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
        fixp_list = ["b0", "p_rew_init", "p_eps_init", "gam", "sw", "alpha", "a_marg"]
        beta, st = params.loc[params["ID"] == id, varp_list].values.ravel()
        b0, p_rew_init, p_eps_init, gam, sw, alpha, a_marg = [
            self.fixed_params[fp] for fp in fixp_list
        ]
        w0 = np.array([p_rew_init, p_eps_init])
        return {
            "b0": b0,
            "w0": w0,
            "gam": gam,
            "sw": sw,
            "alpha": alpha,
            "a_marg": a_marg,
            "beta": beta,
            "st": st,
        }

    def update_w(self, b, w, c_t, rpe_t, params_i):
        # No update!
        # v8: update
        # v7: no update
        # <= v8: gam = 0
        return w


class PCBRL(PCModel):

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
                "sim_marg": True,
            }
        )
        # used fixed for hyperparam_tuning
        self.param_dict = {
            "beta": CogParam(scipy.stats.expon(1), lb=0),
            "st": CogParam(scipy.stats.gamma(2, scale=0.2), lb=0),
            "a_marg": CogParam(scipy.stats.uniform(), lb=0, ub=1),
        }

    def __str__(self):
        return "PCBRL"

    def id_param_init(self, params, id):
        varp_list = ["beta", "st", "a_marg"]
        fixp_list = ["b0", "p_rew_init", "p_eps_init", "gam", "sw", "alpha"]
        beta, st, a_marg = params.loc[params["ID"] == id, varp_list].values.ravel()
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
            "a_marg": a_marg,
        }

    def update_w(self, b, w, c_t, rpe_t, params_i):
        # No update!
        # v8: update
        # v7: no update
        # <= v8: gam = 0
        return w

    def update_m(self, c_t, m, params):
        if c_t == -1:
            return m
        a_marg = params["a_marg"]
        m = (1 - a_marg) * m + a_marg * c_t
        eps = 1e-4
        m = np.minimum(np.maximum(eps, m), 1 - eps)
        return m
