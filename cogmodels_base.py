import numpy as np
import pandas as pd
from scipy.special import expit
import scipy
from numbers import Number

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

    def __init__(self):
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


class PCModel(CogModel):
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
        self.k_action = 2
        self.fixed_params = {"b0": 0.5,
                             "K_marginal": 50}
        self.param_dict = {"alpha": CogParam(scipy.stats.uniform()),
                           "beta": CogParam(scipy.stats.expon(1)),
                           "p_rew_init": CogParam(scipy.stats.beta(90, 30)),
                           "p_eps_init": CogParam(scipy.stats.beta(1, 30)),
                           "st": CogParam(scipy.stats.gamma(2, scale=0.2)),
                           "sw": CogParam(scipy.stats.uniform(loc=0, scale=0.05)),
                           "gam": CogParam(scipy.stats.uniform())}
        self.fitted_params = None
        self.summary = {}

    def calc_q(self, b, w):
        """
        Function to calculate action values given b, belief states, and w, the weights
        b = probability believing in state 1
        """
        f_b = np.array([1-b, b]).reshape((1, 2))
        W = np.diag([w[0]-w[1]] * len(w)) + w[1]
        return f_b @ W

    def calc_qdiff(self, b, w):
        """
        Function to calculate q value differences given b, w
        """
        return (2*b-1) * (w[0]-w[1])

    def fit_marginal(self, data, K=50):
        # Fit marginal choice probability
        #
        # INPUTS:
        #   data - dataframe containing all the relevant data
        #   K - number of different exponential weighting values
        #
        # OUTPUTS:
        #   data with new column 'logodds' (choice log odds)

        alpha = np.linspace(0.001, 0.3, num=K)
        N = data.shape[0]
        m = np.zeros((N, K)) + 0.5
        c = data['Decision']
        sess = data['Session']

        for n in range(N):
            if (n > 0) and (sess.iat[n] == sess.iat[n - 1]):
                if np.isnan(c.iat[n - 1]):
                    # Handling miss decisions
                    m[n,] = m[n - 1,]
                else:
                    m[n,] = (1 - alpha) * m[n - 1,] + alpha * c.iat[n - 1]

        m[m == 0] = 0.001
        m[m == 1] = 0.999
        c_vec = c[~c.isnull()].values
        m_mat = m[~c.isnull()]
        L = np.dot(c_vec, np.log(m_mat)) + np.dot((1 - c_vec), np.log(1 - m_mat))
        m = m[:, np.argmax(L)]
        data['logodds'] = np.log(m) - np.log(1 - m)
        data['marg'] = m
        print("alpha =", alpha[np.argmax(L)])

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
        qdiff = np.zeros(N)
        rpe = np.zeros(N)
        b_arr = np.zeros(N)
        w_arr = np.zeros((N, 2))

        c = data['Decision']
        # TODO: handle miss decisions
        sess = data['Session']
        subj = data['Subject']
        id_i = data['ID']
        r = data['Reward']

        # replace later with generic function
        plist = ['p_rew_init', 'p_eps_init', 'gam', 'sw', 'alpha']
        p_rew_init, p_eps_init, gam, sw, alpha = params.loc[params["ID"] == id_i.iat[0], plist].values.ravel()
        w0 = np.array([p_rew_init, p_eps_init])
        b0 = self.fixed_params['b0']

        # add later with generic function
        # wt = np.zeros((N, len(w0.ravel())))
        # bdim = 1 if isinstance(b0, Number) else len(b0)
        # bt = np.zeros((N, bdim))
        b, w = b0, w0

        # when ID changes: certain parameter changes

        for n in range(N):
            # initializing latents
            if id_i.iat[n] != id_i.iat[n-1]:
                p_rew_init, p_eps_init, gam, sw, alpha = params.loc[params["ID"] == id_i.iat[n], plist].values.ravel()
                w0 = np.array([p_rew_init, p_eps_init])

            if subj.iat[n] != subj.iat[n-1]:
                b = b0
                w = np.copy(w0)
            elif sess.iat[n] != sess.iat[n - 1]:
                b = b0
                w = w0 * gam + (1-gam) * w
            ## Action value calculation
            qs = self.calc_q(b, w).ravel()
            # compute value difference
            qdiff[n] = qs[1] - qs[0]
            ## Model update
            if c.iat[n] == -1:
                # handling miss trials
                rpe[n] = np.nan
                w_arr[n, :] = w
                b_arr[n] = b
                # w, b remains the same
            else:
                rpe[n] = r.iat[n] - qs[c.iat[n]]
                # w, b reflects information prior to reward
                w_arr[n, :] = w
                b_arr[n] = b

                # update w according to reward prediction error, uncomment later
                f_b = np.array([1-b, b])
                # change the update rule
                # w[c.iat[n]] = w[c.iat[n]] + alpha * rpe[n] * f_b
                dw = np.array([1-b, b]) # alternatively p2 = b+c-2bc
                if c.iat[n] == 1:
                    dw = np.array([b, 1-b])
                w += alpha * rpe[n] * dw
                w = np.minimum(np.maximum(w, 0), 1)

                def projected_rew(r_i, c_i, z):
                    if c_i == 1:
                        P = w[::-1]
                    else:
                        P = w
                    if r_i == 0:
                        P = 1 - P
                    return P[int(z)]

                p1, p0 = projected_rew(r.iat[n], c.iat[n], 1), projected_rew(r.iat[n], c.iat[n], 0)
                b = ((1-sw) * b * p1 + sw * (1 - b) * p0) / (b * p1 + (1 - b) * p0)

        data['qdiff'] = qdiff
        data['rpe'] = rpe
        data['b'] = b_arr
        data[['p_rew', 'p_eps']] = w_arr
        # qdiff[n], rpe[n], b, w, b_arr[n], w_arr[n]
        return data

    def get_proba(self, data, params=None):
        if 'loglik' in data.columns:
            return np.exp(data['loglik'].values)
        else:
            params_in = self.fitted_params[['ID', 'beta', 'st']] if params is None else params
            new_data = data.merge(params_in, on='ID')
            return expit(new_data['qdiff'] * new_data['beta'] + new_data['logodds'] * new_data['st'])

    def create_params(self, data):
        uniq_ids = data['ID'].unique()
        n_id = len(uniq_ids)
        new_params = {}
        for p in self.param_dict:
            self.param_dict[p].n_id = n_id
            new_params[p] = self.param_dict[p].eval()
        new_params['ID'] = uniq_ids
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

        # data <- fit_marginal
        # params <- create_params
        # params2x, x2params
        # x0 = params2x
        # nll(x, *data, *params): x2params(x), sim, get_prob, negsum
        # results = scipy.optimize.minimize(nll, x0)
        # self.fitted_params = x2params(results)

        data = self.fit_marginal(data, K=self.fixed_params['K_marginal'])
        params_df = self.create_params(data)
        id_list = params_df['ID']
        param_names = [c for c in params_df if c != "ID"]
        params2x = lambda params_df: params_df.drop(columns='ID').values.ravel(order='C')
        x2params = lambda x: pd.DataFrame(x.reshape((-1, len(self.param_dict)), order='C'),
                                          columns=param_names, index=id_list).reset_index().rename({'index': 'ID'})
        x0 = params2x(params_df)

        def nll(x):
            params = x2params(x)
            data_sim = self.sim(data, params, *args, **kwargs)
            return - np.sum(np.log(self.get_proba(data_sim, params)))

        res = scipy.optimize.minimize(nll, x0, method='BFGS', tol=1e-6)
        # constrain optimize
        self.fitted_params = x2params(res.x)
        negloglik = res.fun
        bic = len(res.x) * np.log(len(data)) + 2 * negloglik
        aic = 2 * len(res.x) + 2 * negloglik

        data_sim_opt = self.sim(data, self.fitted_params, *args, **kwargs)
        data['choice_p'] = self.get_proba(data_sim_opt)

        latent_names = ['qdiff', 'rpe', 'p_rew', 'p_eps', 'b', 'choice_p']
        self.summary = {'bic': bic, 'aic': aic,
                        'latents': data[latent_names].reset_index(drop=True)}
        return self

    # pseudo R-squared: http://courses.atlas.illinois.edu/fall2016/STAT/STAT200/RProgramming/LogisticRegression.html#:~:text=Null%20Model,-The%20simplest%20model&text=The%20fitted%20equation%20is%20ln,e%E2%88%923.36833)%3D0.0333.


class CogParam:

    def __init__(self, value=None, fixed_init=False, n_id=1):
        self.value = value
        self.fixed_init = fixed_init
        self.n_id = n_id

    def eval(self):
        if isinstance(self.value, Number):
            return np.full(self.n_id, self.value)
        elif hasattr(self.value, "rvs"):
            return self.value.rvs(size=self.n_id)
        else:
            return ValueError()



