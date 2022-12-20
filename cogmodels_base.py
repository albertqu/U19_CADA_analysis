import numpy as np
from scipy.special import expit
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

    def sim(self, data, *args, **kwargs):
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

    """

    def __init__(self):
        super().__init__()
        self.k_action = 2
        self.params = {}
        pass

    def calc_q(self, b, w):
        """
        Function to calculate action values given b, belief states, and w, the weights
        b = probability of choosing action 1,
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
                m[n,] = (1 - alpha) * m[n - 1,] + alpha * c.iat[n - 1]

        m[m == 0] = 0.001
        m[m == 1] = 0.999
        L = np.dot(c, np.log(m)) + np.dot((1 - c), np.log(1 - m))
        m = m[:, np.argmax(L)]
        data['logodds'] = np.log(m) - np.log(1 - m)
        print("alpha =", alpha[np.argmax(L)])

        return data


    def sim(self, data, beta=1, p_rew=0.75, p_eps=0, sw=0.02, st=1, gam=0, *args, **kwargs):
        """
        Simulates the model that matches the data

        params:
            data: pd.DataFrame
            ... params listed in class description

        Returns:
            data with new columns filled with latents listed in class description
        """

        N = data.shape[0]
        qdiff = np.zeros(N)
        rpe = np.zeros(N)

        c = data['Decision']
        sess = data['Session']
        subj = data['Subject']
        r = data['Reward']

        # replace later with generic function
        w0 = np.array([p_rew, p_eps])
        b0 = 0.5

        # add later with generic function
        # wt = np.zeros((N, len(w0.ravel())))
        # bdim = 1 if isinstance(b0, Number) else len(b0)
        # bt = np.zeros((N, bdim))
        b, w = b0, w0

        for n in range(N):
            # initializing latents
            if n == 0:
                b = b0
                w = w0
            elif subj.iat[n] != subj.iat[n-1]:
                b = b0
                w = w0
            elif sess.iat[n] != sess.iat[n - 1]:
                b = b0
                w = w0 * gam + (1-gam) * w
            ## Action value calculation
            qs = self.calc_q(b, w)
            # compute value difference
            qdiff[n] = qs[1] - qs[0]
            ## Model update
            rpe[n] = r.iat[n] - qs[c.iat[n]]
            # update w according to reward prediction error, uncomment later
            # f_b = np.array([1-b, b])
            # w[c.iat[n]] = w[c.iat[n]] + alpha * rpe[n] * f_b

            def projected_rew(r, c, z):
                if c == 1:
                    P = w[::-1]
                else:
                    P = w
                if r == 0:
                    P = 1 - P
                return P[int(z)]

            p1, p0 = projected_rew(r.iat[n], c.iat[n], 1), projected_rew(r.iat[n], c.iat[n], 0)
            b = ((1-sw) * b * p1 + sw * (1 - b) * p0) / (b * p1 + (1 - b) * (1 - p0))

        data['qdiff'] = qdiff
        data['rpe'] = rpe

        return data

    def get_proba(self, data):
        beta = self.params['beta']
        st = self.params['st']
        return np.sum(expit(data['qdiff'] * beta + data['logodds'] * st))

    def fit(self, data, *args, **kwargs):
        """
        Fit models to all subjects
        """
        pass


    # pseudo R-squared: http://courses.atlas.illinois.edu/fall2016/STAT/STAT200/RProgramming/LogisticRegression.html#:~:text=Null%20Model,-The%20simplest%20model&text=The%20fitted%20equation%20is%20ln,e%E2%88%923.36833)%3D0.0333.




