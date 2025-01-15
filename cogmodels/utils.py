import numpy as np

import statsmodels.api as sm

__author__ = "Albert Qü"
__version__ = "0.0.1"
__email__ = "albert_qu@berkeley.edu"
__status__ = "Dev"


def regularize(x, lb, ub, eps=1e-15):
    return np.clip(x, lb+eps, ub-eps)

class Task:

    def __init__(self):
        self.k_action = 0


class Probswitch_2ABT(Task):

    def __init__(self, **kwargs):
        super(Probswitch_2ABT, self).__init__()
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

        if 'blockLen_range' in kwargs:
            self.lb, self.ub = kwargs['blockLen_range']
        self.p_cor = kwargs['p_cor']
        self.p_inc = kwargs['p_inc']
        self.initialize()

    def initialize(self):
        self.blockN = 1
        self.bLen = np.random.randint(self.lb, self.ub+1)
        self.btrial = 0
        self.target = np.random.randint(2)
        self.trialN = 0
        self.nRew = 0

    def checkNewBlock(self):
        if self.nRew == self.bLen:
            self.blockN += 1
            self.bLen = np.random.randint(self.lb, self.ub + 1)
            self.btrial = 0
            self.target = 1-self.target
            self.nRew = 0

    def getOutcome(self, action):
        p = np.random.random()
        if action == self.target:
            reward = int(p < self.p_cor)
        else:
            reward = int(p<self.p_inc)
        self.trialN += 1
        self.btrial += 1
        if reward:
            self.nRew += 1
            self.checkNewBlock()
        return reward


def add_switch(data):
    data['Switch'] = 0
    data.loc[data['Decision'] != data['Decision'].shift(1), 'Switch'] = 1
    data.loc[data['Trial'] == 1, 'Switch'] = 0
    return data


def cogmodel_nll(model, data, params_df):
    params = params_df
    data_sim = model.sim(data, params)
    c_valid_sel = data_sim['Decision'] != -1
    p_s = model.get_proba(data_sim, params)[c_valid_sel].values

    c_vs = data_sim.loc[c_valid_sel, 'Decision'].values
    epsilon = 1e-15 # 0.001
    p_s = np.minimum(np.maximum(epsilon, p_s), 1-epsilon)
    # try out class_weights
    return -(c_vs @ np.log(p_s) + (1-c_vs) @ np.log(1-p_s))

