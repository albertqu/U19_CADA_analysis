from calendar import c
import numpy as np
import pandas as pd
import scipy
import patsy
from scipy.special import expit

from cogmodels.base import CogModel2ABT_BWQ, CogParam
from cogmodels.utils import Probswitch_2ABT


class WSLS(CogModel2ABT_BWQ):
    
    """
    Win-stay-lose-shift model can be derived as a RLCF model with high beta,
    and alpha=1. But here we get a barebone version
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.param_dict = {'eps': CogParam(scipy.stats.uniform(), lb=0, ub=1)}

    def id_param_init(self, params, id):
        varp_list = ['eps']
        fixp_list = ["b0", "w0", "gam"]
        return {"b0": 0.5, "gam": 1, "w0": 0, 
                "eps": params.loc[params["ID"] == id, 'eps'].values[0]}
    
    def __str__(self):
        return "WSLS"

    def calc_q(self, b, w):
        return np.array([0.0, b])
    
    def update_b(self, b, w, c_t, r_t, params_i):
        if r_t == 1:
            if c_t == 1:
                b= 1 - params_i["eps"]/2
            else:
                b= params_i["eps"]/2
        else:
            if c_t == 1:
                b= params_i["eps"]/2
            else:
                b= 1 - params_i["eps"]/2
        
        return b
    
    def update_w(self, b, w, c_t, rpe_t, params_i):
        return w
    
    def assign_latents(self, data, qdiff, rpe, b_arr, w_arr):
        data["b"] = b_arr
        data['rpe'] = rpe
        return data
    
    def get_proba(self, data, params=None):
        return data['b']
    
    def select_action(self, qdiff, m_1back, params):
        return int(np.random.random() <= qdiff)
    
    # def fit(self, data, *args, **kwargs):
    #     def nll():
    #         params = {}
    #         data_sim = self.sim(data, params, *args, **kwargs)
    #         # weight by class weights
    #         c_valid_sel = data_sim["Decision"] != -1
    #         p_s = self.get_proba(data_sim, params)[c_valid_sel].values

    #         c_vs = data_sim.loc[c_valid_sel, "Decision"].values
    #         epsilon = 1e-15  # 0.001
    #         p_s = np.minimum(np.maximum(epsilon, p_s), 1 - epsilon)
    #         # try out class_weights
    #         return -(c_vs @ np.log(p_s) + (1 - c_vs) @ np.log(1 - p_s))

    #     negloglik = nll()
    #     bic = 2 * negloglik
    #     aic = 2 * negloglik
    #     self.fitted_params = self.create_params(data)

    #     data_sim_opt = self.sim(data, self.fitted_params, *args, **kwargs)
    #     data["choice_p"] = self.get_proba(data_sim_opt)

    #     self.summary = {
    #         "bic": bic,
    #         "aic": aic,
    #         "latents": data[self.latent_names + ["choice_p"]].reset_index(drop=True),
    #     }

    #     return self
