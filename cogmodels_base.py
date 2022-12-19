import numpy as np
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

    params:
        .beta: policy compression parameter
        .p_rew: reward prob in correct choice
        .p_eps: reward prob in wrong choice
        .st: stickiness to previous choice
        .sw: switch probability

    latents:
        .b: belief vector, propagating and updated via bayesian inference
        .w: weights for belief vector
        .rpe: reward prediction error
        .Q: action values

    """

    def __init__(self):
        super().__init__()
        pass

    def sim(self, data, beta=1, p_rew=0.75, p_eps=0, sw=0.98, st=1, *args, **kwargs):
        """
        Simulates the model that matches the data

        params:
            data: pd.DataFrame
            ... params listed in class description

        Returns:
            data with new columns filled with latents listed in class description
        """

        

        pass


    def fit(self, data, *args, **kwargs):
        """
        Fit models to all subjects
        """




