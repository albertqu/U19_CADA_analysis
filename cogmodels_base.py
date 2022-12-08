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
    """





