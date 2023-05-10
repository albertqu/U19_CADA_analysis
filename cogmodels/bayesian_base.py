import numpy as np
from cogmodels.base import CogModel2ABT_BWQ


class BayesianModel(CogModel2ABT_BWQ):
    def __init__(self):
        super().__init__()

    @staticmethod
    def bayesian_step(b, ps, c_t, r_t, sw):
        # performs bayesian step to update b
        def projected_rew(r_i, c_i, z):
            if c_i == 1:
                P = ps[::-1]
            else:
                P = ps
            if r_i == 0:
                P = 1 - P
            return P[int(z)]

        p1, p0 = projected_rew(r_t, c_t, 1), projected_rew(r_t, c_t, 0)
        eps = 1e-15
        b = ((1 - sw) * b * p1 + sw * (1 - b) * p0) / (
            b * p1 + (1 - b) * p0
        )  # catch when b goes to 1
        b = np.maximum(np.minimum(b, 1 - eps), eps)
        return b
