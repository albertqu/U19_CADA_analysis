from cogmodels.base import *
from cogmodels.bayesian_base import *
from cogmodels.bimodel import *
from cogmodels.lr import *
from cogmodels.pcmodel import *
from cogmodels.rl import *
from cogmodels.utils import *

__author__ = "Albert Qü"
__version__ = "0.0.1"
__email__ = "albert_qu@berkeley.edu"
__status__ = "Dev"


def load_model(desc):
    # Loads model from string description
    model_dict = {
        "RLCF": RLCF,
        "RL4p": RL_4p,
        "BI": BIModel,
        "BIfp": BIModel_fixp,
        "BIlog": BI_log,
        "BRL": BRL,
        "LR": LR,
        "PC": PCModel,
        "PCf": PCModel_fixpswgam,
        "PCBRL": PCBRL,
    }
    return model_dict[desc]
