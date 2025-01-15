from cogmodels.base import *
from cogmodels.bayesian_base import *
from cogmodels.bimodel import *
from cogmodels.lr import *
from cogmodels.pcmodel import *
from cogmodels.rl import *
from cogmodels.utils import *
from cogmodels.meta_rl import *
from cogmodels.wsls import *

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
        "BRLfp": BRL_fp,
        "BRLfwr": BRL_fwr,
        "BRLwr": BRL_wr,
        "BRLfw": BRL_fw,
        "BRLwrp": BRL_wrp,
        "LR": LR,
        "RFLR": RFLR,
        "PC": PCModel,
        "PCf": PCModel_fixpswgam,
        "PCBRL": PCBRL,
        "RL_meta": RL_Grossman,
        "RL_meta_nof": RL_Grossman_nof,
        "RLFQ3p": RL_Forgetting3p,
        "RLFQST": RL_FQST,
        "RL_metaP": RL_Grossman_prime,
        "PearceHall": Pearce_Hall,
        "WSLS": WSLS
    }
    return model_dict[desc]
