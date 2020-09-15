# Utils
from behaviors import *
from peristimulus import *

# Data
from pandas.api.types import is_numeric_dtype

##################################################
####################### RL #######################
##################################################
def get_session_RL_features(method='BRL'):
    """
    :param method: SRL (standard RL), BRL (Belief State RL), Bayesian Model (Maria Eckstein),
        ITI RL (Maria Eckstein)
    :return:
    """
    pass


def get_RL_features(actions, outcomes, times, method='BRL'):
    """
    :param actions:
    :param outcomes:
    :param times:
    :param method: SRL (standard RL), BRL (Belief State RL)
    :return: RPE, V, Q (figure out model contrast)
    """
    pass


##################################################
############## Logistic Regressions ##############
##################################################
def get_session_logistic_regression(actions, outcomes, times):
    """
    :param actions:
    :param outcomes:
    :param times:
    :return:
    """
    model = None
    accuracy = {'train': 0, 'test': 0}
    return model, accuracy


##################################################
################# Neural Modeling ################
##################################################
def get_decision_tree_modeling(fp_sigs, fp_times, behavior_times, trial_feature, model='RandomForest'):
    """
    :param fp_sigs: by trial
    :param fp_times: uniform time window
    :param event_types: aligned windows
    :param trial_feature: continuous: RF/XG regression; discrete: RF/XG classification
    :return:

    """
    # regression_multi_models(models, Y, method='linear', N_iters=100, raw_features_names=None,
    # reg_params=None, feature_importance=True, confidence_level=0.95, show=True)
    # classifier_LD_multimodels(models, labels, LD_dim=None, N_iters=100, mode='true',
    #                           ignore_labels=None, clf_models='all', clf_params=None,
    #                           cluster_param=3, label_alias=None, show=True)
    time_window = np.arange(-2000, 2001, 50)

    pass


def get_session_decision_tree_modeling(folder, animal, session, event_types, trial_feature,
                                       model='RandomForest', zscore=True,
                                       base_method='robust', denoise=True):
    """
    :param fp_sigs: by trial
    :param fp_times: uniform time window
    :param event_types: aligned windows
    :param trial_feature: continuous: RF/XG regression; discrete: RF/XG classification
    :return:

    """
    # regression_multi_models(models, Y, method='linear', N_iters=100, raw_features_names=None,
    # reg_params=None, feature_importance=True, confidence_level=0.95, show=True)
    # classifier_LD_multimodels(models, labels, LD_dim=None, N_iters=100, mode='true',
    #                           ignore_labels=None, clf_models='all', clf_params=None,
    #                           cluster_param=3, label_alias=None, show=True)
    time_window = np.arange(-2000, 2001, 50)
    tags = ['DA', 'Ca']

    files = encode_to_filename(folder, animal, session)
    matfile, green, red, fp = files['behavior'], files['green'], files['red'], files['FP']
    # Load FP
    if fp is not None:
        with h5py.File(fp, 'r') as fp_hdf5:
            fp_sigs = [access_mat_with_path(fp_hdf5, f'{tags[i]}/dff/{base_method}')
                       for i in range(len(tags))]
            if zscore:
                fp_sigs = [(fp_sigs[i] - np.mean(fp_sigs[i])) / np.std(fp_sigs[i], ddof=1)
                           for i in range(len(fp_sigs))]
            fp_times = [access_mat_with_path(fp_hdf5, f'{tags[i]}/time') for i in
                        range(len(tags))]
    else:
        print(f"Warning {animal} {session} does not have photometry processed!")
        fp_times, fp_sigs, iso_times, iso_sigs = get_sources_from_csvs([green, red],
                                                                       tags=('DA', 'Ca'), show=False)

        fp_sigs = [raw_fluor_to_dff(fp_times[i], fp_sigs[i], iso_times[i], iso_sigs[i], base_method,
                                    zscore=zscore) for i in range(len(fp_sigs))]

    if denoise:
        L = len(fp_times)
        new_times, new_sigs = [None] * L, [None] * L
        for i in range(L):
            new_sigs[i], new_times[i] = denoise_quasi_uniform(fp_sigs[i], fp_times[i])
        fp_sigs, fp_times = new_sigs, new_times

    # TODO: for now just do plots for one session
    mat = h5py.File(matfile, 'r')
    # Get aligned signals to behaviors
    aligned = [[] for _ in range(len(fp_sigs))]
    for beh in event_types:
        behavior_times = get_behavior_times(mat, beh)
        nonan_sel = ~np.any(np.isnan(behavior_times), axis=0)
        behavior_times_nonan = behavior_times[:, nonan_sel]
        # TODO: ADD caps for multiple behavior time latencies
        for i in range(len(fp_sigs)):
            aligned[i].append(align_activities_with_event(fp_sigs[i], fp_times[i], behavior_times_nonan,
                                               time_window, False))

    ys = pd.DataFrame({trial_feature: get_trial_features(mat, trial_feature, True)[nonan_sel]})
    mat.close()

    for i in range(len(fp_sigs)):
        aligned[i] = np.hstack(aligned[i])
        raw_features = np.concatenate([[beh + f'{ts:.0f}ms' for ts in time_window] for beh in event_types])
        if is_numeric_dtype(ys[trial_feature]):
            models = {'raw': (None, aligned)}
            reg_results = regression_multi_models(models, ys, method=['XGBoost', 'RandomForests'],
                                    N_iters=100, raw_features_names=raw_features,
                                    reg_params=None, show=False)
        else:
            # TODO: clf add feature importances
            noNA = ys != 'NA'
            models = {'raw': (None, aligned[noNA])}
            clfs, confs = classifier_LD_multimodels(models, raw_features, LD_dim=None, N_iters=100,
                                                    mode='true', ignore_labels=None,
                                                    clf_models='all', clf_params=None,
                                                    cluster_param=3, label_alias=None, show=False)
            # plot feature importances
            # plot accuracy




def get_GLM_modeling(fp_sigs, features, model='RandomForest'):
    """ Model from chris code
    :param fp_sigs: by trials? (Oversample to 1ms (fastest behavior time signature))
    :param fp_times: uniform time window
    :param features: maybe a pd.Dataframe with trials information and pauses
    :return:
    """
    # build design matrix with trial feature, y with fp_sigs

    # build basis functions and history kernel

    # divide to train and test set

    # train model on training set

    # evaluate on test set
    pass


def get_session_GLM_modeling(fp_sigs, fp_times, matfile, trial_features, fr=1000, model='RandomForest'):
    """ Model from chris code
    :param fp_sigs: by trials? (Oversample to 1ms (fastest behavior time signature))
    :param fp_times: uniform time window
    :param trial_feature:
    :param fr: upsampling rate
    :return:
    """
    # upsample fp_sigs

    # convert behavior times vectors to feature vectors with trial information
    pass


class NeuroMat:
    """
    A class to take trial-structured neural recording and other multi-modal measurement and featurize
    them for modelling purpose.
    credit: The design is strongly influenced by https://github.com/pillowlab/neuroGLM
    """

    def __init__(self, unit, dt, id, n_trial, **kwargs):
        self.unit = unit
        self.dt = dt
        self.id = id
        self.N = n_trial
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    def add_variable(self, data, name, verbose, vtype, trial_inds=None):
        """
        :param data:
        :param name:
        :param verbose:
        :param vtype: 'trace', 'timing' (including spikes), 'value'
        :param trial_inds:
        :return:
        """
        # TODO: enable adding new trial on live later for online experiments
        # TODO: handle decretization
        if not hasattr(self, name):
            setattr(self, name, {'data': [None] * self.N,
                                 'verbose': verbose,
                                 'type': vtype})
        var_dict = getattr(self, name)
        if trial_inds is not None:
            trial_inds = np.arange(len(self.N))
        for i, ind in enumerate(trial_inds):
            var_dict['data'][ind] = data[i]

    def add_temporal_basis(self, varname, basis_type='rucosine', **kwargs):
        """
        # TODO: Add Covariate Boxcar
        :param varname:
        :param basis_type: ['rcosine', 'rbf', 'boxcar' (width 1 boxcar is delta)]
        :return:
        from statsmodels.tsa.tsatools import lagmat
        lagmat(np.arange(10), maxlag=2, trim='forward', original='in')
        """
        if basis_type == 'rucosine':
            defaults = {'start': 0,
                        'width': 200,
                        'interval': 50,
                        'end': 500}
            if kwargs is not None:
                defaults.update(kwargs)
            s, w, itv, e = defaults['start'], defaults['width'], defaults['interval'], defaults['end']
            # convolve with base to obtain shifted signal
            base = ((np.arange(s, w + self.dt, self.dt) - w / 2) * 2 * np.pi / w) / 2 + 0.5

        pass

    def get_all_variables(self):
        return [d for d in dir(self) if (not callable(getattr(self, d))) and (not d.startswith('__'))]



