# Utils
from behaviors import *
from peristimulus import *
from utils_models import *

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


def output_data_for_ITI_DA_RL_model():
    root = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/"
    save_folder = os.path.join(root, 'maria_model')
    folder = os.path.join(root, "ProbSwitch_FP_data")
    event_types = ['center_in', 'center_out', 'side_out']
    #event_types = ['center_in', 'center_out', 'outcome', 'side_out']
    zscore = True
    base_method = 'robust_fast'
    denoise = True
    smooth = 0
    time_window_dict = {'center_in': np.arange(-500, 501, 50),
                        'center_out': np.arange(-500, 501, 50),
                        'outcome': np.arange(-500, 2001, 50),
                        'side_out': np.arange(-500, 1001, 50)}

    choices = {'A2A': {'A2A-15B-B_RT': ["p153_FP_LH", "p238_FP_LH"],
                       'A2A-19B_RT': ['p139_FP_LH', 'p148_FP_LH'],
                       'A2A-19B_RV': ['p142_FP_RH', 'p156_FP_LH']},
               "D1": {"D1-27H_LT": ["p103_FP_RH", "p189_FP_RH"],
                      "D1-28B_LT": ["p135_session2_FP_LH"]}}



    # hue: ITI, row,  col: laterality
    for group in ['D1', 'A2A']:
        neur_type = group if group == 'D1' else 'D2'
        sessions = choices[group]
        for animal in sessions:
            for session in sessions[animal]:

                tags = ['DA', 'Ca']

                files = encode_to_filename(folder, animal, session)
                matfile, green, red, fp = files['behavior'], files['green'], files['red'], files['FP']
                # Load FP
                if fp is not None:
                    with h5py.File(fp, 'r') as fp_hdf5:
                        fp_sigs = [access_mat_with_path(fp_hdf5, f'{tags[i]}/dff/{base_method}')
                                   for i in range(len(tags))]
                        fp_times = [access_mat_with_path(fp_hdf5, f'{tags[i]}/time') for i in
                                    range(len(tags))]
                else:
                    print(f"Warning {animal} {session} does not have photometry processed!")
                    fp_times, fp_sigs, iso_times, iso_sigs = get_sources_from_csvs([green, red],
                                                                                   tags=('DA', 'Ca'),
                                                                                   show=False)

                    fp_sigs = [
                        raw_fluor_to_dff(fp_times[i], fp_sigs[i], iso_times[i], iso_sigs[i], base_method,
                                         zscore=False) for i in range(len(fp_sigs))]
                if denoise:
                    L = len(fp_times)
                    new_times, new_sigs = [None] * L, [None] * L
                    for i in range(L):
                        new_sigs[i], new_times[i] = denoise_quasi_uniform(fp_sigs[i], fp_times[i])
                    fp_sigs, fp_times = new_sigs, new_times
                    # TODO: for now just do plots for one session
                if zscore:
                    fp_sigs = [(fp_sigs[i] - np.mean(fp_sigs[i])) / np.std(fp_sigs[i], ddof=1)
                               for i in range(len(fp_sigs))]

                mat = h5py.File(matfile, 'r')
                # Get aligned signals to behaviors
                N = get_trial_num(mat)
                aligned = [np.full((len(event_types), N), np.nan) for _ in range(len(fp_sigs))]
                for ib, beh in enumerate(event_types):
                    ibtimes = get_behavior_times(mat, beh).ravel()
                    nonan_sel = ~np.isnan(ibtimes)
                    behavior_times_nonan = ibtimes[nonan_sel]
                    for i in range(len(fp_sigs)):
                        align_i = align_activities_with_event(fp_sigs[i], fp_times[i], behavior_times_nonan,
                                                        time_window_dict[beh.split('{')[0]], False)
                        if smooth > 0:
                            aligned[i][ib, nonan_sel] = np.mean(align_i[:, 10-smooth:11+smooth], axis=1)
                        else:
                            aligned[i][ib, nonan_sel] = align_i[:, 11]
                itis = get_trial_features(mat, 'ITI_raw', True)
                rewards_str = get_trial_features(mat, 'R', True)
                rewards = vectorize_with_map(rewards_str, {'Rewarded': 1, 'Unrewarded': 0})
                lats_str = get_trial_features(mat, 'A', True)
                lats = vectorize_with_map(lats_str, {'contra': 1, 'ipsi': 0})
                mat.close()
                datamat = np.vstack([np.arange(1, N+1), lats, rewards, itis, np.vstack(aligned)])
                columns = np.concatenate([['trial', 'A_contra', 'R', 'ITI'],
                                         np.concatenate([[f"{tags[i]}_{ev}" for ev in event_types]
                                                         for i in range(len(tags))])])
                subf = os.path.join(save_folder, neur_type, animal)
                if not os.path.exists(subf):
                    os.makedirs(subf)
                fname = os.path.join(subf, f"{animal}_{session}_RL_DA_smooth{(smooth*2+1)}.csv")
                pdf = pd.DataFrame(datamat.T, columns=columns)
                pdf.to_csv(fname, index=False)




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


def get_session_decision_tree_modeling(folder, animal, session, event_types, trial_feature, zscore=True,
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
    #time_window = np.arange(-2000, 2001, 50)
    time_window_dict = {'center_in': np.arange(-500, 501, 50),
                        'center_out': np.arange(-500, 501, 50),
                        'outcome': np.arange(-500, 2001, 50),
                        'side_out': np.arange(-500, 1001, 50)}
    tags = ['DA', 'Ca']
    fit_models = ['RandomForests']

    files = encode_to_filename(folder, animal, session)
    matfile, green, red, fp = files['behavior'], files['green'], files['red'], files['FP']
    # Load FP
    if fp is not None:
        with h5py.File(fp, 'r') as fp_hdf5:
            fp_sigs = [access_mat_with_path(fp_hdf5, f'{tags[i]}/dff/{base_method}')
                       for i in range(len(tags))]

            fp_times = [access_mat_with_path(fp_hdf5, f'{tags[i]}/time') for i in
                        range(len(tags))]
    else:
        print(f"Warning {animal} {session} does not have photometry processed!")
        fp_times, fp_sigs, iso_times, iso_sigs = get_sources_from_csvs([green, red],
                                                                       tags=('DA', 'Ca'), show=False)

        fp_sigs = [raw_fluor_to_dff(fp_times[i], fp_sigs[i], iso_times[i], iso_sigs[i], base_method,
                                    zscore=False) for i in range(len(fp_sigs))]
    if denoise:
        L = len(fp_times)
        new_times, new_sigs = [None] * L, [None] * L
        for i in range(L):
            new_sigs[i], new_times[i] = denoise_quasi_uniform(fp_sigs[i], fp_times[i])
        fp_sigs, fp_times = new_sigs, new_times
        # TODO: for now just do plots for one session

    if zscore:
        fp_sigs = [(fp_sigs[i] - np.mean(fp_sigs[i])) / np.std(fp_sigs[i], ddof=1)
                   for i in range(len(fp_sigs))]

    mat = h5py.File(matfile, 'r')
    # Get aligned signals to behaviors
    aligned = [[] for _ in range(len(fp_sigs))]
    behavior_times = np.vstack([get_behavior_times(mat, beh) for beh in event_types])
    nonan_sel = ~np.any(np.isnan(behavior_times), axis=0)
    behavior_times_nonan = behavior_times[:, nonan_sel]
    for ib, beh in enumerate(event_types):
        # TODO: ADD caps for multiple behavior time latencies
        for i in range(len(fp_sigs)):
            aligned[i].append(align_activities_with_event(fp_sigs[i], fp_times[i], behavior_times_nonan[ib],
                                                          time_window_dict[beh.split('{')[0]], False))

    ys = pd.DataFrame({trial_feature: get_trial_features(mat, trial_feature, True)[nonan_sel]})
    mat.close()
    for i in range(len(fp_sigs)):
        aligned[i] = np.hstack(aligned[i])

    regrs, clfrs = [], []
    for i in range(len(fp_sigs)):
        raw_features = np.concatenate(
            [[beh + f'{ts:.0f}ms' for ts in time_window_dict[beh.split('{')[0]]] for beh in event_types])
        if is_numeric_dtype(ys[trial_feature]):
            models = {'raw': (None, aligned[i])}
            reg_results = regression_multi_models(models, ys, method=fit_models,
                                                  N_iters=3, raw_features_names=raw_features,
                                                  reg_params=None, show=True)
            regrs.append(reg_results)
            # plot feature importances
            for md in fit_models:
                visualize_feature_importance(regrs[i]['raw'][md]['f_importance'], raw_features,
                                             tag=f'{tags[i]}_{md}_{trial_feature}')
        else:
            # TODO: clf add feature importances
            noNA = (ys != '').values.ravel()
            models = {'raw': (None, aligned[i][noNA])}
            clfs, confs = classifier_LD_multimodels(models, ys[trial_feature][noNA], LD_dim=None,
                                                    N_iters=100,
                                                    mode='true', ignore_labels=None,
                                                    clf_models=fit_models, clf_params=None,
                                                    cluster_param=3, label_alias=None, show=True)
            clfrs.append((clfs, confs))

            # plot feature importances
            for md in fit_models:
                visualize_feature_importance(clfrs[i][0]['raw'][md]['f_importance'], raw_features,
                                             tag=f'{tags[i]}_{md}_{trial_feature}')

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



