from neuro_series import *
from nb_viz import *
from peristimulus import *
from abc import abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import logging
logging.basicConfig(level=logging.INFO)
#sns.set_context("talk")
RAND_STATE = 230


##################################################
##################### NBMat ######################
##################################################
class NeuroBehaviorMat:
    # always be careful of applying lagging operations or dim reductions and check if:
    # 1. operations are only carried over rows with the same (animal, session) tuple
    # 2. df is in wide format, where data is organized trial by trials

    behavior_events = []
    event_features = {}
    trial_features = []
    id_vars = []
    time_multiplier = 1

    # example id_vars: animal, session, trial,
    # make a state representation?
    def __init__(self, neural=True):
        self.neural = neural
        # self.id_vars = id_vars
        self.event_time_windows = {}
        self.nb_cols = None
        self.nb_lag_cols = None

    def series_time(self, event, t):
        ev_neur = lambda ev: ev + '_neur'
        return f'{ev_neur(event)}|{t * self.time_multiplier :.2f}'

    def extend_features(self, nb_df, *args, **kwargs):
        return nb_df

    def merge_rois_wide(self, df, pivot_col):
        # nb_df must be of wide form: meaning each trial must only have 1
        id_cols = self.id_vars
        ind_ics = list(np.setdiff1d(id_cols, [pivot_col]))
        return self.apply_to_idgroups(df, self.merge_rois_wide_ID, id_vars=ind_ics, pivot_col=pivot_col)

    def merge_rois_wide_ID(self, df, pivot_col):
        id_cols = self.id_vars

        ind_ics = list(np.setdiff1d(id_cols, [pivot_col]))
        nb_cols = [c for c in df.columns if ('_neur' in c) and ('|' in c)]
        pivot_values = df[pivot_col].unique()
        # print(ind_ics, pivot_values)

        result_df = df.loc[df[pivot_col] == pivot_values[0]].drop(columns=pivot_col)
        result_df.rename(columns={nbc: f'{pivot_values[0]}--{nbc}' for nbc in nb_cols}, inplace=True)
        for i, pv in enumerate(pivot_values):
            slice_df = df.loc[df[pivot_col] == pv, ind_ics + nb_cols].reset_index(drop=True)
            slice_df.rename(columns={nbc: f'{pv}--{nbc}' for nbc in nb_cols}, inplace=True)
            result_df = result_df.merge(slice_df, how='left', on=ind_ics)

    def align_B2N_dff_ID(self, behavior_df, neur_df, events, form='wide'):
        # align behavior to neural on a single session basis, DO NOT use it across multiple sessions
        # event_windows: dictionary with event to window series
        # Now default to aligning to dff, assumes that df is already in tidy format,
        # for raw fluorescence should first melt then compare

        """
        by default takes in neur_df
        Now method only supports alignment for single animal session, need to develop ID system
        Returns:
            aligned_df: dataframe with behavior features
            nb_cols: column names of the new aligned cols
        """
        id_set = np.setdiff1d(self.id_vars, ['roi'])
        assert np.all([len(np.unique(behavior_df[v])) == 1 for v in id_set]), 'not unique'
        logging.info('Using jove zdff method for dff calculation by default')
        nb_dfs = []
        dff_cname = 'ZdFF'
        result_dfs = []
        ev_neur = lambda ev: ev + '_neur'
        nb_cols = {
        ev_neur(ev): [f'{ev_neur(ev)}|{ts * self.time_multiplier :.2f}' for ts in self.event_time_windows[ev]]
        for ev in events}
        for roi in np.unique(neur_df['roi']):
            ev_nb_df = behavior_df.copy()
            roi_sig = neur_df.loc[neur_df['roi'] == roi, dff_cname].values
            roi_time = neur_df.loc[neur_df['roi'] == roi, 'time'].values
            for event in events:
                if event == 'side_out':
                    event = 'first_side_out'
                assert event in self.behavior_events, f"unknown event {event}"
                ev_sel = ~np.isnan(behavior_df[event]) # TODO: maybe use pd.isnull instead
                aROI = align_activities_with_event(roi_sig, roi_time,
                                                   behavior_df.loc[ev_sel, event].values,
                                                   self.event_time_windows[event])
                ev_nb_df.loc[ev_sel, 'roi'] = roi
                ev_nb_df.loc[ev_sel, nb_cols[ev_neur(event)]] = aROI

            nb_dfs.append(ev_nb_df)
        nb_df = pd.concat(nb_dfs, axis=0)
        if form == 'long':
            all_evneur_cols = np.concatenate(list(nb_cols.values()))
            id_cols = list(np.setdiff1d(nb_df.columns, all_evneur_cols))
            # recursively reduce event column groups
            for event in events:
                ecols = nb_cols[ev_neur(event)]
                nb_df = pd.melt(nb_df, id_vars=id_cols, value_vars=ecols,
                                var_name=f'{ev_neur(event)}_time', value_name=f'{ev_neur(event)}_{dff_cname}')
                nb_df[f'{ev_neur(event)}_time'] = nb_df[f'{ev_neur(event)}_time'].str.extract(
                    f'{ev_neur(event)}' + '\|(?P<ts>-?(\d|\.)+)').ts
                id_cols = id_cols + [f'{ev_neur(event)}_time', f'{ev_neur(event)}_{dff_cname}']
                nb_df[f'{ev_neur(event)}_time'] = nb_df[f'{ev_neur(event)}_time'].astype(float)
        self.nb_cols = nb_cols
        return nb_df

    def lag_wide_df_ID(self, nb_df, features):
        # lagging option differs in behavior for trial/event features and neural features
        # features = {feature: {'long': T/F (default F), 'pre': int (default 0), 'post': int (default 0)}}
        # TODO: debug
        assert np.all([len(np.unique(nb_df[v])) == 1 for v in self.id_vars]), 'not unique'
        assert len(nb_df.trial) == len(np.unique(nb_df.trial)), 'suspect data is not in wide form'
        for feat in features:
            feat_opt = features[feat]
            flex_t = lambda i, mode: '{t-%d}' % i if mode == 'pre' else '{t+%d}' % i

            def te_colf(fts, lag, mode):
                if len(fts) > 1:
                    logging.warning("unexpected")
                ft = fts[0]
                assert mode in ['pre', 'post'], f'Bad Mode {mode}'
                return [ft + flex_t(i, mode) for i in range(1, lag + 1)]

            def neur_colf(fts, lag, mode):
                assert mode in ['pre', 'post'], f'Bad Mode {mode}'
                sif = lambda x, i, mode: x.split('|')[0] + flex_t(i, mode) + '|%s' % x.split('|')[1]
                return np.concatenate([[sif(ft, i, mode) for ft in fts] for i in range(1, lag + 1)])

            if '_neur' in feat:
                feat_ev = '_'.join(feat.split('_')[:-1])
                assert feat in self.nb_cols, f'event selected {feat_ev} not aligned yet'
                cols_to_shifts = self.nb_cols[feat]
                colf = neur_colf
            else:
                if not ((feat in self.trial_features + self.behavior_events) or (feat in self.event_features)):
                    print(f'Lagging derived feature {feat}, can lead to unexpected behavior')
                    assert feat in nb_df.columns, f'unknown option {feat}'
                cols_to_shifts = [feat]
                colf = te_colf
            values_to_shifts = nb_df[cols_to_shifts]
            shifted = []
            shifted_cols = []
            # add method for interval shifting if needed
            if 'pre' in feat_opt and feat_opt['pre']:
                # lagmat pads with 0 use df.shift instead
                shifted.append(
                    np.concatenate([values_to_shifts.shift(i) for i in range(1, feat_opt['pre'] + 1)],
                                   axis=1))
                # add shifted_cols
                shifted_cols.append(colf(cols_to_shifts, feat_opt['pre'], 'pre'))

            if 'post' in feat_opt and feat_opt['post']:
                shifted.append(
                    np.concatenate([values_to_shifts.shift(-i) for i in range(1, feat_opt['post'] + 1)],
                                   axis=1))
                shifted_cols.append(colf(cols_to_shifts, feat_opt['post'], 'post'))
            if shifted:
                assert shifted, 'didnt do anything?'
                shifted_cols = list(np.concatenate(shifted_cols))
                nb_df[shifted_cols] = np.concatenate(shifted, axis=1)
            if 'long' in feat_opt and feat_opt['long']:
                features[feat]['to_melt'] = cols_to_shifts + shifted_cols
        for feat in features:
            # check melt: TODO: currently assume all event_neur are lagged
            # update lagged cols
            if 'long' in features[feat] and features[feat]['long']:
                melt_cols = features[feat]['to_melt']
                id_cols = np.setdiff1d(nb_df.columns, melt_cols)
                # assert '_neur' in feat, 'currently only support aligned neural'
                if '_neur' in feat:
                    # TODO: flexible _ZdFF treatment
                    nb_df = nb_df.melt(id_vars=id_cols, value_vars=melt_cols, var_name=f'{feat}_arg',
                                       value_name=f'{feat}_ZdFF')
                    nb_df[[f'{feat}_lag', f'{feat}_time']] = nb_df[f'{feat}_arg'].str.replace(feat, '').str.split(
                        r'|', expand=True)
                    nb_df[f'{feat}_time'] = nb_df[f'{feat}_time'].astype(float)
                else:
                    nb_df = nb_df.melt(id_vars=id_cols, value_vars=melt_cols, var_name=f'{feat}_arg',
                                       value_name=f'{feat}_value')
                    sample_val = nb_df.loc[0, f'{feat}_arg']
                    if '|' in sample_val:
                        nb_df[[f'{feat}_lag', f'{feat}_time']] = nb_df[f'{feat}_arg'].str.replace(feat, '').str.split(
                            r'|', expand=True)
                        nb_df[f'{feat}_time'] = nb_df[f'{feat}_time'].astype(float)
                    else:
                        nb_df[f'{feat}_lag'] = nb_df[f'{feat}_arg'].str.replace(feat, '')
                nb_df[f'{feat}_lag'] = nb_df[f'{feat}_lag'].apply(lambda x: x[1:-1].replace('t', ''))
                nb_df.loc[nb_df[f'{feat}_lag'] == '', f'{feat}_lag'] = 0
                nb_df[f'{feat}_lag'] = nb_df[f'{feat}_lag'].astype(int)
                nb_df.drop(columns=f'{feat}_arg', inplace=True)
                uniq_lag = np.unique(nb_df[f'{feat}_lag'])
                if (len(uniq_lag) == 1) and (uniq_lag[0] == 0):
                    nb_df.drop(columns=f'{feat}_lag', inplace=True)
        return nb_df

    def lag_wide_df(self, nb_df, features):
        # nb_df must be of wide form: meaning each trial must only have 1
        return self.apply_to_idgroups(nb_df, self.lag_wide_df_ID, features=features)

    def apply_to_idgroups(self, nb_df, func, id_vars=None, *args, **kwargs):
        """ func: must takes in nb_df,
        *args, **kwargs: additional argument for func
        """
        #
        if id_vars is None:
            id_vars = self.id_vars
        # id_groups = [np.unique(nb_df[v]) for v in id_vars]
        all_dfs = []
        # better runtime with
        nb_df['combined'] = nb_df[id_vars].values.tolist()
        nb_df['combined'] = nb_df['combined'].str.join(',')
        for id_group in np.unique(nb_df['combined'].dropna()):
            id_group_vars = id_group.split(',')
            uniq_sel = np.logical_and.reduce(
                [nb_df[idv] == id_group_vars[i] for i, idv in enumerate(id_vars)])
            all_dfs.append(func(nb_df[uniq_sel].reset_index(drop=True), *args, **kwargs))
        nb_df.drop(columns='combined', inplace=True)
        return pd.concat(all_dfs, axis=0)


class PS_NBMat(NeuroBehaviorMat):
    behavior_events = ['center_in', 'center_out', 'side_in', 'outcome',
                       'zeroth_side_out', 'first_side_out', 'last_side_out']

    event_features = {'action': ['side_in', 'outcome',
                                 'zeroth_side_out', 'first_side_out'],
                      'last_side_out_side': ['last_side_out']}

    trial_features = ['explore_complex', 'struct_complex', 'state', 'rewarded',
                      'trial_in_block', 'block_num', 'quality']

    id_vars = ['animal', 'session', 'roi']

    def __init__(self, neural=True):
        super().__init__(neural)
        self.event_time_windows = {'center_in': np.arange(-1, 1.001, 0.05),
                                   'center_out': np.arange(-1, 1.001, 0.05),
                                   'outcome': np.arange(-0.5, 2.001, 0.05),
                                   'side_in': np.arange(-0.5, 2.001, 0.05),
                                   'zeroth_side_out': np.arange(-0.5, 1.001, 0.05),
                                   'first_side_out': np.arange(-0.5, 1.001, 0.05),
                                   'last_side_out': np.arange(-0.5, 1.001, 0.05)}

    def get_perc_trial_in_block(self, nb_df):
        # TODO: shift this function to behaviorMat
        def ptib(nb_df):
            nb_df['perc_TIB'] = 0
            for ibn in np.unique(nb_df['block_num']):
                bn_sel = nb_df['block_num'] == ibn
                mTIB = np.max(nb_df.loc[bn_sel, 'trial_in_block'].values)
                nb_df.loc[bn_sel, 'perc_TIB'] = nb_df.loc[bn_sel, 'trial_in_block'].values / mTIB
            return nb_df

        return self.apply_to_idgroups(nb_df, ptib, id_vars=['animal', 'session'])

    def extend_features(self, nb_df, *args, **kwargs):
        nb_df = self.get_perc_trial_in_block(nb_df)
        nb_df['pTIB_Q'] = pd.cut(nb_df['perc_TIB'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        return nb_df


class RR_NBMat(NeuroBehaviorMat):

    fields = ['tone_onset', 'T_Entry', 'choice', 'outcome',
              'quit', 'collection', 'trial_end', 'exit']
    # Add capacity for only behavior
    behavior_events = RRBehaviorMat.fields
    # tone_onset -> T_Entry -> choice (-> {0: quit,
    #                                      1: outcome (-> collection)}) -> trial_end

    event_features = {'accept': ['tone_onset', 'T_entry', 'choice', 'trial_end', 'exit'],
                      'reward': ['outcome', 'trial_end', 'exit']}

    trial_features = ['tone_prob', 'restaurant', 'lapIndex', 'blockIndex']

    id_vars = ['animal', 'session', 'roi']

    def __init__(self, neural=True):
        super().__init__(neural)
        if not neural:
            self.id_vars = ['animal', 'session']
        self.event_time_windows = {'tone_onset': np.arange(-1, 1.001, 0.05),
                                   'T_Entry': np.arange(-1, 1.001, 0.05),
                                   'choice': np.arange(-1, 1.001, 0.05),
                                   'outcome': np.arange(-1, 1.001, 0.05),
                                   'quit': np.arange(-1, 1.001, 0.05),
                                   'collection': np.arange(-1, 1.001, 0.05),
                                   'trial_end': np.arange(-1, 1.001, 0.05),
                                   'exit': np.arange(-1, 1.001, 0.05)}

    def fit_action_value_function(self, df):
        # endogs
        exog = 'accept'
        to_convert = ['restaurant']
        X = pd.concat([pd.get_dummies(df['restaurant']), df['tone_prob']], axis=1)
        reg_df = pd.concat([X, df[to_convert]], axis=1)
        y = df[exog].values
        # Use held out dataset to evaluate score
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RAND_STATE)
        # clf = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train, y_train)
        clf = LogisticRegression(random_state=RAND_STATE).fit(X_train, y_train)
        cv_score = clf.score(X_test, y_test)
        # use full dataset to calculate action logits
        clf_psy = clf.fit(X, y)
        base_df = reg_df.drop_duplicates().reset_index(drop=True)
        X_base = base_df.drop(columns=to_convert)
        logits = X_base.values @ clf_psy.coef_.T + clf_psy.intercept_
        base_df['action_logit'] = logits

        def func(endog_df):
            return endog_df.merge(base_df, how='left', on=list(endog_df.columns))['action_logit']

        return {'score': cv_score, 'func': func, 'name': 'action_logit', 'debug': base_df}

    def add_action_value_feature(self, df, endog_map):
        endog = ['restaurant', 'tone_prob']
        df[endog_map['name']] = endog_map['func'](df[endog])
        return df


#########################################################
##################### NBExperiment ######################
#########################################################
class NBExperiment:
    info_name = None
    spec_name = None

    def __init__(self, folder=''):
        self.meta = None
        self.nbm = NeuroBehaviorMat()
        self.nbviz = NBVisualizer(self)
        self.plot_path = folder

    def meta_safe_drop(self, nb_df, inplace=False):
        subcols = list(np.setdiff1d(nb_df.columns, self.meta.columns))
        return nb_df.dropna(subset=subcols, inplace=inplace)

    @abstractmethod
    def load_animal_session(self, animal_arg, session):
        return NotImplemented

    @abstractmethod
    def encode_to_filename(self, animal, session, ftypes="processed_all"):
        return NotImplemented

    def align_lagged_view(self, proj, events, laglist=None, **kwargs):
        """
        Given the project code specified in `proj`, look through metadata loaded through `self.info_name`
        and align each session's neural data to `events`. If specified, lag the neural data according to
        `laglist`.
        :param proj: 3 letter project code: e.g. 'UUM', 'RRM', 'BSD', 'JUV', 'FIP'
        :param events: list, list of events to be aligned to,
                            results stored in wideform columns f'{ event}_neur|...'
        :param laglist: Dict[str, Dict]
                            dictionary specifying all column features that need to be trial lagged.
                            Each column is mapped to an additional dictionary:
                             {'pre': [lags backward in time],
                              'post': ['lags forward in time'],
                              'long': [boolean for whether pivoting the df to a long-form df]}
        :param kwargs: provide additional conditions on what sessions to include through selecting certain
        variable value in metadata
        :return: all_nb_df: pd.DataFrame
                    dataframe containing neural and behavior data
        E.g.
        ```{python}
        pse_rrm = PS_Expr(data_root)
        laglist = {'action': {'pre': 2, 'post': 3},
                   'rewarded': {'pre': 2, 'post': 3},
                   f'outcome_neur': {'pre': 2, 'post': 3, 'long': True}
        laglist.update({ev: {'pre': 1, 'post': 1} for ev in pse_rrm.nbm.behavior_events})
        nb_df_rrm = pse_rrm.align_lagged_view('RRM', ['outcome'], laglist=None, cell_type='A2A')
        ```
        """
        proj_sel = self.meta['animal'].str.startswith(proj)
        meta_sel = df_select_kwargs(self.meta, return_index=True, **kwargs)
        all_nb_dfs = []

        for animal, session in self.meta.loc[meta_sel & proj_sel, ['animal', 'session']].values:
            try:
                bmat, neuro_series = self.load_animal_session(animal, session)
            except Exception:
                logging.warning(f'Error in {animal} {session}')
                bmat, ps_series = None, None

            if bmat is None:
                logging.info(f'skipping {animal} {session}')
            else:
                try:
                    bdf, dff_df = bmat.todf(), neuro_series.calculate_dff(method='dZF_jove')
                    nb_df = self.nbm.align_B2N_dff_ID(bdf, dff_df, events, form='wide')
                    nb_df = self.nbm.extend_features(nb_df)
                    sig_scores = neuro_series.diagnose_multi_channels(viz=False)
                    for sigch in sig_scores:
                        sigch_score_col = neuro_series.quality_metric + f'_{sigch}'
                        self.meta.loc[(self.meta['animal'] == animal)
                                      & (self.meta['session'] == session),
                                      sigch_score_col] = sig_scores[sigch]
                    if laglist is not None:
                        nb_df = self.nbm.lag_wide_df_ID(nb_df, laglist)
                    all_nb_dfs.append(nb_df)
                except Exception:
                    logging.warning(f'Error in calculating dff or AUC-score for {animal}, {session}, check signal')

        all_nb_df = pd.concat(all_nb_dfs, axis=0)
        return all_nb_df.merge(self.meta, how='left', on=['animal', 'session'])

    def behavior_lagged_view(self, proj, laglist=None, **kwargs):
        """
        Given the project code specified in `proj`, look through metadata loaded through `self.info_name`
        and merge all behavioral data. If specified, lag data according to `laglist`.
        :param proj: 3 letter project code: e.g. 'UUM', 'RRM', 'BSD', 'JUV', 'FIP'
        :param laglist: Dict[str, Dict]
                            dictionary specifying all column features that need to be trial lagged.
                            Each column is mapped to an additional dictionary:
                             {'pre': [lags backward in time],
                              'post': ['lags forward in time'],
                              'long': [boolean for whether pivoting the df to a long-form df]}
        :param kwargs: provide additional conditions on what sessions to include through selecting certain
        variable value in metadata
        :return: all_nb_df: pd.DataFrame
                    dataframe containing neural and behavior data
        E.g.
        ```{python}
        pse_rrm = PS_Expr(data_root)
        laglist = {'action': {'pre': 2, 'post': 3},
                   'rewarded': {'pre': 2, 'post': 3},
                   f'outcome_neur': {'pre': 2, 'post': 3, 'long': True}
        laglist.update({ev: {'pre': 1, 'post': 1} for ev in pse_rrm.nbm.behavior_events})
        nb_df_rrm = pse_rrm.align_lagged_view('RRM', ['outcome'], laglist=None, cell_type='A2A')
        ```
        """
        proj_sel = self.meta['animal'].str.startswith(proj)
        meta_sel = df_select_kwargs(self.meta, return_index=True, **kwargs)
        all_bdfs = []

        for animal, session in self.meta.loc[meta_sel & proj_sel, ['animal', 'session']].values:
            try:
                bmat, _ = self.load_animal_session(animal, session)
            except Exception:
                logging.warning(f'Error in {animal} {session}')
                bmat, ps_series = None, None

            if bmat is None:
                logging.info(f'skipping {animal} {session}')
            else:
                try:
                    bdf = bmat.todf()
                    all_bdfs.append(bdf)
                except Exception:
                    logging.warning(f'Error in calculating dff or AUC-score for {animal}, {session}, check signal')

        all_bdf = pd.concat(all_bdfs, axis=0)
        final_bdf = all_bdf.merge(self.meta, how='left', on=['animal', 'session'])
        if 'stimulation_on' in all_bdf.columns:
            final_bdf.loc[final_bdf['opto_stim'] != 1, 'stimulation_on'] = None
            final_bdf.loc[final_bdf['opto_stim'] != 1, 'stimulation_off'] = None
        return final_bdf

    def neural_sig_check(self, proj, **kwargs):
        proj_sel = self.meta['animal'].str.startswith(proj)
        meta_sel = df_select_kwargs(self.meta, return_index=True, **kwargs)

        for animal, session in self.meta.loc[meta_sel & proj_sel, ['animal', 'session']].values:
            try:
                _, neuro_series = self.load_animal_session(animal, session)
                qual_check_folder = oj(self.plot_path, 'FP_quality_check')
                sig_scores = neuro_series.diagnose_multi_channels(plot_path=qual_check_folder)
            except:
                logging.warning(f'Error in {animal} {session}')
                bmat, ps_series = None, None


class PS_Expr(NBExperiment):
    info_name = 'probswitch_neural_subset.csv'
    spec_name = 'probswitch_animal_specs.csv'

    def __init__(self, folder, **kwargs):
        # age and animal_ID must be filled
        super().__init__(folder)
        self.folder = folder
        pathlist = folder.split(os.sep)[:-1] + ['plots']
        self.plot_path = oj(os.sep, *pathlist)
        print(f'Changing plot_path as {self.plot_path}')
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        for kw in kwargs:
            if hasattr(self, kw):
                setattr(self, kw, kwargs[kw])
        info = pd.read_csv(os.path.join(folder, self.info_name))
        spec = pd.read_csv(os.path.join(folder, self.spec_name))
        self.meta = info.merge(spec, left_on='animal', right_on='alias', how='left')
        self.meta['animal_ID'] = self.meta['animal_ID_y']
        # self.meta.loc[self.meta['session_num']]
        self.meta['cell_type'] = self.meta['animal_ID'].str.split('-', expand=True)[0]
        self.meta['session'] = self.meta['age'].apply(self.cvt_age_to_session)
        self.meta['hemi'] = ''
        self.meta.loc[self.meta['plug_in'] == 'R', 'hemi'] = 'right'
        self.meta.loc[self.meta['plug_in'] == 'L', 'hemi'] = 'left'
        self.nbm = PS_NBMat()

        # TODO: modify this later
        if 'trig_mode' not in self.meta.columns:
            self.meta['trig_mode'] = 'BSC1'

    def cvt_age_to_session(self, age):
        DIG_LIMIT = 2  # limit the digits allowed for age representation (max 99)
        age = float(age)
        if np.allclose(age % 1, 0):
            return f'p{int(age)}'
        else:
            digit = np.around(age % 1, DIG_LIMIT)
            agenum = int(age // 1)
            if np.allclose(digit, 0.05):
                return f'p{agenum}_session0'
            else:
                snum = str(digit).split('.')[1]
                return f'p{agenum}_session{snum}'

    def load_animal_session(self, animal_arg, session, options='all'):
        # load animal session according to info sheet and align to neural signal
        # left location
        # right location
        # left virus
        # right virus
        file_found = False

        arg_type = 'animal_ID' if (animal_arg in np.unique(self.meta.animal_ID)) else 'animal'
        for aopt in ('animal_ID', 'animal'):
            filearg = self.meta.loc[self.meta[arg_type] == animal_arg, aopt].values[0]
            filemap = self.encode_to_filename(filearg, session, ['behaviorLOG', 'FP', 'FPTS'])
            if filemap['behaviorLOG'] is not None:
                file_found = True
                break
        if not file_found:
            logging.warning(f'Cannot find files for {animal_arg}, {session}')
            return None, None

        hfile = h5py.File(filemap['behaviorLOG'], 'r')
        animal_alias = self.meta.loc[self.meta[arg_type] == animal_arg, 'animal'].values[0]
        bmat = PSBehaviorMat(animal_alias, session, hfile, STAGE=1)
        fp_file = filemap['FP']
        fp_timestamps = filemap['FPTS']

        if (fp_file is not None) and (fp_timestamps is not None):
            session_sel = self.meta['session'] == session
            trig_mode = self.meta.loc[(self.meta[arg_type] == animal_arg) & session_sel, 'trig_mode'].values[0]
            ps_series = BonsaiPS1Hemi2Ch(fp_file, fp_timestamps, trig_mode, animal_alias, session)
            ps_series.merge_channels()
            ps_series.realign_time(bmat)
        else:
            ps_series = None
        return bmat, ps_series

    def encode_to_filename(self, animal, session, ftypes="processed_all"):
        """
        :param folder: str
                folder for data storage
        :param animal: str
                animal name: e.g. A2A-15B-B_RT
        :param session: str
                session name: e.g. p151_session1_FP_RH
        :param ftype: list or str:
                list (or a single str) of typed files to return
                'exper': .mat files
                'bin_mat': binary file
                'green': green fluorescence
                'red': red FP
                'behavior': .mat behavior file
                'FP': processed dff hdf5 file
                if ftypes=="all"
        :return:
                returns all 5 files in a dictionary; otherwise return all file types
                in a dictionary, None if not found
        """
        folder = self.folder
        paths = [os.path.join(folder, animal, session), os.path.join(folder, animal + '_' + session),
                 os.path.join(folder, animal), folder]
        if ftypes == "raw all":
            ftypes = ["exper", "bin_mat", "green", "red"]
        elif ftypes == "processed_all":
            ftypes = ["processed", "green", "red", "FP"]
        elif isinstance(ftypes, str):
            ftypes = [ftypes]
        results = {ft: None for ft in ftypes}
        registers = 0
        for p in paths:
            if os.path.exists(p):
                for f in os.listdir(p):
                    for ift in ftypes:
                        if ift == 'FP':
                            ift_arg = 'FP_'
                        else:
                            ift_arg = ift
                        if (ift_arg in f) and (animal in f) and (session in f):
                            results[ift] = os.path.join(p, f)
                            registers += 1
                            if registers == len(ftypes):
                                return results if len(results) > 1 else results[ift]
        return results if len(results) > 1 else list(results.values())[0]

    def select_region(self, nb_df, region):
        left_sel = (nb_df['hemi'] == 'left') & (nb_df['left_region'] == region)
        right_sel = (nb_df['hemi'] == 'right') & (nb_df['right_region'] == region)
        return nb_df[left_sel | right_sel].reset_index(drop=True)


class RR_Expr(NBExperiment):
    # TODO: for decoding, add functions to merge multiple rois
    info_name = 'rr_neural_subset.csv'
    spec_name = 'rr_animal_specs.csv'

    def __init__(self, folder, **kwargs):
        super().__init__(folder)
        self.folder = folder
        pathlist = folder.split(os.sep)[:-1] + ['plots']
        self.plot_path = oj(os.sep, *pathlist)
        print(f'Changing plot_path as {self.plot_path}')
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        for kw in kwargs:
            if hasattr(self, kw):
                setattr(self, kw, kwargs[kw])
        info = pd.read_csv(os.path.join(folder, self.info_name))
        spec = pd.read_csv(os.path.join(folder, self.spec_name))
        self.meta = info.merge(spec, left_on='animal', right_on='alias', how='left')
        # # self.meta.loc[self.meta['session_num']]
        self.meta['cell_type'] = self.meta['animal_ID'].str.split('-', expand=True)[0]
        self.meta['session'] = self.meta['age'].apply(self.cvt_age_to_session)
        self.nbm = RR_NBMat()
        self.nbviz = RR_NBViz(self)

        # # TODO: modify this later
        if ('trig_mode' not in self.meta.columns) and ('fp_recorded' in self.meta.columns):
            self.meta.loc[self.meta['fp_recorded'] == 1, 'trig_mode'] = 'TRIG1'

    def cvt_age_to_session(self, age):
        DIG_LIMIT = 2  # limit the digits allowed for age representation (max 99)
        age = float(age)
        if np.allclose(age % 1, 0):
            return f'Day{int(age)}'
        else:
            digit = np.around(age % 1, DIG_LIMIT)
            agenum = int(age // 1)
            if np.allclose(digit, 0.05):
                return f'Day{agenum}_session0'
            else:
                snum = str(digit).split('.')[1]
                return f'Day{agenum}_session{snum}'

    def load_animal_session(self, animal_arg, session, options='all'):
        # load animal session according to info sheet and align to neural signal
        # left location
        # right location
        # left virus
        # right virus
        file_found = False

        arg_type = 'animal_ID' if (animal_arg in np.unique(self.meta.animal_ID)) else 'animal'
        for aopt in ('animal_ID', 'animal'):
            filearg = self.meta.loc[self.meta[arg_type] == animal_arg, aopt].values[0]
            filemap = self.encode_to_filename(filearg, session, ['RR_', 'FP', 'FPTS'])
            if filemap['RR_'] is not None:
                file_found = True
                break
        if not file_found:
            logging.warning(f'Cannot find files for {animal_arg}, {session}')
            return None, None

        animal_alias = self.meta.loc[self.meta[arg_type] == animal_arg, 'animal'].values[0]
        bmat = RRBehaviorMat(animal_alias, session, filemap['RR_'], STAGE=1)
        fp_file = filemap['FP']
        fp_timestamps = filemap['FPTS']

        if (fp_file is not None) and (fp_timestamps is not None):
            session_sel = self.meta['session'] == session
            trig_mode = self.meta.loc[(self.meta[arg_type] == animal_arg) & session_sel, 'trig_mode'].values[0]
            rr_series = BonsaiRR2Hemi2Ch(fp_file, fp_timestamps, trig_mode, animal_alias, session)
            rr_series.merge_channels()
            rr_series.realign_time(bmat)
        else:
            rr_series = None
        return bmat, rr_series

    def encode_to_filename(self, animal, session, ftypes="all"):
        """
        :param folder: str
                folder for data storage
        :param animal: str
                animal name: e.g. A2A-15B-B_RT
        :param session: str
                session name: e.g. p151_session1_FP_RH
        :param ftype: list or str:
                list (or a single str) of typed files to return
                'RR_': behavior files
                'bin_mat': binary file
                'green': green fluorescence
                'red': red FP
                'behavior': .mat behavior file
                'FP': processed dff hdf5 file
                if ftypes=="all"
        :return:
                returns all 5 files in a dictionary; otherwise return all file types
                in a dictionary, None if not found
        """
        folder = self.folder
        # bfolder = oj(folder, 'RR_Behavior_Data')
        # fpfolder = oj(folder, 'RR_FP_Data')
        # paths = [os.path.join(bfolder, animal, session), os.path.join(bfolder, animal + '_' + session),
        #          os.path.join(bfolder, animal), bfolder,
        #          oj(fpfolder, animal, session), oj(fpfolder, animal + '_' + session),
        #          oj(fpfolder, animal), fpfolder]
        paths = [oj(folder, animal, session), oj(folder, animal + '_' + session),
                 oj(folder, animal), folder]
        if ftypes == "all":
            ftypes = ['RR_', "FP", 'FPTS']
        elif isinstance(ftypes, str):
            ftypes = [ftypes]
        results = {ft: None for ft in ftypes}
        registers = 0
        for p in paths:
            if os.path.exists(p):
                for f in os.listdir(p):
                    for ift in ftypes:
                        if ift == 'FP':
                            ift_arg = 'FP_'
                        else:
                            ift_arg = ift
                        if (ift_arg in f) and (animal in f) and (session in f):
                            results[ift] = os.path.join(p, f)
                            registers += 1
                            if registers == len(ftypes):
                                return results if len(results) > 1 else results[ift]
        return results if len(results) > 1 else list(results.values())[0]


class RR_Opto(RR_Expr):

    info_name = 'rr_opto_subset.csv'
    spec_name = 'rr_animal_specs.csv'

    def __init__(self, folder, **kwargs):
        super().__init__(folder, **kwargs)
        self.nbm = RR_NBMat(neural=False)
