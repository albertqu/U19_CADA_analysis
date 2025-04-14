from enum import Enum
import numpy as np
import pandas as pd


class TaskState(Enum):
    HALL = 0
    TJ = 1
    WAIT = 2
    REWARD = 3
    UNREWARD = 4

class EventProcessor:
    events = ['tone_onset', 'acceptTJ', 'rejectTJ', 'quit',
               'rewardT', 'unrewardT', 'collection']
    nodes = [i for i in range(len(events))]
    ev2node = {e: n for n, e in enumerate(events)}
    graph = {
        0: {1: TaskState.HALL, 2: TaskState.HALL},
        1: {3: TaskState.WAIT, 4: TaskState.WAIT, 5: TaskState.WAIT},
        2: {0: TaskState.TJ},
        3: {0: TaskState.TJ},
        4: {0: TaskState.REWARD, 6: TaskState.REWARD},
        5: {0: TaskState.UNREWARD},
        6: {0: TaskState.TJ}
    }

    event_taus = {
        'tone_onset': (0, 1),
        'acceptTJ': (-1, 1.5),
        'rejectTJ': (-1, 1.5),
        'quit': (-1.5, 1),
        'rewardT': (0, 1.5),
        'unrewardT': (0, 1.5),
        'collection': (0, 1)
    }

    id_cols = ['animal', 'session', 'trial']

    def __init__(self, bdf):
        self.trial_bdf = self.prep_behavior_df(bdf)
        event_bdf = self.trial_bdf.melt(id_vars=['animal', 'session', 'trial'], 
                                             value_vars=self.trial_bdf.columns[3:], var_name='event', 
                                             value_name='time').sort_values(['animal', 'session', 'trial', 'time'])
        self.event_bdf = event_bdf.dropna(subset=['time']).reset_index(drop=True)
        self.bdf = bdf
        
    def strip_event_prefix(self, x):
        if x.startswith('prev_') or x.startswith('next_'):
            return x[5:]
        return x

    def prep_behavior_df(self, bdf):
        bdf['acceptTJ'] = np.nan
        acc_sel = bdf['slp_accept'] == 1
        bdf.loc[acc_sel, 'acceptTJ'] = bdf.loc[acc_sel, 'slp_T_Entry_time'].values

        bdf['rejectTJ'] = np.nan
        rej_sel = bdf['slp_decision'] == 'reject'
        bdf.loc[rej_sel, 'rejectTJ'] = bdf.loc[rej_sel, 'slp_T_Entry_time'].values

        bdf['rewardT'] = np.nan
        reward_sel = bdf['reward'] == 1
        bdf.loc[reward_sel, 'rewardT'] = bdf.loc[reward_sel, 'outcome'].values

        bdf['unrewardT'] = np.nan
        unreward_sel = (bdf['reward'] == 0) & (~bdf['outcome'].isnull())
        bdf.loc[unreward_sel, 'unrewardT'] = bdf.loc[unreward_sel, 'outcome'].values
        return bdf[['animal', 'session', 'trial', 'tone_onset', 'acceptTJ', 'rejectTJ', 'slp_quit_time', 
                        'rewardT', 'unrewardT', 'collection']].rename(columns={'slp_quit_time': 'quit'})
    
    def update_event_taus(self, event_taus):
        self.event_taus.update(event_taus)

    def get_GAM_bdf(self):
        trial_bdf = self.trial_bdf
        # lag event front and back
        metadata = self.bdf[['animal', 'session', 'trial', 'offer_prob', 'slp_accept',
                             'lastTrial', 'slp_hall_time']].rename(columns={'slp_hall_time': 'hall_time',
                                                                            'slp_accept': 'ACC'})
        trial_bdf = trial_bdf.merge(metadata, how='left', on=['animal', 'session', 'trial'])
        prev_events = ['prev_' + e for e in self.events]
        post_events = ['next_' + e for e in self.events]
        trial_bdf[prev_events] = trial_bdf[self.events].shift(1)
        trial_bdf[post_events] = trial_bdf[self.events].shift(-1)
        trial_bdf.loc[trial_bdf['trial'] == 1, prev_events] = np.nan
        trial_bdf.loc[trial_bdf['lastTrial'], post_events] = np.nan
        # gam variables
        self.gam_aux = ['offer_prob', 'hall_time', 'ACC']
        self.gam_events = self.events + prev_events + post_events
        trial_bdf['trial_start'] = trial_bdf['tone_onset']
        trial_bdf[self.gam_events] = trial_bdf[self.gam_events] - trial_bdf['trial_start'].values.reshape(-1, 1)
        gam_bdf = trial_bdf[['animal', 'session', 'trial'] + self.gam_aux + self.gam_events].reset_index(drop=True)
        return gam_bdf
    
    def get_GAM_mat(self, neur_df_long, regularize=True):
        """ 
        GAM mat is carefully constructed such that the columns are in the following order:
        id_cols: len=3
        gam_neur_cols: len=4
        gam_aux: len=2
        gam_events: len=7*3
        """
        gam_neur_meta = ['hemi', 'cell_type', 'neur_time', 'ZdF']
        self.gam_neur_cols = gam_neur_meta
        neur_df_long = neur_df_long.dropna(subset=gam_neur_meta).reset_index(drop=True)
        gam_bdf = self.get_GAM_bdf()
        gam_mat = neur_df_long.merge(gam_bdf, how='left', on=['animal', 'session', 'trial'])
        gam_mat = gam_mat.dropna(subset=['tone_onset']).reset_index(drop=True)
        gam_mat[self.gam_events] = gam_mat['neur_time'].values.reshape(-1, 1) - gam_mat[self.gam_events].values
        # regularize time
        if regularize:
            for e in self.gam_events:
                event = self.strip_event_prefix(e)
                tau0, tauf = self.event_taus[event]
                gam_mat[e] = np.minimum(np.maximum(gam_mat[e], tau0), tauf)
                gam_mat[e] = gam_mat[e].fillna(tau0)
        gam_mat =gam_mat[self.id_cols+self.gam_neur_cols+self.gam_aux+self.gam_events]
        return gam_mat.sort_values(['animal', 'session', 'hemi', 'trial', 'neur_time']).reset_index(drop=True)

    def get_trial_event_count(self, preT=1, postT=1):
        def helper(edf):
            edf['idx'] = edf.index
            event_counts = []
            trial_starts = edf[edf['event'] == 'tone_onset'].sort_values('trial')
            for i, row in trial_starts.iterrows():
                start = row['time']
                # each trial will have a subset df
                node_df = edf[(edf['time'] >= start - preT) & (edf['time'] <= start + postT)].sort_values('time')
                # id0, idf = node_df['idx'].iat[0], node_df['idx'].iat[-1]
                ecounts = node_df['event'].value_counts().to_dict()
                icounts = {k: 0 for k in self.events}
                for k in ecounts:
                    icounts[k] = ecounts[k]
                icounts['trial'] = row['trial']
                icounts['animal'] = row['animal']
                icounts['session'] = row['session']
                event_counts.append(icounts)
            return pd.DataFrame(event_counts)
        return self.event_bdf.groupby(['animal', 'session'], as_index=False).apply(helper)
    

### Utils

def parse_lag_expr(expression):
    """
    Parse lag expressions of the form L(x) or L(x1:x2), handling nested parentheses.
    Args:
        expression (str): A string containing lag expressions like "L(x)" or "L(x1:x2)"
    Returns:
        list: List of tuples containing (lag_depth, variable_name) for each lag operation
    """
    result = []
    i = 0
    
    while i < len(expression):
        # Find the start of a lag operation
        if expression[i:i+2] == "L(":
            # Count the lag depth (how many consecutive L's)
            lag_depth = 0
            while i + lag_depth < len(expression) and expression[i + lag_depth] == 'L':
                lag_depth += 1
            # Move past the 'L' characters
            i += lag_depth
            # Find the matching closing parenthesis
            if expression[i] == '(':
                open_count = 1
                start_pos = i + 1
                i += 1
                while i < len(expression) and open_count > 0:
                    if expression[i] == '(':
                        open_count += 1
                    elif expression[i] == ')':
                        open_count -= 1
                    i += 1
                
                # Extract the variable name from within the parentheses
                if open_count == 0:
                    variable = expression[start_pos:i-1].strip()
                    return variable
                    #result.append((lag_depth, variable))
            else:
                i += 1
        else:
            i += 1
            
    return result

##############################################
##### PREPROCESSING FOR REGRESSION ANALYSIS ##
##############################################
from statsmodels.tsa.tsatools import lagmat
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline   
from pandas.api.types import is_numeric_dtype
from utils import df_select_kwargs

def get_baseline_activities(y, X, future, past, fr=None):
    """ Select regions where covariate X has no effect on y after lagging.
    y: recording time series (neural data)
    X: 2D matrix, various covariations
    future/past = pre/post: Lag X backward by `pre` frames and forward by `post` frames, if fr
    is specified, otherwise pre/post specifies the lag in seconds.
    pre = time excluded from baseline estimation before cue onset, post = time excluded after cue onset
    NOTE: **drops 'edges' automatically, see 'valid' mode in np.convolve**
    """
    # uniform lag everywhere
    pre, post = future, past
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    assert len(X.shape) == 2, 'reshape covariate to 2D'
    if fr is not None:
        pre, post = int(pre * fr), int(post * fr)
    y_t = y[pre:len(y)-post]
    X_lag = lagmat(X, pre+post, trim="none", original='in')
    X_lag = X_lag[post+pre:len(X_lag)-(pre+post)]
    return y_t[np.sum(np.abs(X_lag), axis=1) == 0]

def samples_to_discrete_timeseries(ts, xs=None, method='integration', fr=20, tmax=None):
    """ Turns potentially unevenly sampled timeseries into evenly sampled time series

    method: str
        integration: discretize through integration
        interp: resample data via interpolation
        state: resample data via interpolation approach using `traces` package
    """
    if xs is None:
        assert (method == 'integration'), 'must use integration method for binary timeseries'
        xs = np.full_like(ts, 1)
    if tmax is None:
        tmax = np.max(ts)
    dt = 1 / fr
    nT = np.floor(tmax / dt) + 1
    T = nT * dt
    ts_new = np.arange(0, T+dt, dt)
    if method == 'integration':
        ys, _ = np.histogram(ts, np.concatenate([ts_new, [ts_new[-1]+dt]]))
    elif method == 'interp':
        # replace with natural splines 
        cs = CubicSpline(ts, xs, bc_type='natural')
        ys = cs(ts_new)
        # ys = interp1d(ts, xs, fill_value='extrapolate')(ts_new)
    elif method == 'state':
        assert 'traces' in globals(), 'traces must be installed'
    else:
        raise NotImplementedError(f'unknown method {method}')
    return ts_new, ys

def get_nb_df_baseline(bdf, dff_df_unmelted, beh_cols=None, fr=8.052):
    behav = bdf
    dff_df = dff_df_unmelted
    frame_time = dff_df['time']
    dFF = dff_df[[c for c in dff_df.columns if c != 'time']].values.T
    if beh_cols is None:
        beh_cols = ['outcome', 'first_lick_in', 'onset'] # select behavioral events
    behavior_times = behav[beh_cols].values.T # processes structure of csv to create time stamps
    tmax = max(60 * 60, np.nanmax(behavior_times))
    X = []
    for i in range(len(behavior_times)): # discretizes time stamps
        btimes = behavior_times[i]
        ti, xi = samples_to_discrete_timeseries(btimes[~np.isnan(btimes)], xs=None, method='integration', fr=fr, tmax=tmax)
        X.append(xi)
    X = np.array(X).T

    # calculate baseline for all cells
    i = 0
    baselines = [None] * len(dFF)
    for i in range(len(dFF)):
        tj,y = samples_to_discrete_timeseries(frame_time, dFF[i], method='interp', fr=fr, tmax=tmax)
        baselines[i] = y
    baselines = np.vstack(baselines).T
    safe_region = tj >= frame_time.min()
    baselines = get_baseline_activities(baselines[safe_region], X[safe_region, :], 2, 2, fr=fr).T
    return dFF, baselines

def merge_resample_dc(discs, conts, fr=None, tmax=None):
    # discs: list of pd.Dataframe containing behavioral timestamps
    # conts: list of pd.Dataframe containing column `time` and measurements
    # Assuming continuous variables in conts has no NaNs!
    if tmax is None:
        tmax = max(np.max([np.nanmax(d.values) for d in discs]), 
                   np.max([np.max(c['time']) for c in conts]))
    if fr is None:
        fr = np.min([(len(c) - 1) / (np.max(c['time']) - np.min(c['time'])) for c in conts])
    
    ts = None
    rs_dfs = []
    for disc in discs:
        rs_ds = [samples_to_discrete_timeseries(disc.loc[~pd.isnull(disc[dcol]), dcol].values, xs=None,
                                                method='integration', fr=fr, tmax=tmax)[1] for dcol in disc.columns]
        rs_df = pd.DataFrame(np.vstack(rs_ds).T, columns=disc.columns)
        rs_dfs.append(rs_df)
    for cont in conts:
        cont_cols = [c for c in cont.columns if c != 'time']
        time_axis = cont['time']
        if np.sum(pd.isnull(cont[cont_cols]).values) > 0:
            raise ValueError("Algorithm requires that continuous series no longer have no NaNs")

        if ts is None:
            ts,y = samples_to_discrete_timeseries(time_axis.values, cont[cont_cols[0]].values, method='interp', fr=fr, tmax=tmax)
            rs_cs = [y] + [samples_to_discrete_timeseries(time_axis.values, cont[ccol].values, method='interp', fr=fr, tmax=tmax)[1] for ccol in cont_cols[1:]]
        else:
            rs_cs = [samples_to_discrete_timeseries(time_axis.values, cont[ccol], method='interp', fr=fr, tmax=tmax) for ccol in cont_cols]
        cs_df = pd.DataFrame(np.vstack(rs_cs).T, columns=cont_cols)
        rs_dfs.append(cs_df)
    
    return pd.concat([pd.Series(ts, name='time')] + rs_dfs, axis=1)

def add_trial_variable_glm_df(glm_df, trial_var_df, trial_starts, trial_ends):
    # assume trial_start trial_ends contain no NaNs
    # use traces implementation for better performance
    for tv in trial_var_df:
        if is_numeric_dtype(trial_var_df[tv]):
            glm_df[tv] = np.nan
        else:
            glm_df[tv] = None
    for i in range(len(trial_starts)):
        t_s, t_e = trial_starts[i], trial_ends[i]
        if not np.isnan(t_s) and not np.isnan(t_e):
            glm_df.loc[(glm_df['time'] >= t_s) & (glm_df['time'] < t_e), trial_var_df.columns] = trial_var_df.iloc[i, :].values
    return glm_df      


def get_glm_df(rse, sleap_bdf, animal, session, method='ZdF_jove'):
    """Takes in sleap_bdf and rse information and build GLM df (before lagging)
    """
    session_df = sleap_bdf[(sleap_bdf['animal'] == animal) & (sleap_bdf['session'] == session)].reset_index(drop=True)
    # key event df: tone_onset
    eprop = EventProcessor(session_df)
    event_df = eprop.event_bdf
    wide_bdf = eprop.trial_bdf
    bmat, neuro_series = rse.load_animal_session(animal, session)
    dff_df = neuro_series.calculate_dff(method=method, melt=False)

    # preparing continuous neural data
    roi_str_op = lambda s: s.replace('_470nm', '').replace('_ZdFF', '_ZdF') if ('_470nm' in s) else s
    dff_df_wide = dff_df.drop(columns=['method'])
    dff_df_wide.columns = [roi_str_op(c) for c in dff_df_wide.columns]
    # preparing discrete event data
    behavior_cols = eprop.events
    glm_df = merge_resample_dc([wide_bdf[behavior_cols]], [dff_df_wide], fr=20, tmax=None)
    # filter out rows earlier than neural end and later than neural signal start
    dff_start, dff_end = dff_df_wide['time'].min(), eprop.bdf['tmax'].max()
    glm_df = df_select_kwargs(glm_df, time=lambda x: (x>=dff_start) & (x<=dff_end)).reset_index(drop=True)

    # adding trial level variables
    select_vars = ['tone_onset', 'offer_prob', 'slp_accept', 'slp_hall_time', 'trial_end', 'lastTrial', 'tmax']
    trial_var_df = eprop.bdf[select_vars].rename(columns={'slp_accept': 'ACC',
                                                        'slp_hall_time': 'hall_time'})
    trial_var_df['trialEnd'] = trial_var_df['tone_onset'].shift(-1)
    trial_var_df.loc[trial_var_df['lastTrial'], 'trialEnd'] = trial_var_df.loc[trial_var_df['lastTrial'], 'tmax']
    trial_vars = ['offer_prob', 'ACC', 'hall_time']
    glm_df = add_trial_variable_glm_df(glm_df, trial_var_df[trial_vars], trial_var_df['tone_onset'], trial_var_df['trialEnd'])
    glm_df[trial_vars] = glm_df[trial_vars].fillna(0)
    glm_df['animal'] = animal
    glm_df['session'] = session
    return glm_df, eprop

####################################
# REGRESSION ANALYSIS UTILS ########
####################################

from scipy.interpolate import make_smoothing_spline
from sklearn.linear_model import LassoCV
import seaborn as sns
import matplotlib.pyplot as plt
import patsy 
from utils_rr.configs import RAND_STATE
from sklearn.model_selection import train_test_split

class FCM_regmat:
    def __init__(self, glm_df, fr, eprop):
        self.glm_df = glm_df
        self.eprop = eprop
        self.fr = fr
        self.lag_map = {}

    def reg_from_formula(self, formula:str):
        """
        regression with lag formula, L() denotes lag operations.
        So far only two functions are supported: L(x) and L(x1:x2)
        formula: str
            formula for the regression model, y ~ L(x1:x2) + x2 + x3
        """
        covars = formula.strip().split('~')[1].split('+')
        yvar = formula.strip().split('~')[0].strip()
        lag_vars = set() # record for lagging operation
        raw_vars = set() # perform patsy on 
        for cv in covars:
            if 'L(' in cv:
                v = cv.strip().split('L(')[1][:-1]
                lag_vars.add(v.strip())
            else:
                raw_vars.add(cv.strip())
        lagcols = [c for c in patsy.dmatrix('+'.join(list(lag_vars)), data=self.glm_df, 
                     return_type='dataframe').columns if c != 'Intercept']

        # use patsy to get intermediate design matrix
        raw_formula = f'{yvar} ~ {"+".join(list(raw_vars|lag_vars))}'
        # print(raw_formula)
        y, X = patsy.dmatrices(raw_formula, data=self.glm_df, return_type='dataframe')
        if not lagcols:
            return y, X
        Xlags = self.get_time_lags(lagcols, X)
        X = pd.concat([X[list(raw_vars)], Xlags], axis=1)
        return y, X

    def get_time_lags(self, lagcols, glm_df):
        if not lagcols:
            return None
        fr = self.fr
        lagdf = []
        def lag_str_func(num):
            if num < 0:
                return f'f{-num}'
            elif num > 0:
                return f'b{num}'
            else:
                return 't0'
        for col in lagcols:
            if ':' in col:
                parts = col.split(':')
                ev = [pt for pt in parts if pt in self.eprop.events][0]
            elif col in self.eprop.events:
                ev = col
            else:
                continue
            sdf = glm_df[col]
            tau0, tauf = EventProcessor.event_taus[ev]
            tau0, tauf = int(tau0*fr), int(tauf*fr)
            forw, back = -tau0, tauf
            cols = [col+'__'+lag_str_func(i) for i in range(-forw, back+1)]
            evlag = pd.DataFrame(lagmat(sdf, maxlag=forw+back, trim='None', original='in')[forw:-back], columns=cols)
            lagdf.append(evlag)
            self.lag_map[col] = list(evlag.columns)
        return pd.concat(lagdf, axis=1)

def list_str_contains(s, lst):
    contents = [l for l in lst if l in s]
    if contents:
        return contents[0]

def lagstr2num(s):
    ts = s.split('__')[-1]
    if ts[0] == 'f':
        return -float(ts[1:])
    elif ts[0] == 'b':
        return float(ts[1:])
    else:
        return 0.0

def getTrialVar(feature):
    if ':' in feature:
        a, b = feature.split(':')
        if a in EventProcessor.events:
            return b
        else:
            return a
    else:
        return 'base'
    
def trialVar_extract(trialV):
    if '[' not in trialV:
        return None
    tvar, value = trialV.split('[')
    value = value[:-1]
    if 'T.' in value:
        value = value.split('T.')[-1]
    return value

def clean_trialVar(trialV):
    if '[' not in trialV:
        return trialV
    tvar, value = trialV.split('[')
    if tvar.startswith('C('):
        tvar = tvar[2:-1]
    return tvar


def clean_reg_results(reg_coef, fr=20, clean_domains='all'):
    reg_coef[['feature', 'lag']] = reg_coef['var'].str.split('__', expand=True)
    reg_coef['lag'] = reg_coef['lag'].apply(lagstr2num) / fr
    reg_coef['trial_var'] = reg_coef['feature'].apply(getTrialVar)
    reg_coef['event'] = reg_coef['feature'].apply(lambda s: s.split(':')[0] if ':' in s else s)
    reg_coef['trialV_value'] = reg_coef['trial_var'].apply(trialVar_extract)
    reg_coef.loc[reg_coef['trialV_value'].isnull(), 'trialV_value'] = reg_coef.loc[reg_coef['trialV_value'].isnull(), 'trial_var']
    reg_coef['trial_var'] = reg_coef['trial_var'].apply(clean_trialVar)

    for ev in reg_coef['event'].unique():
        ev_df = reg_coef[reg_coef['event'] == ev]
        base_vals = ev_df.loc[ev_df['trial_var'] == 'base', 'coef'].values
        for cond in ev_df['trialV_value'].unique():
            if cond == 'base':
                print('#', ev, cond)
                continue
            print(ev, cond)
            vs = reg_coef.loc[(reg_coef['event'] == ev) & (reg_coef['trialV_value'] == cond), 'coef']
            reg_coef.loc[(reg_coef['event'] == ev) & (reg_coef['trialV_value'] == cond), 'coef'] = vs.values + base_vals

    # domain cleaning
    domain_events = {'TJ': ['acceptTJ', 'rejectTJ'], 'outcome': ['rewardT', 'unrewardT']}
    if isinstance(clean_domains, list):
        domain_events = {ev: domain_events[ev] for ev in clean_domains}
    for ev_group in domain_events:
        elist = domain_events[ev_group]
        reg_coef.loc[reg_coef['event'].isin(elist), 'trialV_value'] = reg_coef.loc[reg_coef['event'].isin(elist), 'event']
        reg_coef.loc[reg_coef['event'].isin(elist), 'event'] = ev_group
    return reg_coef

def smooth_helper(df):
    df = df.sort_values('lag')
    ts = df['lag'].values
    ss = make_smoothing_spline(ts, df['coef'], lam=0.001)
    df['coef'] = ss(ts)
    return df

def glm_model_and_plot(X, y, smooth=True, fr=20, clean_domains='all'):
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=RAND_STATE)
    verbose=True
    reg = LassoCV(cv=5, random_state=0).fit(X_train, y_train)
    train_score, test_score = reg.score(X_train, y_train), reg.score(X_test, y_test)
    # CV
    if verbose:
        print(f'Train score: {train_score:.4f}, Test score: {test_score:.4f}')

    reg_coef = pd.DataFrame({'coef': reg.coef_, 'var': list(X.columns)})
    reg_coef = clean_reg_results(reg_coef, fr=fr, clean_domains=clean_domains)

    all_kernels = reg_coef[['trial_var', 'event']].drop_duplicates().reset_index(drop=True)
    def get_best_row_col(n):
        col = int(np.ceil(np.sqrt(n)))
        row = int(np.ceil(n / col))
        return row, col
    row, col = get_best_row_col(all_kernels.shape[0])
    fig, axes = plt.subplots(row, col, figsize=(col*4, row*4))
    for i, row in all_kernels.iterrows():
        trialV, ev = row['trial_var'], row['event']
        plotdata = reg_coef[(reg_coef['trial_var'] == trialV) & (reg_coef['event'] == ev)].reset_index(drop=True)
        if trialV != 'base':
            plotdata = pd.concat([reg_coef[(reg_coef['trial_var'] == 'base') & (reg_coef['event'] == ev)].reset_index(drop=True),
                                  plotdata], axis=0)
        if smooth:
            plotdata = plotdata.groupby('trialV_value', as_index=True).apply(smooth_helper, include_groups=False).reset_index()
        plotdata.rename(columns={'trialV_value': trialV}, inplace=True)
        ax = axes.ravel()[i]
        sns.lineplot(data=plotdata, x='lag', y='coef', hue=trialV, ax=ax)
        ax.set_title(f'{ev} {trialV}')
        ax.axvline(0, c='gray', ls='--')
        ax.legend(loc='upper left', bbox_to_anchor=(0.5, 1.1))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.despine()
    return fig, {'coef': reg_coef, 'score': (train_score, test_score)}
