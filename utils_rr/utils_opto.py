from nb_viz import plot_correlation
# import utilities
from utils import *
from behaviors import *
from peristimulus import *
from neuro_series import *
from neurobehavior_base import *
from os.path import join as oj
import logging
import itertools
logging.basicConfig(level=logging.INFO)


def plot_missing_value_information(dataset: pd.DataFrame, figsize = (6, 8)) -> None:
    col_missing_vals = dataset.isna().astype(int).sum(axis=0).sort_values(ascending=False)
    col_missing_vals = col_missing_vals[col_missing_vals > 0]

    plt.figure(figsize=figsize)
    sns.set_theme(font_scale=1)
    sns.barplot(y=col_missing_vals.index, x=col_missing_vals)
    plt.show()

def print_missing_value_information(dataset: pd.DataFrame, lrange: int, rrange: int) -> pd.Series:
    col_missing_vals = dataset.isna().astype(int).sum(axis=0).sort_values(ascending=False)
    col_missing_vals = col_missing_vals[col_missing_vals > 0]

    print("shape if we remove all rows with missing values: ", 
          dataset.dropna(axis=0).shape)

    print("drop n columns highest in missing values, then remove all rows with missing values: ")
    for n in range(lrange, rrange):
        print(f'n = {n}', dataset.drop(col_missing_vals.index[:n], axis=1).dropna(axis=0).shape)

    return col_missing_vals


def check_sequence_rule(nb_df, sequence_rule):
    drop_mask = np.full(len(nb_df), False, dtype=bool)
    sa = nb_df[sequence_rule[0]]
    null_a = pd.isnull(sa)
    for i in range(len(sequence_rule)-1):
        sb = nb_df[sequence_rule[i+1]]
        null_b = pd.isnull(sb)
        null_mask = (~null_b) & null_a
        drop_mask = (drop_mask | null_mask) | (sb < sa)
        sa = sb
        null_a = null_b
    return drop_mask
        

def clean_data(nb_df, expr):
    """
    data column doc, D: delete in cleaned dataframe, A: Add derived feature using this column,
    C: features used for correlation, R: convert to relative time
    data_cols = ['animal', # identifier, subject ID nonverbose
                'session', # identifier, training day ID
                'age', # age for animals at the session, C
                'rr_day', # training day number, C
                'plugged_in', # variable for plug in location (L/R/0/null)
                'opto_stim', # indicator for whether opto stim is on, A
                'vid_saved', # indicator for whether videos are saved
                'epoch', # indicator for training epoch
                'trial', # trial number within sessions, C
                'lapIndex', # index of lap, C
                'blockIndex', # index of block, C
                # trial level performance information
                'tone_prob', # tone reward probability, C 
                'restaurant', # restaurant ID, C
                'accept', # boolean for accept or reject choices, C
                'reward', # indicator for whether a trial is rewarded, C
                # trial timing information
                'stimulation_on', # stimulation on timestamp, adding stim, stim{t-1}, ARC
                'stimulation_off', # similar to prev, ARC
                'tone_onset', # timestamp for onset of offer tones, cannot be null, RC
                'T_Entry', # timestamp for T junction entry in RR_maze, cannot be null, RC
                'choice', # timestamp for T junction entry in RR_maze, cannot be null, RC
                'outcome', # timestamp for when outcomes are revealed (pellets or no pellet), RC
                'quit', # timestamp for when animals quit an offer, RC
                'collection', # timestamp for when animals collect pellets, RC
                'trial_end', # timestamp when trial ends, RC
                'exit', # timestamp when animal exits the restaurant, RC
                # task level timing information
                'date', # `express_day = date - implant_date`, D-A
                'time_in', # time for the start of session, 24-format time HH:MM:00, derive to floating number A
                'time_out', # time for the end of session, 24-format time HH:MM:00
                'duration', # duration in HH:mm:ss format, derive to floating number A
                'tmax', # max time duration in `bmat.time_unit`, should be close to `duration`
                # task level performance information
                'pre_task_weight', # similar to post task weight, D
                'pre_task_weight_percent', # similar to post task weight percentage, C
                'banana', # number of banana pellets earned, C
                'plain', # number of plain pellets earned, C
                'grape', # number of grape pellets earned, C
                'chocolate', # number of chocolate pellets earned, C
                'pellets_earned ', # beware of the space: total pellets earned, D
                'supplement_g', # amount of food supplement, in grams, C
                'total_food_g', # amount of food obtained in total, in grams
                'post_task_weight', # weight after task, D
                'post_task_weight_percent',  # percentage of post task weight compared to adlib
                'hazard', # indicator variable for problematic data -> current strategy drop hazard != 0
                'animal_ID', # identifier, subject ID verbose, D
                'alias', # alias for subject identifier, same as animal, D
                'DOB', # date of birth for animal, D
                'sex', # sex of animal, same for all animals
                'implant_date',  # surgery date, -> derive feature `express_day` to assess expression level, D-A
                'curr_age', # current age of the animal, irrelevant, D
                'post_surg_day', # post surgery days current, D
                'weightS0', # Starting weight, D
                'weightP0', # Starting weight post treatment
                '80pc',  # 80% weight line, D
                '85pc', # similar to prev, D
                '90pc', # similar to prev, D
                'left_region', # implant location for ROI `left hemisphere` -> potentially need more detailed coordinate
                'left_virus', # virus choice for ROI `left hemisphere`
                'right_region', # similar to prev
                'right_virus', # similar to left ROI
                'left_fiber_eff', # irrelevant, D
                'right_fiber_eff', # irrelevant, D
                'fiber_NA', # NA for fiber, useful for quality assurance
                'cell_type' # cell type information
                ]
    """

    data_cols = ['animal', # identifier, subject ID nonverbose
                 'session', # identifier, training day ID
                 'age', # age for animals at the session, C
                 'rr_day', # training day number, C
                 'plugged_in', # variable for plug in location (L/R/0/null)
                 'opto_stim', # indicator for whether opto stim is on, A
                 'vid_saved', # indicator for whether videos are saved
                 'epoch', # indicator for training epoch
                 'trial', # trial number within sessions, C
                 'lapIndex', # index of lap, C
                 'blockIndex', # index of block, C
                 # trial level performance information
                 'tone_prob', # tone reward probability, C 
                 'restaurant', # restaurant ID, C
                 'accept', # boolean for accept or reject choices, C
                 'reward', # indicator for whether a trial is rewarded, C
                 # trial timing information
                 'stimulation_on', # stimulation on timestamp, adding stim, stim{t-1}, ARC
                 'stimulation_off', # similar to prev, ARC
                 'tone_onset', # timestamp for onset of offer tones, cannot be null, RC
                 'T_Entry', # timestamp for T junction entry in RR_maze, cannot be null, RC
                 'choice', # timestamp for T junction entry in RR_maze, cannot be null, RC
                 'outcome', # timestamp for when outcomes are revealed (pellets or no pellet), RC
                 'quit', # timestamp for when animals quit an offer, RC
                 'collection', # timestamp for when animals collect pellets, RC
                 'trial_end', # timestamp when trial ends, RC
                 'exit', # timestamp when animal exits the restaurant, RC
                #  # task level timing information
                #  'time_in', # time for the start of session, 24-format time HH:MM:00, derive to floating number A
                #  'time_out', # time for the end of session, 24-format time HH:MM:00
                #  'duration', # duration in HH:mm:ss format, derive to floating number A
                #  'tmax', # max time duration in `bmat.time_unit`, should be close to `duration`
                #  # task level performance information
                #  'pre_task_weight_percent', # similar to post task weight percentage, C
                #  'banana', # number of banana pellets earned, C
                #  'plain', # number of plain pellets earned, C
                #  'grape', # number of grape pellets earned, C
                #  'chocolate', # number of chocolate pellets earned, C
                #  'supplement_g', # amount of food supplement, in grams, C
                #  'total_food_g', # amount of food obtained in total, in grams
                #  'post_task_weight_percent',  # percentage of post task weight compared to adlib
                 'hazard', # indicator variable for problematic data -> current strategy drop hazard != 0
                 'sex', # sex of animal, same for all animals
                 'weightP0', # Starting weight post treatment
                 'left_region', # implant location for ROI `left hemisphere` -> potentially need more detailed coordinate
                 'left_virus', # virus choice for ROI `left hemisphere`
                 'right_region', # similar to prev
                 'right_virus', # similar to left ROI
                 'fiber_NA', # NA for fiber, useful for quality assurance
                 'cell_type', # cell type information
                 # Derived features
                 'roi', # value: None, specifically added for optostim
                 'express_day', # number of day since expression, C
                 'stimON', # derive feature stimType AC
                 'stimType', # describes stimulation type for analysis, baseline, postStim, stim, nostim
                 'exit{t-1}', # timestamp for previous exit, C
                 'hall_time', # timestamp for animal to traverse the hall after tone onset, C
                 'decision_time', # timestamp for animal to make a decision after tone onset, C
                 'quit_time', # time it takes for animal to make quit decision after accepting, C
                 'stim_dur' # duration for stimulation, C
                ]
    # nb_df 
    for t_field in ['date', 'implant_date']:
        if nb_df[t_field].dtype == 'object':
            nb_df[t_field] = pd.to_datetime(nb_df[t_field], format='%m/%d/%Y')
    nb_df['express_day'] = (nb_df['date'] - nb_df['implant_date']).dt.days
    # classify stim trials
    nb_df['roi'] = 'none'
    nb_df['stimON'] = ~nb_df['stimulation_on'].isnull()
    nb_df = expr.nbm.lag_wide_df(nb_df, {'stimON': {'pre': 1}, 'exit': {'pre': 1}}).reset_index(drop=True)
    nb_df.loc[pd.isnull(nb_df['stimON{t-1}']), 'stimON{t-1}'] = False
    nb_df['stimON{t-1}'] = nb_df['stimON{t-1}'].astype(bool)
    nb_df['stimType'] = 'postStim'
    nb_df.loc[nb_df['opto_stim'] == 0, 'stimType'] = 'baseline'
    stimday_sel = nb_df['opto_stim'] == 1
    # assert not np.any(nb_df['stimON'] & nb_df['stimON{t-1}']), 'should never have two consecutive stimONs'
    nb_df.loc[stimday_sel & (~nb_df['stimON{t-1}']) & (~nb_df['stimON']), 'stimType'] = 'nostim'
    nb_df.loc[stimday_sel & nb_df['stimON'], 'stimType'] = 'stim'
    # Dropping trials with null T_Entry/choice
    original = len(nb_df)
    nb_df = nb_df.dropna(subset=['tone_onset', 'T_Entry', 'choice'])
    if len(nb_df) < original:
        logging.warning(f'dropping {original - len(nb_df)} rows that contains NaN timestamps')
    drop_mask1 = check_sequence_rule(nb_df, ['tone_onset', 'T_Entry', 'choice', 'exit'])
    drop_mask2 = check_sequence_rule(nb_df, ['tone_onset', 'T_Entry', 'choice', 'outcome', 'collection'])
    drop_mask3 = check_sequence_rule(nb_df, ['tone_onset', 'T_Entry', 'choice', 'quit'])
    drop_mask = drop_mask1 | drop_mask2 | drop_mask3
    if np.sum(drop_mask) > 0:
        logging.warning(f'dropping {np.sum(drop_mask)} rows that violate task structure')
    nb_df = nb_df[~drop_mask].reset_index(drop=True)
    
    # obtain derived task feature
    nb_df['hall_time'] = nb_df['T_Entry'] - nb_df['tone_onset']
    nb_df['decision_time'] = nb_df['choice'] - nb_df['tone_onset']
    nb_df['quit_time'] = nb_df['quit'] - nb_df['choice']
    nb_df['stim_dur'] = nb_df['stimulation_off'] - nb_df['stimulation_on']

    # convert_to_relative_time, TODO
    time_cols = ['stimulation_on', # stimulation on timestamp, adding stim, stim{t-1}, ARC
                 'stimulation_off', # similar to prev, ARC
                 'tone_onset', # timestamp for onset of offer tones, cannot be null, RC
                 'T_Entry', # timestamp for T junction entry in RR_maze, cannot be null, RC
                 'choice', # timestamp for T junction entry in RR_maze, cannot be null, RC
                 'outcome', # timestamp for when outcomes are revealed (pellets or no pellet), RC
                 'quit', # timestamp for when animals quit an offer, RC
                 'collection', # timestamp for when animals collect pellets, RC
                 'trial_end', # timestamp when trial ends, RC
                 'exit', # timestamp when animal exits the restaurant, RC
                 'exit{t-1}' # timestamp when animal 
                 ]
    
    return nb_df.loc[nb_df['hazard'] == 0, data_cols].reset_index(drop=True)


def get_rr_correlation_matrix(nb_df, dendo=True):
    data_corr_cols = ['age', # age for animals at the session, C
                    'rr_day', # training day number, C
                    'opto_stim', # indicator for whether opto stim is on, A
                    'trial', # trial number within sessions, C
                    'lapIndex', # index of lap, C
                    'blockIndex', # index of block, C
                    # trial level performance information
                    'tone_prob', # tone reward probability, C 
                    'restaurant', # restaurant ID, C
                    'accept', # boolean for accept or reject choices, C
                    'reward', # indicator for whether a trial is rewarded, C
                    # task level timing information
                    # 'tmax', # max time duration in `bmat.time_unit`, should be close to `duration`, C
                    # task level performance information
                    # 'pre_task_weight_percent', # similar to post task weight percentage, C
                    # 'banana', # number of banana pellets earned, C
                    # 'plain', # number of plain pellets earned, C
                    # 'grape', # number of grape pellets earned, C
                    # 'chocolate', # number of chocolate pellets earned, C
                    # 'supplement_g', # amount of food supplement, in grams, C
                    # 'post_task_weight_percent',  # percentage of post task weight compared to adlib, C
                    # 'weightP0', # Starting weight post treatment, C
                    # # Derived features
                    # 'express_day', # number of day since expression, C
                    'hall_time', # timestamp for animal to traverse the hall after tone onset, C
                    'decision_time', # timestamp for animal to make a decision after tone onset, C
                    'quit_time', # time it takes for animal to make quit decision after accepting, C
                    'stimON', # derive feature stimType AC
                    'stim_dur' # duration for stimulation, C
                    ]
    
    corr_df = nb_df[data_corr_cols].astype(float).reset_index(drop=True)
    return plot_correlation(corr_df, dendo)


def preprocess_data(nb_df, expr, dtime_thres=95):
    nb_df = nb_df.groupby('animal').apply(lambda x: x[x['decision_time'] <= np.percentile(x['decision_time'].values, dtime_thres)]).reset_index(drop=True)
    return nb_df