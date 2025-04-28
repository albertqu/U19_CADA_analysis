import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline

def smooth_data(df, columns):
    """
    Apply a smoothing spline to each trial segment.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data to be smoothed and a 'trial' column.
    columns: A list of columns in df to be smoothed
    
    Returns:
    pandas.DataFrame: A new smoothed DataFrame containing only the rows that meet the trial requirements.
    """
    def smooth_trial(trial_df):
        trial_len = len(trial_df)
        grid = np.linspace(1, trial_len, trial_len)
        
        for column in columns:
            smoothing_column = trial_df[column].values
            spline = make_smoothing_spline(grid, smoothing_column)
            trial_df[column] = spline(grid)  # Update the smoothed column in place
        
        return trial_df

    smoothed_df = df.groupby('trial', group_keys=True).apply(smooth_trial).reset_index(drop=True)
    
    return smoothed_df