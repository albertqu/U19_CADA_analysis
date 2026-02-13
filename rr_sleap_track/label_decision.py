import numpy as np
import pandas as pd

def label_decision(df):
    """
    TODO should use groupby and apply function
    Updates the DataFrame in real time, adding a new column with the current state
    (T-entry, Acc, Rej, quit) for each timepoint.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the smoothed data.
                           It is assumed to have 'warped Head x' and 'warped Head y' columns.

    Returns:
    pandas.DataFrame: A DataFrame with the decision for each timepoint.
    """
    current_trial = None
    decisions = []
    decision = None
    last_decision = None
    
    for index, row in df.iterrows():
        x = row['warped Head x']
        y = row['warped Head y']
        trial = row['trial']
        
        if current_trial != trial: # 0>_000_0 Start of a new trial
            decision = None
            last_decision = None
            current_trial = trial

        if (decision == None):
            if y < 46: # entering T-entry
                decision = 'T-Entry'

        if decision == 'T-Entry':
            if x > 309:
                decision = 'ACC'
            elif x < 282:
                decision = 'REJ'

        if decision == 'ACC':
            if x < 282:
                decision = 'quit'

        # Only append the decision if it has changed, otherwise append None
        if decision != last_decision:
            decisions.append(decision)
            last_decision = decision
        else:
            decisions.append(None)     
    
    # Add the decisions column to the DataFrame
    df['decision'] = decisions

    return df
