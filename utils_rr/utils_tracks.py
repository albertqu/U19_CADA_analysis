import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import make_interp_spline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
Track Data Processing Library
============================

This library contains a series of functions for processing, interpolating, and visualizing track data.
Track data is stored in pandas DataFrames where each track is identified by a unique 'track_id' 
and organized in long format.

The library is organized into four main categories of functions:
1. Helper Functions - Utility functions for data manipulation
2. ID Track Functions - Functions that operate on single tracks identified by track_id
3. Group-wise Functions - Functions that operate on groups of tracks
4. Plot Functions - Functions for visualizing track data

Typical function signature:
    track_func(tracks_df, **kwargs)

This library is designed for processing motion tracking data, particularly for analyzing
movement trajectories with multiple keypoints (Head, Torso, Tailhead).
"""

###############################################
############## Helper Functions ###############
###############################################

def combine_apply_trackid_functions(tracks_df, funcs):
    """ 
    Apply multiple processing functions in sequence to each track in a DataFrame.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame containing multiple tracks identified by 'track_id'.
    funcs : list of tuples
        List of tuples in the format (function, kwargs), where function is a processing
        function and kwargs is a dictionary of keyword arguments to pass to the function.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with all processing functions applied to each track.
    """
    def idtrack_helper(idtrack):
        result = idtrack.copy()
        for f, kwargs in funcs:
            result = f(result, **kwargs)
        return result
    return tracks_df.groupby('track_id', as_index=False).apply(idtrack_helper).reset_index(drop=True)

def mirror_pad(ts, ys):
    """
    Pad a time series with mirrored values to improve interpolation at boundaries.
    
    Parameters
    ----------
    ts : numpy.ndarray
        Array of time points.
    ys : numpy.ndarray
        Array of values corresponding to the time points.
        
    Returns
    -------
    new_t : numpy.ndarray
        Padded time array with mirrored points added before the original time points.
    new_y : numpy.ndarray
        Padded values array with mirrored points added before the original values.
    
    Notes
    -----
    This function mirrors the early portion of the time series around ts[0] to create
    a padded version that helps improve interpolation accuracy at the boundaries.
    """
    # Assuming ts is sorted, pad ys with mirrored values around ts[0] with the minimum entry time < 0
    t0 = ts[0]
    t1_ind = np.where(ts > 2*t0)[0][0]
    tseg = ts[1:t1_ind+2]
    yseg = ys[1:t1_ind+2]
    pad_t = 2 * t0 - tseg
    new_t = np.concatenate((pad_t[::-1], ts))
    new_y = np.concatenate((yseg[::-1], ys))
    return new_t, new_y

def interp_xy(x, y, xnew, method='cubicn'):
    """
    Interpolate time series data to new time points using various methods.
    
    Parameters
    ----------
    x : numpy.ndarray
        Original x-values (typically time points).
    y : numpy.ndarray
        Original y-values to be interpolated.
    xnew : numpy.ndarray
        New x-values to interpolate to.
    method : str, default='cubicn'
        Interpolation method to use. Options:
        - 'linear': Linear interpolation
        - 'bspline': B-spline interpolation with k=3
        - 'mirror_bsp': B-spline interpolation with mirrored padding
        - 'cubicn': Cubic spline with natural boundary conditions
        - 'cubicc': Cubic spline with clamped boundary conditions
        
    Returns
    -------
    ynew : numpy.ndarray
        Interpolated y-values corresponding to xnew.
    """
    if method == 'linear':
        ynew = np.interp(xnew, x, y)
    if method == 'bspline':
        bspl = make_interp_spline(x, y, k=3)
        ynew = bspl(xnew)
    elif method == 'mirror_bsp':
        x_p, y_p = mirror_pad(x, y)
        bspl = make_interp_spline(x_p, y_p, k=3)
        ynew = bspl(xnew)
    elif method == 'cubicn':
        cs = CubicSpline(x, y, bc_type='natural')
        ynew = cs(xnew)
    elif method == 'cubicc':
        cs = CubicSpline(x, y, bc_type='clamped')
        ynew = cs(xnew)
    return ynew

def tracks_in_bounds(tracks_df):
    """ 
    This is a useful utility function that checks whether a track is in the bounds of the arena

    Examples
    --------
    >>> warped_tracks[~tracks_in_bounds(warped_tracks)] # should return an empty dataframe
    """
    ybound = [0, 400]
    xbound = [200, 800]
    x_in = (tracks_df['warped_Head_x'] >= xbound[0]) & (tracks_df['warped_Head_x'] < xbound[1])
    y_in = (tracks_df['warped_Head_y'] > ybound[0]) & (tracks_df['warped_Head_y'] < ybound[1])
    return x_in & y_in

########################################
########### ID Track Functions #########
########################################

def timewarp_interp_idtrack(tracks_df, value_cols, dt, method='cubicn'):
    """ 
    Normalize time to [0, 1] range and interpolate data for a single track.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame containing a single track.
    value_cols : list of str
        List of column names to interpolate.
    dt : float
        Time step for the new interpolated time points.
    method : str, default='cubicc'
        Interpolation method passed to interp_xy.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with normalized time and interpolated values.
        
    Notes
    -----
    This function normalizes the 'rel_time' column by dividing by 'slp_hall_time',
    then creates a new time grid from 0 to 1 and interpolates all value columns
    to this new time grid.
    """
    tracks_df['rel_time'] = tracks_df['rel_time'] / tracks_df['slp_hall_time'].iloc[0]
        
    # Create new time points for interpolation
    new_time_points = np.arange(0, 1+dt, dt)
    
    # Interpolate each value column
    interpolated_data = {'rel_time': new_time_points}
    for col in value_cols:
        # TODO: replace np.interp with constrained spline interpolation (natural splines)
        interpolated_data[col] = interp_xy(tracks_df['rel_time'].values, tracks_df[col].values, new_time_points, method=method)
    
    # Create a new DataFrame for the interpolated data
    interpolated_df = pd.DataFrame(interpolated_data)
    interpolated_df['track_id'] = tracks_df['track_id'].iloc[0]
    return interpolated_df

def interp_idtrack(tracks_df, value_cols, dt, method='cubicc'):
    """
    Interpolate data for a single track to a regular time grid.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame containing a single track.
    value_cols : list of str
        List of column names to interpolate.
    dt : float
        Time step for the new interpolated time points.
    method : str, default='cubicc'
        Interpolation method passed to interp_xy.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with interpolated values at regular time intervals.
        
    Notes
    -----
    Unlike timewarp_interp_idtrack, this function does not normalize time to [0, 1].
    Instead, it creates a new time grid from 0 to the maximum time and interpolates
    all value columns to this new time grid.
    """
    max_t = tracks_df['rel_time'].max()
    new_time_points = np.arange(0, np.floor(max_t / dt) * dt +dt, dt)
    # Interpolate each value column
    interpolated_data = {'rel_time': new_time_points}
    for col in value_cols:
        # TODO: replace np.interp with constrained spline interpolation (natural splines)
        interpolated_data[col] = interp_xy(tracks_df['rel_time'].values, tracks_df[col].values, new_time_points, method=method)
    
    # Create a new DataFrame for the interpolated data
    interpolated_df = pd.DataFrame(interpolated_data)
    interpolated_df['track_id'] = tracks_df['track_id'].iloc[0]
    return interpolated_df

def subtract_initial_idtrack(tracks_df):
    """
    Subtract the initial position from all positions in a single track.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame containing a single track.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with positions relative to the initial position.
        
    Notes
    -----
    This function subtracts the initial x and y values for each keypoint
    (Head, Torso, Tailhead) from all subsequent values, effectively
    making the track start at the origin.
    """
    temp = tracks_df.sort_values('rel_time')
    keypoints = ['Head', 'Torso', 'Tailhead']
    for kp in keypoints:
        for c in 'xy':
            temp[f'warped_{kp}_{c}'] = temp[f'warped_{kp}_{c}']- temp[f'warped_{kp}_{c}'].iat[0]
    return temp

#############################################
######### Group wise functions ##############
#############################################
""" Use combine_apply_trackid_func() if needed to use multiple functions"""

def resamp_interp_tracks(tracks_df, value_cols, dt=0.05, method='cubicc'):
    """
    Resample and interpolate all tracks in a DataFrame to a regular time grid.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame containing multiple tracks identified by 'track_id'.
    value_cols : list of str
        List of column names to interpolate.
    dt : float, default=0.05
        Time step for the new interpolated time points.
    method : str, default='cubicc'
        Interpolation method passed to interp_xy.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with all tracks interpolated to regular time intervals.
    """
    return tracks_df.groupby('track_id', as_index=False).apply(interp_idtrack, value_cols=value_cols, 
                                                              dt=dt, method=method).reset_index(drop=True)

def linear_warp_interp_tracks(tracks_df, value_cols, dt=0.05, method='cubicc'):
    """
    Normalize time to [0, 1] and interpolate data for all tracks in a DataFrame.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame containing multiple tracks identified by 'track_id'.
    value_cols : list of str
        List of column names to interpolate.
    dt : float, default=0.05
        Time step for the new interpolated time points.
    method : str, default='cubicc'
        Interpolation method passed to interp_xy.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with normalized time and interpolated values for all tracks.
        
    Notes
    -----
    This function applies timewarp_interp_idtrack to each track in the DataFrame,
    normalizing all tracks to a common time scale from 0 to 1, which facilitates
    comparison between tracks of different durations.
    """
    return tracks_df.groupby('track_id', as_index=False).apply(timewarp_interp_idtrack, #include_groups=False,
                                                               value_cols=value_cols, dt=dt, 
                                                              method=method).reset_index(drop=True)

def subtract_initial(tracks_df: pd.DataFrame):
    """
    Subtract the initial position from all positions for all tracks in a DataFrame.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame with tracks identified by 'track_id', containing columns:
            track_id : str
                Unique identifier for each track.
            rel_time : float
                Time relative to the start of the trial.
            warped_[kp]_x/y : float
                xy-coordinate of the keypoint kp.
                
    Returns
    -------
    pandas.DataFrame
        DataFrame with positions relative to the initial position for all tracks.
    """
    tracks_df = tracks_df.groupby('track_id').apply(subtract_initial_idtrack, 
                                                    include_groups=False).reset_index()
    tracks_df.drop(columns='level_1', inplace=True)
    return tracks_df

#############################################################
##################### Plot functions ########################       
#############################################################

def plot_tracks_xyt(sample_tracks, decision_palette, keypoints=None, legend=None,
                    hue_col='slp_decision', hue_order=None, height=3, width=5):
    """
    Plot x and y coordinates over time for each keypoint.
    
    Parameters
    ----------
    sample_tracks : pandas.DataFrame
        DataFrame containing track data with columns:
        - 'track_id': Identifier for each track
        - 'rel_time': Time points
        - 'warped_[kp]_x/y': x and y coordinates for each keypoint
        - 'slp_decision': Decision category for coloring
    decision_palette : dict or list
        Color palette for the 'slp_decision' categories.
    keypoints : list of str, optional
        List of keypoints to plot. If None, defaults to ['Head', 'Torso', 'Tailhead'].
    hue_col : str, optional
        Column to use for hue encoding. Default is 'slp_decision'.
    width : float, optional
        Width of the figure. Default is 3.
    height : float, optional
        Height of the figure. Default is 2.5.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plots.
    axes : numpy.ndarray
        Array of axes objects in the figure.
        
    Notes
    -----
    Creates a 2xN grid of plots, where N is the number of keypoints.
    The top row shows x-coordinates over time, and the bottom row shows
    y-coordinates over time, for each keypoint.
    """
    if keypoints is None:
        keypoints = ['Head', 'Torso', 'Tailhead']
    fig, axes = plt.subplots(nrows=2, ncols=len(keypoints), figsize=(len(keypoints) * width, height*2))
    if len(keypoints) == 1:
        axes = axes.reshape(2, 1)
    for j in range(len(keypoints)):
        kp = keypoints[j]
        sns.lineplot(data=sample_tracks, x='rel_time', y=f'warped_{kp}_x', hue=hue_col, hue_order=hue_order,
                     palette=decision_palette, style='track_id', linewidth=0.8, legend=legend, ax=axes[0, j])
        sns.lineplot(data=sample_tracks, x='rel_time', y=f'warped_{kp}_y', hue=hue_col, hue_order=hue_order,
                     palette=decision_palette,style='track_id', linewidth=0.8, legend=legend, ax=axes[1, j])
        axes[1, j].invert_yaxis()
    sns.despine()
    fig.subplots_adjust(hspace=0.3, wspace=0.4)
    return fig, axes

def plot_tracks_2D(sample_tracks, decision_palette, keypoints=None, 
                   hue_col='slp_decision', hue_order=None, width=3, height=2.5,
                   legend=None):
    """
    Plot tracks as 2D trajectories for each keypoint.
    
    Parameters
    ----------
    sample_tracks : pandas.DataFrame
        DataFrame containing track data with columns:
        - 'track_id': Identifier for each track
        - 'warped_[kp]_x/y': x and y coordinates for each keypoint
        - 'slp_decision': Decision category for coloring
    decision_palette : dict or list
        Color palette for the 'slp_decision' categories.
    keypoints : list of str, optional
        List of keypoints to plot. If None, defaults to ['Head', 'Torso', 'Tailhead'].
    hue_col : str, optional
        Column to use for hue encoding. Default is 'slp_decision'.
    width : float, optional
        Width of the figure. Default is 3.
    height : float, optional
        Height of the figure. Default is 2.5.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plots.
    axes : numpy.ndarray
        Array of axes objects in the figure.
        
    Notes
    -----
    Creates a 1xN grid of plots, where N is the number of keypoints.
    Each plot shows the 2D trajectory (x vs y) for a specific keypoint.
    """
    from matplotlib.axes import Axes
    if keypoints is None:
        keypoints = ['Head', 'Torso', 'Tailhead']
    fig, axes = plt.subplots(nrows=1, ncols=len(keypoints), figsize=(len(keypoints) * width, height))
    if isinstance(axes, Axes):
        axes = np.array([axes])
    for j in range(len(keypoints)):
        kp = keypoints[j]
        sns.lineplot(data=sample_tracks, x=f'warped_{kp}_x', y=f'warped_{kp}_y', hue=hue_col, 
                     hue_order=hue_order, sort=False,
                     palette=decision_palette, style='track_id', linewidth=0.8, legend=legend, ax=axes[j])
        axes[j].invert_yaxis()
    sns.despine()
    fig.subplots_adjust(hspace=0.3, wspace=0.4)
    return fig, axes

def random_sample(df, id_col, N=100, seed=None):
    """
    Randomly sample a subset of tracks from a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing track data.
    id_col : str
        Column name that identifies unique tracks.
    N : int, default=100
        Number of unique tracks to sample.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing only the randomly sampled tracks.
    """
    if seed is not None:
        np.random.seed(seed)
    ids = df[id_col].unique()
    return df[df[id_col].isin(np.random.choice(ids, N))].reset_index(drop=True)