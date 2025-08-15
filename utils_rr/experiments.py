import sys
sys.path.append("..")
from utils_rr.utils_causal import *
from os.path import join as oj
from utils_rr.utils_tracks import linear_warp_interp_tracks
from neurobehavior_base import RR_Expr
from utils_rr.utils_prep import *
from utils_rr.utils_loading import load_data_sessions_gcamp, load_neural_data, load_sleap_data, add_hall_summary_stats
RESAMP_INTV = 50/ 1000


def track_propensity_fitmodels():

    def pivot_tracks_long2wide(warp_tracks, value_cols):
        # assume time column is 'rel_time', and has been renormalized to [0, 1] w.r.t. hall time for all tracks
        # or consistent across tracks respectively
        # Pivot the dataframe to get x and y values for each time point as separate columns
        pvt_dfs = []
        for vcol in value_cols:
            pivoted = warp_tracks.pivot(index='track_id', columns='rel_time', values=vcol)
            # Rename the columns to x_t1, x_t2, ..., x_t21
            pivoted.columns = [f'{vcol}_t{col:.2f}' for col in pivoted.columns]
            pvt_dfs.append(pivoted)
        # Merge the pivoted x and y dataframes
        pivoted_tracks = pd.concat(pvt_dfs, axis=1).reset_index()
        # Add the 'accept' column
        aux_cols = ['track_id', 'slp_accept', 'slp_decision', 'restaurant']
        accept_col = warp_tracks[aux_cols].drop_duplicates()
        pivoted_tracks = pivoted_tracks.merge(accept_col, on='track_id')
        return pivoted_tracks
    
    # load data
    # sleap_bdf_root = r'D:\U19\data\RR\ARJ_raw'
    # track_root = r"Z:\Restaurant Row\Data\processed_tracks"
    # cache_folder = r'D:\U19\data\RR\caching'
    data_root = r'C:\wilbrechtlab\U19\data\RR\ARJ_raw'
    track_root = r"Z:\Restaurant Row\Data\processed_tracks"
    cache_folder = r'C:\wilbrechtlab\U19\data\RR\caching'
    plot_folder = r"C:\wilbrechtlab\U19\data\RR\plots\RR_paper"
    rse = RR_Expr(data_root)
    sessions = load_data_sessions_gcamp()
    sleap_bdf, all_track_df, raw_track_df = load_sleap_data(rse, cache_folder, 
                                                            track_root, sessions, RESAMP_INTV)
    method = "ZdF_jove" #"lossless"
    nb_df = load_neural_data(rse, cache_folder, sessions, method='ZdF_jove')

    id_cols = ['animal', 'session', 'trial']
    aug_cols = list(np.setdiff1d(sleap_bdf.columns, nb_df.columns))
    neur_df = nb_df.merge(sleap_bdf[id_cols+aug_cols], on=id_cols, how='left')
    neur_df = add_hall_summary_stats(rse, neur_df)

    pd.set_option('future.no_silent_downcasting', True)
    sleap_bdf['track_id'] = sleap_bdf['animal'] + '_' + sleap_bdf['session'] + '_' + sleap_bdf['trial'].astype(str)
    sleap_bdf['slp_DT'] = sleap_bdf['slp_choice_time'] - sleap_bdf['slp_T_Entry_time']
    lastdf = sleap_bdf[['animal', 'session', 'trial', 'track_id']].groupby(['animal', 'session']).last().reset_index()
    lastdf['lastTrial'] = True
    sleap_bdf = sleap_bdf.merge(lastdf[['track_id', 'lastTrial']], on='track_id', how='left')
    sleap_bdf['lastTrial'] = sleap_bdf['lastTrial'].infer_objects(copy=False).fillna(False)
    sleap_bdf['offer_rank'] = sleap_bdf['offer_prob'].rank(method='dense').astype(float)
    sleap_bdf['decision2'] = sleap_bdf['slp_decision'].map({'accept': 'acc', 'reject': 'rej', 'quit': 'acc'})
    sleap_bdf['commit'] = sleap_bdf['outcome'].notnull().astype(float)
    sleap_bdf['ACC'] = ((sleap_bdf['commit'] == 1) | (sleap_bdf['slp_accept'] == 1)).astype(float)
    sleap_bdf['REJ'] = ((sleap_bdf['commit'] == 0) & (sleap_bdf['slp_accept'] == 0)).astype(float)
        
    # get raw track df
    hall_time_cutoff = 1.5
    raw_track_df['track_id'] = raw_track_df['animal'] + '_' + raw_track_df['session'] + '_' + raw_track_df['trial'].astype(str)
    sleap_bdf['slp_DT'] = sleap_bdf['slp_choice_time'] - sleap_bdf['slp_T_Entry_time']
    task_variables = ['slp_hall_time', 'restaurant', 'offer_prob', 'slp_decision', 'slp_accept', 'slp_DT', 'reward']
    tracks_df = raw_track_df.merge(sleap_bdf[id_cols+task_variables], on=id_cols, how='left')
    warp_col_map = {c: '_'.join(c.split(' ')) for c in tracks_df.columns if 'warped' in c}
    tracks_df = tracks_df.rename(columns=warp_col_map)
    DT = 0.05
    halltracks = tracks_df[(tracks_df['rel_time'] <= tracks_df['slp_hall_time'] + DT) &
                            (tracks_df['slp_hall_time'] <= hall_time_cutoff)].reset_index(drop=True)
    value_cols = ['warped_Head_x', 'warped_Head_y', 'warped_Torso_x', 'warped_Torso_y',
                                                                    'warped_Tailhead_x', 'warped_Tailhead_y']
    value_cols = ['warped_Head_x', 'warped_Head_y', 'warped_Neck_x', 'warped_Neck_y', 
                'warped_Torso_x', 'warped_Torso_y', 'warped_Tailhead_x', 'warped_Tailhead_y']
    warped_tracks = linear_warp_interp_tracks(halltracks, value_cols, dt=DT, method='cubicn')
    warped_tracks = warped_tracks.merge(sleap_bdf[['track_id'] + task_variables], on='track_id', how='left')
    pivot_tracks = pivot_tracks_long2wide(warped_tracks, value_cols)
    model_folder = oj(cache_folder, 'models')
    model_combined_grid_search(X, y, model_folder=model_folder, restaurant_id=None)


if __name__ == "__main__":
    track_propensity_fitmodels()