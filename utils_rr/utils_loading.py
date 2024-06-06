import pandas as pd


def calculate_neur_diffs_RR(nb_df, events, sessions, expr):
    """
    Customary function to calculate differences between the left and right ROI neural responses
    around `events` for all `sessions`
    nb_df: pd.DataFrame
        neurobehavior dataframe
    events: list
        list of behavior events
    sessions: dict
        dictionary describing all animal, sessions, 3 means bilateral recordings
    expr: neurobehavior experiment object
        experiment object used to organize data frames
    """
    expr.nbm.nb_cols, expr.nbm.nb_lag_cols = expr.nbm.parse_nb_cols(nb_df)
    nb_diff_dfs = []

    for animal in sessions:
        for sess in sessions[animal]:
            if sessions[animal][sess] == 3:
                session = f"Day{sess}"
                nb_df_session = nb_df[
                    (nb_df["animal"] == animal) & (nb_df["session"] == session)
                ].reset_index(drop=True)
                nb_df_session["roi"] = nb_df_session["roi"].str.replace("_470nm", "")
                basic_cols = [c for c in nb_df_session.columns if "_neur|" not in c]
                nb_result = nb_df_session.loc[
                    nb_df_session["roi"] == "left", basic_cols
                ].reset_index(drop=True)
                nb_result["roi"] = "diff"
                for event in events:
                    ev_cols = expr.nbm.nb_cols[f"{event}_neur"]
                    leftVs = nb_df_session.loc[
                        nb_df_session["roi"] == "left", ev_cols
                    ].values
                    rightVs = nb_df_session.loc[
                        nb_df_session["roi"] == "right", ev_cols
                    ].values
                    nb_result[ev_cols] = 0
                    nb_result[ev_cols] = leftVs - rightVs
                nb_diff_dfs.append(nb_result)
    nb_diff_df = pd.concat(nb_diff_dfs, axis=0)
    return nb_diff_df
