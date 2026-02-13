import os
import pandas as pd

def append_exp_info(df, video_folder, filename, logger):
    """
    Appends specific columns from a CSV file to the input DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to which columns will be appended.
    - video_folder (str): The directory where the CSV file is located.
    - h5_file (str): The H5 file name, which determines the CSV file name.

    Returns:
    - pd.DataFrame: The DataFrame with the appended columns.
    """
    # Construct the CSV file name based on the H5 file name
    csv_file_name = filename.split('.')[0] + '.csv'
    csv_file_path = os.path.join(video_folder, csv_file_name)

    if not os.path.exists(csv_file_path):
        logger.info(f"CSV file {csv_file_path} does not exist.")
        return df

    # Read the CSV file into a DataFrame
    csv_df = pd.read_csv(csv_file_path)

    # Check if the number of rows match
    if len(df) != len(csv_df):
        logger.info("The number of rows in the input DataFrame and the CSV file do not match.")
        return df

    # Append the columns to the input DataFrame
    appended_df = pd.concat([df, csv_df], axis=1)

    return appended_df
