import os
import pandas as pd
from logger import setup_logger
from label_decision import label_decision
from smooth_data import smooth_data

def track_processing(track_root):
    '''
    Access all '_tracks_raw' csv files, smooth the data and calculate the kinematic 
    parameters and output a csv file for each raw track (all output files are stored in 
    a different folder in the root video folder).
    '''
    
    # Create the processed_tracks folder in the root video folder
    processed_tracks_folder = os.path.join(track_root, 'processed_tracks')
    if not os.path.isdir(processed_tracks_folder):
        os.makedirs(processed_tracks_folder)

    # Logger setup
    logger_location = os.path.join(processed_tracks_folder, 'logger')
    if not os.path.isdir(logger_location):
        os.makedirs(logger_location)
    logger = setup_logger(os.path.basename(track_root), logger_location)

    # Warped columns to be smoothed
    columns_to_smooth = [
        'warped Head x', 'warped Head y',
        'warped Neck x', 'warped Neck y',
        'warped Torso x', 'warped Torso y',
        'warped Tailhead x', 'warped Tailhead y'
    ]
    
    # Bodyparts to calculate velocity
    bodyparts = [
        'Head',
        'Neck',
        'Torso',
        'Tailhead'
    ]
    
    # Iterate through animal folders and session folders in root folder
    for animal in os.listdir(track_root):
        animal_path = os.path.join(track_root, animal)
        if os.path.isdir(animal_path):
            for session in os.listdir(animal_path):
                session_path = os.path.join(animal_path, session)
                if os.path.isdir(session_path):
                    for file in os.listdir(session_path):
                        if file.endswith('_tracks_raw.csv'):
                            raw_filepath = os.path.join(session_path, file)
                            raw_df = pd.read_csv(raw_filepath)
                            
                            # smooth the raw track data
                            smoothed_df = smooth_data(raw_df, columns_to_smooth)
                            
                            # label animal decisions according to coordinates
                            labeled_df = label_decision(smoothed_df)
                            
                            # Assign the output path in the processed_tracks folder
                            parts = file.split('_')
                            base = '_'.join(parts[:3])
                            output_folder = os.path.join(processed_tracks_folder, animal, session)
                            if not os.path.isdir(output_folder):
                                os.makedirs(output_folder)
                            output_path = os.path.join(output_folder, base + '_tracks_processed.csv')
                            
                            labeled_df.to_csv(output_path, index=False)
                            
                            logger.info(f'Preprocess done for: {base}')

