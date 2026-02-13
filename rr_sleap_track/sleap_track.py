import os
from logger import setup_logger 
from transform_coordinates import warp_coordinates
from track_videos import track_videos
from convert_slp_files import convert_slp_files
from append_exp_info import append_exp_info

def sleap_track(video_root, model_path, output_root_folder, video_list=None):
    '''
    Track all videos under the input folder with the assigned model. Then output a '_predictions.slp' file and an 
    '_analysis.h5' file for each video. Then warp the coordinates and output a csv file for each video.
    '''
    
    # Create the raw_track folder in the root video folder
    raw_track_folder = os.path.join(output_root_folder, 'raw_track')
    if not os.path.isdir(raw_track_folder):
        os.makedirs(raw_track_folder)
        
    # Logger setup
    logger_location = os.path.join(video_root, 'logger')
    if not os.path.isdir(logger_location):
        os.makedirs(logger_location)
    logger = setup_logger(os.path.basename(video_root), logger_location)
    
    # Iterate through animal folders and session folders in root folder
    for animal in os.listdir(video_root):
        animal_path = os.path.join(video_root, animal)
        if os.path.isdir(animal_path):
            for session in os.listdir(animal_path):
                session_path = os.path.join(animal_path, session)
                if os.path.isdir(session_path):
                    # Track videos in the folder
                    track_videos(session_path, model_path, logger, video_list)

                    # Convert SLEAP files in the folder
                    convert_slp_files(session_path, logger)

                    for filename in os.listdir(session_path):
                        if filename.endswith('analysis.h5'):
                            input_h5 = os.path.join(session_path, filename)
                            
                            # Warp all coordinates to align the videos from different cameras
                            df = warp_coordinates(input_h5, logger)
                            
                            appended_df = append_exp_info(df, session_path, filename, logger)

                            # Video name should start with RRMxxx_Dayxxx_Rx_
                            output_folder = os.path.join(raw_track_folder, animal, session)
                            if not os.path.isdir(output_folder):
                                os.makedirs(output_folder)
                                
                            parts = filename.split('_')
                            base = '_'.join(parts[:3])
                            output_path = os.path.join(output_folder, base + '_tracks_raw.csv')
                            
                            appended_df.to_csv(output_path, index=False)
                            
                            logger.info(f'Preprocess done for: {base}')
                            