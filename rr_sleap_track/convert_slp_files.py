import os
import subprocess

def convert_slp_files(video_path, logger):
    '''
    Convert all '_prediction.slp' files into '_analysis.h5' files
    '''
    for filename in os.listdir(video_path):
        if filename.endswith('predictions.slp'):
            input_video = os.path.join(video_path, filename)
            base, _ = os.path.splitext(filename)
            output_path = os.path.join(video_path, base + '_analysis.h5')
            
            logger.info(f"Converting file: {filename}")
            
            # Build the command
            command = [
                'sleap-convert',
                input_video,
                '--format', 'analysis',
                '-o', output_path
            ]

            # Run the command
            try:
                subprocess.run(command, check=True)
                logger.info(f"Successfully converted to: {str(filename) + '_analysis.h5'}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error converting {filename}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error converting {filename}: {e}")

    logger.info(f"---Conversion complete for all files in {os.path.split(video_path)[1]}.---")