from sleap_track import sleap_track

# make sure the video name is RRM0xx_Dayxxx_Rx_turns.avi
# make sure there is a CSV file recording the bonsai metadata in the same name and at the same folder as the video
# this program should run in sleap_env
# output raw_track files will be in raw_track folder in output_root_folder
sleap_video_root = r'/home/wholebrain/sleap_models/20240402_RR_n24/Videos'
model_path = r'/home/wholebrain/sleap_models/20240402_RR_n24/models/v003.3_240430_165906.single_instance.n=732'
output_root_folder = r''
video_list = None

sleap_track(sleap_video_root, model_path, output_root_folder, video_list)
