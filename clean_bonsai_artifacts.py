import os, shutil, stat, sys
from datetime import datetime

def clean_file_with_keywords(keyword, filepaths=None):
    if filepaths is None:
        filepaths = [os.path.join(*keyword.split(os.sep)[:-1])]
        keyword = keyword.split(os.sep)[-1]
    for fpath in filepaths:
        for f in os.listdir(fpath):
            targetf = os.path.join(fpath, f)
            tdelta = datetime.now() - datetime.fromtimestamp(os.path.getctime())
            if (keyword in f) and (tdelta.seconds < 300):
                print("Deleting", targetf)
                os.remove(targetf)

                
if __name__ == '__main__':
    clean_file_with_keywords(sys.argv[1], ['rr_data', 'rr_data_FP', 'rr_video'])
    
