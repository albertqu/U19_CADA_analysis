import os, shutil, stat, errno, time, datetime

def filter_by_date(src_folder, dest_folder, archive_date):
    os.chdir(src_folder)
    for name in os.listdir(src_folder):
        if datetime.datetime.fromtimestamp(os.path.getmtime(name)) < archive_date:
            print(name, datetime.datetime.fromtimestamp(os.path.getmtime(name)))