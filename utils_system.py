import os, shutil, stat, errno, time, datetime


def archive_by_date(src_folder, dest_folder, archive_date):
    assert os.path.exists(dest_folder)
    os.chdir(src_folder)
    tomoves = []
    for name in os.listdir(src_folder):
        if (name[0] != '_') and (datetime.datetime.fromtimestamp(os.path.getatime(name)) < archive_date):
            #atime: access time, mtime: modify time
            # print(name, datetime.datetime.fromtimestamp(os.path.getmtime(name)))
            tomoves.append(name)
    for name in tomoves:
        try:
            print("moving", name)
            shutil.move(name, os.path.join(dest_folder, name))
        except Exception:
            print("skipping", name)


if __name__ == '__main__':
    archive_date = datetime.datetime.strptime("2019/05/31", "%Y/%m/%d")
    src_folder = "/Volumes/Wilbrecht_file_server"
    archive = os.path.join(src_folder, '_Archive')
    if not os.path.exists(archive):
        os.makedirs(archive)
    archive_by_date(src_folder, archive, archive_date)

