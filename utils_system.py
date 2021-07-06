import os, shutil, stat, errno, time, datetime
from utils import path_prefix_free
import imageio


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


def chunk_video_sample_from_file(filename, out_folder, fps, duration):
    assert path_prefix_free(filename).count('.') == 1, 'has to contain only one .'
    file_code = path_prefix_free(filename).split('.')[0]
    suffix = path_prefix_free(filename).split('.')[1]
    w = imageio.get_writer(os.path.join(out_folder, f'{file_code}_sample.{suffix}'), format='FFMPEG',
                           mode='I', fps=fps) # figure out what mode means
    vid = imageio.get_reader(filename)
    for ith, img in enumerate(vid):
        if ith < duration * fps:
            print('writing', img.shape, ith)
            w.append_data(img)
        if ith >= duration * fps:
            print('all done')
            break
    w.close()


if __name__ == '__main__':
    archive_date = datetime.datetime.strptime("2019/05/31", "%Y/%m/%d")
    src_folder = "/Volumes/Wilbrecht_file_server"
    archive = os.path.join(src_folder, '_Archive')
    if not os.path.exists(archive):
        os.makedirs(archive)
    archive_by_date(src_folder, archive, archive_date)

