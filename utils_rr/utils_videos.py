import os
import imageio
import re
from os.path import join as oj


def list_files(dir, type):
    """
    List all files of a certain type in the given dir
    :param dir: directory
    :param type: str
    :return:
    """
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith(type):
                r.append(os.path.join(root, name))
    return r


def comp(o):
    return int(re.findall(r"\d+", o)[-1])


def find_vid_folder_animal(root, animal):
    d1, d2 = oj(root, "D1"), oj(root, "A2A")
    if animal in os.listdir(d1):
        return oj(d1, animal, "video")
    elif animal in os.listdir(d2):
        return oj(d2, animal, "video")
    else:
        return None


def jpg_to_vid_folder(folder, output):
    """converts a folder of jpg files to video"""
    img_array = []
    mouseID = str(folder).split(os.sep + "video")[0].split(os.sep)[-1]
    name = output + os.sep + mouseID + "_" + str(folder).split(os.sep)[-1] + ".avi"
    if os.path.isfile(name):
        return None
    if os.path.isdir(folder):
        if not any(fname.endswith(r".jpg") for fname in os.listdir(folder)):
            return None
        jpg = [oj(folder, f) for f in os.listdir(folder) if f.endswith("jpg")]
        print(jpg)
        jpg.sort(key=comp)
    else:
        return None
    return jpg


def jpg_to_vid(folder, output):
    """converts a folder of jpg files to video"""
    img_array = []
    mouseID = str(folder).split(os.sep + "video")[0].split(os.sep)[-1]
    if not os.path.exists(output):
        os.makedirs(output)
    outname = output + os.sep + mouseID + "_" + str(folder).split(os.sep)[-1] + ".avi"
    if os.path.isfile(outname):
        return None

    if os.path.isdir(folder):
        if not any(fname.endswith(r".jpg") for fname in os.listdir(folder)):
            return None
        tImg_inds = sorted(
            [
                int(re.match(r"frame(\d+).jpg", f).group(1)) - 1
                for f in os.listdir(folder)
                if f.endswith(".jpg")
            ]
        )
    else:
        return None

    print(tImg_inds[:5], tImg_inds[-5:])
    w = imageio.get_writer(
        outname, format="FFMPEG", mode="I", fps=30, macro_block_size=8
    )

    fparts = folder.split(os.path.sep)
    dpath = os.path.join(*fparts[:-1])
    name = fparts[-1]
    cam, sess, _ = name.split("_")
    vidf = [
        oj(dpath, f)
        for f in os.listdir(dpath)
        if f.startswith(cam[3:5] + f"_cam_{sess}")
    ][0]
    print("processing ", vidf, "and", folder)
    vidObj = imageio.get_reader(vidf)

    for i in tImg_inds:
        iframe = vidObj.get_data(i)
        w.append_data(iframe)
    w.close()
    # out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'XVID'), 15, size)
    # for i in range(len(img_array)):
    #     out.write(img_array[i])
    # out.release()
    print("saving to ", name)


def main():
    root_save = r"Z:\Restaurant Row\Data\D1\RRM030\video"
    out_path = r"D:\U19\data\sleap_videos"
    for fd in os.listdir(root_save):
        folder = oj(root_save, fd)
        if os.path.isdir(folder) and fd.endswith("_Turns"):
            print(folder)
            jpg_to_vid(folder, out_path)


if __name__ == "__main__":
    main()
