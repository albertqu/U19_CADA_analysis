#!/usr/bin/env python3

from utils import decode_from_filename
import os
import shutil
import pandas as pd
import numpy as np

IN_PATH = "/Volumes/Wilbrecht_file_server/2ABT/ProbSwitch/ProbSwitch_Raw"
OUT_PATH = os.path.join(os.getcwd(), "animals")

decoded_files = []


def group_by(data, key_extractor):
    groups = {}
    for d in data:
        key = key_extractor(d)

        data_for_key = []
        if key in groups:
            data_for_key = groups[key]
        else:
            groups[key] = data_for_key

        data_for_key.append(d)
    return groups


def summarize_sessions(implant_csv_path=None, sort_key="aID"):
    """
    implant_csv: pd.DataFrame from implant csv file
    """
    data = {
        "animal": [],
        "aID": [],
        "session": [],
        "date": [],
        "ftype": [],
        "age": [],
        "FP": [],
        "region": [],
        "note": [],
    }

    implant_lookup = {}
    if implant_csv_path:
        implant_df = pd.read_csv(implant_csv_path)
        for i in range(len(implant_df)):
            animal_name = implant_df.loc[i, "Name"]
            if animal_name and (str(animal_name) != "nan"):
                LH_target = implant_df.loc[i, "LH Target"]
                RH_target = implant_df.loc[i, "RH Target"]
                print(animal_name)
                name_first, name_sec = animal_name.split(" ")
                name_first = "-".join(name_first.split("-")[:2])
                implant_lookup[name_first + "_" + name_sec] = {
                    "LH": LH_target,
                    "RH": RH_target,
                }

    for decoded_file in decoded_files:
        f = decoded_file["name"]
        options = decoded_file["decoded"]
        if options is None:
            pass
        elif ("FP_" in f) and ("FP_" not in options["session"]):
            print(f, options["session"])
        else:
            for q in ["animal", "ftype", "session"]:
                data[q].append(options[q])

            name_first2, name_sec2 = options["animal"].split("_")
            name_first2 = "-".join(name_first2.split("-")[:2])
            aID = name_first2 + "_" + name_sec2
            data["aID"].append(aID)
            data["date"].append(options["T"])
            opts = options["session"].split("_FP_")
            data["age"].append(opts[0])
            if len(opts) > 1:
                data["FP"].append(opts[1])
                if aID not in implant_lookup:
                    # print('skipping', options['animal'], options['session'])
                    data["region"].append("")
                else:
                    data["region"].append(implant_lookup[aID][opts[1]])
            else:
                data["FP"].append("")
                data["region"].append("")
            data["note"].append(options["DN"] + options["SP"])

    apdf = pd.DataFrame(data)
    sorted_pdf = apdf.sort_values(["date", "session"], ascending=True)
    sorted_pdf["S_no"] = 0
    for anim in sorted_pdf[sort_key].unique():
        tempslice = sorted_pdf[sorted_pdf[sort_key] == anim]
        sorted_pdf.loc[sorted_pdf[sort_key] == anim, "S_no"] = np.arange(
            1, len(tempslice) + 1
        )

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    sorted_pdf.to_csv(
        os.path.join(OUT_PATH, f"exper_list_final_{sort_key}.csv"), index=False
    )


def copy_files():
    grouped_by_animal = group_by(
        decoded_files,
        lambda data: data["decoded"]["animal"]
        if data["decoded"] and "animal" in data["decoded"]
        else "_NONE",
    )

    print("Copying files...")

    for animal, animal_files in grouped_by_animal.items():
        animal_folder = os.path.join(OUT_PATH, animal)
        if not os.path.exists(animal_folder):
            os.makedirs(animal_folder)

        grouped_by_session = group_by(
            animal_files,
            lambda data: data["decoded"]["session"]
            if data["decoded"] and "session" in data["decoded"]
            else "_NONE",
        )

        for session, session_files in grouped_by_session.items():
            session_folder = os.path.join(animal_folder, session)
            if not os.path.exists(session_folder):
                os.makedirs(session_folder)

            for decoded_file in session_files:
                out_filepath = os.path.join(session_folder, decoded_file["name"])
                shutil.copy2(decoded_file["path"], out_filepath)
                print("Copied")


def decode_files_in_folder(root):
    for filename in os.listdir(root):
        # gross
        if filename == ".DS_Store":
            continue

        filepath = os.path.join(root, filename)

        # recurse on folders
        if os.path.isdir(filepath):
            decode_files_in_folder(filepath)
            continue

        decoded_files.append(
            {
                "name": filename,
                "path": filepath,
                "decoded": decode_from_filename(filepath),
            }
        )


# prepare files

print("Decoding files...")
decode_files_in_folder(IN_PATH)

# do stuff

print("Summarizing...")
summarize_sessions()

# print("Copying...")
# copy_files()
