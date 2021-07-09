#!/usr/bin/env python3

from utils import decode_from_filename
import os
import shutil
import pandas as pd
import numpy as np
import re

IN_PATH = "/Volumes/Wilbrecht_file_server/2ABT/ProbSwitch/ProbSwitch_Raw"
IMPLANT_CSV_PATH = "/Volumes/Wilbrecht_file_server/Albert/CADA_data/ProbSwitch_FP_data/ImplantedMice_ProbSwitch.csv"
OUT_PATH = os.path.join(os.getcwd(), "animals")

decoded_files = []
decode_failed = []

txt_files = []


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


def summarize_sessions(sort_key="aID"):
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
        "FP_note": [],
        "trigger_mode": [],
        "LED_power_415": [],
        "LED_power_470": [],
        "LED_power_560": [],
    }

    implant_lookup = {}
    if IMPLANT_CSV_PATH:
        implant_df = pd.read_csv(IMPLANT_CSV_PATH)
        for i in range(len(implant_df)):
            animal_name = implant_df.loc[i, "Name"]
            if animal_name and (str(animal_name) != "nan"):
                LH_target = implant_df.loc[i, "LH Target"]
                RH_target = implant_df.loc[i, "RH Target"]
                # print(animal_name)
                name_first, name_sec = animal_name.split(" ")
                name_first = "-".join(name_first.split("-")[:2])
                implant_lookup[name_first + "_" + name_sec] = {
                    "LH": LH_target,
                    "RH": RH_target,
                }

    txt_matches = 0
    txt_misses = 0
    for decoded_file in decoded_files:
        f = decoded_file["name"]

        options = decoded_file["decoded"]
        if options is None:
            continue
        elif ("FP_" in f) and ("FP_" not in options["session"]):
            print(f, options["session"])
            continue

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

        data["FP_note"].append("")
        data["LED_power_415"].append("")
        data["LED_power_470"].append("")
        data["LED_power_560"].append("")

        # parse text files for information

        #                        replace underscores with underscore or variable whitespace
        name_search = options["animal"].replace("_", r"(?:_|\s+)")
        # try each file and use first found
        found = False
        trigger_mode = ""
        for txt_file in txt_files:
            # get line containing animal session and two following
            matched_lines = re.search(
                r"^\s*" + name_search + r"[^\n]*\n[^\n]*\n",
                txt_file["data"],
                re.MULTILINE,
            )
            if not matched_lines:
                continue

            found = True
            lines = matched_lines.group(0)

            matched_trigger_mode = re.search(r"(BSC|Trg)\s*\d", lines, re.I)
            if matched_trigger_mode:
                trigger_mode = matched_trigger_mode.group(0)

            break

        data["trigger_mode"].append(trigger_mode)

        if found:
            txt_matches += 1
        else:
            txt_misses += 1
            # print("missed txt data", options["animal"])

    print(f"{txt_matches} matches, {txt_misses} misses")

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
    global txt_files
    for filename in os.listdir(root):
        # gross
        if filename == ".DS_Store":
            continue

        filepath = os.path.join(root, filename)

        # store for later parsing
        if ".txt" in filename:
            with open(filepath, "r") as f:
                txt_data = f.read()

                # txt files either have BSC or Trg notation, but not both
                dtm = "BSC1" if "BSC" in txt_data else "Trg1"

                txt_files.append({"default_trigger_mode": dtm, "data": txt_data})
            continue

        # recurse on folders
        if os.path.isdir(filepath):
            decode_files_in_folder(filepath)
            continue

        decoded = decode_from_filename(filepath)

        if decoded:
            if decoded["ftype"] == "exper":
                decoded_files.append(
                    {
                        "name": filename,
                        "path": filepath,
                        "decoded": decoded,
                    }
                )
        else:
            decode_failed.append(filepath)


# prepare files

print("Decoding files...")
decode_files_in_folder(IN_PATH)

# do stuff

print("Summarizing...")
summarize_sessions()

# print("Copying...")
# copy_files()
