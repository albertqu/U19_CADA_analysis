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


def decode_text_files():
    for txt_file in txt_files:
        txt_file_date_entry_map = {}
        curr_date = ""
        curr_lines = []

        for line in txt_file["lines"]:
            date_match = re.search(
                r"\s*([0-9]{1,2})/([0-9]{1,2})/([0-9]{1,2})\s*", line
            )
            if date_match:
                if len(date_match.groups()) == 3:
                    # save current state and reset
                    if len(curr_lines):
                        txt_file_date_entry_map[curr_date] = "\n".join(
                            curr_lines
                        ).strip()
                        curr_lines = []

                    month = date_match.group(1).zfill(2)
                    day = date_match.group(2).zfill(2)
                    year = "20" + date_match.group(3)
                    curr_date = f"{year}{month}{day}"
                else:
                    print("Invalid date match:", line, date_match.groups())
            else:
                curr_lines.append(line)

        # save current state
        if len(curr_lines):
            txt_file_date_entry_map[curr_date] = "\n".join(curr_lines).strip()

        # save to file dict
        txt_file["entries"] = txt_file_date_entry_map


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

        # parse text files for information

        name_search = options["animal"]
        # replace underscores with underscore or variable whitespace
        name_search = name_search.replace("_", r"(?:_|\s+)")
        # -B and -1 are optional in animal name
        name_search = re.sub(r"(-(?:B|1))\b", r"(?:\1)?", name_search)

        # try each file and use first found
        found = False
        fp_note = ""
        trigger_mode = ""
        power415 = ""
        power470 = ""
        power560 = ""
        for txt_file in txt_files:
            # find sessions for date
            entry_data = txt_file["entries"].get(options["T"])
            if not entry_data:
                continue

            # get line containing animal session and two following
            matched_lines = re.search(
                r"^\s*" + name_search + r"[^\n]*(?:\n[^\n]*)?",
                entry_data,
                re.MULTILINE,
            )
            if not matched_lines:
                continue

            found = True
            lines = matched_lines.group(0)

            matched_trigger_mode = re.search(r"(BSC|Trg)\s*\d", lines, re.I)
            if matched_trigger_mode:
                trigger_mode = matched_trigger_mode.group(0)
            else:
                trigger_mode = txt_file["default_trigger_mode"]

            # powers
            power_regex = (
                r"[^;,0-9a-zA-Z\.]*([0-9a-zA-Z=\.\t ]+)[^;,\.]*[;,\.]?[^\S\n]*"
            )
            matched_power = re.search(
                fr"^[^\n]*415{power_regex}470{power_regex}560{power_regex}[^\n]*$",
                entry_data,
                re.MULTILINE,
            )
            if matched_power:
                clean_match = (
                    lambda s: re.sub(r"(?:green|top|red|bottom)", "", s)
                    .strip()
                    .replace(" ", "")
                )

                power415 = clean_match(matched_power.group(1))
                power470 = clean_match(matched_power.group(2))
                power560 = clean_match(matched_power.group(3))
            else:
                print("missed power match", entry_data)

            # FP note

            fp_note_regex1 = r"([0-9]{3}=[a-zA-Z]+,[0-9]{3}=[a-zA-Z]+)"

            matched_fp_note = re.search(fp_note_regex1, entry_data)
            if matched_fp_note:
                txt_matches += 1

                fp_note = matched_fp_note.group(0)
            else:

                def get_fp_note_keywords(inner):
                    matched = re.search(
                        r"(?:(?:top|bottom|green|red)(?: |\/)?)+",
                        inner,
                    )
                    if matched:
                        return matched.group(0)

                fp_note_items = []
                try:
                    after_415 = entry_data.split("415")[1].split("470")[0]
                    after_470 = entry_data.split("470")[1].split("560")[0]
                    after_560 = entry_data.split("560")[1].split("\n")[0]

                    fp_note_415 = get_fp_note_keywords(after_415)
                    fp_note_470 = get_fp_note_keywords(after_470)
                    fp_note_560 = get_fp_note_keywords(after_560)

                    if fp_note_415:
                        fp_note_items.append(f"415={fp_note_415}")
                    if fp_note_470:
                        fp_note_items.append(f"470={fp_note_470}")
                    if fp_note_560:
                        fp_note_items.append(f"560={fp_note_560}")
                except:
                    pass

                if len(fp_note_items):
                    fp_note = ",".join(fp_note_items)
                else:
                    print("missed fp note match", entry_data)

            break

        data["FP_note"].append(fp_note)
        data["trigger_mode"].append(trigger_mode)
        data["LED_power_415"].append(power415)
        data["LED_power_470"].append(power470)
        data["LED_power_560"].append(power560)

        if found:
            txt_matches += 1
        else:
            txt_misses += 1
            # print('missed animal', options["animal"], options["session"], options["T"])

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
                lines = f.readlines()

                # txt files either have BSC or Trg notation, but not both
                dtm = "BSC1" if "BSC" in "\n".join(lines) else "Trg1"

                txt_files.append({"default_trigger_mode": dtm, "lines": lines})
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
decode_text_files()

# do stuff

print("Summarizing...")
summarize_sessions()

# print("Copying...")
# copy_files()
