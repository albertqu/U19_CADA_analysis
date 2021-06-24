#!/usr/bin/env python3

from utils import decode_from_filename
import os
import shutil

IN_PATH = '/Volumes/Wilbrecht_file_server/2ABT/ProbSwitch/ProbSwitch_Raw'
OUT_PATH = os.path.join(os.getcwd(), 'animals')

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


def decode_files_in_folder(root):
    for filename in os.listdir(root):
        # gross
        if filename == '.DS_Store':
            continue

        filepath = os.path.join(root, filename)

        # recurse on folders
        if os.path.isdir(filepath):
            decode_files_in_folder(filepath)
            continue

        decoded_files.append(
            {'name': filename, 'path': filepath, 'decoded': decode_from_filename(filepath)})


decode_files_in_folder(IN_PATH)

grouped_by_animal = group_by(
    decoded_files, lambda data: data['decoded']['animal'] if data['decoded'] and 'animal' in data['decoded'] else '_NONE')

print('Copying files...')

for animal, animal_files in grouped_by_animal.items():
    animal_folder = os.path.join(OUT_PATH, animal)
    if not os.path.exists(animal_folder):
        os.makedirs(animal_folder)

    grouped_by_session = group_by(
        animal_files, lambda data: data['decoded']['session'] if data['decoded'] and 'session' in data['decoded'] else '_NONE')

    for session, session_files in grouped_by_session.items():
        session_folder = os.path.join(animal_folder, session)
        if not os.path.exists(session_folder):
            os.makedirs(session_folder)

        for decoded_file in session_files:
            out_filepath = os.path.join(session_folder, decoded_file['name'])
            shutil.copy2(decoded_file['path'], out_filepath)
            print('Copied')
