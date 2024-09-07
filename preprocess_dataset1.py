import os
import zipfile
import yaml
from pathlib import Path

def unzip_into_dir(zip_data, extraction_dir):
    os.makedirs(extraction_dir, exist_ok=True)

    if os.path.exists(zip_data):
        with zipfile.ZipFile(zip_data, 'r') as zip_ref:
            zip_ref.extractall(extraction_dir)

        print("Successfully extracted the zip file in", extraction_dir)
    else:
        print("data.zip file doesn't exist! Try downloading manually from the link given in README.md")

def change_class_id(file_name, label_dict):
    with open(file_name, 'r') as fptr:
        lines = fptr.readlines()
        class_name = Path(file_name).stem.split('_')[0]

    fixed_lines = []
    for line in lines:
        fixed_line = str(label_dict[class_name]) + line[1:]
        fixed_lines.append(fixed_line)

    with open(file_name, 'w') as fptr:
        for line in fixed_lines:
            fptr.write(line)

def list_txt_files(directory):
    try:
        entries = os.listdir(directory)

        txt_files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry)) and entry.endswith('.txt')]
        return txt_files
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return []
    except PermissionError:
        print(f"Permission denied to access the directory: {directory}")
        return []

if __name__ == '__main__':
    label_dict = {'Automatic Rifle': 0, 
              'Bazooka': 1, 
              'Grenade Launcher': 2, 
              'Handgun': 3, 
              'Knife': 4, 
              'SMG': 6, 
              'Shotgun': 5, 
              'Sniper': 7, 
              'Sword': 8}

    with open('labels.yaml', 'w') as file:
        yaml.dump(label_dict, file, default_flow_style=False)

    print("Successfully created labels.yaml file")

    zip_data = 'archive.zip'
    extraction_dir = "datasets/set1/"

    os.makedirs(extraction_dir, exist_ok=True)

    unzip_into_dir(zip_data, extraction_dir)

    paths = [r"datasets\set1\weapon_detection\train\labels", 
        r"datasets\set1\weapon_detection\val\labels"]

    with open('labels.yaml', 'r') as file:
        label_dict = yaml.safe_load(file)

    for path in paths:
        filenames = list_txt_files(path)

        if filenames:
            for filename in filenames:
                file_dir = os.path.join(path, filename)
                change_class_id(file_dir, label_dict)

    print("Successfully updated annotation files with respective class IDs!")