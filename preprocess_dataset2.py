"""
This script extracts a specific dataset from a ZIP file, creates annotations for images, and copies images to a specified directory. 
It also allows for optional removal of the ZIP file after extraction.

Usage:
    python script_name.py -d <destination_path> [-r]

Arguments:
    -d, --destination_path  Path to the directory where the dataset should be extracted and processed.
    -r, --remove_zip        Optional. If specified, deletes the original ZIP file after extraction.

Functions:
    extract_dataset(zip_path, required_file_path, extract_to)
        Extracts files from the ZIP archive that match the required directory path.
        
    get_labels(dataset_path)
        Generates a dictionary mapping directory names to integer labels.
        
    create_annotations(dataset_path, destination_path, label_dict)
        Creates annotation files for each image based on its directory label.
        
    copy_images(dataset_dir, destination_dir)
        Copies all .jpg images from the dataset directory to the destination directory.

    move_files_to_folder(list_of_files, destination_folder)
        Moves files from the list_of_files to the destination_folder.

    split_data(dataset_dir)
        Splits data into train, validation, and test sets and organizes them into respective folders.
"""

import zipfile
import os
import argparse
from pathlib import Path
import yaml
import shutil
from sklearn.model_selection import train_test_split

def extract_dataset(zip_path, required_file_path, extract_to):
    """Extracts files from a ZIP archive to the specified directory, filtering by required path."""

    print("Attempting to extract the specified dataset into destination directory...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            # Check if the current file path starts with the required directory path
            if file_info.filename.startswith(required_file_path):
                # Ensure the destination directory exists
                os.makedirs(extract_to, exist_ok=True)
                
                # Extract the file to the target directory, maintaining the directory structure
                zip_ref.extract(file_info.filename, extract_to)
        
    print("Successfully extracted all the files from", zip_path)

def get_labels(dataset_path):
    """Generates a dictionary mapping directory names to integer labels."""
    try:
        # Get a list of all subdirectories (labels) in the dataset directory
        labels = [entry.name for entry in dataset_path.iterdir() if entry.is_dir()]
        label_dict = {label: i for i, label in enumerate(labels)}
        return label_dict
    except Exception as e:
        print(e)
        return {}
    
def create_annotations(dataset_path, destination_path, label_dict):
    """Creates annotation files for each image based on its directory label."""

    print("Attempting to create necessary annotations for the images found in dataset...")

    label_path = Path(destination_path, 'labels')
    parent_path = Path(dataset_path)

    # Ensure the 'labels' directory exists
    label_path.mkdir(parents=True, exist_ok=True)

    # Iterate through all .jpg files in the dataset directory
    for file_path in parent_path.rglob('*.jpg'):
        base = file_path.stem  # Get the base name of the file without extension

        # Get the label from the parent directory name
        label = file_path.parts[-2]
        annotation = f"{label_dict[label]} 0.5 0.5 1 1"
        annotation_file = base + '.txt'
        annotation_path = label_path / annotation_file

        # Write the annotation to the file
        with open(annotation_path, 'w') as file:
            file.write(annotation)

    print("Successfully created the annotations in the directory:", label_path.name)

def copy_images(dataset_dir, destination_dir):
    """Copies all .jpg images from the dataset directory to the destination directory."""

    print("Attempting to copy images from the temporary dataset directory to images...")

    dataset_path = Path(dataset_dir)
    destination_path = Path(destination_dir, 'images')

    # Ensure the 'images' directory exists
    destination_path.mkdir(parents=True, exist_ok=True)

    # Iterate through all .jpg files and copy them to the destination directory
    for file_path in dataset_path.rglob('*.jpg'):
        dest_file_path = destination_path / file_path.name
        shutil.copy(file_path, dest_file_path)
    
    print(f"Copied all jpg files from {dataset_dir} to {destination_dir}")

def move_files_to_folder(files, destination_dir):
    """Moves the specified files to the destination directory."""
    destination_path = Path(destination_dir)

    # Iterate through the list of files and move each to the destination directory
    for file in files:
        try:
            dest_file_path = destination_path / file.name
            
            # Remove the destination file if it already exists
            if dest_file_path.exists():
                dest_file_path.unlink()

            shutil.move(str(file), str(destination_path))
        except Exception as e:
            print(f"Failed to move file: {file} due to {e}")

def split_data(dataset_dir):
    """
    Splits data into training, validation, and test sets, and organizes them into respective folders.

    Parameters:
    dataset_dir -- the root directory containing 'images' and 'labels' subdirectories.
    """

    print("Attempting to split the dataset into train, validation and test sets...")

    img_path = Path(dataset_dir, 'images')
    annotations_path = Path(dataset_dir, 'labels')

    # Define subdirectories for train, validation, and test sets
    img_path_train = img_path / 'train'
    img_path_val = img_path / 'val'
    img_path_test = img_path / 'test'

    annotations_path_train = annotations_path / 'train'
    annotations_path_val = annotations_path / 'val'
    annotations_path_test = annotations_path / 'test'

    # Create directories for train, validation, and test sets
    img_path_train.mkdir(parents=True, exist_ok=True)
    img_path_val.mkdir(parents=True, exist_ok=True)
    img_path_test.mkdir(parents=True, exist_ok=True)

    annotations_path_train.mkdir(parents=True, exist_ok=True)
    annotations_path_val.mkdir(parents=True, exist_ok=True)
    annotations_path_test.mkdir(parents=True, exist_ok=True)

    # Get a sorted list of image and annotation file names
    images = [file for file in img_path.rglob('*.jpg')]
    annotations = [file for file in annotations_path.rglob('*.txt')]

    images.sort()
    annotations.sort()

    # Split the data into training, validation, and test sets (80-10-10 split)
    train_images, val_images, train_annotations, val_annotations = train_test_split(
        images, annotations, test_size=0.2, random_state=1
    )
    val_images, test_images, val_annotations, test_annotations = train_test_split(
        val_images, val_annotations, test_size=0.5, random_state=1
    )

    # Move the split files to their respective folders
    move_files_to_folder(train_images, img_path_train)
    move_files_to_folder(train_annotations, annotations_path_train)
    move_files_to_folder(val_images, img_path_val)
    move_files_to_folder(val_annotations, annotations_path_val)
    move_files_to_folder(test_images, img_path_test)
    move_files_to_folder(test_annotations, annotations_path_test)

    print("Successfully split the dataset into train, val and test sets")


if __name__ == '__main__':
    zip_path = 'OD-WeaponDetection-master.zip'  # Path to your zip file
    required_file_path = "OD-WeaponDetection-master/Weapons and similar handled objects/Sohas_weapon-Classification/"  # Specific dataset only

    parser = argparse.ArgumentParser(description="Extract the required dataset from the zip file.")
    parser.add_argument('-d', "--destination_path", type=str, required=True, help="Path to extraction directory")
    parser.add_argument('-r', "--remove_zip", action='store_true', help="Delete the original zip file after extraction")
    
    args = parser.parse_args()

    if Path(args.destination_path).is_dir():
        shutil.rmtree(str(Path(args.destination_path)))

    # Extract dataset from ZIP file
    extract_dataset(zip_path, required_file_path, args.destination_path)

    # Optionally remove the ZIP file after extraction
    if args.remove_zip:
        os.remove(zip_path)
        print("Successfully deleted the zip file!")

    # Define the path to the dataset directory
    dataset_dir = Path(args.destination_path) / required_file_path

    # Create label dictionary and save to YAML file
    label_dict = get_labels(dataset_dir)
    with open('labels.yaml', 'w') as file:
        yaml.dump(label_dict, file, default_flow_style=False)
    print("Created labels.yaml with updated labels")

    # Create annotation files for images
    create_annotations(dataset_dir, args.destination_path, label_dict)

    # Copy all images to the destination directory
    copy_images(dataset_dir, args.destination_path)

    # Split the dataset into training, validation, and test sets
    split_data(args.destination_path)

    # Delete the temporary dataset directory
    shutil.rmtree(str(Path(args.destination_path, required_file_path.split('/')[0])))
