from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

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
    Splits data into training and validation sets, and organizes them into respective folders.

    Parameters:
    dataset_dir -- the root directory containing 'images' and 'labels' subdirectories.
    """

    print("Attempting to split the dataset into train and validation sets...")

    img_path = Path(dataset_dir, 'images')
    annotations_path = Path(dataset_dir, 'labels')

    # Create a temporary directory
    temp_dir = Path(dataset_dir, 'temp')
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Create directories for train and validation sets
    img_path_train = img_path / 'train/'
    img_path_val = img_path / 'val/'
    
    annotations_path_train = annotations_path / 'train/'
    annotations_path_val = annotations_path / 'val/'

    # Define paths in the temporary directory
    temp_img_path = temp_dir / 'images'
    temp_annotations_path = temp_dir / 'labels'

    # Move images and annotations to the temporary directory
    shutil.move(str(img_path_train), str(temp_img_path))
    shutil.move(str(annotations_path_train), str(temp_annotations_path))

    img_path_train.mkdir(parents=True, exist_ok=True)
    img_path_val.mkdir(parents=True, exist_ok=True)

    annotations_path_train.mkdir(parents=True, exist_ok=True)
    annotations_path_val.mkdir(parents=True, exist_ok=True)

    # Get a sorted list of image and annotation file names
    images = sorted(
        [file for file in temp_img_path.rglob('*') if file.suffix.lower() in ['.jpg', '.jpeg']]
    )
    annotations = sorted(
        [file for file in temp_annotations_path.rglob('*') if file.suffix.lower() in ['.txt']]
    )


    # Split the data into training and validation sets (80-20 split)
    train_images, val_images, train_annotations, val_annotations = train_test_split(
        images, annotations, test_size=0.2, random_state=1
    )

    # Move the split files back to their respective folders
    move_files_to_folder(train_images, img_path_train)
    move_files_to_folder(train_annotations, annotations_path_train)
    move_files_to_folder(val_images, img_path_val)
    move_files_to_folder(val_annotations, annotations_path_val)

    # Remove the temporary directory
    shutil.rmtree(temp_dir)

    print("Successfully split the dataset into train and validation sets.")

if __name__ == '__main__':
    dataset_dir = Path("datasets/")

    split_data(dataset_dir)

    print("Validation split successful!")
