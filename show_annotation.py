import cv2
import matplotlib.pyplot as plt
import os
import random

def read_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            annotations.append(list(map(float, line.strip().split())))
    return annotations

def draw_annotations(image_path, annotations):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    for annotation in annotations:
        class_id, x_center, y_center, width, height = annotation
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put the class ID on the rectangle
        cv2.putText(image, f'ID: {int(class_id)}', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        print(f'Class ID: {int(class_id)}, Coordinates: ({x1}, {y1}), ({x2}, {y2})')

    return image

def show_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def process_random_images(image_dir, label_dir, num_images=5):
    # Get a list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpeg') or f.endswith('.jpg')]
    
    # Select random images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    for image_file in selected_images:
        # Construct full image and annotation file paths
        image_path = os.path.join(image_dir, image_file)
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        annotation_path = os.path.join(label_dir, annotation_file)

        # Read and draw annotations if the annotation file exists
        if os.path.exists(annotation_path):
            annotations = read_annotations(annotation_path)
            annotated_image = draw_annotations(image_path, annotations)

            # Show the annotated image
            show_image(annotated_image)
        else:
            print(f"No annotation found for {image_file}")

# Directories for images and annotations
image_directory = r"datasets/images/train"
label_directory = r"datasets/labels/train"

# Process random images
process_random_images(image_directory, label_directory)
