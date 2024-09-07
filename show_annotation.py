import cv2
import matplotlib.pyplot as plt

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

        # Optionally, put the class ID on the rectangle
        cv2.putText(image, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def show_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# File paths
image_path = r"F:\Surveillence_System_Project_(BCSIR)\Deep_Learning\object_detection\weapon_detection\train\images\Sniper\Sniper_80.jpeg"  # Your image file
annotation_path = r"F:\Surveillence_System_Project_(BCSIR)\Deep_Learning\object_detection\weapon_detection\train\labels\Sniper_80.txt"  # Your annotation file

# Read and draw annotations
annotations = read_annotations(annotation_path)
annotated_image = draw_annotations(image_path, annotations)

# Show the image
show_image(annotated_image)
