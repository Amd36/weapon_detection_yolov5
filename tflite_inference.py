import numpy as np
import cv2
import tensorflow as tf

# Load the TFLite model and allocate tensors
model_path = '/home/du/Junayed/weapon_detection_yolov5/exported_models/best-fp16-yolov5m.tflite'  # Path to your TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class names
with open('classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Function to preprocess the image
def preprocess_image(image_path):
    original_image = cv2.imread(image_path)
    original_height, original_width = original_image.shape[:2]
    image = cv2.resize(original_image, (640, 640))  # Resize to model input size
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image, axis=0), original_image, original_width, original_height  # Return original image and dimensions

# Function to perform Non-Maximum Suppression (NMS)
def non_max_suppression(detections, iou_threshold=0.4):
    boxes = []
    confidences = []
    class_ids = []
    
    for detection in detections:
        box = detection[:4]
        confidence = detection[4]
        class_id = np.argmax(detection[5:])

        if confidence > 0.5:  # Confidence threshold
            boxes.append(box)
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # Perform NMS
    indices = tf.image.non_max_suppression(
        boxes, confidences, max_output_size=50, iou_threshold=iou_threshold
    )

    return [(boxes[i], confidences[i], class_ids[i]) for i in indices.numpy()]

# Function to draw predictions on the image
def draw_predictions(image, results, original_width, original_height):
    for box, confidence, class_id in results:
        # Convert from center-width-height to xmin, ymin, xmax, ymax
        x_center, y_center, width, height = box
        x_min = int((x_center - width / 2) * original_width)
        y_min = int((y_center - height / 2) * original_height)
        x_max = int((x_center + width / 2) * original_width)
        y_max = int((y_center + height / 2) * original_height)

        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Prepare label
        label = f"{class_names[class_id]}: {confidence:.2f}"
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_bg = (x_min, y_min - label_size[1] - 10, x_min + label_size[0], y_min)
        cv2.rectangle(image, (label_bg[0], label_bg[1]), (label_bg[2], label_bg[3]), (255, 0, 0), -1)
        # Draw text
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Function to run inference and display class probabilities
def infer_and_display(image_path):
    input_data, original_image, original_width, original_height = preprocess_image(image_path)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run the model
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.array(output_data)

    # Apply NMS and get results
    results = non_max_suppression(output_data[0])

    # Draw predictions on the image
    draw_predictions(original_image, results, original_width, original_height)

    # Display the image
    cv2.imshow("Detections", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'sample2.png'  # Path to your image
infer_and_display(image_path)
