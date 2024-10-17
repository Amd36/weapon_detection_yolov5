import numpy as np
import tensorflow as tf
import cv2
import os
import argparse

def preprocess_image(image):
    image = cv2.resize(image, (640, 640))  # Adjust according to your model's input size
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def postprocess_output(output_data):
    boxes = output_data[0]
    scores = output_data[1]
    classes = output_data[2]
    
    valid_boxes = []
    confidence_threshold = 0.5
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            valid_boxes.append((boxes[i], scores[i], classes[i]))
    return valid_boxes

def main(model_path, image_path):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    image = cv2.imread(image_path)
    input_image = preprocess_image(image)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run inference
    interpreter.invoke()

    # Get output data
    output_data = [interpreter.get_tensor(detail['index']) for detail in output_details]

    # Postprocess the output
    detections = postprocess_output(output_data)

    # Draw bounding boxes on the image
    for box, score, class_id in detections:
        x1, y1, x2, y2 = box  # Adjust according to your model output
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image, f"ID: {int(class_id)} Score: {score:.2f}", 
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Inference', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with TFLite model.')
    parser.add_argument('--model', type=str, required=True, help='Path to the TFLite model file.')
    parser.add_argument('--image', type=str, required=True, help='Path to the image for inference.')

    args = parser.parse_args()
    
    main(args.model, args.image)
