import cv2
import numpy as np
from keras.models import load_model
from ultralytics import YOLO

# Load your custom face recognition model
face_recognition_model = load_model('E:/Face_Recognition/Recognition/VGG16/Result/custom_vggface_model_thang.h5')

# Load the YOLOv8 model
yolo_model = YOLO('E:/Face_Recognition/Detection/Yolo/runs/detect/train/weights/last.pt')

# Load labels for face recognition
# Assuming you have labels corresponding to the output classes of your face recognition model
# Modify this according to your actual label format
labels = ["Empty", "Thang"]  # Example labels

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Predict objects in the frame using YOLOv8
    results = yolo_model(frame)

    # Process each detected object
    for result in results:
        boxes = result.boxes.xyxy  # Get box coordinates in (x1, y1, x2, y2) format
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Crop the detected face region
            face_region = frame[y1:y2, x1:x2]

            # Resize face region to fit your face recognition model input size (if needed)
            desired_width = 224
            desired_height = 224
            face_region_resized = cv2.resize(face_region, (desired_width, desired_height))

            # Perform face recognition prediction
            prediction = face_recognition_model.predict(np.expand_dims(face_region_resized, axis=0))

            # Get the predicted label and confidence
            confidence = prediction[0][np.argmax(prediction)]
            if prediction >= 0.5:
                predicted_label = 1
            else:
                predicted_label = 0

            predicted_label = labels[predicted_label]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label and confidence
            label_text = f"{predicted_label}: {confidence:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('YOLOv8 Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
