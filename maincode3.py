import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import mediapipe as mp

# Dataset location
DATASET_PATH = r'C:\Users\acer\OneDrive\Documents\leapGestRecog'

# Load data
def load_data(data_dir):
    images = []
    labels = []
    label_map = {}  # To store the mapping from label index to gesture name
    gesture_labels = ['draw', 'erase', 'clear']  # Adjusted to match dataset

    for idx, gesture in enumerate(gesture_labels):
        folder_path = os.path.join(data_dir, gesture)
        label_map[idx] = gesture  # Map the label index to the gesture name
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, image_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue  # Skip empty or unreadable images
                img = cv2.resize(img, (64, 64))  # Resize to 64x64 pixels
                img = img / 255.0  # Normalize pixel values
                images.append(img)
                labels.append(idx)  # Append the label index corresponding to the gesture
    return np.array(images), np.array(labels), label_map

# Build and train the CNN model
def build_and_train_model():
    train_images, train_labels, label_map = load_data(os.path.join(DATASET_PATH, 'train'))
    test_images, test_labels, _ = load_data(os.path.join(DATASET_PATH, 'test'))

    # Split training data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )

    # Build CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(label_map), activation='softmax')  # Output layer
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, batch_size=32, epochs=10, validation_data=(val_images, val_labels))

    # Evaluate on test data
    model.evaluate(test_images, test_labels)

    # Save the model
    model.save('hand_sign_model.keras')
    return model, label_map

# Air Canvas application
def air_canvas(model, label_map):
    # Initialize MediaPipe for hand tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255
    gesture_to_action = {0: 'draw', 1: 'erase', 2: 'clear'}  # Updated mapping
    drawing = False
    last_pos = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for better user interaction
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Draw the landmarks on the frame
                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the bounding box for the hand (region of interest)
                x_min = min([landmark.x for landmark in landmarks.landmark])
                x_max = max([landmark.x for landmark in landmarks.landmark])
                y_min = min([landmark.y for landmark in landmarks.landmark])
                y_max = max([landmark.y for landmark in landmarks.landmark])

                # Define a region of interest (ROI) around the hand
                h, w, _ = frame.shape
                roi = frame[int(y_min * h):int(y_max * h), int(x_min * w):int(x_max * w)]

                # Resize the ROI to match the input shape of the CNN model (64x64)
                roi_resized = cv2.resize(roi, (64, 64))
                roi_resized = roi_resized / 255.0  # Normalize the pixel values
                roi_resized = np.expand_dims(roi_resized, axis=0)  # Add batch dimension

                # Predict the gesture using the trained CNN model
                prediction = model.predict(roi_resized)
                gesture = np.argmax(prediction)
                confidence = np.max(prediction)

                # Only proceed if the confidence is above a threshold (e.g., 0.7)
                if confidence > 0.7:
                    action = gesture_to_action.get(gesture, None)
                else:
                    action = None  # Ignore if confidence is low

                # Debugging: Print the prediction result
                print(f"Prediction: {gesture}, Confidence: {confidence}")

                # Map the predicted gesture to an action
                cv2.putText(frame, f"Gesture: {label_map[gesture]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Perform the action based on the detected gesture
                if action == 'draw':
                    drawing = True
                    if last_pos is not None:
                        cv2.line(canvas, last_pos, (250, 250), (0, 0, 0), 5)  # Draw on the canvas
                    last_pos = (250, 250)
                elif action == 'erase' and drawing:
                    canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255  # Erase canvas
                    last_pos = None
                elif action == 'clear':
                    canvas.fill(255)  # Clear canvas
                    last_pos = None

        # Show the updated canvas and hand gesture feed
        cv2.imshow('Air Canvas', canvas)
        cv2.imshow('Hand Gesture Feed', frame)

        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    model, label_map = build_and_train_model()
    air_canvas(model, label_map)
