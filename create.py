'''Explanation:

	•	Initialization: The code initializes MediaPipe’s Hand model to detect hand landmarks in images.
	•	Directory Setup: It specifies the directory containing images and prepares empty lists to store the processed data and corresponding labels.
	•	Image Processing: The process_images function iterates through each class directory and image, using MediaPipe to detect hand landmarks in each image.
	•	Feature Extraction: For each detected hand, it extracts the x and y coordinates of the landmarks, normalizes them, and stores them as features.
	•	Padding: The code ensures that all feature vectors have the same length (84 features), padding with zeros if necessary.
	•	Output: Finally, it returns the processed data and labels as NumPy arrays, ready for further processing or model training.'''

import os
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands for hand landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the directory containing the collected images
DATA_DIR = './data'

# Lists to store processed data and their corresponding labels
data = []
labels = []

# Function to process images in the dataset
def process_images():
    # Loop through each class directory in the data directory
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):  # Skip if it's not a directory
            continue

        # Loop through each image in the class directory
        for img_path in os.listdir(dir_path):
            img = cv2.imread(os.path.join(dir_path, img_path))  # Read the image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB

            # Process the image with MediaPipe Hands to detect hand landmarks
            results = hands.process(img_rgb)

            # Lists to store the hand landmark coordinates (features)
            data_aux = []
            all_x = []
            all_y = []

            # Check if hand landmarks were detected in the image
            if results.multi_hand_landmarks:
                # Loop through each detected hand (for multi-hand cases)
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract the x and y coordinates of each landmark
                    x_ = [hand_landmarks.landmark[i].x for i in range(len(hand_landmarks.landmark))]
                    y_ = [hand_landmarks.landmark[i].y for i in range(len(hand_landmarks.landmark))]
                    all_x.extend(x_)  # Store all x coordinates
                    all_y.extend(y_)  # Store all y coordinates

                    # Normalize the x and y coordinates by subtracting the minimum value (top-left corner)
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))  # Normalize x and append to the feature list
                        data_aux.append(y - min(y_))  # Normalize y and append to the feature list

                # Handle padding based on the number of detected features (landmarks)
                if len(data_aux) == 42:
                    data_aux.extend([0] * 42)  # Pad with zeros to reach 84 features (single hand case)
                elif len(data_aux) == 84:
                    pass  # No padding needed (already has 84 features for both hands)

                # Add the processed features and corresponding label to the dataset
                if len(data_aux) == 84:
                    data.append(data_aux)
                    labels.append(dir_)

    # Return the processed data and labels as NumPy arrays
    return np.asarray(data), np.asarray(labels)