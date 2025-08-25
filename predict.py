import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model and labels from the saved file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']  # Extract the model from the loaded dictionary

# Initialize MediaPipe Hands for hand landmark detection and the webcam for capturing video
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or 1 for an external webcam
mp_hands = mp.solutions.hands  # Initialize the MediaPipe Hands solution
mp_drawing = mp.solutions.drawing_utils  # Utility for drawing the landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Utility for drawing styles
hands = mp_hands.Hands(static_image_mode=False,
                       min_detection_confidence=0.3)  # Set up hand detection with a minimum confidence threshold

# Define the label dictionary mapping predicted labels to corresponding characters
labels_dict = {
    'A': 'A',
    'B': 'B',
    'C': 'C',
    'D': 'D',
    'E': 'E',
    'F': 'F',
    'G': 'G',
    'H': 'H',
    'I': 'I',
    'J': 'J',
    'K': 'K',
    'L': 'L',
    'M': 'M',
    'N': 'N',
    'O': 'O',
    'P': 'P',
    'Q': 'Q',
    'R': 'R',
    'S': 'S',
    'T': 'T',
    'U': 'U',
    'V': 'V',
    'W': 'W',
    'X': 'X',
    'Y': 'Y',
    'Z': 'Z',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    '0': '0',
    # Add more labels according to your dataset
}

confidence_threshold = 0.4  # Define the confidence threshold

while True:
    ret, frame = cap.read()  # Capture a frame from the webcam

    if not ret:
        print("Failed to capture image from webcam. Exiting...")
        break

    H, W, _ = frame.shape  # Get the height and width of the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB format for MediaPipe
    results = hands.process(frame_rgb)  # Process the frame to detect hand landmarks

    data_aux = []  # List to store normalized hand landmark features
    all_x = []  # List to store all x coordinates of landmarks
    all_y = []  # List to store all y coordinates of landmarks

    if results.multi_hand_landmarks:
        # Loop through each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the x and y coordinates of each landmark
            x_ = [hand_landmarks.landmark[i].x for i in range(len(hand_landmarks.landmark))]
            y_ = [hand_landmarks.landmark[i].y for i in range(len(hand_landmarks.landmark))]
            all_x.extend(x_)  # Store all x coordinates
            all_y.extend(y_)  # Store all y coordinates

            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Normalize the x and y coordinates by subtracting the minimum value
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalize x and append to the feature list
                data_aux.append(y - min(y_))  # Normalize y and append to the feature list

        # Ensure the feature list has the correct number of elements for the model
        if len(data_aux) == 42:
            data_aux.extend([0] * 42)  # Pad to 84 features if only one hand is detected
        elif len(data_aux) == 84:
            pass  # No padding needed if both hands are detected

        # If the feature list is of the correct length, make a prediction using the model
        if len(data_aux) == 84:
            # Get class probabilities
            probs = model.predict_proba([np.asarray(data_aux)])
            max_prob = np.max(probs)
            predicted_label_index = np.argmax(probs)
            predicted_label = model.classes_[predicted_label_index]  # Get the predicted label

            # Check if the maximum probability meets the confidence threshold
            if max_prob >= confidence_threshold:
                predicted_character = labels_dict.get(predicted_label,
                                                      '')  # Map the label to the corresponding character
                display_text = f"{predicted_character} ({max_prob:.1f})"
            else:
                predicted_character = ''
                display_text = f" ({max_prob:.1f})"

            # Calculate the bounding box that encloses the detected hands
            if all_x and all_y:
                x1 = int(min(all_x) * W) - 10  # Top-left corner x coordinate
                y1 = int(min(all_y) * H) - 10  # Top-left corner y coordinate
                x2 = int(max(all_x) * W) + 10  # Bottom-right corner x coordinate
                y2 = int(max(all_y) * H) + 10  # Bottom-right corner y coordinate

                # Draw the bounding box and the predicted character on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)  # Draw a black rectangle
                cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255),
                            3, cv2.LINE_AA)  # Display the predicted character and probability above the bounding box

    cv2.imshow('frame', frame)  # Show the frame with the detected hands and predictions

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop if the 'q' key is pressed
        break

cap.release()  # Release the webcam resource
cv2.destroyAllWindows()  # Close all OpenCV windows