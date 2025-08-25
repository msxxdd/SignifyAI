'''Explanation:

	•	Directory Setup: The code first checks and creates the necessary directories for storing the images.
	•	Camera Initialization: It then tries to access the camera to capture video frames. If the camera isn’t accessible, the program exits.
	•	Data Collection: The code enters a loop for each class, where it prompts the user to get ready before collecting images. Once ready, it captures a specified number of images and saves them in the appropriate directory.
	•	Completion: After collecting images for all classes, the program releases the camera and closes any OpenCV windows.'''

import os
import cv2

# Define the directory where the data will be stored
DATA_DIR = './data'

# Create the data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Set the number of classes (e.g., different hand signs) and the size of the dataset for each class
number_of_classes = 26
dataset_size = 100

# Initialize the camera (Use 0 for the default camera, change to 1 if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Check if the camera is accessible
if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

# Loop through each class to collect data
for j in range(number_of_classes):
    # Create a directory for each class inside the main data directory
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Display a "Ready?" prompt on the screen before starting data collection
    while True:
        ret, frame = cap.read()  # Capture a frame from the camera
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        # Display the "Ready?" prompt on the captured frame
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Show the frame with the prompt

        if cv2.waitKey(25) == ord('q'):
            # Capture a new frame to clear the previous text
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                continue

            # Display the "Get ready..." message on the new frame
            cv2.putText(frame, 'Get ready...', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.waitKey(2000)  # 2-second delay to give the user time to get ready
            break

    # Start collecting images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()  # Capture a frame from the camera
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        cv2.imshow('frame', frame)  # Show the frame in a window
        cv2.waitKey(25)  # Wait for 25 milliseconds between frames

        # Save the captured frame as an image file in the corresponding class directory
        file_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(file_path, frame)
        counter += 1  # Increment the counter to track the number of images collected

print("Data collection complete.")  # Indicate that data collection is finished
cap.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close all OpenCV windows