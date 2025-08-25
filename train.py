'''Explanation:

	•	Data Loading and Processing: The code starts by loading and processing the data using the process_images function from another module (create). This function returns the data and corresponding labels.
	•	Data Splitting: The data is then split into training and test sets, with 80% of the data used for training the model and 20% reserved for testing. The stratify parameter ensures that the class distribution is the same in both the training and test sets.
	•	Model Initialization and Training: A RandomForestClassifier model is created and trained using the training data (x_train, y_train).
	•	Prediction and Evaluation: The trained model makes predictions on the test set (x_test), and the accuracy of these predictions is calculated by comparing them to the actual labels (y_test). The accuracy score is printed as a percentage.
	•	Model Saving: Finally, the trained model is saved to a file (model.p) using pickle, so it can be loaded and used later without retraining.'''

import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from create import process_images  # Import the process_images function from the create module

start_time = time.time()
# Load and process data using the process_images function
data, labels = process_images()

# Split the data into training and test sets
# 20% of the data will be used for testing, and 80% for training
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Initialize a RandomForestClassifier model
model = RandomForestClassifier()

# Train the model using the training data
model.fit(x_train, y_train)

# Use the trained model to make predictions on the test data
y_predict = model.predict(x_test)

# Calculate the accuracy of the model by comparing predictions with the actual test labels
score = accuracy_score(y_test, y_predict)

# Print the accuracy score as a percentage
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a file using pickle
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

# Stop the timer
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print('Time taken: {:.2f} seconds'.format(elapsed_time))