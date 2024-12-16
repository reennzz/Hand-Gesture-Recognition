import os
import cv2
import numpy as np

# Paths
dataset_path = './HandGesture/images/'

# Parameters
img_size = 64  # Resize all images to 64x64

# Function to load dataset with ORB features
def load_dataset_with_orb(dataset_path):
    X = []
    y = []
    orb = cv2.ORB_create()
    
    # Iterate through gesture directories
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                
                # Read and process the image
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.GaussianBlur(img, (5, 5), 0)  # Apply Gaussian blur
                    img = cv2.resize(img, (img_size, img_size))

                    # Extract ORB features
                    keypoints, descriptors = orb.detectAndCompute(img, None)
                    if descriptors is not None:
                        X.append(descriptors.flatten())
                        y.append(label)

    return np.array(X, dtype=object), np.array(y)

# Load dataset
X, y = load_dataset_with_orb(dataset_path)

# Flatten descriptors to uniform length
max_length = max([len(x) for x in X])
X = np.array([np.pad(x, (0, max_length - len(x)), 'constant') if len(x) < max_length else x for x in X])

# Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier (e.g., Random Forest)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the model and label encoder
import joblib
joblib.dump(classifier, 'hand_gesture_rf_model.pkl')
with open('label_encoder.npy', 'wb') as f:
    np.save(f, label_encoder.classes_)

print("Model training with ORB features complete and saved.")

# Real-time gesture recognition using webcam
def real_time_gesture_recognition():
    # Load the trained model and label encoder
    classifier = joblib.load('hand_gesture_rf_model.pkl')
    label_classes = np.load('label_encoder.npy', allow_pickle=True)
    orb = cv2.ORB_create()

    cap = cv2.VideoCapture(0)
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)  # Apply Gaussian blur
        resized_frame = cv2.resize(gray_frame, (img_size, img_size))

        # Extract ORB features
        keypoints, descriptors = orb.detectAndCompute(resized_frame, None)
        if descriptors is not None:
            descriptors = descriptors.flatten()
            descriptors = np.pad(descriptors, (0, max_length - len(descriptors)), 'constant') if len(descriptors) < max_length else descriptors[:max_length]
            prediction = classifier.predict([descriptors])
            gesture = label_classes[prediction[0]]

            # Display the prediction on the video feed
            cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run real-time gesture recognition
if __name__ == "__main__":
    real_time_gesture_recognition()
