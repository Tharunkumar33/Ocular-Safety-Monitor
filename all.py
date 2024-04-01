# Import necessary libraries
import cv2
import numpy as np
import dlib
from scipy.spatial import distance
import pygame
from imutils import face_utils

# Initialize Pygame for sound alerts
pygame.mixer.init()
pygame.mixer.music.load("beep-01a.wav")

# Load the face detector and facial landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate the EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    A = distance.euclidean((eye[1].x, eye[1].y), (eye[5].x, eye[5].y))
    B = distance.euclidean((eye[2].x, eye[2].y), (eye[4].x, eye[4].y))
    C = distance.euclidean((eye[0].x, eye[0].y), (eye[3].x, eye[3].y))
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for EAR thresholds and consecutive frames
EAR_THRESHOLD = 0.25
CONSEC_FRAMES_CLOSED = 60  # 3 seconds for eyes not detected
CONSEC_FRAMES_DROWSY = 60  # 3 seconds for eyes closed

# Initialize variables
frames_closed = 0
frames_drowsy = 0
cap = cv2.VideoCapture(0)  # Change to 1 if you want to use an external webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    # Check if no faces are detected
    if len(faces) == 0:
        frames_closed += 1
        frames_drowsy = 0

        if frames_closed >= CONSEC_FRAMES_CLOSED:
            # No eyes detected for 3 seconds, play beep sound
            pygame.mixer.music.play(-1)  # Loop the alarm sound
            cv2.putText(frame, "Eyes Not Recognized!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        frames_closed = 0

        for face in faces:
            landmarks = predictor(gray, face)

            left_eye = [landmarks.part(i) for i in range(36, 42)]
            right_eye = [landmarks.part(i) for i in range(42, 48)]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # Calculate the average EAR for both eyes
            avg_ear = (left_ear + right_ear) / 2.0

            # Draw the detected face and landmarks on the frame
            for i in range(36, 48):
                x, y = landmarks.part(i).x, landmarks.part(i).y
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            # Check if the EAR is below the threshold
            if avg_ear < EAR_THRESHOLD:
                frames_drowsy += 1

                if frames_drowsy >= CONSEC_FRAMES_DROWSY:
                    # Drowsiness detected for 3 seconds, sound an alarm
                    pygame.mixer.music.play(-1)  # Loop the alarm sound
                    cv2.putText(frame, "Drowsiness Detected!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                frames_drowsy = 0
                pygame.mixer.music.fadeout(2500)  # Stop the alarm sound

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
