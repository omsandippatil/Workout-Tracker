import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate distance between two points
def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Function to check plank position
def check_plank(shoulder, hip, ankle, threshold=0.1):
    # Check if the shoulders, hips, and ankles are aligned
    shoulder_hip_distance = calculate_distance(shoulder, hip)
    hip_ankle_distance = calculate_distance(hip, ankle)

    # If the distances are within a threshold, assume plank position
    if shoulder_hip_distance < threshold and hip_ankle_distance < threshold:
        return True
    else:
        return False

# Function to count plank time
def count_plank_time(shoulder, hip, ankle, in_plank, start_time):
    # Check if in plank position
    if check_plank(shoulder, hip, ankle):
        # If not already in plank, start counting time
        if not in_plank:
            start_time = time.time()
            in_plank = True
        else:
            # If already in plank, calculate elapsed time
            elapsed_time = time.time() - start_time
            return in_plank, start_time, elapsed_time
    else:
        # If not in plank position, reset timer
        in_plank = False
        start_time = None

    return in_plank, start_time, 0

plank_timer = 0
in_plank = False
start_time = None

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            # Extract landmarks for plank
            landmarks = results.pose_landmarks.landmark
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Count plank time
            in_plank, start_time, elapsed_time = count_plank_time(left_shoulder, left_hip, left_ankle, in_plank, start_time)
            plank_timer += elapsed_time

        except:
            pass

        # Render plank timer
        cv2.rectangle(image, (0, 0), (250, 60), (0, 0, 0), -1)
        cv2.putText(image, 'Plank Time:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "{:.2f}".format(plank_timer) + ' s', (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
