import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate distance between two points
def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = math.degrees(radians)
    angle = angle + 360 if angle < 0 else angle
    return angle

# Function to check squat position
def check_squat(hip, knee, ankle, threshold=0.1):
    # Check if the knees are bent at a proper angle
    hip_knee_distance = calculate_distance(hip, knee)
    knee_ankle_distance = calculate_distance(knee, ankle)

    # If the distances are within a threshold, assume squat position
    if knee_ankle_distance < hip_knee_distance * threshold:
        return True
    else:
        return False

# Function to count squats
def count_squats(hip, knee, ankle, in_squat, squat_count):
    # Check if in squat position
    if check_squat(hip, knee, ankle):
        # If not already in squat, count as a squat
        if not in_squat:
            squat_count += 1
            print("Squat Count:", squat_count)
            in_squat = True
    else:
        in_squat = False

    return in_squat, squat_count

# Simple smoothing over n frames
def smooth_values(values, window_size=5):
    smoothed_values = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        end = i + 1
        smoothed_value = np.mean(values[start:end])
        smoothed_values.append(smoothed_value)
    return smoothed_values

squat_count = 0
in_squat = False

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
            # Extract landmarks for squats
            landmarks = results.pose_landmarks.landmark
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Smooth the landmark values
            smoothed_left_hip = smooth_values(left_hip)
            smoothed_left_knee = smooth_values(left_knee)
            smoothed_left_ankle = smooth_values(left_ankle)

            # Calculate the angle between hip, knee, and ankle
            angle = calculate_angle(smoothed_left_hip, smoothed_left_knee, smoothed_left_ankle)

            # Count squats
            in_squat, squat_count = count_squats(smoothed_left_hip, smoothed_left_knee, smoothed_left_ankle, in_squat, squat_count)

            # Debugging output
            print("Hip-Knee Distance:", calculate_distance(smoothed_left_hip, smoothed_left_knee))
            print("Knee-Ankle Distance:", calculate_distance(smoothed_left_knee, smoothed_left_ankle))
            print("In Squat:", in_squat)
            print("Angle:", angle)

            # Display the angle on the video frame
            cv2.putText(image, 'Angle: {:.2f}'.format(angle),
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print("Error:", e)

        # Render squat count
        cv2.putText(image, 'Squat Count: ' + str(squat_count),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
