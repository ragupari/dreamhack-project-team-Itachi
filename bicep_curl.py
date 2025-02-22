from flask import Blueprint, render_template, Response, jsonify
import cv2
import numpy as np
import time
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

bicep_curls = Blueprint('bicep_curls', __name__)

rep_count = 0  # To track repetitions
feedback = "Let's go"  # To store feedback message

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Capture from the default webcam

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def process_bicep_curls():
    """Main function to track bicep curls and provide feedback."""
    global rep_count
    global feedback
    
    stage = None
    arm_extended = False
    holding_arm_not_extended = False
    last_time_arm_not_extended = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 3)  # Flip the frame horizontally for mirror view
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate the angle between shoulder, elbow, and wrist
                angle = calculate_angle(shoulder, elbow, wrist)

                # Bicep curl stage logic
                if angle > 160:  # Arm is fully extended (down position)
                    stage = "down"
                    arm_extended = True
                    holding_arm_not_extended = False
                elif angle < 30 and stage == "down":  # Arm is curled up (up position)
                    stage = "up"
                    rep_count += 1  # Increment rep count when curl is completed
                    arm_extended = False

                # Feedback logic
                if arm_extended:
                    feedback = "Good job! Extend your arm fully after each rep."
                elif angle < 30:
                    feedback = "Well done! Keep curling your arm back up!"
                elif not arm_extended and stage == "up":
                    feedback = "Don't forget to extend your arm fully between reps."

                # Extra feedback if arm stays bent for too long
                if not arm_extended and not holding_arm_not_extended:
                    last_time_arm_not_extended = time.time()
                    holding_arm_not_extended = True

                if holding_arm_not_extended and (time.time() - last_time_arm_not_extended) > 3:
                    feedback = "Hold on, try to extend your arm properly to complete the rep."

            # Draw the pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Encode frame to send via the streaming server
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

# Route to get the current rep count
@bicep_curls.route("/get_reps")
def get_reps():
    return jsonify({"reps": rep_count})

# Route to get the current feedback message
@bicep_curls.route("/get_feedback")
def get_feedback():
    return jsonify({"feedback": feedback})

# Route to stream video feed
@bicep_curls.route("/video_feed")
def video_feed():
    return Response(process_bicep_curls(), mimetype="multipart/x-mixed-replace; boundary=frame")
