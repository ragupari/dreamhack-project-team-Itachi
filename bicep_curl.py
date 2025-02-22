import cv2
import numpy as np
import time
import mediapipe as mp
from flask import Flask, Response, render_template,request
from flask import jsonify


def bicep_curls():

    global rep_count
    global feedback
    rep_count = 0
    stage = None
    arm_extended = False
    holding_arm_not_extended = False
    last_time_arm_not_extended = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 3)  # Flip the frame horizontally
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

                # Calculate angle between shoulder, elbow, and wrist
                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:  # Arm is extended
                    stage = "down"
                    arm_extended = True
                    holding_arm_not_extended = False
                elif angle < 30 and stage == "down":  # Arm is curled up
                    stage = "up"
                    rep_count += 1  # Increment rep count
                    arm_extended = False

                # Feedback logic: Ensure priority for important messages
                if arm_extended:
                    feedback = "Good job! Extend your arm fully after each rep."
                elif angle < 30:
                    feedback = "Well done! Keep curling your arm back up!"
                elif not arm_extended and stage == "up":
                    feedback = "Don't forget to extend your arm fully between reps."

                # Additional feedback if the arm has been held in a bent position for too long
                if not arm_extended and not holding_arm_not_extended:
                    last_time_arm_not_extended = time.time()
                    holding_arm_not_extended = True

                if holding_arm_not_extended and (time.time() - last_time_arm_not_extended) > 3:
                    feedback = "Hold on, try to extend your arm properly to complete the rep."

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Encode frame to send via a streaming server
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
