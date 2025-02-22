from flask import Blueprint, render_template, Response, jsonify, request
import cv2
import numpy as np
import time
import mediapipe as mp
import math

hand_physio = Blueprint('hand_physio', __name__)

# Fixed angle calculation function
def calculate_angle(p1, p2, p3):
    # Extract coordinates
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Vectors p2p1 and p2p3
    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x2, y3 - y2)

    # Dot product and magnitudes of the vectors
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

    # Avoid division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0

    # Angle in radians
    angle = math.acos(dot_product / (magnitude_v1 * magnitude_v2))
    
    # Convert to degrees
    return math.degrees(angle)

# Global variables
vid = None  # Declare it globally so it can be accessed from stop route
counter = 0
feedback = "Let's go"

def hand_physio_process():
    global vid, counter, feedback

    if vid is None:
        vid = cv2.VideoCapture(0)
        vid.set(3, 1280)  # Set the capture resolution

    mphands = mp.solutions.hands
    Hands = mphands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    mpdraw = mp.solutions.drawing_utils

    previous_state = "Closed"
    open_state_detected = False
    closed_state_detected = False
    suggestion_start_time = None

    while True:
        ret, frame = vid.read()
        if not ret:
            feedback = "Could not capture frame."
            break

        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = Hands.process(RGBframe)
        
        if result.multi_hand_landmarks:
            for handLm in result.multi_hand_landmarks:
                mpdraw.draw_landmarks(frame, handLm, mphands.HAND_CONNECTIONS,
                                    mpdraw.DrawingSpec(color=(0, 0, 255), circle_radius=7, thickness=cv2.FILLED),
                                    mpdraw.DrawingSpec(color=(0, 255, 0), thickness=7))

                h, w, _ = frame.shape
                wrist = handLm.landmark[0]
                thumb_base = handLm.landmark[1]
                thumb_tip = handLm.landmark[4]

                wrist_coords = (int(wrist.x * w), int(wrist.y * h))
                thumb_base_coords = (int(thumb_base.x * w), int(thumb_base.y * h))
                thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))

                angle = calculate_angle(wrist_coords, thumb_base_coords, thumb_tip_coords)

                if angle < 130:
                    hand_state = "Closed"
                elif angle > 165:
                    hand_state = "Open"
                else:
                    hand_state = previous_state

                if hand_state != previous_state:
                    if hand_state == "Open" and closed_state_detected:
                        counter += 1
                        open_state_detected = False
                        closed_state_detected = False
                    elif hand_state == "Closed" and open_state_detected:
                        closed_state_detected = True

                    previous_state = hand_state

                    if hand_state == "Open":
                        open_state_detected = True

                current_time = time.time()

                if 135 < angle < 180:
                    if suggestion_start_time is None:
                        suggestion_start_time = current_time
                    elif current_time - suggestion_start_time >= 1:
                        feedback = "Slowly close your fingers"
                else:
                    suggestion_start_time = None

                if angle < 135:
                    feedback = "Slowly release your fingers"
        
        cv2.imshow("Hand Gesture Tracker", frame)
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@hand_physio.route("/get_reps")
def get_reps():
    global counter
    return jsonify({"reps": counter})

@hand_physio.route("/start")
def reset():
    global counter, feedback
    counter = 0  # Reset to 0
    feedback = "Let's go"
    return Response("Reset successful")

@hand_physio.route("/stop")
def stop():
    global vid
    if vid is not None:
        vid.release()
        vid = None  # Set it to None to prevent future accesses
    return Response("Video capture stopped")

@hand_physio.route("/get_feedback")
def get_feedback():
    global feedback
    return jsonify({"feedback": feedback})

@hand_physio.route("/video_feed")
def video_feed():
    try:
        return Response(hand_physio_process(), mimetype="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        return Response(f"Error occurred: {str(e)}")
