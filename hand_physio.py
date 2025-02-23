from flask import Flask, render_template, Response, Blueprint, jsonify
import cv2
import mediapipe as mp
import math
import time
from openai import OpenAI
import os

hand_physio = Blueprint('hand_physio', __name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is missing. Set it as an environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)
# Fixed angle calculation function
def calculate_angle(p1, p2, p3):
    try:
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
    except Exception as e:
        print(f"Error in angle calculation: {e}")
        return 0

# Function to generate feedback using OpenAI
def generate_feedback(angle, hand_state):
    try:
        prompt = f"The user hand is currently {hand_state} at {int(angle)}°. If closed, open slowly; if open, close gently. Provide a short, constructive command (10 words or less) to improve hand movement."

        # Use the new API interface
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.7,
        )
    
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Feedback unavailable."



# Initialize MediaPipe hand detector
mphands = mp.solutions.hands
Hands = mphands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
mpdraw = mp.solutions.drawing_utils

# Initialize variables
counter = 0
previous_state = "Closed"
open_state_detected = False
closed_state_detected = False
suggestion_start_time = None
suggestion_displayed = False
previous_angle = 0
feedback = "Let's Go!"
vid = None
def generate_frames():
    # Initialize video capture
    global vid
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        print("Error: Could not open camera.")
        exit()
    global previous_angle, feedback, suggestion_start_time, counter, previous_state, open_state_detected, closed_state_detected

    while True:
        ret, frame = vid.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to RGB
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = Hands.process(RGBframe)
        
        if result.multi_hand_landmarks:
            for handLm in result.multi_hand_landmarks:
                try:
                    # Draw landmarks
                    mpdraw.draw_landmarks(frame, handLm, mphands.HAND_CONNECTIONS,
                                          mpdraw.DrawingSpec(color=(0, 0, 255), circle_radius=7, thickness=cv2.FILLED),
                                          mpdraw.DrawingSpec(color=(0, 255, 0), thickness=7))

                    # Get frame dimensions
                    h, w, _ = frame.shape
                    
                    # Extract key points for angle calculation
                    wrist = handLm.landmark[0]
                    thumb_base = handLm.landmark[1]
                    thumb_tip = handLm.landmark[4]

                    # Convert to pixel coordinates
                    wrist_coords = (int(wrist.x * w), int(wrist.y * h))
                    thumb_base_coords = (int(thumb_base.x * w), int(thumb_base.y * h))
                    thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))

                    # Calculate angle
                    angle = calculate_angle(wrist_coords, thumb_base_coords, thumb_tip_coords)

                    # Smooth the angle (optional)
                    angle = 0.7 * angle + 0.3 * previous_angle
                    previous_angle = angle

                    # Determine hand state based on angle
                    if angle < 130:
                        hand_state = "Closed"
                    elif angle > 165:
                        hand_state = "Open"
                    else:
                        hand_state = previous_state

                    # Handle state change for counting
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
    
                    # Generate feedback using OpenAI
                    if time.time() - (suggestion_start_time or 0) >= 3:  # Generate feedback every 1 second
                        feedback = generate_feedback(angle, hand_state)
                        suggestion_start_time = time.time()

                    # Display feedback on frame

                    # Display information on frame

                except Exception as e:
                    print(f"Error processing hand landmarks: {e}")

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@hand_physio.route("/get_feedback")
def get_feedback():
    return jsonify({"feedback": feedback})

@hand_physio.route("/start")
def reset():
    global counter, feedback
    counter = 0  # Reset to 0
    feedback = "Let's go"
    return Response("Reset successful")

@hand_physio.route("/stop")
def stop():
    global vid
    vid.release()
    return Response("caps released")

@hand_physio.route("/get_reps")
def get_reps():
    return jsonify({"reps": counter})

@hand_physio.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
