import cv2
import mediapipe as mp
import math
import time

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

# Initialize video capture
vid = cv2.VideoCapture(0)
vid.set(3, 1280)

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

while True:
    ret, frame = vid.read()
    if not ret:
        break

    # Convert frame to RGB
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = Hands.process(RGBframe)
    
    if result.multi_hand_landmarks:
        for handLm in result.multi_hand_landmarks:
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

            # Display encouragement messages
            current_time = time.time()

            if 135 < angle < 180:
                if suggestion_start_time is None:
                    suggestion_start_time = current_time
                elif current_time - suggestion_start_time >= 1:
                    cv2.putText(frame, "Slowly close your fingers", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    suggestion_displayed = True
            else:
                suggestion_start_time = None
                suggestion_displayed = False

            if angle < 135:
                cv2.putText(frame, "Slowly release your fingers", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display information on frame
            cv2.putText(frame, f'Angle: {int(angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Hand is: {hand_state}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Counter: {counter}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the video frame
    cv2.imshow("Hand Gesture Tracker", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
