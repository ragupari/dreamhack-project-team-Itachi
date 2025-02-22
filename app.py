import cv2
import numpy as np
import time
import mediapipe as mp
from flask import Flask, Response, render_template,request
from flask import jsonify


app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)  # Capture from webcam

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)



rep_count = 0  # Global variable to track reps
feedback = "hiiiii"  # Global variable to store feedback


@app.route("/get_reps")
def get_reps():
    """API to get the current rep count"""
    return jsonify({"reps": rep_count})

@app.route("/get_feedback")
def get_feedback():

    """API to get the current feedback"""
    return jsonify({"feedback": feedback})

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/exercisepage")
def exercisepage():
    exercise = request.args.get("exercise")  # Get parameter from URL
    return render_template("exercise.html", exercise=exercise)


@app.route("/video_feed")
def video_feed():
    return Response(bicep_curls(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
