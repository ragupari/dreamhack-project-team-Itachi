import cv2
import numpy as np
import time
import mediapipe as mp
from flask import Flask, Response, render_template, request, jsonify
from bicep_curl import bicep_curls



# Set up the Flask app
app = Flask(__name__)
app.register_blueprint(bicep_curls, url_prefix='/bicep_curls')

# Route to serve the main exercise page
@app.route("/")
def index():
    return render_template("index.html")





# Route for the exercise page with specific exercise selected
@app.route("/exercisepage")
def exercisepage():
    exercise = request.args.get("exercise")  # Get exercise from the URL
    return render_template("exercise.html", exercise=exercise)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)