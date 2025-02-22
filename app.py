import cv2
import numpy as np
import time
import mediapipe as mp
from flask import Flask, Response, render_template, request, jsonify
from bicep_curl import bicep_curls
import asyncio
from chat import agent
from hand_physio import hand_physio


# Set up the Flask app
app = Flask(__name__)
app.register_blueprint(bicep_curls, url_prefix='/bicep_curls')
app.register_blueprint(hand_physio, url_prefix='/hand_physio')

# Route to serve the main exercise page
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chat.html")





# Route for the exercise page with specific exercise selected
@app.route("/exercisepage")
def exercisepage():
    exercise = request.args.get("exercise")  # Get exercise from the URL
    url = request.args.get("url")  # Get video URL from the URL
    return render_template("exercise.html", exercise=exercise, url = url)

async def get_response(query, messages):
    response = await agent.run(query, message_history=messages)
    return response.data, response.all_messages()


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    
    if not user_input:
        return jsonify({"error": "No input received"})

    # Maintain session messages (You can use a database for real users)
    if "messages" not in request.json:
        messages = []
    else:
        messages = request.json["messages"]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response, messages = loop.run_until_complete(get_response(user_input, messages))

    return jsonify({"response": response, "messages": messages})



# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)


