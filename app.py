from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embedding
from src.helper import chatbot
import os
from dotenv import load_dotenv

import markdown

app = Flask(__name__)


@app.route("/")

def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    chat_history=[]
    msg = request.form["msg"]
    input = msg
    print(input)
    result,chat_history=chatbot(input,chat_history)
    result= markdown.markdown(result)
    print('history:',chat_history)
    print("Response : ", result)
    return str(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
