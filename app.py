import os
from flask import Flask
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World! </p>"

@app.route("/api")
def api_key():
    return f"API KEY: ${os.getenv("API_KEY")}"