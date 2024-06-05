import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from rag.rag_setup import rag_chain

load_dotenv()

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World! </p>"


@app.route("/api")
def api_key():
    return f"API KEY: ${os.getenv("API_KEY")}"


@app.route("/vector-search")
def vector_search():
    query = request.args.get('query')

    custom_projection = {'$project': {
        '_id': 0,
        os.getenv("EMBEDDING_KEY"): 0}}

    chain = rag_chain(custom_projection)

    response = chain.invoke(query)

    return jsonify(response)
