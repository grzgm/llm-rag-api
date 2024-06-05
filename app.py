import os
import json
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from rag.rag_setup import vector_search_chain, rag_chain

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
    docs_num = int(request.args.get('docs_num')) if request.args.get('docs_num') else 3

    custom_projection = {'$project': {
        '_id': 0,
        os.getenv("EMBEDDING_KEY"): 0}}

    chain = vector_search_chain(custom_projection, docs_num)

    response = chain.invoke(query)

    return jsonify(response)


@app.route("/rag")
def rag():
    query = request.args.get('query')
    custom_projection = request.args.get('projection')
    docs_num = int(request.args.get('docs_num')) if request.args.get('docs_num') else 3

    if not custom_projection:
        custom_projection = {'$project': {
            '_id': 0,
            os.getenv("EMBEDDING_KEY"): 0}}
    else:
        custom_projection = json.loads(custom_projection)

    chain = rag_chain(custom_projection, docs_num)

    response = chain.invoke(query)

    return jsonify(response)
