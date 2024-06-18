import os
import json
import langchain_core.exceptions
import lark.exceptions
import pymongo.errors
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from langchain_core.documents import Document
from rag.rag_setup import vector_search_chain, rag_chain, self_querying_vector_search_chain, self_querying_rag_chain

load_dotenv()

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World! </p>"


@app.route("/vector-search")
def vector_search():
    query = request.args.get('query')
    docs_num = int(request.args.get('docs_num')) if request.args.get('docs_num') else 3

    custom_projection = {'$project': {
        '_id': 0,
        os.getenv("EMBEDDING_KEY"): 0}}

    chain = vector_search_chain(custom_projection, docs_num)

    try:
        docs = chain.invoke(query)
    except Exception as e:
        print("An error occurred:", e)
        return "There was an unknown problem"

    docs = docs_to_json(docs)

    return jsonify(docs)


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

    try:
        response = chain.invoke(query)
    except Exception as e:
        print("An error occurred:", e)
        return "There was an unknown problem"

    return jsonify(response)


@app.route("/sq-vector-search")
def self_querying_vector_search():
    query = request.args.get('query')
    custom_projection = request.args.get('projection')
    docs_num = int(request.args.get('docs_num')) if request.args.get('docs_num') else 3

    if not custom_projection:
        custom_projection = {'$project': {
            '_id': 0,
            os.getenv("EMBEDDING_KEY"): 0}}
    else:
        custom_projection = json.loads(custom_projection)

    chain = self_querying_vector_search_chain(custom_projection, docs_num)

    try:
        docs = chain.invoke(query)
    except langchain_core.exceptions.OutputParserException as e:
        print("An error occurred:", e)
        return "There was a problem with parsing filters"
    except lark.exceptions.UnexpectedToken as e:
        print("An error occurred:", e)
        return "There was a problem with parsing filters"
    except pymongo.errors.OperationFailure as e:
        print("An error occurred:", e)
        return "There was a problem with filters in MongoDB"
    except Exception as e:
        print("An error occurred:", e)
        return "There was an unknown problem"

    docs = docs_to_json(docs)

    return jsonify(docs)


@app.route("/sq-rag")
def self_querying_rag():
    query = request.args.get('query')
    custom_projection = request.args.get('projection')
    docs_num = int(request.args.get('docs_num')) if request.args.get('docs_num') else 3

    if not custom_projection:
        custom_projection = {'$project': {
            '_id': 0,
            os.getenv("EMBEDDING_KEY"): 0}}
    else:
        custom_projection = json.loads(custom_projection)

    chain = self_querying_rag_chain(custom_projection, docs_num)

    try:
        response = chain.invoke(query)
    except langchain_core.exceptions.OutputParserException as e:
        print("An error occurred:", e)
        return "There was a problem with parsing filters"
    except lark.exceptions.UnexpectedToken as e:
        print("An error occurred:", e)
        return "There was a problem with parsing filters"
    except pymongo.errors.OperationFailure as e:
        print("An error occurred:", e)
        return "There was a problem with filters in MongoDB"
    except Exception as e:
        print("An error occurred:", e)
        return "There was an unknown problem"

    return jsonify(response)


def docs_to_json(docs: list[Document]) -> list:
    """Convert Documents to JSON format.

    Removes '_id' field for proper JSON conversion.

    Returns:
        List of Documents converted to JSON format.
    """
    json_docs = []
    for doc in docs:
        doc.metadata.pop('_id', None)
        json_docs.append(doc.to_json())
    return json_docs
