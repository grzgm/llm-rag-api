import os
import json
import langchain_core.exceptions
import lark.exceptions
import pymongo.errors
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from langchain_core.documents import Document
from rag.rag_setup import (
    vector_search_chain,
    rag_chain,
    self_querying_vector_search_chain,
    self_querying_rag_chain
)

load_dotenv()

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World! </p>"

def get_custom_projection(custom_projection):
    if not custom_projection:
        return {'$project': {
            '_id': 0,
            os.getenv("EMBEDDING_KEY"): 0}}
    return json.loads(custom_projection)

def process_request(chain_func, query, custom_projection, docs_num):
    custom_projection = get_custom_projection(custom_projection)
    chain = chain_func(custom_projection, docs_num)
    try:
        result = chain.invoke(query)
    except langchain_core.exceptions.OutputParserException as e:
        print("An error occurred:", e)
        return "There was a problem with parsing filters", 400
    except lark.exceptions.UnexpectedToken as e:
        print("An error occurred:", e)
        return "There was a problem with parsing filters", 400
    except pymongo.errors.OperationFailure as e:
        print("An error occurred:", e)
        return "There was a problem with filters in MongoDB", 500
    except Exception as e:
        print("An error occurred:", e)
        return "There was an unknown problem", 500
    if isinstance(result, list):
        result = docs_to_json(result)
    return jsonify(result), 200

@app.route("/vector-search")
def vector_search():
    query = request.args.get('query')
    custom_projection = request.args.get('projection')
    docs_num = int(request.args.get('docs_num')) if request.args.get('docs_num') else 3
    return process_request(vector_search_chain, query, custom_projection, docs_num)

@app.route("/rag")
def rag():
    query = request.args.get('query')
    custom_projection = request.args.get('projection')
    docs_num = int(request.args.get('docs_num')) if request.args.get('docs_num') else 3
    return process_request(rag_chain, query, custom_projection, docs_num)

@app.route("/sq-vector-search")
def self_querying_vector_search():
    query = request.args.get('query')
    custom_projection = request.args.get('projection')
    docs_num = int(request.args.get('docs_num')) if request.args.get('docs_num') else 3
    return process_request(self_querying_vector_search_chain, query, custom_projection, docs_num)

@app.route("/sq-rag")
def self_querying_rag():
    query = request.args.get('query')
    custom_projection = request.args.get('projection')
    docs_num = int(request.args.get('docs_num')) if request.args.get('docs_num') else 3
    return process_request(self_querying_rag_chain, query, custom_projection, docs_num)

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

if __name__ == "__main__":
    app.run()
