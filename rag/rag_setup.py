"""Collection of functions used to set up RAG infrastructure"""
import os
from typing import (
    Dict,
    Optional,
)

from pymongo import MongoClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSerializable
from langchain_community.llms.ollama import Ollama
from langchain.chains.query_constructor.base import AttributeInfo
from rag.projection_self_query_retriever import SelfQueryRetriever
from rag.projection_vector_store import MongoDBAtlasProjectionVectorStore
from rag.projection_retriever import MongoDBAtlasProjectionRetriever
from rag.prompt_template import PROMPT

CLIENT = MongoClient(os.getenv("MONGO_URI"))

EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

LLM = Ollama(model="phi3:3.8b")

# LLM model with enabled feature to format response into JSON object
JSON_LLM = Ollama(model="phi3:3.8b", format="json")

# Document content description for self query vector search
DOCUMENT_CONTENT_DESCRIPTION = os.getenv("DOCUMENT_CONTENT_DESCRIPTION")

# Metadata required for self query vector search
METADATA_FIELD_INFO = [
    AttributeInfo(
        name="year",
        description="Not a Date, just the INTEGER represending year. The year the movie was released, represented as an integer",
        type="integer",
    ),
    AttributeInfo(
        name="imdb.rating", description="A 1-10 rating for the movie", type="integer"
    ),
    AttributeInfo(
        name="genres",
        description="The genres of the movie. One of ['Science fiction', 'Comedy', 'Drama', 'Thriller', 'Romance', 'Action', 'Animated']",
        type="string",
    ),
]

OUTPUT_PARSER = StrOutputParser()


def mongo_connection():
    """Return MongoDB Collection.

    Uses the values form .env file.

    Returns:
        Connection to MongoDB Collection specified in .env file.
    """
    db_name = os.getenv("DB_NAME")
    collection_name = os.getenv("COLL_NAME")
    collection = CLIENT[db_name][collection_name]

    return collection


def vector_search_chain(custom_projection: Optional[Dict] = None, k: int = 4) -> MongoDBAtlasProjectionRetriever:
    """Return Chain consisting of retriever for MongoDB Vector Search.

    Uses `MongoDBAtlasProjectionVectorStore` and `MongoDBAtlasProjectionRetriever`.

    Args:
        custom_projection: (Optional) Custom document projection returned from
            the MongoDB collection. Defaults to None.
        k: (Optional) number of documents to return. Defaults to 4.

    Returns:
        Chain for MongoDB Vector Search.
    """
    collection = mongo_connection()
    vectorstore = MongoDBAtlasProjectionVectorStore(
        collection, EMBEDDING_MODEL, embedding_key=os.getenv("EMBEDDING_KEY"), index_name=os.getenv("INDEX_NAME"))

    retriever = MongoDBAtlasProjectionRetriever(movie_vectorstore=vectorstore, search_kwargs={
        "custom_projection": custom_projection, "k": k})

    return retriever


def rag_chain(custom_projection: Optional[Dict] = None, k: int = 4) -> RunnableSerializable[str, str]:
    """Return Chain consisting of retriever, prompt template, LLM and output parser for RAG based on MongoDB Documents.

    Uses `MongoDBAtlasProjectionVectorStore`, `MongoDBAtlasProjectionRetriever`, `RunnableParallel`.

    Args:
        custom_projection: (Optional) Custom document projection returned from
            the MongoDB collection. Defaults to None.
        k: (Optional) number of documents to return. Defaults to 4.

    Returns:
        Chain for RAG.
    """
    collection = mongo_connection()
    vectorstore = MongoDBAtlasProjectionVectorStore(
        collection, EMBEDDING_MODEL, embedding_key=os.getenv("EMBEDDING_KEY"), index_name=os.getenv("INDEX_NAME"))

    retriever = MongoDBAtlasProjectionRetriever(movie_vectorstore=vectorstore, search_kwargs={
        "custom_projection": custom_projection, "k": k})

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    chain = setup_and_retrieval | PROMPT | LLM | OUTPUT_PARSER

    return chain


def self_querying_vector_search_chain(custom_projection: Optional[Dict] = None, k: int = 4) -> SelfQueryRetriever:
    """Return Chain consisting of self query retriever for self querying MongoDB Vector Search.

    Uses `MongoDBAtlasProjectionVectorStore` and `SelfQueryRetriever`.

    Args:
        custom_projection: (Optional) Custom document projection returned from
            the MongoDB collection. Defaults to None.
        k: (Optional) number of documents to return. Defaults to 4.

    Returns:
        Chain for MongoDB self query Vector Search.
    """
    collection = mongo_connection()
    vectorstore = MongoDBAtlasProjectionVectorStore(
        collection, EMBEDDING_MODEL, embedding_key=os.getenv("EMBEDDING_KEY"), index_name=os.getenv("INDEX_NAME"))

    document_content_description = "Brief summary of a movie"
    retriever = SelfQueryRetriever.from_llm(
        JSON_LLM,
        vectorstore,
        document_content_description,
        METADATA_FIELD_INFO,
        search_kwargs={
            "custom_projection": custom_projection, "k": k}
    )

    return retriever


def self_querying_rag_chain(custom_projection: Optional[Dict] = None, k: int = 4) -> RunnableSerializable[str, str]:
    """Return Chain consisting of self query retriever, prompt template, LLM and output parser for
    self querying RAG based on MongoDB Documents.

    Uses `MongoDBAtlasProjectionVectorStore`, `SelfQueryRetriever`, `RunnableParallel`.

    Args:
        custom_projection: (Optional) Custom document projection returned from
            the MongoDB collection. Defaults to None.
        k: (Optional) number of documents to return. Defaults to 4.

    Returns:
        Chain for self query RAG.
    """
    collection = mongo_connection()
    vectorstore = MongoDBAtlasProjectionVectorStore(
        collection, EMBEDDING_MODEL, embedding_key=os.getenv("EMBEDDING_KEY"), index_name=os.getenv("INDEX_NAME"))

    retriever = SelfQueryRetriever.from_llm(
        JSON_LLM,
        vectorstore,
        DOCUMENT_CONTENT_DESCRIPTION,
        METADATA_FIELD_INFO,
        search_kwargs={
            "custom_projection": custom_projection, "k": k}
    )

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    chain = setup_and_retrieval | PROMPT | LLM | OUTPUT_PARSER

    return chain
