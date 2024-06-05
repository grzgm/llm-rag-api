import os
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from pymongo import MongoClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from rag.projection_vector_store import MongoDBAtlasProjectionVectorStore
from rag.projection_retriever import MongoDBAtlasProjectionRetriever
from rag.prompt_template import PROMPT

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        print(f"LLM Formated Prompt: \n {prompts}")


LLM = Ollama(
    model="llama2", callback_manager=CallbackManager([MyCustomHandler()]))

output_parser = StrOutputParser()


def mongo_connection():
    client = MongoClient(os.getenv("MONGO_URI"))
    db_name = os.getenv("DB_NAME")
    collection_name = os.getenv("COLL_NAME")
    collection = client[db_name][collection_name]

    return collection


def vector_search_chain(custom_projection=None, k=4):
    collection = mongo_connection()
    vectorstore = MongoDBAtlasProjectionVectorStore(
        collection, embedding_model, embedding_key=os.getenv("EMBEDDING_KEY"), index_name=os.getenv("INDEX_NAME"))

    retriever = MongoDBAtlasProjectionRetriever(movie_vectorstore=vectorstore, search_kwargs={
        "custom_projection": custom_projection, "k": k})

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    chain = setup_and_retrieval

    return chain
def rag_chain(custom_projection=None, k=4):
    collection = mongo_connection()
    vectorstore = MongoDBAtlasProjectionVectorStore(
        collection, embedding_model, embedding_key=os.getenv("EMBEDDING_KEY"), index_name=os.getenv("INDEX_NAME"))

    retriever = MongoDBAtlasProjectionRetriever(movie_vectorstore=vectorstore, search_kwargs={
        "custom_projection": custom_projection, "k": k})

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    chain = setup_and_retrieval | PROMPT | LLM | output_parser

    return chain
