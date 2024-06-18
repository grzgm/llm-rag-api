""" Custom MongoDBAtlasProjectionVectorStore based on MongoDBAtlasVectorSearch
"""
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
)
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.documents import Document

MongoDBDocumentType = TypeVar("MongoDBDocumentType", bound=Dict[str, Any])


class MongoDBAtlasProjectionVectorStore(MongoDBAtlasVectorSearch):
    """Modifed `MongoDB Atlas Vector Search` vector store.
    """

    def _similarity_search_with_score(
            self,
            embedded_query: List[float],
            k: int = 4,
            pre_filter: Optional[Dict] = None,
            post_filter_pipeline: Optional[List[Dict]] = None,
            custom_projection: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """Return MongoDB documents most similar to the given query and their scores.

        Uses the vectorSearch operator available in MongoDB Atlas Search.
        For more: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/

        Args:
            embedded_query: Embedded query to look up documents similar to.
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: (Optional) dictionary of argument(s) to prefilter document
                fields on.
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages
                following the vectorSearch stage.
            custom_projection: (Optional) Custom document projection returned from
                the MongoDB collection.

        Returns:
            List of documents most similar to the query and their scores.
        """
        params = {
            "index": self._index_name,
            "path": self._embedding_key,
            "queryVector": embedded_query,
            "numCandidates": k * 10,
            "limit": k,
        }
        if pre_filter:
            params["filter"] = pre_filter
        query = {"$vectorSearch": params}

        pipeline = [
            query,
            {"$set": {"score": {"$meta": "vectorSearchScore"}}},
        ]

        if custom_projection:
            pipeline.append(custom_projection)

        if post_filter_pipeline is not None:
            pipeline.extend(post_filter_pipeline)

        cursor = self._collection.aggregate(pipeline)  # type: ignore[arg-type]
        docs = []
        for res in cursor:
            score = res.pop("score")
            text = str(res)
            docs.append((Document(page_content=text), score))
        return docs

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            pre_filter: Optional[Dict] = None,
            post_filter_pipeline: Optional[List[Dict]] = None,
            custom_projection: Optional[Dict] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return MongoDB documents most similar to the given query and their scores.

        Uses the vectorSearch operator available in MongoDB Atlas Search.
        For more: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/

        Args:
            query: Text to look up documents similar to.
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: (Optional) dictionary of argument(s) to prefilter document
                fields on.
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages
                following the vectorSearch stage.
            custom_projection: (Optional) Custom document projection returned from
                the MongoDB collection.

        Returns:
            List of documents most similar to the query and their scores.
        """
        embedded_query = self._embedding.embed_query(query)
        docs = self._similarity_search_with_score(
            embedded_query,
            k=k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            custom_projection=custom_projection,
            **kwargs,
        )
        return docs

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            pre_filter: Optional[Dict] = None,
            post_filter_pipeline: Optional[List[Dict]] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return MongoDB documents most similar to the given query.

        Uses the vectorSearch operator available in MongoDB Atlas Search.
        For more: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/

        Args:
            query: Text to look up documents similar to.
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: (Optional) dictionary of argument(s) to prefilter document
                fields on.
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages
                following the vectorSearch stage.

        Returns:
            List of documents most similar to the query and their scores.
        """
        additional = kwargs.get("additional")
        docs_and_scores = self.similarity_search_with_score(
            query,
            k=k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            **kwargs,
        )

        if additional and "similarity_score" in additional:
            for doc, score in docs_and_scores:
                doc.metadata["score"] = score
        return [doc for doc, _ in docs_and_scores]
