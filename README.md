# RAG-based Flask Application

This project is a Flask-based application providing multiple endpoints for performing vector searches and RAG (Retrieval-Augmented Generation) using MongoDB and Langchain. The endpoints leverage embeddings for document retrieval and query augmentation.

## Table of Contents
- [Installation](#installation)
- [Configuration](#configuration)
- [Endpoints](#endpoints)
  - [Hello World](#hello-world)
  - [Vector Search](#vector-search)
  - [RAG](#rag)
  - [Self-Querying Vector Search](#self-querying-vector-search)
  - [Self-Querying RAG](#self-querying-rag)
- [Usage](#usage)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/grzgm/llm-rag-api
    cd llm-rag-api
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Copy contents from `.env.example` to a `.env` file in the project root and populate it with the necessary environment variables:
    ```
    MONGO_URI=<your_mongodb_uri>
    DB_NAME=<your_database_name>
    COLL_NAME=<your_collection_name>
    INDEX_NAME=<your_index_name>
    EMBEDDING_KEY=<your_embedding_key>
    DOCUMENT_CONTENT_DESCRIPTION=<document_content_description>
    ```

4. Run the Flask application:
    ```bash
    flask run
    ```

## Configuration

Make sure to configure the `.env` file with the correct values for your MongoDB instance and document metadata.

## Endpoints

### Hello World

- **URL:** `/`
- **Method:** `GET`
- **Description:** Returns a simple greeting message to ensure the server is running.
- **Parameters:** None
- **Response:**
    ```html
    <p>Hello, World!</p>
    ```

### Vector Search

- **URL:** `/vector-search`
- **Method:** `GET`
- **Description:** Performs a vector search on the MongoDB collection.
- **Parameters:**

| Parameter     | Type      | Required | Description                                           |
|---------------|-----------|----------|-------------------------------------------------------|
| `query`       | string    | Yes      | The query to search for.                              |
| `projection`  | JSON string | No     | Custom projection for the MongoDB query.              |
| `docs_num`    | integer   | No       | Number of documents to return (default: 3).           |

- **Response:** JSON array of documents matching the search query.

### RAG

- **URL:** `/rag`
- **Method:** `GET`
- **Description:** Performs a RAG operation to retrieve documents and generate responses based on the query.
- **Parameters:**

| Parameter     | Type      | Required | Description                                           |
|---------------|-----------|----------|-------------------------------------------------------|
| `query`       | string    | Yes      | The query to search for.                              |
| `projection`  | JSON string | No     | Custom projection for the MongoDB query.              |
| `docs_num`    | integer   | No       | Number of documents to return (default: 3).           |

- **Response:** JSON object containing the RAG-generated response.

### Self-Querying Vector Search

- **URL:** `/sq-vector-search`
- **Method:** `GET`
- **Description:** Performs a self-querying vector search, which automatically generates filters based on the query.
- **Parameters:**

| Parameter     | Type      | Required | Description                                           |
|---------------|-----------|----------|-------------------------------------------------------|
| `query`       | string    | Yes      | The query to search for.                              |
| `projection`  | JSON string | No     | Custom projection for the MongoDB query.              |
| `docs_num`    | integer   | No       | Number of documents to return (default: 3).           |

- **Response:** JSON array of documents matching the search query with auto-generated filters.

### Self-Querying RAG

- **URL:** `/sq-rag`
- **Method:** `GET`
- **Description:** Performs a self-querying RAG operation to retrieve documents and generate responses based on the query with auto-generated filters.
- **Parameters:**

| Parameter     | Type      | Required | Description                                           |
|---------------|-----------|----------|-------------------------------------------------------|
| `query`       | string    | Yes      | The query to search for.                              |
| `projection`  | JSON string | No     | Custom projection for the MongoDB query.              |
| `docs_num`    | integer   | No       | Number of documents to return (default: 3).           |

- **Response:** JSON object containing the self-querying RAG-generated response.

## Usage

To use the endpoints, send HTTP GET requests with the appropriate parameters to the Flask server. For example, to perform a vector search, use the following curl command:

```bash
curl -G \
  --data-urlencode "query=your_search_query" \
  --data-urlencode "projection={\"field\": 1}" \
  --data-urlencode "docs_num=5" \
  http://localhost:5000/vector-search
```

Make sure to adjust the parameters as needed.