import os
from pymongo import MongoClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


def embed_collection(collection):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for doc in collection.find({"embedding": {"$exists": False}}):
        if "vector" not in doc:
            process_document(collection, doc, model)
        else:
            print(f"Vector already computed for document ID: {doc['_id']}")


def process_document(collection, doc, model):
    if "title" in doc:
        movie_id = doc["_id"]
        title = doc["title"]
        print(f"Computing vector for title: {title}")

        text = f'Title: "{title}"\n'
        fullplot = doc.get("fullplot")
        if fullplot:
            text += f'Fullplot: {fullplot}'

        vector = model.encode(text).tolist()
        update_fields = {
            "embedding": vector,
            "title": title,
            "fullplot": fullplot
        }
        collection.update_one({"_id": movie_id}, {"$set": update_fields}, upsert=True)
        print(f"Vector computed and stored for document ID: {movie_id}")


def clone_collection(db, old_coll_name, new_coll_name):
    cloned_collection = db[old_coll_name].aggregate([{"$match": {}}])
    db[new_coll_name].insert_many(cloned_collection)
    print(f"Cloned collection '{old_coll_name}' to '{new_coll_name}'")


def delete_and_output_documents(collection, query):
    documents = collection.find(query)
    for document in documents:
        print(
            document.get("text"),
            document.get("fullplot"),
            document.get("description"),
            document.get("year"),
            document.get("rating"),
            document.get("director"),
            document.get("genre"),
        )
    result = collection.delete_many(query)
    print(f"Deleted {result.deleted_count} documents.")


def main():
    client = MongoClient(os.getenv("MONGO_URI"))
    db_name = os.getenv("DB_NAME")
    collection_name = os.getenv("COLL_NAME")
    collection = client[db_name][collection_name]

    try:
        embed_collection(collection)
        query = {"year": 1993, "rating": 7.7, "genre": "science fiction"}
        delete_and_output_documents(collection, query)
    finally:
        client.close()


if __name__ == "__main__":
    main()
