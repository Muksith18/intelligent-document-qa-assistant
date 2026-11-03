
import chromadb
import numpy as np

DATABASE_PATH = "./chroma_db"


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=DATABASE_PATH
        )

    def create_collection(self, document_id):

        collection = self.client.get_or_create_collection(
            name=f"document_{document_id}",
            metadata={"hnsw:space": "cosine"}
        )

        return collection

    def add_documents(self, collection, chunks, embeddings):

        batch_size = 500

        for start in range(0, len(chunks), batch_size):

            batch_chunks = chunks[start:start + batch_size]

            batch_embeddings = embeddings[
                start:start + batch_size
            ]

            collection.upsert(

                ids=[
                    str(chunk["chunk_id"])
                    for chunk in batch_chunks
                ],

                embeddings=batch_embeddings.tolist(),

                documents=[
                    chunk["text"]
                    for chunk in batch_chunks
                ],

                metadatas=[
                    {
                        "chunk_id": chunk["chunk_id"],
                        "page_number": chunk["page_number"],
                        "document_id": str(collection.name)
                    }
                    for chunk in batch_chunks
                ]
            )

        return collection.count()
