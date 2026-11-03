
import chromadb

from embedding_model import get_embedding_model


DATABASE_PATH = "./chroma_db"

DEFAULT_DISTANCE_THRESHOLD = 0.30
FALLBACK_DISTANCE_THRESHOLD = 0.35

DEFAULT_CONTEXT_WINDOW = 5


class Retriever:

    def __init__(self, document_id):

        self.embedding_model = get_embedding_model()

        print("Connecting to ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=DATABASE_PATH
        )

        collection_name = f"document_{document_id}"

        self.collection = self.client.get_collection(
            name=collection_name
        )

        print("Database records:", self.collection.count())

    def expand_context(
        self,
        retrieved_chunks,
        context_window=DEFAULT_CONTEXT_WINDOW
    ):

        """
        Expand context around the best-matching retrieved chunk.

        Original retrieved chunks are preserved.
        Additional neighboring chunks are marked as context.
        """

        if not retrieved_chunks:
            return []

        # Preserve all original retrieved chunks.
        expanded_chunks = {}

        for chunk in retrieved_chunks:

            chunk_id = int(chunk["chunk_id"])

            expanded_chunks[chunk_id] = {
                **chunk,
                "is_context": False
            }

        # Select the best-matching chunk.
        # Lower cosine distance means greater similarity.
        anchor_chunk = min(
            retrieved_chunks,
            key=lambda chunk: chunk["distance"]
        )

        anchor_id = int(anchor_chunk["chunk_id"])

        print("\n========== CONTEXT EXPANSION ==========")
        print("Anchor chunk:", anchor_id)
        print("Anchor page:", anchor_chunk["page_number"])
        print("Context window:", context_window)

        # Calculate neighboring chunk IDs.
        start_id = max(1, anchor_id - context_window)

        end_id = min(
            self.collection.count(),
            anchor_id + context_window
        )

        neighbor_ids = [
            str(chunk_id)
            for chunk_id in range(start_id, end_id + 1)
        ]

        # Fetch neighboring chunks from ChromaDB.
        results = self.collection.get(
            ids=neighbor_ids,
            include=["documents", "metadatas"]
        )

        # Add neighboring chunks without replacing original matches.
        for i, document in enumerate(results["documents"]):

            metadata = results["metadatas"][i]

            chunk_id = int(metadata["chunk_id"])

            if chunk_id not in expanded_chunks:

                expanded_chunks[chunk_id] = {
                    "text": document,
                    "page_number": metadata["page_number"],
                    "chunk_id": chunk_id,
                    "distance": None,
                    "is_context": True
                }

        # Sort chunks by their original document order.
        final_chunks = sorted(
            expanded_chunks.values(),
            key=lambda chunk: chunk["chunk_id"]
        )

        original_count = len(retrieved_chunks)
        expanded_count = len(final_chunks)

        print("Original chunks:", original_count)
        print("Expanded chunks:", expanded_count)
        print("Additional chunks:", expanded_count - original_count)
        print("=======================================\n")

        return final_chunks

    def retrieve(
        self,
        question,
        top_k=5,
        distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
        adaptive=True,
        expand_context=True,
        context_window=DEFAULT_CONTEXT_WINDOW
    ):

        if not question.strip():
            return []

        total_records = self.collection.count()

        if total_records == 0:
            return []

        query_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True
        )

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, total_records),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        candidates = []

        for i, document in enumerate(results["documents"][0]):

            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            candidates.append({
                "text": document,
                "page_number": metadata["page_number"],
                "chunk_id": metadata["chunk_id"],
                "distance": distance,
                # Keep vector matches distinguishable even when context
                # expansion is disabled for retrieval diagnostics.
                "is_context": False,
            })

        # First attempt: strict threshold.
        retrieved_chunks = [
            chunk
            for chunk in candidates
            if chunk["distance"] <= distance_threshold
        ]

        print("\n========== RETRIEVAL FILTER ==========")
        print("Candidate chunks:", len(candidates))
        print("Strict threshold:", distance_threshold)
        print("Strictly accepted:", len(retrieved_chunks))

        # Second attempt: relaxed threshold.
        if not retrieved_chunks and adaptive:

            print("\nNo chunks passed strict filtering.")
            print("Activating adaptive retrieval...")

            retrieved_chunks = [
                chunk
                for chunk in candidates
                if chunk["distance"] <= FALLBACK_DISTANCE_THRESHOLD
            ]

            print(
                "Fallback threshold:",
                FALLBACK_DISTANCE_THRESHOLD
            )

            print(
                "Chunks accepted after fallback:",
                len(retrieved_chunks)
            )

        print("Final retrieved chunks:", len(retrieved_chunks))
        print("======================================\n")

        if not retrieved_chunks:
            return []

        # Expand context around the best-matching chunk.
        if expand_context:

            retrieved_chunks = self.expand_context(
                retrieved_chunks,
                context_window=context_window
            )

        return retrieved_chunks
