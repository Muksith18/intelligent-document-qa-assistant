
from embedding_model import get_embedding_model
from document_processor import DocumentProcessor
from vector_store import VectorStore


class DocumentIngestion:

    def __init__(self):

        self.embedding_model = get_embedding_model()

        self.processor = DocumentProcessor()

        self.vector_store = VectorStore()

    def ingest_pdf(self, pdf_path, document_id):

        print("\nProcessing PDF...")

        result = self.processor.process_pdf(pdf_path)

        chunks = result["chunks"]

        if not chunks:
            raise ValueError("No extractable text found in this PDF.")

        print("Total pages:", result["total_pages"])
        print("Total chunks:", result["total_chunks"])

        print("\nGenerating embeddings...")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        print("\nCreating vector collection...")

        collection = self.vector_store.create_collection(document_id)

        print("\nStoring vectors in ChromaDB...")

        total_records = self.vector_store.add_documents(
            collection,
            chunks,
            embeddings
        )

        print("\nDocument ingestion completed!")

        return {
            "document_id": document_id,
            "total_pages": result["total_pages"],
            "total_chunks": result["total_chunks"],
            "total_records": total_records,
            "empty_pages": result["empty_pages"]
        }
