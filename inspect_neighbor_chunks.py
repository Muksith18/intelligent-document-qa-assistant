
import chromadb
from evaluation_config import CANONICAL_DOCUMENT_ID

DOCUMENT_ID = CANONICAL_DOCUMENT_ID

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name=f"document_{DOCUMENT_ID}"
)

# Page 295's retrieved chunk has ID 833.
# Inspect nearby chunks to find the missing beginning.
chunk_ids = [
    str(i) for i in range(828, 839)
]

results = collection.get(
    ids=chunk_ids,
    include=["documents", "metadatas"]
)

print("\n========== NEIGHBORING CHUNKS ==========")

for chunk_id, document, metadata in zip(
    results["ids"],
    results["documents"],
    results["metadatas"]
):

    print("\n" + "=" * 75)
    print("CHUNK ID:", chunk_id)
    print("PAGE:", metadata["page_number"])
    print("=" * 75)

    print(document)

print("\nInspection completed.")
