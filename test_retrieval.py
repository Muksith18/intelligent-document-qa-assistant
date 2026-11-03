
from retrieval import Retriever

# Initialize retriever
retriever = Retriever()

# Ask a question
question = "How does TCP establish a connection?"

# Retrieve relevant passages
results = retriever.retrieve(question, top_k=5)

# Display results
print("\n========== RETRIEVED RESULTS ==========")

for result in results:

    print("\nPage:", result["page_number"])
    print("Chunk ID:", result["chunk_id"])
    print("Distance:", round(result["distance"], 4))
    print("Text:", result["text"])
    print("-" * 60)
