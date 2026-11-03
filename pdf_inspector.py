
import re
from pypdf import PdfReader

def create_chunks(text, chunk_size=1000, overlap=150):

    chunks = []

    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end]

        chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks

sample_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 5

test_chunks = create_chunks(
    sample_text,
    chunk_size=50,
    overlap=10
)

print("Total chunks:", len(test_chunks))

for i, chunk in enumerate(test_chunks):

    print(f"\nChunk {i + 1}:")
    print(chunk)
    print("Length:", len(chunk))

def clean_text(text):

    # Normalize unusual Unicode whitespace
    text = text.replace("\xa0", " ")

    # Replace whitespace other than line breaks
    # with a single ordinary space
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove spaces around line breaks
    text = re.sub(r" *\n *", "\n", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# Load PDF
reader = PdfReader("CN_RefereneceBook.pdf")

all_pages = []
empty_pages = []

# Extract text from every page
for page_number, page in enumerate(reader.pages):

    text = page.extract_text()

    if text and text.strip():

        page_data = {
            "page_number": page_number + 1,
            "text": text
        }

        all_pages.append(page_data)

    else:
        empty_pages.append(page_number + 1)


# Clean extracted text
cleaned_pages = []

for page_data in all_pages:

    cleaned = clean_text(page_data["text"])

    cleaned_page = {
        "page_number": page_data["page_number"],
        "text": cleaned
    }

    cleaned_pages.append(cleaned_page)


# Display results
print("Total PDF pages:", len(reader.pages))
print("Pages with extracted text:", len(all_pages))
print("Pages without text:", len(empty_pages))
print("Total cleaned pages:", len(cleaned_pages))

print("\nOriginal text:")
print(all_pages[0]["text"][:500])

print("\nCleaned text:")
print(cleaned_pages[0]["text"][:500])


# Create chunks from all cleaned pages

document_chunks = []

chunk_id = 1

for page_data in cleaned_pages:

    text = page_data["text"]

    page_chunks = create_chunks(
        text,
        chunk_size=1000,
        overlap=150
    )

    for chunk in page_chunks:

        chunk_data = {
            "chunk_id": chunk_id,
            "page_number": page_data["page_number"],
            "text": chunk
        }

        document_chunks.append(chunk_data)

        chunk_id += 1


# Display results

print("\nCHUNKING RESULTS")

print("Total cleaned pages:", len(cleaned_pages))
print("Total chunks generated:", len(document_chunks))

print("\nFirst chunk:")

print("Chunk ID:", document_chunks[0]["chunk_id"])
print("Page number:", document_chunks[0]["page_number"])
print("Text:", document_chunks[0]["text"])

print("\nFirst chunk length:",
      len(document_chunks[0]["text"]))


# Validate our generated chunks

chunk_lengths = [
    len(chunk["text"])
    for chunk in document_chunks
]

print("\nCHUNK VALIDATION")

print("Total chunks:", len(document_chunks))

print("Smallest chunk:", min(chunk_lengths))

print("Largest chunk:", max(chunk_lengths))

print(
    "Chunks exceeding 1000 characters:",
    sum(length > 1000 for length in chunk_lengths)
)

print(
    "Chunks with empty text:",
    sum(length == 0 for length in chunk_lengths)
)

# Inspect the first five chunks
print("\nFIRST FIVE CHUNKS")

for chunk in document_chunks[:5]:

    print(
        "ID:", chunk["chunk_id"],
        "| Page:", chunk["page_number"],
        "| Length:", len(chunk["text"])
    )

# Find chunks containing fewer than 100 characters

short_chunks = [
    chunk
    for chunk in document_chunks
    if len(chunk["text"].strip()) < 100
]

print("\nSHORT CHUNK ANALYSIS")

print("Chunks below 100 characters:", len(short_chunks))

print("\nFirst 10 short chunks:")

for chunk in short_chunks[:10]:

    print("\nChunk ID:", chunk["chunk_id"])
    print("Page:", chunk["page_number"])
    print("Length:", len(chunk["text"]))
    print("Text:", repr(chunk["text"]))

import json

# Save chunks into a JSON file
with open("chunks.json", "w", encoding="utf-8") as file:

    json.dump(
        document_chunks,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nChunks successfully saved!")

print("Total chunks saved:", len(document_chunks))

# Load the saved chunks
with open("chunks.json", "r", encoding="utf-8") as file:

    loaded_chunks = json.load(file)

print("\nJSON VERIFICATION")

print("Loaded chunks:", len(loaded_chunks))

print("First chunk ID:", loaded_chunks[0]["chunk_id"])

print("First chunk page:", loaded_chunks[0]["page_number"])
