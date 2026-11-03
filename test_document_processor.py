from document_processor import DocumentProcessor

processor = DocumentProcessor()

result = processor.process_pdf("CN_RefereneceBook.pdf")

print("Total pages:", result["total_pages"])
print("Total chunks:", result["total_chunks"])
print("Pages without extractable text:", result["empty_pages"])

print("\nFirst chunk:")
print(result["chunks"][0])