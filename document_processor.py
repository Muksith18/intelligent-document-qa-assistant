
import re
from pypdf import PdfReader


class DocumentProcessor:

    # ----------------------------------
    # 1. CLEAN EXTRACTED TEXT
    # ----------------------------------

    def clean_text(self, text):

        text = text.replace("\xa0", " ")

        # Remove unnecessary spaces and tabs
        text = re.sub(r"[^\S\n]+", " ", text)

        # Remove spaces around line breaks
        text = re.sub(r" *\n *", "\n", text)

        # Replace excessive line breaks
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


    # ----------------------------------
    # 2. CREATE TEXT CHUNKS
    # ----------------------------------

    def create_chunks(self, text, chunk_size=1000, overlap=150):

        chunks = []

        start = 0

        while start < len(text):

            end = min(start + chunk_size, len(text))

            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            if end == len(text):
                break

            start = end - overlap

        return chunks


    # ----------------------------------
    # 3. PROCESS PDF
    # ----------------------------------

    def process_pdf(self, pdf_path):

        # Support both file paths and uploaded file objects
        if hasattr(pdf_path, "seek"):
            pdf_path.seek(0)

        reader = PdfReader(pdf_path)

        document_chunks = []

        chunk_id = 1

        empty_pages = []

        # Extract text page by page
        for page_number, page in enumerate(reader.pages, start=1):

            extracted_text = page.extract_text() or ""

            cleaned_text = self.clean_text(extracted_text)

            # Track pages without extractable text
            if not cleaned_text:

                empty_pages.append(page_number)

                continue

            # Split extracted text into chunks
            page_chunks = self.create_chunks(cleaned_text)

            for chunk in page_chunks:

                document_chunks.append({

                    "chunk_id": chunk_id,

                    "page_number": page_number,

                    "text": chunk

                })

                chunk_id += 1

        # Return processing results
        return {

            "chunks": document_chunks,

            "total_pages": len(reader.pages),

            "total_chunks": len(document_chunks),

            "empty_pages": empty_pages

        }
