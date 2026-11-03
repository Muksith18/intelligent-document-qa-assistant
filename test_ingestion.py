"""Legacy manual ingestion fixture, excluded from Phase 6 evaluation tooling.

It retains its historical fixture ID intentionally: running it creates or
replaces that fixture collection, so it must never be used as an automated
test or as a Phase 6 evaluation command.
"""


LEGACY_DIAGNOSTIC = True
LEGACY_DOCUMENT_ID = "networking_test_001"
LEGACY_PDF_PATH = "CN_RefereneceBook.pdf"


def main():
    from ingestion import DocumentIngestion

    ingestion = DocumentIngestion()
    result = ingestion.ingest_pdf(
        pdf_path=LEGACY_PDF_PATH,
        document_id=LEGACY_DOCUMENT_ID,
    )
    print("\n========== INGESTION REPORT ==========")
    print("Document ID:", result["document_id"])
    print("Total pages:", result["total_pages"])
    print("Total chunks:", result["total_chunks"])
    print("Stored records:", result["total_records"])
    print("Empty pages:", result["empty_pages"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
