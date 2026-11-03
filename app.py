
import uuid
import streamlit as st

from ingestion import DocumentIngestion
from rag_engine import RAGEngine


# ----------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Intelligent Document Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Intelligent Document Q&A Assistant")

st.write(
    "Upload a PDF, ask questions, and receive "
    "AI-generated answers grounded in your document."
)

st.divider()


# ----------------------------------
# 2. INITIALIZE SHARED RESOURCES
# ----------------------------------

@st.cache_resource
def get_ingestion_engine():
    return DocumentIngestion()


@st.cache_resource
def get_rag_engine(document_id):
    return RAGEngine(document_id)


# ----------------------------------
# 3. SESSION STATE
# ----------------------------------

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------------
# HELPER: DISPLAY SOURCE DISTANCE
# ----------------------------------

def display_source(source):

    distance = source.get("distance")

    if distance is None:
        distance_display = "Context only"
    else:
        distance_display = f"{distance:.4f}"

    st.write(
        f"Page: {source['page_number']} | "
        f"Chunk: {source['chunk_id']} | "
        f"Distance: {distance_display}"
    )


# ----------------------------------
# 4. PDF UPLOAD SECTION
# ----------------------------------

st.header("Upload your document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:

    st.write("Selected file:", uploaded_file.name)

    if st.button("Process PDF", type="primary"):

        # Generate a unique ID for this document
        document_id = uuid.uuid4().hex

        try:
            with st.spinner(
                "Processing PDF and generating embeddings..."
            ):

                ingestion = get_ingestion_engine()

                result = ingestion.ingest_pdf(
                    uploaded_file,
                    document_id
                )

            st.session_state.document_id = document_id
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.messages = []

            print(
                "Active document ID:",
                st.session_state.document_id
            )

            st.success("PDF processed successfully!")

            st.write("**Document details**")
            st.write("Filename:", uploaded_file.name)
            st.write("Total pages:", result["total_pages"])
            st.write("Total chunks:", result["total_chunks"])
            st.write("Stored vectors:", result["total_records"])
            st.write(
                "Pages without extractable text:",
                result["empty_pages"]
            )

        except Exception as error:
            st.error(f"PDF processing failed: {error}")


# ----------------------------------
# 5. DOCUMENT STATUS
# ----------------------------------

if st.session_state.document_id:

    st.divider()

    st.success(
        f"Active document: {st.session_state.uploaded_filename}"
    )

    # Initialize the RAG engine for this document
    try:
        rag = get_rag_engine(
            st.session_state.document_id
        )

    except Exception as error:
        st.error(f"Unable to initialize RAG engine: {error}")
        st.stop()


    # ----------------------------------
    # 6. DISPLAY CHAT HISTORY
    # ----------------------------------

    st.header("Ask your document")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and "sources" in message
            ):

                with st.expander("View Sources"):

                    for source in message["sources"]:

                        display_source(source)


    # ----------------------------------
    # 7. QUESTION INPUT
    # ----------------------------------

    question = st.chat_input(
        "Ask a question about your uploaded PDF..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching your document and generating an answer..."
            ):

                result = rag.ask_question(question)

                answer = result["answer"]
                sources = result["sources"]

            st.markdown(answer)

            with st.expander("View Sources"):

                if sources:

                    for source in sources:

                        display_source(source)

                else:
                    st.write("No sources available.")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


# ----------------------------------
# 8. SIDEBAR
# ----------------------------------

with st.sidebar:

    st.header("About")

    st.write(
        "This application uses Retrieval-Augmented Generation "
        "(RAG) to answer questions from uploaded PDF documents."
    )

    st.divider()

    st.subheader("Current Document")

    if st.session_state.document_id:

        st.write(st.session_state.uploaded_filename)

    else:

        st.write("No document uploaded yet.")

    if st.button("Clear Chat", use_container_width=True):

        st.session_state.messages = []
        st.rerun()
