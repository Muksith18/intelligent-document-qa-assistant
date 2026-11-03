
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embedding_model = None


def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        print("Loading shared embedding model...")

        _embedding_model = SentenceTransformer(MODEL_NAME)

    return _embedding_model
