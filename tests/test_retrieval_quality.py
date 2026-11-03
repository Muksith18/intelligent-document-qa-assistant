import sys
import types
import unittest


chromadb_module = types.ModuleType("chromadb")
sys.modules.setdefault("chromadb", chromadb_module)

embedding_model_module = types.ModuleType("embedding_model")
embedding_model_module.get_embedding_model = lambda: None
sys.modules.setdefault("embedding_model", embedding_model_module)

from retrieval import Retriever


class FakeEmbeddingModel:
    def encode(self, question, normalize_embeddings):
        return types.SimpleNamespace(tolist=lambda: [0.1, 0.2])


class FakeCollection:
    def __init__(self, candidates, neighbors, count=12):
        self.candidates = candidates
        self.neighbors = neighbors
        self._count = count

    def count(self):
        return self._count

    def query(self, **kwargs):
        return self.candidates

    def get(self, ids, include):
        selected = [self.neighbors[item] for item in ids if item in self.neighbors]
        return {
            "documents": [item["text"] for item in selected],
            "metadatas": [
                {"page_number": item["page_number"], "chunk_id": item["chunk_id"]}
                for item in selected
            ],
        }


def candidate_response(items):
    return {
        "documents": [[item["text"] for item in items]],
        "metadatas": [[
            {"page_number": item["page_number"], "chunk_id": item["chunk_id"]}
            for item in items
        ]],
        "distances": [[item["distance"] for item in items]],
    }


class RetrievalQualityTests(unittest.TestCase):
    def build_retriever(self, candidates, neighbors=None, count=12):
        retriever = Retriever.__new__(Retriever)
        retriever.embedding_model = FakeEmbeddingModel()
        retriever.collection = FakeCollection(
            candidate_response(candidates), neighbors or {}, count=count
        )
        return retriever

    def test_strict_matches_are_explicitly_marked_as_original_hits(self):
        retriever = self.build_retriever([
            {"text": "strict", "page_number": 3, "chunk_id": 3, "distance": 0.20},
            {"text": "rejected", "page_number": 4, "chunk_id": 4, "distance": 0.36},
        ])

        chunks = retriever.retrieve("TCP", expand_context=False)

        self.assertEqual([chunk["chunk_id"] for chunk in chunks], [3])
        self.assertEqual(chunks[0]["distance"], 0.20)
        self.assertFalse(chunks[0]["is_context"])

    def test_adaptive_fallback_keeps_original_hit_provenance(self):
        retriever = self.build_retriever([
            {"text": "fallback", "page_number": 5, "chunk_id": 5, "distance": 0.33},
            {"text": "rejected", "page_number": 6, "chunk_id": 6, "distance": 0.36},
        ])

        chunks = retriever.retrieve("TCP", expand_context=False)

        self.assertEqual([chunk["chunk_id"] for chunk in chunks], [5])
        self.assertFalse(chunks[0]["is_context"])

    def test_context_expansion_deduplicates_anchor_and_marks_neighbors(self):
        candidates = [
            {"text": "anchor", "page_number": 3, "chunk_id": 3, "distance": 0.20},
            {"text": "second match", "page_number": 5, "chunk_id": 5, "distance": 0.25},
        ]
        neighbors = {
            "1": {"text": "one", "page_number": 1, "chunk_id": 1},
            "2": {"text": "two", "page_number": 2, "chunk_id": 2},
            "3": {"text": "anchor", "page_number": 3, "chunk_id": 3},
            "4": {"text": "four", "page_number": 4, "chunk_id": 4},
            "5": {"text": "second match", "page_number": 5, "chunk_id": 5},
        }
        retriever = self.build_retriever(candidates, neighbors, count=5)

        chunks = retriever.retrieve("TCP", expand_context=True, context_window=2)

        self.assertEqual([chunk["chunk_id"] for chunk in chunks], [1, 2, 3, 4, 5])
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), len(chunks))
        self.assertFalse(next(chunk for chunk in chunks if chunk["chunk_id"] == 3)["is_context"])
        self.assertFalse(next(chunk for chunk in chunks if chunk["chunk_id"] == 5)["is_context"])
        neighbor = next(chunk for chunk in chunks if chunk["chunk_id"] == 4)
        self.assertTrue(neighbor["is_context"])
        self.assertIsNone(neighbor["distance"])

    def test_context_expansion_respects_document_start_boundary(self):
        candidates = [{"text": "first", "page_number": 1, "chunk_id": 1, "distance": 0.20}]
        neighbors = {
            "1": {"text": "first", "page_number": 1, "chunk_id": 1},
            "2": {"text": "second", "page_number": 1, "chunk_id": 2},
            "3": {"text": "third", "page_number": 2, "chunk_id": 3},
        }
        retriever = self.build_retriever(candidates, neighbors, count=3)

        chunks = retriever.retrieve("TCP", expand_context=True, context_window=5)

        self.assertEqual([chunk["chunk_id"] for chunk in chunks], [1, 2, 3])
        self.assertFalse(chunks[0]["is_context"])
        self.assertTrue(all(chunk["distance"] is None for chunk in chunks[1:]))


if __name__ == "__main__":
    unittest.main()
