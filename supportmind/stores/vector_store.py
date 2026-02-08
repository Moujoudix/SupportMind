"""
FAISS Vector Store with GPU support.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from supportmind.config.settings import get_config
from supportmind.models.schemas import Document

# Try importing ML libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False


class VectorStore:
    """GPU-accelerated FAISS vector store."""

    def __init__(self, dimension: int = None):
        config = get_config()
        self.dimension = dimension or config.embedding_dim
        self.doc_ids: List[str] = []
        self.doc_metadata: Dict[str, Dict] = {}
        self.index = None
        self.index_gpu = None
        self.gpu_resources = None

        self._init_encoder()
        self._init_index()

    def _init_encoder(self):
        """Initialize embedding model."""
        config = get_config()
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.encoder = SentenceTransformer(
                config.embedding_model,
                device=config.device
            )
            if config.use_fp16 and config.device == "cuda":
                self.encoder.half()
        else:
            self.encoder = None

    def _init_index(self):
        """Initialize FAISS index."""
        config = get_config()
        if not FAISS_AVAILABLE:
            self._mock_store = {}
            return

        self.index = faiss.IndexFlatIP(self.dimension)

        if config.use_gpu_faiss and TORCH_AVAILABLE:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                self.gpu_resources.setTempMemory(256 * 1024 * 1024)
                self.index_gpu = faiss.index_cpu_to_gpu(
                    self.gpu_resources, 0, self.index
                )
            except Exception:
                pass

    @property
    def active_index(self):
        return self.index_gpu if self.index_gpu else self.index

    def encode(self, texts: Union[str, List[str]],
               show_progress: bool = False) -> np.ndarray:
        """Generate embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        if self.encoder:
            config = get_config()
            embeddings = self.encoder.encode(
                texts,
                batch_size=config.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            if embeddings.dtype == np.float16:
                embeddings = embeddings.astype(np.float32)
            return embeddings
        else:
            return np.random.randn(len(texts), self.dimension).astype("float32")

    def add_documents(self, documents: List[Document],
                      show_progress: bool = True) -> int:
        """Add documents to index."""
        if not documents:
            return 0

        texts = [doc.get_searchable_text() for doc in documents]
        embeddings = self.encode(texts, show_progress=show_progress)

        if FAISS_AVAILABLE and self.active_index:
            faiss.normalize_L2(embeddings)
            self.active_index.add(embeddings)
        else:
            if not hasattr(self, "_mock_store"):
                self._mock_store = {}
            for i, doc in enumerate(documents):
                self._mock_store[doc.doc_id] = embeddings[i]

        for doc in documents:
            self.doc_ids.append(doc.doc_id)
            self.doc_metadata[doc.doc_id] = {
                "doc_type": doc.doc_type,
                "source_id": doc.source_id,
                "title": doc.title
            }

        return len(documents)

    def search(self, query: str, top_k: int = 5,
               doc_type: str = None) -> List[Tuple[str, float]]:
        """Semantic search."""
        if not self.doc_ids:
            return []

        query_emb = self.encode(query)

        if FAISS_AVAILABLE and self.active_index:
            faiss.normalize_L2(query_emb)
            search_k = top_k * 3 if doc_type else top_k
            scores, indices = self.active_index.search(
                query_emb, min(search_k, len(self.doc_ids))
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.doc_ids):
                    continue
                doc_id = self.doc_ids[idx]

                if doc_type and self.doc_metadata.get(doc_id, {}).get("doc_type") != doc_type:
                    continue

                results.append((doc_id, float(score)))
                if len(results) >= top_k:
                    break
            return results
        else:
            filtered = self.doc_ids
            if doc_type:
                filtered = [d for d in self.doc_ids
                           if self.doc_metadata.get(d, {}).get("doc_type") == doc_type]
            scores = np.random.rand(len(filtered))
            idx = np.argsort(scores)[::-1][:top_k]
            return [(filtered[i], float(scores[i])) for i in idx]

    def count(self) -> int:
        return len(self.doc_ids)

    def counts_by_type(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for m in self.doc_metadata.values():
            counts[m.get("doc_type", "UNKNOWN")] += 1
        return dict(counts)

    def save(self, path: str = None):
        """Save index to disk."""
        config = get_config()
        path = Path(path or config.paths.index_dir)
        path.mkdir(parents=True, exist_ok=True)

        if FAISS_AVAILABLE:
            if self.index_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index_gpu)
                faiss.write_index(cpu_index, str(path / "faiss.index"))
            elif self.index:
                faiss.write_index(self.index, str(path / "faiss.index"))

        with open(path / "metadata.json", "w") as f:
            json.dump({
                "doc_ids": self.doc_ids,
                "doc_metadata": self.doc_metadata
            }, f)

    def load(self, path: str = None) -> bool:
        """Load index from disk."""
        config = get_config()
        path = Path(path or config.paths.index_dir)

        try:
            if FAISS_AVAILABLE and (path / "faiss.index").exists():
                self.index = faiss.read_index(str(path / "faiss.index"))
                if config.use_gpu_faiss and self.gpu_resources:
                    self.index_gpu = faiss.index_cpu_to_gpu(
                        self.gpu_resources, 0, self.index
                    )

            if (path / "metadata.json").exists():
                with open(path / "metadata.json", "r") as f:
                    data = json.load(f)
                    self.doc_ids = data["doc_ids"]
                    self.doc_metadata = data["doc_metadata"]
            return True
        except Exception:
            return False