#!/usr/bin/env python3
"""
Generate SupportMind repository structure.
Run this script to create all necessary files.
"""

import os
from pathlib import Path

# Define file contents
FILES = {
    # Vector Store
    "supportmind/stores/vector_store.py": '''"""
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
''',

    # CLI
    "supportmind/cli.py": '''"""
Command-line interface for SupportMind.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="SupportMind: Self-Learning AI Support Intelligence"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data")
    ingest_parser.add_argument("--data-path", help="Path to data directory")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo")

    # Server command
    server_parser = subparsers.add_parser("serve", help="Start API server")
    server_parser.add_argument("--host", default="0.0.0.0")
    server_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "ingest":
        from scripts.ingest_data import main as ingest_main
        ingest_main(args.data_path)
    elif args.command == "query":
        from supportmind import RAGGenerator
        rag = RAGGenerator()
        response = rag.generate(args.question)
        print(f"\\nAnswer: {response.answer}")
        print(f"\\nSources: {response.get_source_citations()}")
    elif args.command == "demo":
        from scripts.demo import main as demo_main
        demo_main()
    elif args.command == "serve":
        import uvicorn
        uvicorn.run(
            "supportmind.api.endpoints:app",
            host=args.host,
            port=args.port,
            reload=True
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
''',

    # README
    "README.md": '''# SupportMind

**Self-Learning AI Support Intelligence System**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

SupportMind is an intelligent support system that:
- 🔍 **Retrieves** relevant knowledge using hybrid search (semantic + keyword)
- 💬 **Generates** accurate responses with RAG (Retrieval Augmented Generation)
- ✅ **Evaluates** response quality with QA scoring
- 📚 **Learns** by automatically creating KB articles from resolved tickets
- 🔗 **Tracks** knowledge lineage and provenance

## Features

- **Unified Retrieval**: Combines FAISS semantic search with SQLite FTS5
- **Evidence-Based Classification**: Determines answer type from retrieved documents
- **QA & Compliance**: Automated quality scoring with compliance checks
- **Self-Learning Loop**: Gap detection → KB generation → Review → Publish
- **Full Traceability**: Every response includes source citations and trace IDs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/supportmind.git
cd supportmind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
Quick Start
python
from supportmind import RAGGenerator, Database, VectorStore

# Initialize components
db = Database()
vs = VectorStore()
rag = RAGGenerator()

# Ask a question
response = rag.generate("How do I reset a user\'s password?")
print(response.answer)
print(response.get_source_citations())
CLI Usage
bash
# Ingest data
supportmind ingest --data-path ./data

# Query the system
supportmind query "How do I reset a password?"

# Run demo
supportmind demo

# Start API server
supportmind serve --port 8000
Project Structure
text
supportmind/
├── supportmind/          # Main package
│   ├── config/           # Configuration
│   ├── models/           # Data models
│   ├── stores/           # Database & vector store
│   ├── pipelines/        # RAG, QA, Learning
│   ├── analytics/        # Metrics
│   └── api/              # REST API
├── app/                  # Streamlit dashboard
├── scripts/              # Utility scripts
├── tests/                # Test suite
└── docs/                 # Documentation
Architecture
text
Query → Unified Retrieval → Evidence-Based Type Detection → RAG Generation → QA Scoring
                                      ↓
                              Gap Detection → KB Draft → Review → Publish → Index Update
License
MIT License - see LICENSE for details.
''',
}

def create_file(path: str, content: str):
    """Create a file with content."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content)
    print(f" ✅ Created: {path}")

def main():
    """Generate all repository files."""
    print("🚀 Generating SupportMind repository structure...\n")

    # Create directories
    dirs = [
        "supportmind/config",
        "supportmind/models",
        "supportmind/stores",
        "supportmind/ingest",
        "supportmind/pipelines",
        "supportmind/analytics",
        "supportmind/api",
        "supportmind/utils",
        "app",
        "scripts",
        "tests",
        "data",
        "artifacts/index",
        "artifacts/kb_versions",
        "artifacts/qa_reports",
        "notebooks",
        "docs",
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        # Create __init__.py for Python packages
        if d.startswith("supportmind") or d in ["app", "tests"]:
            init_file = Path(d) / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""{}"""'.format(d.replace("/", ".")))

    print("📁 Created directories\n")

    # Create files
    print("📄 Creating files...")
    for path, content in FILES.items():
        create_file(path, content.strip())

    # Create .gitkeep files
    for d in ["data", "artifacts/index", "artifacts/kb_versions", "artifacts/qa_reports"]:
        gitkeep = Path(d) / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("")

    print("\n✅ Repository structure generated!")
    print("\nNext steps:")
    print("  1. Copy your notebook code into the appropriate modules")
    print("  2. Run: pip install -e .")
    print("  3. Run: supportmind demo")

if __name__ == "__main__":
    main()
