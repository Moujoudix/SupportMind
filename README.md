# SupportMind

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
source venv/bin/activate  # On Windows: venv\Scripts\activate

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
response = rag.generate("How do I reset a user's password?")
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