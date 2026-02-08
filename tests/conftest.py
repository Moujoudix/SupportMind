"""
Pytest configuration and fixtures.
"""

import pytest
import tempfile
import os

from supportmind.config.settings import Config, set_config, Paths
from supportmind.stores.database import Database
from supportmind.stores.vector_store import VectorStore


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="session")
def test_config(temp_dir):
    """Create test configuration."""
    config = Config()
    config.db_path = os.path.join(temp_dir, "test.db")
    config.paths = Paths(
        base_path=temp_dir,
        artifacts_path=os.path.join(temp_dir, "artifacts")
    )
    config.paths.ensure_dirs()
    set_config(config)
    return config


@pytest.fixture
def db(test_config):
    """Create test database."""
    # Clear singleton for fresh database
    Database._instances.clear()
    return Database(test_config.db_path)


@pytest.fixture
def vector_store(test_config):
    """Create test vector store."""
    return VectorStore(dimension=test_config.embedding_dim)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    from supportmind.models.schemas import Document

    return [
        Document(
            doc_id="KB_001",
            doc_type="KB",
            source_id="001",
            title="Password Reset Guide",
            content="To reset your password, go to Settings > Security > Reset Password.",
            metadata={"category": "Security", "product": "Main App"}
        ),
        Document(
            doc_id="KB_002",
            doc_type="KB",
            source_id="002",
            title="Account Setup",
            content="Create a new account by clicking Sign Up and filling in your details.",
            metadata={"category": "Getting Started", "product": "Main App"}
        ),
        Document(
            doc_id="SCRIPT_001",
            doc_type="SCRIPT",
            source_id="S001",
            title="Troubleshoot Login Issues",
            content="Step 1: Check credentials. Step 2: Clear browser cache.",
            metadata={"category": "Troubleshooting"}
        ),
    ]
