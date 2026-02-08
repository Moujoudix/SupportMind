"""
SupportMind Configuration
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DocType(Enum):
    """Document types in the unified index."""
    KB = "KB"
    SCRIPT = "SCRIPT"
    TICKET = "TICKET"
    CONVERSATION = "CONVERSATION"


class GapType(Enum):
    """Types of knowledge gaps."""
    MISSING_KB = "missing_kb"
    OUTDATED_KB = "outdated_kb"
    INCOMPLETE_KB = "incomplete_kb"
    NONE = "none"


class ReviewStatus(Enum):
    """Review workflow status."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class QAAction(Enum):
    """QA decision actions."""
    ALLOW = "allow"
    REVISE = "revise"
    ESCALATE = "escalate"
    BLOCK = "block"


class Paths:
    """Data and artifact paths."""

    def __init__(self, base_path: str = None, artifacts_path: str = None):
        self.base_path = Path(base_path or os.getenv("DATA_PATH", "./data"))
        self.artifacts_path = Path(artifacts_path or os.getenv("ARTIFACTS_PATH", "./artifacts"))

        # CSV file paths
        self.conversations = self.base_path / "Conversations.csv"
        self.existing_kb = self.base_path / "Existing_Knowledge_Articles.csv"
        self.kb_lineage = self.base_path / "KB_Lineage.csv"
        self.knowledge_articles = self.base_path / "Knowledge_Articles.csv"
        self.learning_events = self.base_path / "Learning_Events.csv"
        self.placeholder_dict = self.base_path / "Placeholder_Dictionary.csv"
        self.qa_eval_prompt = self.base_path / "QA_Evaluation_Prompt.csv"
        self.questions = self.base_path / "Questions.csv"
        self.readme = self.base_path / "README.csv"
        self.scripts_master = self.base_path / "Scripts_Master.csv"
        self.tickets = self.base_path / "Tickets.csv"

        # Artifact directories
        self.index_dir = self.artifacts_path / "index"
        self.kb_versions_dir = self.artifacts_path / "kb_versions"
        self.qa_reports_dir = self.artifacts_path / "qa_reports"
        self.logs_dir = self.artifacts_path / "logs"

    def ensure_dirs(self):
        """Create all necessary directories."""
        for dir_path in [
            self.artifacts_path,
            self.index_dir,
            self.kb_versions_dir,
            self.qa_reports_dir,
            self.logs_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_kaggle(cls) -> "Paths":
        """Create paths for Kaggle environment."""
        base = "/kaggle/input/support-mind"
        prefix = "SupportMind__Final_Data.xlsx - "

        paths = cls(base_path=base)
        paths.conversations = Path(f"{base}/{prefix}Conversations.csv")
        paths.existing_kb = Path(f"{base}/{prefix}Existing_Knowledge_Articles.csv")
        paths.kb_lineage = Path(f"{base}/{prefix}KB_Lineage.csv")
        paths.knowledge_articles = Path(f"{base}/{prefix}Knowledge_Articles.csv")
        paths.learning_events = Path(f"{base}/{prefix}Learning_Events.csv")
        paths.placeholder_dict = Path(f"{base}/{prefix}Placeholder_Dictionary.csv")
        paths.qa_eval_prompt = Path(f"{base}/{prefix}QA_Evaluation_Prompt.csv")
        paths.questions = Path(f"{base}/{prefix}Questions.csv")
        paths.readme = Path(f"{base}/{prefix}README.csv")
        paths.scripts_master = Path(f"{base}/{prefix}Scripts_Master.csv")
        paths.tickets = Path(f"{base}/{prefix}Tickets.csv")

        return paths


@dataclass
class Config:
    """Main configuration class."""

    # Device
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "cpu"))
    use_fp16: bool = True

    # Embedding
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embedding_dim: int = 384
    embedding_batch_size: int = 64

    # FAISS
    use_gpu_faiss: bool = False

    # Retrieval
    top_k: int = 5
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    confidence_threshold: float = 0.4

    # Learning
    gap_confidence_threshold: float = 0.5
    score_margin_threshold: float = 0.15
    require_human_review: bool = True

    # QA
    auto_zero_threshold: float = 0.3

    # Database
    db_path: str = field(
        default_factory=lambda: os.getenv("DATABASE_PATH", "./supportmind.db")
    )

    # LLM
    use_mock_llm: bool = True
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    )

    # Paths
    paths: Paths = field(default_factory=Paths)

    def __post_init__(self):
        """Post-initialization setup."""
        # Detect GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                self.use_gpu_faiss = True
                self.embedding_batch_size = 128
        except ImportError:
            pass

        # Ensure directories exist
        self.paths.ensure_dirs()

    @classmethod
    def for_kaggle(cls) -> "Config":
        """Create configuration for Kaggle environment."""
        config = cls()
        config.paths = Paths.for_kaggle()
        config.paths.ensure_dirs()
        return config


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global configuration."""
    global _config
    if _config is None:
        # Auto-detect environment
        if os.path.exists("/kaggle/input"):
            _config = Config.for_kaggle()
        else:
            _config = Config()
    return _config


def set_config(config: Config):
    """Set global configuration."""
    global _config
    _config = config
