"""
SupportMind: Self-Learning AI Support Intelligence System
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from supportmind.config.settings import Config, get_config
from supportmind.stores.database import Database
from supportmind.stores.vector_store import VectorStore
from supportmind.pipelines.retrieval import UnifiedRetriever
from supportmind.pipelines.rag import RAGGenerator
from supportmind.pipelines.quality import QAScorer
from supportmind.pipelines.learning import LearningPipeline
from supportmind.analytics.metrics import Analytics
from supportmind.api.copilot import AgentCopilot

__all__ = [
    "Config",
    "get_config",
    "Database",
    "VectorStore",
    "UnifiedRetriever",
    "RAGGenerator",
    "QAScorer",
    "LearningPipeline",
    "Analytics",
    "AgentCopilot",
]
