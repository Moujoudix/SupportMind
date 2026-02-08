#!/usr/bin/env python3
"""
Data Ingestion Script.
Loads CSV data and populates the database and vector store.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from supportmind.config.settings import get_config, set_config, Config, Paths
from supportmind.ingest.loader import DataIngester
from supportmind.stores.database import Database
from supportmind.stores.vector_store import VectorStore


def main(data_path: str = None):
    """
    Run data ingestion.

    Args:
        data_path: Optional path to data directory
    """
    print("=" * 60)
    print("SupportMind Data Ingestion")
    print("=" * 60)

    # Configure paths if provided
    if data_path:
        config = Config()
        config.paths = Paths(base_path=data_path)
        set_config(config)

    config = get_config()

    print(f"\nConfiguration:")
    print(f"  Data path: {config.paths.base_path}")
    print(f"  Database: {config.db_path}")
    print(f"  Device: {config.device}")

    # Check if data files exist
    print(f"\nChecking data files...")
    files_exist = True
    for name in ['knowledge_articles', 'tickets', 'conversations', 'scripts_master']:
        path = getattr(config.paths, name, None)
        if path and Path(path).exists():
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} not found at {path}")
            files_exist = False

    if not files_exist:
        print("\n⚠️ Some data files are missing. Ingestion may be incomplete.")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Initialize components
    print(f"\nInitializing components...")
    db = Database()
    vs = VectorStore()

    # Run ingestion
    ingester = DataIngester(db=db, vector_store=vs, paths=config.paths)
    stats = ingester.run_full_ingestion()

    # Print summary
    print("\n" + "=" * 60)
    print("Ingestion Summary")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ Ingestion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data into SupportMind")
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to data directory containing CSV files"
    )

    args = parser.parse_args()
    main(args.data_path)
