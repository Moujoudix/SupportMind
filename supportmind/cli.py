"""
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
        print(f"\nAnswer: {response.answer}")
        print(f"\nSources: {response.get_source_citations()}")
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