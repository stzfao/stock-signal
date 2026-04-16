"""CLI entry point: uv run python -m stock_signal"""

import argparse
import asyncio
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock Signal Engine")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch all data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Override universe with specific symbols (e.g., --symbols AAPL MSFT GOOGL)",
    )
    parser.add_argument(
        "--market",
        default="us",
        choices=["us"],
        help="Market to run (default: us)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from stock_signal.config import FMP_API_KEY

    if not FMP_API_KEY:
        print("Error: FMP_API_KEY not set. Copy .env.example to .env and add your key.", file=sys.stderr)
        sys.exit(1)

    from stock_signal.pipeline import run_us_pipeline

    output = asyncio.run(run_us_pipeline(refresh=args.refresh, symbols=args.symbols))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
