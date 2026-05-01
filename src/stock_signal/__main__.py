"""CLI entry point: uv run python -m stock_signal"""

import argparse
import asyncio
import logging
from pathlib import Path

from .config import Config, Scope
from .pipeline import run_us_pipeline

_DEFAULT_CONFIG = Path(__file__).parent / "config.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock Signal Engine")
    parser.add_argument(
        "--symbols", nargs="+",
        help="Override universe with specific symbols (e.g. --symbols AAPL MSFT)",
    )
    parser.add_argument(
        "--scope", nargs="+",
        default=[Scope.SP500.value],
        choices=[s.value for s in Scope],
        help="Universe scope (default: sp500)",
    )
    parser.add_argument(
        "--config", default=str(_DEFAULT_CONFIG),
        help="Path to config.toml (default: bundled config)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config.new(args.config)
    out = asyncio.run(run_us_pipeline(
        config=config,
        scope=args.scope,
        symbols=args.symbols,
    ))
    print(out)


if __name__ == "__main__":
    main()
