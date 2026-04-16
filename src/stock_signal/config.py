import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "stock_signal.duckdb"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

FMP_API_KEY: str = os.environ.get("FMP_API_KEY", "")
FMP_BASE_URL: str = "https://financialmodelingprep.com/api/v3"
FMP_RATE_LIMIT_PER_MIN: int = int(os.getenv("FMP_RATE_LIMIT", "250"))
FMP_MAX_CONCURRENT: int = int(os.getenv("FMP_MAX_CONCURRENT", "5"))

STALENESS_DAYS: int = 7
