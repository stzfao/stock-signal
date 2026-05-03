import os
from pathlib import Path
import tomllib

from enum import Enum
from dotenv import load_dotenv
from dataclasses import field

from pydantic import ConfigDict, RootModel, TypeAdapter, field_validator
from pydantic.dataclasses import dataclass
from pydantic.alias_generators import to_camel

load_dotenv()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Scope(Enum):
    RUSSEL2000 = "russel2000"
    SP500 = "sp500"
    NASDAQ = "nasdaq"
    NYSE = "nyse"

    @classmethod
    def members(cls):
        return [m.name for m in cls]

@dataclass(frozen=True)
class Schedule:
    price_staleness_hours: int
    fundamentals_staleness_days: int

@dataclass(frozen=True)
class DataStore:
    db_path: Path
    output_dir: Path

    @field_validator("db_path", "output_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str) -> Path:
        return _PROJECT_ROOT / Path(v)

@dataclass(config=ConfigDict(alias_generator=to_camel,populate_by_name=True),frozen=True)
class NYSEBody:
    instrument_type: str
    sort_column: str
    sort_order: str
    max_results_per_page: int
    page_number: int
    filter_token: str

@dataclass(frozen=True)
class NYSE:
    base_url: str
    body: NYSEBody = field(default_factory=NYSEBody)

    @property
    def serialized_body(self) -> dict[str, str | int]:
        return RootModel(self.body).model_dump(by_alias=True)

@dataclass(frozen=True)
class FMP:
    base_url: str
    rpm: int
    max_conn: int
    api_key: str = field(default_factory=lambda: os.environ["FMP_API_KEY"], repr=False)

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("FMP_API_KEY not set")

@dataclass(frozen=True)
class NASDAQ:
    base_url: str
    table_only: bool
    limit: int

@dataclass(frozen=True)
class SlickCharts:
    base_url: str

@dataclass(frozen=True)
class StockAnalysis:
    base_url: str
    rpm: int = 30
    max_conn: int = 5

@dataclass
class Config:
    schedule: Schedule = field(default_factory=Schedule)
    datastore: DataStore = field(default_factory=DataStore)
    nasdaq: NASDAQ = field(default_factory=NASDAQ)
    fmp: FMP = field(default_factory=FMP)
    nyse: NYSE = field(default_factory=NYSE)
    slickcharts: SlickCharts = field(default_factory=SlickCharts)
    stockanalysis: StockAnalysis = field(default_factory=StockAnalysis)

    @classmethod
    def new(cls, path: str) -> "Config":
        config_path = _PROJECT_ROOT / path
        with open(str(config_path), "rb") as f:
            data = tomllib.load(f)
        adapter = TypeAdapter(cls)
        return adapter.validate_python(data)


# DB_PATH = PROJECT_ROOT / "data" / "stock_signal.duckdb"
# OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
# #
# FMP_API_KEY: str = os.environ.get("FMP_API_KEY", "")
# FMP_BASE_URL: str = "https://financialmodelingprep.com/api/v3"
# FMP_RATE_LIMIT_PER_MIN: int = int(os.getenv("FMP_RATE_LIMIT", "250"))
# FMP_MAX_CONCURRENT: int = int(os.getenv("FMP_MAX_CONCURRENT", "5"))
#
# STALENESS_DAYS: int = 7

# config = Config.new(os.environ["CONFIG_PATH"])
#
# print(config.nasdaq.base_url)
# print(config.nyse.base_url)
# print(config.nyse.serialized_params)
# print(config.nyse.params.instrument_type)