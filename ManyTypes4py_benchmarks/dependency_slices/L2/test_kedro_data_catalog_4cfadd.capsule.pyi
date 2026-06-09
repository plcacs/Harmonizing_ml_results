from typing import Any

# === Internal dependency: kedro ===
io: Any

# === Internal dependency: kedro.io ===
# re-export: from .core import DatasetAlreadyExistsError
# re-export: from .core import DatasetError
# re-export: from .core import DatasetNotFoundError
# re-export: from .kedro_data_catalog import KedroDataCatalog
# re-export: from .lambda_dataset import LambdaDataset
# re-export: from .memory_dataset import MemoryDataset

# === Internal dependency: kedro.io.core ===
def generate_timestamp() -> str: ...
def parse_dataset_definition(config: dict[str, Any], load_version: str | None = ..., save_version: str | None = ...) -> tuple[type[AbstractDataset], dict[str, Any]]: ...
VERSION_FORMAT: str
_DEFAULT_PACKAGES: Any

# === Third-party dependency: kedro_datasets.pandas ===
# Used symbols: CSVDataset, ParquetDataset

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Third-party dependency: pandas.testing ===
# Used symbols: assert_frame_equal

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises