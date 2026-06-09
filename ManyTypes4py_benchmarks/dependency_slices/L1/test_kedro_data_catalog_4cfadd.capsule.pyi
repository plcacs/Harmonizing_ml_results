from typing import Any

# === Internal dependency: kedro ===
io: Any

# === Internal dependency: kedro.io ===
from .core import DatasetAlreadyExistsError
from .core import DatasetError
from .core import DatasetNotFoundError
from .kedro_data_catalog import KedroDataCatalog
from .lambda_dataset import LambdaDataset
from .memory_dataset import MemoryDataset

# === Internal dependency: kedro.io.core ===
def generate_timestamp(): ...
def parse_dataset_definition(config, load_version=..., save_version=...): ...
VERSION_FORMAT = '%Y-%m-%dT%H.%M.%S.%fZ'
_DEFAULT_PACKAGES = ['kedro.io.', 'kedro_datasets.', '']

# === Third-party dependency: kedro_datasets.pandas ===
# Used symbols: CSVDataset, ParquetDataset

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Third-party dependency: pandas.testing ===
# Used symbols: assert_frame_equal

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises