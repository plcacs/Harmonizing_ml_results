from __future__ import annotations
import difflib
import logging
import pprint
import re
from typing import Any
from kedro.io.catalog_config_resolver import CREDENTIALS_KEY, CatalogConfigResolver, Patterns
from kedro.io.core import AbstractDataset, AbstractVersionedDataset, DatasetAlreadyExistsError, DatasetError, DatasetNotFoundError, Version, _validate_versions, generate_timestamp
from kedro.io.memory_dataset import MemoryDataset
from kedro.utils import _format_rich, _has_rich_handler

CATALOG_KEY: str = 'catalog'
WORDS_REGEX_PATTERN: re.Pattern = re.compile('\\W+')

def _sub_nonword_chars(dataset_name: str) -> str:
    return re.sub(WORDS_REGEX_PATTERN, '__', dataset_name)

class _FrozenDatasets:
    def __init__(self, *datasets_collections: Any) -> None:
        self._original_names: dict = {}
    
    def __setattr__(self, key: str, value: Any) -> None:
    
    def _ipython_key_completions_(self) -> list:
    
    def __getitem__(self, key: str) -> Any:
    
    def __repr__(self) -> str:

class DataCatalog:
    def __init__(self, datasets: dict = None, feed_dict: dict = None, dataset_patterns: dict = None, load_versions: dict = None, save_version: str = None, default_pattern: dict = None, config_resolver: CatalogConfigResolver = None) -> None:
    
    def __repr__(self) -> str:
    
    def __contains__(self, dataset_name: str) -> bool:
    
    def __eq__(self, other: DataCatalog) -> bool:
    
    @property
    def config_resolver(self) -> CatalogConfigResolver:
    
    @property
    def _logger(self) -> logging.Logger:

    @classmethod
    def from_config(cls, catalog: dict, credentials: dict = None, load_versions: dict = None, save_version: str = None) -> DataCatalog:
    
    def _get_dataset(self, dataset_name: str, version: Version = None, suggest: bool = True) -> AbstractDataset:
    
    def load(self, name: str, version: Version = None) -> Any:
    
    def save(self, name: str, data: Any) -> None:
    
    def exists(self, name: str) -> bool:
    
    def release(self, name: str) -> None:
    
    def add(self, dataset_name: str, dataset: AbstractDataset, replace: bool = False) -> None:
    
    def add_all(self, datasets: dict, replace: bool = False) -> None:
    
    def add_feed_dict(self, feed_dict: dict, replace: bool = False) -> None:
    
    def list(self, regex_search: str = None) -> list:
    
    def shallow_copy(self, extra_dataset_patterns: dict = None) -> DataCatalog:
    
    def confirm(self, name: str) -> None:
