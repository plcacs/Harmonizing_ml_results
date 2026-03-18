```python
from __future__ import annotations

import re
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Pattern, Union

from kedro.io.catalog_config_resolver import CatalogConfigResolver
from kedro.io.core import (
    AbstractDataset,
    AbstractVersionedDataset,
    DatasetAlreadyExistsError,
    DatasetError,
    DatasetNotFoundError,
    Version,
)

CATALOG_KEY: str = ...
WORDS_REGEX_PATTERN: Pattern[str] = ...

def _sub_nonword_chars(dataset_name: str) -> str: ...

class _FrozenDatasets:
    _original_names: Dict[str, str]
    
    def __init__(self, *datasets_collections: Any) -> None: ...
    def __setattr__(self, key: str, value: Any) -> None: ...
    def _ipython_key_completions_(self) -> List[str]: ...
    def __getitem__(self, key: str) -> Any: ...
    def __repr__(self) -> str: ...

class DataCatalog:
    _config_resolver: CatalogConfigResolver
    _load_versions: Dict[str, Version]
    _save_version: Optional[str]
    _datasets: Dict[str, AbstractDataset]
    datasets: Optional[_FrozenDatasets]
    _use_rich_markup: bool
    
    def __init__(
        self,
        datasets: Optional[Dict[str, AbstractDataset]] = ...,
        feed_dict: Optional[Dict[str, Any]] = ...,
        dataset_patterns: Optional[Dict[str, Any]] = ...,
        load_versions: Optional[Dict[str, str]] = ...,
        save_version: Optional[str] = ...,
        default_pattern: Optional[Dict[str, Any]] = ...,
        config_resolver: Optional[CatalogConfigResolver] = ...,
    ) -> None: ...
    
    def __repr__(self) -> str: ...
    def __contains__(self, dataset_name: str) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...
    
    @property
    def config_resolver(self) -> CatalogConfigResolver: ...
    @property
    def _logger(self) -> Any: ...
    
    @classmethod
    def from_config(
        cls,
        catalog: Optional[Dict[str, Any]],
        credentials: Optional[Dict[str, Any]] = ...,
        load_versions: Optional[Dict[str, str]] = ...,
        save_version: Optional[str] = ...,
    ) -> DataCatalog: ...
    
    def _get_dataset(
        self,
        dataset_name: str,
        version: Optional[Version] = ...,
        suggest: bool = ...,
    ) -> AbstractDataset: ...
    
    def load(self, name: str, version: Optional[str] = ...) -> Any: ...
    def save(self, name: str, data: Any) -> None: ...
    def exists(self, name: str) -> bool: ...
    def release(self, name: str) -> None: ...
    
    def add(
        self,
        dataset_name: str,
        dataset: AbstractDataset,
        replace: bool = ...,
    ) -> None: ...
    
    def add_all(
        self,
        datasets: Dict[str, AbstractDataset],
        replace: bool = ...,
    ) -> None: ...
    
    def add_feed_dict(
        self,
        feed_dict: Dict[str, Any],
        replace: bool = ...,
    ) -> None: ...
    
    def list(self, regex_search: Optional[str] = ...) -> List[str]: ...
    
    def shallow_copy(
        self,
        extra_dataset_patterns: Optional[Dict[str, Any]] = ...,
    ) -> DataCatalog: ...
    
    def confirm(self, name: str) -> None: ...
```