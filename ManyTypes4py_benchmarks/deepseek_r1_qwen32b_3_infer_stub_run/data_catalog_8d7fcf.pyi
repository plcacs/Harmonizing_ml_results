"""Stub file for data_catalog_8d7fcf module."""

from __future__ import annotations
import difflib
import logging
import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    ClassVar,
    Pattern,
    overload,
    Sequence,
    Tuple,
)

from kedro.io.core import (
    AbstractDataset,
    AbstractVersionedDataset,
    DatasetAlreadyExistsError,
    DatasetError,
    DatasetNotFoundError,
    Version,
)
from kedro.io.catalog_config_resolver import CatalogConfigResolver

CATALOG_KEY: str
WORDS_REGEX_PATTERN: Pattern[str]

def _sub_nonword_chars(dataset_name: str) -> str: ...

class _FrozenDatasets:
    """Helper class to access underlying loaded datasets."""
    def __init__(self, *datasets_collections: Union[_FrozenDatasets, Dict[str, Any]]) -> None: ...
    def __setattr__(self, key: str, value: Any) -> None: ...
    def _ipython_key_completions_(self) -> List[str]: ...
    def __getitem__(self, key: str) -> Any: ...
    def __repr__(self) -> str: ...
    _original_names: Dict[str, str]

class DataCatalog:
    """``DataCatalog`` stores instances of ``AbstractDataset`` implementations
    to provide ``load`` and ``save`` capabilities from anywhere in the
    program.
    """
    _config_resolver: CatalogConfigResolver
    _load_versions: Dict[str, Version]
    _save_version: Version
    _datasets: Dict[str, AbstractDataset]
    datasets: _FrozenDatasets
    _logger: logging.Logger
    _use_rich_markup: bool

    def __init__(
        self,
        datasets: Optional[Dict[str, AbstractDataset]] = None,
        feed_dict: Optional[Dict[str, Union[Any, AbstractDataset]]] = None,
        dataset_patterns: Optional[Dict[str, Any]] = None,
        load_versions: Optional[Dict[str, Union[str, Version]]] = None,
        save_version: Optional[str] = None,
        default_pattern: Optional[Dict[str, Any]] = None,
        config_resolver: Optional[CatalogConfigResolver] = None,
    ) -> None: ...

    @classmethod
    def from_config(
        cls,
        catalog: Dict[str, Dict[str, Any]],
        credentials: Optional[Dict[str, Dict[str, Any]]] = None,
        load_versions: Optional[Dict[str, Union[str, Version]]] = None,
        save_version: Optional[str] = None,
    ) -> DataCatalog: ...

    @property
    def config_resolver(self) -> CatalogConfigResolver: ...

    def __repr__(self) -> str: ...
    def __contains__(self, dataset_name: str) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...

    def _get_dataset(
        self,
        dataset_name: str,
        version: Optional[Union[str, Version]] = None,
        suggest: bool = True,
    ) -> AbstractDataset: ...

    def load(self, name: str, version: Optional[Union[str, Version]] = None) -> Any: ...
    def save(self, name: str, data: Any) -> None: ...
    def exists(self, name: str) -> bool: ...
    def release(self, name: str) -> None: ...

    def add(
        self,
        dataset_name: str,
        dataset: AbstractDataset,
        replace: bool = False,
    ) -> None: ...
    def add_all(
        self,
        datasets: Dict[str, AbstractDataset],
        replace: bool = False,
    ) -> None: ...
    def add_feed_dict(
        self,
        feed_dict: Dict[str, Union[Any, AbstractDataset]],
        replace: bool = False,
    ) -> None: ...

    def list(self, regex_search: Optional[str] = None) -> List[str]: ...
    def shallow_copy(
        self,
        extra_dataset_patterns: Optional[Dict[str, Any]] = None,
    ) -> DataCatalog: ...
    def confirm(self, name: str) -> None: ...