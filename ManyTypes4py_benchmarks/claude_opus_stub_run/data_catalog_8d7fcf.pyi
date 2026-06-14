from __future__ import annotations

import logging
import re
from typing import Any

from kedro.io.catalog_config_resolver import CatalogConfigResolver, Patterns
from kedro.io.core import AbstractDataset, Version

CATALOG_KEY: str
WORDS_REGEX_PATTERN: re.Pattern[str]

def _sub_nonword_chars(dataset_name: str) -> str: ...

class _FrozenDatasets:
    _original_names: dict[str, str]
    def __init__(self, *datasets_collections: _FrozenDatasets | dict[str, AbstractDataset] | None) -> None: ...
    def __setattr__(self, key: str, value: Any) -> None: ...
    def _ipython_key_completions_(self) -> list[str]: ...
    def __getitem__(self, key: str) -> Any: ...
    def __repr__(self) -> str: ...

class DataCatalog:
    _config_resolver: CatalogConfigResolver
    _load_versions: dict[str, str]
    _save_version: str | None
    _datasets: dict[str, AbstractDataset]
    datasets: _FrozenDatasets | None
    _use_rich_markup: bool

    def __init__(
        self,
        datasets: dict[str, AbstractDataset] | None = ...,
        feed_dict: dict[str, Any] | None = ...,
        dataset_patterns: Patterns | None = ...,
        load_versions: dict[str, str] | None = ...,
        save_version: str | None = ...,
        default_pattern: Patterns | None = ...,
        config_resolver: CatalogConfigResolver | None = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __contains__(self, dataset_name: str) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    @property
    def config_resolver(self) -> CatalogConfigResolver: ...
    @property
    def _logger(self) -> logging.Logger: ...
    @classmethod
    def from_config(
        cls,
        catalog: dict[str, dict[str, Any]] | None,
        credentials: dict[str, dict[str, Any]] | None = ...,
        load_versions: dict[str, str] | None = ...,
        save_version: str | None = ...,
    ) -> DataCatalog: ...
    def _get_dataset(
        self,
        dataset_name: str,
        version: Version | None = ...,
        suggest: bool = ...,
    ) -> AbstractDataset: ...
    def load(self, name: str, version: str | None = ...) -> Any: ...
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
        datasets: dict[str, AbstractDataset],
        replace: bool = ...,
    ) -> None: ...
    def add_feed_dict(
        self,
        feed_dict: dict[str, Any],
        replace: bool = ...,
    ) -> None: ...
    def list(self, regex_search: str | None = ...) -> list[str]: ...
    def shallow_copy(
        self,
        extra_dataset_patterns: Patterns | None = ...,
    ) -> DataCatalog: ...
    def confirm(self, name: str) -> None: ...