from __future__ import annotations

import logging
import re
from typing import Any, Iterator, List

from kedro.io.catalog_config_resolver import CatalogConfigResolver, Patterns
from kedro.io.core import (
    AbstractDataset,
    AbstractVersionedDataset,
    CatalogProtocol,
    Version,
)
from kedro.io.memory_dataset import MemoryDataset


class _LazyDataset:
    name: str
    config: dict[str, Any]
    load_version: str | None
    save_version: str | None

    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        load_version: str | None = None,
        save_version: str | None = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def materialize(self) -> AbstractDataset: ...


class KedroDataCatalog(CatalogProtocol):
    _config_resolver: CatalogConfigResolver
    _lazy_datasets: dict[str, _LazyDataset]
    _load_versions: dict[str, str]
    _save_version: str | None
    _use_rich_markup: bool

    def __init__(
        self,
        datasets: dict[str, AbstractDataset] | None = None,
        raw_data: dict[str, Any] | None = None,
        config_resolver: CatalogConfigResolver | None = None,
        load_versions: dict[str, str] | None = None,
        save_version: str | None = None,
    ) -> None: ...

    @property
    def datasets(self) -> dict[str, AbstractDataset | _LazyDataset]: ...

    @datasets.setter
    def datasets(self, value: Any) -> None: ...

    def __getattribute__(self, key: str) -> Any: ...

    @property
    def config_resolver(self) -> CatalogConfigResolver: ...

    def __repr__(self) -> str: ...
    def __contains__(self, dataset_name: str) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def keys(self) -> list[str]: ...
    def values(self) -> list[AbstractDataset | None]: ...
    def items(self) -> list[tuple[str, AbstractDataset | None]]: ...
    def __iter__(self) -> Iterator[str]: ...
    def __getitem__(self, ds_name: str) -> AbstractDataset: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __len__(self) -> int: ...
    def get(self, key: str, default: AbstractDataset | None = None) -> AbstractDataset | None: ...
    def _ipython_key_completions_(self) -> list[str]: ...

    @property
    def _logger(self) -> logging.Logger: ...

    @classmethod
    def from_config(
        cls,
        catalog: dict[str, dict[str, Any]] | None,
        credentials: dict[str, dict[str, Any]] | None = None,
        load_versions: dict[str, str] | None = None,
        save_version: str | None = None,
    ) -> KedroDataCatalog: ...

    def to_config(
        self,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, str | None],
        str | None,
    ]: ...

    @staticmethod
    def _validate_dataset_config(ds_name: str, ds_config: Any) -> None: ...

    def _add_from_config(self, ds_name: str, ds_config: dict[str, Any]) -> None: ...

    def get_dataset(
        self,
        ds_name: str,
        version: Version | None = None,
        suggest: bool = True,
    ) -> AbstractDataset: ...

    def _get_dataset(
        self,
        dataset_name: str,
        version: Version | None = None,
        suggest: bool = True,
    ) -> AbstractDataset: ...

    def add(
        self,
        ds_name: str,
        dataset: AbstractDataset | _LazyDataset,
        replace: bool = False,
    ) -> None: ...

    def filter(
        self,
        name_regex: str | re.Pattern[str] | None = None,
        type_regex: str | re.Pattern[str] | None = None,
        by_type: type[AbstractDataset] | list[type[AbstractDataset]] | None = None,
    ) -> list[str]: ...

    def list(
        self,
        regex_search: str | None = None,
        regex_flags: int = 0,
    ) -> list[str]: ...

    def save(self, name: str, data: Any) -> None: ...
    def load(self, name: str, version: str | None = None) -> Any: ...
    def release(self, name: str) -> None: ...
    def confirm(self, name: str) -> None: ...

    def add_feed_dict(
        self,
        feed_dict: dict[str, Any],
        replace: bool = False,
    ) -> None: ...

    def shallow_copy(
        self,
        extra_dataset_patterns: Patterns | None = None,
    ) -> KedroDataCatalog: ...

    def exists(self, name: str) -> bool: ...