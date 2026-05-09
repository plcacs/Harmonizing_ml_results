"""Type stubs for kedro_data_catalog_148318 module."""

from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Iterator,
    Iterable,
    Callable,
    Pattern,
    Set,
    Any,
    AnyStr,
    overload,
)
from typing import TYPE_CHECKING
import difflib
import re
from datetime import datetime
from pathlib import Path
from kedro.io.core import (
    AbstractDataset,
    AbstractVersionedDataset,
    CatalogProtocol,
    DatasetAlreadyExistsError,
    DatasetError,
    DatasetNotFoundError,
    Version,
)
from kedro.io import (
    MemoryDataset,
    _is_memory_dataset,
)
from kedro.io.catalog_config_resolver import CatalogConfigResolver
from kedro.utils import _format_rich, _has_rich_handler

class _LazyDataset:
    def __init__(self, name: str, config: dict, load_version: Optional[str] = None, save_version: Optional[str] = None) -> None: ...
    def materialize(self) -> AbstractDataset: ...

class KedroDataCatalog(CatalogProtocol):
    def __init__(
        self,
        datasets: Optional[Dict[str, AbstractDataset]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
        config_resolver: Optional[CatalogConfigResolver] = None,
        load_versions: Optional[Dict[str, str]] = None,
        save_version: Optional[str] = None,
    ) -> None: ...

    @classmethod
    def from_config(
        cls,
        catalog: dict,
        credentials: Optional[dict] = None,
        load_versions: Optional[dict] = None,
        save_version: Optional[str] = None,
    ) -> KedroDataCatalog: ...

    def to_config(self) -> Tuple[dict, dict, dict, Optional[str]]: ...

    def __getitem__(self, ds_name: str) -> AbstractDataset: ...
    def __setitem__(self, key: str, value: Union[AbstractDataset, _LazyDataset]) -> None: ...
    def __contains__(self, dataset_name: str) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

    def keys(self) -> List[str]: ...
    def values(self) -> List[AbstractDataset]: ...
    def items(self) -> List[Tuple[str, AbstractDataset]]: ...

    def get(self, key: str, default: Optional[Any] = None) -> Optional[AbstractDataset]: ...

    def add(
        self,
        ds_name: str,
        dataset: Union[AbstractDataset, _LazyDataset],
        replace: bool = False,
    ) -> None: ...

    def filter(
        self,
        name_regex: Optional[Union[str, Pattern]] = None,
        type_regex: Optional[Union[str, Pattern]] = None,
        by_type: Optional[Union[Type[AbstractDataset], List[Type[AbstractDataset]]]] = None,
    ) -> List[str]: ...

    def list(
        self,
        regex_search: Optional[str] = None,
        regex_flags: int = re.IGNORECASE,
    ) -> List[str]: ...

    def save(self, name: str, data: Any) -> None: ...
    def load(self, name: str, version: Optional[str] = None) -> Any: ...
    def release(self, name: str) -> None: ...
    def confirm(self, name: str) -> None: ...

    def add_feed_dict(self, feed_dict: Dict[str, Any], replace: bool = False) -> None: ...
    def shallow_copy(self, extra_dataset_patterns: Optional[List[Patterns]] = None) -> KedroDataCatalog: ...
    def exists(self, name: str) -> bool: ...

    @property
    def datasets(self) -> Dict[str, AbstractDataset]: ...
    @property
    def config_resolver(self) -> CatalogConfigResolver: ...
    @property
    def _logger(self) -> logging.Logger: ...