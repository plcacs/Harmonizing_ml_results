"""Type stubs for kedro_data_catalog_148318 module."""

from __future__ import annotations
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from logging import Logger
import difflib
import re
from typing import Any, Iterator, List
from kedro.io.core import (
    AbstractDataset,
    AbstractVersionedDataset,
    CatalogProtocol,
    DatasetAlreadyExistsError,
    DatasetError,
    DatasetNotFoundError,
    Version,
)
from kedro.io.catalog_config_resolver import CatalogConfigResolver, Patterns

class _LazyDataset:
    """A helper class to store AbstractDataset configuration and materialize dataset object."""
    def __init__(self, name: str, config: Dict[str, Any], load_version: Optional[str] = None, save_version: Optional[str] = None) -> None: ...
    def __repr__(self) -> str: ...
    def materialize(self) -> AbstractDataset: ...

class KedroDataCatalog(CatalogProtocol):
    """A catalog for managing datasets in Kedro."""
    _config_resolver: CatalogConfigResolver
    _lazy_datasets: Dict[str, _LazyDataset]
    __datasets: Dict[str, AbstractDataset]
    _load_versions: Dict[str, str]
    _save_version: str
    _use_rich_markup: bool
    _logger: ClassVar[Logger]

    def __init__(self, datasets: Optional[Dict[str, AbstractDataset]] = None, raw_data: Optional[Dict[str, Any]] = None, config_resolver: Optional[CatalogConfigResolver] = None, load_versions: Optional[Dict[str, str]] = None, save_version: Optional[str] = None) -> None: ...

    def __getattribute__(self, key: str) -> Any: ...

    def __getitem__(self, ds_name: str) -> AbstractDataset: ...

    def __setitem__(self, key: str, value: Union[AbstractDataset, _LazyDataset, Any]) -> None: ...

    @property
    def datasets(self) -> Dict[str, AbstractDataset]: ...
    @datasets.setter
    def datasets(self, value: Any) -> None: ...

    @property
    def config_resolver(self) -> CatalogConfigResolver: ...

    def __repr__(self) -> str: ...

    def __contains__(self, dataset_name: str) -> bool: ...

    def __eq__(self, other: KedroDataCatalog) -> bool: ...

    def keys(self) -> List[str]: ...

    def values(self) -> List[AbstractDataset]: ...

    def items(self) -> List[Tuple[str, AbstractDataset]]: ...

    def __iter__(self) -> Iterator[str]: ...

    def __len__(self) -> int: ...

    def get(self, key: str, default: Optional[Any] = None) -> Optional[AbstractDataset]: ...

    def _ipython_key_completions_(self) -> List[str]: ...

    @classmethod
    def from_config(cls, catalog: Dict[str, Any], credentials: Optional[Dict[str, Any]] = None, load_versions: Optional[Dict[str, str]] = None, save_version: Optional[str] = None) -> KedroDataCatalog: ...

    def to_config(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str], Optional[str]]: ...

    def get_dataset(self, ds_name: str, version: Optional[str] = None, suggest: bool = True) -> AbstractDataset: ...

    def _get_dataset(self, dataset_name: str, version: Optional[str] = None, suggest: bool = True) -> AbstractDataset: ...

    def add(self, ds_name: str, dataset: Union[AbstractDataset, _LazyDataset], replace: bool = False) -> None: ...

    def filter(self, name_regex: Optional[Union[str, re.Pattern]] = None, type_regex: Optional[Union[str, re.Pattern]] = None, by_type: Optional[Union[type, List[type]]] = None) -> List[str]: ...

    def list(self, regex_search: Optional[str] = None, regex_flags: int = re.IGNORECASE) -> List[str]: ...

    def save(self, name: str, data: Any) -> None: ...

    def load(self, name: str, version: Optional[str] = None) -> Any: ...

    def release(self, name: str) -> None: ...

    def confirm(self, name: str) -> None: ...

    def add_feed_dict(self, feed_dict: Dict[str, Any], replace: bool = False) -> None: ...

    def shallow_copy(self, extra_dataset_patterns: Optional[List[Patterns]] = None) -> KedroDataCatalog: ...

    def exists(self, name: str) -> bool: ...