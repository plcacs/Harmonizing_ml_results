"""Type stubs for kedro_data_catalog_148318 module."""

from __future__ import annotations
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    Type,
    ClassVar,
    Iterable,
    Pattern,
)
from typing import TYPE_CHECKING
from difflib import SequenceMatcher
from logging import Logger
from re import Pattern
from uuid import UUID

from kedro.io.core import (
    CatalogProtocol,
    AbstractDataset,
    AbstractVersionedDataset,
    DatasetAlreadyExistsError,
    DatasetError,
    DatasetNotFoundError,
    Version,
)
from kedro.io.catalog_config_resolver import CatalogConfigResolver
from kedro.utils import _HasRichHandler

if TYPE_CHECKING:
    from kedro.io.memory_dataset import MemoryDataset

class _LazyDataset:
    name: str
    config: Dict[str, Any]
    load_version: Optional[str]
    save_version: Optional[str]

    def __init__(self, name: str, config: Dict[str, Any], load_version: Optional[str] = None, save_version: Optional[str] = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def materialize(self) -> AbstractDataset:
        ...

class KedroDataCatalog(CatalogProtocol):
    _config_resolver: CatalogConfigResolver
    __datasets: Dict[str, AbstractDataset]
    _lazy_datasets: Dict[str, _LazyDataset]
    _load_versions: Dict[str, str]
    _save_version: Optional[str]
    _use_rich_markup: bool
    _logger: ClassVar[Logger]

    def __init__(
        self,
        datasets: Optional[Dict[str, AbstractDataset]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
        config_resolver: Optional[CatalogConfigResolver] = None,
        load_versions: Optional[Dict[str, str]] = None,
        save_version: Optional[str] = None,
    ) -> None:
        ...

    @property
    def datasets(self) -> Dict[str, AbstractDataset]:
        ...

    @datasets.setter
    def datasets(self, value: Any) -> None:
        ...

    def __getattribute__(self, key: str) -> Any:
        ...

    @property
    def config_resolver(self) -> CatalogConfigResolver:
        ...

    def __repr__(self) -> str:
        ...

    def __contains__(self, dataset_name: str) -> bool:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def keys(self) -> List[str]:
        ...

    def values(self) -> List[AbstractDataset]:
        ...

    def items(self) -> List[Tuple[str, AbstractDataset]]:
        ...

    def __iter__(self) -> Iterator[str]:
        ...

    def __getitem__(self, ds_name: str) -> AbstractDataset:
        ...

    def __setitem__(self, key: str, value: Union[AbstractDataset, _LazyDataset, Any]) -> None:
        ...

    def __len__(self) -> int:
        ...

    def get(self, key: str, default: Optional[Any] = None) -> Optional[AbstractDataset]:
        ...

    def _ipython_key_completions_(self) -> List[str]:
        ...

    @classmethod
    def from_config(
        cls,
        catalog: Optional[Dict[str, Any]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        load_versions: Optional[Dict[str, str]] = None,
        save_version: Optional[str] = None,
    ) -> KedroDataCatalog:
        ...

    def to_config(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str], Optional[str]]:
        ...

    def _validate_dataset_config(self, ds_name: str, ds_config: Dict[str, Any]) -> None:
        ...

    def _add_from_config(self, ds_name: str, ds_config: Dict[str, Any]) -> None:
        ...

    def get_dataset(self, ds_name: str, version: Optional[str] = None, suggest: bool = True) -> AbstractDataset:
        ...

    def _get_dataset(self, dataset_name: str, version: Optional[str] = None, suggest: bool = True) -> AbstractDataset:
        ...

    def add(self, ds_name: str, dataset: Union[AbstractDataset, _LazyDataset], replace: bool = False) -> None:
        ...

    def filter(
        self,
        name_regex: Optional[Union[Pattern[str], str]] = None,
        type_regex: Optional[Union[Pattern[str], str]] = None,
        by_type: Optional[Union[Type[AbstractDataset], List[Type[AbstractDataset]]]] = None,
    ) -> List[str]:
        ...

    def list(self, regex_search: Optional[str] = None, regex_flags: int = 0) -> List[str]:
        ...

    def save(self, name: str, data: Any) -> None:
        ...

    def load(self, name: str, version: Optional[str] = None) -> Any:
        ...

    def release(self, name: str) -> None:
        ...

    def confirm(self, name: str) -> None:
        ...

    def add_feed_dict(self, feed_dict: Dict[str, Any], replace: bool = False) -> None:
        ...

    def shallow_copy(self, extra_dataset_patterns: Optional[Iterable[str]] = None) -> KedroDataCatalog:
        ...

    def exists(self, name: str) -> bool:
        ...