```pyi
from __future__ import annotations

import logging
import re
from typing import Any, Iterator, List, Optional, Dict, Tuple, Union, Pattern

from kedro.io.catalog_config_resolver import CatalogConfigResolver, Patterns
from kedro.io.core import AbstractDataset, AbstractVersionedDataset, CatalogProtocol, Version

class _LazyDataset:
    name: str
    config: Dict[str, Any]
    load_version: Optional[Version]
    save_version: Optional[str]
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        load_version: Optional[Version] = None,
        save_version: Optional[str] = None,
    ) -> None: ...
    
    def __repr__(self) -> str: ...
    
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
    
    @property
    def datasets(self) -> Dict[str, Union[_LazyDataset, AbstractDataset]]: ...
    
    @datasets.setter
    def datasets(self, value: Any) -> None: ...
    
    @property
    def config_resolver(self) -> CatalogConfigResolver: ...
    
    def __repr__(self) -> str: ...
    
    def __contains__(self, dataset_name: str) -> bool: ...
    
    def __eq__(self, other: Any) -> bool: ...
    
    def keys(self) -> List[str]: ...
    
    def values(self) -> List[AbstractDataset]: ...
    
    def items(self) -> List[Tuple[str, AbstractDataset]]: ...
    
    def __iter__(self) -> Iterator[str]: ...
    
    def __getitem__(self, ds_name: str) -> AbstractDataset: ...
    
    def __setitem__(self, key: str, value: Union[AbstractDataset, _LazyDataset, Any]) -> None: ...
    
    def __len__(self) -> int: ...
    
    def get(self, key: str, default: Optional[AbstractDataset] = None) -> Optional[AbstractDataset]: ...
    
    def _ipython_key_completions_(self) -> List[str]: ...
    
    @property
    def _logger(self) -> logging.Logger: ...
    
    @classmethod
    def from_config(
        cls,
        catalog: Optional[Dict[str, Dict[str, Any]]] = None,
        credentials: Optional[Dict[str, Dict[str, Any]]] = None,
        load_versions: Optional[Dict[str, str]] = None,
        save_version: Optional[str] = None,
    ) -> KedroDataCatalog: ...
    
    def to_config(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Optional[str]], Optional[str]]: ...
    
    @staticmethod
    def _validate_dataset_config(ds_name: str, ds_config: Any) -> None: ...
    
    def _add_from_config(self, ds_name: str, ds_config: Dict[str, Any]) -> None: ...
    
    def get_dataset(
        self,
        ds_name: str,
        version: Optional[Union[str, Version]] = None,
        suggest: bool = True,
    ) -> AbstractDataset: ...
    
    def _get_dataset(
        self,
        dataset_name: str,
        version: Optional[Union[str, Version]] = None,
        suggest: bool = True,
    ) -> AbstractDataset: ...
    
    def add(self, ds_name: str, dataset: Union[AbstractDataset, _LazyDataset], replace: bool = False) -> None: ...
    
    def filter(
        self,
        name_regex: Optional[Union[str, Pattern[str]]] = None,
        type_regex: Optional[Union[str, Pattern[str]]] = None,
        by_type: Optional[Union[type, List[type]]] = None,
    ) -> List[str]: ...
    
    def list(self, regex_search: Optional[str] = None, regex_flags: int = 0) -> List[str]: ...
    
    def save(self, name: str, data: Any) -> None: ...
    
    def load(self, name: str, version: Optional[str] = None) -> Any: ...
    
    def release(self, name: str) -> None: ...
    
    def confirm(self, name: str) -> None: ...
    
    def add_feed_dict(self, feed_dict: Dict[str, Any], replace: bool = False) -> None: ...
    
    def shallow_copy(self, extra_dataset_patterns: Optional[Dict[str, Any]] = None) -> KedroDataCatalog: ...
    
    def exists(self, name: str) -> bool: ...
```