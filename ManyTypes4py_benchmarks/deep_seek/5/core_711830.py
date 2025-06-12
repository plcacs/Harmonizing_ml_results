from __future__ import annotations
import abc
import copy
import logging
import pprint
import re
import sys
import warnings
from collections import namedtuple
from datetime import datetime, timezone
from functools import partial, wraps
from glob import iglob
from inspect import getcallargs
from operator import attrgetter
from pathlib import Path, PurePath, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, List, Optional, Tuple, Type, Union, cast
from urllib.parse import urlsplit
from cachetools import Cache, cachedmethod
from cachetools.keys import hashkey
from typing_extensions import Self, Protocol, runtime_checkable, TypeVar

if TYPE_CHECKING:
    import os
    from kedro.io.catalog_config_resolver import CatalogConfigResolver, Patterns

VERSION_FORMAT = '%Y-%m-%dT%H.%M.%S.%fZ'
VERSIONED_FLAG_KEY = 'versioned'
VERSION_KEY = 'version'
HTTP_PROTOCOLS = ('http', 'https')
PROTOCOL_DELIMITER = '://'
CLOUD_PROTOCOLS = ('abfs', 'abfss', 'adl', 'gcs', 'gdrive', 'gs', 'oci', 'oss', 's3', 's3a', 's3n')
TYPE_KEY = 'type'

_DI = TypeVar('_DI')
_DO = TypeVar('_O')
_C = TypeVar('_C')

class DatasetError(Exception):
    pass

class DatasetNotFoundError(DatasetError):
    pass

class DatasetAlreadyExistsError(DatasetError):
    pass

class VersionNotFoundError(DatasetError):
    pass

class VersionAlreadyExistsError(DatasetError):
    pass

class Version(namedtuple('Version', ['load', 'save'])):
    load: Optional[str]
    save: Optional[str]
    __slots__ = ()

class AbstractDataset(abc.ABC, Generic[_DI, _DO]):
    _EPHEMERAL: bool = False
    _init_args: Dict[str, Any]

    @classmethod
    def from_config(
        cls,
        name: str,
        config: Dict[str, Any],
        load_version: Optional[str] = None,
        save_version: Optional[str] = None
    ) -> AbstractDataset:
        pass

    def to_config(self) -> Dict[str, Any]:
        pass

    @property
    def _logger(self) -> logging.Logger:
        pass

    def __str__(self) -> str:
        pass

    @classmethod
    def _load_wrapper(cls, load_func: Callable[..., _DO]) -> Callable[..., _DO]:
        pass

    @classmethod
    def _save_wrapper(cls, save_func: Callable[..., None]) -> Callable[..., None]:
        pass

    def __init_subclass__(cls, **kwargs: Any) -> None:
        pass

    def _pretty_repr(self, object_description: Dict[str, Any]) -> str:
        pass

    def __repr__(self) -> str:
        pass

    @abc.abstractmethod
    def load(self) -> _DO:
        pass

    @abc.abstractmethod
    def save(self, data: _DI) -> None:
        pass

    @abc.abstractmethod
    def _describe(self) -> Dict[str, Any]:
        pass

    def exists(self) -> bool:
        pass

    def _exists(self) -> bool:
        pass

    def release(self) -> None:
        pass

    def _release(self) -> None:
        pass

    def _copy(self, **overwrite_params: Any) -> Self:
        pass

def generate_timestamp() -> str:
    pass

_CONSISTENCY_WARNING = "Save version '{}' did not match load version '{}' for {}. This is strongly discouraged due to inconsistencies it may cause between 'save' and 'load' operations. Please refrain from setting exact load version for intermediate datasets where possible to avoid this warning."
_DEFAULT_PACKAGES = ['kedro.io.', 'kedro_datasets.', '']

def parse_dataset_definition(
    config: Dict[str, Any],
    load_version: Optional[str] = None,
    save_version: Optional[str] = None
) -> Tuple[Type[AbstractDataset], Dict[str, Any]]:
    pass

def _load_obj(class_path: str) -> Optional[Type[Any]]:
    pass

def _local_exists(filepath: str) -> bool:
    pass

class AbstractVersionedDataset(AbstractDataset[_DI, _DO], abc.ABC):
    def __init__(
        self,
        filepath: PurePath,
        version: Optional[Version],
        exists_function: Optional[Callable[[str], bool]] = None,
        glob_function: Optional[Callable[[str], List[str]]] = None
    ) -> None:
        pass

    @cachedmethod(cache=attrgetter('_version_cache'), key=partial(hashkey, 'load'))
    def _fetch_latest_load_version(self) -> str:
        pass

    @cachedmethod(cache=attrgetter('_version_cache'), key=partial(hashkey, 'save'))
    def _fetch_latest_save_version(self) -> str:
        pass

    def resolve_load_version(self) -> Optional[str]:
        pass

    def _get_load_path(self) -> PurePath:
        pass

    def resolve_save_version(self) -> Optional[str]:
        pass

    def _get_save_path(self) -> PurePath:
        pass

    def _get_versioned_path(self, version: str) -> PurePath:
        pass

    @classmethod
    def _save_wrapper(cls, save_func: Callable[..., None]) -> Callable[..., None]:
        pass

    def exists(self) -> bool:
        pass

    def _release(self) -> None:
        pass

def _parse_filepath(filepath: str) -> Dict[str, str]:
    pass

def get_protocol_and_path(
    filepath: Union[str, PurePath],
    version: Optional[Version] = None
) -> Tuple[str, str]:
    pass

def get_filepath_str(raw_path: PurePath, protocol: str) -> str:
    pass

def validate_on_forbidden_chars(**kwargs: str) -> None:
    pass

@runtime_checkable
class CatalogProtocol(Protocol[_C]):
    def __contains__(self, ds_name: str) -> bool:
        pass

    @property
    def config_resolver(self) -> CatalogConfigResolver:
        pass

    @classmethod
    def from_config(cls, catalog: Dict[str, Any]) -> CatalogProtocol:
        pass

    def _get_dataset(
        self,
        dataset_name: str,
        version: Optional[Version] = None,
        suggest: bool = True
    ) -> AbstractDataset:
        pass

    def list(self, regex_search: Optional[str] = None) -> List[str]:
        pass

    def save(self, name: str, data: Any) -> None:
        pass

    def load(self, name: str, version: Optional[Version] = None) -> Any:
        pass

    def add(self, ds_name: str, dataset: AbstractDataset, replace: bool = False) -> None:
        pass

    def add_feed_dict(self, datasets: Dict[str, AbstractDataset], replace: bool = False) -> None:
        pass

    def exists(self, name: str) -> bool:
        pass

    def release(self, name: str) -> None:
        pass

    def confirm(self, name: str) -> None:
        pass

    def shallow_copy(self, extra_dataset_patterns: Optional[Patterns] = None) -> CatalogProtocol:
        pass

def _validate_versions(
    datasets: Optional[Dict[str, AbstractDataset]],
    load_versions: Dict[str, str],
    save_version: Optional[str]
) -> Tuple[Dict[str, str], Optional[str]]:
    pass
