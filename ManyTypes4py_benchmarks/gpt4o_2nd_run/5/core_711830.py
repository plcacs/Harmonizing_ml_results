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
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type, TypeVar, Union, runtime_checkable
from urllib.parse import urlsplit
from cachetools import Cache, cachedmethod
from cachetools.keys import hashkey
from typing_extensions import Self
from kedro.utils import load_obj

if TYPE_CHECKING:
    import os
    from kedro.io.catalog_config_resolver import CatalogConfigResolver, Patterns

VERSION_FORMAT: str = '%Y-%m-%dT%H.%M.%S.%fZ'
VERSIONED_FLAG_KEY: str = 'versioned'
VERSION_KEY: str = 'version'
HTTP_PROTOCOLS: tuple[str, str] = ('http', 'https')
PROTOCOL_DELIMITER: str = '://'
CLOUD_PROTOCOLS: tuple[str, ...] = ('abfs', 'abfss', 'adl', 'gcs', 'gdrive', 'gs', 'oci', 'oss', 's3', 's3a', 's3n')
TYPE_KEY: str = 'type'

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

_DI = TypeVar('_DI')
_DO = TypeVar('_DO')

class AbstractDataset(abc.ABC, Generic[_DI, _DO]):
    _EPHEMERAL: bool = False

    @classmethod
    def from_config(cls: Type[Self], name: str, config: Dict[str, Any], load_version: Optional[str] = None, save_version: Optional[str] = None) -> Self:
        try:
            class_obj, config = parse_dataset_definition(config, load_version, save_version)
        except Exception as exc:
            raise DatasetError(f"An exception occurred when parsing config for dataset '{name}':\n{exc!s}") from exc
        try:
            dataset = class_obj(**config)
        except TypeError as err:
            raise DatasetError(f"\n{err}.\nDataset '{name}' must only contain arguments valid for the constructor of '{class_obj.__module__}.{class_obj.__qualname__}'.") from err
        except Exception as err:
            raise DatasetError(f"\n{err}.\nFailed to instantiate dataset '{name}' of type '{class_obj.__module__}.{class_obj.__qualname__}'.") from err
        return dataset

    def to_config(self) -> Dict[str, Any]:
        return_config = {f'{TYPE_KEY}': f'{type(self).__module__}.{type(self).__name__}'}
        if self._init_args:
            self._init_args.pop('self', None)
            return_config.update(self._init_args)
        if type(self).__name__ == 'CachedDataset':
            cached_ds = return_config.pop('dataset')
            cached_ds_return_config = {}
            if isinstance(cached_ds, dict):
                cached_ds_return_config = cached_ds
            elif isinstance(cached_ds, AbstractDataset):
                cached_ds_return_config = cached_ds.to_config()
            if VERSIONED_FLAG_KEY in cached_ds_return_config:
                return_config[VERSIONED_FLAG_KEY] = cached_ds_return_config.pop(VERSIONED_FLAG_KEY)
            cached_ds_return_config.pop('metadata', None)
            return_config['dataset'] = cached_ds_return_config
        if return_config.pop(VERSION_KEY, None):
            return_config[VERSIONED_FLAG_KEY] = True
        return_config.pop('metadata', None)
        return return_config

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __str__(self) -> str:
        def _to_str(obj: Any, is_root: bool = False) -> str:
            fmt = '{}={}' if is_root else "'{}': {}"
            if isinstance(obj, dict):
                sorted_dict = sorted(obj.items(), key=lambda pair: str(pair[0]))
                text = ', '.join((fmt.format(key, _to_str(value)) for key, value in sorted_dict if value is not None))
                return text if is_root else '{' + text + '}'
            return str(obj)
        return f'{type(self).__name__}({_to_str(self._describe(), True)})'

    @classmethod
    def _load_wrapper(cls: Type[Self], load_func: Callable[[Self], _DO]) -> Callable[[Self], _DO]:
        @wraps(load_func)
        def load(self: Self) -> _DO:
            self._logger.debug('Loading %s', str(self))
            try:
                return load_func(self)
            except DatasetError:
                raise
            except Exception as exc:
                message = f'Failed while loading data from dataset {self!s}.\n{exc!s}'
                raise DatasetError(message) from exc
        load.__annotations__['return'] = load_func.__annotations__.get('return')
        load.__loadwrapped__ = True
        return load

    @classmethod
    def _save_wrapper(cls: Type[Self], save_func: Callable[[Self, _DI], None]) -> Callable[[Self, _DI], None]:
        @wraps(save_func)
        def save(self: Self, data: _DI) -> None:
            if data is None:
                raise DatasetError("Saving 'None' to a 'Dataset' is not allowed")
            try:
                self._logger.debug('Saving %s', str(self))
                save_func(self, data)
            except (DatasetError, FileNotFoundError, NotADirectoryError):
                raise
            except Exception as exc:
                message = f'Failed while saving data to dataset {self!s}.\n{exc!s}'
                raise DatasetError(message) from exc
        save.__annotations__['data'] = save_func.__annotations__.get('data', Any)
        save.__annotations__['return'] = save_func.__annotations__.get('return')
        save.__savewrapped__ = True
        return save

    def __init_subclass__(cls, **kwargs: Any) -> None:
        init_func = cls.__init__

        @wraps(init_func)
        def new_init(self: Self, *args: Any, **kwargs: Any) -> None:
            init_func(self, *args, **kwargs)
            self._init_args = getcallargs(init_func, self, *args, **kwargs)
        cls.__init__ = new_init
        super().__init_subclass__(**kwargs)
        if hasattr(cls, '_load') and (not cls._load.__qualname__.startswith('Abstract')):
            cls.load = cls._load
        if hasattr(cls, '_save') and (not cls._save.__qualname__.startswith('Abstract')):
            cls.save = cls._save
        if hasattr(cls, 'load') and (not cls.load.__qualname__.startswith('Abstract')):
            cls.load = cls._load_wrapper(cls.load if not getattr(cls.load, '__loadwrapped__', False) else cls.load.__wrapped__)
        if hasattr(cls, 'save') and (not cls.save.__qualname__.startswith('Abstract')):
            cls.save = cls._save_wrapper(cls.save if not getattr(cls.save, '__savewrapped__', False) else cls.save.__wrapped__)

    def _pretty_repr(self, object_description: Dict[str, Any]) -> str:
        str_keys = []
        for arg_name, arg_descr in object_description.items():
            if arg_descr is not None:
                descr = pprint.pformat(arg_descr, sort_dicts=False, compact=True, depth=2, width=sys.maxsize)
                str_keys.append(f'{arg_name}={descr}')
        return f'{type(self).__module__}.{type(self).__name__}({', '.join(str_keys)})'

    def __repr__(self) -> str:
        object_description = self._describe()
        if isinstance(object_description, dict) and all((isinstance(key, str) for key in object_description)):
            return self._pretty_repr(object_description)
        self._logger.warning(f"'{type(self).__module__}.{type(self).__name__}' is a subclass of AbstractDataset and it must implement the '_describe' method following the signature of AbstractDataset's '_describe'.")
        return f'{type(self).__module__}.{type(self).__name__}()'

    @abc.abstractmethod
    def load(self) -> _DO:
        raise NotImplementedError(f"'{self.__class__.__name__}' is a subclass of AbstractDataset and it must implement the 'load' method")

    @abc.abstractmethod
    def save(self, data: _DI) -> None:
        raise NotImplementedError(f"'{self.__class__.__name__}' is a subclass of AbstractDataset and it must implement the 'save' method")

    @abc.abstractmethod
    def _describe(self) -> Dict[str, Any]:
        raise NotImplementedError(f"'{self.__class__.__name__}' is a subclass of AbstractDataset and it must implement the '_describe' method")

    def exists(self) -> bool:
        try:
            self._logger.debug('Checking whether target of %s exists', str(self))
            return self._exists()
        except Exception as exc:
            message = f'Failed during exists check for dataset {self!s}.\n{exc!s}'
            raise DatasetError(message) from exc

    def _exists(self) -> bool:
        self._logger.warning("'exists()' not implemented for '%s'. Assuming output does not exist.", self.__class__.__name__)
        return False

    def release(self) -> None:
        try:
            self._logger.debug('Releasing %s', str(self))
            self._release()
        except Exception as exc:
            message = f'Failed during release for dataset {self!s}.\n{exc!s}'
            raise DatasetError(message) from exc

    def _release(self) -> None:
        pass

    def _copy(self, **overwrite_params: Any) -> Self:
        dataset_copy = copy.deepcopy(self)
        for name, value in overwrite_params.items():
            setattr(dataset_copy, name, value)
        return dataset_copy

def generate_timestamp() -> str:
    current_ts = datetime.now(tz=timezone.utc).strftime(VERSION_FORMAT)
    return current_ts[:-4] + current_ts[-1:]

class Version(namedtuple('Version', ['load', 'save'])):
    __slots__ = ()

_CONSISTENCY_WARNING: str = "Save version '{}' did not match load version '{}' for {}. This is strongly discouraged due to inconsistencies it may cause between 'save' and 'load' operations. Please refrain from setting exact load version for intermediate datasets where possible to avoid this warning."
_DEFAULT_PACKAGES: list[str] = ['kedro.io.', 'kedro_datasets.', '']

def parse_dataset_definition(config: Dict[str, Any], load_version: Optional[str] = None, save_version: Optional[str] = None) -> tuple[Type[AbstractDataset], Dict[str, Any]]:
    save_version = save_version or generate_timestamp()
    config = copy.deepcopy(config)
    if TYPE_KEY not in config:
        raise DatasetError("'type' is missing from dataset catalog configuration.\nHint: If this catalog entry is intended for variable interpolation, make sure that the top level key is preceded by an underscore.")
    dataset_type = config.pop(TYPE_KEY)
    class_obj: Optional[Type[AbstractDataset]] = None
    if isinstance(dataset_type, str):
        if len(dataset_type.strip('.')) != len(dataset_type):
            raise DatasetError("'type' class path does not support relative paths or paths ending with a dot.")
        class_paths = (prefix + dataset_type for prefix in _DEFAULT_PACKAGES)
        for class_path in class_paths:
            tmp = _load_obj(class_path)
            if tmp is not None:
                class_obj = tmp
                break
        else:
            hint = 'Hint: If you are trying to use a dataset from `kedro-datasets`, make sure that the package is installed in your current environment. You can do so by running `pip install kedro-datasets` or `pip install kedro-datasets[<dataset-group>]` to install `kedro-datasets` along with related dependencies for the specific dataset group.'
            raise DatasetError(f"Class '{dataset_type}' not found, is this a typo?\n{hint}")
    if not class_obj:
        class_obj = dataset_type
    if not issubclass(class_obj, AbstractDataset):
        raise DatasetError(f"Dataset type '{class_obj.__module__}.{class_obj.__qualname__}' is invalid: all dataset types must extend 'AbstractDataset'.")
    if VERSION_KEY in config:
        message = "'%s' attribute removed from dataset configuration since it is a reserved word and cannot be directly specified"
        logging.getLogger(__name__).warning(message, VERSION_KEY)
        del config[VERSION_KEY]
    if config.pop(VERSIONED_FLAG_KEY, False) or getattr(class_obj, VERSIONED_FLAG_KEY, False):
        config[VERSION_KEY] = Version(load_version, save_version)
    return (class_obj, config)

def _load_obj(class_path: str) -> Optional[Type[AbstractDataset]]:
    mod_path, _, class_name = class_path.rpartition('.')
    try:
        available_classes = load_obj(f'{mod_path}.__all__')
    except (ModuleNotFoundError, AttributeError, ValueError):
        available_classes = None
    try:
        class_obj = load_obj(class_path)
    except (ModuleNotFoundError, ValueError, AttributeError) as exc:
        if available_classes and class_name in available_classes:
            raise DatasetError(f'{exc}. Please see the documentation on how to install relevant dependencies for {class_path}:\nhttps://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#install-dependencies-related-to-the-data-catalog') from exc
        return None
    return class_obj

def _local_exists(filepath: str) -> bool:
    return Path(filepath).exists()

class AbstractVersionedDataset(AbstractDataset[_DI, _DO], abc.ABC):
    def __init__(self, filepath: PurePath, version: Optional[Version], exists_function: Optional[Callable[[str], bool]] = None, glob_function: Optional[Callable[[str], Any]] = None) -> None:
        self._filepath = filepath
        self._version = version
        self._exists_function = exists_function or _local_exists
        self._glob_function = glob_function or iglob
        self._version_cache = Cache(maxsize=2)

    @cachedmethod(cache=attrgetter('_version_cache'), key=partial(hashkey, 'load'))
    def _fetch_latest_load_version(self) -> str:
        pattern = str(self._get_versioned_path('*'))
        try:
            version_paths = sorted(self._glob_function(pattern), reverse=True)
        except Exception as exc:
            message = f'Did not find any versions for {self}. This could be due to insufficient permission. Exception: {exc}'
            raise VersionNotFoundError(message) from exc
        most_recent = next((path for path in version_paths if self._exists_function(path)), None)
        if not most_recent:
            message = f'Did not find any versions for {self}'
            raise VersionNotFoundError(message)
        return PurePath(most_recent).parent.name

    @cachedmethod(cache=attrgetter('_version_cache'), key=partial(hashkey, 'save'))
    def _fetch_latest_save_version(self) -> str:
        return generate_timestamp()

    def resolve_load_version(self) -> Optional[str]:
        if not self._version:
            return None
        if self._version.load:
            return self._version.load
        return self._fetch_latest_load_version()

    def _get_load_path(self) -> PurePath:
        if not self._version:
            return self._filepath
        load_version = self.resolve_load_version()
        return self._get_versioned_path(load_version)

    def resolve_save_version(self) -> Optional[str]:
        if not self._version:
            return None
        if self._version.save:
            return self._version.save
        return self._fetch_latest_save_version()

    def _get_save_path(self) -> PurePath:
        if not self._version:
            return self._filepath
        save_version = self.resolve_save_version()
        versioned_path = self._get_versioned_path(save_version)
        if self._exists_function(str(versioned_path)):
            raise DatasetError(f"Save path '{versioned_path}' for {self!s} must not exist if versioning is enabled.")
        return versioned_path

    def _get_versioned_path(self, version: str) -> PurePath:
        return self._filepath / version / self._filepath.name

    @classmethod
    def _save_wrapper(cls: Type[Self], save_func: Callable[[Self, _DI], None]) -> Callable[[Self, _DI], None]:
        @wraps(save_func)
        def save(self: Self, data: _DI) -> None:
            self._version_cache.clear()
            save_version = self.resolve_save_version()
            try:
                super()._save_wrapper(save_func)(self, data)
            except (FileNotFoundError, NotADirectoryError) as err:
                _default_version = 'YYYY-MM-DDThh.mm.ss.sssZ'
                raise DatasetError(f"Cannot save versioned dataset '{self._filepath.name}' to '{self._filepath.parent.as_posix()}' because a file with the same name already exists in the directory. This is likely because versioning was enabled on a dataset already saved previously. Either remove '{self._filepath.name}' from the directory or manually convert it into a versioned dataset by placing it in a versioned directory (e.g. with default versioning format '{self._filepath.as_posix()}/{_default_version}/{self._filepath.name}').") from err
            load_version = self.resolve_load_version()
            if load_version != save_version:
                warnings.warn(_CONSISTENCY_WARNING.format(save_version, load_version, str(self)))
                self._version_cache.clear()
        return save

    def exists(self) -> bool:
        self._logger.debug('Checking whether target of %s exists', str(self))
        try:
            return self._exists()
        except VersionNotFoundError:
            return False
        except Exception as exc:
            message = f'Failed during exists check for dataset {self!s}.\n{exc!s}'
            raise DatasetError(message) from exc

    def _release(self) -> None:
        super()._release()
        self._version_cache.clear()

def _parse_filepath(filepath: str) -> Dict[str, str]:
    if re.match('^[a-zA-Z]:[\\\\/]', filepath) or re.match('^[a-zA-Z0-9]+://', filepath) is None:
        return {'protocol': 'file', 'path': filepath}
    parsed_path = urlsplit(filepath)
    protocol = parsed_path.scheme or 'file'
    if protocol in HTTP_PROTOCOLS:
        return {'protocol': protocol, 'path': filepath}
    path = parsed_path.path
    if protocol == 'file':
        windows_path = re.match('^/([a-zA-Z])[:|]([\\\\/].*)$', path)
        if windows_path:
            path = ':'.join(windows_path.groups())
    if parsed_path.query:
        path = f'{path}?{parsed_path.query}'
    if parsed_path.fragment:
        path = f'{path}#{parsed_path.fragment}'
    options = {'protocol': protocol, 'path': path}
    if parsed_path.netloc and protocol in CLOUD_PROTOCOLS:
        host_with_port = parsed_path.netloc.rsplit('@', 1)[-1]
        host = host_with_port.rsplit(':', 1)[0]
        options['path'] = host + options['path']
        if protocol in ['abfss', 'oci'] and parsed_path.username:
            options['path'] = parsed_path.username + '@' + options['path']
    return options

def get_protocol_and_path(filepath: Union[str, PurePath], version: Optional[Version] = None) -> tuple[str, str]:
    options_dict = _parse_filepath(str(filepath))
    path = options_dict['path']
    protocol = options_dict['protocol']
    if protocol in HTTP_PROTOCOLS:
        if version is not None:
            raise DatasetError('Versioning is not supported for HTTP protocols. Please remove the `versioned` flag from the dataset configuration.')
        path = path.split(PROTOCOL_DELIMITER, 1)[-1]
    return (protocol, path)

def get_filepath_str(raw_path: PurePath, protocol: str) -> str:
    path = raw_path.as_posix()
    if protocol in HTTP_PROTOCOLS:
        path = ''.join((protocol, PROTOCOL_DELIMITER, path))
    return path

def validate_on_forbidden_chars(**kwargs: str) -> None:
    for key, value in kwargs.items():
        if ' ' in value or ';' in value:
            raise DatasetError(f"Neither white-space nor semicolon are allowed in '{key}'.")

_C = TypeVar('_C')

@runtime_checkable
class CatalogProtocol(Protocol[_C]):

    def __contains__(self, ds_name: str) -> bool:
        ...

    @property
    def config_resolver(self) -> CatalogConfigResolver:
        ...

    @classmethod
    def from_config(cls: Type[Self], catalog: Dict[str, Any]) -> Self:
        ...

    def _get_dataset(self, dataset_name: str, version: Optional[str] = None, suggest: bool = True) -> _C:
        ...

    def list(self, regex_search: Optional[str] = None) -> list[str]:
        ...

    def save(self, name: str, data: Any) -> None:
        ...

    def load(self, name: str, version: Optional[str] = None) -> Any:
        ...

    def add(self, ds_name: str, dataset: _C, replace: bool = False) -> None:
        ...

    def add_feed_dict(self, datasets: Dict[str, _C], replace: bool = False) -> None:
        ...

    def exists(self, name: str) -> bool:
        ...

    def release(self, name: str) -> None:
        ...

    def confirm(self, name: str) -> None:
        ...

    def shallow_copy(self, extra_dataset_patterns: Optional[Patterns] = None) -> Self:
        ...

def _validate_versions(datasets: Optional[Dict[str, AbstractDataset]], load_versions: Dict[str, str], save_version: Optional[str]) -> tuple[Dict[str, str], Optional[str]]:
    if not datasets:
        return (load_versions, save_version)
    cur_load_versions = load_versions.copy()
    cur_save_version = save_version
    for ds_name, ds in datasets.items():
        cur_ds = ds._dataset if ds.__class__.__name__ == 'CachedDataset' else ds
        if isinstance(cur_ds, AbstractVersionedDataset) and cur_ds._version:
            if cur_ds._version.load:
                cur_load_versions[ds_name] = cur_ds._version.load
            if cur_ds._version.save:
                cur_save_version = cur_save_version or cur_ds._version.save
                if cur_save_version != cur_ds._version.save:
                    raise VersionAlreadyExistsError(f'Cannot add a dataset `{ds_name}` with `{cur_ds._version.save}` save version. Save version set for the catalog is `{cur_save_version}`All datasets in the catalog must have the same save version.')
    return (cur_load_versions, cur_save_version)
