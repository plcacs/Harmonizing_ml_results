#!/usr/bin/env python
"""This module provides a set of classes which underpin the data loading and
saving functionality provided by ``kedro.io``.
"""

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
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, Optional, Tuple, Type, TypeVar, Union
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
HTTP_PROTOCOLS: Tuple[str, str] = ('http', 'https')
PROTOCOL_DELIMITER: str = '://'
CLOUD_PROTOCOLS: Tuple[str, ...] = (
    'abfs', 'abfss', 'adl', 'gcs', 'gdrive', 'gs', 'oci', 'oss', 's3', 's3a', 's3n'
)
TYPE_KEY: str = 'type'

class DatasetError(Exception):
    """``DatasetError`` raised by ``AbstractDataset`` implementations
    in case of failure of input/output methods.

    ``AbstractDataset`` implementations should provide instructive
    information in case of failure.
    """
    pass

class DatasetNotFoundError(DatasetError):
    """``DatasetNotFoundError`` raised by ```DataCatalog`` and ``KedroDataCatalog``
    classes in case of trying to use a non-existing dataset.
    """
    pass

class DatasetAlreadyExistsError(DatasetError):
    """``DatasetAlreadyExistsError`` raised by ```DataCatalog`` and ``KedroDataCatalog``
    classes in case of trying to add a dataset which already exists in the ``DataCatalog``.
    """
    pass

class VersionNotFoundError(DatasetError):
    """``VersionNotFoundError`` raised by ``AbstractVersionedDataset`` implementations
    in case of no load versions available for the dataset.
    """
    pass

class VersionAlreadyExistsError(DatasetError):
    """``VersionAlreadyExistsError`` raised by ``DataCatalog`` and ``KedroDataCatalog``
    classes when attempting to add a dataset to a catalog with a save version
    that conflicts with the save version already set for the catalog.
    """
    pass

_DI = TypeVar("_DI")
_DO = TypeVar("_DO")

class AbstractDataset(abc.ABC, Generic[_DI, _DO]):
    """``AbstractDataset`` is the base class for all dataset implementations.

    All dataset implementations should extend this abstract class
    and implement the methods marked as abstract.
    If a specific dataset implementation cannot be used in conjunction with
    the ``ParallelRunner``, such user-defined dataset should have the
    attribute `_SINGLE_PROCESS = True`.
    Example:
    ::

        >>> from pathlib import Path, PurePosixPath
        >>> import pandas as pd
        >>> from kedro.io import AbstractDataset
        >>>
        >>>
        >>> class MyOwnDataset(AbstractDataset[pd.DataFrame, pd.DataFrame]):
        >>>     def __init__(self, filepath, param1, param2=True):
        >>>         self._filepath = PurePosixPath(filepath)
        >>>         self._param1 = param1
        >>>         self._param2 = param2
        >>>
        >>>     def load(self) -> pd.DataFrame:
        >>>         return pd.read_csv(self._filepath)
        >>>
        >>>     def save(self, df: pd.DataFrame) -> None:
        >>>         df.to_csv(str(self._filepath))
        >>>
        >>>     def _exists(self) -> bool:
        >>>         return Path(self._filepath.as_posix()).exists()
        >>>
        >>>     def _describe(self):
        >>>         return dict(param1=self._param1, param2=self._param2)

    Example catalog.yml specification:
    ::

        my_dataset:
            type: <path-to-my-own-dataset>.MyOwnDataset
            filepath: data/01_raw/my_data.csv
            param1: <param1-value> # param1 is a required argument
            # param2 will be True by default
    """
    "\n    Datasets are persistent by default. User-defined datasets that\n    are not made to be persistent, such as instances of `MemoryDataset`,\n    need to change the `_EPHEMERAL` attribute to 'True'.\n    "
    _EPHEMERAL: bool = False

    @classmethod
    def from_config(
        cls: Type[Self],
        name: str,
        config: Dict[str, Any],
        load_version: Optional[str] = None,
        save_version: Optional[str] = None,
    ) -> Self:
        """Create a dataset instance using the configuration provided.

        Args:
            name: Data set name.
            config: Data set config dictionary.
            load_version: Version string to be used for ``load`` operation if
                the dataset is versioned. Has no effect on the dataset
                if versioning was not enabled.
            save_version: Version string to be used for ``save`` operation if
                the dataset is versioned. Has no effect on the dataset
                if versioning was not enabled.

        Returns:
            An instance of an ``AbstractDataset`` subclass.

        Raises:
            DatasetError: When the function fails to create the dataset
                from its config.

        """
        try:
            class_obj, config = parse_dataset_definition(config, load_version, save_version)
        except Exception as exc:
            raise DatasetError(
                f"An exception occurred when parsing config for dataset '{name}':\n{exc!s}"
            ) from exc
        try:
            dataset = class_obj(**config)  # type: ignore
        except TypeError as err:
            raise DatasetError(
                f"\n{err}.\nDataset '{name}' must only contain arguments valid for the constructor of '{class_obj.__module__}.{class_obj.__qualname__}'."
            ) from err
        except Exception as err:
            raise DatasetError(
                f"\n{err}.\nFailed to instantiate dataset '{name}' of type '{class_obj.__module__}.{class_obj.__qualname__}'."
            ) from err
        return dataset

    def to_config(self) -> Dict[str, Any]:
        """Converts the dataset instance into a dictionary-based configuration for
        serialization. Ensures that any subclass-specific details are handled, with
        additional logic for versioning and caching implemented for `CachedDataset`.

        Adds a key for the dataset's type using its module and class name and
        includes the initialization arguments.

        For `CachedDataset` it extracts the underlying dataset's configuration,
        handles the `versioned` flag and removes unnecessary metadata. It also
        ensures the embedded dataset's configuration is appropriately flattened
        or transformed.

        If the dataset has a version key, it sets the `versioned` flag in the
        configuration.

        Removes the `metadata` key from the configuration if present.

        Returns:
            A dictionary containing the dataset's type and initialization arguments.
        """
        return_config: Dict[str, Any] = {f'{TYPE_KEY}': f'{type(self).__module__}.{type(self).__name__}'}
        if self._init_args:
            self._init_args.pop('self', None)
            return_config.update(self._init_args)
        if type(self).__name__ == 'CachedDataset':
            cached_ds = return_config.pop('dataset')
            cached_ds_return_config: Dict[str, Any] = {}
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
            """Returns a string representation where
            1. The root level (i.e. the Dataset.__init__ arguments) are
            formatted like Dataset(key=value).
            2. Dictionaries have the keys alphabetically sorted recursively.
            3. None values are not shown.
            """
            fmt = '{}={}' if is_root else "'{}': {}"
            if isinstance(obj, dict):
                sorted_dict = sorted(obj.items(), key=lambda pair: str(pair[0]))
                text = ', '.join((fmt.format(key, _to_str(value)) for key, value in sorted_dict if value is not None))
                return text if is_root else '{' + text + '}'
            return str(obj)
        return f'{type(self).__name__}({_to_str(self._describe(), True)})'

    @classmethod
    def _load_wrapper(cls, load_func: Callable[..., _DO]) -> Callable[..., _DO]:
        """Decorate `load_func` with logging and error handling code."""
        @wraps(load_func)
        def load(self: AbstractDataset[_DI, _DO]) -> _DO:
            self._logger.debug('Loading %s', str(self))
            try:
                return load_func(self)
            except DatasetError:
                raise
            except Exception as exc:
                message = f'Failed while loading data from dataset {self!s}.\n{exc!s}'
                raise DatasetError(message) from exc
        load.__annotations__['return'] = load_func.__annotations__.get('return')
        load.__loadwrapped__ = True  # type: ignore[attr-defined]
        return load

    @classmethod
    def _save_wrapper(cls, save_func: Callable[..., None]) -> Callable[[AbstractDataset[_DI, _DO], _DI], None]:
        """Decorate `save_func` with logging and error handling code."""
        @wraps(save_func)
        def save(self: AbstractDataset[_DI, _DO], data: _DI) -> None:
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
        save.__savewrapped__ = True  # type: ignore[attr-defined]
        return save

    def __init_subclass__(cls: Type[Any], **kwargs: Any) -> None:
        """Customizes the behavior of subclasses of AbstractDataset during
        their creation. This method is automatically invoked when a subclass
        of AbstractDataset is defined.

        Decorates the `load` and `save` methods provided by the class.
        If `_load` or `_save` are defined, alias them as a prerequisite.
        """
        init_func = cls.__init__

        @wraps(init_func)
        def new_init(self: AbstractDataset[_DI, _DO], *args: Any, **kwargs: Any) -> None:
            """Executes the original __init__, then save the arguments used
            to initialize the instance.
            """
            init_func(self, *args, **kwargs)
            self._init_args = getcallargs(init_func, self, *args, **kwargs)
        cls.__init__ = new_init
        super().__init_subclass__(**kwargs)
        if hasattr(cls, '_load') and (not cls._load.__qualname__.startswith('Abstract')):
            cls.load = cls._load  # type: ignore
        if hasattr(cls, '_save') and (not cls._save.__qualname__.startswith('Abstract')):
            cls.save = cls._save  # type: ignore
        if hasattr(cls, 'load') and (not cls.load.__qualname__.startswith('Abstract')):
            cls.load = cls._load_wrapper(cls.load if not getattr(cls.load, '__loadwrapped__', False) else cls.load.__wrapped__)  # type: ignore
        if hasattr(cls, 'save') and (not cls.save.__qualname__.startswith('Abstract')):
            cls.save = cls._save_wrapper(cls.save if not getattr(cls.save, '__savewrapped__', False) else cls.save.__wrapped__)  # type: ignore

    def _pretty_repr(self, object_description: Dict[str, Any]) -> str:
        str_keys: list[str] = []
        for arg_name, arg_descr in object_description.items():
            if arg_descr is not None:
                descr = pprint.pformat(arg_descr, sort_dicts=False, compact=True, depth=2, width=sys.maxsize)
                str_keys.append(f'{arg_name}={descr}')
        return f'{type(self).__module__}.{type(self).__name__}({", ".join(str_keys)})'

    def __repr__(self) -> str:
        object_description = self._describe()
        if isinstance(object_description, dict) and all((isinstance(key, str) for key in object_description)):
            return self._pretty_repr(object_description)
        self._logger.warning(f"'{type(self).__module__}.{type(self).__name__}' is a subclass of AbstractDataset and it must implement the '_describe' method following the signature of AbstractDataset's '_describe'.")
        return f'{type(self).__module__}.{type(self).__name__}()'

    @abc.abstractmethod
    def load(self) -> _DO:
        """Loads data by delegation to the provided load method.

        Returns:
            Data returned by the provided load method.

        Raises:
            DatasetError: When underlying load method raises error.
        """
        raise NotImplementedError(f"'{self.__class__.__name__}' is a subclass of AbstractDataset and it must implement the 'load' method")

    @abc.abstractmethod
    def save(self, data: _DI) -> None:
        """Saves data by delegation to the provided save method.

        Args:
            data: the value to be saved by provided save method.

        Raises:
            DatasetError: when underlying save method raises error.
            FileNotFoundError: when save method got file instead of dir, on Windows.
            NotADirectoryError: when save method got file instead of dir, on Unix.
        """
        raise NotImplementedError(f"'{self.__class__.__name__}' is a subclass of AbstractDataset and it must implement the 'save' method")

    @abc.abstractmethod
    def _describe(self) -> Dict[str, Any]:
        raise NotImplementedError(f"'{self.__class__.__name__}' is a subclass of AbstractDataset and it must implement the '_describe' method")

    def exists(self) -> bool:
        """Checks whether a dataset's output already exists by calling
        the provided _exists() method.

        Returns:
            Flag indicating whether the output already exists.

        Raises:
            DatasetError: when underlying exists method raises error.
        """
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
        """Release any cached data.

        Raises:
            DatasetError: when underlying release method raises error.
        """
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
    """Generate the timestamp to be used by versioning.

    Returns:
        String representation of the current timestamp.
    """
    current_ts: str = datetime.now(tz=timezone.utc).strftime(VERSION_FORMAT)
    return current_ts[:-4] + current_ts[-1:]

class Version(namedtuple('Version', ['load', 'save'])):
    """This namedtuple is used to provide load and save versions for versioned
    datasets. If ``Version.load`` is None, then the latest available version
    is loaded. If ``Version.save`` is None, then save version is formatted as
    YYYY-MM-DDThh.mm.ss.sssZ of the current timestamp.
    """
    __slots__ = ()

_DEFAULT_PACKAGES: list[str] = ['kedro.io.', 'kedro_datasets.', '']

def parse_dataset_definition(
    config: Dict[str, Any],
    load_version: Optional[str] = None,
    save_version: Optional[str] = None,
) -> Tuple[Type[AbstractDataset[Any, Any]], Dict[str, Any]]:
    """Parse and instantiate a dataset class using the configuration provided.

    Args:
        config: Data set config dictionary. It *must* contain the `type` key
            with fully qualified class name or the class object.
        load_version: Version string to be used for ``load`` operation if
            the dataset is versioned. Has no effect on the dataset
            if versioning was not enabled.
        save_version: Version string to be used for ``save`` operation if
            the dataset is versioned. Has no effect on the dataset
            if versioning was not enabled.

    Raises:
        DatasetError: If the function fails to parse the configuration provided.

    Returns:
        2-tuple: (Dataset class object, configuration dictionary)
    """
    save_version = save_version or generate_timestamp()
    config = copy.deepcopy(config)
    if TYPE_KEY not in config:
        raise DatasetError("'type' is missing from dataset catalog configuration.\nHint: If this catalog entry is intended for variable interpolation, make sure that the top level key is preceded by an underscore.")
    dataset_type = config.pop(TYPE_KEY)
    class_obj: Optional[Any] = None
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
            hint = ('Hint: If you are trying to use a dataset from `kedro-datasets`, make sure that '
                    'the package is installed in your current environment. You can do so by running '
                    '`pip install kedro-datasets` or `pip install kedro-datasets[<dataset-group>]` to install '
                    '`kedro-datasets` along with related dependencies for the specific dataset group.')
            raise DatasetError(f"Class '{dataset_type}' not found, is this a typo?\n{hint}")
    if not class_obj:
        class_obj = dataset_type
    if not issubclass(class_obj, AbstractDataset):
        raise DatasetError(f"Dataset type '{class_obj.__module__}.{class_obj.__qualname__}' is invalid: all dataset types must extend 'AbstractDataset'.")
    if VERSION_KEY in config:
        message: str = "'%s' attribute removed from dataset configuration since it is a reserved word and cannot be directly specified"
        logging.getLogger(__name__).warning(message, VERSION_KEY)
        del config[VERSION_KEY]
    if config.pop(VERSIONED_FLAG_KEY, False) or getattr(class_obj, VERSIONED_FLAG_KEY, False):
        config[VERSION_KEY] = Version(load_version, save_version)
    return (class_obj, config)

def _load_obj(class_path: str) -> Optional[Any]:
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

def _local_exists(filepath: Union[str, Path]) -> bool:
    return Path(filepath).exists()

class AbstractVersionedDataset(AbstractDataset[_DI, _DO], abc.ABC):
    """
    ``AbstractVersionedDataset`` is the base class for all versioned dataset
    implementations.

    All datasets that implement versioning should extend this
    abstract class and implement the methods marked as abstract.

    Example:
    ::

        >>> from pathlib import Path, PurePosixPath
        >>> import pandas as pd
        >>> from kedro.io import AbstractVersionedDataset
        >>>
        >>>
        >>> class MyOwnDataset(AbstractVersionedDataset):
        >>>     def __init__(self, filepath, version, param1, param2=True):
        >>>         super().__init__(PurePosixPath(filepath), version)
        >>>         self._param1 = param1
        >>>         self._param2 = param2
        >>>
        >>>     def load(self) -> pd.DataFrame:
        >>>         load_path = self._get_load_path()
        >>>         return pd.read_csv(load_path)
        >>>
        >>>     def save(self, df: pd.DataFrame) -> None:
        >>>         save_path = self._get_save_path()
        >>>         df.to_csv(str(save_path))
        >>>
        >>>     def _exists(self) -> bool:
        >>>         path = self._get_load_path()
        >>>         return Path(path.as_posix()).exists()
        >>>
        >>>     def _describe(self):
        >>>         return dict(version=self._version, param1=self._param1, param2=self._param2)

    Example catalog.yml specification:
    ::

        my_dataset:
            type: <path-to-my-own-dataset>.MyOwnDataset
            filepath: data/01_raw/my_data.csv
            versioned: true
            param1: <param1-value> # param1 is a required argument
            # param2 will be True by default
    """

    def __init__(
        self,
        filepath: Union[Path, PurePath],
        version: Version,
        exists_function: Optional[Callable[[Any], bool]] = None,
        glob_function: Optional[Callable[..., Iterable[str]]] = None,
    ) -> None:
        """Creates a new instance of ``AbstractVersionedDataset``.

        Args:
            filepath: Filepath in POSIX format to a file.
            version: If specified, should be an instance of
                ``kedro.io.core.Version``. If its ``load`` attribute is
                None, the latest version will be loaded. If its ``save``
                attribute is None, save version will be autogenerated.
            exists_function: Function that is used for determining whether
                a path exists in a filesystem.
            glob_function: Function that is used for finding all paths
                in a filesystem, which match a given pattern.
        """
        self._filepath: Union[Path, PurePath] = filepath
        self._version: Version = version
        self._exists_function: Callable[[Any], bool] = exists_function or _local_exists
        self._glob_function: Callable[..., Iterable[str]] = glob_function or iglob
        self._version_cache: Cache = Cache(maxsize=2)

    @cachedmethod(cache=attrgetter('_version_cache'), key=partial(hashkey, 'load'))
    def _fetch_latest_load_version(self) -> str:
        pattern: str = str(self._get_versioned_path('*'))
        try:
            version_paths: Iterable[str] = sorted(self._glob_function(pattern), reverse=True)
        except Exception as exc:
            message: str = f'Did not find any versions for {self}. This could be due to insufficient permission. Exception: {exc}'
            raise VersionNotFoundError(message) from exc
        most_recent: Optional[str] = next((path for path in version_paths if self._exists_function(path)), None)
        if not most_recent:
            message = f'Did not find any versions for {self}'
            raise VersionNotFoundError(message)
        return PurePath(most_recent).parent.name

    @cachedmethod(cache=attrgetter('_version_cache'), key=partial(hashkey, 'save'))
    def _fetch_latest_save_version(self) -> str:
        """Generate and cache the current save version"""
        return generate_timestamp()

    def resolve_load_version(self) -> Optional[str]:
        """Compute the version the dataset should be loaded with."""
        if not self._version:
            return None
        if self._version.load:
            return self._version.load
        return self._fetch_latest_load_version()

    def _get_load_path(self) -> Union[Path, PurePath]:
        if not self._version:
            return self._filepath
        load_version: Optional[str] = self.resolve_load_version()
        return self._get_versioned_path(load_version)  # type: ignore

    def resolve_save_version(self) -> Optional[str]:
        """Compute the version the dataset should be saved with."""
        if not self._version:
            return None
        if self._version.save:
            return self._version.save
        return self._fetch_latest_save_version()

    def _get_save_path(self) -> Union[Path, PurePath]:
        if not self._version:
            return self._filepath
        save_version: Optional[str] = self.resolve_save_version()
        versioned_path: Union[Path, PurePath] = self._get_versioned_path(save_version)  # type: ignore
        if self._exists_function(str(versioned_path)):
            raise DatasetError(f"Save path '{versioned_path}' for {self!s} must not exist if versioning is enabled.")
        return versioned_path

    def _get_versioned_path(self, version: str) -> Union[Path, PurePath]:
        return self._filepath / version / self._filepath.name

    @classmethod
    def _save_wrapper(cls, save_func: Callable[..., None]) -> Callable[[AbstractVersionedDataset, _DI], None]:
        """Decorate `save_func` with logging and error handling code."""
        @wraps(save_func)
        def save(self: AbstractVersionedDataset, data: _DI) -> None:
            self._version_cache.clear()
            save_version: Optional[str] = self.resolve_save_version()
            try:
                super(AbstractVersionedDataset, self)._save_wrapper(save_func)(self, data)
            except (FileNotFoundError, NotADirectoryError) as err:
                _default_version: str = 'YYYY-MM-DDThh.mm.ss.sssZ'
                raise DatasetError(
                    f"Cannot save versioned dataset '{self._filepath.name}' to '{self._filepath.parent.as_posix()}' because a file with the same name already exists in the directory. This is likely because versioning was enabled on a dataset already saved previously. Either remove '{self._filepath.name}' from the directory or manually convert it into a versioned dataset by placing it in a versioned directory (e.g. with default versioning format '{self._filepath.as_posix()}/{_default_version}/{self._filepath.name}')."
                ) from err
            load_version: Optional[str] = self.resolve_load_version()
            if load_version != save_version:
                warnings.warn(_CONSISTENCY_WARNING.format(save_version, load_version, str(self)))
                self._version_cache.clear()
        return save

    def exists(self) -> bool:
        """Checks whether a dataset's output already exists by calling
        the provided _exists() method.

        Returns:
            Flag indicating whether the output already exists.

        Raises:
            DatasetError: when underlying exists method raises error.
        """
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
    """Split filepath on protocol and path. Based on `fsspec.utils.infer_storage_options`.

    Args:
        filepath: Either local absolute file path or URL (s3://bucket/file.csv)

    Returns:
        Parsed filepath.
    """
    if re.match('^[a-zA-Z]:[\\\\/]', filepath) or re.match('^[a-zA-Z0-9]+://', filepath) is None:
        return {'protocol': 'file', 'path': filepath}
    parsed_path = urlsplit(filepath)
    protocol: str = parsed_path.scheme or 'file'
    if protocol in HTTP_PROTOCOLS:
        return {'protocol': protocol, 'path': filepath}
    path: str = parsed_path.path
    if protocol == 'file':
        windows_path = re.match('^/([a-zA-Z])[:|]([\\\\/].*)$', path)
        if windows_path:
            path = ':'.join(windows_path.groups())
    if parsed_path.query:
        path = f'{path}?{parsed_path.query}'
    if parsed_path.fragment:
        path = f'{path}#{parsed_path.fragment}'
    options: Dict[str, str] = {'protocol': protocol, 'path': path}
    if parsed_path.netloc and protocol in CLOUD_PROTOCOLS:
        host_with_port: str = parsed_path.netloc.rsplit('@', 1)[-1]
        host: str = host_with_port.rsplit(':', 1)[0]
        options['path'] = host + options['path']
        if protocol in ['abfss', 'oci'] and parsed_path.username:
            options['path'] = f'{parsed_path.username}@' + options['path']
    return options

def get_protocol_and_path(filepath: str, version: Optional[Version] = None) -> Tuple[str, str]:
    """Parses filepath on protocol and path.

    .. warning::
        Versioning is not supported for HTTP protocols.

    Args:
        filepath: raw filepath e.g.: ``gcs://bucket/test.json``.
        version: instance of ``kedro.io.core.Version`` or None.

    Returns:
        Protocol and path.

    Raises:
        DatasetError: when protocol is http(s) and version is not None.
    """
    options_dict: Dict[str, str] = _parse_filepath(str(filepath))
    path: str = options_dict['path']
    protocol: str = options_dict['protocol']
    if protocol in HTTP_PROTOCOLS:
        if version is not None:
            raise DatasetError('Versioning is not supported for HTTP protocols. Please remove the `versioned` flag from the dataset configuration.')
        path = path.split(PROTOCOL_DELIMITER, 1)[-1]
    return (protocol, path)

def get_filepath_str(raw_path: PurePath, protocol: str) -> str:
    """Returns filepath. Returns full filepath (with protocol) if protocol is HTTP(s).

    Args:
        raw_path: filepath without protocol.
        protocol: protocol.

    Returns:
        Filepath string.
    """
    path: str = raw_path.as_posix()
    if protocol in HTTP_PROTOCOLS:
        path = ''.join((protocol, PROTOCOL_DELIMITER, path))
    return path

def validate_on_forbidden_chars(**kwargs: str) -> None:
    """Validate that string values do not include white-spaces or ;"""
    for key, value in kwargs.items():
        if ' ' in value or ';' in value:
            raise DatasetError(f"Neither white-space nor semicolon are allowed in '{key}'.")

_C = TypeVar("_C")

from typing import Protocol, runtime_checkable

@runtime_checkable
class CatalogProtocol(Protocol[_C]):

    def __contains__(self, ds_name: str) -> bool:
        """Check if a dataset is in the catalog."""
        ...

    @property
    def config_resolver(self) -> Any:
        """Return a copy of the datasets dictionary."""
        ...

    @classmethod
    def from_config(cls, catalog: Dict[str, Any]) -> CatalogProtocol[Any]:
        """Create a catalog instance from configuration."""
        ...

    def _get_dataset(self, dataset_name: str, version: Optional[Version] = None, suggest: bool = True) -> Any:
        """Retrieve a dataset by its name."""
        ...

    def list(self, regex_search: Optional[str] = None) -> Iterable[str]:
        """List all dataset names registered in the catalog."""
        ...

    def save(self, name: str, data: Any) -> None:
        """Save data to a registered dataset."""
        ...

    def load(self, name: str, version: Optional[Version] = None) -> Any:
        """Load data from a registered dataset."""
        ...

    def add(self, ds_name: str, dataset: Any, replace: bool = False) -> None:
        """Add a new dataset to the catalog."""
        ...

    def add_feed_dict(self, datasets: Dict[str, Any], replace: bool = False) -> None:
        """Add datasets to the catalog using the data provided through the `feed_dict`."""
        ...

    def exists(self, name: str) -> bool:
        """Checks whether registered dataset exists by calling its `exists()` method."""
        ...

    def release(self, name: str) -> None:
        """Release any cached data associated with a dataset."""
        ...

    def confirm(self, name: str) -> None:
        """Confirm a dataset by its name."""
        ...

    def shallow_copy(self, extra_dataset_patterns: Optional[Any] = None) -> CatalogProtocol[Any]:
        """Returns a shallow copy of the current object."""
        ...

def _validate_versions(
    datasets: Optional[Dict[str, Any]],
    load_versions: Dict[str, str],
    save_version: Optional[str],
) -> Tuple[Dict[str, str], Optional[str]]:
    """Validates and synchronises dataset versions for loading and saving.

    Ensures consistency of dataset versions across a catalog, particularly
    for versioned datasets. It updates load versions and validates that all
    save versions are consistent.

    Args:
        datasets: A dictionary mapping dataset names to their instances.
            if None, no validation occurs.
        load_versions: A mapping between dataset names and versions
            to load.
        save_version: Version string to be used for ``save`` operations
            by all datasets with versioning enabled.

    Returns:
        Updated ``load_versions`` with load versions specified in the ``datasets``
            and resolved ``save_version``.

    Raises:
        VersionAlreadyExistsError: If a dataset's save version conflicts with
            the catalog's save version.
    """
    if not datasets:
        return (load_versions, save_version)
    cur_load_versions: Dict[str, str] = load_versions.copy()
    cur_save_version: Optional[str] = save_version
    for ds_name, ds in datasets.items():
        cur_ds = ds._dataset if ds.__class__.__name__ == 'CachedDataset' else ds
        if isinstance(cur_ds, AbstractVersionedDataset) and cur_ds._version:
            if cur_ds._version.load:
                cur_load_versions[ds_name] = cur_ds._version.load  # type: ignore
            if cur_ds._version.save:
                cur_save_version = cur_save_version or cur_ds._version.save  # type: ignore
                if cur_save_version != cur_ds._version.save:
                    raise VersionAlreadyExistsError(
                        f'Cannot add a dataset `{ds_name}` with `{cur_ds._version.save}` save version. Save version set for the catalog is `{cur_save_version}`All datasets in the catalog must have the same save version.'
                    )
    return (cur_load_versions, cur_save_version)