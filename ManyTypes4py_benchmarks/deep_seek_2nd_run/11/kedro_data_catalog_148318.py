"""``KedroDataCatalog`` stores instances of ``AbstractDataset`` implementations to
provide ``load`` and ``save`` capabilities from anywhere in the program. To
use a ``KedroDataCatalog``, you need to instantiate it with a dictionary of datasets.
Then it will act as a single point of reference for your calls, relaying load and
save functions to the underlying datasets.

``KedroDataCatalog`` is an experimental feature aimed to replace ``DataCatalog`` in the future.
Expect possible breaking changes while using it.
"""
from __future__ import annotations
import difflib
import logging
import re
from typing import Any, Iterator, List, Dict, Optional, Union, Set, Tuple, Type, TypeVar, overload
from kedro.io.catalog_config_resolver import CatalogConfigResolver, Patterns
from kedro.io.core import TYPE_KEY, AbstractDataset, AbstractVersionedDataset, CatalogProtocol, DatasetAlreadyExistsError, DatasetError, DatasetNotFoundError, Version, _validate_versions, generate_timestamp, parse_dataset_definition
from kedro.io.memory_dataset import MemoryDataset, _is_memory_dataset
from kedro.utils import _format_rich, _has_rich_handler

T = TypeVar('T', bound=AbstractDataset)

class _LazyDataset:
    """A helper class to store AbstractDataset configuration and materialize dataset object."""

    def __init__(self, name: str, config: Dict[str, Any], load_version: Optional[str] = None, save_version: Optional[str] = None) -> None:
        self.name = name
        self.config = config
        self.load_version = load_version
        self.save_version = save_version

    def __repr__(self) -> str:
        class_type, _ = parse_dataset_definition(self.config)
        return f'{class_type.__module__}.{class_type.__qualname__}'

    def materialize(self) -> AbstractDataset:
        return AbstractDataset.from_config(self.name, self.config, self.load_version, self.save_version)

class KedroDataCatalog(CatalogProtocol):

    def __init__(
        self,
        datasets: Optional[Dict[str, AbstractDataset]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
        config_resolver: Optional[CatalogConfigResolver] = None,
        load_versions: Optional[Dict[str, str]] = None,
        save_version: Optional[str] = None
    ) -> None:
        """``KedroDataCatalog`` stores instances of ``AbstractDataset``
        implementations to provide ``load`` and ``save`` capabilities from
        anywhere in the program. To use a ``KedroDataCatalog``, you need to
        instantiate it with a dictionary of datasets. Then it will act as a
        single point of reference for your calls, relaying load and save
        functions to the underlying datasets.

        Note: ``KedroDataCatalog`` is an experimental feature and is under active development. Therefore, it is possible we'll introduce breaking changes to this class, so be mindful of that if you decide to use it already.

        Args:
            datasets: A dictionary of dataset names and dataset instances.
            raw_data: A dictionary with data to be added in memory as `MemoryDataset`` instances.
                Keys represent dataset names and the values are raw data.
            config_resolver: An instance of CatalogConfigResolver to resolve dataset patterns and configurations.
            load_versions: A mapping between dataset names and versions
                to load. Has no effect on datasets without enabled versioning.
            save_version: Version string to be used for ``save`` operations
                by all datasets with enabled versioning. It must: a) be a
                case-insensitive string that conforms with operating system
                filename limitations, b) always return the latest version when
                sorted in lexicographical order.

        Example:
        ::
            >>> from kedro_datasets.pandas import CSVDataset
            >>>
            >>> cars = CSVDataset(filepath="cars.csv",
            >>>                   load_args=None,
            >>>                   save_args={"index": False})
            >>> catalog = KedroDataCatalog(datasets={"cars": cars})
        """
        self._config_resolver = config_resolver or CatalogConfigResolver()
        self.__datasets = datasets or {}
        self._lazy_datasets: Dict[str, _LazyDataset] = {}
        self._load_versions, self._save_version = _validate_versions(datasets, load_versions or {}, save_version)
        self._use_rich_markup = _has_rich_handler()
        for ds_name, ds_config in self._config_resolver.config.items():
            self._add_from_config(ds_name, ds_config)
        raw_data = raw_data or {}
        for ds_name, data in raw_data.items():
            self[ds_name] = data

    @property
    def datasets(self) -> Dict[str, Union[AbstractDataset, _LazyDataset]]:
        return self._lazy_datasets | self.__datasets

    @datasets.setter
    def datasets(self, value: Dict[str, Union[AbstractDataset, _LazyDataset]]) -> None:
        raise AttributeError('Operation not allowed. Please use KedroDataCatalog.add() instead.')

    def __getattribute__(self, key: str) -> Any:
        if key == '_datasets':
            return self.datasets
        else:
            return super().__getattribute__(key)

    @property
    def config_resolver(self) -> CatalogConfigResolver:
        return self._config_resolver

    def __repr__(self) -> str:
        return repr(self._lazy_datasets | self.__datasets)

    def __contains__(self, dataset_name: str) -> bool:
        """Check if an item is in the catalog as a materialised dataset or pattern."""
        return dataset_name in self.__datasets or dataset_name in self._lazy_datasets or self._config_resolver.match_pattern(dataset_name) is not None

    def __eq__(self, other: object) -> bool:
        """Compares two catalogs based on materialised datasets and datasets patterns."""
        if not isinstance(other, KedroDataCatalog):
            return False
        return (self.__datasets, self._lazy_datasets, self._config_resolver.list_patterns()) == (other.__datasets, other._lazy_datasets, other.config_resolver.list_patterns())

    def keys(self) -> List[str]:
        """List all dataset names registered in the catalog."""
        return list(self._lazy_datasets.keys()) + list(self.__datasets.keys())

    def values(self) -> List[AbstractDataset]:
        """List all datasets registered in the catalog."""
        return [self.get(key) for key in self]

    def items(self) -> List[Tuple[str, AbstractDataset]]:
        """List all dataset names and datasets registered in the catalog."""
        return [(key, self.get(key)) for key in self]

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def __getitem__(self, ds_name: str) -> AbstractDataset:
        """Get a dataset by name from an internal collection of datasets.

        If a dataset is not in the collection but matches any pattern
        it is instantiated and added to the collection first, then returned.

        Args:
            ds_name: A dataset name.

        Returns:
            An instance of AbstractDataset.

        Raises:
            DatasetNotFoundError: When a dataset with the given name
                is not in the collection and does not match patterns.
        """
        return self.get_dataset(ds_name)

    def __setitem__(self, key: str, value: Union[AbstractDataset, _LazyDataset, Any]) -> None:
        """Add dataset to the ``KedroDataCatalog`` using the given key as a datsets name
        and the provided data as the value.

        The value can either be raw data or a Kedro dataset (i.e., an instance of a class
        inheriting from ``AbstractDataset``). If raw data is provided, it will be automatically
        wrapped in a ``MemoryDataset`` before being added to the catalog.

        Args:
            key: Name of the dataset.
            value: Raw data or an instance of a class inheriting from ``AbstractDataset``.

        Example:
        ::

            >>> from kedro_datasets.pandas import CSVDataset
            >>> import pandas as pd
            >>>
            >>> df = pd.DataFrame({"col1": [1, 2],
            >>>                    "col2": [4, 5],
            >>>                    "col3": [5, 6]})
            >>>
            >>> catalog = KedroDataCatalog()
            >>> catalog["data_df"] = df  # Add raw data as a MemoryDataset
            >>>
            >>> assert catalog.load("data_df").equals(df)
            >>>
            >>> csv_dataset = CSVDataset(filepath="test.csv")
            >>> csv_dataset.save(df)
            >>> catalog["data_csv_dataset"] = csv_dataset  # Add a dataset instance
            >>>
            >>> assert catalog.load("data_csv_dataset").equals(df)
        """
        if key in self.__datasets:
            self._logger.warning("Replacing dataset '%s'", key)
        if isinstance(value, AbstractDataset):
            self._load_versions, self._save_version = _validate_versions({key: value}, self._load_versions, self._save_version)
            self.__datasets[key] = value
        elif isinstance(value, _LazyDataset):
            self._lazy_datasets[key] = value
        else:
            self._logger.info(f'Adding input data as a MemoryDataset - {key}')
            self.__datasets[key] = MemoryDataset(data=value)

    def __len__(self) -> int:
        return len(self.keys())

    def get(self, key: str, default: Optional[AbstractDataset] = None) -> Optional[AbstractDataset]:
        """Get a dataset by name from an internal collection of datasets.

        If a dataset is not in the collection but matches any pattern
        it is instantiated and added to the collection first, then returned.

        Args:
            key: A dataset name.
            default: Optional argument for default dataset to return in case
                requested dataset not in the catalog.

        Returns:
            An instance of AbstractDataset.
        """
        if key not in self.__datasets and key not in self._lazy_datasets:
            ds_config = self._config_resolver.resolve_pattern(key)
            if ds_config:
                self._add_from_config(key, ds_config)
        lazy_dataset = self._lazy_datasets.pop(key, None)
        if lazy_dataset:
            self[key] = lazy_dataset.materialize()
        dataset = self.__datasets.get(key, None)
        return dataset or default

    def _ipython_key_completions_(self) -> List[str]:
        return self.keys()

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    @classmethod
    def from_config(
        cls,
        catalog: Optional[Dict[str, Dict[str, Any]]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        load_versions: Optional[Dict[str, str]] = None,
        save_version: Optional[str] = None
    ) -> KedroDataCatalog:
        """Create a ``KedroDataCatalog`` instance from configuration. This is a
        factory method used to provide developers with a way to instantiate
        ``KedroDataCatalog`` with configuration parsed from configuration files.

        Args:
            catalog: A dictionary whose keys are the dataset names and
                the values are dictionaries with the constructor arguments
                for classes implementing ``AbstractDataset``. The dataset
                class to be loaded is specified with the key ``type`` and their
                fully qualified class name. All ``kedro.io`` dataset can be
                specified by their class name only, i.e. their module name
                can be omitted.
            credentials: A dictionary containing credentials for different
                datasets. Use the ``credentials`` key in a ``AbstractDataset``
                to refer to the appropriate credentials as shown in the example
                below.
            load_versions: A mapping between dataset names and versions
                to load. Has no effect on datasets without enabled versioning.
            save_version: Version string to be used for ``save`` operations
                by all datasets with enabled versioning. It must: a) be a
                case-insensitive string that conforms with operating system
                filename limitations, b) always return the latest version when
                sorted in lexicographical order.

        Returns:
            An instantiated ``KedroDataCatalog`` containing all specified
            datasets, created and ready to use.

        Raises:
            DatasetNotFoundError: When `load_versions` refers to a dataset that doesn't
                exist in the catalog.

        Example:
        ::

            >>> config = {
            >>>     "cars": {
            >>>         "type": "pandas.CSVDataset",
            >>>         "filepath": "cars.csv",
            >>>         "save_args": {
            >>>             "index": False
            >>>         }
            >>>     },
            >>>     "boats": {
            >>>         "type": "pandas.CSVDataset",
            >>>         "filepath": "s3://aws-bucket-name/boats.csv",
            >>>         "credentials": "boats_credentials",
            >>>         "save_args": {
            >>>             "index": False
            >>>         }
            >>>     }
            >>> }
            >>>
            >>> credentials = {
            >>>     "boats_credentials": {
            >>>         "client_kwargs": {
            >>>             "aws_access_key_id": "<your key id>",
            >>>             "aws_secret_access_key": "<your secret>"
            >>>         }
            >>>      }
            >>> }
            >>>
            >>> catalog = KedroDataCatalog.from_config(config, credentials)
            >>>
            >>> df = catalog.load("cars")
            >>> catalog.save("boats", df)
        """
        catalog = catalog or {}
        config_resolver = CatalogConfigResolver(catalog, credentials)
        save_version = save_version or generate_timestamp()
        load_versions = load_versions or {}
        missing_keys = [ds_name for ds_name in load_versions if not (ds_name in config_resolver.config or config_resolver.match_pattern(ds_name))]
        if missing_keys:
            raise DatasetNotFoundError(f"'load_versions' keys [{', '.join(sorted(missing_keys))}] are not found in the catalog.")
        return cls(load_versions=load_versions, save_version=save_version, config_resolver=config_resolver)

    def to_config(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Optional[str]], Optional[str]]:
        """Converts the `KedroDataCatalog` instance into a configuration format suitable for
        serialization. This includes datasets, credentials, and versioning information.

        This method is only applicable to catalogs that contain datasets initialized with static, primitive
        parameters. For example, it will work fine if one passes credentials as dictionary to
        `GBQQueryDataset` but not as `google.auth.credentials.Credentials` object. See
        https://github.com/kedro-org/kedro-plugins/issues/950 for the details.

        Returns:
            A tuple containing:
                catalog: A dictionary mapping dataset names to their unresolved configurations,
                    excluding in-memory datasets.
                credentials: A dictionary of unresolved credentials extracted from dataset configurations.
                load_versions: A dictionary mapping dataset names to specific versions to be loaded,
                    or `None` if no version is set.
                save_version: A global version identifier for saving datasets, or `None` if not specified.
        Example:
        ::

            >>> from kedro.io import KedroDataCatalog
            >>> from kedro_datasets.pandas import CSVDataset
            >>>
            >>> cars = CSVDataset(
            >>>     filepath="cars.csv",
            >>>     load_args=None,
            >>>     save_args={"index": False}
            >>> )
            >>> catalog = KedroDataCatalog(datasets={'cars': cars})
            >>>
            >>> config, credentials, load_versions, save_version = catalog.to_config()
            >>>
            >>> new_catalog = KedroDataCatalog.from_config(config, credentials, load_versions, save_version)
        """
        catalog: Dict[str, Dict[str, Any]] = {}
        credentials: Dict[str, Any] = {}
        load_versions: Dict[str, Optional[str]] = {}
        for ds_name, ds in self._lazy_datasets.items():
            if _is_memory_dataset(ds.config.get(TYPE_KEY, '')):
                continue
            unresolved_config, unresolved_credentials = self._config_resolver.unresolve_credentials(ds_name, ds.config)
            catalog[ds_name] = unresolved_config
            credentials.update(unresolved_credentials)
            load_versions[ds_name] = self._load_versions.get(ds_name, None)
        for ds_name, ds in self.__datasets.items():
            if _is_memory_dataset(ds):
                continue
            resolved_config = ds.to_config()
            unresolved_config, unresolved_credentials = self._config_resolver.unresolve_credentials(ds_name, resolved_config)
            catalog[ds_name] = unresolved_config
            credentials.update(unresolved_credentials)
            load_versions[ds_name] = self._load_versions.get(ds_name, None)
        return (catalog, credentials, load_versions, self._save_version)

    @staticmethod
    def _validate_dataset_config(ds_name: str, ds_config: Dict[str, Any]) -> None:
        if not isinstance(ds_config, dict):
            raise DatasetError(f"Catalog entry '{ds_name}' is not a valid dataset configuration. \nHint: If this catalog entry is intended for variable interpolation, make sure that the key is preceded by an underscore.")
        if 'type' not in ds_config:
            raise DatasetError(f"An exception occurred when parsing config for dataset '{ds_name}':\n'type' is missing from dataset catalog configuration.\nHint: If this catalog entry is intended for variable interpolation, make sure that the top level key is preceded by an underscore.")

    def _add_from_config(self, ds_name: str, ds_config: Dict[str, Any]) -> None:
        """Create a LazyDataset instance and add it to the catalog.

        Args:
            ds_name: A dataset name.
            ds_config: A dataset configuration.

        Raises:
            DatasetError: When a dataset configuration provided is not valid.
        """
        self._validate_dataset_config(ds_name, ds_config)
        ds = _LazyDataset(ds_name, ds_config, self._load_versions.get(ds_name), self._save_version)
        self.add(ds_name, ds)

    def get_dataset(self, ds_name: str, version: Optional[Union[str, Version]] = None, suggest: bool = True) -> AbstractDataset:
        """Get a dataset by name from an internal collection of datasets.

       