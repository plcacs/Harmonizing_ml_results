from __future__ import annotations
import difflib
import logging
import re
from typing import Any, Iterator, List, Dict, Optional, Tuple, Union

from kedro.io.catalog_config_resolver import CatalogConfigResolver, Patterns
from kedro.io.core import (
    TYPE_KEY,
    AbstractDataset,
    AbstractVersionedDataset,
    CatalogProtocol,
    DatasetAlreadyExistsError,
    DatasetError,
    DatasetNotFoundError,
    Version,
    _validate_versions,
    generate_timestamp,
    parse_dataset_definition,
)
from kedro.io.memory_dataset import MemoryDataset, _is_memory_dataset
from kedro.utils import _format_rich, _has_rich_handler


class _LazyDataset:
    def __init__(
        self, 
        name: str, 
        config: Dict[str, Any], 
        load_version: Optional[Any] = None, 
        save_version: Optional[Any] = None
    ) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.load_version: Optional[Any] = load_version
        self.save_version: Optional[Any] = save_version

    def __repr__(self) -> str:
        class_type, _ = parse_dataset_definition(self.config)
        return f"{class_type.__module__}.{class_type.__qualname__}"

    def materialize(self) -> AbstractDataset:
        return AbstractDataset.from_config(self.name, self.config, self.load_version, self.save_version)


class KedroDataCatalog(CatalogProtocol):
    def __init__(
        self,
        datasets: Optional[Dict[str, AbstractDataset]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
        config_resolver: Optional[CatalogConfigResolver] = None,
        load_versions: Optional[Dict[str, Any]] = None,
        save_version: Optional[str] = None,
    ) -> None:
        """
        KedroDataCatalog stores instances of AbstractDataset implementations to provide load and save capabilities from
        anywhere in the program. To use a KedroDataCatalog, you need to instantiate it with a dictionary of datasets. Then
        it will act as a single point of reference for your calls, relaying load and save functions to the underlying
        datasets.

        Note: KedroDataCatalog is an experimental feature and is under active development. Therefore, it is possible we'll
        introduce breaking changes to this class, so be mindful of that if you decide to use it already.
        """
        self._config_resolver: CatalogConfigResolver = config_resolver or CatalogConfigResolver()
        self.__datasets: Dict[str, AbstractDataset] = datasets or {}
        self._lazy_datasets: Dict[str, _LazyDataset] = {}
        self._load_versions, self._save_version = _validate_versions(datasets, load_versions or {}, save_version)
        self._use_rich_markup: bool = _has_rich_handler()
        for ds_name, ds_config in self._config_resolver.config.items():
            self._add_from_config(ds_name, ds_config)
        raw_data = raw_data or {}
        for ds_name, data in raw_data.items():
            self[ds_name] = data

    @property
    def datasets(self) -> Dict[str, Any]:
        return self._lazy_datasets | self.__datasets

    @datasets.setter
    def datasets(self, value: Any) -> None:
        raise AttributeError("Operation not allowed. Please use KedroDataCatalog.add() instead.")

    def __getattribute__(self, key: str) -> Any:
        if key == "_datasets":
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
        return (
            dataset_name in self.__datasets
            or dataset_name in self._lazy_datasets
            or self._config_resolver.match_pattern(dataset_name) is not None
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KedroDataCatalog):
            return False
        return (
            (self.__datasets, self._lazy_datasets, self._config_resolver.list_patterns())
            == (other.__datasets, other._lazy_datasets, other.config_resolver.list_patterns())
        )

    def keys(self) -> List[str]:
        """List all dataset names registered in the catalog."""
        return list(self._lazy_datasets.keys()) + list(self.__datasets.keys())

    def values(self) -> List[Any]:
        """List all datasets registered in the catalog."""
        return [self.get(key) for key in self]

    def items(self) -> List[Tuple[str, Any]]:
        """List all dataset names and datasets registered in the catalog."""
        return [(key, self.get(key)) for key in self]

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def __getitem__(self, ds_name: str) -> AbstractDataset:
        """Get a dataset by name from an internal collection of datasets.

        If a dataset is not in the collection but matches any pattern
        it is instantiated and added to the collection first, then returned.
        """
        return self.get_dataset(ds_name)

    def __setitem__(self, key: str, value: Any) -> None:
        """Add dataset to the KedroDataCatalog using the given key as a dataset name
        and the provided data as the value.
        """
        if key in self.__datasets:
            self._logger.warning("Replacing dataset '%s'", key)
        if isinstance(value, AbstractDataset):
            self._load_versions, self._save_version = _validate_versions({key: value}, self._load_versions, self._save_version)
            self.__datasets[key] = value
        elif isinstance(value, _LazyDataset):
            self._lazy_datasets[key] = value
        else:
            self._logger.info(f"Adding input data as a MemoryDataset - {key}")
            self.__datasets[key] = MemoryDataset(data=value)

    def __len__(self) -> int:
        return len(self.keys())

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a dataset by name from an internal collection of datasets.

        If a dataset is not in the collection but matches any pattern
        it is instantiated and added to the collection first, then returned.
        """
        if key not in self.__datasets and key not in self._lazy_datasets:
            ds_config = self._config_resolver.resolve_pattern(key)
            if ds_config:
                self._add_from_config(key, ds_config)
        lazy_dataset: Optional[_LazyDataset] = self._lazy_datasets.pop(key, None)
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
        catalog: Optional[Dict[str, Any]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        load_versions: Optional[Dict[str, Any]] = None,
        save_version: Optional[str] = None,
    ) -> KedroDataCatalog:
        """Create a KedroDataCatalog instance from configuration."""
        catalog = catalog or {}
        config_resolver = CatalogConfigResolver(catalog, credentials)
        save_version = save_version or generate_timestamp()
        load_versions = load_versions or {}
        missing_keys = [
            ds_name
            for ds_name in load_versions
            if not (ds_name in config_resolver.config or config_resolver.match_pattern(ds_name))
        ]
        if missing_keys:
            raise DatasetNotFoundError(
                f"'load_versions' keys [{', '.join(sorted(missing_keys))}] are not found in the catalog."
            )
        return cls(load_versions=load_versions, save_version=save_version, config_resolver=config_resolver)

    def to_config(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[str]]:
        """Converts the KedroDataCatalog instance into a configuration format suitable for serialization."""
        catalog: Dict[str, Any] = {}
        credentials: Dict[str, Any] = {}
        load_versions: Dict[str, Any] = {}
        for ds_name, ds in self._lazy_datasets.items():
            if _is_memory_dataset(ds.config.get(TYPE_KEY, "")):
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
    def _validate_dataset_config(ds_name: str, ds_config: Any) -> None:
        if not isinstance(ds_config, dict):
            raise DatasetError(
                f"Catalog entry '{ds_name}' is not a valid dataset configuration. \nHint: If this catalog entry is intended for variable interpolation, make sure that the key is preceded by an underscore."
            )
        if "type" not in ds_config:
            raise DatasetError(
                f"An exception occurred when parsing config for dataset '{ds_name}':\n'type' is missing from dataset catalog configuration.\nHint: If this catalog entry is intended for variable interpolation, make sure that the top level key is preceded by an underscore."
            )

    def _add_from_config(self, ds_name: str, ds_config: Dict[str, Any]) -> None:
        """Create a LazyDataset instance and add it to the catalog."""
        self._validate_dataset_config(ds_name, ds_config)
        ds = _LazyDataset(ds_name, ds_config, self._load_versions.get(ds_name), self._save_version)
        self.add(ds_name, ds)

    def get_dataset(self, ds_name: str, version: Optional[Any] = None, suggest: bool = True) -> AbstractDataset:
        """Get a dataset by name from an internal collection of datasets.

        If a dataset is not in the collection but matches any pattern it is instantiated and added to the collection first, then returned.
        """
        dataset = self.get(ds_name)
        if dataset is None:
            error_msg = f"Dataset '{ds_name}' not found in the catalog"
            if suggest:
                matches = difflib.get_close_matches(ds_name, self.keys())
                if matches:
                    suggestions = ", ".join(matches)
                    error_msg += f" - did you mean one of these instead: {suggestions}"
            raise DatasetNotFoundError(error_msg)
        if version and isinstance(dataset, AbstractVersionedDataset):
            dataset = dataset._copy(_version=version)
        return dataset

    def _get_dataset(self, dataset_name: str, version: Optional[Any] = None, suggest: bool = True) -> AbstractDataset:
        return self.get_dataset(dataset_name, version, suggest)

    def add(self, ds_name: str, dataset: Union[AbstractDataset, _LazyDataset], replace: bool = False) -> None:
        """Adds a new AbstractDataset object to the KedroDataCatalog."""
        if (ds_name in self.__datasets or ds_name in self._lazy_datasets) and (not replace):
            raise DatasetAlreadyExistsError(f"Dataset '{ds_name}' has already been registered")
        self.__setitem__(ds_name, dataset)

    def filter(
        self,
        name_regex: Optional[Union[str, re.Pattern]] = None,
        type_regex: Optional[Union[str, re.Pattern]] = None,
        by_type: Optional[Union[type, List[type]]] = None,
    ) -> List[str]:
        """Filter dataset names registered in the catalog based on name and/or type."""
        filtered: List[str] = self.keys()
        if name_regex:
            filtered = [ds_name for ds_name in filtered if re.search(name_regex, ds_name)]
        by_type_set = set()
        if by_type:
            if not isinstance(by_type, list):
                by_type = [by_type]
            for _type in by_type:
                by_type_set.add(f"{_type.__module__}.{_type.__qualname__}")
        if by_type_set or type_regex:
            filtered_types: List[str] = []
            for ds_name in filtered:
                if ds_name in self._lazy_datasets:
                    str_type = str(self._lazy_datasets[ds_name])
                else:
                    class_type = type(self.__datasets[ds_name])
                    str_type = f"{class_type.__module__}.{class_type.__qualname__}"
                if (not type_regex or re.search(type_regex, str_type)) and (not by_type_set or str_type in by_type_set):
                    filtered_types.append(ds_name)
            return filtered_types
        return filtered

    def list(self, regex_search: Optional[str] = None, regex_flags: int = 0) -> List[str]:
        """List all dataset names registered in the catalog, optionally filtered by a regex pattern."""
        if regex_search is None:
            return self.keys()
        if regex_search == "":
            self._logger.warning("The empty string will not match any datasets")
            return []
        if not regex_flags:
            regex_flags = re.IGNORECASE
        try:
            pattern = re.compile(regex_search, flags=regex_flags)
        except re.error as exc:
            raise SyntaxError(f"Invalid regular expression provided: '{regex_search}'") from exc
        return [ds_name for ds_name in self.__iter__() if pattern.search(ds_name)]

    def save(self, name: str, data: Any) -> None:
        """Save data to a registered dataset."""
        dataset = self.get_dataset(name)
        self._logger.info(
            "Saving data to %s (%s)...",
            _format_rich(name, "dark_orange") if self._use_rich_markup else name,
            type(dataset).__name__,
            extra={"markup": True},
        )
        dataset.save(data)

    def load(self, name: str, version: Optional[Any] = None) -> Any:
        """Loads a registered dataset."""
        load_version = Version(version, None) if version else None
        dataset = self.get_dataset(name, version=load_version)
        self._logger.info(
            "Loading data from %s (%s)...",
            _format_rich(name, "dark_orange") if self._use_rich_markup else name,
            type(dataset).__name__,
            extra={"markup": True},
        )
        return dataset.load()

    def release(self, name: str) -> None:
        """Release any cached data associated with a dataset."""
        dataset = self.get_dataset(name)
        dataset.release()

    def confirm(self, name: str) -> None:
        """Confirm a dataset by its name."""
        self._logger.info("Confirming dataset '%s'", name)
        dataset = self.get_dataset(name)
        if hasattr(dataset, "confirm"):
            dataset.confirm()
        else:
            raise DatasetError(f"Dataset '{name}' does not have 'confirm' method")

    def add_feed_dict(self, feed_dict: Dict[str, Any], replace: bool = False) -> None:
        for ds_name, ds_data in feed_dict.items():
            self.add(ds_name, MemoryDataset(data=ds_data), replace)

    def shallow_copy(self, extra_dataset_patterns: Optional[Dict[str, Any]] = None) -> KedroDataCatalog:
        """Returns a shallow copy of the current object."""
        if extra_dataset_patterns:
            self._config_resolver.add_runtime_patterns(extra_dataset_patterns)
        return self

    def exists(self, name: str) -> bool:
        """Checks whether registered dataset exists by calling its exists() method."""
        try:
            dataset = self._get_dataset(name)
        except DatasetNotFoundError:
            return False
        return dataset.exists()