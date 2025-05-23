from __future__ import annotations
import difflib
import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from kedro.io.catalog_config_resolver import CatalogConfigResolver, Patterns
from kedro.io.core import TYPE_KEY, AbstractDataset, AbstractVersionedDataset, CatalogProtocol, DatasetAlreadyExistsError, DatasetError, DatasetNotFoundError, Version, _validate_versions, generate_timestamp, parse_dataset_definition
from kedro.io.memory_dataset import MemoryDataset, _is_memory_dataset
from kedro.utils import _format_rich, _has_rich_handler

class _LazyDataset:
    """A helper class to store AbstractDataset configuration and materialize dataset object."""

    def __init__(self, name: str, config: Dict[str, Any], load_version: Optional[str] = None, save_version: Optional[str] = None):
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

    def __init__(self, datasets: Optional[Dict[str, AbstractDataset]] = None, raw_data: Optional[Dict[str, Any]] = None, config_resolver: Optional[CatalogConfigResolver] = None, load_versions: Optional[Dict[str, str]] = None, save_version: Optional[str] = None):
        self._config_resolver = config_resolver or CatalogConfigResolver()
        self.__datasets: Dict[str, AbstractDataset] = datasets or {}
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
    def datasets(self, value: Any) -> None:
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
        return dataset_name in self.__datasets or dataset_name in self._lazy_datasets or self._config_resolver.match_pattern(dataset_name) is not None

    def __eq__(self, other: Any) -> bool:
        return (self.__datasets, self._lazy_datasets, self._config_resolver.list_patterns()) == (other.__datasets, other._lazy_datasets, other.config_resolver.list_patterns())

    def keys(self) -> List[str]:
        return list(self._lazy_datasets.keys()) + list(self.__datasets.keys())

    def values(self) -> List[AbstractDataset]:
        return [self.get(key) for key in self]

    def items(self) -> List[Tuple[str, AbstractDataset]]:
        return [(key, self.get(key)) for key in self]

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def __getitem__(self, ds_name: str) -> AbstractDataset:
        return self.get_dataset(ds_name)

    def __setitem__(self, key: str, value: Union[AbstractDataset, _LazyDataset, Any]) -> None:
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
    def from_config(cls, catalog: Dict[str, Any], credentials: Optional[Dict[str, Any]] = None, load_versions: Optional[Dict[str, str]] = None, save_version: Optional[str] = None) -> KedroDataCatalog:
        catalog = catalog or {}
        config_resolver = CatalogConfigResolver(catalog, credentials)
        save_version = save_version or generate_timestamp()
        load_versions = load_versions or {}
        missing_keys = [ds_name for ds_name in load_versions if not (ds_name in config_resolver.config or config_resolver.match_pattern(ds_name))]
        if missing_keys:
            raise DatasetNotFoundError(f"'load_versions' keys [{', '.join(sorted(missing_keys))}] are not found in the catalog.")
        return cls(load_versions=load_versions, save_version=save_version, config_resolver=config_resolver)

    def to_config(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Optional[str]], Optional[str]]:
        catalog = {}
        credentials = {}
        load_versions = {}
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
        self._validate_dataset_config(ds_name, ds_config)
        ds = _LazyDataset(ds_name, ds_config, self._load_versions.get(ds_name), self._save_version)
        self.add(ds_name, ds)

    def get_dataset(self, ds_name: str, version: Optional[Version] = None, suggest: bool = True) -> AbstractDataset:
        dataset = self.get(ds_name)
        if dataset is None:
            error_msg = f"Dataset '{ds_name}' not found in the catalog"
            if suggest:
                matches = difflib.get_close_matches(ds_name, self.keys())
                if matches:
                    suggestions = ', '.join(matches)
                    error_msg += f' - did you mean one of these instead: {suggestions}'
            raise DatasetNotFoundError(error_msg)
        if version and isinstance(dataset, AbstractVersionedDataset):
            dataset = dataset._copy(_version=version)
        return dataset

    def _get_dataset(self, dataset_name: str, version: Optional[Version] = None, suggest: bool = True) -> AbstractDataset:
        return self.get_dataset(dataset_name, version, suggest)

    def add(self, ds_name: str, dataset: Union[AbstractDataset, _LazyDataset], replace: bool = False) -> None:
        if (ds_name in self.__datasets or ds_name in self._lazy_datasets) and (not replace):
            raise DatasetAlreadyExistsError(f"Dataset '{ds_name}' has already been registered")
        self.__setitem__(ds_name, dataset)

    def filter(self, name_regex: Optional[Union[str, re.Pattern]] = None, type_regex: Optional[Union[str, re.Pattern]] = None, by_type: Optional[Union[type, List[type]]] = None) -> List[str]:
        filtered = self.keys()
        if name_regex:
            filtered = [ds_name for ds_name in filtered if re.search(name_regex, ds_name)]
        by_type_set = set()
        if by_type:
            if not isinstance(by_type, list):
                by_type = [by_type]
            for _type in by_type:
                by_type_set.add(f'{_type.__module__}.{_type.__qualname__}')
        if by_type_set or type_regex:
            filtered_types = []
            for ds_name in filtered:
                if ds_name in self._lazy_datasets:
                    str_type = str(self._lazy_datasets[ds_name])
                else:
                    class_type = type(self.__datasets[ds_name])
                    str_type = f'{class_type.__module__}.{class_type.__qualname__}'
                if (not type_regex or re.search(type_regex, str_type)) and (not by_type_set or str_type in by_type_set):
                    filtered_types.append(ds_name)
            return filtered_types
        return filtered

    def list(self, regex_search: Optional[str] = None, regex_flags: int = 0) -> List[str]:
        if regex_search is None:
            return self.keys()
        if regex_search == '':
            self._logger.warning('The empty string will not match any datasets')
            return []
        if not regex_flags:
            regex_flags = re.IGNORECASE
        try:
            pattern = re.compile(regex_search, flags=regex_flags)
        except re.error as exc:
            raise SyntaxError(f"Invalid regular expression provided: '{regex_search}'") from exc
        return [ds_name for ds_name in self.__iter__() if pattern.search(ds_name)]

    def save(self, name: str, data: Any) -> None:
        dataset = self.get_dataset(name)
        self._logger.info('Saving data to %s (%s)...', _format_rich(name, 'dark_orange') if self._use_rich_markup else name, type(dataset).__name__, extra={'markup': True})
        dataset.save(data)

    def load(self, name: str, version: Optional[str] = None) -> Any:
        load_version = Version(version, None) if version else None
        dataset = self.get_dataset(name, version=load_version)
        self._logger.info('Loading data from %s (%s)...', _format_rich(name, 'dark_orange') if self._use_rich_markup else name, type(dataset).__name__, extra={'markup': True})
        return dataset.load()

    def release(self, name: str) -> None:
        dataset = self.get_dataset(name)
        dataset.release()

    def confirm(self, name: str) -> None:
        self._logger.info("Confirming dataset '%s'", name)
        dataset = self.get_dataset(name)
        if hasattr(dataset, 'confirm'):
            dataset.confirm()
        else:
            raise DatasetError(f"Dataset '{name}' does not have 'confirm' method")

    def add_feed_dict(self, feed_dict: Dict[str, Any], replace: bool = False) -> None:
        for ds_name, ds_data in feed_dict.items():
            self.add(ds_name, MemoryDataset(data=ds_data), replace)

    def shallow_copy(self, extra_dataset_patterns: Optional[Dict[str, Any]] = None) -> KedroDataCatalog:
        if extra_dataset_patterns:
            self._config_resolver.add_runtime_patterns(extra_dataset_patterns)
        return self

    def exists(self, name: str) -> bool:
        try:
            dataset = self._get_dataset(name)
        except DatasetNotFoundError:
            return False
        return dataset.exists()
