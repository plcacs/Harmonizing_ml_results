from __future__ import annotations
import difflib
import logging
import pprint
import re
from typing import Any, Optional, Dict, List, Union, Mapping

from kedro.io.catalog_config_resolver import CREDENTIALS_KEY, CatalogConfigResolver, Patterns
from kedro.io.core import (
    AbstractDataset,
    AbstractVersionedDataset,
    DatasetAlreadyExistsError,
    DatasetError,
    DatasetNotFoundError,
    Version,
    _validate_versions,
    generate_timestamp,
)
from kedro.io.memory_dataset import MemoryDataset
from kedro.utils import _format_rich, _has_rich_handler

CATALOG_KEY = "catalog"
WORDS_REGEX_PATTERN = re.compile(r"\W+")


def _sub_nonword_chars(dataset_name: str) -> str:
    return re.sub(WORDS_REGEX_PATTERN, "__", dataset_name)


class _FrozenDatasets:
    def __init__(
        self, *datasets_collections: Optional[Union[_FrozenDatasets, Dict[str, AbstractDataset]]]
    ) -> None:
        self._original_names: Dict[str, str] = {}
        for collection in datasets_collections:
            if collection is None:
                continue
            if isinstance(collection, _FrozenDatasets):
                self.__dict__.update(collection.__dict__)
                self._original_names.update(collection._original_names)
            else:
                for dataset_name, dataset in collection.items():
                    self.__dict__[_sub_nonword_chars(dataset_name)] = dataset
                    self._original_names[dataset_name] = ""

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_original_names":
            super().__setattr__(key, value)
            return
        msg = "Operation not allowed. "
        if key in self.__dict__:
            msg += "Please change datasets through configuration."
        else:
            msg += "Please use DataCatalog.add() instead."
        raise AttributeError(msg)

    def _ipython_key_completions_(self) -> List[str]:
        return list(self._original_names.keys())

    def __getitem__(self, key: str) -> AbstractDataset:
        return self.__dict__[_sub_nonword_chars(key)]

    def __repr__(self) -> str:
        datasets_repr: Dict[str, str] = {}
        for ds_name in self._original_names.keys():
            datasets_repr[ds_name] = self.__dict__[_sub_nonword_chars(ds_name)].__repr__()
        return pprint.pformat(datasets_repr, sort_dicts=False)


class DataCatalog:
    def __init__(
        self,
        datasets: Optional[Dict[str, AbstractDataset]] = None,
        feed_dict: Optional[Dict[str, Any]] = None,
        dataset_patterns: Optional[Dict[str, Any]] = None,
        load_versions: Optional[Dict[str, Any]] = None,
        save_version: Optional[str] = None,
        default_pattern: Optional[Dict[str, Any]] = None,
        config_resolver: Optional[CatalogConfigResolver] = None,
    ) -> None:
        self._config_resolver: CatalogConfigResolver = config_resolver or CatalogConfigResolver()
        if not config_resolver:
            self._config_resolver._dataset_patterns = dataset_patterns or {}
            self._config_resolver._default_pattern = default_pattern or {}
        self._load_versions, self._save_version = _validate_versions(
            datasets, load_versions or {}, save_version
        )
        self._datasets: Dict[str, AbstractDataset] = {}
        self.datasets: Optional[_FrozenDatasets] = None
        self.add_all(datasets or {})
        self._use_rich_markup: bool = _has_rich_handler()
        if feed_dict:
            self.add_feed_dict(feed_dict)

    def __repr__(self) -> str:
        return self.datasets.__repr__()  # type: ignore

    def __contains__(self, dataset_name: str) -> bool:
        return dataset_name in self._datasets or self._config_resolver.match_pattern(dataset_name) is not None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataCatalog):
            return NotImplemented
        return (self._datasets, self._config_resolver.list_patterns()) == (
            other._datasets,
            other.config_resolver.list_patterns(),
        )

    @property
    def config_resolver(self) -> CatalogConfigResolver:
        return self._config_resolver

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
    ) -> DataCatalog:
        catalog = catalog or {}
        datasets: Dict[str, AbstractDataset] = {}
        config_resolver = CatalogConfigResolver(catalog, credentials)
        save_version = save_version or generate_timestamp()
        load_versions = load_versions or {}
        for ds_name in catalog:
            if not config_resolver.is_pattern(ds_name):
                datasets[ds_name] = AbstractDataset.from_config(
                    ds_name,
                    config_resolver.config.get(ds_name, {}),
                    load_versions.get(ds_name),
                    save_version,
                )
        missing_keys = [
            ds_name
            for ds_name in load_versions
            if not (ds_name in config_resolver.config or config_resolver.match_pattern(ds_name))
        ]
        if missing_keys:
            raise DatasetNotFoundError(
                f"'load_versions' keys [{', '.join(sorted(missing_keys))}] are not found in the catalog."
            )
        return cls(
            datasets=datasets,
            dataset_patterns=config_resolver._dataset_patterns,
            load_versions=load_versions,
            save_version=save_version,
            default_pattern=config_resolver._default_pattern,
            config_resolver=config_resolver,
        )

    def _get_dataset(
        self, dataset_name: str, version: Optional[Version] = None, suggest: bool = True
    ) -> AbstractDataset:
        ds_config = self._config_resolver.resolve_pattern(dataset_name)
        if dataset_name not in self._datasets and ds_config:
            ds = AbstractDataset.from_config(
                dataset_name, ds_config, self._load_versions.get(dataset_name), self._save_version
            )
            self.add(dataset_name, ds)
        if dataset_name not in self._datasets:
            error_msg = f"Dataset '{dataset_name}' not found in the catalog"
            if suggest:
                matches = difflib.get_close_matches(dataset_name, self._datasets.keys())
                if matches:
                    suggestions = ", ".join(matches)
                    error_msg += f" - did you mean one of these instead: {suggestions}"
            raise DatasetNotFoundError(error_msg)
        dataset = self._datasets[dataset_name]
        if version and isinstance(dataset, AbstractVersionedDataset):
            dataset = dataset._copy(_version=version)
        return dataset

    def load(self, name: str, version: Optional[str] = None) -> Any:
        load_version: Optional[Version] = Version(version, None) if version else None
        dataset = self._get_dataset(name, version=load_version)
        self._logger.info(
            "Loading data from %s (%s)...",
            _format_rich(name, "dark_orange") if self._use_rich_markup else name,
            type(dataset).__name__,
            extra={"markup": True},
        )
        result = dataset.load()
        return result

    def save(self, name: str, data: Any) -> None:
        dataset = self._get_dataset(name)
        self._logger.info(
            "Saving data to %s (%s)...",
            _format_rich(name, "dark_orange") if self._use_rich_markup else name,
            type(dataset).__name__,
            extra={"markup": True},
        )
        dataset.save(data)

    def exists(self, name: str) -> bool:
        try:
            dataset = self._get_dataset(name)
        except DatasetNotFoundError:
            return False
        return dataset.exists()

    def release(self, name: str) -> None:
        dataset = self._get_dataset(name)
        dataset.release()

    def add(self, dataset_name: str, dataset: AbstractDataset, replace: bool = False) -> None:
        if dataset_name in self._datasets:
            if replace:
                self._logger.warning("Replacing dataset '%s'", dataset_name)
            else:
                raise DatasetAlreadyExistsError(f"Dataset '{dataset_name}' has already been registered")
        self._load_versions, self._save_version = _validate_versions({dataset_name: dataset}, self._load_versions, self._save_version)
        self._datasets[dataset_name] = dataset
        self.datasets = _FrozenDatasets(self.datasets, {dataset_name: dataset})

    def add_all(self, datasets: Dict[str, AbstractDataset], replace: bool = False) -> None:
        for ds_name, ds in datasets.items():
            self.add(ds_name, ds, replace)

    def add_feed_dict(self, feed_dict: Dict[str, Any], replace: bool = False) -> None:
        for ds_name, ds_data in feed_dict.items():
            dataset = ds_data if isinstance(ds_data, AbstractDataset) else MemoryDataset(data=ds_data)
            self.add(ds_name, dataset, replace)

    def list(self, regex_search: Optional[str] = None) -> List[str]:
        if regex_search is None:
            return list(self._datasets.keys())
        if not regex_search.strip():
            self._logger.warning("The empty string will not match any datasets")
            return []
        try:
            pattern = re.compile(regex_search, flags=re.IGNORECASE)
        except re.error as exc:
            raise SyntaxError(f"Invalid regular expression provided: '{regex_search}'") from exc
        return [ds_name for ds_name in self._datasets if pattern.search(ds_name)]

    def shallow_copy(self, extra_dataset_patterns: Optional[Dict[str, Any]] = None) -> DataCatalog:
        if extra_dataset_patterns:
            self._config_resolver.add_runtime_patterns(extra_dataset_patterns)
        return self.__class__(
            datasets=self._datasets,
            dataset_patterns=self._config_resolver._dataset_patterns,
            default_pattern=self._config_resolver._default_pattern,
            load_versions=self._load_versions,
            save_version=self._save_version,
            config_resolver=self._config_resolver,
        )

    def confirm(self, name: str) -> None:
        self._logger.info("Confirming dataset '%s'", name)
        dataset = self._get_dataset(name)
        if hasattr(dataset, "confirm"):
            dataset.confirm()
        else:
            raise DatasetError(f"Dataset '{name}' does not have 'confirm' method")