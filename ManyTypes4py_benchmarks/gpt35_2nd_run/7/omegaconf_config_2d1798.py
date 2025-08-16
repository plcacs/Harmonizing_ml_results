from __future__ import annotations
import logging
import mimetypes
import typing
from collections.abc import Iterable, KeysView
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError, UnsupportedInterpolationType
from omegaconf.resolvers import oc
from yaml.parser import ParserError
from yaml.scanner import ScannerError

import fsspec

_config_logger: logging.Logger = logging.getLogger(__name__)
_NO_VALUE: object = object()

class MergeStrategies(Enum):
    SOFT: int = auto()
    DESTRUCTIVE: int = auto()

MERGING_IMPLEMENTATIONS: dict[MergeStrategies, str] = {MergeStrategies.SOFT: '_soft_merge', MergeStrategies.DESTRUCTIVE: '_destructive_merge'}

class OmegaConfigLoader(AbstractConfigLoader):
    def __init__(self, conf_source: str, env: str = None, runtime_params: dict[str, Any] = None, *,
                 config_patterns: dict[str, list[str]] = None, base_env: str = None, default_run_env: str = None,
                 custom_resolvers: dict[str, Callable] = None, merge_strategy: dict[str, str] = None):
        self.base_env: str = base_env or ''
        self.default_run_env: str = default_run_env or ''
        self.merge_strategy: dict[str, str] = merge_strategy or {}
        self._globals_oc: OmegaConf = None
        self._runtime_params_oc: OmegaConf = None
        self.config_patterns: dict[str, list[str]] = {'catalog': ['catalog*', 'catalog*/**', '**/catalog*'],
                                                     'parameters': ['parameters*', 'parameters*/**', '**/parameters*'],
                                                     'credentials': ['credentials*', 'credentials*/**', '**/credentials*'],
                                                     'globals': ['globals.yml']}
        self.config_patterns.update(config_patterns or {})
        OmegaConf.clear_resolver('oc.env')
        self._custom_resolvers: dict[str, Callable] = custom_resolvers
        if custom_resolvers:
            self._register_new_resolvers(custom_resolvers)
        self._register_globals_resolver()
        self._fs, self._protocol = self._initialise_filesystem_and_protocol(conf_source)
        super().__init__(conf_source=conf_source, env=env, runtime_params=runtime_params)
        try:
            self._globals: dict[str, Any] = self['globals']
        except MissingConfigException:
            self._globals = {}

    def __setitem__(self, key: str, value: Any) -> None:
        if key == 'globals':
            self._globals = value
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        ...

    def __repr__(self) -> str:
        ...

    def keys(self) -> KeysView[str]:
        ...

    @typing.no_type_check
    def load_and_merge_dir_config(self, conf_path: str, patterns: list[str], key: str, processed_files: set, read_environment_variables: bool = False) -> dict:
        ...

    @staticmethod
    def _initialise_filesystem_and_protocol(conf_source: str) -> tuple[fsspec.AbstractFileSystem, str]:
        ...

    def _merge_configs(self, config: dict, env_config: dict, key: str, env_path: str) -> dict:
        ...

    def _get_all_keys(self, cfg: dict, parent_key: str = '') -> set[str]:
        ...

    def _is_valid_config_path(self, path: Path) -> bool:
        ...

    def _register_globals_resolver(self) -> None:
        ...

    def _register_runtime_params_resolver(self) -> None:
        ...

    def _get_globals_value(self, variable: str, default_value: Any = _NO_VALUE) -> Any:
        ...

    def _get_runtime_value(self, variable: str, default_value: Any = _NO_VALUE) -> Any:
        ...

    @staticmethod
    def _register_new_resolvers(resolvers: dict[str, Callable]) -> None:
        ...

    def _check_duplicates(self, key: str, config_per_file: dict) -> None:
        ...

    @staticmethod
    def _resolve_environment_variables(config: dict[str, Any]) -> None:
        ...

    @staticmethod
    def _destructive_merge(config: dict, env_config: dict, env_path: str) -> dict:
        ...

    @staticmethod
    def _soft_merge(config: dict, env_config: dict, env_path: str = None) -> dict:
        ...
