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
    SOFT = auto()
    DESTRUCTIVE = auto()

MERGING_IMPLEMENTATIONS: typing.Dict[MergeStrategies, str] = {MergeStrategies.SOFT: '_soft_merge', MergeStrategies.DESTRUCTIVE: '_destructive_merge'}

class OmegaConfigLoader(AbstractConfigLoader):
    def __init__(self, conf_source: str, env: str = None, runtime_params: Any = None, *, config_patterns: typing.Optional[typing.Dict[str, typing.List[str]]] = None, base_env: str = None, default_run_env: str = None, custom_resolvers: typing.Optional[typing.Dict[str, Callable]] = None, merge_strategy: typing.Optional[typing.Dict[str, str]] = None):
        self.base_env: str = base_env or ''
        self.default_run_env: str = default_run_env or ''
        self.merge_strategy: typing.Dict[str, str] = merge_strategy or {}
        self._globals_oc: typing.Optional[DictConfig] = None
        self._runtime_params_oc: typing.Optional[DictConfig] = None
        self.config_patterns: typing.Dict[str, typing.List[str]] = {'catalog': ['catalog*', 'catalog*/**', '**/catalog*'], 'parameters': ['parameters*', 'parameters*/**', '**/parameters*'], 'credentials': ['credentials*', 'credentials*/**', '**/credentials*'], 'globals': ['globals.yml']}
        self.config_patterns.update(config_patterns or {})
        OmegaConf.clear_resolver('oc.env')
        self._custom_resolvers: typing.Optional[typing.Dict[str, Callable]] = custom_resolvers
        if custom_resolvers:
            self._register_new_resolvers(custom_resolvers)
        self._register_globals_resolver()
        self._fs, self._protocol = self._initialise_filesystem_and_protocol(conf_source)
        super().__init__(conf_source=conf_source, env=env, runtime_params=runtime_params)
        try:
            self._globals = self['globals']
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

    def keys(self) -> KeysView:
        ...

    @typing.no_type_check
    def load_and_merge_dir_config(self, conf_path: str, patterns: typing.List[str], key: str, processed_files: set, read_environment_variables: bool = False) -> Dict[str, Any]:
        ...

    @staticmethod
    def _initialise_filesystem_and_protocol(conf_source: str) -> typing.Tuple[fsspec.AbstractFileSystem, str]:
        ...

    def _merge_configs(self, config: Dict[str, Any], env_config: Dict[str, Any], key: str, env_path: str) -> Dict[str, Any]:
        ...

    def _get_all_keys(self, cfg: DictConfig, parent_key: str = '') -> set:
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
    def _register_new_resolvers(resolvers: typing.Dict[str, Callable]) -> None:
        ...

    def _check_duplicates(self, key: str, config_per_file: Dict[Path, Dict[str, Any]]) -> None:
        ...

    @staticmethod
    def _resolve_environment_variables(config: Dict[str, Any]) -> None:
        ...

    @staticmethod
    def _destructive_merge(config: Dict[str, Any], env_config: Dict[str, Any], env_path: str) -> Dict[str, Any]:
        ...

    @staticmethod
    def _soft_merge(config: Dict[str, Any], env_config: Dict[str, Any], env_path: str = None) -> Dict[str, Any]:
        ...

    def _is_hidden(self, path_str: str) -> bool:
        ...
