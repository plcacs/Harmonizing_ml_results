"""This module provides ``kedro.config`` with the functionality to load one
or more configuration files of yaml or json type from specified paths through OmegaConf.
"""

from __future__ import annotations

import io
import logging
import mimetypes
import typing
from collections.abc import Iterable, KeysView
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import fsspec
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError, UnsupportedInterpolationType
from omegaconf.resolvers import oc
from yaml.parser import ParserError
from yaml.scanner import ScannerError

from kedro.config.abstract_config import AbstractConfigLoader, MissingConfigException

_config_logger = logging.getLogger(__name__)

_NO_VALUE = object()


class MergeStrategies(Enum):
    SOFT = auto()
    DESTRUCTIVE = auto()


MERGING_IMPLEMENTATIONS: Dict[MergeStrategies, str] = {
    MergeStrategies.SOFT: "_soft_merge",
    MergeStrategies.DESTRUCTIVE: "_destructive_merge",
}


class OmegaConfigLoader(AbstractConfigLoader):
    def __init__(  # noqa: PLR0913
        self,
        conf_source: str,
        env: Optional[str] = None,
        runtime_params: Optional[Dict[str, Any]] = None,
        *,
        config_patterns: Optional[Dict[str, List[str]]] = None,
        base_env: Optional[str] = None,
        default_run_env: Optional[str] = None,
        custom_resolvers: Optional[Dict[str, Callable]] = None,
        merge_strategy: Optional[Dict[str, str]] = None,
    ):
        self.base_env: str = base_env or ""
        self.default_run_env: str = default_run_env or ""
        self.merge_strategy: Dict[str, str] = merge_strategy or {}
        self._globals_oc: Optional[DictConfig] = None
        self._runtime_params_oc: Optional[DictConfig] = None
        self.config_patterns: Dict[str, List[str]] = {
            "catalog": ["catalog*", "catalog*/**", "**/catalog*"],
            "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
            "credentials": ["credentials*", "credentials*/**", "**/credentials*"],
            "globals": ["globals.yml"],
        }
        self.config_patterns.update(config_patterns or {})

        OmegaConf.clear_resolver("oc.env")
        self._custom_resolvers: Optional[Dict[str, Callable]] = custom_resolvers
        if custom_resolvers:
            self._register_new_resolvers(custom_resolvers)
        self._register_globals_resolver()

        self._fs, self._protocol = self._initialise_filesystem_and_protocol(conf_source)

        super().__init__(
            conf_source=conf_source,
            env=env,
            runtime_params=runtime_params,
        )
        try:
            self._globals: Dict[str, Any] = self["globals"]
        except MissingConfigException:
            self._globals = {}

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "globals":
            self._globals = value
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Dict[str, Any]:  # noqa: PLR0912
        self._register_runtime_params_resolver()

        if key in self:
            return super().__getitem__(key)  # type: ignore[no-any-return]

        if key not in self.config_patterns:
            raise KeyError(
                f"No config patterns were found for '{key}' in your config loader"
            )
        patterns: List[str] = [*self.config_patterns[key]]

        if key == "globals":
            OmegaConf.clear_resolver("runtime_params")

        read_environment_variables: bool = key == "credentials"

        processed_files: Set[Path] = set()
        if self._protocol == "file":
            base_path: str = str(Path(self.conf_source) / self.base_env)
        else:
            base_path = str(Path(self._fs.ls("", detail=False)[-1]) / self.base_env)
        try:
            base_config: Dict[str, Any] = self.load_and_merge_dir_config(
                base_path, patterns, key, processed_files, read_environment_variables
            )
        except UnsupportedInterpolationType as exc:
            if "runtime_params" in str(exc):
                raise UnsupportedInterpolationType(
                    "The `runtime_params:` resolver is not supported for globals."
                )
            else:
                raise exc

        config: Dict[str, Any] = base_config

        run_env: str = self.env or self.default_run_env

        if run_env == self.base_env:
            return config  # type: ignore[no-any-return]

        if self._protocol == "file":
            env_path: str = str(Path(self.conf_source) / run_env)
        else:
            env_path = str(Path(self._fs.ls("", detail=False)[-1]) / run_env)
        try:
            env_config: Dict[str, Any] = self.load_and_merge_dir_config(
                env_path, patterns, key, processed_files, read_environment_variables
            )
        except UnsupportedInterpolationType as exc:
            if "runtime_params" in str(exc):
                raise UnsupportedInterpolationType(
                    "The `runtime_params:` resolver is not supported for globals."
                )
            else:
                raise exc

        resulting_config: Dict[str, Any] = self._merge_configs(config, env_config, key, env_path)

        if not processed_files and key != "globals":
            raise MissingConfigException(
                f"No files of YAML or JSON format found in {base_path} or {env_path} matching"
                f" the glob pattern(s): {[*self.config_patterns[key]]}"
            )

        return resulting_config  # type: ignore[no-any-return]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OmegaConfigLoader(conf_source={self.conf_source}, env={self.env}, "
            f"runtime_params={self.runtime_params}, "
            f"config_patterns={self.config_patterns}, "
            f"base_env={self.base_env}), "
            f"default_run_env={self.default_run_env}), "
            f"custom_resolvers={self._custom_resolvers}), "
            f"merge_strategy={self.merge_strategy})"
        )

    def keys(self) -> KeysView:
        return KeysView(self.config_patterns)

    @typing.no_type_check
    def load_and_merge_dir_config(
        self,
        conf_path: str,
        patterns: Iterable[str],
        key: str,
        processed_files: Set[Path],
        read_environment_variables: Optional[bool] = False,
    ) -> Dict[str, Any]:
        if not self._fs.isdir(Path(conf_path).as_posix()):
            raise MissingConfigException(
                f"Given configuration path either does not exist "
                f"or is not a valid directory: {conf_path}"
            )

        paths: List[Path] = []
        for pattern in patterns:
            for each in self._fs.glob(Path(f"{conf_path!s}/{pattern}").as_posix()):
                if not self._is_hidden(each):
                    paths.append(Path(each))

        deduplicated_paths: Set[Path] = set(paths)
        config_files_filtered: List[Path] = [
            path for path in deduplicated_paths if self._is_valid_config_path(path)
        ]

        config_per_file: Dict[Path, Any] = {}
        for config_filepath in config_files_filtered:
            try:
                with self._fs.open(str(config_filepath.as_posix())) as open_config:
                    tmp_fo = io.StringIO(open_config.read().decode("utf8"))
                    config: DictConfig = OmegaConf.load(tmp_fo)
                    processed_files.add(config_filepath)
                if read_environment_variables:
                    self._resolve_environment_variables(config)
                config_per_file[config_filepath] = config
            except (ParserError, ScannerError) as exc:
                line = exc.problem_mark.line
                cursor = exc.problem_mark.column
                raise ParserError(
                    f"Invalid YAML or JSON file {Path(config_filepath).as_posix()},"
                    f" unable to read line {line}, position {cursor}."
                ) from exc

        aggregate_config: List[DictConfig] = list(config_per_file.values())
        self._check_duplicates(key, config_per_file)

        if not aggregate_config:
            return {}

        if key == "parameters":
            return OmegaConf.to_container(
                OmegaConf.merge(*aggregate_config, self.runtime_params), resolve=True
            )

        merged_config_container: Dict[str, Any] = OmegaConf.to_container(
            OmegaConf.merge(*aggregate_config), resolve=True
        )
        return {
            k: v for k, v in merged_config_container.items() if not k.startswith("_")
        }

    @staticmethod
    def _initialise_filesystem_and_protocol(
        conf_source: str,
    ) -> Tuple[fsspec.AbstractFileSystem, str]:
        file_mimetype, _ = mimetypes.guess_type(conf_source)
        if file_mimetype == "application/x-tar":
            protocol = "tar"
        elif file_mimetype in (
            "application/zip",
            "application/x-zip-compressed",
            "application/zip-compressed",
        ):
            protocol = "zip"
        else:
            protocol = "file"
        fs = fsspec.filesystem(protocol=protocol, fo=conf_source)
        return fs, protocol

    def _merge_configs(
        self,
        config: Dict[str, Any],
        env_config: Dict[str, Any],
        key: str,
        env_path: str,
    ) -> Any:
        merging_strategy: str = self.merge_strategy.get(key, "destructive")
        try:
            strategy: MergeStrategies = MergeStrategies[merging_strategy.upper()]

            merge_function_name: str = MERGING_IMPLEMENTATIONS[strategy]
            merge_function: Callable = getattr(self, merge_function_name)
            return merge_function(config, env_config, env_path)
        except KeyError:
            allowed_strategies: List[str] = [strategy.name.lower() for strategy in MergeStrategies]
            raise ValueError(
                f"Merging strategy {merging_strategy} not supported. The accepted merging "
                f"strategies are {allowed_strategies}."
            )

    def _get_all_keys(self, cfg: Any, parent_key: str = "") -> Set[str]:
        keys: Set[str] = set()

        for key, value in cfg.items():
            full_key: str = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                keys.update(self._get_all_keys(value, full_key))
            else:
                keys.add(full_key)
        return keys

    def _is_valid_config_path(self, path: Path) -> bool:
        posix_path: str = path.as_posix()
        return self._fs.isfile(str(posix_path)) and path.suffix in [
            ".yml",
            ".yaml",
            ".json",
        ]

    def _register_globals_resolver(self) -> None:
        OmegaConf.register_new_resolver(
            "globals",
            self._get_globals_value,
            replace=True,
        )

    def _register_runtime_params_resolver(self) -> None:
        OmegaConf.register_new_resolver(
            "runtime_params",
            self._get_runtime_value,
            replace=True,
        )

    def _get_globals_value(self, variable: str, default_value: Any = _NO_VALUE) -> Any:
        if variable.startswith("_"):
            raise InterpolationResolutionError(
                "Keys starting with '_' are not supported for globals."
            )

        if not self._globals_oc:
            self._globals_oc = OmegaConf.create(self._globals)

        interpolated_value: Any = OmegaConf.select(
            self._globals_oc, variable, default=default_value
        )
        if interpolated_value != _NO_VALUE:
            return interpolated_value
        else:
            raise InterpolationResolutionError(
                f"Globals key '{variable}' not found and no default value provided."
            )

    def _get_runtime_value(self, variable: str, default_value: Any = _NO_VALUE) -> Any:
        if not self._runtime_params_oc:
            self._runtime_params_oc = OmegaConf.create(self.runtime_params)

        interpolated_value: Any = OmegaConf.select(
            self._runtime_params_oc, variable, default=default_value
        )
        if interpolated_value != _NO_VALUE:
            return interpolated_value
        else:
            raise InterpolationResolutionError(
                f"Runtime parameter '{variable}' not found and no default value provided."
            )

    @staticmethod
    def _register_new_resolvers(resolvers: Dict[str, Callable]) -> None:
        for name, resolver in resolvers.items():
            if not OmegaConf.has_resolver(name):
                msg: str = f"Registering new custom resolver: {name}"
                _config_logger.debug(msg)
                OmegaConf.register_new_resolver(name=name, resolver=resolver)

    def _check_duplicates(self, key: str, config_per_file: Dict[Path, Any]) -> None:
        if key == "parameters":
            seen_files_to_keys: Dict[Path, Set[str]] = {
                file: self._get_all_keys(OmegaConf.to_container(config, resolve=False))
                for file, config in config_per_file.items()
            }
        else:
            seen_files_to_keys = {
                file: set(config.keys()) for file, config in config_per_file.items()
            }

        duplicates: List[str] = []

        filepaths: List[Path] = list(seen_files_to_keys.keys())
        for i, filepath1 in enumerate(filepaths, 1):
            config1: Set[str] = seen_files_to_keys[filepath1]
            for filepath2 in filepaths[i:]:
                config2: Set[str] = seen_files_to_keys[filepath2]

                combined_keys: Set[str] = config1 & config2
                overlapping_keys: Set[str] = {
                    key for key in combined_keys if not key.startswith("_")
                }

                if overlapping_keys:
                    sorted_keys: str = ", ".join(sorted(overlapping_keys))
                    if len(sorted_keys) > 100:  # noqa: PLR2004
                        sorted_keys = sorted_keys[:100] + "..."
                    duplicates.append(
                        f"Duplicate keys found in {filepath1} and {filepath2}: {sorted_keys}"
                    )

        if duplicates:
            dup_str: str = "\n".join(duplicates)
            raise ValueError(f"{dup_str}")

    @staticmethod
    def _resolve_environment_variables(config: DictConfig) -> None:
        if not OmegaConf.has_resolver("oc.env"):
            OmegaConf.register_new_resolver("oc.env", oc.env)
            OmegaConf.resolve(config)
            OmegaConf.clear_resolver("oc.env")
        else:
            OmegaConf.resolve(config)

    @staticmethod
    def _destructive_merge(
        config: Dict[str, Any], env_config: Dict[str, Any], env_path: str
    ) -> Dict[str, Any]:
        common_keys: Set[str] = config.keys() & env_config.keys()
        if common_keys:
            sorted_keys: str = ", ".join(sorted(common_keys))
            msg: str = (
                "Config from path '%s' will override the following "
                "existing top-level config keys: %s"
            )
            _config_logger.debug(msg, env_path, sorted_keys)

        config.update(env_config)
        return config

    @staticmethod
    def _soft_merge(
        config: Dict[str, Any], env_config: Dict[str, Any], env_path: Optional[str] = None
    ) -> Any:
        return OmegaConf.to_container(OmegaConf.merge(config, env_config))

    def _is_hidden(self, path_str: str) -> bool:
        path: Path = Path(path_str)
        conf_path: str = Path(self.conf_source).resolve().as_posix()
        if self._protocol == "file":
            path = path.resolve()
        posix_path: str = path.as_posix()
        if posix_path.startswith(conf_path):
            posix_path = posix_path.replace(conf_path, "")
        parts: List[str] = posix_path.split(self._fs.sep)
        HIDDEN: str = "."
        return any(part.startswith(HIDDEN) for part in parts)
