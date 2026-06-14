import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union

from pydantic.v1.config import BaseConfig, Extra
from pydantic.v1.fields import ModelField
from pydantic.v1.main import BaseModel
from pydantic.v1.typing import StrPath

env_file_sentinel: str
SettingsSourceCallable = Callable[["BaseSettings"], Dict[str, Any]]
DotenvType = Union[StrPath, List[StrPath], Tuple[StrPath, ...]]

class SettingsError(ValueError): ...

class BaseSettings(BaseModel):
    def __init__(
        __pydantic_self__,
        _env_file: Optional[DotenvType] = ...,
        _env_file_encoding: Optional[str] = ...,
        _env_nested_delimiter: Optional[str] = ...,
        _secrets_dir: Optional[StrPath] = ...,
        **values: Any,
    ) -> None: ...
    def _build_values(
        self,
        init_kwargs: Dict[str, Any],
        _env_file: Optional[DotenvType] = ...,
        _env_file_encoding: Optional[str] = ...,
        _env_nested_delimiter: Optional[str] = ...,
        _secrets_dir: Optional[StrPath] = ...,
    ) -> Dict[str, Any]: ...

    class Config(BaseConfig):
        env_prefix: str
        env_file: Optional[DotenvType]
        env_file_encoding: Optional[str]
        env_nested_delimiter: Optional[str]
        secrets_dir: Optional[StrPath]
        validate_all: bool
        extra: Extra
        arbitrary_types_allowed: bool
        case_sensitive: bool

        @classmethod
        def prepare_field(cls, field: ModelField) -> None: ...
        @classmethod
        def customise_sources(
            cls,
            init_settings: "InitSettingsSource",
            env_settings: "EnvSettingsSource",
            file_secret_settings: "SecretsSettingsSource",
        ) -> Tuple[SettingsSourceCallable, ...]: ...
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any: ...

class InitSettingsSource:
    __slots__ = ("init_kwargs",)
    init_kwargs: Dict[str, Any]

    def __init__(self, init_kwargs: Dict[str, Any]) -> None: ...
    def __call__(self, settings: BaseSettings) -> Dict[str, Any]: ...
    def __repr__(self) -> str: ...

class EnvSettingsSource:
    __slots__ = ("env_file", "env_file_encoding", "env_nested_delimiter", "env_prefix_len")
    env_file: Optional[DotenvType]
    env_file_encoding: Optional[str]
    env_nested_delimiter: Optional[str]
    env_prefix_len: int

    def __init__(
        self,
        env_file: Optional[DotenvType],
        env_file_encoding: Optional[str],
        env_nested_delimiter: Optional[str] = ...,
        env_prefix_len: int = ...,
    ) -> None: ...
    def __call__(self, settings: BaseSettings) -> Dict[str, Any]: ...
    def _read_env_files(self, case_sensitive: bool) -> Dict[str, Optional[str]]: ...
    def field_is_complex(self, field: ModelField) -> Tuple[bool, bool]: ...
    def explode_env_vars(self, field: ModelField, env_vars: Mapping[str, Optional[str]]) -> Dict[str, Any]: ...
    def __repr__(self) -> str: ...

class SecretsSettingsSource:
    __slots__ = ("secrets_dir",)
    secrets_dir: Optional[StrPath]

    def __init__(self, secrets_dir: Optional[StrPath]) -> None: ...
    def __call__(self, settings: BaseSettings) -> Dict[str, Any]: ...
    def __repr__(self) -> str: ...

def read_env_file(
    file_path: StrPath,
    *,
    encoding: Optional[str] = ...,
    case_sensitive: bool = ...,
) -> Dict[str, Optional[str]]: ...

def find_case_path(
    dir_path: Path,
    file_name: str,
    case_sensitive: bool,
) -> Optional[Path]: ...