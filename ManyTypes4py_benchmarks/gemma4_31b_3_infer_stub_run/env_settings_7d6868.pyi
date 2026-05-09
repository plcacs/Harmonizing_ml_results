import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload
from pydantic.v1.config import BaseConfig, Extra
from pydantic.v1.fields import ModelField
from pydantic.v1.main import BaseModel
from pydantic.v1.typing import StrPath

env_file_sentinel: str = ...
SettingsSourceCallable: Callable[['BaseSettings'], Dict[str, Any]]
DotenvType: Union[StrPath, List[StrPath], Tuple[StrPath, ...]]

class SettingsError(ValueError): ...

class BaseSettings(BaseModel):
    """
    Base class for settings, allowing values to be overridden by environment variables.

    This is useful in production for secrets you do not wish to save in code, it plays nicely with docker(-compose),
    Heroku and any 12 factor app design.
    """
    def __init__(
        self,
        _env_file: DotenvType = ...,
        _env_file_encoding: Optional[str] = None,
        _env_nested_delimiter: Optional[str] = None,
        _secrets_dir: Optional[StrPath] = None,
        **values: Any,
    ) -> None: ...

    def _build_values(
        self,
        init_kwargs: Dict[str, Any],
        _env_file: Optional[DotenvType] = None,
        _env_file_encoding: Optional[str] = None,
        _env_nested_delimiter: Optional[str] = None,
        _secrets_dir: Optional[StrPath] = None,
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
            init_settings: 'InitSettingsSource',
            env_settings: 'EnvSettingsSource',
            file_secret_settings: 'SecretsSettingsSource',
        ) -> Tuple['InitSettingsSource', 'EnvSettingsSource', 'SecretsSettingsSource']: ...

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any: ...

class InitSettingsSource:
    init_kwargs: Dict[str, Any]

    def __init__(self, init_kwargs: Dict[str, Any]) -> None: ...
    def __call__(self, settings: BaseSettings) -> Dict[str, Any]: ...
    def __repr__(self) -> str: ...

class EnvSettingsSource:
    env_file: Optional[DotenvType]
    env_file_encoding: Optional[str]
    env_nested_delimiter: Optional[str]
    env_prefix_len: int

    def __init__(
        self,
        env_file: Optional[DotenvType],
        env_file_encoding: Optional[str],
        env_nested_delimiter: Optional[str] = None,
        env_prefix_len: int = 0,
    ) -> None: ...

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]: ...
    def _read_env_files(self, case_sensitive: bool) -> Dict[str, str]: ...
    def field_is_complex(self, field: ModelField) -> Union[Tuple[bool, bool], Tuple[bool, bool]]: ...
    def explode_env_vars(self, field: ModelField, env_vars: Dict[str, str]) -> Dict[str, Any]: ...
    def __repr__(self) -> str: ...

class SecretsSettingsSource:
    secrets_dir: Optional[StrPath]

    def __init__(self, secrets_dir: Optional[StrPath]) -> None: ...
    def __call__(self, settings: BaseSettings) -> Dict[str, Any]: ...
    def __repr__(self) -> str: ...

def read_env_file(
    file_path: Union[str, Path],
    *,
    encoding: Optional[str] = None,
    case_sensitive: bool = False,
) -> Dict[str, str]: ...

def find_case_path(
    dir_path: Path,
    file_name: str,
    case_sensitive: bool,
) -> Optional[Path]: ...