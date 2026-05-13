import os
from pathlib import Path
from typing import AbstractSet, Any, Callable, ClassVar, Dict, List, Mapping, Optional, Tuple, Type, Union
from pydantic.v1.config import BaseConfig, Extra
from pydantic.v1.fields import ModelField
from pydantic.v1.main import BaseModel
from pydantic.v1.types import JsonWrapper
from pydantic.v1.typing import StrPath, display_as_type, get_origin, is_union
from pydantic.v1.utils import deep_update, lenient_issubclass, path_type, sequence_like

env_file_sentinel: str

SettingsSourceCallable = Callable[['BaseSettings'], Dict[str, Any]]
DotenvType = Union[StrPath, List[StrPath], Tuple[StrPath, ...]]

class SettingsError(ValueError):
    pass

class BaseSettings(BaseModel):
    def __init__(
        __pydantic_self__,
        _env_file: Union[StrPath, List[StrPath], Tuple[StrPath, ...], str] = ...,
        _env_file_encoding: Optional[str] = None,
        _env_nested_delimiter: Optional[str] = None,
        _secrets_dir: Optional[StrPath] = None,
        **values: Any
    ) -> None: ...

    def _build_values(
        self,
        init_kwargs: Dict[str, Any],
        _env_file: Optional[Union[StrPath, List[StrPath], Tuple[StrPath, ...]]] = None,
        _env_file_encoding: Optional[str] = None,
        _env_nested_delimiter: Optional[str] = None,
        _secrets_dir: Optional[StrPath] = None
    ) -> Dict[str, Any]: ...

    class Config(BaseConfig):
        env_prefix: ClassVar[str]
        env_file: ClassVar[Optional[Union[StrPath, List[StrPath], Tuple[StrPath, ...]]]]
        env_file_encoding: ClassVar[Optional[str]]
        env_nested_delimiter: ClassVar[Optional[str]]
        secrets_dir: ClassVar[Optional[StrPath]]
        validate_all: ClassVar[bool]
        extra: ClassVar[Extra]
        arbitrary_types_allowed: ClassVar[bool]
        case_sensitive: ClassVar[bool]

        @classmethod
        def prepare_field(cls, field: ModelField) -> None: ...

        @classmethod
        def customise_sources(
            cls,
            init_settings: InitSettingsSource,
            env_settings: EnvSettingsSource,
            file_secret_settings: SecretsSettingsSource
        ) -> Tuple[SettingsSourceCallable, ...]: ...

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any: ...

class InitSettingsSource:
    __slots__ = ('init_kwargs',)
    init_kwargs: Dict[str, Any]

    def __init__(self, init_kwargs: Dict[str, Any]) -> None: ...
    def __call__(self, settings: BaseSettings) -> Dict[str, Any]: ...
    def __repr__(self) -> str: ...

class EnvSettingsSource:
    __slots__ = ('env_file', 'env_file_encoding', 'env_nested_delimiter', 'env_prefix_len')
    env_file: Optional[Union[StrPath, List[StrPath], Tuple[StrPath, ...]]]
    env_file_encoding: Optional[str]
    env_nested_delimiter: Optional[str]
    env_prefix_len: int

    def __init__(
        self,
        env_file: Optional[Union[StrPath, List[StrPath], Tuple[StrPath, ...]]],
        env_file_encoding: Optional[str],
        env_nested_delimiter: Optional[str] = None,
        env_prefix_len: int = 0
    ) -> None: ...
    def __call__(self, settings: BaseSettings) -> Dict[str, Any]: ...
    def _read_env_files(self, case_sensitive: bool) -> Dict[str, str]: ...
    def field_is_complex(self, field: ModelField) -> Tuple[bool, bool]: ...
    def explode_env_vars(self, field: ModelField, env_vars: Dict[str, str]) -> Dict[str, Any]: ...
    def __repr__(self) -> str: ...

class SecretsSettingsSource:
    __slots__ = ('secrets_dir',)
    secrets_dir: Optional[StrPath]

    def __init__(self, secrets_dir: Optional[StrPath]) -> None: ...
    def __call__(self, settings: BaseSettings) -> Dict[str, Any]: ...
    def __repr__(self) -> str: ...

def read_env_file(file_path: Path, *, encoding: Optional[str] = None, case_sensitive: bool = False) -> Dict[str, Optional[str]]: ...
def find_case_path(dir_path: Path, file_name: str, case_sensitive: bool) -> Optional[Path]: ...