```python
import os
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

from pydantic.v1.config import BaseConfig, Extra
from pydantic.v1.fields import ModelField
from pydantic.v1.main import BaseModel
from pydantic.v1.types import JsonWrapper
from pydantic.v1.typing import StrPath

env_file_sentinel: str = ...
SettingsSourceCallable = Callable[["BaseSettings"], Dict[str, Any]]
DotenvType = Union[StrPath, List[StrPath], Tuple[StrPath, ...]]


class SettingsError(ValueError):
    ...


class BaseSettings(BaseModel):
    def __init__(
        self,
        _env_file: Any = ...,
        _env_file_encoding: Optional[str] = None,
        _env_nested_delimiter: Optional[str] = None,
        _secrets_dir: Optional[StrPath] = None,
        **values: Any,
    ) -> None: ...

    def _build_values(
        self,
        init_kwargs: Dict[str, Any],
        _env_file: Any = ...,
        _env_file_encoding: Optional[str] = None,
        _env_nested_delimiter: Optional[str] = None,
        _secrets_dir: Optional[StrPath] = None,
    ) -> Dict[str, Any]: ...

    class Config(BaseConfig):
        env_prefix: ClassVar[str] = ...
        env_file: ClassVar[Any] = ...
        env_file_encoding: ClassVar[Optional[str]] = ...
        env_nested_delimiter: ClassVar[Optional[str]] = ...
        secrets_dir: ClassVar[Optional[StrPath]] = ...
        validate_all: ClassVar[bool] = ...
        extra: ClassVar[Extra] = ...
        arbitrary_types_allowed: ClassVar[bool] = ...
        case_sensitive: ClassVar[bool] = ...

        @classmethod
        def prepare_field(cls, field: ModelField) -> None: ...

        @classmethod
        def customise_sources(
            cls,
            init_settings: "InitSettingsSource",
            env_settings: "EnvSettingsSource",
            file_secret_settings: "SecretsSettingsSource",
        ) -> Tuple["InitSettingsSource", "EnvSettingsSource", "SecretsSettingsSource"]: ...

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any: ...


class InitSettingsSource:
    __slots__: Tuple[str, ...] = ...

    def __init__(self, init_kwargs: Dict[str, Any]) -> None: ...

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]: ...

    def __repr__(self) -> str: ...


class EnvSettingsSource:
    __slots__: Tuple[str, ...] = ...

    def __init__(
        self,
        env_file: Any,
        env_file_encoding: Optional[str],
        env_nested_delimiter: Optional[str] = None,
        env_prefix_len: int = 0,
    ) -> None: ...

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]: ...

    def _read_env_files(self, case_sensitive: bool) -> Dict[str, str]: ...

    def field_is_complex(self, field: ModelField) -> Tuple[bool, bool]: ...

    def explode_env_vars(self, field: ModelField, env_vars: Dict[str, str]) -> Dict[str, Any]: ...

    def __repr__(self) -> str: ...


class SecretsSettingsSource:
    __slots__: Tuple[str, ...] = ...

    def __init__(self, secrets_dir: Optional[StrPath]) -> None: ...

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]: ...

    def __repr__(self) -> str: ...


def read_env_file(
    file_path: StrPath,
    *,
    encoding: Optional[str] = None,
    case_sensitive: bool = False,
) -> Dict[str, Optional[str]]: ...


def find_case_path(
    dir_path: Path,
    file_name: str,
    case_sensitive: bool,
) -> Optional[Path]: ...
```