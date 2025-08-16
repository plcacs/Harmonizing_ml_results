from typing import Any, Callable, ClassVar, Generic, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union
from mode import Seconds as _Seconds
from mode.utils.imports import SymbolArg, symbol_by_name
from mode.utils.logging import Severity as _Severity
from yarl import URL as _URL
from faust.exceptions import ImproperlyConfigured
from faust.utils import json
from faust.utils.urls import URIListArg, urllist
from faust.types.auth import CredentialsArg, CredentialsT, to_credentials
from faust.types.codecs import CodecArg, CodecT

AutodiscoverArg = Union[bool, Iterable[str], Callable[[], Iterable[str]]]
DictArg = Union[str, Mapping[str, T]]
URLArg = Union[str, _URL]
BrokerArg = URIListArg
NumberInputArg = Union[str, int, float]

T = TypeVar('T')
IT = TypeVar('IT')
OT = TypeVar('OT')

class Param(Generic[IT, OT], property):
    text_type: Tuple[Type]
    default: IT
    env_name: Optional[str]
    default_alias: Optional[str]
    default_template: Optional[str]
    allow_none: bool
    ignore_default: bool
    version_introduced: Optional[str]
    version_deprecated: Optional[str]
    deprecation_reason: Optional[str]
    version_changed: Optional[str]
    version_removed: Optional[str]
    deprecation_warning_template: str
    deprecation_removal_warning: str

    def __init__(self, *, name: str, env_name: Optional[str] = None, default: Optional[IT] = None, default_alias: Optional[str] = None, default_template: Optional[str] = None, allow_none: Optional[bool] = None, ignore_default: Optional[bool] = None, section: Optional[str] = None, version_introduced: Optional[str] = None, version_deprecated: Optional[str] = None, version_removed: Optional[str] = None, version_changed: Optional[str] = None, deprecation_reason: Optional[str] = None, related_cli_options: Optional[Mapping[str, str]] = None, related_settings: Optional[List[str]] = None, help: Optional[str] = None, **kwargs: Any) -> None:
        ...

    def _init_options(self, **kwargs: Any) -> None:
        ...

    def on_get_value(self, fun: Callable[[_Settings, OT], OT]) -> Callable[[_Settings, OT], OT]:
        ...

    def on_set_default(self, fun: Callable[[_Settings], IT]) -> Callable[[_Settings], IT]:
        ...

    def on_get(self, conf: _Settings) -> OT:
        ...

    def prepare_get(self, conf: _Settings, value: IT) -> OT:
        ...

    def on_set(self, settings: _Settings, value: IT) -> None:
        ...

    def set_class_default(self, cls: Type) -> None:
        ...

    def on_init_set_value(self, conf: _Settings, provided_value: Optional[IT]) -> None:
        ...

    def on_init_set_default(self, conf: _Settings, provided_value: Optional[IT]) -> None:
        ...

    def build_deprecation_warning(self) -> str:
        ...

    def validate_before(self, value: Optional[IT] = None) -> None:
        ...

    def validate_after(self, value: OT) -> None:
        ...

    def prepare_set(self, conf: _Settings, value: IT) -> OT:
        ...

    def prepare_init_default(self, conf: _Settings, value: IT) -> Optional[OT]:
        ...

    def to_python(self, conf: _Settings, value: IT) -> OT:
        ...

    @property
    def active(self) -> bool:
        ...

    @property
    def deprecated(self) -> bool:
        ...

    @property
    def class_name(self) -> str:
        ...

class Bool(Param[Any, bool]):
    text_type: Tuple[Type]

class Str(Param[str, str]):
    text_type: Tuple[Type]

class Severity(Param[_Severity, _Severity]):
    text_type: Tuple[Type]

class Number(Param[IT, OT]):
    min_value: Optional[IT]
    max_value: Optional[IT]

class _Int(Number[NumberInputArg, int]):
    text_type: Tuple[Type]

class Int(_Int[NumberInputArg, int]):
    ...

class UnsignedInt(_Int[NumberInputArg, int]):
    ...

class Version(Int):
    ...

class Port(UnsignedInt):
    ...

class Seconds(Param[_Seconds, float]):
    text_type: Tuple[Type]

class Credentials(Param[CredentialsArg, Optional[CredentialsT]]):
    text_type: Tuple[Type]

class SSLContext(Param[ssl.SSLContext, Optional[ssl.SSLContext]]):
    text_type: Tuple[Type]

class Dict(Param[DictArg[T], Mapping[str, T]]):
    text_type: Tuple[Type]

class LogHandlers(Param[List[logging.Handler], List[logging.Handler]]):
    text_type: Tuple[Type]

class Timezone(Param[Union[str, tzinfo], tzinfo]):
    text_type: Tuple[Type]

class BrokerList(Param[BrokerArg, List[_URL]]):
    text_type: Tuple[Type]
    default_scheme: str

class URL(Param[URLArg, _URL]):
    text_type: Tuple[Type]

class Path(Param[Union[str, _Path], _Path]):
    text_type: Tuple[Type]
    expanduser: bool

class Codec(Param[CodecArg, CodecArg]):
    text_type: Tuple[Type]

def Enum(typ: Type) -> Type[EnumParam]:
    ...

class _Symbol(Param[IT, OT]):
    text_type: Tuple[Type]

def Symbol(typ: Type) -> Type[_Symbol[SymbolArg[T], T]]:
    ...
