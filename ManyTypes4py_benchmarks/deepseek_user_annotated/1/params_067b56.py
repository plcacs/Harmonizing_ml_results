import abc
import logging
import ssl
import typing
import warnings
from datetime import timedelta, timezone, tzinfo
from pathlib import Path as _Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from mode import Seconds as _Seconds, want_seconds
from mode.utils.imports import SymbolArg, symbol_by_name
from mode.utils.logging import Severity as _Severity
from yarl import URL as _URL

from faust.exceptions import ImproperlyConfigured
from faust.utils import json
from faust.utils.urls import URIListArg, urllist

from faust.types.auth import CredentialsArg, CredentialsT, to_credentials
from faust.types.codecs import CodecArg, CodecT

if typing.TYPE_CHECKING:
    from .settings import Settings as _Settings
    from .sections import Section as _Section
else:
    class _Section: ...       # noqa
    class _Settings: ...      # noqa

__all__ = [
    'AutodiscoverArg',
    'DictArg',
    'URLArg',
    'BrokerArg',
    'Param',
    'Bool',
    'Str',
    'Severity',
    'Int',
    'UnsignedInt',
    'Version',
    'Port',
    'Seconds',
    'Credentials',
    'SSLContext',
    'Dict',
    'LogHandlers',
    'Timezone',
    'BrokerList',
    'URL',
    'Path',
    'Codec',
    'Enum',
    'Symbol',
    'to_bool',
]

#: Default transport used when no scheme specified.
DEFAULT_BROKER_SCHEME: str = 'kafka'

T = TypeVar('T')
IT = TypeVar('IT')   # Input type.
OT = TypeVar('OT')   # Output type.

BOOLEAN_TERMS: Mapping[str, bool] = {
    '': False,
    'false': False,
    'no': False,
    '0': False,
    'true': True,
    'yes': True,
    '1': True,
    'on': True,
    'off': False,
}

AutodiscoverArg = Union[
    bool,
    Iterable[str],
    Callable[[], Iterable[str]],
]

DictArg = Union[str, Mapping[str, T]]

URLArg = Union[str, _URL]
BrokerArg = URIListArg

DEPRECATION_WARNING_TEMPLATE: str = '''
Setting {self.name} is deprecated since Faust version \
{self.version_deprecated}: {self.deprecation_reason}. {alt_removal}
'''.strip()

DEPRECATION_REMOVAL_WARNING: str = '''
Further the setting is scheduled to be removed in Faust version \
{self.version_removal}.
'''.strip()


def to_bool(term: Union[str, bool], *,
            table: Mapping[str, bool] = BOOLEAN_TERMS) -> bool:
    """Convert common terms for true/false to bool.

    Examples (true/false/yes/no/on/off/1/0).
    """
    if table is None:
        table = BOOLEAN_TERMS
    if isinstance(term, str):
        try:
            return table[term.lower()]
        except KeyError:
            raise TypeError('Cannot coerce {0!r} to type bool'.format(term))
    return term


OutputCallable = Callable[[_Settings, OT], OT]
OnDefaultCallable = Callable[[_Settings], IT]


class Param(Generic[IT, OT], property):
    """Faust setting desscription."""

    text_type: ClassVar[Tuple[Any, ...]] = (Any,)
    name: str
    storage_name: str
    default: IT = cast(IT, None)
    env_name: Optional[str] = None
    default_alias: Optional[str] = None
    default_template: Optional[str] = None
    allow_none: bool = False
    ignore_default: bool = False
    section: _Section
    version_introduced: Optional[str] = None
    version_deprecated: Optional[str] = None
    deprecation_reason: Optional[str] = None
    version_changed: Optional[Mapping[str, str]] = None
    version_removed: Optional[str] = None
    related_cli_options: Mapping[str, List[str]] = {}
    related_settings: List[Any] = []
    deprecation_warning_template: str = DEPRECATION_WARNING_TEMPLATE
    deprecation_removal_warning: str = DEPRECATION_REMOVAL_WARNING

    def __init__(self, *,
                 name: str,
                 env_name: str = None,
                 default: IT = None,
                 default_alias: str = None,
                 default_template: str = None,
                 allow_none: bool = None,
                 ignore_default: bool = None,
                 section: _Section = None,
                 version_introduced: str = None,
                 version_deprecated: str = None,
                 version_removed: str = None,
                 version_changed: Mapping[str, str] = None,
                 deprecation_reason: str = None,
                 related_cli_options: Mapping[str, List[str]] = None,
                 related_settings: List[Any] = None,
                 help: str = None,
                 **kwargs: Any) -> None:
        ...

    def _init_options(self, **kwargs: Any) -> None:
        ...

    def on_get_value(self, fun: OutputCallable) -> OutputCallable:
        ...

    def on_set_default(self, fun: OnDefaultCallable) -> OnDefaultCallable:
        ...

    def __get__(self, obj: Any, type: Type = None) -> OT:
        ...

    def __set__(self, obj: Any, value: IT) -> None:
        ...

    def on_get(self, conf: _Settings) -> OT:
        ...

    def prepare_get(self, conf: _Settings, value: OT) -> OT:
        ...

    def on_set(self, settings: Any, value: OT) -> None:
        ...

    def set_class_default(self, cls: Type) -> None:
        ...

    def on_init_set_value(self,
                          conf: _Settings,
                          provided_value: Optional[IT]) -> None:
        ...

    def on_init_set_default(self,
                            conf: _Settings,
                            provided_value: Optional[IT]) -> None:
        ...

    def build_deprecation_warning(self) -> str:
        ...

    def validate_before(self, value: IT = None) -> None:
        ...

    def validate_after(self, value: OT) -> None:
        ...

    def prepare_set(self, conf: _Settings, value: IT) -> OT:
        ...

    def prepare_init_default(self, conf: _Settings, value: IT) -> OT:
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
    """Boolean setting type."""
    text_type: ClassVar[Tuple[Type[bool], ...]] = (bool,)

    def to_python(self, conf: _Settings, value: Any) -> bool:
        ...


class Str(Param[str, str]):
    """String setting type."""
    text_type: ClassVar[Tuple[Type[str], ...]] = (str,)


class Severity(Param[_Severity, _Severity]):
    """Logging severity setting type."""
    text_type: ClassVar[Tuple[Union[Type[str], Type[int]], ...]] = (str, int)


class Number(Param[IT, OT]):
    """Number setting type (baseclass for int/float)."""
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    number_aliases: Mapping[IT, OT]

    def _init_options(self,
                      min_value: int = None,
                      max_value: int = None,
                      number_aliases: Mapping[IT, OT] = None,
                      **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def convert(self, conf: _Settings, value: IT) -> OT:
        ...

    def to_python(self,
                  conf: _Settings,
                  value: IT) -> OT:
        ...

    def validate_after(self, value: OT) -> None:
        ...

    def _out_of_range(self, value: float) -> ImproperlyConfigured:
        ...


NumberInputArg = Union[str, int, float]


class _Int(Number[IT, OT]):
    text_type: ClassVar[Tuple[Type[int], ...]] = (int,)

    def convert(self,
                conf: _Settings,
                value: IT) -> OT:
        ...


class Int(_Int[NumberInputArg, int]):
    """Signed integer setting type."""


class UnsignedInt(_Int[NumberInputArg, int]):
    """Unsigned integer setting type."""
    min_value: int = 0


class Version(Int):
    """Version setting type."""
    min_value: int = 1


class Port(UnsignedInt):
    """Network port setting type."""
    min_value: int = 1
    max_value: int = 65535


class Seconds(Param[_Seconds, float]):
    """Seconds setting type."""
    text_type: ClassVar[Tuple[Union[Type[float], Type[timedelta]], ...]] = (float, timedelta)

    def to_python(self, conf: _Settings, value: _Seconds) -> float:
        ...


class Credentials(Param[CredentialsArg, Optional[CredentialsT]]):
    """Authentication credentials setting type."""
    text_type: ClassVar[Tuple[Type[CredentialsT], ...]] = (CredentialsT,)

    def to_python(self,
                  conf: _Settings,
                  value: CredentialsArg) -> Optional[CredentialsT]:
        ...


class SSLContext(Param[ssl.SSLContext, Optional[ssl.SSLContext]]):
    """SSL context setting type."""
    text_type: ClassVar[Tuple[Type[ssl.SSLContext], ...]] = (ssl.SSLContext,)


class Dict(Param[DictArg[T], Mapping[str, T]]):
    """Dictionary setting type."""
    text_type: ClassVar[Tuple[Type[dict], ...]] = (dict,)

    def to_python(self,
                  conf: _Settings,
                  value: DictArg[T]) -> Mapping[str, T]:
        ...


class LogHandlers(Param[List[logging.Handler], List[logging.Handler]]):
    """Log handler list setting type."""
    text_type: ClassVar[Tuple[Type[List[logging.Handler]], ...]] = (List[logging.Handler],)

    def prepare_init_default(
            self, conf: _Settings, value: Any) -> List[logging.Handler]:
        ...


class Timezone(Param[Union[str, tzinfo], tzinfo]):
    """Timezone setting type."""
    text_type: ClassVar[Tuple[Type[tzinfo], ...]] = (tzinfo,)
    builtin_timezones: Dict[str, tzinfo] = {'UTC': timezone.utc}

    def to_python(self, conf: _Settings, value: Union[str, tzinfo]) -> tzinfo:
        ...


class BrokerList(Param[BrokerArg, List[_URL]]):
    """Broker URL list setting type."""
    text_type: ClassVar[Tuple[Union[Type[str], Type[_URL], Type[List[str]]], ...]] = (str, _URL, List[str])
    default_scheme: str = DEFAULT_BROKER_SCHEME

    def to_python(self, conf: _Settings, value: BrokerArg) -> List[_URL]:
        ...

    def broker_list(self, value: BrokerArg) -> List[_URL]:
        ...


class URL(Param[URLArg, _URL]):
    """URL setting type."""
    text_type: ClassVar[Tuple[Union[Type[str], Type[_URL]], ...]] = (str, _URL)

    def to_python(self, conf: _Settings, value: URLArg) -> _URL:
        ...


class Path(Param[Union[str, _Path], _Path]):
    """Path setting type."""
    text_type: ClassVar[Tuple[Union[Type[str], Type[_Path]], ...]] = (str, _Path)
    expanduser: bool = True

    def to_python(self, conf: _Settings, value: Union[str, _Path]) -> _Path:
        ...

    def prepare_path(self, conf: _Settings, path: _Path) -> _Path:
        ...


class Codec(Param[CodecArg, CodecArg]):
    """Serialization codec setting type."""
    text_type: ClassVar[Tuple[Union[Type[str], Type[CodecT]], ...]] = (str, CodecT)


def Enum(typ: T) -> Type[Param[Union[str, T], T]]:
    """Generate new enum setting type."""

    class EnumParam(Param[Union[str, T], T]):
        text_type: ClassVar[Tuple[Type[str], ...]] = (str,)

        def to_python(self, conf: _Settings, value: Union[str, T]) -> T:
            ...

    return EnumParam


class _Symbol(Param[IT, OT]):
    text_type: ClassVar[Tuple[Union[Type[str], Type[Type]], ...]] = (str, Type)

    def to_python(self, conf: _Settings, value: IT) -> OT:
        ...


def Symbol(typ: T) -> Type[Param[SymbolArg[T], T]]:
    """Generate new symbol setting type."""
    return _Symbol[SymbolArg[T], T]
