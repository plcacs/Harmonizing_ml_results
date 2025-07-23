import abc
import logging
import ssl
import typing
import warnings
from datetime import timedelta, timezone, tzinfo
from pathlib import Path as _Path
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, List, 
                    Mapping, Optional, Tuple, Type, TypeVar, Union, cast, 
                    overload)
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
    class _Section: ...
    class _Settings: ...

__all__ = [
    'AutodiscoverArg', 'DictArg', 'URLArg', 'BrokerArg', 'Param', 'Bool', 
    'Str', 'Severity', 'Int', 'UnsignedInt', 'Version', 'Port', 'Seconds', 
    'Credentials', 'SSLContext', 'Dict', 'LogHandlers', 'Timezone', 
    'BrokerList', 'URL', 'Path', 'Codec', 'Enum', 'Symbol', 'to_bool'
]

DEFAULT_BROKER_SCHEME = 'kafka'
T = TypeVar('T')
IT = TypeVar('IT')
OT = TypeVar('OT')
BOOLEAN_TERMS: Dict[str, bool] = {
    '': False, 'false': False, 'no': False, '0': False, 
    'true': True, 'yes': True, '1': True, 'on': True, 'off': False
}

AutodiscoverArg = Union[bool, Iterable[str], Callable[[], Iterable[str]]]
DictArg = Union[str, Mapping[str, T]]
URLArg = Union[str, _URL]
BrokerArg = URIListArg
DEPRECATION_WARNING_TEMPLATE = '\nSetting {self.name} is deprecated since Faust version {self.version_deprecated}: {self.deprecation_reason}. {alt_removal}\n'.strip()
DEPRECATION_REMOVAL_WARNING = '\nFurther the setting is scheduled to be removed in Faust version {self.version_removal}.\n'.strip()

def to_bool(term: Any, *, table: Optional[Dict[str, bool]] = BOOLEAN_TERMS) -> bool:
    if table is None:
        table = BOOLEAN_TERMS
    if isinstance(term, str):
        try:
            return table[term.lower()]
        except KeyError:
            raise TypeError(f'Cannot coerce {term!r} to type bool')
    return term

OutputCallable = Callable[[_Settings, OT], OT]
OnDefaultCallable = Callable[[_Settings], IT]

class Param(Generic[IT, OT], property):
    text_type: ClassVar[Tuple[Type[Any], ...]] = (Any,)
    default: IT = cast(IT, None)
    env_name: Optional[str] = None
    default_alias: Optional[str] = None
    default_template: Optional[str] = None
    allow_none: bool = False
    ignore_default: bool = False
    version_introduced: Optional[str] = None
    version_deprecated: Optional[str] = None
    deprecation_reason: Optional[str] = None
    version_changed: Optional[str] = None
    version_removed: Optional[str] = None
    deprecation_warning_template: str = DEPRECATION_WARNING_TEMPLATE
    deprecation_removal_warning: str = DEPRECATION_REMOVAL_WARNING
    section: _Section

    def __init__(
        self,
        *,
        name: str,
        env_name: Optional[str] = None,
        default: Optional[IT] = None,
        default_alias: Optional[str] = None,
        default_template: Optional[str] = None,
        allow_none: Optional[bool] = None,
        ignore_default: Optional[bool] = None,
        section: Optional[_Section] = None,
        version_introduced: Optional[str] = None,
        version_deprecated: Optional[str] = None,
        version_removed: Optional[str] = None,
        version_changed: Optional[str] = None,
        deprecation_reason: Optional[str] = None,
        related_cli_options: Optional[Dict[str, Any]] = None,
        related_settings: Optional[List[str]] = None,
        help: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        assert name
        self.name = name
        self.storage_name = f'_{name}'
        if env_name is not None:
            self.env_name = env_name
        if default is not None:
            self.default = default
        if default_alias is not None:
            self.default_alias = default_alias
        if default_template is not None:
            self.default_template = default_template
        if allow_none is not None:
            self.allow_none = allow_none
        if ignore_default is not None:
            self.ignore_default = ignore_default
        if section is not None:
            self.section = section
        assert self.section
        if version_introduced is not None:
            self.version_introduced = version_introduced
        if version_deprecated is not None:
            self.version_deprecated = version_deprecated
        if version_removed is not None:
            self.version_removed = version_removed
        if version_changed is not None:
            self.version_changed = version_changed
        if deprecation_reason is not None:
            self.deprecation_reason = deprecation_reason
        if help is not None:
            self.__doc__ = help
        self._on_get_value_: Optional[OutputCallable] = None
        self._on_set_default_: Optional[OnDefaultCallable] = None
        self.options: Dict[str, Any] = kwargs
        self.related_cli_options: Dict[str, Any] = related_cli_options or {}
        self.related_settings: List[str] = related_settings or []
        self._init_options(**self.options)
        if self.version_deprecated:
            assert self.deprecation_reason

    def _init_options(self, **kwargs: Any) -> None: ...

    def on_get_value(self, fun: OutputCallable) -> OutputCallable:
        self._on_get_value_ = fun
        return fun

    def on_set_default(self, fun: OnDefaultCallable) -> OnDefaultCallable:
        self._on_set_default_ = fun
        return fun

    def __get__(self, obj: Any, type: Optional[Type[Any]] = None) -> Any:
        if obj is None:
            return self
        if self.version_deprecated:
            warnings.warn(UserWarning(self.build_deprecation_warning()))
        return self.on_get(obj)

    def __set__(self, obj: Any, value: Any) -> None:
        self.on_set(obj, self.prepare_set(obj, value))

    def on_get(self, conf: _Settings) -> Any:
        value = getattr(conf, self.storage_name)
        if value is None and self.default_alias:
            retval = getattr(conf, self.default_alias)
        else:
            retval = self.prepare_get(conf, value)
        if self._on_get_value_ is not None:
            return self._on_get_value_(conf, retval)
        return retval

    def prepare_get(self, conf: _Settings, value: Any) -> Any: return value

    def on_set(self, settings: _Settings, value: Any) -> None:
        settings.__dict__[self.storage_name] = value
        assert getattr(settings, self.storage_name) == value

    def set_class_default(self, cls: Type[Any]) -> None:
        setattr(cls, self.storage_name, self.default)

    def on_init_set_value(self, conf: _Settings, provided_value: Any) -> None:
        if provided_value is not None:
            self.__set__(conf, provided_value)

    def on_init_set_default(self, conf: _Settings, provided_value: Any) -> None:
        if provided_value is None:
            default_value = self.default
            if self._on_set_default_:
                default_value = self._on_set_default_(conf)
            if default_value is None and self.default_template:
                default_value = self.default_template.format(conf=conf)
            setattr(conf, self.storage_name, self.prepare_init_default(conf, default_value))

    def build_deprecation_warning(self) -> str:
        alt_removal = ''
        if self.version_removed:
            alt_removal = self.deprecation_removal_warning.format(self=self)
        return self.deprecation_warning_template.format(self=self, alt_removal=alt_removal)

    def validate_before(self, value: Any = None) -> None: ...

    def validate_after(self, value: Any) -> None: ...

    def prepare_set(self, conf: _Settings, value: Any) -> Any:
        skip_validate = value is None and self.allow_none
        if not skip_validate:
            self.validate_before(value)
        if value is not None:
            new_value = self.to_python(conf, value)
        else:
            new_value = value
        if not skip_validate:
            self.validate_after(new_value)
        return new_value

    def prepare_init_default(self, conf: _Settings, value: Any) -> Any:
        if value is not None:
            return self.to_python(conf, value)
        return None

    def to_python(self, conf: _Settings, value: IT) -> OT:
        return cast(OT, value)

    @property
    def active(self) -> bool:
        return not bool(self.version_removed)

    @property
    def deprecated(self) -> bool:
        return bool(self.version_deprecated)

    @property
    def class_name(self) -> str:
        return type(self).__name__

class Bool(Param[Any, bool]):
    text_type = (bool,)

    def to_python(self, conf: _Settings, value: Any) -> bool:
        if isinstance(value, str):
            return to_bool(value)
        return bool(value)

class Str(Param[str, str]):
    text_type = (str,)

class Severity(Param[_Severity, _Severity]):
    text_type = (str, int)

class Number(Param[IT, OT]):
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    number_aliases: Dict[Any, Any]

    def _init_options(
        self, 
        min_value: Optional[int] = None, 
        max_value: Optional[int] = None, 
        number_aliases: Optional[Dict[Any, Any]] = None, 
        **kwargs: Any
    ) -> None:
        if min_value is not None:
            self.min_value = min_value
        if max_value is not None:
            self.max_value = max_value
        self.number_aliases = number_aliases or {}

    @abc.abstractmethod
    def convert(self, conf: _Settings, value: Any) -> Any: ...

    def to_python(self, conf: _Settings, value: Any) -> Any:
        try:
            return self.number_aliases[value]
        except KeyError:
            return self.convert(conf, value)

    def validate_after(self, value: Any) -> None:
        v = cast(int, value)
        min_ = self.min_value
        max_ = self.max_value
        if min_ is not None and v < min_:
            raise self._out_of_range(v)
        if max_ is not None and v > max_:
            raise self._out_of_range(v)

    def _out_of_range(self, value: Any) -> ImproperlyConfigured:
        return ImproperlyConfigured(
            f'Value {value} is out of range for {self.class_name} '
            f'(min={self.min_value} max={self.max_value})'
        )

NumberInputArg = Union[str, int, float]

class _Int(Number[IT, OT]):
    text_type = (int,)

    def convert(self, conf: _Settings, value: Any) -> Any:
        return cast(OT, int(cast(int, value)))

class Int(_Int[NumberInputArg, int]): pass

class UnsignedInt(_Int[NumberInputArg, int]):
    min_value = 0

class Version(Int):
    min_value = 1

class Port(UnsignedInt):
    min_value = 1
    max_value = 65535

class Seconds(Param[_Seconds, float]):
    text_type = (float, timedelta)

    def to_python(self, conf: _Settings, value: Any) -> float:
        return want_seconds(value)

class Credentials(Param[CredentialsArg, Optional[CredentialsT]]):
    text_type = (CredentialsT,)

    def to_python(self, conf: _Settings, value: Any) -> Optional[CredentialsT]:
        return to_credentials(value)

class SSLContext(Param[ssl.SSLContext, Optional[ssl.SSLContext]]):
    text_type = (ssl.SSLContext,)

class Dict(Param[DictArg[T], Mapping[str, T]]):
    text_type = (dict,)

    def to_python(self, conf: _Settings, value: Any) -> Mapping[str, T]:
        if isinstance(value, str):
            return json.loads(value)
        elif isinstance(value, Mapping):
            return value
        return dict(value)

class LogHandlers(Param[List[logging.Handler], List[logging.Handler]]):
    text_type = (List[logging.Handler],)

    def prepare_init_default(self, conf: _Settings, value: Any) -> List[logging.Handler]:
        return []

class Timezone(Param[Union[str, tzinfo], tzinfo]):
    text_type = (tzinfo,)
    builtin_timezones: Dict[str, tzinfo] = {'UTC': timezone.utc}

    def to_python(self, conf: _Settings, value: Any) -> tzinfo:
        if isinstance(value, str):
            try:
                return cast(tzinfo, self.builtin_timezones[value.lower()])
            except KeyError:
                import pytz
                return cast(tzinfo, pytz.timezone(value))
        else:
            return value

class BrokerList(Param[BrokerArg, List[_URL]]):
    text_type = (str, _URL, List[str])
    default_scheme = DEFAULT_BROKER_SCHEME

    def to_python(self, conf: _Settings, value: Any) -> List[_URL]:
        return self.broker_list(value)

    def broker_list(self, value: Any) -> List[_URL]:
        return urllist(value, default_scheme=self.default_scheme)

class URL(Param[URLArg, _URL]):
    text_type = (str, _URL)

    def to_python(self, conf: _Settings, value: Any) -> _URL:
        return _URL(value)

class Path(Param[Union[str, _Path], _Path]):
    text_type = (str, _Path)
    expanduser = True

    def to_python(self, conf: _Settings, value: Any) -> _Path:
        p = _Path(value)
        if self.expanduser:
            p = p.expanduser()
        return self.prepare_path(conf, p)

    def prepare_path(self, conf: _Settings, path: _Path) -> _Path:
        return path

class Codec(Param[CodecArg, CodecArg]):
    text_type = (str, CodecT)

def Enum(typ: Type[T]) -> Type[Param[Union[str, T], T]]:
    class EnumParam(Param[Union[str, T], T]):
        text_type = (str,)

        def to_python(self, conf: _Settings, value: Any) -> T:
            return typ(value)
    return EnumParam

class _Symbol(Param[IT, OT]):
    text_type = (str, Type)

    def to_python(self, conf: _Settings, value: Any) -> OT:
        return cast(OT, symbol_by_name(value))

def Symbol(typ: Type[T]) -> Type[_Symbol[SymbolArg[T], T]]:
    return _Symbol[SymbolArg[T], T]
