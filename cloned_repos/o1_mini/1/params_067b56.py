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

    class _Section:
        ...

    class _Settings:
        ...


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

DEFAULT_BROKER_SCHEME: str = 'kafka'

T = TypeVar('T')
IT = TypeVar('IT')
OT = TypeVar('OT')

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

AutodiscoverArg = Union[bool, Iterable[str], Callable[[], Iterable[str]]]
DictArg = Union[str, Mapping[str, T]]
URLArg = Union[str, _URL]
BrokerArg = URIListArg

DEPRECATION_WARNING_TEMPLATE: str = '\nSetting {self.name} is deprecated since Faust version {self.version_deprecated}: {self.deprecation_reason}. {alt_removal}\n'.strip()
DEPRECATION_REMOVAL_WARNING: str = '\nFurther the setting is scheduled to be removed in Faust version {self.version_removal}.\n'.strip()


def to_bool(term: Union[str, bool], *, table: Optional[Mapping[str, bool]] = None) -> bool:
    """Convert common terms for true/false to bool.

    Examples (true/false/yes/no/on/off/1/0).
    """
    if table is None:
        table = BOOLEAN_TERMS
    if isinstance(term, str):
        try:
            return table[term.lower()]
        except KeyError:
            raise TypeError(f'Cannot coerce {term!r} to type bool')
    return cast(bool, term)


OutputCallable = Callable[[_Settings, OT], OT]
OnDefaultCallable = Callable[[_Settings], IT]


class Param(Generic[IT, OT], property):
    """Faust setting description.

    Describes a Faust setting, how to read it from environment
    variables or from a configuration object.

    """
    text_type: Tuple[Any, ...] = (Any,)
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
        section: Optional[str] = None,
        version_introduced: Optional[str] = None,
        version_deprecated: Optional[str] = None,
        version_removed: Optional[str] = None,
        version_changed: Optional[str] = None,
        deprecation_reason: Optional[str] = None,
        related_cli_options: Optional[Mapping[str, Any]] = None,
        related_settings: Optional[List[str]] = None,
        help: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        assert name
        self.name: str = name
        self.storage_name: str = f'_{name}'
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
        self.options: Mapping[str, Any] = kwargs
        self.related_cli_options: Mapping[str, Any] = related_cli_options or {}
        self.related_settings: List[str] = related_settings or []
        self._init_options(**self.options)
        if self.version_deprecated:
            assert self.deprecation_reason

    def _init_options(self, **kwargs: Any) -> None:
        """Use in subclasses to quickly override ``__init__``."""
        ...

    def on_get_value(self, fun: OutputCallable) -> OutputCallable:
        """Decorator that adds a callback when this setting is retrieved."""
        assert self._on_get_value_ is None
        self._on_get_value_ = fun
        return fun

    def on_set_default(self, fun: OnDefaultCallable) -> OnDefaultCallable:
        """Decorator that adds a callback when a default value is used."""
        assert self._on_set_default_ is None
        self._on_set_default_ = fun
        return fun

    def __get__(self, obj: Any, type: Optional[Type[Any]] = None) -> OT:
        if obj is None:
            return cast(OT, self)
        if self.version_deprecated:
            warnings.warn(UserWarning(self.build_deprecation_warning()))
        return self.on_get(obj)

    def __set__(self, obj: Any, value: IT) -> None:
        self.on_set(obj, self.prepare_set(obj, value))

    def on_get(self, conf: _Settings) -> OT:
        """What happens when the setting is accessed/retrieved."""
        value: Optional[IT] = getattr(conf, self.storage_name)
        if value is None and self.default_alias:
            retval: IT = getattr(conf, self.default_alias)
        else:
            retval = self.prepare_get(conf, value)
        if self._on_get_value_ is not None:
            return self._on_get_value_(conf, retval)
        return cast(OT, retval)

    def prepare_get(self, conf: _Settings, value: Optional[IT]) -> IT:
        """Prepare value when accessed/retrieved."""
        return cast(IT, value)

    def on_set(self, settings: _Settings, value: OT) -> None:
        """What happens when the setting is stored/set."""
        settings.__dict__[self.storage_name] = value
        assert getattr(settings, self.storage_name) == value

    def set_class_default(self, cls: Type[Any]) -> None:
        """Set class default value for storage attribute."""
        setattr(cls, self.storage_name, self.default)

    def on_init_set_value(self, conf: _Settings, provided_value: Optional[IT]) -> None:
        """What happens at ``Settings.__init__`` to store provided value.

        Arguments:
            conf: Settings object.
            provided_value: Provided configuration value passed to
                            ``Settings.__init__`` or :const:`None` if not set.
        """
        if provided_value is not None:
            self.__set__(conf, provided_value)

    def on_init_set_default(self, conf: _Settings, provided_value: Optional[IT]) -> None:
        """What happens at ``Settings.__init__`` to set default value.

        Arguments:
            conf: Settings object.
            provided_value: Provided configuration value passed to
                            ``Settings.__init__`` or :const:`None` if not set.
        """
        if provided_value is None:
            default_value: Optional[IT] = self.default
            if self._on_set_default_:
                default_value = self._on_set_default_(conf)
            if default_value is None and self.default_template:
                default_value = self.default_template.format(conf=conf)
            setattr(conf, self.storage_name, self.prepare_init_default(conf, default_value))

    def build_deprecation_warning(self) -> str:
        """Build deprecation warning for this setting."""
        alt_removal: str = ''
        if self.version_removed:
            alt_removal = self.deprecation_removal_warning.format(self=self)
        return self.deprecation_warning_template.format(self=self, alt_removal=alt_removal)

    def validate_before(self, value: Optional[IT] = None) -> None:
        """Validate value before setting is converted to the target type."""
        ...

    def validate_after(self, value: OT) -> None:
        """Validate value after it has been converted to its target type."""
        ...

    def prepare_set(self, conf: _Settings, value: IT) -> OT:
        """Prepare value for storage."""
        skip_validate: bool = value is None and self.allow_none
        if not skip_validate:
            self.validate_before(value)
        if value is not None:
            new_value: OT = self.to_python(conf, value)
        else:
            new_value = cast(OT, value)
        if not skip_validate:
            self.validate_after(new_value)
        return new_value

    def prepare_init_default(self, conf: _Settings, value: Optional[IT]) -> Optional[OT]:
        """Prepare default value for storage."""
        if value is not None:
            return self.to_python(conf, value)
        return cast(Optional[OT], None)

    def to_python(self, conf: _Settings, value: IT) -> OT:
        """Convert value in input type to its output type."""
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
    """Boolean setting type."""
    text_type: Tuple[Any, ...] = (bool,)

    def to_python(self, conf: _Settings, value: Union[str, bool]) -> bool:
        """Convert given value to :class:`bool`."""
        if isinstance(value, str):
            return to_bool(value)
        return bool(value)


class Str(Param[str, str]):
    """String setting type."""
    text_type: Tuple[Any, ...] = (str,)


class Severity(Param[_Severity, _Severity]):
    """Logging severity setting type."""
    text_type: Tuple[Any, ...] = (str, int)


class Number(Param[IT, OT]):
    """Number setting type (baseclass for int/float)."""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None

    def _init_options(
        self,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        number_aliases: Optional[Mapping[Union[str, int, float], OT]] = None,
        **kwargs: Any,
    ) -> None:
        if min_value is not None:
            self.min_value = min_value
        if max_value is not None:
            self.max_value = max_value
        self.number_aliases: Mapping[Union[str, int, float], OT] = number_aliases or {}

    @abc.abstractmethod
    def convert(self, conf: _Settings, value: IT) -> OT:
        ...

    def to_python(self, conf: _Settings, value: IT) -> OT:
        """Convert given value to number."""
        try:
            return self.number_aliases[value]
        except KeyError:
            return self.convert(conf, value)

    def validate_after(self, value: OT) -> None:
        """Validate number value."""
        v = cast(Union[int, float], value)
        min_ = self.min_value
        max_ = self.max_value
        if min_ is not None and v < min_:
            raise self._out_of_range(v)
        if max_ is not None and v > max_:
            raise self._out_of_range(v)

    def _out_of_range(self, value: Union[int, float]) -> ImproperlyConfigured:
        return ImproperlyConfigured(
            f'Value {value} is out of range for {self.class_name} (min={self.min_value} max={self.max_value})'
        )


NumberInputArg = Union[str, int, float]


class _Int(Number[NumberInputArg, int]):
    text_type: Tuple[Any, ...] = (int,)

    def convert(self, conf: _Settings, value: NumberInputArg) -> int:
        """Convert given value to int."""
        return cast(int, int(cast(int, value)))


class Int(_Int):
    """Signed integer setting type."""
    pass


class UnsignedInt(_Int):
    """Unsigned integer setting type."""
    min_value: Union[int, float] = 0


class Version(Int):
    """Version setting type.

    Versions must be greater than ``1``.
    """
    min_value: Union[int, float] = 1


class Port(UnsignedInt):
    """Network port setting type.

    Ports must be in the range 1-65535.
    """
    min_value: Union[int, float] = 1
    max_value: Union[int, float] = 65535


class Seconds(Param[_Seconds, float]):
    """Seconds setting type.

    Converts from :class:`float`/:class:`~datetime.timedelta` to
    :class:`float`.
    """
    text_type: Tuple[Any, ...] = (float, timedelta)

    def to_python(self, conf: _Settings, value: Union[float, timedelta]) -> float:
        return want_seconds(value)


class Credentials(Param[CredentialsArg, Optional[CredentialsT]]):
    """Authentication credentials setting type."""
    text_type: Tuple[Any, ...] = (CredentialsT,)

    def to_python(self, conf: _Settings, value: CredentialsArg) -> Optional[CredentialsT]:
        return to_credentials(value)


class SSLContext(Param[ssl.SSLContext, Optional[ssl.SSLContext]]):
    """SSL context setting type."""
    text_type: Tuple[Any, ...] = (ssl.SSLContext,)


class Dict(Param[DictArg[T], Mapping[str, T]]):
    """Dictionary setting type."""
    text_type: Tuple[Any, ...] = (dict,)

    def to_python(self, conf: _Settings, value: DictArg[T]) -> Mapping[str, T]:
        if isinstance(value, str):
            return json.loads(value)
        elif isinstance(value, Mapping):
            return value
        return dict(value)


class LogHandlers(Param[List[logging.Handler], List[logging.Handler]]):
    """Log handler list setting type."""
    text_type: Tuple[Any, ...] = (List[logging.Handler],)

    def prepare_init_default(self, conf: _Settings, value: Optional[List[logging.Handler]]) -> List[logging.Handler]:
        return []


class Timezone(Param[Union[str, tzinfo], tzinfo]):
    """Timezone setting type."""
    text_type: Tuple[Any, ...] = (tzinfo,)
    builtin_timezones: Mapping[str, tzinfo] = {'utc': timezone.utc}

    def to_python(self, conf: _Settings, value: Union[str, tzinfo]) -> tzinfo:
        if isinstance(value, str):
            try:
                return cast(tzinfo, self.builtin_timezones[value.lower()])
            except KeyError:
                import pytz

                return cast(tzinfo, pytz.timezone(value))
        else:
            return value


class BrokerList(Param[BrokerArg, List[_URL]]):
    """Broker URL list setting type."""
    text_type: Tuple[Any, ...] = (str, _URL, List[str])
    default_scheme: str = DEFAULT_BROKER_SCHEME

    def to_python(self, conf: _Settings, value: BrokerArg) -> List[_URL]:
        return self.broker_list(value)

    def broker_list(self, value: BrokerArg) -> List[_URL]:
        return urllist(value, default_scheme=self.default_scheme)


class URL(Param[URLArg, _URL]):
    """URL setting type."""
    text_type: Tuple[Any, ...] = (str, _URL)

    def to_python(self, conf: _Settings, value: URLArg) -> _URL:
        return _URL(value)


class Path(Param[Union[str, _Path], _Path]):
    """Path setting type."""
    text_type: Tuple[Any, ...] = (str, _Path)
    expanduser: bool = True

    def to_python(self, conf: _Settings, value: Union[str, _Path]) -> _Path:
        p = _Path(value)
        if self.expanduser:
            p = p.expanduser()
        return self.prepare_path(conf, p)

    def prepare_path(self, conf: _Settings, path: _Path) -> _Path:
        return path


class Codec(Param[CodecArg, CodecArg]):
    """Serialization codec setting type."""
    text_type: Tuple[Any, ...] = (str, CodecT)


def Enum(typ: Type[T]) -> Type[Param[Union[str, T], T]]:
    """Generate new enum setting type."""

    class EnumParam(Param[Union[str, T], T]):
        text_type: Tuple[Any, ...] = (str,)

        def to_python(self, conf: _Settings, value: Union[str, T]) -> T:
            return typ(value)

    return EnumParam


class _Symbol(Param[IT, OT]):
    text_type: Tuple[Any, ...] = (str, Type)

    def to_python(self, conf: _Settings, value: IT) -> OT:
        return cast(OT, symbol_by_name(value))


def Symbol(typ: Type[T]) -> Type[Param[SymbolArg[T], T]]:
    """Generate new symbol setting type."""
    return _Symbol[SymbolArg[T], T]
