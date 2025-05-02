import abc
import logging
import ssl
import typing
import warnings
from datetime import timedelta, timezone, tzinfo
from pathlib import Path as _Path
from typing import Any, Callable, ClassVar, Generic, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union, cast
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
__all__ = ['AutodiscoverArg', 'DictArg', 'URLArg', 'BrokerArg', 'Param', 'Bool', 'Str', 'Severity', 'Int', 'UnsignedInt', 'Version', 'Port', 'Seconds', 'Credentials', 'SSLContext', 'Dict', 'LogHandlers', 'Timezone', 'BrokerList', 'URL', 'Path', 'Codec', 'Enum', 'Symbol', 'to_bool']
DEFAULT_BROKER_SCHEME = 'kafka'
T = TypeVar('T')
IT = TypeVar('IT')
OT = TypeVar('OT')
BOOLEAN_TERMS: Mapping[str, bool] = {'': False, 'false': False, 'no': False, '0': False, 'true': True, 'yes': True, '1': True, 'on': True, 'off': False}
AutodiscoverArg = Union[bool, Iterable[str], Callable[[], Iterable[str]]]
DictArg = Union[str, Mapping[str, T]]
URLArg = Union[str, _URL]
BrokerArg = URIListArg
DEPRECATION_WARNING_TEMPLATE = '\nSetting {self.name} is deprecated since Faust version {self.version_deprecated}: {self.deprecation_reason}. {alt_removal}\n'.strip()
DEPRECATION_REMOVAL_WARNING = '\nFurther the setting is scheduled to be removed in Faust version {self.version_removal}.\n'.strip()

def to_bool(term, *, table: Mapping[str, bool]=BOOLEAN_TERMS):
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
    """Faust setting description.

    Describes a Faust setting, how to read it from environment
    variables or from a configuration object.

    """
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
    related_cli_options: Mapping[str, List[str]]
    related_settings: List[Any]
    deprecation_warning_template: str = DEPRECATION_WARNING_TEMPLATE
    deprecation_removal_warning: str = DEPRECATION_REMOVAL_WARNING

    def __init__(self, *, name: str, env_name: str=None, default: IT=None, default_alias: str=None, default_template: str=None, allow_none: bool=None, ignore_default: bool=None, section: _Section=None, version_introduced: str=None, version_deprecated: str=None, version_removed: str=None, version_changed: Mapping[str, str]=None, deprecation_reason: str=None, related_cli_options: Mapping[str, List[str]]=None, related_settings: List[Any]=None, help: str=None, **kwargs: Any):
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
        self.options = kwargs
        self.related_cli_options = related_cli_options or {}
        self.related_settings = related_settings or []
        self._init_options(**self.options)
        if self.version_deprecated:
            assert self.deprecation_reason

    def _init_options(self, **kwargs: Any):
        """Use in subclasses to quickly override ``__init__``."""
        ...

    def on_get_value(self, fun):
        """Decorator that adds a callback when this setting is retrieved."""
        assert self._on_get_value_ is None
        self._on_get_value_ = fun
        return fun

    def on_set_default(self, fun):
        """Decorator that adds a callback when a default value is used."""
        assert self._on_set_default_ is None
        self._on_set_default_ = fun
        return fun

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        if self.version_deprecated:
            warnings.warn(UserWarning(self.build_deprecation_warning()))
        return self.on_get(obj)

    def __set__(self, obj, value):
        self.on_set(obj, self.prepare_set(obj, value))

    def on_get(self, conf):
        """What happens when the setting is accessed/retrieved."""
        value = getattr(conf, self.storage_name)
        if value is None and self.default_alias:
            retval = getattr(conf, self.default_alias)
        else:
            retval = self.prepare_get(conf, value)
        if self._on_get_value_ is not None:
            return self._on_get_value_(conf, retval)
        return retval

    def prepare_get(self, conf, value):
        """Prepare value when accessed/retrieved."""
        return value

    def on_set(self, settings, value):
        """What happens when the setting is stored/set."""
        settings.__dict__[self.storage_name] = value
        assert getattr(settings, self.storage_name) == value

    def set_class_default(self, cls):
        """Set class default value for storage attribute."""
        setattr(cls, self.storage_name, self.default)

    def on_init_set_value(self, conf, provided_value):
        """What happens at ``Settings.__init__`` to store provided value.

        Arguments:
            conf: Settings object.
            provided_value: Provided configuration value passed to
                            ``Settings.__init__`` or :const:`None` if not set.
        """
        if provided_value is not None:
            self.__set__(conf, provided_value)

    def on_init_set_default(self, conf, provided_value):
        """What happens at ``Settings.__init__`` to set default value.

        Arguments:
            conf: Settings object.
            provided_value: Provided configuration value passed to
                            ``Settings.__init__`` or :const:`None` if not set.
        """
        if provided_value is None:
            default_value = self.default
            if self._on_set_default_:
                default_value = self._on_set_default_(conf)
            if default_value is None and self.default_template:
                default_value = self.default_template.format(conf=conf)
            setattr(conf, self.storage_name, self.prepare_init_default(conf, default_value))

    def build_deprecation_warning(self):
        """Build deprecation warning for this setting."""
        alt_removal = ''
        if self.version_removed:
            alt_removal = self.deprecation_removal_warning.format(self=self)
        return self.deprecation_warning_template.format(self=self, alt_removal=alt_removal)

    def validate_before(self, value=None):
        """Validate value before setting is converted to the target type."""
        ...

    def validate_after(self, value):
        """Validate value after it has been converted to its target type."""
        ...

    def prepare_set(self, conf, value):
        """Prepare value for storage."""
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

    def prepare_init_default(self, conf, value):
        """Prepare default value for storage."""
        if value is not None:
            return self.to_python(conf, value)
        return None

    def to_python(self, conf, value):
        """Convert value in input type to its output type."""
        return cast(OT, value)

    @property
    def active(self):
        return not bool(self.version_removed)

    @property
    def deprecated(self):
        return bool(self.version_deprecated)

    @property
    def class_name(self):
        return type(self).__name__

class Bool(Param[Any, bool]):
    """Boolean setting type."""
    text_type = (bool,)

    def to_python(self, conf, value):
        """Convert given value to :class:`bool`."""
        if isinstance(value, str):
            return to_bool(value)
        return bool(value)

class Str(Param[str, str]):
    """String setting type."""
    text_type = (str,)

class Severity(Param[_Severity, _Severity]):
    """Logging severity setting type."""
    text_type = (str, int)

class Number(Param[IT, OT]):
    """Number setting type (baseclass for int/float)."""
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    number_aliases: Mapping[IT, OT]

    def _init_options(self, min_value=None, max_value=None, number_aliases=None, **kwargs: Any):
        if min_value is not None:
            self.min_value = min_value
        if max_value is not None:
            self.max_value = max_value
        self.number_aliases = number_aliases or {}

    @abc.abstractmethod
    def convert(self, conf, value):
        ...

    def to_python(self, conf, value):
        """Convert given value to number."""
        try:
            return self.number_aliases[value]
        except KeyError:
            return self.convert(conf, value)

    def validate_after(self, value):
        """Validate number value."""
        v = cast(int, value)
        min_ = self.min_value
        max_ = self.max_value
        if min_ is not None and v < min_:
            raise self._out_of_range(v)
        if max_ is not None and v > max_:
            raise self._out_of_range(v)

    def _out_of_range(self, value):
        return ImproperlyConfigured(f'Value {value} is out of range for {self.class_name} (min={self.min_value} max={self.max_value})')
NumberInputArg = Union[str, int, float]

class _Int(Number[IT, OT]):
    text_type = (int,)

    def convert(self, conf, value):
        """Convert given value to int."""
        return cast(OT, int(cast(int, value)))

class Int(_Int[NumberInputArg, int]):
    """Signed integer setting type."""

class UnsignedInt(_Int[NumberInputArg, int]):
    """Unsigned integer setting type."""
    min_value = 0

class Version(Int):
    """Version setting type.

    Versions must be greater than ``1``.
    """
    min_value = 1

class Port(UnsignedInt):
    """Network port setting type.

    Ports must be in the range 1-65535.
    """
    min_value = 1
    max_value = 65535

class Seconds(Param[_Seconds, float]):
    """Seconds setting type.

    Converts from :class:`float`/:class:`~datetime.timedelta` to
    :class:`float`.
    """
    text_type = (float, timedelta)

    def to_python(self, conf, value):
        return want_seconds(value)

class Credentials(Param[CredentialsArg, Optional[CredentialsT]]):
    """Authentication credentials setting type."""
    text_type = (CredentialsT,)

    def to_python(self, conf, value):
        return to_credentials(value)

class SSLContext(Param[ssl.SSLContext, Optional[ssl.SSLContext]]):
    """SSL context setting type."""
    text_type = (ssl.SSLContext,)

class Dict(Param[DictArg[T], Mapping[str, T]]):
    """Dictionary setting type."""
    text_type = (dict,)

    def to_python(self, conf, value):
        if isinstance(value, str):
            return json.loads(value)
        elif isinstance(value, Mapping):
            return value
        return dict(value)

class LogHandlers(Param[List[logging.Handler], List[logging.Handler]]):
    """Log handler list setting type."""
    text_type = (List[logging.Handler],)

    def prepare_init_default(self, conf, value):
        return []

class Timezone(Param[Union[str, tzinfo], tzinfo]):
    """Timezone setting type."""
    text_type = (tzinfo,)
    builtin_timezones = {'UTC': timezone.utc}

    def to_python(self, conf, value):
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
    text_type = (str, _URL, List[str])
    default_scheme = DEFAULT_BROKER_SCHEME

    def to_python(self, conf, value):
        return self.broker_list(value)

    def broker_list(self, value):
        return urllist(value, default_scheme=self.default_scheme)

class URL(Param[URLArg, _URL]):
    """URL setting type."""
    text_type = (str, _URL)

    def to_python(self, conf, value):
        return _URL(value)

class Path(Param[Union[str, _Path], _Path]):
    """Path setting type."""
    text_type = (str, _Path)
    expanduser: bool = True

    def to_python(self, conf, value):
        p = _Path(value)
        if self.expanduser:
            p = p.expanduser()
        return self.prepare_path(conf, p)

    def prepare_path(self, conf, path):
        return path

class Codec(Param[CodecArg, CodecArg]):
    """Serialization codec setting type."""
    text_type = (str, CodecT)

def Enum(typ):
    """Generate new enum setting type."""

    class EnumParam(Param[Union[str, T], T]):
        text_type = (str,)

        def to_python(self, conf, value):
            return typ(value)
    return EnumParam

class _Symbol(Param[IT, OT]):
    text_type = (str, Type)

    def to_python(self, conf, value):
        return cast(OT, symbol_by_name(value))

def Symbol(typ):
    """Generate new symbol setting type."""
    return _Symbol[SymbolArg[T], T]