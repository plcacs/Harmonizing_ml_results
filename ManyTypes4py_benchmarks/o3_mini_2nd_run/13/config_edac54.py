#!/usr/bin/env python
"""
The config module holds package-wide configurables and provides
a uniform API for working with them.

Overview
========

This module supports the following requirements:
- options are referenced using keys in dot.notation, e.g. "x.y.option - z".
- keys are case-insensitive.
- functions should accept partial/regex keys, when unambiguous.
- options can be registered by modules at import time.
- options can be registered at init-time (via core.config_init)
- options have a default value, and (optionally) a description and
  validation function associated with them.
- options can be deprecated, in which case referencing them
  should produce a warning.
- deprecated options can optionally be rerouted to a replacement
  so that accessing a deprecated option reroutes to a differently
  named option.
- options can be reset to their default value.
- all option can be reset to their default value at once.
- all options in a certain sub - namespace can be reset at once.
- the user can set / get / reset or ask for the description of an option.
- a developer can register and mark an option as deprecated.
- you can register a callback to be invoked when the option value
  is set or reset. Changing the stored value is considered misuse, but
  is not verboten.

Implementation
==============

- Data is stored using nested dictionaries, and should be accessed
  through the provided API.

- "Registered options" and "Deprecated options" have metadata associated
  with them, which are stored in auxiliary dictionaries keyed on the
  fully-qualified key, e.g. "x.y.z.option".

- the config_init module is imported by the package's __init__.py file.
  placing any register_option() calls there will ensure those options
  are available as soon as pandas is loaded. If you use register_option
  in a module, it will only be available after that module is imported,
  which you should be aware of.

- `config_prefix` is a context_manager (for use with the `with` keyword)
  which can save developers some typing, see the docstring.

"""
from __future__ import annotations
from contextlib import contextmanager
import re
from typing import TYPE_CHECKING, Any, Callable, Generator, List, Optional, Tuple, cast
import warnings
from pandas._typing import F
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from collections.abc import Sequence

class DeprecatedOption(NamedTuple):
    key: str
    msg: Optional[str]
    rkey: Optional[str]
    removal_ver: Optional[str]

class RegisteredOption(NamedTuple):
    key: str
    defval: Any
    doc: str
    validator: Optional[Callable[[Any], None]]
    cb: Optional[Callable[[str], None]]

_deprecated_options: dict[str, DeprecatedOption] = {}
_registered_options: dict[str, RegisteredOption] = {}
_global_config: dict[str, Any] = {}
_reserved_keys: List[str] = ['all']

class OptionError(AttributeError, KeyError):
    """
    Exception raised for pandas.options.

    Backwards compatible with KeyError checks.

    See Also
    --------
    options : Access and modify global pandas settings.

    Examples
    --------
    >>> pd.options.context
    Traceback (most recent call last):
    OptionError: No such option
    """

def _get_single_key(pat: str) -> str:
    keys: List[str] = _select_options(pat)
    if len(keys) == 0:
        _warn_if_deprecated(pat)
        raise OptionError(f'No such keys(s): {pat!r}')
    if len(keys) > 1:
        raise OptionError('Pattern matched multiple keys')
    key: str = keys[0]
    _warn_if_deprecated(key)
    key = _translate_key(key)
    return key

def get_option(pat: str) -> Any:
    """
    Retrieve the value of the specified option.

    ...
    """
    key: str = _get_single_key(pat)
    root, k = _get_root(key)
    return root[k]

def set_option(*args: Any) -> None:
    """
    Set the value of the specified option or options.

    ...
    """
    nargs: int = len(args)
    if not nargs or nargs % 2 != 0:
        raise ValueError('Must provide an even number of non-keyword arguments')
    for k, v in zip(args[::2], args[1::2]):
        key: str = _get_single_key(k)
        opt: Optional[RegisteredOption] = _get_registered_option(key)
        if opt and opt.validator:
            opt.validator(v)
        root, k_root = _get_root(key)
        root[k_root] = v
        if opt and opt.cb:
            opt.cb(key)

def describe_option(pat: str = '', _print_desc: bool = True) -> Optional[str]:
    """
    Print the description for one or more registered options.

    ...
    """
    keys: List[str] = _select_options(pat)
    if len(keys) == 0:
        raise OptionError(f'No such keys(s) for pat={pat!r}')
    s: str = '\n'.join([_build_option_description(k) for k in keys])
    if _print_desc:
        print(s)
        return None
    return s

def reset_option(pat: str) -> None:
    """
    Reset one or more options to their default value.

    ...
    """
    keys: List[str] = _select_options(pat)
    if len(keys) == 0:
        raise OptionError(f'No such keys(s) for pat={pat!r}')
    if len(keys) > 1 and len(pat) < 4 and (pat != 'all'):
        raise ValueError('You must specify at least 4 characters when resetting multiple keys, use the special keyword "all" to reset all the options to their default value')
    for k in keys:
        set_option(k, _registered_options[k].defval)

def get_default_val(pat: str) -> Any:
    key: str = _get_single_key(pat)
    return _get_registered_option(key).defval

class DictWrapper:
    """provide attribute-style access to a nested dict"""

    def __init__(self, d: dict[str, Any], prefix: str = '') -> None:
        object.__setattr__(self, 'd', d)
        object.__setattr__(self, 'prefix', prefix)

    def __setattr__(self, key: str, val: Any) -> None:
        prefix: str = object.__getattribute__(self, 'prefix')
        if prefix:
            prefix += '.'
        prefix += key
        if key in self.d and (not isinstance(self.d[key], dict)):
            set_option(prefix, val)
        else:
            raise OptionError('You can only set the value of existing options')

    def __getattr__(self, key: str) -> Any:
        prefix: str = object.__getattribute__(self, 'prefix')
        if prefix:
            prefix += '.'
        prefix += key
        try:
            v: Any = object.__getattribute__(self, 'd')[key]
        except KeyError as err:
            raise OptionError('No such option') from err
        if isinstance(v, dict):
            return DictWrapper(v, prefix)
        else:
            return get_option(prefix)

    def __dir__(self) -> List[str]:
        return list(self.d.keys())

options: DictWrapper = DictWrapper(_global_config)

@contextmanager
def option_context(*args: Any) -> Generator[None, None, None]:
    """
    Context manager to temporarily set options in a ``with`` statement.

    ...
    """
    if len(args) % 2 != 0 or len(args) < 2:
        raise ValueError('Provide an even amount of arguments as option_context(pat, val, pat, val...).')
    ops: Tuple[Tuple[Any, Any], ...] = tuple(zip(args[::2], args[1::2]))
    try:
        undo: Tuple[Tuple[str, Any], ...] = tuple(((pat, get_option(pat)) for pat, _ in ops))
        for pat, val in ops:
            set_option(pat, val)
        yield
    finally:
        for pat, val in undo:
            set_option(pat, val)

def register_option(key: str, defval: Any, doc: str = '', 
                    validator: Optional[Callable[[Any], None]] = None, 
                    cb: Optional[Callable[[str], None]] = None) -> None:
    """
    Register an option in the package-wide pandas config object

    ...
    """
    import keyword
    import tokenize
    key = key.lower()
    if key in _registered_options:
        raise OptionError(f"Option '{key}' has already been registered")
    if key in _reserved_keys:
        raise OptionError(f"Option '{key}' is a reserved key")
    if validator:
        validator(defval)
    path: List[str] = key.split('.')
    for k in path:
        if not re.match('^' + tokenize.Name + '$', k):
            raise ValueError(f'{k} is not a valid identifier')
        if keyword.iskeyword(k):
            raise ValueError(f'{k} is a python keyword')
    cursor: Any = _global_config
    msg: str = "Path prefix to option '{option}' is already an option"
    for i, p in enumerate(path[:-1]):
        if not isinstance(cursor, dict):
            raise OptionError(msg.format(option='.'.join(path[:i])))
        if p not in cursor:
            cursor[p] = {}
        cursor = cursor[p]
    if not isinstance(cursor, dict):
        raise OptionError(msg.format(option='.'.join(path[:-1])))
    cursor[path[-1]] = defval
    _registered_options[key] = RegisteredOption(key=key, defval=defval, doc=doc, validator=validator, cb=cb)

def deprecate_option(key: str, msg: Optional[str] = None, rkey: Optional[str] = None, removal_ver: Optional[str] = None) -> None:
    """
    Mark option `key` as deprecated, if code attempts to access this option,
    a warning will be produced, using `msg` if given, or a default message
    if not.
    if `rkey` is given, any access to the key will be re-routed to `rkey`.

    ...
    """
    key = key.lower()
    if key in _deprecated_options:
        raise OptionError(f"Option '{key}' has already been defined as deprecated.")
    _deprecated_options[key] = DeprecatedOption(key, msg, rkey, removal_ver)

def _select_options(pat: str) -> List[str]:
    """
    returns a list of keys matching `pat`

    if pat=="all", returns all registered options
    """
    if pat in _registered_options:
        return [pat]
    keys: List[str] = sorted(_registered_options.keys())
    if pat == 'all':
        return keys
    return [k for k in keys if re.search(pat, k, re.I)]

def _get_root(key: str) -> Tuple[dict[str, Any], str]:
    path: List[str] = key.split('.')
    cursor: Any = _global_config
    for p in path[:-1]:
        cursor = cursor[p]
    return (cursor, path[-1])

def _get_deprecated_option(key: str) -> Optional[DeprecatedOption]:
    """
    Retrieves the metadata for a deprecated option, if `key` is deprecated.

    Returns
    -------
    DeprecatedOption (namedtuple) if key is deprecated, None otherwise
    """
    try:
        d: DeprecatedOption = _deprecated_options[key]
    except KeyError:
        return None
    else:
        return d

def _get_registered_option(key: str) -> Optional[RegisteredOption]:
    """
    Retrieves the option metadata if `key` is a registered option.

    Returns
    -------
    RegisteredOption (namedtuple) if key is registered, None otherwise
    """
    return _registered_options.get(key)

def _translate_key(key: str) -> str:
    """
    if key is deprecated and a replacement key defined, will return the
    replacement key, otherwise returns `key` as-is
    """
    d: Optional[DeprecatedOption] = _get_deprecated_option(key)
    if d:
        return d.rkey or key
    else:
        return key

def _warn_if_deprecated(key: str) -> bool:
    """
    Checks if `key` is a deprecated option and if so, prints a warning.

    Returns
    -------
    bool - True if `key` is deprecated, False otherwise.
    """
    d: Optional[DeprecatedOption] = _get_deprecated_option(key)
    if d:
        if d.msg:
            warnings.warn(d.msg, FutureWarning, stacklevel=find_stack_level())
        else:
            msg: str = f"'{key}' is deprecated"
            if d.removal_ver:
                msg += f' and will be removed in {d.removal_ver}'
            if d.rkey:
                msg += f", please use '{d.rkey}' instead."
            else:
                msg += ', please refrain from using it.'
            warnings.warn(msg, FutureWarning, stacklevel=find_stack_level())
        return True
    return False

def _build_option_description(k: str) -> str:
    """Builds a formatted description of a registered option and prints it"""
    o: Optional[RegisteredOption] = _get_registered_option(k)
    d: Optional[DeprecatedOption] = _get_deprecated_option(k)
    s: str = f'{k} '
    if o and o.doc:
        s += '\n'.join(o.doc.strip().split('\n'))
    else:
        s += 'No description available.'
    if o:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            warnings.simplefilter('ignore', DeprecationWarning)
            s += f'\n    [default: {o.defval}] [currently: {get_option(k)}]'
    if d:
        rkey: str = d.rkey or ''
        s += '\n    (Deprecated'
        s += f', use `{rkey}` instead.'
        s += ')'
    return s

@contextmanager
def config_prefix(prefix: str) -> Generator[None, None, None]:
    """
    contextmanager for multiple invocations of API with a common prefix

    supported API functions: (register / get / set) __option

    Warning: This is not thread-safe, and won't work properly if you import
    the API functions into your module using the "from x import y" construct.

    Example
    -------
    import pandas._config.config as cf
    with cf.config_prefix("display.font"):
        cf.register_option("color", "red")
        cf.register_option("size", " 5 pt")
        cf.set_option("size", " 6 pt")
        cf.get_option("size")
        ...
    """
    global register_option, get_option, set_option

    def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        def inner(key: str, *args: Any, **kwds: Any) -> Any:
            pkey: str = f'{prefix}.{key}'
            return func(pkey, *args, **kwds)
        return cast(F, inner)
    _register_option: Callable[..., Any] = register_option
    _get_option: Callable[..., Any] = get_option
    _set_option: Callable[..., Any] = set_option
    set_option = wrap(set_option)
    get_option = wrap(get_option)
    register_option = wrap(register_option)
    try:
        yield
    finally:
        set_option = _set_option
        get_option = _get_option
        register_option = _register_option

def is_type_factory(_type: type) -> Callable[[Any], None]:
    """
    Parameters
    ----------
    _type: type
        A type to be compared against.

    Returns
    -------
    Callable[[Any], None]
        A validator function which raises ValueError if type(x) is not equal to `_type`.
    """
    def inner(x: Any) -> None:
        if type(x) != _type:
            raise ValueError(f"Value must have type '{_type}'")
    return inner

def is_instance_factory(_type: type | Tuple[type, ...]) -> Callable[[Any], None]:
    """
    Parameters
    ----------
    _type: type or tuple of types
        The type(s) to be checked against.

    Returns
    -------
    Callable[[Any], None]
        A validator function which raises ValueError if x is not an instance of `_type`.
    """
    if isinstance(_type, tuple):
        type_repr: str = '|'.join(map(str, _type))
    else:
        type_repr = f"'{_type}'"
    def inner(x: Any) -> None:
        if not isinstance(x, _type):
            raise ValueError(f'Value must be an instance of {type_repr}')
    return inner

def is_one_of_factory(legal_values: List[Any]) -> Callable[[Any], None]:
    callables = [c for c in legal_values if callable(c)]
    legal_non_callable: List[Any] = [c for c in legal_values if not callable(c)]
    def inner(x: Any) -> None:
        if x not in legal_non_callable:
            if not any((c(x) for c in callables)):
                uvals: List[str] = [str(lval) for lval in legal_non_callable]
                pp_values: str = '|'.join(uvals)
                msg: str = f'Value must be one of {pp_values}'
                if len(callables):
                    msg += ' or a callable'
                raise ValueError(msg)
    return inner

def is_nonnegative_int(value: Optional[int]) -> None:
    """
    Verify that value is None or a nonnegative int.

    Parameters
    ----------
    value : None or int
        The value to be checked.

    Raises
    ------
    ValueError
        When the value is not None or is a negative integer.
    """
    if value is None:
        return
    elif isinstance(value, int):
        if value >= 0:
            return
    msg: str = 'Value must be a nonnegative integer or None'
    raise ValueError(msg)

is_int: Callable[[Any], None] = is_type_factory(int)
is_bool: Callable[[Any], None] = is_type_factory(bool)
is_float: Callable[[Any], None] = is_type_factory(float)
is_str: Callable[[Any], None] = is_type_factory(str)
is_text: Callable[[Any], None] = is_instance_factory((str, bytes))

def is_callable(obj: Any) -> bool:
    """
    Parameters
    ----------
    obj : Any
        The object to be checked.

    Returns
    -------
    bool
         True if object is callable, raises ValueError otherwise.
    """
    if not callable(obj):
        raise ValueError('Value must be a callable')
    return True
