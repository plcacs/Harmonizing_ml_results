from __future__ import annotations
from contextlib import contextmanager
import re
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, cast
import warnings
from pandas._typing import F
from pandas.util._exceptions import find_stack_level

# Define NamedTuple classes with annotations
class DeprecatedOption(Tuple[str, Optional[str], Optional[str], Optional[str]]):
    key: str
    msg: Optional[str]
    rkey: Optional[str]
    removal_ver: Optional[str]

    def __new__(cls, key: str, msg: Optional[str], rkey: Optional[str], removal_ver: Optional[str]) -> DeprecatedOption:
        return super().__new__(cls, key, msg, rkey, removal_ver)


class RegisteredOption(Tuple[str, Any, str, Optional[Callable[[Any], None]], Optional[Callable[[str], None]]]):
    key: str
    defval: Any
    doc: str
    validator: Optional[Callable[[Any], None]]
    cb: Optional[Callable[[str], None]]

    def __new__(
        cls,
        key: str,
        defval: Any,
        doc: str,
        validator: Optional[Callable[[Any], None]],
        cb: Optional[Callable[[str], None]]
    ) -> RegisteredOption:
        return super().__new__(cls, key, defval, doc, validator, cb)


_deprecated_options: Dict[str, DeprecatedOption] = {}
_registered_options: Dict[str, RegisteredOption] = {}
_global_config: Dict[str, Any] = {}
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
    """
    key: str = _get_single_key(pat)
    root, k = _get_root(key)
    return root[k]


def set_option(*args: Any) -> None:
    """
    Set the value of the specified option or options.
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
    """
    keys: List[str] = _select_options(pat)
    if len(keys) == 0:
        raise OptionError(f'No such keys(s) for pat={pat!r}')
    if len(keys) > 1 and len(pat) < 4 and (pat != 'all'):
        raise ValueError('You must specify at least 4 characters when resetting multiple keys, use the special keyword "all" to reset all the options to their default value')
    for k in keys:
        reg_opt: RegisteredOption = _registered_options[k]
        set_option(k, reg_opt.defval)


def get_default_val(pat: str) -> Any:
    key: str = _get_single_key(pat)
    reg_opt: RegisteredOption = _get_registered_option(key)  # type: ignore
    return reg_opt.defval


class DictWrapper:
    """provide attribute-style access to a nested dict"""

    def __init__(self, d: Dict[str, Any], prefix: str = '') -> None:
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
    """
    if len(args) % 2 != 0 or len(args) < 2:
        raise ValueError('Provide an even amount of arguments as option_context(pat, val, pat, val...).')
    ops: Tuple[Tuple[Any, Any], ...] = tuple(zip(args[::2], args[1::2]))
    try:
        undo: Tuple[Tuple[str, Any], ...] = tuple(((pat, get_option(pat)) for pat, val in ops))  # type: ignore
        for pat, val in ops:
            set_option(pat, val)
        yield
    finally:
        for pat, val in undo:
            set_option(pat, val)


def register_option(key: str, defval: Any, doc: str = '', validator: Optional[Callable[[Any], None]] = None, cb: Optional[Callable[[str], None]] = None) -> None:
    """
    Register an option in the package-wide pandas config object
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
    cursor: Dict[str, Any] = _global_config
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
    _registered_options[key] = RegisteredOption(key, defval, doc, validator, cb)


def deprecate_option(key: str, msg: Optional[str] = None, rkey: Optional[str] = None, removal_ver: Optional[str] = None) -> None:
    """
    Mark option `key` as deprecated.
    """
    key = key.lower()
    if key in _deprecated_options:
        raise OptionError(f"Option '{key}' has already been defined as deprecated.")
    _deprecated_options[key] = DeprecatedOption(key, msg, rkey, removal_ver)


def _select_options(pat: str) -> List[str]:
    """
    returns a list of keys matching `pat`
    """
    if pat in _registered_options:
        return [pat]
    keys: List[str] = sorted(_registered_options.keys())
    if pat == 'all':
        return keys
    return [k for k in keys if re.search(pat, k, re.I)]


def _get_root(key: str) -> Tuple[Dict[str, Any], str]:
    path: List[str] = key.split('.')
    cursor: Dict[str, Any] = _global_config
    for p in path[:-1]:
        cursor = cursor[p]
    return (cursor, path[-1])


def _get_deprecated_option(key: str) -> Optional[DeprecatedOption]:
    """
    Retrieves the metadata for a deprecated option.
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
    """
    return _registered_options.get(key)


def _translate_key(key: str) -> str:
    """
    if key is deprecated and a replacement key defined, will return the
    replacement key, otherwise returns `key` as-is.
    """
    d: Optional[DeprecatedOption] = _get_deprecated_option(key)
    if d:
        return d.rkey or key
    else:
        return key


def _warn_if_deprecated(key: str) -> bool:
    """
    Checks if `key` is a deprecated option and if so, prints a warning.
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
    """
    global register_option, get_option, set_option

    def wrap(func: Callable[..., Any]) -> F:
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


def is_type_factory(_type: Type[Any]) -> Callable[[Any], None]:
    """
    Parameters
    ----------
    _type : type
        the type that a value must exactly match.
    Returns
    -------
    validator - a function that raises ValueError if the type doesn't match.
    """
    def inner(x: Any) -> None:
        if type(x) != _type:
            raise ValueError(f"Value must have type '{_type}'")
    return inner


def is_instance_factory(_type: Any) -> Callable[[Any], None]:
    """
    Parameters
    ----------
    _type : type or tuple of types
        the type(s) to be checked.
    Returns
    -------
    validator - a function that raises ValueError if x is not an instance of _type.
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
    legal_values_non_callable = [c for c in legal_values if not callable(c)]

    def inner(x: Any) -> None:
        if x not in legal_values_non_callable:
            if not any((c(x) for c in callables)):
                uvals = [str(lval) for lval in legal_values_non_callable]
                pp_values = '|'.join(uvals)
                msg = f'Value must be one of {pp_values}'
                if len(callables):
                    msg += ' or a callable'
                raise ValueError(msg)
    return inner


def is_nonnegative_int(value: Optional[int]) -> None:
    """
    Verify that value is None or a positive int.
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
    bool - True if object is callable, raises ValueError otherwise.
    """
    if not callable(obj):
        raise ValueError('Value must be a callable')
    return True