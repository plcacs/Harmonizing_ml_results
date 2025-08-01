#!/usr/bin/env python3
"""
Infrastructure of options for Koalas.
"""
from contextlib import contextmanager
import json
from typing import Union, Any, Tuple, Callable, List, Dict, Iterator
from pyspark._globals import _NoValue, _NoValueType
from databricks.koalas.utils import default_session

__all__ = ['get_option', 'set_option', 'reset_option', 'options', 'option_context']

class Option:
    """
    Option class that defines an option with related properties.

    This class holds all information relevant to the one option. Also,
    its instance can validate if the given value is acceptable or not.

    It is currently for internal usage only.

    Parameters
    ----------
    key: str, keyword-only argument
        the option name to use.
    doc: str, keyword-only argument
        the documentation for the current option.
    default: Any, keyword-only argument
        default value for this option.
    types: Union[Tuple[type, ...], type], keyword-only argument
        default is str. It defines the expected types for this option. It is
        used with `isinstance` to validate the given value to this option.
    check_func: Tuple[Callable[[Any], bool], str], keyword-only argument
        default is a function that always returns `True` with an empty string.
        It defines:
          - a function to check the given value to this option
          - the error message to show when this check is failed.
        When a new value is set to this option, this function is called to check
        if the given value is valid.
    """
    def __init__(self, *, key: str, doc: str, default: Any, 
                 types: Union[Tuple[type, ...], type] = str, 
                 check_func: Tuple[Callable[[Any], bool], str] = (lambda v: True, '')) -> None:
        self.key: str = key
        self.doc: str = doc
        self.default: Any = default
        self.types: Union[Tuple[type, ...], type] = types
        self.check_func: Tuple[Callable[[Any], bool], str] = check_func

    def validate(self, v: Any) -> None:
        """
        Validate the given value and throw an exception with related information such as key.
        """
        if not isinstance(v, self.types):
            raise ValueError("The value for option '%s' was %s; however, expected types are [%s]." % (self.key, type(v), str(self.types)))
        if not self.check_func[0](v):
            raise ValueError(self.check_func[1])

_options: List[Option] = [
    Option(
        key='display.max_rows', 
        doc='This sets the maximum number of rows Koalas should output when printing out various output. For example, this value determines the number of rows to be shown at the repr() in a dataframe. Set `None` to unlimit the input length. Default is 1000.',
        default=1000, 
        types=(int, type(None)), 
        check_func=(lambda v: v is None or v >= 0, "'display.max_rows' should be greater than or equal to 0.")
    ),
    Option(
        key='compute.max_rows', 
        doc="'compute.max_rows' sets the limit of the current Koalas DataFrame. Set `None` to unlimit the input length. When the limit is set, it is executed by the shortcut by collecting the data into the driver, and then using the pandas API. If the limit is unset, the operation is executed by PySpark. Default is 1000.",
        default=1000, 
        types=(int, type(None)), 
        check_func=(lambda v: v is None or v >= 0, "'compute.max_rows' should be greater than or equal to 0.")
    ),
    Option(
        key='compute.shortcut_limit', 
        doc="'compute.shortcut_limit' sets the limit for a shortcut. It computes specified number of rows and use its schema. When the dataframe length is larger than this limit, Koalas uses PySpark to compute.",
        default=1000, 
        types=int, 
        check_func=(lambda v: v >= 0, "'compute.shortcut_limit' should be greater than or equal to 0.")
    ),
    Option(
        key='compute.ops_on_diff_frames', 
        doc="This determines whether or not to operate between two different dataframes. For example, 'combine_frames' function internally performs a join operation which can be expensive in general. So, if `compute.ops_on_diff_frames` variable is not True, that method throws an exception.",
        default=False, 
        types=bool
    ),
    Option(
        key='compute.default_index_type', 
        doc='This sets the default index type: sequence, distributed and distributed-sequence.',
        default='sequence', 
        types=str, 
        check_func=(lambda v: v in ('sequence', 'distributed', 'distributed-sequence'), "Index type should be one of 'sequence', 'distributed', 'distributed-sequence'.")
    ),
    Option(
        key='compute.ordered_head', 
        doc="'compute.ordered_head' sets whether or not to operate head with natural ordering. Koalas does not guarantee the row ordering so `head` could return some rows from distributed partitions. If 'compute.ordered_head' is set to True, Koalas performs natural ordering beforehand, but it will cause a performance overhead.",
        default=False, 
        types=bool
    ),
    Option(
        key='plotting.max_rows', 
        doc="'plotting.max_rows' sets the visual limit on top-n-based plots such as `plot.bar` and `plot.pie`. If it is set to 1000, the first 1000 data points will be used for plotting. Default is 1000.",
        default=1000, 
        types=int, 
        check_func=(lambda v: v is v >= 0, "'plotting.max_rows' should be greater than or equal to 0.")
    ),
    Option(
        key='plotting.sample_ratio', 
        doc="'plotting.sample_ratio' sets the proportion of data that will be plotted for sample-based plots such as `plot.line` and `plot.area`. This option defaults to 'plotting.max_rows' option.",
        default=None, 
        types=(float, type(None)), 
        check_func=(lambda v: v is None or 1 >= v >= 0, "'plotting.sample_ratio' should be 1.0 >= value >= 0.0.")
    ),
    Option(
        key='plotting.backend', 
        doc='Backend to use for plotting. Default is plotly. Supports any package that has a top-level `.plot` method. Known options are: [matplotlib, plotly].',
        default='plotly', 
        types=str
    )
]
_options_dict: Dict[str, Option] = dict(zip((option.key for option in _options), _options))
_key_format = 'koalas.{}'.format

class OptionError(AttributeError, KeyError):
    pass

def show_options() -> None:
    """
    Make a pretty table that can be copied and pasted into public documentation.
    This is currently for an internal purpose.

    Examples
    --------
    >>> show_options()  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    ================... =======... =====================...
    Option              Default    Description
    ================... =======... =====================...
    display.max_rows    1000       This sets the maximum...
    ...
    ================... =======... =====================...
    """
    import textwrap
    header: List[str] = ['Option', 'Default', 'Description']
    row_format: str = '{:<31} {:<14} {:<53}'
    print(row_format.format('=' * 31, '=' * 14, '=' * 53))
    print(row_format.format(*header))
    print(row_format.format('=' * 31, '=' * 14, '=' * 53))
    for option in _options:
        doc: str = textwrap.fill(option.doc, 53)
        formatted: str = ''.join([line + '\n' + ' ' * 47 for line in doc.split('\n')]).rstrip()
        print(row_format.format(option.key, repr(option.default), formatted))
    print(row_format.format('=' * 31, '=' * 14, '=' * 53))

def get_option(key: str, default: Any = _NoValue) -> Any:
    """
    Retrieves the value of the specified option.

    Parameters
    ----------
    key : str
        The key which should match a single option.
    default : object
        The default value if the option is not set yet. The value should be JSON serializable.

    Returns
    -------
    result : the value of the option

    Raises
    ------
    OptionError : if no such option exists and the default is not provided
    """
    _check_option(key)
    if default is _NoValue:
        default = _options_dict[key].default
    _options_dict[key].validate(default)
    return json.loads(default_session().conf.get(_key_format(key), default=json.dumps(default)))

def set_option(key: str, value: Any) -> None:
    """
    Sets the value of the specified option.

    Parameters
    ----------
    key : str
        The key which should match a single option.
    value : object
        New value of option. The value should be JSON serializable.
    """
    _check_option(key)
    _options_dict[key].validate(value)
    default_session().conf.set(_key_format(key), json.dumps(value))

def reset_option(key: str) -> None:
    """
    Reset one option to their default value.

    Pass "all" as argument to reset all options.

    Parameters
    ----------
    key : str
        If specified only option will be reset.
    """
    _check_option(key)
    default_session().conf.unset(_key_format(key))

@contextmanager
def option_context(*args: Any) -> Iterator[None]:
    """
    Context manager to temporarily set options in the `with` statement context.

    You need to invoke as ``option_context(pat, val, [(pat, val), ...])``.

    Examples
    --------
    >>> with option_context('display.max_rows', 10, 'compute.max_rows', 5):
    ...     print(get_option('display.max_rows'), get_option('compute.max_rows'))
    10 5
    >>> print(get_option('display.max_rows'), get_option('compute.max_rows'))
    1000 1000
    """
    if len(args) == 0 or len(args) % 2 != 0:
        raise ValueError('Need to invoke as option_context(pat, val, [(pat, val), ...]).')
    opts: Dict[str, Any] = dict(zip(args[::2], args[1::2]))
    orig_opts: Dict[str, Any] = {key: get_option(key) for key in opts}
    try:
        for key, value in opts.items():
            set_option(key, value)
        yield
    finally:
        for key, value in orig_opts.items():
            set_option(key, value)

def _check_option(key: str) -> None:
    if key not in _options_dict:
        raise OptionError("No such option: '{}'. Available options are [{}]".format(key, ', '.join(list(_options_dict.keys()))))

class DictWrapper:
    """ Provide attribute-style access to a nested dict. """

    def __init__(self, d: Dict[str, Option], prefix: str = '') -> None:
        object.__setattr__(self, 'd', d)
        object.__setattr__(self, 'prefix', prefix)

    def __setattr__(self, key: str, val: Any) -> None:
        prefix: str = object.__getattribute__(self, 'prefix')
        d: Dict[str, Option] = object.__getattribute__(self, 'd')
        if prefix:
            prefix += '.'
        canonical_key: str = prefix + key
        candidates: List[str] = [k for k in d.keys() if all((x in k.split('.') for x in canonical_key.split('.')))]
        if len(candidates) == 1 and candidates[0] == canonical_key:
            return set_option(canonical_key, val)
        else:
            raise OptionError("No such option: '{}'. Available options are [{}]".format(key, ', '.join(list(_options_dict.keys()))))

    def __getattr__(self, key: str) -> Any:
        prefix: str = object.__getattribute__(self, 'prefix')
        d: Dict[str, Option] = object.__getattribute__(self, 'd')
        if prefix:
            prefix += '.'
        canonical_key: str = prefix + key
        candidates: List[str] = [k for k in d.keys() if all((x in k.split('.') for x in canonical_key.split('.')))]
        if len(candidates) == 1 and candidates[0] == canonical_key:
            return get_option(canonical_key)
        elif len(candidates) == 0:
            raise OptionError("No such option: '{}'. Available options are [{}]".format(key, ', '.join(list(_options_dict.keys()))))
        else:
            return DictWrapper(d, canonical_key)

    def __dir__(self) -> List[str]:
        prefix: str = object.__getattribute__(self, 'prefix')
        d: Dict[str, Option] = object.__getattribute__(self, 'd')
        if prefix == '':
            candidates = list(d.keys())
            offset = 0
        else:
            candidates = [k for k in d.keys() if all((x in k.split('.') for x in prefix.split('.')))]
            offset = len(prefix) + 1
        return [c[offset:] for c in candidates]

options: DictWrapper = DictWrapper(_options_dict)