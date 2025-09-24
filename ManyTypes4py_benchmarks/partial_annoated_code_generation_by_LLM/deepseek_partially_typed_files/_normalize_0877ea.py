from __future__ import annotations
from collections import abc, defaultdict
import copy
from typing import TYPE_CHECKING, Any, DefaultDict, overload, Union, List, Optional
import numpy as np
from pandas._libs.writers import convert_json_to_lines
import pandas as pd
from pandas import DataFrame, Series
if TYPE_CHECKING:
    from collections.abc import Iterable
    from pandas._typing import IgnoreRaise, Scalar

def convert_to_line_delimits(s: str) -> str:
    """
    Helper function that converts JSON lists to line delimited JSON.
    """
    if not s[0] == '[' and s[-1] == ']':
        return s
    s = s[1:-1]
    return convert_json_to_lines(s)

@overload
def nested_to_record(ds: dict[str, Any], prefix: str = ..., sep: str = ..., level: int = ..., max_level: Optional[int] = ...) -> dict[str, Any]:
    ...

@overload
def nested_to_record(ds: list[dict[str, Any]], prefix: str = ..., sep: str = ..., level: int = ..., max_level: Optional[int] = ...) -> list[dict[str, Any]]:
    ...

def nested_to_record(ds: Union[dict[str, Any], list[dict[str, Any]]], prefix: str = '', sep: str = '.', level: int = 0, max_level: Optional[int] = None) -> Union[dict[str, Any], list[dict[str, Any]]]:
    """
    A simplified json_normalize

    Converts a nested dict into a flat dict ("record"), unlike json_normalize,
    it does not attempt to extract a subset of the data.

    Parameters
    ----------
    ds : dict or list of dicts
    prefix: the prefix, optional, default: ""
    sep : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar
    level: int, optional, default: 0
        The number of levels in the json string.

    max_level: int, optional, default: None
        The max depth to normalize.

    Returns
    -------
    d - dict or list of dicts, matching `ds`

    Examples
    --------
    >>> nested_to_record(
    ...     dict(flat1=1, dict1=dict(c=1, d=2), nested=dict(e=dict(c=1, d=2), d=2))
    ... )
    {'flat1': 1, 'dict1.c': 1, 'dict1.d': 2, 'nested.e.c': 1, 'nested.e.d': 2, 'nested.d': 2}
    """
    singleton = False
    if isinstance(ds, dict):
        ds = [ds]
        singleton = True
    new_ds: list[dict[str, Any]] = []
    for d in ds:
        new_d = copy.deepcopy(d)
        for (k, v) in d.items():
            if not isinstance(k, str):
                k = str(k)
            if level == 0:
                newkey = k
            else:
                newkey = prefix + sep + k
            if not isinstance(v, dict) or (max_level is not None and level >= max_level):
                if level != 0:
                    v = new_d.pop(k)
                    new_d[newkey] = v
                continue
            v = new_d.pop(k)
            new_d.update(nested_to_record(v, newkey, sep, level + 1, max_level))
        new_ds.append(new_d)
    if singleton:
        return new_ds[0]
    return new_ds

def _normalise_json(data: Any, key_string: str, normalized_dict: dict[str, Any], separator: str) -> dict[str, Any]:
    """
    Main recursive function
    Designed for the most basic use case of pd.json_normalize(data)
    intended as a performance improvement, see #15621

    Parameters
    ----------
    data : Any
        Type dependent on types contained within nested Json
    key_string : str
        New key (with separator(s) in) for data
    normalized_dict : dict
        The new normalized/flattened Json dict
    separator : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar
    """
    if isinstance(data, dict):
        for (key, value) in data.items():
            new_key = f'{key_string}{separator}{key}'
            if not key_string:
                new_key = new_key.removeprefix(separator)
            _normalise_json(data=value, key_string=new_key, normalized_dict=normalized_dict, separator=separator)
    else:
        normalized_dict[key_string] = data
    return normalized_dict

def _normalise_json_ordered(data: dict[str, Any], separator: str) -> dict[str, Any]:
    """
    Order the top level keys and then recursively go to depth

    Parameters
    ----------
    data : dict or list of dicts
    separator : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar

    Returns
    -------
    dict or list of dicts, matching `normalised_json_object`
    """
    top_dict_: dict[str, Any] = {k: v for (k, v) in data.items() if not isinstance(v, dict)}
    nested_dict_: dict[str, Any] = _normalise_json(data={k: v for (k, v) in data.items() if isinstance(v, dict)}, key_string='', normalized_dict={}, separator=separator)
    return {**top_dict_, **nested_dict_}

def _simple_json_normalize(ds: Union[dict[str, Any], list[dict[str, Any]]], sep: str = '.') -> Union[dict[str, Any], list[dict[str, Any]], Any]:
    """
    A optimized basic json_normalize

    Converts a nested dict into a flat dict ("record"), unlike
    json_normalize and nested_to_record it doesn't do anything clever.
    But for the most basic use cases it enhances performance.
    E.g. pd.json_normalize(data)

    Parameters
    ----------
    ds : dict or list of dicts
    sep : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar

    Returns
    -------
    frame : DataFrame
    d - dict or list of dicts, matching `normalised_json_object`

    Examples
    --------
    >>> _simple_json_normalize(
    ...     {
    ...         "flat1": 1,
    ...         "dict1": {"c": 1, "d": 2},
    ...         "nested": {"e": {"c": 1, "d": 2}, "d": 2},
    ...     }
    ... )
    {'flat1': 1, 'dict1.c': 1, 'dict1.d': 2, 'nested.e.c': 1, 'nested.e.d': 2, 'nested.d': 2}

    """
    normalised_json_object: dict[str, Any] = {}
    if isinstance(ds, dict):
        normalised_json_object = _normalise_json_ordered(data=ds, separator=sep)
    elif isinstance(ds, list):
        normalised_json_list: list[dict[str, Any]] = [_simple_json_normalize(row, sep=sep) for row in ds]
        return normalised_json_list
    return normalised_json_object

def json_normalize(data: Union[dict[str, Any], list[dict[str, Any]], Series], record_path: Optional[Union[str, list[str]]] = None, meta: Optional[Union[str, list[Union[str, list[str]]]]] = None, meta_prefix: Optional[str] = None, record_prefix: Optional[str] = None, errors: str = 'raise', sep: str = '.', max_level: Optional[int] = None) -> DataFrame:
    """
    Normalize semi-structured JSON data into a flat table.

    This method is designed to transform semi-structured JSON data, such as nested
    dictionaries or lists, into a flat table. This is particularly useful when
    handling JSON-like data structures that contain deeply nested fields.

    Parameters
    ----------
    data : dict, list of dicts, or Series of dicts
        Unserialized JSON objects.
    record_path : str or list of str, default None
        Path in each object to list of records. If not passed, data will be
        assumed to be an array of records.
    meta : list of paths (str or list of str), default None
        Fields to use as metadata for each record in resulting table.
    meta_prefix : str, default None
        If True, prefix records with dotted path, e.g. foo.bar.field if
        meta is ['foo', 'bar'].
    record_prefix : str, default None
        If True, prefix records with dotted path, e.g. foo.bar.field if
        path to records is ['foo', 'bar'].
    errors : {'raise', 'ignore'}, default 'raise'
        Configures error handling.

        * 'ignore' : will ignore KeyError if keys listed in meta are not
          always present.
        * 'raise' : will raise KeyError if keys listed in meta are not
          always present.
    sep : str, default '.'
        Nested records will generate names separated by sep.
        e.g., for sep='.', {'foo': {'bar': 0}} -> foo.bar.
    max_level : int, default None
        Max number of levels(depth of dict) to normalize.
        if None, normalizes all levels.

    Returns
    -------
    DataFrame
        The normalized data, represented as a pandas DataFrame.

    See Also
    --------
    DataFrame : Two-dimensional, size-mutable, potentially heterogeneous tabular data.
    Series : One-dimensional ndarray with axis labels (including time series).

    Examples
    --------
    >>> data = [
    ...     {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
    ...     {"name": {"given": "Mark", "family": "Regner"}},
    ...     {"id": 2, "name": "Faye Raker"},
    ... ]
    >>> pd.json_normalize(data)
        id name.first name.last name.given name.family        name
    0  1.0     Coleen      Volk        NaN         NaN         NaN
    1  NaN        NaN       NaN       Mark      Regner         NaN
    2  2.0        NaN       NaN        NaN         NaN  Faye Raker

    >>> data = [
    ...     {
    ...         "id": 1,
    ...         "name": "Cole Volk",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
    ...     {
    ...         "id": 2,
    ...         "name": "Faye Raker",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ... ]
    >>> pd.json_normalize(data, max_level=0)
        id        name                        fitness
    0  1.0   Cole Volk  {'height': 130, 'weight': 60}
    1  NaN    Mark Reg  {'height': 130, 'weight': 60}
    2  2.0  Faye Raker  {'height': 130, 'weight': 60}

    Normalizes nested data up to level 1.

    >>> data = [
    ...     {
    ...         "id": 1,
    ...         "name": "Cole Volk",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
    ...     {
    ...         "id": 2,
    ...         "name": "Faye Raker",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ... ]
    >>> pd.json_normalize(data, max_level=1)
        id        name  fitness.height  fitness.weight
    0  1.0   Cole Volk             130              60
    1  NaN    Mark Reg             130              60
    2  2.0  Faye Raker             130              60

    >>> data = [
    ...     {
    ...         "id": 1,
    ...         "name": "Cole Volk",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
    ...     {
    ...         "id": 2,
    ...         "name": "Faye Raker",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ... ]
    >>> series = pd.Series(data, index=pd.Index(["a", "b", "c"]))
    >>> pd.json_normalize(series)
        id        name  fitness.height  fitness.weight
    a  1.0   Cole Volk             130              60
    b  NaN    Mark Reg             130              60
    c  2.0  Faye Raker             130              60

    >>> data = [
    ...     {
    ...         "state": "Florida",
    ...         "shortname": "FL",
    ...         "info": {"governor": "Rick Scott"},
    ...         "counties": [
    ...             {"name": "Dade", "population": 12345},
    ...             {"name": "Broward", "population": 40000},
    ...             {"name": "Palm Beach", "population": 60000},
    ...         ],
    ...     },
    ...     {
    ...         "state": "Ohio",
    ...         "shortname": "OH",
    ...         "info": {"governor": "John Kasich"},
    ...         "counties": [
    ...             {"name": "Summit", "population": 1234},
    ...             {"name": "Cuyahoga", "population": 1337},
    ...         ],
    ...     },
    ... ]
    >>> result = pd.json_normalize(
    ...     data, "counties", ["state", "shortname", ["info", "governor"]]
    ... )
    >>> result
             name  population    state shortname info.governor
    0        Dade       12345   Florida    FL    Rick Scott
    1     Broward       40000   Florida    FL    Rick Scott
    2  Palm Beach       60000   Florida    FL    Rick Scott
    3      Summit        1234   Ohio       OH    John Kasich
    4    Cuyahoga        1337   Ohio       OH    John Kasich

    >>> data = {"A": [1, 2]}
    >>> pd.json_normalize(data, "A", record_prefix="Prefix.")
        Prefix.0
    0          1
    1          2

    Returns normalized data with columns prefixed with the given string.
    """

    def _pull_field(js: dict[str, Any], spec: Union[list[str], str], extract_record: bool = False) -> Union[Scalar, Iterable]:
        """Internal function to pull field"""
        result = js
        try:
            if isinstance(spec, list):
                for field in spec:
                    if result is None:
                        raise KeyError(field)
                    result = result[field]
            else:
                result = result[spec]
        except KeyError as e:
            if extract_record:
                raise KeyError(f'Key {e} not found. If specifying a record_path, all elements of data should have the path.') from e
            if errors == 'ignore':
                return np.nan
            else:
                raise KeyError(f"Key {e} not found. To replace missing values of {e} with np.nan, pass in errors='ignore'") from e
        return result

    def _pull_records(js: dict[str, Any], spec: Union[list[str], str]) -> list:
        """
        Internal function to pull field for records, and similar to
        _pull_field, but require to return list. And will raise error
        if has non iterable value.
        """
        result = _pull_field(js, spec, extract_record=True)
        if not isinstance(result, list):
            if pd.isnull(result):
                result = []
            else:
                raise TypeError(f'Path must contain list or null, but got {type(result).__name__} at {spec!r}')
        return result
    index: Optional[pd.Index] = None
    if isinstance(data, Series):
        index = data.index
    else:
        index = None
    if isinstance(data, list) and (not data):
        return DataFrame()
    elif isinstance(data, dict):
        data = [data]
    elif isinstance(data, abc.Iterable) and (not isinstance(data, str)):
        data = list(data)
    else:
        raise NotImplementedError
    if record_path is None and meta is None and (meta_prefix is None) and (record_prefix is None) and (max_level is None):
        return DataFrame(_simple_json_normalize(data, sep=sep), index=index)
    if record_path is None:
        if any(([isinstance(x, dict) for x in y.values()] for y in data)):
            data = nested_to_record(data, sep=sep, max_level=max_level)
        return DataFrame(data, index=index)
    elif not isinstance(record_path, list):
        record_path = [record_path]
    if meta is None:
        meta = []
    elif not isinstance(meta, list):
        meta = [meta]
    _meta: list[list[str]] = [m if isinstance(m, list) else [m] for m in meta]
    records: list[dict[str, Any]] = []
    lengths: list[int] = []
    meta_vals: DefaultDict[str, list] = defaultdict(list)
    meta_keys: list[str