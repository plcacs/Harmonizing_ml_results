from __future__ import annotations
from collections import abc, defaultdict
import copy
from typing import Any, Dict, List, Union, Optional, overload
import numpy as np
from pandas._libs.writers import convert_json_to_lines
import pandas as pd
from pandas import DataFrame, Series

if False:
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
def nested_to_record(
    ds: Union[Dict[Any, Any], List[Dict[Any, Any]]],
    prefix: str = '',
    sep: str = '.',
    level: int = 0,
    max_level: Optional[int] = None
) -> Dict[Any, Any]:
    ...


@overload
def nested_to_record(
    ds: Union[Dict[Any, Any], List[Dict[Any, Any]]],
    prefix: str = '',
    sep: str = '.',
    level: int = 0,
    max_level: Optional[int] = None
) -> List[Dict[Any, Any]]:
    ...


def nested_to_record(
    ds: Union[Dict[Any, Any], List[Dict[Any, Any]]],
    prefix: str = '',
    sep: str = '.',
    level: int = 0,
    max_level: Optional[int] = None
) -> Union[Dict[Any, Any], List[Dict[Any, Any]]]:
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
    """
    singleton = False
    if isinstance(ds, dict):
        ds = [ds]
        singleton = True
    new_ds: List[Dict[Any, Any]] = []
    for d in ds:
        new_d = copy.deepcopy(d)
        for k, v in d.items():
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


def _normalise_json(
    data: Any,
    key_string: str,
    normalized_dict: Dict[str, Any],
    separator: str
) -> Dict[str, Any]:
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
        for key, value in data.items():
            new_key = f'{key_string}{separator}{key}'
            if not key_string:
                new_key = new_key.removeprefix(separator)
            _normalise_json(
                data=value, key_string=new_key, normalized_dict=normalized_dict, separator=separator
            )
    else:
        normalized_dict[key_string] = data
    return normalized_dict


def _normalise_json_ordered(data: Dict[Any, Any], separator: str) -> Dict[str, Any]:
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
    top_dict_ = {k: v for k, v in data.items() if not isinstance(v, dict)}
    nested_dict_ = _normalise_json(
        data={k: v for k, v in data.items() if isinstance(v, dict)},
        key_string='',
        normalized_dict={},
        separator=separator
    )
    return {**top_dict_, **nested_dict_}


def _simple_json_normalize(
    ds: Union[Dict[Any, Any], List[Any]],
    sep: str = '.'
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
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
    """
    normalised_json_object: Dict[str, Any] = {}
    if isinstance(ds, dict):
        normalised_json_object = _normalise_json_ordered(data=ds, separator=sep)
    elif isinstance(ds, list):
        normalised_json_list: List[Dict[str, Any]] = [_simple_json_normalize(row, sep=sep) for row in ds]  # type: ignore
        return normalised_json_list
    return normalised_json_object


def json_normalize(
    data: Union[Dict[Any, Any], List[Any], Series],
    record_path: Optional[Union[str, List[str]]] = None,
    meta: Optional[Union[str, List[Union[str, List[str]]]]] = None,
    meta_prefix: Optional[str] = None,
    record_prefix: Optional[str] = None,
    errors: str = 'raise',
    sep: str = '.',
    max_level: Optional[int] = None
) -> DataFrame:
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
    """
    def _pull_field(js: Any, spec: Union[str, List[str]], extract_record: bool = False) -> Any:
        """Internal function to pull field"""
        result: Any = js
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

    def _pull_records(js: Any, spec: Union[str, List[str]]) -> List[Any]:
        """
        Internal function to pull field for records, and similar to
        _pull_field, but require to return list. And will raise error
        if has non iterable value.
        """
        result: Any = _pull_field(js, spec, extract_record=True)
        if not isinstance(result, list):
            if pd.isnull(result):
                result = []
            else:
                raise TypeError(f'Path must contain list or null, but got {type(result).__name__} at {spec!r}')
        return result

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

    if (
        record_path is None and meta is None and (meta_prefix is None)
        and (record_prefix is None) and (max_level is None)
    ):
        return DataFrame(_simple_json_normalize(data, sep=sep), index=index)

    if record_path is None:
        if any(([isinstance(x, dict) for x in y.values()] for y in data)):
            data = nested_to_record(data, sep=sep, max_level=max_level)  # type: ignore
        return DataFrame(data, index=index)
    elif not isinstance(record_path, list):
        record_path = [record_path]
    if meta is None:
        meta = []
    elif not isinstance(meta, list):
        meta = [meta]
    _meta: List[List[str]] = [m if isinstance(m, list) else [m] for m in meta]  # type: ignore
    records: List[Any] = []
    lengths: List[int] = []
    meta_vals: defaultdict[str, List[Any]] = defaultdict(list)
    meta_keys: List[str] = [sep.join(val) for val in _meta]

    def _recursive_extract(
        data: Union[Dict[Any, Any], List[Dict[Any, Any]]],
        path: List[str],
        seen_meta: Dict[str, Any],
        level: int = 0
    ) -> None:
        if isinstance(data, dict):
            data = [data]
        if len(path) > 1:
            for obj in data:
                for val, key in zip(_meta, meta_keys):
                    if level + 1 == len(val):
                        seen_meta[key] = _pull_field(obj, val[-1])
                _recursive_extract(obj[path[0]], path[1:], seen_meta, level=level + 1)
        else:
            for obj in data:
                recs: List[Any] = _pull_records(obj, path[0])
                recs = [
                    nested_to_record(r, sep=sep, max_level=max_level) if isinstance(r, dict) else r
                    for r in recs
                ]
                lengths.append(len(recs))
                for val, key in zip(_meta, meta_keys):
                    if level + 1 > len(val):
                        meta_val = seen_meta[key]
                    else:
                        meta_val = _pull_field(obj, val[level:])
                    meta_vals[key].append(meta_val)
                records.extend(recs)

    _recursive_extract(data, record_path, {}, level=0)
    result = DataFrame(records)
    if record_prefix is not None:
        result = result.rename(columns=lambda x: f'{record_prefix}{x}')
    for k, v in meta_vals.items():
        key_name = k
        if meta_prefix is not None:
            key_name = meta_prefix + key_name
        if key_name in result:
            raise ValueError(f'Conflicting metadata name {key_name}, need distinguishing prefix ')
        values = np.array(v, dtype=object)
        if values.ndim > 1:
            values = np.empty((len(v),), dtype=object)
            for i, val in enumerate(v):
                values[i] = val
        result[key_name] = values.repeat(lengths)
    if index is not None:
        result.index = index.repeat(lengths)
    return result