from __future__ import annotations

from collections import (
    abc,
    defaultdict,
)
import copy
from typing import (
    Any,
    DefaultDict,
    Iterable,
    List,
    Union,
    overload,
    Optional,
)
import numpy as np

from pandas._libs.writers import convert_json_to_lines

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)

if False:  # TYPE_CHECKING block
    from collections.abc import Iterable
    from pandas._typing import (
        IgnoreRaise,
        Scalar,
    )
else:
    IgnoreRaise = str
    Scalar = Any


def convert_to_line_delimits(s: str) -> str:
    """
    Helper function that converts JSON lists to line delimited JSON.
    """
    # Determine we have a JSON list to turn to lines otherwise just return the
    # json object, only lists can
    if not (s[0] == "[" and s[-1] == "]"):
        return s
    s = s[1:-1]
    return convert_json_to_lines(s)


@overload
def nested_to_record(
    ds: dict,
    prefix: str = "",
    sep: str = ".",
    level: int = 0,
    max_level: int | None = None,
) -> dict[str, Any]:
    ...


@overload
def nested_to_record(
    ds: list[dict],
    prefix: str = "",
    sep: str = ".",
    level: int = 0,
    max_level: int | None = None,
) -> list[dict[str, Any]]:
    ...


def nested_to_record(
    ds: dict | list[dict],
    prefix: str = "",
    sep: str = ".",
    level: int = 0,
    max_level: int | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    A simplified json_normalize

    Converts a nested dict into a flat dict ("record"), unlike json_normalize,
    it does not attempt to extract a subset of the data.
    """
    singleton: bool = False
    if isinstance(ds, dict):
        ds = [ds]
        singleton = True
    new_ds: list[dict[str, Any]] = []
    for d in ds:
        new_d: dict[str, Any] = copy.deepcopy(d)
        for k, v in d.items():
            # each key gets renamed with prefix
            if not isinstance(k, str):
                k = str(k)
            if level == 0:
                newkey: str = k
            else:
                newkey = prefix + sep + k

            # flatten if type is dict and
            # current dict level  < maximum level provided and
            # only dicts gets recurse-flattened
            # only at level>1 do we rename the rest of the keys
            if not isinstance(v, dict) or (max_level is not None and level >= max_level):
                if level != 0:  # so we skip copying for top level, common case
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
    normalized_dict: dict[str, Any],
    separator: str,
) -> dict[str, Any]:
    """
    Main recursive function to flatten JSON.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            new_key: str = f"{key_string}{separator}{key}"
            if not key_string:
                new_key = new_key.removeprefix(separator)
            _normalise_json(
                data=value,
                key_string=new_key,
                normalized_dict=normalized_dict,
                separator=separator,
            )
    else:
        normalized_dict[key_string] = data
    return normalized_dict


def _normalise_json_ordered(data: dict[str, Any], separator: str) -> dict[str, Any]:
    """
    Order the top level keys and then recursively go to depth.
    """
    top_dict_: dict[str, Any] = {k: v for k, v in data.items() if not isinstance(v, dict)}
    nested_dict_: dict[str, Any] = _normalise_json(
        data={k: v for k, v in data.items() if isinstance(v, dict)},
        key_string="",
        normalized_dict={},
        separator=separator,
    )
    return {**top_dict_, **nested_dict_}


def _simple_json_normalize(
    ds: dict | list[dict],
    sep: str = ".",
) -> dict | list[dict] | Any:
    """
    A optimized basic json_normalize.
    """
    normalised_json_object: dict[str, Any] = {}
    if isinstance(ds, dict):
        normalised_json_object = _normalise_json_ordered(data=ds, separator=sep)
    elif isinstance(ds, list):
        normalised_json_list = [_simple_json_normalize(row, sep=sep) for row in ds]
        return normalised_json_list
    return normalised_json_object


def json_normalize(
    data: dict[str, Any] | list[dict[str, Any]] | Series,
    record_path: str | List[Any] | None = None,
    meta: str | List[str | List[str]] | None = None,
    meta_prefix: Optional[str] = None,
    record_prefix: Optional[str] = None,
    errors: IgnoreRaise = "raise",
    sep: str = ".",
    max_level: int | None = None,
) -> DataFrame:
    """
    Normalize semi-structured JSON data into a flat table.
    """
    def _pull_field(
        js: dict[str, Any], spec: list | str, extract_record: bool = False
    ) -> Union[Scalar, Iterable]:
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
                raise KeyError(
                    f"Key {e} not found. If specifying a record_path, all elements of "
                    f"data should have the path."
                ) from e
            if errors == "ignore":
                return np.nan
            else:
                raise KeyError(
                    f"Key {e} not found. To replace missing values of {e} with "
                    f"np.nan, pass in errors='ignore'"
                ) from e
        return result

    def _pull_records(js: dict[str, Any], spec: list | str) -> list[Any]:
        result: Any = _pull_field(js, spec, extract_record=True)
        if not isinstance(result, list):
            if pd.isnull(result):
                result = []
            else:
                raise TypeError(
                    f"Path must contain list or null, but got {type(result).__name__} at {spec!r}"
                )
        return result

    if isinstance(data, Series):
        index = data.index
    else:
        index = None

    if isinstance(data, list) and not data:
        return DataFrame()
    elif isinstance(data, dict):
        data = [data]
    elif isinstance(data, abc.Iterable) and not isinstance(data, str):
        data = list(data)
    else:
        raise NotImplementedError

    if (
        record_path is None
        and meta is None
        and meta_prefix is None
        and record_prefix is None
        and max_level is None
    ):
        return DataFrame(_simple_json_normalize(data, sep=sep), index=index)

    if record_path is None:
        if any(isinstance(x, dict) for y in data for x in y.values()):
            data = nested_to_record(data, sep=sep, max_level=max_level)
        return DataFrame(data, index=index)
    elif not isinstance(record_path, list):
        record_path = [record_path]

    if meta is None:
        meta = []
    elif not isinstance(meta, list):
        meta = [meta]

    _meta: List[List[Any]] = [m if isinstance(m, list) else [m] for m in meta]
    meta_vals: DefaultDict[str, list[Any]] = defaultdict(list)
    meta_keys: List[str] = [sep.join(val) for val in _meta]
    lengths: List[int] = []

    def _recursive_extract(
        data_item: Any, path: List[Any], seen_meta: dict[str, Any], level: int = 0
    ) -> None:
        if isinstance(data_item, dict):
            data_list: List[Any] = [data_item]
        else:
            data_list = data_item
        if len(path) > 1:
            for obj in data_list:
                for val, key in zip(_meta, meta_keys):
                    if level + 1 == len(val):
                        seen_meta[key] = _pull_field(obj, val[-1])
                _recursive_extract(obj[path[0]], path[1:], seen_meta, level=level + 1)
        else:
            for obj in data_list:
                recs: List[Any] = _pull_records(obj, path[0])
                recs = [
                    nested_to_record(r, sep=sep, max_level=max_level)
                    if isinstance(r, dict)
                    else r
                    for r in recs
                ]
                lengths.append(len(recs))
                for val, key in zip(_meta, meta_keys):
                    if level + 1 > len(val):
                        meta_val = seen_meta[key]
                    else:
                        meta_val = _pull_field(obj, val[level:])
                    meta_vals[key].append(meta_val)
                nonlocal records
                records.extend(recs)

    records: List[Any] = []
    _recursive_extract(data, record_path, {}, level=0)

    result: DataFrame = DataFrame(records)
    if record_prefix is not None:
        result = result.rename(columns=lambda x: f"{record_prefix}{x}")

    for k, v in meta_vals.items():
        if meta_prefix is not None:
            k = meta_prefix + k
        if k in result:
            raise ValueError(
                f"Conflicting metadata name {k}, need distinguishing prefix "
            )
        values = np.array(v, dtype=object)
        if values.ndim > 1:
            values = np.empty((len(v),), dtype=object)
            for i, val in enumerate(v):
                values[i] = val
        result[k] = values.repeat(lengths)
    if index is not None:
        result.index = index.repeat(lengths)
    return result