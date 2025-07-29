from __future__ import annotations
import collections
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict, Tuple, Optional, Union

import numpy as np
from pandas import Series, Index


def _from_nested_dict(data: Mapping[Any, Mapping[Any, Any]]) -> Dict[Any, Dict[Any, Any]]:
    new_data: Dict[Any, Dict[Any, Any]] = defaultdict(dict)
    for index, s in data.items():
        for col, v in s.items():
            new_data[col][index] = v
    return new_data


def _reindex_for_setitem(value: Any, index: Index) -> Tuple[Any, Optional[Any]]:
    if value.index.equals(index) or not len(index):
        if isinstance(value, Series):
            return (value._values, value._references)
        return (value._values.copy(), None)
    try:
        reindexed_value = value.reindex(index)._values
    except ValueError as err:
        if not value.index.is_unique:
            raise err
        raise TypeError('incompatible index of inserted column with frame index') from err
    return (reindexed_value, None)