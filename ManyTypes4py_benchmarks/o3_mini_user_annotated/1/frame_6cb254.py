from collections import defaultdict
from typing import Any, DefaultDict, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union, TypeVar

from pandas import DataFrame, Series, Index
from pandas._typing import ArrayLike
# Assume BlockValuesRefs is defined somewhere in pandas internals; for annotation purposes:
BlockValuesRefs = Any

T = TypeVar("T")
HashableT = TypeVar("HashableT", bound=Hashable)
HashableT2 = TypeVar("HashableT2", bound=Hashable)


def _from_nested_dict(
    data: Mapping[HashableT, Mapping[HashableT2, T]]
) -> DefaultDict[HashableT2, Dict[HashableT, T]]:
    new_data: DefaultDict[HashableT2, Dict[HashableT, T]] = defaultdict(dict)
    for index, s in data.items():
        for col, v in s.items():
            new_data[col][index] = v
    return new_data


def _reindex_for_setitem(
    value: Union[DataFrame, Series], index: Index
) -> Tuple[ArrayLike, Optional[BlockValuesRefs]]:
    if value.index.equals(index) or not len(index):
        if isinstance(value, Series):
            return value._values, value._references  # type: ignore
        return value._values.copy(), None  # type: ignore
    try:
        reindexed_value = value.reindex(index)._values  # type: ignore
    except ValueError as err:
        if not value.index.is_unique:
            raise err
        raise TypeError("incompatible index of inserted column with frame index") from err
    return reindexed_value, None