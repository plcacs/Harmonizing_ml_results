from collections import OrderedDict, abc
from collections.abc import Sequence
from copy import copy
from datetime import datetime, timedelta
from typing import Any, Generic, Optional, Union, TypeVar, Dict, List, Set, Tuple, Iterable, Callable, cast
import attr
import numpy as np
import pandas
from hypothesis import strategies as st
from hypothesis._settings import note_deprecation
from hypothesis.control import reject
from hypothesis.errors import InvalidArgument
from hypothesis.extra import numpy as npst
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.coverage import check, check_function
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.internal.validation import check_type, check_valid_interval, check_valid_size, try_convert
from hypothesis.strategies._internal.strategies import Ex, check_strategy, SearchStrategy
from hypothesis.strategies._internal.utils import cacheable, defines_strategy
try:
    from pandas.core.arrays.integer import IntegerDtype
except ImportError:
    IntegerDtype = ()  # type: ignore

T = TypeVar('T')
Ex = TypeVar('Ex')

def dtype_for_elements_strategy(s: SearchStrategy[Any]) -> SearchStrategy[Any]:
    return st.shared(s.map(lambda x: pandas.Series([x]).dtype), key=('hypothesis.extra.pandas.dtype_for_elements_strategy', s))

def infer_dtype_if_necessary(dtype: Any, values: List[Any], elements: SearchStrategy[Any], draw: Callable[[SearchStrategy[Any]], Any) -> Any:
    if dtype is None and (not values):
        return draw(dtype_for_elements_strategy(elements))
    return dtype

@check_function
def elements_and_dtype(elements: Optional[SearchStrategy[Any]], dtype: Any, source: Optional[str] = None) -> Tuple[SearchStrategy[Any]], Any]:
    if source is None:
        prefix = ''
    else:
        prefix = f'{source}.'
    if elements is not None:
        check_strategy(elements, f'{prefix}elements')
    else:
        with check('dtype is not None'):
            if dtype is None:
                raise InvalidArgument(f'At least one of {prefix}elements or {prefix}dtype must be provided.')
    with check('isinstance(dtype, CategoricalDtype)'):
        if pandas.api.types.CategoricalDtype.is_dtype(dtype):
            raise InvalidArgument(f'{prefix}dtype is categorical, which is currently unsupported')
    if isinstance(dtype, type) and issubclass(dtype, IntegerDtype):
        raise InvalidArgument(f'Passed dtype={dtype!r} is a dtype class, please pass in an instance of this class.Otherwise it would be treated as dtype=object')
    if isinstance(dtype, type) and np.dtype(dtype).kind == 'O' and (dtype is not object):
        err_msg = f'Passed dtype={dtype!r} is not a valid Pandas dtype.'
        if issubclass(dtype, datetime):
            err_msg += ' To generate valid datetimes, pass `dtype="datetime64[ns]"`'
            raise InvalidArgument(err_msg)
        elif issubclass(dtype, timedelta):
            err_msg += ' To generate valid timedeltas, pass `dtype="timedelta64[ns]"`'
            raise InvalidArgument(err_msg)
        note_deprecation(f"{err_msg}  We'll treat it as dtype=object for now, but this will be an error in a future version.", since='2021-12-31', has_codemod=False, stacklevel=1)
    if isinstance(dtype, st.SearchStrategy):
        raise InvalidArgument(f'Passed dtype={dtype!r} is a strategy, but we require a concrete dtype here.  See https://stackoverflow.com/q/74355937 for workaround patterns.')
    _get_subclasses = getattr(IntegerDtype, '__subclasses__', list)
    dtype = {t.name: t() for t in _get_subclasses()}.get(dtype, dtype)
    if isinstance(dtype, IntegerDtype):
        is_na_dtype = True
        dtype = np.dtype(dtype.name.lower())
    elif dtype is not None:
        is_na_dtype = False
        dtype = try_convert(np.dtype, dtype, 'dtype')
    else:
        is_na_dtype = False
    if elements is None:
        elements = npst.from_dtype(dtype)
        if is_na_dtype:
            elements = st.none() | elements
    elif dtype is not None:
        def convert_element(value: Any) -> Any:
            if is_na_dtype and value is None:
                return None
            name = f'draw({prefix}elements)'
            try:
                return np.array([value], dtype=dtype)[0]
            except (TypeError, ValueError):
                raise InvalidArgument('Cannot convert %s=%r of type %s to dtype %s' % (name, value, type(value).__name__, dtype.str)) from None
        elements = elements.map(convert_element)
    assert elements is not None
    return (elements, dtype)

class ValueIndexStrategy(st.SearchStrategy):
    def __init__(
        self,
        elements: SearchStrategy[Any],
        dtype: Any,
        min_size: int,
        max_size: int,
        unique: bool,
        name: SearchStrategy[Optional[str]]
    ) -> None:
        super().__init__()
        self.elements = elements
        self.dtype = dtype
        self.min_size = min_size
        self.max_size = max_size
        self.unique = unique
        self.name = name

    def do_draw(self, data: Any) -> Any:
        result = []
        seen = set()
        iterator = cu.many(data, min_size=self.min_size, max_size=self.max_size, average_size=(self.min_size + self.max_size) / 2)
        while iterator.more():
            elt = data.draw(self.elements)
            if self.unique:
                if elt in seen:
                    iterator.reject()
                    continue
                seen.add(elt)
            result.append(elt)
        dtype = infer_dtype_if_necessary(dtype=self.dtype, values=result, elements=self.elements, draw=data.draw)
        return pandas.Index(result, dtype=dtype, tupleize_cols=False, name=data.draw(self.name))

DEFAULT_MAX_SIZE = 10

@cacheable
@defines_strategy()
def range_indexes(
    min_size: int = 0,
    max_size: Optional[int] = None,
    name: SearchStrategy[Optional[str]] = st.none()
) -> SearchStrategy[pandas.RangeIndex]:
    check_valid_size(min_size, 'min_size')
    check_valid_size(max_size, 'max_size')
    if max_size is None:
        max_size = min([min_size + DEFAULT_MAX_SIZE, 2 ** 63 - 1])
    check_valid_interval(min_size, max_size, 'min_size', 'max_size')
    check_strategy(name)
    return st.builds(pandas.RangeIndex, st.integers(min_size, max_size), name=name)

@cacheable
@defines_strategy()
def indexes(
    *,
    elements: Optional[SearchStrategy[Any]] = None,
    dtype: Any = None,
    min_size: int = 0,
    max_size: Optional[int] = None,
    unique: bool = True,
    name: SearchStrategy[Optional[str]] = st.none()
) -> SearchStrategy[pandas.Index]:
    check_valid_size(min_size, 'min_size')
    check_valid_size(max_size, 'max_size')
    check_valid_interval(min_size, max_size, 'min_size', 'max_size')
    check_type(bool, unique, 'unique')
    elements, dtype = elements_and_dtype(elements, dtype)
    if max_size is None:
        max_size = min_size + DEFAULT_MAX_SIZE
    return ValueIndexStrategy(elements, dtype, min_size, max_size, unique, name)

@defines_strategy()
def series(
    *,
    elements: Optional[SearchStrategy[Any]] = None,
    dtype: Any = None,
    index: Optional[SearchStrategy[Any]] = None,
    fill: Optional[SearchStrategy[Any]] = None,
    unique: bool = False,
    name: SearchStrategy[Optional[str]] = st.none()
) -> SearchStrategy[pandas.Series]:
    if index is None:
        index = range_indexes()
    else:
        check_strategy(index, 'index')
    elements, np_dtype = elements_and_dtype(elements, dtype)
    index_strategy = index
    if np_dtype is not None and np_dtype.kind == 'O' and (not isinstance(dtype, IntegerDtype)):
        dtype = np_dtype

    @st.composite
    def result(draw: Any) -> pandas.Series:
        index = draw(index_strategy)
        if len(index) > 0:
            if dtype is not None:
                result_data = draw(npst.arrays(dtype=object, elements=elements, shape=len(index), fill=fill, unique=unique)).tolist()
            else:
                result_data = list(draw(npst.arrays(dtype=object, elements=elements, shape=len(index), fill=fill, unique=unique)).tolist())
            return pandas.Series(result_data, index=index, dtype=dtype, name=draw(name))
        else:
            return pandas.Series((), index=index, dtype=dtype if dtype is not None else draw(dtype_for_elements_strategy(elements)), name=draw(name))
    return result()

@attr.s(slots=True)
class column(Generic[Ex]):
    name: Any = attr.ib(default=None)
    elements: Optional[SearchStrategy[Any]] = attr.ib(default=None)
    dtype: Any = attr.ib(default=None, repr=get_pretty_function_description)
    fill: Optional[SearchStrategy[Any]] = attr.ib(default=None)
    unique: bool = attr.ib(default=False)

def columns(
    names_or_number: Union[int, float, Sequence[Any]],
    *,
    dtype: Any = None,
    elements: Optional[SearchStrategy[Any]] = None,
    fill: Optional[SearchStrategy[Any]] = None,
    unique: bool = False
) -> List[column[Any]]:
    if isinstance(names_or_number, (int, float)):
        names = [None] * int(names_or_number)
    else:
        names = list(names_or_number)
    return [column(name=n, dtype=dtype, elements=elements, fill=fill, unique=unique) for n in names]

@defines_strategy()
def data_frames(
    columns: Optional[Sequence[column[Any]]] = None,
    *,
    rows: Optional[SearchStrategy[Any]] = None,
    index: Optional[SearchStrategy[Any]] = None
) -> SearchStrategy[pandas.DataFrame]:
    if index is None:
        index = range_indexes()
    else:
        check_strategy(index, 'index')
    index_strategy = index
    if columns is None:
        if rows is None:
            raise InvalidArgument('At least one of rows and columns must be provided')
        else:
            @st.composite
            def rows_only(draw: Any) -> pandas.DataFrame:
                index = draw(index_strategy)

                def row() -> Any:
                    result = draw(rows)
                    check_type(abc.Iterable, result, 'draw(row)')
                    return result
                if len(index) > 0:
                    return pandas.DataFrame([row() for _ in index], index=index)
                else:
                    base = pandas.DataFrame([row()])
                    return base.drop(0)
            return rows_only()
    assert columns is not None
    cols = try_convert(tuple, columns, 'columns')
    rewritten_columns = []
    column_names = set()
    for i, c in enumerate(cols):
        check_type(column, c, f'columns[{i}]')
        c = copy(c)
        if c.name is None:
            label = f'columns[{i}]'
            c.name = i
        else:
            label = c.name
            try:
                hash(c.name)
            except TypeError:
                raise InvalidArgument(f'Column names must be hashable, but columns[{i}].name was {c.name!r} of type {type(c.name).__name__}, which cannot be hashed.') from None
        if c.name in column_names:
            raise InvalidArgument(f'duplicate definition of column name {c.name!r}')
        column_names.add(c.name)
        c.elements, _ = elements_and_dtype(c.elements, c.dtype, label)
        if c.dtype is None and rows is not None:
            raise InvalidArgument('Must specify a dtype for all columns when combining rows with columns.')
        c.fill = npst.fill_for(fill=c.fill, elements=c.elements, unique=c.unique, name=label)
        rewritten_columns.append(c)
    if rows is None:
        @st.composite
        def just_draw_columns(draw: Any) -> pandas.DataFrame:
            index = draw(index_strategy)
            local_index_strategy = st.just(index)
            data = OrderedDict(((c.name, None) for c in rewritten_columns))
            columns_without_fill = [c for c in rewritten_columns if c.fill.is_empty]
            if columns_without_fill:
                for c in columns_without_fill:
                    data[c.name] = pandas.Series(np.zeros(shape=len(index), dtype=object), index=index, dtype=c.dtype)
                seen = {c.name: set() for c in columns_without_fill if c.unique}
                for i in range(len(index)):
                    for c in columns_without_fill:
                        if c.unique:
                            for _ in range(5):
                                value = draw(c.elements)
                                if value not in seen[c.name]:
                                    seen[c.name].add(value)
                                    break
                            else:
                                reject()
                        else:
                            value = draw(c.elements)
                        try:
                            data[c.name][i] = value
                        except ValueError as err:
                            if c.dtype is None and (not isinstance(value, (float, int, str, bool, datetime, timedelta))):
                                raise ValueError(f'Failed to add value={value!r} to column {c.name} with dtype=None.  Maybe passing dtype=object would help?') from err
                            raise
            for c in rewritten_columns:
                if not c.fill.is_empty:
                    data[c.name] = draw(series(index=local_index_strategy, dtype=c.dtype, elements=c.elements, fill=c.fill, unique=c.unique))
            return pandas.DataFrame(data, index=index)
        return just_draw_columns()
    else:
        @st.composite
        def assign_rows(draw: Any) -> pandas.DataFrame:
            index = draw(index_strategy)
            result = pandas.DataFrame(OrderedDict(((c.name, pandas.Series(np.zeros(dtype=c.dtype, shape=len(index)), dtype=c.dtype)) for c in rewritten_columns)), index=index)
            fills = {}
            any_unique = any((c.unique for c in rewritten_columns))
            if any_unique:
                all_seen = [set() if c.unique else None for c in rewritten_columns]
                while all_seen[-1] is None:
                    all_seen.pop()
            for row_index in range(len(index)):
                for _ in range(5):
                    original_row = draw(rows)
                    row = original_row
                    if isinstance(row, dict):
                        as_list = [None] * len(rewritten_columns)
                        for i, c in enumerate(rewritten_columns):
                            try:
                                as_list[i] = row[c.name]
                            except KeyError:
                                try:
                                    as_list[i] = fills[i]
                                except KeyError:
                                    if c.fill.is_empty:
                                        raise InvalidArgument(f'Empty fill strategy in {c!r} cannot complete row {original_row!r}') from None
                                    fills[i] = draw(c.fill)
                                    as_list[i] = fills[i]
                        for k in row:
                            if k not in column_names:
                                raise InvalidArgument('Row %r contains column %r not in columns %r)' % (row, k, [c.name for c in rewritten_columns]))
                        row = as_list
                    if any_unique:
                        has_duplicate = False
                        for seen, value in zip(all_seen, row):
                            if seen is None:
                                continue
                            if value in seen:
                                has_duplicate = True
                                break
                            seen.add(value)
                        if has_duplicate:
                            continue
                    row = list(try_convert(tuple, row, 'draw(rows)'))
                    if len(row) > len(rewritten_columns):
                        raise InvalidArgument(f'Row {original_row!r} contains too many entries. Has {len(row)} but expected at most {len(rewritten_columns)}')
                    while len(row) < len(rewritten_columns):
                        c = rewritten_columns[len(row)]
                        if c.fill.is_empty:
                            raise InvalidArgument(f'Empty fill strategy in {c!r} cannot complete row {original_row!r}')
                        row.append(draw(c.fill))
                    result.iloc[row_index] = row
                    break
                else:
                    reject()
            return result
        return assign_rows()
