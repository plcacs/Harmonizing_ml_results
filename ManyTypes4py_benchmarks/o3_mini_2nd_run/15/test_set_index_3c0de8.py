#!/usr/bin/env python3
"""
See also: test_reindex.py:TestReindexSetIndex
"""

from datetime import datetime, timedelta
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pytest
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    date_range,
    period_range,
    to_datetime,
)
import pandas._testing as tm


@pytest.fixture
def frame_of_index_cols() -> DataFrame:
    """
    Fixture for DataFrame of columns that can be used for indexing

    Columns are ['A', 'B', 'C', 'D', 'E', ('tuple', 'as', 'label')];
    'A' & 'B' contain duplicates (but are jointly unique), the rest are unique.

         A      B  C         D         E  (tuple, as, label)
    0  foo    one  a  0.608477 -0.012500           -1.664297
    1  foo    two  b -0.633460  0.249614           -0.364411
    2  foo  three  c  0.615256  2.154968           -0.834666
    3  bar    one  d  0.234246  1.085675            0.718445
    4  bar    two  e  0.533841 -0.005702           -3.533912
    """
    df: DataFrame = DataFrame({
        'A': ['foo', 'foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'three', 'one', 'two'],
        'C': ['a', 'b', 'c', 'd', 'e'],
        'D': np.random.default_rng(2).standard_normal(5),
        'E': np.random.default_rng(2).standard_normal(5),
        ('tuple', 'as', 'label'): np.random.default_rng(2).standard_normal(5),
    })
    return df


class TestSetIndex:

    def test_set_index_multiindex(self) -> None:
        d: dict[str, List[Union[int, float]]] = {'t1': [2, 2.5, 3], 't2': [4, 5, 6]}
        df: DataFrame = DataFrame(d)
        tuples: List[Tuple[int, int]] = [(0, 1), (0, 2), (1, 2)]
        df['tuples'] = tuples
        index: MultiIndex = MultiIndex.from_tuples(df['tuples'])
        df.set_index(index)

    def test_set_index_empty_column(self) -> None:
        df: DataFrame = DataFrame(
            [{'a': 1, 'p': 0}, {'a': 2, 'm': 10}, {'a': 3, 'm': 11, 'p': 20}, {'a': 4, 'm': 12, 'p': 21}],
            columns=['a', 'm', 'p', 'x']
        )
        result: DataFrame = df.set_index(['a', 'x'])
        expected: DataFrame = df[['m', 'p']]
        expected.index = MultiIndex.from_arrays([df['a'], df['x']], names=['a', 'x'])
        tm.assert_frame_equal(result, expected)

    def test_set_index_empty_dataframe(self) -> None:
        df1: DataFrame = DataFrame({
            'a': Series(dtype='datetime64[ns]'),
            'b': Series(dtype='int64'),
            'c': []
        })
        df2: DataFrame = df1.set_index(['a', 'b'])
        result: Any = df2.index.to_frame().dtypes
        expected: Any = df1[['a', 'b']].dtypes
        tm.assert_series_equal(result, expected)

    def test_set_index_multiindexcolumns(self) -> None:
        columns: MultiIndex = MultiIndex.from_tuples([('foo', 1), ('foo', 2), ('bar', 1)])
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=columns)
        result: DataFrame = df.set_index(df.columns[0])
        expected: DataFrame = df.iloc[:, 1:]
        expected.index = df.iloc[:, 0].values
        expected.index.names = [df.columns[0]]
        tm.assert_frame_equal(result, expected)

    def test_set_index_timezone(self) -> None:
        idx: DatetimeIndex = DatetimeIndex(['2014-01-01 10:10:10'], tz='UTC').tz_convert('Europe/Rome')
        df: DataFrame = DataFrame({'A': idx})
        assert df.set_index(idx).index[0].hour == 11
        assert DatetimeIndex(Series(df.A))[0].hour == 11
        assert df.set_index(df.A).index[0].hour == 11

    def test_set_index_cast_datetimeindex(self) -> None:
        df: DataFrame = DataFrame({
            'A': [datetime(2000, 1, 1) + timedelta(i) for i in range(1000)],
            'B': np.random.default_rng(2).standard_normal(1000)
        })
        idf: DataFrame = df.set_index('A')
        assert isinstance(idf.index, DatetimeIndex)

    def test_set_index_dst(self) -> None:
        di: DatetimeIndex = date_range('2006-10-29 00:00:00', periods=3, freq='h', tz='US/Pacific')
        df: DataFrame = DataFrame(data={'a': [0, 1, 2], 'b': [3, 4, 5]}, index=di).reset_index()
        res: DataFrame = df.set_index('index')
        exp: DataFrame = DataFrame(data={'a': [0, 1, 2], 'b': [3, 4, 5]}, index=Index(di, name='index'))
        exp.index = exp.index._with_freq(None)
        tm.assert_frame_equal(res, exp)
        res = df.set_index(['index', 'a'])
        exp_index: MultiIndex = MultiIndex.from_arrays([di, [0, 1, 2]], names=['index', 'a'])
        exp = DataFrame({'b': [3, 4, 5]}, index=exp_index)
        tm.assert_frame_equal(res, exp)

    def test_set_index(self, float_string_frame: DataFrame) -> None:
        df: DataFrame = float_string_frame
        idx: Index = Index(np.arange(len(df) - 1, -1, -1, dtype=np.int64))
        df = df.set_index(idx)
        tm.assert_index_equal(df.index, idx)
        with pytest.raises(ValueError, match='Length mismatch'):
            df.set_index(idx[::2])

    def test_set_index_names(self) -> None:
        df: DataFrame = DataFrame(np.ones((10, 4)), columns=Index(list('ABCD'), dtype=object),
                                  index=Index([f'i-{i}' for i in range(10)], dtype=object))
        df.index.name = 'name'
        assert df.set_index(df.index).index.names == ['name']
        mi: MultiIndex = MultiIndex.from_arrays(df[['A', 'B']].T.values, names=['A', 'B'])
        mi2: MultiIndex = MultiIndex.from_arrays(df[['A', 'B', 'A', 'B']].T.values, names=['A', 'B', 'C', 'D'])
        df = df.set_index(['A', 'B'])
        assert df.set_index(df.index).index.names == ['A', 'B']
        assert isinstance(df.set_index(df.index).index, MultiIndex)
        tm.assert_index_equal(df.set_index(df.index).index, mi)
        idx2: Index = df.index.rename(['C', 'D'])
        assert isinstance(df.set_index([df.index, idx2]).index, MultiIndex)
        tm.assert_index_equal(df.set_index([df.index, idx2]).index, mi2)

    @pytest.mark.parametrize('keys', ['A', 'C', ['A', 'B'], ('tuple', 'as', 'label')])
    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('drop', [True, False])
    def test_set_index_drop_inplace(
        self,
        frame_of_index_cols: DataFrame,
        drop: bool,
        inplace: bool,
        keys: Union[str, List[str], Tuple[str, ...]]
    ) -> None:
        df: DataFrame = frame_of_index_cols
        if isinstance(keys, list):
            idx: MultiIndex = MultiIndex.from_arrays([df[x] for x in keys], names=keys)
        else:
            idx: Index = Index(df[keys], name=keys)
        expected: DataFrame = df.drop(keys, axis=1) if drop else df
        expected.index = idx
        if inplace:
            result: DataFrame = df.copy()
            return_value: Any = result.set_index(keys, drop=drop, inplace=True)
            assert return_value is None
        else:
            result = df.set_index(keys, drop=drop)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('keys', ['A', 'C', ['A', 'B'], ('tuple', 'as', 'label')])
    @pytest.mark.parametrize('drop', [True, False])
    def test_set_index_append(
        self,
        frame_of_index_cols: DataFrame,
        drop: bool,
        keys: Union[str, List[str], Tuple[str, ...]]
    ) -> None:
        df: DataFrame = frame_of_index_cols
        keys_list: List[Any] = keys if isinstance(keys, list) else [keys]
        idx: MultiIndex = MultiIndex.from_arrays([df.index] + [df[x] for x in keys_list], names=[None] + keys_list)
        expected: DataFrame = df.drop(keys_list, axis=1) if drop else df.copy()
        expected.index = idx
        result: DataFrame = df.set_index(keys, drop=drop, append=True)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('keys', ['A', 'C', ['A', 'B'], ('tuple', 'as', 'label')])
    @pytest.mark.parametrize('drop', [True, False])
    def test_set_index_append_to_multiindex(
        self,
        frame_of_index_cols: DataFrame,
        drop: bool,
        keys: Union[str, List[str], Tuple[str, ...]]
    ) -> None:
        df: DataFrame = frame_of_index_cols.set_index(['D'], drop=drop, append=True)
        keys_list: List[Any] = keys if isinstance(keys, list) else [keys]
        expected: DataFrame = frame_of_index_cols.set_index(['D'] + keys_list, drop=drop, append=True)
        result: DataFrame = df.set_index(keys, drop=drop, append=True)
        tm.assert_frame_equal(result, expected)

    def test_set_index_after_mutation(self) -> None:
        df: DataFrame = DataFrame({'val': [0, 1, 2], 'key': ['a', 'b', 'c']})
        expected: DataFrame = DataFrame({'val': [1, 2]}, Index(['b', 'c'], name='key'))
        df2: DataFrame = df.loc[df.index.map(lambda indx: indx >= 1)]
        result: DataFrame = df2.set_index('key')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('box', [Series, Index, np.array, list, lambda x: [list(x)], lambda x: MultiIndex.from_arrays([x])])
    @pytest.mark.parametrize('append, index_name', [(True, None), (True, 'B'), (True, 'test'), (False, None)])
    @pytest.mark.parametrize('drop', [True, False])
    def test_set_index_pass_single_array(
        self,
        frame_of_index_cols: DataFrame,
        drop: bool,
        append: bool,
        index_name: Optional[str],
        box: Callable[[Any], Any]
    ) -> None:
        df: DataFrame = frame_of_index_cols
        df.index.name = index_name
        key: Any = box(df['B'])
        if box == list:
            msg: str = "['one', 'two', 'three', 'one', 'two']"
            with pytest.raises(KeyError, match=msg):
                df.set_index(key, drop=drop, append=append)
        else:
            name_mi: Any = getattr(key, 'names', None)
            name: List[Optional[str]] = [getattr(key, 'name', None)] if name_mi is None else name_mi
            result: DataFrame = df.set_index(key, drop=drop, append=append)
            expected: DataFrame = df.set_index(['B'], drop=False, append=append)
            expected.index.names = ([index_name] + name) if append else name
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('box', [Series, Index, np.array, list, lambda x: MultiIndex.from_arrays([x])])
    @pytest.mark.parametrize('append, index_name', [(True, None), (True, 'A'), (True, 'B'), (True, 'test'), (False, None)])
    @pytest.mark.parametrize('drop', [True, False])
    def test_set_index_pass_arrays(
        self,
        frame_of_index_cols: DataFrame,
        drop: bool,
        append: bool,
        index_name: Optional[str],
        box: Callable[[Any], Any]
    ) -> None:
        df: DataFrame = frame_of_index_cols
        df.index.name = index_name
        keys: List[Any] = ['A', box(df['B'])]
        names: List[Optional[str]] = ['A', None if box in [np.array, list, tuple, iter] else 'B']
        result: DataFrame = df.set_index(keys, drop=drop, append=append)
        expected: DataFrame = df.set_index(['A', 'B'], drop=False, append=append)
        expected = expected.drop('A', axis=1) if drop else expected
        expected.index.names = ([index_name] + names) if append else names
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'box2', [Series, Index, np.array, list, iter, lambda x: MultiIndex.from_arrays([x]), lambda x: x.name],
        ids=['Series', 'Index', 'np.array', 'iter', 'MultiIndex']
    )
    @pytest.mark.parametrize(
        'box1', [Series, Index, np.array, list, iter, lambda x: MultiIndex.from_arrays([x]), lambda x: x.name]
    )
    @pytest.mark.parametrize('append, index_name', [(True, None), (True, 'A'), (True, 'test'), (False, None)])
    @pytest.mark.parametrize('drop', [True, False])
    def test_set_index_pass_arrays_duplicate(
        self,
        frame_of_index_cols: DataFrame,
        drop: bool,
        append: bool,
        index_name: Optional[str],
        box1: Callable[[Any], Any],
        box2: Callable[[Any], Any]
    ) -> None:
        df: DataFrame = frame_of_index_cols
        df.index.name = index_name
        keys: List[Any] = [box1(df['A']), box2(df['A'])]
        result: DataFrame = df.set_index(keys, drop=drop, append=append)
        keys = [box1(df['A']), box2(df['A'])]
        first_drop: bool = False if (isinstance(keys[0], str) and keys[0] == 'A' and
                                       isinstance(keys[1], str) and (keys[1] == 'A')) else drop
        expected: DataFrame = df.set_index([keys[0]], drop=first_drop, append=append)
        expected = expected.set_index([keys[1]], drop=drop, append=True)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('append', [True, False])
    @pytest.mark.parametrize('drop', [True, False])
    def test_set_index_pass_multiindex(
        self,
        frame_of_index_cols: DataFrame,
        drop: bool,
        append: bool
    ) -> None:
        df: DataFrame = frame_of_index_cols
        keys: MultiIndex = MultiIndex.from_arrays([df['A'], df['B']], names=['A', 'B'])
        result: DataFrame = df.set_index(keys, drop=drop, append=append)
        expected: DataFrame = df.set_index(['A', 'B'], drop=False, append=append)
        tm.assert_frame_equal(result, expected)

    def test_construction_with_categorical_index(self) -> None:
        ci: CategoricalIndex = CategoricalIndex(list('ab') * 5, name='B')
        df: DataFrame = DataFrame({'A': np.random.default_rng(2).standard_normal(10), 'B': ci.values})
        idf: DataFrame = df.set_index('B')
        tm.assert_index_equal(idf.index, ci)
        df = DataFrame({'A': np.random.default_rng(2).standard_normal(10), 'B': ci})
        idf = df.set_index('B')
        tm.assert_index_equal(idf.index, ci)
        idf = idf.reset_index().set_index('B')
        tm.assert_index_equal(idf.index, ci)

    def test_set_index_preserve_categorical_dtype(self) -> None:
        df: DataFrame = DataFrame({
            'A': [1, 2, 1, 1, 2],
            'B': [10, 16, 22, 28, 34],
            'C1': Categorical(list('abaab'), categories=list('bac'), ordered=False),
            'C2': Categorical(list('abaab'), categories=list('bac'), ordered=True)
        })
        for cols in ['C1', 'C2', ['A', 'C1'], ['A', 'C2'], ['C1', 'C2']]:
            result: DataFrame = df.set_index(cols).reset_index()
            result = result.reindex(columns=df.columns)
            tm.assert_frame_equal(result, df)

    def test_set_index_datetime(self) -> None:
        df: DataFrame = DataFrame({
            'label': ['a', 'a', 'a', 'b', 'b', 'b'],
            'datetime': ['2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00',
                         '2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00'],
            'value': list(range(6))
        })
        df.index = to_datetime(df.pop('datetime'), utc=True)
        df.index = df.index.tz_convert('US/Pacific')
        expected: DatetimeIndex = DatetimeIndex(['2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00'], name='datetime')
        expected = expected.tz_localize('UTC').tz_convert('US/Pacific')
        df = df.set_index('label', append=True)
        tm.assert_index_equal(df.index.levels[0], expected)
        tm.assert_index_equal(df.index.levels[1], Index(['a', 'b'], name='label'))
        assert df.index.names == ['datetime', 'label']
        df = df.swaplevel(0, 1)
        tm.assert_index_equal(df.index.levels[0], Index(['a', 'b'], name='label'))
        tm.assert_index_equal(df.index.levels[1], expected)
        assert df.index.names == ['label', 'datetime']
        df = DataFrame(np.random.default_rng(2).random(6))
        idx1: DatetimeIndex = DatetimeIndex(
            ['2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00',
             '2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00'], tz='US/Eastern'
        )
        idx2: DatetimeIndex = DatetimeIndex(
            ['2012-04-01 09:00', '2012-04-01 09:00', '2012-04-01 09:00',
             '2012-04-02 09:00', '2012-04-02 09:00', '2012-04-02 09:00'], tz='US/Eastern'
        )
        idx3: DatetimeIndex = date_range('2011-01-01 09:00', periods=6, tz='Asia/Tokyo')
        idx3 = idx3._with_freq(None)
        df = df.set_index(idx1)
        df = df.set_index(idx2, append=True)
        df = df.set_index(idx3, append=True)
        expected1: DatetimeIndex = DatetimeIndex(['2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00'], tz='US/Eastern')
        expected2: DatetimeIndex = DatetimeIndex(['2012-04-01 09:00', '2012-04-02 09:00'], tz='US/Eastern')
        tm.assert_index_equal(df.index.levels[0], expected1)
        tm.assert_index_equal(df.index.levels[1], expected2)
        tm.assert_index_equal(df.index.levels[2], idx3)
        tm.assert_index_equal(df.index.get_level_values(0), idx1)
        tm.assert_index_equal(df.index.get_level_values(1), idx2)
        tm.assert_index_equal(df.index.get_level_values(2), idx3)

    def test_set_index_period(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random(6))
        idx1 = period_range('2011-01-01', periods=3, freq='M')
        idx1 = idx1.append(idx1)
        idx2 = period_range('2013-01-01 09:00', periods=2, freq='h')
        idx2 = idx2.append(idx2).append(idx2)
        idx3 = period_range('2005', periods=6, freq='Y')
        df = df.set_index(idx1)
        df = df.set_index(idx2, append=True)
        df = df.set_index(idx3, append=True)
        expected1 = period_range('2011-01-01', periods=3, freq='M')
        expected2 = period_range('2013-01-01 09:00', periods=2, freq='h')
        tm.assert_index_equal(df.index.levels[0], expected1)
        tm.assert_index_equal(df.index.levels[1], expected2)
        tm.assert_index_equal(df.index.levels[2], idx3)
        tm.assert_index_equal(df.index.get_level_values(0), idx1)
        tm.assert_index_equal(df.index.get_level_values(1), idx2)
        tm.assert_index_equal(df.index.get_level_values(2), idx3)


class TestSetIndexInvalid:

    def test_set_index_verify_integrity(self, frame_of_index_cols: DataFrame) -> None:
        df: DataFrame = frame_of_index_cols
        with pytest.raises(ValueError, match='Index has duplicate keys'):
            df.set_index('A', verify_integrity=True)
        with pytest.raises(ValueError, match='Index has duplicate keys'):
            df.set_index([df['A'], df['A']], verify_integrity=True)

    @pytest.mark.parametrize('append', [True, False])
    @pytest.mark.parametrize('drop', [True, False])
    def test_set_index_raise_keys(
        self, frame_of_index_cols: DataFrame, drop: bool, append: bool
    ) -> None:
        df: DataFrame = frame_of_index_cols
        with pytest.raises(KeyError, match="['foo', 'bar', 'baz']"):
            df.set_index(['foo', 'bar', 'baz'], drop=drop, append=append)
        with pytest.raises(KeyError, match='X'):
            df.set_index([df['A'], df['B'], 'X'], drop=drop, append=append)
        msg: str = "[('foo', 'foo', 'foo', 'bar', 'bar')]"
        with pytest.raises(KeyError, match=msg):
            df.set_index(tuple(df['A']), drop=drop, append=append)
        with pytest.raises(KeyError, match=msg):
            df.set_index(['A', df['A'], tuple(df['A'])], drop=drop, append=append)

    @pytest.mark.parametrize('append', [True, False])
    @pytest.mark.parametrize('drop', [True, False])
    def test_set_index_raise_on_type(
        self, frame_of_index_cols: DataFrame, drop: bool, append: bool
    ) -> None:
        box: type = set
        df: DataFrame = frame_of_index_cols
        msg: str = 'The parameter "keys" may be a column key, .*'
        with pytest.raises(TypeError, match=msg):
            df.set_index(box(df['A']), drop=drop, append=append)
        with pytest.raises(TypeError, match=msg):
            df.set_index(['A', df['A'], box(df['A'])], drop=drop, append=append)

    @pytest.mark.parametrize('box', [Series, Index, np.array, iter, lambda x: MultiIndex.from_arrays([x])], ids=['Series', 'Index', 'np.array', 'iter', 'MultiIndex'])
    @pytest.mark.parametrize('length', [4, 6], ids=['too_short', 'too_long'])
    @pytest.mark.parametrize('append', [True, False])
    @pytest.mark.parametrize('drop', [True, False])
    def test_set_index_raise_on_len(
        self,
        frame_of_index_cols: DataFrame,
        box: Callable[[Any], Any],
        length: int,
        drop: bool,
        append: bool
    ) -> None:
        df: DataFrame = frame_of_index_cols
        values: np.ndarray = np.random.default_rng(2).integers(0, 10, (length,))
        msg: str = 'Length mismatch: Expected 5 rows, received array of length.*'
        with pytest.raises(ValueError, match=msg):
            df.set_index(box(values), drop=drop, append=append)
        with pytest.raises(ValueError, match=msg):
            df.set_index(['A', df.A, box(values)], drop=drop, append=append)


class TestSetIndexCustomLabelType:

    class Thing:
        def __init__(self, name: str, color: str) -> None:
            self.name: str = name
            self.color: str = color

        def __str__(self) -> str:
            return f'<Thing {self.name!r}>'
        __repr__ = __str__

    def test_set_index_custom_label_type(self) -> None:
        thing1: TestSetIndexCustomLabelType.Thing = self.Thing('One', 'red')
        thing2: TestSetIndexCustomLabelType.Thing = self.Thing('Two', 'blue')
        df: DataFrame = DataFrame({thing1: [0, 1], thing2: [2, 3]})
        expected: DataFrame = DataFrame({thing1: [0, 1]}, index=Index([2, 3], name=thing2))
        result: DataFrame = df.set_index(thing2)
        tm.assert_frame_equal(result, expected)
        result = df.set_index([thing2])
        tm.assert_frame_equal(result, expected)
        thing3: TestSetIndexCustomLabelType.Thing = self.Thing('Three', 'pink')
        msg: str = "<Thing 'Three'>"
        with pytest.raises(KeyError, match=msg):
            df.set_index(thing3)
        with pytest.raises(KeyError, match=msg):
            df.set_index([thing3])

    class ThingFrozenset(frozenset):
        def __repr__(self) -> str:
            tmp = sorted(self)
            joined_reprs = ', '.join(map(repr, tmp))
            return f'frozenset({{{joined_reprs}}})'

    def test_set_index_custom_label_hashable_iterable(self) -> None:
        thing1: TestSetIndexCustomLabelType.ThingFrozenset = self.ThingFrozenset(['One', 'red'])
        thing2: TestSetIndexCustomLabelType.ThingFrozenset = self.ThingFrozenset(['Two', 'blue'])
        df: DataFrame = DataFrame({thing1: [0, 1], thing2: [2, 3]})
        expected: DataFrame = DataFrame({thing1: [0, 1]}, index=Index([2, 3], name=thing2))
        result: DataFrame = df.set_index(thing2)
        tm.assert_frame_equal(result, expected)
        result = df.set_index([thing2])
        tm.assert_frame_equal(result, expected)
        thing3: TestSetIndexCustomLabelType.ThingFrozenset = self.ThingFrozenset(['Three', 'pink'])
        import re
        msg: str = r"frozenset\(\{'Three', 'pink'\}\)"
        with pytest.raises(KeyError, match=msg):
            df.set_index(thing3)
        with pytest.raises(KeyError, match=msg):
            df.set_index([thing3])

    class ThingSet(set):
        def __init__(self, name: str, color: str) -> None:
            self.name: str = name
            self.color: str = color
            super().__init__()

        def __str__(self) -> str:
            return f'<Thing {self.name!r}>'

    def test_set_index_custom_label_type_raises(self) -> None:
        thing1: TestSetIndexCustomLabelType.ThingSet = self.ThingSet('One', 'red')
        thing2: TestSetIndexCustomLabelType.ThingSet = self.ThingSet('Two', 'blue')
        df: DataFrame = DataFrame([[0, 2], [1, 3]], columns=[thing1, thing2])
        msg: str = 'The parameter "keys" may be a column key, .*'
        with pytest.raises(TypeError, match=msg):
            df.set_index(thing2)
        with pytest.raises(TypeError, match=msg):
            df.set_index([thing2])

    def test_set_index_periodindex(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random(6))
        idx1 = period_range('2011/01/01', periods=6, freq='M')
        idx2 = period_range('2013', periods=6, freq='Y')
        df = df.set_index(idx1)
        tm.assert_index_equal(df.index, idx1)
        df = df.set_index(idx2)
        tm.assert_index_equal(df.index, idx2)


# End of module
