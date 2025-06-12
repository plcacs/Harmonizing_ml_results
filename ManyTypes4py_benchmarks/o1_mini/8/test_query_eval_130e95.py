import operator
from tokenize import TokenError
import numpy as np
import pytest
from pandas.errors import NumExprClobberingError, UndefinedVariableError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series, date_range
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
from typing import Any, Callable, Dict, List, Optional, Union

@pytest.fixture(params=['python', 'pandas'], ids=lambda x: x)
def parser(request: pytest.FixtureRequest) -> str:
    return request.param

@pytest.fixture(params=['python', pytest.param('numexpr', marks=td.skip_if_no('numexpr'))], ids=lambda x: x)
def engine(request: pytest.FixtureRequest) -> str:
    return request.param

def skip_if_no_pandas_parser(parser: str) -> None:
    if parser != 'pandas':
        pytest.skip(f'cannot evaluate with parser={parser}')

class TestCompat:

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return DataFrame({'A': [1, 2, 3]})

    @pytest.fixture
    def expected1(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df.A > 0]

    @pytest.fixture
    def expected2(self, df: pd.DataFrame) -> pd.Series:
        return df.A + 1

    def test_query_default(self, df: pd.DataFrame, expected1: pd.DataFrame, expected2: pd.Series) -> None:
        result: pd.DataFrame = df.query('A>0')
        tm.assert_frame_equal(result, expected1)
        result = df.eval('A+1')
        tm.assert_series_equal(result, expected2)

    def test_query_None(self, df: pd.DataFrame, expected1: pd.DataFrame, expected2: pd.Series) -> None:
        result: pd.DataFrame = df.query('A>0', engine=None)
        tm.assert_frame_equal(result, expected1)
        result = df.eval('A+1', engine=None)
        tm.assert_series_equal(result, expected2)

    def test_query_python(self, df: pd.DataFrame, expected1: pd.DataFrame, expected2: pd.Series) -> None:
        result: pd.DataFrame = df.query('A>0', engine='python')
        tm.assert_frame_equal(result, expected1)
        result = df.eval('A+1', engine='python')
        tm.assert_series_equal(result, expected2)

    def test_query_numexpr(self, df: pd.DataFrame, expected1: pd.DataFrame, expected2: pd.Series) -> None:
        if NUMEXPR_INSTALLED:
            result: pd.DataFrame = df.query('A>0', engine='numexpr')
            tm.assert_frame_equal(result, expected1)
            result = df.eval('A+1', engine='numexpr')
            tm.assert_series_equal(result, expected2)
        else:
            msg: str = "'numexpr' is not installed or an unsupported version. Cannot use engine='numexpr' for query/eval if 'numexpr' is not installed"
            with pytest.raises(ImportError, match=msg):
                df.query('A>0', engine='numexpr')
            with pytest.raises(ImportError, match=msg):
                df.eval('A+1', engine='numexpr')

class TestDataFrameEval:

    @pytest.mark.parametrize('n', [4, 4000])
    @pytest.mark.parametrize('op_str,op,rop', [
        ('+', '__add__', '__radd__'),
        ('-', '__sub__', '__rsub__'),
        ('*', '__mul__', '__rmul__'),
        ('/', '__truediv__', '__rtruediv__')
    ])
    def test_ops(self, op_str: str, op: str, rop: str, n: int) -> None:
        df: pd.DataFrame = DataFrame(1, index=range(n), columns=list('abcd'))
        df.iloc[0] = 2
        m: pd.Series = df.mean()
        base: pd.DataFrame = DataFrame(np.tile(m.values, n).reshape(n, -1), columns=list('abcd'))
        expected: pd.DataFrame = eval(f'base {op_str} df')
        result: pd.DataFrame = eval(f'm {op_str} df')
        tm.assert_frame_equal(result, expected)
        if op in ['+', '*']:
            result = getattr(df, op)(m)
            tm.assert_frame_equal(result, expected)
        elif op in ['-', '/']:
            result = getattr(df, rop)(m)
            tm.assert_frame_equal(result, expected)

    def test_dataframe_sub_numexpr_path(self) -> None:
        df: pd.DataFrame = DataFrame({'A': np.random.default_rng(2).standard_normal(25000)})
        df.iloc[0:5] = np.nan
        expected: pd.DataFrame = 1 - np.isnan(df.iloc[0:25])
        result: pd.DataFrame = (1 - np.isnan(df)).iloc[0:25]
        tm.assert_frame_equal(result, expected)

    def test_query_non_str(self) -> None:
        df: pd.DataFrame = DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'b']})
        msg: str = 'expr must be a string to be evaluated'
        with pytest.raises(ValueError, match=msg):
            df.query(lambda x: x.B == 'b')
        with pytest.raises(ValueError, match=msg):
            df.query(111)

    def test_query_empty_string(self) -> None:
        df: pd.DataFrame = DataFrame({'A': [1, 2, 3]})
        msg: str = 'expr cannot be an empty string'
        with pytest.raises(ValueError, match=msg):
            df.query('')

    def test_query_duplicate_column_name(self, engine: str, parser: str) -> None:
        df: pd.DataFrame = DataFrame({'A': range(3), 'B': range(3), 'C': range(3)}).rename(columns={'B': 'A'})
        res: pd.DataFrame = df.query('C == 1', engine=engine, parser=parser)
        expect: pd.DataFrame = DataFrame([[1, 1, 1]], columns=['A', 'A', 'C'], index=[1])
        tm.assert_frame_equal(res, expect)

    def test_eval_resolvers_as_list(self) -> None:
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=list('ab'))
        dict1: Dict[str, int] = {'a': 1}
        dict2: Dict[str, int] = {'b': 2}
        assert df.eval('a + b', resolvers=[dict1, dict2]) == dict1['a'] + dict2['b']
        assert pd.eval('a + b', resolvers=[dict1, dict2]) == dict1['a'] + dict2['b']

    def test_eval_resolvers_combined(self) -> None:
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=list('ab'))
        dict1: Dict[str, int] = {'c': 2}
        result: pd.Series = df.eval('a + b * c', resolvers=[dict1])
        expected: pd.Series = df['a'] + df['b'] * dict1['c']
        tm.assert_series_equal(result, expected)

    def test_eval_object_dtype_binop(self) -> None:
        df: pd.DataFrame = DataFrame({'a1': ['Y', 'N']})
        res: pd.DataFrame = df.eval("c = ((a1 == 'Y') & True)")
        expected: pd.DataFrame = DataFrame({'a1': ['Y', 'N'], 'c': [True, False]})
        tm.assert_frame_equal(res, expected)

    def test_using_numpy(self, engine: str, parser: str) -> None:
        skip_if_no_pandas_parser(parser)
        df: pd.DataFrame = Series([0.2, 1.5, 2.8], name='a').to_frame()
        res: pd.Series = df.eval('@np.floor(a)', engine=engine, parser=parser)
        expected: pd.Series = np.floor(df['a'])
        tm.assert_series_equal(expected, res)

    def test_eval_simple(self, engine: str, parser: str) -> None:
        df: pd.DataFrame = Series([0.2, 1.5, 2.8], name='a').to_frame()
        res: pd.Series = df.eval('a', engine=engine, parser=parser)
        expected: pd.Series = df['a']
        tm.assert_series_equal(expected, res)

    def test_extension_array_eval(self, engine: str, parser: str, request: pytest.FixtureRequest) -> None:
        if engine == 'numexpr':
            mark = pytest.mark.xfail(reason='numexpr does not support extension array dtypes')
            request.applymarker(mark)
        df: pd.DataFrame = DataFrame({'a': pd.array([1, 2, 3]), 'b': pd.array([4, 5, 6])})
        result: pd.Series = df.eval('a / b', engine=engine, parser=parser)
        expected: pd.Series = Series(pd.array([0.25, 0.4, 0.5]))
        tm.assert_series_equal(result, expected)

    def test_complex_eval(self, engine: str, parser: str) -> None:
        df: pd.DataFrame = DataFrame({'a': [1 + 2j], 'b': [1 + 1j]})
        result: pd.Series = df.eval('a/b', engine=engine, parser=parser)
        expected: pd.Series = Series([1.5 + 0.5j])
        tm.assert_series_equal(result, expected)

class TestDataFrameQueryWithMultiIndex:

    def test_query_with_named_multiindex(self, parser: str, engine: str) -> None:
        skip_if_no_pandas_parser(parser)
        a: np.ndarray = np.random.default_rng(2).choice(['red', 'green'], size=10)
        b: np.ndarray = np.random.default_rng(2).choice(['eggs', 'ham'], size=10)
        index: pd.MultiIndex = MultiIndex.from_arrays([a, b], names=['color', 'food'])
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        ind: pd.Series = Series(df.index.get_level_values('color').values, index=index, name='color')
        res1: pd.DataFrame = df.query('color == "red"', parser=parser, engine=engine)
        res2: pd.DataFrame = df.query('"red" == color', parser=parser, engine=engine)
        exp: pd.DataFrame = df[ind == 'red']
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('color != "red"', parser=parser, engine=engine)
        res2 = df.query('"red" != color', parser=parser, engine=engine)
        exp = df[ind != 'red']
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('color == ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] == color', parser=parser, engine=engine)
        exp = df[ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('color != ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] != color', parser=parser, engine=engine)
        exp = df[~ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('["red"] in color', parser=parser, engine=engine)
        res2 = df.query('"red" in color', parser=parser, engine=engine)
        exp = df[ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('["red"] not in color', parser=parser, engine=engine)
        res2 = df.query('"red" not in color', parser=parser, engine=engine)
        exp = df[~ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

    def test_query_with_unnamed_multiindex(self, parser: str, engine: str) -> None:
        skip_if_no_pandas_parser(parser)
        a: np.ndarray = np.random.default_rng(2).choice(['red', 'green'], size=10)
        b: np.ndarray = np.random.default_rng(2).choice(['eggs', 'ham'], size=10)
        index: pd.MultiIndex = MultiIndex.from_arrays([a, b])
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        ind: pd.Series = Series(df.index.get_level_values(0).values, index=index)
        res1: pd.DataFrame = df.query('ilevel_0 == "red"', parser=parser, engine=engine)
        res2: pd.DataFrame = df.query('"red" == ilevel_0', parser=parser, engine=engine)
        exp: pd.DataFrame = df[ind == 'red']
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('ilevel_0 != "red"', parser=parser, engine=engine)
        res2 = df.query('"red" != ilevel_0', parser=parser, engine=engine)
        exp = df[ind != 'red']
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('ilevel_0 == ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] == ilevel_0', parser=parser, engine=engine)
        exp = df[ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('ilevel_0 != ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] != ilevel_0', parser=parser, engine=engine)
        exp = df[~ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('["red"] in ilevel_0', parser=parser, engine=engine)
        res2 = df.query('"red" in ilevel_0', parser=parser, engine=engine)
        exp = df[ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('["red"] not in ilevel_0', parser=parser, engine=engine)
        res2 = df.query('"red" not in ilevel_0', parser=parser, engine=engine)
        exp = df[~ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        ind = Series(df.index.get_level_values(1).values, index=index)
        res1 = df.query('ilevel_1 == "eggs"', parser=parser, engine=engine)
        res2 = df.query('"eggs" == ilevel_1', parser=parser, engine=engine)
        exp = df[ind == 'eggs']
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('ilevel_1 != "eggs"', parser=parser, engine=engine)
        res2 = df.query('"eggs" != ilevel_1', parser=parser, engine=engine)
        exp = df[ind != 'eggs']
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('ilevel_1 == ["eggs"]', parser=parser, engine=engine)
        res2 = df.query('["eggs"] == ilevel_1', parser=parser, engine=engine)
        exp = df[ind.isin(['eggs'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('ilevel_1 != ["eggs"]', parser=parser, engine=engine)
        res2 = df.query('["eggs"] != ilevel_1', parser=parser, engine=engine)
        exp = df[~ind.isin(['eggs'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('["eggs"] in ilevel_1', parser=parser, engine=engine)
        res2 = df.query('"eggs" in ilevel_1', parser=parser, engine=engine)
        exp = df[ind.isin(['eggs'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('["eggs"] not in ilevel_1', parser=parser, engine=engine)
        res2 = df.query('"eggs" not in ilevel_1', parser=parser, engine=engine)
        exp = df[~ind.isin(['eggs'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

    def test_query_with_partially_named_multiindex(self, parser: str, engine: str) -> None:
        skip_if_no_pandas_parser(parser)
        a: np.ndarray = np.random.default_rng(2).choice(['red', 'green'], size=10)
        b: np.ndarray = np.arange(10)
        index: pd.MultiIndex = MultiIndex.from_arrays([a, b])
        index.names = [None, 'rating']
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        res: pd.DataFrame = df.query('rating == 1', parser=parser, engine=engine)
        ind: pd.Series = Series(df.index.get_level_values('rating').values, index=index, name='rating')
        exp: pd.DataFrame = df[ind == 1]
        tm.assert_frame_equal(res, exp)
        res = df.query('rating != 1', parser=parser, engine=engine)
        ind = Series(df.index.get_level_values('rating').values, index=index, name='rating')
        exp = df[ind != 1]
        tm.assert_frame_equal(res, exp)
        res = df.query('ilevel_0 == "red"', parser=parser, engine=engine)
        ind = Series(df.index.get_level_values(0).values, index=index)
        exp = df[ind == 'red']
        tm.assert_frame_equal(res, exp)
        res = df.query('ilevel_0 != "red"', parser=parser, engine=engine)
        ind = Series(df.index.get_level_values(0).values, index=index)
        exp = df[ind != 'red']
        tm.assert_frame_equal(res, exp)

    def test_query_multiindex_get_index_resolvers(self) -> None:
        df: pd.DataFrame = DataFrame(
            np.ones((10, 3)),
            index=MultiIndex.from_arrays([range(10) for _ in range(2)], names=['spam', 'eggs'])
        )
        resolvers: Dict[str, Union[pd.Index, pd.Series]] = df._get_index_resolvers()

        def to_series(mi: pd.MultiIndex, level: str) -> pd.Series:
            level_values: np.ndarray = mi.get_level_values(level).values
            s: pd.Series = pd.Series(level_values, index=mi)
            return s

        col_series: pd.Series = df.columns.to_series()
        expected: Dict[str, Union[pd.Index, pd.Series]] = {
            'index': df.index,
            'columns': col_series,
            'spam': to_series(df.index, 'spam'),
            'eggs': to_series(df.index, 'eggs'),
            'clevel_0': col_series
        }
        for k, v in resolvers.items():
            if isinstance(v, Index):
                assert v.is_(expected[k])  # type: ignore
            elif isinstance(v, Series):
                tm.assert_series_equal(v, expected[k])  # type: ignore
            else:
                raise AssertionError('object must be a Series or Index')

@td.skip_if_no('numexpr')
class TestDataFrameQueryNumExprPandas:

    @pytest.fixture
    def engine(self) -> str:
        return 'numexpr'

    @pytest.fixture
    def parser(self) -> str:
        return 'pandas'

    def test_date_query_with_attribute_access(self, engine: str, parser: str) -> None:
        skip_if_no_pandas_parser(parser)
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df['dates1']: pd.DatetimeIndex = date_range('1/1/2012', periods=5)
        df['dates2']: pd.DatetimeIndex = date_range('1/1/2013', periods=5)
        df['dates3']: pd.DatetimeIndex = date_range('1/1/2014', periods=5)
        res: pd.DataFrame = df.query('@df.dates1 < 20130101 < @df.dates3', engine=engine, parser=parser)
        expec: pd.DataFrame = df[(df.dates1 < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_no_attribute_access(self, engine: str, parser: str) -> None:
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df['dates1'] = date_range('1/1/2012', periods=5)
        df['dates2'] = date_range('1/1/2013', periods=5)
        df['dates3'] = date_range('1/1/2014', periods=5)
        res: pd.DataFrame = df.query('dates1 < 20130101 < dates3', engine=engine, parser=parser)
        expec: pd.DataFrame = df[(df.dates1 < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_NaT(self, engine: str, parser: str) -> None:
        n: int = 10
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df['dates1'] = date_range('1/1/2012', periods=n)
        df['dates2'] = date_range('1/1/2013', periods=n)
        df['dates3'] = date_range('1/1/2014', periods=n)
        df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates1'] = pd.NaT
        df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates3'] = pd.NaT
        res: pd.DataFrame = df.query('dates1 < 20130101 < dates3', engine=engine, parser=parser)
        expec: pd.DataFrame = df[(df.dates1 < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query(self, engine: str, parser: str) -> None:
        n: int = 10
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df['dates1'] = date_range('1/1/2012', periods=n)
        df['dates3'] = date_range('1/1/2014', periods=n)
        return_value: Optional[bool] = df.set_index('dates1', inplace=True, drop=True)
        assert return_value is None
        res: pd.DataFrame = df.query('index < 20130101 < dates3', engine=engine, parser=parser)
        expec: pd.DataFrame = df[(df.index < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT(self, engine: str, parser: str) -> None:
        n: int = 10
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((n, 3))).astype({0: object})
        df['dates1'] = date_range('1/1/2012', periods=n)
        df['dates3'] = date_range('1/1/2014', periods=n)
        df.iloc[0, 0] = pd.NaT
        return_value = df.set_index('dates1', inplace=True, drop=True)
        assert return_value is None
        res: pd.DataFrame = df.query('index < 20130101 < dates3', engine=engine, parser=parser)
        expec: pd.DataFrame = df[(df.index < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT_duplicates(self, engine: str, parser: str) -> None:
        n: int = 10
        d: Dict[str, pd.DatetimeIndex] = {}
        d['dates1'] = date_range('1/1/2012', periods=n)
        d['dates3'] = date_range('1/1/2014', periods=n)
        df: pd.DataFrame = DataFrame(d)
        df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates1'] = pd.NaT
        return_value: Optional[bool] = df.set_index('dates1', inplace=True, drop=True)
        assert return_value is None
        res: pd.DataFrame = df.query('dates1 < 20130101 < dates3', engine=engine, parser=parser)
        expec: pd.DataFrame = df[(df.index.to_series() < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_non_date(self, engine: str, parser: str) -> None:
        n: int = 10
        df: pd.DataFrame = DataFrame({'dates': date_range('1/1/2012', periods=n), 'nondate': np.arange(n)})
        result: pd.DataFrame = df.query('dates == nondate', parser=parser, engine=engine)
        assert len(result) == 0
        result = df.query('dates != nondate', parser=parser, engine=engine)
        tm.assert_frame_equal(result, df)
        msg: str = 'Invalid comparison between dtype=datetime64\\[ns\\] and ndarray'
        for op in ['<', '>', '<=', '>=']:
            with pytest.raises(TypeError, match=msg):
                df.query(f'dates {op} nondate', parser=parser, engine=engine)

    def test_query_syntax_error(self, engine: str, parser: str) -> None:
        df: pd.DataFrame = DataFrame({'i': range(10), '+': range(3, 13), 'r': range(4, 14)})
        msg: str = 'invalid syntax'
        with pytest.raises(SyntaxError, match=msg):
            df.query('i - +', engine=engine, parser=parser)

    def test_query_scope(self, engine: str, parser: str) -> None:
        skip_if_no_pandas_parser(parser)
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((20, 2)), columns=list('ab'))
        a: int = 1
        b: int = 2
        res: pd.DataFrame = df.query('a > b', engine=engine, parser=parser)
        expected: pd.DataFrame = df[df.a > df.b]
        tm.assert_frame_equal(res, expected)
        res = df.query('@a > b', engine=engine, parser=parser)
        expected = df[a > df.b]
        tm.assert_frame_equal(res, expected)
        with pytest.raises(UndefinedVariableError, match="local variable 'c' is not defined"):
            df.query('@a > b > @c', engine=engine, parser=parser)
        with pytest.raises(UndefinedVariableError, match="name 'c' is not defined"):
            df.query('@a > b > c', engine=engine, parser=parser)

    def test_query_doesnt_pickup_local(self, engine: str, parser: str) -> None:
        n: int = m = 10
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).integers(m, size=(n, 3)), columns=list('abc'))
        with pytest.raises(UndefinedVariableError, match="name 'sin' is not defined"):
            df.query('sin > 5', engine=engine, parser=parser)

    def test_query_builtin(self, engine: str, parser: str) -> None:
        n: int = m = 10
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).integers(m, size=(n, 3)), columns=list('abc'))
        df.index.name = 'sin'
        msg: str = 'Variables in expression.+'
        with pytest.raises(NumExprClobberingError, match=msg):
            df.query('sin > 5', engine=engine, parser=parser)

    def test_query(self, engine: str, parser: str) -> None:
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=['a', 'b', 'c'])
        tm.assert_frame_equal(
            df.query('a < b', engine=engine, parser=parser),
            df[df.a < df.b]
        )
        tm.assert_frame_equal(
            df.query('a + b > b * c', engine=engine, parser=parser),
            df[df.a + df.b > df.b * df.c]
        )

    def test_query_index_with_name(self, engine: str, parser: str) -> None:
        df: pd.DataFrame = DataFrame(
            np.random.default_rng(2).integers(10, size=(10, 3)),
            index=Index(range(10), name='blob'),
            columns=['a', 'b', 'c']
        )
        res: pd.DataFrame = df.query('(blob < 5) & (a < b)', engine=engine, parser=parser)
        expec: pd.DataFrame = df[(df.index < 5) & (df.a < df.b)]
        tm.assert_frame_equal(res, expec)
        res = df.query('blob < b', engine=engine, parser=parser)
        expec = df[df.index < df.b]
        tm.assert_frame_equal(res, expec)

    def test_query_index_without_name(self, engine: str, parser: str) -> None:
        df: pd.DataFrame = DataFrame(
            np.random.default_rng(2).integers(10, size=(10, 3)),
            index=range(10),
            columns=['a', 'b', 'c']
        )
        res: pd.DataFrame = df.query('index < b', engine=engine, parser=parser)
        expec: pd.DataFrame = df[df.index < df.b]
        tm.assert_frame_equal(res, expec)
        res = df.query('index < 5', engine=engine, parser=parser)
        expec = df[df.index < 5]
        tm.assert_frame_equal(res, expec)

    def test_nested_scope(self, engine: str, parser: str) -> None:
        skip_if_no_pandas_parser(parser)
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df2: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        expected: pd.DataFrame = df[(df > 0) & (df2 > 0)]
        result: pd.DataFrame = df.query('(@df > 0) & (@df2 > 0)', engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)
        result = pd.eval('df[df > 0 and df2 > 0]', engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)
        result = pd.eval('df[df > 0 and df2 > 0 and df[df > 0] > 0]', engine=engine, parser=parser)
        expected = df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]
        tm.assert_frame_equal(result, expected)
        result = pd.eval('df[(df>0) & (df2>0)]', engine=engine, parser=parser)
        expected = df.query('(@df>0) & (@df2>0)', engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

    def test_nested_raises_on_local_self_reference(self, engine: str, parser: str) -> None:
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        with pytest.raises(UndefinedVariableError, match="name 'df' is not defined"):
            df.query('df > 0', engine=engine, parser=parser)

    def test_local_syntax(self, engine: str, parser: str) -> None:
        skip_if_no_pandas_parser(parser)
        df: pd.DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((100, 10)),
            columns=list('abcdefghij')
        )
        b: int = 1
        expect: pd.DataFrame = df[df.a < b]
        result: pd.DataFrame = df.query('a < @b', engine=engine, parser=parser)
        tm.assert_frame_equal(result, expect)
        expect = df[df.a < df.b]
        result = df.query('a < b', engine=engine, parser=parser)
        tm.assert_frame_equal(result, expect)

    def test_chained_cmp_and_in(self, engine: str, parser: str) -> None:
        skip_if_no_pandas_parser(parser)
        cols: List[str] = list('abc')
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((100, len(cols))), columns=cols)
        res: pd.DataFrame = df.query('a < b < c and a not in b not in c', engine=engine, parser=parser)
        ind: pd.Series = (df.a < df.b) & (df.b < df.c) & ~df.b.isin(df.a) & ~df.c.isin(df.b)
        expec: pd.DataFrame = df[ind]
        tm.assert_frame_equal(res, expec)

    def test_local_variable_with_in(self, engine: str, parser: str) -> None:
        skip_if_no_pandas_parser(parser)
        a: pd.Series = Series(np.random.default_rng(2).integers(3, size=15), name='a')
        b: pd.Series = Series(np.random.default_rng(2).integers(10, size=15), name='b')
        df: pd.DataFrame = DataFrame({'a': a, 'b': b})
        expected: pd.DataFrame = df.loc[(df.b - 1).isin(a)]
        result: pd.DataFrame = df.query('b - 1 in a', engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)
        b = Series(np.random.default_rng(2).integers(10, size=15), name='b')
        expected = df.loc[(b - 1).isin(a)]
        result = df.query('@b - 1 in a', engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

    def test_at_inside_string(self, engine: str, parser: str) -> None:
        skip_if_no_pandas_parser(parser)
        c: int = 1
        df: pd.DataFrame = DataFrame({'a': ['a', 'a', 'b', 'b', '@c', '@c']})
        result: pd.DataFrame = df.query('a == "@c"', engine=engine, parser=parser)
        expected: pd.DataFrame = df[df.a == '@c']
        tm.assert_frame_equal(result, expected)

    def test_query_undefined_local(self, engine: str, parser: str) -> None:
        skip_if_no_pandas_parser(parser)
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)), columns=list('ab'))
        with pytest.raises(UndefinedVariableError, match="local variable 'c' is not defined"):
            df.query('a == @c', engine=engine, parser=parser)

    def test_index_resolvers_come_after_columns_with_the_same_name(self, engine: str, parser: str) -> None:
        n: int = 1
        a: np.ndarray = np.r_[20:101:20]
        df: pd.DataFrame = DataFrame({'index': a, 'b': np.random.default_rng(2).standard_normal(a.size)})
        df.index.name = 'index'
        result: pd.DataFrame = df.query('index > 5', engine=engine, parser=parser)
        expected: pd.DataFrame = df[df['index'] > 5]
        tm.assert_frame_equal(result, expected)
        df = DataFrame({'index': a, 'b': np.random.default_rng(2).standard_normal(a.size)})
        result = df.query('ilevel_0 > 5', engine=engine, parser=parser)
        expected = df.loc[df.index[df.index > 5]]
        tm.assert_frame_equal(result, expected)
        df = DataFrame({'a': a, 'b': np.random.default_rng(2).standard_normal(a.size)})
        df.index.name = 'a'
        result = df.query('a > 5', engine=engine, parser=parser)
        expected = df[df.a > 5]
        tm.assert_frame_equal(result, expected)
        result = df.query('index > 5', engine=engine, parser=parser)
        expected = df.loc[df.index[df.index > 5]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('op, f', [['==', operator.eq], ['!=', operator.ne]])
    def test_inf(self, op: str, f: Callable[[Any, Any], bool], engine: str, parser: str) -> None:
        n: int = 10
        df: pd.DataFrame = DataFrame({'a': np.random.default_rng(2).random(n), 'b': np.random.default_rng(2).random(n)})
        df.loc[::2, 'a'] = np.inf
        q: str = f'a {op} inf'
        expected: pd.DataFrame = df[f(df.a, np.inf)]
        result: pd.DataFrame = df.query(q, engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

    def test_check_tz_aware_index_query(self, tz_aware_fixture: str) -> None:
        tz: str = tz_aware_fixture
        df_index: pd.DatetimeIndex = date_range(start='2019-01-01', freq='1D', periods=10, tz=tz, name='time')
        expected: pd.DataFrame = DataFrame(index=df_index)
        df: pd.DataFrame = DataFrame(index=df_index)
        result: pd.DataFrame = df.query('"2018-01-03 00:00:00+00" < time')
        tm.assert_frame_equal(result, expected)
        expected = DataFrame(df_index)
        expected.columns = expected.columns.astype(object)
        result = df.reset_index().query('"2018-01-03 00:00:00+00" < time')
        tm.assert_frame_equal(result, expected)

    def test_method_calls_in_query(self, engine: str, parser: str) -> None:
        n: int = 10
        df: pd.DataFrame = DataFrame({
            'a': 2 * np.random.default_rng(2).random(n),
            'b': np.random.default_rng(2).random(n)
        })
        expected: pd.Series = df[df.a.astype('int') == 0]
        result: pd.Series = df.query("a.astype('int') == 0", engine=engine, parser=parser)
        tm.assert_series_equal(result, expected)
        df = DataFrame({
            'a': np.where(np.random.default_rng(2).random(n) < 0.5, np.nan, np.random.default_rng(2).standard_normal(n)),
            'b': np.random.default_rng(2).standard_normal(n)
        })
        expected = df[df.a.notnull()]
        result = df.query('a.notnull()', engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

@td.skip_if_no('numexpr')
class TestDataFrameQueryNumExprPython(TestDataFrameQueryNumExprPandas):

    @pytest.fixture
    def engine(self) -> str:
        return 'numexpr'

    @pytest.fixture
    def parser(self) -> str:
        return 'python'

    def test_date_query_no_attribute_access(self, engine: str, parser: str) -> None:
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df['dates1'] = date_range('1/1/2012', periods=5)
        df['dates2'] = date_range('1/1/2013', periods=5)
        df['dates3'] = date_range('1/1/2014', periods=5)
        res: pd.DataFrame = df.query('(dates1 < 20130101) & (20130101 < dates3)', engine=engine, parser=parser)
        expec: pd.DataFrame = df[(df.dates1 < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_NaT(self, engine: str, parser: str) -> None:
        n: int = 10
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df['dates1'] = date_range('1/1/2012', periods=n)
        df['dates2'] = date_range('1/1/2013', periods=n)
        df['dates3'] = date_range('1/1/2014', periods=n)
        df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates1'] = pd.NaT
        df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates3'] = pd.NaT
        res: pd.DataFrame = df.query('(dates1 < 20130101) & (20130101 < dates3)', engine=engine, parser=parser)
        expec: pd.DataFrame = df[(df.dates1 < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query(self, engine: str, parser: str) -> None:
        n: int = 10
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df['dates1'] = date_range('1/1/2012', periods=n)
        df['dates3'] = date_range('1/1/2014', periods=n)
        return_value: Optional[bool] = df.set_index('dates1', inplace=True, drop=True)
        assert return_value is None
        res: pd.DataFrame = df.query('(index < 20130101) & (20130101 < dates3)', engine=engine, parser=parser)
        expec: pd.DataFrame = df[(df.index < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT(self, engine: str, parser: str) -> None:
        n: int = 10
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((n, 3))).astype({0: object})
        df['dates1'] = date_range('1/1/2012', periods=n)
        df['dates3'] = date_range('1/1/2014', periods=n)
        df.iloc[0, 0] = pd.NaT
        return_value: Optional[bool] = df.set_index('dates1', inplace=True, drop=True)
        assert return_value is None
        res: pd.DataFrame = df.query('(index < 20130101) & (20130101 < dates3)', engine=engine, parser=parser)
        expec: pd.DataFrame = df[(df.index < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT_duplicates(self, engine: str, parser: str) -> None:
        n: int = 10
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df['dates1'] = date_range('1/1/2012', periods=n)
        df['dates3'] = date_range('1/1/2014', periods=n)
        df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates1'] = pd.NaT
        return_value: Optional[bool] = df.set_index('dates1', inplace=True, drop=True)
        assert return_value is None
        msg: str = "'BoolOp' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            df.query('index < 20130101 < dates3', engine=engine, parser=parser)

    def test_nested_scope(self, engine: str, parser: str) -> None:
        x: int = 1
        result: int = pd.eval('x + 1', engine=engine, parser=parser)
        assert result == 2
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df2: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        msg: str = "The '@' prefix is only supported by the pandas parser"
        with pytest.raises(SyntaxError, match=msg):
            df.query('(@df>0) & (@df2>0)', engine=engine, parser=parser)
        with pytest.raises(UndefinedVariableError, match="name 'df' is not defined"):
            df.query('(df>0) & (df2>0)', engine=engine, parser=parser)
        expected: pd.DataFrame = df[(df > 0) & (df2 > 0)]
        result: pd.DataFrame = pd.eval('df[(df > 0) & (df2 > 0)]', engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)
        expected = df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]
        result = pd.eval('df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]', engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)
        expected = df.query('(@df>0) & (@df2>0)', engine=engine, parser=parser)
        result = pd.eval('df[(df > 0) & (df2 > 0)]', engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

    def test_query_numexpr_with_min_and_max_columns(self) -> None:
        df: pd.DataFrame = DataFrame({'min': [1, 2, 3], 'max': [4, 5, 6]})
        regex_to_match: str = 'Variables in expression \\"\\(min\\) == \\(1\\)\\" overlap with builtins: \\(\'min\'\\)'
        with pytest.raises(NumExprClobberingError, match=regex_to_match):
            df.query('min == 1')
        regex_to_match = 'Variables in expression \\"\\(max\\) == \\(1\\)\\" overlap with builtins: \\(\'max\'\\)'
        with pytest.raises(NumExprClobberingError, match=regex_to_match):
            df.query('max == 1')

class TestDataFrameQueryPythonPandas(TestDataFrameQueryNumExprPandas):

    @pytest.fixture
    def engine(self) -> str:
        return 'python'

    @pytest.fixture
    def parser(self) -> str:
        return 'pandas'

    def test_query_builtin(self, engine: str, parser: str) -> None:
        n: int = m = 10
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).integers(m, size=(n, 3)), columns=list('abc'))
        df.index.name = 'sin'
        expected: pd.DataFrame = df[df.index > 5]
        result: pd.DataFrame = df.query('sin > 5', engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

class TestDataFrameQueryPythonPython(TestDataFrameQueryNumExprPython):

    @pytest.fixture
    def engine(self) -> str:
        return 'python'

    @pytest.fixture
    def parser(self) -> str:
        return 'python'

    def test_query_builtin(self, engine: str, parser: str) -> None:
        n: int = m = 10
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).integers(m, size=(n, 3)), columns=list('abc'))
        df.index.name = 'sin'
        expected: pd.DataFrame = df[df.index > 5]
        result: pd.DataFrame = df.query('sin > 5', engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

class TestDataFrameQueryStrings:

    def test_str_query_method(self, parser: str, engine: str) -> None:
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 1)), columns=['b'])
        df['strings'] = Series(list('aabbccddee'))
        expect: pd.DataFrame = df[df.strings == 'a']
        if parser != 'pandas':
            col: str = 'strings'
            lst: str = '"a"'
            lhs: List[str] = [col] * 2 + [lst] * 2
            rhs: List[str] = lhs[::-1]
            eq, ne = ('==', '!=')
            ops: List[str] = 2 * ([eq] + [ne])
            msg: str = "'(Not)?In' nodes are not implemented"
            for lh, op_, rh in zip(lhs, ops, rhs):
                ex: str = f'{lh} {op_} {rh}'
                with pytest.raises(NotImplementedError, match=msg):
                    df.query(ex, engine=engine, parser=parser, local_dict={'strings': df.strings})
        else:
            res: pd.DataFrame = df.query('"a" == strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            res = df.query('strings == "a"', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            tm.assert_frame_equal(res, df[df.strings.isin(['a'])])
            expect = df[df.strings != 'a']
            res = df.query('strings != "a"', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            res = df.query('"a" != strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            tm.assert_frame_equal(res, df[~df.strings.isin(['a'])])

    def test_str_list_query_method(self, parser: str, engine: str) -> None:
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 1)), columns=['b'])
        df['strings'] = Series(list('aabbccddee'))
        expect: pd.DataFrame = df[df.strings.isin(['a', 'b'])]
        if parser != 'pandas':
            col: str = 'strings'
            lst: str = '["a", "b"]'
            lhs: List[str] = [col] * 2 + [lst] * 2
            rhs: List[str] = lhs[::-1]
            eq, ne = ('==', '!=')
            ops: List[str] = 2 * ([eq] + [ne])
            msg: str = "'(Not)?In' nodes are not implemented"
            for lh, ops_, rh in zip(lhs, ops, rhs):
                ex: str = f'{lh} {ops_} {rh}'
                with pytest.raises(NotImplementedError, match=msg):
                    df.query(ex, engine=engine, parser=parser)
        else:
            res: pd.DataFrame = df.query('strings == ["a", "b"]', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            res = df.query('["a", "b"] == strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            expect = df[~df.strings.isin(['a', 'b'])]
            res = df.query('strings != ["a", "b"]', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            res = df.query('["a", "b"] != strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

    def test_query_with_string_columns(self, parser: str, engine: str) -> None:
        df: pd.DataFrame = DataFrame({
            'a': list('aaaabbbbcccc'),
            'b': list('aabbccddeeff'),
            'c': np.random.default_rng(2).integers(5, size=12),
            'd': np.random.default_rng(2).integers(9, size=12)
        })
        if parser == 'pandas':
            res: pd.DataFrame = df.query('a in b', parser=parser, engine=engine)
            expec: pd.DataFrame = df[df.a.isin(df.b)]
            tm.assert_frame_equal(res, expec)
            res = df.query('a in b and c < d', parser=parser, engine=engine)
            expec = df[df.a.isin(df.b) & (df.c < df.d)]
            tm.assert_frame_equal(res, expec)
        else:
            msg: str = "'(Not)?In' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                df.query('a in b', parser=parser, engine=engine)
            msg = "'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                df.query('a in b and c < d', parser=parser, engine=engine)

    def test_object_array_eq_ne(self, parser: str, engine: str) -> None:
        df: pd.DataFrame = DataFrame({
            'a': list('aaaabbbbcccc'),
            'b': list('aabbccddeeff'),
            'c': np.random.default_rng(2).integers(5, size=12),
            'd': np.random.default_rng(2).integers(9, size=12)
        })
        res: pd.DataFrame = df.query('a == b', parser=parser, engine=engine)
        exp: pd.DataFrame = df[df.a == df.b]
        tm.assert_frame_equal(res, exp)
        res = df.query('a != b', parser=parser, engine=engine)
        exp = df[df.a != df.b]
        tm.assert_frame_equal(res, exp)

    def test_query_with_nested_strings(self, parser: str, engine: str) -> None:
        skip_if_no_pandas_parser(parser)
        events: List[str] = [f'page {n} {act}' for n in range(1, 4) for act in ['load', 'exit']] * 2
        stamps1: pd.DatetimeIndex = date_range('2014-01-01 0:00:01', freq='30s', periods=6)
        stamps2: pd.DatetimeIndex = date_range('2014-02-01 1:00:01', freq='30s', periods=6)
        df: pd.DataFrame = DataFrame({
            'id': np.arange(1, 7).repeat(2),
            'event': events,
            'timestamp': stamps1.append(stamps2)
        })
        expected: pd.DataFrame = df[df.event == '"page 1 load"']
        res: pd.DataFrame = df.query('\'"page 1 load"\' in event', parser=parser, engine=engine)
        tm.assert_frame_equal(expected, res)

    def test_query_with_nested_special_character(self, parser: str, engine: str) -> None:
        skip_if_no_pandas_parser(parser)
        df: pd.DataFrame = DataFrame({'a': ['a', 'b', 'test & test'], 'b': [1, 2, 3]})
        res: pd.DataFrame = df.query('a == "test & test"', parser=parser, engine=engine)
        expec: pd.DataFrame = df[df.a == 'test & test']
        tm.assert_frame_equal(res, expec)

    @pytest.mark.parametrize('op, func', [
        ('<', operator.lt),
        ('>', operator.gt),
        ('<=', operator.le),
        ('>=', operator.ge)
    ])
    def test_query_lex_compare_strings(self, parser: str, engine: str, op: str, func: Callable[[Any, Any], bool]) -> None:
        a: pd.Series = Series(np.random.default_rng(2).choice(list('abcde'), 20))
        b: pd.Series = Series(np.arange(a.size))
        df: pd.DataFrame = DataFrame({'X': a, 'Y': b})
        res: pd.DataFrame = df.query(f'X {op} "d"', engine=engine, parser=parser)
        expected: pd.DataFrame = df[func(df.X, 'd')]
        tm.assert_frame_equal(res, expected)

    def test_query_single_element_booleans(self, parser: str, engine: str) -> None:
        columns: tuple = ('bid', 'bidsize', 'ask', 'asksize')
        data: np.ndarray = np.random.default_rng(2).integers(2, size=(1, len(columns))).astype(bool)
        df: pd.DataFrame = DataFrame(data, columns=columns)
        res: pd.DataFrame = df.query('bid & ask', engine=engine, parser=parser)
        expected: pd.DataFrame = df[df.bid & df.ask]
        tm.assert_frame_equal(res, expected)

    def test_query_string_scalar_variable(self, parser: str, engine: str) -> None:
        skip_if_no_pandas_parser(parser)
        df: pd.DataFrame = DataFrame({
            'Symbol': ['BUD US', 'BUD US', 'IBM US', 'IBM US'],
            'Price': [109.7, 109.72, 183.3, 183.35]
        })
        e: pd.DataFrame = df[df.Symbol == 'BUD US']
        symb: str = 'BUD US'
        r: pd.DataFrame = df.query('Symbol == @symb', parser=parser, engine=engine)
        tm.assert_frame_equal(e, r)

    @pytest.mark.parametrize('in_list', [
        [None, 'asdf', 'ghjk'],
        ['asdf', None, 'ghjk'],
        ['asdf', 'ghjk', None],
        [None, None, 'asdf'],
        ['asdf', None, None],
        [None, None, None]
    ])
    def test_query_string_null_elements(self, in_list: List[Optional[str]]) -> None:
        parser: str = 'pandas'
        engine: str = 'python'
        expected: Dict[int, Optional[str]] = {i: value for i, value in enumerate(in_list) if value == 'asdf'}
        df_expected: pd.DataFrame = DataFrame({'a': expected}, dtype='string')
        df_expected.index = df_expected.index.astype('int64')
        df: pd.DataFrame = DataFrame({'a': in_list}, dtype='string')
        df.index = Index(list(df.index), dtype=df.index.dtype)
        res1: pd.DataFrame = df.query("a == 'asdf'", parser=parser, engine=engine)
        res2: pd.DataFrame = df[df['a'] == 'asdf']
        res3: pd.DataFrame = df.query("a <= 'asdf'", parser=parser, engine=engine)
        tm.assert_frame_equal(res1, df_expected)
        tm.assert_frame_equal(res1, res2)
        tm.assert_frame_equal(res1, res3)
        tm.assert_frame_equal(res2, res3)

    def test_query_with_string_columns_with_hashtag_character(self) -> None:
        df: pd.DataFrame = DataFrame((1, 2, 3), columns=['a#'])
        result: pd.DataFrame = df.query('`a#` < 2')
        expected: pd.DataFrame = df[df['a#'] < 2]
        tm.assert_frame_equal(result, expected)

    def test_expr_with_comment(self) -> None:
        df: pd.DataFrame = DataFrame((1, 2, 3), columns=['a#'])
        result: pd.DataFrame = df.query('`a#` < 2  # This is a comment')
        expected: pd.DataFrame = df[df['a#'] < 2]
        tm.assert_frame_equal(result, expected)

    def test_expr_with_column_name_with_backtick_and_hash(self) -> None:
        df: pd.DataFrame = DataFrame((1, 2, 3), columns=['a`#b'])
        result: pd.DataFrame = df.query('`a``#b` < 2')
        expected: pd.DataFrame = df[df['a`#b'] < 2]
        tm.assert_frame_equal(result, expected)

    def test_expr_with_column_name_with_backtick(self) -> None:
        df: pd.DataFrame = DataFrame({'a`b': (1, 2, 3), 'ab': (4, 5, 6)})
        result: pd.DataFrame = df.query('`a``b` < 2')
        expected: pd.DataFrame = df[df['a`b'] < 2]
        tm.assert_frame_equal(result, expected)

    def test_expr_with_string_with_backticks(self) -> None:
        df: pd.DataFrame = DataFrame(('`', '