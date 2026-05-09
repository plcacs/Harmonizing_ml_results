from typing import Any, Callable, Dict, List, Optional, Tuple, Union

class TestLoc:
    def test_not_change_nan_loc(
        self,
        series: List[Union[float, str, bool]],
        new_series: List[Union[float, str, bool]],
        expected_ser: List[Union[float, str, bool]]
    ) -> None:
        """Test that setting a new series doesn't change NaN values."""
        df = DataFrame({'A': series})
        df.loc[:, 'A'] = new_series
        expected = DataFrame({'A': expected_ser})
        tm.assert_frame_equal(df.isna(), expected)
        tm.assert_frame_equal(df.notna(), ~expected)

class TestLocBaseIndependent:
    def test_loc_npstr(
        self,
        df: DataFrame
    ) -> None:
        """Test that np.array indexing works with string indices."""
        result = df.loc[np.array(['2021/6/1'])[0]:]
        expected = df.iloc[151:]
        tm.assert_frame_equal(result, expected)

class TestLocWithEllipsis:
    def test_loc_iloc_getitem_ellipsis(
        self,
        obj: Union[Series, DataFrame],
        indexer: Callable[[Union[Series, DataFrame]], Union[Series, DataFrame]]
    ) -> None:
        """Test that .loc and .iloc indexing with ellipsis returns the original object."""
        result = indexer(obj)[...]
        tm.assert_equal(result, obj)

class TestLocWithMultiIndex:
    def test_loc_getitem_multilevel_index_order(
        self,
        dim: str,
        keys: Tuple[Optional[Union[str, Tuple[str, str]]], ...],
        expected: Tuple[Union[str, Tuple[str, str]], ...],
        df: DataFrame
    ) -> None:
        """Test that .loc indexing with multi-index keys returns the correct order."""
        kwargs = {dim: [['c', 'a', 'a', 'b', 'b'], [1, 1, 2, 1, 2]]}
        df = DataFrame(np.arange(25).reshape(5, 5), **kwargs)
        exp_index = MultiIndex.from_arrays(expected)
        if dim == 'index':
            res = df.loc[keys, :]
            tm.assert_index_equal(res.index, exp_index)
        elif dim == 'columns':
            res = df.loc[:, keys]
            tm.assert_index_equal(res.columns, exp_index)

class TestLocSetitemWithExpansion:
    def test_loc_setitem_with_expansion_large_dataframe(
        self,
        monkeypatch: Any,
        result: DataFrame,
        expected: DataFrame
    ) -> None:
        """Test that .loc indexing with expansion works with large DataFrames."""
        size_cutoff = 50
        with monkeypatch.context():
            monkeypatch.setattr(libindex, '_SIZE_CUTOFF', size_cutoff)
            result = DataFrame({'x': range(size_cutoff)}, dtype='int64')
            result.loc[size_cutoff] = size_cutoff
        expected = DataFrame({'x': range(size_cutoff + 1)}, dtype='int64')
        tm.assert_frame_equal(result, expected)

class TestLocCallable:
    def test_frame_loc_getitem_callable(
        self,
        df: DataFrame
    ) -> None:
        """Test that .loc indexing with callable key works."""
        res = df.loc[lambda x: x.A > 2]
        tm.assert_frame_equal(res, df.loc[df.A > 2])

class TestLocBooleanLabelsAndSlices:
    def test_loc_bool_incompatible_index_raises(
        self,
        bool_value: bool,
        obj: Union[Series, DataFrame]
    ) -> None:
        """Test that .loc indexing with boolean labels raises an error."""
        message = f'{bool_value}: boolean label can not be used without a boolean index'
        if obj.index.inferred_type != 'boolean':
            obj = obj.astype('object')
            with pytest.raises(KeyError, match=message):
                obj.loc[bool_value]

class TestLocListlike:
    def test_loc_getitem_list_of_labels_categoricalindex_with_na(
        self,
        box: Callable[[Union[Series, Index]], Union[Series, Index]],
        ci: CategoricalIndex,
        ser: Series
    ) -> None:
        """Test that .loc indexing with list of labels and categorical index works."""
        result = ser.loc[box(ci)]
        tm.assert_series_equal(result, ser)
        result = ser[box(ci)]
        tm.assert_series_equal(result, ser)

class TestLocSeries:
    def test_loc_getitem(
        self,
        ser: Series
    ) -> None:
        """Test that .loc indexing on a Series works."""
        inds = ser.index[[3, 4, 7]]
        tm.assert_series_equal(ser.loc[inds], ser.reindex(inds))

def test_loc_getitem_label_list_integer_labels(
    columns: List[Union[int, str]],
    column_key: List[Union[int, str]],
    expected_columns: List[int],
    df: DataFrame
) -> None:
    """Test that .loc indexing with integer labels on a DataFrame works."""
    expected = df.iloc[:, expected_columns]
    result = df.loc[['A', 'B', 'C'], column_key]
    tm.assert_frame_equal(result, expected, check_column_type=True)

def test_loc_setitem_float_intindex(
    rand_data: np.ndarray,
    result: DataFrame,
    expected: DataFrame
) -> None:
    """Test that .loc indexing with float index on a DataFrame works."""
    result = DataFrame(rand_data)
    result.loc[:, 0.5] = np.nan
    expected_data = np.hstack((rand_data, np.array([np.nan] * 8).reshape(8, 1)))
    expected = DataFrame(expected_data, columns=[0.0, 1.0, 2.0, 3.0, 0.5])
    tm.assert_frame_equal(result, expected)

def test_loc_axis_1_slice(
    cols: List[Tuple[int, int]],
    df: DataFrame,
    result: DataFrame,
    expected: DataFrame
) -> None:
    """Test that .loc indexing with axis=1 and slice on a DataFrame works."""
    result = df.loc(axis=1)[(2014, 9):(2015, 8)]
    expected = DataFrame(np.ones((10, 4)), index=tuple('ABCDEFGHIJ'), columns=MultiIndex.from_tuples([(2014, 9), (2014, 10), (2015, 7), (2015, 8)]))
    tm.assert_frame_equal(result, expected)

def test_loc_set_dataframe_multiindex(
    expected: DataFrame,
    result: DataFrame
) -> None:
    """Test that .loc indexing with multi-index on a DataFrame works."""
    result = expected.copy()
    result.loc[0, [(0, 1)]] = result.loc[0, [(0, 1)]]
    tm.assert_frame_equal(result, expected)

def test_loc_mixed_int_float(
    ser: Series
) -> None:
    """Test that .loc indexing with mixed int and float index on a Series works."""
    result = ser.loc[1]
    assert result == 0

def test_loc_with_positional_slice_raises(
    ser: Series
) -> None:
    """Test that .loc indexing with positional slice raises an error."""
    with pytest.raises(TypeError, match='Slicing a positional slice with .loc'):
        ser.loc[:3] = 2

def test_loc_slice_disallows_positional(
    dti: DatetimeIndex,
    df: DataFrame,
    ser: Series
) -> None:
    """Test that .loc indexing with positional slice disallows position."""
    msg = 'cannot do slice indexing on DatetimeIndex with these indexers \\[1\\] of type int'
    for obj in [df, ser]:
        with pytest.raises(TypeError, match=msg):
            obj.loc[1:3]
        with pytest.raises(TypeError, match='Slicing a positional slice with .loc'):
            obj.loc[1:3] = 1
    with pytest.raises(TypeError, match=msg):
        df.loc[1:3, 1]
    with pytest.raises(TypeError, match='Slicing a positional slice with .loc'):
        df.loc[1:3, 1] = 2

def test_loc_datetimelike_mismatched_dtypes(
    df: DataFrame
) -> None:
    """Test that .loc indexing with datetimelike mismatched dtypes raises an error."""
    msg = 'None of \\[TimedeltaIndex.* are in the \\[index\\]'
    with pytest.raises(KeyError, match=msg):
        df.loc[pd.TimedeltaIndex(dti.asi8)]

def test_loc_with_period_index_indexer(
    idx: PeriodIndex,
    df: DataFrame
) -> None:
    """Test that .loc indexing with period index works."""
    tm.assert_frame_equal(df, df.loc[idx])
    tm.assert_frame_equal(df, df.loc[list(idx)])
    tm.assert_frame_equal(df, df.loc[list(idx)])
    tm.assert_frame_equal(df.iloc[0:5], df.loc[idx[0:5]])
    tm.assert_frame_equal(df, df.loc[list(idx)])

def test_loc_setitem_multiindex_timestamp(
    vals: np.ndarray,
    idx: DatetimeIndex,
    cols: List[str],
    exp: DataFrame,
    res: DataFrame
) -> None:
    """Test that .loc indexing with multi-index and timestamp works."""
    res = DataFrame(vals, index=idx, columns=cols)
    tm.assert_frame_equal(res, exp)

def test_loc_getitem_multiindex_tuple_level(
    lev1: List[str],
    lev2: List[Tuple[int, int]],
    lev3: List[int],
    cols: MultiIndex,
    df: DataFrame,
    result: DataFrame,
    expected: DataFrame,
    alt: DataFrame,
    ser: Series,
    expected2: Series,
    alt2: Series,
    result2: Series
) -> None:
    """Test that .loc indexing with multi-index tuple level works."""
    result = df.loc[:, (lev1[0], lev2[0], lev3[0])]
    expected = df.iloc[:, :1]
    tm.assert_frame_equal(result, expected)
    alt = df.xs((lev1[0], lev2[0], lev3[0]), level=[0, 1, 2], axis=1)
    tm.assert_frame_equal(alt, expected)
    ser = df.iloc[0]
    expected2 = ser.iloc[:1]
    alt2 = ser.xs((lev1[0], lev2[0], lev3[0]), level=[0, 1, 2], axis=0)
    tm.assert_series_equal(alt2, expected2)
    result2 = ser.loc[lev1[0], lev2[0], lev3[0]]
    assert result2 == 6

def test_loc_getitem_nullable_index_with_duplicates(
    df: DataFrame,
    df2: DataFrame
) -> None:
    """Test that .loc indexing with nullable index and duplicates works."""
    res = df2.loc[1]
    expected = Series([1, 5], index=df2.columns, dtype='Int64', name=1)
    tm.assert_series_equal(res, expected)
    df2.index = df2.index.astype(object)
    res = df2.loc[1]
    tm.assert_series_equal(res, expected)

def test_loc_setitem_uint8_upcast(
    value: Union[int, np.uint16, np.int16],
    df: DataFrame
) -> None:
    """Test that .loc indexing with uint8 upcast raises an error."""
    with pytest.raises(TypeError, match='Invalid value'):
        df.loc[2, 'col1'] = value

def test_loc_set_int_dtype(
    df: DataFrame
) -> None:
    """Test that .loc indexing with int dtype works."""
    df.loc[:, 'col1'] = 5
    expected = DataFrame({0: ['a'], 1: ['b'], 2: ['c'], 'col1': [5]})
    tm.assert_frame_equal(df, expected)

def test_loc_periodindex_3_levels(
    p_index: PeriodIndex,
    mi_series: DataFrame
) -> None:
    """Test that .loc indexing with period index and 3 levels works."""
    assert mi_series.loc[p_index[0], 'A', 'B'] == 1.0

def test_loc_setitem_pyarrow_strings(
    df: DataFrame
) -> None:
    """Test that .loc indexing with pyarrow strings works."""
    df.loc[df.ids, 'strings'] = Series(['X', 'Y'])
    expected_df = DataFrame({'strings': Series(['X', 'Y', 'C'], dtype='string[pyarrow]'), 'ids': Series([True, True, False])})
    tm.assert_frame_equal(df, expected_df)

def test_loc_series(
    ser: Series
) -> None:
    """Test that .loc indexing on a Series works."""
    inds = ser.index[[3, 4, 7]]
    tm.assert_series_equal(ser.loc[inds], ser.reindex(inds))

def test_loc_setitem(
    ser: Series
) -> None:
    """Test that .loc indexing with setitem on a Series works."""
    inds = ser.index[[3, 4, 7]]
    result = ser.copy()
    result.loc[inds] = 5
    expected = ser.copy()
    expected.iloc[[3, 4, 7]] = 5
    tm.assert_series_equal(result, expected)
