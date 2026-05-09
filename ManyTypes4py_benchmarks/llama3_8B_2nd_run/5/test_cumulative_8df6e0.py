class TestSeriesCumulativeOps:
    @pytest.mark.parametrize('func', [np.cumsum, np.cumprod])
    def test_datetime_series(self, datetime_series: pd.Series, func: np.ufunc) -> None:
        tm.assert_numpy_array_equal(func(datetime_series).values, func(np.array(datetime_series)), check_dtype=True)
        ts = datetime_series.copy()
        ts[::2] = np.nan
        result = func(ts)[1::2]
        expected = func(np.array(ts.dropna()))
        tm.assert_numpy_array_equal(result.values, expected, check_dtype=False)

    @pytest.mark.parametrize('method', ['cummin', 'cummax'])
    def test_cummin_cummax(self, datetime_series: pd.Series, method: str) -> None:
        ufunc = methods[method]
        result = getattr(datetime_series, method)().values
        expected = ufunc(np.array(datetime_series))
        tm.assert_numpy_array_equal(result, expected)
        ts = datetime_series.copy()
        ts[::2] = np.nan
        result = getattr(ts, method)()[1::2]
        expected = ufunc(ts.dropna())
        result.index = result.index._with_freq(None)
        tm.assert_series_equal(result, expected)

    # ... (rest of the code remains the same)
