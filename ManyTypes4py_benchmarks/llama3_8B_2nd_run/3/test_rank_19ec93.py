class TestRank:
    s: Series
    df: DataFrame
    results: dict

    def test_rank(self, float_frame: DataFrame) -> None:
        # ...

    def test_rank2(self) -> None:
        # ...

    def test_rank_does_not_mutate(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), dtype='float64')
        expected: DataFrame = df.copy()
        df.rank()
        result: DataFrame = df
        tm.assert_frame_equal(result, expected)

    def test_rank_mixed_frame(self, float_string_frame: DataFrame) -> None:
        # ...

    def test_rank_na_option(self, float_frame: DataFrame) -> None:
        # ...

    def test_rank_axis(self) -> None:
        df: DataFrame = DataFrame([[2, 1], [4, 3]])
        tm.assert_frame_equal(df.rank(axis=0), df.rank(axis='index'))
        tm.assert_frame_equal(df.rank(axis=1), df.rank(axis='columns'))

    @pytest.mark.parametrize('ax', [0, 1])
    def test_rank_methods_frame(self, ax: int, rank_method: str) -> None:
        # ...

    @pytest.mark.parametrize('dtype', ['O', 'f8', 'i8'])
    def test_rank_descending(self, rank_method: str, dtype: str) -> None:
        # ...

    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.parametrize('dtype', [None, object])
    def test_rank_2d_tie_methods(self, rank_method: str, axis: int, dtype: str) -> None:
        # ...

    @pytest.mark.parametrize('rank_method,exp', [('dense', [[1.0, 1.0, 1.0], [1.0, 0.5, 2.0 / 3], [1.0, 0.5, 1.0 / 3]]), # ...
