    def test_drop_duplicates_metadata(self, idx: PeriodIndex) -> None:
    def test_drop_duplicates(self, keep: str, expected: np.ndarray, index: np.ndarray, idx: PeriodIndex) -> None:
    def freq(self, request: pytest.FixtureRequest) -> str:
    def idx(self, freq: str) -> PeriodIndex:
    def idx(self, freq_sample: str) -> DatetimeIndex:
    def idx(self, freq_sample: str) -> TimedeltaIndex:
