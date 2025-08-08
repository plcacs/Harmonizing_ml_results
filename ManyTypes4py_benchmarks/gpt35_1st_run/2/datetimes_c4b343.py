    def __new__(cls, data: Optional[Union[Series, Index]] = None, freq: Any = _NoValue, normalize: bool = False, closed: Optional[str] = None, ambiguous: Union[str, bool, Any] = 'raise', dayfirst: bool = False, yearfirst: bool = False, dtype: Optional[Union[str, Any]] = None, copy: bool = False, name: Optional[Any] = None) -> Index:
    def __getattr__(self, item: Any) -> Any:
    @property
    def year(self) -> Index:
    @property
    def month(self) -> Index:
    @property
    def day(self) -> Index:
    @property
    def hour(self) -> Index:
    @property
    def minute(self) -> Index:
    @property
    def second(self) -> Index:
    @property
    def microsecond(self) -> Index:
    @property
    def week(self) -> Index:
    @property
    def weekofyear(self) -> Index:
    @property
    def dayofweek(self) -> Index:
    @property
    def day_of_week(self) -> Index:
    @property
    def weekday(self) -> Index:
    @property
    def dayofyear(self) -> Index:
    @property
    def day_of_year(self) -> Index:
    @property
    def quarter(self) -> Index:
    @property
    def is_month_start(self) -> Index:
    @property
    def is_month_end(self) -> Index:
    @property
    def is_quarter_start(self) -> Index:
    @property
    def is_quarter_end(self) -> Index:
    @property
    def is_year_start(self) -> Index:
    @property
    def is_year_end(self) -> Index:
    @property
    def is_leap_year(self) -> Index:
    @property
    def daysinmonth(self) -> Index:
    @property
    def days_in_month(self) -> Index:
    def ceil(self, freq: Any, *args: Any, **kwargs: Any) -> DatetimeIndex:
    def floor(self, freq: Any, *args: Any, **kwargs: Any) -> DatetimeIndex:
    def round(self, freq: Any, *args: Any, **kwargs: Any) -> DatetimeIndex:
    def month_name(self, locale: Optional[str] = None) -> Index:
    def day_name(self, locale: Optional[str] = None) -> Index:
    def normalize(self) -> DatetimeIndex:
    def strftime(self, date_format: str) -> Index:
    def indexer_between_time(self, start_time: Any, end_time: Any, include_start: bool = True, include_end: bool = True) -> Index:
    def indexer_at_time(self, time: Any, asof: bool = False) -> Index:
