def _new_PeriodIndex(cls, **d: dict) -> PeriodIndex:
    values: np.ndarray = d.pop('data')
    if values.dtype == 'int64':
        freq = d.pop('freq', None)
        dtype = PeriodDtype(freq)
        values = PeriodArray(values, dtype=dtype)
        return cls._simple_new(values, **d)
    else:
        return cls(values, **d)

def asfreq(self, freq=None, how='E') -> PeriodIndex:
    arr = self._data.asfreq(freq, how)
    return type(self)._simple_new(arr, name=self.name)

def to_timestamp(self, freq=None, how='start') -> DatetimeIndex:
    arr = self._data.to_timestamp(freq, how)
    return DatetimeIndex._simple_new(arr, name=self.name)

def from_fields(cls, *, year=None, quarter=None, month=None, day=None, hour=None, minute=None, second=None, freq=None) -> PeriodIndex:
    fields = {'year': year, 'quarter': quarter, 'month': month, 'day': day, 'hour': hour, 'minute': minute, 'second': second}
    fields = {key: value for key, value in fields.items() if value is not None}
    arr = PeriodArray._from_fields(fields=fields, freq=freq)
    return cls._simple_new(arr)

def from_ordinals(cls, ordinals: np.ndarray, freq, name=None) -> PeriodIndex:
    ordinals = np.asarray(ordinals, dtype=np.int64)
    dtype = PeriodDtype(freq)
    data = PeriodArray._simple_new(ordinals, dtype=dtype)
    return cls._simple_new(data, name=name)

def period_range(start=None, end=None, periods=None, freq=None, name=None) -> PeriodIndex:
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError('Of the three parameters: start, end, and periods, exactly two must be specified')
    if freq is None and (not isinstance(start, Period) and (not isinstance(end, Period))):
        freq = 'D'
    data, freq = PeriodArray._generate_range(start, end, periods, freq)
    dtype = PeriodDtype(freq)
    data = PeriodArray(data, dtype=dtype)
    return PeriodIndex(data, name=name)
