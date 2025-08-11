import datetime
import re
import pytest
from mimesis import Datetime
from mimesis.datasets import GMT_OFFSETS, TIMEZONES
from mimesis.enums import DurationUnit, TimestampFormat, TimezoneRegion
from mimesis.exceptions import NonEnumerableError
from . import patterns

class TestDatetime:

    @pytest.fixture
    def _datetime(self) -> Datetime:
        return Datetime()

    def test_str(self, dt: Union[str, datetime.date]) -> None:
        assert re.match(patterns.DATA_PROVIDER_STR_REGEX, str(dt))

    @pytest.mark.parametrize('days, objects_count', [(7, 169), (14, 337), (28, 673)])
    def test_bulk_create_datetimes(self, _datetime, days, objects_count) -> None:
        date_start = datetime.datetime.now()
        date_end = date_start + datetime.timedelta(days=days)
        datetime_objects = _datetime.bulk_create_datetimes(date_start=date_start, date_end=date_end, minutes=60)
        assert len(datetime_objects) == objects_count

    def test_bulk_create_datetimes_error(self, _datetime: Union[datetime.datetime, None]) -> None:
        date_start = datetime.datetime.now()
        date_end = date_start - datetime.timedelta(days=7)
        with pytest.raises(ValueError):
            _datetime.bulk_create_datetimes(date_start, date_end)
        with pytest.raises(ValueError):
            _datetime.bulk_create_datetimes(None, None)
        with pytest.raises(ValueError):
            _datetime.bulk_create_datetimes(date_start, date_start + datetime.timedelta(days=1))

    def test_year(self, _datetime) -> None:
        result = _datetime.year(minimum=2000, maximum=_datetime._CURRENT_YEAR)
        assert result >= 2000
        assert result <= _datetime._CURRENT_YEAR

    def test_gmt_offset(self, _datetime) -> None:
        result = _datetime.gmt_offset()
        assert result in GMT_OFFSETS

    def test_day_of_month(self, _datetime) -> None:
        result = _datetime.day_of_month()
        assert 1 <= result <= 31

    def test_date(self, dt) -> None:
        date_object = dt.date(start=dt._CURRENT_YEAR, end=dt._CURRENT_YEAR)
        assert isinstance(date_object, datetime.date)
        assert date_object.year == dt._CURRENT_YEAR

    def test_formatted_date(self, _datetime) -> None:
        fmt_date = _datetime.formatted_date('%Y', start=2000, end=2000)
        assert fmt_date == '2000'
        assert isinstance(fmt_date, str)

    def test_formatted_datetime(self, dt) -> None:
        year, hour = dt.formatted_datetime('%Y %H', start=2000, end=2000).split(' ')
        assert int(year.strip()) == 2000
        assert int(hour.strip()) <= 23
        assert isinstance(year, str)
        result = dt.formatted_datetime(fmt='')
        assert result

    def test_time(self, dt) -> None:
        default = dt.time()
        assert isinstance(default, datetime.time)

    def test_formatted_time(self, dt) -> None:
        default = dt.formatted_time()
        assert isinstance(default, str)

    def test_century(self, _datetime) -> None:
        result = _datetime.century()
        assert result is not None
        assert isinstance(result, str)

    def test_day_of_week(self, dt) -> None:
        result = dt.day_of_week()
        assert result in dt._dataset['day']['name']
        result_abbr = dt.day_of_week(abbr=True)
        assert result_abbr in dt._dataset['day']['abbr']

    def test_month(self, dt) -> None:
        result = dt.month()
        assert result is not None
        result_abbr = dt.month(abbr=True)
        assert isinstance(result_abbr, str)

    def test_periodicity(self, dt) -> None:
        result = dt.periodicity()
        assert result in dt._dataset['periodicity']

    @pytest.mark.parametrize('region', [TimezoneRegion.AFRICA, TimezoneRegion.AMERICA, TimezoneRegion.ANTARCTICA, TimezoneRegion.ARCTIC, TimezoneRegion.ASIA, TimezoneRegion.ATLANTIC, TimezoneRegion.AUSTRALIA, TimezoneRegion.EUROPE, TimezoneRegion.INDIAN, TimezoneRegion.PACIFIC])
    def test_timezone(self, _datetime, region) -> None:
        result = _datetime.timezone(region=region)
        assert result in TIMEZONES
        assert result.startswith(region.value)

    def test_timezone_without_region(self, _datetime: Union[datetime.datetime, None, int]) -> None:
        result = _datetime.timezone()
        region = result.split('/')[0]
        assert region in set([tz.split('/')[0] for tz in TIMEZONES])

    @pytest.mark.parametrize('fmt, out_type, kwargs', [(TimestampFormat.POSIX, int, {}), (TimestampFormat.RFC_3339, str, {'start': 2023, 'end': 2023}), (TimestampFormat.ISO_8601, str, {})])
    def test_timestamp(self, _datetime, fmt: Union[datetime.date, datetime.datetime.datetime, None, str], out_type, kwargs) -> None:
        result = _datetime.timestamp(fmt=fmt, **kwargs)
        assert result is not None
        assert isinstance(result, out_type)
        start = kwargs.get('start')
        end = kwargs.get('end')
        if start and end:
            year = int(result[:4])
            assert start <= year <= end
        with pytest.raises(NonEnumerableError):
            _datetime.timestamp(fmt='Blabla')

    @pytest.mark.parametrize('start, end, timezone', [(2014, 2019, 'Europe/Paris'), (2014, 2019, None)])
    def test_datetime(self, _datetime: Union[datetime.datetime.datetime, int], start: Union[datetime.datetime.datetime, int, datetime.time], end: Union[datetime.datetime.datetime, int, datetime.time], timezone: Union[datetime.datetime.datetime, int]) -> None:
        dt_obj = _datetime.datetime(start=start, end=end, timezone=timezone)
        assert start <= dt_obj.year <= end
        assert isinstance(dt_obj, datetime.datetime)
        if timezone:
            assert dt_obj.tzinfo is not None
        else:
            assert dt_obj.tzinfo is None

    @pytest.mark.parametrize('start, end', [(2018, 2018), (2019, 2019)])
    def test_formatted_datetime(self, _datetime, start, end) -> None:
        dt_str = _datetime.formatted_date(fmt='%Y', start=start, end=end)
        assert isinstance(dt_str, str)
        assert start <= int(dt_str) <= end
        dt_without_fmt = _datetime.formatted_date(fmt=None, start=start, end=end)
        assert dt_without_fmt

    @pytest.mark.parametrize('fmt', [None])
    def test_formatted_datetime_without_fmt(self, dt: Union[str, datetime.date, datetime.datetime.datetime], fmt: Union[str, datetime.date, datetime.datetime.datetime]) -> None:
        dt_str = dt.formatted_date(fmt=fmt, start=2010, end=2030)
        assert isinstance(dt_str, str)

    def test_week_date(self, _datetime) -> None:
        result = _datetime.week_date(start=2017, end=_datetime._CURRENT_YEAR)
        result = result.replace('-', ' ').replace('W', '')
        year, week = result.split(' ')
        assert int(year) >= 2017 and int(year) <= _datetime._CURRENT_YEAR
        assert int(week) <= 52

    @pytest.mark.parametrize('min_duration, max_duration, duration_unit', [(1, 10, DurationUnit.WEEKS), (1, 10, DurationUnit.DAYS), (1, 10, DurationUnit.HOURS), (1, 10, DurationUnit.MINUTES), (1, 10, DurationUnit.SECONDS), (1, 10, DurationUnit.MILLISECONDS), (1, 10, DurationUnit.MICROSECONDS), (1, 10, None)])
    def test_duration(self, _datetime: Union[int, float, datetime.date], min_duration: Union[int, float, datetime.date], max_duration: Union[int, float, datetime.date], duration_unit: Union[int, float, datetime.date]) -> None:
        result = _datetime.duration(min_duration=min_duration, max_duration=max_duration, duration_unit=duration_unit)
        assert isinstance(result, datetime.timedelta)

    def test_duration_error(self, _datetime: Union[datetime.datetime, datetime.date.time.date.time]) -> None:
        with pytest.raises(ValueError):
            _datetime.duration(min_duration=10, max_duration=1, duration_unit=DurationUnit.WEEKS)
        with pytest.raises(NonEnumerableError):
            _datetime.duration(min_duration=10, max_duration=10, duration_unit='Blabla')
        with pytest.raises(TypeError):
            _datetime.duration(min_duration=10.5, max_duration=10.9, duration_unit=DurationUnit.WEEKS)

class TestSeededDatetime:

    @pytest.fixture
    def d1(self, seed: Union[int, None, str]) -> Datetime:
        return Datetime(seed=seed)

    @pytest.fixture
    def d2(self, seed: Union[int, None, str]) -> Datetime:
        return Datetime(seed=seed)

    def test_year(self, d1: Union[datetime.datetime.datetime, int, datetime.datetime.date], d2: Union[datetime.datetime.datetime, int, datetime.datetime.date]) -> None:
        assert d1.year() == d2.year()
        assert d1.year(1942, 2048) == d2.year(1942, 2048)

    def test_gmt_offset(self, d1: Union[int, None, float], d2: Union[int, None, float]) -> None:
        assert d1.gmt_offset() == d2.gmt_offset()

    def test_day_of_month(self, d1: Union[int, datetime.datetime.datetime, datetime.datetime.date], d2: Union[int, datetime.datetime.datetime, datetime.datetime.date]) -> None:
        assert d1.day_of_month() == d2.day_of_month()

    def test_date(self, d1: Union[datetime.date, datetime.datetime], d2: Union[datetime.date, datetime.datetime]) -> None:
        assert d1.date() == d2.date()
        assert d1.date(start=1024, end=2048) == d2.date(start=1024, end=2048)

    def test_formatted_date(self, d1: Union[datetime.date, datetime.datetime, int], d2: Union[datetime.date, datetime.datetime, int]) -> None:
        assert d1.formatted_date() == d2.formatted_date()
        assert d1.formatted_date(start=1024, end=2048) == d2.formatted_date(start=1024, end=2048)

    def test_time(self, d1: Union[float, int, datetime.datetime.datetime], d2: Union[float, int, datetime.datetime.datetime]) -> None:
        assert d1.time() == d2.time()

    def test_formatted_time(self, d1: Union[datetime.datetime, datetime.date, int], d2: Union[datetime.datetime, datetime.date, int]) -> None:
        assert d1.formatted_time() == d2.formatted_time()

    def test_century(self, d1: Union[dict, str, float], d2: Union[dict, str, float]) -> None:
        assert d1.century() == d2.century()

    def test_day_of_week(self, d1: Union[float, datetime.datetime, datetime.datetime.timedelta, None], d2: Union[float, datetime.datetime, datetime.datetime.timedelta, None]) -> None:
        assert d1.day_of_week() == d2.day_of_week()
        assert d1.day_of_week(abbr=True) == d2.day_of_week(abbr=True)

    def test_month(self, d1: Union[datetime.date.time.date, datetime.datetime, float], d2: Union[datetime.date.time.date, datetime.datetime, float]) -> None:
        assert d1.month() == d2.month()
        assert d1.month(abbr=True) == d2.month(abbr=True)

    def test_periodicity(self, d1: Union[dict[str, typing.Any], str, dict], d2: Union[dict[str, typing.Any], str, dict]) -> None:
        assert d1.periodicity() == d2.periodicity()

    def test_timezone(self, d1: Union[datetime.date, datetime.datetime], d2: Union[datetime.date, datetime.datetime]) -> None:
        assert d1.timezone() == d2.timezone()
        assert d1.timezone(region=None) == d2.timezone(region=None)
        assert d1.timezone(region=TimezoneRegion.EUROPE) == d2.timezone(region=TimezoneRegion.EUROPE)

    @pytest.mark.parametrize('fmt', TimestampFormat)
    def test_timestamp(self, d1: Union[datetime.date, datetime.datetime.datetime, None, str], d2: Union[datetime.date, datetime.datetime.datetime, None, str], fmt: Union[datetime.date, datetime.datetime.datetime, None, str]) -> None:
        assert d1.timestamp(fmt) == d2.timestamp(fmt)

    def test_formatted_datetime(self, d1: Union[datetime.datetime, datetime.date.time.date, dict], d2: Union[datetime.datetime, datetime.date.time.date, dict]) -> None:
        assert d1.formatted_datetime() == d2.formatted_datetime()

    def test_week_date(self, d1: Union[datetime.datetime, datetime.date, int], d2: Union[datetime.datetime, datetime.date, int]) -> None:
        assert d1.week_date() == d2.week_date()
        assert d1.week_date(start=2007, end=2018) == d2.week_date(start=2007, end=2018)

    def test_bulk_create_datetimes(self, d1: Union[datetime.datetime, datetime.date, int], d2: Union[datetime.datetime, datetime.date, int]) -> None:
        date_start = datetime.datetime.now()
        date_end = date_start + datetime.timedelta(days=7)
        assert d1.bulk_create_datetimes(date_start, date_end, minutes=10) == d2.bulk_create_datetimes(date_start, date_end, minutes=10)

    def test_duratioh(self, d1: Union[dict, datetime, typing.Callable, None], d2: Union[dict, datetime, typing.Callable, None]) -> None:
        assert d1.duration() == d2.duration()
        assert d1.duration(10, 20, DurationUnit.WEEKS) == d2.duration(10, 20, DurationUnit.WEEKS)