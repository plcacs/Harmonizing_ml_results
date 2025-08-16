from datetime import datetime
from typing import List

class TestDatetimeIndexOps:

    def test_dti_no_millisecond_field(self) -> None:

    def test_dti_time(self) -> None:

    def test_dti_date(self) -> None:

    def test_dti_date2(self, dtype: str) -> None:

    def test_dti_time2(self, dtype: str) -> None:

    def test_dti_timetz(self, tz_naive_fixture: str) -> None:

    def test_dti_timestamp_fields(self, field: str) -> None:

    def test_dti_nanosecond(self) -> None:

    def test_dti_hour_tzaware(self, prefix: str) -> None:

    def test_day_name_month_name(self, time_locale: str) -> None:

    def test_dti_week(self) -> None:

    def test_dti_fields(self, tz: str) -> None:

    def test_dti_is_year_quarter_start(self) -> None:

    def test_dti_is_month_start(self) -> None:

    def test_dti_is_month_start_custom(self) -> None:

    def test_dti_dr_is_year_start(self, timestamp: str, freq: str, periods: int, expected_values: List[bool]) -> None:

    def test_dti_dr_is_year_end(self, timestamp: str, freq: str, periods: int, expected_values: List[bool]) -> None:

    def test_dti_dr_is_quarter_start(self, timestamp: str, freq: str, periods: int, expected_values: List[bool]) -> None:

    def test_dti_dr_is_quarter_end(self, timestamp: str, freq: str, periods: int, expected_values: List[bool]) -> None:

    def test_dti_dr_is_month_start(self, timestamp: str, freq: str, periods: int, expected_values: List[bool]) -> None:

    def test_dti_dr_is_month_end(self, timestamp: str, freq: str, periods: int, expected_values: List[bool]) -> None:

    def test_dti_is_year_quarter_start_doubledigit_freq(self) -> None:

    def test_dti_is_year_start_freq_custom_business_day_with_digit(self) -> None:

    def test_dti_is_year_quarter_start_freq_business_month_begin(self, freq: str) -> None:

