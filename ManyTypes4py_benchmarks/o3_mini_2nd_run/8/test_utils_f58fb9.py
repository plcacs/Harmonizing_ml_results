import os
import re
from unittest import mock
import sys
import click
import pytest
from six import StringIO
from hypothesis.strategies import text
from hypothesis import given
import string
from dateutil import tz
from datetime import datetime
from chalice import utils
from typing import Any

class TestUI:
    def setup_method(self) -> None:
        self.out: StringIO = StringIO()
        self.err: StringIO = StringIO()
        self.ui: utils.UI = utils.UI(self.out, self.err)

    def test_write_goes_to_out_obj(self) -> None:
        self.ui.write('Foo')
        assert self.out.getvalue() == 'Foo'
        assert self.err.getvalue() == ''

    def test_error_goes_to_err_obj(self) -> None:
        self.ui.error('Foo')
        assert self.err.getvalue() == 'Foo'
        assert self.out.getvalue() == ''

    def test_confirm_raises_own_exception(self) -> None:
        confirm: Any = mock.Mock(spec=click.confirm)
        confirm.side_effect = click.Abort()
        ui: utils.UI = utils.UI(self.out, self.err, confirm)
        with pytest.raises(utils.AbortedError):
            ui.confirm('Confirm?')

    def test_confirm_returns_value(self) -> None:
        confirm: Any = mock.Mock(spec=click.confirm)
        confirm.return_value = 'foo'
        ui: utils.UI = utils.UI(self.out, self.err, confirm)
        return_value = ui.confirm('Confirm?')
        assert return_value == 'foo'

class TestChaliceZip:
    def test_chalice_zip_file(self, tmpdir: Any) -> None:
        tmpdir.mkdir('foo').join('app.py').write('# Test app')
        zip_path = tmpdir.join('app.zip')
        app_filename: str = str(tmpdir.join('foo', 'app.py'))
        script_obj = tmpdir.join('foo', 'myscript.sh')
        script_obj.write('echo foo')
        script_file: str = str(script_obj)
        os.chmod(script_file, 0o755)
        with utils.ChaliceZipFile(str(zip_path), 'w') as z:
            z.write(app_filename)
            z.write(script_file)
        with utils.ChaliceZipFile(str(zip_path)) as z:
            assert len(z.infolist()) == 2
            app = z.getinfo(app_filename[1:])
            assert app.date_time == (1980, 1, 1, 0, 0, 0)
            assert app.external_attr >> 16 == os.stat(app_filename).st_mode
            script = z.getinfo(script_file[1:])
            assert script.date_time == (1980, 1, 1, 0, 0, 0)
            assert script.external_attr >> 16 == os.stat(script_file).st_mode

class TestPipeReader:
    def test_pipe_reader_does_read_pipe(self) -> None:
        mock_stream: Any = mock.Mock(spec=sys.stdin)
        mock_stream.isatty.return_value = False
        mock_stream.read.return_value = 'foobar'
        reader: utils.PipeReader = utils.PipeReader(mock_stream)
        value: Any = reader.read()
        assert value == 'foobar'

    def test_pipe_reader_does_not_read_tty(self) -> None:
        mock_stream: Any = mock.Mock(spec=sys.stdin)
        mock_stream.isatty.return_value = True
        mock_stream.read.return_value = 'foobar'
        reader: utils.PipeReader = utils.PipeReader(mock_stream)
        value: Any = reader.read()
        assert value is None

def test_serialize_json() -> None:
    assert utils.serialize_to_json({'foo': 'bar'}) == '{\n  "foo": "bar"\n}\n'

@pytest.mark.parametrize(
    'name,cfn_name',
    [
        ('f', 'F'),
        ('foo', 'Foo'),
        ('foo_bar', 'FooBar'),
        ('foo_bar_baz', 'FooBarBaz'),
        ('F', 'F'),
        ('FooBar', 'FooBar'),
        ('S3Bucket', 'S3Bucket'),
        ('s3Bucket', 'S3Bucket'),
        ('123', '123'),
        ('foo-bar-baz', 'FooBarBaz'),
        ('foo_bar-baz', 'FooBarBaz'),
        ('foo-bar_baz', 'FooBarBaz'),
        ('foo_bar!?', 'FooBar'),
        ('_foo_bar', 'FooBar'),
    ]
)
def test_to_cfn_resource_name(name: str, cfn_name: str) -> None:
    assert utils.to_cfn_resource_name(name) == cfn_name

@given(name=text(alphabet=string.ascii_letters + string.digits + '-_'))
def test_to_cfn_resource_name_properties(name: str) -> None:
    try:
        result: str = utils.to_cfn_resource_name(name)
    except ValueError:
        pass
    else:
        assert re.search('[^A-Za-z0-9]', result) is None

class TestTimestampUtils:
    def setup_method(self) -> None:
        self.mock_now: Any = mock.Mock(spec=datetime.utcnow)
        self.set_now()
        self.timestamp_convert: utils.TimestampConverter = utils.TimestampConverter(self.mock_now)

    def set_now(
        self,
        year: int = 2020,
        month: int = 1,
        day: int = 1,
        hour: int = 0,
        minute: int = 0,
        sec: int = 0
    ) -> None:
        self.now: datetime = datetime(year, month, day, hour, minute, sec, tzinfo=tz.tzutc())
        self.mock_now.return_value = self.now

    def test_iso_no_timezone(self) -> None:
        result: datetime = self.timestamp_convert.timestamp_to_datetime('2020-01-01T00:00:01.000000')
        assert result == datetime(2020, 1, 1, 0, 0, 1)

    def test_iso_with_timezone(self) -> None:
        result: datetime = self.timestamp_convert.timestamp_to_datetime('2020-01-01T00:00:01.000000-01:00')
        expected: datetime = datetime(2020, 1, 1, 0, 0, 1, tzinfo=tz.tzoffset(None, -3600))
        assert result == expected

    def test_to_datetime_relative_second(self) -> None:
        self.set_now(sec=2)
        result: datetime = self.timestamp_convert.timestamp_to_datetime('1s')
        expected: datetime = datetime(2020, 1, 1, 0, 0, 1, tzinfo=tz.tzutc())
        assert result == expected

    def test_to_datetime_relative_multiple_seconds(self) -> None:
        self.set_now(sec=5)
        result: datetime = self.timestamp_convert.timestamp_to_datetime('2s')
        expected: datetime = datetime(2020, 1, 1, 0, 0, 3, tzinfo=tz.tzutc())
        assert result == expected

    def test_to_datetime_relative_minute(self) -> None:
        self.set_now(minute=2)
        result: datetime = self.timestamp_convert.timestamp_to_datetime('1m')
        expected: datetime = datetime(2020, 1, 1, 0, 1, 0, tzinfo=tz.tzutc())
        assert result == expected

    def test_to_datetime_relative_hour(self) -> None:
        self.set_now(hour=2)
        result: datetime = self.timestamp_convert.timestamp_to_datetime('1h')
        expected: datetime = datetime(2020, 1, 1, 1, 0, 0, tzinfo=tz.tzutc())
        assert result == expected

    def test_to_datetime_relative_day(self) -> None:
        self.set_now(day=3)
        result: datetime = self.timestamp_convert.timestamp_to_datetime('1d')
        expected: datetime = datetime(2020, 1, 2, 0, 0, 0, tzinfo=tz.tzutc())
        assert result == expected

    def test_to_datetime_relative_week(self) -> None:
        self.set_now(day=14)
        result: datetime = self.timestamp_convert.timestamp_to_datetime('1w')
        expected: datetime = datetime(2020, 1, 7, 0, 0, 0, tzinfo=tz.tzutc())
        assert result == expected

@pytest.mark.parametrize(
    'timestamp,expected',
    [
        ('2020-01-01', datetime(2020, 1, 1)),
        ('2020-01-01T00:00:01', datetime(2020, 1, 1, 0, 0, 1)),
        ('2020-02-02T01:02:03', datetime(2020, 2, 2, 1, 2, 3)),
        ('2020-01-01T00:00:00Z', datetime(2020, 1, 1, 0, 0, tzinfo=tz.tzutc())),
        ('2020-01-01T00:00:00-04:00', datetime(2020, 1, 1, 0, 0, 0, tzinfo=tz.tzoffset('EDT', -14400))),
    ]
)
def test_parse_iso8601_timestamp(timestamp: str, expected: datetime) -> None:
    timestamp_convert: utils.TimestampConverter = utils.TimestampConverter()
    result: datetime = timestamp_convert.parse_iso8601_timestamp(timestamp)
    assert result == expected