import re
from datetime import datetime, timezone
from time import time
import humanize
from freqtrade.constants import DATETIME_PRINT_FORMAT

def dt_now():
    """Return the current datetime in UTC."""
    return datetime.now(timezone.utc)

def dt_utc(year, month, day, hour=0, minute=0, second=0, microsecond=0):
    """Return a datetime in UTC."""
    return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc)

def dt_ts(dt=None):
    """
    Return dt in ms as a timestamp in UTC.
    If dt is None, return the current datetime in UTC.
    """
    if dt:
        return int(dt.timestamp() * 1000)
    return int(time() * 1000)

def dt_ts_def(dt, default=0):
    """
    Return dt in ms as a timestamp in UTC.
    If dt is None, return the given default.
    """
    if dt:
        return int(dt.timestamp() * 1000)
    return default

def dt_ts_none(dt):
    """
    Return dt in ms as a timestamp in UTC.
    If dt is None, return the given default.
    """
    if dt:
        return int(dt.timestamp() * 1000)
    return None

def dt_floor_day(dt):
    """Return the floor of the day for the given datetime."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def dt_from_ts(timestamp):
    """
    Return a datetime from a timestamp.
    :param timestamp: timestamp in seconds or milliseconds
    """
    if timestamp > 10000000000.0:
        timestamp /= 1000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)

def shorten_date(_date):
    """
    Trim the date so it fits on small screens
    """
    new_date = re.sub('seconds?', 'sec', _date)
    new_date = re.sub('minutes?', 'min', new_date)
    new_date = re.sub('hours?', 'h', new_date)
    new_date = re.sub('days?', 'd', new_date)
    new_date = re.sub('^an?', '1', new_date)
    return new_date

def dt_humanize_delta(dt):
    """
    Return a humanized string for the given timedelta.
    """
    return humanize.naturaltime(dt)

def format_date(date):
    """
    Return a formatted date string.
    Returns an empty string if date is None.
    :param date: datetime to format
    """
    if date:
        return date.strftime(DATETIME_PRINT_FORMAT)
    return ''

def format_ms_time(date):
    """
    convert MS date to readable format.
    : epoch-string in ms
    """
    return dt_from_ts(date).strftime('%Y-%m-%dT%H:%M:%S')

def format_ms_time_det(date):
    """
    convert MS date to readable format - detailed.
    : epoch-string in ms
    """
    return dt_from_ts(date).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]