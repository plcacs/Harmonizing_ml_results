import datetime
from typing import Optional, Union, List
import dateutil
import dateutil.rrule
import dateutil.tz
from pydantic import Field, field_validator, model_validator
from prefect._internal.schemas.bases import PrefectBaseModel
from prefect._internal.schemas.validators import default_anchor_date, default_timezone, validate_cron_string, validate_rrule_string
from prefect.types._datetime import Date, DateTime

MAX_ITERATIONS: int = 1000
MAX_RRULE_LENGTH: int = 6500

def is_valid_timezone(v: str) -> bool:
    from prefect._internal.pytz import HAS_PYTZ
    if HAS_PYTZ:
        import pytz
    else:
        from prefect._internal import pytz
    return v in pytz.all_timezones_set

class IntervalSchedule(PrefectBaseModel):
    model_config = ConfigDict(extra='forbid')
    interval: datetime.timedelta = Field(gt=datetime.timedelta(0))
    anchor_date: DateTime = Field(default_factory=lambda: DateTime.now('UTC'), examples=['2020-01-01T00:00:00Z'])
    timezone: Optional[str] = Field(default=None, examples=['America/New_York'])

    @model_validator(mode='after')
    def validate_timezone(self) -> 'IntervalSchedule':
        self.timezone = default_timezone(self.timezone, self.model_dump())
        return self

class CronSchedule(PrefectBaseModel):
    model_config = ConfigDict(extra='forbid')
    cron: str = Field(default=..., examples=['0 0 * * *'])
    timezone: Optional[str] = Field(default=None, examples=['America/New_York'])
    day_or: bool = Field(default=True, description='Control croniter behavior for handling day and day_of_week entries.')

    @field_validator('timezone')
    @classmethod
    def valid_timezone(cls, v: Optional[str]) -> str:
        return default_timezone(v)

    @field_validator('cron')
    @classmethod
    def valid_cron_string(cls, v: str) -> str:
        return validate_cron_string(v)

DEFAULT_ANCHOR_DATE: Date = Date(2020, 1, 1)

def _rrule_dt(rrule: dateutil.rrule.rrule, name: str = '_dtstart') -> Optional[datetime.datetime]:
    return getattr(rrule, name, None)

def _rrule(rruleset: dateutil.rrule.rruleset, name: str = '_rrule') -> List[dateutil.rrule.rrule]:
    return getattr(rruleset, name, [])

def _rdates(rrule: dateutil.rrule.rrule, name: str = '_rdate') -> List[datetime.datetime]:
    return getattr(rrule, name, [])

class RRuleSchedule(PrefectBaseModel):
    model_config = ConfigDict(extra='forbid')
    rrule: str
    timezone: str = Field(default='UTC', examples=['America/New_York'], validate_default=True)

    @field_validator('rrule')
    @classmethod
    def validate_rrule_str(cls, v: str) -> str:
        return validate_rrule_string(v)

    @classmethod
    def from_rrule(cls, rrule: Union[dateutil.rrule.rrule, dateutil.rrule.rruleset]) -> 'RRuleSchedule':
        if isinstance(rrule, dateutil.rrule.rrule):
            dtstart = _rrule_dt(rrule)
            if dtstart and dtstart.tzinfo is not None:
                timezone = dtstart.tzinfo.tzname(dtstart)
            else:
                timezone = 'UTC'
            return RRuleSchedule(rrule=str(rrule), timezone=timezone)
        rrules = _rrule(rrule)
        dtstarts = [dts for rr in rrules if (dts := _rrule_dt(rr)) is not None]
        unique_dstarts = set((DateTime.instance(d).in_tz('UTC') for d in dtstarts))
        unique_timezones = set((d.tzinfo for d in dtstarts if d.tzinfo is not None))
        if len(unique_timezones) > 1:
            raise ValueError(f'rruleset has too many dtstart timezones: {unique_timezones}')
        if len(unique_dstarts) > 1:
            raise ValueError(f'rruleset has too many dtstarts: {unique_dstarts}')
        if unique_dstarts and unique_timezones:
            [unique_tz] = unique_timezones
            timezone = unique_tz.tzname(dtstarts[0])
        else:
            timezone = 'UTC'
        rruleset_string = ''
        if rrules:
            rruleset_string += '\n'.join((str(r) for r in rrules))
        if (exrule := _rrule(rrule, '_exrule')):
            rruleset_string += '\n' if rruleset_string else ''
            rruleset_string += '\n'.join((str(r) for r in exrule)).replace('RRULE', 'EXRULE')
        if (rdates := _rdates(rrule)):
            rruleset_string += '\n' if rruleset_string else ''
            rruleset_string += 'RDATE:' + ','.join((rd.strftime('%Y%m%dT%H%M%SZ') for rd in rdates))
        if (exdates := _rdates(rrule, '_exdate')):
            rruleset_string += '\n' if rruleset_string else ''
            rruleset_string += 'EXDATE:' + ','.join((exd.strftime('%Y%m%dT%H%M%SZ') for exd in exdates))
        return RRuleSchedule(rrule=rruleset_string, timezone=timezone)

    def to_rrule(self) -> Union[dateutil.rrule.rrule, dateutil.rrule.rruleset]:
        rrule = dateutil.rrule.rrulestr(self.rrule, dtstart=DEFAULT_ANCHOR_DATE, cache=True)
        timezone = dateutil.tz.gettz(self.timezone)
        if isinstance(rrule, dateutil.rrule.rrule):
            dtstart = _rrule_dt(rrule)
            assert dtstart is not None
            kwargs = dict(dtstart=dtstart.replace(tzinfo=timezone))
            if (until := _rrule_dt(rrule, '_until')):
                kwargs.update(until=until.replace(tzinfo=timezone))
            return rrule.replace(**kwargs)
        localized_rrules = []
        for rr in _rrule(rrule):
            dtstart = _rrule_dt(rr)
            assert dtstart is not None
            kwargs = dict(dtstart=dtstart.replace(tzinfo=timezone))
            if (until := _rrule_dt(rr, '_until')):
                kwargs.update(until=until.replace(tzinfo=timezone))
            localized_rrules.append(rr.replace(**kwargs))
        setattr(rrule, '_rrule', localized_rrules)
        localized_exrules = []
        for exr in _rrule(rrule, '_exrule'):
            dtstart = _rrule_dt(exr)
            assert dtstart is not None
            kwargs = dict(dtstart=dtstart.replace(tzinfo=timezone))
            if (until := _rrule_dt(exr, '_until')):
                kwargs.update(until=until.replace(tzinfo=timezone))
            localized_exrules.append(exr.replace(**kwargs))
        setattr(rrule, '_exrule', localized_exrules)
        localized_rdates = []
        for rd in _rdates(rrule):
            localized_rdates.append(rd.replace(tzinfo=timezone))
        setattr(rrule, '_rdate', localized_rdates)
        localized_exdates = []
        for exd in _rdates(rrule, '_exdate'):
            localized_exdates.append(exd.replace(tzinfo=timezone))
        setattr(rrule, '_exdate', localized_exdates)
        return rrule

    @field_validator('timezone')
    def valid_timezone(cls, v: Optional[str]) -> str:
        if v is None:
            return 'UTC'
        if is_valid_timezone(v):
            return v
        raise ValueError(f'Invalid timezone: "{v}"')

class NoSchedule(PrefectBaseModel):
    model_config = ConfigDict(extra='forbid')

SCHEDULE_TYPES = Union[IntervalSchedule, CronSchedule, RRuleSchedule, NoSchedule]

def is_schedule_type(obj: Any) -> bool:
    return isinstance(obj, (IntervalSchedule, CronSchedule, RRuleSchedule, NoSchedule))

def construct_schedule(
    interval: Optional[Union[int, float, datetime.timedelta]] = None,
    anchor_date: Optional[DateTime] = None,
    cron: Optional[str] = None,
    rrule: Optional[str] = None,
    timezone: Optional[str] = None
) -> SCHEDULE_TYPES:
    num_schedules = sum((1 for entry in (interval, cron, rrule) if entry is not None))
    if num_schedules > 1:
        raise ValueError('Only one of interval, cron, or rrule can be provided.')
    if anchor_date and (not interval):
        raise ValueError('An anchor date can only be provided with an interval schedule')
    if timezone and (not (interval or cron or rrule)):
        raise ValueError('A timezone can only be provided with interval, cron, or rrule')
    schedule: Optional[SCHEDULE_TYPES] = None
    if interval:
        if isinstance(interval, (int, float)):
            interval = datetime.timedelta(seconds=interval)
        if not anchor_date:
            anchor_date = DateTime.now()
        schedule = IntervalSchedule(interval=interval, anchor_date=anchor_date, timezone=timezone)
    elif cron:
        schedule = CronSchedule(cron=cron, timezone=timezone)
    elif rrule:
        schedule = RRuleSchedule(rrule=rrule, timezone=timezone)
    if schedule is None:
        raise ValueError('Either interval, cron, or rrule must be provided')
    return schedule
