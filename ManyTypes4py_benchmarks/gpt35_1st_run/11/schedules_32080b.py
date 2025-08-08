from typing import Union
import datetime

class IntervalSchedule(PrefectBaseModel):
    interval: datetime.timedelta
    anchor_date: Optional[DateTime]
    timezone: Optional[str]

class CronSchedule(PrefectBaseModel):
    cron: str
    timezone: Optional[str]
    day_or: Optional[bool]

class RRuleSchedule(PrefectBaseModel):
    rrule: str
    timezone: Optional[str]

class NoSchedule(PrefectBaseModel):
    pass

SCHEDULE_TYPES = Union[IntervalSchedule, CronSchedule, RRuleSchedule, NoSchedule]
