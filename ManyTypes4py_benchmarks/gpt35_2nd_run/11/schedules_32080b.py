from typing import Union
from prefect._internal.schemas.bases import PrefectBaseModel

class IntervalSchedule(PrefectBaseModel):
    interval: datetime.timedelta
    anchor_date: Optional[DateTime]
    timezone: Optional[str]

class CronSchedule(PrefectBaseModel):
    cron: str
    timezone: Optional[str]
    day_or: Optional[bool]

class RRuleSchedule(PrefectBaseModel):
    timezone: str
    rrule: str

class NoSchedule(PrefectBaseModel):
    pass

SCHEDULE_TYPES = Union[IntervalSchedule, CronSchedule, RRuleSchedule, NoSchedule]
