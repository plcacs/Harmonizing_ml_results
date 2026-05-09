import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

def to_datetime(
    arg: Union[object, List[object], np.ndarray, Index, Series, np.str_, np.int64],
    unit: Optional[str] = None,
    errors: str = "raise",
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: Optional[bool] = None,
    format: Optional[str] = None,
    exact: bool = True,
    origin: Optional[Union[datetime.datetime, datetime.date, int, str, np.datetime64]] = None,
    cache: bool = False,
) -> Union[Index, Series]:
    ...
