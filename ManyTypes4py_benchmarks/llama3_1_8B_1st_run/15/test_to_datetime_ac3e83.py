from typing import Union, List, Tuple, Any, Optional

def to_datetime(
    arg: Union[Series, Index, np.ndarray, List, Tuple, str, int, float, datetime],
    errors: str = 'raise',
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: Optional[bool] = None,
    cache: bool = False,
    unit: str = 'ns',
    format: Optional[str] = None,
    exact: bool = True,
    origin: Any = 0,
    infer_datetime_format: bool = True,
    coerce: bool = False,
    tz: Optional[str] = None,
) -> Union[Series, Index]:
