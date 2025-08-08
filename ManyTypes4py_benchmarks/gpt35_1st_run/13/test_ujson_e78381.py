from typing import Any, Dict, List, Union

def _clean_dict(d: Dict[Any, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in d.items()}

def test_dataframe(orient: Union[None, str]) -> None:
    ...

def test_series(orient: Union[None, str]) -> None:
    ...

def test_index() -> None:
    ...

def test_datetime_index() -> None:
    ...

def test_encode_timedelta_iso(td: Timedelta) -> str:
    ...
