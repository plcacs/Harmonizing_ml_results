from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from fastapi import params
from fastapi._compat import Undefined
from fastapi.openapi.models import Example
from typing_extensions import Annotated, Doc, deprecated
_Unset = Undefined

def Path(
    default: Any = ...,
    *,
    default_factory: Optional[Callable[[], Any]] = _Unset,
    alias: Optional[str] = None,
    alias_priority: Optional[int] = _Unset,
    validation_alias: Optional[str] = None,
    serialization_alias: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    gt: Optional[float] = None,
    ge: Optional[float] = None,
    lt: Optional[float] = None,
    le: Optional[float] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    regex: Optional[str] = None,
    discriminator: Optional[str] = None,
    strict: Optional[bool] = _Unset,
    multiple_of: Optional[float] = _Unset,
    allow_inf_nan: Optional[bool] = _Unset,
    max_digits: Optional[int] = _Unset,
    decimal_places: Optional[int] = _Unset,
    examples: Optional[List[Example]] = None,
    example: Optional[Any] = _Unset,
    openapi_examples: Optional[Dict[str, Example]] = None,
    deprecated: Optional[bool] = None,
    include_in_schema: Optional[bool] = True,
    json_schema_extra: Optional[Dict[str, Any]] = None,
    **extra: Any
) -> params.Path:
    """
    Declare a path parameter for a *path operation*.

    Read more about it in the
    [FastAPI docs for Path Parameters and Numeric Validations](https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/).

    