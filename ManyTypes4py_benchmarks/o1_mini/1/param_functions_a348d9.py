from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from fastapi import params
from fastapi._compat import Undefined
from fastapi.openapi.models import Example
from typing_extensions import Annotated, Doc, deprecated

_Unset = Undefined

def Path(
    default: Any = ...,
    *,
    default_factory: Union[Callable[[], Any], Undefined] = _Unset,
    alias: Optional[str] = None,
    alias_priority: Union[int, Undefined] = _Unset,
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
    discriminator: Optional[bool] = None,
    strict: Union[bool, Undefined] = _Unset,
    multiple_of: Union[float, Undefined] = _Unset,
    allow_inf_nan: Union[bool, Undefined] = _Unset,
    max_digits: Union[int, Undefined] = _Unset,
    decimal_places: Union[int, Undefined] = _Unset,
    examples: Optional[List[Example]] = None,
    example: Union[Any, Undefined] = _Unset,
    openapi_examples: Optional[List[Example]] = None,
    deprecated: Optional[bool] = None,
    include_in_schema: bool = True,
    json_schema_extra: Optional[Dict[str, Any]] = None,
    **extra: Any
) -> params.Path:
    """
    Declare a path parameter for a *path operation*.

    Read more about it in the
    [FastAPI docs for Path Parameters and Numeric Validations](https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/).

    