from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from fastapi import params
from fastapi._compat import Undefined
from fastapi.openapi.models import Example
from typing_extensions import Annotated, Doc, deprecated
_Unset = Undefined

def Path(
    default: Any = ..., 
    *,
    default_factory: Callable[..., Any] = _Unset, 
    alias: Optional[str] = None, 
    alias_priority: Optional[Union[int, float]] = _Unset, 
    validation_alias: Optional[str] = None, 
    serialization_alias: Optional[str] = None, 
    title: Optional[str] = None, 
    description: Optional[str] = None, 
    gt: Optional[Union[int, float]] = None, 
    ge: Optional[Union[int, float]] = None, 
    lt: Optional[Union[int, float]] = None, 
    le: Optional[Union[int, float]] = None, 
    min_length: Optional[int] = None, 
    max_length: Optional[int] = None, 
    pattern: Optional[str] = None, 
    regex: Optional[str] = None, 
    discriminator: Optional[str] = None, 
    strict: Optional[bool] = _Unset, 
    multiple_of: Optional[Union[int, float]] = _Unset, 
    allow_inf_nan: Optional[bool] = _Unset, 
    max_digits: Optional[int] = _Unset, 
    decimal_places: Optional[int] = _Unset, 
    examples: Optional[Union[List[Example], Dict[str, Example]]] = None, 
    example: Optional[Any] = _Unset, 
    openapi_examples: Optional[Dict[str, Example]] = None, 
    deprecated: Optional[bool] = None, 
    include_in_schema: bool = True, 
    json_schema_extra: Optional[Dict[str, Any]] = None, 
    **extra: Any
) -> params.Path:
    """
    Declare a path parameter for a *path operation*.

    Read more about it in the
    [FastAPI docs for Path Parameters and Numeric Validations](https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/).

    