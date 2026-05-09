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
    alias: str = None,
    alias_priority: Union[int, _Unset] = _Unset,
    validation_alias: str = None,
    serialization_alias: str = None,
    title: str = None,
    description: str = None,
    gt: Union[int, float, _Unset] = _Unset,
    ge: Union[int, float, _Unset] = _Unset,
    lt: Union[int, float, _Unset] = _Unset,
    le: Union[int, float, _Unset] = _Unset,
    min_length: Union[int, _Unset] = _Unset,
    max_length: Union[int, _Unset] = _Unset,
    pattern: Union[str, _Unset] = _Unset,
    regex: Union[str, _Unset] = _Unset,
    discriminator: Union[str, _Unset] = _Unset,
    strict: bool = _Unset,
    multiple_of: Union[int, float, _Unset] = _Unset,
    allow_inf_nan: bool = _Unset,
    max_digits: Union[int, _Unset] = _Unset,
    decimal_places: Union[int, _Unset] = _Unset,
    examples: Optional[List[Example]] = None,
    example: Union[Any, _Unset] = _Unset,
    openapi_examples: Optional[Dict[str, Example]] = None,
    deprecated: Union[bool, _Unset] = _Unset,
    include_in_schema: bool = True,
    json_schema_extra: Optional[Dict[str, Any]] = None,
    **extra: Any
) -> params.Path:
    """
    Declare a path parameter for a *path operation*.

    Read more about it in the
    [FastAPI docs for Path Parameters and Numeric Validations](https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/).

    