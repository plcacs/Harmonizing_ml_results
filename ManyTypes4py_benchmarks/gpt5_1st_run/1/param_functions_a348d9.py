from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from fastapi import params
from fastapi._compat import Undefined
from fastapi.openapi.models import Example
from typing_extensions import Annotated, Doc, deprecated
_Unset = Undefined

def Path(
    default: Any = ...,
    *,
    default_factory: Any = _Unset,
    alias: Optional[str] = None,
    alias_priority: Any = _Unset,
    validation_alias: Any = None,
    serialization_alias: Any = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    gt: Any = None,
    ge: Any = None,
    lt: Any = None,
    le: Any = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    regex: Optional[str] = None,
    discriminator: Any = None,
    strict: Any = _Unset,
    multiple_of: Any = _Unset,
    allow_inf_nan: Any = _Unset,
    max_digits: Any = _Unset,
    decimal_places: Any = _Unset,
    examples: Optional[List[Any]] = None,
    example: Any = _Unset,
    openapi_examples: Optional[Dict[str, Example]] = None,
    deprecated: Optional[bool] = None,
    include_in_schema: bool = True,
    json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any]], None]]] = None,
    **extra: Any,
) -> Any:
    """
    Declare a path parameter for a *path operation*.

    Read more about it in the
    [FastAPI docs for Path Parameters and Numeric Validations](https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/).

    ```python
    from typing import Annotated

    from fastapi import FastAPI, Path

    app = FastAPI()


    @app.get("/items/{item_id}")
    async def read_items(
        item_id: Annotated[int, Path(title="The ID of the item to get")],
    ):
        return {"item_id": item_id}
    ```
    """
    return params.Path(default=default, default_factory=default_factory, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Query(
    default: Any = Undefined,
    *,
    default_factory: Any = _Unset,
    alias: Optional[str] = None,
    alias_priority: Any = _Unset,
    validation_alias: Any = None,
    serialization_alias: Any = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    gt: Any = None,
    ge: Any = None,
    lt: Any = None,
    le: Any = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    regex: Optional[str] = None,
    discriminator: Any = None,
    strict: Any = _Unset,
    multiple_of: Any = _Unset,
    allow_inf_nan: Any = _Unset,
    max_digits: Any = _Unset,
    decimal_places: Any = _Unset,
    examples: Optional[List[Any]] = None,
    example: Any = _Unset,
    openapi_examples: Optional[Dict[str, Example]] = None,
    deprecated: Optional[bool] = None,
    include_in_schema: bool = True,
    json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any]], None]]] = None,
    **extra: Any,
) -> Any:
    return params.Query(default=default, default_factory=default_factory, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Header(
    default: Any = Undefined,
    *,
    default_factory: Any = _Unset,
    alias: Optional[str] = None,
    alias_priority: Any = _Unset,
    validation_alias: Any = None,
    serialization_alias: Any = None,
    convert_underscores: bool = True,
    title: Optional[str] = None,
    description: Optional[str] = None,
    gt: Any = None,
    ge: Any = None,
    lt: Any = None,
    le: Any = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    regex: Optional[str] = None,
    discriminator: Any = None,
    strict: Any = _Unset,
    multiple_of: Any = _Unset,
    allow_inf_nan: Any = _Unset,
    max_digits: Any = _Unset,
    decimal_places: Any = _Unset,
    examples: Optional[List[Any]] = None,
    example: Any = _Unset,
    openapi_examples: Optional[Dict[str, Example]] = None,
    deprecated: Optional[bool] = None,
    include_in_schema: bool = True,
    json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any]], None]]] = None,
    **extra: Any,
) -> Any:
    return params.Header(default=default, default_factory=default_factory, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, convert_underscores=convert_underscores, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Cookie(
    default: Any = Undefined,
    *,
    default_factory: Any = _Unset,
    alias: Optional[str] = None,
    alias_priority: Any = _Unset,
    validation_alias: Any = None,
    serialization_alias: Any = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    gt: Any = None,
    ge: Any = None,
    lt: Any = None,
    le: Any = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    regex: Optional[str] = None,
    discriminator: Any = None,
    strict: Any = _Unset,
    multiple_of: Any = _Unset,
    allow_inf_nan: Any = _Unset,
    max_digits: Any = _Unset,
    decimal_places: Any = _Unset,
    examples: Optional[List[Any]] = None,
    example: Any = _Unset,
    openapi_examples: Optional[Dict[str, Example]] = None,
    deprecated: Optional[bool] = None,
    include_in_schema: bool = True,
    json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any]], None]]] = None,
    **extra: Any,
) -> Any:
    return params.Cookie(default=default, default_factory=default_factory, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Body(
    default: Any = Undefined,
    *,
    default_factory: Any = _Unset,
    embed: Optional[bool] = None,
    media_type: str = 'application/json',
    alias: Optional[str] = None,
    alias_priority: Any = _Unset,
    validation_alias: Any = None,
    serialization_alias: Any = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    gt: Any = None,
    ge: Any = None,
    lt: Any = None,
    le: Any = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    regex: Optional[str] = None,
    discriminator: Any = None,
    strict: Any = _Unset,
    multiple_of: Any = _Unset,
    allow_inf_nan: Any = _Unset,
    max_digits: Any = _Unset,
    decimal_places: Any = _Unset,
    examples: Optional[List[Any]] = None,
    example: Any = _Unset,
    openapi_examples: Optional[Dict[str, Example]] = None,
    deprecated: Optional[bool] = None,
    include_in_schema: bool = True,
    json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any]], None]]] = None,
    **extra: Any,
) -> Any:
    return params.Body(default=default, default_factory=default_factory, embed=embed, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Form(
    default: Any = Undefined,
    *,
    default_factory: Any = _Unset,
    media_type: str = 'application/x-www-form-urlencoded',
    alias: Optional[str] = None,
    alias_priority: Any = _Unset,
    validation_alias: Any = None,
    serialization_alias: Any = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    gt: Any = None,
    ge: Any = None,
    lt: Any = None,
    le: Any = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    regex: Optional[str] = None,
    discriminator: Any = None,
    strict: Any = _Unset,
    multiple_of: Any = _Unset,
    allow_inf_nan: Any = _Unset,
    max_digits: Any = _Unset,
    decimal_places: Any = _Unset,
    examples: Optional[List[Any]] = None,
    example: Any = _Unset,
    openapi_examples: Optional[Dict[str, Example]] = None,
    deprecated: Optional[bool] = None,
    include_in_schema: bool = True,
    json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any]], None]]] = None,
    **extra: Any,
) -> Any:
    return params.Form(default=default, default_factory=default_factory, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def File(
    default: Any = Undefined,
    *,
    default_factory: Any = _Unset,
    media_type: str = 'multipart/form-data',
    alias: Optional[str] = None,
    alias_priority: Any = _Unset,
    validation_alias: Any = None,
    serialization_alias: Any = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    gt: Any = None,
    ge: Any = None,
    lt: Any = None,
    le: Any = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    regex: Optional[str] = None,
    discriminator: Any = None,
    strict: Any = _Unset,
    multiple_of: Any = _Unset,
    allow_inf_nan: Any = _Unset,
    max_digits: Any = _Unset,
    decimal_places: Any = _Unset,
    examples: Optional[List[Any]] = None,
    example: Any = _Unset,
    openapi_examples: Optional[Dict[str, Example]] = None,
    deprecated: Optional[bool] = None,
    include_in_schema: bool = True,
    json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any]], None]]] = None,
    **extra: Any,
) -> Any:
    return params.File(default=default, default_factory=default_factory, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Depends(
    dependency: Optional[Callable[..., Any]] = None,
    *,
    use_cache: bool = True,
) -> Any:
    """
    Declare a FastAPI dependency.

    It takes a single "dependable" callable (like a function).

    Don't call it directly, FastAPI will call it for you.

    Read more about it in the
    [FastAPI docs for Dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/).

    **Example**

    ```python
    from typing import Annotated

    from fastapi import Depends, FastAPI

    app = FastAPI()


    async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
        return {"q": q, "skip": skip, "limit": limit}


    @app.get("/items/")
    async def read_items(commons: Annotated[dict, Depends(common_parameters)]):
        return commons
    ```
    """
    return params.Depends(dependency=dependency, use_cache=use_cache)

def Security(
    dependency: Optional[Callable[..., Any]] = None,
    *,
    scopes: Optional[Sequence[str]] = None,
    use_cache: bool = True,
) -> Any:
    """
    Declare a FastAPI Security dependency.

    The only difference with a regular dependency is that it can declare OAuth2
    scopes that will be integrated with OpenAPI and the automatic UI docs (by default
    at `/docs`).

    It takes a single "dependable" callable (like a function).

    Don't call it directly, FastAPI will call it for you.

    Read more about it in the
    [FastAPI docs for Security](https://fastapi.tiangolo.com/tutorial/security/) and
    in the
    [FastAPI docs for OAuth2 scopes](https://fastapi.tiangolo.com/advanced/security/oauth2-scopes/).

    **Example**

    ```python
    from typing import Annotated

    from fastapi import Security, FastAPI

    from .db import User
    from .security import get_current_active_user

    app = FastAPI()

    @app.get("/users/me/items/")
    async def read_own_items(
        current_user: Annotated[User, Security(get_current_active_user, scopes=["items"])]
    ):
        return [{"item_id": "Foo", "owner": current_user.username}]
    ```
    """
    return params.Security(dependency=dependency, scopes=scopes, use_cache=use_cache)