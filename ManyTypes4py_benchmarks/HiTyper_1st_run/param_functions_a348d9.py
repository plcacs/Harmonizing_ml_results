from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from fastapi import params
from fastapi._compat import Undefined
from fastapi.openapi.models import Example
from typing_extensions import Annotated, Doc, deprecated
_Unset = Undefined

def Path(default: Union[bool, str, dict[str, typing.Any]]=..., *, default_factory: Union[str, int, typing.Type]=_Unset, alias: Union[None, bool, str, dict[str, typing.Any]]=None, alias_priority: Union[str, int, typing.Type]=_Unset, validation_alias: Union[None, bool, str, dict[str, typing.Any]]=None, serialization_alias: Union[None, bool, str, dict[str, typing.Any]]=None, title: Union[None, bool, str, dict[str, typing.Any]]=None, description: Union[None, bool, str, dict[str, typing.Any]]=None, gt: Union[None, bool, str, dict[str, typing.Any]]=None, ge: Union[None, bool, str, dict[str, typing.Any]]=None, lt: Union[None, bool, str, dict[str, typing.Any]]=None, le: Union[None, bool, str, dict[str, typing.Any]]=None, min_length: Union[None, bool, str, dict[str, typing.Any]]=None, max_length: Union[None, bool, str, dict[str, typing.Any]]=None, pattern: Union[None, bool, str, dict[str, typing.Any]]=None, regex: Union[None, bool, str, dict[str, typing.Any]]=None, discriminator: Union[None, bool, str, dict[str, typing.Any]]=None, strict: Union[str, int, typing.Type]=_Unset, multiple_of: Union[str, int, typing.Type]=_Unset, allow_inf_nan: Union[str, int, typing.Type]=_Unset, max_digits: Union[str, int, typing.Type]=_Unset, decimal_places: Union[str, int, typing.Type]=_Unset, examples: Union[None, bool, str, dict[str, typing.Any]]=None, example: Union[str, int, typing.Type]=_Unset, openapi_examples: Union[None, bool, str, dict[str, typing.Any]]=None, deprecated: Union[None, bool, str, dict[str, typing.Any]]=None, include_in_schema: bool=True, json_schema_extra: Union[None, bool, str, dict[str, typing.Any]]=None, **extra):
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

def Query(default: Any=Undefined, *, default_factory: Union[str, int, typing.Type]=_Unset, alias: Union[None, bool, str]=None, alias_priority: Union[str, int, typing.Type]=_Unset, validation_alias: Union[None, bool, str]=None, serialization_alias: Union[None, bool, str]=None, title: Union[None, bool, str]=None, description: Union[None, bool, str]=None, gt: Union[None, bool, str]=None, ge: Union[None, bool, str]=None, lt: Union[None, bool, str]=None, le: Union[None, bool, str]=None, min_length: Union[None, bool, str]=None, max_length: Union[None, bool, str]=None, pattern: Union[None, bool, str]=None, regex: Union[None, bool, str]=None, discriminator: Union[None, bool, str]=None, strict: Union[str, int, typing.Type]=_Unset, multiple_of: Union[str, int, typing.Type]=_Unset, allow_inf_nan: Union[str, int, typing.Type]=_Unset, max_digits: Union[str, int, typing.Type]=_Unset, decimal_places: Union[str, int, typing.Type]=_Unset, examples: Union[None, bool, str]=None, example: Union[str, int, typing.Type]=_Unset, openapi_examples: Union[None, bool, str]=None, deprecated: Union[None, bool, str]=None, include_in_schema: bool=True, json_schema_extra: Union[None, bool, str]=None, **extra):
    return params.Query(default=default, default_factory=default_factory, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Header(default: Any=Undefined, *, default_factory: Union[str, int, typing.Type]=_Unset, alias: Union[None, bool]=None, alias_priority: Union[str, int, typing.Type]=_Unset, validation_alias: Union[None, bool]=None, serialization_alias: Union[None, bool]=None, convert_underscores: bool=True, title: Union[None, bool]=None, description: Union[None, bool]=None, gt: Union[None, bool]=None, ge: Union[None, bool]=None, lt: Union[None, bool]=None, le: Union[None, bool]=None, min_length: Union[None, bool]=None, max_length: Union[None, bool]=None, pattern: Union[None, bool]=None, regex: Union[None, bool]=None, discriminator: Union[None, bool]=None, strict: Union[str, int, typing.Type]=_Unset, multiple_of: Union[str, int, typing.Type]=_Unset, allow_inf_nan: Union[str, int, typing.Type]=_Unset, max_digits: Union[str, int, typing.Type]=_Unset, decimal_places: Union[str, int, typing.Type]=_Unset, examples: Union[None, bool]=None, example: Union[str, int, typing.Type]=_Unset, openapi_examples: Union[None, bool]=None, deprecated: Union[None, bool]=None, include_in_schema: bool=True, json_schema_extra: Union[None, bool]=None, **extra):
    return params.Header(default=default, default_factory=default_factory, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, convert_underscores=convert_underscores, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Cookie(default: Any=Undefined, *, default_factory: Union[str, int, typing.Type]=_Unset, alias: Union[None, bool]=None, alias_priority: Union[str, int, typing.Type]=_Unset, validation_alias: Union[None, bool]=None, serialization_alias: Union[None, bool]=None, title: Union[None, bool]=None, description: Union[None, bool]=None, gt: Union[None, bool]=None, ge: Union[None, bool]=None, lt: Union[None, bool]=None, le: Union[None, bool]=None, min_length: Union[None, bool]=None, max_length: Union[None, bool]=None, pattern: Union[None, bool]=None, regex: Union[None, bool]=None, discriminator: Union[None, bool]=None, strict: Union[str, int, typing.Type]=_Unset, multiple_of: Union[str, int, typing.Type]=_Unset, allow_inf_nan: Union[str, int, typing.Type]=_Unset, max_digits: Union[str, int, typing.Type]=_Unset, decimal_places: Union[str, int, typing.Type]=_Unset, examples: Union[None, bool]=None, example: Union[str, int, typing.Type]=_Unset, openapi_examples: Union[None, bool]=None, deprecated: Union[None, bool]=None, include_in_schema: bool=True, json_schema_extra: Union[None, bool]=None, **extra):
    return params.Cookie(default=default, default_factory=default_factory, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Body(default: Any=Undefined, *, default_factory: Union[str, int, typing.Type]=_Unset, embed: Union[None, bool, dict]=None, media_type: typing.Text='application/json', alias: Union[None, bool, dict]=None, alias_priority: Union[str, int, typing.Type]=_Unset, validation_alias: Union[None, bool, dict]=None, serialization_alias: Union[None, bool, dict]=None, title: Union[None, bool, dict]=None, description: Union[None, bool, dict]=None, gt: Union[None, bool, dict]=None, ge: Union[None, bool, dict]=None, lt: Union[None, bool, dict]=None, le: Union[None, bool, dict]=None, min_length: Union[None, bool, dict]=None, max_length: Union[None, bool, dict]=None, pattern: Union[None, bool, dict]=None, regex: Union[None, bool, dict]=None, discriminator: Union[None, bool, dict]=None, strict: Union[str, int, typing.Type]=_Unset, multiple_of: Union[str, int, typing.Type]=_Unset, allow_inf_nan: Union[str, int, typing.Type]=_Unset, max_digits: Union[str, int, typing.Type]=_Unset, decimal_places: Union[str, int, typing.Type]=_Unset, examples: Union[None, bool, dict]=None, example: Union[str, int, typing.Type]=_Unset, openapi_examples: Union[None, bool, dict]=None, deprecated: Union[None, bool, dict]=None, include_in_schema: bool=True, json_schema_extra: Union[None, bool, dict]=None, **extra):
    return params.Body(default=default, default_factory=default_factory, embed=embed, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Form(default: Any=Undefined, *, default_factory: Union[str, int, typing.Type]=_Unset, media_type: typing.Text='application/x-www-form-urlencoded', alias: Union[None, bool, str, faustypes.models.CoercionMapping]=None, alias_priority: Union[str, int, typing.Type]=_Unset, validation_alias: Union[None, bool, str, faustypes.models.CoercionMapping]=None, serialization_alias: Union[None, bool, str, faustypes.models.CoercionMapping]=None, title: Union[None, bool, str, faustypes.models.CoercionMapping]=None, description: Union[None, bool, str, faustypes.models.CoercionMapping]=None, gt: Union[None, bool, str, faustypes.models.CoercionMapping]=None, ge: Union[None, bool, str, faustypes.models.CoercionMapping]=None, lt: Union[None, bool, str, faustypes.models.CoercionMapping]=None, le: Union[None, bool, str, faustypes.models.CoercionMapping]=None, min_length: Union[None, bool, str, faustypes.models.CoercionMapping]=None, max_length: Union[None, bool, str, faustypes.models.CoercionMapping]=None, pattern: Union[None, bool, str, faustypes.models.CoercionMapping]=None, regex: Union[None, bool, str, faustypes.models.CoercionMapping]=None, discriminator: Union[None, bool, str, faustypes.models.CoercionMapping]=None, strict: Union[str, int, typing.Type]=_Unset, multiple_of: Union[str, int, typing.Type]=_Unset, allow_inf_nan: Union[str, int, typing.Type]=_Unset, max_digits: Union[str, int, typing.Type]=_Unset, decimal_places: Union[str, int, typing.Type]=_Unset, examples: Union[None, bool, str, faustypes.models.CoercionMapping]=None, example: Union[str, int, typing.Type]=_Unset, openapi_examples: Union[None, bool, str, faustypes.models.CoercionMapping]=None, deprecated: Union[None, bool, str, faustypes.models.CoercionMapping]=None, include_in_schema: bool=True, json_schema_extra: Union[None, bool, str, faustypes.models.CoercionMapping]=None, **extra):
    return params.Form(default=default, default_factory=default_factory, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def File(default: Any=Undefined, *, default_factory: Union[str, int, typing.Type]=_Unset, media_type: typing.Text='multipart/form-data', alias: Union[None, bool, str, typing.Type]=None, alias_priority: Union[str, int, typing.Type]=_Unset, validation_alias: Union[None, bool, str, typing.Type]=None, serialization_alias: Union[None, bool, str, typing.Type]=None, title: Union[None, bool, str, typing.Type]=None, description: Union[None, bool, str, typing.Type]=None, gt: Union[None, bool, str, typing.Type]=None, ge: Union[None, bool, str, typing.Type]=None, lt: Union[None, bool, str, typing.Type]=None, le: Union[None, bool, str, typing.Type]=None, min_length: Union[None, bool, str, typing.Type]=None, max_length: Union[None, bool, str, typing.Type]=None, pattern: Union[None, bool, str, typing.Type]=None, regex: Union[None, bool, str, typing.Type]=None, discriminator: Union[None, bool, str, typing.Type]=None, strict: Union[str, int, typing.Type]=_Unset, multiple_of: Union[str, int, typing.Type]=_Unset, allow_inf_nan: Union[str, int, typing.Type]=_Unset, max_digits: Union[str, int, typing.Type]=_Unset, decimal_places: Union[str, int, typing.Type]=_Unset, examples: Union[None, bool, str, typing.Type]=None, example: Union[str, int, typing.Type]=_Unset, openapi_examples: Union[None, bool, str, typing.Type]=None, deprecated: Union[None, bool, str, typing.Type]=None, include_in_schema: bool=True, json_schema_extra: Union[None, bool, str, typing.Type]=None, **extra):
    return params.File(default=default, default_factory=default_factory, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Depends(dependency: Union[None, bool, typing.Callable, typing.Sequence[str]]=None, *, use_cache: bool=True):
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

def Security(dependency: Union[None, bool, typing.Callable, typing.Sequence[str]]=None, *, scopes: Union[None, bool, typing.Callable, typing.Sequence[str]]=None, use_cache: bool=True):
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