from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from fastapi import params
from fastapi._compat import Undefined
from fastapi.openapi.models import Example
from typing_extensions import Annotated, Doc, deprecated
_Unset = Undefined

def Path(default=..., *, default_factory=_Unset, alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
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

def Query(default=Undefined, *, default_factory=_Unset, alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
    return params.Query(default=default, default_factory=default_factory, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Header(default=Undefined, *, default_factory=_Unset, alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, convert_underscores=True, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
    return params.Header(default=default, default_factory=default_factory, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, convert_underscores=convert_underscores, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Cookie(default=Undefined, *, default_factory=_Unset, alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
    return params.Cookie(default=default, default_factory=default_factory, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Body(default=Undefined, *, default_factory=_Unset, embed=None, media_type='application/json', alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
    return params.Body(default=default, default_factory=default_factory, embed=embed, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Form(default=Undefined, *, default_factory=_Unset, media_type='application/x-www-form-urlencoded', alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
    return params.Form(default=default, default_factory=default_factory, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def File(default=Undefined, *, default_factory=_Unset, media_type='multipart/form-data', alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
    return params.File(default=default, default_factory=default_factory, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, example=example, examples=examples, openapi_examples=openapi_examples, deprecated=deprecated, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

def Depends(dependency=None, *, use_cache=True):
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

def Security(dependency=None, *, scopes=None, use_cache=True):
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