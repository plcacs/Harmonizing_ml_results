from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from fastapi import params
from fastapi._compat import Undefined
from fastapi.openapi.models import Example
from typing_extensions import Annotated, Doc, deprecated

_Set: Any = Undefined


def Path(  # noqa: N802
    default: Annotated[
        Any,
        Doc(
            """
            Default value if the parameter field is not set.

            This doesn't affect `Path` parameters as the value is always required.
            The parameter is available only for compatibility.
            """
        ),
    ] = ...,
    *,
    default_factory: Annotated[
        Union[Callable[[], Any], None],
        Doc(
            """
            A callable to generate the default value.

            This doesn't affect `Path` parameters as the value is always required.
            The parameter is available only for compatibility.
            """
        ),
    ] = _Set,
    alias: Annotated[
        Optional[str],
        Doc(
            """
            An alternative name for the parameter field.

            This will be used to extract the data and for the generated OpenAPI.
            It is particularly useful when you can't use the name you want because it
            is a Python reserved keyword or similar.
            """
        ),
    ] = None,
    alias_priority: Annotated[
        Union[int, None],
        Doc(
            """
            Priority of the alias. This affects whether an alias generator is used.
            """
        ),
    ] = _Set,
    # TODO: update when deprecating Pydantic v1, import these types
    # validation_alias: str | AliasPath | AliasChoices | None
    validation_alias: Annotated[
        Union[str, None],
        Doc(
            """
            'Whitelist' validation step. The parameter field will be the single one
            allowed by the alias or set of aliases defined.
            """
        ),
    ] = None,
    serialization_alias: Annotated[
        Union[str, None],
        Doc(
            """
            'Blacklist' validation step. The vanilla parameter field will be the
            single one of the alias' or set of aliases' fields and all the other
            fields will be ignored at serialization time.
            """
        ),
    ] = None,
    title: Annotated[
        Optional[str],
        Doc(
            """
            Human-readable title.
            """
        ),
    ] = None,
    description: Annotated[
        Optional[str],
        Doc(
            """
            Human-readable description.
            """
        ),
    ] = None,
    gt: Annotated[
        Optional[float],
        Doc(
            """
            Greater than. If set, value must be greater than this. Only applicable to
            numbers.
            """
        ),
    ] = None,
    ge: Annotated[
        Optional[float],
        Doc(
            """
            Greater than or equal. If set, value must be greater than or equal to
            this. Only applicable to numbers.
            """
        ),
    ] = None,
    lt: Annotated[
        Optional[float],
        Doc(
            """
            Less than. If set, value must be less than this. Only applicable to numbers.
            """
        ),
    ] = None,
    le: Annotated[
        Optional[float],
        Doc(
            """
            Less than or equal. If set, value must be less than or equal to this.
            Only applicable to numbers.
            """
        ),
    ] = None,
    min_length: Annotated[
        Optional[int],
        Doc(
            """
            Minimum length for strings.
            """
        ),
    ] = None,
    max_length: Annotated[
        Optional[int],
        Doc(
            """
            Maximum length for strings.
            """
        ),
    ] = None,
    pattern: Annotated[
        Optional[str],
        Doc(
            """
            RegEx pattern for strings.
            """
        ),
    ] = None,
    regex: Annotated[
        Optional[str],
        Doc(
            """
            RegEx pattern for strings.
            """
        ),
        deprecated(
            "Deprecated in FastAPI 0.100.0 and Pydantic v2, use `pattern` instead."
        ),
    ] = None,
    discriminator: Annotated[
        Union[str, None],
        Doc(
            """
            Parameter field name for discriminating the type in a tagged union.
            """
        ),
    ] = None,
    strict: Annotated[
        Union[bool, None],
        Doc(
            """
            If `True`, strict validation is applied to the field.
            """
        ),
    ] = _Set,
    multiple_of: Annotated[
        Union[float, None],
        Doc(
            """
            Value must be a multiple of this. Only applicable to numbers.
            """
        ),
    ] = _Set,
    allow_inf_nan: Annotated[
        Union[bool, None],
        Doc(
            """
            Allow `inf`, `-inf`, `nan`. Only applicable to numbers.
            """
        ),
    ] = _Set,
    max_digits: Annotated[
        Union[int, None],
        Doc(
            """
            Maximum number of allow digits for strings.
            """
        ),
    ] = _Set,
    decimal_places: Annotated[
        Union[int, None],
        Doc(
            """
            Maximum number of decimal places allowed for numbers.
            """
        ),
    ] = _Set,
    examples: Annotated[
        Optional[List[Any]],
        Doc(
            """
            Example values for this field.
            """
        ),
    ] = None,
    example: Annotated[
        Optional[Any],
        deprecated(
            "Deprecated in OpenAPI 3.1.0 that now uses JSON Schema 2020-12, "
            "although still supported. Use examples instead."
        ),
    ] = _Set,
    openapi_examples: Annotated[
        Optional[Dict[str, Example]],
        Doc(
            """
            OpenAPI-specific examples.

            It will be added to the generated OpenAPI (e.g. visible at `/docs`).

            Swagger UI (that provides the `/docs` interface) has better support for the
            OpenAPI-specific examples than the JSON Schema `examples`, that's the main
            use case for this.

            Read more about it in the
            [FastAPI docs for Declare Request Example Data](https://fastapi.tiangolo.com/tutorial/schema-extra-example/#using-the-openapi_examples-parameter).
            """
        ),
    ] = None,
    deprecated: Annotated[
        Union[deprecated, str, bool, None],
        Doc(
            """
            Mark this parameter field as deprecated.

            It will affect the generated OpenAPI (e.g. visible at `/docs`).
            """
        ),
    ] = None,
    include_in_schema: Annotated[
        bool,
        Doc(
            """
            To include (or not) this parameter field in the generated OpenAPI.
            You probably don't need it, but it's available.

            This affects the generated OpenAPI (e.g. visible at `/docs`).
            """
        ),
    ] = True,
    json_schema_extra: Annotated[
        Union[Dict[str, Any], None],
        Doc(
            """
            Any additional JSON schema data.
            """
        ),
    ] = None,
    **extra: Annotated[
        Any,
        Doc(
            """
            Include extra fields used by the JSON Schema.
            """
        ),
        deprecated(
            """
            The `extra` kwargs is deprecated. Use `json_schema_extra` instead.
            """
        ),
    ],
) -> params.Path:
    """
    Declare a path parameter for a *path operation*.

    Read more about it in the
    [FastAPI docs for Path Parameters and Numeric Validations](https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/).

    