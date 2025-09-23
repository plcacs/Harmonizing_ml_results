from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from fastapi import params
from fastapi._compat import Undefined
from fastapi.openapi.models import Example
from typing_extensions import Annotated, Doc, deprecated

_Unset = Undefined

def Path(
    default: Any = ...,
    *,
    default_factory: Annotated[Union[Callable[[], Any], None], Doc("\n            A callable to generate the default value.\n\n            This doesn't affect `Path` parameters as the value is always required.\n            The parameter is available only for compatibility.\n            ")] = _Unset,
    alias: Annotated[Optional[str], Doc("\n            An alternative name for the parameter field.\n\n            This will be used to extract the data and for the generated OpenAPI.\n            It is particularly useful when you can't use the name you want because it\n            is a Python reserved keyword or similar.\n            ")] = None,
    alias_priority: Annotated[Union[int, None], Doc('\n            Priority of the alias. This affects whether an alias generator is used.\n            ')] = _Unset,
    validation_alias: Annotated[Union[str, None], Doc("\n            'Whitelist' validation step. The parameter field will be the single one\n            allowed by the alias or set of aliases defined.\n            ")] = None,
    serialization_alias: Annotated[Union[str, None], Doc("\n            'Blacklist' validation step. The vanilla parameter field will be the\n            single one of the alias' or set of aliases' fields and all the other\n            fields will be ignored at serialization time.\n            ")] = None,
    title: Annotated[Optional[str], Doc('\n            Human-readable title.\n            ')] = None,
    description: Annotated[Optional[str], Doc('\n            Human-readable description.\n            ')] = None,
    gt: Annotated[Optional[float], Doc('\n            Greater than. If set, value must be greater than this. Only applicable to\n            numbers.\n            ')] = None,
    ge: Annotated[Optional[float], Doc('\n            Greater than or equal. If set, value must be greater than or equal to\n            this. Only applicable to numbers.\n            ')] = None,
    lt: Annotated[Optional[float], Doc('\n            Less than. If set, value must be less than this. Only applicable to numbers.\n            ')] = None,
    le: Annotated[Optional[float], Doc('\n            Less than or equal. If set, value must be less than or equal to this.\n            Only applicable to numbers.\n            ')] = None,
    min_length: Annotated[Optional[int], Doc('\n            Minimum length for strings.\n            ')] = None,
    max_length: Annotated[Optional[int], Doc('\n            Maximum length for strings.\n            ')] = None,
    pattern: Annotated[Optional[str], Doc('\n            RegEx pattern for strings.\n            ')] = None,
    regex: Annotated[Optional[str], Doc('\n            RegEx pattern for strings.\n            '), deprecated('Deprecated in FastAPI 0.100.0 and Pydantic v2, use `pattern` instead.')] = None,
    discriminator: Annotated[Union[str, None], Doc('\n            Parameter field name for discriminating the type in a tagged union.\n            ')] = None,
    strict: Annotated[Union[bool, None], Doc('\n            If `True`, strict validation is applied to the field.\n            ')] = _Unset,
    multiple_of: Annotated[Union[float, None], Doc('\n            Value must be a multiple of this. Only applicable to numbers.\n            ')] = _Unset,
    allow_inf_nan: Annotated[Union[bool, None], Doc('\n            Allow `inf`, `-inf`, `nan`. Only applicable to numbers.\n            ')] = _Unset,
    max_digits: Annotated[Union[int, None], Doc('\n            Maximum number of allow digits for strings.\n            ')] = _Unset,
    decimal_places: Annotated[Union[int, None], Doc('\n            Maximum number of decimal places allowed for numbers.\n            ')] = _Unset,
    examples: Annotated[Optional[List[Any]], Doc('\n            Example values for this field.\n            ')] = None,
    example: Annotated[Optional[Any], deprecated('Deprecated in OpenAPI 3.1.0 that now uses JSON Schema 2020-12, although still supported. Use examples instead.')] = _Unset,
    openapi_examples: Annotated[Optional[Dict[str, Example]], Doc("\n            OpenAPI-specific examples.\n\n            It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n            Swagger UI (that provides the `/docs` interface) has better support for the\n            OpenAPI-specific examples than the JSON Schema `examples`, that's the main\n            use case for this.\n\n            Read more about it in the\n            [FastAPI docs for Declare Request Example Data](https://fastapi.tiangolo.com/tutorial/schema-extra-example/#using-the-openapi_examples-parameter).\n            ")] = None,
    deprecated: Annotated[Union[deprecated, str, bool, None], Doc('\n            Mark this parameter field as deprecated.\n\n            It will affect the generated OpenAPI (e.g. visible at `/docs`).\n            ')] = None,
    include_in_schema: Annotated[bool, Doc("\n            To include (or not) this parameter field in the generated OpenAPI.\n            You probably don't need it, but it's available.\n\n            This affects the generated OpenAPI (e.g. visible at `/docs`).\n            ")] = True,
    json_schema_extra: Annotated[Union[Dict[str, Any], None], Doc('\n            Any additional JSON schema data.\n            ')] = None,
    **extra: Annotated[Any, Doc('\n            Include extra fields used by the JSON Schema.\n            '), deprecated('\n            The `extra` kwargs is deprecated. Use `json_schema_extra` instead.\n            ')]
) -> Any:
    """
    Declare a path parameter for a *path operation*.

    Read more about it in the
    [FastAPI docs for Path Parameters and Numeric Validations](https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/).

    