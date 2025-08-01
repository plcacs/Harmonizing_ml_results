from __future__ import annotations as _annotations
import functools
import inspect
from collections.abc import Awaitable
from functools import partial
from typing import Any, Callable, Optional, TypeVar, Union, cast
import pydantic_core
from ..config import ConfigDict
from ..plugin._schema_validator import create_schema_validator
from ._config import ConfigWrapper
from ._generate_schema import GenerateSchema, ValidateCallSupportedTypes
from ._namespace_utils import MappingNamespace, NsResolver, ns_for_function

T = TypeVar('T')
R = TypeVar('R')
FuncT = TypeVar('FuncT', bound=Callable[..., Any])

def extract_function_name(func: ValidateCallSupportedTypes) -> str:
    """Extract the name of a `ValidateCallSupportedTypes` object."""
    return f'partial({func.func.__name__})' if isinstance(func, functools.partial) else func.__name__

def extract_function_qualname(func: ValidateCallSupportedTypes) -> str:
    """Extract the qualname of a `ValidateCallSupportedTypes` object."""
    return f'partial({func.func.__qualname__})' if isinstance(func, functools.partial) else func.__qualname__

def update_wrapper_attributes(
    wrapped: Callable[..., Any], 
    wrapper: Callable[..., Any]
) -> Callable[..., Any]:
    """Update the `wrapper` function with the attributes of the `wrapped` function. Return the updated function."""
    if inspect.iscoroutinefunction(wrapped):
        @functools.wraps(wrapped)
        async def wrapper_function(*args: Any, **kwargs: Any) -> Any:
            return await wrapper(*args, **kwargs)
    else:
        @functools.wraps(wrapped)
        def wrapper_function(*args: Any, **kwargs: Any) -> Any:
            return wrapper(*args, **kwargs)
    wrapper_function.__name__ = extract_function_name(wrapped)
    wrapper_function.__qualname__ = extract_function_qualname(wrapped)
    wrapper_function.raw_function = wrapped
    return wrapper_function

class ValidateCallWrapper:
    """This is a wrapper around a function that validates the arguments passed to it, and optionally the return value."""
    __slots__ = ('__pydantic_validator__', '__return_pydantic_validator__')

    def __init__(
        self,
        function: ValidateCallSupportedTypes,
        config: Union[ConfigDict, None],
        validate_return: bool,
        parent_namespace: Optional[MappingNamespace]
    ) -> None:
        if isinstance(function, partial):
            schema_type = function.func
            module = function.func.__module__
        else:
            schema_type = function
            module = function.__module__
        qualname = extract_function_qualname(function)
        ns_resolver = NsResolver(namespaces_tuple=ns_for_function(schema_type, parent_namespace=parent_namespace))
        config_wrapper = ConfigWrapper(config)
        gen_schema = GenerateSchema(config_wrapper, ns_resolver)
        schema = gen_schema.clean_schema(gen_schema.generate_schema(function))
        core_config = config_wrapper.core_config(title=qualname)
        self.__pydantic_validator__ = create_schema_validator(schema, schema_type, module, qualname, 'validate_call', core_config, config_wrapper.plugin_settings)
        if validate_return:
            signature = inspect.signature(function)
            return_type = signature.return_annotation if signature.return_annotation is not signature.empty else Any
            gen_schema = GenerateSchema(config_wrapper, ns_resolver)
            schema = gen_schema.clean_schema(gen_schema.generate_schema(return_type))
            validator = create_schema_validator(schema, schema_type, module, qualname, 'validate_call', core_config, config_wrapper.plugin_settings)
            if inspect.iscoroutinefunction(function):
                async def return_val_wrapper(aw: Awaitable[Any]) -> Any:
                    return validator.validate_python(await aw)
                self.__return_pydantic_validator__ = return_val_wrapper
            else:
                self.__return_pydantic_validator__ = validator.validate_python
        else:
            self.__return_pydantic_validator__ = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        res = self.__pydantic_validator__.validate_python(pydantic_core.ArgsKwargs(args, kwargs))
        if self.__return_pydantic_validator__:
            return self.__return_pydantic_validator__(res)
        else:
            return res
