import warnings
from collections.abc import Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union, overload, Dict, List, Set, Tuple, Type
from typing_extensions import deprecated
from .._internal import _config, _typing_extra
from ..alias_generators import to_pascal
from ..errors import PydanticUserError
from ..functional_validators import field_validator
from ..main import BaseModel, create_model
from ..warnings import PydanticDeprecatedSince20

if not TYPE_CHECKING:
    DeprecationWarning = PydanticDeprecatedSince20

__all__ = ('validate_arguments',)

if TYPE_CHECKING:
    AnyCallable = Callable[..., Any]
    AnyCallableT = TypeVar('AnyCallableT', bound=AnyCallable)
    ConfigType = Union[None, Type[Any], Dict[str, Any]]

@overload
def validate_arguments(func: None = None, *, config: Optional[ConfigType] = None) -> Callable[[AnyCallableT], AnyCallableT]:
    ...

@overload
def validate_arguments(func: AnyCallableT) -> AnyCallableT:
    ...

@deprecated('The `validate_arguments` method is deprecated; use `validate_call` instead.', category=None)
def validate_arguments(
    func: Optional[AnyCallableT] = None, *, config: Optional[ConfigType] = None
) -> Union[AnyCallableT, Callable[[AnyCallableT], AnyCallableT]]:
    """Decorator to validate the arguments passed to a function."""
    warnings.warn(
        'The `validate_arguments` method is deprecated; use `validate_call` instead.',
        PydanticDeprecatedSince20,
        stacklevel=2
    )

    def validate(_func: AnyCallableT) -> AnyCallableT:
        vd = ValidatedFunction(_func, config)

        @wraps(_func)
        def wrapper_function(*args: Any, **kwargs: Any) -> Any:
            return vd.call(*args, **kwargs)
        
        wrapper_function.vd = vd
        wrapper_function.validate = vd.init_model_instance
        wrapper_function.raw_function = vd.raw_function
        wrapper_function.model = vd.model
        return wrapper_function  # type: ignore

    if func:
        return validate(func)
    else:
        return validate

ALT_V_ARGS: str = 'v__args'
ALT_V_KWARGS: str = 'v__kwargs'
V_POSITIONAL_ONLY_NAME: str = 'v__positional_only'
V_DUPLICATE_KWARGS: str = 'v__duplicate_kwargs'

class ValidatedFunction:
    def __init__(self, function: Callable[..., Any], config: Optional[ConfigType]) -> None:
        from inspect import Parameter, signature
        
        parameters = signature(function).parameters
        if parameters.keys() & {ALT_V_ARGS, ALT_V_KWARGS, V_POSITIONAL_ONLY_NAME, V_DUPLICATE_KWARGS}:
            raise PydanticUserError(
                f'"{ALT_V_ARGS}", "{ALT_V_KWARGS}", "{V_POSITIONAL_ONLY_NAME}" and "{V_DUPLICATE_KWARGS}" are not permitted as argument names when using the "{validate_arguments.__name__}" decorator',
                code=None
            )
        
        self.raw_function: Callable[..., Any] = function
        self.arg_mapping: Dict[int, str] = {}
        self.positional_only_args: Set[str] = set()
        self.v_args_name: str = 'args'
        self.v_kwargs_name: str = 'kwargs'
        
        type_hints = _typing_extra.get_type_hints(function, include_extras=True)
        takes_args: bool = False
        takes_kwargs: bool = False
        fields: Dict[str, Tuple[Any, Any]] = {}
        
        for i, (name, p) in enumerate(parameters.items()):
            if p.annotation is p.empty:
                annotation: Any = Any
            else:
                annotation = type_hints[name]
            
            default: Any = ... if p.default is p.empty else p.default
            
            if p.kind == Parameter.POSITIONAL_ONLY:
                self.arg_mapping[i] = name
                fields[name] = (annotation, default)
                fields[V_POSITIONAL_ONLY_NAME] = (List[str], None)
                self.positional_only_args.add(name)
            elif p.kind == Parameter.POSITIONAL_OR_KEYWORD:
                self.arg_mapping[i] = name
                fields[name] = (annotation, default)
                fields[V_DUPLICATE_KWARGS] = (List[str], None)
            elif p.kind == Parameter.KEYWORD_ONLY:
                fields[name] = (annotation, default)
            elif p.kind == Parameter.VAR_POSITIONAL:
                self.v_args_name = name
                fields[name] = (Tuple[annotation, ...], None)
                takes_args = True
            else:
                assert p.kind == Parameter.VAR_KEYWORD, p.kind
                self.v_kwargs_name = name
                fields[name] = (Dict[str, annotation], None)
                takes_kwargs = True
        
        if not takes_args and self.v_args_name in fields:
            self.v_args_name = ALT_V_ARGS
        if not takes_kwargs and self.v_kwargs_name in fields:
            self.v_kwargs_name = ALT_V_KWARGS
        if not takes_args:
            fields[self.v_args_name] = (List[Any], None)
        if not takes_kwargs:
            fields[self.v_kwargs_name] = (Dict[Any, Any], None)
        
        self.create_model(fields, takes_args, takes_kwargs, config)

    def init_model_instance(self, *args: Any, **kwargs: Any) -> BaseModel:
        values = self.build_values(args, kwargs)
        return self.model(**values)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        m = self.init_model_instance(*args, **kwargs)
        return self.execute(m)

    def build_values(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        if args:
            arg_iter = enumerate(args)
            while True:
                try:
                    i, a = next(arg_iter)
                except StopIteration:
                    break
                arg_name = self.arg_mapping.get(i)
                if arg_name is not None:
                    values[arg_name] = a
                else:
                    values[self.v_args_name] = [a] + [a for _, a in arg_iter]
                    break
        
        var_kwargs: Dict[str, Any] = {}
        wrong_positional_args: List[str] = []
        duplicate_kwargs: List[str] = []
        
        fields_alias: List[str] = [
            field.alias for name, field in self.model.__pydantic_fields__.items() 
            if name not in (self.v_args_name, self.v_kwargs_name)
        ]
        non_var_fields: Set[str] = set(self.model.__pydantic_fields__) - {self.v_args_name, self.v_kwargs_name}
        
        for k, v in kwargs.items():
            if k in non_var_fields or k in fields_alias:
                if k in self.positional_only_args:
                    wrong_positional_args.append(k)
                if k in values:
                    duplicate_kwargs.append(k)
                values[k] = v
            else:
                var_kwargs[k] = v
        
        if var_kwargs:
            values[self.v_kwargs_name] = var_kwargs
        if wrong_positional_args:
            values[V_POSITIONAL_ONLY_NAME] = wrong_positional_args
        if duplicate_kwargs:
            values[V_DUPLICATE_KWARGS] = duplicate_kwargs
        
        return values

    def execute(self, m: BaseModel) -> Any:
        d = {
            k: v for k, v in m.__dict__.items() 
            if k in m.__pydantic_fields_set__ or m.__pydantic_fields__[k].default_factory
        }
        var_kwargs: Dict[str, Any] = d.pop(self.v_kwargs_name, {})
        
        if self.v_args_name in d:
            args_: List[Any] = []
            in_kwargs = False
            kwargs: Dict[str, Any] = {}
            
            for name, value in d.items():
                if in_kwargs:
                    kwargs[name] = value
                elif name == self.v_args_name:
                    args_ += value
                    in_kwargs = True
                else:
                    args_.append(value)
            return self.raw_function(*args_, **kwargs, **var_kwargs)
        elif self.positional_only_args:
            args_: List[Any] = []
            kwargs = {}
            
            for name, value in d.items():
                if name in self.positional_only_args:
                    args_.append(value)
                else:
                    kwargs[name] = value
            return self.raw_function(*args_, **kwargs, **var_kwargs)
        else:
            return self.raw_function(**d, **var_kwargs)

    def create_model(
        self, 
        fields: Dict[str, Tuple[Any, Any]], 
        takes_args: bool, 
        takes_kwargs: bool, 
        config: Optional[ConfigType]
    ) -> None:
        pos_args = len(self.arg_mapping)
        config_wrapper = _config.ConfigWrapper(config)
        
        if config_wrapper.alias_generator:
            raise PydanticUserError(
                'Setting the "alias_generator" property on custom Config for @validate_arguments is not yet supported, please remove.',
                code=None
            )
        
        if config_wrapper.extra is None:
            config_wrapper.config_dict['extra'] = 'forbid'

        class DecoratorBaseModel(BaseModel):
            @field_validator(self.v_args_name, check_fields=False)
            @classmethod
            def check_args(cls, v: Optional[List[Any]]) -> Optional[List[Any]]:
                if takes_args or v is None:
                    return v
                raise TypeError(f'{pos_args} positional arguments expected but {pos_args + len(v)} given')

            @field_validator(self.v_kwargs_name, check_fields=False)
            @classmethod
            def check_kwargs(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
                if takes_kwargs or v is None:
                    return v
                plural = '' if len(v) == 1 else 's'
                keys = ', '.join(map(repr, v.keys()))
                raise TypeError(f'unexpected keyword argument{plural}: {keys}')

            @field_validator(V_POSITIONAL_ONLY_NAME, check_fields=False)
            @classmethod
            def check_positional_only(cls, v: Optional[List[str]]) -> None:
                if v is None:
                    return
                plural = '' if len(v) == 1 else 's'
                keys = ', '.join(map(repr, v))
                raise TypeError(f'positional-only argument{plural} passed as keyword argument{plural}: {keys}')

            @field_validator(V_DUPLICATE_KWARGS, check_fields=False)
            @classmethod
            def check_duplicate_kwargs(cls, v: Optional[List[str]]) -> None:
                if v is None:
                    return
                plural = '' if len(v) == 1 else 's'
                keys = ', '.join(map(repr, v))
                raise TypeError(f'multiple values for argument{plural}: {keys}')
            
            model_config = config_wrapper.config_dict
        
        self.model = create_model(
            to_pascal(self.raw_function.__name__),
            __base__=DecoratorBaseModel,
            **fields
        )
