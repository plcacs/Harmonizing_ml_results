import warnings
from abc import ABCMeta
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from types import FunctionType, prepare_class, resolve_bases
from typing import (
    TYPE_CHECKING, AbstractSet, Any, Callable, ClassVar, Dict, List, Mapping, 
    Optional, Tuple, Type, TypeVar, Union, cast, no_type_check, overload
)
from typing_extensions import dataclass_transform
from pydantic.v1.class_validators import ValidatorGroup, extract_root_validators, extract_validators, inherit_validators
from pydantic.v1.config import BaseConfig, Extra, inherit_config, prepare_config
from pydantic.v1.error_wrappers import ErrorWrapper, ValidationError
from pydantic.v1.errors import ConfigError, DictError, ExtraError, MissingError
from pydantic.v1.fields import MAPPING_LIKE_SHAPES, Field, ModelField, ModelPrivateAttr, PrivateAttr, Undefined, is_finalvar_with_default_val
from pydantic.v1.json import custom_pydantic_encoder, pydantic_encoder
from pydantic.v1.parse import Protocol, load_file, load_str_bytes
from pydantic.v1.schema import default_ref_template, model_schema
from pydantic.v1.types import PyObject, StrBytes
from pydantic.v1.typing import (
    AnyCallable, get_args, get_origin, is_classvar, is_namedtuple, is_union, 
    resolve_annotations, update_model_forward_refs
)
from pydantic.v1.utils import (
    DUNDER_ATTRIBUTES, ROOT_KEY, ClassAttribute, GetterDict, Representation, 
    ValueItems, generate_model_signature, is_valid_field, is_valid_private_name, 
    lenient_issubclass, sequence_like, smart_deepcopy, unique_list, validate_field_name
)

if TYPE_CHECKING:
    from inspect import Signature
    from pydantic.v1.class_validators import ValidatorListDict
    from pydantic.v1.types import ModelOrDc
    from pydantic.v1.typing import (
        AbstractSetIntStr, AnyClassMethod, CallableGenerator, DictAny, DictStrAny, 
        MappingIntStrAny, ReprArgs, SetStr, TupleGenerator
    )
    Model = TypeVar('Model', bound='BaseModel')

__all__ = ('BaseModel', 'create_model', 'validate_model')
_T = TypeVar('_T')

def validate_custom_root_type(fields: Dict[str, Any]) -> None:
    if len(fields) > 1:
        raise ValueError(f'{ROOT_KEY} cannot be mixed with other fields')

def generate_hash_function(frozen: bool) -> Optional[Callable[[Any], int]]:
    def hash_function(self_: Any) -> int:
        return hash(self_.__class__) + hash(tuple(self_.__dict__.values()))
    return hash_function if frozen else None

ANNOTATED_FIELD_UNTOUCHED_TYPES = (property, type, classmethod, staticmethod)
UNTOUCHED_TYPES = (FunctionType,) + ANNOTATED_FIELD_UNTOUCHED_TYPES
_is_base_model_class_defined: bool = False

@dataclass_transform(kw_only_default=True, field_specifiers=(Field,))
class ModelMetaclass(ABCMeta):
    @no_type_check
    def __new__(
        mcs, 
        name: str, 
        bases: Tuple[type, ...], 
        namespace: Dict[str, Any], 
        **kwargs: Any
    ) -> 'ModelMetaclass':
        fields: Dict[str, Any] = {}
        config: Type[BaseConfig] = BaseConfig
        validators: Dict[str, Any] = {}
        pre_root_validators: List[Any] = []
        post_root_validators: List[Any] = []
        private_attributes: Dict[str, Any] = {}
        base_private_attributes: Dict[str, Any] = {}
        slots: Any = namespace.get('__slots__', ())
        slots = {slots} if isinstance(slots, str) else set(slots)
        class_vars: Set[str] = set()
        hash_func: Optional[Callable[[Any], int]] = None
        
        for base in reversed(bases):
            if _is_base_model_class_defined and issubclass(base, BaseModel) and (base != BaseModel):
                fields.update(smart_deepcopy(base.__fields__))
                config = inherit_config(base.__config__, config)
                validators = inherit_validators(base.__validators__, validators)
                pre_root_validators += base.__pre_root_validators__
                post_root_validators += base.__post_root_validators__
                base_private_attributes.update(base.__private_attributes__)
                class_vars.update(base.__class_vars__)
                hash_func = base.__hash__
        
        resolve_forward_refs: bool = kwargs.pop('__resolve_forward_refs__', True)
        allowed_config_kwargs: Set[str] = {key for key in dir(config) if not (key.startswith('__') and key.endswith('__'))}
        config_kwargs: Dict[str, Any] = {key: kwargs.pop(key) for key in kwargs.keys() & allowed_config_kwargs}
        config_from_namespace: Optional[Type[BaseConfig]] = namespace.get('Config')
        
        if config_kwargs and config_from_namespace:
            raise TypeError('Specifying config in two places is ambiguous, use either Config attribute or class kwargs')
        
        config = inherit_config(config_from_namespace, config, **config_kwargs)
        validators = inherit_validators(extract_validators(namespace), validators)
        vg: ValidatorGroup = ValidatorGroup(validators)
        
        for f in fields.values():
            f.set_config(config)
            extra_validators = vg.get_validators(f.name)
            if extra_validators:
                f.class_validators.update(extra_validators)
                f.populate_validators()
        
        prepare_config(config, name)
        untouched_types: Tuple[type, ...] = ANNOTATED_FIELD_UNTOUCHED_TYPES

        def is_untouched(v: Any) -> bool:
            return isinstance(v, untouched_types) or v.__class__.__name__ == 'cython_function_or_method'
        
        if (namespace.get('__module__'), namespace.get('__qualname__')) != ('pydantic.main', 'BaseModel'):
            annotations: Dict[str, Any] = resolve_annotations(namespace.get('__annotations__', {}), namespace.get('__module__', None))
            for ann_name, ann_type in annotations.items():
                if is_classvar(ann_type):
                    class_vars.add(ann_name)
                elif is_finalvar_with_default_val(ann_type, namespace.get(ann_name, Undefined)):
                    class_vars.add(ann_name)
                elif is_valid_field(ann_name):
                    validate_field_name(bases, ann_name)
                    value = namespace.get(ann_name, Undefined)
                    allowed_types = get_args(ann_type) if is_union(get_origin(ann_type)) else (ann_type,)
                    if is_untouched(value) and ann_type != PyObject and (not any((lenient_issubclass(get_origin(allowed_type), Type) for allowed_type in allowed_types))):
                        continue
                    fields[ann_name] = ModelField.infer(
                        name=ann_name, 
                        value=value, 
                        annotation=ann_type, 
                        class_validators=vg.get_validators(ann_name), 
                        config=config
                    )
                elif ann_name not in namespace and config.underscore_attrs_are_private:
                    private_attributes[ann_name] = PrivateAttr()
            
            untouched_types = UNTOUCHED_TYPES + config.keep_untouched
            for var_name, value in namespace.items():
                can_be_changed = var_name not in class_vars and (not is_untouched(value))
                if isinstance(value, ModelPrivateAttr):
                    if not is_valid_private_name(var_name):
                        raise NameError(f'Private attributes "{var_name}" must not be a valid field name; Use sunder or dunder names, e. g. "_{var_name}" or "__{var_name}__"')
                    private_attributes[var_name] = value
                elif config.underscore_attrs_are_private and is_valid_private_name(var_name) and can_be_changed:
                    private_attributes[var_name] = PrivateAttr(default=value)
                elif is_valid_field(var_name) and var_name not in annotations and can_be_changed:
                    validate_field_name(bases, var_name)
                    inferred = ModelField.infer(
                        name=var_name, 
                        value=value, 
                        annotation=annotations.get(var_name, Undefined), 
                        class_validators=vg.get_validators(var_name), 
                        config=config
                    )
                    if var_name in fields:
                        if lenient_issubclass(inferred.type_, fields[var_name].type_):
                            inferred.type_ = fields[var_name].type_
                        else:
                            raise TypeError(f'The type of {name}.{var_name} differs from the new default value; if you wish to change the type of this field, please use a type annotation')
                    fields[var_name] = inferred
        
        _custom_root_type: bool = ROOT_KEY in fields
        if _custom_root_type:
            validate_custom_root_type(fields)
        
        vg.check_for_unused()
        
        if config.json_encoders:
            json_encoder: Callable[[Any], Any] = partial(custom_pydantic_encoder, config.json_encoders)
        else:
            json_encoder = pydantic_encoder
        
        pre_rv_new, post_rv_new = extract_root_validators(namespace)
        if hash_func is None:
            hash_func = generate_hash_function(config.frozen)
        
        exclude_from_namespace: Set[str] = fields.keys() | private_attributes.keys() | {'__slots__'}
        new_namespace: Dict[str, Any] = {
            '__config__': config,
            '__fields__': fields,
            '__exclude_fields__': {name: field.field_info.exclude for name, field in fields.items() if field.field_info.exclude is not None} or None,
            '__include_fields__': {name: field.field_info.include for name, field in fields.items() if field.field_info.include is not None} or None,
            '__validators__': vg.validators,
            '__pre_root_validators__': unique_list(pre_root_validators + pre_rv_new, name_factory=lambda v: v.__name__),
            '__post_root_validators__': unique_list(post_root_validators + post_rv_new, name_factory=lambda skip_on_failure_and_v: skip_on_failure_and_v[1].__name__),
            '__schema_cache__': {},
            '__json_encoder__': staticmethod(json_encoder),
            '__custom_root_type__': _custom_root_type,
            '__private_attributes__': {**base_private_attributes, **private_attributes},
            '__slots__': slots | private_attributes.keys(),
            '__hash__': hash_func,
            '__class_vars__': class_vars,
            **{n: v for n, v in namespace.items() if n not in exclude_from_namespace}
        }
        
        cls: 'ModelMetaclass' = super().__new__(mcs, name, bases, new_namespace, **kwargs)
        cls.__signature__ = ClassAttribute('__signature__', generate_model_signature(cls.__init__, fields, config))
        
        if resolve_forward_refs:
            cls.__try_update_forward_refs__()
        
        for name, obj in namespace.items():
            if name not in new_namespace:
                set_name = getattr(obj, '__set_name__', None)
                if callable(set_name):
                    set_name(cls, name)
        
        return cls

    def __instancecheck__(self, instance: Any) -> bool:
        return hasattr(instance, '__post_root_validators__') and super().__instancecheck__(instance)

object_setattr: Callable[[object, str, Any], None] = object.__setattr__

class BaseModel(Representation, metaclass=ModelMetaclass):
    if TYPE_CHECKING:
        __fields__: Dict[str, ModelField] = {}
        __include_fields__: Optional[Dict[str, Any]] = None
        __exclude_fields__: Optional[Dict[str, Any]] = None
        __validators__: Dict[str, Any] = {}
        __config__: Type[BaseConfig] = BaseConfig
        __json_encoder__: Callable[[Any], Any] = lambda x: x
        __schema_cache__: Dict[Any, Any] = {}
        __custom_root_type__: bool = False
        __fields_set__: Set[str] = set()
    
    Config: Type[BaseConfig] = BaseConfig
    __slots__: Tuple[str, ...] = ('__dict__', '__fields_set__')
    __doc__: str = ''

    def __init__(__pydantic_self__, **data: Any) -> None:
        values, fields_set, validation_error = validate_model(__pydantic_self__.__class__, data)
        if validation_error:
            raise validation_error
        try:
            object_setattr(__pydantic_self__, '__dict__', values)
        except TypeError as e:
            raise TypeError('Model values must be a dict; you may not have returned a dictionary from a root validator') from e
        object_setattr(__pydantic_self__, '__fields_set__', fields_set)
        __pydantic_self__._init_private_attributes()

    @no_type_check
    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__private_attributes__ or name in DUNDER_ATTRIBUTES:
            return object_setattr(self, name, value)
        if self.__config__.extra is not Extra.allow and name not in self.__fields__:
            raise ValueError(f'"{self.__class__.__name__}" object has no field "{name}"')
        elif not self.__config__.allow_mutation or self.__config__.frozen:
            raise TypeError(f'"{self.__class__.__name__}" is immutable and does not support item assignment')
        elif name in self.__fields__ and self.__fields__[name].final:
            raise TypeError(f'"{self.__class__.__name__}" object "{name}" field is final and does not support reassignment')
        elif self.__config__.validate_assignment:
            new_values = {**self.__dict__, name: value}
            for validator in self.__pre_root_validators__:
                try:
                    new_values = validator(self.__class__, new_values)
                except (ValueError, TypeError, AssertionError) as exc:
                    raise ValidationError([ErrorWrapper(exc, loc=ROOT_KEY)], self.__class__)
            
            known_field = self.__fields__.get(name, None)
            if known_field:
                if not known_field.field_info.allow_mutation:
                    raise TypeError(f'"{known_field.name}" has allow_mutation set to False and cannot be assigned')
                dict_without_original_value = {k: v for k, v in self.__dict__.items() if k != name}
                value, error_ = known_field.validate(value, dict_without_original_value, loc=name, cls=self.__class__)
                if error_:
                    raise ValidationError([error_], self.__class__)
                else:
                    new_values[name] = value
            
            errors = []
            for skip_on_failure, validator in self.__post_root_validators__:
                if skip_on_failure and errors:
                    continue
                try:
                    new_values = validator(self.__class__, new_values)
                except (ValueError, TypeError, AssertionError) as exc:
                    errors.append(ErrorWrapper(exc, loc=ROOT_KEY))
            
            if errors:
                raise ValidationError(errors, self.__class__)
            object_setattr(self, '__dict__', new_values)
        else:
            self.__dict__[name] = value
        self.__fields_set__.add(name)

    def __getstate__(self) -> Dict[str, Any]:
        private_attrs = ((k, getattr(self, k, Undefined)) for k in self.__private_attributes__)
        return {
            '__dict__': self.__dict__,
            '__fields_set__': self.__fields_set__,
            '__private_attribute_values__': {k: v for k, v in private_attrs if v is not Undefined}
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        object_setattr(self, '__dict__', state['__dict__'])
        object_setattr(self, '__fields_set__', state['__fields_set__'])
        for name, value in state.get('__private_attribute_values__', {}).items():
            object_setattr(self, name, value)

    def _init_private_attributes(self) -> None:
        for name, private_attr in self.__private_attributes__.items():
            default = private_attr.get_default()
            if default is not Undefined:
                object_setattr(self, name, default)

    def dict(
        self,
        *,
        include: Optional[Union[AbstractSet[Union[str, int]], Mapping[Union[str, int], Any]]] = None,
        exclude: Optional[Union[AbstractSet[Union[str, int]], Mapping[Union[str, int], Any]]] = None,
        by_alias: bool = False,
        skip_defaults: Optional[bool] = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False
    ) -> Dict[str, Any]:
        if skip_defaults is not None:
            warnings.warn(f'{self.__class__.__name__}.dict(): "skip_defaults" is deprecated and replaced by "exclude_unset"', DeprecationWarning)
            exclude_unset = skip_defaults
        return dict(self._iter(
            to_dict=True,
            by_alias=by_alias,
            include=include,
            exclude=exclude,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none
        ))

    def json(
        self,
        *,
        include: Optional[Union[AbstractSet[Union[str, int]], Mapping[Union[str, int], Any]]] = None,
        exclude: Optional[