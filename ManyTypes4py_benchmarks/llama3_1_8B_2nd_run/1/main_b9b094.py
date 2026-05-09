"""Logic for creating models."""
from __future__ import annotations as _annotations
import operator
import sys
import types
import typing
import warnings
from collections.abc import Generator, Mapping
from copy import copy, deepcopy
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Literal, TypeVar, Union, cast, overload
import pydantic_core
import typing_extensions
from pydantic_core import PydanticUndefined, ValidationError
from typing_extensions import Self, TypeAlias, Unpack
from . import PydanticDeprecatedSince20, PydanticDeprecatedSince211
from ._internal import _config, _decorators, _fields, _forward_ref, _generics, _mock_val_ser, _model_construction, _namespace_utils, _repr, _typing_extra, _utils
from ._migration import getattr_migration
from .aliases import AliasChoices, AliasPath
from .annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from .config import ConfigDict
from .errors import PydanticUndefinedAnnotation, PydanticUserError
from .json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaMode, JsonSchemaValue, model_json_schema
from .plugin._schema_validator import PluggableSchemaValidator
if TYPE_CHECKING:
    from inspect import Signature
    from pathlib import Path
    from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator
    from ._internal._namespace_utils import MappingNamespace
    from ._internal._utils import AbstractSetIntStr, MappingIntStrAny
    from .deprecated.parse import Protocol as DeprecatedParseProtocol
    from .fields import ComputedFieldInfo, FieldInfo, ModelPrivateAttr
else:
    DeprecationWarning = PydanticDeprecatedSince20
__all__ = ('BaseModel', 'create_model')
TupleGenerator = Generator[tuple[str, Any], None, None]
IncEx = Union[set[int], set[str], Mapping[int, Union['IncEx', bool]], Mapping[str, Union['IncEx', bool]]]
_object_setattr = _model_construction.object_setattr

def _check_frozen(model_cls: type, name: str, value: Any) -> None:
    if model_cls.model_config.get('frozen'):
        error_type = 'frozen_instance'
    elif getattr(model_cls.__pydantic_fields__.get(name), 'frozen', False):
        error_type = 'frozen_field'
    else:
        return
    raise ValidationError.from_exception_data(model_cls.__name__, [{'type': error_type, 'loc': (name,), 'input': value}])

def _model_field_setattr_handler(model: BaseModel, name: str, val: Any) -> None:
    model.__dict__[name] = val
    model.__pydantic_fields_set__.add(name)
_SIMPLE_SETATTR_HANDLERS: Dict[str, Callable[[BaseModel, str, Any], None]] = {'model_field': _model_field_setattr_handler, 'validate_assignment': lambda model: model.__pydantic_validator__.validate_assignment(model, name, value), 'private': lambda model, name, val: model.__pydantic_private__.__setitem__(name, val), 'cached_property': lambda model, name, val: model.__dict__.__setitem__(name, val), 'extra_known': lambda model, name, val: _object_setattr(model, name, val)}

class BaseModel(metaclass=_model_construction.ModelMetaclass):
    """!!! abstract "Usage Documentation"
        [Models](../concepts/models.md)

    A base class for creating Pydantic models.

    Attributes:
        __class_vars__: The names of the class variables defined on the model.
        __private_attributes__: Metadata about the private attributes of the model.
        __signature__: The synthesized `__init__` [`Signature`][inspect.Signature] of the model.

        __pydantic_complete__: Whether model building is completed, or if there are still undefined fields.
        __pydantic_core_schema__: The core schema of the model.
        __pydantic_custom_init__: Whether the model has a custom `__init__` function.
        __pydantic_decorators__: Metadata containing the decorators defined on the model.
            This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.
        __pydantic_generic_metadata__: Metadata for generic models; contains data used for a similar purpose to
            __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.
        __pydantic_parent_namespace__: Parent namespace of the model, used for automatic rebuilding of models.
        __pydantic_post_init__: The name of the post-init method for the model, if defined.
        __pydantic_root_model__: Whether the model is a [`RootModel`][pydantic.root_model.RootModel].
        __pydantic_serializer__: The `pydantic-core` `SchemaSerializer` used to dump instances of the model.
        __pydantic_validator__: The `pydantic-core` `SchemaValidator` used to validate instances of the model.

        __pydantic_fields__: A dictionary of field names and their corresponding [`FieldInfo`][pydantic.fields.FieldInfo] objects.
        __pydantic_computed_fields__: A dictionary of computed field names and their corresponding [`ComputedFieldInfo`][pydantic.fields.ComputedFieldInfo] objects.

        __pydantic_extra__: A dictionary containing extra values, if [`extra`][pydantic.config.ConfigDict.extra]
            is set to `'allow'`.
        __pydantic_fields_set__: The names of fields explicitly set during instantiation.
        __pydantic_private__: Values of private attributes set on the model instance.
    """
    model_config: ConfigDict = ConfigDict()  # type: ignore
    """Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict]."""
    __class_vars__: ClassVar[AbstractSetIntStr] = set()  # type: ignore
    """The names of the class variables defined on the model."""
    __private_attributes__: ClassVar[MappingIntStrAny] = {}  # type: ignore
    """Metadata about the private attributes of the model."""
    __signature__: ClassVar[Signature] = Signature()  # type: ignore
    """The synthesized `__init__` [`Signature`][inspect.Signature] of the model."""
    __pydantic_complete__: ClassVar[bool] = False  # type: ignore
    """Whether model building is completed, or if there are still undefined fields."""
    __pydantic_core_schema__: ClassVar[CoreSchema] = _mock_val_ser.MockCoreSchema('Pydantic models should inherit from BaseModel, BaseModel cannot be instantiated directly', code='base-model-instantiated')
    """The core schema of the model."""
    __pydantic_custom_init__: ClassVar[bool] = False  # type: ignore
    """Whether the model has a custom `__init__` method."""
    __pydantic_decorators__: ClassVar[_decorators.DecoratorInfos] = _decorators.DecoratorInfos()  # type: ignore
    """Metadata containing the decorators defined on the model.\n    This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1."""
    __pydantic_generic_metadata__: ClassVar[MappingIntStrAny] = {}  # type: ignore
    """Metadata for generic models; contains data used for a similar purpose to\n    __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these."""
    __pydantic_parent_namespace__: ClassVar[MappingNamespace] = None  # type: ignore
    """Parent namespace of the model, used for automatic rebuilding of models."""
    __pydantic_post_init__: ClassVar[str] = ''  # type: ignore
    """The name of the post-init method for the model, if defined."""
    __pydantic_root_model__: ClassVar[bool] = False  # type: ignore
    """Whether the model is a [`RootModel`][pydantic.root_model.RootModel]."""
    __pydantic_serializer__: ClassVar[SchemaSerializer] = _mock_val_ser.MockValSer('Pydantic models should inherit from BaseModel, BaseModel cannot be instantiated directly', val_or_ser='serializer', code='base-model-instantiated')
    """The `pydantic-core` `SchemaSerializer` used to dump instances of the model."""
    __pydantic_validator__: ClassVar[SchemaValidator] = _mock_val_ser.MockValSer('Pydantic models should inherit from BaseModel, BaseModel cannot be instantiated directly', val_or_ser='validator', code='base-model-instantiated')
    """The `pydantic-core` `SchemaValidator` used to validate instances of the model."""
    __pydantic_fields__: ClassVar[MappingIntStrFieldInfo] = {}  # type: ignore
    """A dictionary of field names and their corresponding [`FieldInfo`][pydantic.fields.FieldInfo] objects.\n    This replaces `Model.__fields__` from Pydantic V1.\n    """
    __pydantic_computed_fields__: ClassVar[MappingIntStrComputedFieldInfo] = {}  # type: ignore
    """A dictionary of computed field names and their corresponding [`ComputedFieldInfo`][pydantic.fields.ComputedFieldInfo] objects."""
    __pydantic_extra__: ClassVar[MappingIntStrAny] = _model_construction.NoInitField(init=False)  # type: ignore
    """A dictionary containing extra values, if [`extra`][pydantic.config.ConfigDict.extra] is set to `'allow'`."""
    __pydantic_fields_set__: ClassVar[AbstractSetIntStr] = _model_construction.NoInitField(init=False)  # type: ignore
    """The names of fields explicitly set during instantiation."""
    __pydantic_private__: ClassVar[MappingIntStrAny] = _model_construction.NoInitField(init=False)  # type: ignore
    """Values of private attributes set on the model instance."""

    def __init__(self: BaseModel, /, **data: Any) -> None:
        """Create a new model by parsing and validating input data from keyword arguments.

        Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
        validated to form a valid model.

        `self` is explicitly positional-only to allow `self` as a field name.
        """
        __tracebackhide__ = True
        validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
        if self is not validated_self:
            warnings.warn("A custom validator is returning a value other than `self`.\nReturning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.\nSee the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.", stacklevel=2)
    __init__.__pydantic_base_init__ = True

    @_utils.deprecated_instance_property
    @classmethod
    def model_fields(cls: type[ModelT]) -> MappingIntStrFieldInfo:
        """A mapping of field names to their respective [`FieldInfo`][pydantic.fields.FieldInfo] instances.

        !!! warning
            Accessing this attribute from a model instance is deprecated, and will not work in Pydantic V3.
            Instead, you should access this attribute from the model class.
        """
        return getattr(cls, '__pydantic_fields__', {})

    @_utils.deprecated_instance_property
    @classmethod
    def model_computed_fields(cls: type[ModelT]) -> MappingIntStrComputedFieldInfo:
        """A mapping of computed field names to their respective [`ComputedFieldInfo`][pydantic.fields.ComputedFieldInfo] instances.

        !!! warning
            Accessing this attribute from a model instance is deprecated, and will not work in Pydantic V3.
            Instead, you should access this attribute from the model class.
        """
        return getattr(cls, '__pydantic_computed_fields__', {})

    @property
    def model_extra(self: BaseModel) -> MappingIntStrAny:
        """Get extra fields set during validation.

        Returns:
            A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.
        """
        return self.__pydantic_extra__

    @property
    def model_fields_set(self: BaseModel) -> AbstractSetIntStr:
        """Returns the set of fields that have been explicitly set on this model instance.

        Returns:
            A set of strings representing the fields that have been set,
                i.e. that were not filled from defaults.
        """
        return self.__pydantic_fields_set__

    @classmethod
    def model_construct(cls: type[ModelT], _fields_set: AbstractSetIntStr = None, **values: Any) -> ModelT:
        """Creates a new instance of the `Model` class with validated data.

        Creates a new model setting `__dict__` and `__pydantic_fields_set__` from trusted or pre-validated data.
        Default values are respected, but no other validation is performed.

        !!! note
            `model_construct()` generally respects the `model_config.extra` setting on the provided model.
            That is, if `model_config.extra == 'allow'`, then all extra passed values are added to the model instance's `__dict__`
            and `__pydantic_extra__` fields. If `model_config.extra == 'ignore'` (the default), then all extra passed values are ignored.
            Because no validation is performed with a call to `model_construct()`, having `model_config.extra == 'forbid'` does not result in
            an error if extra values are passed, but they will be ignored.

        Args:
            _fields_set: A set of field names that were originally explicitly set during instantiation. If provided,
                this is directly used for the [`model_fields_set`][pydantic.BaseModel.model_fields_set] attribute.
                Otherwise, the field names from the `values` argument will be used.
            values: Trusted or pre-validated data dictionary.

        Returns:
            A new instance of the `Model` class with validated data.
        """
        m = cls.__new__(cls)
        fields_values = {}
        fields_set = set()
        for name, field in cls.__pydantic_fields__.items():
            if field.alias is not None and field.alias in values:
                fields_values[name] = values.pop(field.alias)
                fields_set.add(name)
            if name not in fields_set and field.validation_alias is not None:
                validation_aliases = field.validation_alias.choices if isinstance(field.validation_alias, AliasChoices) else [field.validation_alias]
                for alias in validation_aliases:
                    if isinstance(alias, str) and alias in values:
                        fields_values[name] = values.pop(alias)
                        fields_set.add(name)
                        break
                    elif isinstance(alias, AliasPath):
                        value = alias.search_dict_for_path(values)
                        if value is not PydanticUndefined:
                            fields_values[name] = value
                            fields_set.add(name)
                            break
            if name not in fields_set:
                if name in values:
                    fields_values[name] = values.pop(name)
                    fields_set.add(name)
                elif not field.is_required():
                    fields_values[name] = field.get_default(call_default_factory=True, validated_data=fields_values)
        if _fields_set is None:
            _fields_set = fields_set
        _extra = values if cls.model_config.get('extra') == 'allow' else None
        _object_setattr(m, '__dict__', fields_values)
        _object_setattr(m, '__pydantic_fields_set__', _fields_set)
        if not cls.__pydantic_root_model__:
            _object_setattr(m, '__pydantic_extra__', _extra)
        if cls.__pydantic_post_init__:
            m.model_post_init(None)
            if hasattr(m, '__pydantic_private__') and m.__pydantic_private__ is not None:
                for k, v in values.items():
                    if k in m.__private_attributes__:
                        m.__pydantic_private__[k] = v
        elif not cls.__pydantic_root_model__:
            _object_setattr(m, '__pydantic_private__', None)
        return m

    def model_copy(self: BaseModel, *, update: MappingIntStrAny = None, deep: bool = False) -> ModelT:
        """!!! abstract "Usage Documentation"
            [`model_copy`](../concepts/serialization.md#model_copy)

        Returns a copy of the model.

        !!! note
            The underlying instance's [`__dict__`][object.__dict__] attribute is copied. This
            might have unexpected side effects if you store anything in it, on top of the model
            fields (e.g. the value of [cached properties][functools.cached_property]).

        Args:
            update: Values to change/add in the new model. Note: the data is not validated
                before creating the new model. You should trust this data.
            deep: Set to `True` to make a deep copy of the model.

        Returns:
            New model instance.
        """
        copied = self.__deepcopy__() if deep else self.__copy__()
        if update:
            if self.model_config.get('extra') == 'allow':
                for k, v in update.items():
                    if k in self.__pydantic_fields__:
                        copied.__dict__[k] = v
                    else:
                        if copied.__pydantic_extra__ is None:
                            copied.__pydantic_extra__ = {}
                        copied.__pydantic_extra__[k] = v
            else:
                copied.__dict__.update(update)
            copied.__pydantic_fields_set__.update(update.keys())
        return copied

    def model_dump(self: BaseModel, *, mode: Literal['python', 'json'] = 'python', include: AbstractSetIntStr = None, exclude: AbstractSetIntStr = None, context: Any = None, by_alias: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False, round_trip: bool = False, warnings: bool = True, fallback: Callable[[Any], Any] = None, serialize_as_any: bool = False) -> Dict[str, Any]:
        """!!! abstract "Usage Documentation"
            [`model_dump`](../concepts/serialization.md#modelmodel_dump)

        Generate a dictionary representation of the model, optionally specifying which fields to include or exclude.

        Args:
            mode: The mode in which `to_python` should run.
                If mode is 'json', the output will only contain JSON serializable types.
                If mode is 'python', the output may contain non-JSON-serializable Python objects.
            include: A set of fields to include in the output.
            exclude: A set of fields to exclude from the output.
            context: Additional context to pass to the serializer.
            by_alias: Whether to use the field's alias in the dictionary key if defined.
            exclude_unset: Whether to exclude fields that have not been explicitly set.
            exclude_defaults: Whether to exclude fields that are set to their default value.
            exclude_none: Whether to exclude fields that have a value of `None`.
            round_trip: If True, dumped values should be valid as input for non-idempotent types such as Json[T].
            warnings: How to handle serialization errors. False/"none" ignores them, True/"warn" logs errors,
                "error" raises a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError].
            fallback: A function to call when an unknown value is encountered. If not provided,
                a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError] error is raised.
            serialize_as_any: Whether to serialize fields with duck-typing serialization behavior.

        Returns:
            A dictionary representation of the model.
        """
        return self.__pydantic_serializer__.to_python(self, mode=mode, by_alias=by_alias, include=include, exclude=exclude, context=context, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, round_trip=round_trip, warnings=warnings, fallback=fallback, serialize_as_any=serialize_as_any)

    def model_dump_json(self: BaseModel, *, indent: int = None, include: AbstractSetIntStr = None, exclude: AbstractSetIntStr = None, context: Any = None, by_alias: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False, round_trip: bool = False, warnings: bool = True, fallback: Callable[[Any], Any] = None, serialize_as_any: bool = False) -> str:
        """!!! abstract "Usage Documentation"
            [`model_dump_json`](../concepts/serialization.md#modelmodel_dump_json)

        Generates a JSON representation of the model using Pydantic's `to_json` method.

        Args:
            indent: Indentation to use in the JSON output. If None is passed, the output will be compact.
            include: Field(s) to include in the JSON output.
            exclude: Field(s) to exclude from the JSON output.
            context: Additional context to pass to the serializer.
            by_alias: Whether to serialize using field aliases.
            exclude_unset: Whether to exclude fields that have not been explicitly set.
            exclude_defaults: Whether to exclude fields that are set to their default value.
            exclude_none: Whether to exclude fields that have a value of `None`.
            round_trip: If True, dumped values should be valid as input for non-idempotent types such as Json[T].
            warnings: How to handle serialization errors. False/"none" ignores them, True/"warn" logs errors,
                "error" raises a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError].
            fallback: A function to call when an unknown value is encountered. If not provided,
                a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError] error is raised.
            serialize_as_any: Whether to serialize fields with duck-typing serialization behavior.

        Returns:
            A JSON string representation of the model.
        """
        return self.__pydantic_serializer__.to_json(self, indent=indent, include=include, exclude=exclude, context=context, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, round_trip=round_trip, warnings=warnings, fallback=fallback, serialize_as_any=serialize_as_any).decode()

    @classmethod
    def model_json_schema(cls: type[ModelT], by_alias: bool = True, ref_template: str = DEFAULT_REF_TEMPLATE, schema_generator: Type[GenerateJsonSchema] = GenerateJsonSchema, mode: Literal['validation', 'mappings'] = 'validation') -> JsonSchemaValue:
        """Generates a JSON schema for a model class.

        Args:
            by_alias: Whether to use attribute aliases or not.
            ref_template: The reference template.
            schema_generator: To override the logic used to generate the JSON schema, as a subclass of
                `GenerateJsonSchema` with your desired modifications
            mode: The mode in which to generate the schema.

        Returns:
            The JSON schema for the given model class.
        """
        return model_json_schema(cls, by_alias=by_alias, ref_template=ref_template, schema_generator=schema_generator, mode=mode)

    @classmethod
    def model_parametrized_name(cls: type[ModelT], params: Tuple[type, ...]) -> str:
        """Compute the class name for parametrizations of generic classes.

        This method can be overridden to achieve a custom naming scheme for generic BaseModels.

        Args:
            params: Tuple of types of the class. Given a generic class
                `Model` with 2 type variables and a concrete model `Model[str, int]`,
                the value `(str, int)` would be passed to `params`.

        Returns:
            String representing the new class where `params` are passed to `cls` as type variables.

        Raises:
            TypeError: Raised when trying to generate concrete names for non-generic models.
        """
        if not issubclass(cls, typing.Generic):
            raise TypeError('Concrete names should only be generated for generic models.')
        param_names = [param if isinstance(param, str) else _repr.display_as_type(param) for param in params]
        params_component = ', '.join(param_names)
        return f'{cls.__name__}[{params_component}]'

    def model_post_init(self: BaseModel, context: Any, /) -> None:
        """Override this method to perform additional initialization after `__init__` and `model_construct`.
        This is useful if you want to do some validation that requires the entire model to be initialized.
        """
        pass

    @classmethod
    def model_rebuild(cls: type[ModelT], *, force: bool = False, raise_errors: bool = True, _parent_namespace_depth: int = 2, _types_namespace: MappingNamespace = None) -> bool:
        """Try to rebuild the pydantic-core schema for the model.

        This may be necessary when one of the annotations is a ForwardRef which could not be resolved during
        the initial attempt to build the schema, and automatic rebuilding fails.

        Args:
            force: Whether to force the rebuilding of the model schema, defaults to `False`.
            raise_errors: Whether to raise errors, defaults to `True`.
            _parent_namespace_depth: The depth level of the parent namespace, defaults to 2.
            _types_namespace: The types namespace, defaults to `None`.

        Returns:
            Returns `None` if the schema is already "complete" and rebuilding was not required.
            If rebuilding _was_ required, returns `True` if rebuilding was successful, otherwise `False`.
        """
        if not force and cls.__pydantic_complete__:
            return None
        for attr in ('__pydantic_core_schema__', '__pydantic_validator__', '__pydantic_serializer__'):
            if attr in cls.__dict__:
                delattr(cls, attr)
        cls.__pydantic_complete__ = False
        if _types_namespace is not None:
            rebuild_ns = _types_namespace
        elif _parent_namespace_depth > 0:
            rebuild_ns = _typing_extra.parent_frame_namespace(parent_depth=_parent_namespace_depth, force=True) or {}
        else:
            rebuild_ns = {}
        parent_ns = _model_construction.unpack_lenient_weakvaluedict(cls.__pydantic_parent_namespace__) or {}
        ns_resolver = _namespace_utils.NsResolver(parent_namespace={**rebuild_ns, **parent_ns})
        if not cls.__pydantic_fields_complete__:
            typevars_map = _generics.get_model_typevars_map(cls)
            try:
                cls.__pydantic_fields__ = _fields.rebuild_model_fields(cls, ns_resolver=ns_resolver, typevars_map=typevars_map)
            except NameError as e:
                exc = PydanticUndefinedAnnotation.from_name_error(e)
                _mock_val_ser.set_model_mocks(cls, f'`{exc.name}`')
                if raise_errors:
                    raise exc from e
            if not raise_errors and (not cls.__pydantic_fields_complete__):
                return False
            assert cls.__pydantic_fields_complete__
        return _model_construction.complete_model_class(cls, _config.ConfigWrapper(cls.model_config, check=False), raise_errors=raise_errors, ns_resolver=ns_resolver)

    @classmethod
    def model_validate(cls: type[ModelT], obj: Any, *, strict: bool = None, from_attributes: bool = None, context: Any = None) -> ModelT:
        """Validate a pydantic model instance.

        Args:
            obj: The object to validate.
            strict: Whether to enforce types strictly.
            from_attributes: Whether to extract data from object attributes.
            context: Additional context to pass to the validator.

        Raises:
            ValidationError: If the object could not be validated.

        Returns:
            The validated model instance.
        """
        __tracebackhide__ = True
        return cls.__pydantic_validator__.validate_python(obj, strict=strict, from_attributes=from_attributes, context=context)

    @classmethod
    def model_validate_json(cls: type[ModelT], json_data: str, *, strict: bool = None, context: Any = None) -> ModelT:
        """!!! abstract "Usage Documentation"
            [JSON Parsing](../concepts/json.md#json-parsing)

        Validate the given JSON data against the Pydantic model.

        Args:
            json_data: The JSON data to validate.
            strict: Whether to enforce types strictly.
            context: Extra variables to pass to the validator.

        Returns:
            The validated Pydantic model.

        Raises:
            ValidationError: If `json_data` is not a JSON string or the object could not be validated.
        """
        __tracebackhide__ = True
        return cls.__pydantic_validator__.validate_json(json_data, strict=strict, context=context)

    @classmethod
    def model_validate_strings(cls: type[ModelT], obj: Any, *, strict: bool = None, context: Any = None) -> ModelT:
        """Validate the given object with string data against the Pydantic model.

        Args:
            obj: The object containing string data to validate.
            strict: Whether to enforce types strictly.
            context: Extra variables to pass to the validator.

        Returns:
            The validated Pydantic model.
        """
        __tracebackhide__ = True
        return cls.__pydantic_validator__.validate_strings(obj, strict=strict, context=context)

    @classmethod
    def __get_pydantic_core_schema__(cls: type[ModelT], source: Any, handler: Callable[[Any], CoreSchema], /) -> CoreSchema:
        warnings.warn('The `__get_pydantic_core_schema__` method of the `BaseModel` class is deprecated. If you are calling `super().__get_pydantic_core_schema__` when overriding the method on a Pydantic model, consider using `handler(source)` instead. However, note that overriding this method on models can lead to unexpected side effects.', PydanticDeprecatedSince211, stacklevel=2)
        schema = cls.__dict__.get('__pydantic_core_schema__')
        if schema is not None and (not isinstance(schema, _mock_val_ser.MockCoreSchema)):
            return cls.__pydantic_core_schema__
        return handler(source)

    @classmethod
    def __get_pydantic_json_schema__(cls: type[ModelT], core_schema: CoreSchema, handler: Callable[[CoreSchema], JsonSchemaValue], /) -> JsonSchemaValue:
        """Hook into generating the model's JSON schema.

        Args:
            core_schema: A `pydantic-core` CoreSchema.
                You can ignore this argument and call the handler with a new CoreSchema,
                wrap this CoreSchema (`{'type': 'nullable', 'schema': current_schema}`),
                or just call the handler with the original schema.
            handler: Call into Pydantic's internal JSON schema generation.
                This will raise a `pydantic.errors.PydanticInvalidForJsonSchema` if JSON schema
                generation fails.
                Since this gets called by `BaseModel.model_json_schema` you can override the
                `schema_generator` argument to that function to change JSON schema generation globally
                for a type.

        Returns:
            A JSON schema, as a Python object.
        """
        return handler(core_schema)

    @classmethod
    def __pydantic_init_subclass__(cls: type[ModelT], **kwargs: Any) -> None:
        """This is intended to behave just like `__init_subclass__`, but is called by `ModelMetaclass`
        only after the class is actually fully initialized. In particular, attributes like `model_fields` will
        be present when this is called.

        This is necessary because `__init_subclass__` will always be called by `type.__new__`,
        and it would require a prohibitively large refactor to the `ModelMetaclass` to ensure that
        `type.__new__` was called in such a manner that the class would already be sufficiently initialized.

        This will receive the same `kwargs` that would be passed to the standard `__init_subclass__`, namely,
        any kwargs passed to the class definition that aren't used internally by pydantic.

        Args:
            **kwargs: Any keyword arguments passed to the class definition that aren't used internally
                by pydantic.
        """
        pass

    def __class_getitem__(cls: type[ModelT], typevar_values: Tuple[type, ...]) -> type[ModelT]:
        cached = _generics.get_cached_generic_type_early(cls, typevar_values)
        if cached is not None:
            return cached
        if cls is BaseModel:
            raise TypeError('Type parameters should be placed on typing.Generic, not BaseModel')
        if not hasattr(cls, '__parameters__'):
            raise TypeError(f'{cls} cannot be parametrized because it does not inherit from typing.Generic')
        if not cls.__pydantic_generic_metadata__['parameters'] and typing.Generic not in cls.__bases__:
            raise TypeError(f'{cls} is not a generic class')
        if not isinstance(typevar_values, tuple):
            typevar_values = (typevar_values,)
        typevars_map = _generics.map_generic_model_arguments(cls, typevar_values)
        typevar_values = tuple((v for v in typevars_map.values()))
        if _utils.all_identical(typevars_map.keys(), typevars_map.values()) and typevars_map:
            submodel = cls
            _generics.set_cached_generic_type(cls, typevar_values, submodel)
        else:
            parent_args = cls.__pydantic_generic_metadata__['args']
            if not parent_args:
                args = typevar_values
            else:
                args = tuple((_generics.replace_types(arg, typevars_map) for arg in parent_args))
            origin = cls.__pydantic_generic_metadata__['origin'] or cls
            model_name = origin.model_parametrized_name(args)
            params = tuple({param: None for param in _generics.iter_contained_typevars(typevars_map.values())})
            with _generics.generic_recursion_self_type(origin, args) as maybe_self_type:
                cached = _generics.get_cached_generic_type_late(cls, typevar_values, origin, args)
                if cached is not None:
                    return cached
                if maybe_self_type is not None:
                    return maybe_self_type
                try:
                    parent_ns = _typing_extra.parent_frame_namespace(parent_depth=2) or {}
                    origin.model_rebuild(_types_namespace=parent_ns)
                except PydanticUndefinedAnnotation:
                    pass
                submodel = _generics.create_generic_submodel(model_name, origin, args, params)
                if len(_generics.recursively_defined_type_refs()) == 1:
                    _generics.set_cached_generic_type(cls, typevar_values, submodel, origin, args)
        return submodel

    def __copy__(self: BaseModel) -> BaseModel:
        """Returns a shallow copy of the model."""
        cls = type(self)
        m = cls.__new__(cls)
        _object_setattr(m, '__dict__', copy(self.__dict__))
        _object_setattr(m, '__pydantic_extra__', copy(self.__pydantic_extra__))
        _object_setattr(m, '__pydantic_fields_set__', copy(self.__pydantic_fields_set__))
        if not hasattr(self, '__pydantic_private__') or self.__pydantic_private__ is None:
            _object_setattr(m, '__pydantic_private__', None)
        else:
            _object_setattr(m, '__pydantic_private__', {k: v for k, v in self.__pydantic_private__.items() if v is not PydanticUndefined})
        return m

    def __deepcopy__(self: BaseModel, memo: Any = None) -> BaseModel:
        """Returns a deep copy of the model."""
        cls = type(self)
        m = cls.__new__(cls)
        _object_setattr(m, '__dict__', deepcopy(self.__dict__, memo=memo))
        _object_setattr(m, '__pydantic_extra__', deepcopy(self.__pydantic_extra__, memo=memo))
        _object_setattr(m, '__pydantic_fields_set__', copy(self.__pydantic_fields_set__))
        if not hasattr(self, '__pydantic_private__') or self.__pydantic_private__ is None:
            _object_setattr(m, '__pydantic_private__', None)
        else:
            _object_setattr(m, '__pydantic_private__', deepcopy({k: v for k, v in self.__pydantic_private__.items() if v is not PydanticUndefined}, memo=memo))
        return m

    def __getattr__(self: BaseModel, item: str) -> Any:
        private_attributes = object.__getattribute__(self, '__private_attributes__')
        if item in private_attributes:
            attribute = private_attributes[item]
            if hasattr(attribute, '__get__'):
                return attribute.__get__(self, type(self))
            try:
                return self.__pydantic_private__[item]
            except KeyError as exc:
                raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}') from exc
        else:
            try:
                pydantic_extra = object.__getattribute__(self, '__pydantic_extra__')
            except AttributeError:
                pydantic_extra = None
            if pydantic_extra:
                try:
                    return pydantic_extra[item]
                except KeyError as exc:
                    raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}') from exc
            elif hasattr(self.__class__, item):
                return super().__getattribute__(item)
            else:
                raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')

    def __setattr__(self: BaseModel, name: str, value: Any) -> None:
        if (setattr_handler := self.__pydantic_setattr_handlers__.get(name)) is not None:
            setattr_handler(self, name, value)
        elif (setattr_handler := self._setattr_handler(name, value)) is not None:
            setattr_handler(self, name, value)
            self.__pydantic_setattr_handlers__[name] = setattr_handler

    def _setattr_handler(self: BaseModel, name: str, value: Any) -> Callable[[BaseModel, str, Any], None]:
        """Get a handler for setting an attribute on the model instance.

        Returns:
            A handler for setting an attribute on the model instance. Used for memoization of the handler.
            Memoizing the handlers leads to a dramatic performance improvement in `__setattr__`
            Returns `None` when memoization is not safe, then the attribute is set directly.
        """
        cls = self.__class__
        if name in cls.__class_vars__:
            raise AttributeError(f'{name!r} is a ClassVar of `{cls.__name__}` and cannot be set on an instance. If you want to set a value on the class, use `{cls.__name__}.{name} = value`.')
        elif not _fields.is_valid_field_name(name):
            if (attribute := cls.__private_attributes__.get(name)) is not None:
                if hasattr(attribute, '__set__'):
                    return lambda model, _name, val: attribute.__set__(model, val)
                else:
                    return _SIMPLE_SETATTR_HANDLERS['private']
            else:
                _object_setattr(self, name, value)
                return None
        attr = getattr(cls, name, None)
        if isinstance(attr, cached_property):
            return _SIMPLE_SETATTR_HANDLERS['cached_property']
        _check_frozen(cls, name, value)
        if isinstance(attr, property):
            return lambda model, _name, val: attr.__set__(model, val)
        elif cls.model_config.get('validate_assignment'):
            return _SIMPLE_SETATTR_HANDLERS['validate_assignment']
        elif name not in cls.__pydantic_fields__:
            if cls.model_config.get('extra') != 'allow':
                raise ValueError(f'"{cls.__name__}" object has no field "{name}"')
            elif attr is None:
                self.__pydantic_extra__[name] = value
                return None
            else:
                return _SIMPLE_SETATTR_HANDLERS['extra_known']
        else:
            return _SIMPLE_SETATTR_HANDLERS['model_field']

    def __delattr__(self: BaseModel, item: str) -> None:
        cls = self.__class__
        if item in self.__private_attributes__:
            attribute = self.__private_attributes__[item]
            if hasattr(attribute, '__delete__'):
                attribute.__delete__(self)
                return
            try:
                del self.__pydantic_private__[item]
                return
            except KeyError as exc:
                raise AttributeError(f'{cls.__name__!r} object has no attribute {item!r}') from exc
        attr = getattr(cls, item, None)
        if isinstance(attr, cached_property):
            return object.__delattr__(self, item)
        _check_frozen(cls, name=item, value=None)
        if item in self.__pydantic_fields__:
            object.__delattr__(self, item)
        elif self.__pydantic_extra__ is not None and item in self.__pydantic_extra__:
            del self.__pydantic_extra__[item]
        else:
            try:
                object.__delattr__(self, item)
            except AttributeError:
                raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')

    def __replace__(self: BaseModel, **changes: Any) -> BaseModel:
        return self.model_copy(update=changes)

    def __getstate__(self: BaseModel) -> Dict[str, Any]:
        private = self.__pydantic_private__
        if private:
            private = {k: v for k, v in private.items() if v is not PydanticUndefined}
        return {'__dict__': self.__dict__, '__pydantic_extra__': self.__pydantic_extra__, '__pydantic_fields_set__': self.__pydantic_fields_set__, '__pydantic_private__': private}

    def __setstate__(self: BaseModel, state: Dict[str, Any]) -> None:
        _object_setattr(self, '__pydantic_fields_set__', state.get('__pydantic_fields_set__', {}))
        _object_setattr(self, '__pydantic_extra__', state.get('__pydantic_extra__', {}))
        _object_setattr(self, '__pydantic_private__', state.get('__pydantic_private__', {}))
        _object_setattr(self, '__dict__', state.get('__dict__', {}))

    def __eq__(self: BaseModel, other: Any) -> bool:
        if isinstance(other, BaseModel):
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
            if not (self_type == other_type and getattr(self, '__pydantic_private__', None) == getattr(other, '__pydantic_private__', None) and (self.__pydantic_extra__ == other.__pydantic_extra__)):
                return False
            if self.__dict__ == other.__dict__:
                return True
            model_fields = type(self).__pydantic_fields__.keys()
            if self.__dict__.keys() <= model_fields and other.__dict__.keys() <= model_fields:
                return False
            getter = operator.itemgetter(*model_fields) if model_fields else lambda _: _utils._SENTINEL
            try:
                return getter(self.__dict__) == getter(other.__dict__)
            except KeyError:
                self_fields_proxy = _utils.SafeGetItemProxy(self.__dict__)
                other_fields_proxy = _utils.SafeGetItemProxy(other.__dict__)
                return getter(self_fields_proxy) == getter(other_fields_proxy)
        else:
            return NotImplemented

    def __iter__(self: BaseModel) -> Iterator[tuple[str, Any]]:
        """So `dict(model)` works."""
        yield from [(k, v) for k, v in self.__dict__.items() if not k.startswith('_')]
        extra = self.__pydantic_extra__
        if extra:
            yield from extra.items()

    def __repr__(self: BaseModel) -> str:
        return f'{self.__repr_name__()}({self.__repr_str__(', ')})'

    def __repr_args__(self: BaseModel) -> Iterator[tuple[str, Any]]:
        for k, v in self.__dict__.items():
            field = self.__pydantic_fields__.get(k)
            if field and field.repr:
                if v is not self:
                    yield (k, v)
                else:
                    yield (k, self.__repr_recursion__(v))
        try:
            pydantic_extra = object.__getattribute__(self, '__pydantic_extra__')
        except AttributeError:
            pydantic_extra = None
        if pydantic_extra is not None:
            yield from ((k, v) for k, v in pydantic_extra.items())
        yield from ((k, getattr(self, k)) for k, v in self.__pydantic_computed_fields__.items() if v.repr)
    __repr_name__ = _repr.Representation.__repr_name__
    __repr_recursion__ = _repr.Representation.__repr_recursion__
    __repr_str__ = _repr.Representation.__repr_str__
    __pretty__ = _repr.Representation.__pretty__
    __rich_repr__ = _repr.Representation.__rich_repr__

    def __str__(self: BaseModel) -> str:
        return self.__repr_str__(' ')

    @property
    @typing_extensions.deprecated('The `__fields__` attribute is deprecated, use `model_fields` instead.', category=None)
    def __fields__(self: BaseModel) -> MappingIntStrFieldInfo:
        warnings.warn('The `__fields__` attribute is deprecated, use `model_fields` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        return getattr(type(self), '__pydantic_fields__', {})

    @property
    @typing_extensions.deprecated('The `__fields_set__` attribute is deprecated, use `model_fields_set` instead.', category=None)
    def __fields_set__(self: BaseModel) -> AbstractSetIntStr:
        warnings.warn('The `__fields_set__` attribute is deprecated, use `model_fields_set` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        return self.__pydantic_fields_set__

    @typing_extensions.deprecated('The `dict` method is deprecated; use `model_dump` instead.', category=None)
    def dict(self: BaseModel, *, include: AbstractSetIntStr = None, exclude: AbstractSetIntStr = None, by_alias: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False) -> Dict[str, Any]:
        warnings.warn('The `dict` method is deprecated; use `model_dump` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        return self.model_dump(include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none)

    @typing_extensions.deprecated('The `json` method is deprecated; use `model_dump_json` instead.', category=None)
    def json(self: BaseModel, *, include: AbstractSetIntStr = None, exclude: AbstractSetIntStr = None, by_alias: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False, encoder: Any = PydanticUndefined, models_as_dict: Any = PydanticUndefined, **dumps_kwargs: Any) -> str:
        warnings.warn('The `json` method is deprecated; use `model_dump_json` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        if encoder is not PydanticUndefined:
            raise TypeError('The `encoder` argument is no longer supported; use field serializers instead.')
        if models_as_dict is not PydanticUndefined:
            raise TypeError('The `models_as_dict` argument is no longer supported; use a model serializer instead.')
        if dumps_kwargs:
            raise TypeError('`dumps_kwargs` keyword arguments are no longer supported.')
        return self.model_dump_json(include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none)

    @classmethod
    @typing_extensions.deprecated('The `parse_obj` method is deprecated; use `model_validate` instead.', category=None)
    def parse_obj(cls: type[ModelT], obj: Any) -> ModelT:
        warnings.warn('The `parse_obj` method is deprecated; use `model_validate` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        return cls.model_validate(obj)

    @classmethod
    @typing_extensions.deprecated('The `parse_raw` method is deprecated; if your data is JSON use `model_validate_json`, otherwise load the data then use `model_validate` instead.', category=None)
    def parse_raw(cls: type[ModelT], b: bytes, *, content_type: Any = None, encoding: str = 'utf8', proto: Any = None, allow_pickle: bool = False) -> ModelT:
        warnings.warn('The `parse_raw` method is deprecated; if your data is JSON use `model_validate_json`, otherwise load the data then use `model_validate` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        from .deprecated import parse
        try:
            obj = parse.load_str_bytes(b, proto=proto, content_type=content_type, encoding=encoding, allow_pickle=allow_pickle)
        except (ValueError, TypeError) as exc:
            import json
            if isinstance(exc, UnicodeDecodeError):
                type_str = 'value_error.unicodedecode'
            elif isinstance(exc, json.JSONDecodeError):
                type_str = 'value_error.jsondecode'
            elif isinstance(exc, ValueError):
                type_str = 'value_error'
            else:
                type_str = 'type_error'
            error = {'type': pydantic_core.PydanticCustomError(type_str, str(exc)), 'loc': ('__root__',), 'input': b}
            raise pydantic_core.ValidationError.from_exception_data(cls.__name__, [error])
        return cls.model_validate(obj)

    @classmethod
    @typing_extensions.deprecated('The `parse_file` method is deprecated; load the data from file, then if your data is JSON use `model_validate_json`, otherwise `model_validate` instead.', category=None)
    def parse_file(cls: type[ModelT], path: Path, *, content_type: Any = None, encoding: str = 'utf8', proto: Any = None, allow_pickle: bool = False) -> ModelT:
        warnings.warn('The `parse_file` method is deprecated; load the data from file, then if your data is JSON use `model_validate_json`, otherwise `model_validate` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        from .deprecated import parse
        obj = parse.load_file(path, proto=proto, content_type=content_type, encoding=encoding, allow_pickle=allow_pickle)
        return cls.parse_obj(obj)

    @classmethod
    @typing_extensions.deprecated("The `from_orm` method is deprecated; set `model_config['from_attributes']=True` and use `model_validate` instead.", category=None)
    def from_orm(cls: type[ModelT], obj: Any) -> ModelT:
        warnings.warn("The `from_orm` method is deprecated; set `model_config['from_attributes']=True` and use `model_validate` instead.", category=PydanticDeprecatedSince20, stacklevel=2)
        if not cls.model_config.get('from_attributes', None):
            raise PydanticUserError('You must set the config attribute `from_attributes=True` to use from_orm', code=None)
        return cls.model_validate(obj)

    @classmethod
    @typing_extensions.deprecated('The `construct` method is deprecated; use `model_construct` instead.', category=None)
    def construct(cls: type[ModelT], _fields_set: AbstractSetIntStr = None, **values: Any) -> ModelT:
        warnings.warn('The `construct` method is deprecated; use `model_construct` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        return cls.model_construct(_fields_set=_fields_set, **values)

    @typing_extensions.deprecated('The `copy` method is deprecated; use `model_copy` instead. See the docstring of `BaseModel.copy` for details about how to handle `include` and `exclude`.', category=None)
    def copy(self: BaseModel, *, include: AbstractSetIntStr = None, exclude: AbstractSetIntStr = None, update: MappingIntStrAny = None, deep: bool = False) -> ModelT:
        """Returns a copy of the model.

        !!! warning "Deprecated"
            This method is now deprecated; use `model_copy` instead.

        If you need `include` or `exclude`, use:

        