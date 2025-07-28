from __future__ import annotations
import dataclasses
import inspect
import sys
import typing
from collections.abc import Mapping
from copy import copy
from dataclasses import Field as DataclassField
from functools import cached_property
from typing import Annotated, Any, Callable, ClassVar, Dict, Iterator, List, Optional, Tuple, TypeVar, overload, cast
from warnings import warn
import annotated_types
import typing_extensions
from pydantic_core import PydanticUndefined
from typing_extensions import TypeAlias, Unpack, deprecated

from . import types
from ._internal import _decorators, _fields, _generics, _internal_dataclass, _repr, _typing_extra, _utils
from ._internal._namespace_utils import GlobalsNamespace, MappingNamespace
from .aliases import AliasChoices, AliasPath
from .config import JsonDict
from .errors import PydanticUserError
from .json_schema import PydanticJsonSchemaWarning
from .warnings import PydanticDeprecatedSince20

if typing.TYPE_CHECKING:
    from ._internal._repr import ReprArgs
else:
    DeprecationWarning = PydanticDeprecatedSince20

__all__ = ('Field', 'PrivateAttr', 'computed_field')
_Unset: Any = PydanticUndefined
if sys.version_info >= (3, 13):
    import warnings
    Deprecated = warnings.deprecated | deprecated
else:
    Deprecated = deprecated


class _FromFieldInfoInputs(typing_extensions.TypedDict, total=False):
    pass


class _FieldInfoInputs(_FromFieldInfoInputs, total=False):
    pass


class FieldInfo(_repr.Representation):
    __slots__ = (
        'annotation',
        'default',
        'default_factory',
        'alias',
        'alias_priority',
        'validation_alias',
        'serialization_alias',
        'title',
        'field_title_generator',
        'description',
        'examples',
        'exclude',
        'discriminator',
        'deprecated',
        'json_schema_extra',
        'frozen',
        'validate_default',
        'repr',
        'init',
        'init_var',
        'kw_only',
        'metadata',
        '_attributes_set',
        '_complete',
        '_original_assignment',
        '_original_annotation'
    )
    metadata_lookup: ClassVar[Dict[str, Optional[Callable[[Any], Any]]]] = {
        'strict': types.Strict,
        'gt': annotated_types.Gt,
        'ge': annotated_types.Ge,
        'lt': annotated_types.Lt,
        'le': annotated_types.Le,
        'multiple_of': annotated_types.MultipleOf,
        'min_length': annotated_types.MinLen,
        'max_length': annotated_types.MaxLen,
        'pattern': None,
        'allow_inf_nan': None,
        'max_digits': None,
        'decimal_places': None,
        'union_mode': None,
        'coerce_numbers_to_str': None,
        'fail_fast': types.FailFast
    }

    def __init__(self, **kwargs: Any) -> None:
        self._attributes_set: Dict[str, Any] = {k: v for k, v in kwargs.items() if v is not _Unset}
        kwargs = {k: _DefaultValues.get(k) if v is _Unset else v for k, v in kwargs.items()}
        self.annotation: Any = kwargs.get('annotation')
        default = kwargs.pop('default', PydanticUndefined)
        if default is Ellipsis:
            self.default = PydanticUndefined
            self._attributes_set.pop('default', None)
        else:
            self.default = default
        self.default_factory: Any = kwargs.pop('default_factory', None)
        if self.default is not PydanticUndefined and self.default_factory is not None:
            raise TypeError('cannot specify both default and default_factory')
        self.alias = kwargs.pop('alias', None)
        self.validation_alias = kwargs.pop('validation_alias', None)
        self.serialization_alias = kwargs.pop('serialization_alias', None)
        alias_is_set = any((alias is not None for alias in (self.alias, self.validation_alias, self.serialization_alias)))
        self.alias_priority = kwargs.pop('alias_priority', None) or 2 if alias_is_set else None
        self.title = kwargs.pop('title', None)
        self.field_title_generator = kwargs.pop('field_title_generator', None)
        self.description = kwargs.pop('description', None)
        self.examples = kwargs.pop('examples', None)
        self.exclude = kwargs.pop('exclude', None)
        self.discriminator = kwargs.pop('discriminator', None)
        self.deprecated = kwargs.pop('deprecated', getattr(self, 'deprecated', None))
        self.repr = kwargs.pop('repr', True)
        self.json_schema_extra = kwargs.pop('json_schema_extra', None)
        self.validate_default = kwargs.pop('validate_default', None)
        self.frozen = kwargs.pop('frozen', None)
        self.init = kwargs.pop('init', None)
        self.init_var = kwargs.pop('init_var', None)
        self.kw_only = kwargs.pop('kw_only', None)
        self.metadata = self._collect_metadata(kwargs)
        self._complete = True
        self._original_annotation = PydanticUndefined
        self._original_assignment = PydanticUndefined

    @staticmethod
    def from_field(default: Any = PydanticUndefined, **kwargs: Any) -> FieldInfo:
        if 'annotation' in kwargs:
            raise TypeError('"annotation" is not permitted as a Field keyword argument')
        return FieldInfo(default=default, **kwargs)

    @staticmethod
    def from_annotation(annotation: Any) -> FieldInfo:
        final = _typing_extra.is_finalvar(annotation)
        if final:
            if _typing_extra.is_generic_alias(annotation):
                annotation = typing_extensions.get_args(annotation)[0]
            else:
                return FieldInfo(annotation=Any, frozen=True)
        annotation, metadata = _typing_extra.unpack_annotated(annotation)
        if metadata:
            if not final:
                final = _typing_extra.is_finalvar(annotation)
                if final and _typing_extra.is_generic_alias(annotation):
                    annotation = typing_extensions.get_args(annotation)[0]
            field_info_annotations = [a for a in metadata if isinstance(a, FieldInfo)]
            field_info = FieldInfo.merge_field_infos(*field_info_annotations, annotation=annotation)
            if field_info:
                new_field_info = copy(field_info)
                new_field_info.annotation = annotation
                new_field_info.frozen = final or field_info.frozen
                field_metadata: List[Any] = []
                for a in metadata:
                    if _typing_extra.is_deprecated_instance(a):
                        new_field_info.deprecated = a.message
                    elif not isinstance(a, FieldInfo):
                        field_metadata.append(a)
                    else:
                        field_metadata.extend(a.metadata)
                new_field_info.metadata = field_metadata
                return new_field_info
        return FieldInfo(annotation=annotation, frozen=final or None)

    @staticmethod
    def from_annotated_attribute(annotation: Any, default: Any) -> FieldInfo:
        if annotation is default:
            raise PydanticUserError(
                "Error when building FieldInfo from annotated attribute. Make sure you don't have any field name clashing with a type annotation.",
                code='unevaluable-type-annotation'
            )
        final = _typing_extra.is_finalvar(annotation)
        if final and _typing_extra.is_generic_alias(annotation):
            annotation = typing_extensions.get_args(annotation)[0]
        if isinstance(default, FieldInfo):
            default.annotation, annotation_metadata = _typing_extra.unpack_annotated(annotation)
            default.metadata += annotation_metadata
            default = default.merge_field_infos(
                *[x for x in annotation_metadata if isinstance(x, FieldInfo)], default, annotation=default.annotation
            )
            default.frozen = final or default.frozen
            return default
        if isinstance(default, dataclasses.Field):
            init_var = False
            if annotation is dataclasses.InitVar:
                init_var = True
                annotation = typing.cast(Any, Any)
            elif isinstance(annotation, dataclasses.InitVar):
                init_var = True
                annotation = annotation.type
            pydantic_field = FieldInfo._from_dataclass_field(default)
            pydantic_field.annotation, annotation_metadata = _typing_extra.unpack_annotated(annotation)
            pydantic_field.metadata += annotation_metadata
            pydantic_field = pydantic_field.merge_field_infos(
                *[x for x in annotation_metadata if isinstance(x, FieldInfo)],
                pydantic_field,
                annotation=pydantic_field.annotation
            )
            pydantic_field.frozen = final or pydantic_field.frozen
            pydantic_field.init_var = init_var
            pydantic_field.init = getattr(default, 'init', None)
            pydantic_field.kw_only = getattr(default, 'kw_only', None)
            return pydantic_field
        annotation, metadata = _typing_extra.unpack_annotated(annotation)
        if metadata:
            field_infos = [a for a in metadata if isinstance(a, FieldInfo)]
            field_info = FieldInfo.merge_field_infos(*field_infos, annotation=annotation, default=default)
            field_metadata: List[Any] = []
            for a in metadata:
                if _typing_extra.is_deprecated_instance(a):
                    field_info.deprecated = a.message
                elif not isinstance(a, FieldInfo):
                    field_metadata.append(a)
                else:
                    field_metadata.extend(a.metadata)
            field_info.metadata = field_metadata
            return field_info
        return FieldInfo(annotation=annotation, default=default, frozen=final or None)

    @staticmethod
    def merge_field_infos(*field_infos: FieldInfo, **overrides: Any) -> FieldInfo:
        if len(field_infos) == 1:
            field_info = copy(field_infos[0])
            field_info._attributes_set.update(overrides)
            default_override = overrides.pop('default', PydanticUndefined)
            if default_override is Ellipsis:
                default_override = PydanticUndefined
            if default_override is not PydanticUndefined:
                field_info.default = default_override
            for k, v in overrides.items():
                setattr(field_info, k, v)
            return field_info
        merged_field_info_kwargs: Dict[str, Any] = {}
        metadata: Dict[Any, Any] = {}
        for field_info in field_infos:
            attributes_set = field_info._attributes_set.copy()
            try:
                json_schema_extra = attributes_set.pop('json_schema_extra')
                existing_json_schema_extra = merged_field_info_kwargs.get('json_schema_extra')
                if existing_json_schema_extra is None:
                    merged_field_info_kwargs['json_schema_extra'] = json_schema_extra
                if isinstance(existing_json_schema_extra, dict):
                    if isinstance(json_schema_extra, dict):
                        merged_field_info_kwargs['json_schema_extra'] = {**existing_json_schema_extra, **json_schema_extra}
                    if callable(json_schema_extra):
                        warn(
                            "Composing `dict` and `callable` type `json_schema_extra` is not supported.The `callable` type is being ignored.If you'd like support for this behavior, please open an issue on pydantic.",
                            PydanticJsonSchemaWarning
                        )
                elif callable(json_schema_extra):
                    merged_field_info_kwargs['json_schema_extra'] = json_schema_extra
            except KeyError:
                pass
            merged_field_info_kwargs.update(attributes_set)
            for x in field_info.metadata:
                if not isinstance(x, FieldInfo):
                    metadata[type(x)] = x
        merged_field_info_kwargs.update(overrides)
        field_info = FieldInfo(**merged_field_info_kwargs)
        field_info.metadata = list(metadata.values())
        return field_info

    @staticmethod
    def _from_dataclass_field(dc_field: dataclasses.Field) -> FieldInfo:
        default = dc_field.default
        if default is dataclasses.MISSING:
            default = _Unset
        if dc_field.default_factory is dataclasses.MISSING:
            default_factory = _Unset
        else:
            default_factory = dc_field.default_factory
        dc_field_metadata = {k: v for k, v in dc_field.metadata.items() if k in _FIELD_ARG_NAMES}
        return Field(default=default, default_factory=default_factory, repr=dc_field.repr, **dc_field_metadata)

    @staticmethod
    def _collect_metadata(kwargs: Dict[str, Any]) -> List[Any]:
        metadata: List[Any] = []
        general_metadata: Dict[str, Any] = {}
        for key, value in list(kwargs.items()):
            try:
                marker = FieldInfo.metadata_lookup[key]
            except KeyError:
                continue
            del kwargs[key]
            if value is not None:
                if marker is None:
                    general_metadata[key] = value
                else:
                    metadata.append(marker(value))
        if general_metadata:
            metadata.append(_fields.pydantic_general_metadata(**general_metadata))
        return metadata

    @property
    def deprecation_message(self) -> Optional[str]:
        if self.deprecated is None:
            return None
        if isinstance(self.deprecated, bool):
            return 'deprecated' if self.deprecated else None
        return self.deprecated if isinstance(self.deprecated, str) else self.deprecated.message

    @property
    def default_factory_takes_validated_data(self) -> Optional[bool]:
        if self.default_factory is not None:
            return _fields.takes_validated_data_argument(self.default_factory)
        return None

    @overload
    def get_default(self, *, call_default_factory: bool, validated_data: Dict[str, Any]) -> Any:
        ...

    @overload
    def get_default(self, *, call_default_factory: bool = ...) -> Any:
        ...

    def get_default(self, *, call_default_factory: bool = False, validated_data: Optional[Dict[str, Any]] = None) -> Any:
        if self.default_factory is None:
            return _utils.smart_deepcopy(self.default)
        elif call_default_factory:
            if self.default_factory_takes_validated_data:
                fac = cast(Callable[[Dict[str, Any]], Any], self.default_factory)
                if validated_data is None:
                    raise ValueError("The default factory requires the 'validated_data' argument, which was not provided when calling 'get_default'.")
                return fac(validated_data)
            else:
                fac = cast(Callable[[], Any], self.default_factory)
                return fac()
        else:
            return None

    def is_required(self) -> bool:
        return self.default is PydanticUndefined and self.default_factory is None

    def rebuild_annotation(self) -> Any:
        if not self.metadata:
            return self.annotation
        else:
            return Annotated[self.annotation, *self.metadata]

    def apply_typevars_map(
        self,
        typevars_map: Mapping[Any, Any],
        globalns: Optional[Mapping[str, Any]] = None,
        localns: Optional[Mapping[str, Any]] = None
    ) -> None:
        annotation, _ = _typing_extra.try_eval_type(self.annotation, globalns, localns)
        self.annotation = _generics.replace_types(annotation, typevars_map)

    def __repr_args__(self) -> Iterator[Tuple[str, Any]]:
        yield ('annotation', _repr.PlainRepr(_repr.display_as_type(self.annotation)))
        yield ('required', self.is_required())
        for s in self.__slots__:
            if s in ('annotation', '_attributes_set', '_complete', '_original_assignment', '_original_annotation'):
                continue
            elif s == 'metadata' and (not self.metadata):
                continue
            elif s == 'repr' and self.repr is True:
                continue
            if s == 'frozen' and self.frozen is False:
                continue
            if s == 'validation_alias' and self.validation_alias == self.alias:
                continue
            if s == 'serialization_alias' and self.serialization_alias == self.alias:
                continue
            if s == 'default' and self.default is not PydanticUndefined:
                yield ('default', self.default)
            elif s == 'default_factory' and self.default_factory is not None:
                yield ('default_factory', _repr.PlainRepr(_repr.display_as_type(self.default_factory)))
            else:
                value = getattr(self, s)
                if value is not None and value is not PydanticUndefined:
                    yield (s, value)


_Field_ARG_NAMES: set = set(inspect.signature(Field).parameters)
_Field_ARG_NAMES.remove('extra')


class ModelPrivateAttr(_repr.Representation):
    __slots__ = ('default', 'default_factory')

    def __init__(self, default: Any = PydanticUndefined, *, default_factory: Optional[Callable[[], Any]] = None) -> None:
        if default is Ellipsis:
            self.default = PydanticUndefined
        else:
            self.default = default
        self.default_factory = default_factory

    if not typing.TYPE_CHECKING:
        def __getattr__(self, item: str) -> Any:
            if item in {'__get__', '__set__', '__delete__'}:
                if hasattr(self.default, item):
                    return getattr(self.default, item)
            raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')

    def __set_name__(self, cls: Any, name: str) -> None:
        default = self.default
        if default is PydanticUndefined:
            return
        set_name = getattr(default, '__set_name__', None)
        if callable(set_name):
            set_name(cls, name)

    def get_default(self) -> Any:
        return _utils.smart_deepcopy(self.default) if self.default_factory is None else self.default_factory()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and (self.default, self.default_factory) == (other.default, other.default_factory)


@overload
def PrivateAttr(default: Any, *, init: bool = False) -> ModelPrivateAttr:
    ...


@overload
def PrivateAttr(*, default_factory: Callable[[], Any], init: bool = False) -> ModelPrivateAttr:
    ...


@overload
def PrivateAttr(*, init: bool = False) -> ModelPrivateAttr:
    ...


def PrivateAttr(default: Any = PydanticUndefined, *, default_factory: Optional[Callable[[], Any]] = None, init: bool = False) -> ModelPrivateAttr:
    if default is not PydanticUndefined and default_factory is not None:
        raise TypeError('cannot specify both default and default_factory')
    return ModelPrivateAttr(default, default_factory=default_factory)


@dataclasses.dataclass(**_internal_dataclass.slots_true)
class ComputedFieldInfo:
    decorator_repr: ClassVar[str] = '@computed_field'
    wrapped_property: Any
    return_type: Any
    alias: Optional[Any]
    alias_priority: Optional[Any]
    title: Optional[Any]
    field_title_generator: Optional[Callable[[str], Any]]
    description: Optional[str]
    deprecated: Optional[Any]
    examples: Optional[Any]
    json_schema_extra: Optional[Any]
    repr: bool

    @property
    def deprecation_message(self) -> Optional[str]:
        if self.deprecated is None:
            return None
        if isinstance(self.deprecated, bool):
            return 'deprecated' if self.deprecated else None
        return self.deprecated if isinstance(self.deprecated, str) else self.deprecated.message


def _wrapped_property_is_private(property_: Any) -> bool:
    wrapped_name = ''
    if isinstance(property_, property):
        wrapped_name = getattr(property_.fget, '__name__', '')
    elif isinstance(property_, cached_property):
        wrapped_name = getattr(property_.func, '__name__', '')
    return wrapped_name.startswith('_') and (not wrapped_name.startswith('__'))


PropertyT = TypeVar('PropertyT')


@overload
def computed_field(func: Callable[..., Any], /) -> Any:
    ...


@overload
def computed_field(*, alias: Optional[str] = None, alias_priority: Optional[int] = None, title: Optional[str] = None, field_title_generator: Optional[Callable[[str], Any]] = None, description: Optional[str] = None, deprecated: Optional[Any] = None, examples: Optional[Any] = None, json_schema_extra: Optional[Any] = None, repr: bool = True, return_type: Any = PydanticUndefined) -> Any:
    ...


def computed_field(
    func: Optional[Callable[..., Any]] = None,
    /,
    *,
    alias: Optional[str] = None,
    alias_priority: Optional[int] = None,
    title: Optional[str] = None,
    field_title_generator: Optional[Callable[[str], Any]] = None,
    description: Optional[str] = None,
    deprecated: Optional[Any] = None,
    examples: Optional[Any] = None,
    json_schema_extra: Optional[Any] = None,
    repr: Optional[bool] = None,
    return_type: Any = PydanticUndefined
) -> Any:
    def dec(f: Callable[..., Any]) -> Any:
        nonlocal description, deprecated, return_type, alias_priority
        unwrapped = _decorators.unwrap_wrapped_function(f)
        if description is None and unwrapped.__doc__:
            description = inspect.cleandoc(unwrapped.__doc__)
        if deprecated is None and hasattr(unwrapped, '__deprecated__'):
            deprecated = unwrapped.__deprecated__
        f = _decorators.ensure_property(f)
        alias_priority = alias_priority or 2 if alias is not None else None
        repr_ = not _wrapped_property_is_private(f) if repr is None else repr
        dec_info = ComputedFieldInfo(
            wrapped_property=f,
            return_type=return_type,
            alias=alias,
            alias_priority=alias_priority,
            title=title,
            field_title_generator=field_title_generator,
            description=description,
            deprecated=deprecated,
            examples=examples,
            json_schema_extra=json_schema_extra,
            repr=repr_
        )
        return _decorators.PydanticDescriptorProxy(f, dec_info)
    if func is None:
        return dec
    else:
        return dec(func)