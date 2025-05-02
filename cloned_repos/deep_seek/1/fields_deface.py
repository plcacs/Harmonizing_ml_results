"""Defining fields on models."""
from __future__ import annotations as _annotations
import dataclasses
import inspect
import sys
import typing
from collections.abc import Mapping
from copy import copy
from dataclasses import Field as DataclassField
from functools import cached_property
from typing import Annotated, Any, Callable, ClassVar, Literal, TypeVar, cast, overload
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
_Unset = PydanticUndefined
if sys.version_info >= (3, 13):
    import warnings
    Deprecated = warnings.deprecated | deprecated
else:
    Deprecated = deprecated

class _FromFieldInfoInputs(typing_extensions.TypedDict, total=False):
    """This class exists solely to add type checking for the `**kwargs` in `FieldInfo.from_field`."""

class _FieldInfoInputs(_FromFieldInfoInputs, total=False):
    """This class exists solely to add type checking for the `**kwargs` in `FieldInfo.__init__`."""

class FieldInfo(_repr.Representation):
    """This class holds information about a field."""
    __slots__ = ('annotation', 'default', 'default_factory', 'alias', 'alias_priority', 'validation_alias', 'serialization_alias', 'title', 'field_title_generator', 'description', 'examples', 'exclude', 'discriminator', 'deprecated', 'json_schema_extra', 'frozen', 'validate_default', 'repr', 'init', 'init_var', 'kw_only', 'metadata', '_attributes_set', '_complete', '_original_assignment', '_original_annotation')
    metadata_lookup: ClassVar[dict[str, type | None]] = {'strict': types.Strict, 'gt': annotated_types.Gt, 'ge': annotated_types.Ge, 'lt': annotated_types.Lt, 'le': annotated_types.Le, 'multiple_of': annotated_types.MultipleOf, 'min_length': annotated_types.MinLen, 'max_length': annotated_types.MaxLen, 'pattern': None, 'allow_inf_nan': None, 'max_digits': None, 'decimal_places': None, 'union_mode': None, 'coerce_numbers_to_str': None, 'fail_fast': types.FailFast}

    def __init__(self, **kwargs: Unpack[_FieldInfoInputs]) -> None:
        self._attributes_set = {k: v for k, v in kwargs.items() if v is not _Unset}
        kwargs = {k: _DefaultValues.get(k) if v is _Unset else v for k, v in kwargs.items()}
        self.annotation: Any = kwargs.get('annotation')
        default: Any = kwargs.pop('default', PydanticUndefined)
        if default is Ellipsis:
            self.default = PydanticUndefined
            self._attributes_set.pop('default', None)
        else:
            self.default = default
        self.default_factory: Callable[..., Any] | None = kwargs.pop('default_factory', None)
        if self.default is not PydanticUndefined and self.default_factory is not None:
            raise TypeError('cannot specify both default and default_factory')
        self.alias: str | None = kwargs.pop('alias', None)
        self.validation_alias: str | AliasChoices | AliasPath | None = kwargs.pop('validation_alias', None)
        self.serialization_alias: str | None = kwargs.pop('serialization_alias', None)
        alias_is_set = any((alias is not None for alias in (self.alias, self.validation_alias, self.serialization_alias)))
        self.alias_priority: int | None = kwargs.pop('alias_priority', None) or 2 if alias_is_set else None
        self.title: str | None = kwargs.pop('title', None)
        self.field_title_generator: Callable[[str], str] | None = kwargs.pop('field_title_generator', None)
        self.description: str | None = kwargs.pop('description', None)
        self.examples: list[Any] | None = kwargs.pop('examples', None)
        self.exclude: bool | None = kwargs.pop('exclude', None)
        self.discriminator: str | None = kwargs.pop('discriminator', None)
        self.deprecated: str | bool | Deprecated | None = kwargs.pop('deprecated', getattr(self, 'deprecated', None))
        self.repr: bool = kwargs.pop('repr', True)
        self.json_schema_extra: dict[str, Any] | Callable[[dict[str, Any]], None] | None = kwargs.pop('json_schema_extra', None)
        self.validate_default: bool | None = kwargs.pop('validate_default', None)
        self.frozen: bool | None = kwargs.pop('frozen', None)
        self.init: bool | None = kwargs.pop('init', None)
        self.init_var: bool | None = kwargs.pop('init_var', None)
        self.kw_only: bool | None = kwargs.pop('kw_only', None)
        self.metadata: list[Any] = self._collect_metadata(kwargs)
        self._complete: bool = True
        self._original_annotation: Any = PydanticUndefined
        self._original_assignment: Any = PydanticUndefined

    @staticmethod
    def from_field(default: Any = PydanticUndefined, **kwargs: Unpack[_FromFieldInfoInputs]) -> FieldInfo:
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
                field_metadata = []
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
            raise PydanticUserError("Error when building FieldInfo from annotated attribute. Make sure you don't have any field name clashing with a type annotation.", code='unevaluable-type-annotation')
        final = _typing_extra.is_finalvar(annotation)
        if final and _typing_extra.is_generic_alias(annotation):
            annotation = typing_extensions.get_args(annotation)[0]
        if isinstance(default, FieldInfo):
            default.annotation, annotation_metadata = _typing_extra.unpack_annotated(annotation)
            default.metadata += annotation_metadata
            default = default.merge_field_infos(*[x for x in annotation_metadata if isinstance(x, FieldInfo)], default, annotation=default.annotation)
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
            pydantic_field = pydantic_field.merge_field_infos(*[x for x in annotation_metadata if isinstance(x, FieldInfo)], pydantic_field, annotation=pydantic_field.annotation)
            pydantic_field.frozen = final or pydantic_field.frozen
            pydantic_field.init_var = init_var
            pydantic_field.init = getattr(default, 'init', None)
            pydantic_field.kw_only = getattr(default, 'kw_only', None)
            return pydantic_field
        annotation, metadata = _typing_extra.unpack_annotated(annotation)
        if metadata:
            field_infos = [a for a in metadata if isinstance(a, FieldInfo)]
            field_info = FieldInfo.merge_field_infos(*field_infos, annotation=annotation, default=default)
            field_metadata = []
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
        merged_field_info_kwargs: dict[str, Any] = {}
        metadata: dict[type, Any] = {}
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
                        warn("Composing `dict` and `callable` type `json_schema_extra` is not supported.The `callable` type is being ignored.If you'd like support for this behavior, please open an issue on pydantic.", PydanticJsonSchemaWarning)
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
    def _from_dataclass_field(dc_field: DataclassField[Any]) -> FieldInfo:
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
    def _collect_metadata(kwargs: dict[str, Any]) -> list[Any]:
        metadata = []
        general_metadata: dict[str, Any] = {}
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
    def deprecation_message(self) -> str | None:
        if self.deprecated is None:
            return None
        if isinstance(self.deprecated, bool):
            return 'deprecated' if self.deprecated else None
        return self.deprecated if isinstance(self.deprecated, str) else self.deprecated.message

    @property
    def default_factory_takes_validated_data(self) -> bool | None:
        if self.default_factory is not None:
            return _fields.takes_validated_data_argument(self.default_factory)

    @overload
    def get_default(self, *, call_default_factory: bool, validated_data: dict[str, Any] | None = ...) -> Any:
        ...

    @overload
    def get_default(self, *, call_default_factory: bool = ...) -> Any:
        ...

    def get_default(self, *, call_default_factory: bool = False, validated_data: dict[str, Any] | None = None) -> Any:
        if self.default_factory is None:
            return _utils.smart_deepcopy(self.default)
        elif call_default_factory:
            if self.default_factory_takes_validated_data:
                fac = cast('Callable[[dict[str, Any]], Any]', self.default_factory)
                if validated_data is None:
                    raise ValueError("The default factory requires the 'validated_data' argument, which was not provided when calling 'get_default'.")
                return fac(validated_data)
            else:
                fac = cast('Callable[[], Any]', self.default_factory)
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

    def apply_typevars_map(self, typevars_map: dict[TypeVar, Any], globalns: dict[str, Any] | None = None, localns: dict[str, Any] | None = None) -> None:
        annotation, _ = _typing_extra.try_eval_type(self.annotation, globalns, localns)
        self.annotation = _generics.replace_types(annotation, typevars_map)

    def __repr_args__(self) -> list[tuple[str, Any]]:
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

class _EmptyKwargs(typing_extensions.TypedDict):
    """This class exists solely to ensure that type checking warns about passing `**extra` in `Field`."""
_DefaultValues: dict[str, Any] = {'default': ..., 'default_factory': None, 'alias': None, 'alias_priority': None, 'validation_alias': None, 'serialization_alias': None, 'title': None, 'description': None, 'examples': None, 'exclude': None, 'discriminator': None, 'json_schema_extra': None, 'frozen': None, 'validate