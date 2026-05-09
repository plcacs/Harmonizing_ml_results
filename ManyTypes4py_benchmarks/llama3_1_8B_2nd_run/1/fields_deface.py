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
    """This class holds information about a field.

    `FieldInfo` is used for any field definition regardless of whether the [`Field()`][pydantic.fields.Field]
    function is explicitly used.

    !!! warning
        You generally shouldn't be creating `FieldInfo` directly, you'll only need to use it when accessing
        [`BaseModel`][pydantic.main.BaseModel] `.model_fields` internals.

    Attributes:
        annotation: The type annotation of the field.
        default: The default value of the field.
        default_factory: A callable to generate the default value. The callable can either take 0 arguments
            (in which case it is called as is) or a single argument containing the already validated data.
        alias: The alias name of the field.
        alias_priority: The priority of the field's alias.
        validation_alias: The validation alias of the field.
        serialization_alias: The serialization alias of the field.
        title: The title of the field.
        field_title_generator: A callable that takes a field name and returns title for it.
        description: The description of the field.
        examples: List of examples of the field.
        exclude: Whether to exclude the field from the model serialization.
        discriminator: Field name or Discriminator for discriminating the type in a tagged union.
        deprecated: A deprecation message, an instance of `warnings.deprecated` or the `typing_extensions.deprecated` backport,
            or a boolean. If `True`, a default deprecation message will be emitted when accessing the field.
        json_schema_extra: A dict or callable to provide extra JSON schema properties.
        frozen: Whether the field is frozen.
        validate_default: Whether to validate the default value of the field.
        repr: Whether to include the field in representation of the model.
        init: Whether the field should be included in the constructor of the dataclass.
        init_var: Whether the field should _only_ be included in the constructor of the dataclass, and not stored.
        kw_only: Whether the field should be a keyword-only argument in the constructor of the dataclass.
        metadata: List of metadata constraints.
    """
    __slots__ = ('annotation', 'default', 'default_factory', 'alias', 'alias_priority', 'validation_alias', 'serialization_alias', 'title', 'field_title_generator', 'description', 'examples', 'exclude', 'discriminator', 'deprecated', 'json_schema_extra', 'frozen', 'validate_default', 'repr', 'init', 'init_var', 'kw_only', 'metadata', '_attributes_set', '_complete', '_original_assignment', '_original_annotation')
    metadata_lookup: typing.Mapping[str, typing.Union[type, None]] = {'strict': types.Strict, 'gt': annotated_types.Gt, 'ge': annotated_types.Ge, 'lt': annotated_types.Lt, 'le': annotated_types.Le, 'multiple_of': annotated_types.MultipleOf, 'min_length': annotated_types.MinLen, 'max_length': annotated_types.MaxLen, 'pattern': None, 'allow_inf_nan': None, 'max_digits': None, 'decimal_places': None, 'union_mode': None, 'coerce_numbers_to_str': None, 'fail_fast': types.FailFast}

    def __init__(self, **kwargs: typing.Any) -> None:
        """This class should generally not be initialized directly; instead, use the `pydantic.fields.Field` function
        or one of the constructor classmethods.

        See the signature of `pydantic.fields.Field` for more details about the expected arguments.
        """
        self._attributes_set = {k: v for k, v in kwargs.items() if v is not _Unset}
        kwargs = {k: _DefaultValues.get(k) if v is _Unset else v for k, v in kwargs.items()}
        self.annotation = kwargs.get('annotation')
        default = kwargs.pop('default', PydanticUndefined)
        if default is Ellipsis:
            self.default = PydanticUndefined
            self._attributes_set.pop('default', None)
        else:
            self.default = default
        self.default_factory = kwargs.pop('default_factory', None)
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
    def from_field(default: typing.Any = PydanticUndefined, **kwargs: typing.Any) -> 'FieldInfo':
        """Create a new `FieldInfo` object with the `Field` function.

        Args:
            default: The default value for the field. Defaults to Undefined.
            **kwargs: Additional arguments dictionary.

        Raises:
            TypeError: If 'annotation' is passed as a keyword argument.

        Returns:
            A new FieldInfo object with the given parameters.

        Example:
            This is how you can create a field with default value like this:

            