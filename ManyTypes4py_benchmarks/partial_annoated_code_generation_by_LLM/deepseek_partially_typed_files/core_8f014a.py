from __future__ import annotations
import hashlib
import html
import inspect
import sys
import uuid
import warnings
from abc import ABC
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, Any, ClassVar, Coroutine, FrozenSet, Optional, TypeVar, Union, get_origin
from uuid import UUID, uuid4
from griffe import Docstring, DocstringSection, DocstringSectionKind, Parser, parse
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, HttpUrl, PrivateAttr, SecretBytes, SecretStr, SerializationInfo, SerializerFunctionWrapHandler, ValidationError, model_serializer
from pydantic.json_schema import GenerateJsonSchema
from typing_extensions import Literal, ParamSpec, Self, TypeGuard, get_args
import prefect.exceptions
from prefect._internal.compatibility.async_dispatch import async_dispatch
from prefect.client.schemas import DEFAULT_BLOCK_SCHEMA_VERSION, BlockDocument, BlockSchema, BlockType, BlockTypeUpdate
from prefect.client.schemas.actions import BlockDocumentCreate, BlockDocumentUpdate, BlockSchemaCreate, BlockTypeCreate
from prefect.client.utilities import inject_client
from prefect.events import emit_event
from prefect.logging.loggers import disable_logger
from prefect.plugins import load_prefect_collections
from prefect.types import SecretDict
from prefect.utilities.asyncutils import run_coro_as_sync, sync_compatible
from prefect.utilities.collections import listrepr, remove_nested_keys, visit_collection
from prefect.utilities.dispatch import lookup_type, register_base_type
from prefect.utilities.hashing import hash_objects
from prefect.utilities.importtools import to_qualified_name
from prefect.utilities.pydantic import handle_secret_render
from prefect.utilities.slugify import slugify
if TYPE_CHECKING:
    from pydantic.main import IncEx
    from prefect.client.orchestration import PrefectClient, SyncPrefectClient
R = TypeVar('R')
P = ParamSpec('P')
ResourceTuple = tuple[dict[str, Any], list[dict[str, Any]]]

def block_schema_to_key(schema: BlockSchema) -> str:
    """
    Defines the unique key used to lookup the Block class for a given schema.
    """
    if schema.block_type is None:
        raise ValueError('Block type is not set')
    return f'{schema.block_type.slug}'

class InvalidBlockRegistration(Exception):
    """
    Raised on attempted registration of the base Block
    class or a Block interface class
    """

def _collect_nested_reference_strings(obj: Any) -> list[str]:
    """
    Collects all nested reference strings (e.g. #/definitions/Model) from a given object.
    """
    found_reference_strings: list[str] = []
    if isinstance(obj, dict):
        if (ref := obj.get('$ref')):
            found_reference_strings.append(ref)
        for value in obj.values():
            found_reference_strings.extend(_collect_nested_reference_strings(value))
    if isinstance(obj, list):
        for item in obj:
            found_reference_strings.extend(_collect_nested_reference_strings(item))
    return found_reference_strings

def _get_non_block_reference_definitions(object_definition: dict[str, Any], definitions: dict[str, Any]) -> dict[str, Any]:
    """
    Given a definition of an object in a block schema OpenAPI spec and the dictionary
    of all reference definitions in that same block schema OpenAPI spec, return the
    definitions for objects that are referenced from the object or any children of
    the object that do not reference a block.
    """
    non_block_definitions: dict[str, Any] = {}
    reference_strings = _collect_nested_reference_strings(object_definition)
    for reference_string in reference_strings:
        if isinstance(reference_string, str):
            definition_key = reference_string.replace('#/definitions/', '')
            definition = definitions.get(definition_key)
            if definition and definition.get('block_type_slug') is None:
                non_block_definitions = {**non_block_definitions, definition_key: definition, **_get_non_block_reference_definitions(definition, definitions)}
    return non_block_definitions

def _is_subclass(cls: Any, parent_cls: type) -> bool:
    """
    Checks if a given class is a subclass of another class. Unlike issubclass,
    this will not throw an exception if cls is an instance instead of a type.
    """
    return inspect.isclass(cls) and (not get_origin(cls)) and issubclass(cls, parent_cls)

def _collect_secret_fields(name: str, type_: Any, secrets: list[str]) -> None:
    """
    Recursively collects all secret fields from a given type and adds them to the
    secrets list, supporting nested Union / Dict / Tuple / List / BaseModel fields.
    Also, note, this function mutates the input secrets list, thus does not return anything.
    """
    if get_origin(type_) in (Union, dict, list, tuple):
        for nested_type in get_args(type_):
            _collect_secret_fields(name, nested_type, secrets)
        return
    elif _is_subclass(type_, BaseModel):
        for (field_name, field) in type_.model_fields.items():
            if field.annotation is not None:
                _collect_secret_fields(f'{name}.{field_name}', field.annotation, secrets)
        return
    if type_ in (SecretStr, SecretBytes) or (isinstance(type_, type) and getattr(type_, '__module__', None) == 'pydantic.types' and (getattr(type_, '__name__', None) == 'Secret')):
        secrets.append(name)
    elif type_ == SecretDict:
        secrets.append(f'{name}.*')
    elif Block.is_block_class(type_):
        secrets.extend((f'{name}.{s}' for s in type_.model_json_schema()['secret_fields']))

def _should_update_block_type(local_block_type: BlockType, server_block_type: BlockType) -> bool:
    """
    Compares the fields of `local_block_type` and `server_block_type`.
    Only compare the possible updatable fields as defined by `BlockTypeUpdate.updatable_fields`
    Returns True if they are different, otherwise False.
    """
    fields = BlockTypeUpdate.updatable_fields()
    local_block_fields = local_block_type.model_dump(include=fields, exclude_unset=True)
    server_block_fields = server_block_type.model_dump(include=fields, exclude_unset=True)
    if local_block_fields.get('description') is not None:
        local_block_fields['description'] = html.unescape(local_block_fields['description'])
    if local_block_fields.get('code_example') is not None:
        local_block_fields['code_example'] = html.unescape(local_block_fields['code_example'])
    if server_block_fields.get('description') is not None:
        server_block_fields['description'] = html.unescape(server_block_fields['description'])
    if server_block_fields.get('code_example') is not None:
        server_block_fields['code_example'] = html.unescape(server_block_fields['code_example'])
    return server_block_fields != local_block_fields

class BlockNotSavedError(RuntimeError):
    """
    Raised when a given block is not saved and an operation that requires
    the block to be saved is attempted.
    """
    pass

def schema_extra(schema: dict[str, Any], model: type[BaseModel]) -> None:
    """
    Customizes Pydantic's schema generation feature to add blocks related information.
    """
    schema['block_type_slug'] = model.get_block_type_slug()
    description = model.get_description()
    if description:
        schema['description'] = description
    else:
        schema.pop('description', None)
    secrets: list[str] = []
    for (name, field) in model.model_fields.items():
        if field.annotation is not None:
            _collect_secret_fields(name, field.annotation, secrets)
    schema['secret_fields'] = secrets
    refs: dict[str, Any] = {}

    def collect_block_schema_references(field_name: str, annotation: type) -> None:
        """Walk through the annotation and collect block schemas for any nested blocks."""
        if Block.is_block_class(annotation):
            if isinstance(refs.get(field_name), list):
                refs[field_name].append(annotation._to_block_schema_reference_dict())
            elif isinstance(refs.get(field_name), dict):
                refs[field_name] = [refs[field_name], annotation._to_block_schema_reference_dict()]
            else:
                refs[field_name] = annotation._to_block_schema_reference_dict()
        if get_origin(annotation) in (Union, list, tuple, dict):
            for type_ in get_args(annotation):
                collect_block_schema_references(field_name, type_)
    for (name, field) in model.model_fields.items():
        if field.annotation is not None:
            collect_block_schema_references(name, field.annotation)
    schema['block_schema_references'] = refs

@register_base_type
class Block(BaseModel, ABC):
    """
    A base class for implementing a block that wraps an external service.

    This class can be defined with an arbitrary set of fields and methods, and
    couples business logic with data contained in an block document.
    `_block_document_name`, `_block_document_id`, `_block_schema_id`, and
    `_block_type_id` are reserved by Prefect as Block metadata fields, but
    otherwise a Block can implement arbitrary logic. Blocks can be instantiated
    without populating these metadata fields, but can only be used interactively,
    not with the Prefect API.

    Instead of the __init__ method, a block implementation allows the
    definition of a `block_initialization` method that is called after
    initialization.
    """
    model_config = ConfigDict(extra='allow', json_schema_extra=schema_extra)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.block_initialization()

    def __str__(self) -> str:
        return self.__repr__()

    def __repr_args__(self) -> list[tuple[str | None, Any]]:
        repr_args = super().__repr_args__()
        data_keys = self.model_json_schema()['properties'].keys()
        return [(key, value) for (key, value) in repr_args if key is None or key in data_keys]

    def block_initialization(self) -> None:
        pass
    _block_type_name: ClassVar[Optional[str]] = None
    _block_type_slug: ClassVar[Optional[str]] = None
    _logo_url: ClassVar[Optional[HttpUrl]] = None
    _documentation_url: ClassVar[Optional[HttpUrl]] = None
    _description: ClassVar[Optional[str]] = None
    _code_example: ClassVar[Optional[str]] = None
    _block_type_id: ClassVar[Optional[UUID]] = None
    _block_schema_id: ClassVar[Optional[UUID]] = None
    _block_schema_capabilities: ClassVar[Optional[list[str]]] = None
    _block_schema_version: ClassVar[Optional[str]] = None
    _block_document_id: Optional[UUID] = PrivateAttr(None)
    _block_document_name: Optional[str] = PrivateAttr(None)
    _is_anonymous: Optional[bool] = PrivateAttr(None)
    _events_excluded_methods: ClassVar[list[str]] = PrivateAttr(default=['block_initialization', 'save', 'dict'])

    @classmethod
    def __dispatch_key__(cls) -> str | None:
        if cls.__name__ == 'Block':
            return None
        return block_schema_to_key(cls._to_block_schema())

    @model_serializer(mode='wrap')
    def ser_model(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) -> Any:
        jsonable_self = handler(self)
        if (ctx := info.context) and ctx.get('include_secrets') is True:
            jsonable_self.update({field_name: visit_collection(expr=getattr(self, field_name), visit_fn=partial(handle_secret_render, context=ctx), return_data=True) for field_name in self.model_fields})
        if (extra_fields := {'block_type_slug': self.get_block_type_slug(), '_block_document_id': self._block_document_id, '_block_document_name': self._block_document_name, '_is_anonymous': self._is_anonymous}):
            jsonable_self |= {key: value for (key, value) in extra_fields.items() if value is not None}
        return jsonable_self

    @classmethod
    def get_block_type_name(cls) -> str:
        return cls._block_type_name or cls.__name__

    @classmethod
    def get_block_type_slug(cls) -> str:
        return slugify(cls._block_type_slug or cls.get_block_type_name())

    @classmethod
    def get_block_capabilities(cls) -> FrozenSet[str]:
        """
        Returns the block capabilities for this Block. Recursively collects all block
        capabilities of all parent classes into a single frozenset.
        """
        return frozenset({c for base in (cls,) + cls.__mro__ for c in getattr(base, '_block_schema_capabilities', []) or []})

    @classmethod
    def _get_current_package_version(cls) -> str:
        current_module = inspect.getmodule(cls)
        if current_module:
            top_level_module = sys.modules[current_module.__name__.split('.')[0] or '__main__']
            try:
                version = Version(top_level_module.__version__)
                return version.base_version
            except (AttributeError, InvalidVersion):
                pass
        return DEFAULT_BLOCK_SCHEMA_VERSION

    @classmethod
    def get_block_schema_version(cls) -> str:
        return cls._block_schema_version or cls._get_current_package_version()

    @classmethod
    def _to_block_schema_reference_dict(cls) -> dict[str, Any]:
        return dict(block_type_slug=cls.get_block_type_slug(), block_schema_checksum=cls._calculate_schema_checksum())

    @classmethod
    def _calculate_schema_checksum(cls, block_schema_fields: dict[str, Any] | None=None) -> str:
        """
        Generates a unique hash for the underlying schema of block.

        Args:
            block_schema_fields: Dictionary detailing block schema fields to generate a
                checksum for. The fields of the current class is used if this parameter
                is not provided.

        Returns:
            str: The calculated checksum prefixed with the hashing algorithm used.
        """
        block_schema_fields = cls.model_json_schema() if block_schema_fields is None else block_schema_fields
        fields_for_checksum = remove_nested_keys(['secret_fields'], block_schema_fields)
        if fields_for_checksum.get('definitions'):
            non_block_definitions = _get_non_block_reference_definitions(fields_for_checksum, fields_for_checksum['definitions'])
            if non_block_definitions:
                fields_for_checksum['definitions'] = non_block_definitions
            else:
                fields_for_checksum.pop('definitions')
        checksum = hash_objects(fields_for_checksum, hash_algo=hashlib.sha256)
        if checksum is None:
            raise ValueError('Unable to compute checksum for block schema')
        else:
            return f'sha256:{checksum}'

    def _to_block_document(self, name: Optional[str]=None, block_schema_id: Optional[UUID]=None, block_type_id: Optional[UUID]=None, is_anonymous: Optional[bool]=None, include_secrets: bool=False) -> BlockDocument:
        """
        Creates the corresponding block document based on the data stored in a block.
        The corresponding block document name, block type ID, and block schema ID must
        either be passed into the method or configured on the block.

        Args:
            name: The name of the created block document. Not required if anonymous.
            block_schema_id: UUID of the corresponding block schema.
            block_type_id: UUID of the corresponding block type.
            is_anonymous: if True, an anonymous block is created. Anonymous
                blocks are not displayed in the UI and used primarily for system
                operations and features that need to automatically generate blocks.

        Returns:
            BlockDocument: Corresponding block document
                populated with the block's configured data.
        """
        if is_anonymous is None:
            is_anonymous = self._is_anonymous or False
        if not is_anonymous and (not name) and (not self._block_document_name):
            raise ValueError('No name provided, either as an argument or on the block.')
        if not block_schema_id and (not self._block_schema_id):
            raise ValueError('No block schema ID provided, either as an argument or on the block.')
        if not block_type_id and (not self._block_type_id):
            raise ValueError('No block type ID provided, either as an argument or on the block.')
        data_keys = self.model_json_schema(by_alias=False)['properties'].keys()
        block_document_data = self.model_dump(by_alias=True, include=data_keys, context={'include_secrets': include_secrets})
        for key in data_keys:
            field_value = getattr(self, key)
            if isinstance(field_value, Block) and field_value._block_document_id is not None:
                block_document_data[key] = {'$ref': {'block_document_id': field_value._block_document_id}}
        block_schema_id = block_schema_id or self._block_schema_id
        block_type_id = block_type_id or self._block_type_id
        if block_schema_id is None:
            raise ValueError('No block schema ID provided, either as an argument or on the block.')
        if block_type_id is None:
            raise ValueError('No block type ID provided, either as an argument or on the block.')
        return BlockDocument(id=self._block_document_id or uuid4(), name=name or self._block_document_name if not is_anonymous else None, block_schema_id=block_schema_id, block_type_id=block_type_id, block_type_name=self._block_type_name, data=block_document_data, block_schema