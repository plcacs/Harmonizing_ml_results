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
from typing import (
    Any,
    Callable,
    ClassVar,
    Coroutine,
    Dict,
    FrozenSet,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)
from uuid import UUID, uuid4

from griffe import Docstring, DocstringSection, DocstringSectionKind, Parser, parse
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, HttpUrl, PrivateAttr, SecretBytes, SecretStr, SerializationInfo, SerializerFunctionWrapHandler, ValidationError, model_serializer
from pydantic.json_schema import GenerateJsonSchema
from typing_extensions import Literal, ParamSpec, Self, TypeGuard, get_args as te_get_args

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

R = TypeVar("R")
P = ParamSpec("P")
ResourceTuple = Tuple[Dict[str, Any], List[Dict[str, Any]]]


def block_schema_to_key(schema: BlockSchema) -> str:
    """
    Defines the unique key used to lookup the Block class for a given schema.
    """
    if schema.block_type is None:
        raise ValueError("Block type is not set")
    return f"{schema.block_type.slug}"


class InvalidBlockRegistration(Exception):
    """
    Raised on attempted registration of the base Block
    class or a Block interface class
    """
    pass


def _collect_nested_reference_strings(obj: Any) -> List[str]:
    """
    Collects all nested reference strings (e.g. #/definitions/Model) from a given object.
    """
    found_reference_strings: List[str] = []
    if isinstance(obj, dict):
        if (ref := obj.get("$ref")):
            found_reference_strings.append(ref)
        for value in obj.values():
            found_reference_strings.extend(_collect_nested_reference_strings(value))
    if isinstance(obj, list):
        for item in obj:
            found_reference_strings.extend(_collect_nested_reference_strings(item))
    return found_reference_strings


def _get_non_block_reference_definitions(
    object_definition: Dict[str, Any],
    definitions: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Given a definition of an object in a block schema OpenAPI spec and the dictionary
    of all reference definitions in that same block schema OpenAPI spec, return the
    definitions for objects that are referenced from the object or any children of
    the object that do not reference a block.
    """
    non_block_definitions: Dict[str, Any] = {}
    reference_strings = _collect_nested_reference_strings(object_definition)
    for reference_string in reference_strings:
        if isinstance(reference_string, str):
            definition_key = reference_string.replace("#/definitions/", "")
            definition = definitions.get(definition_key)
            if definition and definition.get("block_type_slug") is None:
                non_block_definitions = {
                    **non_block_definitions,
                    definition_key: definition,
                    **_get_non_block_reference_definitions(definition, definitions),
                }
    return non_block_definitions


def _is_subclass(cls: Any, parent_cls: type) -> bool:
    """
    Checks if a given class is a subclass of another class. Unlike issubclass,
    this will not throw an exception if cls is an instance instead of a type.
    """
    return inspect.isclass(cls) and (not get_origin(cls)) and issubclass(cls, parent_cls)


def _collect_secret_fields(name: str, type_: Any, secrets: List[str]) -> None:
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
        for field_name, field in type_.__fields__.items() if hasattr(type_, "__fields__") else {}.items():
            if field.annotation is not None:
                _collect_secret_fields(f"{name}.{field_name}", field.annotation, secrets)
        # Alternatively, for pydantic v2:
        for field_name, field in getattr(type_, "model_fields", {}).items():
            if field.annotation is not None:
                _collect_secret_fields(f"{name}.{field_name}", field.annotation, secrets)
        return
    if type_ in (SecretStr, SecretBytes) or (isinstance(type_, type) and getattr(type_, "__module__", None) == "pydantic.types" and (getattr(type_, "__name__", None) == "Secret")):
        secrets.append(name)
    elif type_ == SecretDict:
        secrets.append(f"{name}.*")
    elif Block.is_block_class(type_):
        secrets.extend((f"{name}.{s}" for s in type_.model_json_schema()["secret_fields"]))


def _should_update_block_type(
    local_block_type: BlockType, server_block_type: BlockType
) -> bool:
    """
    Compares the fields of `local_block_type` and `server_block_type`.
    Only compare the possible updatable fields as defined by `BlockTypeUpdate.updatable_fields`
    Returns True if they are different, otherwise False.
    """
    fields = BlockTypeUpdate.updatable_fields()
    local_block_fields = local_block_type.model_dump(include=fields, exclude_unset=True)
    server_block_fields = server_block_type.model_dump(include=fields, exclude_unset=True)
    if local_block_fields.get("description") is not None:
        local_block_fields["description"] = html.unescape(local_block_fields["description"])
    if local_block_fields.get("code_example") is not None:
        local_block_fields["code_example"] = html.unescape(local_block_fields["code_example"])
    if server_block_fields.get("description") is not None:
        server_block_fields["description"] = html.unescape(server_block_fields["description"])
    if server_block_fields.get("code_example") is not None:
        server_block_fields["code_example"] = html.unescape(server_block_fields["code_example"])
    return server_block_fields != local_block_fields


class BlockNotSavedError(RuntimeError):
    """
    Raised when a given block is not saved and an operation that requires
    the block to be saved is attempted.
    """
    pass


def schema_extra(schema: Dict[str, Any], model: Any) -> None:
    """
    Customizes Pydantic's schema generation feature to add blocks related information.
    """
    schema["block_type_slug"] = model.get_block_type_slug()
    description = model.get_description()
    if description:
        schema["description"] = description
    else:
        schema.pop("description", None)
    secrets: List[str] = []
    for name, field in model.model_fields.items():
        if field.annotation is not None:
            _collect_secret_fields(name, field.annotation, secrets)
    schema["secret_fields"] = secrets
    refs: Dict[str, Any] = {}

    def collect_block_schema_references(field_name: str, annotation: Any) -> None:
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

    for name, field in model.model_fields.items():
        if field.annotation is not None:
            collect_block_schema_references(name, field.annotation)
    schema["block_schema_references"] = refs


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
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", json_schema_extra=schema_extra)

    # Block metadata
    _block_type_name: ClassVar[Optional[str]] = None
    _block_type_slug: ClassVar[Optional[str]] = None
    _logo_url: ClassVar[Optional[str]] = None
    _documentation_url: ClassVar[Optional[str]] = None
    _description: ClassVar[Optional[str]] = None
    _code_example: ClassVar[Optional[str]] = None
    _block_type_id: ClassVar[Optional[UUID]] = None
    _block_schema_id: ClassVar[Optional[UUID]] = None
    _block_schema_capabilities: ClassVar[Optional[List[str]]] = None

    _block_document_id: UUID = PrivateAttr(None)
    _block_document_name: Optional[str] = PrivateAttr(None)
    _is_anonymous: Optional[bool] = PrivateAttr(None)
    _events_excluded_methods: List[str] = PrivateAttr(default=["block_initialization", "save", "dict"])

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.block_initialization()

    def __str__(self) -> str:
        return self.__repr__()

    def __repr_args__(self) -> List[Tuple[Optional[str], Any]]:
        repr_args = super().__repr_args__()
        data_keys = self.model_json_schema()["properties"].keys()
        return [(key, value) for key, value in repr_args if key is None or key in data_keys]

    def block_initialization(self) -> None:
        pass

    @classmethod
    def __dispatch_key__(cls) -> Optional[str]:
        if cls.__name__ == "Block":
            return None
        return block_schema_to_key(cls._to_block_schema())

    @model_serializer(mode="wrap")
    def ser_model(self, handler: Callable[[Any], Any], info: Any) -> Any:
        jsonable_self = handler(self)
        if (ctx := info.context) and ctx.get("include_secrets") is True:
            jsonable_self.update(
                {
                    field_name: visit_collection(
                        expr=getattr(self, field_name),
                        visit_fn=partial(handle_secret_render, context=ctx),
                        return_data=True,
                    )
                    for field_name in self.model_fields
                }
            )
        extra_fields = {
            "block_type_slug": self.get_block_type_slug(),
            "_block_document_id": self._block_document_id,
            "_block_document_name": self._block_document_name,
            "_is_anonymous": self._is_anonymous,
        }
        if extra_fields:
            jsonable_self |= {key: value for key, value in extra_fields.items() if value is not None}
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
        return frozenset(
            {
                c
                for base in (cls,) + cls.__mro__
                for c in getattr(base, "_block_schema_capabilities", []) or []
            }
        )

    @classmethod
    def _get_current_package_version(cls) -> str:
        current_module = inspect.getmodule(cls)
        if current_module:
            top_level_module = sys.modules[current_module.__name__.split(".")[0] or "__main__"]
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
    def _to_block_schema_reference_dict(cls) -> Dict[str, Any]:
        return dict(
            block_type_slug=cls.get_block_type_slug(),
            block_schema_checksum=cls._calculate_schema_checksum(),
        )

    @classmethod
    def _calculate_schema_checksum(cls, block_schema_fields: Optional[Dict[str, Any]] = None) -> str:
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
        fields_for_checksum = remove_nested_keys(["secret_fields"], block_schema_fields)
        if fields_for_checksum.get("definitions"):
            non_block_definitions = _get_non_block_reference_definitions(fields_for_checksum, fields_for_checksum["definitions"])
            if non_block_definitions:
                fields_for_checksum["definitions"] = non_block_definitions
            else:
                fields_for_checksum.pop("definitions")
        checksum = hash_objects(fields_for_checksum, hash_algo=hashlib.sha256)
        if checksum is None:
            raise ValueError("Unable to compute checksum for block schema")
        else:
            return f"sha256:{checksum}"

    def _to_block_document(
        self,
        name: Optional[str] = None,
        block_schema_id: Optional[UUID] = None,
        block_type_id: Optional[UUID] = None,
        is_anonymous: Optional[bool] = None,
        include_secrets: bool = False,
    ) -> BlockDocument:
        """
        Creates the corresponding block document based on the data stored in a block.
        The corresponding block document name, block type ID, and block schema ID must
        either be passed into the method or configured on the block.
        """
        if is_anonymous is None:
            is_anonymous = self._is_anonymous or False
        if not is_anonymous and (not name) and (not self._block_document_name):
            raise ValueError("No name provided, either as an argument or on the block.")
        if not block_schema_id and (not self._block_schema_id):
            raise ValueError("No block schema ID provided, either as an argument or on the block.")
        if not block_type_id and (not self._block_type_id):
            raise ValueError("No block type ID provided, either as an argument or on the block.")
        data_keys = self.model_json_schema(by_alias=False)["properties"].keys()
        block_document_data: Dict[str, Any] = self.model_dump(
            by_alias=True, include=data_keys, context={"include_secrets": include_secrets}
        )
        for key in data_keys:
            field_value = getattr(self, key)
            if isinstance(field_value, Block) and field_value._block_document_id is not None:
                block_document_data[key] = {"$ref": {"block_document_id": field_value._block_document_id}}
        block_schema_id = block_schema_id or self._block_schema_id
        block_type_id = block_type_id or self._block_type_id
        if block_schema_id is None:
            raise ValueError("No block schema ID provided, either as an argument or on the block.")
        if block_type_id is None:
            raise ValueError("No block type ID provided, either as an argument or on the block.")
        return BlockDocument(
            id=self._block_document_id or uuid4(),
            name=name or self._block_document_name if not is_anonymous else None,
            block_schema_id=block_schema_id,
            block_type_id=block_type_id,
            block_type_name=self._block_type_name,
            data=block_document_data,
            block_schema=self._to_block_schema(block_type_id=block_type_id or self._block_type_id),
            block_type=self._to_block_type(),
            is_anonymous=is_anonymous,
        )

    @classmethod
    def _to_block_schema(cls, block_type_id: Optional[UUID] = None) -> BlockSchema:
        """
        Creates the corresponding block schema of the block.
        The corresponding block_type_id must either be passed into
        the method or configured on the block.
        """
        fields = cls.model_json_schema()
        return BlockSchema(
            id=cls._block_schema_id if cls._block_schema_id is not None else uuid4(),
            checksum=cls._calculate_schema_checksum(),
            fields=fields,
            block_type_id=block_type_id or cls._block_type_id,
            block_type=cls._to_block_type(),
            capabilities=list(cls.get_block_capabilities()),
            version=cls.get_block_schema_version(),
        )

    @classmethod
    def _parse_docstring(cls) -> List[DocstringSection]:
        """
        Parses the docstring into list of DocstringSection objects.
        Helper method used primarily to suppress irrelevant logs, e.g.
        `<module>:11: No type or annotation for parameter 'write_json'`
        because griffe is unable to parse the types from pydantic.BaseModel.
        """
        if cls.__doc__ is None:
            return []
        with disable_logger("griffe"):
            docstring = Docstring(cls.__doc__)
            parsed = parse(docstring, Parser.google)
        return parsed

    @classmethod
    def get_description(cls) -> Optional[str]:
        """
        Returns the description for the current block. Attempts to parse
        description from class docstring if an override is not defined.
        """
        description: Optional[str] = cls._description
        if description is None and cls.__doc__ is not None:
            parsed = cls._parse_docstring()
            parsed_description = next((section.as_dict().get("value") for section in parsed if section.kind == DocstringSectionKind.text), None)
            if isinstance(parsed_description, str):
                description = parsed_description.strip()
        return description

    @classmethod
    def get_code_example(cls) -> Optional[str]:
        """
        Returns the code example for the given block. Attempts to parse
        code example from the class docstring if an override is not provided.
        """
        code_example: Optional[str] = dedent(text=cls._code_example) if cls._code_example is not None else None
        if code_example is None and cls.__doc__ is not None:
            parsed = cls._parse_docstring()
            for section in parsed:
                if section.kind == DocstringSectionKind.examples:
                    code_example = "\n".join((part[1] for part in section.as_dict().get("value", [])))
                    break
                if section.kind == DocstringSectionKind.admonition:
                    value = section.as_dict().get("value", {})
                    if value.get("annotation") == "example":
                        code_example = value.get("description")
                        break
        if code_example is None:
            code_example = cls._generate_code_example()
        return code_example

    @classmethod
    def _generate_code_example(cls) -> str:
        """Generates a default code example for the current class"""
        qualified_name = to_qualified_name(cls)
        module_str = ".".join(qualified_name.split(".")[:-1])
        origin: Any = cls.__pydantic_generic_metadata__.get("origin") or cls
        class_name = origin.__name__
        block_variable_name = f'{cls.get_block_type_slug().replace("-", "_")}_block'
        return dedent(
            f"        ```python\n        from {module_str} import {class_name}\n\n        {block_variable_name} = {class_name}.load(\"BLOCK_NAME\")\n        ```"
        )

    @classmethod
    def _to_block_type(cls) -> BlockType:
        """
        Creates the corresponding block type of the block.
        """
        return BlockType(
            id=cls._block_type_id or uuid4(),
            slug=cls.get_block_type_slug(),
            name=cls.get_block_type_name(),
            logo_url=cls._logo_url,
            documentation_url=cls._documentation_url,
            description=cls.get_description(),
            code_example=cls.get_code_example(),
        )

    @classmethod
    def _from_block_document(cls, block_document: BlockDocument) -> Block:
        """
        Instantiates a block from a given block document. The corresponding block class
        will be looked up in the block registry based on the corresponding block schema
        of the provided block document.
        """
        if block_document.block_schema is None:
            raise ValueError("Unable to determine block schema for provided block document")
        block_cls: Type[Block] = cls if cls.__name__ != "Block" else cls.get_block_class_from_schema(block_document.block_schema)
        block: Block = block_cls.model_validate(block_document.data)
        block._block_document_id = block_document.id
        block.__class__._block_schema_id = block_document.block_schema_id
        block.__class__._block_type_id = block_document.block_type_id
        block._block_document_name = block_document.name
        block._is_anonymous = block_document.is_anonymous
        block._define_metadata_on_nested_blocks(block_document.block_document_references)
        resources = block._event_method_called_resources()
        if resources:
            kind = block._event_kind()
            resource, related = resources
            emit_event(event=f"{kind}.loaded", resource=resource, related=related)
        return block

    def _event_kind(self) -> str:
        return f"prefect.block.{self.get_block_type_slug()}"

    def _event_method_called_resources(self) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        if not (self._block_document_id and self._block_document_name):
            return None
        return (
            {
                "prefect.resource.id": f"prefect.block-document.{self._block_document_id}",
                "prefect.resource.name": self._block_document_name,
            },
            [
                {
                    "prefect.resource.id": f"prefect.block-type.{self.get_block_type_slug()}",
                    "prefect.resource.role": "block-type",
                }
            ],
        )

    @classmethod
    def get_block_class_from_schema(cls, schema: BlockSchema) -> Type[Block]:
        """
        Retrieve the block class implementation given a schema.
        """
        return cls.get_block_class_from_key(block_schema_to_key(schema))

    @classmethod
    def get_block_class_from_key(cls, key: str) -> Type[Block]:
        """
        Retrieve the block class implementation given a key.
        """
        load_prefect_collections()
        return cast(Type[Block], lookup_type(cls, key))

    def _define_metadata_on_nested_blocks(self, block_document_references: Dict[str, Any]) -> None:
        """
        Recursively populates metadata fields on nested blocks based on the
        provided block document references.
        """
        for field_name, block_document_reference in block_document_references.items():
            nested_block = getattr(self, field_name)
            if isinstance(nested_block, Block):
                nested_block_document_info: Dict[str, Any] = block_document_reference.get("block_document", {})
                nested_block._define_metadata_on_nested_blocks(nested_block_document_info.get("block_document_references", {}))
                nested_block_document_id = nested_block_document_info.get("id")
                nested_block._block_document_id = UUID(nested_block_document_id) if nested_block_document_id else None
                nested_block._block_document_name = nested_block_document_info.get("name")
                nested_block._is_anonymous = nested_block_document_info.get("is_anonymous")

    @classmethod
    async def _aget_block_document(cls, name: str, client: Any) -> Tuple[BlockDocument, str]:
        if cls.__name__ == "Block":
            block_type_slug, block_document_name = name.split("/", 1)
        else:
            block_type_slug = cls.get_block_type_slug()
            block_document_name = name
        try:
            block_document = await client.read_block_document_by_name(name=block_document_name, block_type_slug=block_type_slug)
        except prefect.exceptions.ObjectNotFound as e:
            raise ValueError(f"Unable to find block document named {block_document_name} for block type {block_type_slug}") from e
        return (block_document, block_document_name)

    @classmethod
    def _get_block_document(cls, name: str, client: Any) -> Tuple[BlockDocument, str]:
        if cls.__name__ == "Block":
            block_type_slug, block_document_name = name.split("/", 1)
        else:
            block_type_slug = cls.get_block_type_slug()
            block_document_name = name
        try:
            block_document = client.read_block_document_by_name(name=block_document_name, block_type_slug=block_type_slug)
        except prefect.exceptions.ObjectNotFound as e:
            raise ValueError(f"Unable to find block document named {block_document_name} for block type {block_type_slug}") from e
        return (block_document, block_document_name)

    @classmethod
    @inject_client
    async def _get_block_document_by_id(cls, block_document_id: Union[str, UUID], client: Any = None) -> Tuple[BlockDocument, str]:
        if TYPE_CHECKING:
            from prefect.client.orchestration import PrefectClient
            assert isinstance(client, PrefectClient)
        if isinstance(block_document_id, str):
            try:
                block_document_id = UUID(block_document_id)
            except ValueError:
                raise ValueError(f"Block document ID {block_document_id!r} is not a valid UUID")
        try:
            block_document = await client.read_block_document(block_document_id=block_document_id)
        except prefect.exceptions.ObjectNotFound:
            raise ValueError(f"Unable to find block document with ID {block_document_id!r}")
        return (block_document, block_document.name)

    @classmethod
    @inject_client
    async def aload(cls, name: str, validate: bool = True, client: Any = None) -> Block:
        """
        Retrieves data from the block document with the given name for the block type
        that corresponds with the current class and returns an instantiated version of
        the current class with the data stored in the block document.
        """
        if TYPE_CHECKING:
            from prefect.client.orchestration import PrefectClient
            assert isinstance(client, PrefectClient)
        block_document, _ = await cls._aget_block_document(name, client=client)
        return cls._load_from_block_document(block_document, validate=validate)

    @classmethod
    @async_dispatch(aload)
    def load(cls, name: str, validate: bool = True, client: Any = None) -> Block:
        """
        Retrieves data from the block document with the given name for the block type
        that corresponds with the current class and returns an instantiated version of
        the current class with the data stored in the block document.
        """
        if client is None:
            from prefect.client.orchestration import get_client
            with get_client(sync_client=True) as sync_client:
                block_document, _ = cls._get_block_document(name, client=sync_client)
        else:
            block_document, _ = run_coro_as_sync(cls._aget_block_document(name, client=client))
        return cls._load_from_block_document(block_document, validate=validate)

    @classmethod
    @sync_compatible
    @inject_client
    async def load_from_ref(cls, ref: Union[str, UUID, Dict[str, Any]], validate: bool = True, client: Any = None) -> Block:
        """
        Retrieves data from the block document by given reference for the block type
        that corresponds with the current class and returns an instantiated version of
        the current class with the data stored in the block document.
        """
        if TYPE_CHECKING:
            from prefect.client.orchestration import PrefectClient
            assert isinstance(client, PrefectClient)
        block_document: Optional[BlockDocument] = None
        if isinstance(ref, (str, UUID)):
            block_document, _ = await cls._get_block_document_by_id(ref, client=client)
        elif (block_document_id := ref.get("block_document_id")):
            block_document, _ = await cls._get_block_document_by_id(block_document_id, client=client)
        elif (block_document_slug := ref.get("block_document_slug")):
            block_document, _ = await cls._aget_block_document(block_document_slug, client=client)
        if not block_document:
            raise ValueError(f"Invalid reference format {ref!r}.")
        return cls._load_from_block_document(block_document, validate=validate)

    @classmethod
    def _load_from_block_document(cls, block_document: BlockDocument, validate: bool = True) -> Block:
        """
        Loads a block from a given block document.
        """
        try:
            return cls._from_block_document(block_document)
        except ValidationError as e:
            if not validate:
                missing_fields = tuple((err["loc"][0] for err in e.errors()))
                missing_block_data = {field: None for field in missing_fields if isinstance(field, str)}
                warnings.warn(
                    f"Could not fully load {block_document.name!r} of block type {cls.get_block_type_slug()!r} - this is likely because one or more required fields were added to the schema for {cls.__name__!r} that did not exist on the class when this block was last saved. Please specify values for new field(s): {listrepr(missing_fields)}, then run `{cls.__name__}.save(\"{block_document.name}\", overwrite=True)`, and load this block again before attempting to use it."
                )
                return cls.model_construct(**block_document.data, **missing_block_data)
            raise RuntimeError(
                f"Unable to load {block_document.name!r} of block type {cls.get_block_type_slug()!r} due to failed validation. To load without validation, try loading again with `validate=False`."
            ) from e

    @staticmethod
    def is_block_class(block: Any) -> bool:
        return _is_subclass(block, Block)

    @staticmethod
    def annotation_refers_to_block_class(annotation: Any) -> bool:
        if Block.is_block_class(annotation):
            return True
        if get_origin(annotation) is Union:
            for inner in get_args(annotation):
                if Block.is_block_class(inner):
                    return True
        return False

    @classmethod
    @sync_compatible
    @inject_client
    async def register_type_and_schema(cls, client: Any = None) -> None:
        """
        Makes block available for configuration with current Prefect API.
        Recursively registers all nested blocks. Registration is idempotent.
        """
        if TYPE_CHECKING:
            from prefect.client.orchestration import PrefectClient
            assert isinstance(client, PrefectClient)
        if cls.__name__ == "Block":
            raise InvalidBlockRegistration("`register_type_and_schema` should be called on a Block subclass and not on the Block class directly.")
        if ABC in getattr(cls, "__bases__", []):
            raise InvalidBlockRegistration("`register_type_and_schema` should be called on a Block subclass and not on a Block interface class directly.")

        async def register_blocks_in_annotation(annotation: Any) -> None:
            """Walk through the annotation and register any nested blocks."""
            if Block.is_block_class(annotation):
                coro = annotation.register_type_and_schema(client=client)
                if TYPE_CHECKING:
                    assert isinstance(coro, Coroutine)
                await coro
            elif get_origin(annotation) in (Union, tuple, list, dict):
                for inner_annotation in get_args(annotation):
                    await register_blocks_in_annotation(inner_annotation)

        for field in cls.model_fields.values():
            if field.annotation is not None:
                await register_blocks_in_annotation(field.annotation)
        try:
            block_type = await client.read_block_type_by_slug(slug=cls.get_block_type_slug())
            cls._block_type_id = block_type.id
            local_block_type = cls._to_block_type()
            if _should_update_block_type(local_block_type=local_block_type, server_block_type=block_type):
                await client.update_block_type(
                    block_type_id=block_type.id,
                    block_type=BlockTypeUpdate(**local_block_type.model_dump(include={"logo_url", "documentation_url", "description", "code_example"})),
                )
        except prefect.exceptions.ObjectNotFound:
            block_type_create = BlockTypeCreate(
                **cls._to_block_type().model_dump(include={"name", "slug", "logo_url", "documentation_url", "description", "code_example"})
            )
            block_type = await client.create_block_type(block_type=block_type_create)
            cls._block_type_id = block_type.id
        try:
            block_schema = await client.read_block_schema_by_checksum(
                checksum=cls._calculate_schema_checksum(), version=cls.get_block_schema_version()
            )
        except prefect.exceptions.ObjectNotFound:
            block_schema_create = BlockSchemaCreate(
                **cls._to_block_schema(block_type_id=block_type.id).model_dump(include={"fields", "block_type_id", "capabilities", "version"})
            )
            block_schema = await client.create_block_schema(block_schema=block_schema_create)
        cls._block_schema_id = block_schema.id

    @inject_client
    async def _save(
        self,
        name: Optional[str] = None,
        is_anonymous: bool = False,
        overwrite: bool = False,
        client: Any = None,
    ) -> UUID:
        """
        Saves the values of a block as a block document with an option to save as an
        anonymous block document.
        """
        if TYPE_CHECKING:
            from prefect.client.orchestration import PrefectClient
            assert isinstance(client, PrefectClient)
        if name is None and (not is_anonymous):
            if self._block_document_name is None:
                raise ValueError("You're attempting to save a block document without a name. Please either call `save` with a `name` or pass `is_anonymous=True` to save an anonymous block.")
            else:
                name = self._block_document_name
        self._is_anonymous = is_anonymous
        coro = self.register_type_and_schema(client=client)
        if TYPE_CHECKING:
            assert isinstance(coro, Coroutine)
        await coro
        block_document: Optional[BlockDocument] = None
        try:
            block_document_create = BlockDocumentCreate(
                **self._to_block_document(name=name, include_secrets=True).model_dump(include={"name", "block_schema_id", "block_type_id", "data", "is_anonymous"})
            )
            block_document = await client.create_block_document(block_document=block_document_create)
        except prefect.exceptions.ObjectAlreadyExists as err:
            if overwrite:
                block_document_id = self._block_document_id
                if block_document_id is None and name is not None:
                    existing_block_document = await client.read_block_document_by_name(name=name, block_type_slug=self.get_block_type_slug())
                    block_document_id = existing_block_document.id
                block_document_update = BlockDocumentUpdate(
                    **self._to_block_document(name=name, include_secrets=True).model_dump(include={"block_schema_id", "data"})
                )
                await client.update_block_document(block_document_id=block_document_id, block_document=block_document_update)
                block_document = await client.read_block_document(block_document_id=block_document_id)
            else:
                raise ValueError(
                    "You are attempting to save values with a name that is already in use for this block type. If you would like to overwrite the values that are saved, then save with `overwrite=True`."
                ) from err
        self._block_document_name = block_document.name
        self._block_document_id = block_document.id
        return self._block_document_id

    @sync_compatible
    async def save(self, name: Optional[str] = None, overwrite: bool = False, client: Any = None) -> UUID:
        """
        Saves the values of a block as a block document.
        """
        document_id = await self._save(name=name, overwrite=overwrite, client=client)
        return document_id

    @classmethod
    @sync_compatible
    @inject_client
    async def delete(cls, name: str, client: Any = None) -> None:
        if TYPE_CHECKING:
            from prefect.client.orchestration import PrefectClient
            assert isinstance(client, PrefectClient)
        block_document, _ = await cls._aget_block_document(name, client=client)
        await client.delete_block_document(block_document.id)

    def __new__(cls, **kwargs: Any) -> Block:
        """
        Create an instance of the Block subclass type if a `block_type_slug` is
        present in the data payload.
        """
        block_type_slug = kwargs.pop("block_type_slug", None)
        if block_type_slug:
            subcls = cls.get_block_class_from_key(block_type_slug)
            return super().__new__(subcls)
        else:
            return super().__new__(cls)

    def get_block_placeholder(self) -> str:
        """
        Returns the block placeholder for the current block which can be used for
        templating.
        """
        block_document_name = self._block_document_name
        if not block_document_name:
            raise BlockNotSavedError("Could not generate block placeholder for unsaved block.")
        return f"prefect.blocks.{self.get_block_type_slug()}.{block_document_name}"

    @classmethod
    def model_json_schema(
        cls,
        *,
        by_alias: bool = True,
        ref_template: str = "#/definitions/{model}",
        schema_generator: Type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: str = "validation",
    ) -> Dict[str, Any]:
        """TODO: stop overriding this method - use GenerateSchema in ConfigDict instead?"""
        schema = super().model_json_schema(by_alias, ref_template, schema_generator, mode)
        if "$defs" in schema:
            schema["definitions"] = schema.pop("$defs")
        if "additionalProperties" in schema:
            schema.pop("additionalProperties")
        for _, definition in schema.get("definitions", {}).items():
            if "additionalProperties" in definition:
                definition.pop("additionalProperties")
        return schema

    @classmethod
    def model_validate(cls, obj: Any, *, strict: Any = None, from_attributes: bool = None, context: Optional[Dict[str, Any]] = None) -> Block:
        if isinstance(obj, dict):
            extra_serializer_fields = {"_block_document_id", "_block_document_name", "_is_anonymous"}.intersection(obj.keys())
            for field in extra_serializer_fields:
                obj.pop(field, None)
        return super().model_validate(obj, strict=strict, from_attributes=from_attributes, context=context)

    def model_dump(
        self,
        *,
        mode: str = "python",
        include: Any = None,
        exclude: Any = None,
        context: Any = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool = True,
        serialize_as_any: bool = False,
    ) -> Dict[str, Any]:
        d = super().model_dump(
            mode=mode,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            serialize_as_any=serialize_as_any,
        )
        extra_serializer_fields = {"block_type_slug", "_block_document_id", "_block_document_name", "_is_anonymous"}.intersection(d.keys())
        for field in extra_serializer_fields:
            if (include and field not in include) or (exclude and field in exclude):
                d.pop(field)
        return d