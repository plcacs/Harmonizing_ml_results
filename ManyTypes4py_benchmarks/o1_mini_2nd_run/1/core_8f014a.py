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
    TYPE_CHECKING,
    Any,
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
)
from uuid import UUID, uuid4
from griffe import Docstring, DocstringSection, DocstringSectionKind, Parser, parse
from packaging.version import InvalidVersion, Version
from pydantic import (
    BaseModel,
    ConfigDict,
    HttpUrl,
    PrivateAttr,
    SecretBytes,
    SecretStr,
    ValidationError,
    model_serializer,
)
from pydantic.json_schema import GenerateJsonSchema
from typing_extensions import Literal, ParamSpec, Self, TypeGuard, get_args
import prefect.exceptions
from prefect._internal.compatibility.async_dispatch import async_dispatch
from prefect.client.schemas import (
    DEFAULT_BLOCK_SCHEMA_VERSION,
    BlockDocument,
    BlockSchema,
    BlockType,
    BlockTypeUpdate,
)
from prefect.client.schemas.actions import (
    BlockDocumentCreate,
    BlockDocumentUpdate,
    BlockSchemaCreate,
    BlockTypeCreate,
)
from prefect.client.utilities import inject_client
from prefect.events import emit_event
from prefect.logging.loggers import disable_logger
from prefect.plugins import load_prefect_collections
from prefect.types import SecretDict
from prefect.utilities.asyncutils import run_coro_as_sync, sync_compatible
from prefect.utilities.collections import (
    listrepr,
    remove_nested_keys,
    visit_collection,
)
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
    reference_strings: List[str] = _collect_nested_reference_strings(object_definition)
    for reference_string in reference_strings:
        if isinstance(reference_string, str):
            definition_key = reference_string.replace("#/definitions/", "")
            definition: Optional[Dict[str, Any]] = definitions.get(definition_key)
            if definition and definition.get("block_type_slug") is None:
                non_block_definitions = {
                    **non_block_definitions,
                    definition_key: definition,
                    **_get_non_block_reference_definitions(definition, definitions),
                }
    return non_block_definitions


def _is_subclass(cls: Any, parent_cls: Type[Any]) -> bool:
    """
    Checks if a given class is a subclass of another class. Unlike issubclass,
    this will not throw an exception if cls is an instance instead of a type.
    """
    return inspect.isclass(cls) and (not get_origin(cls)) and issubclass(cls, parent_cls)


def _collect_secret_fields(
    name: str,
    type_: Any,
    secrets: List[str],
) -> None:
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
        for field_name, field in type_.model_fields.items():
            if field.annotation is not None:
                _collect_secret_fields(f"{name}.{field_name}", field.annotation, secrets)
        return
    if (
        type_ in (SecretStr, SecretBytes)
        or (
            isinstance(type_, type)
            and getattr(type_, "__module__", None) == "pydantic.types"
            and getattr(type_, "__name__", None) == "Secret"
        )
    ):
        secrets.append(name)
    elif type_ == SecretDict:
        secrets.append(f"{name}.*")
    elif Block.is_block_class(type_):
        secrets.extend(
            (f"{name}.{s}" for s in type_.model_json_schema()["secret_fields"])
        )


def _should_update_block_type(
    local_block_type: BlockType, server_block_type: BlockType
) -> bool:
    """
    Compares the fields of `local_block_type` and `server_block_type`.
    Only compare the possible updatable fields as defined by `BlockTypeUpdate.updatable_fields`
    Returns True if they are different, otherwise False.
    """
    fields = BlockTypeUpdate.updatable_fields()
    local_block_fields: Dict[str, Any] = local_block_type.model_dump(
        include=fields, exclude_unset=True
    )
    server_block_fields: Dict[str, Any] = server_block_type.model_dump(
        include=fields, exclude_unset=True
    )
    if local_block_fields.get("description") is not None:
        local_block_fields["description"] = html.unescape(
            local_block_fields["description"]
        )
    if local_block_fields.get("code_example") is not None:
        local_block_fields["code_example"] = html.unescape(
            local_block_fields["code_example"]
        )
    if server_block_fields.get("description") is not None:
        server_block_fields["description"] = html.unescape(
            server_block_fields["description"]
        )
    if server_block_fields.get("code_example") is not None:
        server_block_fields["code_example"] = html.unescape(
            server_block_fields["code_example"]
        )
    return server_block_fields != local_block_fields


class BlockNotSavedError(RuntimeError):
    """
    Raised when a given block is not saved and an operation that requires
    the block to be saved is attempted.
    """
    pass


def schema_extra(schema: Dict[str, Any], model: Type[BaseModel]) -> None:
    """
    Customizes Pydantic's schema generation feature to add blocks related information.
    """
    schema["block_type_slug"] = model.get_block_type_slug()
    description: Optional[str] = model.get_description()
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

    def collect_block_schema_references(
        field_name: str, annotation: Any
    ) -> None:
        """Walk through the annotation and collect block schemas for any nested blocks."""
        if Block.is_block_class(annotation):
            if isinstance(refs.get(field_name), list):
                refs[field_name].append(
                    annotation._to_block_schema_reference_dict()
                )
            elif isinstance(refs.get(field_name), dict):
                refs[field_name] = [
                    refs[field_name],
                    annotation._to_block_schema_reference_dict(),
                ]
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
    model_config: ConfigDict = ConfigDict(
        extra="allow", json_schema_extra=schema_extra
    )

    _block_type_name: Optional[str] = None
    _block_type_slug: Optional[str] = None
    _logo_url: Optional[HttpUrl] = None
    _documentation_url: Optional[HttpUrl] = None
    _description: Optional[str] = None
    _code_example: Optional[str] = None
    _block_type_id: Optional[UUID] = None
    _block_schema_id: Optional[UUID] = None
    _block_schema_capabilities: Optional[List[str]] = None
    _block_schema_version: Optional[str] = None
    _block_document_id: UUID | None = PrivateAttr(default=None)
    _block_document_name: Optional[str] = PrivateAttr(default=None)
    _is_anonymous: Optional[bool] = PrivateAttr(default=None)
    _events_excluded_methods: List[str] = PrivateAttr(
        default_factory=lambda: [
            "block_initialization",
            "save",
            "dict",
        ]
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.block_initialization()

    def __str__(self) -> str:
        return self.__repr__()

    def __repr_args__(self) -> List[Tuple[Optional[str], Any]]:
        repr_args = super().__repr_args__()
        data_keys = self.model_json_schema()["properties"].keys()
        return [
            (key, value)
            for key, value in repr_args
            if key is None or key in data_keys
        ]

    def block_initialization(self) -> None:
        pass

    @classmethod
    def __dispatch_key__(cls) -> Optional[str]:
        if cls.__name__ == "Block":
            return None
        return block_schema_to_key(cls._to_block_schema())

    @model_serializer(mode="wrap")
    def ser_model(
        self,
        handler: Any,
        info: SerializationInfo,
    ) -> Dict[str, Any]:
        jsonable_self: Dict[str, Any] = handler(self)
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
        if (
            extra_fields := {
                "block_type_slug": self.get_block_type_slug(),
                "_block_document_id": self._block_document_id,
                "_block_document_name": self._block_document_name,
                "_is_anonymous": self._is_anonymous,
            }
        ):
            jsonable_self |= {
                key: value
                for key, value in extra_fields.items()
                if value is not None
            }
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
            top_level_module = sys.modules[
                current_module.__name__.split(".")[0] or "__main__"
            ]
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
        return {
            "block_type_slug": cls.get_block_type_slug(),
            "block_schema_checksum": cls._calculate_schema_checksum(),
        }

    @classmethod
    def _calculate_schema_checksum(
        cls, block_schema_fields: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates a unique hash for the underlying schema of block.

        Args:
            block_schema_fields: Dictionary detailing block schema fields to generate a
                checksum for. The fields of the current class is used if this parameter
                is not provided.

        Returns:
            str: The calculated checksum prefixed with the hashing algorithm used.
        """
        block_schema_fields = (
            cls.model_json_schema() if block_schema_fields is None else block_schema_fields
        )
        fields_for_checksum = remove_nested_keys(["secret_fields"], block_schema_fields)
        if fields_for_checksum.get("definitions"):
            non_block_definitions = _get_non_block_reference_definitions(
                fields_for_checksum, fields_for_checksum["definitions"]
            )
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

        Args:
            name: The name of the created block document. Not required if anonymous.
            block_schema_id: UUID of the corresponding block schema.
            block_type_id: UUID of the corresponding block type.
            is_anonymous: if True, an anonymous block is created. Anonymous
                blocks are not displayed in the UI and used primarily for system
                operations and features that need to automatically generate blocks.
            include_secrets: Whether to include secret fields in the data.

        Returns:
            BlockDocument: Corresponding block document
                populated with the block's configured data.
        """
        if is_anonymous is None:
            is_anonymous = self._is_anonymous or False
        if not is_anonymous and (not name) and (not self._block_document_name):
            raise ValueError(
                "No name provided, either as an argument or on the block."
            )
        if not block_schema_id and (not self._block_schema_id):
            raise ValueError(
                "No block schema ID provided, either as an argument or on the block."
            )
        if not block_type_id and (not self._block_type_id):
            raise ValueError(
                "No block type ID provided, either as an argument or on the block."
            )
        data_keys = self.model_json_schema(by_alias=False)["properties"].keys()
        block_document_data: Dict[str, Any] = self.model_dump(
            by_alias=True,
            include=data_keys,
            context={"include_secrets": include_secrets},
        )
        for key in data_keys:
            field_value = getattr(self, key)
            if isinstance(field_value, Block) and field_value._block_document_id is not None:
                block_document_data[key] = {
                    "$ref": {"block_document_id": field_value._block_document_id}
                }
        block_schema_id = block_schema_id or self._block_schema_id
        block_type_id = block_type_id or self._block_type_id
        if block_schema_id is None:
            raise ValueError(
                "No block schema ID provided, either as an argument or on the block."
            )
        if block_type_id is None:
            raise ValueError(
                "No block type ID provided, either as an argument or on the block."
            )
        return BlockDocument(
            id=self._block_document_id or uuid4(),
            name=name
            or (self._block_document_name if not is_anonymous else None),
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

        Args:
            block_type_id: UUID of the corresponding block type.

        Returns:
            BlockSchema: The corresponding block schema.
        """
        fields: Dict[str, Any] = cls.model_json_schema()
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
            parsed: List[DocstringSection] = parse(docstring, Parser.google)
        return parsed

    @classmethod
    def get_description(cls) -> Optional[str]:
        """
        Returns the description for the current block. Attempts to parse
        description from class docstring if an override is not defined.
        """
        description: Optional[str] = cls._description
        if description is None and cls.__doc__ is not None:
            parsed: List[DocstringSection] = cls._parse_docstring()
            parsed_description: Optional[str] = next(
                (
                    section.as_dict().get("value")
                    for section in parsed
                    if section.kind == DocstringSectionKind.text
                ),
                None,
            )
            if isinstance(parsed_description, str):
                description = parsed_description.strip()
        return description

    @classmethod
    def get_code_example(cls) -> str:
        """
        Returns the code example for the given block. Attempts to parse
        code example from the class docstring if an override is not provided.
        """
        code_example: Optional[str] = (
            dedent(text=cls._code_example) if cls._code_example is not None else None
        )
        if code_example is None and cls.__doc__ is not None:
            parsed: List[DocstringSection] = cls._parse_docstring()
            for section in parsed:
                if section.kind == DocstringSectionKind.examples:
                    code_example = "\n".join(
                        (part[1] for part in section.as_dict().get("value", []))
                    )
                    break
                if section.kind == DocstringSectionKind.admonition:
                    value: Dict[str, Any] = section.as_dict().get("value", {})
                    if value.get("annotation") == "example":
                        code_example = value.get("description")
                        break
        if code_example is None:
            code_example = cls._generate_code_example()
        return code_example

    @classmethod
    def _generate_code_example(cls) -> str:
        """Generates a default code example for the current class"""
        qualified_name: str = to_qualified_name(cls)
        module_str: str = ".".join(qualified_name.split(".")[:-1])
        origin: Type[Any] = cls.__pydantic_generic_metadata__.get("origin") or cls
        class_name: str = origin.__name__
        block_variable_name: str = f"{cls.get_block_type_slug().replace('-', '_')}_block"
        return dedent(
            f"""\
            