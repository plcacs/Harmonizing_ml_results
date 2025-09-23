"""
Functions for interacting with block schema ORM objects.
Intended for internal use by the Prefect REST API.
"""
import json
from copy import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast
from uuid import UUID
import sqlalchemy as sa
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.engine import Result
from prefect.server import schemas
from prefect.server.database import PrefectDBInterface, db_injector, orm_models
from prefect.server.models.block_types import read_block_type_by_slug
from prefect.server.schemas.actions import BlockSchemaCreate
from prefect.server.schemas.core import BlockSchema, BlockSchemaReference
from prefect.server.schemas.filters import BlockSchemaFilter
if TYPE_CHECKING:
    from prefect.client.schemas.actions import BlockSchemaCreate as ClientBlockSchemaCreate
    from prefect.client.schemas.objects import BlockSchema as ClientBlockSchema

class MissingBlockTypeException(Exception):
    """Raised when the block type corresponding to a block schema cannot be found"""

@db_injector
async def create_block_schema(db: PrefectDBInterface, session: AsyncSession, block_schema: Union[schemas.actions.BlockSchemaCreate, schemas.core.BlockSchema, 'ClientBlockSchemaCreate', 'ClientBlockSchema'], override: bool=False, definitions: Optional[dict[str, Any]]=None) -> Union[BlockSchema, orm_models.BlockSchema]:
    """
    Create a new block schema.

    Args:
        session: A database session
        block_schema: a block schema object
        definitions: Definitions of fields from block schema fields
            attribute. Used when recursively creating nested block schemas

    Returns:
        block_schema: an ORM block schema model
    """
    from prefect.blocks.core import Block, _get_non_block_reference_definitions
    if not isinstance(block_schema, schemas.actions.BlockSchemaCreate):
        block_schema = schemas.actions.BlockSchemaCreate.model_validate(block_schema.model_dump(mode='json', exclude={'id', 'created', 'updated', 'checksum', 'block_type'}))
    insert_values: Dict[str, Any] = block_schema.model_dump_for_orm(exclude_unset=False, exclude={'block_type', 'id', 'created', 'updated'})
    definitions = definitions or block_schema.fields.get('definitions')
    fields_for_checksum: Dict[str, Any] = insert_values['fields']
    if definitions:
        fields_for_checksum['definitions'] = definitions
    checksum: str = Block._calculate_schema_checksum(fields_for_checksum)
    existing_block_schema: Optional[BlockSchema] = await read_block_schema_by_checksum(session=session, checksum=checksum, version=block_schema.version)
    if existing_block_schema:
        return existing_block_schema
    insert_values['checksum'] = checksum
    if definitions:
        non_block_definitions: Optional[Dict[str, Any]] = _get_non_block_reference_definitions(insert_values['fields'], definitions)
        if non_block_definitions:
            insert_values['fields']['definitions'] = _get_non_block_reference_definitions(insert_values['fields'], definitions)
        else:
            insert_values['fields'].pop('definitions', None)
    block_schema_references: Dict[str, Any] = insert_values['fields'].pop('block_schema_references', {})
    insert_stmt = db.queries.insert(db.BlockSchema).values(**insert_values)
    if override:
        insert_stmt = insert_stmt.on_conflict_do_update(index_elements=db.orm.block_schema_unique_upsert_columns, set_=insert_values)
    await session.execute(insert_stmt)
    query: sa.Select = sa.select(db.BlockSchema).where(db.BlockSchema.checksum == insert_values['checksum']).order_by(db.BlockSchema.created.desc()).limit(1).execution_options(populate_existing=True)
    if block_schema.version is not None:
        query = query.where(db.BlockSchema.version == block_schema.version)
    result: Result = await session.execute(query)
    created_block_schema: Union[BlockSchema, orm_models.BlockSchema] = copy(result.scalar_one())
    await _register_nested_block_schemas(db, session=session, parent_block_schema_id=created_block_schema.id, block_schema_references=block_schema_references, base_fields=insert_values['fields'], definitions=definitions, override=override)
    created_block_schema.fields['block_schema_references'] = block_schema_references
    if definitions is not None:
        created_block_schema.fields['definitions'] = definitions
    return created_block_schema

async def _register_nested_block_schemas(db: PrefectDBInterface, session: AsyncSession, parent_block_schema_id: UUID, block_schema_references: dict[str, Union[dict[str, str], List[dict[str, str]]]], base_fields: Dict[str, Any], definitions: Optional[Dict[str, Any]], override: bool=False) -> None:
    """
    Iterates through each of the block schema references declared on the block schema.
    Attempts to register each of the nested block schemas if they have not already been
    registered. An error is thrown if the corresponding block type for a block schema
    has not been registered.

    Args:
        session: A database session.
        parent_block_schema_id: The ID of the parent block schema.
        block_schema_references: A dictionary containing the block schema references for
            the child block schemas of the parent block schema.
        base_fields: The field name and type declarations for the parent block schema.
        definitions: A dictionary of the field name and type declarations of each
            child block schema.
        override: Flag controlling if a block schema should updated in place.
    """
    for (reference_name, reference_values) in block_schema_references.items():
        reference_values_list: List[dict[str, str]] = reference_values if isinstance(reference_values, list) else [reference_values]
        for reference_values_entry in reference_values_list:
            reference_block_type: Optional[orm_models.BlockType] = await read_block_type_by_slug(session=session, block_type_slug=reference_values_entry['block_type_slug'])
            if reference_block_type is None:
                raise MissingBlockTypeException(f"Cannot create block schema because block type {reference_values_entry['block_type_slug']!r} was not found.Did you forget to register the block type?")
            reference_block_schema: Union[BlockSchema, orm_models.BlockSchema, None]
            reference_block_schema = await read_block_schema_by_checksum(session=session, checksum=reference_values_entry['block_schema_checksum'])
            if reference_block_schema is None:
                if definitions is None:
                    raise ValueError('Unable to create nested block schema due to missing definitions in root block schema fields')
                sub_block_schema_fields: Optional[Dict[str, Any]] = _get_fields_for_child_schema(db, definitions, base_fields, reference_name, reference_block_type)
                if sub_block_schema_fields is None:
                    raise ValueError(f'Unable to create nested block schema for block type {reference_block_type.name!r} due to missing definition.')
                reference_block_schema = await create_block_schema(session=session, block_schema=BlockSchemaCreate(fields=sub_block_schema_fields, block_type_id=reference_block_type.id), override=override, definitions=definitions)
            await create_block_schema_reference(session=session, block_schema_reference=BlockSchemaReference(parent_block_schema_id=parent_block_schema_id, reference_block_schema_id=reference_block_schema.id, name=reference_name))

def _get_fields_for_child_schema(db: PrefectDBInterface, definitions: Dict[str, Any], base_fields: Dict[str, Any], reference_name: str, reference_block_type: orm_models.BlockType) -> Optional[Dict[str, Any]]:
    """
    Returns the field definitions for a child schema. The fields definitions are pulled from the provided `definitions`
    dictionary based on the information extracted from `base_fields` using the `reference_name`. `reference_block_type`
    is used to disambiguate fields that have a union type.
    """
    from prefect.blocks.core import _collect_nested_reference_strings
    spec_reference: Dict[str, Any] = base_fields['properties'][reference_name]
    sub_block_schema_fields: Optional[Dict[str, Any]] = None
    reference_strings: List[str] = _collect_nested_reference_strings(spec_reference)
    if len(reference_strings) == 1:
        sub_block_schema_fields = definitions.get(reference_strings[0].replace('#/definitions/', ''))
    else:
        for reference_string in reference_strings:
            definition_key: str = reference_string.replace('#/definitions/', '')
            potential_sub_block_schema_fields: Dict[str, Any] = definitions[definition_key]
            if definitions[definition_key]['block_type_slug'] == reference_block_type.slug:
                sub_block_schema_fields = potential_sub_block_schema_fields
                break
    return sub_block_schema_fields

@db_injector
async def delete_block_schema(db: PrefectDBInterface, session: AsyncSession, block_schema_id: UUID) -> bool:
    """
    Delete a block schema by id.

    Args:
        session: A database session
        block_schema_id: a block schema id

    Returns:
        bool: whether or not the block schema was deleted
    """
    result: Result = await session.execute(delete(db.BlockSchema).where(db.BlockSchema.id == block_schema_id))
    return result.rowcount > 0

@db_injector
async def read_block_schema(db: PrefectDBInterface, session: AsyncSession, block_schema_id: UUID) -> Optional[BlockSchema]:
    """
    Reads a block schema by id. Will reconstruct the block schema's fields attribute
    to include block schema references.

    Args:
        session: A database session
        block_schema_id: a block_schema id

    Returns:
        orm_models..BlockSchema: the block_schema
    """
    block_schema_references_query: sa.Select = sa.select(db.BlockSchemaReference).select_from(db.BlockSchemaReference).filter_by(parent_block_schema_id=block_schema_id).cte('block_schema_references', recursive=True)
    block_schema_references_join: sa.Select = sa.select(db.BlockSchemaReference).select_from(db.BlockSchemaReference).join(block_schema_references_query, db.BlockSchemaReference.parent_block_schema_id == block_schema_references_query.c.reference_block_schema_id)
    recursive_block_schema_references_cte: sa.CTE = block_schema_references_query.union_all(block_schema_references_join)
    nested_block_schemas_query: sa.Select = sa.select(db.BlockSchema, recursive_block_schema_references_cte.c.name, recursive_block_schema_references_cte.c.parent_block_schema_id).select_from(db.BlockSchema).join(recursive_block_schema_references_cte, db.BlockSchema.id == recursive_block_schema_references_cte.c.reference_block_schema_id, isouter=True).filter(sa.or_(db.BlockSchema.id == block_schema_id, recursive_block_schema_references_cte.c.parent_block_schema_id.is_not(None)))
    result: Result = await session.execute(nested_block_schemas_query)
    return _construct_full_block_schema(result.all())

def _construct_full_block_schema(block_schemas_with_references: List[Tuple[BlockSchema, Optional[str], Optional[UUID]]], root_block_schema: Optional[BlockSchema]=None) -> Optional[BlockSchema]:
    """
    Takes a list of block schemas along with reference information and reconstructs
    the root block schema's fields attribute to contain block schema references for
    client consumption.

    Args:
        block_schema_with_references: A list of tuples with the structure:
            - A block schema object
            - The name the block schema lives under in the parent block schema
            - The ID of the block schema's parent block schema
        root_block_schema: Optional block schema to start traversal. Will attempt to
            determine root block schema if not provided.

    Returns:
        BlockSchema: A block schema with a fully reconstructed fields attribute
    """
    if len(block_schemas_with_references) == 0:
        return None
    root_block_schema_copy: Optional[BlockSchema] = copy(root_block_schema) if root_block_schema is not None else _find_root_block_schema(block_schemas_with_references)
    if root_block_schema_copy is None:
        raise ValueError('Unable to determine root block schema during schema reconstruction.')
    root_block_schema_copy.fields = _construct_block_schema_fields_with_block_references(root_block_schema_copy, block_schemas_with_references)
    definitions: Dict[str, Any] = _construct_block_schema_spec_definitions(root_block_schema_copy, block_schemas_with_references)
    if definitions or root_block_schema_copy.fields.get('definitions'):
        root_block_schema_copy.fields['definitions'] = {**root_block_schema_copy.fields.get('definitions', {}), **definitions}
    return root_block_schema_copy

def _find_root_block_schema(block_schemas_with_references: List[Tuple[BlockSchema, Optional[str], Optional[UUID]]]) -> Optional[BlockSchema]:
    """
    Attempts to find the root block schema from a list of block schemas
    with references. Returns None if a root block schema is not found.
    Returns only the first potential root block schema if multiple are found.
    """
    return next((copy(block_schema) for (block_schema, _, parent_block_schema_id) in block_schemas_with_references if parent_block_schema_id is None), None)

def _construct_block_schema_spec_definitions(root_block_schema: BlockSchema, block_schemas_with_references: List[Tuple[BlockSchema, Optional[str], Optional[UUID]]]) -> Dict[str, Any]:
    """
    Constructs field definitions for a block schema based on the nested block schemas
    as defined in the block_schemas_with_references list.
    """
    definitions: Dict[str, Any] = {}
    for (_, block_schema_references) in root_block_schema.fields['block_schema_references'].items():
        block_schema_references_list: List[Dict[str, str]] = block_schema_references if isinstance(block_schema_references, list) else [block_schema_references]
        for block_schema_reference in block_schema_references_list:
            child_block_schema: Optional[BlockSchema] = _find_block_schema_via_checksum(block_schemas_with_references, block_schema_reference['block_schema_checksum'])
            if child_block_schema is not None:
                child_block_schema_constructed: Optional[BlockSchema] = _construct_full_block_schema(block_schemas_with_references=block_schemas_with_references, root_block_schema=child_block_schema)
                assert child_block_schema_constructed
                definitions = _add_block_schemas_fields_to_definitions(definitions, child_block_schema_constructed)
    return definitions

def _find_block_schema_via_checksum(block_schemas_with_references: List[Tuple[BlockSchema, Optional[str], Optional[UUID]]], checksum: str) -> Optional[BlockSchema]:
    """Attempt to find a block schema via a given checksum. Returns None if not found."""
    return next((block_schema for (block_schema, _, _) in block_schemas_with_references if block_schema.checksum == checksum), None)

def _add_block_schemas_fields_to_definitions(definitions: Dict[str, Any], child_block_schema: BlockSchema) -> Dict[str, Any]:
    """
    Returns a new definitions dict with the fields of a block schema and it's child
    block schemas added to the existing definitions.
    """
    block_schema_title: Optional[str] = child_block_schema.fields.get('title')
    if block_schema_title is not None:
        child_definitions: Dict[str, Any] = child_block_schema.fields.pop('definitions', {})
        return {**definitions, **{block_schema_title: child_block_schema.fields}, **child_definitions}
    else:
        return definitions

def _construct_block_schema_fields_with_block_references(parent_block_schema: BlockSchema, block_schemas_with_references: List[Tuple[BlockSchema, Optional[str], Optional[UUID]]]) -> Dict[str, Any]:
    """
    Constructs the block_schema_references in a block schema's fields attributes. Returns
    a copy of the block schema with block_schema_references added.

    Args:
        parent_block_schema: The block schema that needs block references populated.
        block_schema_with_references: A list of tuples with the structure:
            - A block schema object
            - The name the block schema lives under in the parent block schema
            - The ID of the block schema's parent block schema

    Returns:
        Dict: Block schema fields with block schema references added.

    """
    block_schema_fields_copy: Dict[str, Any] = {**parent_block_schema.fields, 'block_schema_references': {}}
    for (nested_block_schema, name, parent_block_schema_id) in block_schemas_with_references:
        if parent_block_schema_id == parent_block_schema.id:
            assert nested_block_schema.block_type, f'{nested_block_schema} has no block type'
            new_block_schema_reference: Dict[str, str] = {'block_schema_checksum': nested_block_schema.checksum, 'block_type_slug': nested_block_schema.block_type.slug}
            if name not in block_schema_fields_copy['block_schema_references']:
                block_schema_fields_copy['block_schema_references'][name] = new_block_schema_reference
            elif isinstance(block_schema_fields_copy['block_schema_references'][name], list) and new_block_schema_reference not in block_schema_fields_copy['block_schema_references'][name]:
                block_schema_fields_copy['block_schema_references'][name].append(new_block_schema_reference)
            elif block_schema_fields_copy['block_schema_references'][name] != new_block_schema_reference:
                block_schema_fields_copy['block