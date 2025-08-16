from prefect.server import schemas
from prefect.server.database import PrefectDBInterface, db_injector
from prefect.server.models.block_types import read_block_type_by_slug
from prefect.server.schemas.actions import BlockSchemaCreate
from prefect.server.schemas.core import BlockSchema, BlockSchemaReference
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict

class MissingBlockTypeException(Exception):
    """Raised when the block type corresponding to a block schema cannot be found"""

@db_injector
async def create_block_schema(db, session: AsyncSession, block_schema: Union[schemas.actions.BlockSchemaCreate, Dict[str, Any]], override: bool = False, definitions: Optional[Dict[str, Any]] = None) -> BlockSchema:
    ...

async def _register_nested_block_schemas(db, session: AsyncSession, parent_block_schema_id: UUID, block_schema_references: Dict[str, Any], base_fields: Dict[str, Any], definitions: Optional[Dict[str, Any]], override: bool = False) -> None:
    ...

def _get_fields_for_child_schema(db, definitions: Dict[str, Any], base_fields: Dict[str, Any], reference_name: str, reference_block_type: Any) -> Optional[Dict[str, Any]]:
    ...

@db_injector
async def delete_block_schema(db, session: AsyncSession, block_schema_id: UUID) -> bool:
    ...

@db_injector
async def read_block_schema(db, session: AsyncSession, block_schema_id: UUID) -> orm_models.BlockSchema:
    ...

def _construct_full_block_schema(block_schemas_with_references: List[Tuple[orm_models.BlockSchema, str, UUID]], root_block_schema: Optional[BlockSchema] = None) -> BlockSchema:
    ...

@db_injector
async def read_block_schemas(db, session: AsyncSession, block_schema_filter: Optional[Any] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> List[orm_models.BlockSchema]:
    ...

@db_injector
async def read_block_schema_by_checksum(db, session: AsyncSession, checksum: str, version: Optional[int] = None) -> orm_models.BlockSchema:
    ...

@db_injector
async def read_available_block_capabilities(db, session: AsyncSession) -> List[str]:
    ...

@db_injector
async def create_block_schema_reference(db, session: AsyncSession, block_schema_reference: BlockSchemaReference) -> orm_models.BlockSchemaReference:
    ...
