class MissingBlockTypeException(Exception):
    """Raised when the block type corresponding to a block schema cannot be found"""

@db_injector
async def create_block_schema(db: PrefectDBInterface, session: AsyncSession, block_schema: BlockSchemaCreate, override: bool = False, definitions: Optional[Dict[str, Any]] = None) -> BlockSchema:
    """
    Create a new block schema.

    Args:
        session: A database session
        block_schema: a block schema object
        definitions: Definitions of fields from block schema fields
            attribute. Used when recursively creating nested block schemas
        override: Flag controlling if a block schema should updated in place

    Returns:
        block_schema: an ORM block schema model
    """
    # ...

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
    # ...

@db_injector
async def read_block_schema(db: PrefectDBInterface, session: AsyncSession, block_schema_id: UUID) -> BlockSchema:
    """
    Reads a block schema by id. Will reconstruct the block schema's fields attribute
    to include block schema references.

    Args:
        session: A database session
        block_schema_id: a block schema id

    Returns:
        BlockSchema: the block_schema
    """
    # ...

@db_injector
async def read_block_schemas(db: PrefectDBInterface, session: AsyncSession, block_schema_filter: Optional[schemas.actions.BlockSchemaFilter] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> List[BlockSchema]:
    """
    Reads block schemas, optionally filtered by type or name.

    Args:
        session: A database session
        block_schema_filter: a block schema filter object
        limit (int): query limit
        offset (int): query offset

    Returns:
        List[BlockSchema]: the block_schemas
    """
    # ...

@db_injector
async def read_block_schema_by_checksum(db: PrefectDBInterface, session: AsyncSession, checksum: str, version: Optional[str] = None) -> BlockSchema:
    """
    Reads a block_schema by checksum. Will reconstruct the block schema's fields
    attribute to include block schema references.

    Args:
        session: A database session
        checksum: a block_schema checksum
        version: A block_schema version

    Returns:
        BlockSchema: the block_schema
    """
    # ...

@db_injector
async def read_available_block_capabilities(db: PrefectDBInterface, session: AsyncSession) -> List[str]:
    """
    Retrieves a list of all available block capabilities.

    Args:
        session: A database session.

    Returns:
        List[str]: List of all available block capabilities.
    """
    # ...

@db_injector
async def create_block_schema_reference(db: PrefectDBInterface, session: AsyncSession, block_schema_reference: BlockSchemaReference) -> BlockSchemaReference:
    """
    Creates a block schema reference.

    Args:
        session: A database session
        block_schema_reference: A block schema reference object

    Returns:
        BlockSchemaReference: The created BlockSchemaReference
    """
    # ...
