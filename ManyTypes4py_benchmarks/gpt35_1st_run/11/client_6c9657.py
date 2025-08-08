from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from httpx import HTTPStatusError
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.exceptions import ObjectAlreadyExists, ObjectNotFound, ProtectedBlockError

if TYPE_CHECKING:
    from uuid import UUID
    from prefect.client.schemas.actions import BlockTypeCreate, BlockTypeUpdate
    from prefect.client.schemas.objects import BlockDocument, BlockType

class BlocksTypeClient(BaseClient):

    def read_block_documents_by_type(self, block_type_slug: str, offset: Optional[int] = None, limit: Optional[int] = None, include_secrets: bool = True) -> List[BlockDocument]:
        ...

    def create_block_type(self, block_type) -> BlockType:
        ...

    def read_block_type_by_slug(self, slug: str) -> BlockType:
        ...

    def update_block_type(self, block_type_id: UUID, block_type) -> None:
        ...

    def delete_block_type(self, block_type_id: UUID) -> None:
        ...

    def read_block_types(self) -> List[BlockType]:
        ...

    def read_block_document_by_name(self, name: str, block_type_slug: str, include_secrets: bool = True) -> BlockDocument:
        ...

class BlocksTypeAsyncClient(BaseAsyncClient):

    async def read_block_documents_by_type(self, block_type_slug: str, offset: Optional[int] = None, limit: Optional[int] = None, include_secrets: bool = True) -> List[BlockDocument]:
        ...

    async def create_block_type(self, block_type) -> BlockType:
        ...

    async def read_block_type_by_slug(self, slug: str) -> BlockType:
        ...

    async def update_block_type(self, block_type_id: UUID, block_type) -> None:
        ...

    async def delete_block_type(self, block_type_id: UUID) -> None:
        ...

    async def read_block_types(self) -> List[BlockType]:
        ...

    async def read_block_document_by_name(self, name: str, block_type_slug: str, include_secrets: bool = True) -> BlockDocument:
        ...
