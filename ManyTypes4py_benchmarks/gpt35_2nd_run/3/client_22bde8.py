from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Union
from httpx import HTTPStatusError
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.exceptions import ObjectAlreadyExists, ObjectNotFound

if TYPE_CHECKING:
    from uuid import UUID
    from prefect.client.schemas.actions import BlockDocumentCreate, BlockDocumentUpdate
    from prefect.client.schemas.objects import BlockDocument

class BlocksDocumentClient(BaseClient):

    def create_block_document(self, block_document: Union[BlockDocumentCreate, BlockDocumentUpdate], include_secrets: bool = True) -> BlockDocument:
        ...

    def update_block_document(self, block_document_id: UUID, block_document: BlockDocumentUpdate) -> None:
        ...

    def delete_block_document(self, block_document_id: UUID) -> None:
        ...

    def read_block_document(self, block_document_id: UUID, include_secrets: bool = True) -> Optional[BlockDocument]:
        ...

    def read_block_documents(self, block_schema_type: Optional[str] = None, offset: Optional[int] = None, limit: Optional[int] = None, include_secrets: bool = True) -> List[BlockDocument]:
        ...

class BlocksDocumentAsyncClient(BaseAsyncClient):

    async def create_block_document(self, block_document: Union[BlockDocumentCreate, BlockDocumentUpdate], include_secrets: bool = True) -> BlockDocument:
        ...

    async def update_block_document(self, block_document_id: UUID, block_document: BlockDocumentUpdate) -> None:
        ...

    async def delete_block_document(self, block_document_id: UUID) -> None:
        ...

    async def read_block_document(self, block_document_id: UUID, include_secrets: bool = True) -> Optional[BlockDocument]:
        ...

    async def read_block_documents(self, block_schema_type: Optional[str] = None, offset: Optional[int] = None, limit: Optional[int] = None, include_secrets: bool = True) -> List[BlockDocument]:
        ...
