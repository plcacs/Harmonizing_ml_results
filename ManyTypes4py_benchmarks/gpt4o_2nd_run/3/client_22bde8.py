from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List
from httpx import HTTPStatusError
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.exceptions import ObjectAlreadyExists, ObjectNotFound

if TYPE_CHECKING:
    from uuid import UUID
    from prefect.client.schemas.actions import BlockDocumentCreate, BlockDocumentUpdate
    from prefect.client.schemas.objects import BlockDocument

class BlocksDocumentClient(BaseClient):

    def create_block_document(self, block_document: BlockDocumentCreate, include_secrets: bool = True) -> BlockDocument:
        block_document_data = block_document.model_dump(
            mode='json',
            exclude_unset=True,
            exclude={'id', 'block_schema', 'block_type'},
            context={'include_secrets': include_secrets},
            serialize_as_any=True
        )
        try:
            response = self.request('POST', '/block_documents/', json=block_document_data)
        except HTTPStatusError as e:
            if e.response.status_code == 409:
                raise ObjectAlreadyExists(http_exc=e) from e
            else:
                raise
        from prefect.client.schemas.objects import BlockDocument
        return BlockDocument.model_validate(response.json())

    def update_block_document(self, block_document_id: UUID, block_document: BlockDocumentUpdate) -> None:
        try:
            self.request(
                'PATCH',
                '/block_documents/{id}',
                path_params={'id': block_document_id},
                json=block_document.model_dump(
                    mode='json',
                    exclude_unset=True,
                    include={'data', 'merge_existing_data', 'block_schema_id'}
                )
            )
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    def delete_block_document(self, block_document_id: UUID) -> None:
        try:
            self.request('DELETE', '/block_documents/{id}', path_params={'id': block_document_id})
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    def read_block_document(self, block_document_id: UUID, include_secrets: bool = True) -> Optional[BlockDocument]:
        try:
            response = self.request(
                'GET',
                '/block_documents/{id}',
                path_params={'id': block_document_id},
                params=dict(include_secrets=include_secrets)
            )
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise
        from prefect.client.schemas.objects import BlockDocument
        return BlockDocument.model_validate(response.json())

    def read_block_documents(
        self,
        block_schema_type: Optional[str] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        include_secrets: bool = True
    ) -> List[BlockDocument]:
        response = self.request(
            'POST',
            '/block_documents/filter',
            json=dict(
                block_schema_type=block_schema_type,
                offset=offset,
                limit=limit,
                include_secrets=include_secrets
            )
        )
        from prefect.client.schemas.objects import BlockDocument
        return BlockDocument.model_validate_list(response.json())

class BlocksDocumentAsyncClient(BaseAsyncClient):

    async def create_block_document(self, block_document: BlockDocumentCreate, include_secrets: bool = True) -> BlockDocument:
        block_document_data = block_document.model_dump(
            mode='json',
            exclude_unset=True,
            exclude={'id', 'block_schema', 'block_type'},
            context={'include_secrets': include_secrets},
            serialize_as_any=True
        )
        try:
            response = await self.request('POST', '/block_documents/', json=block_document_data)
        except HTTPStatusError as e:
            if e.response.status_code == 409:
                raise ObjectAlreadyExists(http_exc=e) from e
            else:
                raise
        from prefect.client.schemas.objects import BlockDocument
        return BlockDocument.model_validate(response.json())

    async def update_block_document(self, block_document_id: UUID, block_document: BlockDocumentUpdate) -> None:
        try:
            await self.request(
                'PATCH',
                '/block_documents/{id}',
                path_params={'id': block_document_id},
                json=block_document.model_dump(
                    mode='json',
                    exclude_unset=True,
                    include={'data', 'merge_existing_data', 'block_schema_id'}
                )
            )
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    async def delete_block_document(self, block_document_id: UUID) -> None:
        try:
            await self.request('DELETE', '/block_documents/{id}', path_params={'id': block_document_id})
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    async def read_block_document(self, block_document_id: UUID, include_secrets: bool = True) -> Optional[BlockDocument]:
        try:
            response = await self.request(
                'GET',
                '/block_documents/{id}',
                path_params={'id': block_document_id},
                params=dict(include_secrets=include_secrets)
            )
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise
        from prefect.client.schemas.objects import BlockDocument
        return BlockDocument.model_validate(response.json())

    async def read_block_documents(
        self,
        block_schema_type: Optional[str] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        include_secrets: bool = True
    ) -> List[BlockDocument]:
        response = await self.request(
            'POST',
            '/block_documents/filter',
            json=dict(
                block_schema_type=block_schema_type,
                offset=offset,
                limit=limit,
                include_secrets=include_secrets
            )
        )
        from prefect.client.schemas.objects import BlockDocument
        return BlockDocument.model_validate_list(response.json())
