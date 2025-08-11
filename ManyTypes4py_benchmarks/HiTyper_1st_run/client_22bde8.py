from __future__ import annotations
from typing import TYPE_CHECKING
from httpx import HTTPStatusError
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.exceptions import ObjectAlreadyExists, ObjectNotFound
if TYPE_CHECKING:
    from uuid import UUID
    from prefect.client.schemas.actions import BlockDocumentCreate, BlockDocumentUpdate
    from prefect.client.schemas.objects import BlockDocument

class BlocksDocumentClient(BaseClient):

    def create_block_document(self, block_document: Union[bool, dict, bytes], include_secrets: bool=True):
        """
        Create a block document in the Prefect API. This data is used to configure a
        corresponding Block.

        Args:
            include_secrets (bool): whether to include secret values
                on the stored Block, corresponding to Pydantic's `SecretStr` and
                `SecretBytes` fields. Note Blocks may not work as expected if
                this is set to `False`.
        """
        block_document_data = block_document.model_dump(mode='json', exclude_unset=True, exclude={'id', 'block_schema', 'block_type'}, context={'include_secrets': include_secrets}, serialize_as_any=True)
        try:
            response = self.request('POST', '/block_documents/', json=block_document_data)
        except HTTPStatusError as e:
            if e.response.status_code == 409:
                raise ObjectAlreadyExists(http_exc=e) from e
            else:
                raise
        from prefect.client.schemas.objects import BlockDocument
        return BlockDocument.model_validate(response.json())

    def update_block_document(self, block_document_id: str, block_document: str) -> None:
        """
        Update a block document in the Prefect API.
        """
        try:
            self.request('PATCH', '/block_documents/{id}', path_params={'id': block_document_id}, json=block_document.model_dump(mode='json', exclude_unset=True, include={'data', 'merge_existing_data', 'block_schema_id'}))
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    def delete_block_document(self, block_document_id: Union[str, int]) -> None:
        """
        Delete a block document.
        """
        try:
            self.request('DELETE', '/block_documents/{id}', path_params={'id': block_document_id})
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    def read_block_document(self, block_document_id: Union[str, dict[str, typing.Any], dict, None], include_secrets: bool=True):
        """
        Read the block document with the specified ID.

        Args:
            block_document_id: the block document id
            include_secrets (bool): whether to include secret values
                on the Block, corresponding to Pydantic's `SecretStr` and
                `SecretBytes` fields. These fields are automatically obfuscated
                by Pydantic, but users can additionally choose not to receive
                their values from the API. Note that any business logic on the
                Block may not work if this is `False`.

        Raises:
            httpx.RequestError: if the block document was not found for any reason

        Returns:
            A block document or None.
        """
        try:
            response = self.request('GET', '/block_documents/{id}', path_params={'id': block_document_id}, params=dict(include_secrets=include_secrets))
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise
        from prefect.client.schemas.objects import BlockDocument
        return BlockDocument.model_validate(response.json())

    def read_block_documents(self, block_schema_type: Union[None, int]=None, offset: Union[None, int]=None, limit: Union[None, int]=None, include_secrets: bool=True) -> Union[collections.abc.AsyncIterator, bytes, bool]:
        """
        Read block documents

        Args:
            block_schema_type: an optional block schema type
            offset: an offset
            limit: the number of blocks to return
            include_secrets (bool): whether to include secret values
                on the Block, corresponding to Pydantic's `SecretStr` and
                `SecretBytes` fields. These fields are automatically obfuscated
                by Pydantic, but users can additionally choose not to receive
                their values from the API. Note that any business logic on the
                Block may not work if this is `False`.

        Returns:
            A list of block documents
        """
        response = self.request('POST', '/block_documents/filter', json=dict(block_schema_type=block_schema_type, offset=offset, limit=limit, include_secrets=include_secrets))
        from prefect.client.schemas.objects import BlockDocument
        return BlockDocument.model_validate_list(response.json())

class BlocksDocumentAsyncClient(BaseAsyncClient):

    async def create_block_document(self, block_document, include_secrets=True):
        """
        Create a block document in the Prefect API. This data is used to configure a
        corresponding Block.

        Args:
            include_secrets (bool): whether to include secret values
                on the stored Block, corresponding to Pydantic's `SecretStr` and
                `SecretBytes` fields. Note Blocks may not work as expected if
                this is set to `False`.
        """
        block_document_data = block_document.model_dump(mode='json', exclude_unset=True, exclude={'id', 'block_schema', 'block_type'}, context={'include_secrets': include_secrets}, serialize_as_any=True)
        try:
            response = await self.request('POST', '/block_documents/', json=block_document_data)
        except HTTPStatusError as e:
            if e.response.status_code == 409:
                raise ObjectAlreadyExists(http_exc=e) from e
            else:
                raise
        from prefect.client.schemas.objects import BlockDocument
        return BlockDocument.model_validate(response.json())

    async def update_block_document(self, block_document_id, block_document):
        """
        Update a block document in the Prefect API.
        """
        try:
            await self.request('PATCH', '/block_documents/{id}', path_params={'id': block_document_id}, json=block_document.model_dump(mode='json', exclude_unset=True, include={'data', 'merge_existing_data', 'block_schema_id'}))
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    async def delete_block_document(self, block_document_id):
        """
        Delete a block document.
        """
        try:
            await self.request('DELETE', '/block_documents/{id}', path_params={'id': block_document_id})
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    async def read_block_document(self, block_document_id, include_secrets=True):
        """
        Read the block document with the specified ID.

        Args:
            block_document_id: the block document id
            include_secrets (bool): whether to include secret values
                on the Block, corresponding to Pydantic's `SecretStr` and
                `SecretBytes` fields. These fields are automatically obfuscated
                by Pydantic, but users can additionally choose not to receive
                their values from the API. Note that any business logic on the
                Block may not work if this is `False`.

        Raises:
            httpx.RequestError: if the block document was not found for any reason

        Returns:
            A block document or None.
        """
        try:
            response = await self.request('GET', '/block_documents/{id}', path_params={'id': block_document_id}, params=dict(include_secrets=include_secrets))
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise
        from prefect.client.schemas.objects import BlockDocument
        return BlockDocument.model_validate(response.json())

    async def read_block_documents(self, block_schema_type=None, offset=None, limit=None, include_secrets=True):
        """
        Read block documents

        Args:
            block_schema_type: an optional block schema type
            offset: an offset
            limit: the number of blocks to return
            include_secrets (bool): whether to include secret values
                on the Block, corresponding to Pydantic's `SecretStr` and
                `SecretBytes` fields. These fields are automatically obfuscated
                by Pydantic, but users can additionally choose not to receive
                their values from the API. Note that any business logic on the
                Block may not work if this is `False`.

        Returns:
            A list of block documents
        """
        response = await self.request('POST', '/block_documents/filter', json=dict(block_schema_type=block_schema_type, offset=offset, limit=limit, include_secrets=include_secrets))
        from prefect.client.schemas.objects import BlockDocument
        return BlockDocument.model_validate_list(response.json())