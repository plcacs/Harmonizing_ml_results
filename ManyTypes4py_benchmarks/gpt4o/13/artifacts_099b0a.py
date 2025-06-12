from __future__ import annotations
import json
import math
import warnings
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional, Union, cast
from uuid import UUID
from typing_extensions import Self
from prefect._internal.compatibility.async_dispatch import async_dispatch
from prefect.client.orchestration import PrefectClient, get_client
from prefect.client.schemas.actions import ArtifactCreate as ArtifactRequest
from prefect.client.schemas.actions import ArtifactUpdate
from prefect.client.schemas.filters import ArtifactFilter, ArtifactFilterKey
from prefect.client.schemas.objects import Artifact as ArtifactResponse
from prefect.client.schemas.sorting import ArtifactSort
from prefect.context import MissingContextError, get_run_context
from prefect.logging.loggers import get_logger
from prefect.utilities.asyncutils import asyncnullcontext
from prefect.utilities.context import get_task_and_flow_run_ids

if TYPE_CHECKING:
    import logging

logger = get_logger('artifacts')

class Artifact(ArtifactRequest):
    async def acreate(self, client: Optional[PrefectClient] = None) -> ArtifactResponse:
        local_client_context = asyncnullcontext(client) if client else get_client()
        async with local_client_context as client:
            task_run_id, flow_run_id = get_task_and_flow_run_ids()
            try:
                get_run_context()
            except MissingContextError:
                warnings.warn('Artifact creation outside of a flow or task run is deprecated and will be removed in a later version.', FutureWarning)
            return await client.create_artifact(artifact=ArtifactRequest(type=self.type, key=self.key, description=self.description, task_run_id=self.task_run_id or task_run_id, flow_run_id=self.flow_run_id or flow_run_id, data=await self.aformat()))

    @async_dispatch(acreate)
    def create(self, client: Optional[PrefectClient] = None) -> ArtifactResponse:
        sync_client = get_client(sync_client=True)
        task_run_id, flow_run_id = get_task_and_flow_run_ids()
        try:
            get_run_context()
        except MissingContextError:
            warnings.warn('Artifact creation outside of a flow or task run is deprecated and will be removed in a later version.', FutureWarning)
        return sync_client.create_artifact(artifact=ArtifactRequest(type=self.type, key=self.key, description=self.description, task_run_id=self.task_run_id or task_run_id, flow_run_id=self.flow_run_id or flow_run_id, data=cast(str, self.format(_sync=True))))

    @classmethod
    async def aget(cls, key: Optional[str] = None, client: Optional[PrefectClient] = None) -> Optional[ArtifactResponse]:
        local_client_context = asyncnullcontext(client) if client else get_client()
        async with local_client_context as client:
            filter_key_value = None if key is None else [key]
            artifacts = await client.read_artifacts(limit=1, sort=ArtifactSort.UPDATED_DESC, artifact_filter=ArtifactFilter(key=ArtifactFilterKey(any_=filter_key_value)))
            return None if not artifacts else artifacts[0]

    @classmethod
    @async_dispatch(aget)
    def get(cls, key: Optional[str] = None, client: Optional[PrefectClient] = None) -> Optional[ArtifactResponse]:
        sync_client = get_client(sync_client=True)
        filter_key_value = None if key is None else [key]
        artifacts = sync_client.read_artifacts(limit=1, sort=ArtifactSort.UPDATED_DESC, artifact_filter=ArtifactFilter(key=ArtifactFilterKey(any_=filter_key_value)))
        return None if not artifacts else artifacts[0]

    @classmethod
    async def aget_or_create(cls, key: Optional[str] = None, description: Optional[str] = None, data: Optional[Any] = None, client: Optional[PrefectClient] = None, **kwargs: Any) -> tuple[ArtifactResponse, bool]:
        artifact = await cls.aget(key, client)
        if artifact:
            return (artifact, False)
        new_artifact = cls(key=key, description=description, data=data, **kwargs)
        created_artifact = await new_artifact.acreate(client)
        return (created_artifact, True)

    @classmethod
    @async_dispatch(aget_or_create)
    def get_or_create(cls, key: Optional[str] = None, description: Optional[str] = None, data: Optional[Any] = None, client: Optional[PrefectClient] = None, **kwargs: Any) -> tuple[ArtifactResponse, bool]:
        artifact = cast(ArtifactResponse, cls.get(key, _sync=True))
        if artifact:
            return (artifact, False)
        new_artifact = cls(key=key, description=description, data=data, **kwargs)
        created_artifact = cast(ArtifactResponse, new_artifact.create(_sync=True))
        return (created_artifact, True)

    async def aformat(self) -> str:
        return json.dumps(self.data)

    @async_dispatch(aformat)
    def format(self) -> str:
        return json.dumps(self.data)

class LinkArtifact(Artifact):
    link_text: Optional[str] = None
    type: str = 'markdown'

    def _format(self) -> str:
        return f'[{self.link_text}]({self.link})' if self.link_text else f'[{self.link}]({self.link})'

    async def aformat(self) -> str:
        return self._format()

    @async_dispatch(aformat)
    def format(self) -> str:
        return self._format()

class MarkdownArtifact(Artifact):
    type: str = 'markdown'

    async def aformat(self) -> str:
        return self.markdown

    @async_dispatch(aformat)
    def format(self) -> str:
        return self.markdown

class TableArtifact(Artifact):
    type: str = 'table'

    @classmethod
    def _sanitize(cls, item: Any) -> Any:
        if isinstance(item, list):
            return [cls._sanitize(sub_item) for sub_item in item]
        elif isinstance(item, dict):
            return {k: cls._sanitize(v) for k, v in item.items()}
        elif isinstance(item, float) and math.isnan(item):
            return None
        else:
            return item

    async def aformat(self) -> str:
        return json.dumps(self._sanitize(self.table))

    @async_dispatch(aformat)
    def format(self) -> str:
        return json.dumps(self._sanitize(self.table))

class ProgressArtifact(Artifact):
    type: str = 'progress'

    def _format(self) -> float:
        min_progress = 0.0
        max_progress = 100.0
        if self.progress < min_progress or self.progress > max_progress:
            logger.warning(f'ProgressArtifact received an invalid value, Progress: {self.progress}%')
            self.progress = max(min_progress, min(self.progress, max_progress))
            logger.warning(f'Interpreting as {self.progress}% progress')
        return self.progress

    async def aformat(self) -> float:
        return self._format()

    @async_dispatch(aformat)
    def format(self) -> float:
        return self._format()

class ImageArtifact(Artifact):
    type: str = 'image'

    async def aformat(self) -> str:
        return self.image_url

    @async_dispatch(aformat)
    def format(self) -> str:
        return self.image_url

async def acreate_link_artifact(link: str, link_text: Optional[str] = None, key: Optional[str] = None, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> UUID:
    new_artifact = LinkArtifact(key=key, description=description, link=link, link_text=link_text)
    artifact = await new_artifact.acreate(client)
    return artifact.id

@async_dispatch(acreate_link_artifact)
def create_link_artifact(link: str, link_text: Optional[str] = None, key: Optional[str] = None, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> UUID:
    new_artifact = LinkArtifact(key=key, description=description, link=link, link_text=link_text)
    artifact = cast(ArtifactResponse, new_artifact.create(_sync=True))
    return artifact.id

async def acreate_markdown_artifact(markdown: str, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    new_artifact = MarkdownArtifact(key=key, description=description, markdown=markdown)
    artifact = await new_artifact.acreate()
    return artifact.id

@async_dispatch(acreate_markdown_artifact)
def create_markdown_artifact(markdown: str, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    new_artifact = MarkdownArtifact(key=key, description=description, markdown=markdown)
    artifact = cast(ArtifactResponse, new_artifact.create(_sync=True))
    return artifact.id

async def acreate_table_artifact(table: Any, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    new_artifact = TableArtifact(key=key, description=description, table=table)
    artifact = await new_artifact.acreate()
    return artifact.id

@async_dispatch(acreate_table_artifact)
def create_table_artifact(table: Any, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    new_artifact = TableArtifact(key=key, description=description, table=table)
    artifact = cast(ArtifactResponse, new_artifact.create(_sync=True))
    return artifact.id

async def acreate_progress_artifact(progress: float, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    new_artifact = ProgressArtifact(key=key, description=description, progress=progress)
    artifact = await new_artifact.acreate()
    return artifact.id

@async_dispatch(acreate_progress_artifact)
def create_progress_artifact(progress: float, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    new_artifact = ProgressArtifact(key=key, description=description, progress=progress)
    artifact = cast(ArtifactResponse, new_artifact.create(_sync=True))
    return artifact.id

async def aupdate_progress_artifact(artifact_id: UUID, progress: float, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> UUID:
    local_client_context = nullcontext(client) if client else get_client()
    async with local_client_context as client:
        artifact = ProgressArtifact(description=description, progress=progress)
        update = ArtifactUpdate(description=artifact.description, data=await artifact.aformat()) if description else ArtifactUpdate(data=await artifact.aformat())
        await client.update_artifact(artifact_id=artifact_id, artifact=update)
        return artifact_id

@async_dispatch(aupdate_progress_artifact)
def update_progress_artifact(artifact_id: UUID, progress: float, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> UUID:
    sync_client = get_client(sync_client=True)
    artifact = ProgressArtifact(description=description, progress=progress)
    update = ArtifactUpdate(description=artifact.description, data=cast(float, artifact.format(_sync=True))) if description else ArtifactUpdate(data=cast(float, artifact.format(_sync=True)))
    sync_client.update_artifact(artifact_id=artifact_id, artifact=update)
    return artifact_id

async def acreate_image_artifact(image_url: str, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    new_artifact = ImageArtifact(key=key, description=description, image_url=image_url)
    artifact = await new_artifact.acreate()
    return artifact.id

@async_dispatch(acreate_image_artifact)
def create_image_artifact(image_url: str, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    new_artifact = ImageArtifact(key=key, description=description, image_url=image_url)
    artifact = cast(ArtifactResponse, new_artifact.create(_sync=True))
    return artifact.id
