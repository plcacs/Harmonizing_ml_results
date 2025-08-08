"""
Interface for creating and reading artifacts.
"""
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
    """
    An artifact is a piece of data that is created by a flow or task run.
    https://docs.prefect.io/latest/develop/artifacts

    Arguments:
        type: A string identifying the type of artifact.
        key: A user-provided string identifier.
          The key must only contain lowercase letters, numbers, and dashes.
        description: A user-specified description of the artifact.
        data: A JSON payload that allows for a result to be retrieved.
    """

    async def acreate(self, client=None):
        """
        An async method to create an artifact.

        Arguments:
            client: The PrefectClient

        Returns:
            - The created artifact.
        """
        local_client_context = asyncnullcontext(client) if client else get_client()
        async with local_client_context as client:
            task_run_id, flow_run_id = get_task_and_flow_run_ids()
            try:
                get_run_context()
            except MissingContextError:
                warnings.warn('Artifact creation outside of a flow or task run is deprecated and will be removed in a later version.', FutureWarning)
            return await client.create_artifact(artifact=ArtifactRequest(type=self.type, key=self.key, description=self.description, task_run_id=self.task_run_id or task_run_id, flow_run_id=self.flow_run_id or flow_run_id, data=await self.aformat()))

    @async_dispatch(acreate)
    def create(self, client=None):
        """
        A method to create an artifact.

        Arguments:
            client: The PrefectClient

        Returns:
            - The created artifact.
        """
        sync_client = get_client(sync_client=True)
        task_run_id, flow_run_id = get_task_and_flow_run_ids()
        try:
            get_run_context()
        except MissingContextError:
            warnings.warn('Artifact creation outside of a flow or task run is deprecated and will be removed in a later version.', FutureWarning)
        return sync_client.create_artifact(artifact=ArtifactRequest(type=self.type, key=self.key, description=self.description, task_run_id=self.task_run_id or task_run_id, flow_run_id=self.flow_run_id or flow_run_id, data=cast(str, self.format(_sync=True))))

    @classmethod
    async def aget(cls, key=None, client=None):
        """
        A async method to get an artifact.

        Arguments:
            key: The key of the artifact to get.
            client: A client to use when calling the Prefect API.

        Returns:
            The artifact (if found).
        """
        local_client_context = asyncnullcontext(client) if client else get_client()
        async with local_client_context as client:
            filter_key_value = None if key is None else [key]
            artifacts = await client.read_artifacts(limit=1, sort=ArtifactSort.UPDATED_DESC, artifact_filter=ArtifactFilter(key=ArtifactFilterKey(any_=filter_key_value)))
            return None if not artifacts else artifacts[0]

    @classmethod
    @async_dispatch(aget)
    def get(cls, key=None, client=None):
        """
        A method to get an artifact.

        Arguments:
            key: The key of the artifact to get.
            client: A client to use when calling the Prefect API.

        Returns:
            The artifact (if found).
        """
        sync_client = get_client(sync_client=True)
        filter_key_value = None if key is None else [key]
        artifacts = sync_client.read_artifacts(limit=1, sort=ArtifactSort.UPDATED_DESC, artifact_filter=ArtifactFilter(key=ArtifactFilterKey(any_=filter_key_value)))
        return None if not artifacts else artifacts[0]

    @classmethod
    async def aget_or_create(cls, key=None, description=None, data=None, client=None, **kwargs):
        """
        A async method to get or create an artifact.

        Arguments:
            key: The key of the artifact to get or create.
            description: The description of the artifact to create.
            data: The data of the artifact to create.
            client: The PrefectClient
            **kwargs: Additional keyword arguments to use when creating the artifact.

        Returns:
            The artifact, either retrieved or created.
        """
        artifact = await cls.aget(key, client)
        if artifact:
            return (artifact, False)
        new_artifact = cls(key=key, description=description, data=data, **kwargs)
        created_artifact = await new_artifact.acreate(client)
        return (created_artifact, True)

    @classmethod
    @async_dispatch(aget_or_create)
    def get_or_create(cls, key=None, description=None, data=None, client=None, **kwargs):
        """
        A method to get or create an artifact.

        Arguments:
            key: The key of the artifact to get or create.
            description: The description of the artifact to create.
            data: The data of the artifact to create.
            client: The PrefectClient
            **kwargs: Additional keyword arguments to use when creating the artifact.

        Returns:
            The artifact, either retrieved or created.
        """
        artifact = cast(ArtifactResponse, cls.get(key, _sync=True))
        if artifact:
            return (artifact, False)
        new_artifact = cls(key=key, description=description, data=data, **kwargs)
        created_artifact = cast(ArtifactResponse, new_artifact.create(_sync=True))
        return (created_artifact, True)

    async def aformat(self):
        return json.dumps(self.data)

    @async_dispatch(aformat)
    def format(self):
        return json.dumps(self.data)

class LinkArtifact(Artifact):
    link_text = None
    type = 'markdown'

    def _format(self):
        return f'[{self.link_text}]({self.link})' if self.link_text else f'[{self.link}]({self.link})'

    async def aformat(self):
        return self._format()

    @async_dispatch(aformat)
    def format(self):
        return self._format()

class MarkdownArtifact(Artifact):
    type = 'markdown'

    async def aformat(self):
        return self.markdown

    @async_dispatch(aformat)
    def format(self):
        return self.markdown

class TableArtifact(Artifact):
    type = 'table'

    @classmethod
    def _sanitize(cls, item):
        """
        Sanitize NaN values in a given item.
        The item can be a dict, list or float.
        """
        if isinstance(item, list):
            return [cls._sanitize(sub_item) for sub_item in item]
        elif isinstance(item, dict):
            return {k: cls._sanitize(v) for k, v in item.items()}
        elif isinstance(item, float) and math.isnan(item):
            return None
        else:
            return item

    async def aformat(self):
        return json.dumps(self._sanitize(self.table))

    @async_dispatch(aformat)
    def format(self):
        return json.dumps(self._sanitize(self.table))

class ProgressArtifact(Artifact):
    type = 'progress'

    def _format(self):
        min_progress = 0.0
        max_progress = 100.0
        if self.progress < min_progress or self.progress > max_progress:
            logger.warning(f'ProgressArtifact received an invalid value, Progress: {self.progress}%')
            self.progress = max(min_progress, min(self.progress, max_progress))
            logger.warning(f'Interpreting as {self.progress}% progress')
        return self.progress

    async def aformat(self):
        return self._format()

    @async_dispatch(aformat)
    def format(self):
        return self._format()

class ImageArtifact(Artifact):
    """
    An artifact that will display an image from a publicly accessible URL in the UI.

    Arguments:
        image_url: The URL of the image to display.
    """
    type = 'image'

    async def aformat(self):
        return self.image_url

    @async_dispatch(aformat)
    def format(self):
        """
        This method is used to format the artifact data so it can be properly sent
        to the API when the .create() method is called.

        Returns:
            str: The image URL.
        """
        return self.image_url

async def acreate_link_artifact(link, link_text=None, key=None, description=None, client=None):
    """
    Create a link artifact.

    Arguments:
        link: The link to create.
        link_text: The link text.
        key: A user-provided string identifier.
          Required for the artifact to show in the Artifacts page in the UI.
          The key must only contain lowercase letters, numbers, and dashes.
        description: A user-specified description of the artifact.


    Returns:
        The table artifact ID.
    """
    new_artifact = LinkArtifact(key=key, description=description, link=link, link_text=link_text)
    artifact = await new_artifact.acreate(client)
    return artifact.id

@async_dispatch(acreate_link_artifact)
def create_link_artifact(link, link_text=None, key=None, description=None, client=None):
    """
    Create a link artifact.

    Arguments:
        link: The link to create.
        link_text: The link text.
        key: A user-provided string identifier.
          Required for the artifact to show in the Artifacts page in the UI.
          The key must only contain lowercase letters, numbers, and dashes.
        description: A user-specified description of the artifact.


    Returns:
        The table artifact ID.
    """
    new_artifact = LinkArtifact(key=key, description=description, link=link, link_text=link_text)
    artifact = cast(ArtifactResponse, new_artifact.create(_sync=True))
    return artifact.id

async def acreate_markdown_artifact(markdown, key=None, description=None):
    """
    Create a markdown artifact.

    Arguments:
        markdown: The markdown to create.
        key: A user-provided string identifier.
          Required for the artifact to show in the Artifacts page in the UI.
          The key must only contain lowercase letters, numbers, and dashes.
        description: A user-specified description of the artifact.

    Returns:
        The table artifact ID.
    """
    new_artifact = MarkdownArtifact(key=key, description=description, markdown=markdown)
    artifact = await new_artifact.acreate()
    return artifact.id

@async_dispatch(acreate_markdown_artifact)
def create_markdown_artifact(markdown, key=None, description=None):
    """
    Create a markdown artifact.

    Arguments:
        markdown: The markdown to create.
        key: A user-provided string identifier.
          Required for the artifact to show in the Artifacts page in the UI.
          The key must only contain lowercase letters, numbers, and dashes.
        description: A user-specified description of the artifact.

    Returns:
        The table artifact ID.
    """
    new_artifact = MarkdownArtifact(key=key, description=description, markdown=markdown)
    artifact = cast(ArtifactResponse, new_artifact.create(_sync=True))
    return artifact.id

async def acreate_table_artifact(table, key=None, description=None):
    """
    Create a table artifact asynchronously.

    Arguments:
        table: The table to create.
        key: A user-provided string identifier.
          Required for the artifact to show in the Artifacts page in the UI.
          The key must only contain lowercase letters, numbers, and dashes.
        description: A user-specified description of the artifact.

    Returns:
        The table artifact ID.
    """
    new_artifact = TableArtifact(key=key, description=description, table=table)
    artifact = await new_artifact.acreate()
    return artifact.id

@async_dispatch(acreate_table_artifact)
def create_table_artifact(table, key=None, description=None):
    """
    Create a table artifact.

    Arguments:
        table: The table to create.
        key: A user-provided string identifier.
          Required for the artifact to show in the Artifacts page in the UI.
          The key must only contain lowercase letters, numbers, and dashes.
        description: A user-specified description of the artifact.

    Returns:
        The table artifact ID.
    """
    new_artifact = TableArtifact(key=key, description=description, table=table)
    artifact = cast(ArtifactResponse, new_artifact.create(_sync=True))
    return artifact.id

async def acreate_progress_artifact(progress, key=None, description=None):
    """
    Create a progress artifact asynchronously.

    Arguments:
        progress: The percentage of progress represented by a float between 0 and 100.
        key: A user-provided string identifier.
          Required for the artifact to show in the Artifacts page in the UI.
          The key must only contain lowercase letters, numbers, and dashes.
        description: A user-specified description of the artifact.

    Returns:
        The progress artifact ID.
    """
    new_artifact = ProgressArtifact(key=key, description=description, progress=progress)
    artifact = await new_artifact.acreate()
    return artifact.id

@async_dispatch(acreate_progress_artifact)
def create_progress_artifact(progress, key=None, description=None):
    """
    Create a progress artifact.

    Arguments:
        progress: The percentage of progress represented by a float between 0 and 100.
        key: A user-provided string identifier.
          Required for the artifact to show in the Artifacts page in the UI.
          The key must only contain lowercase letters, numbers, and dashes.
        description: A user-specified description of the artifact.

    Returns:
        The progress artifact ID.
    """
    new_artifact = ProgressArtifact(key=key, description=description, progress=progress)
    artifact = cast(ArtifactResponse, new_artifact.create(_sync=True))
    return artifact.id

async def aupdate_progress_artifact(artifact_id, progress, description=None, client=None):
    """
    Update a progress artifact asynchronously.

    Arguments:
        artifact_id: The ID of the artifact to update.
        progress: The percentage of progress represented by a float between 0 and 100.
        description: A user-specified description of the artifact.

    Returns:
        The progress artifact ID.
    """
    local_client_context = nullcontext(client) if client else get_client()
    async with local_client_context as client:
        artifact = ProgressArtifact(description=description, progress=progress)
        update = ArtifactUpdate(description=artifact.description, data=await artifact.aformat()) if description else ArtifactUpdate(data=await artifact.aformat())
        await client.update_artifact(artifact_id=artifact_id, artifact=update)
        return artifact_id

@async_dispatch(aupdate_progress_artifact)
def update_progress_artifact(artifact_id, progress, description=None, client=None):
    """
    Update a progress artifact.

    Arguments:
        artifact_id: The ID of the artifact to update.
        progress: The percentage of progress represented by a float between 0 and 100.
        description: A user-specified description of the artifact.

    Returns:
        The progress artifact ID.
    """
    sync_client = get_client(sync_client=True)
    artifact = ProgressArtifact(description=description, progress=progress)
    update = ArtifactUpdate(description=artifact.description, data=cast(float, artifact.format(_sync=True))) if description else ArtifactUpdate(data=cast(float, artifact.format(_sync=True)))
    sync_client.update_artifact(artifact_id=artifact_id, artifact=update)
    return artifact_id

async def acreate_image_artifact(image_url, key=None, description=None):
    """
    Create an image artifact asynchronously.

    Arguments:
        image_url: The URL of the image to display.
        key: A user-provided string identifier.
          Required for the artifact to show in the Artifacts page in the UI.
          The key must only contain lowercase letters, numbers, and dashes.
        description: A user-specified description of the artifact.

    Returns:
        The image artifact ID.
    """
    new_artifact = ImageArtifact(key=key, description=description, image_url=image_url)
    artifact = await new_artifact.acreate()
    return artifact.id

@async_dispatch(acreate_image_artifact)
def create_image_artifact(image_url, key=None, description=None):
    """
    Create an image artifact.

    Arguments:
        image_url: The URL of the image to display.
        key: A user-provided string identifier.
          Required for the artifact to show in the Artifacts page in the UI.
          The key must only contain lowercase letters, numbers, and dashes.
        description: A user-specified description of the artifact.

    Returns:
        The image artifact ID.
    """
    new_artifact = ImageArtifact(key=key, description=description, image_url=image_url)
    artifact = cast(ArtifactResponse, new_artifact.create(_sync=True))
    return artifact.id