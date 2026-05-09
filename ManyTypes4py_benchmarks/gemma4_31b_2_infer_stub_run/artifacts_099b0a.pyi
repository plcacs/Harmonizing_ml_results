"""
Interface for creating and reading artifacts.
"""
from __future__ import annotations

from typing import Any, Optional, Union, Tuple, overload
from uuid import UUID
from prefect.client.orchestration import PrefectClient
from prefect.client.schemas.actions import ArtifactCreate as ArtifactRequest
from prefect.client.schemas.objects import Artifact as ArtifactResponse

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
    async def acreate(self, client: Optional[PrefectClient] = ...) -> ArtifactResponse: ...
    def create(self, client: Optional[PrefectClient] = ...) -> ArtifactResponse: ...
    @classmethod
    async def aget(cls, key: Optional[str] = ..., client: Optional[PrefectClient] = ...) -> Optional[ArtifactResponse]: ...
    @classmethod
    def get(cls, key: Optional[str] = ..., client: Optional[PrefectClient] = ...) -> Optional[ArtifactResponse]: ...
    @classmethod
    async def aget_or_create(
        cls,
        key: Optional[str] = ...,
        description: Optional[str] = ...,
        data: Optional[Any] = ...,
        client: Optional[PrefectClient] = ...,
        **kwargs: Any,
    ) -> Tuple[ArtifactResponse, bool]: ...
    @classmethod
    def get_or_create(
        cls,
        key: Optional[str] = ...,
        description: Optional[str] = ...,
        data: Optional[Any] = ...,
        client: Optional[PrefectClient] = ...,
        **kwargs: Any,
    ) -> Tuple[ArtifactResponse, bool]: ...
    async def aformat(self) -> str: ...
    def format(self) -> str: ...

class LinkArtifact(Artifact):
    link_text: Optional[str]
    type: str
    link: str
    def _format(self) -> str: ...
    async def aformat(self) -> str: ...
    def format(self) -> str: ...

class MarkdownArtifact(Artifact):
    type: str
    markdown: str
    async def aformat(self) -> str: ...
    def format(self) -> str: ...

class TableArtifact(Artifact):
    type: str
    table: Any
    @classmethod
    def _sanitize(cls, item: Any) -> Any: ...
    async def aformat(self) -> str: ...
    def format(self) -> str: ...

class ProgressArtifact(Artifact):
    type: str
    progress: float
    def _format(self) -> float: ...
    async def aformat(self) -> float: ...
    def format(self) -> float: ...

class ImageArtifact(Artifact):
    """
    An artifact that will display an image from a publicly accessible URL in the UI.

    Arguments:
        image_url: The URL of the image to display.
    """
    type: str
    image_url: str
    async def aformat(self) -> str: ...
    def format(self) -> str: ...

async def acreate_link_artifact(
    link: str,
    link_text: Optional[str] = ...,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
    client: Optional[PrefectClient] = ...,
) -> UUID: ...

def create_link_artifact(
    link: str,
    link_text: Optional[str] = ...,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
    client: Optional[PrefectClient] = ...,
) -> UUID: ...

async def acreate_markdown_artifact(
    markdown: str,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
) -> UUID: ...

def create_markdown_artifact(
    markdown: str,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
) -> UUID: ...

async def acreate_table_artifact(
    table: Any,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
) -> UUID: ...

def create_table_artifact(
    table: Any,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
) -> UUID: ...

async def acreate_progress_artifact(
    progress: float,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
) -> UUID: ...

def create_progress_artifact(
    progress: float,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
) -> UUID: ...

async def aupdate_progress_artifact(
    artifact_id: UUID,
    progress: float,
    description: Optional[str] = ...,
    client: Optional[PrefectClient] = ...,
) -> UUID: ...

def update_progress_artifact(
    artifact_id: UUID,
    progress: float,
    description: Optional[str] = ...,
    client: Optional[PrefectClient] = ...,
) -> UUID: ...

async def acreate_image_artifact(
    image_url: str,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
) -> UUID: ...

def create_image_artifact(
    image_url: str,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
) -> UUID: ...