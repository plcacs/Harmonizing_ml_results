"""
Stub file for 'artifacts_099b0a' module
"""

from __future__ import annotations
from typing import Any, Optional, Union, ClassVar, Tuple, List, Dict, Any
from uuid import UUID
from prefect.client.schemas.objects import Artifact as ArtifactResponse
from prefect.client.schemas.actions import ArtifactCreate as ArtifactRequest
from prefect.client.schemas.actions import ArtifactUpdate
from prefect.client.schemas.filters import ArtifactFilter, ArtifactFilterKey
from prefect.client.schemas.sorting import ArtifactSort

class Artifact(ArtifactRequest):
    type: str
    key: str
    description: Optional[str]
    data: Any
    task_run_id: Optional[UUID]
    flow_run_id: Optional[UUID]

    async def acreate(self, client: Optional[PrefectClient] = None) -> Self: ...
    def create(self, client: Optional[PrefectClient] = None) -> Self: ...

    @classmethod
    async def aget(cls, key: Optional[str] = None, client: Optional[PrefectClient] = None) -> Optional[ArtifactResponse]: ...
    @classmethod
    def get(cls, key: Optional[str] = None, client: Optional[PrefectClient] = None) -> Optional[ArtifactResponse]: ...

    @classmethod
    async def aget_or_create(cls, key: Optional[str] = None, description: Optional[str] = None, data: Any = None, client: Optional[PrefectClient] = None, **kwargs: Any) -> Tuple[ArtifactResponse, bool]: ...
    @classmethod
    def get_or_create(cls, key: Optional[str] = None, description: Optional[str] = None, data: Any = None, client: Optional[PrefectClient] = None, **kwargs: Any) -> Tuple[ArtifactResponse, bool]: ...

    async def aformat(self) -> str: ...
    def format(self) -> str: ...

class LinkArtifact(Artifact):
    link: str
    link_text: Optional[str]

class MarkdownArtifact(Artifact):
    markdown: str

class TableArtifact(Artifact):
    table: Any

    @classmethod
    def _sanitize(cls, item: Union[dict, list, float]) -> Union[dict, list, None, float]: ...

class ProgressArtifact(Artifact):
    progress: float

class ImageArtifact(Artifact):
    image_url: str

async def acreate_link_artifact(link: str, link_text: Optional[str] = None, key: Optional[str] = None, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> UUID: ...
def create_link_artifact(link: str, link_text: Optional[str] = None, key: Optional[str] = None, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> UUID: ...

async def acreate_markdown_artifact(markdown: str, key: Optional[str] = None, description: Optional[str] = None) -> UUID: ...
def create_markdown_artifact(markdown: str, key: Optional[str] = None, description: Optional[str] = None) -> UUID: ...

async def acreate_table_artifact(table: Any, key: Optional[str] = None, description: Optional[str] = None) -> UUID: ...
def create_table_artifact(table: Any, key: Optional[str] = None, description: Optional[str] = None) -> UUID: ...

async def acreate_progress_artifact(progress: float, key: Optional[str] = None, description: Optional[str] = None) -> UUID: ...
def create_progress_artifact(progress: float, key: Optional[str] = None, description: Optional[str] = None) -> UUID: ...

async def aupdate_progress_artifact(artifact_id: UUID, progress: float, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> UUID: ...
def update_progress_artifact(artifact_id: UUID, progress: float, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> UUID: ...

async def acreate_image_artifact(image_url: str, key: Optional[str] = None, description: Optional[str] = None) -> UUID: ...
def create_image_artifact(image_url: str, key: Optional[str] = None, description: Optional[str] = None) -> UUID: ...