"""
Stub file for 'artifacts_099b0a' module
"""

from __future__ import annotations
from typing import Any, Optional, Union, List, Tuple, Dict, AnyStr, overload
from typing_extensions import Self
from prefect.client.schemas.objects import Artifact as ArtifactResponse
from prefect.client.orchestration import PrefectClient
from prefect.client.schemas.actions import ArtifactUpdate
from prefect.client.schemas.filters import ArtifactFilter, ArtifactFilterKey
from prefect.client.schemas.sorting import ArtifactSort
from prefect.context import MissingContextError
from prefect.utilities.asyncutils import asyncnullcontext
from prefect.utilities.context import get_task_and_flow_run_ids
from uuid import UUID
import json
import math
import warnings
import logging

class Artifact(ArtifactResponse):
    type: str
    key: str
    description: Optional[str]
    data: Any
    task_run_id: Optional[UUID]
    flow_run_id: Optional[UUID]

    async def acreate(self, client: Optional[PrefectClient] = None) -> ArtifactResponse:
        ...

    def create(self, client: Optional[PrefectClient] = None) -> ArtifactResponse:
        ...

    @classmethod
    async def aget(cls, key: Optional[str] = None, client: Optional[PrefectClient] = None) -> Optional[ArtifactResponse]:
        ...

    @classmethod
    def get(cls, key: Optional[str] = None, client: Optional[PrefectClient] = None) -> Optional[ArtifactResponse]:
        ...

    @classmethod
    async def aget_or_create(cls, key: Optional[str] = None, description: Optional[str] = None, data: Any = None, client: Optional[PrefectClient] = None, **kwargs: Any) -> Tuple[ArtifactResponse, bool]:
        ...

    @classmethod
    def get_or_create(cls, key: Optional[str] = None, description: Optional[str] = None, data: Any = None, client: Optional[PrefectClient] = None, **kwargs: Any) -> Tuple[ArtifactResponse, bool]:
        ...

    async def aformat(self) -> str:
        ...

    def format(self) -> str:
        ...

class LinkArtifact(Artifact):
    link: str
    link_text: Optional[str]
    type: str = 'markdown'

    def _format(self) -> str:
        ...

    async def aformat(self) -> str:
        ...

    def format(self) -> str:
        ...

class MarkdownArtifact(Artifact):
    type: str = 'markdown'
    markdown: str

    async def aformat(self) -> str:
        ...

    def format(self) -> str:
        ...

class TableArtifact(Artifact):
    type: str = 'table'
    table: Any

    @classmethod
    def _sanitize(cls, item: Any) -> Any:
        ...

    async def aformat(self) -> str:
        ...

    def format(self) -> str:
        ...

class ProgressArtifact(Artifact):
    type: str = 'progress'
    progress: float

    def _format(self) -> float:
        ...

    async def aformat(self) -> float:
        ...

    def format(self) -> float:
        ...

class ImageArtifact(Artifact):
    type: str = 'image'
    image_url: str

    async def aformat(self) -> str:
        ...

    def format(self) -> str:
        ...

async def acreate_link_artifact(link: str, link_text: Optional[str] = None, key: Optional[str] = None, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> str:
    ...

def create_link_artifact(link: str, link_text: Optional[str] = None, key: Optional[str] = None, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> str:
    ...

async def acreate_markdown_artifact(markdown: str, key: Optional[str] = None, description: Optional[str] = None) -> str:
    ...

def create_markdown_artifact(markdown: str, key: Optional[str] = None, description: Optional[str] = None) -> str:
    ...

async def acreate_table_artifact(table: Any, key: Optional[str] = None, description: Optional[str] = None) -> str:
    ...

def create_table_artifact(table: Any, key: Optional[str] = None, description: Optional[str] = None) -> str:
    ...

async def acreate_progress_artifact(progress: float, key: Optional[str] = None, description: Optional[str] = None) -> str:
    ...

def create_progress_artifact(progress: float, key: Optional[str] = None, description: Optional[str] = None) -> str:
    ...

async def aupdate_progress_artifact(artifact_id: str, progress: float, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> str:
    ...

def update_progress_artifact(artifact_id: str, progress: float, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> str:
    ...

async def acreate_image_artifact(image_url: str, key: Optional[str] = None, description: Optional[str] = None) -> str:
    ...

def create_image_artifact(image_url: str, key: Optional[str] = None, description: Optional[str] = None) -> str:
    ...