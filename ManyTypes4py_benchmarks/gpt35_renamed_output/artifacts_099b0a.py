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
    async def func_4dc4nl8d(self, client: Optional[PrefectClient] = None) -> ArtifactResponse:
    
    @async_dispatch(acreate)
    def func_rto8w76v(self, client: Optional[PrefectClient] = None) -> ArtifactResponse:
    
    @classmethod
    async def func_ypekj1f2(cls, key: Optional[str] = None, client: Optional[PrefectClient] = None) -> Optional[ArtifactResponse]:
    
    @classmethod
    @async_dispatch(aget)
    def func_38lb0xah(cls, key: Optional[str] = None, client: Optional[PrefectClient] = None) -> Optional[ArtifactResponse]:
    
    @classmethod
    async def func_87132h8n(cls, key: Optional[str] = None, description: Optional[str] = None, data: Optional[Any] = None, client: Optional[PrefectClient] = None, **kwargs: Any) -> Tuple[ArtifactResponse, bool]:
    
    @classmethod
    @async_dispatch(aget_or_create)
    def func_6ujs1wo8(cls, key: Optional[str] = None, description: Optional[str] = None, data: Optional[Any] = None, client: Optional[PrefectClient] = None, **kwargs: Any) -> Tuple[ArtifactResponse, bool]:
    
    async def func_ije75yr8(self) -> str:
    
    @async_dispatch(aformat)
    def func_91esi2nz(self) -> str:


class LinkArtifact(Artifact):
    def func_85h0s1ip(self) -> str:
    
    async def func_ije75yr8(self) -> str:
    
    @async_dispatch(aformat)
    def func_91esi2nz(self) -> str:


class MarkdownArtifact(Artifact):
    async def func_ije75yr8(self) -> str:
    
    @async_dispatch(aformat)
    def func_91esi2nz(self) -> str:


class TableArtifact(Artifact):
    @classmethod
    def func_0ugfj1o8(cls, item: Any) -> Any:
    
    async def func_ije75yr8(self) -> str:
    
    @async_dispatch(aformat)
    def func_91esi2nz(self) -> str:


class ProgressArtifact(Artifact):
    def func_85h0s1ip(self) -> float:
    
    async def func_ije75yr8(self) -> str:
    
    @async_dispatch(aformat)
    def func_91esi2nz(self) -> str:


class ImageArtifact(Artifact):
    async def func_ije75yr8(self) -> str:
    
    @async_dispatch(aformat)
    def func_91esi2nz(self) -> str:


async def func_9hi7knju(link: str, link_text: Optional[str] = None, key: Optional[str] = None, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> UUID:
    
@async_dispatch(acreate_link_artifact)
def func_c32yw0ql(link: str, link_text: Optional[str] = None, key: Optional[str] = None, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> UUID:
    
async def func_i472ne3e(markdown: str, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    
@async_dispatch(acreate_markdown_artifact)
def func_vce9815q(markdown: str, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    
async def func_5jb1si62(table: Any, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    
@async_dispatch(acreate_table_artifact)
def func_8hkwu0qr(table: Any, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    
async def func_h5wy1uh0(progress: float, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    
@async_dispatch(acreate_progress_artifact)
def func_6z07d3g7(progress: float, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    
async def func_64ll5ci1(artifact_id: UUID, progress: float, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> UUID:
    
@async_dispatch(aupdate_progress_artifact)
def func_dvl3b83y(artifact_id: UUID, progress: float, description: Optional[str] = None, client: Optional[PrefectClient] = None) -> UUID:
    
async def func_2ky1fg8m(image_url: str, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
    
@async_dispatch(acreate_image_artifact)
def func_i3od9zus(image_url: str, key: Optional[str] = None, description: Optional[str] = None) -> UUID:
