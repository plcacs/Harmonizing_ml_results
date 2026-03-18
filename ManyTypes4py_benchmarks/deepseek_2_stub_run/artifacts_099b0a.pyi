```python
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Union, overload
from uuid import UUID

if TYPE_CHECKING:
    import logging
    from prefect.client.orchestration import PrefectClient
    from prefect.client.schemas.objects import Artifact as ArtifactResponse

class Artifact:
    type: str
    key: Optional[str]
    description: Optional[str]
    data: Any
    task_run_id: Optional[UUID]
    flow_run_id: Optional[UUID]
    
    def __init__(
        self,
        type: str,
        key: Optional[str] = ...,
        description: Optional[str] = ...,
        data: Any = ...,
        task_run_id: Optional[UUID] = ...,
        flow_run_id: Optional[UUID] = ...
    ) -> None: ...
    
    async def acreate(self, client: Optional[PrefectClient] = ...) -> ArtifactResponse: ...
    
    @overload
    def create(self, client: Optional[PrefectClient] = ...) -> ArtifactResponse: ...
    
    @overload
    def create(self, _sync: bool = ...) -> ArtifactResponse: ...
    
    @classmethod
    async def aget(
        cls,
        key: Optional[str] = ...,
        client: Optional[PrefectClient] = ...
    ) -> Optional[ArtifactResponse]: ...
    
    @classmethod
    @overload
    def get(
        cls,
        key: Optional[str] = ...,
        client: Optional[PrefectClient] = ...
    ) -> Optional[ArtifactResponse]: ...
    
    @classmethod
    @overload
    def get(
        cls,
        key: Optional[str] = ...,
        _sync: bool = ...
    ) -> Optional[ArtifactResponse]: ...
    
    @classmethod
    async def aget_or_create(
        cls,
        key: Optional[str] = ...,
        description: Optional[str] = ...,
        data: Any = ...,
        client: Optional[PrefectClient] = ...,
        **kwargs: Any
    ) -> tuple[ArtifactResponse, bool]: ...
    
    @classmethod
    @overload
    def get_or_create(
        cls,
        key: Optional[str] = ...,
        description: Optional[str] = ...,
        data: Any = ...,
        client: Optional[PrefectClient] = ...,
        **kwargs: Any
    ) -> tuple[ArtifactResponse, bool]: ...
    
    @classmethod
    @overload
    def get_or_create(
        cls,
        key: Optional[str] = ...,
        description: Optional[str] = ...,
        data: Any = ...,
        _sync: bool = ...,
        **kwargs: Any
    ) -> tuple[ArtifactResponse, bool]: ...
    
    async def aformat(self) -> str: ...
    
    @overload
    def format(self) -> str: ...
    
    @overload
    def format(self, _sync: bool = ...) -> str: ...

class LinkArtifact(Artifact):
    link_text: Optional[str]
    type: str
    link: str
    
    def __init__(
        self,
        key: Optional[str] = ...,
        description: Optional[str] = ...,
        link: str = ...,
        link_text: Optional[str] = ...,
        **kwargs: Any
    ) -> None: ...
    
    async def aformat(self) -> str: ...
    
    @overload
    def format(self) -> str: ...
    
    @overload
    def format(self, _sync: bool = ...) -> str: ...

class MarkdownArtifact(Artifact):
    type: str
    markdown: str
    
    def __init__(
        self,
        key: Optional[str] = ...,
        description: Optional[str] = ...,
        markdown: str = ...,
        **kwargs: Any
    ) -> None: ...
    
    async def aformat(self) -> str: ...
    
    @overload
    def format(self) -> str: ...
    
    @overload
    def format(self, _sync: bool = ...) -> str: ...

class TableArtifact(Artifact):
    type: str
    table: Any
    
    def __init__(
        self,
        key: Optional[str] = ...,
        description: Optional[str] = ...,
        table: Any = ...,
        **kwargs: Any
    ) -> None: ...
    
    async def aformat(self) -> str: ...
    
    @overload
    def format(self) -> str: ...
    
    @overload
    def format(self, _sync: bool = ...) -> str: ...

class ProgressArtifact(Artifact):
    type: str
    progress: float
    
    def __init__(
        self,
        key: Optional[str] = ...,
        description: Optional[str] = ...,
        progress: float = ...,
        **kwargs: Any
    ) -> None: ...
    
    async def aformat(self) -> float: ...
    
    @overload
    def format(self) -> float: ...
    
    @overload
    def format(self, _sync: bool = ...) -> float: ...

class ImageArtifact(Artifact):
    type: str
    image_url: str
    
    def __init__(
        self,
        key: Optional[str] = ...,
        description: Optional[str] = ...,
        image_url: str = ...,
        **kwargs: Any
    ) -> None: ...
    
    async def aformat(self) -> str: ...
    
    @overload
    def format(self) -> str: ...
    
    @overload
    def format(self, _sync: bool = ...) -> str: ...

async def acreate_link_artifact(
    link: str,
    link_text: Optional[str] = ...,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
    client: Optional[PrefectClient] = ...
) -> UUID: ...

@overload
def create_link_artifact(
    link: str,
    link_text: Optional[str] = ...,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
    client: Optional[PrefectClient] = ...
) -> UUID: ...

@overload
def create_link_artifact(
    link: str,
    link_text: Optional[str] = ...,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
    _sync: bool = ...
) -> UUID: ...

async def acreate_markdown_artifact(
    markdown: str,
    key: Optional[str] = ...,
    description: Optional[str] = ...
) -> UUID: ...

@overload
def create_markdown_artifact(
    markdown: str,
    key: Optional[str] = ...,
    description: Optional[str] = ...
) -> UUID: ...

@overload
def create_markdown_artifact(
    markdown: str,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
    _sync: bool = ...
) -> UUID: ...

async def acreate_table_artifact(
    table: Any,
    key: Optional[str] = ...,
    description: Optional[str] = ...
) -> UUID: ...

@overload
def create_table_artifact(
    table: Any,
    key: Optional[str] = ...,
    description: Optional[str] = ...
) -> UUID: ...

@overload
def create_table_artifact(
    table: Any,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
    _sync: bool = ...
) -> UUID: ...

async def acreate_progress_artifact(
    progress: float,
    key: Optional[str] = ...,
    description: Optional[str] = ...
) -> UUID: ...

@overload
def create_progress_artifact(
    progress: float,
    key: Optional[str] = ...,
    description: Optional[str] = ...
) -> UUID: ...

@overload
def create_progress_artifact(
    progress: float,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
    _sync: bool = ...
) -> UUID: ...

async def aupdate_progress_artifact(
    artifact_id: UUID,
    progress: float,
    description: Optional[str] = ...,
    client: Optional[PrefectClient] = ...
) -> UUID: ...

@overload
def update_progress_artifact(
    artifact_id: UUID,
    progress: float,
    description: Optional[str] = ...,
    client: Optional[PrefectClient] = ...
) -> UUID: ...

@overload
def update_progress_artifact(
    artifact_id: UUID,
    progress: float,
    description: Optional[str] = ...,
    _sync: bool = ...
) -> UUID: ...

async def acreate_image_artifact(
    image_url: str,
    key: Optional[str] = ...,
    description: Optional[str] = ...
) -> UUID: ...

@overload
def create_image_artifact(
    image_url: str,
    key: Optional[str] = ...,
    description: Optional[str] = ...
) -> UUID: ...

@overload
def create_image_artifact(
    image_url: str,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
    _sync: bool = ...
) -> UUID: ...

logger: logging.Logger = ...
```