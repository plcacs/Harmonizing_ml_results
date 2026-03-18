```python
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Union, Tuple
from uuid import UUID
from typing_extensions import Self

if TYPE_CHECKING:
    import logging
    from prefect.client.orchestration import PrefectClient
    from prefect.client.schemas.objects import Artifact as ArtifactResponse

class Artifact:
    type: str
    key: str
    description: Optional[str]
    data: Any
    task_run_id: Optional[UUID]
    flow_run_id: Optional[UUID]
    
    def __init__(
        self,
        type: str,
        key: str,
        description: Optional[str] = ...,
        data: Any = ...,
        **kwargs: Any
    ) -> None: ...
    
    async def acreate(self, client: Optional[Any] = ...) -> ArtifactResponse: ...
    
    def create(self, client: Optional[Any] = ..., _sync: bool = ...) -> ArtifactResponse: ...
    
    @classmethod
    async def aget(
        cls,
        key: Optional[str] = ...,
        client: Optional[Any] = ...
    ) -> Optional[ArtifactResponse]: ...
    
    @classmethod
    def get(
        cls,
        key: Optional[str] = ...,
        client: Optional[Any] = ...,
        _sync: bool = ...
    ) -> Optional[ArtifactResponse]: ...
    
    @classmethod
    async def aget_or_create(
        cls,
        key: Optional[str] = ...,
        description: Optional[str] = ...,
        data: Any = ...,
        client: Optional[Any] = ...,
        **kwargs: Any
    ) -> Tuple[ArtifactResponse, bool]: ...
    
    @classmethod
    def get_or_create(
        cls,
        key: Optional[str] = ...,
        description: Optional[str] = ...,
        data: Any = ...,
        client: Optional[Any] = ...,
        **kwargs: Any,
        _sync: bool = ...
    ) -> Tuple[ArtifactResponse, bool]: ...
    
    async def aformat(self) -> str: ...
    
    def format(self, _sync: bool = ...) -> str: ...

class LinkArtifact(Artifact):
    link_text: Optional[str]
    link: str
    type: str
    
    def __init__(
        self,
        key: str,
        description: Optional[str] = ...,
        link: str = ...,
        link_text: Optional[str] = ...,
        **kwargs: Any
    ) -> None: ...
    
    async def aformat(self) -> str: ...
    
    def format(self, _sync: bool = ...) -> str: ...

class MarkdownArtifact(Artifact):
    markdown: str
    type: str
    
    def __init__(
        self,
        key: str,
        description: Optional[str] = ...,
        markdown: str = ...,
        **kwargs: Any
    ) -> None: ...
    
    async def aformat(self) -> str: ...
    
    def format(self, _sync: bool = ...) -> str: ...

class TableArtifact(Artifact):
    table: Any
    type: str
    
    def __init__(
        self,
        key: str,
        description: Optional[str] = ...,
        table: Any = ...,
        **kwargs: Any
    ) -> None: ...
    
    async def aformat(self) -> str: ...
    
    def format(self, _sync: bool = ...) -> str: ...

class ProgressArtifact(Artifact):
    progress: float
    type: str
    
    def __init__(
        self,
        key: str,
        description: Optional[str] = ...,
        progress: float = ...,
        **kwargs: Any
    ) -> None: ...
    
    async def aformat(self) -> float: ...
    
    def format(self, _sync: bool = ...) -> float: ...

class ImageArtifact(Artifact):
    image_url: str
    type: str
    
    def __init__(
        self,
        key: str,
        description: Optional[str] = ...,
        image_url: str = ...,
        **kwargs: Any
    ) -> None: ...
    
    async def aformat(self) -> str: ...
    
    def format(self, _sync: bool = ...) -> str: ...

async def acreate_link_artifact(
    link: str,
    link_text: Optional[str] = ...,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
    client: Optional[Any] = ...
) -> UUID: ...

def create_link_artifact(
    link: str,
    link_text: Optional[str] = ...,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
    client: Optional[Any] = ...,
    _sync: bool = ...
) -> UUID: ...

async def acreate_markdown_artifact(
    markdown: str,
    key: Optional[str] = ...,
    description: Optional[str] = ...
) -> UUID: ...

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
    client: Optional[Any] = ...
) -> UUID: ...

def update_progress_artifact(
    artifact_id: UUID,
    progress: float,
    description: Optional[str] = ...,
    client: Optional[Any] = ...,
    _sync: bool = ...
) -> UUID: ...

async def acreate_image_artifact(
    image_url: str,
    key: Optional[str] = ...,
    description: Optional[str] = ...
) -> UUID: ...

def create_image_artifact(
    image_url: str,
    key: Optional[str] = ...,
    description: Optional[str] = ...,
    _sync: bool = ...
) -> UUID: ...

logger: logging.Logger = ...
```