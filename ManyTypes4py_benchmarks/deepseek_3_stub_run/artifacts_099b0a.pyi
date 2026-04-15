"""
Interface for creating and reading artifacts.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Union, overload
from uuid import UUID
from typing_extensions import Self

if TYPE_CHECKING:
    import logging
    from prefect.client.orchestration import PrefectClient
    from prefect.client.schemas.objects import Artifact as ArtifactResponse

logger: logging.Logger = ...

class Artifact:
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
    type: str
    key: Optional[str]
    description: Optional[str]
    data: Any
    task_run_id: Optional[UUID]
    flow_run_id: Optional[UUID]
    
    def __init__(
        self,
        type: str,
        key: Optional[str] = None,
        description: Optional[str] = None,
        data: Any = None,
        **kwargs: Any
    ) -> None: ...
    
    async def acreate(self, client: Optional[PrefectClient] = None) -> ArtifactResponse: ...
    
    @overload
    def create(self, client: Optional[PrefectClient] = None) -> ArtifactResponse: ...
    
    @overload
    def create(self, _sync: bool = True) -> ArtifactResponse: ...
    
    def create(self, client: Optional[PrefectClient] = None, _sync: bool = True) -> ArtifactResponse: ...
    
    @classmethod
    async def aget(
        cls,
        key: Optional[str] = None,
        client: Optional[PrefectClient] = None
    ) -> Optional[ArtifactResponse]: ...
    
    @classmethod
    @overload
    def get(
        cls,
        key: Optional[str] = None,
        client: Optional[PrefectClient] = None
    ) -> Optional[ArtifactResponse]: ...
    
    @classmethod
    @overload
    def get(
        cls,
        key: Optional[str] = None,
        _sync: bool = True
    ) -> Optional[ArtifactResponse]: ...
    
    @classmethod
    def get(
        cls,
        key: Optional[str] = None,
        client: Optional[PrefectClient] = None,
        _sync: bool = True
    ) -> Optional[ArtifactResponse]: ...
    
    @classmethod
    async def aget_or_create(
        cls,
        key: Optional[str] = None,
        description: Optional[str] = None,
        data: Any = None,
        client: Optional[PrefectClient] = None,
        **kwargs: Any
    ) -> tuple[ArtifactResponse, bool]: ...
    
    @classmethod
    @overload
    def get_or_create(
        cls,
        key: Optional[str] = None,
        description: Optional[str] = None,
        data: Any = None,
        client: Optional[PrefectClient] = None,
        **kwargs: Any
    ) -> tuple[ArtifactResponse, bool]: ...
    
    @classmethod
    @overload
    def get_or_create(
        cls,
        key: Optional[str] = None,
        description: Optional[str] = None,
        data: Any = None,
        _sync: bool = True,
        **kwargs: Any
    ) -> tuple[ArtifactResponse, bool]: ...
    
    @classmethod
    def get_or_create(
        cls,
        key: Optional[str] = None,
        description: Optional[str] = None,
        data: Any = None,
        client: Optional[PrefectClient] = None,
        _sync: bool = True,
        **kwargs: Any
    ) -> tuple[ArtifactResponse, bool]: ...
    
    async def aformat(self) -> str: ...
    
    @overload
    def format(self) -> str: ...
    
    @overload
    def format(self, _sync: bool = True) -> str: ...
    
    def format(self, _sync: bool = True) -> str: ...

class LinkArtifact(Artifact):
    link_text: Optional[str]
    link: str
    type: str = ...
    
    def __init__(
        self,
        key: Optional[str] = None,
        description: Optional[str] = None,
        link: Optional[str] = None,
        link_text: Optional[str] = None,
        **kwargs: Any
    ) -> None: ...
    
    def _format(self) -> str: ...
    
    async def aformat(self) -> str: ...
    
    @overload
    def format(self) -> str: ...
    
    @overload
    def format(self, _sync: bool = True) -> str: ...
    
    def format(self, _sync: bool = True) -> str: ...

class MarkdownArtifact(Artifact):
    markdown: str
    type: str = ...
    
    def __init__(
        self,
        key: Optional[str] = None,
        description: Optional[str] = None,
        markdown: Optional[str] = None,
        **kwargs: Any
    ) -> None: ...
    
    async def aformat(self) -> str: ...
    
    @overload
    def format(self) -> str: ...
    
    @overload
    def format(self, _sync: bool = True) -> str: ...
    
    def format(self, _sync: bool = True) -> str: ...

class TableArtifact(Artifact):
    table: Union[list[Any], dict[str, Any]]
    type: str = ...
    
    def __init__(
        self,
        key: Optional[str] = None,
        description: Optional[str] = None,
        table: Optional[Union[list[Any], dict[str, Any]]] = None,
        **kwargs: Any
    ) -> None: ...
    
    @classmethod
    def _sanitize(cls, item: Any) -> Any: ...
    
    async def aformat(self) -> str: ...
    
    @overload
    def format(self) -> str: ...
    
    @overload
    def format(self, _sync: bool = True) -> str: ...
    
    def format(self, _sync: bool = True) -> str: ...

class ProgressArtifact(Artifact):
    progress: float
    type: str = ...
    
    def __init__(
        self,
        key: Optional[str] = None,
        description: Optional[str] = None,
        progress: Optional[float] = None,
        **kwargs: Any
    ) -> None: ...
    
    def _format(self) -> float: ...
    
    async def aformat(self) -> float: ...
    
    @overload
    def format(self) -> float: ...
    
    @overload
    def format(self, _sync: bool = True) -> float: ...
    
    def format(self, _sync: bool = True) -> float: ...

class ImageArtifact(Artifact):
    """
    An artifact that will display an image from a publicly accessible URL in the UI.

    Arguments:
        image_url: The URL of the image to display.
    """
    image_url: str
    type: str = ...
    
    def __init__(
        self,
        key: Optional[str] = None,
        description: Optional[str] = None,
        image_url: Optional[str] = None,
        **kwargs: Any
    ) -> None: ...
    
    async def aformat(self) -> str: ...
    
    @overload
    def format(self) -> str: ...
    
    @overload
    def format(self, _sync: bool = True) -> str: ...
    
    def format(self, _sync: bool = True) -> str: ...

async def acreate_link_artifact(
    link: str,
    link_text: Optional[str] = None,
    key: Optional[str] = None,
    description: Optional[str] = None,
    client: Optional[PrefectClient] = None
) -> UUID: ...

@overload
def create_link_artifact(
    link: str,
    link_text: Optional[str] = None,
    key: Optional[str] = None,
    description: Optional[str] = None,
    client: Optional[PrefectClient] = None
) -> UUID: ...

@overload
def create_link_artifact(
    link: str,
    link_text: Optional[str] = None,
    key: Optional[str] = None,
    description: Optional[str] = None,
    _sync: bool = True
) -> UUID: ...

def create_link_artifact(
    link: str,
    link_text: Optional[str] = None,
    key: Optional[str] = None,
    description: Optional[str] = None,
    client: Optional[PrefectClient] = None,
    _sync: bool = True
) -> UUID: ...

async def acreate_markdown_artifact(
    markdown: str,
    key: Optional[str] = None,
    description: Optional[str] = None
) -> UUID: ...

@overload
def create_markdown_artifact(
    markdown: str,
    key: Optional[str] = None,
    description: Optional[str] = None
) -> UUID: ...

@overload
def create_markdown_artifact(
    markdown: str,
    key: Optional[str] = None,
    description: Optional[str] = None,
    _sync: bool = True
) -> UUID: ...

def create_markdown_artifact(
    markdown: str,
    key: Optional[str] = None,
    description: Optional[str] = None,
    _sync: bool = True
) -> UUID: ...

async def acreate_table_artifact(
    table: Union[list[Any], dict[str, Any]],
    key: Optional[str] = None,
    description: Optional[str] = None
) -> UUID: ...

@overload
def create_table_artifact(
    table: Union[list[Any], dict[str, Any]],
    key: Optional[str] = None,
    description: Optional[str] = None
) -> UUID: ...

@overload
def create_table_artifact(
    table: Union[list[Any], dict[str, Any]],
    key: Optional[str] = None,
    description: Optional[str] = None,
    _sync: bool = True
) -> UUID: ...

def create_table_artifact(
    table: Union[list[Any], dict[str, Any]],
    key: Optional[str] = None,
    description: Optional[str] = None,
    _sync: bool = True
) -> UUID: ...

async def acreate_progress_artifact(
    progress: float,
    key: Optional[str] = None,
    description: Optional[str] = None
) -> UUID: ...

@overload
def create_progress_artifact(
    progress: float,
    key: Optional[str] = None,
    description: Optional[str] = None
) -> UUID: ...

@overload
def create_progress_artifact(
    progress: float,
    key: Optional[str] = None,
    description: Optional[str] = None,
    _sync: bool = True
) -> UUID: ...

def create_progress_artifact(
    progress: float,
    key: Optional[str] = None,
    description: Optional[str] = None,
    _sync: bool = True
) -> UUID: ...

async def aupdate_progress_artifact(
    artifact_id: UUID,
    progress: float,
    description: Optional[str] = None,
    client: Optional[PrefectClient] = None
) -> UUID: ...

@overload
def update_progress_artifact(
    artifact_id: UUID,
    progress: float,
    description: Optional[str] = None,
    client: Optional[PrefectClient] = None
) -> UUID: ...

@overload
def update_progress_artifact(
    artifact_id: UUID,
    progress: float,
    description: Optional[str] = None,
    _sync: bool = True
) -> UUID: ...

def update_progress_artifact(
    artifact_id: UUID,
    progress: float,
    description: Optional[str] = None,
    client: Optional[PrefectClient] = None,
    _sync: bool = True
) -> UUID: ...

async def acreate_image_artifact(
    image_url: str,
    key: Optional[str] = None,
    description: Optional[str] = None
) -> UUID: ...

@overload
def create_image_artifact(
    image_url: str,
    key: Optional[str] = None,
    description: Optional[str] = None
) -> UUID: ...

@overload
def create_image_artifact(
    image_url: str,
    key: Optional[str] = None,
    description: Optional[str] = None,
    _sync: bool = True
) -> UUID: ...

def create_image_artifact(
    image_url: str,
    key: Optional[str] = None,
    description: Optional[str] = None,
    _sync: bool = True
) -> UUID: ...