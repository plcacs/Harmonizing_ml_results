from __future__ import annotations
import os
import socket
import threading
import uuid
from functools import partial
from operator import methodcaller
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Callable, ClassVar, Optional, TypeVar, Union
from uuid import UUID
from cachetools import LRUCache
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag
from typing_extensions import ParamSpec, Self
import prefect
from prefect._internal.compatibility.async_dispatch import async_dispatch
from prefect._result_records import R, ResultRecord, ResultRecordMetadata
from prefect.blocks.core import Block
from prefect.exceptions import ConfigurationError, MissingContextError
from prefect.filesystems import LocalFileSystem, NullFileSystem, WritableFileSystem
from prefect.locking.protocol import LockManager
from prefect.logging import get_logger
from prefect.serializers import Serializer
from prefect.settings.context import get_current_settings
from prefect.types import DateTime
from prefect.utilities.annotations import NotSet
from prefect.utilities.asyncutils import sync_compatible

ResultStorage = Union[WritableFileSystem, str]
ResultSerializer = Union[Serializer, str]
LITERAL_TYPES = {type(None), bool, UUID}

def DEFAULT_STORAGE_KEY_FN() -> str:
    return uuid.uuid4().hex

_default_storages: dict = {}

async def aget_default_result_storage() -> WritableFileSystem:
    ...

@async_dispatch(aget_default_result_storage)
def get_default_result_storage() -> WritableFileSystem:
    ...

async def aresolve_result_storage(result_storage: Union[Block, Path, str, UUID]) -> WritableFileSystem:
    ...

@async_dispatch(aresolve_result_storage)
def resolve_result_storage(result_storage: Union[Block, Path, str, UUID]) -> WritableFileSystem:
    ...

def resolve_serializer(serializer: Union[Serializer, str]) -> Serializer:
    ...

async def get_or_create_default_task_scheduling_storage() -> WritableFileSystem:
    ...

def get_default_result_serializer() -> Serializer:
    ...

def get_default_persist_setting() -> bool:
    ...

def get_default_persist_setting_for_tasks() -> bool:
    ...

def should_persist_result() -> bool:
    ...

def _format_user_supplied_storage_key(key: str) -> str:
    ...

async def _call_explicitly_async_block_method(block: Block, method: str, args: tuple, kwargs: dict) -> Any:
    ...

def default_cache() -> LRUCache:
    ...

def result_storage_discriminator(x: Any) -> str:
    ...

class ResultStore(BaseModel):
    ...

    @classmethod
    async def _from_metadata(cls, metadata: ResultRecordMetadata) -> ResultRecord:
        ...

    @sync_compatible
    async def update_for_flow(self, flow: Flow) -> ResultStore:
        ...

    @sync_compatible
    async def update_for_task(self, task: Task) -> ResultStore:
        ...

    @staticmethod
    def generate_default_holder() -> str:
        ...

    @sync_compatible
    async def _exists(self, key: str) -> bool:
        ...

    def exists(self, key: str) -> bool:
        ...

    async def aexists(self, key: str) -> bool:
        ...

    def _resolved_key_path(self, key: str) -> str:
        ...

    @sync_compatible
    async def _read(self, key: str, holder: str) -> ResultRecord:
        ...

    def read(self, key: str, holder: str = None) -> ResultRecord:
        ...

    async def aread(self, key: str, holder: str = None) -> ResultRecord:
        ...

    def create_result_record(self, obj: Any, key: str = None, expiration: Optional[DateTime] = None) -> ResultRecord:
        ...

    def write(self, obj: Any, key: str = None, expiration: Optional[DateTime] = None, holder: str = None) -> None:
        ...

    async def awrite(self, obj: Any, key: str = None, expiration: Optional[DateTime] = None, holder: str = None) -> None:
        ...

    @sync_compatible
    async def _persist_result_record(self, result_record: ResultRecord, holder: str) -> None:
        ...

    def persist_result_record(self, result_record: ResultRecord, holder: str = None) -> None:
        ...

    async def apersist_result_record(self, result_record: ResultRecord, holder: str = None) -> None:
        ...

    def supports_isolation_level(self, level: IsolationLevel) -> bool:
        ...

    def acquire_lock(self, key: str, holder: str = None, timeout: Optional[float] = None) -> bool:
        ...

    async def aacquire_lock(self, key: str, holder: str = None, timeout: Optional[float] = None) -> bool:
        ...

    def release_lock(self, key: str, holder: str = None) -> None:
        ...

    def is_locked(self, key: str) -> bool:
        ...

    def is_lock_holder(self, key: str, holder: str = None) -> bool:
        ...

    def wait_for_lock(self, key: str, timeout: Optional[float] = None) -> None:
        ...

    async def await_for_lock(self, key: str, timeout: Optional[float] = None) -> None:
        ...

    @sync_compatible
    async def store_parameters(self, identifier: str, parameters: Any) -> None:
        ...

    @sync_compatible
    async def read_parameters(self, identifier: str) -> Any:
        ...

def get_result_store() -> ResultStore:
    ...
