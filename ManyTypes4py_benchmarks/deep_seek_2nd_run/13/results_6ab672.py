from __future__ import annotations
import inspect
import os
import socket
import threading
import uuid
from functools import partial
from operator import methodcaller
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Callable, ClassVar, Dict, Optional, Tuple, TypeVar, Union
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

if TYPE_CHECKING:
    import logging
    from prefect import Flow, Task
    from prefect.transactions import IsolationLevel

ResultStorage = Union[WritableFileSystem, str]
ResultSerializer = Union[Serializer, str]
LITERAL_TYPES = {type(None), bool, UUID}

def DEFAULT_STORAGE_KEY_FN() -> str:
    return uuid.uuid4().hex

logger = get_logger('results')
P = ParamSpec('P')
_default_storages: Dict[Tuple[str, str], WritableFileSystem] = {}

async def aget_default_result_storage() -> WritableFileSystem:
    """
    Generate a default file system for result storage.
    """
    settings = get_current_settings()
    default_block = settings.results.default_storage_block
    basepath = settings.results.local_storage_path
    cache_key = (str(default_block), str(basepath))
    if cache_key in _default_storages:
        return _default_storages[cache_key]
    if default_block is not None:
        storage = await aresolve_result_storage(default_block)
    else:
        storage = LocalFileSystem(basepath=str(basepath))
    _default_storages[cache_key] = storage
    return storage

@async_dispatch(aget_default_result_storage)
def get_default_result_storage() -> WritableFileSystem:
    """
    Generate a default file system for result storage.
    """
    settings = get_current_settings()
    default_block = settings.results.default_storage_block
    basepath = settings.results.local_storage_path
    cache_key = (str(default_block), str(basepath))
    if cache_key in _default_storages:
        return _default_storages[cache_key]
    if default_block is not None:
        storage = resolve_result_storage(default_block, _sync=True)
        if TYPE_CHECKING:
            assert isinstance(storage, WritableFileSystem)
    else:
        storage = LocalFileSystem(basepath=str(basepath))
    _default_storages[cache_key] = storage
    return storage

async def aresolve_result_storage(result_storage: Union[Block, Path, str, UUID]) -> WritableFileSystem:
    """
    Resolve one of the valid `ResultStorage` input types into a saved block
    document id and an instance of the block.
    """
    from prefect.client.orchestration import get_client
    client = get_client()
    if isinstance(result_storage, Block):
        storage_block = result_storage
    elif isinstance(result_storage, Path):
        storage_block = LocalFileSystem(basepath=str(result_storage))
    elif isinstance(result_storage, str):
        block = await Block.aload(result_storage, client=client)
        if TYPE_CHECKING:
            assert isinstance(block, WritableFileSystem)
        storage_block = block
    elif isinstance(result_storage, UUID):
        block_document = await client.read_block_document(result_storage)
        from_block_document = methodcaller('_from_block_document', block_document)
        block = from_block_document(Block)
        if TYPE_CHECKING:
            assert isinstance(block, WritableFileSystem)
        storage_block = block
    else:
        raise TypeError(f"Result storage must be one of the following types: 'UUID', 'Block', 'str'. Got unsupported type {type(result_storage).__name__!r}.")
    return storage_block

@async_dispatch(aresolve_result_storage)
def resolve_result_storage(result_storage: Union[Block, Path, str, UUID]) -> WritableFileSystem:
    """
    Resolve one of the valid `ResultStorage` input types into a saved block
    document id and an instance of the block.
    """
    from prefect.client.orchestration import get_client
    client = get_client(sync_client=True)
    if isinstance(result_storage, Block):
        storage_block = result_storage
    elif isinstance(result_storage, Path):
        storage_block = LocalFileSystem(basepath=str(result_storage))
    elif isinstance(result_storage, str):
        block = Block.load(result_storage, _sync=True)
        if TYPE_CHECKING:
            assert isinstance(block, WritableFileSystem)
        storage_block = block
    elif isinstance(result_storage, UUID):
        block_document = client.read_block_document(result_storage)
        from_block_document = methodcaller('_from_block_document', block_document)
        block = from_block_document(Block)
        if TYPE_CHECKING:
            assert isinstance(block, WritableFileSystem)
        storage_block = block
    else:
        raise TypeError(f"Result storage must be one of the following types: 'UUID', 'Block', 'str'. Got unsupported type {type(result_storage).__name__!r}.")
    return storage_block

def resolve_serializer(serializer: ResultSerializer) -> Serializer:
    """
    Resolve one of the valid `ResultSerializer` input types into a serializer
    instance.
    """
    if isinstance(serializer, Serializer):
        return serializer
    elif isinstance(serializer, str):
        return Serializer(type=serializer)
    else:
        raise TypeError(f"Result serializer must be one of the following types: 'Serializer', 'str'. Got unsupported type {type(serializer).__name__!r}.")

async def get_or_create_default_task_scheduling_storage() -> WritableFileSystem:
    """
    Generate a default file system for background task parameter/result storage.
    """
    settings = get_current_settings()
    default_block = settings.tasks.scheduling.default_storage_block
    if default_block is not None:
        block = await Block.aload(default_block)
        if TYPE_CHECKING:
            assert isinstance(block, WritableFileSystem)
        return block
    basepath = settings.results.local_storage_path
    return LocalFileSystem(basepath=str(basepath))

def get_default_result_serializer() -> Serializer:
    """
    Generate a default file system for result storage.
    """
    settings = get_current_settings()
    return resolve_serializer(settings.results.default_serializer)

def get_default_persist_setting() -> bool:
    """
    Return the default option for result persistence.
    """
    settings = get_current_settings()
    return settings.results.persist_by_default

def get_default_persist_setting_for_tasks() -> bool:
    """
    Return the default option for result persistence for tasks.
    """
    settings = get_current_settings()
    return settings.tasks.default_persist_result if settings.tasks.default_persist_result is not None else settings.results.persist_by_default

def should_persist_result() -> bool:
    """
    Return the default option for result persistence determined by the current run context.

    If there is no current run context, the value of `results.persist_by_default` on the
    current settings will be returned.
    """
    from prefect.context import FlowRunContext, TaskRunContext
    task_run_context = TaskRunContext.get()
    if task_run_context is not None:
        return task_run_context.persist_result
    flow_run_context = FlowRunContext.get()
    if flow_run_context is not None:
        return flow_run_context.persist_result
    return get_default_persist_setting()

def _format_user_supplied_storage_key(key: str) -> str:
    runtime_vars = {key: getattr(prefect.runtime, key) for key in dir(prefect.runtime)}
    return key.format(**runtime_vars, parameters=prefect.runtime.task_run.parameters)

async def _call_explicitly_async_block_method(block: Block, method: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """
    TODO: remove this once we have explicit async methods on all storage blocks

    see https://github.com/PrefectHQ/prefect/issues/15008
    """
    if hasattr(block, f'a{method}'):
        return await getattr(block, f'a{method}')(*args, **kwargs)
    elif hasattr(getattr(block, method, None), 'aio'):
        return await getattr(block, method).aio(block, *args, **kwargs)
    else:
        maybe_coro = getattr(block, method)(*args, **kwargs)
        if inspect.isawaitable(maybe_coro):
            return await maybe_coro
        else:
            return maybe_coro

T = TypeVar('T')

def default_cache() -> LRUCache:
    return LRUCache(maxsize=1000)

def result_storage_discriminator(x: Any) -> str:
    if isinstance(x, dict):
        if 'block_type_slug' in x:
            return 'WritableFileSystem'
        else:
            return 'NullFileSystem'
    if isinstance(x, WritableFileSystem):
        return 'WritableFileSystem'
    if isinstance(x, NullFileSystem):
        return 'NullFileSystem'
    return 'None'

class ResultStore(BaseModel):
    """
    Manages the storage and retrieval of results.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    result_storage: Optional[WritableFileSystem] = Field(default=None)
    metadata_storage: Optional[WritableFileSystem] = Field(default=None)
    lock_manager: Optional[LockManager] = Field(default=None)
    cache_result_in_memory: bool = Field(default=True)
    serializer: Serializer = Field(default_factory=get_default_result_serializer)
    storage_key_fn: Callable[[], str] = Field(default=DEFAULT_STORAGE_KEY_FN)
    cache: LRUCache = Field(default_factory=default_cache)

    @property
    def result_storage_block_id(self) -> Optional[UUID]:
        if self.result_storage is None:
            return None
        return getattr(self.result_storage, '_block_document_id', None)

    @classmethod
    async def _from_metadata(cls, metadata: ResultRecordMetadata) -> ResultRecord[R]:
        """
        Create a result record from metadata.
        """
        if metadata.storage_block_id is None:
            storage_block = None
        else:
            storage_block = await aresolve_result_storage(metadata.storage_block_id)
        store = cls(result_storage=storage_block, serializer=metadata.serializer)
        if metadata.storage_key is None:
            raise ValueError('storage_key is required to hydrate a result record from metadata')
        result = await store.aread(metadata.storage_key)
        return result

    @sync_compatible
    async def update_for_flow(self, flow: Flow) -> Self:
        """
        Create a new result store for a flow with updated settings.
        """
        update = {}
        update['cache_result_in_memory'] = flow.cache_result_in_memory
        if flow.result_storage is not None:
            update['result_storage'] = await aresolve_result_storage(flow.result_storage)
        if flow.result_serializer is not None:
            update['serializer'] = resolve_serializer(flow.result_serializer)
        if self.result_storage is None and update.get('result_storage') is None:
            update['result_storage'] = await aget_default_result_storage()
        update['metadata_storage'] = NullFileSystem()
        return self.model_copy(update=update)

    @sync_compatible
    async def update_for_task(self, task: Task) -> Self:
        """
        Create a new result store for a task.
        """
        from prefect.transactions import get_transaction
        update = {}
        update['cache_result_in_memory'] = task.cache_result_in_memory
        if task.result_storage is not None:
            update['result_storage'] = await aresolve_result_storage(task.result_storage)
        if task.result_serializer is not None:
            update['serializer'] = resolve_serializer(task.result_serializer)
        if task.result_storage_key is not None:
            update['storage_key_fn'] = partial(_format_user_supplied_storage_key, task.result_storage_key)
        if (current_txn := get_transaction()) and isinstance(current_txn.store, ResultStore):
            update['lock_manager'] = current_txn.store.lock_manager
        from prefect.cache_policies import CachePolicy
        if isinstance(task.cache_policy, CachePolicy):
            if task.cache_policy.key_storage is not None:
                storage = task.cache_policy.key_storage
                if isinstance(storage, str) and (not len(storage.split('/')) == 2):
                    storage = Path(storage)
                update['metadata_storage'] = await aresolve_result_storage(storage)
            if task.cache_policy.lock_manager is not None:
                update['lock_manager'] = task.cache_policy.lock_manager
        if self.result_storage is None and update.get('result_storage') is None:
            update['result_storage'] = await aget_default_result_storage()
        if isinstance(self.metadata_storage, NullFileSystem) and update.get('metadata_storage', NotSet) is NotSet:
            update['metadata_storage'] = None
        return self.model_copy(update=update)

    @staticmethod
    def generate_default_holder() -> str:
        """
        Generate a default holder string using hostname, PID, and thread ID.
        """
        hostname = socket.gethostname()
        pid = os.getpid()
        thread_name = threading.current_thread().name
        thread_id = threading.get_ident()
        return f'{hostname}:{pid}:{thread_id}:{thread_name}'

    @sync_compatible
    async def _exists(self, key: str) -> bool:
        """
        Check if a result record exists in storage.
        """
        if self.metadata_storage is not None:
            try:
                metadata_content = await _call_explicitly_async_block_method(self.metadata_storage, 'read_path', (key,), {})
                if metadata_content is None:
                    return False
                metadata = ResultRecordMetadata.load_bytes(metadata_content)
            except Exception:
                return False
        else:
            try:
                content = await _call_explicitly_async_block_method(self.result_storage, 'read_path', (key,), {})
                if content is None:
                    return False
                record = ResultRecord.deserialize(content)
                metadata = record.metadata
            except Exception:
                return False
        if metadata.expiration:
            exists = metadata.expiration > DateTime.now('utc')
        else:
            exists = True
        return exists

    def exists(self, key: str) -> bool:
        """
        Check if a result record exists in storage.
        """
        return self._exists(key=key, _sync=True)

    async def aexists(self, key: str) -> bool:
        """
        Check if a result record exists in storage.
        """
        return await self._exists(key=key, _sync=False)

    def _resolved_key_path(self, key: str) -> str:
        if self.result_storage_block_id is None and (_resolve_path := getattr(self.result_storage, '_resolve_path', None)):
            return str(_resolve_path(key))
        return key

    @sync_compatible
    async def _read(self, key: str, holder: str) -> ResultRecord[R]:
        """
        Read a result record from storage.
        """
        from prefect._experimental.lineage import emit_result_read_event
        if self.lock_manager is not None and (not self.is_lock_holder(key, holder)):
            await self.await_for_lock(key)
        resolved_key_path = self._resolved_key_path(key)
        if resolved_key_path in self.cache:
            cached_result = self.cache[resolved_key_path]
            await emit_result_read_event(self, resolved_key_path, cached=True)
            return cached_result
        if self.result_storage is None:
            self.result_storage = await aget_default_result_storage()
        if self.metadata_storage is not None:
            metadata_content = await _call_explicitly_async_block_method(self.metadata_storage, 'read_path', (key,), {})
            metadata = ResultRecordMetadata.load_bytes(metadata_content)
            assert metadata.storage_key is not None, 'Did not find storage key in metadata'
            result_content = await _call_explicitly_async_block_method(self.result_storage, 'read_path', (metadata.storage_key,), {})
            result_record = ResultRecord.deserialize_from_result_and_metadata(result=result_content, metadata=metadata_content)
            await emit_result_read_event(self, resolved_key_path)
        else:
            content = await _call_explicitly_async_block_method(self.result_storage, 'read_path', (key,), {})
            result_record = ResultRecord.deserialize(content, backup_serializer=self.serializer)
            await emit_result_read_event(self, resolved_key_path)
        if self.cache_result_in_memory:
            self.cache[resolved_key_path] = result_record
        return result_record

    def read(self, key: str, holder: Optional[str] = None) -> ResultRecord[R]:
        """
        Read a result record from storage.
        """
        holder = holder or self.generate_default_holder()
        return self._read(key=key, holder=holder, _sync=True)

    async def aread(self, key: str, holder: Optional[str] = None) -> ResultRecord[R]:
        """
        Read a result record from storage.
        """
        holder = holder or self.generate_default_holder()
        return await self._read(key=key, holder=holder, _sync=False)

    def create_result_record(self,