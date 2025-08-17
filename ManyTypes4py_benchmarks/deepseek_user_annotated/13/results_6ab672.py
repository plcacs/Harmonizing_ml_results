from __future__ import annotations

import inspect
import os
import socket
import threading
import uuid
from functools import partial
from operator import methodcaller
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    ClassVar,
    Dict,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from uuid import UUID

from cachetools import LRUCache
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
)
from typing_extensions import ParamSpec, Self

import prefect
from prefect._internal.compatibility.async_dispatch import async_dispatch
from prefect._result_records import R, ResultRecord, ResultRecordMetadata
from prefect.blocks.core import Block
from prefect.exceptions import (
    ConfigurationError,
    MissingContextError,
)
from prefect.filesystems import (
    LocalFileSystem,
    NullFileSystem,
    WritableFileSystem,
)
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
LITERAL_TYPES: set[type] = {type(None), bool, UUID}


def DEFAULT_STORAGE_KEY_FN() -> str:
    return uuid.uuid4().hex


logger: "logging.Logger" = get_logger("results")
P = ParamSpec("P")

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
        # Use the local file system
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
        # Use the local file system
        storage = LocalFileSystem(basepath=str(basepath))

    _default_storages[cache_key] = storage
    return storage


async def aresolve_result_storage(
    result_storage: Union[ResultStorage, UUID, Path],
) -> WritableFileSystem:
    """
    Resolve one of the valid `ResultStorage` input types into a saved block
    document id and an instance of the block.
    """
    from prefect.client.orchestration import get_client

    client = get_client()
    storage_block: WritableFileSystem
    if isinstance(result_storage, Block):
        storage_block = result_storage
    elif isinstance(result_storage, Path):
        storage_block = LocalFileSystem(basepath=str(result_storage))
    elif isinstance(result_storage, str):
        block = await Block.aload(result_storage, client=client)
        if TYPE_CHECKING:
            assert isinstance(block, WritableFileSystem)
        storage_block = block
    elif isinstance(result_storage, UUID):  # pyright: ignore[reportUnnecessaryIsInstance]
        block_document = await client.read_block_document(result_storage)
        from_block_document = methodcaller("_from_block_document", block_document)
        block = from_block_document(Block)
        if TYPE_CHECKING:
            assert isinstance(block, WritableFileSystem)
        storage_block = block
    else:
        raise TypeError(
            "Result storage must be one of the following types: 'UUID', 'Block', "
            f"'str'. Got unsupported type {type(result_storage).__name__!r}."
        )

    return storage_block


@async_dispatch(aresolve_result_storage)
def resolve_result_storage(
    result_storage: Union[ResultStorage, UUID, Path],
) -> WritableFileSystem:
    """
    Resolve one of the valid `ResultStorage` input types into a saved block
    document id and an instance of the block.
    """
    from prefect.client.orchestration import get_client

    client = get_client(sync_client=True)
    storage_block: WritableFileSystem
    if isinstance(result_storage, Block):
        storage_block = result_storage
    elif isinstance(result_storage, Path):
        storage_block = LocalFileSystem(basepath=str(result_storage))
    elif isinstance(result_storage, str):
        block = Block.load(result_storage, _sync=True)
        if TYPE_CHECKING:
            assert isinstance(block, WritableFileSystem)
        storage_block = block
    elif isinstance(result_storage, UUID):  # pyright: ignore[reportUnnecessaryIsInstance]
        block_document = client.read_block_document(result_storage)
        from_block_document = methodcaller("_from_block_document", block_document)
        block = from_block_document(Block)
        if TYPE_CHECKING:
            assert isinstance(block, WritableFileSystem)
        storage_block = block
    else:
        raise TypeError(
            "Result storage must be one of the following types: 'UUID', 'Block', "
            f"'str'. Got unsupported type {type(result_storage).__name__!r}."
        )
    return storage_block


def resolve_serializer(serializer: ResultSerializer) -> Serializer:
    """
    Resolve one of the valid `ResultSerializer` input types into a serializer
    instance.
    """
    if isinstance(serializer, Serializer):
        return serializer
    elif isinstance(serializer, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        return Serializer(type=serializer)
    else:
        raise TypeError(
            "Result serializer must be one of the following types: 'Serializer', "
            f"'str'. Got unsupported type {type(serializer).__name__!r}."
        )


async def get_or_create_default_task_scheduling_storage() -> ResultStorage:
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

    # otherwise, use the local file system
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
    return (
        settings.tasks.default_persist_result
        if settings.tasks.default_persist_result is not None
        else settings.results.persist_by_default
    )


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
    # Note here we are pinning to task runs since flow runs do not support storage keys
    # yet; we'll need to split logic in the future or have two separate functions
    runtime_vars = {key: getattr(prefect.runtime, key) for key in dir(prefect.runtime)}
    return key.format(**runtime_vars, parameters=prefect.runtime.task_run.parameters)


async def _call_explicitly_async_block_method(
    block: Union[WritableFileSystem, NullFileSystem],
    method: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """
    TODO: remove this once we have explicit async methods on all storage blocks

    see https://github.com/PrefectHQ/prefect/issues/15008
    """
    if hasattr(block, f"a{method}"):  # explicit async method
        return await getattr(block, f"a{method}")(*args, **kwargs)
    elif hasattr(getattr(block, method, None), "aio"):  # sync_compatible
        return await getattr(block, method).aio(block, *args, **kwargs)
    else:  # should not happen in prefect, but users can override impls
        maybe_coro = getattr(block, method)(*args, **kwargs)
        if inspect.isawaitable(maybe_coro):
            return await maybe_coro
        else:
            return maybe_coro


T = TypeVar("T")


def default_cache() -> LRUCache[str, "ResultRecord[Any]"]:
    return LRUCache(maxsize=1000)


def result_storage_discriminator(x: Any) -> str:
    if isinstance(x, dict):
        if "block_type_slug" in x:
            return "WritableFileSystem"
        else:
            return "NullFileSystem"
    if isinstance(x, WritableFileSystem):
        return "WritableFileSystem"
    if isinstance(x, NullFileSystem):
        return "NullFileSystem"
    return "None"


class ResultStore(BaseModel):
    """
    Manages the storage and retrieval of results.

    Attributes:
        result_storage: The storage for result records. If not provided, the default
            result storage will be used.
        metadata_storage: The storage for result record metadata. If not provided,
            the metadata will be stored alongside the results.
        lock_manager: The lock manager to use for locking result records. If not provided,
            the store cannot be used in transactions with the SERIALIZABLE isolation level.
        cache_result_in_memory: Whether to cache results in memory.
        serializer: The serializer to use for results.
        storage_key_fn: The function to generate storage keys.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    result_storage: Optional[WritableFileSystem] = Field(default=None)
    metadata_storage: Annotated[
        Union[
            Annotated[WritableFileSystem, Tag("WritableFileSystem")],
            Annotated[NullFileSystem, Tag("NullFileSystem")],
            Annotated[None, Tag("None")],
        ],
        Discriminator(result_storage_discriminator),
    ] = Field(default=None)
    lock_manager: Optional[LockManager] = Field(default=None)
    cache_result_in_memory: bool = Field(default=True)
    serializer: Serializer = Field(default_factory=get_default_result_serializer)
    storage_key_fn: Callable[[], str] = Field(default=DEFAULT_STORAGE_KEY_FN)
    cache: LRUCache[str, "ResultRecord[Any]"] = Field(default_factory=default_cache)

    @property
    def result_storage_block_id(self) -> Optional[UUID]:
        if self.result_storage is None:
            return None
        return getattr(self.result_storage, "_block_document_id", None)

    @classmethod
    async def _from_metadata(cls, metadata: ResultRecordMetadata) -> "ResultRecord[R]":
        """
        Create a result record from metadata.

        Will use the result record metadata to fetch data via a result store.

        Args:
            metadata: The metadata to create the result record from.

        Returns:
            ResultRecord: The result record.
        """
        if metadata.storage_block_id is None:
            storage_block = None
        else:
            storage_block = await aresolve_result_storage(metadata.storage_block_id)
        store = cls(result_storage=storage_block, serializer=metadata.serializer)
        if metadata.storage_key is None:
            raise ValueError(
                "storage_key is required to hydrate a result record from metadata"
            )
        result = await store.aread(metadata.storage_key)
        return result

    @sync_compatible
    async def update_for_flow(self, flow: "Flow[..., Any]") -> Self:
        """
        Create a new result store for a flow with updated settings.

        Args:
            flow: The flow to update the result store for.

        Returns:
            An updated result store.
        """
        update: Dict[str, Any] = {}
        update["cache_result_in_memory"] = flow.cache_result_in_memory
        if flow.result_storage is not None:
            update["result_storage"] = await aresolve_result_storage(
                flow.result_storage
            )
        if flow.result_serializer is not None:
            update["serializer"] = resolve_serializer(flow.result_serializer)
        if self.result_storage is None and update.get("result_storage") is None:
            update["result_storage"] = await aget_default_result_storage()
        update["metadata_storage"] = NullFileSystem()
        return self.model_copy(update=update)

    @sync_compatible
    async def update_for_task(self: Self, task: "Task[P, R]") -> Self:
        """
        Create a new result store for a task.

        Args:
            task: The task to update the result store for.

        Returns:
            An updated result store.
        """
        from prefect.transactions import get_transaction

        update: Dict[str, Any] = {}
        update["cache_result_in_memory"] = task.cache_result_in_memory
        if task.result_storage is not None:
            update["result_storage"] = await aresolve_result_storage(
                task.result_storage
            )
        if task.result_serializer is not None:
            update["serializer"] = resolve_serializer(task.result_serializer)
        if task.result_storage_key is not None:
            update["storage_key_fn"] = partial(
                _format_user_supplied_storage_key, task.result_storage_key
            )

        # use the lock manager from a parent transaction if it exists
        if (current_txn := get_transaction()) and isinstance(
            current_txn.store, ResultStore
        ):
            update["lock_manager"] = current_txn.store.lock_manager

        from prefect.cache_policies import CachePolicy

        if isinstance(task.cache_policy, CachePolicy):
            if task.cache_policy.key_storage is not None:
                storage = task.cache_policy.key_storage
                if isinstance(storage, str) and not len(storage.split("/")) == 2:
                    storage = Path(storage)
                update["metadata_storage"] = await aresolve_result_storage(storage)
            # if the cache policy has a lock manager, it takes precedence over the parent transaction
            if task.cache_policy.lock_manager is not None:
                update["lock_manager"] = task.cache_policy.lock_manager

        if self.result_storage is None and update.get("result_storage") is None:
            update["result_storage"] = await aget_default_result_storage()
        if (
            isinstance(self.metadata_storage, NullFileSystem)
            and update.get("metadata_storage", NotSet) is NotSet
        ):
            update["metadata_storage"] = None
        return self.model_copy(update=update)

    @staticmethod
    def generate_default_holder() -> str:
        """
        Generate a default holder string using hostname, PID, and thread ID.

        Returns:
            str: A unique identifier string.
        """
        hostname = socket.gethostname()
        pid = os.getpid()
        thread_name = threading.current_thread().name
        thread_id = threading.get_ident()
        return f"{hostname}:{pid}:{thread_id}:{thread_name}"

    @sync_compatible
    async def _exists(self, key: str) -> bool:
        """
        Check if a result record exists in storage.

        Args:
            key: The key to check for the existence of a result record.

        Returns:
            bool: True if the result record exists, False otherwise.
        """
        if self.metadata_storage is not None:
            # TODO: Add an `exists` method to commonly used storage blocks
            # so the entire payload doesn't need to be read
            try:
                metadata_content = await _call_explicitly_async_block_method(
                    self.metadata_storage, "read_path", (key,), {}
                )
                if metadata_content is None:
                    return False
                metadata = ResultRecordMetadata.load_bytes(metadata_content)

            except Exception:
                return False
        else:
            try:
                content = await _call_explicitly_async_block_method(
                    self.result_storage, "read_path", (key,), {}
                )
                if content is None:
                    return False
                record: ResultRecord[Any] = ResultRecord.deserialize(content)
                metadata = record.m