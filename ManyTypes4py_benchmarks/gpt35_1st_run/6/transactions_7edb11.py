import copy
import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Type, Union
from pydantic import Field, PrivateAttr
from typing_extensions import Self
from prefect.context import ContextModel
from prefect.exceptions import ConfigurationError, MissingContextError, SerializationError
from prefect.logging.loggers import LoggingAdapter, get_logger, get_run_logger
from prefect.results import ResultRecord, ResultStore, get_result_store
from prefect.utilities._engine import get_hook_name
from prefect.utilities.annotations import NotSet
from prefect.utilities.collections import AutoEnum

class IsolationLevel(AutoEnum):
    READ_COMMITTED = AutoEnum.auto()
    SERIALIZABLE = AutoEnum.auto()

class CommitMode(AutoEnum):
    EAGER = AutoEnum.auto()
    LAZY = AutoEnum.auto()
    OFF = AutoEnum.auto()

class TransactionState(AutoEnum):
    PENDING = AutoEnum.auto()
    ACTIVE = AutoEnum.auto()
    STAGED = AutoEnum.auto()
    COMMITTED = AutoEnum.auto()
    ROLLED_BACK = AutoEnum.auto()

class Transaction(ContextModel):
    store: Optional[ResultStore] = None
    key: Optional[str] = None
    children: List['Transaction'] = Field(default_factory=list)
    commit_mode: Optional[CommitMode] = None
    isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED
    state: TransactionState = TransactionState.PENDING
    on_commit_hooks: List[Callable[['Transaction'], Any]] = Field(default_factory=list)
    on_rollback_hooks: List[Callable[['Transaction'], Any]] = Field(default_factory=list)
    overwrite: bool = False
    logger: LoggingAdapter = Field(default_factory=partial(get_logger, 'transactions'))
    write_on_commit: bool = True
    _stored_values: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _staged_value: Any = None
    __var__: ContextVar['Transaction'] = ContextVar('transaction')

    def set(self, name: str, value: Any) -> None:
        ...

    def get(self, name: str, default: Any = NotSet) -> Any:
        ...

    def is_committed(self) -> bool:
        ...

    def is_rolled_back(self) -> bool:
        ...

    def is_staged(self) -> bool:
        ...

    def is_pending(self) -> bool:
        ...

    def is_active(self) -> bool:
        ...

    def __enter__(self) -> 'Transaction':
        ...

    def __exit__(self, *exc_info) -> None:
        ...

    def begin(self) -> None:
        ...

    def read(self) -> Optional[ResultRecord]:
        ...

    def reset(self) -> None:
        ...

    def add_child(self, transaction: 'Transaction') -> None:
        ...

    def get_parent(self) -> Optional['Transaction']:
        ...

    def commit(self) -> bool:
        ...

    def run_hook(self, hook: Callable[['Transaction'], Any], hook_type: str) -> None:
        ...

    def stage(self, value: Any, on_rollback_hooks: Optional[List[Callable[['Transaction'], Any]]] = None, on_commit_hooks: Optional[List[Callable[['Transaction'], Any]]] = None) -> None:
        ...

    def rollback(self) -> bool:
        ...

    @classmethod
    def get_active(cls) -> Optional['Transaction']:
        ...

def get_transaction() -> Optional[Transaction]:
    ...

@contextmanager
def transaction(key: Optional[str] = None, store: Optional[ResultStore] = None, commit_mode: Optional[CommitMode] = None, isolation_level: Optional[IsolationLevel] = None, overwrite: bool = False, write_on_commit: bool = True, logger: Optional[LoggingAdapter] = None) -> Generator[Transaction, None, None]:
    ...
