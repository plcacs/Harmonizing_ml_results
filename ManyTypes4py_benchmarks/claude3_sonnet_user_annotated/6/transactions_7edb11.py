import copy
import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Type,
    Union,
    TypeVar,
    cast,
)

from pydantic import Field, PrivateAttr
from typing_extensions import Self

from prefect.context import ContextModel
from prefect.exceptions import (
    ConfigurationError,
    MissingContextError,
    SerializationError,
)
from prefect.logging.loggers import LoggingAdapter, get_logger, get_run_logger
from prefect.results import (
    ResultRecord,
    ResultStore,
    get_result_store,
)
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
    """
    A base model for transaction state.
    """

    store: Optional[ResultStore] = None
    key: Optional[str] = None
    children: List["Transaction"] = Field(default_factory=list)
    commit_mode: Optional[CommitMode] = None
    isolation_level: Optional[IsolationLevel] = IsolationLevel.READ_COMMITTED
    state: TransactionState = TransactionState.PENDING
    on_commit_hooks: List[Callable[["Transaction"], None]] = Field(default_factory=list)
    on_rollback_hooks: List[Callable[["Transaction"], None]] = Field(
        default_factory=list
    )
    overwrite: bool = False
    logger: Union[logging.Logger, LoggingAdapter] = Field(
        default_factory=partial(get_logger, "transactions")
    )
    write_on_commit: bool = True
    _stored_values: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _staged_value: Any = None
    __var__: ContextVar[Self] = ContextVar("transaction")

    def set(self, name: str, value: Any) -> None:
        """
        Set a stored value in the transaction.

        Args:
            name: The name of the value to set
            value: The value to set

        Examples:
            Set a value for use later in the transaction:
            