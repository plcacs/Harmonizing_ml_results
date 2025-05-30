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
    """
    A base model for transaction state.
    """
    store = None
    key = None
    children = Field(default_factory=list)
    commit_mode = None
    isolation_level = IsolationLevel.READ_COMMITTED
    state = TransactionState.PENDING
    on_commit_hooks = Field(default_factory=list)
    on_rollback_hooks = Field(default_factory=list)
    overwrite = False
    logger = Field(default_factory=partial(get_logger, 'transactions'))
    write_on_commit = True
    _stored_values = PrivateAttr(default_factory=dict)
    _staged_value = None
    __var__ = ContextVar('transaction')

    def set(self, name, value):
        """
        Set a stored value in the transaction.

        Args:
            name: The name of the value to set
            value: The value to set

        Examples:
            Set a value for use later in the transaction:
            ```python
            with transaction() as txn:
                txn.set("key", "value")
                ...
                assert txn.get("key") == "value"
            ```
        """
        self._stored_values[name] = value

    def get(self, name, default=NotSet):
        """
        Get a stored value from the transaction.

        Child transactions will return values from their parents unless a value with
        the same name is set in the child transaction.

        Direct changes to returned values will not update the stored value. To update the
        stored value, use the `set` method.

        Args:
            name: The name of the value to get
            default: The default value to return if the value is not found

        Returns:
            The value from the transaction

        Examples:
            Get a value from the transaction:
            ```python
            with transaction() as txn:
                txn.set("key", "value")
                ...
                assert txn.get("key") == "value"
            ```

            Get a value from a parent transaction:
            ```python
            with transaction() as parent:
                parent.set("key", "parent_value")
                with transaction() as child:
                    assert child.get("key") == "parent_value"
            ```

            Update a stored value:
            ```python
            with transaction() as txn:
                txn.set("key", [1, 2, 3])
                value = txn.get("key")
                value.append(4)
                # Stored value is not updated until `.set` is called
                assert value == [1, 2, 3, 4]
                assert txn.get("key") == [1, 2, 3]

                txn.set("key", value)
                assert txn.get("key") == [1, 2, 3, 4]
            ```
        """
        value = copy.deepcopy(self._stored_values.get(name, NotSet))
        if value is NotSet:
            parent = self.get_parent()
            if parent is not None:
                value = parent.get(name, default)
            elif default is not NotSet:
                value = default
            else:
                raise ValueError(f'Could not retrieve value for unknown key: {name}')
        return value

    def is_committed(self):
        return self.state == TransactionState.COMMITTED

    def is_rolled_back(self):
        return self.state == TransactionState.ROLLED_BACK

    def is_staged(self):
        return self.state == TransactionState.STAGED

    def is_pending(self):
        return self.state == TransactionState.PENDING

    def is_active(self):
        return self.state == TransactionState.ACTIVE

    def __enter__(self):
        if self._token is not None:
            raise RuntimeError('Context already entered. Context enter calls cannot be nested.')
        parent = get_transaction()
        if self.commit_mode is None:
            self.commit_mode = parent.commit_mode if parent else CommitMode.LAZY
        if self.isolation_level is None:
            self.isolation_level = parent.isolation_level if parent else IsolationLevel.READ_COMMITTED
        assert self.isolation_level is not None, 'Isolation level was not set correctly'
        if self.store and self.key and (not self.store.supports_isolation_level(self.isolation_level)):
            raise ConfigurationError(f"Isolation level {self.isolation_level.name} is not supported by provided configuration. Please ensure you've provided a lock file directory or lock manager when using the SERIALIZABLE isolation level.")
        self.state = TransactionState.ACTIVE
        self.begin()
        self._token = self.__var__.set(self)
        return self

    def __exit__(self, *exc_info):
        exc_type, exc_val, _ = exc_info
        if not self._token:
            raise RuntimeError('Asymmetric use of context. Context exit called without an enter.')
        if exc_type:
            self.rollback()
            self.reset()
            raise exc_val
        if self.commit_mode == CommitMode.EAGER:
            self.commit()
        if self.get_parent():
            self.reset()
            return
        if self.commit_mode == CommitMode.OFF:
            self.rollback()
        elif self.commit_mode == CommitMode.LAZY:
            self.commit()
        self.reset()

    def begin(self):
        if self.store and self.key and (self.isolation_level == IsolationLevel.SERIALIZABLE):
            self.logger.debug(f'Acquiring lock for transaction {self.key!r}')
            self.store.acquire_lock(self.key)
        if not self.overwrite and self.store and self.key and self.store.exists(key=self.key):
            self.state = TransactionState.COMMITTED

    def read(self):
        if self.store and self.key:
            record = self.store.read(key=self.key)
            if isinstance(record, ResultRecord):
                return record
        return None

    def reset(self):
        parent = self.get_parent()
        if parent:
            parent.add_child(self)
        if self._token:
            self.__var__.reset(self._token)
            self._token = None
        if parent and self.state == TransactionState.ROLLED_BACK:
            parent.rollback()

    def add_child(self, transaction):
        self.children.append(transaction)

    def get_parent(self):
        parent = None
        if self._token:
            prev_var = getattr(self._token, 'old_value')
            if prev_var != Token.MISSING:
                parent = prev_var
        else:
            parent = self.get_active()
        return parent

    def commit(self):
        if self.state in [TransactionState.ROLLED_BACK, TransactionState.COMMITTED]:
            if self.store and self.key and (self.isolation_level == IsolationLevel.SERIALIZABLE):
                self.logger.debug(f'Releasing lock for transaction {self.key!r}')
                self.store.release_lock(self.key)
            return False
        try:
            for child in self.children:
                child.commit()
            for hook in self.on_commit_hooks:
                self.run_hook(hook, 'commit')
            if self.store and self.key and self.write_on_commit:
                if isinstance(self.store, ResultStore):
                    if isinstance(self._staged_value, ResultRecord):
                        self.store.persist_result_record(result_record=self._staged_value)
                    else:
                        self.store.write(key=self.key, obj=self._staged_value)
                else:
                    self.store.write(key=self.key, result=self._staged_value)
            self.state = TransactionState.COMMITTED
            if self.store and self.key and (self.isolation_level == IsolationLevel.SERIALIZABLE):
                self.logger.debug(f'Releasing lock for transaction {self.key!r}')
                self.store.release_lock(self.key)
            return True
        except SerializationError as exc:
            if self.logger:
                self.logger.warning(f'Encountered an error while serializing result for transaction {self.key!r}: {exc} Code execution will continue, but the transaction will not be committed.')
            self.rollback()
            return False
        except Exception:
            if self.logger:
                self.logger.exception(f'An error was encountered while committing transaction {self.key!r}', exc_info=True)
            self.rollback()
            return False

    def run_hook(self, hook, hook_type):
        hook_name = get_hook_name(hook)
        should_log = getattr(hook, 'log_on_run', True)
        if should_log:
            self.logger.info(f'Running {hook_type} hook {hook_name!r}')
        try:
            hook(self)
        except Exception as exc:
            if should_log:
                self.logger.error(f'An error was encountered while running {hook_type} hook {hook_name!r}')
            raise exc
        else:
            if should_log:
                self.logger.info(f'{hook_type.capitalize()} hook {hook_name!r} finished running successfully')

    def stage(self, value, on_rollback_hooks=None, on_commit_hooks=None):
        """
        Stage a value to be committed later.
        """
        on_commit_hooks = on_commit_hooks or []
        on_rollback_hooks = on_rollback_hooks or []
        if self.state != TransactionState.COMMITTED:
            self._staged_value = value
            self.on_rollback_hooks += on_rollback_hooks
            self.on_commit_hooks += on_commit_hooks
            self.state = TransactionState.STAGED

    def rollback(self):
        if self.state in [TransactionState.ROLLED_BACK, TransactionState.COMMITTED]:
            return False
        try:
            for hook in reversed(self.on_rollback_hooks):
                self.run_hook(hook, 'rollback')
            self.state = TransactionState.ROLLED_BACK
            for child in reversed(self.children):
                child.rollback()
            return True
        except Exception:
            if self.logger:
                self.logger.exception(f'An error was encountered while rolling back transaction {self.key!r}', exc_info=True)
            return False
        finally:
            if self.store and self.key and (self.isolation_level == IsolationLevel.SERIALIZABLE):
                self.logger.debug(f'Releasing lock for transaction {self.key!r}')
                self.store.release_lock(self.key)

    @classmethod
    def get_active(cls):
        return cls.__var__.get(None)

def get_transaction():
    return Transaction.get_active()

@contextmanager
def transaction(key=None, store=None, commit_mode=None, isolation_level=None, overwrite=False, write_on_commit=True, logger=None):
    """
    A context manager for opening and managing a transaction.

    Args:
        - key: An identifier to use for the transaction
        - store: The store to use for persisting the transaction result. If not provided,
            a default store will be used based on the current run context.
        - commit_mode: The commit mode controlling when the transaction and
            child transactions are committed
        - overwrite: Whether to overwrite an existing transaction record in the store
        - write_on_commit: Whether to write the result to the store on commit. If not provided,
            will default will be determined by the current run context. If no run context is
            available, the value of `PREFECT_RESULTS_PERSIST_BY_DEFAULT` will be used.

    Yields:
        - Transaction: An object representing the transaction state
    """
    if key and (not store):
        store = get_result_store()
    try:
        _logger = logger or get_run_logger()
    except MissingContextError:
        _logger = get_logger('transactions')
    with Transaction(key=key, store=store, commit_mode=commit_mode, isolation_level=isolation_level, overwrite=overwrite, write_on_commit=write_on_commit, logger=_logger) as txn:
        yield txn