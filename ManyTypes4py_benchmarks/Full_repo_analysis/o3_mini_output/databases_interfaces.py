import typing
from collections.abc import Sequence
from sqlalchemy.sql import ClauseElement
from typing import Any, Mapping, Optional, Sequence as TypingSequence, Union, List, AsyncIterator

class DatabaseBackend:
    async def connect(self) -> None:
        raise NotImplementedError()

    async def disconnect(self) -> None:
        raise NotImplementedError()

    def connection(self) -> "ConnectionBackend":
        raise NotImplementedError()


class ConnectionBackend:
    async def acquire(self) -> "ConnectionBackend":
        raise NotImplementedError()

    async def release(self) -> None:
        raise NotImplementedError()

    async def fetch_all(self, query: ClauseElement) -> List["Record"]:
        raise NotImplementedError()

    async def fetch_one(self, query: ClauseElement) -> Optional["Record"]:
        raise NotImplementedError()

    async def fetch_val(self, query: ClauseElement, column: int = 0) -> Any:
        row: Optional["Record"] = await self.fetch_one(query)
        return None if row is None else row[column]

    async def execute(self, query: ClauseElement) -> None:
        raise NotImplementedError()

    async def execute_many(self, queries: TypingSequence[ClauseElement]) -> None:
        raise NotImplementedError()

    async def iterate(self, query: ClauseElement) -> AsyncIterator["Record"]:
        raise NotImplementedError()
        yield True  # This line will never be reached

    def transaction(self) -> "TransactionBackend":
        raise NotImplementedError()

    @property
    def raw_connection(self) -> Any:
        raise NotImplementedError()


class TransactionBackend:
    async def start(self, is_root: bool, extra_options: Mapping[str, Any]) -> None:
        raise NotImplementedError()

    async def commit(self) -> None:
        raise NotImplementedError()

    async def rollback(self) -> None:
        raise NotImplementedError()


class Record(Sequence[Any]):
    @property
    def _mapping(self) -> Mapping[str, Any]:
        raise NotImplementedError()

    def __getitem__(self, key: Union[int, slice, str]) -> Any:
        raise NotImplementedError()