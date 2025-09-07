import typing
from collections.abc import Sequence
from sqlalchemy.sql import ClauseElement
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

T = typing.TypeVar('T')

class DatabaseBackend:

    async def connect(self) -> None:
        raise NotImplementedError()

    async def disconnect(self) -> None:
        raise NotImplementedError()

    def connection(self) -> 'ConnectionBackend':
        raise NotImplementedError()

class ConnectionBackend:

    async def acquire(self) -> None:
        raise NotImplementedError()

    async def release(self) -> None:
        raise NotImplementedError()

    async def fetch_all(self, query: ClauseElement) -> List['Record']:
        raise NotImplementedError()

    async def fetch_one(self, query: ClauseElement) -> Optional['Record']:
        raise NotImplementedError()

    async def fetch_val(self, query: ClauseElement, column: int = 0) -> Any:
        row = await self.fetch_one(query)
        return None if row is None else row[column]

    async def execute(self, query: ClauseElement) -> Any:
        raise NotImplementedError()

    async def execute_many(self, queries: List[ClauseElement]) -> Any:
        raise NotImplementedError()

    async def iterate(self, query: ClauseElement) -> AsyncGenerator['Record', None]:
        raise NotImplementedError()
        yield True

    def transaction(self) -> 'TransactionBackend':
        raise NotImplementedError()

    @property
    def raw_connection(self) -> Any:
        raise NotImplementedError()

class TransactionBackend:

    async def start(self, is_root: bool, extra_options: Dict[str, Any]) -> None:
        raise NotImplementedError()

    async def commit(self) -> None:
        raise NotImplementedError()

    async def rollback(self) -> None:
        raise NotImplementedError()

class Record(Sequence):

    @property
    def _mapping(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def __getitem__(self, key: Union[int, str]) -> Any:
        raise NotImplementedError()
