import typing
from collections.abc import Sequence
from sqlalchemy.sql import ClauseElement
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, TypeVar, Union, overload

T = TypeVar('T')

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

    async def fetch_all(self, query: Union[str, ClauseElement]) -> List['Record']:
        raise NotImplementedError()

    async def fetch_one(self, query: Union[str, ClauseElement]) -> Optional['Record']:
        raise NotImplementedError()

    async def fetch_val(self, query: Union[str, ClauseElement], column: int = 0) -> Any:
        row = await self.fetch_one(query)
        return None if row is None else row[column]

    async def execute(self, query: Union[str, ClauseElement]) -> Any:
        raise NotImplementedError()

    async def execute_many(self, queries: List[Union[str, ClauseElement]]) -> None:
        raise NotImplementedError()

    async def iterate(self, query: Union[str, ClauseElement]) -> AsyncIterator['Record']:
        raise NotImplementedError()
        yield True  # type: ignore

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

class Record(Sequence[Any]):

    @property
    def _mapping(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def __getitem__(self, key: Union[int, str]) -> Any:
        raise NotImplementedError()
