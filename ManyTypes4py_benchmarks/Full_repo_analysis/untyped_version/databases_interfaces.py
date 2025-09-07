import typing
from collections.abc import Sequence
from sqlalchemy.sql import ClauseElement

class DatabaseBackend:

    async def connect(self):
        raise NotImplementedError()

    async def disconnect(self):
        raise NotImplementedError()

    def connection(self):
        raise NotImplementedError()

class ConnectionBackend:

    async def acquire(self):
        raise NotImplementedError()

    async def release(self):
        raise NotImplementedError()

    async def fetch_all(self, query):
        raise NotImplementedError()

    async def fetch_one(self, query):
        raise NotImplementedError()

    async def fetch_val(self, query, column=0):
        row = await self.fetch_one(query)
        return None if row is None else row[column]

    async def execute(self, query):
        raise NotImplementedError()

    async def execute_many(self, queries):
        raise NotImplementedError()

    async def iterate(self, query):
        raise NotImplementedError()
        yield True

    def transaction(self):
        raise NotImplementedError()

    @property
    def raw_connection(self):
        raise NotImplementedError()

class TransactionBackend:

    async def start(self, is_root, extra_options):
        raise NotImplementedError()

    async def commit(self):
        raise NotImplementedError()

    async def rollback(self):
        raise NotImplementedError()

class Record(Sequence):

    @property
    def _mapping(self):
        raise NotImplementedError()

    def __getitem__(self, key):
        raise NotImplementedError()