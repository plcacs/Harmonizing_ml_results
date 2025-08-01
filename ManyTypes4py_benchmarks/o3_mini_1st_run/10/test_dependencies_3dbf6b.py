#!/usr/bin/env python3
from __future__ import annotations
import asyncio
from typing import Any, Type, Callable
from uuid import UUID
import pendulum
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from prefect.server.database import PrefectDBInterface, dependencies
from prefect.server.database.configurations import (
    AioSqliteConfiguration,
    AsyncPostgresConfiguration,
    BaseDatabaseConfiguration,
)
from prefect.server.database.orm_models import AioSqliteORMConfiguration, AsyncPostgresORMConfiguration
from prefect.server.database.query_components import (
    AioSqliteQueryComponents,
    AsyncPostgresQueryComponents,
    BaseQueryComponents,
)
from prefect.server.schemas.graph import Graph

@pytest.mark.parametrize('ConnectionConfig', (AsyncPostgresConfiguration, AioSqliteConfiguration))
async def test_injecting_an_existing_database_database_config(ConnectionConfig: Type[BaseDatabaseConfiguration]) -> None:
    with dependencies.temporary_database_config(ConnectionConfig(connection_url=None)):
        db: PrefectDBInterface = dependencies.provide_database_interface()
        assert type(db.database_config) is ConnectionConfig

async def test_injecting_a_really_dumb_database_database_config() -> None:

    class UselessConfiguration(BaseDatabaseConfiguration):
        async def engine(self) -> Any:
            ...

        async def session(self, engine: Any) -> Any:
            ...

        async def create_db(self, connection: Any, base_metadata: Any) -> Any:
            ...

        async def drop_db(self, connection: Any, base_metadata: Any) -> Any:
            ...

        def is_inmemory(self, engine: Any) -> Any:
            ...

        async def begin_transaction(self, session: Any, locked: bool) -> Any:
            ...

    with dependencies.temporary_database_config(UselessConfiguration(connection_url=None)):
        db: PrefectDBInterface = dependencies.provide_database_interface()
        assert type(db.database_config) is UselessConfiguration

@pytest.mark.parametrize('QueryComponents', (AsyncPostgresQueryComponents, AioSqliteQueryComponents))
async def test_injecting_existing_query_components(QueryComponents: Type[BaseQueryComponents]) -> None:
    with dependencies.temporary_query_components(QueryComponents()):
        db: PrefectDBInterface = dependencies.provide_database_interface()
        assert type(db.queries) is QueryComponents

async def test_injecting_really_dumb_query_components() -> None:

    class ReallyBrokenQueries(BaseQueryComponents):
        def insert(self, obj: Any) -> Any:
            ...

        def greatest(self, *values: Any) -> Any:
            ...

        def least(self, *values: Any) -> Any:
            ...

        def uses_json_strings(self) -> Any:
            ...

        def cast_to_json(self, json_obj: Any) -> Any:
            ...

        def build_json_object(self, *args: Any) -> Any:
            ...

        def json_arr_agg(self, json_array: Any) -> Any:
            ...

        def make_timestamp_intervals(self, start_time: Any, end_time: Any, interval: Any) -> Any:
            ...

        def set_state_id_on_inserted_flow_runs_statement(self, fr_model: Any, frs_model: Any, inserted_flow_run_ids: Any, insert_flow_run_states: Any) -> Any:
            ...

        async def get_flow_run_notifications_from_queue(self, session: AsyncSession, limit: int) -> Any:
            pass

        def get_scheduled_flow_runs_from_work_queues(self, limit_per_queue: int, work_queue_ids: Any, scheduled_before: Any) -> Any:
            ...

        def _get_scheduled_flow_runs_from_work_pool_template_path(self) -> Any:
            ...

        def _build_flow_run_graph_v2_query(self) -> Any:
            ...

        async def flow_run_graph_v2(self, session: AsyncSession, flow_run_id: Any, since: Any, max_nodes: int) -> Any:
            raise NotImplementedError()

    with dependencies.temporary_query_components(ReallyBrokenQueries()):
        db: PrefectDBInterface = dependencies.provide_database_interface()
        assert type(db.queries) is ReallyBrokenQueries

@pytest.mark.parametrize('ORMConfig', (AsyncPostgresORMConfiguration, AioSqliteORMConfiguration))
async def test_injecting_existing_orm_configs(ORMConfig: Type[Any]) -> None:
    with dependencies.temporary_orm_config(ORMConfig()):
        db: PrefectDBInterface = dependencies.provide_database_interface()
        assert type(db.orm) is ORMConfig

async def test_inject_interface_class() -> None:

    class TestInterface(PrefectDBInterface):
        @property
        def new_property(self) -> int:
            return 42

    with dependencies.temporary_interface_class(TestInterface):
        db: PrefectDBInterface = dependencies.provide_database_interface()
        assert isinstance(db, TestInterface)

class TestDBInject:
    db: PrefectDBInterface

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.db = dependencies.provide_database_interface()

    def test_decorated_function(self) -> None:

        @dependencies.db_injector
        def function_with_injected_db(db: PrefectDBInterface, foo: Any) -> PrefectDBInterface:
            """The documentation is sublime"""
            return db

        assert function_with_injected_db(42) is self.db
        unwrapped: Callable[..., Any] = function_with_injected_db.__wrapped__  # type: ignore
        assert function_with_injected_db.__doc__ == unwrapped.__doc__
        function_with_injected_db.__doc__ = 'Something else'
        assert function_with_injected_db.__doc__ == 'Something else'
        assert unwrapped.__doc__ == function_with_injected_db.__doc__
        del function_with_injected_db.__doc__
        assert function_with_injected_db.__doc__ is None
        assert unwrapped.__doc__ is function_with_injected_db.__doc__

    class SomeClass:
        @dependencies.db_injector
        def method_with_injected_db(self, db: PrefectDBInterface, foo: Any) -> PrefectDBInterface:
            """The documentation is sublime"""
            return db

    def test_decorated_method(self) -> None:
        instance = self.SomeClass()
        assert instance.method_with_injected_db(42) is self.db

    def test_unbound_decorated_method(self) -> None:
        instance = self.SomeClass()
        bound: Callable[..., PrefectDBInterface] = self.SomeClass.method_with_injected_db.__get__(instance)
        assert bound(42) is self.db

    def test_bound_method_attributes(self) -> None:
        instance = self.SomeClass()
        bound: Callable[..., PrefectDBInterface] = instance.method_with_injected_db
        assert bound.__self__ is instance
        assert bound.__func__ is self.SomeClass.method_with_injected_db.__wrapped__  # type: ignore
        unwrapped: Callable[..., Any] = bound.__wrapped__  # type: ignore
        assert bound.__doc__ == unwrapped.__doc__
        before: Any = bound.__doc__
        with pytest.raises(AttributeError, match='is not writable$'):
            bound.__doc__ = 'Something else'
        with pytest.raises(AttributeError, match='is not writable$'):
            del bound.__doc__
        assert unwrapped.__doc__ == before

    def test_decorated_coroutine_function(self) -> None:

        @dependencies.db_injector
        async def coroutine_with_injected_db(db: PrefectDBInterface, foo: Any) -> PrefectDBInterface:
            return db

        assert asyncio.iscoroutinefunction(coroutine_with_injected_db)
        result: PrefectDBInterface = asyncio.run(coroutine_with_injected_db(42))
        assert result is self.db
