from typing import Type, Any

async def test_injecting_an_existing_database_database_config(ConnectionConfig: Type[BaseDatabaseConfiguration]) -> None:
async def test_injecting_a_really_dumb_database_database_config() -> None:
async def test_injecting_existing_query_components(QueryComponents: Type[BaseQueryComponents]) -> None:
async def test_injecting_really_dumb_query_components() -> None:
async def test_injecting_existing_orm_configs(ORMConfig: Type[BaseORMConfiguration]) -> None:
async def test_inject_interface_class() -> None:
def test_decorated_function() -> None:
def test_decorated_method() -> None:
def test_unbound_decorated_method() -> None:
def test_bound_method_attributes() -> None:
def test_decorated_coroutine_function() -> None:
