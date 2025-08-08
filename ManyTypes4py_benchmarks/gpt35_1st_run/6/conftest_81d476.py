from typing import List, Union
import pytest
from sqlalchemy.engine import Engine
from homeassistant.core import HomeAssistant

def pytest_configure(config: pytest.Config):
    """Add custom skip_on_db_engine marker."""
    config.addinivalue_line('markers', 'skip_on_db_engine(engine): mark test to run only on named DB engine(s)')

def skip_by_db_engine(request: pytest.FixtureRequest, recorder_db_url: str):
    """Fixture to skip tests on unsupported DB engines."""
    ...

def recorder_dialect_name(hass: HomeAssistant, db_engine: str):
    """Patch the recorder dialect."""
    ...

class InstrumentedMigration:
    """Container to aid controlling migration progress."""

def instrument_migration_fixture(hass: HomeAssistant) -> InstrumentedMigration:
    """Instrument recorder migration."""
    ...

def instrument_migration(hass: HomeAssistant) -> Generator[InstrumentedMigration, None, None]:
    """Instrument recorder migration."""
    ...

def _instrument_migrate_schema_live(real_func: callable, *args: List[Union[str, InstrumentedMigration]]) -> None:
    """Control migration progress and check results."""
    ...

def _instrument_migrate_schema_non_live(real_func: callable, *args: List[Union[str, InstrumentedMigration]]) -> None:
    """Control migration progress and check results."""
    ...

def _instrument_migrate_schema(real_func: callable, args: List[Union[str, InstrumentedMigration]], migration_done: threading.Event, migration_done_stall: threading.Event) -> None:
    """Control migration progress and check results."""
    ...

def _instrument_apply_update(instance: object, hass: HomeAssistant, engine: Engine, session_maker: callable, new_version: int, old_version: int) -> None:
    """Control migration progress."""
    ...
