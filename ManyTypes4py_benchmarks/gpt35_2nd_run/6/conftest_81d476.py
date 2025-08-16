from typing import List, Union
import pytest
from sqlalchemy.engine import Engine
from homeassistant.core import HomeAssistant

def pytest_configure(config: pytest.Config):
    """Add custom skip_on_db_engine marker."""
    config.addinivalue_line('markers', 'skip_on_db_engine(engine): mark test to run only on named DB engine(s)')

def skip_by_db_engine(request: pytest.FixtureRequest, recorder_db_url: str):
    """Fixture to skip tests on unsupported DB engines."""
    pass

def recorder_dialect_name(hass: HomeAssistant, db_engine: str):
    """Patch the recorder dialect."""
    pass

class InstrumentedMigration:
    """Container to aid controlling migration progress."""
    live_migration_done: threading.Event
    live_migration_done_stall: threading.Event
    non_live_migration_done: threading.Event
    non_live_migration_done_stall: threading.Event
    migration_stall: threading.Event
    migration_started: threading.Event
    migration_version: Union[int, None]
    apply_update_mock: Mock
    stall_on_schema_version: Union[int, None]
    apply_update_stalled: threading.Event
    apply_update_version: Union[int, None]

def instrument_migration_fixture(hass: HomeAssistant):
    """Instrument recorder migration."""
    pass
