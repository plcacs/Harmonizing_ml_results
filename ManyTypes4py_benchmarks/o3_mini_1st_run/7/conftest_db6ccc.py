"""Fixtures for PrusaLink."""
from collections.abc import Generator
from typing import Any, Dict
from unittest.mock import patch
import pytest
from homeassistant.components.prusalink import DOMAIN
from homeassistant.core import HomeAssistant
from tests.common import MockConfigEntry

@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Mock a PrusaLink config entry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={'host': 'http://example.com', 'username': 'dummy', 'password': 'dummypw'},
        version=1,
        minor_version=2,
    )
    entry.add_to_hass(hass)
    return entry

@pytest.fixture
def mock_version_api() -> Generator[Dict[str, Any], None, None]:
    """Mock PrusaLink version API."""
    resp: Dict[str, Any] = {
        'api': '2.0.0',
        'server': '2.1.2',
        'text': 'PrusaLink',
        'hostname': 'PrusaXL'
    }
    with patch('pyprusalink.PrusaLink.get_version', return_value=resp):
        yield resp

@pytest.fixture
def mock_info_api() -> Generator[Dict[str, Any], None, None]:
    """Mock PrusaLink info API."""
    resp: Dict[str, Any] = {
        'nozzle_diameter': 0.4,
        'mmu': False,
        'serial': 'serial-1337',
        'hostname': 'PrusaXL',
        'min_extrusion_temp': 170
    }
    with patch('pyprusalink.PrusaLink.get_info', return_value=resp):
        yield resp

@pytest.fixture
def mock_get_legacy_printer() -> Generator[Dict[str, Any], None, None]:
    """Mock PrusaLink printer API."""
    resp: Dict[str, Any] = {'telemetry': {'material': 'PLA'}}
    with patch('pyprusalink.PrusaLink.get_legacy_printer', return_value=resp):
        yield resp

@pytest.fixture
def mock_get_status_idle() -> Generator[Dict[str, Any], None, None]:
    """Mock PrusaLink printer API."""
    resp: Dict[str, Any] = {
        'storage': {'path': '/usb/', 'name': 'usb', 'read_only': False},
        'printer': {
            'state': 'IDLE',
            'temp_bed': 41.9,
            'target_bed': 60.5,
            'temp_nozzle': 47.8,
            'target_nozzle': 210.1,
            'axis_z': 1.8,
            'axis_x': 7.9,
            'axis_y': 8.4,
            'flow': 100,
            'speed': 100,
            'fan_hotend': 100,
            'fan_print': 75
        }
    }
    with patch('pyprusalink.PrusaLink.get_status', return_value=resp):
        yield resp

@pytest.fixture
def mock_get_status_printing() -> Generator[Dict[str, Any], None, None]:
    """Mock PrusaLink printer API."""
    resp: Dict[str, Any] = {
        'job': {'id': 129, 'progress': 37.0, 'time_remaining': 73020, 'time_printing': 43987},
        'storage': {'path': '/usb/', 'name': 'usb', 'read_only': False},
        'printer': {
            'state': 'PRINTING',
            'temp_bed': 53.9,
            'target_bed': 85.0,
            'temp_nozzle': 6.0,
            'target_nozzle': 0.0,
            'axis_z': 5.0,
            'flow': 100,
            'speed': 100,
            'fan_hotend': 5000,
            'fan_print': 2500
        }
    }
    with patch('pyprusalink.PrusaLink.get_status', return_value=resp):
        yield resp

@pytest.fixture
def mock_job_api_idle() -> Generator[Dict[str, Any], None, None]:
    """Mock PrusaLink job API having no job."""
    resp: Dict[str, Any] = {}
    with patch('pyprusalink.PrusaLink.get_job', return_value=resp):
        yield resp

@pytest.fixture
def mock_job_api_idle_mk3() -> Generator[Dict[str, Any], None, None]:
    """Mock PrusaLink job API having a job with idle state (MK3)."""
    resp: Dict[str, Any] = {
        'id': 129,
        'state': 'IDLE',
        'progress': 0.0,
        'time_remaining': None,
        'time_printing': 0,
        'file': {
            'refs': {
                'icon': '/thumb/s/usb/TabletStand3~4.BGC',
                'thumbnail': '/thumb/l/usb/TabletStand3~4.BGC',
                'download': '/usb/TabletStand3~4.BGC'
            },
            'name': 'TabletStand3~4.BGC',
            'display_name': 'TabletStand3.bgcode',
            'path': '/usb',
            'size': 754535,
            'm_timestamp': 1698686881
        }
    }
    with patch('pyprusalink.PrusaLink.get_job', return_value=resp):
        yield resp

@pytest.fixture
def mock_job_api_printing() -> Generator[Dict[str, Any], None, None]:
    """Mock PrusaLink printing."""
    resp: Dict[str, Any] = {
        'id': 129,
        'state': 'PRINTING',
        'progress': 37.0,
        'time_remaining': 73020,
        'time_printing': 43987,
        'file': {
            'refs': {
                'icon': '/thumb/s/usb/TabletStand3~4.BGC',
                'thumbnail': '/thumb/l/usb/TabletStand3~4.BGC',
                'download': '/usb/TabletStand3~4.BGC'
            },
            'name': 'TabletStand3~4.BGC',
            'display_name': 'TabletStand3.bgcode',
            'path': '/usb',
            'size': 754535,
            'm_timestamp': 1698686881
        }
    }
    with patch('pyprusalink.PrusaLink.get_job', return_value=resp):
        yield resp

@pytest.fixture
def mock_job_api_paused(
    mock_get_status_printing: Dict[str, Any], 
    mock_job_api_printing: Dict[str, Any]
) -> None:
    """Mock PrusaLink paused printing."""
    mock_job_api_printing['state'] = 'PAUSED'
    mock_get_status_printing['printer']['state'] = 'PAUSED'

@pytest.fixture
def mock_api(
    mock_version_api: Dict[str, Any],
    mock_info_api: Dict[str, Any],
    mock_get_legacy_printer: Dict[str, Any],
    mock_get_status_idle: Dict[str, Any],
    mock_job_api_idle: Dict[str, Any]
) -> None:
    """Mock PrusaLink API."""
    # This fixture might perform additional API mocking setup.
    pass
