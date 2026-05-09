from typing import Any

def test_setup_api_ping(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, supervisor_client: SupervisorClient) -> None:
    ...

async def test_service_register(hass: HomeAssistant) -> None:
    ...

async def test_service_calls_core(hass: HomeAssistant, device_registry: DeviceRegistry) -> None:
    ...

async def test_coordinator_updates(hass: HomeAssistant, caplog: LogCaptureFixture, supervisor_client: SupervisorClient) -> None:
    ...

async def test_setup_hardware_integration(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, supervisor_client: SupervisorClient, integration: str) -> None:
    ...

def test_hostname_from_addon_slug() -> None:
    ...

def test_deprecated_function_is_hassio(hass: HomeAssistant, caplog: LogCaptureFixture) -> None:
    ...

def test_deprecated_function_get_supervisor_ip() -> None:
    ...

@pytest.mark.parametrize(('constant_name', 'replacement_name', 'replacement'), [('HassioServiceInfo', 'homeassistant.helpers.service_info.hassio.HassioServiceInfo', HassioServiceInfo)])
def test_deprecated_constants(caplog: LogCaptureFixture, constant_name: str, replacement_name: str, replacement