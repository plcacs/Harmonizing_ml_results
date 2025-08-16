def _create_location() -> Mock:
    ...

def _create_installed_app(location_id: str, app_id: str) -> Mock:
    ...

async def setup_platform(hass: HomeAssistant, platform: str, *, devices: Any = None, scenes: Any = None) -> MockConfigEntry:
    ...

async def setup_component(hass: HomeAssistant, config_file: dict, hass_storage: dict) -> None:
    ...

@pytest.fixture(name='location')
def location_fixture() -> Mock:
    ...

@pytest.fixture(name='locations')
def locations_fixture(location: Mock) -> List[Mock]:
    ...

@pytest.fixture(name='app')
async def app_fixture(hass: HomeAssistant, config_file: dict) -> Mock:
    ...

@pytest.fixture(name='app_oauth_client')
def app_oauth_client_fixture() -> Mock:
    ...

@pytest.fixture(name='app_settings')
def app_settings_fixture(app: Mock, config_file: dict) -> Mock:
    ...

@pytest.fixture(name='installed_app')
def installed_app_fixture(location: Mock, app: Mock) -> Mock:
    ...

@pytest.fixture(name='installed_apps')
def installed_apps_fixture(installed_app: Mock, locations: List[Mock], app: Mock) -> List[Mock]:
    ...

@pytest.fixture(name='config_file')
def config_file_fixture() -> dict:
    ...

@pytest.fixture(name='smartthings_mock')
def smartthings_mock_fixture(locations: List[Mock]) -> Mock:
    ...

@pytest.fixture(name='device')
def device_fixture(location: Mock) -> Mock:
    ...

@pytest.fixture(name='config_entry')
def config_entry_fixture(installed_app: Mock, location: Mock) -> Mock:
    ...

@pytest.fixture(name='subscription_factory')
def subscription_factory_fixture() -> Callable[[str], Subscription]:
    ...

@pytest.fixture(name='device_factory')
def device_factory_fixture() -> Callable[[str, List[str], Optional[dict]], DeviceEntity]:
    ...

@pytest.fixture(name='scene_factory')
def scene_factory_fixture(location: Mock) -> Callable[[str], SceneEntity]:
    ...

@pytest.fixture(name='scene')
def scene_fixture(scene_factory: Callable[[str], SceneEntity]) -> SceneEntity:
    ...

@pytest.fixture(name='event_factory')
def event_factory_fixture() -> Callable[[str, str, str, str, str, Optional[dict]], Mock]:
    ...

@pytest.fixture(name='event_request_factory')
def event_request_factory_fixture(event_factory: Callable[[str, str, str, str, str, Optional[dict]], Mock]) -> Callable[[Optional[List[str]], Optional[List[Mock]]], Mock]:
    ...
