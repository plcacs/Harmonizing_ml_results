async def setup_platform(hass: HomeAssistant, platform: str, *, devices: list[DeviceEntity] = None, scenes: list[SceneEntity] = None) -> MockConfigEntry:
    ...

@pytest.fixture(autouse=True)
async def setup_component(hass: HomeAssistant, config_file: dict[str, str], hass_storage: dict[str, dict[str, str]]) -> None:
    ...

def _create_location() -> Location:
    ...

@pytest.fixture(name='location')
def location_fixture() -> Location:
    ...

@pytest.fixture(name='locations')
def locations_fixture(location: Location) -> list[Location]:
    ...

@pytest.fixture(name='app')
async def app_fixture(hass: HomeAssistant, config_file: dict[str, str]) -> AppEntity:
    ...

@pytest.fixture(name='app_oauth_client')
def app_oauth_client_fixture() -> AppOAuthClient:
    ...

@pytest.fixture(name='app_settings')
def app_settings_fixture(app: AppEntity, config_file: dict[str, str]) -> AppSettings:
    ...

def _create_installed_app(location_id: str, app_id: str) -> InstalledApp:
    ...

@pytest.fixture(name='installed_app')
def installed_app_fixture(location: Location, app: AppEntity) -> InstalledApp:
    ...

@pytest.fixture(name='installed_apps')
def installed_apps_fixture(installed_app: InstalledApp, locations: list[Location], app: AppEntity) -> list[InstalledApp]:
    ...

@pytest.fixture(name='config_file')
def config_file_fixture() -> dict[str, str]:
    ...

@pytest.fixture(name='smartthings_mock')
def smartthings_mock_fixture(locations: list[Location]) -> SmartThings:
    ...

@pytest.fixture(name='device')
def device_fixture(location: Location) -> DeviceEntity:
    ...

@pytest.fixture(name='config_entry')
def config_entry_fixture(installed_app: InstalledApp, location: Location) -> MockConfigEntry:
    ...

@pytest.fixture(name='subscription_factory')
def subscription_factory_fixture() -> callable[[str], Subscription]:
    ...

@pytest.fixture(name='device_factory')
def device_factory_fixture() -> callable[[str, list[str], dict[str, str]], DeviceEntity]:
    ...

@pytest.fixture(name='scene_factory')
def scene_factory_fixture(location: Location) -> callable[[str], SceneEntity]:
    ...

@pytest.fixture(name='scene')
def scene_fixture(scene_factory: callable[[str], SceneEntity]) -> SceneEntity:
    ...

@pytest.fixture(name='event_factory')
def event_factory_fixture() -> callable[[str, str, str, str, str, dict[str, str]], Event]:
    ...

@pytest.fixture(name='event_request_factory')
def event_request_factory_fixture(event_factory: callable[[str, str, str, str, str, dict[str, str]], Event]) -> callable[[list[str], list[Event]], EventRequest]:
    ...
