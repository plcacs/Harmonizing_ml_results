from __future__ import annotations
from http import HTTPStatus
from typing import Any
from unittest.mock import patch
from google_nest_sdm.exceptions import AuthException
import pytest
from homeassistant import config_entries
from homeassistant.components.nest.const import DOMAIN, OAUTH2_AUTHORIZE, OAUTH2_TOKEN
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult, FlowResultType
from homeassistant.helpers import config_entry_oauth2_flow
from homeassistant.helpers.service_info.dhcp import DhcpServiceInfo
from .common import CLIENT_ID, CLOUD_PROJECT_ID, PROJECT_ID, SUBSCRIBER_ID, TEST_CONFIG_APP_CREDS, TEST_CONFIGFLOW_APP_CREDS, NestTestConfig
from .conftest import FakeAuth, PlatformSetup
from tests.common import MockConfigEntry
from tests.test_util.aiohttp import AiohttpClientMocker
from tests.typing import ClientSessionGenerator

WEB_REDIRECT_URL: str = 'https://example.com/auth/external/callback'
APP_REDIRECT_URL: str = 'urn:ietf:wg:oauth:2.0:oob'
RAND_SUFFIX: str = 'ABCDEF'
FAKE_DHCP_DATA: DhcpServiceInfo = DhcpServiceInfo(ip='127.0.0.2', macaddress='001122334455', hostname='fake_hostname')

@pytest.fixture
def nest_test_config() -> Any:
    """Fixture with empty configuration and no existing config entry."""
    return TEST_CONFIGFLOW_APP_CREDS

@pytest.fixture(autouse=True)
def mock_rand_topic_name_fixture() -> Any:
    """Set the topic name random string to a constant."""
    with patch('homeassistant.components.nest.config_flow.get_random_string', return_value=RAND_SUFFIX):
        yield

@pytest.fixture(autouse=True)
def mock_request_setup(auth: FakeAuth) -> Any:
    """Fixture to ensure fake requests are setup."""

class OAuthFixture:
    """Simulate the oauth flow used by the config flow."""

    def __init__(self, hass: HomeAssistant, hass_client_no_auth: ClientSessionGenerator, aioclient_mock: AiohttpClientMocker) -> None:
        """Initialize OAuthFixture."""
        self.hass = hass
        self.hass_client = hass_client_no_auth
        self.aioclient_mock = aioclient_mock

    async def async_app_creds_flow(self, result: FlowResult, cloud_project_id: str = CLOUD_PROJECT_ID, project_id: str = PROJECT_ID) -> None:
        """Invoke multiple steps in the app credentials based flow."""
        assert result.get('type') is FlowResultType.FORM
        assert result.get('step_id') == 'cloud_project'
        result = await self.async_configure(result, {'cloud_project_id': CLOUD_PROJECT_ID})
        assert result.get('type') is FlowResultType.FORM
        assert result.get('step_id') == 'device_project'
        result = await self.async_configure(result, {'project_id': project_id})
        await self.async_oauth_web_flow(result, project_id=project_id)

    async def async_oauth_web_flow(self, result: FlowResult, project_id: str = PROJECT_ID) -> None:
        """Invoke the oauth flow for Web Auth with fake responses."""
        state = self.create_state(result, WEB_REDIRECT_URL)
        assert result['type'] is FlowResultType.EXTERNAL_STEP
        assert result['url'] == self.authorize_url(state, WEB_REDIRECT_URL, CLIENT_ID, project_id)
        client = await self.hass_client()
        resp = await client.get(f'/auth/external/callback?code=abcd&state={state}')
        assert resp.status == 200
        assert resp.headers['content-type'] == 'text/html; charset=utf-8'

    async def async_reauth(self, config_entry: ConfigEntry) -> FlowResult:
        """Initiate a reuath flow."""
        config_entry.async_start_reauth(self.hass)
        await self.hass.async_block_till_done()
        result = self.async_progress()
        assert result['step_id'] == 'reauth_confirm'
        return await self.hass.config_entries.flow.async_configure(result['flow_id'], {})

    def async_progress(self) -> FlowResult:
        """Return the current step of the config flow."""
        flows = self.hass.config_entries.flow.async_progress()
        assert len(flows) == 1
        return flows[0]

    def create_state(self, result: FlowResult, redirect_url: str) -> Any:
        """Create state object based on redirect url."""
        return config_entry_oauth2_flow._encode_jwt(self.hass, {'flow_id': result['flow_id'], 'redirect_uri': redirect_url})

    def authorize_url(self, state: str, redirect_url: str, client_id: str, project_id: str) -> str:
        """Generate the expected authorization url."""
        oauth_authorize = OAUTH2_AUTHORIZE.format(project_id=project_id)
        return f'{oauth_authorize}?response_type=code&client_id={client_id}&redirect_uri={redirect_url}&state={state}&scope=https://www.googleapis.com/auth/sdm.service+https://www.googleapis.com/auth/pubsub&access_type=offline&prompt=consent'

    def async_mock_refresh(self) -> None:
        """Finish the OAuth flow exchanging auth token for refresh token."""
        self.aioclient_mock.post(OAUTH2_TOKEN, json={'refresh_token': 'mock-refresh-token', 'access_token': 'mock-access-token', 'type': 'Bearer', 'expires_in': 60})

    async def async_complete_pubsub_flow(self, result: FlowResult, selected_topic: str, selected_subscription: str = 'create_new_subscription', user_input: Any = None, existing_errors: Any = None) -> Any:
        """Fixture to walk through the Pub/Sub topic and subscription steps.

        This picks a simple set of steps that are reusable for most flows without
        exercising the corner cases.
        """
        assert result.get('type') is FlowResultType.FORM
        assert result.get('step_id') == 'pubsub_topic'
        assert not result.get('errors')
        result = await self.async_configure(result, {'topic_name': selected_topic})
        assert result.get('type') is FlowResultType.FORM
        assert result.get('step_id') == 'pubsub_topic_confirm'
        assert not result.get('errors')
        result = await self.async_configure(result, {})
        assert result.get('type') is FlowResultType.FORM
        assert result.get('step_id') == 'pubsub_subscription'
        assert not result.get('errors')
        return await self.async_finish_setup(result, {'subscription_name': selected_subscription})

    async def async_finish_setup(self, result: FlowResult, user_input: Any = None) -> Any:
        """Finish the OAuth flow exchanging auth token for refresh token."""
        with patch('homeassistant.components.nest.async_setup_entry', return_value=True) as mock_setup:
            await self.async_configure(result, user_input)
            assert len(mock_setup.mock_calls) == 1
            await self.hass.async_block_till_done()
        return self.get_config_entry()

    async def async_configure(self, result: FlowResult, user_input: Any) -> FlowResult:
        """Advance to the next step in the config flow."""
        return await self.hass.config_entries.flow.async_configure(result['flow_id'], user_input)

    def get_config_entry(self) -> ConfigEntry:
        """Get the config entry."""
        entries = self.hass.config_entries.async_entries(DOMAIN)
        assert len(entries) >= 1
        return entries[0]

@pytest.fixture
async def oauth(hass: HomeAssistant, hass_client_no_auth: ClientSessionGenerator, aioclient_mock: AiohttpClientMocker, current_request_with_host: Any) -> OAuthFixture:
    """Create the simulated oauth flow."""
    return OAuthFixture(hass, hass_client_no_auth, aioclient_mock)

@pytest.fixture(name='sdm_managed_topic')
def mock_sdm_managed_topic() -> bool:
    """Fixture to configure fake server responses for SDM owend Pub/Sub topics."""
    return False

@pytest.fixture(name='user_managed_topics')
def mock_user_managed_topics() -> list:
    """Fixture to configure fake server response for user owned Pub/Sub topics."""
    return []

@pytest.fixture(name='subscriptions')
def mock_subscriptions() -> list:
    """Fixture to configure fake server response for user subscriptions that exist."""
    return []

@pytest.fixture(name='cloud_project_id')
def mock_cloud_project_id() -> str:
    """Fixture to configure the cloud console project id used in tests."""
    return CLOUD_PROJECT_ID

@pytest.fixture(name='create_topic_status')
def mock_create_topic_status() -> HTTPStatus:
    """Fixture to configure the return code when creating the topic."""
    return HTTPStatus.OK

@pytest.fixture(name='create_subscription_status')
def mock_create_subscription_status() -> HTTPStatus:
    """Fixture to configure the return code when creating the subscription."""
    return HTTPStatus.OK

@pytest.fixture(name='list_topics_status')
def mock_list_topics_status() -> HTTPStatus:
    """Fixture to configure the return code when listing topics."""
    return HTTPStatus.OK

@pytest.fixture(name='list_subscriptions_status')
def mock_list_subscriptions_status() -> HTTPStatus:
    """Fixture to configure the return code when listing subscriptions."""
    return HTTPStatus.OK

def setup_mock_list_subscriptions_responses(aioclient_mock: AiohttpClientMocker, cloud_project_id: str, subscriptions: list, list_subscriptions_status: HTTPStatus = HTTPStatus.OK) -> None:
    """Configure the mock responses for listing Pub/Sub subscriptions."""
    aioclient_mock.get(f'https://pubsub.googleapis.com/v1/projects/{cloud_project_id}/subscriptions', json={'subscriptions': [{'name': subscription_name, 'topic': topic, 'pushConfig': {}, 'ackDeadlineSeconds': 10, 'messageRetentionDuration': '604800s', 'expirationPolicy': {'ttl': '2678400s'}, 'state': 'ACTIVE'} for subscription_name, topic in subscriptions or ()]}, status=list_subscriptions_status)

def setup_mock_create_topic_responses(aioclient_mock: AiohttpClientMocker, cloud_project_id: str, create_topic_status: HTTPStatus = HTTPStatus.OK) -> None:
    """Configure the mock responses for creating a Pub/Sub topic."""
    aioclient_mock.put(f'https://pubsub.googleapis.com/v1/projects/{cloud_project_id}/topics/home-assistant-{RAND_SUFFIX}', json={}, status=create_topic_status)
    aioclient_mock.post(f'https://pubsub.googleapis.com/v1/projects/{cloud_project_id}/topics/home-assistant-{RAND_SUFFIX}:setIamPolicy', json={}, status=create_topic_status)

def setup_mock_create_subscription_responses(aioclient_mock: AiohttpClientMocker, cloud_project_id: str, create_subscription_status: HTTPStatus = HTTPStatus.OK) -> None:
    """Configure the mock responses for creating a Pub/Sub subscription."""
    aioclient_mock.put(f'https://pubsub.googleapis.com/v1/projects/{cloud_project_id}/subscriptions/home-assistant-{RAND_SUFFIX}', json={}, status=create_subscription_status)

@pytest.fixture(autouse=True)
def mock_pubsub_api_responses(aioclient_mock: AiohttpClientMocker, sdm_managed_topic: bool, user_managed_topics: list, subscriptions: list, device_access_project_id: str, cloud_project_id: str, create_topic_status: HTTPStatus, create_subscription_status: HTTPStatus, list_topics_status: HTTPStatus, list_subscriptions_status: HTTPStatus) -> None:
    """Configure a server response for an SDM managed Pub/Sub topic.

    We check for a topic created by the SDM Device Access Console (but note we don't have permission to read it)
    or the user has created one themselves in the Google Cloud Project.
    """
    aioclient_mock.get(f'https://pubsub.googleapis.com/v1/projects/sdm-prod/topics/enterprise-{device_access_project_id}', status=HTTPStatus.FORBIDDEN if sdm_managed_topic else HTTPStatus.NOT_FOUND)
    aioclient_mock.get(f'https://pubsub.googleapis.com/v1/projects/{cloud_project_id}/topics', json={'topics': [{'name': topic_name} for topic_name in user_managed_topics or ()]}, status=list_topics_status)
    setup_mock_list_subscriptions_responses(aioclient_mock, cloud_project_id, subscriptions, list_subscriptions_status)
    setup_mock_create_topic_responses(aioclient_mock, cloud_project_id, create_topic_status)
    setup_mock_create_subscription_responses(aioclient_mock, cloud_project_id, create_subscription_status)

@pytest.mark.parametrize('sdm_managed_topic', [True])
async def test_app_credentials(hass: HomeAssistant, oauth: OAuthFixture) -> None:
    """Check full flow."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={'source': config_entries.SOURCE_USER})
    await oauth.async_app_creds_flow(result)
    oauth.async_mock_refresh()
    result = await oauth.async_configure(result, None)
    entry = await oauth.async_complete_pubsub_flow(result, selected_topic=f'projects/sdm-prod/topics/enterprise-{PROJECT_ID}')
    data = dict(entry.data)
    assert 'token' in data
    data['token'].pop('expires_in')
    data['token'].pop('expires_at')
    assert data == {'sdm': {}, 'auth_implementation': 'imported-cred', 'cloud_project_id': CLOUD_PROJECT_ID, 'project_id': PROJECT_ID, 'subscription_name': f'projects/{CLOUD_PROJECT_ID}/subscriptions/home-assistant-{RAND_SUFFIX}', 'topic_name': f'projects/sdm-prod/topics/enterprise-{PROJECT_ID}', 'token': {'refresh_token': 'mock-refresh-token', 'access_token': 'mock-access-token', 'type': 'Bearer'}}

# Additional test cases can be annotated in a similar manner
