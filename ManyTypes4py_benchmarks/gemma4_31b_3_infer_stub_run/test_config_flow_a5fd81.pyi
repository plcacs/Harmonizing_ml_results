"""Test the Google Nest Device Access config flow."""

from __future__ import annotations
from http import HTTPStatus
from typing import Any, Optional, Union, Protocol, Callable, Awaitable
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult, FlowResultType
from homeassistant.helpers.service_info.dhcp import DhcpServiceInfo

WEB_REDIRECT_URL: str = 'https://example.com/auth/external/callback'
APP_REDIRECT_URL: str = 'urn:ietf:wg:oauth:2.0:oob'
RAND_SUFFIX: str = 'ABCDEF'
FAKE_DHCP_DATA: DhcpServiceInfo = ...

class OAuthFixture:
    """Simulate the oauth flow used by the config flow."""

    def __init__(self, hass: HomeAssistant, hass_client_no_auth: Callable[[], Awaitable[Any]], aioclient_mock: Any) -> None:
        """Initialize OAuthFixture."""
        self.hass: HomeAssistant
        self.hass_client: Callable[[], Awaitable[Any]]
        self.aioclient_mock: Any

    async def async_app_creds_flow(self, result: FlowResult, cloud_project_id: str = ..., project_id: str = ...) -> None:
        """Invoke multiple steps in the app credentials based flow."""
        ...

    async def async_oauth_web_flow(self, result: FlowResult, project_id: str = ...) -> None:
        """Invoke the oauth flow for Web Auth with fake responses."""
        ...

    async def async_reauth(self, config_entry: ConfigEntry) -> FlowResult:
        """Initiate a reuath flow."""
        ...

    def async_progress(self) -> FlowResult:
        """Return the current step of the config flow."""
        ...

    def create_state(self, result: FlowResult, redirect_url: str) -> str:
        """Create state object based on redirect url."""
        ...

    def authorize_url(self, state: str, redirect_url: str, client_id: str, project_id: str) -> str:
        """Generate the expected authorization url."""
        ...

    def async_mock_refresh(self) -> None:
        """Finish the OAuth flow exchanging auth token for refresh token."""
        ...

    async def async_complete_pubsub_flow(
        self, 
        result: FlowResult, 
        selected_topic: str, 
        selected_subscription: str = 'create_new_subscription', 
        user_input: Optional[dict[str, Any]] = None, 
        existing_errors: Optional[dict[str, Any]] = None
    ) -> ConfigEntry:
        """Fixture to walk through the Pub/Sub topic and subscription steps."""
        ...

    async def async_finish_setup(self, result: FlowResult, user_input: Optional[dict[str, Any]] = None) -> ConfigEntry:
        """Finish the OAuth flow exchanging auth token for refresh token."""
        ...

    async def async_configure(self, result: FlowResult, user_input: Optional[dict[str, Any]]) -> FlowResult:
        """Advance to the next step in the config flow."""
        ...

    def get_config_entry(self) -> ConfigEntry:
        """Get the config entry."""
        ...

def setup_mock_list_subscriptions_responses(
    aioclient_mock: Any, 
    cloud_project_id: str, 
    subscriptions: Optional[list[tuple[str, str]]], 
    list_subscriptions_status: HTTPStatus = HTTPStatus.OK
) -> None:
    """Configure the mock responses for listing Pub/Sub subscriptions."""
    ...

def setup_mock_create_topic_responses(
    aioclient_mock: Any, 
    cloud_project_id: str, 
    create_topic_status: HTTPStatus = HTTPStatus.OK
) -> None:
    """Configure the mock responses for creating a Pub/Sub topic."""
    ...

def setup_mock_create_subscription_responses(
    aioclient_mock: Any, 
    cloud_project_id: str, 
    create_subscription_status: HTTPStatus = HTTPStatus.OK
) -> None:
    """Configure the mock responses for creating a Pub/Sub subscription."""
    ...

def nest_test_config() -> Any: ...
def mock_rand_topic_name_fixture() -> None: ...
def mock_request_setup(auth: Any) -> None: ...
def oauth(hass: HomeAssistant, hass_client_no_auth: Any, aioclient_mock: Any, current_request_with_host: Any) -> OAuthFixture: ...
def mock_sdm_managed_topic() -> bool: ...
def mock_user_managed_topics() -> list[str]: ...
def mock_subscriptions() -> list[Any]: ...
def mock_cloud_project_id() -> str: ...
def mock_create_topic_status() -> HTTPStatus: ...
def mock_create_subscription_status() -> HTTPStatus: ...
def mock_list_topics_status() -> HTTPStatus: ...
def mock_list_subscriptions_status() -> HTTPStatus: ...
def mock_pubsub_api_responses(
    aioclient_mock: Any, 
    sdm_managed_topic: bool, 
    user_managed_topics: Optional[list[str]], 
    subscriptions: Optional[list[Any]], 
    device_access_project_id: str, 
    cloud_project_id: str, 
    create_topic_status: HTTPStatus, 
    create_subscription_status: HTTPStatus, 
    list_topics_status: HTTPStatus, 
    list_subscriptions_status: HTTPStatus
) -> None: ...