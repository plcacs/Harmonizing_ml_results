from __future__ import annotations

from http import HTTPStatus
from typing import Any, Awaitable, Iterable, Optional

from google_nest_sdm.exceptions import AuthException
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult, FlowResultType
from homeassistant.helpers.service_info.dhcp import DhcpServiceInfo
from tests.common import MockConfigEntry
from tests.test_util.aiohttp import AiohttpClientMocker
from tests.typing import ClientSessionGenerator

from .common import (
    CLIENT_ID,
    CLOUD_PROJECT_ID,
    PROJECT_ID,
    SUBSCRIBER_ID,
    TEST_CONFIG_APP_CREDS,
    TEST_CONFIGFLOW_APP_CREDS,
    NestTestConfig,
)
from .conftest import FakeAuth, PlatformSetup

WEB_REDIRECT_URL: str = ...
APP_REDIRECT_URL: str = ...
RAND_SUFFIX: str = ...
FAKE_DHCP_DATA: DhcpServiceInfo = ...


def nest_test_config() -> NestTestConfig: ...
def mock_rand_topic_name_fixture() -> None: ...
def mock_request_setup(auth: FakeAuth) -> None: ...


class OAuthFixture:
    hass: HomeAssistant
    hass_client: ClientSessionGenerator
    aioclient_mock: AiohttpClientMocker

    def __init__(
        self,
        hass: HomeAssistant,
        hass_client_no_auth: ClientSessionGenerator,
        aioclient_mock: AiohttpClientMocker,
    ) -> None: ...
    def async_app_creds_flow(
        self,
        result: FlowResult[Any],
        cloud_project_id: str = ...,
        project_id: str = ...,
    ) -> Awaitable[None]: ...
    def async_oauth_web_flow(
        self,
        result: FlowResult[Any],
        project_id: str = ...,
    ) -> Awaitable[None]: ...
    def async_reauth(self, config_entry: ConfigEntry) -> Awaitable[FlowResult[Any]]: ...
    def async_progress(self) -> dict[str, Any]: ...
    def create_state(self, result: FlowResult[Any], redirect_url: str) -> str: ...
    def authorize_url(
        self, state: str, redirect_url: str, client_id: str, project_id: str
    ) -> str: ...
    def async_mock_refresh(self) -> None: ...
    def async_complete_pubsub_flow(
        self,
        result: FlowResult[Any],
        selected_topic: str,
        selected_subscription: str = ...,
        user_input: Optional[dict[str, Any]] = ...,
        existing_errors: Optional[dict[str, Any]] = ...,
    ) -> Awaitable[ConfigEntry]: ...
    def async_finish_setup(
        self, result: FlowResult[Any], user_input: Optional[dict[str, Any]] = ...
    ) -> Awaitable[ConfigEntry]: ...
    def async_configure(
        self, result: FlowResult[Any], user_input: Optional[dict[str, Any]]
    ) -> Awaitable[FlowResult[Any]]: ...
    def get_config_entry(self) -> ConfigEntry: ...


async def oauth(
    hass: HomeAssistant,
    hass_client_no_auth: ClientSessionGenerator,
    aioclient_mock: AiohttpClientMocker,
    current_request_with_host: Any,
) -> OAuthFixture: ...
def sdm_managed_topic() -> bool: ...
def user_managed_topics() -> list[str]: ...
def subscriptions() -> list[tuple[str, str]]: ...
def cloud_project_id() -> str: ...
def create_topic_status() -> HTTPStatus: ...
def create_subscription_status() -> HTTPStatus: ...
def list_topics_status() -> HTTPStatus: ...
def list_subscriptions_status() -> HTTPStatus: ...
def setup_mock_list_subscriptions_responses(
    aioclient_mock: AiohttpClientMocker,
    cloud_project_id: str,
    subscriptions: Optional[Iterable[tuple[str, str]]],
    list_subscriptions_status: HTTPStatus = ...,
) -> None: ...
def setup_mock_create_topic_responses(
    aioclient_mock: AiohttpClientMocker,
    cloud_project_id: str,
    create_topic_status: HTTPStatus = ...,
) -> None: ...
def setup_mock_create_subscription_responses(
    aioclient_mock: AiohttpClientMocker,
    cloud_project_id: str,
    create_subscription_status: HTTPStatus = ...,
) -> None: ...
def mock_pubsub_api_responses(
    aioclient_mock: AiohttpClientMocker,
    sdm_managed_topic: bool,
    user_managed_topics: list[str],
    subscriptions: list[tuple[str, str]],
    device_access_project_id: str,
    cloud_project_id: str,
    create_topic_status: HTTPStatus,
    create_subscription_status: HTTPStatus,
    list_topics_status: HTTPStatus,
    list_subscriptions_status: HTTPStatus,
) -> None: ...
async def test_app_credentials(hass: HomeAssistant, oauth: OAuthFixture) -> None: ...
async def test_config_flow_restart(hass: HomeAssistant, oauth: OAuthFixture) -> None: ...
async def test_config_flow_wrong_project_id(
    hass: HomeAssistant, oauth: OAuthFixture
) -> None: ...
async def test_config_flow_pubsub_configuration_error(
    hass: HomeAssistant, oauth: OAuthFixture
) -> None: ...
async def test_config_flow_pubsub_subscriber_error(
    hass: HomeAssistant, oauth: OAuthFixture
) -> None: ...
async def test_multiple_config_entries(
    hass: HomeAssistant, oauth: OAuthFixture, setup_platform: PlatformSetup
) -> None: ...
async def test_duplicate_config_entries(
    hass: HomeAssistant, oauth: OAuthFixture, setup_platform: PlatformSetup
) -> None: ...
async def test_reauth_multiple_config_entries(
    hass: HomeAssistant,
    oauth: OAuthFixture,
    setup_platform: PlatformSetup,
    config_entry: ConfigEntry,
) -> None: ...
async def test_pubsub_subscription_auth_failure(
    hass: HomeAssistant, oauth: OAuthFixture, mock_subscriber: Any
) -> None: ...
async def test_pubsub_subscriber_config_entry_reauth(
    hass: HomeAssistant,
    oauth: OAuthFixture,
    setup_platform: PlatformSetup,
    config_entry: ConfigEntry,
) -> None: ...
async def test_config_entry_title_from_home(
    hass: HomeAssistant, oauth: OAuthFixture, auth: FakeAuth
) -> None: ...
async def test_config_entry_title_multiple_homes(
    hass: HomeAssistant, oauth: OAuthFixture, auth: FakeAuth
) -> None: ...
async def test_title_failure_fallback(
    hass: HomeAssistant, oauth: OAuthFixture, mock_subscriber: Any
) -> None: ...
async def test_structure_missing_trait(
    hass: HomeAssistant, oauth: OAuthFixture, auth: FakeAuth
) -> None: ...
async def test_dhcp_discovery(
    hass: HomeAssistant, oauth: OAuthFixture, nest_test_config: NestTestConfig
) -> None: ...
async def test_dhcp_discovery_with_creds(
    hass: HomeAssistant, oauth: OAuthFixture
) -> None: ...
async def test_token_error(
    hass: HomeAssistant, oauth: OAuthFixture, status_code: HTTPStatus, error_reason: str
) -> None: ...
async def test_existing_topic_and_subscription(
    hass: HomeAssistant, oauth: OAuthFixture
) -> None: ...
async def test_no_eligible_topics(hass: HomeAssistant, oauth: OAuthFixture) -> None: ...
async def test_list_topics_failure(
    hass: HomeAssistant, oauth: OAuthFixture
) -> None: ...
async def test_create_topic_failed(
    hass: HomeAssistant,
    oauth: OAuthFixture,
    aioclient_mock: AiohttpClientMocker,
    cloud_project_id: str,
    subscriptions: list[tuple[str, str]],
    auth: FakeAuth,
) -> None: ...
async def test_list_subscriptions_failure(
    hass: HomeAssistant, oauth: OAuthFixture
) -> None: ...