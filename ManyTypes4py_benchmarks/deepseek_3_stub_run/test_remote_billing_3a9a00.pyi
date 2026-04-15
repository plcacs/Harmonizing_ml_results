from datetime import datetime
from typing import Any, Optional, Union
from unittest.mock import Mock
from uuid import UUID

from django.conf import settings
from django.http import HttpRequest
from django.test.client import Client
from django.utils.timezone import datetime as DateTime
from typing_extensions import override

from corporate.lib.remote_billing_util import (
    REMOTE_BILLING_SESSION_VALIDITY_SECONDS,
    LegacyServerIdentityDict,
    RemoteBillingIdentityDict,
    RemoteBillingUserDict,
)
from corporate.lib.stripe import RemoteRealmBillingSession, RemoteServerBillingSession
from corporate.models import CustomerPlan, LicenseLedger
from zerver.lib.exceptions import RemoteRealmServerMismatchError
from zerver.lib.rate_limiter import RateLimitedIPAddr
from zerver.lib.remote_server import RemoteZulipServer
from zerver.lib.test_classes import BouncerTestCase
from zerver.lib.test_helpers import activate_push_notification_service, ratelimit_rule
from zerver.models import Realm, UserProfile
from zilencer.models import (
    PreregistrationRemoteRealmBillingUser,
    PreregistrationRemoteServerBillingUser,
    RateLimitedRemoteZulipServer,
    RemoteRealm,
    RemoteRealmBillingUser,
    RemoteServerBillingUser,
)

if False:
    from django.test.client import _MonkeyPatchedWSGIResponse as TestHttpResponse


class RemoteRealmBillingTestCase(BouncerTestCase):
    def execute_remote_billing_authentication_flow(
        self,
        user: UserProfile,
        next_page: Optional[str] = None,
        expect_tos: bool = True,
        confirm_tos: bool = True,
        first_time_login: bool = True,
        return_without_clicking_confirmation_link: bool = False,
        return_from_auth_url: bool = False,
    ) -> "TestHttpResponse": ...


@activate_push_notification_service()
class SelfHostedBillingEndpointBasicTest(RemoteRealmBillingTestCase):
    @responses.activate
    def test_self_hosted_billing_endpoints(self) -> None: ...


@activate_push_notification_service()
class RemoteBillingAuthenticationTest(RemoteRealmBillingTestCase):
    def test_self_hosted_config_error_page(self) -> None: ...

    @responses.activate
    def test_remote_billing_authentication_flow(self) -> None: ...

    @ratelimit_rule(10, 3, domain="sends_email_by_remote_server")
    @ratelimit_rule(10, 2, domain="sends_email_by_ip")
    @responses.activate
    def test_remote_billing_authentication_flow_rate_limited(self) -> None: ...

    @responses.activate
    def test_remote_billing_authentication_flow_realm_not_registered(self) -> None: ...

    @responses.activate
    def test_remote_billing_authentication_flow_tos_consent_failure(self) -> None: ...

    @responses.activate
    def test_remote_billing_authentication_flow_tos_consent_update(self) -> None: ...

    @responses.activate
    def test_remote_billing_authentication_flow_expired_session(self) -> None: ...

    @responses.activate
    def test_remote_billing_unauthed_access(self) -> None: ...

    @responses.activate
    def test_remote_billing_authentication_flow_to_sponsorship_page(self) -> None: ...

    @responses.activate
    def test_remote_billing_authentication_flow_to_upgrade_page(self) -> None: ...

    @responses.activate
    def test_remote_billing_authentication_flow_cant_access_billing_without_finishing_confirmation(
        self,
    ) -> None: ...

    @responses.activate
    def test_remote_billing_authentication_flow_generate_two_confirmation_links_before_confirming(
        self,
    ) -> None: ...

    @responses.activate
    def test_transfer_complimentary_access_plan_scheduled_for_upgrade_from_server_to_realm(
        self,
    ) -> None: ...

    @responses.activate
    def test_transfer_plan_from_server_to_realm_when_realm_has_customer(self) -> None: ...

    @responses.activate
    def test_transfer_business_plan_from_server_to_realm(self) -> None: ...

    @responses.activate
    def test_transfer_plan_from_server_to_realm_edge_cases(self) -> None: ...


class RemoteServerTestCase(BouncerTestCase):
    uuid: UUID
    secret: str

    @override
    def setUp(self) -> None: ...

    def execute_remote_billing_authentication_flow(
        self,
        email: str,
        full_name: str,
        next_page: Optional[str] = None,
        expect_tos: bool = True,
        confirm_tos: bool = True,
        return_without_clicking_confirmation_link: bool = False,
    ) -> "TestHttpResponse": ...


class LegacyServerLoginTest(RemoteServerTestCase):
    @ratelimit_rule(10, 3, domain="sends_email_by_remote_server")
    @ratelimit_rule(10, 2, domain="sends_email_by_ip")
    def test_remote_billing_authentication_flow_rate_limited(self) -> None: ...

    def test_server_login_get(self) -> None: ...

    def test_server_login_invalid_zulip_org_id(self) -> None: ...

    def test_server_login_invalid_zulip_org_key(self) -> None: ...

    def test_server_login_deactivated_server(self) -> None: ...

    def test_server_login_success_with_no_plan(self) -> None: ...

    def test_server_login_success_consent_is_not_re_asked(self) -> None: ...

    def test_server_login_success_with_next_page(self) -> None: ...

    def test_server_login_next_page_in_form_persists(self) -> None: ...

    def test_server_billing_unauthed(self) -> None: ...

    def test_remote_billing_authentication_flow_tos_consent_failure(self) -> None: ...


class TestGenerateDeactivationLink(BouncerTestCase):
    def test_generate_deactivation_link(self) -> None: ...