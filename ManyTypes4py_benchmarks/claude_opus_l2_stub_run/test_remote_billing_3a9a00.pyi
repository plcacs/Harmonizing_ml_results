from datetime import timedelta
from typing import TYPE_CHECKING
from unittest import mock
from uuid import UUID

import responses
import time_machine
from django.conf import settings
from django.utils.timezone import now as timezone_now
from typing_extensions import override

from corporate.lib.remote_billing_util import (
    REMOTE_BILLING_SESSION_VALIDITY_SECONDS,
    LegacyServerIdentityDict,
    RemoteBillingIdentityDict,
    RemoteBillingUserDict,
)
from corporate.lib.stripe import (
    RemoteRealmBillingSession,
    RemoteServerBillingSession,
    add_months,
)
from corporate.models import (
    CustomerPlan,
    LicenseLedger,
    get_current_plan_by_customer,
    get_customer_by_remote_realm,
    get_customer_by_remote_server,
)
from corporate.views.remote_billing_page import (
    generate_confirmation_link_for_server_deactivation,
)
from zerver.lib.exceptions import RemoteRealmServerMismatchError
from zerver.lib.rate_limiter import RateLimitedIPAddr
from zerver.lib.remote_server import send_server_data_to_push_bouncer
from zerver.lib.send_email import FromAddress
from zerver.lib.test_classes import BouncerTestCase
from zerver.lib.test_helpers import activate_push_notification_service, ratelimit_rule
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.models import Realm, UserProfile
from zerver.models.realms import get_realm
from zilencer.models import (
    PreregistrationRemoteRealmBillingUser,
    PreregistrationRemoteServerBillingUser,
    RateLimitedRemoteZulipServer,
    RemoteRealm,
    RemoteRealmBillingUser,
    RemoteServerBillingUser,
    RemoteZulipServer,
)

if TYPE_CHECKING:
    from django.test.client import _MonkeyPatchedWSGIResponse as TestHttpResponse

class RemoteRealmBillingTestCase(BouncerTestCase):
    def execute_remote_billing_authentication_flow(
        self,
        user: UserProfile,
        next_page: str | None = ...,
        expect_tos: bool = ...,
        confirm_tos: bool = ...,
        first_time_login: bool = ...,
        return_without_clicking_confirmation_link: bool = ...,
        return_from_auth_url: bool = ...,
    ) -> TestHttpResponse: ...

class SelfHostedBillingEndpointBasicTest(RemoteRealmBillingTestCase):
    def test_self_hosted_billing_endpoints(self) -> None: ...

class RemoteBillingAuthenticationTest(RemoteRealmBillingTestCase):
    def test_self_hosted_config_error_page(self) -> None: ...
    def test_remote_billing_authentication_flow(self) -> None: ...
    def test_remote_billing_authentication_flow_rate_limited(self) -> None: ...
    def test_remote_billing_authentication_flow_realm_not_registered(self) -> None: ...
    def test_remote_billing_authentication_flow_tos_consent_failure(self) -> None: ...
    def test_remote_billing_authentication_flow_tos_consent_update(self) -> None: ...
    def test_remote_billing_authentication_flow_expired_session(self) -> None: ...
    def test_remote_billing_unauthed_access(self) -> None: ...
    def test_remote_billing_authentication_flow_to_sponsorship_page(self) -> None: ...
    def test_remote_billing_authentication_flow_to_upgrade_page(self) -> None: ...
    def test_remote_billing_authentication_flow_cant_access_billing_without_finishing_confirmation(self) -> None: ...
    def test_remote_billing_authentication_flow_generate_two_confirmation_links_before_confirming(self) -> None: ...
    def test_transfer_complimentary_access_plan_scheduled_for_upgrade_from_server_to_realm(self) -> None: ...
    def test_transfer_plan_from_server_to_realm_when_realm_has_customer(self) -> None: ...
    def test_transfer_business_plan_from_server_to_realm(self) -> None: ...
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
        next_page: str | None = ...,
        expect_tos: bool = ...,
        confirm_tos: bool = ...,
        return_without_clicking_confirmation_link: bool = ...,
    ) -> TestHttpResponse: ...

class LegacyServerLoginTest(RemoteServerTestCase):
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