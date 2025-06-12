from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional, cast
from unittest import mock
import orjson
import time_machine
from django.conf import settings
from django.utils.timezone import now as timezone_now
from typing_extensions import override
from corporate.lib.stripe import (
    RealmBillingSession,
    RemoteRealmBillingSession,
    add_months,
    get_configured_fixed_price_plan_offer,
    start_of_next_billing_cycle,
)
from corporate.models import (
    Customer,
    CustomerPlan,
    CustomerPlanOffer,
    LicenseLedger,
    SponsoredPlanTypes,
    ZulipSponsorshipRequest,
    get_current_plan_by_customer,
    get_customer_by_realm,
)
from zerver.actions.create_realm import do_create_realm
from zerver.actions.invites import do_create_multiuse_invite_link
from zerver.actions.realm_settings import do_change_realm_org_type, do_send_realm_reactivation_email
from zerver.actions.user_settings import do_change_user_setting
from zerver.lib.test_classes import ZulipTestCase
from zerver.lib.test_helpers import reset_email_visibility_to_everyone_in_zulip_realm
from zerver.models import MultiuseInvite, PreregistrationUser, Realm, UserMessage, UserProfile
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.realms import OrgTypeEnum, get_org_type_display_name, get_realm
from zilencer.lib.remote_counts import MissingDataError

if TYPE_CHECKING:
    from django.test.client import _MonkeyPatchedWSGIResponse as TestHttpResponse
import uuid
from zilencer.models import (
    RemoteRealm,
    RemoteRealmAuditLog,
    RemoteRealmBillingUser,
    RemoteServerBillingUser,
    RemoteZulipServer,
    RemoteZulipServerAuditLog,
)


class TestRemoteServerSupportEndpoint(ZulipTestCase):
    @override
    def setUp(self) -> None:
        def add_sponsorship_request(
            name: str, org_type: int, website: str, paid_users: str, plan: str
        ) -> None:
            remote_realm = RemoteRealm.objects.get(name=name)
            customer = Customer.objects.create(
                remote_realm=remote_realm, sponsorship_pending=True
            )
            ZulipSponsorshipRequest.objects.create(
                customer=customer,
                org_type=org_type,
                org_website=website,
                org_description="We help people.",
                expected_total_users="20-35",
                plan_to_use_zulip="For communication on moon.",
                paid_users_count=paid_users,
                paid_users_description="",
                requested_plan=plan,
            )

        def upgrade_complimentary_access_plan(
            complimentary_access_plan: CustomerPlan,
        ) -> None:
            billed_licenses = 10
            assert complimentary_access_plan.end_date is not None
            last_ledger_entry = (
                LicenseLedger.objects.filter(plan=complimentary_access_plan)
                .order_by("-id")
                .first()
            )
            assert last_ledger_entry is not None
            last_ledger_entry.licenses_at_next_renewal = billed_licenses
            last_ledger_entry.save(update_fields=["licenses_at_next_renewal"])
            complimentary_access_plan.status = (
                CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END
            )
            complimentary_access_plan.save(update_fields=["status"])
            plan_params = {
                "automanage_licenses": True,
                "charge_automatically": False,
                "price_per_license": 100,
                "billing_cycle_anchor": complimentary_access_plan.end_date,
                "billing_schedule": CustomerPlan.BILLING_SCHEDULE_MONTHLY,
                "tier": CustomerPlan.TIER_SELF_HOSTED_BASIC,
                "status": CustomerPlan.NEVER_STARTED,
            }
            CustomerPlan.objects.create(
                customer=complimentary_access_plan.customer,
                next_invoice_date=complimentary_access_plan.end_date,
                **plan_params,
            )

        def add_complimentary_access_plan(name: str, upgrade: bool) -> None:
            complimentary_access_plan_anchor = datetime(
                2050, 1, 1, tzinfo=timezone.utc
            )
            next_plan_anchor = datetime(2050, 2, 1, tzinfo=timezone.utc)
            remote_realm = RemoteRealm.objects.get(name=name)
            billing_session = RemoteRealmBillingSession(remote_realm)
            billing_session.create_complimentary_access_plan(
                complimentary_access_plan_anchor, next_plan_anchor
            )
            customer = billing_session.get_customer()
            assert customer is not None
            complimentary_access_plan = billing_session.get_complimentary_access_plan(
                customer
            )
            assert complimentary_access_plan is not None
            assert complimentary_access_plan.end_date is not None
            if upgrade:
                upgrade_complimentary_access_plan(complimentary_access_plan)

        super().setUp()
        for i in range(6):
            hostname = f"zulip-{i}.example.com"
            remote_server = RemoteZulipServer.objects.create(
                hostname=hostname,
                contact_email=f"admin@{hostname}",
                uuid=uuid.uuid4(),
            )
            RemoteZulipServerAuditLog.objects.create(
                event_type=AuditLogEventType.REMOTE_SERVER_CREATED,
                server=remote_server,
                event_time=remote_server.last_updated,
            )
            if i > 1:
                realm_name = f"realm-name-{i}"
                realm_host = f"realm-host-{i}"
                realm_uuid = uuid.uuid4()
                RemoteRealm.objects.create(
                    server=remote_server,
                    uuid=realm_uuid,
                    host=realm_host,
                    name=realm_name,
                    realm_date_created=datetime(2023, 12, 1, tzinfo=timezone.utc),
                )
        server = RemoteZulipServer.objects.get(hostname="zulip-0.example.com")
        server.deactivated = True
        server.save(update_fields=["deactivated"])
        add_sponsorship_request(
            name="realm-name-2",
            org_type=OrgTypeEnum.Community.value,
            website="",
            paid_users="None",
            plan=SponsoredPlanTypes.BUSINESS.value,
        )
        add_sponsorship_request(
            name="realm-name-3",
            org_type=OrgTypeEnum.OpenSource.value,
            website="example.org",
            paid_users="",
            plan=SponsoredPlanTypes.COMMUNITY.value,
        )
        add_complimentary_access_plan(name="realm-name-4", upgrade=True)
        add_complimentary_access_plan(name="realm-name-5", upgrade=False)
        remote_realm = RemoteRealm.objects.get(name="realm-name-3")
        RemoteRealmBillingUser.objects.create(
            remote_realm=remote_realm,
            email="realm-admin@example.com",
            user_uuid=uuid.uuid4(),
        )
        RemoteServerBillingUser.objects.create(
            remote_server=remote_realm.server, email="server-admin@example.com"
        )

    def test_remote_support_view_queries(self) -> None:
        iago = self.example_user("iago")
        self.login_user(iago)
        with self.assert_database_query_count(28):
            result = self.client_get(
                "/activity/remote/support", {"q": "zulip-3.example.com"}
            )
            self.assertEqual(result.status_code, 200)

    def test_search(self) -> None:
        def assert_server_details_in_response(
            html_response: "TestHttpResponse", hostname: str
        ) -> None:
            self.assert_in_success_response(
                [
                    '<span class="remote-label">Remote server</span>',
                    f"<h3>{hostname} <a",
                    f"<b>Contact email</b>: admin@{hostname}",
                    "<b>Billing users</b>:",
                    "<b>Date created</b>:",
                    "<b>UUID</b>:",
                    "<b>Zulip version</b>:",
                    "<b>Plan type</b>: Free<br />",
                    "<b>Non-guest user count</b>: 0<br />",
                    "<b>Guest user count</b>: 0<br />",
                    "📶 Push notification status:",
                ],
                html_response,
            )

        def assert_realm_details_in_response(
            html_response: "TestHttpResponse", name: str, host: str
        ) -> None:
            self.assert_in_success_response(
                [
                    '<span class="remote-label">Remote realm</span>',
                    f"<h3>{name}</h3>",
                    f"<b>Remote realm host:</b> {host}<br />",
                    "<b>Date created</b>: 01 December 2023",
                    "<b>Organization type</b>: Unspecified<br />",
                    "<b>Has remote realms</b>: True<br />",
                    "📶 Push notification status:",
                ],
                html_response,
            )
            self.assert_not_in_success_response(
                ["<h3>zulip-1.example.com"], html_response
            )

        def check_deactivated_server(
            result: "TestHttpResponse", hostname: str
        ) -> None:
            self.assert_not_in_success_response(
                [
                    "<b>Sponsorship pending</b>:<br />",
                    "⏱️ Schedule fixed price plan:",
                ],
                result,
            )
            self.assert_in_success_response(
                [
                    '<span class="remote-label">Remote server: deactivated</span>',
                    f"<h3>{hostname} <a",
                    f"<b>Contact email</b>: admin@{hostname}",
                    "<b>Billing users</b>:",
                    "<b>Date created</b>:",
                    "<b>UUID</b>:",
                    "<b>Zulip version</b>:",
                    "📶 Push notification status:",
                    "💸 Discounts and sponsorship information:",
                ],
                result,
            )

        def check_remote_server_with_no_realms(
            result: "TestHttpResponse",
        ) -> None:
            assert_server_details_in_response(result, "zulip-1.example.com")
            self.assert_not_in_success_response(
                ["<h3>zulip-2.example.com", "<b>Remote realm host:</b>"], result
            )
            self.assert_in_success_response(
                ["<b>Has remote realms</b>: False<br />"], result
            )

        def check_sponsorship_request_no_website(
            result: "TestHttpResponse",
        ) -> None:
            self.assert_in_success_response(
                [
                    "<li><b>Organization type</b>: Community</li>",
                    "<li><b>Organization website</b>: No website submitted</li>",
                    "<li><b>Paid staff</b>: None</li>",
                    "<li><b>Requested plan</b>: Business</li>",
                    "<li><b>Organization description</b>: We help people.</li>",
                    "<li><b>Estimated total users</b>: 20-35</li>",
                    "<li><b>Description of paid staff</b>: </li>",
                ],
                result,
            )

        def check_sponsorship_request_with_website(
            result: "TestHttpResponse",
        ) -> None:
            self.assert_in_success_response(
                [
                    "<li><b>Organization type</b>: Open-source project</li>",
                    "<li><b>Organization website</b>: example.org</li>",
                    "<li><b>Paid staff</b>: </li>",
                    "<li><b>Requested plan</b>: Community</li>",
                    "<li><b>Organization description</b>: We help people.</li>",
                    "<li><b>Estimated total users</b>: 20-35</li>",
                    "<li><b>Description of paid staff</b>: </li>",
                ],
                result,
            )

        def check_no_sponsorship_request(result: "TestHttpResponse") -> None:
            self.assert_not_in_success_response(
                [
                    "<li><b>Organization description</b>: We help people.</li>",
                    "<li><b>Estimated total users</b>: 20-35</li>",
                    "<li><b>Description of paid staff</b>: </li>",
                ],
                result,
            )

        def check_complimentary_access_plan_with_upgrade(
            result: "TestHttpResponse",
        ) -> None:
            self.assert_in_success_response(
                [
                    "Current plan information:",
                    "<b>Plan name</b>: Zulip Basic (complimentary)<br />",
                    "<b>Status</b>: New plan scheduled<br />",
                    "<b>End date</b>: 01 February 2050<br />",
                    "⏱️ Next plan information:",
                    "<b>Plan name</b>: Zulip Basic<br />",
                    "<b>Status</b>: Never started<br />",
                    "<b>Start date</b>: 01 February 2050<br />",
                    "<b>Billing schedule</b>: Monthly<br />",
                    "<b>Price per license</b>: $1.00<br />",
                    "<b>Estimated billed licenses</b>: 10<br />",
                    "<b>Estimated annual revenue</b>: $120.00<br />",
                ],
                result,
            )

        def check_complimentary_access_plan_without_upgrade(
            result: "TestHttpResponse",
        ) -> None:
            self.assert_in_success_response(
                [
                    "Current plan information:",
                    "<b>Plan name</b>: Zulip Basic (complimentary)<br />",
                    "<b>Status</b>: Active<br />",
                    "<b>End date</b>: 01 February 2050<br />",
                ],
                result,
            )
            self.assert_not_in_success_response(
                ["⏱️ Next plan information:"], result
            )

        def check_for_billing_users_emails(
            result: "TestHttpResponse",
        ) -> None:
            self.assert_in_success_response(
                [
                    "<b>Billing users</b>: realm-admin@example.com",
                    "<b>Billing users</b>: server-admin@example.com",
                ],
                result,
            )

        self.login("cordelia")
        result = self.client_get("/activity/remote/support")
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result["Location"], "/login/")
        self.login("iago")
        assert self.example_user("iago").is_staff
        result = self.client_get("/activity/remote/support")
        self.assert_in_success_response(
            [
                'input type="text" name="q" class="input-xxlarge search-query" placeholder="hostname, UUID or contact email"'
            ],
            result,
        )
        result = self.client_get(
            "/activity/remote/support", {"q": "example.com"}
        )
        for i in range(6):
            self.assert_in_success_response(
                [f"<h3>zulip-{i}.example.com <a"], result
            )
        server = 0
        result = self.client_get(
            "/activity/remote/support", {"q": f"zulip-{server}.example.com"}
        )
        check_deactivated_server(result, f"zulip-{server}.example.com")
        server = 1
        result = self.client_get(
            "/activity/remote/support", {"q": f"zulip-{server}.example.com"}
        )
        check_remote_server_with_no_realms(result)
        result = self.client_get(
            "/activity/remote/support", {"q": "realm-host-"}
        )
        for i in range(6):
            if i > server:
                assert_server_details_in_response(
                    result, f"zulip-{i}.example.com"
                )
                assert_realm_details_in_response(
                    result, f"realm-name-{i}", f"realm-host-{i}"
                )
        server = 2
        with mock.patch(
            "corporate.views.support.compute_max_monthly_messages",
            return_value=1000,
        ):
            result = self.client_get(
                "/activity/remote/support",
                {"q": f"zulip-{server}.example.com"},
            )
        self.assert_in_success_response(
            ["<b>Max monthly messages</b>: 1000"], result
        )
        assert_server_details_in_response(result, f"zulip-{server}.example.com")
        assert_realm_details_in_response(
            result, f"realm-name-{server}", f"realm-host-{server}"
        )
        check_sponsorship_request_no_website(result)
        with mock.patch(
            "corporate.views.support.compute_max_monthly_messages",
            side_effect=MissingDataError,
        ):
            result = self.client_get(
                "/activity/remote/support",
                {"q": f"zulip-{server}.example.com"},
            )
        self.assert_in_success_response(
            ["<b>Max monthly messages</b>: Recent analytics data missing"],
            result,
        )
        assert_server_details_in_response(result, f"zulip-{server}.example.com")
        assert_realm_details_in_response(
            result, f"realm-name-{server}", f"realm-host-{server}"
        )
        check_sponsorship_request_no_website(result)
        server = 3
        result = self.client_get(
            "/activity/remote/support", {"q": f"zulip-{server}.example.com"}
        )
        assert_server_details_in_response(result, f"zulip-{server}.example.com")
        assert_realm_details_in_response(
            result, f"realm-name-{server}", f"realm-host-{server}"
        )
        check_sponsorship_request_with_website(result)
        check_for_billing_users_emails(result)
        result = self.client_get(
            "/activity/remote/support", {"q": "realm-admin@example.com"}
        )
        assert_server_details_in_response(result, f"zulip-{server}.example.com")
       