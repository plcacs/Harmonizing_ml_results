#!/usr/bin/env python3
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, TYPE_CHECKING
from unittest import mock
import orjson
import time_machine

from django.conf import settings
from django.utils.timezone import now as timezone_now
from typing_extensions import override

from corporate.lib.stripe import RealmBillingSession, RemoteRealmBillingSession, add_months, get_configured_fixed_price_plan_offer, start_of_next_billing_cycle
from corporate.models import Customer, CustomerPlan, CustomerPlanOffer, LicenseLedger, SponsoredPlanTypes, ZulipSponsorshipRequest, get_current_plan_by_customer, get_customer_by_realm
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
import uuid
from zilencer.models import RemoteRealm, RemoteRealmAuditLog, RemoteRealmBillingUser, RemoteServerBillingUser, RemoteZulipServer, RemoteZulipServerAuditLog

if TYPE_CHECKING:
    from django.test.client import _MonkeyPatchedWSGIResponse as TestHttpResponse

class TestRemoteServerSupportEndpoint(ZulipTestCase):
    @override
    def setUp(self) -> None:
        def add_sponsorship_request(name: str, org_type: int, website: str, paid_users: str, plan: str) -> None:
            remote_realm: RemoteRealm = RemoteRealm.objects.get(name=name)
            customer: Customer = Customer.objects.create(remote_realm=remote_realm, sponsorship_pending=True)
            ZulipSponsorshipRequest.objects.create(customer=customer, org_type=org_type, org_website=website, org_description='We help people.', expected_total_users='20-35', plan_to_use_zulip='For communication on moon.', paid_users_count=paid_users, paid_users_description='', requested_plan=plan)

        def upgrade_complimentary_access_plan(complimentary_access_plan: CustomerPlan) -> None:
            billed_licenses: int = 10
            assert complimentary_access_plan.end_date is not None
            last_ledger_entry: Optional[LicenseLedger] = LicenseLedger.objects.filter(plan=complimentary_access_plan).order_by('-id').first()
            assert last_ledger_entry is not None
            last_ledger_entry.licenses_at_next_renewal = billed_licenses
            last_ledger_entry.save(update_fields=['licenses_at_next_renewal'])
            complimentary_access_plan.status = CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END
            complimentary_access_plan.save(update_fields=['status'])
            plan_params: dict[str, Any] = {
                'automanage_licenses': True,
                'charge_automatically': False,
                'price_per_license': 100,
                'billing_cycle_anchor': complimentary_access_plan.end_date,
                'billing_schedule': CustomerPlan.BILLING_SCHEDULE_MONTHLY,
                'tier': CustomerPlan.TIER_SELF_HOSTED_BASIC,
                'status': CustomerPlan.NEVER_STARTED
            }
            CustomerPlan.objects.create(customer=complimentary_access_plan.customer, next_invoice_date=complimentary_access_plan.end_date, **plan_params)

        def add_complimentary_access_plan(name: str, upgrade: bool) -> None:
            complimentary_access_plan_anchor: datetime = datetime(2050, 1, 1, tzinfo=timezone.utc)
            next_plan_anchor: datetime = datetime(2050, 2, 1, tzinfo=timezone.utc)
            remote_realm: RemoteRealm = RemoteRealm.objects.get(name=name)
            billing_session: RemoteRealmBillingSession = RemoteRealmBillingSession(remote_realm)
            billing_session.create_complimentary_access_plan(complimentary_access_plan_anchor, next_plan_anchor)
            customer: Customer = billing_session.get_customer()
            assert customer is not None
            complimentary_access_plan: Optional[CustomerPlan] = billing_session.get_complimentary_access_plan(customer)
            assert complimentary_access_plan is not None
            assert complimentary_access_plan.end_date is not None
            if upgrade:
                upgrade_complimentary_access_plan(complimentary_access_plan)

        super().setUp()
        for i in range(6):
            hostname: str = f'zulip-{i}.example.com'
            remote_server = RemoteZulipServer.objects.create(hostname=hostname, contact_email=f'admin@{hostname}', uuid=uuid.uuid4())
            RemoteZulipServerAuditLog.objects.create(event_type=AuditLogEventType.REMOTE_SERVER_CREATED, server=remote_server, event_time=remote_server.last_updated)
            if i > 1:
                realm_name: str = f'realm-name-{i}'
                realm_host: str = f'realm-host-{i}'
                realm_uuid = uuid.uuid4()
                RemoteRealm.objects.create(server=remote_server, uuid=realm_uuid, host=realm_host, name=realm_name, realm_date_created=datetime(2023, 12, 1, tzinfo=timezone.utc))
        server = RemoteZulipServer.objects.get(hostname='zulip-0.example.com')
        server.deactivated = True
        server.save(update_fields=['deactivated'])
        add_sponsorship_request(name='realm-name-2', org_type=OrgTypeEnum.Community.value, website='', paid_users='None', plan=SponsoredPlanTypes.BUSINESS.value)
        add_sponsorship_request(name='realm-name-3', org_type=OrgTypeEnum.OpenSource.value, website='example.org', paid_users='', plan=SponsoredPlanTypes.COMMUNITY.value)
        add_complimentary_access_plan(name='realm-name-4', upgrade=True)
        add_complimentary_access_plan(name='realm-name-5', upgrade=False)
        remote_realm = RemoteRealm.objects.get(name='realm-name-3')
        RemoteRealmBillingUser.objects.create(remote_realm=remote_realm, email='realm-admin@example.com', user_uuid=uuid.uuid4())
        RemoteServerBillingUser.objects.create(remote_server=remote_realm.server, email='server-admin@example.com')

    def test_remote_support_view_queries(self) -> None:
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        with self.assert_database_query_count(28):
            result = self.client_get('/activity/remote/support', {'q': 'zulip-3.example.com'})
            self.assertEqual(result.status_code, 200)

    def test_search(self) -> None:
        def assert_server_details_in_response(html_response: str, hostname: str) -> None:
            self.assert_in_success_response([
                '<span class="remote-label">Remote server</span>',
                f'<h3>{hostname} <a',
                f'<b>Contact email</b>: admin@{hostname}',
                '<b>Billing users</b>:',
                '<b>Date created</b>:',
                '<b>UUID</b>:',
                '<b>Zulip version</b>:',
                '<b>Plan type</b>: Free<br />',
                '<b>Non-guest user count</b>: 0<br />',
                '<b>Guest user count</b>: 0<br />',
                '📶 Push notification status:'
            ], html_response)

        def assert_realm_details_in_response(html_response: str, name: str, host: str) -> None:
            self.assert_in_success_response([
                '<span class="remote-label">Remote realm</span>',
                f'<h3>{name}</h3>',
                f'<b>Remote realm host:</b> {host}<br />',
                '<b>Date created</b>: 01 December 2023',
                '<b>Organization type</b>: Unspecified<br />',
                '<b>Has remote realms</b>: True<br />',
                '📶 Push notification status:'
            ], html_response)
            self.assert_not_in_success_response(['<h3>zulip-1.example.com'], html_response)

        def check_deactivated_server(result: str, hostname: str) -> None:
            self.assert_not_in_success_response(['<b>Sponsorship pending</b>:<br />', '⏱️ Schedule fixed price plan:'], result)
            self.assert_in_success_response([
                '<span class="remote-label">Remote server: deactivated</span>',
                f'<h3>{hostname} <a',
                f'<b>Contact email</b>: admin@{hostname}',
                '<b>Billing users</b>:',
                '<b>Date created</b>:',
                '<b>UUID</b>:',
                '<b>Zulip version</b>:',
                '📶 Push notification status:',
                '💸 Discounts and sponsorship information:'
            ], result)

        def check_remote_server_with_no_realms(result: str) -> None:
            assert_server_details_in_response(result, 'zulip-1.example.com')
            self.assert_not_in_success_response(['<h3>zulip-2.example.com', '<b>Remote realm host:</b>'], result)
            self.assert_in_success_response(['<b>Has remote realms</b>: False<br />'], result)

        def check_sponsorship_request_no_website(result: str) -> None:
            self.assert_in_success_response([
                '<li><b>Organization type</b>: Community</li>',
                '<li><b>Organization website</b>: No website submitted</li>',
                '<li><b>Paid staff</b>: None</li>',
                '<li><b>Requested plan</b>: Business</li>',
                '<li><b>Organization description</b>: We help people.</li>',
                '<li><b>Estimated total users</b>: 20-35</li>',
                '<li><b>Description of paid staff</b>: </li>'
            ], result)

        def check_sponsorship_request_with_website(result: str) -> None:
            self.assert_in_success_response([
                '<li><b>Organization type</b>: Open-source project</li>',
                '<li><b>Organization website</b>: example.org</li>',
                '<li><b>Paid staff</b>: </li>',
                '<li><b>Requested plan</b>: Community</li>',
                '<li><b>Organization description</b>: We help people.</li>',
                '<li><b>Estimated total users</b>: 20-35</li>',
                '<li><b>Description of paid staff</b>: </li>'
            ], result)

        def check_no_sponsorship_request(result: str) -> None:
            self.assert_not_in_success_response([
                '<li><b>Organization description</b>: We help people.</li>',
                '<li><b>Estimated total users</b>: 20-35</li>',
                '<li><b>Description of paid staff</b>: </li>'
            ], result)

        def check_complimentary_access_plan_with_upgrade(result: str) -> None:
            self.assert_in_success_response([
                'Current plan information:',
                '<b>Plan name</b>: Zulip Basic (complimentary)<br />',
                '<b>Status</b>: New plan scheduled<br />',
                '<b>End date</b>: 01 February 2050<br />',
                '⏱️ Next plan information:',
                '<b>Plan name</b>: Zulip Basic<br />',
                '<b>Status</b>: Never started<br />',
                '<b>Start date</b>: 01 February 2050<br />',
                '<b>Billing schedule</b>: Monthly<br />',
                '<b>Price per license</b>: $1.00<br />',
                '<b>Estimated billed licenses</b>: 10<br />',
                '<b>Estimated annual revenue</b>: $120.00<br />'
            ], result)

        def check_complimentary_access_plan_without_upgrade(result: str) -> None:
            self.assert_in_success_response([
                'Current plan information:',
                '<b>Plan name</b>: Zulip Basic (complimentary)<br />',
                '<b>Status</b>: Active<br />',
                '<b>End date</b>: 01 February 2050<br />'
            ], result)
            self.assert_not_in_success_response(['⏱️ Next plan information:'], result)

        def check_for_billing_users_emails(result: str) -> None:
            self.assert_in_success_response([
                '<b>Billing users</b>: realm-admin@example.com',
                '<b>Billing users</b>: server-admin@example.com'
            ], result)

        self.login('cordelia')
        result = self.client_get('/activity/remote/support')
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        self.login('iago')
        assert self.example_user('iago').is_staff
        result = self.client_get('/activity/remote/support')
        self.assert_in_success_response(['input type="text" name="q" class="input-xxlarge search-query" placeholder="hostname, UUID or contact email"'], result)
        result = self.client_get('/activity/remote/support', {'q': 'example.com'})
        for i in range(6):
            self.assert_in_success_response([f'<h3>zulip-{i}.example.com <a'], result)
        server = 0
        result = self.client_get('/activity/remote/support', {'q': f'zulip-{server}.example.com'})
        check_deactivated_server(result, f'zulip-{server}.example.com')
        server = 1
        result = self.client_get('/activity/remote/support', {'q': f'zulip-{server}.example.com'})
        check_remote_server_with_no_realms(result)
        result = self.client_get('/activity/remote/support', {'q': 'realm-host-'})
        for i in range(6):
            if i > server:
                assert_server_details_in_response(result, f'zulip-{i}.example.com')
                assert_realm_details_in_response(result, f'realm-name-{i}', f'realm-host-{i}')
        server = 2
        with mock.patch('corporate.views.support.compute_max_monthly_messages', return_value=1000):
            result = self.client_get('/activity/remote/support', {'q': f'zulip-{server}.example.com'})
        self.assert_in_success_response(['<b>Max monthly messages</b>: 1000'], result)
        assert_server_details_in_response(result, f'zulip-{server}.example.com')
        assert_realm_details_in_response(result, f'realm-name-{server}', f'realm-host-{server}')
        check_sponsorship_request_no_website(result)
        with mock.patch('corporate.views.support.compute_max_monthly_messages', side_effect=MissingDataError):
            result = self.client_get('/activity/remote/support', {'q': f'zulip-{server}.example.com'})
        self.assert_in_success_response(['<b>Max monthly messages</b>: Recent analytics data missing'], result)
        assert_server_details_in_response(result, f'zulip-{server}.example.com')
        assert_realm_details_in_response(result, f'realm-name-{server}', f'realm-host-{server}')
        check_sponsorship_request_no_website(result)
        server = 3
        result = self.client_get('/activity/remote/support', {'q': f'zulip-{server}.example.com'})
        assert_server_details_in_response(result, f'zulip-{server}.example.com')
        assert_realm_details_in_response(result, f'realm-name-{server}', f'realm-host-{server}')
        check_sponsorship_request_with_website(result)
        check_for_billing_users_emails(result)
        result = self.client_get('/activity/remote/support', {'q': 'realm-admin@example.com'})
        assert_server_details_in_response(result, f'zulip-{server}.example.com')
        assert_realm_details_in_response(result, f'realm-name-{server}', f'realm-host-{server}')
        check_for_billing_users_emails(result)
        result = self.client_get('/activity/remote/support', {'q': 'server-admin@example.com'})
        assert_server_details_in_response(result, f'zulip-{server}.example.com')
        assert_realm_details_in_response(result, f'realm-name-{server}', f'realm-host-{server}')
        check_for_billing_users_emails(result)
        server = 4
        result = self.client_get('/activity/remote/support', {'q': f'zulip-{server}.example.com'})
        assert_server_details_in_response(result, f'zulip-{server}.example.com')
        assert_realm_details_in_response(result, f'realm-name-{server}', f'realm-host-{server}')
        check_no_sponsorship_request(result)
        check_complimentary_access_plan_with_upgrade(result)
        server = 5
        result = self.client_get('/activity/remote/support', {'q': f'zulip-{server}.example.com'})
        assert_server_details_in_response(result, f'zulip-{server}.example.com')
        assert_realm_details_in_response(result, f'realm-name-{server}', f'realm-host-{server}')
        check_no_sponsorship_request(result)
        check_complimentary_access_plan_without_upgrade(result)
        remote_server = RemoteZulipServer.objects.get(hostname=f'zulip-{server}.example.com')
        result = self.client_get('/activity/remote/support', {'q': f'{remote_server.uuid}'})
        assert_server_details_in_response(result, f'zulip-{server}.example.com')
        assert_realm_details_in_response(result, f'realm-name-{server}', f'realm-host-{server}')
        remote_realm = RemoteRealm.objects.get(host=f'realm-host-{server}')
        result = self.client_get('/activity/remote/support', {'q': f'{remote_realm.uuid}'})
        assert_server_details_in_response(result, f'zulip-{server}.example.com')
        assert_realm_details_in_response(result, f'realm-name-{server}', f'realm-host-{server}')
        server = 0
        remote_server = RemoteZulipServer.objects.get(hostname=f'zulip-{server}.example.com')
        result = self.client_get('/activity/remote/support', {'q': f'{remote_server.uuid}'})
        check_deactivated_server(result, f'zulip-{server}.example.com')
        unknown_uuid = uuid.uuid4()
        result = self.client_get('/activity/remote/support', {'q': f'{unknown_uuid}'})
        self.assert_not_in_success_response(['<span class="remote-label">Remote server</span>', '<span class="remote-label">Remote realm</span>'], result)

    def test_extend_current_plan_end_date(self) -> None:
        remote_realm: RemoteRealm = RemoteRealm.objects.get(name='realm-name-5')
        customer: Customer = Customer.objects.get(remote_realm=remote_realm)
        plan: Optional[CustomerPlan] = get_current_plan_by_customer(customer)
        assert plan is not None
        plan.reminder_to_review_plan_email_sent = True
        plan.save(update_fields=['reminder_to_review_plan_email_sent'])
        self.assertEqual(plan.status, CustomerPlan.ACTIVE)
        self.assertEqual(plan.end_date, datetime(2050, 2, 1, tzinfo=timezone.utc))
        self.assertEqual(plan.next_invoice_date, datetime(2050, 2, 1, tzinfo=timezone.utc))
        cordelia: Any = self.example_user('cordelia')
        self.login_user(cordelia)
        result = self.client_post('/activity/remote/support', {'realm_id': f'{remote_realm.id}', 'end_date': '2050-03-01'})
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        result = self.client_post('/activity/remote/support', {'remote_realm_id': f'{remote_realm.id}', 'plan_end_date': '2050-03-01'})
        self.assert_in_success_response(['Current plan for realm-name-5 updated to end on 2050-03-01.'], result)
        plan.refresh_from_db()
        self.assertEqual(plan.end_date, datetime(2050, 3, 1, tzinfo=timezone.utc))
        self.assertEqual(plan.next_invoice_date, datetime(2050, 3, 1, tzinfo=timezone.utc))
        self.assertFalse(plan.reminder_to_review_plan_email_sent)
        audit_logs = RemoteRealmAuditLog.objects.filter(event_type=AuditLogEventType.CUSTOMER_PLAN_PROPERTY_CHANGED).order_by('-id')
        assert audit_logs.exists()
        reminder_to_review_plan_email_sent_changed_audit_log = audit_logs[0]
        next_invoice_date_changed_audit_log = audit_logs[1]
        end_date_changed_audit_log = audit_logs[2]
        expected_extra_data = {'old_value': '2050-02-01T00:00:00Z', 'new_value': '2050-03-01T00:00:00Z', 'property': 'end_date', 'plan_id': plan.id}
        self.assertEqual(end_date_changed_audit_log.extra_data, expected_extra_data)
        expected_extra_data['property'] = 'next_invoice_date'
        self.assertEqual(next_invoice_date_changed_audit_log.extra_data, expected_extra_data)
        expected_extra_data.update({'old_value': True, 'new_value': False, 'property': 'reminder_to_review_plan_email_sent'})
        self.assertEqual(reminder_to_review_plan_email_sent_changed_audit_log.extra_data, expected_extra_data)
        result = self.client_post('/activity/remote/support', {'remote_realm_id': f'{remote_realm.id}', 'plan_end_date': '2020-01-01'})
        self.assert_in_success_response(['Cannot update current plan for realm-name-5 to end on 2020-01-01.'], result)

    def test_configure_complimentary_access_plan(self) -> None:
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        remote_realm: RemoteRealm = RemoteRealm.objects.get(name='realm-name-4')
        result = self.client_post('/activity/remote/support', {'remote_realm_id': f'{remote_realm.id}', 'complimentary_access_plan': '2010-03-01'})
        self.assert_in_success_response(['Cannot configure a complimentary access plan for realm-name-4 to end on 2010-03-01.'], result)
        result = self.client_post('/activity/remote/support', {'remote_realm_id': f'{remote_realm.id}', 'complimentary_access_plan': '2050-03-01'})
        self.assert_in_success_response(['Cannot configure a complimentary access plan for realm-name-4 because of current plan.'], result)
        remote_realm = RemoteRealm.objects.get(name='realm-name-2')
        assert remote_realm.plan_type == RemoteRealm.PLAN_TYPE_SELF_MANAGED
        result = self.client_post('/activity/remote/support', {'remote_realm_id': f'{remote_realm.id}', 'complimentary_access_plan': '2050-03-01'})
        self.assert_in_success_response(['Complimentary access plan for realm-name-2 configured to end on 2050-03-01.'], result)
        remote_realm.refresh_from_db()
        assert remote_realm.plan_type == RemoteRealm.PLAN_TYPE_SELF_MANAGED_LEGACY

    def test_discount_support_actions_when_upgrade_scheduled(self) -> None:
        remote_realm: RemoteRealm = RemoteRealm.objects.get(name='realm-name-4')
        billing_session = RemoteRealmBillingSession(remote_realm=remote_realm)
        customer: Optional[Customer] = billing_session.get_customer()
        assert customer is not None
        plan: Optional[CustomerPlan] = get_current_plan_by_customer(customer)
        assert plan is not None
        next_plan: Optional[CustomerPlan] = billing_session.get_next_plan(plan)
        assert next_plan is not None
        self.assertEqual(plan.status, CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END)
        self.assertEqual(next_plan.status, CustomerPlan.NEVER_STARTED)
        self.assertEqual(customer.monthly_discounted_price, 0)
        self.assertEqual(customer.annual_discounted_price, 0)
        self.assertIsNone(plan.discount)
        self.assertIsNone(next_plan.discount)
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        self.assertIsNone(customer.required_plan_tier)
        result = self.client_post('/activity/remote/support', {'remote_realm_id': f'{remote_realm.id}', 'required_plan_tier': f'{CustomerPlan.TIER_SELF_HOSTED_BUSINESS}'})
        self.assert_in_success_response(['Required plan tier for realm-name-4 set to Zulip Business.'], result)
        customer.refresh_from_db()
        next_plan.refresh_from_db()
        self.assertEqual(customer.required_plan_tier, CustomerPlan.TIER_SELF_HOSTED_BUSINESS)
        self.assertEqual(plan.tier, CustomerPlan.TIER_SELF_HOSTED_LEGACY)
        self.assertEqual(next_plan.tier, CustomerPlan.TIER_SELF_HOSTED_BASIC)
        result = self.client_post('/activity/remote/support', {'remote_realm_id': f'{remote_realm.id}', 'required_plan_tier': f'{CustomerPlan.TIER_SELF_HOSTED_BASIC}'})
        self.assert_in_success_response(['Required plan tier for realm-name-4 set to Zulip Basic.'], result)
        result = self.client_post('/activity/remote/support', {'remote_realm_id': f'{remote_realm.id}', 'monthly_discounted_price': '50', 'annual_discounted_price': '500'})
        self.assert_in_success_response(['Monthly price for realm-name-4 changed to 50 from 0. Annual price changed to 500 from 0.'], result)
        customer.refresh_from_db()
        plan.refresh_from_db()
        next_plan.refresh_from_db()
        self.assertEqual(customer.monthly_discounted_price, 50)
        self.assertEqual(customer.annual_discounted_price, 500)
        self.assertEqual(plan.discount, None)
        self.assertEqual(next_plan.discount, '85.71')
        self.assertEqual(plan.tier, CustomerPlan.TIER_SELF_HOSTED_LEGACY)
        self.assertEqual(next_plan.tier, CustomerPlan.TIER_SELF_HOSTED_BASIC)
        self.assertIsNone(customer.minimum_licenses)
        result = self.client_post('/activity/remote/support', {'remote_realm_id': f'{remote_realm.id}', 'minimum_licenses': '50'})
        self.assert_in_success_response(['Cannot set minimum licenses; upgrade to new plan already scheduled for realm-name-4.'], result)
        customer.refresh_from_db()
        self.assertIsNone(customer.minimum_licenses)

    def test_support_deactivate_remote_server(self) -> None:
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        remote_server_with_upgrade: RemoteZulipServer = RemoteZulipServer.objects.get(hostname='zulip-4.example.com')
        self.assertFalse(remote_server_with_upgrade.deactivated)
        result = self.client_post('/activity/remote/support', {'remote_server_id': f'{remote_server_with_upgrade.id}', 'remote_server_status': 'deactivated'})
        self.assert_in_success_response([f'Cannot deactivate remote server ({remote_server_with_upgrade.hostname}) that has active or scheduled plans.'], result)
        remote_server_with_upgrade.refresh_from_db()
        self.assertFalse(remote_server_with_upgrade.deactivated)
        remote_server_no_upgrade: RemoteZulipServer = RemoteZulipServer.objects.get(hostname='zulip-5.example.com')
        self.assertFalse(remote_server_no_upgrade.deactivated)
        result = self.client_post('/activity/remote/support', {'remote_server_id': f'{remote_server_no_upgrade.id}', 'remote_server_status': 'deactivated'})
        self.assert_in_success_response([f'Remote server ({remote_server_no_upgrade.hostname}) deactivated.'], result)
        remote_server_no_upgrade.refresh_from_db()
        self.assertTrue(remote_server_no_upgrade.deactivated)
        audit_log = RemoteZulipServerAuditLog.objects.filter(event_type=AuditLogEventType.REMOTE_SERVER_DEACTIVATED).last()
        assert audit_log is not None
        self.assertEqual(audit_log.server, remote_server_no_upgrade)

    def test_support_reactivate_remote_server(self) -> None:
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        remote_server: RemoteZulipServer = RemoteZulipServer.objects.get(hostname='zulip-0.example.com')
        self.assertTrue(remote_server.deactivated)
        result = self.client_post('/activity/remote/support', {'remote_server_id': f'{remote_server.id}', 'remote_server_status': 'active'})
        self.assert_in_success_response([f'Remote server ({remote_server.hostname}) reactivated.'], result)
        remote_server.refresh_from_db()
        self.assertFalse(remote_server.deactivated)
        audit_log = RemoteZulipServerAuditLog.objects.filter(event_type=AuditLogEventType.REMOTE_SERVER_REACTIVATED).last()
        assert audit_log is not None
        self.assertEqual(audit_log.server, remote_server)


class TestSupportEndpoint(ZulipTestCase):

    def create_customer_and_plan(self, realm: Realm, monthly: bool = False) -> Customer:
        now: datetime = datetime(2016, 1, 2, tzinfo=timezone.utc)
        billing_schedule: str = CustomerPlan.BILLING_SCHEDULE_ANNUAL
        price_per_license: int = 8000
        months: int = 12
        if monthly:
            billing_schedule = CustomerPlan.BILLING_SCHEDULE_MONTHLY
            price_per_license = 800
            months = 1
        customer: Customer = Customer.objects.create(realm=realm)
        plan: CustomerPlan = CustomerPlan.objects.create(
            customer=customer,
            billing_cycle_anchor=now,
            billing_schedule=billing_schedule,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
            price_per_license=price_per_license,
            next_invoice_date=add_months(now, months)
        )
        LicenseLedger.objects.create(licenses=10, licenses_at_next_renewal=10, event_time=now, is_renewal=True, plan=plan)
        return customer

    def test_realm_support_view_queries(self) -> None:
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        with self.assert_database_query_count(18):
            result = self.client_get('/activity/support', {'q': 'zulip'}, subdomain='zulip')
            self.assertEqual(result.status_code, 200)

    def test_search(self) -> None:
        reset_email_visibility_to_everyone_in_zulip_realm()
        lear_user: Any = self.lear_user('king')
        lear_user.is_staff = True
        lear_user.save(update_fields=['is_staff'])
        lear_realm: Realm = get_realm('lear')

        def assert_user_details_in_html_response(html_response: str, full_name: str, email: str, role: str) -> None:
            self.assert_in_success_response([
                '<span class="cloud-label">Cloud user</span>\n',
                f'<h3>{full_name}</h3>',
                f'<b>Email</b>: {email}',
                '<b>Is active</b>: True<br />',
                f'<b>Role</b>: {role}<br />'
            ], html_response)

        def create_invitation(stream: str, invitee_email: str, realm: Optional[Realm] = None) -> None:
            invite_expires_in_minutes: int = 10 * 24 * 60
            with self.captureOnCommitCallbacks(execute=True):
                self.client_post('/json/invites', {
                    'invitee_emails': [invitee_email],
                    'stream_ids': orjson.dumps([self.get_stream_id(stream, realm)]).decode(),
                    'invite_expires_in_minutes': invite_expires_in_minutes,
                    'invite_as': PreregistrationUser.INVITE_AS['MEMBER']
                }, subdomain=realm.string_id if realm is not None else 'zulip')

        def check_hamlet_user_query_result(result: str) -> None:
            assert_user_details_in_html_response(result, 'King Hamlet', self.example_email('hamlet'), 'Member')
            self.assert_in_success_response([
                f'<b>Admins</b>: {self.example_email("iago")}\n',
                f'<b>Owners</b>: {self.example_email("desdemona")}\n',
                'class="copy-button" data-clipboard-text="{}">'.format(self.example_email("iago")),
                'class="copy-button" data-clipboard-text="{}">'.format(self.example_email("desdemona"))
            ], result)

        def check_lear_user_query_result(result: str) -> None:
            assert_user_details_in_html_response(result, lear_user.full_name, lear_user.email, 'Member')

        def check_othello_user_query_result(result: str) -> None:
            assert_user_details_in_html_response(result, 'Othello, the Moor of Venice', self.example_email('othello'), 'Member')

        def check_polonius_user_query_result(result: str) -> None:
            assert_user_details_in_html_response(result, 'Polonius', self.example_email('polonius'), 'Guest')

        def check_zulip_realm_query_result(result: str) -> None:
            zulip_realm: Realm = get_realm('zulip')
            first_human_user = zulip_realm.get_first_human_user()
            assert first_human_user is not None
            self.assert_in_success_response([
                f'<b>First human user</b>: {first_human_user.delivery_email}\n',
                f'<input type="hidden" name="realm_id" value="{zulip_realm.id}"',
                'Zulip Dev</h3>',
                '<option value="1" selected>Self-hosted</option>',
                '<option value="2">Limited</option>',
                'input type="number" name="monthly_discounted_price" value="None"',
                'input type="number" name="annual_discounted_price" value="None"',
                '<option value="active" selected>Active</option>',
                '<option value="deactivated" >Deactivated</option>',
                f'<option value="{zulip_realm.org_type}" selected>',
                'scrub-realm-button">',
                'data-string-id="zulip"'
            ], result)

        def check_lear_realm_query_result(result: str) -> None:
            self.assert_in_success_response([
                f'<input type="hidden" name="realm_id" value="{lear_realm.id}"',
                'Lear &amp; Co.</h3>',
                '<option value="1" selected>Self-hosted</option>',
                '<option value="2">Limited</option>',
                'input type="number" name="monthly_discounted_price" value="None"',
                'input type="number" name="annual_discounted_price" value="None"',
                '<option value="active" selected>Active</option>',
                '<option value="deactivated" >Deactivated</option>',
                'scrub-realm-button">',
                'data-string-id="lear"',
                '<b>Plan name</b>: Zulip Cloud Standard',
                '<b>Status</b>: Active',
                '<b>Billing schedule</b>: Annual',
                '<b>Licenses</b>: 2/10 (Manual)',
                '<b>Price per license</b>: $80.00',
                '<b>Annual recurring revenue</b>: $800.00',
                '<b>Start of next billing cycle</b>:',
                '<option value="send_invoice" selected>',
                '<option value="charge_automatically" >'
            ], result)

        def check_preregistration_user_query_result(result: str, email: str, invite: bool = False) -> None:
            self.assert_in_success_response([
                '<span class="cloud-label">Cloud confirmation</span>\n',
                f'<b>Email</b>: {email}'
            ], result)
            if invite:
                self.assert_in_success_response(['<h3>Invite</h3>'], result)
                self.assert_in_success_response([
                    '<b>Expires in</b>: 1\xa0week, 3\xa0days',
                    '<b>Status</b>: Link has not been used'
                ], result)
                self.assert_in_success_response([], result)
            else:
                self.assert_not_in_success_response(['<h3>Invite</h3>'], result)
                self.assert_in_success_response([
                    '<b>Expires in</b>: 1\xa0day',
                    '<b>Status</b>: Link has not been used'
                ], result)

        def check_realm_creation_query_result(result: str, email: str) -> None:
            self.assert_in_success_response([
                '<span class="cloud-label">Cloud confirmation</span>\n',
                '<h3>Realm creation</h3>\n',
                '<b>Link</b>: http://testserver/accounts/do_confirm/',
                '<b>Expires in</b>: 1\xa0day'
            ], result)

        def check_multiuse_invite_link_query_result(result: str) -> None:
            self.assert_in_success_response([
                '<span class="cloud-label">Cloud confirmation</span>\n',
                '<h3>Multiuse invite</h3>\n',
                '<b>Link</b>: http://zulip.testserver/join/',
                '<b>Expires in</b>: 1\xa0week, 3\xa0days'
            ], result)

        def check_realm_reactivation_link_query_result(result: str) -> None:
            self.assert_in_success_response([
                '<span class="cloud-label">Cloud confirmation</span>\n',
                '<h3>Realm reactivation</h3>\n',
                '<b>Link</b>: http://zulip.testserver/reactivate/',
                '<b>Expires in</b>: 1\xa0day'
            ], result)

        def get_check_query_result(query: str, count: int, subdomain: str = 'zulip') -> "TestHttpResponse":
            result = self.client_get('/activity/support', {'q': query}, subdomain=subdomain)
            self.assertEqual(result.content.decode().count('support-query-result'), count)
            return result

        self.login('cordelia')
        result = self.client_get('/activity/support')
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        self.login('iago')
        do_change_user_setting(self.example_user('hamlet'), 'email_address_visibility', UserProfile.EMAIL_ADDRESS_VISIBILITY_NOBODY, acting_user=None)
        self.create_customer_and_plan(lear_realm)
        result = self.client_get('/activity/support')
        self.assert_in_success_response(['<input type="text" name="q" class="input-xxlarge search-query"'], result)
        result = get_check_query_result(self.example_email('hamlet'), 1)
        check_hamlet_user_query_result(result)
        check_zulip_realm_query_result(result)
        assert self.example_email('hamlet') != self.example_email('hamlet').upper()
        result = get_check_query_result(self.example_email('hamlet').upper(), 1)
        check_hamlet_user_query_result(result)
        check_zulip_realm_query_result(result)
        result = get_check_query_result(lear_user.email, 1)
        check_lear_user_query_result(result)
        check_lear_realm_query_result(result)
        result = get_check_query_result(self.example_email('polonius'), 1)
        check_polonius_user_query_result(result)
        check_zulip_realm_query_result(result)
        result = get_check_query_result('lear', 1)
        check_lear_realm_query_result(result)
        result = get_check_query_result('http://lear.testserver', 1)
        check_lear_realm_query_result(result)
        with self.settings(REALM_HOSTS={'zulip': 'localhost'}):
            result = get_check_query_result('http://localhost', 1)
            check_zulip_realm_query_result(result)
        result = get_check_query_result('hamlet@zulip.com, lear', 2)
        check_hamlet_user_query_result(result)
        check_zulip_realm_query_result(result)
        check_lear_realm_query_result(result)
        result = get_check_query_result('King hamlet,lear', 2)
        check_hamlet_user_query_result(result)
        check_zulip_realm_query_result(result)
        check_lear_realm_query_result(result)
        result = get_check_query_result('Othello, the Moor of Venice', 1)
        check_othello_user_query_result(result)
        check_zulip_realm_query_result(result)
        result = get_check_query_result('lear, Hamlet <hamlet@zulip.com>', 2)
        check_hamlet_user_query_result(result)
        check_zulip_realm_query_result(result)
        check_lear_realm_query_result(result)
        self.client_post('/accounts/home/', {'email': self.nonreg_email('test')})
        self.login('iago')

        def query_result_from_before(*args: Any, **kwargs: Any) -> "TestHttpResponse":
            with time_machine.travel(timezone_now() - timedelta(minutes=50), tick=False):
                return get_check_query_result(*args, **kwargs)
        result = query_result_from_before(self.nonreg_email('test'), 1)
        check_preregistration_user_query_result(result, self.nonreg_email('test'))
        check_zulip_realm_query_result(result)
        create_invitation('Denmark', self.nonreg_email('test1'))
        result = query_result_from_before(self.nonreg_email('test1'), 1)
        check_preregistration_user_query_result(result, self.nonreg_email('test1'), invite=True)
        check_zulip_realm_query_result(result)
        email = self.nonreg_email('alice')
        self.submit_realm_creation_form(email, realm_subdomain='custom-test', realm_name='Zulip test')
        result = query_result_from_before(email, 1)
        check_realm_creation_query_result(result, email)
        invite_expires_in_minutes = 10 * 24 * 60
        do_create_multiuse_invite_link(self.example_user('hamlet'), invited_as=1, invite_expires_in_minutes=invite_expires_in_minutes, include_realm_default_subscriptions=False)
        result = query_result_from_before('zulip', 2)
        check_multiuse_invite_link_query_result(result)
        check_zulip_realm_query_result(result)
        MultiuseInvite.objects.all().delete()
        do_send_realm_reactivation_email(get_realm('zulip'), acting_user=None)
        result = query_result_from_before('zulip', 2)
        check_realm_reactivation_link_query_result(result)
        check_zulip_realm_query_result(result)
        lear_nonreg_email = 'newguy@lear.org'
        self.client_post('/accounts/home/', {'email': lear_nonreg_email}, subdomain='lear')
        result = query_result_from_before(lear_nonreg_email, 1)
        check_preregistration_user_query_result(result, lear_nonreg_email)
        check_lear_realm_query_result(result)
        self.login_user(lear_user)
        create_invitation('general', 'newguy2@lear.org', lear_realm)
        result = query_result_from_before('newguy2@lear.org', 1, lear_realm.string_id)
        check_preregistration_user_query_result(result, 'newguy2@lear.org', invite=True)
        check_lear_realm_query_result(result)

    def test_get_org_type_display_name(self) -> None:
        self.assertEqual(get_org_type_display_name(Realm.ORG_TYPES['business']['id']), 'Business')
        self.assertEqual(get_org_type_display_name(883), '')

    def test_unspecified_org_type_correctly_displayed(self) -> None:
        """
        Unspecified org type is special in that it is marked to not be shown
        on the registration page (because organitions are not meant to be able to choose it),
        but should be correctly shown at the /support/ endpoint.
        """
        realm: Realm = get_realm('zulip')
        do_change_realm_org_type(realm, 0, acting_user=None)
        self.assertEqual(realm.org_type, 0)
        self.login('iago')
        result = self.client_get('/activity/support', {'q': 'zulip'}, subdomain='zulip')
        self.assert_in_success_response([f'<input type="hidden" name="realm_id" value="{realm.id}"', '<option value="0" selected>'], result)

    def test_change_billing_modality(self) -> None:
        realm: Realm = get_realm('zulip')
        customer: Customer = self.create_customer_and_plan(realm)
        cordelia: Any = self.example_user('cordelia')
        self.login_user(cordelia)
        result = self.client_post('/activity/support', {'realm_id': f'{realm.id}', 'billing_method': 'charge_automatically'})
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        result = self.client_post('/activity/support', {'realm_id': f'{realm.id}', 'billing_modality': 'charge_automatically'})
        self.assert_in_success_response(['Billing collection method of zulip updated to charge automatically'], result)
        plan: Optional[CustomerPlan] = get_current_plan_by_customer(customer)
        assert plan is not None
        self.assertEqual(plan.charge_automatically, True)
        result = self.client_post('/activity/support', {'realm_id': f'{realm.id}', 'billing_modality': 'send_invoice'})
        self.assert_in_success_response(['Billing collection method of zulip updated to send invoice'], result)
        plan.refresh_from_db()
        self.assertEqual(plan.charge_automatically, False)

    def test_change_realm_plan_type(self) -> None:
        cordelia: Any = self.example_user('cordelia')
        self.login_user(cordelia)
        result = self.client_post('/activity/support', {'realm_id': f'{cordelia.realm_id}', 'plan_type': '2'})
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        with mock.patch('corporate.views.support.do_change_realm_plan_type') as m:
            result = self.client_post('/activity/support', {'realm_id': f'{iago.realm_id}', 'plan_type': '2'})
            m.assert_called_once_with(get_realm('zulip'), 2, acting_user=iago)
            self.assert_in_success_response(['Plan type of zulip changed from Self-hosted to Limited'], result)
        with mock.patch('corporate.views.support.do_change_realm_plan_type') as m:
            result = self.client_post('/activity/support', {'realm_id': f'{iago.realm_id}', 'plan_type': '10'})
            m.assert_called_once_with(get_realm('zulip'), 10, acting_user=iago)
            self.assert_in_success_response(['Plan type of zulip changed from Self-hosted to Plus'], result)

    def test_change_org_type(self) -> None:
        cordelia: Any = self.example_user('cordelia')
        self.login_user(cordelia)
        result = self.client_post('/activity/support', {'realm_id': f'{cordelia.realm_id}', 'org_type': '70'})
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        with mock.patch('corporate.views.support.do_change_realm_org_type') as m:
            result = self.client_post('/activity/support', {'realm_id': f'{iago.realm_id}', 'org_type': '70'})
            m.assert_called_once_with(get_realm('zulip'), 70, acting_user=iago)
            self.assert_in_success_response(['Organization type of zulip changed from Business to Government'], result)

    def test_change_max_invites(self) -> None:
        realm: Realm = get_realm('zulip')
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        self.assertEqual(realm.max_invites, settings.INVITES_DEFAULT_REALM_DAILY_MAX)
        result = self.client_post('/activity/support', {'realm_id': f'{realm.id}', 'max_invites': '1'})
        self.assert_in_success_response(['Cannot update maximum number of daily invitations for zulip, because 1 is less than the default for the current plan type.'], result)
        realm.refresh_from_db()
        self.assertEqual(realm.max_invites, settings.INVITES_DEFAULT_REALM_DAILY_MAX)
        result = self.client_post('/activity/support', {'realm_id': f'{realm.id}', 'max_invites': '700'})
        self.assert_in_success_response(['Maximum number of daily invitations for zulip updated to 700.'], result)
        realm.refresh_from_db()
        self.assertEqual(realm.max_invites, 700)
        result = self.client_post('/activity/support', {'realm_id': f'{realm.id}', 'max_invites': '0'})
        self.assert_in_success_response(['Maximum number of daily invitations for zulip updated to the default for the current plan type.'], result)
        realm.refresh_from_db()
        self.assertEqual(realm.max_invites, settings.INVITES_DEFAULT_REALM_DAILY_MAX)
        result = self.client_post('/activity/support', {'realm_id': f'{realm.id}', 'max_invites': '0'})
        self.assert_in_success_response(['Cannot update maximum number of daily invitations for zulip, because the default for the current plan type is already set.'], result)

    def test_attach_discount(self) -> None:
        lear_realm: Realm = get_realm('lear')
        customer: Customer = self.create_customer_and_plan(lear_realm, True)
        cordelia: Any = self.example_user('cordelia')
        self.login_user(cordelia)
        discount_change_data: dict[str, str] = {'realm_id': f'{lear_realm.id}', 'monthly_discounted_price': '600', 'annual_discounted_price': '6000'}
        result = self.client_post('/activity/support', discount_change_data)
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'required_plan_tier': f'{CustomerPlan.TIER_CLOUD_STANDARD}'})
        self.assert_in_success_response(['Required plan tier for lear set to Zulip Cloud Standard.'], result)
        result = self.client_post('/activity/support', discount_change_data)
        self.assert_in_success_response(['Monthly price for lear changed to 600 from 0. Annual price changed to 6000 from 0.'], result)
        customer.refresh_from_db()
        plan: Optional[CustomerPlan] = get_current_plan_by_customer(customer)
        assert plan is not None
        self.assertEqual(customer.monthly_discounted_price, 600)
        self.assertEqual(customer.annual_discounted_price, 6000)
        self.assertEqual(plan.discount, '25')
        start_next_billing_cycle: datetime = start_of_next_billing_cycle(plan, timezone_now())
        billing_cycle_string: str = start_next_billing_cycle.strftime('%d %B %Y')
        twenty_five_percent_discounted_response = [
            '<b>Plan name</b>: Zulip Cloud Standard',
            '<b>Status</b>: Active',
            '<b>Discount</b>: 25%',
            '<b>Billing schedule</b>: Monthly',
            '<b>Licenses</b>: 2/10 (Manual)',
            '<b>Price per license</b>: $6.00',
            '<b>Annual recurring revenue</b>: $720.00',
            f'<b>Start of next billing cycle</b>: {billing_cycle_string}'
        ]
        result = self.client_get('/activity/support', {'q': 'lear'})
        self.assert_in_success_response(twenty_five_percent_discounted_response, result)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'monthly_discounted_price': '800', 'annual_discounted_price': '8000'})
        no_discount_response = [
            '<b>Plan name</b>: Zulip Cloud Standard',
            '<b>Status</b>: Active',
            '<b>Billing schedule</b>: Monthly',
            '<b>Licenses</b>: 2/10 (Manual)',
            '<b>Price per license</b>: $8.00',
            '<b>Annual recurring revenue</b>: $960.00',
            f'<b>Start of next billing cycle</b>: {billing_cycle_string}'
        ]
        result = self.client_get('/activity/support', {'q': 'lear'})
        self.assert_in_success_response(no_discount_response, result)
        self.client_post('/activity/support', discount_change_data)
        result = self.client_get('/activity/support', {'q': 'lear'})
        self.assert_in_success_response(twenty_five_percent_discounted_response, result)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'monthly_discounted_price': '0', 'annual_discounted_price': '0'})
        result = self.client_get('/activity/support', {'q': 'lear'})
        self.assert_in_success_response(no_discount_response, result)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'monthly_discounted_price': '600', 'annual_discounted_price': '0'})
        monthly_discounted_response = [
            '<b>Plan name</b>: Zulip Cloud Standard',
            '<b>Status</b>: Active',
            '<b>Discount</b>: 25%',
            '<b>Billing schedule</b>: Monthly',
            '<b>Licenses</b>: 2/10 (Manual)',
            '<b>Price per license</b>: $6.00',
            '<b>Annual recurring revenue</b>: $720.00',
            f'<b>Start of next billing cycle</b>: {billing_cycle_string}'
        ]
        result = self.client_get('/activity/support', {'q': 'lear'})
        self.assert_in_success_response(monthly_discounted_response, result)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'monthly_discounted_price': '0', 'annual_discounted_price': '6000'})
        result = self.client_get('/activity/support', {'q': 'lear'})
        self.assert_in_success_response(no_discount_response, result)
        plan = get_current_plan_by_customer(customer)
        assert plan is not None
        plan.status = CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE
        plan.save(update_fields=['status'])
        support_admin: Any = self.example_user('iago')
        assert plan.next_invoice_date is not None
        RealmBillingSession(support_admin, lear_realm, support_session=True).make_end_of_cycle_updates_if_needed(plan, plan.next_invoice_date)
        result = self.client_get('/activity/support', {'q': 'lear'})
        self.assert_in_success_response([
            '<b>Plan name</b>: Zulip Cloud Standard',
            '<b>Status</b>: Active',
            '<b>Discount</b>: 25%',
            '<b>Billing schedule</b>: Annual',
            '<b>Licenses</b>: 2/10 (Manual)',
            '<b>Price per license</b>: $60.00',
            '<b>Annual recurring revenue</b>: $600.00'
        ], result)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'monthly_discounted_price': '600', 'annual_discounted_price': '0'})
        result = self.client_get('/activity/support', {'q': 'lear'})
        self.assert_in_success_response([
            '<b>Plan name</b>: Zulip Cloud Standard',
            '<b>Status</b>: Active',
            '<b>Billing schedule</b>: Annual',
            '<b>Licenses</b>: 2/10 (Manual)',
            '<b>Price per license</b>: $80.00',
            '<b>Annual recurring revenue</b>: $800.00'
        ], result)

    def test_change_sponsorship_status(self) -> None:
        lear_realm: Realm = get_realm('lear')
        self.assertIsNone(get_customer_by_realm(lear_realm))
        cordelia: Any = self.example_user('cordelia')
        self.login_user(cordelia)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'sponsorship_pending': 'true'})
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'sponsorship_pending': 'true'})
        self.assert_in_success_response(['lear marked as pending sponsorship.'], result)
        customer: Optional[Customer] = get_customer_by_realm(lear_realm)
        assert customer is not None
        self.assertTrue(customer.sponsorship_pending)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'sponsorship_pending': 'false'})
        self.assert_in_success_response(['lear is no longer pending sponsorship.'], result)
        customer = get_customer_by_realm(lear_realm)
        assert customer is not None
        self.assertFalse(customer.sponsorship_pending)

    def test_approve_sponsorship(self) -> None:
        support_admin: Any = self.example_user('iago')
        lear_realm: Realm = get_realm('lear')
        billing_session = RealmBillingSession(user=support_admin, realm=lear_realm, support_session=True)
        billing_session.update_customer_sponsorship_status(True)
        king_user: Any = self.lear_user('king')
        king_user.role = UserProfile.ROLE_REALM_OWNER
        king_user.save()
        cordelia: Any = self.example_user('cordelia')
        self.login_user(cordelia)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'approve_sponsorship': 'true'})
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'approve_sponsorship': 'true'})
        self.assert_in_success_response(['Sponsorship approved for lear'], result)
        lear_realm.refresh_from_db()
        self.assertEqual(lear_realm.plan_type, Realm.PLAN_TYPE_STANDARD_FREE)
        customer: Optional[Customer] = get_customer_by_realm(lear_realm)
        assert customer is not None
        self.assertFalse(customer.sponsorship_pending)
        messages = UserMessage.objects.filter(user_profile=king_user)
        self.assertIn('request for sponsored hosting has been approved', messages[0].message.content)
        self.assert_length(messages, 1)

    def test_update_current_plan_end_date(self) -> None:
        lear_realm: Realm = get_realm('lear')
        customer: Optional[Customer] = get_customer_by_realm(lear_realm)
        assert customer is None
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'plan_end_date': '2020-01-01'})
        self.assert_in_success_response(['Cannot update current plan for lear to end on 2020-01-01.'], result)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'plan_end_date': '2050-03-01'})
        self.assert_in_success_response(['No current plan for lear.'], result)
        customer = self.create_customer_and_plan(lear_realm, True)
        assert customer is not None
        current_plan: Optional[CustomerPlan] = get_current_plan_by_customer(customer)
        assert current_plan is not None
        self.assertIsNone(current_plan.end_date)
        next_invoice_date = current_plan.next_invoice_date
        assert next_invoice_date is not None
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'plan_end_date': '2050-03-01'})
        self.assert_in_success_response(['Current plan for lear updated to end on 2050-03-01.'], result)
        current_plan.refresh_from_db()
        self.assertEqual(current_plan.end_date, datetime(2050, 3, 1, tzinfo=timezone.utc))
        self.assertEqual(current_plan.next_invoice_date, next_invoice_date)

    def test_configure_fixed_price_plan_without_current_plan(self) -> None:
        plan_offer_tier: int = CustomerPlanOffer.TIER_CLOUD_STANDARD
        plan_offer_tier_name: str = CustomerPlanOffer.name_from_tier(plan_offer_tier)
        lear_realm: Realm = get_realm('lear')
        customer: Optional[Customer] = get_customer_by_realm(lear_realm)
        assert customer is None
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'fixed_price': 360})
        self.assert_in_success_response(['Required plan tier should not be set to None'], result)
        customer = get_customer_by_realm(lear_realm)
        assert customer is not None
        self.assertIsNone(customer.required_plan_tier)
        plan = get_current_plan_by_customer(customer)
        assert plan is None
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'required_plan_tier': f'{plan_offer_tier}'})
        self.assert_in_success_response([f'Required plan tier for lear set to {plan_offer_tier_name}.'], result)
        customer.refresh_from_db()
        assert customer.required_plan_tier is not None
        plan_offer = get_configured_fixed_price_plan_offer(customer, customer.required_plan_tier)
        assert plan_offer is None
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'fixed_price': 360})
        self.assert_in_success_response([f'Customer can now buy a fixed price {plan_offer_tier_name} plan.'], result)
        customer.refresh_from_db()
        plan_offer = get_configured_fixed_price_plan_offer(customer, customer.required_plan_tier)
        assert plan_offer is not None
        self.assertEqual(plan_offer.status, CustomerPlanOffer.CONFIGURED)
        self.assertEqual(plan_offer.fixed_price, 36000)
        self.assertIsNone(plan_offer.sent_invoice_id)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'delete_fixed_price_next_plan': 'true'})
        self.assert_in_success_response(['Fixed-price plan offer deleted'], result)
        customer.refresh_from_db()
        plan_offer = get_configured_fixed_price_plan_offer(customer, customer.required_plan_tier)
        self.assertIsNone(plan_offer)

    def test_configure_upgrade_fixed_price_plan_with_current_plan(self) -> None:
        plan_offer_tier: int = CustomerPlanOffer.TIER_CLOUD_PLUS
        plan_offer_tier_name: str = CustomerPlanOffer.name_from_tier(plan_offer_tier)
        lear_realm: Realm = get_realm('lear')
        customer: Customer = self.create_customer_and_plan(lear_realm, True)
        plan: Optional[CustomerPlan] = get_current_plan_by_customer(customer)
        assert plan is not None
        self.assertIsNone(plan.end_date)
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'fixed_price': 360})
        self.assert_in_success_response(['Required plan tier should not be set to None'], result)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'required_plan_tier': f'{plan_offer_tier}'})
        self.assert_in_success_response([f'Required plan tier for lear set to {plan_offer_tier_name}.'], result)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'fixed_price': 360})
        self.assert_in_success_response(['Configure lear current plan end-date, before scheduling a new plan.'], result)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'plan_end_date': '2050-03-01'})
        self.assert_in_success_response(['Current plan for lear updated to end on 2050-03-01.'], result)
        plan.refresh_from_db()
        self.assertEqual(plan.end_date, datetime(2050, 3, 1, tzinfo=timezone.utc))
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'required_plan_tier': f'{plan_offer_tier}'})
        self.assert_in_success_response([f'Required plan tier for lear set to {plan_offer_tier_name}.'], result)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'fixed_price': 360})
        self.assert_in_success_response(['New plan for lear cannot be scheduled until all the invoices of the current plan are processed.'], result)
        billing_session = RealmBillingSession(user=iago, realm=lear_realm, support_session=True)
        next_billing_cycle = billing_session.get_next_billing_cycle(plan).strftime('%Y-%m-%d')
        with time_machine.travel(datetime(2016, 1, 3, tzinfo=timezone.utc), tick=False):
            result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'plan_end_date': next_billing_cycle})
        self.assert_in_success_response([f'Current plan for lear updated to end on {next_billing_cycle}.'], result)
        plan.refresh_from_db()
        self.assertEqual(plan.end_date, datetime(2016, 2, 2, tzinfo=timezone.utc))
        self.assertEqual(plan.status, CustomerPlan.ACTIVE)
        next_plan = billing_session.get_next_plan(plan)
        assert next_plan is None
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'fixed_price': 360})
        self.assert_in_success_response([f'Fixed price {plan_offer_tier_name} plan scheduled to start on {next_billing_cycle}.'], result)
        plan.refresh_from_db()
        self.assertEqual(plan.end_date, datetime(2016, 2, 2, tzinfo=timezone.utc))
        self.assertEqual(plan.status, CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END)
        next_plan = billing_session.get_next_plan(plan)
        assert next_plan is not None
        self.assertEqual(next_plan.fixed_price, 36000)
        self.assertEqual(next_plan.billing_cycle_anchor, plan.end_date)
        self.assertEqual(next_plan.charge_automatically, plan.charge_automatically)
        self.assertTrue(next_plan.automanage_licenses)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'delete_fixed_price_next_plan': 'true'})
        self.assert_in_success_response(['Fixed-price scheduled plan deleted'], result)
        next_plan = billing_session.get_next_plan(plan)
        self.assertIsNone(next_plan)

    def test_approve_sponsorship_deactivated_realm(self) -> None:
        support_admin: Any = self.example_user('iago')
        with self.settings(BILLING_ENABLED=True):
            limited_realm: Realm = do_create_realm('limited', 'limited')
            self.assertEqual(limited_realm.plan_type, Realm.PLAN_TYPE_LIMITED)
            billing_session = RealmBillingSession(user=support_admin, realm=limited_realm, support_session=True)
            billing_session.update_customer_sponsorship_status(True)
        limited_realm.deactivated = True
        limited_realm.save()
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        result = self.client_post('/activity/support', {'realm_id': f'{limited_realm.id}', 'approve_sponsorship': 'true'})
        self.assertIn(b'Realm has been deactivated', result.content)
        limited_realm.refresh_from_db()
        self.assertEqual(limited_realm.plan_type, Realm.PLAN_TYPE_LIMITED)

    def test_activate_or_deactivate_realm(self) -> None:
        cordelia: Any = self.example_user('cordelia')
        lear_realm: Realm = get_realm('lear')
        self.login_user(cordelia)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'status': 'deactivated'})
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        self.login('iago')
        with mock.patch('corporate.views.support.do_deactivate_realm') as m:
            result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'status': 'deactivated'})
            m.assert_called_once_with(lear_realm, acting_user=self.example_user('iago'), deactivation_reason='owner_request', email_owners=True)
            self.assert_in_success_response(['lear deactivated'], result)
        with mock.patch('corporate.views.support.do_send_realm_reactivation_email') as m:
            result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'status': 'active'})
            m.assert_called_once_with(lear_realm, acting_user=self.example_user('iago'))
            self.assert_in_success_response(['Realm reactivation email sent to admins of lear'], result)

    def test_change_subdomain(self) -> None:
        cordelia: Any = self.example_user('cordelia')
        lear_realm: Realm = get_realm('lear')
        self.login_user(cordelia)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'new_subdomain': 'new_name'})
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        self.login('iago')
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'new_subdomain': 'new-name'})
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/activity/support?q=new-name')
        realm_id = lear_realm.id
        lear_realm = get_realm('new-name')
        self.assertEqual(lear_realm.id, realm_id)
        self.assertTrue(Realm.objects.filter(string_id='lear').exists())
        self.assertTrue(Realm.objects.filter(string_id='lear')[0].deactivated)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'new_subdomain': 'new-name'})
        self.assert_in_success_response(['Subdomain already in use. Please choose a different one.'], result)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'new_subdomain': 'zulip'})
        self.assert_in_success_response(['Subdomain already in use. Please choose a different one.'], result)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'new_subdomain': 'lear'})
        self.assert_in_success_response(['Subdomain already in use. Please choose a different one.'], result)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'new_subdomain': 'your-org'})
        self.assert_in_success_response(['Subdomain reserved. Please choose a different one.'], result)

    def test_modify_plan_for_downgrade_at_end_of_billing_cycle(self) -> None:
        realm: Realm = get_realm('zulip')
        customer: Customer = self.create_customer_and_plan(realm)
        cordelia: Any = self.example_user('cordelia')
        self.login_user(cordelia)
        result = self.client_post('/activity/support', {'realm_id': f'{realm.id}', 'modify_plan': 'downgrade_at_billing_cycle_end'})
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        with self.assertLogs('corporate.stripe', 'INFO') as m:
            result = self.client_post('/activity/support', {'realm_id': f'{realm.id}', 'modify_plan': 'downgrade_at_billing_cycle_end'})
            self.assert_in_success_response(['zulip marked for downgrade at the end of billing cycle'], result)
            customer.refresh_from_db()
            plan: Optional[CustomerPlan] = get_current_plan_by_customer(customer)
            assert plan is not None
            self.assertEqual(plan.status, CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE)
            expected_log = f'INFO:corporate.stripe:Change plan status: Customer.id: {customer.id}, CustomerPlan.id: {plan.id}, status: {CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE}'
            self.assertEqual(m.output[0], expected_log)

    def test_modify_plan_for_downgrade_now_without_additional_licenses(self) -> None:
        realm: Realm = get_realm('zulip')
        customer: Customer = self.create_customer_and_plan(realm)
        plan: Optional[CustomerPlan] = get_current_plan_by_customer(customer)
        assert plan is not None
        cordelia: Any = self.example_user('cordelia')
        self.login_user(cordelia)
        result = self.client_post('/activity/support', {'realm_id': f'{realm.id}', 'modify_plan': 'downgrade_now_without_additional_licenses'})
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        iago: Any = self.example_user('iago')
        self.login_user(iago)
        result = self.client_post('/activity/support', {'realm_id': f'{iago.realm_id}', 'modify_plan': 'downgrade_now_without_additional_licenses'})
        self.assert_in_success_response(['zulip downgraded without creating additional invoices'], result)
        plan.refresh_from_db()
        self.assertEqual(plan.status, CustomerPlan.ENDED)
        realm.refresh_from_db()
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_LIMITED)

    def test_scrub_realm(self) -> None:
        cordelia: Any = self.example_user('cordelia')
        lear_realm: Realm = get_realm('lear')
        self.login_user(cordelia)
        result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'discount': '25'})
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        self.login('iago')
        with mock.patch('corporate.views.support.do_scrub_realm') as m:
            result = self.client_post('/activity/support', {'realm_id': f'{lear_realm.id}', 'scrub_realm': 'true'})
            m.assert_called_once_with(lear_realm, acting_user=self.example_user('iago'))
            self.assert_in_success_response(['lear scrubbed'], result)

    def test_delete_user(self) -> None:
        cordelia: Any = self.example_user('cordelia')
        hamlet: Any = self.example_user('hamlet')
        hamlet_email: str = hamlet.delivery_email
        realm: Realm = get_realm('zulip')
        self.login_user(cordelia)
        result = self.client_post('/activity/support', {'realm_id': f'{realm.id}', 'delete_user_by_id': hamlet.id})
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/login/')
        self.login('iago')
        with mock.patch('corporate.views.support.do_delete_user_preserving_messages') as m:
            result = self.client_post('/activity/support', {'realm_id': f'{realm.id}', 'delete_user_by_id': hamlet.id})
            m.assert_called_once_with(hamlet)
            self.assert_in_success_response([f'{hamlet_email} in zulip deleted'], result)
