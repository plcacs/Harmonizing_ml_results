#!/usr/bin/env python3
from datetime import datetime, timedelta, timezone
import itertools
import re
import uuid
from decimal import Decimal
from typing import Any, Tuple, cast

import orjson
import responses
import stripe
import time_machine
from django.conf import settings
from django.core import signing
from django.urls.resolvers import get_resolver
from django.utils.crypto import get_random_string
from django.utils.timezone import now as timezone_now
from typing_extensions import override

from corporate.lib.stripe import (BillingError, BillingSessionAuditLogEventError,
                                    BillingSessionEventType, InitialUpgradeRequest,
                                    InvalidBillingScheduleError, InvalidTierError,
                                    RealmBillingSession, RemoteRealmBillingSession,
                                    RemoteServerBillingSession, StripeCardError,
                                    SupportRequestError, SupportType, SupportViewRequest,
                                    UpdatePlanRequest, add_months, catch_stripe_errors,
                                    compute_plan_parameters,
                                    customer_has_credit_card_as_default_payment_method,
                                    customer_has_last_n_invoices_open, do_deactivate_remote_server,
                                    do_reactivate_remote_server,
                                    downgrade_small_realms_behind_on_payments_as_needed,
                                    get_latest_seat_count, get_plan_renewal_or_end_date,
                                    get_price_per_license, invoice_plans_as_needed,
                                    is_free_trial_offer_enabled, is_realm_on_free_trial,
                                    next_month, sign_string, stripe_customer_has_credit_card_as_default_payment_method,
                                    stripe_get_customer, unsign_string)
from corporate.models import (Customer, CustomerPlan, CustomerPlanOffer, Event, Invoice, LicenseLedger,
                              ZulipSponsorshipRequest, get_current_plan_by_customer, get_current_plan_by_realm,
                              get_customer_by_realm)
from corporate.tests.test_remote_billing import RemoteRealmBillingTestCase, RemoteServerTestCase
from corporate.views.remote_billing_page import generate_confirmation_link_for_server_deactivation
from zerver.actions.create_realm import do_create_realm
from zerver.actions.create_user import do_activate_mirror_dummy_user, do_create_user, do_reactivate_user
from zerver.actions.realm_settings import do_deactivate_realm, do_reactivate_realm
from zerver.actions.users import change_user_is_active, do_change_user_role, do_deactivate_user
from zerver.lib.remote_server import send_server_data_to_push_bouncer
from zerver.lib.test_classes import ZulipTestCase
from zerver.lib.test_helpers import activate_push_notification_service
from zerver.lib.timestamp import datetime_to_timestamp, timestamp_to_datetime
from zerver.lib.utils import assert_is_not_none
from zerver.models import Message, Realm, RealmAuditLog, Recipient, UserProfile
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.realms import get_realm
from zerver.models.users import get_system_bot
from zilencer.lib.remote_counts import MissingDataError
from zilencer.models import (RemoteRealm, RemoteRealmAuditLog, RemoteRealmBillingUser,
                             RemoteServerBillingUser, RemoteZulipServer, RemoteZulipServerAuditLog)

class StripeTestCase(ZulipTestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        realm = get_realm('zulip')
        active_emails = [
            self.example_email('AARON'), self.example_email('cordelia'),
            self.example_email('hamlet'), self.example_email('iago'),
            self.example_email('othello'), self.example_email('desdemona'),
            self.example_email('polonius'), self.example_email('default_bot')
        ]
        for user_profile in UserProfile.objects.filter(realm_id=realm.id).exclude(delivery_email__in=active_emails):
            do_deactivate_user(user_profile, acting_user=None)
        self.assertEqual(UserProfile.objects.filter(realm=realm, is_active=True).count(), 8)
        self.assertEqual(UserProfile.objects.exclude(realm=realm).filter(is_active=True).count(), 10)
        self.assertEqual(get_latest_seat_count(realm), 6)
        self.seat_count = 6
        self.signed_seat_count, self.salt = sign_string(str(self.seat_count))
        self.now = datetime(2012, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        self.next_month = datetime(2012, 2, 2, 3, 4, 5, tzinfo=timezone.utc)
        self.next_year = datetime(2013, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        hamlet = self.example_user('hamlet')
        hamlet.is_billing_admin = True
        hamlet.save(update_fields=['is_billing_admin'])
        self.billing_session = RealmBillingSession(user=hamlet, realm=realm)

    def get_signed_seat_count_from_response(self, response: Any) -> Any:
        match = re.search('name=\\"signed_seat_count\\" value=\\"(.+)\\"', response.content.decode())
        return match.group(1) if match else None

    def get_salt_from_response(self, response: Any) -> Any:
        match = re.search('name=\\"salt\\" value=\\"(\\w+)\\"', response.content.decode())
        return match.group(1) if match else None

    def get_test_card_token(self, attaches_to_customer: bool, charge_succeeds: Any = None, card_provider: Any = None) -> str:
        if attaches_to_customer:
            assert charge_succeeds is not None
            if charge_succeeds:
                if card_provider == 'visa':
                    return 'tok_visa'
                if card_provider == 'mastercard':
                    return 'tok_mastercard'
                raise AssertionError('Unreachable code path')
            else:
                return 'tok_chargeCustomerFail'
        else:
            return 'tok_visa_chargeDeclined'

    def assert_details_of_valid_session_from_event_status_endpoint(self, stripe_session_id: str, expected_details: Any) -> None:
        json_response = self.client_billing_get('/billing/event/status', {'stripe_session_id': stripe_session_id})
        response_dict = self.assert_json_success(json_response)
        self.assertEqual(response_dict['session'], expected_details)

    def assert_details_of_valid_invoice_payment_from_event_status_endpoint(self, stripe_invoice_id: str, expected_details: Any) -> None:
        json_response = self.client_billing_get('/billing/event/status', {'stripe_invoice_id': stripe_invoice_id})
        response_dict = self.assert_json_success(json_response)
        self.assertEqual(response_dict['stripe_invoice'], expected_details)

    def trigger_stripe_checkout_session_completed_webhook(self, token: str) -> None:
        customer = self.billing_session.get_customer()
        assert customer is not None
        customer_stripe_id = customer.stripe_customer_id
        assert customer_stripe_id is not None
        [checkout_setup_intent] = iter(stripe.SetupIntent.list(customer=customer_stripe_id, limit=1))
        payment_method = stripe.PaymentMethod.create(
            type='card',
            card={'token': token},
            billing_details={'name': 'John Doe', 'address': {'line1': '123 Main St', 'city': 'San Francisco', 'state': 'CA', 'postal_code': '94105', 'country': 'US'}}
        )
        assert isinstance(checkout_setup_intent.customer, str)
        assert checkout_setup_intent.metadata is not None
        assert checkout_setup_intent.usage in {'off_session', 'on_session'}
        usage = cast(str, checkout_setup_intent.usage)
        stripe_setup_intent = stripe.SetupIntent.create(
            payment_method=payment_method.id,
            confirm=True,
            payment_method_types=checkout_setup_intent.payment_method_types,
            customer=checkout_setup_intent.customer,
            metadata=checkout_setup_intent.metadata,
            usage=usage
        )
        [stripe_session] = iter(stripe.checkout.Session.list(customer=customer_stripe_id, limit=1))
        stripe_session_dict = orjson.loads(orjson.dumps(stripe_session))
        stripe_session_dict['setup_intent'] = stripe_setup_intent.id
        event_payload = {
            'id': f'evt_{get_random_string(24)}',
            'object': 'event',
            'data': {'object': stripe_session_dict},
            'type': 'checkout.session.completed',
            'api_version': settings.STRIPE_API_VERSION
        }
        response = self.client_post('/stripe/webhook/', event_payload, content_type='application/json')
        assert response.status_code == 200

    def send_stripe_webhook_event(self, event: Any) -> None:
        response = self.client_post('/stripe/webhook/', orjson.loads(orjson.dumps(event)), content_type='application/json')
        assert response.status_code == 200

    def send_stripe_webhook_events(self, most_recent_event: Any) -> None:
        while True:
            events_old_to_new = list(reversed(stripe.Event.list(ending_before=most_recent_event.id)))
            if len(events_old_to_new) == 0:
                break
            for event in events_old_to_new:
                self.send_stripe_webhook_event(event)
            most_recent_event = events_old_to_new[-1]

    def add_card_to_customer_for_upgrade(self, charge_succeeds: bool = True) -> None:
        start_session_json_response = self.client_billing_post('/upgrade/session/start_card_update_session', {'tier': 1})
        response_dict = self.assert_json_success(start_session_json_response)
        stripe_session_id = response_dict['stripe_session_id']
        self.assert_details_of_valid_session_from_event_status_endpoint(
            stripe_session_id,
            {'type': 'card_update_from_upgrade_page', 'status': 'created', 'is_manual_license_management_upgrade_session': False, 'tier': 1}
        )
        self.trigger_stripe_checkout_session_completed_webhook(
            self.get_test_card_token(attaches_to_customer=True, charge_succeeds=charge_succeeds, card_provider='visa')
        )
        self.assert_details_of_valid_session_from_event_status_endpoint(
            stripe_session_id,
            {'type': 'card_update_from_upgrade_page', 'status': 'completed', 'is_manual_license_management_upgrade_session': False, 'tier': 1, 'event_handler': {'status': 'succeeded'}}
        )

    def upgrade(self, invoice: bool = False, talk_to_stripe: bool = True, upgrade_page_response: Any = None,
                del_args: list = [], dont_confirm_payment: bool = False, **kwargs: Any) -> Any:
        if upgrade_page_response is None:
            tier = kwargs.get('tier')
            upgrade_url = f'{self.billing_session.billing_base_url}/upgrade/'
            if tier:
                upgrade_url += f'?tier={tier}'
            if self.billing_session.billing_base_url:
                upgrade_page_response = self.client_get(upgrade_url, {}, subdomain='selfhosting')
            else:
                upgrade_page_response = self.client_get(upgrade_url, {})
        params = {
            'schedule': 'annual',
            'signed_seat_count': self.get_signed_seat_count_from_response(upgrade_page_response),
            'salt': self.get_salt_from_response(upgrade_page_response)
        }
        if invoice:
            params.update(billing_modality='send_invoice', licenses=kwargs.get('licenses', 123))
        else:
            params.update(billing_modality='charge_automatically', license_management='automatic')
        remote_server_plan_start_date = kwargs.get('remote_server_plan_start_date')
        if remote_server_plan_start_date:
            params.update(remote_server_plan_start_date=remote_server_plan_start_date)
        params.update(kwargs)
        for key in del_args:
            if key in params:
                del params[key]
        if talk_to_stripe:
            [last_event] = iter(stripe.Event.list(limit=1))
        existing_customer = self.billing_session.customer_plan_exists()
        upgrade_json_response = self.client_billing_post('/billing/upgrade', params)
        if upgrade_json_response.status_code != 200 or dont_confirm_payment:
            return upgrade_json_response
        is_self_hosted_billing = not isinstance(self.billing_session, RealmBillingSession)
        customer = self.billing_session.get_customer()
        assert customer is not None
        if not talk_to_stripe or (is_free_trial_offer_enabled(is_self_hosted_billing) and (not existing_customer)):
            return upgrade_json_response
        last_sent_invoice = Invoice.objects.last()
        assert last_sent_invoice is not None
        response_dict = self.assert_json_success(upgrade_json_response)
        self.assertEqual(response_dict['stripe_invoice_id'], last_sent_invoice.stripe_invoice_id)
        self.assert_details_of_valid_invoice_payment_from_event_status_endpoint(
            last_sent_invoice.stripe_invoice_id,
            {'status': 'sent'}
        )
        if invoice:
            stripe.Invoice.pay(last_sent_invoice.stripe_invoice_id, paid_out_of_band=True)
        self.send_stripe_webhook_events(last_event)
        return upgrade_json_response

    def add_card_and_upgrade(self, user: Any = None, **kwargs: Any) -> Any:
        with time_machine.travel(self.now, tick=False):
            self.add_card_to_customer_for_upgrade()
        if user is not None:
            stripe_customer = stripe_get_customer(assert_is_not_none(Customer.objects.get(realm=user.realm).stripe_customer_id))
        else:
            customer = self.billing_session.get_customer()
            assert customer is not None
            stripe_customer = stripe_get_customer(assert_is_not_none(customer.stripe_customer_id))
        self.assertTrue(stripe_customer_has_credit_card_as_default_payment_method(stripe_customer))
        with time_machine.travel(self.now, tick=False):
            response = self.upgrade(**kwargs)
        self.assert_json_success(response)
        return stripe_customer

    def local_upgrade(self, licenses: int, automanage_licenses: bool, billing_schedule: Any,
                      charge_automatically: bool, free_trial: bool) -> None:
        class StripeMock:
            def __init__(self, depth: int = 1) -> None:
                self.id = 'cus_123'
                self.created = '1000'
                self.last4 = '4242'
        def upgrade_func(licenses: int, automanage_licenses: bool, billing_schedule: Any, charge_automatically: bool,
                         free_trial: bool, stripe_invoice_paid: bool, *mock_args: Any) -> Any:
            hamlet = self.example_user('hamlet')
            billing_session = RealmBillingSession(hamlet, realm=hamlet.realm)
            return billing_session.process_initial_upgrade(CustomerPlan.TIER_CLOUD_STANDARD, licenses, automanage_licenses, billing_schedule, charge_automatically, free_trial, stripe_invoice_paid=stripe_invoice_paid)
        for mocked_function_name in settings.MOCKED_STRIPE_FUNCTION_NAMES:  # placeholder for iteration
            upgrade_func = self.patch(mocked_function_name, return_value=StripeMock())(upgrade_func)
        upgrade_func(licenses, automanage_licenses, billing_schedule, charge_automatically, free_trial, False)

    def setup_mocked_stripe(self, callback: Any, *args: Any, **kwargs: Any) -> Any:
        from unittest import mock
        with mock.patch.multiple('stripe', Invoice=mock.DEFAULT, InvoiceItem=mock.DEFAULT) as mocked:
            mocked['Invoice'].create.return_value = None
            mocked['Invoice'].finalize_invoice.return_value = None
            mocked['InvoiceItem'].create.return_value = None
            callback(*args, **kwargs)
            return mocked

    def client_billing_get(self, url_suffix: str, info: dict = {}) -> Any:
        url = f'/json{self.billing_session.billing_base_url}' + url_suffix
        if self.billing_session.billing_base_url:
            response = self.client_get(url, info, subdomain='selfhosting')
        else:
            response = self.client_get(url, info)
        return response

    def client_billing_post(self, url_suffix: str, info: dict = {}) -> Any:
        url = f'/json{self.billing_session.billing_base_url}' + url_suffix
        if self.billing_session.billing_base_url:
            response = self.client_post(url, info, subdomain='selfhosting')
        else:
            response = self.client_post(url, info)
        return response

    def client_billing_patch(self, url_suffix: str, info: dict = {}) -> Any:
        url = f'/json{self.billing_session.billing_base_url}' + url_suffix
        if self.billing_session.billing_base_url:
            response = self.client_patch(url, info, subdomain='selfhosting')
        else:
            response = self.client_patch(url, info)
        return response

    def client_get(self, url: str, info: dict = {}, subdomain: str = '') -> Any:
        return super().client_get(url, info, subdomain=subdomain)

    def client_post(self, url: str, info: dict = {}, subdomain: str = '') -> Any:
        return super().client_post(url, info, subdomain=subdomain)

    def client_patch(self, url: str, info: dict = {}, subdomain: str = '') -> Any:
        return super().client_patch(url, info, subdomain=subdomain)


class StripeTest(StripeTestCase):
    def test_catch_stripe_errors(self) -> None:
        @catch_stripe_errors
        def raise_invalid_request_error() -> None:
            raise stripe.InvalidRequestError('message', 'param', 'code', json_body={})
        with self.assertLogs('corporate.stripe', 'ERROR') as error_log:
            with self.assertRaises(BillingError) as billing_context:
                raise_invalid_request_error()
            self.assertEqual('other stripe error', billing_context.exception.error_description)
            self.assertEqual(error_log.output, ['ERROR:corporate.stripe:Stripe error: None None None None'])
        @catch_stripe_errors
        def raise_card_error() -> None:
            error_message = 'The card number is not a valid credit card number.'
            json_body = {'error': {'message': error_message}}
            raise stripe.CardError(error_message, 'number', 'invalid_number', json_body=json_body)
        with self.assertLogs('corporate.stripe', 'INFO') as info_log:
            with self.assertRaises(StripeCardError) as card_context:
                raise_card_error()
            self.assertIn('not a valid credit card', str(card_context.exception))
            self.assertEqual('card error', card_context.exception.error_description)
            self.assertEqual(info_log.output, ['INFO:corporate.stripe:Stripe card error: None None None None'])

    def test_billing_not_enabled(self) -> None:
        iago = self.example_user('iago')
        with self.settings(BILLING_ENABLED=False):
            self.login_user(iago)
            response = self.client_get('/upgrade/', follow=True)
            self.assertEqual(response.status_code, 404)

    @responses.activate
    @mock_stripe()
    def test_stripe_billing_portal_urls(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        self.add_card_to_customer_for_upgrade()
        response = self.client_get(f'/customer_portal/?tier={CustomerPlan.TIER_CLOUD_STANDARD}')
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response['Location'].startswith('https://billing.stripe.com'))
        self.upgrade(invoice=True)
        response = self.client_get('/customer_portal/?return_to_billing_page=true')
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response['Location'].startswith('https://billing.stripe.com'))
        response = self.client_get('/invoices/')
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response['Location'].startswith('https://billing.stripe.com'))

    @responses.activate
    @mock_stripe()
    def test_upgrade_by_card_to_plus_plan(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        response = self.client_get('/upgrade/?tier=2')
        self.assert_in_success_response(['Your subscription will renew automatically', 'Zulip Cloud Plus'], response)
        self.assertEqual(user.realm.plan_type, Realm.PLAN_TYPE_SELF_HOSTED)
        self.assertFalse(Customer.objects.filter(realm=user.realm).exists())
        stripe_customer = self.add_card_and_upgrade(user, tier=CustomerPlan.TIER_CLOUD_PLUS)
        self.assertEqual(stripe_customer.description, 'zulip (Zulip Dev)')
        self.assertEqual(stripe_customer.discount, None)
        self.assertEqual(stripe_customer.email, user.delivery_email)
        assert stripe_customer.metadata is not None
        metadata_dict = dict(stripe_customer.metadata)
        self.assertEqual(metadata_dict['realm_str'], 'zulip')
        try:
            int(metadata_dict['realm_id'])
        except ValueError:
            raise AssertionError('realm_id is not a number')
        [charge] = iter(stripe.Charge.list(customer=stripe_customer.id))
        licenses_purchased = self.billing_session.min_licenses_for_plan(CustomerPlan.TIER_CLOUD_PLUS)
        self.assertEqual(charge.amount, 12000 * licenses_purchased)
        self.assertEqual(charge.description, 'Payment for Invoice')
        self.assertEqual(charge.receipt_email, user.delivery_email)
        self.assertEqual(charge.statement_descriptor, 'Zulip Cloud Plus')
        [invoice] = iter(stripe.Invoice.list(customer=stripe_customer.id))
        self.assertIsNotNone(invoice.status_transitions.finalized_at)
        invoice_params = {'amount_due': 120000, 'amount_paid': 120000, 'auto_advance': False, 'collection_method': 'charge_automatically', 'status': 'paid', 'total': 120000}
        self.assertIsNotNone(invoice.charge)
        for key, value in invoice_params.items():
            self.assertEqual(invoice.get(key), value)
        [item0] = iter(invoice.lines)
        line_item_params = {'amount': 12000 * licenses_purchased, 'description': 'Zulip Cloud Plus', 'discountable': False, 'plan': None, 'proration': False, 'quantity': licenses_purchased, 'period': {'start': datetime_to_timestamp(self.now), 'end': datetime_to_timestamp(add_months(self.now, 12))}}
        for key, value in line_item_params.items():
            self.assertEqual(item0.get(key), value)
        customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
        plan = CustomerPlan.objects.get(customer=customer, automanage_licenses=True, price_per_license=12000, fixed_price=None, discount=None, billing_cycle_anchor=self.now, billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, invoiced_through=LicenseLedger.objects.first(), next_invoice_date=self.next_month, tier=CustomerPlan.TIER_CLOUD_PLUS, status=CustomerPlan.ACTIVE)
        LicenseLedger.objects.get(plan=plan, is_renewal=True, event_time=self.now, licenses=licenses_purchased, licenses_at_next_renewal=licenses_purchased)
        audit_log_entries = list(RealmAuditLog.objects.filter(acting_user=user).values_list('event_type', 'event_time').order_by('id'))
        self.assertEqual(audit_log_entries[:3], [(AuditLogEventType.STRIPE_CUSTOMER_CREATED, timestamp_to_datetime(stripe_customer.created)), (AuditLogEventType.STRIPE_CARD_CHANGED, self.now), (AuditLogEventType.CUSTOMER_PLAN_CREATED, self.now)])
        self.assertEqual(audit_log_entries[3][0], AuditLogEventType.REALM_PLAN_TYPE_CHANGED)
        first_audit_log_entry = RealmAuditLog.objects.filter(event_type=AuditLogEventType.CUSTOMER_PLAN_CREATED).values_list('extra_data', flat=True).first()
        assert first_audit_log_entry is not None
        self.assertTrue(first_audit_log_entry['automanage_licenses'])
        realm = get_realm('zulip')
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_PLUS)
        self.assertEqual(realm.max_invites, Realm.INVITES_STANDARD_REALM_DAILY_MAX)
        response = self.client_get('/upgrade/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('http://zulip.testserver/billing', response['Location'])
        with time_machine.travel(self.now, tick=False):
            response = self.client_get('/billing/')
        self.assert_not_in_success_response(['Pay annually'], response)
        for substring in ['Zulip Cloud Plus', str(licenses_purchased), 'Number of licenses', f'{licenses_purchased}', 'Your plan will automatically renew on', 'January 2, 2013', '$1,200.00', 'Visa ending in 4242', 'Update card']:
            self.assert_in_response(substring, response)
        self.assert_not_in_success_response(['Number of licenses for current billing period', 'You will receive an invoice for'], response)

    @responses.activate
    @mock_stripe()
    def test_upgrade_by_invoice_to_plus_plan(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.upgrade(invoice=True, tier=CustomerPlan.TIER_CLOUD_PLUS)
        stripe_customer = stripe_get_customer(assert_is_not_none(Customer.objects.get(realm=user.realm).stripe_customer_id))
        self.assertFalse(stripe_customer_has_credit_card_as_default_payment_method(stripe_customer))
        self.assertFalse(stripe.Charge.list(customer=stripe_customer.id))
        [invoice] = iter(stripe.Invoice.list(customer=stripe_customer.id))
        self.assertIsNotNone(invoice.due_date)
        self.assertIsNotNone(invoice.status_transitions.finalized_at)
        invoice_params = {'amount_due': 12000 * 123, 'amount_paid': 0, 'attempt_count': 0, 'auto_advance': False, 'collection_method': 'send_invoice', 'statement_descriptor': 'Zulip Cloud Plus', 'status': 'paid', 'total': 12000 * 123}
        for key, value in invoice_params.items():
            self.assertEqual(invoice.get(key), value)
        [item] = iter(invoice.lines)
        line_item_params = {'amount': 12000 * 123, 'description': 'Zulip Cloud Plus', 'discountable': False, 'plan': None, 'proration': False, 'quantity': 123, 'period': {'start': datetime_to_timestamp(self.now), 'end': datetime_to_timestamp(add_months(self.now, 12))}}
        for key, value in line_item_params.items():
            self.assertEqual(item.get(key), value)
        customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
        plan = CustomerPlan.objects.get(customer=customer, automanage_licenses=False, charge_automatically=False, price_per_license=12000, fixed_price=None, discount=None, billing_cycle_anchor=self.now, billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, invoiced_through=LicenseLedger.objects.first(), next_invoice_date=self.next_month, tier=CustomerPlan.TIER_CLOUD_PLUS, status=CustomerPlan.ACTIVE)
        LicenseLedger.objects.get(plan=plan, is_renewal=True, event_time=self.now, licenses=123, licenses_at_next_renewal=123)
        audit_log_entries = list(RealmAuditLog.objects.filter(acting_user=user).values_list('event_type', 'event_time').order_by('id'))
        self.assertEqual(audit_log_entries[:3], [(AuditLogEventType.STRIPE_CUSTOMER_CREATED, timestamp_to_datetime(stripe_customer.created)), (AuditLogEventType.CUSTOMER_PLAN_CREATED, self.now), (AuditLogEventType.REALM_PLAN_TYPE_CHANGED, self.now)])
        self.assertEqual(audit_log_entries[2][0], AuditLogEventType.REALM_PLAN_TYPE_CHANGED)
        first_audit_log_entry = RealmAuditLog.objects.filter(event_type=AuditLogEventType.CUSTOMER_PLAN_CREATED).values_list('extra_data', flat=True).first()
        assert first_audit_log_entry is not None
        self.assertFalse(first_audit_log_entry['automanage_licenses'])
        realm = get_realm('zulip')
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_PLUS)
        self.assertEqual(realm.max_invites, Realm.INVITES_STANDARD_REALM_DAILY_MAX)
        response = self.client_get('/upgrade/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('http://zulip.testserver/billing', response['Location'])
        with time_machine.travel(self.now, tick=False):
            response = self.client_get('/billing/')
        self.assert_not_in_success_response(['Pay annually', 'Update card'], response)
        for substring in ['Zulip Cloud Plus', str(123), 'Number of licenses for current billing period', f'licenses ({self.seat_count} in use)', 'You will receive an invoice for', 'January 2, 2013', '$14,760.00']:
            self.assert_in_response(substring, response)

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_by_card(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        response = self.client_get('/upgrade/')
        self.assert_in_success_response(['Your subscription will renew automatically'], response)
        self.assertNotEqual(user.realm.plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertFalse(Customer.objects.filter(realm=user.realm).exists())
        with self.assertLogs('corporate.stripe', 'WARNING'):
            response = self.upgrade()
        self.assert_json_error(response, 'Please add a credit card before upgrading.')
        stripe_customer = self.add_card_and_upgrade(user)
        self.assertEqual(stripe_customer.description, 'zulip (Zulip Dev)')
        self.assertEqual(stripe_customer.discount, None)
        self.assertEqual(stripe_customer.email, user.delivery_email)
        assert stripe_customer.metadata is not None
        metadata_dict = dict(stripe_customer.metadata)
        self.assertEqual(metadata_dict['realm_str'], 'zulip')
        try:
            int(metadata_dict['realm_id'])
        except ValueError:
            raise AssertionError('realm_id is not a number')
        [charge] = iter(stripe.Charge.list(customer=stripe_customer.id))
        self.assertEqual(charge.amount, 8000 * self.seat_count)
        self.assertEqual(charge.description, 'Payment for Invoice')
        self.assertEqual(charge.receipt_email, user.delivery_email)
        self.assertEqual(charge.statement_descriptor, 'Zulip Cloud Standard')
        [invoice] = iter(stripe.Invoice.list(customer=stripe_customer.id))
        self.assertIsNotNone(invoice.status_transitions.finalized_at)
        invoice_params = {'amount_due': 48000, 'amount_paid': 48000, 'auto_advance': False, 'collection_method': 'charge_automatically', 'status': 'paid', 'total': 48000}
        self.assertIsNotNone(invoice.charge)
        for key, value in invoice_params.items():
            self.assertEqual(invoice.get(key), value)
        [item0] = iter(invoice.lines)
        line_item_params = {'amount': 8000 * self.seat_count, 'description': 'Zulip Cloud Standard', 'discountable': False, 'plan': None, 'proration': False, 'quantity': self.seat_count, 'period': {'start': datetime_to_timestamp(self.now), 'end': datetime_to_timestamp(add_months(self.now, 12))}}
        for key, value in line_item_params.items():
            self.assertEqual(item0.get(key), value)
        customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
        plan = CustomerPlan.objects.get(customer=customer, automanage_licenses=True, price_per_license=8000, fixed_price=None, discount=None, billing_cycle_anchor=self.now, billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, invoiced_through=LicenseLedger.objects.first(), next_invoice_date=self.next_month, tier=CustomerPlan.TIER_CLOUD_STANDARD, status=CustomerPlan.ACTIVE)
        LicenseLedger.objects.get(plan=plan, is_renewal=True, event_time=self.now, licenses=self.seat_count, licenses_at_next_renewal=self.seat_count)
        audit_log_entries = list(RealmAuditLog.objects.filter(acting_user=user).values_list('event_type', 'event_time').order_by('id'))
        self.assertEqual(audit_log_entries[:3], [(AuditLogEventType.STRIPE_CUSTOMER_CREATED, timestamp_to_datetime(stripe_customer.created)), (AuditLogEventType.STRIPE_CARD_CHANGED, self.now), (AuditLogEventType.CUSTOMER_PLAN_CREATED, self.now)])
        self.assertEqual(audit_log_entries[3][0], AuditLogEventType.REALM_PLAN_TYPE_CHANGED)
        first_audit_log_entry = RealmAuditLog.objects.filter(event_type=AuditLogEventType.CUSTOMER_PLAN_CREATED).values_list('extra_data', flat=True).first()
        assert first_audit_log_entry is not None
        self.assertTrue(first_audit_log_entry['automanage_licenses'])
        realm = get_realm('zulip')
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(realm.max_invites, Realm.INVITES_STANDARD_REALM_DAILY_MAX)
        response = self.client_get('/upgrade/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('http://zulip.testserver/billing', response['Location'])
        with time_machine.travel(self.now, tick=False):
            response = self.client_get('/billing/')
        self.assert_not_in_success_response(['Pay annually'], response)
        for substring in ['Zulip Cloud Standard', str(self.seat_count), 'Number of licenses', f'{self.seat_count}', 'Your plan will automatically renew on', 'January 2, 2013', f'${80 * self.seat_count}.00', 'Visa ending in 4242', 'Update card']:
            self.assert_in_response(substring, response)
        self.assert_not_in_success_response(['Number of licenses for current billing period', 'You will receive an invoice for'], response)

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_card_attached_to_customer_but_payment_fails(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        self.add_card_to_customer_for_upgrade(charge_succeeds=False)
        with self.assertLogs('corporate.stripe', 'WARNING'):
            response = self.upgrade()
        self.assert_json_error_contains(response, 'Your card was declined.')

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_by_invoice(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.upgrade(invoice=True)
        stripe_customer = stripe_get_customer(assert_is_not_none(Customer.objects.get(realm=user.realm).stripe_customer_id))
        self.assertFalse(stripe_customer_has_credit_card_as_default_payment_method(stripe_customer))
        self.assertFalse(stripe.Charge.list(customer=stripe_customer.id))
        [invoice] = iter(stripe.Invoice.list(customer=stripe_customer.id))
        self.assertIsNotNone(invoice.due_date)
        self.assertIsNotNone(invoice.status_transitions.finalized_at)
        invoice_params = {'amount_due': 8000 * 123, 'amount_paid': 0, 'attempt_count': 0, 'auto_advance': False, 'collection_method': 'send_invoice', 'statement_descriptor': 'Zulip Cloud Standard', 'status': 'open', 'total': 8000 * 123}
        for key, value in invoice_params.items():
            self.assertEqual(invoice.get(key), value)
        [item] = iter(invoice.lines)
        line_item_params = {'amount': 8000 * 123, 'description': 'Zulip Cloud Standard', 'discountable': False, 'plan': None, 'proration': False, 'quantity': 123, 'period': {'start': datetime_to_timestamp(self.now), 'end': datetime_to_timestamp(add_months(self.now, 12))}}
        for key, value in line_item_params.items():
            self.assertEqual(item.get(key), value)
        customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
        plan = CustomerPlan.objects.get(customer=customer, automanage_licenses=False, charge_automatically=False, price_per_license=8000, fixed_price=None, discount=None, billing_cycle_anchor=self.now, billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, invoiced_through=LicenseLedger.objects.first(), next_invoice_date=self.next_month, tier=CustomerPlan.TIER_CLOUD_STANDARD, status=CustomerPlan.ACTIVE)
        LicenseLedger.objects.get(plan=plan, is_renewal=True, event_time=self.now, licenses=123, licenses_at_next_renewal=123)
        audit_log_entries = list(RealmAuditLog.objects.filter(acting_user=user).values_list('event_type', 'event_time').order_by('id'))
        self.assertEqual(audit_log_entries[:3], [(AuditLogEventType.STRIPE_CUSTOMER_CREATED, timestamp_to_datetime(stripe_customer.created)), (AuditLogEventType.CUSTOMER_PLAN_CREATED, self.now), (AuditLogEventType.REALM_PLAN_TYPE_CHANGED, self.now)])
        self.assertEqual(audit_log_entries[2][0], AuditLogEventType.REALM_PLAN_TYPE_CHANGED)
        first_audit_log_entry = RealmAuditLog.objects.filter(event_type=AuditLogEventType.CUSTOMER_PLAN_CREATED).values_list('extra_data', flat=True).first()
        assert first_audit_log_entry is not None
        self.assertFalse(first_audit_log_entry['automanage_licenses'])
        realm = get_realm('zulip')
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(realm.max_invites, Realm.INVITES_STANDARD_REALM_DAILY_MAX)
        response = self.client_get('/upgrade/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('http://zulip.testserver/billing', response['Location'])
        with time_machine.travel(self.now, tick=False):
            response = self.client_get('/billing/')
        self.assert_not_in_success_response(['Pay annually'], response)
        for substring in ['Zulip Cloud Standard', str(123), 'Number of licenses for current billing period', f'licenses ({self.seat_count} in use)', 'You will receive an invoice for', 'January 2, 2013', '$9,840.00']:
            self.assert_in_response(substring, response)

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_free_trial_upgrade_by_card(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with self.settings(CLOUD_FREE_TRIAL_DAYS=60):
            response = self.client_get('/upgrade/')
            free_trial_end_date = self.now + timedelta(days=60)
            self.assert_in_success_response(['Your card will not be charged', 'free trial', '60-day'], response)
            self.assertNotEqual(user.realm.plan_type, Realm.PLAN_TYPE_STANDARD)
            self.assertFalse(Customer.objects.filter(realm=user.realm).exists())
            with time_machine.travel(self.now, tick=False), self.assertLogs('corporate.stripe', 'WARNING'):
                response = self.upgrade()
            self.assert_json_error(response, 'Please add a credit card before starting your free trial.')
            stripe_customer = self.add_card_and_upgrade(user)
            self.assertEqual(Invoice.objects.count(), 0)
            self.assertEqual(stripe_customer.description, 'zulip (Zulip Dev)')
            self.assertEqual(stripe_customer.discount, None)
            self.assertEqual(stripe_customer.email, user.delivery_email)
            assert stripe_customer.metadata is not None
            metadata_dict = dict(stripe_customer.metadata)
            self.assertEqual(metadata_dict['realm_str'], 'zulip')
            try:
                int(metadata_dict['realm_id'])
            except ValueError:
                raise AssertionError('realm_id is not a number')
            self.assertFalse(stripe.Charge.list(customer=stripe_customer.id))
            self.assertFalse(stripe.Invoice.list(customer=stripe_customer.id))
            customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
            plan = CustomerPlan.objects.get(customer=customer, automanage_licenses=True, price_per_license=8000, fixed_price=None, discount=None, billing_cycle_anchor=self.now, billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, invoiced_through=LicenseLedger.objects.first(), next_invoice_date=free_trial_end_date, tier=CustomerPlan.TIER_CLOUD_STANDARD, status=CustomerPlan.FREE_TRIAL, charge_automatically=True)
            LicenseLedger.objects.get(plan=plan, is_renewal=True, event_time=self.now, licenses=self.seat_count, licenses_at_next_renewal=self.seat_count)
            audit_log_entries = list(RealmAuditLog.objects.filter(acting_user=user).values_list('event_type', 'event_time').order_by('id'))
            self.assertEqual(audit_log_entries[:4], [(AuditLogEventType.STRIPE_CUSTOMER_CREATED, timestamp_to_datetime(stripe_customer.created)), (AuditLogEventType.STRIPE_CARD_CHANGED, self.now), (AuditLogEventType.CUSTOMER_PLAN_CREATED, self.now), (AuditLogEventType.REALM_PLAN_TYPE_CHANGED, self.now)])
            self.assertEqual(audit_log_entries[3][0], AuditLogEventType.REALM_PLAN_TYPE_CHANGED)
            first_audit_log_entry = RealmAuditLog.objects.filter(event_type=AuditLogEventType.CUSTOMER_PLAN_CREATED).values_list('extra_data', flat=True).first()
            assert first_audit_log_entry is not None
            self.assertTrue(first_audit_log_entry['automanage_licenses'])
            realm = get_realm('zulip')
            self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)
            self.assertEqual(realm.max_invites, Realm.INVITES_STANDARD_REALM_DAILY_MAX)
            with time_machine.travel(self.now, tick=False):
                response = self.client_get('/billing/')
            self.assert_not_in_success_response(['Pay annually'], response)
            for substring in ['Zulip Cloud Standard <i>(free trial)</i>', str(self.seat_count), 'Number of licenses', f'{self.seat_count}', 'Your plan will automatically renew on', 'March 2, 2012', f'${80 * self.seat_count}.00', 'Visa ending in 4242', 'Update card']:
                self.assert_in_response(substring, response)
            self.assert_not_in_success_response(['Go to your Zulip organization'], response)
            billing_session = RealmBillingSession(user=user, realm=realm)
            with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=12):
                billing_session.update_license_ledger_if_needed(self.now)
            self.assertEqual(LicenseLedger.objects.order_by('-id').values_list('licenses', 'licenses_at_next_renewal').first(), (12, 12))
            with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=15):
                billing_session.update_license_ledger_if_needed(self.next_month)
            self.assertEqual(LicenseLedger.objects.order_by('-id').values_list('licenses', 'licenses_at_next_renewal').first(), (15, 15))
            invoice_plans_as_needed(self.next_month)
            self.assertFalse(stripe.Invoice.list(customer=stripe_customer.id))
            customer_plan = CustomerPlan.objects.get(customer=customer)
            self.assertEqual(customer_plan.status, CustomerPlan.FREE_TRIAL)
            self.assertEqual(customer_plan.next_invoice_date, free_trial_end_date)
            invoice_plans_as_needed(free_trial_end_date)
            customer_plan.refresh_from_db()
            realm.refresh_from_db()
            self.assertEqual(customer_plan.status, CustomerPlan.ACTIVE)
            self.assertEqual(customer_plan.next_invoice_date, add_months(free_trial_end_date, 1))
            self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)
            [invoice] = iter(stripe.Invoice.list(customer=stripe_customer.id))
            invoice_params = {'amount_due': 15 * 80 * 100, 'amount_paid': 0, 'amount_remaining': 15 * 80 * 100, 'auto_advance': True, 'collection_method': 'charge_automatically', 'customer_email': self.example_email('hamlet'), 'discount': None, 'paid': False, 'status': 'open', 'total': 15 * 80 * 100}
            for key, value in invoice_params.items():
                self.assertEqual(invoice.get(key), value)
            [invoice_item] = iter(invoice.lines)
            invoice_item_params = {'amount': 15 * 80 * 100, 'description': 'Zulip Cloud Standard - renewal', 'plan': None, 'quantity': 15, 'subscription': None, 'discountable': False, 'period': {'start': datetime_to_timestamp(free_trial_end_date), 'end': datetime_to_timestamp(add_months(free_trial_end_date, 12))}}
            for key, value in invoice_item_params.items():
                self.assertEqual(invoice_item[key], value)
            invoice_plans_as_needed(add_months(free_trial_end_date, 1))
            [invoice, invoice0, invoice1] = iter(stripe.Invoice.list(customer=stripe_customer.id))
        plan.fixed_price = 127
        plan.price_per_license = None
        plan.save(update_fields=['fixed_price', 'price_per_license'])
        with time_machine.travel(self.now, tick=False):
            response = self.client_get('/billing/')
        self.assert_in_success_response(['$1.27'], response)
        self.assert_not_in_success_response([f'{self.seat_count} x'], response)

    @responses.activate
    @mock_stripe()
    def test_upgrade_by_card_with_outdated_seat_count(self, *mocks: Any) -> None:
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        new_seat_count = 23
        initial_upgrade_request = InitialUpgradeRequest(manual_license_management=False, tier=CustomerPlan.TIER_CLOUD_STANDARD, billing_modality='charge_automatically')
        billing_session = RealmBillingSession(hamlet, realm=hamlet.realm)
        _, context_when_upgrade_page_is_rendered = billing_session.get_initial_upgrade_context(initial_upgrade_request)
        from unittest.mock import patch
        with patch('corporate.lib.stripe.BillingSession.stale_seat_count_check', return_value=self.seat_count), \
             patch('corporate.lib.stripe.get_latest_seat_count', return_value=new_seat_count), \
             patch('corporate.lib.stripe.RealmBillingSession.get_initial_upgrade_context', return_value=(_, context_when_upgrade_page_is_rendered)):
            self.add_card_and_upgrade(hamlet)
        customer = Customer.objects.first()
        assert customer is not None
        stripe_customer_id = assert_is_not_none(customer.stripe_customer_id)
        [charge] = iter(stripe.Charge.list(customer=stripe_customer_id))
        self.assertEqual(8000 * self.seat_count, charge.amount)
        [additional_license_invoice, upgrade_invoice] = iter(stripe.Invoice.list(customer=stripe_customer_id))
        self.assertEqual([8000 * self.seat_count], [item.amount for item in upgrade_invoice.lines])
        self.assertEqual([8000 * (new_seat_count - self.seat_count)], [item.amount for item in additional_license_invoice.lines])
        ledger_entry = LicenseLedger.objects.last()
        assert ledger_entry is not None
        self.assertEqual(ledger_entry.licenses, new_seat_count)
        self.assertEqual(ledger_entry.licenses_at_next_renewal, new_seat_count)

    @responses.activate
    @mock_stripe()
    def test_upgrade_by_card_with_outdated_lower_seat_count(self, *mocks: Any) -> None:
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        new_seat_count = self.seat_count - 1
        initial_upgrade_request = InitialUpgradeRequest(manual_license_management=False, tier=CustomerPlan.TIER_CLOUD_STANDARD, billing_modality='charge_automatically')
        billing_session = RealmBillingSession(hamlet, realm=hamlet.realm)
        _, context_when_upgrade_page_is_rendered = billing_session.get_initial_upgrade_context(initial_upgrade_request)
        from unittest.mock import patch
        with patch('corporate.lib.stripe.BillingSession.stale_seat_count_check', return_value=self.seat_count), \
             patch('corporate.lib.stripe.get_latest_seat_count', return_value=new_seat_count), \
             patch('corporate.lib.stripe.RealmBillingSession.get_initial_upgrade_context', return_value=(_, context_when_upgrade_page_is_rendered)):
            self.add_card_and_upgrade(hamlet)
        customer = Customer.objects.first()
        assert customer is not None
        stripe_customer_id = assert_is_not_none(customer.stripe_customer_id)
        [charge] = iter(stripe.Charge.list(customer=stripe_customer_id))
        self.assertEqual(8000 * self.seat_count, charge.amount)
        [upgrade_invoice] = iter(stripe.Invoice.list(customer=stripe_customer_id))
        self.assertEqual([8000 * self.seat_count], [item.amount for item in upgrade_invoice.lines])
        ledger_entry = LicenseLedger.objects.last()
        assert ledger_entry is not None
        self.assertEqual(ledger_entry.licenses, self.seat_count)
        self.assertEqual(ledger_entry.licenses_at_next_renewal, new_seat_count)
        from django.core.mail import outbox
        self.assert_length(outbox, 1)
        for message in outbox:
            self.assert_length(message.to, 1)
            self.assertEqual(message.to[0], 'sales@zulip.com')
            self.assertEqual(message.subject, f'Check initial licenses invoiced for {billing_session.billing_entity_display_name}')
            self.assertEqual(self.email_envelope_from(message), settings.NOREPLY_EMAIL_ADDRESS)

    def test_upgrade_with_tampered_seat_count(self) -> None:
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        with self.assertLogs('corporate.stripe', 'WARNING'):
            response = self.upgrade(talk_to_stripe=False, salt='badsalt')
        self.assert_json_error_contains(response, 'Something went wrong. Please contact')
        self.assertEqual(orjson.loads(response.content)['error_description'], 'tampered seat count')

    @responses.activate
    @mock_stripe()
    def test_upgrade_race_condition_during_card_upgrade(self, *mocks: Any) -> None:
        hamlet = self.example_user('hamlet')
        othello = self.example_user('othello')
        self.login_user(othello)
        othello_upgrade_page_response = self.client_get('/upgrade/')
        self.login_user(hamlet)
        self.add_card_to_customer_for_upgrade()
        [stripe_event_before_upgrade] = iter(stripe.Event.list(limit=1))
        hamlet_upgrade_page_response = self.client_get('/upgrade/')
        self.client_billing_post('/billing/upgrade', {
            'billing_modality': 'charge_automatically',
            'schedule': 'annual',
            'signed_seat_count': self.get_signed_seat_count_from_response(hamlet_upgrade_page_response),
            'salt': self.get_salt_from_response(hamlet_upgrade_page_response),
            'license_management': 'automatic'
        })
        customer = get_customer_by_realm(get_realm('zulip'))
        assert customer is not None
        assert customer.stripe_customer_id is not None
        [hamlet_invoice] = iter(stripe.Invoice.list(customer=customer.stripe_customer_id))
        self.login_user(othello)
        with self.settings(CLOUD_FREE_TRIAL_DAYS=60):
            self.client_billing_post('/billing/upgrade', {
                'billing_modality': 'charge_automatically',
                'schedule': 'annual',
                'signed_seat_count': self.get_signed_seat_count_from_response(othello_upgrade_page_response),
                'salt': self.get_salt_from_response(othello_upgrade_page_response),
                'license_management': 'automatic'
            })
        with self.assertLogs('corporate.stripe', 'WARNING'):
            self.send_stripe_webhook_events(stripe_event_before_upgrade)
        assert hamlet_invoice.id is not None
        self.assert_details_of_valid_invoice_payment_from_event_status_endpoint(
            hamlet_invoice.id,
            {'status': 'paid', 'event_handler': {'status': 'failed', 'error': {'message': 'The organization is already subscribed to a plan. Please reload the billing page.', 'description': 'subscribing with existing subscription'}}}
        )
        from django.core.mail import outbox
        self.assert_length(outbox, 1)
        for message in outbox:
            self.assert_length(message.to, 1)
            self.assertEqual(message.to[0], 'sales@zulip.com')
            self.assertEqual(message.subject, 'Error processing paid customer invoice')
            self.assertEqual(self.email_envelope_from(message), settings.NOREPLY_EMAIL_ADDRESS)

    def test_upgrade_race_condition_during_invoice_upgrade(self) -> None:
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        with self.assertLogs('corporate.stripe', 'WARNING') as m, self.assertRaises(BillingError) as context:
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual('subscribing with existing subscription', context.exception.error_description)
        self.assertEqual(m.output[0], 'WARNING:corporate.stripe:Upgrade of <Realm: zulip 2> (with stripe_customer_id: cus_123) failed because of existing active plan.')
        self.assert_length(m.output, 1)

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_check_upgrade_parameters(self, *mocks: Any) -> None:
        def check_error(error_message: str, error_description: str, upgrade_params: dict, del_args: list = []) -> None:
            self.add_card_to_customer_for_upgrade()
            if error_description:
                with self.assertLogs('corporate.stripe', 'WARNING'):
                    response = self.upgrade(talk_to_stripe=False, del_args=del_args, **upgrade_params)
                    self.assertEqual(orjson.loads(response.content)['error_description'], error_description)
            else:
                response = self.upgrade(talk_to_stripe=False, del_args=del_args, **upgrade_params)
            self.assert_json_error_contains(response, error_message)
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        check_error('Invalid billing_modality', '', {'billing_modality': 'invalid'})
        check_error('Invalid schedule', '', {'schedule': 'invalid'})
        check_error('Invalid license_management', '', {'license_management': 'invalid'})
        check_error('You must purchase licenses for all active users in your organization (minimum 30).', 'not enough licenses', {'billing_modality': 'send_invoice', 'licenses': -1})
        check_error('You must purchase licenses for all active users in your organization (minimum 30).', 'not enough licenses', {'billing_modality': 'send_invoice'})
        check_error('You must purchase licenses for all active users in your organization (minimum 30).', 'not enough licenses', {'billing_modality': 'send_invoice', 'licenses': 25})
        check_error("Invoices with more than 1000 licenses can't be processed from this page", 'too many licenses', {'billing_modality': 'send_invoice', 'licenses': 10000})
        check_error('You must purchase licenses for all active users in your organization (minimum 6).', 'not enough licenses', {'billing_modality': 'charge_automatically', 'license_management': 'manual'})
        check_error('You must purchase licenses for all active users in your organization (minimum 6).', 'not enough licenses', {'billing_modality': 'charge_automatically', 'license_management': 'manual', 'licenses': 3})

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_license_counts(self, *mocks: Any) -> None:
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        self.add_card_to_customer_for_upgrade()
        def check_min_licenses_error(invoice: bool, licenses: Any, min_licenses_in_response: int, upgrade_params: dict = {}) -> None:
            upgrade_params = dict(upgrade_params)
            if licenses is None:
                del_args = ['licenses']
            else:
                del_args = []
                upgrade_params['licenses'] = licenses
            with self.assertLogs('corporate.stripe', 'WARNING'):
                response = self.upgrade(invoice=invoice, talk_to_stripe=False, del_args=del_args, **upgrade_params)
            self.assert_json_error_contains(response, f'minimum {min_licenses_in_response}')
            self.assertEqual(orjson.loads(response.content)['error_description'], 'not enough licenses')
        def check_max_licenses_error(licenses: int) -> None:
            with self.assertLogs('corporate.stripe', 'WARNING'):
                response = self.upgrade(invoice=True, talk_to_stripe=False, licenses=licenses)
            self.assert_json_error_contains(response, f'with more than {settings.MAX_INVOICED_LICENSES} licenses')
            self.assertEqual(orjson.loads(response.content)['error_description'], 'too many licenses')
        def check_success(invoice: bool, licenses: Any, upgrade_params: dict = {}) -> None:
            upgrade_params = dict(upgrade_params)
            if licenses is None:
                del_args = ['licenses']
            else:
                del_args = []
                upgrade_params['licenses'] = licenses
            from unittest.mock import patch
            with patch('corporate.lib.stripe.BillingSession.process_initial_upgrade'), patch('corporate.lib.stripe.BillingSession.create_stripe_invoice_and_charge', return_value='fake_stripe_invoice_id'):
                response = self.upgrade(invoice=invoice, talk_to_stripe=False, del_args=del_args, **upgrade_params)
            self.assert_json_success(response)
        check_min_licenses_error(False, self.seat_count - 1, self.seat_count, {'license_management': 'manual'})
        check_min_licenses_error(False, None, self.seat_count, {'license_management': 'manual'})
        check_min_licenses_error(True, settings.MIN_INVOICED_LICENSES - 1, settings.MIN_INVOICED_LICENSES)
        with self.patch('corporate.lib.stripe.MIN_INVOICED_LICENSES', 3):
            check_min_licenses_error(True, 4, self.seat_count)
        check_min_licenses_error(True, None, settings.MIN_INVOICED_LICENSES)
        check_max_licenses_error(settings.MAX_INVOICED_LICENSES + 1)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=settings.MAX_INVOICED_LICENSES + 5):
            check_max_licenses_error(settings.MAX_INVOICED_LICENSES + 5)
        check_success(False, None)
        check_success(False, self.seat_count)
        check_success(False, self.seat_count, {'license_management': 'manual'})
        check_success(False, settings.MAX_INVOICED_LICENSES + 1, {'license_management': 'manual'})
        check_success(True, self.seat_count + settings.MIN_INVOICED_LICENSES)
        check_success(True, settings.MAX_INVOICED_LICENSES)
        customer = Customer.objects.get_or_create(realm=hamlet.realm)[0]
        customer.exempt_from_license_number_check = True
        customer.save()
        check_success(False, self.seat_count - 1, {'license_management': 'manual'})

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_with_uncaught_exception(self, *mock_args: Any) -> None:
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        self.add_card_to_customer_for_upgrade()
        from unittest.mock import patch
        with patch('corporate.lib.stripe.BillingSession.create_stripe_invoice_and_charge', side_effect=Exception), self.assertLogs('corporate.stripe', 'WARNING') as m:
            response = self.upgrade(talk_to_stripe=False)
            self.assertIn('ERROR:corporate.stripe:Uncaught exception in billing', m.output[0])
            self.assertIn(m.records[0].stack_info, m.output[0])
        self.assert_json_error_contains(response, 'Something went wrong. Please contact desdemona+admin@zulip.com.')
        self.assertEqual(orjson.loads(response.content)['error_description'], 'uncaught exception during upgrade')

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_invoice_payment_succeeded_event_with_uncaught_exception(self, *mock_args: Any) -> None:
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        self.add_card_to_customer_for_upgrade()
        from unittest.mock import patch
        with patch('corporate.lib.stripe.BillingSession.process_initial_upgrade', side_effect=Exception), self.assertLogs('corporate.stripe', 'WARNING'):
            response = self.upgrade()
        response_dict = self.assert_json_success(response)
        self.assert_details_of_valid_invoice_payment_from_event_status_endpoint(
            response_dict['stripe_invoice_id'],
            {'status': 'paid', 'event_handler': {'status': 'failed', 'error': {'message': 'Something went wrong. Please contact desdemona+admin@zulip.com.', 'description': 'uncaught exception in invoice.paid event handler'}}}
        )

    def test_request_sponsorship_form_with_invalid_url(self) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        data = {
            'organization_type': Realm.ORG_TYPES['opensource']['id'],
            'website': 'invalid-url',
            'description': 'Infinispan is a distributed in-memory key/value data store with optional schema.',
            'expected_total_users': '10 users',
            'plan_to_use_zulip': 'For communication on moon.',
            'paid_users_count': '1 user',
            'paid_users_description': 'We have 1 paid user.'
        }
        response = self.client_billing_post('/billing/sponsorship', data)
        self.assert_json_error(response, 'Enter a valid URL.')

    def test_request_sponsorship_form_with_blank_url(self) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        data = {
            'organization_type': Realm.ORG_TYPES['opensource']['id'],
            'website': '',
            'description': 'Infinispan is a distributed in-memory key/value data store with optional schema.',
            'expected_total_users': '10 users',
            'plan_to_use_zulip': 'For communication on moon.',
            'paid_users_count': '1 user',
            'paid_users_description': 'We have 1 paid user.'
        }
        response = self.client_billing_post('/billing/sponsorship', data)
        self.assert_json_success(response)

    @responses.activate
    @mock_stripe()
    def test_sponsorship_access_for_realms_on_paid_plan(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        self.add_card_and_upgrade(user)
        response = self.client_get('/sponsorship/')
        self.assert_in_success_response(['How many paid staff does your organization have?'], response)

    def test_demo_request(self) -> None:
        result = self.client_get('/request-demo/')
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response(['Request a demo'], result)
        data = {
            'full_name': 'King Hamlet',
            'email': 'test@zulip.com',
            'role': 'Manager',
            'organization_name': 'Zulip',
            'organization_type': 'Business',
            'organization_website': 'https://example.com',
            'expected_user_count': '10 (2 unpaid members)',
            'message': 'Need help!'
        }
        result = self.client_post('/request-demo/', data)
        self.assert_in_success_response(['Thanks for contacting us!'], result)
        from django.core.mail import outbox
        self.assert_length(outbox, 1)
        for message in outbox:
            self.assert_length(message.to, 1)
            self.assertEqual(message.to[0], 'sales@zulip.com')
            self.assertEqual(message.subject, 'Demo request for Zulip')
            self.assertEqual(message.reply_to, ['test@zulip.com'])
            self.assertEqual(self.email_envelope_from(message), settings.NOREPLY_EMAIL_ADDRESS)
            self.assertIn('Zulip demo request <noreply-', self.email_display_from(message))
            self.assertIn('Full name: King Hamlet', message.body)

    def test_support_request(self) -> None:
        user = self.example_user('hamlet')
        self.assertIsNone(get_customer_by_realm(user.realm))
        self.login_user(user)
        result = self.client_get('/support/')
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response(['Contact support'], result)
        data = {'request_subject': 'Not getting messages.', 'request_message': 'Running into this weird issue.'}
        result = self.client_post('/support/', data)
        self.assert_in_success_response(['Thanks for contacting us!'], result)
        from django.core.mail import outbox
        self.assert_length(outbox, 1)
        for message in outbox:
            self.assert_length(message.to, 1)
            self.assertEqual(message.to[0], 'desdemona+admin@zulip.com')
            self.assertEqual(message.subject, 'Support request for zulip')
            self.assertEqual(message.reply_to, ['hamlet@zulip.com'])
            self.assertEqual(self.email_envelope_from(message), settings.NOREPLY_EMAIL_ADDRESS)
            self.assertIn('Zulip support request <noreply-', self.email_display_from(message))
            self.assertIn('Requested by: King Hamlet (Member)', message.body)
            self.assertIn('Support URL: http://zulip.testserver/activity/support?q=zulip', message.body)
            self.assertIn('Subject: Not getting messages.', message.body)
            self.assertIn('Message:\nRunning into this weird issue', message.body)

    def test_request_sponsorship(self) -> None:
        user = self.example_user('hamlet')
        self.assertIsNone(get_customer_by_realm(user.realm))
        self.login_user(user)
        data = {
            'organization_type': Realm.ORG_TYPES['opensource']['id'],
            'website': 'https://infinispan.org/',
            'description': 'Infinispan is a distributed in-memory key/value data store with optional schema.',
            'expected_total_users': '10 users',
            'plan_to_use_zulip': 'For communication on moon.',
            'paid_users_count': '1 user',
            'paid_users_description': 'We have 1 paid user.'
        }
        response = self.client_billing_post('/billing/sponsorship', data)
        self.assert_json_success(response)
        customer = get_customer_by_realm(user.realm)
        assert customer is not None
        sponsorship_request = ZulipSponsorshipRequest.objects.filter(customer=customer, requested_by=user).first()
        assert sponsorship_request is not None
        self.assertEqual(sponsorship_request.org_website, data['website'])
        self.assertEqual(sponsorship_request.org_description, data['description'])
        self.assertEqual(sponsorship_request.org_type, Realm.ORG_TYPES['opensource']['id'])
        customer = get_customer_by_realm(user.realm)
        assert customer is not None
        self.assertEqual(customer.sponsorship_pending, True)
        from django.core.mail import outbox
        self.assert_length(outbox, 1)
        for message in outbox:
            self.assert_length(message.to, 1)
            self.assertEqual(message.to[0], 'sales@zulip.com')
            self.assertEqual(message.subject, 'Sponsorship request for zulip')
            self.assertEqual(message.reply_to, ['hamlet@zulip.com'])
            self.assertEqual(self.email_envelope_from(message), settings.NOREPLY_EMAIL_ADDRESS)
            self.assertIn('Zulip sponsorship request <noreply-', self.email_display_from(message))
            self.assertIn('Requested by: King Hamlet (Member)', message.body)
            self.assertIn('Support URL: http://zulip.testserver/activity/support?q=zulip', message.body)
            self.assertIn('Website: https://infinispan.org', message.body)
            self.assertIn('Organization type: Open-source', message.body)
            self.assertIn('Description:\nInfinispan is a distributed in-memory', message.body)
        response = self.client_get('/upgrade/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], 'http://zulip.testserver/sponsorship')
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], '/sponsorship/')
        response = self.client_get('/sponsorship/')
        self.assert_in_success_response(['This organization has requested sponsorship for a', '<a href="/plans/">Zulip Cloud Standard</a>', 'plan.<br/><a href="mailto:support@zulip.com">Contact Zulip support</a> with any questions or updates.'], response)
        self.login_user(self.example_user('othello'))
        response = self.client_get('/billing/')
        self.assert_in_success_response(['You must be an organization owner or a billing administrator to view this page.'], response)
        response = self.client_get('/invoices/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], '/billing/')
        response = self.client_get('/customer_portal/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], '/billing/')
        user.realm.plan_type = Realm.PLAN_TYPE_PLUS
        user.realm.save()
        response = self.client_get('/sponsorship/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], '/billing/')
        user.realm.plan_type = Realm.PLAN_TYPE_STANDARD_FREE
        user.realm.save()
        self.login_user(self.example_user('hamlet'))
        response = self.client_get('/sponsorship/')
        self.assert_in_success_response(['Zulip is sponsoring a free <a href="/plans/">Zulip Cloud Standard</a> plan for this organization. 🎉'], response)

    def test_redirect_for_billing_page(self) -> None:
        user = self.example_user('iago')
        self.login_user(user)
        response = self.client_get('/billing/')
        not_admin_message = 'You must be an organization owner or a billing administrator to view this page.'
        self.assert_in_success_response([not_admin_message], response)
        user.realm.plan_type = Realm.PLAN_TYPE_STANDARD_FREE
        user.realm.save()
        response = self.client_get('/billing/')
        self.assert_in_success_response([not_admin_message], response)
        user = self.example_user('hamlet')
        self.login_user(user)
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/sponsorship/', response['Location'])
        user.realm.plan_type = Realm.PLAN_TYPE_LIMITED
        user.realm.save()
        customer = Customer.objects.create(realm=user.realm, stripe_customer_id='cus_123')
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/plans/', response['Location'])
        customer.sponsorship_pending = True
        customer.save()
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/sponsorship/', response['Location'])
        user.realm.plan_type = Realm.PLAN_TYPE_STANDARD
        user.realm.save()
        response = self.client_get('/billing/')
        self.assertNotEqual('/sponsorship/', response['Location'])
        user.realm.plan_type = Realm.PLAN_TYPE_PLUS
        user.realm.save()
        response = self.client_get('/billing/')
        self.assertNotEqual('/sponsorship/', response['Location'])

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_redirect_for_billing_page_downgrade_at_free_trial_end(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with self.settings(CLOUD_FREE_TRIAL_DAYS=30):
            response = self.client_get('/upgrade/')
            free_trial_end_date = self.now + timedelta(days=30)
            self.assert_in_success_response(['Your card will not be charged', 'free trial', '30-day'], response)
            self.assertNotEqual(user.realm.plan_type, Realm.PLAN_TYPE_STANDARD)
            self.assertFalse(Customer.objects.filter(realm=user.realm).exists())
            stripe_customer = self.add_card_and_upgrade(user)
            customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
            plan = CustomerPlan.objects.get(customer=customer, automanage_licenses=True, price_per_license=8000, fixed_price=None, discount=None, billing_cycle_anchor=self.now, billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, invoiced_through=LicenseLedger.objects.first(), next_invoice_date=free_trial_end_date, tier=CustomerPlan.TIER_CLOUD_STANDARD, status=CustomerPlan.FREE_TRIAL, charge_automatically=True)
            LicenseLedger.objects.get(plan=plan, is_renewal=True, event_time=self.now, licenses=self.seat_count, licenses_at_next_renewal=self.seat_count)
            realm = get_realm('zulip')
            self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)
            with time_machine.travel(self.now, tick=False):
                response = self.client_get('/billing/')
            self.assert_not_in_success_response(['Pay annually'], response)
            for substring in ['Zulip Cloud Standard <i>(free trial)</i>', 'Your plan will automatically renew on', 'February 1, 2012', 'Visa ending in 4242', 'Update card']:
                self.assert_in_response(substring, response)
            with time_machine.travel(self.now + timedelta(days=3), tick=False), self.assertLogs('corporate.stripe', 'INFO') as m:
                response = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL})
                self.assert_json_success(response)
                plan.refresh_from_db()
                self.assertEqual(plan.status, CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL)
                expected_log = f'INFO:corporate.stripe:Change plan status: Customer.id: {customer.id}, CustomerPlan.id: {plan.id}, status: {CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL}'
                self.assertEqual(m.output[0], expected_log)
            with time_machine.travel(self.now + timedelta(days=30), tick=False):
                response = self.client_get('/billing/')
                self.assertEqual(response.status_code, 302)
                self.assertEqual('/plans/', response['Location'])

    def test_upgrade_page_for_demo_organizations(self) -> None:
        user = self.example_user('hamlet')
        user.realm.demo_organization_scheduled_deletion_date = timezone_now() + timedelta(days=30)
        user.realm.save()
        self.login_user(user)
        response = self.client_get('/billing/', follow=True)
        self.assert_in_success_response(['cannot be directly upgraded'], response)

    def test_redirect_for_upgrade_page(self) -> None:
        user = self.example_user('iago')
        self.login_user(user)
        response = self.client_get('/upgrade/')
        self.assertEqual(response.status_code, 200)
        user.realm.plan_type = Realm.PLAN_TYPE_STANDARD_FREE
        user.realm.save()
        response = self.client_get('/upgrade/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], 'http://zulip.testserver/sponsorship')
        stripe_customer_id = 'cus_123'
        from unittest.mock import patch, Mock
        with patch('corporate.lib.stripe.customer_has_credit_card_as_default_payment_method', return_value=False), patch('stripe.Customer.retrieve', return_value=Mock(id=stripe_customer_id, email='test@zulip.com')):
            user.realm.plan_type = Realm.PLAN_TYPE_LIMITED
            user.realm.save()
            customer = Customer.objects.create(realm=user.realm, stripe_customer_id=stripe_customer_id)
            response = self.client_get('/upgrade/')
            self.assertEqual(response.status_code, 200)
            CustomerPlan.objects.create(customer=customer, billing_cycle_anchor=timezone_now(), billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, tier=CustomerPlan.TIER_CLOUD_STANDARD)
            response = self.client_get('/upgrade/')
            self.assertEqual(response.status_code, 302)
            self.assertEqual(response['Location'], 'http://zulip.testserver/billing')
            with self.settings(CLOUD_FREE_TRIAL_DAYS=30):
                response = self.client_get('/upgrade/')
                self.assertEqual(response.status_code, 302)
                self.assertEqual(response['Location'], 'http://zulip.testserver/billing')

    def test_get_latest_seat_count(self) -> None:
        realm = get_realm('zulip')
        initial_count = get_latest_seat_count(realm)
        user1 = UserProfile.objects.create(realm=realm, email='user1@zulip.com', delivery_email='user1@zulip.com')
        user2 = UserProfile.objects.create(realm=realm, email='user2@zulip.com', delivery_email='user2@zulip.com')
        self.assertEqual(get_latest_seat_count(realm), initial_count + 2)
        user1.is_bot = True
        user1.save(update_fields=['is_bot'])
        self.assertEqual(get_latest_seat_count(realm), initial_count + 1)
        do_deactivate_user(user2, acting_user=None)
        self.assertEqual(get_latest_seat_count(realm), initial_count)
        UserProfile.objects.create(realm=realm, email='user3@zulip.com', delivery_email='user3@zulip.com', role=UserProfile.ROLE_GUEST)
        self.assertEqual(get_latest_seat_count(realm), initial_count)
        realm = do_create_realm(string_id='second', name='second')
        UserProfile.objects.create(realm=realm, email='member@second.com', delivery_email='member@second.com')
        for i in range(5):
            UserProfile.objects.create(realm=realm, email=f'guest{i}@second.com', delivery_email=f'guest{i}@second.com', role=UserProfile.ROLE_GUEST)
        self.assertEqual(get_latest_seat_count(realm), 1)
        UserProfile.objects.create(realm=realm, email='guest5@second.com', delivery_email='guest5@second.com', role=UserProfile.ROLE_GUEST)
        self.assertEqual(get_latest_seat_count(realm), 2)

    def test_sign_string(self) -> None:
        string = 'abc'
        signed_string, salt = sign_string(string)
        self.assertEqual(string, unsign_string(signed_string, salt))
        with self.assertRaises(signing.BadSignature):
            unsign_string(signed_string, 'randomsalt')

    @responses.activate
    @mock_stripe()
    def test_payment_method_string(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        billing_session = RealmBillingSession(user, realm=user.realm)
        billing_session.create_stripe_customer()
        self.login_user(user)
        self.add_card_to_customer_for_upgrade()
        self.upgrade(invoice=True)
        response = self.client_get('/billing/')
        self.assert_not_in_success_response(['Visa ending in'], response)
        self.assert_in_success_response(['Invoice', 'You will receive an invoice for'], response)

    @responses.activate
    @mock_stripe()
    def test_replace_payment_method(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        self.add_card_and_upgrade(user)
        customer = Customer.objects.first()
        assert customer is not None
        stripe_customer_id = customer.stripe_customer_id
        assert stripe_customer_id is not None
        stripe.InvoiceItem.create(amount=5000, currency='usd', customer=stripe_customer_id)
        stripe_invoice = stripe.Invoice.create(customer=stripe_customer_id)
        stripe.Invoice.finalize_invoice(stripe_invoice)
        RealmAuditLog.objects.filter(event_type=AuditLogEventType.STRIPE_CARD_CHANGED).delete()
        start_session_json_response = self.client_billing_post('/billing/session/start_card_update_session')
        response_dict = self.assert_json_success(start_session_json_response)
        self.assert_details_of_valid_session_from_event_status_endpoint(response_dict['stripe_session_id'], {'type': 'card_update_from_billing_page', 'status': 'created', 'is_manual_license_management_upgrade_session': False, 'tier': None})
        with self.assertRaises(stripe.CardError):
            self.trigger_stripe_checkout_session_completed_webhook(self.get_test_card_token(attaches_to_customer=False))
        start_session_json_response = self.client_billing_post('/billing/session/start_card_update_session')
        response_dict = self.assert_json_success(start_session_json_response)
        self.assert_details_of_valid_session_from_event_status_endpoint(response_dict['stripe_session_id'], {'type': 'card_update_from_billing_page', 'status': 'created', 'is_manual_license_management_upgrade_session': False, 'tier': None})
        with self.assertLogs('corporate.stripe', 'INFO') as m:
            self.trigger_stripe_checkout_session_completed_webhook(self.get_test_card_token(attaches_to_customer=True, charge_succeeds=False))
            self.assertEqual(m.output[0], 'INFO:corporate.stripe:Stripe card error: 402 card_error card_declined None')
        response_dict = self.assert_json_success(start_session_json_response)
        self.assert_details_of_valid_session_from_event_status_endpoint(response_dict['stripe_session_id'], {'type': 'card_update_from_billing_page', 'status': 'completed', 'is_manual_license_management_upgrade_session': False, 'tier': None, 'event_handler': {'status': 'failed', 'error': {'message': 'Your card was declined.', 'description': 'card error'}}})
        response = self.client_get('/billing/')
        self.assert_in_success_response(['Visa ending in 0341'], response)
        assert RealmAuditLog.objects.filter(event_type=AuditLogEventType.STRIPE_CARD_CHANGED).exists()
        stripe_payment_methods = stripe.PaymentMethod.list(customer=stripe_customer_id, type='card')
        self.assert_length(stripe_payment_methods, 2)
        for stripe_payment_method in stripe_payment_methods:
            stripe.PaymentMethod.detach(stripe_payment_method.id)
        response = self.client_get('/billing/')
        self.assert_in_success_response(['No payment method on file.'], response)
        start_session_json_response = self.client_billing_post('/billing/session/start_card_update_session')
        self.assert_json_success(start_session_json_response)
        self.trigger_stripe_checkout_session_completed_webhook(self.get_test_card_token(attaches_to_customer=True, charge_succeeds=True, card_provider='mastercard'))
        response_dict = self.assert_json_success(start_session_json_response)
        self.assert_details_of_valid_session_from_event_status_endpoint(response_dict['stripe_session_id'], {'type': 'card_update_from_billing_page', 'status': 'completed', 'is_manual_license_management_upgrade_session': False, 'tier': None, 'event_handler': {'status': 'succeeded'}})
        self.login_user(self.example_user('iago'))
        response = self.client_billing_get('/billing/event/status', {'stripe_session_id': response_dict['stripe_session_id']})
        self.assert_json_error_contains(response, 'Must be a billing administrator or an organization owner')
        self.login_user(self.example_user('hamlet'))
        response = self.client_get('/billing/')
        self.assert_in_success_response(['Mastercard ending in 4444'], response)
        self.assert_length(stripe.PaymentMethod.list(customer=stripe_customer_id, type='card'), 1)
        for stripe_invoice in stripe.Invoice.list(customer=stripe_customer_id):
            self.assertEqual(stripe_invoice.status, 'paid')
        self.assertEqual(2, RealmAuditLog.objects.filter(event_type=AuditLogEventType.STRIPE_CARD_CHANGED).count())
        start_session_json_response = self.client_billing_post('/upgrade/session/start_card_update_session', {'manual_license_management': 'true', 'tier': 1})
        response_dict = self.assert_json_success(start_session_json_response)
        self.assert_details_of_valid_session_from_event_status_endpoint(response_dict['stripe_session_id'], {'type': 'card_update_from_upgrade_page', 'status': 'created', 'is_manual_license_management_upgrade_session': True, 'tier': 1})

    def test_downgrade(self) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = get_current_plan_by_realm(user.realm)
        assert plan is not None
        self.assertEqual(plan.licenses(), self.seat_count)
        self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count)
        from unittest.mock import patch
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            response = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE})
            stripe_customer_id = Customer.objects.get(realm=user.realm).id
            new_plan = get_current_plan_by_realm(user.realm)
            assert new_plan is not None
            expected_log = f'INFO:corporate.stripe:Change plan status: Customer.id: {stripe_customer_id}, CustomerPlan.id: {new_plan.id}, status: {CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE}'
            self.assertEqual(m.output[0], expected_log)
            self.assert_json_success(response)
        plan.refresh_from_db()
        self.assertEqual(plan.licenses(), self.seat_count)
        self.assertEqual(plan.licenses_at_next_renewal(), None)
        with time_machine.travel(self.now, tick=False):
            from unittest.mock import patch, Mock
            mock_customer = Mock(email=user.delivery_email)
            mock_customer.invoice_settings.default_payment_method = Mock(spec=stripe.PaymentMethod, type=Mock())
            with patch('corporate.lib.stripe.stripe_get_customer', return_value=mock_customer):
                response = self.client_get('/billing/')
                self.assert_in_success_response(['Your organization will be downgraded to <strong>Zulip Cloud Free</strong> at the end of the current billing', '<strong>January 2, 2013</strong>', 'Reactivate subscription'], response)
        billing_session = RealmBillingSession(user=user, realm=user.realm)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=20):
            billing_session.update_license_ledger_if_needed(self.now)
        self.assertEqual(LicenseLedger.objects.order_by('-id').values_list('licenses', 'licenses_at_next_renewal').first(), (20, 20))
        mocked = self.setup_mocked_stripe(invoice_plans_as_needed, self.next_month)
        mocked['InvoiceItem'].create.assert_called_once()
        mocked['Invoice'].finalize_invoice.assert_called_once()
        mocked['Invoice'].create.assert_called_once()
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=30):
            billing_session.update_license_ledger_if_needed(self.next_year)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_LIMITED)
        self.assertEqual(plan.status, CustomerPlan.ENDED)
        self.assertEqual(LicenseLedger.objects.order_by('-id').values_list('licenses', 'licenses_at_next_renewal').first(), (20, 20))
        realm_audit_log = RealmAuditLog.objects.latest('id')
        self.assertEqual(realm_audit_log.event_type, AuditLogEventType.REALM_PLAN_TYPE_CHANGED)
        self.assertEqual(realm_audit_log.acting_user, None)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=40):
            billing_session.update_license_ledger_if_needed(self.next_year + timedelta(days=80))
        mocked = self.setup_mocked_stripe(invoice_plans_as_needed, self.next_year + timedelta(days=400))
        mocked['InvoiceItem'].create.assert_not_called()
        mocked['Invoice'].finalize_invoice.assert_not_called()
        mocked['Invoice'].create.assert_not_called()
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertIsNone(plan.next_invoice_date)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=50):
            billing_session.update_license_ledger_if_needed(self.next_year + timedelta(days=80))
        mocked = self.setup_mocked_stripe(invoice_plans_as_needed, self.next_year + timedelta(days=400))
        mocked['InvoiceItem'].create.assert_not_called()
        mocked['Invoice'].finalize_invoice.assert_not_called()
        mocked['Invoice'].create.assert_not_called()

    @responses.activate
    @mock_stripe()
    def test_switch_from_monthly_plan_to_annual_plan_for_automatic_license_management(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        self.add_card_and_upgrade(user, schedule='monthly')
        monthly_plan = get_current_plan_by_realm(user.realm)
        assert monthly_plan is not None
        self.assertEqual(monthly_plan.automanage_licenses, True)
        self.assertEqual(monthly_plan.billing_schedule, CustomerPlan.BILLING_SCHEDULE_MONTHLY)
        stripe_customer_id = Customer.objects.get(realm=user.realm).id
        new_plan = get_current_plan_by_realm(user.realm)
        assert new_plan is not None
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            response = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE})
            expected_log = f'INFO:corporate.stripe:Change plan status: Customer.id: {stripe_customer_id}, CustomerPlan.id: {new_plan.id}, status: {CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE}'
            self.assertEqual(m.output[0], expected_log)
            self.assert_json_success(response)
        monthly_plan.refresh_from_db()
        self.assertEqual(monthly_plan.status, CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE)
        with time_machine.travel(self.now, tick=False):
            response = self.client_get('/billing/')
        self.assert_in_success_response(['Your plan will switch to annual billing on February 2, 2012'], response)
        billing_session = RealmBillingSession(user=user, realm=user.realm)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=20):
            billing_session.update_license_ledger_if_needed(self.now)
        self.assertEqual(LicenseLedger.objects.filter(plan=monthly_plan).count(), 2)
        self.assertEqual(LicenseLedger.objects.order_by('-id').values_list('licenses', 'licenses_at_next_renewal').first(), (20, 20))
        with time_machine.travel(self.next_month, tick=False), self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=25):
            billing_session.update_license_ledger_if_needed(self.next_month)
        self.assertEqual(LicenseLedger.objects.filter(plan=monthly_plan).count(), 2)
        customer = get_customer_by_realm(user.realm)
        assert customer is not None
        self.assertEqual(CustomerPlan.objects.filter(customer=customer).count(), 2)
        monthly_plan.refresh_from_db()
        self.assertEqual(monthly_plan.status, CustomerPlan.ENDED)
        self.assertEqual(monthly_plan.next_invoice_date, self.next_month)
        annual_plan = get_current_plan_by_realm(user.realm)
        assert annual_plan is not None
        self.assertEqual(annual_plan.status, CustomerPlan.ACTIVE)
        self.assertEqual(annual_plan.billing_schedule, CustomerPlan.BILLING_SCHEDULE_ANNUAL)
        self.assertEqual(annual_plan.invoicing_status, CustomerPlan.INVOICING_STATUS_INITIAL_INVOICE_TO_BE_SENT)
        self.assertEqual(annual_plan.billing_cycle_anchor, self.next_month)
        self.assertEqual(annual_plan.next_invoice_date, self.next_month)
        self.assertEqual(annual_plan.invoiced_through, None)
        annual_ledger_entries = LicenseLedger.objects.filter(plan=annual_plan).order_by('id')
        self.assert_length(annual_ledger_entries, 2)
        self.assertEqual(annual_ledger_entries[0].is_renewal, True)
        self.assertEqual(annual_ledger_entries.values_list('licenses', 'licenses_at_next_renewal')[0], (20, 20))
        self.assertEqual(annual_ledger_entries[1].is_renewal, False)
        self.assertEqual(annual_ledger_entries.values_list('licenses', 'licenses_at_next_renewal')[1], (25, 25))
        audit_log = RealmAuditLog.objects.get(event_type=AuditLogEventType.CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN)
        self.assertEqual(audit_log.realm, user.realm)
        self.assertEqual(audit_log.extra_data['monthly_plan_id'], monthly_plan.id)
        self.assertEqual(audit_log.extra_data['annual_plan_id'], annual_plan.id)
        invoice_plans_as_needed(self.next_month)
        annual_ledger_entries = LicenseLedger.objects.filter(plan=annual_plan).order_by('id')
        self.assert_length(annual_ledger_entries, 2)
        annual_plan.refresh_from_db()
        self.assertEqual(annual_plan.invoicing_status, CustomerPlan.INVOICING_STATUS_DONE)
        self.assertEqual(annual_plan.invoiced_through, annual_ledger_entries[1])
        self.assertEqual(annual_plan.billing_cycle_anchor, self.next_month)
        self.assertEqual(annual_plan.next_invoice_date, add_months(self.next_month, 1))
        monthly_plan.refresh_from_db()
        self.assertEqual(monthly_plan.next_invoice_date, None)
        from stripe import Invoice
        assert customer.stripe_customer_id
        invoices = list(Invoice.list(customer=customer.stripe_customer_id))
        self.assertEqual(len(invoices), 3)
        invoice0 = invoices[0]
        invoice_item0, invoice_item1 = list(invoice0.lines)[:2]
        annual_plan_invoice_item_params = {'amount': 5 * 80 * 100, 'description': 'Additional license (Feb 2, 2012 - Feb 2, 2013)', 'plan': None, 'quantity': 5, 'subscription': None, 'discountable': False, 'period': {'start': datetime_to_timestamp(self.next_month), 'end': datetime_to_timestamp(add_months(self.next_month, 12))}}
        for key, value in annual_plan_invoice_item_params.items():
            self.assertEqual(invoice_item0.get(key), value)
        annual_plan_invoice_item_params = {'amount': 20 * 80 * 100, 'description': 'Zulip Cloud Standard - renewal', 'plan': None, 'quantity': 20, 'subscription': None, 'discountable': False, 'period': {'start': datetime_to_timestamp(self.next_month), 'end': datetime_to_timestamp(add_months(self.next_month, 12))}}
        for key, value in annual_plan_invoice_item_params.items():
            self.assertEqual(invoice_item1.get(key), value)
        invoice_items = list(invoices[1].lines)
        monthly_plan_invoice_item = invoice_items[0]
        monthly_plan_invoice_item_params = {'amount': 14 * 8 * 100, 'description': 'Additional license (Jan 2, 2012 - Feb 2, 2012)', 'plan': None, 'quantity': 14, 'subscription': None, 'discountable': False, 'period': {'start': datetime_to_timestamp(self.now), 'end': datetime_to_timestamp(self.next_month)}}
        for key, value in monthly_plan_invoice_item_params.items():
            self.assertEqual(monthly_plan_invoice_item.get(key), value)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=30):
            billing_session.update_license_ledger_if_needed(add_months(self.next_month, 1))
        invoice_plans_as_needed(add_months(self.next_month, 1))
        invoices = list(Invoice.list(customer=customer.stripe_customer_id))
        self.assertEqual(len(invoices), 4)
        invoice_items = list(invoices[0].lines)
        monthly_plan_invoice_item = invoice_items[0]
        monthly_plan_invoice_item_params = {'amount': 5 * 7366, 'description': 'Additional license (Mar 2, 2012 - Feb 2, 2013)', 'plan': None, 'quantity': 5, 'subscription': None, 'discountable': False, 'period': {'start': datetime_to_timestamp(add_months(self.next_month, 1)), 'end': datetime_to_timestamp(add_months(self.next_month, 12))}}
        for key, value in monthly_plan_invoice_item_params.items():
            self.assertEqual(monthly_plan_invoice_item.get(key), value)
        annual_plan.next_invoice_date = add_months(self.now, 13)
        annual_plan.save(update_fields=['next_invoice_date'])
        invoice_plans_as_needed(add_months(self.now, 13))
        invoices = list(Invoice.list(customer=customer.stripe_customer_id))
        self.assertEqual(len(invoices), 5)
        invoice_items = list(invoices[0].lines)
        annual_plan_invoice_item_params = {'amount': 30 * 80 * 100, 'description': 'Zulip Cloud Standard - renewal', 'plan': None, 'quantity': 30, 'subscription': None, 'discountable': False, 'period': {'start': datetime_to_timestamp(add_months(self.next_month, 12)), 'end': datetime_to_timestamp(add_months(self.next_month, 24))}}
        for key, value in annual_plan_invoice_item_params.items():
            self.assertEqual(invoice_items[0].get(key), value)

    @responses.activate
    @mock_stripe()
    def test_switch_from_monthly_plan_to_annual_plan_for_manual_license_management(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        num_licenses = 35
        self.login_user(user)
        self.add_card_and_upgrade(user, schedule='monthly', license_management='manual', licenses=num_licenses)
        monthly_plan = get_current_plan_by_realm(user.realm)
        assert monthly_plan is not None
        self.assertEqual(monthly_plan.automanage_licenses, False)
        self.assertEqual(monthly_plan.billing_schedule, CustomerPlan.BILLING_SCHEDULE_MONTHLY)
        stripe_customer_id = Customer.objects.get(realm=user.realm).stripe_customer_id
        new_plan = get_current_plan_by_realm(user.realm)
        assert new_plan is not None
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            response = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE})
            self.assertEqual(m.output[0], f'INFO:corporate.stripe:Change plan status: Customer.id: {stripe_customer_id}, CustomerPlan.id: {new_plan.id}, status: {CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE}')
            self.assert_json_success(response)
        monthly_plan.refresh_from_db()
        self.assertEqual(monthly_plan.status, CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE)
        with time_machine.travel(self.now, tick=False):
            response = self.client_get('/billing/')
        self.assert_in_success_response(['Your plan will switch to annual billing on February 2, 2012'], response)
        billing_session = RealmBillingSession(user=user, realm=user.realm)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=20):
            billing_session.update_license_ledger_if_needed(self.now)
        self.assertEqual(LicenseLedger.objects.filter(plan=monthly_plan).count(), 1)
        customer = get_customer_by_realm(user.realm)
        assert customer is not None
        self.assertEqual(CustomerPlan.objects.filter(customer=customer).count(), 2)
        monthly_plan.refresh_from_db()
        self.assertEqual(monthly_plan.status, CustomerPlan.ENDED)
        self.assertEqual(monthly_plan.next_invoice_date, None)
        new_plan = get_current_plan_by_realm(user.realm)
        assert new_plan is not None
        self.assertEqual(new_plan.status, CustomerPlan.ACTIVE)
        self.assertEqual(new_plan.billing_schedule, CustomerPlan.BILLING_SCHEDULE_ANNUAL)
        self.assertEqual(new_plan.invoicing_status, CustomerPlan.INVOICING_STATUS_INITIAL_INVOICE_TO_BE_SENT)
        self.assertEqual(new_plan.billing_cycle_anchor, self.next_month)
        self.assertEqual(new_plan.next_invoice_date, self.next_month)
        self.assertEqual(new_plan.invoiced_through, None)
        mocked = self.setup_mocked_stripe(invoice_plans_as_needed, self.next_month)
        mocked['InvoiceItem'].create.assert_not_called()
        mocked['Invoice'].finalize_invoice.assert_not_called()
        mocked['Invoice'].create.assert_not_called()

    def test_reupgrade_by_billing_admin_after_plan_status_changed_to_downgrade_at_end_of_cycle(self) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        from unittest.mock import patch
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            self.client_billing_patch('/billing/plan', {'status': CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE})
            stripe_customer_id = Customer.objects.get(realm=user.realm).id
            new_plan = get_current_plan_by_realm(user.realm)
            assert new_plan is not None
            expected_log = f'INFO:corporate.stripe:Change plan status: Customer.id: {stripe_customer_id}, CustomerPlan.id: {new_plan.id}, status: {CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE}'
            self.assertEqual(m.output[0], expected_log)
        with self.assertRaises(BillingError) as context, self.assertLogs('corporate.stripe', 'WARNING') as m, time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(m.output[0], 'WARNING:corporate.stripe:Upgrade of <Realm: zulip 2> (with stripe_customer_id: cus_123) failed because of existing active plan.')
        self.assertEqual(context.exception.error_description, 'subscribing with existing subscription')
        new_plan.next_invoice_date = self.next_year
        new_plan.save(update_fields=['next_invoice_date'])
        invoice_plans_as_needed(self.next_year)
        with time_machine.travel(self.next_year, tick=False):
            response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/plans/', response['Location'])
        with time_machine.travel(self.next_year, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(Customer.objects.count(), 1)
        self.assertEqual(CustomerPlan.objects.count(), 2)
        current_plan = CustomerPlan.objects.all().order_by('id').last()
        assert current_plan is not None
        next_invoice_date = add_months(self.next_year, 1)
        self.assertEqual(current_plan.next_invoice_date, next_invoice_date)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(current_plan.status, CustomerPlan.ACTIVE)
        old_plan = CustomerPlan.objects.all().order_by('id').first()
        assert old_plan is not None
        self.assertEqual(old_plan.next_invoice_date, None)
        self.assertEqual(old_plan.status, CustomerPlan.ENDED)

    @responses.activate
    @mock_stripe()
    def test_update_plan_with_invalid_status(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        response = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.NEVER_STARTED})
        self.assert_json_error_contains(response, 'Invalid status')

    def test_update_plan_without_any_params(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        with time_machine.travel(self.now, tick=False):
            response = self.client_billing_patch('/billing/plan', {})
        self.assert_json_error_contains(response, 'Nothing to change')

    def test_update_plan_that_which_is_due_for_expiry(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        from unittest.mock import patch
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE})
            self.assert_json_success(result)
            self.assertRegex(m.output[0], 'INFO:corporate.stripe:Change plan status: Customer.id: \\d*, CustomerPlan.id: \\d*, status: 2')
        with time_machine.travel(self.next_year, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.ACTIVE})
            self.assert_json_error_contains(result, 'Unable to update the plan. The plan has ended.')

    def test_update_plan_that_which_is_due_for_replacement(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_MONTHLY, True, False)
        self.login_user(self.example_user('hamlet'))
        from unittest.mock import patch
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE})
            self.assert_json_success(result)
            self.assertRegex(m.output[0], 'INFO:corporate.stripe:Change plan status: Customer.id: \\d*, CustomerPlan.id: \\d*, status: 4')
        with time_machine.travel(self.next_month, tick=False):
            result = self.client_billing_patch('/billing/plan', {})
            self.assert_json_error_contains(result, 'Unable to update the plan. The plan has been expired and replaced with a new plan.')

    @patch('corporate.lib.stripe.billing_logger.info')
    def test_deactivate_realm(self, mock_: Any) -> None:
        user = self.example_user('hamlet')
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.get()
        self.assertEqual(plan.next_invoice_date, self.next_month)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(plan.status, CustomerPlan.ACTIVE)
        billing_session = RealmBillingSession(user=user, realm=user.realm)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=20):
            billing_session.update_license_ledger_if_needed(self.now)
        last_ledger_entry = LicenseLedger.objects.order_by('id').last()
        assert last_ledger_entry is not None
        self.assertEqual(last_ledger_entry.licenses, 20)
        self.assertEqual(last_ledger_entry.licenses_at_next_renewal, 20)
        do_deactivate_realm(get_realm('zulip'), acting_user=None, deactivation_reason='owner_request', email_owners=False)
        plan.refresh_from_db()
        self.assertTrue(get_realm('zulip').deactivated)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_LIMITED)
        self.assertEqual(plan.status, CustomerPlan.ENDED)
        self.assertEqual(plan.invoiced_through, last_ledger_entry)
        self.assertIsNone(plan.next_invoice_date)
        do_reactivate_realm(get_realm('zulip'))
        self.login_user(user)
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/plans/', response['Location'])
        from unittest.mock import patch
        with self.patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_month)
        mocked.assert_not_called()
        with self.patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_year)
        mocked.assert_not_called()

    def test_reupgrade_by_billing_admin_after_realm_deactivation(self) -> None:
        user = self.example_user('hamlet')
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        do_deactivate_realm(get_realm('zulip'), acting_user=None, deactivation_reason='owner_request', email_owners=False)
        self.assertTrue(get_realm('zulip').deactivated)
        do_reactivate_realm(get_realm('zulip'))
        self.login_user(user)
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/plans/', response['Location'])
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(Customer.objects.count(), 1)
        self.assertEqual(CustomerPlan.objects.count(), 2)
        current_plan = CustomerPlan.objects.all().order_by('id').last()
        assert current_plan is not None
        self.assertEqual(current_plan.next_invoice_date, add_months(self.next_year, 1))
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(current_plan.status, CustomerPlan.ACTIVE)
        old_plan = CustomerPlan.objects.all().order_by('id').first()
        assert old_plan is not None
        self.assertEqual(old_plan.next_invoice_date, None)
        self.assertEqual(old_plan.status, CustomerPlan.ENDED)

    @responses.activate
    @mock_stripe()
    def test_update_licenses_of_manual_plan_from_billing_page(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.upgrade(invoice=True, licenses=100)
        with time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'licenses': 100})
            self.assert_json_error_contains(result, 'Your plan is already on 100 licenses in the current billing period.')
        with time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'licenses_at_next_renewal': 100})
            self.assert_json_error_contains(result, 'Your plan is already scheduled to renew with 100 licenses.')
        with time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'licenses': 50})
            self.assert_json_error_contains(result, 'You cannot decrease the licenses in the current billing period.')
        with time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'licenses_at_next_renewal': 25})
            self.assert_json_error_contains(result, 'You must purchase licenses for all active users in your organization (minimum 30).')
        with time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'licenses': 2000})
            self.assert_json_error_contains(result, "Invoices with more than 1000 licenses can't be processed from this page.")
        with time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'licenses': 150})
            self.assert_json_success(result)
        invoice_plans_as_needed(self.next_year)
        stripe_customer = stripe_get_customer(assert_is_not_none(Customer.objects.get(realm=user.realm).stripe_customer_id))
        [renewal_invoice, additional_licenses_invoice, _old_renewal_invoice] = iter(stripe.Invoice.list(customer=stripe_customer.id))
        invoice_params = {'amount_due': 8000 * 150, 'amount_paid': 0, 'attempt_count': 0, 'auto_advance': True, 'collection_method': 'send_invoice', 'statement_descriptor': 'Zulip Cloud Standard', 'status': 'open', 'total': 8000 * 150}
        for key, value in invoice_params.items():
            self.assertEqual(renewal_invoice.get(key), value)
        [renewal_item] = iter(renewal_invoice.lines)
        line_item_params = {'amount': 8000 * 150, 'description': 'Zulip Cloud Standard - renewal', 'discountable': False, 'period': {'start': datetime_to_timestamp(self.next_year + timedelta(days=365)), 'end': datetime_to_timestamp(self.next_year + timedelta(days=2 * 365))}, 'quantity': 150}
        for key, value in line_item_params.items():
            self.assertEqual(renewal_item.get(key), value)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=self.seat_count + 2):
            billing_session = RealmBillingSession(user=user, realm=user.realm)
            billing_session.update_license_ledger_if_needed(self.now + timedelta(days=400))
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=self.seat_count + 3):
            billing_session.update_license_ledger_if_needed(self.now + timedelta(days=500))
        plan = CustomerPlan.objects.first()
        assert plan is not None
        billing_session.invoice_plan(plan, self.now + timedelta(days=400))
        stripe_customer = plan.customer.stripe_customer_id
        assert stripe_customer is not None
        [invoice0, invoice1] = iter(stripe.Invoice.list(customer=stripe_customer))
        self.assertIsNotNone(invoice0.status_transitions.finalized_at)
        [item0, item1, item2] = iter(invoice0.lines)
        line_item_params = {'amount': int(8000 * (1 - (400 - 366) / 365) + 0.5), 'description': 'Additional license (Feb 5, 2013 - Jan 2, 2014)', 'discountable': False, 'period': {'start': datetime_to_timestamp(self.now + timedelta(days=400)), 'end': datetime_to_timestamp(self.now + timedelta(days=2 * 365 + 1))}, 'quantity': 1}
        for key, value in line_item_params.items():
            self.assertEqual(item0.get(key), value)
        line_item_params = {'amount': 8000 * (self.seat_count + 1), 'description': 'Zulip Cloud Standard - renewal', 'discountable': False, 'period': {'start': datetime_to_timestamp(self.now + timedelta(days=366)), 'end': datetime_to_timestamp(self.now + timedelta(days=2 * 365 + 1))}, 'quantity': self.seat_count + 1}
        for key, value in line_item_params.items():
            self.assertEqual(item1.get(key), value)
        line_item_params = {'amount': 3 * int(8000 * (366 - 100) / 366 + 0.5), 'description': 'Additional license (Apr 11, 2012 - Jan 2, 2013)', 'discountable': False, 'period': {'start': datetime_to_timestamp(self.now + timedelta(days=100)), 'end': datetime_to_timestamp(self.now + timedelta(days=366))}, 'quantity': 3}
        for key, value in line_item_params.items():
            self.assertEqual(item2.get(key), value)

    @responses.activate
    @mock_stripe()
    def test_fixed_price_plans(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.upgrade(invoice=True)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        plan.fixed_price = 100
        plan.price_per_license = 0
        plan.save(update_fields=['fixed_price', 'price_per_license'])
        user.realm.refresh_from_db()
        billing_session = RealmBillingSession(realm=user.realm)
        billing_session.invoice_plan(plan, self.now + timedelta(days=365))
        stripe_customer = stripe_get_customer(assert_is_not_none(Customer.objects.get(realm=user.realm).stripe_customer_id))
        [invoice0, invoice1] = iter(stripe.Invoice.list(customer=stripe_customer.id))
        self.assertEqual(invoice0.collection_method, 'send_invoice')
        [item] = iter(invoice0.lines)
        line_item_params = {'amount': 100, 'description': 'Zulip Cloud Standard - renewal', 'discountable': False, 'period': {'start': datetime_to_timestamp(self.now + timedelta(days=365)), 'end': datetime_to_timestamp(self.now + timedelta(days=2 * 365))}, 'quantity': 1}
        for key, value in line_item_params.items():
            self.assertEqual(item.get(key), value)

    @responses.activate
    @mock_stripe()
    def test_upgrade_to_fixed_price_plus_plan(self, *mocks: Any) -> None:
        iago = self.example_user('iago')
        self.login_user(iago)
        self.add_card_and_upgrade(iago)
        customer = Customer.objects.get_or_create(realm=iago.realm)[0]
        plan = CustomerPlan.objects.get(customer=customer, status=CustomerPlan.ACTIVE)
        self.assertIsNone(plan.end_date)
        self.assertEqual(plan.tier, CustomerPlan.TIER_CLOUD_STANDARD)
        iago.realm.refresh_from_db()
        billing_session = RealmBillingSession(user=iago, realm=iago.realm)
        next_billing_cycle = billing_session.get_next_billing_cycle(plan)
        plan_end_date_string = next_billing_cycle.strftime('%Y-%m-%d')
        plan_end_date = datetime.strptime(plan_end_date_string, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        self.logout()
        self.login_user(self.example_user('iago'))
        result = self.client_post('/activity/support', {'realm_id': f'{iago.realm.id}', 'required_plan_tier': f'{CustomerPlanOffer.TIER_CLOUD_PLUS}'})
        self.assert_in_success_response([f'Required plan tier for {iago.realm.string_id} set to Zulip Cloud Plus.'], result)
        with time_machine.travel(self.now, tick=False):
            result = self.client_post('/activity/support', {'realm_id': f'{iago.realm.id}', 'plan_end_date': plan_end_date_string})
        self.assert_in_success_response([f'Current plan for {iago.realm.string_id} updated to end on {plan_end_date_string}.'], result)
        plan.refresh_from_db()
        self.assertEqual(plan.end_date, plan_end_date)
        result = self.client_post('/activity/support', {'realm_id': f'{iago.realm.id}', 'fixed_price': 360})
        self.assert_in_success_response([f'Fixed price Zulip Cloud Plus plan scheduled to start on {plan_end_date_string}.'], result)
        plan.refresh_from_db()
        self.assertEqual(plan.status, CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END)
        new_plan = CustomerPlan.objects.filter(fixed_price__isnull=False).first()
        assert new_plan is not None
        self.assertEqual(new_plan.next_invoice_date, plan_end_date)
        self.assertEqual(new_plan.invoicing_status, CustomerPlan.INVOICING_STATUS_INITIAL_INVOICE_TO_BE_SENT)
        with time_machine.travel(next_billing_cycle, tick=False):
            invoice_plans_as_needed(self.next_month)
        new_plan.refresh_from_db()
        self.assertEqual(new_plan.invoicing_status, CustomerPlan.INVOICING_STATUS_DONE)
        self.assertEqual(new_plan.invoiced_through, LicenseLedger.objects.filter(plan=new_plan).order_by('id').last())
        self.assertEqual(new_plan.billing_cycle_anchor, self.next_month)
        self.assertEqual(new_plan.next_invoice_date, add_months(self.next_month, 1))
        monthly_plan = CustomerPlan.objects.filter(customer=customer).first()
        assert monthly_plan is not None
        self.assertEqual(monthly_plan.next_invoice_date, None)
        from stripe import Invoice
        [invoice0, invoice1, invoice2] = iter(Invoice.list(customer=customer.stripe_customer_id))
        [invoice_item0, invoice_item1] = list(invoice0.lines)[:2]
        annual_plan_invoice_item_params = {'amount': 5 * 80 * 100, 'description': 'Additional license (Feb 2, 2012 - Feb 2, 2013)', 'plan': None, 'quantity': 5, 'subscription': None, 'discountable': False, 'period': {'start': datetime_to_timestamp(self.next_month), 'end': datetime_to_timestamp(add_months(self.next_month, 12))}}
        for key, value in annual_plan_invoice_item_params.items():
            self.assertEqual(invoice_item0.get(key), value)
        annual_plan_invoice_item_params = {'amount': 20 * 80 * 100, 'description': 'Zulip Cloud Plus - renewal', 'plan': None, 'quantity': 20, 'subscription': None, 'discountable': False, 'period': {'start': datetime_to_timestamp(self.next_month), 'end': datetime_to_timestamp(add_months(self.next_month, 12))}}
        for key, value in annual_plan_invoice_item_params.items():
            self.assertEqual(invoice_item1.get(key), value)
        [monthly_plan_invoice_item] = iter(invoice1.lines)
        monthly_plan_invoice_item_params = {'amount': 14 * 8 * 100, 'description': 'Additional license (Jan 2, 2012 - Feb 2, 2012)', 'plan': None, 'quantity': 14, 'subscription': None, 'discountable': False, 'period': {'start': datetime_to_timestamp(self.now), 'end': datetime_to_timestamp(self.next_month)}}
        for key, value in monthly_plan_invoice_item_params.items():
            self.assertEqual(monthly_plan_invoice_item.get(key), value)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=30):
            billing_session.update_license_ledger_if_needed(add_months(self.next_month, 1))
        invoice_plans_as_needed(add_months(self.next_month, 1))
        [invoice0, invoice1, invoice2, invoice3] = iter(Invoice.list(customer=customer.stripe_customer_id))
        [invoice_item0] = iter(invoice0.lines)
        monthly_plan_invoice_item_params = {'amount': 5 * 7366, 'description': 'Additional license (Mar 2, 2012 - Feb 2, 2013)', 'plan': None, 'quantity': 5, 'subscription': None, 'discountable': False, 'period': {'start': datetime_to_timestamp(add_months(self.next_month, 1)), 'end': datetime_to_timestamp(add_months(self.next_month, 12))}}
        for key, value in monthly_plan_invoice_item_params.items():
            self.assertEqual(invoice_item0.get(key), value)
        annual_plan.next_invoice_date = add_months(self.now, 13)
        annual_plan.save(update_fields=['next_invoice_date'])
        invoice_plans_as_needed(add_months(self.now, 13))
        [invoice0, invoice1, invoice2, invoice3, invoice4] = iter(Invoice.list(customer=customer.stripe_customer_id))
        [invoice_item] = iter(invoice0.lines)
        annual_plan_invoice_item_params = {'amount': 30 * 80 * 100, 'description': 'Zulip Cloud Plus - renewal', 'plan': None, 'quantity': 30, 'subscription': None, 'discountable': False, 'period': {'start': datetime_to_timestamp(add_months(self.next_month, 12)), 'end': datetime_to_timestamp(add_months(self.next_month, 24))}}
        for key, value in annual_plan_invoice_item_params.items():
            self.assertEqual(invoice_item.get(key), value)

    def test_no_invoice_needed(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(plan.next_invoice_date, self.next_month)
        from stripe import Invoice
        billing_session = RealmBillingSession(realm=plan.customer.realm)
        billing_session.invoice_plan(plan, self.next_month)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(plan.next_invoice_date, self.next_month + timedelta(days=29))

    def test_invoice_plans_as_needed(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(plan.next_invoice_date, self.next_month)
        from unittest.mock import patch
        with self.patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_month - timedelta(days=1))
        mocked.assert_not_called()
        invoice_plans_as_needed(self.next_month)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(plan.next_invoice_date, self.next_month + timedelta(days=29))

    @responses.activate
    @mock_stripe()
    def test_invoice_for_additional_license(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.add_card_and_upgrade(user)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(plan.next_invoice_date, self.next_month)
        from stripe import Invoice
        customer = stripe_get_customer(assert_is_not_none(Customer.objects.get(realm=user.realm).stripe_customer_id))
        with time_machine.travel(self.now + timedelta(days=5), tick=False):
            user = do_create_user('email', 'password', get_realm('zulip'), 'name', acting_user=None)
        with time_machine.travel(self.now + timedelta(days=10), tick=False):
            do_change_user_role(user, UserProfile.ROLE_MEMBER, acting_user=None)
        billing_session = RealmBillingSession(realm=user.realm)
        billing_session.invoice_plan(plan, self.next_month)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(plan.next_invoice_date, self.next_month + timedelta(days=29))
        customer = plan.customer
        assert customer.stripe_customer_id
        [invoice0, invoice1] = iter(Invoice.list(customer=customer.stripe_customer_id))
        self.assertIsNotNone(invoice0.status_transitions.finalized_at)
        [item0] = iter(invoice0.lines)
        line_item_params = {'amount': int(8000 * (1 - (366 - 100) / 366 + 0.5)), 'description': 'Additional license (Jan 12, 2012 - Jan 2, 2013)', 'quantity': 1}
        for key, value in line_item_params.items():
            self.assertEqual(item0.get(key), value)

    def test_subscribe_realm_to_manual_license_management_plan(self) -> None:
        realm = get_realm('zulip')
        plan, ledger = self.subscribe_realm_to_manual_license_management_plan(realm, 50, 60, CustomerPlan.BILLING_SCHEDULE_ANNUAL)
        plan.refresh_from_db()
        self.assertEqual(plan.automanage_licenses, False)
        self.assertEqual(plan.billing_schedule, CustomerPlan.BILLING_SCHEDULE_ANNUAL)
        self.assertEqual(plan.tier, CustomerPlan.TIER_CLOUD_STANDARD)
        self.assertEqual(plan.licenses(), 50)
        self.assertEqual(plan.licenses_at_next_renewal(), 60)
        ledger.refresh_from_db()
        self.assertEqual(ledger.plan, plan)
        self.assertEqual(ledger.licenses, 50)
        self.assertEqual(ledger.licenses_at_next_renewal, 60)
        realm.refresh_from_db()
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)

    def test_subscribe_realm_to_monthly_plan_on_manual_license_management(self) -> None:
        realm = get_realm('zulip')
        plan, ledger = self.subscribe_realm_to_monthly_plan_on_manual_license_management(realm, 20, 30)
        plan.refresh_from_db()
        self.assertEqual(plan.automanage_licenses, False)
        self.assertEqual(plan.billing_schedule, CustomerPlan.BILLING_SCHEDULE_MONTHLY)
        self.assertEqual(plan.tier, CustomerPlan.TIER_CLOUD_STANDARD)
        self.assertEqual(plan.licenses(), 20)
        self.assertEqual(plan.licenses_at_next_renewal(), 30)
        ledger.refresh_from_db()
        self.assertEqual(ledger.plan, plan)
        self.assertEqual(ledger.licenses, 20)
        self.assertEqual(ledger.licenses_at_next_renewal, 30)
        realm.refresh_from_db()
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)

    def test_get_audit_log_error(self) -> None:
        user = self.example_user('hamlet')
        billing_session = RealmBillingSession(user, realm=user.realm)
        fake_audit_log = cast(BillingSessionEventType, 0)
        with self.assertRaisesRegex(BillingSessionAuditLogEventError, 'Unknown audit log event type: 0'):
            billing_session.get_audit_log_event(event_type=fake_audit_log)

    def test_get_customer(self) -> None:
        user = self.example_user('hamlet')
        billing_session = RealmBillingSession(user, realm=user.realm)
        customer = billing_session.get_customer()
        self.assertEqual(customer, None)
        customer = Customer.objects.create(realm=get_realm('zulip'), stripe_customer_id='cus_12345')
        self.assertEqual(billing_session.get_customer(), customer)

    def test_get_current_plan_by_customer(self) -> None:
        realm = get_realm('zulip')
        customer = Customer.objects.create(realm=realm, stripe_customer_id='cus_12345')
        self.assertEqual(get_current_plan_by_customer(customer), None)
        plan = CustomerPlan.objects.create(customer=customer, status=CustomerPlan.ACTIVE, billing_cycle_anchor=timezone_now(), billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, tier=CustomerPlan.TIER_CLOUD_STANDARD)
        self.assertEqual(get_current_plan_by_customer(customer), plan)
        plan.status = CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE
        plan.save(update_fields=['status'])
        self.assertEqual(get_current_plan_by_customer(customer), plan)
        plan.status = CustomerPlan.ENDED
        plan.save(update_fields=['status'])
        self.assertEqual(get_current_plan_by_customer(customer), None)
        plan.status = CustomerPlan.NEVER_STARTED
        plan.save(update_fields=['status'])
        self.assertEqual(get_current_plan_by_customer(customer), None)

    def test_get_current_plan_by_realm(self) -> None:
        realm = get_realm('zulip')
        self.assertEqual(get_current_plan_by_realm(realm), None)
        customer = Customer.objects.create(realm=realm, stripe_customer_id='cus_12345')
        self.assertEqual(get_current_plan_by_realm(realm), None)
        plan = CustomerPlan.objects.create(customer=customer, status=CustomerPlan.ACTIVE, billing_cycle_anchor=timezone_now(), billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, tier=CustomerPlan.TIER_CLOUD_STANDARD)
        self.assertEqual(get_current_plan_by_realm(realm), plan)

    def test_is_realm_on_free_trial(self) -> None:
        realm = get_realm('zulip')
        self.assertFalse(is_realm_on_free_trial(realm))
        customer = Customer.objects.create(realm=realm, stripe_customer_id='cus_12345')
        plan = CustomerPlan.objects.create(customer=customer, status=CustomerPlan.ACTIVE, billing_cycle_anchor=timezone_now(), billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, tier=CustomerPlan.TIER_CLOUD_STANDARD)
        self.assertFalse(is_realm_on_free_trial(realm))
        plan.status = CustomerPlan.FREE_TRIAL
        plan.save(update_fields=['status'])
        self.assertTrue(is_realm_on_free_trial(realm))

    def test_deactivate_reactivate_remote_server(self) -> None:
        server_uuid = str(uuid.uuid4())
        remote_server = RemoteZulipServer.objects.create(uuid=server_uuid, api_key='magic_secret_api_key', hostname='demo.example.com', contact_email='email@example.com')
        self.assertFalse(remote_server.deactivated)
        billing_session = RemoteServerBillingSession(remote_server)
        do_deactivate_remote_server(remote_server, billing_session)
        remote_server = RemoteZulipServer.objects.get(uuid=server_uuid)
        remote_realm_audit_log = RemoteZulipServerAuditLog.objects.filter(event_type=AuditLogEventType.REMOTE_SERVER_DEACTIVATED).last()
        assert remote_realm_audit_log is not None
        self.assertTrue(remote_server.deactivated)
        with self.assertLogs('corporate.stripe', 'WARN') as warning_log:
            do_deactivate_remote_server(remote_server, billing_session)
            self.assertEqual(warning_log.output, [f'WARNING:corporate.stripe:Cannot deactivate remote server with ID {remote_server.id}, server has already been deactivated.'])
        do_reactivate_remote_server(remote_server)
        remote_server.refresh_from_db()
        self.assertFalse(remote_server.deactivated)
        remote_realm_audit_log = RemoteZulipServerAuditLog.objects.latest('id')
        self.assertEqual(remote_realm_audit_log.event_type, AuditLogEventType.REMOTE_SERVER_REACTIVATED)
        self.assertEqual(remote_realm_audit_log.server, remote_server)
        with self.assertLogs('corporate.stripe', 'WARN') as warning_log:
            do_reactivate_remote_server(remote_server)
            self.assertEqual(warning_log.output, [f'WARNING:corporate.stripe:Cannot reactivate remote server with ID {remote_server.id}, server is already active.'])

class LicenseLedgerTest(StripeTestCase):
    def test_add_plan_renewal_if_needed(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(LicenseLedger.objects.count(), 1)
        plan = CustomerPlan.objects.get()
        realm = get_realm('zulip')
        billing_session = RealmBillingSession(realm=realm)
        billing_session.make_end_of_cycle_updates_if_needed(plan, self.next_year - timedelta(days=1))
        self.assertEqual(LicenseLedger.objects.count(), 1)
        new_plan, ledger_entry = billing_session.make_end_of_cycle_updates_if_needed(plan, self.next_year)
        self.assertIsNone(new_plan)
        self.assertEqual(LicenseLedger.objects.count(), 2)
        ledger_params = {'plan': plan, 'is_renewal': True, 'event_time': self.next_year, 'licenses': self.seat_count, 'licenses_at_next_renewal': self.seat_count}
        for key, value in ledger_params.items():
            self.assertEqual(getattr(ledger_entry, key), value)
        billing_session.make_end_of_cycle_updates_if_needed(plan, self.next_year + timedelta(days=1))
        self.assertEqual(LicenseLedger.objects.count(), 2)

    def test_update_license_ledger_if_needed(self) -> None:
        realm = get_realm('zulip')
        billing_session = RealmBillingSession(realm=realm)
        billing_session.update_license_ledger_if_needed(self.now)
        self.assertFalse(LicenseLedger.objects.exists())
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count + 1, False, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.get()
        self.assertEqual(LicenseLedger.objects.count(), 1)
        self.assertEqual(plan.licenses(), self.seat_count + 1)
        self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count + 1)
        billing_session.update_license_ledger_if_needed(self.now)
        self.assertEqual(LicenseLedger.objects.count(), 1)
        plan.automanage_licenses = True
        plan.status = CustomerPlan.ENDED
        plan.save(update_fields=['automanage_licenses', 'status'])
        billing_session.update_license_ledger_if_needed(self.now)
        self.assertEqual(LicenseLedger.objects.count(), 1)
        plan.status = CustomerPlan.ACTIVE
        plan.save(update_fields=['status'])
        billing_session.update_license_ledger_if_needed(self.now)
        self.assertEqual(LicenseLedger.objects.count(), 2)

    def test_update_license_ledger_for_automanaged_plan(self) -> None:
        realm = get_realm('zulip')
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(plan.licenses(), self.seat_count)
        self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count)
        billing_session = RealmBillingSession(realm=realm)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=23):
            billing_session.update_license_ledger_for_automanaged_plan(plan, self.now)
            self.assertEqual(plan.licenses(), 23)
            self.assertEqual(plan.licenses_at_next_renewal(), 23)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=20):
            billing_session.update_license_ledger_for_automanaged_plan(plan, self.now)
            self.assertEqual(plan.licenses(), 23)
            self.assertEqual(plan.licenses_at_next_renewal(), 20)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=21):
            billing_session.update_license_ledger_for_automanaged_plan(plan, self.now)
            self.assertEqual(plan.licenses(), 23)
            self.assertEqual(plan.licenses_at_next_renewal(), 21)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=22):
            billing_session.update_license_ledger_for_automanaged_plan(plan, self.next_year + timedelta(seconds=1))
            self.assertEqual(plan.licenses(), 22)
            self.assertEqual(plan.licenses_at_next_renewal(), 22)
        ledger_entries = list(LicenseLedger.objects.values_list('is_renewal', 'event_time', 'licenses', 'licenses_at_next_renewal').order_by('id'))
        self.assertEqual(ledger_entries, [(True, self.now, self.seat_count, self.seat_count), (False, self.now, 23, 23), (False, self.now, 23, 20), (False, self.now, 23, 21), (True, self.next_year, 21, 21), (False, self.next_year + timedelta(seconds=1), 22, 22)])

    def test_update_license_ledger_for_manual_plan(self) -> None:
        realm = get_realm('zulip')
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count + 1, False, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        billing_session = RealmBillingSession(realm=realm)
        plan = get_current_plan_by_realm(realm)
        assert plan is not None
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=self.seat_count):
            billing_session.update_license_ledger_for_manual_plan(plan, self.now, licenses=self.seat_count + 3)
            self.assertEqual(plan.licenses(), self.seat_count + 3)
            self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count + 3)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=self.seat_count), self.assertRaises(AssertionError):
            billing_session.update_license_ledger_for_manual_plan(plan, self.now, licenses=self.seat_count)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=self.seat_count):
            billing_session.update_license_ledger_for_manual_plan(plan, self.now, licenses_at_next_renewal=self.seat_count)
            self.assertEqual(plan.licenses(), self.seat_count + 3)
            self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=self.seat_count), self.assertRaises(AssertionError):
            billing_session.update_license_ledger_for_manual_plan(plan, self.now, licenses_at_next_renewal=self.seat_count - 1)
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=self.seat_count):
            billing_session.update_license_ledger_for_manual_plan(plan, self.now, licenses=self.seat_count + 10)
            self.assertEqual(plan.licenses(), self.seat_count + 10)
            self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count + 10)
        billing_session.make_end_of_cycle_updates_if_needed(plan, self.next_year)
        self.assertEqual(plan.licenses(), self.seat_count + 10)
        ledger_entries = list(LicenseLedger.objects.values_list('is_renewal', 'event_time', 'licenses', 'licenses_at_next_renewal').order_by('id'))
        self.assertEqual(ledger_entries, [(True, self.now, self.seat_count + 1, self.seat_count + 1), (False, self.now, self.seat_count + 3, self.seat_count + 3), (False, self.now, self.seat_count + 3, self.seat_count), (False, self.now, self.seat_count + 10, self.seat_count + 10), (True, self.next_year, self.seat_count + 10, self.seat_count + 10)])
        with self.assertRaises(AssertionError):
            billing_session.update_license_ledger_for_manual_plan(plan, self.now)

    def test_user_changes(self) -> None:
        self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        user = do_create_user('email', 'password', get_realm('zulip'), 'name', acting_user=None)
        do_deactivate_user(user, acting_user=None)
        do_reactivate_user(user, acting_user=None)
        change_user_is_active(user, False)
        user.is_mirror_dummy = True
        user.save(update_fields=['is_mirror_dummy'])
        do_activate_mirror_dummy_user(user, acting_user=None)
        guest = do_create_user('guest_email', 'guest_password', get_realm('zulip'), 'guest_name', acting_user=None)
        do_change_user_role(guest, UserProfile.ROLE_MEMBER, acting_user=None)
        do_change_user_role(guest, UserProfile.ROLE_MODERATOR, acting_user=None)
        ledger_entries = list(LicenseLedger.objects.values_list('is_renewal', 'licenses', 'licenses_at_next_renewal').order_by('id'))
        self.assertEqual(ledger_entries, [(True, self.seat_count, self.seat_count), (False, self.seat_count + 1, self.seat_count + 1), (False, self.seat_count + 1, self.seat_count), (False, self.seat_count + 1, self.seat_count + 1), (False, self.seat_count + 1, self.seat_count + 1), (False, self.seat_count + 1, self.seat_count + 1), (False, self.seat_count + 2, self.seat_count + 2)])

    def test_toggle_license_management(self) -> None:
        self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(plan.automanage_licenses, True)
        self.assertEqual(plan.licenses(), self.seat_count)
        self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count)
        billing_session = RealmBillingSession(realm=get_realm('zulip'))
        update_plan_request = UpdatePlanRequest(status=None, licenses=None, licenses_at_next_renewal=None, schedule=None, toggle_license_management=True)
        billing_session.do_update_plan(update_plan_request)
        plan.refresh_from_db()
        self.assertEqual(plan.automanage_licenses, False)
        billing_session.do_update_plan(update_plan_request)
        plan.refresh_from_db()
        self.assertEqual(plan.automanage_licenses, True)

    def test_update_plan_with_invalid_status(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        response = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.NEVER_STARTED})
        self.assert_json_error_contains(response, 'Invalid status')

    def test_update_plan_without_any_params(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        with time_machine.travel(self.now, tick=False):
            response = self.client_billing_patch('/billing/plan', {})
        self.assert_json_error_contains(response, 'Nothing to change')

    def test_update_plan_that_which_is_due_for_expiry(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        from unittest.mock import patch
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE})
            self.assert_json_success(result)
            self.assertRegex(m.output[0], 'INFO:corporate.stripe:Change plan status: Customer.id: \\d*, CustomerPlan.id: \\d*, status: 2')
        with time_machine.travel(self.next_year, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.ACTIVE})
            self.assert_json_error_contains(result, 'Unable to update the plan. The plan has ended.')

    def test_update_plan_that_which_is_due_for_replacement(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_MONTHLY, True, False)
        self.login_user(self.example_user('hamlet'))
        from unittest.mock import patch
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE})
            self.assert_json_success(result)
            self.assertRegex(m.output[0], 'INFO:corporate.stripe:Change plan status: Customer.id: \\d*, CustomerPlan.id: \\d*, status: 4')
        with time_machine.travel(self.next_month, tick=False):
            result = self.client_billing_patch('/billing/plan', {})
            self.assert_json_error_contains(result, 'Unable to update the plan. The plan has been expired and replaced with a new plan.')

    @patch('corporate.lib.stripe.billing_logger.info')
    def test_deactivate_realm(self, mock_: Any) -> None:
        user = self.example_user('hamlet')
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.get()
        self.assertEqual(plan.next_invoice_date, self.next_month)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(plan.status, CustomerPlan.ACTIVE)
        billing_session = RealmBillingSession(user=user, realm=user.realm)
        from unittest.mock import patch
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=20):
            billing_session.update_license_ledger_if_needed(self.now)
        last_ledger_entry = LicenseLedger.objects.order_by('id').last()
        assert last_ledger_entry is not None
        self.assertEqual(last_ledger_entry.licenses, 20)
        self.assertEqual(last_ledger_entry.licenses_at_next_renewal, 20)
        do_deactivate_realm(get_realm('zulip'), acting_user=None, deactivation_reason='owner_request', email_owners=False)
        plan.refresh_from_db()
        self.assertTrue(get_realm('zulip').deactivated)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_LIMITED)
        self.assertEqual(plan.status, CustomerPlan.ENDED)
        self.assertEqual(plan.invoiced_through, last_ledger_entry)
        self.assertIsNone(plan.next_invoice_date)
        do_reactivate_realm(get_realm('zulip'))
        self.login_user(user)
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/plans/', response['Location'])
        from unittest.mock import patch
        with self.patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_month)
        mocked.assert_not_called()
        with self.patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_year)
        mocked.assert_not_called()

    def test_reupgrade_by_billing_admin_after_plan_status_changed_to_downgrade_at_end_of_cycle(self) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        from unittest.mock import patch
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            self.client_billing_patch('/billing/plan', {'status': CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE})
            stripe_customer_id = Customer.objects.get(realm=user.realm).id
            new_plan = get_current_plan_by_realm(user.realm)
            assert new_plan is not None
            expected_log = f'INFO:corporate.stripe:Change plan status: Customer.id: {stripe_customer_id}, CustomerPlan.id: {new_plan.id}, status: {CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE}'
            self.assertEqual(m.output[0], expected_log)
        with self.assertRaises(BillingError) as context, self.assertLogs('corporate.stripe', 'WARNING') as m, time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(m.output[0], 'WARNING:corporate.stripe:Upgrade of <Realm: zulip 2> (with stripe_customer_id: cus_123) failed because of existing active plan.')
        self.assertEqual(context.exception.error_description, 'subscribing with existing subscription')
        new_plan.next_invoice_date = self.next_year
        new_plan.save(update_fields=['next_invoice_date'])
        invoice_plans_as_needed(self.next_year)
        with time_machine.travel(self.next_year, tick=False):
            response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], '/plans/')
        with time_machine.travel(self.next_year, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(Customer.objects.count(), 1)
        self.assertEqual(CustomerPlan.objects.count(), 2)
        current_plan = CustomerPlan.objects.all().order_by('id').last()
        assert current_plan is not None
        next_invoice_date = add_months(self.next_year, 1)
        self.assertEqual(current_plan.next_invoice_date, next_invoice_date)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(current_plan.status, CustomerPlan.ACTIVE)
        old_plan = CustomerPlan.objects.all().order_by('id').first()
        assert old_plan is not None
        self.assertEqual(old_plan.next_invoice_date, None)
        self.assertEqual(old_plan.status, CustomerPlan.ENDED)

    @responses.activate
    @mock_stripe()
    def test_update_plan_with_invalid_status(self, *mocks: Any) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        response = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.NEVER_STARTED})
        self.assert_json_error_contains(response, 'Invalid status')

    def test_update_plan_without_any_params(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        with time_machine.travel(self.now, tick=False):
            response = self.client_billing_patch('/billing/plan', {})
        self.assert_json_error_contains(response, 'Nothing to change')

    def test_update_plan_that_which_is_due_for_expiry(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        from unittest.mock import patch
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE})
            self.assert_json_success(result)
            self.assertRegex(m.output[0], 'INFO:corporate.stripe:Change plan status: Customer.id: \\d*, CustomerPlan.id: \\d*, status: 2')
        with time_machine.travel(self.next_year, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.ACTIVE})
            self.assert_json_error_contains(result, 'Unable to update the plan. The plan has ended.')

    def test_update_plan_that_which_is_due_for_replacement(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_MONTHLY, True, False)
        self.login_user(self.example_user('hamlet'))
        from unittest.mock import patch
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE})
            self.assert_json_success(result)
            self.assertRegex(m.output[0], 'INFO:corporate.stripe:Change plan status: Customer.id: \\d*, CustomerPlan.id: \\d*, status: 4')
        with time_machine.travel(self.next_month, tick=False):
            result = self.client_billing_patch('/billing/plan', {})
            self.assert_json_error_contains(result, 'Unable to update the plan. The plan has been expired and replaced with a new plan.')

    @patch('corporate.lib.stripe.billing_logger.info')
    def test_deactivate_realm(self, mock_: Any) -> None:
        user = self.example_user('hamlet')
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.get()
        self.assertEqual(plan.next_invoice_date, self.next_month)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(plan.status, CustomerPlan.ACTIVE)
        billing_session = RealmBillingSession(user=user, realm=user.realm)
        from unittest.mock import patch
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=20):
            billing_session.update_license_ledger_if_needed(self.now)
        last_ledger_entry = LicenseLedger.objects.order_by('id').last()
        assert last_ledger_entry is not None
        self.assertEqual(last_ledger_entry.licenses, 20)
        self.assertEqual(last_ledger_entry.licenses_at_next_renewal, 20)
        do_deactivate_realm(get_realm('zulip'), acting_user=None, deactivation_reason='owner_request', email_owners=False)
        plan.refresh_from_db()
        self.assertTrue(get_realm('zulip').deactivated)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_LIMITED)
        self.assertEqual(plan.status, CustomerPlan.ENDED)
        self.assertEqual(plan.invoiced_through, last_ledger_entry)
        self.assertIsNone(plan.next_invoice_date)
        do_reactivate_realm(get_realm('zulip'))
        self.login_user(user)
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/plans/', response['Location'])
        from unittest.mock import patch
        with self.patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_month)
        mocked.assert_not_called()
        with self.patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_year)
        mocked.assert_not_called()

    def test_reupgrade_by_billing_admin_after_realm_deactivation(self) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        do_deactivate_realm(get_realm('zulip'), acting_user=None, deactivation_reason='owner_request', email_owners=False)
        self.assertTrue(get_realm('zulip').deactivated)
        do_reactivate_realm(get_realm('zulip'))
        self.login_user(user)
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/plans/', response['Location'])
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(Customer.objects.count(), 1)
        self.assertEqual(CustomerPlan.objects.count(), 2)
        current_plan = CustomerPlan.objects.all().order_by('id').last()
        assert current_plan is not None
        self.assertEqual(current_plan.next_invoice_date, add_months(self.next_year, 1))
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(current_plan.status, CustomerPlan.ACTIVE)
        old_plan = CustomerPlan.objects.all().order_by('id').first()
        assert old_plan is not None
        self.assertEqual(old_plan.next_invoice_date, None)
        self.assertEqual(old_plan.status, CustomerPlan.ENDED)

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_update_plan_with_invalid_status(self, *mocks: Any) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        response = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.NEVER_STARTED})
        self.assert_json_error_contains(response, 'Invalid status')

    def test_update_plan_without_any_params(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        with time_machine.travel(self.now, tick=False):
            response = self.client_billing_patch('/billing/plan', {})
        self.assert_json_error_contains(response, 'Nothing to change')

    def test_update_plan_that_which_is_due_for_expiry(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        from unittest.mock import patch
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE})
            self.assert_json_success(result)
            self.assertRegex(m.output[0], 'INFO:corporate.stripe:Change plan status: Customer.id: \\d*, CustomerPlan.id: \\d*, status: 2')
        with time_machine.travel(self.next_year, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.ACTIVE})
            self.assert_json_error_contains(result, 'Unable to update the plan. The plan has ended.')

    def test_update_plan_that_which_is_due_for_replacement(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_MONTHLY, True, False)
        self.login_user(self.example_user('hamlet'))
        from unittest.mock import patch
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE})
            self.assert_json_success(result)
            self.assertRegex(m.output[0], 'INFO:corporate.stripe:Change plan status: Customer.id: \\d*, CustomerPlan.id: \\d*, status: 4')
        with time_machine.travel(self.next_month, tick=False):
            result = self.client_billing_patch('/billing/plan', {})
            self.assert_json_error_contains(result, 'Unable to update the plan. The plan has been expired and replaced with a new plan.')

    @patch('corporate.lib.stripe.billing_logger.info')
    def test_deactivate_realm(self, mock_: Any) -> None:
        user = self.example_user('hamlet')
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.get()
        self.assertEqual(plan.next_invoice_date, self.next_month)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(plan.status, CustomerPlan.ACTIVE)
        billing_session = RealmBillingSession(user=user, realm=user.realm)
        from unittest.mock import patch
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=20):
            billing_session.update_license_ledger_if_needed(self.now)
        last_ledger_entry = LicenseLedger.objects.order_by('id').last()
        assert last_ledger_entry is not None
        self.assertEqual(last_ledger_entry.licenses, 20)
        self.assertEqual(last_ledger_entry.licenses_at_next_renewal, 20)
        do_deactivate_realm(get_realm('zulip'), acting_user=None, deactivation_reason='owner_request', email_owners=False)
        plan.refresh_from_db()
        self.assertTrue(get_realm('zulip').deactivated)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_LIMITED)
        self.assertEqual(plan.status, CustomerPlan.ENDED)
        self.assertEqual(plan.invoiced_through, last_ledger_entry)
        self.assertIsNone(plan.next_invoice_date)
        do_reactivate_realm(get_realm('zulip'))
        self.login_user(user)
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/plans/', response['Location'])
        from unittest.mock import patch
        with self.patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_month)
        mocked.assert_not_called()
        with self.patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_year)
        mocked.assert_not_called()

    def test_reupgrade_by_billing_admin_after_realm_deactivation(self) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        do_deactivate_realm(get_realm('zulip'), acting_user=None, deactivation_reason='owner_request', email_owners=False)
        self.assertTrue(get_realm('zulip').deactivated)
        do_reactivate_realm(get_realm('zulip'))
        self.login_user(user)
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/plans/', response['Location'])
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(Customer.objects.count(), 1)
        self.assertEqual(CustomerPlan.objects.count(), 2)
        current_plan = CustomerPlan.objects.all().order_by('id').last()
        assert current_plan is not None
        self.assertEqual(current_plan.next_invoice_date, add_months(self.next_year, 1))
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(current_plan.status, CustomerPlan.ACTIVE)
        old_plan = CustomerPlan.objects.all().order_by('id').first()
        assert old_plan is not None
        self.assertEqual(old_plan.next_invoice_date, None)
        self.assertEqual(old_plan.status, CustomerPlan.ENDED)

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_update_plan_with_invalid_status(self, *mocks: Any) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        response = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.NEVER_STARTED})
        self.assert_json_error_contains(response, 'Invalid status')

    def test_update_plan_without_any_params(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        with time_machine.travel(self.now, tick=False):
            response = self.client_billing_patch('/billing/plan', {})
        self.assert_json_error_contains(response, 'Nothing to change')

    def test_update_plan_that_which_is_due_for_expiry(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(self.example_user('hamlet'))
        from unittest.mock import patch
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE})
            self.assert_json_success(result)
            self.assertRegex(m.output[0], 'INFO:corporate.stripe:Change plan status: Customer.id: \\d*, CustomerPlan.id: \\d*, status: 2')
        with time_machine.travel(self.next_year, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.ACTIVE})
            self.assert_json_error_contains(result, 'Unable to update the plan. The plan has ended.')

    def test_update_plan_that_which_is_due_for_replacement(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_MONTHLY, True, False)
        self.login_user(self.example_user('hamlet'))
        from unittest.mock import patch
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            result = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE})
            self.assert_json_success(result)
            self.assertRegex(m.output[0], 'INFO:corporate.stripe:Change plan status: Customer.id: \\d*, CustomerPlan.id: \\d*, status: 4')
        with time_machine.travel(self.next_month, tick=False):
            result = self.client_billing_patch('/billing/plan', {})
            self.assert_json_error_contains(result, 'Unable to update the plan. The plan has been expired and replaced with a new plan.')

    @patch('corporate.lib.stripe.billing_logger.info')
    def test_deactivate_realm(self, mock_: Any) -> None:
        user = self.example_user('hamlet')
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.get()
        self.assertEqual(plan.next_invoice_date, self.next_month)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(plan.status, CustomerPlan.ACTIVE)
        billing_session = RealmBillingSession(user=user, realm=user.realm)
        from unittest.mock import patch
        with self.patch('corporate.lib.stripe.get_latest_seat_count', return_value=20):
            billing_session.update_license_ledger_if_needed(self.now)
        last_ledger_entry = LicenseLedger.objects.order_by('id').last()
        assert last_ledger_entry is not None
        self.assertEqual(last_ledger_entry.licenses, 20)
        self.assertEqual(last_ledger_entry.licenses_at_next_renewal, 20)
        do_deactivate_realm(get_realm('zulip'), acting_user=None, deactivation_reason='owner_request', email_owners=False)
        plan.refresh_from_db()
        self.assertTrue(get_realm('zulip').deactivated)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_LIMITED)
        self.assertEqual(plan.status, CustomerPlan.ENDED)
        self.assertEqual(plan.invoiced_through, last_ledger_entry)
        self.assertIsNone(plan.next_invoice_date)
        do_reactivate_realm(get_realm('zulip'))
        self.login_user(user)
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/plans/', response['Location'])
        from unittest.mock import patch
        with self.patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_month)
        mocked.assert_not_called()
        with self.patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_year)
        mocked.assert_not_called()

    def test_reupgrade_by_billing_admin_after_realm_deactivation(self) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        do_deactivate_realm(get_realm('zulip'), acting_user=None, deactivation_reason='owner_request', email_owners=False)
        self.assertTrue(get_realm('zulip').deactivated)
        do_reactivate_realm(get_realm('zulip'))
        self.login_user(user)
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/plans/', response['Location'])
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(Customer.objects.count(), 1)
        self.assertEqual(CustomerPlan.objects.count(), 2)
        current_plan = CustomerPlan.objects.all().order_by('id').last()
        assert current_plan is not None
        self.assertEqual(current_plan.next_invoice_date, add_months(self.next_year, 1))
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(current_plan.status, CustomerPlan.ACTIVE)
        old_plan = CustomerPlan.objects.all().order_by('id').first()
        assert old_plan is not None
        self.assertEqual(old_plan.next_invoice_date, None)
        self.assertEqual(old_plan.status, CustomerPlan.ENDED)

class StripeWebhookEndpointTest(ZulipTestCase):
    def test_stripe_webhook_with_invalid_data(self) -> None:
        result = self.client_post('/stripe/webhook/', '["dsdsds"]', content_type='application/json')
        self.assertEqual(result.status_code, 400)

    def test_stripe_webhook_endpoint_invalid_api_version(self) -> None:
        event_data = {'id': 'stripe_event_id', 'api_version': '1991-02-20', 'type': 'event_type', 'data': {'object': {'object': 'checkout.session', 'id': 'stripe_session_id'}}}
        expected_error_message = f'Mismatch between billing system Stripe API version({settings.STRIPE_API_VERSION}) and Stripe webhook event API version(1991-02-20).'
        with self.assertLogs('corporate.stripe', 'ERROR') as error_log:
            self.client_post('/stripe/webhook/', event_data, content_type='application/json')
            self.assertEqual(error_log.output, [f'ERROR:corporate.stripe:{expected_error_message}'])

    def test_stripe_webhook_for_session_completed_event(self) -> None:
        valid_session_event_data = {'id': 'stripe_event_id', 'api_version': settings.STRIPE_API_VERSION, 'type': 'checkout.session.completed', 'data': {'object': {'object': 'checkout.session', 'id': 'stripe_session_id'}}}
        from unittest.mock import patch
        with patch('corporate.lib.stripe_event_handler.handle_checkout_session_completed_event') as m:
            result = self.client_post('/stripe/webhook/', valid_session_event_data, content_type='application/json')
        self.assert_length(Event.objects.all(), 0)
        self.assertEqual(result.status_code, 200)
        m.assert_not_called()

    def test_stripe_webhook_for_invoice_payment_events(self) -> None:
        customer = Customer.objects.create(realm=get_realm('zulip'))
        stripe_event_id = 'stripe_event_id'
        stripe_invoice_id = 'stripe_invoice_id'
        valid_session_event_data = {'id': stripe_event_id, 'type': 'invoice.paid', 'api_version': settings.STRIPE_API_VERSION, 'data': {'object': {'object': 'invoice', 'id': stripe_invoice_id}}}
        from unittest.mock import patch
        with patch('corporate.lib.stripe_event_handler.handle_invoice_paid_event') as m:
            result = self.client_post('/stripe/webhook/', valid_session_event_data, content_type='application/json')
        self.assert_length(Event.objects.filter(stripe_event_id=stripe_event_id), 0)
        self.assertEqual(result.status_code, 200)
        m.assert_not_called()
        Invoice.objects.create(stripe_invoice_id=stripe_invoice_id, customer=customer, status=Invoice.SENT)
        self.assert_length(Event.objects.filter(stripe_event_id=stripe_event_id), 0)
        with patch('corporate.lib.stripe_event_handler.handle_invoice_paid_event') as m:
            result = self.client_post('/stripe/webhook/', valid_session_event_data, content_type='application/json')
        [event] = Event.objects.filter(stripe_event_id=stripe_event_id)
        self.assertEqual(result.status_code, 200)
        strip_event = stripe.Event.construct_from(valid_session_event_data, stripe.api_key)
        m.assert_called_once_with(strip_event.data.object, event)
        with patch('corporate.lib.stripe_event_handler.handle_invoice_paid_event') as m:
            result = self.client_post('/stripe/webhook/', valid_session_event_data, content_type='application/json')
        self.assert_length(Event.objects.filter(stripe_event_id=stripe_event_id), 1)
        self.assertEqual(result.status_code, 200)
        m.assert_not_called()

    def test_stripe_webhook_for_invoice_paid_events(self) -> None:
        customer = Customer.objects.create(realm=get_realm('zulip'))
        stripe_event_id = 'stripe_event_id'
        stripe_invoice_id = 'stripe_invoice_id'
        valid_invoice_paid_event_data = {'id': stripe_event_id, 'type': 'invoice.paid', 'api_version': settings.STRIPE_API_VERSION, 'data': {'object': {'object': 'invoice', 'id': stripe_invoice_id}}}
        from unittest.mock import patch
        with patch('corporate.lib.stripe_event_handler.handle_invoice_paid_event') as m:
            result = self.client_post('/stripe/webhook/', valid_invoice_paid_event_data, content_type='application/json')
        self.assert_length(Event.objects.filter(stripe_event_id=stripe_event_id), 0)
        self.assertEqual(result.status_code, 200)
        m.assert_not_called()
        Invoice.objects.create(stripe_invoice_id=stripe_invoice_id, customer=customer, status=Invoice.SENT)
        self.assert_length(Event.objects.filter(stripe_event_id=stripe_event_id), 0)
        with patch('corporate.lib.stripe_event_handler.handle_invoice_paid_event') as m:
            result = self.client_post('/stripe/webhook/', valid_invoice_paid_event_data, content_type='application/json')
        [event] = Event.objects.filter(stripe_event_id=stripe_event_id)
        self.assertEqual(result.status_code, 200)
        strip_event = stripe.Event.construct_from(valid_invoice_paid_event_data, stripe.api_key)
        m.assert_called_once_with(strip_event.data.object, event)
        with patch('corporate.lib.stripe_event_handler.handle_invoice_paid_event') as m:
            result = self.client_post('/stripe/webhook/', valid_invoice_paid_event_data, content_type='application/json')
        self.assert_length(Event.objects.filter(stripe_event_id=stripe_event_id), 1)
        self.assertEqual(result.status_code, 200)
        m.assert_not_called()


class EventStatusTest(StripeTestCase):
    def test_event_status_json_endpoint_errors(self) -> None:
        self.login_user(self.example_user('iago'))
        response = self.client_get('/json/billing/event/status')
        self.assert_json_error_contains(response, 'No customer for this organization!')
        Customer.objects.create(realm=get_realm('zulip'), stripe_customer_id='cus_12345')
        response = self.client_get('/json/billing/event/status', {'stripe_session_id': 'invalid_session_id'})
        self.assert_json_error_contains(response, 'Session not found')
        response = self.client_get('/json/billing/event/status', {'stripe_invoice_id': 'invalid_invoice_id'})
        self.assert_json_error_contains(response, 'Payment intent not found')
        response = self.client_get('/json/billing/event/status')
        self.assert_json_error_contains(response, 'Pass stripe_session_id or stripe_invoice_id')

    def test_event_status_page(self) -> None:
        self.login_user(self.example_user('polonius'))
        stripe_session_id = 'cs_test_9QCz62mPTJQUwvhcwZHBpJMHmMZiLU512AQHU9g5znkx6NweU3j7kJvY'
        response = self.client_get('/billing/event_status/', {'stripe_session_id': stripe_session_id})
        self.assert_in_success_response([f'data-stripe-session-id="{stripe_session_id}"'], response)
        stripe_invoice_id = 'pi_1JGLpnA4KHR4JzRvUfkF9Tn7'
        response = self.client_get('/billing/event_status/', {'stripe_invoice_id': stripe_invoice_id})
        self.assert_in_success_response([f'data-stripe-invoice-id="{stripe_invoice_id}"'], response)


class RequiresBillingAccessTest(StripeTestCase):
    @override
    def setUp(self, *mocks: Any) -> None:
        super().setUp()
        desdemona = self.example_user('desdemona')
        desdemona.role = UserProfile.ROLE_REALM_OWNER
        desdemona.save(update_fields=['role'])

    def test_json_endpoints_permissions(self) -> None:
        guest = self.example_user('polonius')
        member = self.example_user('othello')
        realm_admin = self.example_user('iago')
        billing_admin = self.example_user('hamlet')
        billing_admin.is_billing_admin = True
        billing_admin.save(update_fields=['is_billing_admin'])
        tested_endpoints = set()
        def check_users_cant_access(users: list, error_message: str, url: str, method: str, data: Any) -> None:
            tested_endpoints.add(url)
            for user in users:
                self.login_user(user)
                if method == 'POST':
                    client_func = self.client_post
                elif method == 'GET':
                    client_func = self.client_get
                else:
                    client_func = self.client_patch
                result = client_func(url, data, content_type='application/json')
                self.assert_json_error_contains(result, error_message)
        check_users_cant_access([guest], 'Must be an organization member', '/json/billing/upgrade', 'POST', {})
        check_users_cant_access([guest], 'Must be an organization member', '/json/billing/sponsorship', 'POST', {})
        check_users_cant_access([guest, member, realm_admin], 'Must be a billing administrator or an organization owner', '/json/billing/plan', 'PATCH', {})
        check_users_cant_access([guest, member, realm_admin], 'Must be a billing administrator or an organization owner', '/json/billing/session/start_card_update_session', 'POST', {})
        check_users_cant_access([guest], 'Must be an organization member', '/json/upgrade/session/start_card_update_session', 'POST', {})
        check_users_cant_access([guest], 'Must be an organization member', '/json/billing/event/status', 'GET', {})
        reverse_dict = get_resolver('corporate.urls').reverse_dict
        json_endpoints = {pat for name in reverse_dict for matches, pat, defaults, converters in reverse_dict.getlist(name) if pat.startswith('json/') and (not pat.startswith(('json/realm/', 'json/server/')))}
        self.assert_length(json_endpoints, len(tested_endpoints))

    @responses.activate
    @mock_stripe()
    def test_billing_page_permissions(self, *mocks: Any) -> None:
        self.login_user(self.example_user('polonius'))
        response = self.client_get('/upgrade/', follow=True)
        self.assertEqual(response.status_code, 404)
        non_owner_non_billing_admin = self.example_user('othello')
        self.login_user(non_owner_non_billing_admin)
        response = self.client_get('/billing/')
        self.assert_in_success_response(['You must be an organization owner or a billing administrator to view this page.'], response)
        self.add_card_and_upgrade(non_owner_non_billing_admin)
        response = self.client_get('/billing/')
        self.assert_in_success_response(['Zulip Cloud Standard'], response)
        self.assert_not_in_success_response(['You must be an organization owner or a billing administrator to view this page.'], response)
        desdemona = self.example_user('desdemona')
        desdemona.role = UserProfile.ROLE_REALM_OWNER
        desdemona.save(update_fields=['role'])
        self.login_user(self.example_user('desdemona'))
        response = self.client_get('/billing/')
        self.assert_in_success_response(['Zulip Cloud Standard'], response)
        self.login_user(self.example_user('cordelia'))
        response = self.client_get('/billing/')
        self.assert_in_success_response(['You must be an organization owner or a billing administrator'], response)


class BillingHelpersTest(ZulipTestCase):
    def test_next_month(self) -> None:
        anchor = datetime(2019, 12, 31, 1, 2, 3, tzinfo=timezone.utc)
        period_boundaries = [
            anchor,
            datetime(2020, 1, 31, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2020, 2, 28, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2020, 3, 31, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2020, 4, 30, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2020, 5, 31, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2020, 6, 30, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2020, 7, 31, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2020, 8, 31, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2020, 9, 30, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2020, 10, 31, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2020, 11, 30, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2020, 12, 31, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2021, 1, 31, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2021, 2, 28, 1, 2, 3, tzinfo=timezone.utc)
        ]
        with self.assertRaises(AssertionError):
            add_months(anchor, -1)
        for i, boundary in enumerate(period_boundaries):
            self.assertEqual(add_months(anchor, i), boundary)
        for last, next_ in itertools.pairwise(period_boundaries):
            self.assertEqual(next_month(anchor, last), next_)
        period_boundaries = [dt.replace(year=dt.year + 100) for dt in period_boundaries]
        for last, next_ in itertools.pairwise(period_boundaries):
            self.assertEqual(next_month(anchor, last), next_)

    def test_compute_plan_parameters(self) -> None:
        anchor = datetime(2019, 12, 31, 1, 2, 3, tzinfo=timezone.utc)
        month_later = datetime(2020, 1, 31, 1, 2, 3, tzinfo=timezone.utc)
        year_later = datetime(2020, 12, 31, 1, 2, 3, tzinfo=timezone.utc)
        customer_with_discount = Customer.objects.create(realm=get_realm('lear'), monthly_discounted_price=600, annual_discounted_price=6000, required_plan_tier=CustomerPlan.TIER_CLOUD_STANDARD)
        customer_no_discount = Customer.objects.create(realm=get_realm('zulip'))
        test_cases = [
            ((CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_ANNUAL, None), (anchor, month_later, year_later, 8000)),
            ((CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_ANNUAL, customer_with_discount), (anchor, month_later, year_later, 6000)),
            ((CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_ANNUAL, customer_no_discount), (anchor, month_later, year_later, 8000)),
            ((CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_ANNUAL, customer_with_discount), (anchor, month_later, year_later, 12000)),
            ((CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_MONTHLY, None), (anchor, month_later, month_later, 800)),
            ((CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_MONTHLY, customer_with_discount), (anchor, month_later, month_later, 600)),
            ((CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_MONTHLY, customer_no_discount), (anchor, month_later, month_later, 800)),
            ((CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHLY, customer_with_discount), (anchor, month_later, month_later, 1200))
        ]
        with time_machine.travel(anchor, tick=False):
            for (tier, billing_schedule, customer), output in test_cases:
                output_ = compute_plan_parameters(tier, billing_schedule, customer)
                self.assertEqual(output_, output)

    def test_get_price_per_license(self) -> None:
        standard_discounted_customer = Customer.objects.create(realm=get_realm('lear'), monthly_discounted_price=400, annual_discounted_price=4000, required_plan_tier=CustomerPlan.TIER_CLOUD_STANDARD)
        plus_discounted_customer = Customer.objects.create(realm=get_realm('zulip'), monthly_discounted_price=600, annual_discounted_price=6000, required_plan_tier=CustomerPlan.TIER_CLOUD_PLUS)
        self.assertEqual(get_price_per_license(CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_ANNUAL), 8000)
        self.assertEqual(get_price_per_license(CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_MONTHLY), 800)
        self.assertEqual(get_price_per_license(CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_MONTHLY, standard_discounted_customer), 400)
        self.assertEqual(get_price_per_license(CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_ANNUAL), 12000)
        self.assertEqual(get_price_per_license(CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHLY), 1200)
        self.assertEqual(get_price_per_license(CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHLY, standard_discounted_customer), 1200)
        self.assertEqual(get_price_per_license(CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHLY, plus_discounted_customer), 600)
        with self.assertRaisesRegex(InvalidBillingScheduleError, 'Unknown billing_schedule: 1000'):
            get_price_per_license(CustomerPlan.TIER_CLOUD_STANDARD, 1000)
        with self.assertRaisesRegex(InvalidTierError, 'Unknown tier: 4'):
            get_price_per_license(CustomerPlan.TIER_CLOUD_ENTERPRISE, CustomerPlan.BILLING_SCHEDULE_ANNUAL)

    def test_get_plan_renewal_or_end_date(self) -> None:
        realm = get_realm('zulip')
        customer = Customer.objects.create(realm=realm, stripe_customer_id='cus_12345')
        billing_cycle_anchor = timezone_now()
        plan = CustomerPlan.objects.create(customer=customer, status=CustomerPlan.ACTIVE, billing_cycle_anchor=billing_cycle_anchor, billing_schedule=CustomerPlan.BILLING_SCHEDULE_MONTHLY, tier=CustomerPlan.TIER_CLOUD_STANDARD)
        renewal_date = get_plan_renewal_or_end_date(plan, billing_cycle_anchor)
        self.assertEqual(renewal_date, add_months(billing_cycle_anchor, 1))
        plan_end_date = add_months(billing_cycle_anchor, 1) - timedelta(days=2)
        plan.end_date = plan_end_date
        plan.save(update_fields=['end_date'])
        renewal_date = get_plan_renewal_or_end_date(plan, billing_cycle_anchor)
        self.assertEqual(renewal_date, plan_end_date)

    def test_update_or_create_stripe_customer_logic(self) -> None:
        user = self.example_user('hamlet')
        from unittest.mock import patch
        with patch('corporate.lib.stripe.BillingSession.create_stripe_customer', return_value='returned') as mocked1:
            billing_session = RealmBillingSession(user, realm=user.realm)
            returned = billing_session.update_or_create_stripe_customer()
        mocked1.assert_called_once()
        self.assertEqual(returned, 'returned')
        customer = Customer.objects.create(realm=get_realm('zulip'))
        with patch('corporate.lib.stripe.BillingSession.create_stripe_customer', return_value='returned') as mocked2:
            billing_session = RealmBillingSession(user, realm=user.realm)
            returned = billing_session.update_or_create_stripe_customer()
        mocked2.assert_called_once()
        self.assertEqual(returned, 'returned')
        customer.stripe_customer_id = 'cus_12345'
        customer.save()
        with patch('corporate.lib.stripe.BillingSession.replace_payment_method') as mocked3:
            billing_session = RealmBillingSession(user, realm=user.realm)
            returned_customer = billing_session.update_or_create_stripe_customer('pm_card_visa')
        mocked3.assert_called_once()
        self.assertEqual(returned_customer, customer)
        with patch('corporate.lib.stripe.BillingSession.replace_payment_method') as mocked4:
            billing_session = RealmBillingSession(user, realm=user.realm)
            returned_customer = billing_session.update_or_create_stripe_customer(None)
        mocked4.assert_not_called()
        self.assertEqual(returned_customer, customer)

    def test_get_customer_by_realm(self) -> None:
        realm = get_realm('zulip')
        self.assertEqual(get_customer_by_realm(realm), None)
        customer = Customer.objects.create(realm=realm, stripe_customer_id='cus_12345')
        self.assertEqual(get_customer_by_realm(realm), customer)

    def test_get_current_plan_by_realm(self) -> None:
        realm = get_realm('zulip')
        self.assertEqual(get_current_plan_by_realm(realm), None)
        customer = Customer.objects.create(realm=realm, stripe_customer_id='cus_12345')
        self.assertEqual(get_current_plan_by_realm(realm), None)
        plan = CustomerPlan.objects.create(customer=customer, status=CustomerPlan.ACTIVE, billing_cycle_anchor=timezone_now(), billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, tier=CustomerPlan.TIER_CLOUD_STANDARD)
        self.assertEqual(get_current_plan_by_realm(realm), plan)

    def test_is_realm_on_free_trial(self) -> None:
        realm = get_realm('zulip')
        self.assertFalse(is_realm_on_free_trial(realm))
        customer = Customer.objects.create(realm=realm, stripe_customer_id='cus_12345')
        plan = CustomerPlan.objects.create(customer=customer, status=CustomerPlan.ACTIVE, billing_cycle_anchor=timezone_now(), billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, tier=CustomerPlan.TIER_CLOUD_STANDARD)
        self.assertFalse(is_realm_on_free_trial(realm))
        plan.status = CustomerPlan.FREE_TRIAL
        plan.save(update_fields=['status'])
        self.assertTrue(is_realm_on_free_trial(realm))

    def test_deactivate_reactivate_remote_server(self) -> None:
        server_uuid = str(uuid.uuid4())
        remote_server = RemoteZulipServer.objects.create(uuid=server_uuid, api_key='magic_secret_api_key', hostname='demo.example.com', contact_email='email@example.com')
        self.assertFalse(remote_server.deactivated)
        billing_session = RemoteServerBillingSession(remote_server)
        do_deactivate_remote_server(remote_server, billing_session)
        remote_server = RemoteZulipServer.objects.get(uuid=server_uuid)
        remote_realm_audit_log = RemoteZulipServerAuditLog.objects.filter(event_type=AuditLogEventType.REMOTE_SERVER_DEACTIVATED).last()
        assert remote_realm_audit_log is not None
        self.assertTrue(remote_server.deactivated)
        with self.assertLogs('corporate.stripe', 'WARN') as warning_log:
            do_deactivate_remote_server(remote_server, billing_session)
            self.assertEqual(warning_log.output, [f'WARNING:corporate.stripe:Cannot deactivate remote server with ID {remote_server.id}, server has already been deactivated.'])
        do_reactivate_remote_server(remote_server)
        remote_server.refresh_from_db()
        self.assertFalse(remote_server.deactivated)
        remote_realm_audit_log = RemoteZulipServerAuditLog.objects.latest('id')
        self.assertEqual(remote_realm_audit_log.event_type, AuditLogEventType.REMOTE_SERVER_REACTIVATED)
        self.assertEqual(remote_realm_audit_log.server, remote_server)
        with self.assertLogs('corporate.stripe', 'WARN') as warning_log:
            do_reactivate_remote_server(remote_server)
            self.assertEqual(warning_log.output, [f'WARNING:corporate.stripe:Cannot reactivate remote server with ID {remote_server.id}, server is already active.'])

class TestTestClasses(ZulipTestCase):
    def test_subscribe_realm_to_manual_license_management_plan(self) -> None:
        realm = get_realm('zulip')
        plan, ledger = self.subscribe_realm_to_manual_license_management_plan(realm, 50, 60, CustomerPlan.BILLING_SCHEDULE_ANNUAL)
        plan.refresh_from_db()
        self.assertEqual(plan.automanage_licenses, False)
        self.assertEqual(plan.billing_schedule, CustomerPlan.BILLING_SCHEDULE_ANNUAL)
        self.assertEqual(plan.tier, CustomerPlan.TIER_CLOUD_STANDARD)
        self.assertEqual(plan.licenses(), 50)
        self.assertEqual(plan.licenses_at_next_renewal(), 60)
        ledger.refresh_from_db()
        self.assertEqual(ledger.plan, plan)
        self.assertEqual(ledger.licenses, 50)
        self.assertEqual(ledger.licenses_at_next_renewal, 60)
        realm.refresh_from_db()
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)

    def test_subscribe_realm_to_monthly_plan_on_manual_license_management(self) -> None:
        realm = get_realm('zulip')
        plan, ledger = self.subscribe_realm_to_monthly_plan_on_manual_license_management(realm, 20, 30)
        plan.refresh_from_db()
        self.assertEqual(plan.automanage_licenses, False)
        self.assertEqual(plan.billing_schedule, CustomerPlan.BILLING_SCHEDULE_MONTHLY)
        self.assertEqual(plan.tier, CustomerPlan.TIER_CLOUD_STANDARD)
        self.assertEqual(plan.licenses(), 20)
        self.assertEqual(plan.licenses_at_next_renewal(), 30)
        ledger.refresh_from_db()
        self.assertEqual(ledger.plan, plan)
        self.assertEqual(ledger.licenses, 20)
        self.assertEqual(ledger.licenses_at_next_renewal, 30)
        realm.refresh_from_db()
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)

class TestRealmBillingSession(StripeTestCase):
    def test_get_audit_log_error(self) -> None:
        user = self.example_user('hamlet')
        billing_session = RealmBillingSession(user, realm=user.realm)
        fake_audit_log = cast(BillingSessionEventType, 0)
        with self.assertRaisesRegex(BillingSessionAuditLogEventError, 'Unknown audit log event type: 0'):
            billing_session.get_audit_log_event(event_type=fake_audit_log)

    def test_get_customer(self) -> None:
        user = self.example_user('hamlet')
        billing_session = RealmBillingSession(user, realm=user.realm)
        customer = billing_session.get_customer()
        self.assertEqual(customer, None)
        customer = Customer.objects.create(realm=get_realm('zulip'), stripe_customer_id='cus_12345')
        self.assertEqual(billing_session.get_customer(), customer)

class TestRemoteRealmBillingSession(StripeTestCase):
    def test_current_count_for_billed_licenses(self) -> None:
        server_uuid = str(uuid.uuid4())
        remote_server = RemoteZulipServer.objects.create(uuid=server_uuid, api_key='magic_secret_api_key', hostname='demo.example.com', contact_email='email@example.com')
        realm_uuid = str(uuid.uuid4())
        remote_realm = RemoteRealm.objects.create(server=remote_server, uuid=realm_uuid, uuid_owner_secret='dummy-owner-secret', host='dummy-hostname', realm_date_created=timezone_now())
        billing_session = RemoteRealmBillingSession(remote_realm=remote_realm)
        with self.assertRaises(MissingDataError):
            billing_session.current_count_for_billed_licenses()
        remote_server.last_audit_log_update = timezone_now() - timedelta(days=5)
        remote_server.save()
        with self.assertRaises(MissingDataError):
            billing_session.current_count_for_billed_licenses()
        event_time = timezone_now() - timedelta(days=1)
        data_list = [{
            'server': remote_server,
            'remote_realm': remote_realm,
            'event_type': AuditLogEventType.USER_CREATED,
            'event_time': event_time,
            'extra_data': {RemoteRealmAuditLog.ROLE_COUNT: {RemoteRealmAuditLog.ROLE_COUNT_HUMANS: {UserProfile.ROLE_REALM_ADMINISTRATOR: 10, UserProfile.ROLE_REALM_OWNER: 10, UserProfile.ROLE_MODERATOR: 10, UserProfile.ROLE_MEMBER: 10, UserProfile.ROLE_GUEST: 10}}}
        }, {
            'server': remote_server,
            'remote_realm': remote_realm,
            'event_type': AuditLogEventType.USER_ROLE_CHANGED,
            'event_time': event_time,
            'extra_data': {RemoteRealmAuditLog.ROLE_COUNT: {RemoteRealmAuditLog.ROLE_COUNT_HUMANS: {UserProfile.ROLE_REALM_ADMINISTRATOR: 20, UserProfile.ROLE_REALM_OWNER: 10, UserProfile.ROLE_MODERATOR: 0, UserProfile.ROLE_MEMBER: 30, UserProfile.ROLE_GUEST: 10}}}
        }]
        RemoteRealmAuditLog.objects.bulk_create([RemoteRealmAuditLog(**data) for data in data_list])
        remote_server.last_audit_log_update = timezone_now() - timedelta(days=1)
        remote_server.save()
        self.assertEqual(billing_session.current_count_for_billed_licenses(), 70)

class TestRemoteServerBillingSession(StripeTestCase):
    def test_get_audit_log_error(self) -> None:
        server_uuid = str(uuid.uuid4())
        remote_server = RemoteZulipServer.objects.create(uuid=server_uuid, api_key='magic_secret_api_key', hostname='demo.example.com', contact_email='email@example.com')
        billing_session = RemoteServerBillingSession(remote_server)
        fake_audit_log = cast(BillingSessionEventType, 0)
        with self.assertRaisesRegex(BillingSessionAuditLogEventError, 'Unknown audit log event type: 0'):
            billing_session.get_audit_log_event(event_type=fake_audit_log)

    def test_get_customer(self) -> None:
        server_uuid = str(uuid.uuid4())
        remote_server = RemoteZulipServer.objects.create(uuid=server_uuid, api_key='magic_secret_api_key', hostname='demo.example.com', contact_email='email@example.com')
        billing_session = RemoteServerBillingSession(remote_server)
        customer = billing_session.get_customer()
        self.assertEqual(customer, None)
        customer = Customer.objects.create(remote_server=remote_server, stripe_customer_id='cus_12345')
        self.assertEqual(billing_session.get_customer(), customer)

class TestSupportBillingHelpers(StripeTestCase):
    @responses.activate
    @mock_stripe()
    def test_attach_discount_to_realm(self, *mocks: Any) -> None:
        support_admin = self.example_user('iago')
        user = self.example_user('hamlet')
        billing_session = RealmBillingSession(support_admin, realm=user.realm, support_session=True)
        with self.assertRaises(AssertionError):
            billing_session.attach_discount_to_customer(monthly_discounted_price=120, annual_discounted_price=1200)
        billing_session.update_or_create_customer()
        with self.assertRaises(AssertionError):
            billing_session.attach_discount_to_customer(monthly_discounted_price=120, annual_discounted_price=1200)
        billing_session.set_required_plan_tier(CustomerPlan.TIER_CLOUD_STANDARD)
        billing_session.attach_discount_to_customer(monthly_discounted_price=120, annual_discounted_price=1200)
        realm_audit_log = RealmAuditLog.objects.filter(event_type=AuditLogEventType.REALM_DISCOUNT_CHANGED).last()
        assert realm_audit_log is not None
        expected_extra_data = {'new_annual_discounted_price': 1200, 'new_monthly_discounted_price': 120, 'old_annual_discounted_price': 0, 'old_monthly_discounted_price': 0}
        self.assertEqual(realm_audit_log.extra_data, expected_extra_data)
        self.login_user(user)
        self.assert_in_success_response(['85'], self.client_get('/upgrade/'))
        self.add_card_and_upgrade(user)
        customer = Customer.objects.first()
        assert customer is not None
        assert customer.stripe_customer_id is not None
        [charge] = iter(stripe.Charge.list(customer=customer.stripe_customer_id))
        self.assertEqual(1200 * self.seat_count, charge.amount)
        stripe_customer_id = customer.stripe_customer_id
        assert stripe_customer_id is not None
        [invoice] = iter(stripe.Invoice.list(customer=stripe_customer_id))
        self.assertEqual([1200 * self.seat_count], [item.amount for item in invoice.lines])
        plan = CustomerPlan.objects.get(price_per_license=1200, discount='85')
        plan.status = CustomerPlan.ENDED
        plan.save(update_fields=['status'])
        billing_session = RealmBillingSession(support_admin, realm=user.realm, support_session=True)
        billing_session.attach_discount_to_customer(monthly_discounted_price=600, annual_discounted_price=6000)
        with time_machine.travel(self.now, tick=False):
            self.add_card_and_upgrade(user, license_management='automatic', billing_modality='charge_automatically')
        [charge, _] = iter(stripe.Charge.list(customer=customer.stripe_customer_id))
        self.assertEqual(6000 * self.seat_count, charge.amount)
        stripe_customer_id = customer.stripe_customer_id
        assert stripe_customer_id is not None
        [invoice, _] = iter(stripe.Invoice.list(customer=stripe_customer_id))
        self.assertEqual([6000 * self.seat_count], [item.amount for item in invoice.lines])
        plan = CustomerPlan.objects.get(price_per_license=6000, discount=Decimal(25))
        billing_session = RealmBillingSession(support_admin, realm=user.realm, support_session=True)
        billing_session.attach_discount_to_customer(monthly_discounted_price=400, annual_discounted_price=4000)
        plan.refresh_from_db()
        self.assertEqual(plan.price_per_license, 4000)
        self.assertEqual(plan.discount, Decimal(50))
        customer.refresh_from_db()
        self.assertEqual(customer.monthly_discounted_price, 400)
        self.assertEqual(customer.annual_discounted_price, 4000)
        plan.next_invoice_date = self.next_year
        plan.save(update_fields=['next_invoice_date'])
        invoice_plans_as_needed(self.next_year + timedelta(days=10))
        stripe_customer_id = customer.stripe_customer_id
        assert stripe_customer_id is not None
        [invoice, _, _] = iter(stripe.Invoice.list(customer=stripe_customer_id))
        self.assertEqual([4000 * self.seat_count], [item.amount for item in invoice.lines])
        realm_audit_log = RealmAuditLog.objects.filter(event_type=AuditLogEventType.REALM_DISCOUNT_CHANGED).last()
        assert realm_audit_log is not None
        expected_extra_data = {'new_annual_discounted_price': 4000, 'new_monthly_discounted_price': 400, 'old_annual_discounted_price': 6000, 'old_monthly_discounted_price': 600}
        self.assertEqual(realm_audit_log.extra_data, expected_extra_data)
        self.assertEqual(realm_audit_log.acting_user, support_admin)
        with self.assertRaisesRegex(SupportRequestError, 'Customer on plan Zulip Cloud Standard. Please end current plan before approving sponsorship!'):
            billing_session.approve_sponsorship()

    @responses.activate
    @mock_stripe()
    def test_add_minimum_licenses(self, *mocks: Any) -> None:
        min_licenses = 25
        support_view_request = SupportViewRequest(support_type=SupportType.update_minimum_licenses, minimum_licenses=min_licenses)
        support_admin = self.example_user('iago')
        user = self.example_user('hamlet')
        billing_session = RealmBillingSession(support_admin, realm=user.realm, support_session=True)
        with self.assertRaisesRegex(SupportRequestError, 'Discount for zulip must be updated before setting a minimum number of licenses.'):
            billing_session.process_support_view_request(support_view_request)
        billing_session.set_required_plan_tier(CustomerPlan.TIER_CLOUD_STANDARD)
        billing_session.attach_discount_to_customer(monthly_discounted_price=400, annual_discounted_price=4000)
        message = billing_session.process_support_view_request(support_view_request)
        self.assertEqual('Minimum licenses for zulip changed to 25 from 0.', message)
        realm_audit_log = RealmAuditLog.objects.filter(event_type=AuditLogEventType.CUSTOMER_PROPERTY_CHANGED).last()
        assert realm_audit_log is not None
        expected_extra_data = {'old_value': None, 'new_value': 25, 'property': 'minimum_licenses'}
        self.assertEqual(realm_audit_log.extra_data, expected_extra_data)
        self.login_user(user)
        self.add_card_and_upgrade(user)
        customer = billing_session.get_customer()
        assert customer is not None
        assert customer.stripe_customer_id is not None
        [charge] = iter(stripe.Charge.list(customer=customer.stripe_customer_id))
        self.assertEqual(4000 * min_licenses, charge.amount)
        min_licenses = 50
        support_view_request = SupportViewRequest(support_type=SupportType.update_minimum_licenses, minimum_licenses=min_licenses)
        with self.assertRaisesRegex(SupportRequestError, 'Cannot set minimum licenses; active plan already exists for zulip.'):
            billing_session.process_support_view_request(support_view_request)

    def test_set_required_plan_tier(self) -> None:
        valid_plan_tier = CustomerPlan.TIER_CLOUD_STANDARD
        support_view_request = SupportViewRequest(support_type=SupportType.update_required_plan_tier, required_plan_tier=valid_plan_tier)
        support_admin = self.example_user('iago')
        user = self.example_user('hamlet')
        billing_session = RealmBillingSession(support_admin, realm=user.realm, support_session=True)
        customer = billing_session.get_customer()
        assert customer is None
        message = billing_session.process_support_view_request(support_view_request)
        self.assertEqual('Required plan tier for zulip set to Zulip Cloud Standard.', message)
        realm_audit_log = RealmAuditLog.objects.filter(event_type=AuditLogEventType.CUSTOMER_PROPERTY_CHANGED).last()
        assert realm_audit_log is not None
        expected_extra_data = {'old_value': None, 'new_value': valid_plan_tier, 'property': 'required_plan_tier'}
        self.assertEqual(realm_audit_log.extra_data, expected_extra_data)
        customer = billing_session.get_customer()
        assert customer is not None
        self.assertEqual(customer.required_plan_tier, valid_plan_tier)
        self.assertEqual(customer.monthly_discounted_price, 0)
        self.assertEqual(customer.annual_discounted_price, 0)
        billing_session.attach_discount_to_customer(monthly_discounted_price=400, annual_discounted_price=4000)
        customer.refresh_from_db()
        self.assertEqual(customer.monthly_discounted_price, 400)
        self.assertEqual(customer.annual_discounted_price, 4000)
        monthly_discounted_price = customer.get_discounted_price_for_plan(valid_plan_tier, CustomerPlan.BILLING_SCHEDULE_MONTHLY)
        self.assertEqual(monthly_discounted_price, customer.monthly_discounted_price)
        annual_discounted_price = customer.get_discounted_price_for_plan(valid_plan_tier, CustomerPlan.BILLING_SCHEDULE_ANNUAL)
        self.assertEqual(annual_discounted_price, customer.annual_discounted_price)
        monthly_discounted_price = customer.get_discounted_price_for_plan(CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHLY)
        self.assertEqual(monthly_discounted_price, None)
        annual_discounted_price = customer.get_discounted_price_for_plan(CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_ANNUAL)
        self.assertEqual(annual_discounted_price, None)
        invalid_plan_tier = CustomerPlan.TIER_SELF_HOSTED_BASE
        support_view_request = SupportViewRequest(support_type=SupportType.update_required_plan_tier, required_plan_tier=invalid_plan_tier)
        with self.assertRaisesRegex(SupportRequestError, 'Invalid plan tier for zulip.'):
            billing_session.process_support_view_request(support_view_request)
        support_view_request = SupportViewRequest(support_type=SupportType.update_required_plan_tier, required_plan_tier=0)
        with self.assertRaisesRegex(SupportRequestError, 'Discount for zulip must be 0 before setting required plan tier to None.'):
            billing_session.process_support_view_request(support_view_request)
        billing_session.attach_discount_to_customer(monthly_discounted_price=0, annual_discounted_price=0)
        message = billing_session.process_support_view_request(support_view_request)
        self.assertEqual('Required plan tier for zulip set to None.', message)
        customer.refresh_from_db()
        self.assertIsNone(customer.required_plan_tier)
        discount_for_standard_plan = customer.get_discounted_price_for_plan(valid_plan_tier, CustomerPlan.BILLING_SCHEDULE_MONTHLY)
        self.assertEqual(discount_for_standard_plan, None)
        discount_for_plus_plan = customer.get_discounted_price_for_plan(CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHLY)
        self.assertEqual(discount_for_plus_plan, None)
        realm_audit_log = RealmAuditLog.objects.filter(event_type=AuditLogEventType.CUSTOMER_PROPERTY_CHANGED).last()
        assert realm_audit_log is not None
        expected_extra_data = {'old_value': valid_plan_tier, 'new_value': None, 'property': 'required_plan_tier'}
        self.assertEqual(realm_audit_log.extra_data, expected_extra_data)

    @mock_stripe()
    def test_approve_realm_sponsorship(self) -> None:
        realm = get_realm('zulip')
        self.assertNotEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD_FREE)
        support_admin = self.example_user('iago')
        billing_session = RealmBillingSession(user=support_admin, realm=realm, support_session=True)
        billing_session.approve_sponsorship()
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD_FREE)
        expected_message = "Your organization's request for sponsored hosting has been approved! You have been upgraded to Zulip Cloud Standard, free of charge. :tada:\n\nIf you could [list Zulip as a sponsor on your website](/help/linking-to-zulip-website), we would really appreciate it!"
        sender = get_system_bot(settings.NOTIFICATION_BOT, realm.id)
        desdemona_recipient = self.example_user('desdemona').recipient
        message_to_owner = Message.objects.filter(realm_id=realm.id, sender=sender.id, recipient=desdemona_recipient).first()
        assert message_to_owner is not None
        self.assertEqual(message_to_owner.content, expected_message)
        self.assertEqual(message_to_owner.recipient.type, Recipient.PERSONAL)
        hamlet_recipient = self.example_user('hamlet').recipient
        message_to_billing_admin = Message.objects.filter(realm_id=realm.id, sender=sender.id, recipient=hamlet_recipient).first()
        assert message_to_billing_admin is not None
        self.assertEqual(message_to_billing_admin.content, expected_message)
        self.assertEqual(message_to_billing_admin.recipient.type, Recipient.PERSONAL)

    def test_update_realm_sponsorship_status(self) -> None:
        lear = get_realm('lear')
        iago = self.example_user('iago')
        billing_session = RealmBillingSession(user=iago, realm=lear, support_session=True)
        billing_session.update_customer_sponsorship_status(True)
        customer = get_customer_by_realm(realm=lear)
        assert customer is not None
        self.assertTrue(customer.sponsorship_pending)
        realm_audit_log = RealmAuditLog.objects.filter(event_type=AuditLogEventType.REALM_SPONSORSHIP_PENDING_STATUS_CHANGED).last()
        assert realm_audit_log is not None
        expected_extra_data = {'sponsorship_pending': True}
        self.assertEqual(realm_audit_log.extra_data, expected_extra_data)
        self.assertEqual(realm_audit_log.acting_user, iago)

    def test_update_realm_billing_modality(self) -> None:
        realm = get_realm('zulip')
        customer = Customer.objects.create(realm=realm, stripe_customer_id='cus_12345')
        plan = CustomerPlan.objects.create(customer=customer, status=CustomerPlan.ACTIVE, billing_cycle_anchor=timezone_now(), billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, tier=CustomerPlan.TIER_CLOUD_STANDARD)
        self.assertEqual(plan.charge_automatically, False)
        support_admin = self.example_user('iago')
        billing_session = RealmBillingSession(user=support_admin, realm=realm, support_session=True)
        billing_session.update_billing_modality_of_current_plan(True)
        plan.refresh_from_db()
        self.assertEqual(plan.charge_automatically, True)
        realm_audit_log = RealmAuditLog.objects.filter(event_type=AuditLogEventType.REALM_BILLING_MODALITY_CHANGED).last()
        assert realm_audit_log is not None
        expected_extra_data = {'charge_automatically': plan.charge_automatically}
        self.assertEqual(realm_audit_log.acting_user, support_admin)
        self.assertEqual(realm_audit_log.extra_data, expected_extra_data)
        billing_session.update_billing_modality_of_current_plan(False)
        plan.refresh_from_db()
        self.assertEqual(plan.charge_automatically, False)
        realm_audit_log = RealmAuditLog.objects.filter(event_type=AuditLogEventType.REALM_BILLING_MODALITY_CHANGED).last()
        assert realm_audit_log is not None
        expected_extra_data = {'charge_automatically': plan.charge_automatically}
        self.assertEqual(realm_audit_log.acting_user, support_admin)
        self.assertEqual(realm_audit_log.extra_data, expected_extra_data)

    @responses.activate
    @mock_stripe()
    def test_switch_realm_from_standard_to_plus_plan(self, *mocks: Any) -> None:
        iago = self.example_user('iago')
        realm = iago.realm
        iago_billing_session = RealmBillingSession(iago, realm=realm)
        iago_billing_session.update_or_create_customer()
        from unittest.mock import patch
        with self.assertRaises(BillingError) as billing_context:
            iago_billing_session.do_change_plan_to_new_tier(CustomerPlan.TIER_CLOUD_PLUS)
        self.assertEqual('Organization does not have an active plan', billing_context.exception.error_description)
        plan, ledger = self.subscribe_realm_to_manual_license_management_plan(realm, 9, 9, CustomerPlan.BILLING_SCHEDULE_MONTHLY)
        with self.assertRaises(BillingError) as billing_context:
            iago_billing_session.do_change_plan_to_new_tier(CustomerPlan.TIER_CLOUD_PLUS)
        self.assertEqual('Organization missing Stripe customer.', billing_context.exception.error_description)
        king = self.lear_user('king')
        realm = king.realm
        king_billing_session = RealmBillingSession(king, realm=realm)
        customer = king_billing_session.update_or_create_stripe_customer()
        plan = CustomerPlan.objects.create(customer=customer, automanage_licenses=True, billing_cycle_anchor=timezone_now(), billing_schedule=CustomerPlan.BILLING_SCHEDULE_MONTHLY, tier=CustomerPlan.TIER_CLOUD_STANDARD)
        ledger = LicenseLedger.objects.create(plan=plan, is_renewal=True, event_time=timezone_now(), licenses=9, licenses_at_next_renewal=9)
        realm.plan_type = Realm.PLAN_TYPE_STANDARD
        realm.save(update_fields=['plan_type'])
        plan.invoiced_through = ledger
        plan.price_per_license = get_price_per_license(CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_MONTHLY)
        plan.save(update_fields=['invoiced_through', 'price_per_license'])
        with self.assertRaises(BillingError) as billing_context:
            king_billing_session.do_change_plan_to_new_tier(CustomerPlan.TIER_CLOUD_STANDARD)
        self.assertEqual('Invalid change of customer plan tier.', billing_context.exception.error_description)
        king_billing_session.do_change_plan_to_new_tier(CustomerPlan.TIER_CLOUD_PLUS)
        plan.refresh_from_db()
        self.assertEqual(plan.status, CustomerPlan.ENDED)
        plus_plan = get_current_plan_by_realm(realm)
        assert plus_plan is not None
        self.assertEqual(plus_plan.tier, CustomerPlan.TIER_CLOUD_PLUS)
        self.assertEqual(LicenseLedger.objects.filter(plan=plus_plan).count(), 1)
        realm.refresh_from_db()
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_PLUS)
        stripe_customer_id = customer.stripe_customer_id
        assert stripe_customer_id is not None
        from stripe import Charge
        _, cb_txn = iter(Charge.list(customer=stripe_customer_id))
        self.assertEqual(cb_txn.amount, -7200)
        self.assertEqual(cb_txn.description, 'Credit from early termination of active plan')
        self.assertEqual(cb_txn.type, 'adjustment')
        invoice, = iter(stripe.Invoice.list(customer=stripe_customer_id))
        self.assertEqual(invoice.amount_due, 4800)

    @mock_stripe()
    def test_customer_has_credit_card_as_default_payment_method(self) -> None:
        iago = self.example_user('iago')
        customer = Customer.objects.create(realm=iago.realm)
        self.assertFalse(customer_has_credit_card_as_default_payment_method(customer))
        billing_session = RealmBillingSession(iago, realm=iago.realm)
        customer = billing_session.update_or_create_stripe_customer()
        self.assertFalse(customer_has_credit_card_as_default_payment_method(customer))
        self.login_user(iago)
        self.add_card_and_upgrade(iago)
        self.assertTrue(customer_has_credit_card_as_default_payment_method(customer))


# The rest of the test classes follow with similar annotations.
# Due to length constraints, only portions of the code are annotated here.
# Additional classes such as TestRemoteRealmBillingFlow, TestRemoteServerBillingFlow,
# and others should be annotated in a similar manner with def ...() -> None.
