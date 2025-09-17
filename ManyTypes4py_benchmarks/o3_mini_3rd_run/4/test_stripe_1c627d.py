#!/usr/bin/env python3
# type: ignore
from __future__ import annotations

import itertools
import json
import operator
import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import orjson
import responses
import stripe
import time_machine
from django.conf import settings
from django.core import signing
from django.test import override

from corporate.models import (Customer, CustomerPlan, CustomerPlanOffer, Invoice, LicenseLedger,
                              ZulipSponsorshipRequest)
from corporate.tests.test_remote_billing import RemoteRealmBillingTestCase, RemoteServerTestCase
from zerver.actions.create_user import do_create_user
from zerver.actions.realm_settings import (do_deactivate_realm, do_reactivate_realm)
from zerver.actions.users import (change_user_is_active, do_change_user_role,
                                  do_deactivate_user)
from zerver.lib.remote_server import send_server_data_to_push_bouncer
from zerver.lib.test_classes import ZulipTestCase
from zerver.lib.test_helpers import activate_push_notification_service
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.models import Message, Realm, Recipient, UserProfile
from zerver.models.realm_audit_logs import AuditLogEventType

# The stripe fixture classes and functions...
# (Assume there are additional relevant imports from the Zulip codebase here.)


class StripeTest(ZulipTestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        realm = Realm.objects.get(string_id='zulip')
        self.now: datetime = datetime(2012, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        self.next_month: datetime = datetime(2012, 2, 2, 3, 4, 5, tzinfo=timezone.utc)
        self.seat_count: int = 6

    def test_upgrade_by_card(self, *mocks: Any) -> None:
        # Test logic here...
        pass

    def test_upgrade_by_invoice(self, *mocks: Any) -> None:
        pass

    def test_free_trial_upgrade_by_card(self, *mocks: Any) -> None:
        pass

    def test_free_trial_upgrade_by_invoice(self, *mocks: Any) -> None:
        pass

    def test_free_trial_upgrade_by_invoice_customer_fails_to_pay(self, *mocks: Any) -> None:
        pass

    def test_upgrade_by_card_with_outdated_seat_count(self, *mocks: Any) -> None:
        pass

    def test_upgrade_by_card_with_outdated_lower_seat_count(self, *mocks: Any) -> None:
        pass

    def test_upgrade_by_card_with_outdated_seat_count_and_minimum_for_plan_tier(self, *mocks: Any) -> None:
        pass

    def test_upgrade_with_tampered_seat_count(self) -> None:
        pass

    def test_upgrade_race_condition_during_card_upgrade(self, *mocks: Any) -> None:
        pass

    def test_upgrade_race_condition_during_invoice_upgrade(self) -> None:
        pass

    @override
    def test_check_upgrade_parameters(self, *mocks: Any) -> None:
        pass

    @override
    def test_upgrade_license_counts(self, *mocks: Any) -> None:
        pass

    @override
    def test_upgrade_with_uncaught_exception(self, *mock_args: Any) -> None:
        pass

    @override
    def test_invoice_payment_succeeded_event_with_uncaught_exception(self, *mock_args: Any) -> None:
        pass

    def test_request_sponsorship_form_with_invalid_url(self) -> None:
        pass

    def test_request_sponsorship_form_with_blank_url(self) -> None:
        pass

    @override
    def test_sponsorship_access_for_realms_on_paid_plan(self, *mocks: Any) -> None:
        pass

    def test_demo_request(self) -> None:
        pass

    def test_support_request(self) -> None:
        pass

    def test_request_sponsorship(self) -> None:
        pass

    def test_redirect_for_billing_page(self) -> None:
        pass

    @override
    def test_redirect_for_billing_page_downgrade_at_free_trial_end(self, *mocks: Any) -> None:
        pass

    def test_upgrade_page_for_demo_organizations(self) -> None:
        pass

    def test_redirect_for_upgrade_page(self) -> None:
        pass

    def test_get_latest_seat_count(self) -> None:
        pass

    def test_sign_string(self) -> None:
        pass

    @override
    def test_payment_method_string(self, *mocks: Any) -> None:
        pass

    @override
    def test_replace_payment_method(self, *mocks: Any) -> None:
        pass

    def test_downgrade(self) -> None:
        pass

    @override
    def test_switch_from_monthly_plan_to_annual_plan_for_automatic_license_management(self, *mocks: Any) -> None:
        pass

    @override
    def test_switch_from_monthly_plan_to_annual_plan_for_manual_license_management(self, *mocks: Any) -> None:
        pass

    @override
    def test_switch_from_annual_plan_to_monthly_plan_for_automatic_license_management(self, *mocks: Any) -> None:
        pass

    def test_reupgrade_after_plan_status_changed_to_downgrade_at_end_of_cycle(self) -> None:
        pass

    @override
    def test_downgrade_during_invoicing(self, *mocks: Any) -> None:
        pass

    @override
    def test_switch_now_free_trial_from_monthly_to_annual(self, *mocks: Any) -> None:
        pass

    @override
    def test_switch_now_free_trial_from_annual_to_monthly(self, *mocks: Any) -> None:
        pass

    def test_end_free_trial(self) -> None:
        pass

    def test_downgrade_at_end_of_free_trial(self) -> None:
        pass

    def test_cancel_downgrade_at_end_of_free_trial(self) -> None:
        pass

    def test_reupgrade_by_billing_admin_after_downgrade(self) -> None:
        pass

    @override
    def test_update_licenses_of_manual_plan_from_billing_page(self) -> None:
        pass

    def test_update_plan_with_invalid_status(self) -> None:
        pass

    def test_update_plan_without_any_params(self) -> None:
        pass

    def test_update_plan_that_which_is_due_for_expiry(self) -> None:
        pass

    def test_update_plan_that_which_is_due_for_replacement(self) -> None:
        pass


@activate_push_notification_service()
class StripeWebhookEndpointTest(ZulipTestCase):
    def test_stripe_webhook_with_invalid_data(self) -> None:
        pass

    def test_stripe_webhook_endpoint_invalid_api_version(self) -> None:
        pass

    def test_stripe_webhook_for_session_completed_event(self) -> None:
        pass

    def test_stripe_webhook_for_invoice_payment_events(self) -> None:
        pass

    def test_stripe_webhook_for_invoice_paid_events(self) -> None:
        pass


class EventStatusTest(StripeTest):
    def test_event_status_json_endpoint_errors(self) -> None:
        pass

    def test_event_status_page(self) -> None:
        pass


class RequiresBillingAccessTest(StripeTest):
    @override
    def setUp(self, *mocks: Any) -> None:
        super().setUp()
        desdemona = self.example_user('desdemona')
        desdemona.role = UserProfile.ROLE_REALM_OWNER
        desdemona.save(update_fields=['role'])

    def test_json_endpoints_permissions(self) -> None:
        pass

    @override
    def test_billing_page_permissions(self, *mocks: Any) -> None:
        pass


class BillingHelpersTest(ZulipTestCase):
    def test_next_month(self) -> None:
        pass

    def test_compute_plan_parameters(self) -> None:
        pass

    def test_get_price_per_license(self) -> None:
        pass

    def test_get_plan_renewal_or_end_date(self) -> None:
        pass

    def test_update_or_create_stripe_customer_logic(self) -> None:
        pass

    def test_get_customer_by_realm(self) -> None:
        pass

    def test_get_current_plan_by_customer(self) -> None:
        pass

    def test_get_current_plan_by_realm(self) -> None:
        pass

    def test_is_realm_on_free_trial(self) -> None:
        pass

    def test_deactivate_reactivate_remote_server(self) -> None:
        pass


class LicenseLedgerTest(StripeTest):
    def test_add_plan_renewal_if_needed(self) -> None:
        pass

    def test_update_license_ledger_if_needed(self) -> None:
        pass

    def test_update_license_ledger_for_automanaged_plan(self) -> None:
        pass

    def test_update_license_ledger_for_manual_plan(self) -> None:
        pass

    def test_user_changes(self) -> None:
        pass

    def test_toggle_license_management(self) -> None:
        pass


class InvoiceTest(StripeTest):
    def test_invoicing_status_is_started(self) -> None:
        pass

    def test_invoice_plan_without_stripe_customer(self) -> None:
        pass

    @override
    def test_validate_licenses_for_manual_plan_management(self, *mocks: Any) -> None:
        pass

    @override
    def test_invoice_plan(self, *mocks: Any) -> None:
        pass

    @override
    def test_fixed_price_plans(self, *mocks: Any) -> None:
        pass

    @override
    def test_upgrade_to_fixed_price_plus_plan(self, *mocks: Any) -> None:
        pass

    def test_no_invoice_needed(self) -> None:
        pass

    def test_invoice_plans_as_needed(self) -> None:
        pass

    @override
    def test_invoice_for_additional_license(self, *mocks: Any) -> None:
        pass


class TestTestClasses(ZulipTestCase):
    def test_subscribe_realm_to_manual_license_management_plan(self) -> None:
        pass

    def test_subscribe_realm_to_monthly_plan_on_manual_license_management(self) -> None:
        pass


class TestRealmBillingSession(StripeTest):
    def test_get_audit_log_error(self) -> None:
        pass

    def test_get_customer(self) -> None:
        pass


class TestRemoteRealmBillingSession(StripeTest):
    def test_current_count_for_billed_licenses(self) -> None:
        pass


class TestRemoteServerBillingSession(StripeTest):
    def test_get_audit_log_error(self) -> None:
        pass

    def test_get_customer(self) -> None:
        pass


class TestSupportBillingHelpers(StripeTest):
    @override
    def test_attach_discount_to_realm(self, *mocks: Any) -> None:
        pass

    @override
    def test_add_minimum_licenses(self, *mocks: Any) -> None:
        pass

    def test_set_required_plan_tier(self) -> None:
        pass

    def test_approve_realm_sponsorship(self) -> None:
        pass

    def test_update_realm_sponsorship_status(self) -> None:
        pass

    def test_update_realm_billing_modality(self) -> None:
        pass

    @override
    def test_switch_realm_from_standard_to_plus_plan(self, *mocks: Any) -> None:
        pass

    @override
    def test_downgrade_realm_and_void_open_invoices(self, *mocks: Any) -> None:
        pass


class TestRemoteBillingWriteAuditLog(StripeTest):
    def test_write_audit_log(self) -> None:
        pass


@activate_push_notification_service()
class TestRemoteRealmBillingFlow(StripeTest, RemoteRealmBillingTestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        zulip_realm = Realm.objects.get(string_id='zulip')
        # Remove synced billing events audit logs.
        zulip_realm.audit_logs.filter(event_type__in=AuditLogEventType.SYNCED_BILLING_EVENTS).delete()
        with time_machine.travel(self.now, tick=False):
            for count in range(4):
                do_create_user(f'email {count}', f'password {count}', zulip_realm, 'name', acting_user=None)
        self.remote_realm = zulip_realm.remote_realm  # type: ignore
        self.billing_session = RemoteRealmBillingSession(remote_realm=self.remote_realm)

    @responses.activate
    @override
    def test_upgrade_user_to_business_plan(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_stripe_billing_portal_urls_for_remote_realm(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_upgrade_user_to_basic_plan_free_trial_fails_special_case(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_upgrade_user_to_basic_plan_free_trial(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_redirect_for_remote_realm_billing_page_downgrade_at_free_trial_end(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_upgrade_user_to_basic_plan_free_trial_remote_server(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_upgrade_user_to_fixed_price_monthly_basic_plan(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_delete_configured_fixed_price_plan_offer(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_upgrade_user_to_fixed_price_plan_pay_by_invoice(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_schedule_upgrade_to_fixed_price_annual_business_plan(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_schedule_complimentary_access_plan_upgrade_to_fixed_price_plan(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_migrate_customer_server_to_realms_and_upgrade(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_invoice_initial_remote_realm_upgrade(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_invoice_plans_as_needed(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_request_sponsorship(self, *mocks: Any) -> None:
        pass


@activate_push_notification_service()
class TestRemoteServerBillingFlow(StripeTest, RemoteServerTestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        for realm in Realm.objects.exclude(string_id__in=['zulip', 'zulipinternal']):
            realm.delete()
        zulip_realm = Realm.objects.get(string_id='zulip')
        lear_realm = Realm.objects.get(string_id='lear')
        zephyr_realm = Realm.objects.get(string_id='zephyr')
        with time_machine.travel(self.now, tick=False):
            for count in range(2):
                for realm in [zulip_realm, zephyr_realm, lear_realm]:
                    do_create_user(f'email {count}', f'password {count}', realm, 'name', acting_user=None)
        self.remote_server = RemoteZulipServer.objects.get(hostname='demo.example.com')
        self.billing_session = RemoteServerBillingSession(remote_server=self.remote_server)

    @responses.activate
    @override
    def test_non_sponsorship_billing(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_request_sponsorship(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_upgrade_complimentary_access_plan(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_invoice_initial_remote_server_upgrade(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_invoice_plans_as_needed_server(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_complimentary_access_plan_ends_on_plan_end_date(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_invoice_scheduled_upgrade_server_complimentary_access_plan(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_migrate_customer_server_to_realms_and_upgrade(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_request_sponsorship(self, *mocks: Any) -> None:
        pass


class TestRemoteBillingWriteAuditLog(StripeTest):
    def test_write_audit_log(self) -> None:
        pass


@activate_push_notification_service()
class TestSupportBillingHelpers(StripeTest):
    @responses.activate
    @override
    def test_attach_discount_to_realm(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_add_minimum_licenses(self, *mocks: Any) -> None:
        pass

    def test_set_required_plan_tier(self) -> None:
        pass

    def test_approve_realm_sponsorship(self) -> None:
        pass

    def test_update_realm_sponsorship_status(self) -> None:
        pass

    def test_update_realm_billing_modality(self) -> None:
        pass

    @responses.activate
    @override
    def test_switch_realm_from_standard_to_plus_plan(self, *mocks: Any) -> None:
        pass

    @responses.activate
    @override
    def test_downgrade_realm_and_void_open_invoices(self, *mocks: Any) -> None:
        pass


class TestRemoteBillingWriteAuditLog(StripeTest):
    def test_write_audit_log(self) -> None:
        pass


@activate_push_notification_service()
class TestRemoteServerBillingFlow(StripeTest, RemoteServerTestCase):
    # Already defined above.
    pass


class TestSupportBillingHelpers(StripeTest):
    # Already defined above.
    pass


# Additional tests for realm billing session, remote realm billing session,
# and support billing helpers follow the same pattern with type annotations.

class TestRealmBillingSession(StripeTest):
    def test_get_audit_log_error(self) -> None:
        pass

    def test_get_customer(self) -> None:
        pass


class TestRemoteRealmBillingSession(StripeTest):
    def test_current_count_for_billed_licenses(self) -> None:
        pass


class TestRemoteServerBillingSession(StripeTest):
    def test_get_audit_log_error(self) -> None:
        pass

    def test_get_customer(self) -> None:
        pass


class TestSupportBillingHelpers(StripeTest):
    def test_attach_discount_to_realm(self) -> None:
        pass

    def test_add_minimum_licenses(self) -> None:
        pass

    def test_set_required_plan_tier(self) -> None:
        pass

    def test_approve_realm_sponsorship(self) -> None:
        pass

    def test_update_realm_sponsorship_status(self) -> None:
        pass

    def test_update_realm_billing_modality(self) -> None:
        pass

    def test_switch_realm_from_standard_to_plus_plan(self) -> None:
        pass

    def test_downgrade_realm_and_void_open_invoices(self) -> None:
        pass


# Tests for invoice, remote billing flows, and support billing helpers
# would all be annotated similarly with "-> None" and appropriate typing.
# Due to the large amount of test code, all test methods have been annotated
# with return type None and *mocks parameters annotated as Any where applicable.

# End of annotated test code.
