#!/usr/bin/env python3
"""
This is the annotated version of the code.
"""

import logging
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import stripe
from django.http import HttpRequest, HttpResponse

# Assume these are imported from appropriate modules
# from .models import Customer, CustomerPlan, Realm, RemoteRealm, RemoteZulipServer, RemoteRealmBillingUser, RemoteServerBillingUser, ZulipSponsorshipRequest
# from .audit_log import AuditLogEventType
# from .exceptions import BillingError, InvalidTierError, InvalidBillingScheduleError, UpgradeWithExistingPlanError, ServerDeactivateWithExistingPlanError
# from .utils import cache_with_key, log_to_file, assert_is_not_none, cents_to_dollar_string, add_months, next_invoice_date, start_of_next_billing_cycle

BILLING_LOG_PATH = "/var/log/billing.log"

billing_logger = logging.getLogger("billing")
log_to_file(billing_logger, BILLING_LOG_PATH)


@dataclass
class PushNotificationsEnabledStatus:
    can_push: bool
    expected_end_timestamp: Optional[int]
    message: str


@dataclass
class StripeCustomerData:
    description: str
    email: str
    metadata: Dict[str, Any]


@dataclass
class UpgradeRequest:
    tier: int
    licenses: int
    billing_modality: str
    schedule: int
    license_management: str
    signed_seat_count: str
    salt: str
    remote_server_plan_start_date: Optional[str]


@dataclass
class InitialUpgradeRequest:
    tier: int
    billing_modality: str
    manual_license_management: bool
    success_message: str = ""


@dataclass
class UpdatePlanRequest:
    status: Optional[int]
    licenses: Optional[int]
    licenses_at_next_renewal: Optional[int]
    toggle_license_management: bool
    schedule: Optional[int]


@dataclass
class EventStatusRequest:
    stripe_session_id: Optional[str]
    stripe_invoice_id: Optional[str]


class BillingSession:
    def get_customer(self) -> Optional["Customer"]:
        raise NotImplementedError

    def get_email(self) -> str:
        raise NotImplementedError

    def current_count_for_billed_licenses(self, event_time: Optional[datetime] = None) -> int:
        raise NotImplementedError

    def get_audit_log_event(self, event_type: Any) -> int:
        raise NotImplementedError

    def write_to_audit_log(
        self,
        event_type: Any,
        event_time: datetime,
        *,
        background_update: bool = False,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError

    def get_data_for_stripe_customer(self) -> StripeCustomerData:
        raise NotImplementedError

    def update_data_for_checkout_session_and_invoice_payment(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def update_or_create_customer(
        self, stripe_customer_id: Optional[str] = None, *, defaults: Optional[Dict[str, Any]] = None
    ) -> "Customer":
        raise NotImplementedError

    def do_change_plan_type(self, *, tier: Optional[int], is_sponsored: bool = False, background_update: bool = False) -> None:
        raise NotImplementedError

    def process_downgrade(self, plan: "CustomerPlan", background_update: bool = False) -> None:
        raise NotImplementedError

    def approve_sponsorship(self) -> str:
        raise NotImplementedError

    def is_sponsored(self) -> bool:
        raise NotImplementedError

    def get_sponsorship_request_session_specific_context(self) -> Dict[str, Any]:
        raise NotImplementedError

    def save_org_type_from_request_sponsorship_session(self, org_type: int) -> None:
        raise NotImplementedError

    def get_upgrade_page_session_type_specific_context(self) -> Dict[str, Any]:
        raise NotImplementedError

    def check_plan_tier_is_billable(self, plan_tier: int) -> bool:
        raise NotImplementedError

    def get_type_of_plan_tier_change(self, current_plan_tier: int, new_plan_tier: int) -> "PlanTierChangeType":
        raise NotImplementedError

    def has_billing_access(self) -> bool:
        raise NotImplementedError

    def on_paid_plan(self) -> bool:
        raise NotImplementedError

    def add_org_type_data_to_sponsorship_context(self, context: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_metadata_for_stripe_update_card(self) -> Dict[str, str]:
        raise NotImplementedError

    def sync_license_ledger_if_needed(self) -> None:
        raise NotImplementedError

    def send_support_admin_realm_internal_message(self, channel_name: str, topic: str, message: str) -> None:
        from zerver.actions.message_send import internal_send_private_message, internal_send_stream_message

        admin_realm = get_realm(settings.SYSTEM_BOT_REALM)
        sender = get_system_bot(settings.NOTIFICATION_BOT, admin_realm.id)
        try:
            channel = get_stream(channel_name, admin_realm)
            internal_send_stream_message(sender, channel, topic, message)
        except Stream.DoesNotExist:
            direct_message = (
                f":red_circle: Channel named '{channel_name}' doesn't exist.\n\n{topic}:\n{message}"
            )
            for user in admin_realm.get_human_admin_users():
                internal_send_private_message(sender, user, direct_message)

    def do_upgrade(self, upgrade_request: UpgradeRequest) -> Dict[str, Any]:
        customer = self.get_customer()
        if customer is not None:
            self.ensure_current_plan_is_upgradable(customer, upgrade_request.tier)
        billing_modality = upgrade_request.billing_modality
        schedule = upgrade_request.schedule

        license_management = upgrade_request.license_management
        if billing_modality == "send_invoice":
            license_management = "manual"

        licenses = upgrade_request.licenses
        request_seat_count = unsign_seat_count(upgrade_request.signed_seat_count, upgrade_request.salt)
        # For automated license management, we check for changes to the billable licenses count made after the billing portal was loaded.
        seat_count = self.stale_seat_count_check(request_seat_count, upgrade_request.tier)
        if billing_modality == "charge_automatically" and license_management == "automatic":
            licenses = seat_count

        exempt_from_license_number_check = customer is not None and customer.exempt_from_license_number_check
        check_upgrade_parameters(
            billing_modality,
            schedule,
            license_management,
            licenses,
            seat_count,
            exempt_from_license_number_check,
            self.min_licenses_for_plan(upgrade_request.tier),
        )
        assert licenses is not None and license_management is not None
        automanage_licenses = license_management == "automatic"
        charge_automatically = billing_modality == "charge_automatically"

        billing_schedule = {"annual": CustomerPlan.BILLING_SCHEDULE_ANNUAL, "monthly": CustomerPlan.BILLING_SCHEDULE_MONTHLY}[schedule]
        data: Dict[str, Any] = {}

        is_self_hosted_billing = not isinstance(self, RealmBillingSession)
        free_trial = is_free_trial_offer_enabled(is_self_hosted_billing, upgrade_request.tier)
        if customer is not None:
            fixed_price_plan_offer = get_configured_fixed_price_plan_offer(customer, upgrade_request.tier)
            if fixed_price_plan_offer is not None:
                free_trial = False

        complimentary_access_plan = self.get_complimentary_access_plan(customer)
        upgrade_when_complimentary_access_plan_ends = (
            complimentary_access_plan is not None and upgrade_request.remote_server_plan_start_date == "billing_cycle_end_date"
        )
        # Directly upgrade free trial orgs.
        # Create NEVER_STARTED plan for complimentary access plans.
        if upgrade_when_complimentary_access_plan_ends or free_trial:
            self.process_initial_upgrade(
                upgrade_request.tier,
                licenses,
                automanage_licenses,
                billing_schedule,
                charge_automatically,
                free_trial,
                complimentary_access_plan,
                upgrade_when_complimentary_access_plan_ends,
            )
            data["organization_upgrade_successful"] = True
        else:
            stripe_invoice_id = self.generate_stripe_invoice(
                upgrade_request.tier,
                licenses,
                license_management,
                billing_schedule,
                billing_modality,
            )
            data["stripe_invoice_id"] = stripe_invoice_id
        return data

    def do_change_schedule_after_free_trial(self, plan: "CustomerPlan", schedule: int) -> None:
        assert plan.charge_automatically
        # Change the billing frequency of the plan after the free trial ends.
        assert schedule in (CustomerPlan.BILLING_SCHEDULE_MONTHLY, CustomerPlan.BILLING_SCHEDULE_ANNUAL)
        last_ledger_entry = LicenseLedger.objects.filter(plan=plan).order_by("-id").first()
        assert last_ledger_entry is not None
        licenses_at_next_renewal = last_ledger_entry.licenses_at_next_renewal
        assert licenses_at_next_renewal is not None
        assert plan.next_invoice_date is not None
        next_billing_cycle = plan.next_invoice_date

        if plan.fixed_price is not None:
            raise BillingError("Customer is already on monthly fixed plan.")

        plan.status = CustomerPlan.ENDED
        plan.next_invoice_date = None
        plan.save(update_fields=["status", "next_invoice_date"])

        price_per_license, discount_for_current_plan = get_price_per_license_and_discount(
            plan.tier, schedule, plan.customer
        )

        new_plan = CustomerPlan.objects.create(
            customer=plan.customer,
            billing_schedule=schedule,
            automanage_licenses=plan.automanage_licenses,
            charge_automatically=plan.charge_automatically,
            price_per_license=price_per_license,
            discount=discount_for_current_plan,
            billing_cycle_anchor=plan.billing_cycle_anchor,
            tier=plan.tier,
            status=CustomerPlan.FREE_TRIAL,
            next_invoice_date=next_billing_cycle,
        )

        ledger_entry = LicenseLedger.objects.create(
            plan=new_plan,
            is_renewal=True,
            event_time=plan.billing_cycle_anchor,
            licenses=licenses_at_next_renewal,
            licenses_at_next_renewal=licenses_at_next_renewal,
        )

        new_plan.invoiced_through = ledger_entry
        new_plan.save(update_fields=["invoiced_through"])

        if schedule == CustomerPlan.BILLING_SCHEDULE_ANNUAL:
            self.write_to_audit_log(
                event_type=BillingSessionEventType.CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN,
                event_time=timezone_now(),
                extra_data={"monthly_plan_id": plan.id, "annual_plan_id": new_plan.id},
            )
        else:
            self.write_to_audit_log(
                event_type=BillingSessionEventType.CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_PLAN,
                event_time=timezone_now(),
                extra_data={"annual_plan_id": plan.id, "monthly_plan_id": new_plan.id},
            )

    def get_next_billing_cycle(self, plan: "CustomerPlan") -> datetime:
        if plan.status in (CustomerPlan.FREE_TRIAL, CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL, CustomerPlan.NEVER_STARTED):
            assert plan.next_invoice_date is not None
            next_billing_cycle = plan.next_invoice_date
        elif plan.status == CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END:
            assert plan.end_date is not None
            next_billing_cycle = plan.end_date
        else:
            last_ledger_renewal = LicenseLedger.objects.filter(plan=plan, is_renewal=True).order_by("-id").first()
            assert last_ledger_renewal is not None
            last_renewal = last_ledger_renewal.event_time
            next_billing_cycle = start_of_next_billing_cycle(plan, last_renewal)

        if plan.end_date is not None:
            next_billing_cycle = min(next_billing_cycle, plan.end_date)

        return next_billing_cycle

    def validate_plan_license_management(self, plan: "CustomerPlan", renewal_license_count: int) -> None:
        if plan.customer.exempt_from_license_number_check:
            return

        if plan.tier not in [CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.TIER_CLOUD_PLUS]:
            return

        min_licenses = self.min_licenses_for_plan(plan.tier)
        if min_licenses > renewal_license_count:
            raise BillingError(
                f"Renewal licenses ({renewal_license_count}) less than minimum licenses ({min_licenses}) required for plan {plan.name}."
            )

        if plan.automanage_licenses:
            return

        if self.current_count_for_billed_licenses() > renewal_license_count:
            raise BillingError(
                f"Customer has not manually updated plan for current license count: {plan.customer!s}"
            )

    def make_end_of_cycle_updates_if_needed(
        self, plan: "CustomerPlan", event_time: datetime
    ) -> Tuple[Optional["CustomerPlan"], Optional["LicenseLedger"]]:
        last_ledger_entry = LicenseLedger.objects.filter(plan=plan, event_time__lte=event_time).order_by("-id").first()
        next_billing_cycle = self.get_next_billing_cycle(plan)
        event_in_next_billing_cycle = next_billing_cycle <= event_time

        if event_in_next_billing_cycle and last_ledger_entry is not None:
            licenses_at_next_renewal = last_ledger_entry.licenses_at_next_renewal
            assert licenses_at_next_renewal is not None

            if plan.end_date == next_billing_cycle and plan.status == CustomerPlan.ACTIVE:
                self.process_downgrade(plan, True)
                return None, None

            if plan.status == CustomerPlan.ACTIVE:
                self.validate_plan_license_management(plan, licenses_at_next_renewal)
                return None, LicenseLedger.objects.create(
                    plan=plan,
                    is_renewal=True,
                    event_time=next_billing_cycle,
                    licenses=licenses_at_next_renewal,
                    licenses_at_next_renewal=licenses_at_next_renewal,
                )
            if plan.is_free_trial():
                is_renewal = True
                if not plan.charge_automatically:
                    last_sent_invoice = Invoice.objects.filter(plan=plan).order_by("-id").first()
                    if last_sent_invoice and last_sent_invoice.status == Invoice.PAID:
                        is_renewal = False
                    else:
                        plan.status = CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL
                        plan.save(update_fields=["status"])
                        self.make_end_of_cycle_updates_if_needed(plan, event_time)
                        return None, None

                plan.invoiced_through = last_ledger_entry
                plan.billing_cycle_anchor = next_billing_cycle.replace(microsecond=0)
                plan.status = CustomerPlan.ACTIVE
                plan.save(update_fields=["invoiced_through", "billing_cycle_anchor", "status"])
                return None, LicenseLedger.objects.create(
                    plan=plan,
                    is_renewal=is_renewal,
                    event_time=next_billing_cycle,
                    licenses=licenses_at_next_renewal,
                    licenses_at_next_renewal=licenses_at_next_renewal,
                )

            if plan.status == CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END:
                plan.status = CustomerPlan.ENDED
                plan.save(update_fields=["status"])

                assert plan.end_date is not None
                new_plan = CustomerPlan.objects.get(
                    customer=plan.customer, billing_cycle_anchor=plan.end_date, status=CustomerPlan.NEVER_STARTED
                )
                self.validate_plan_license_management(new_plan, licenses_at_next_renewal)
                new_plan.status = CustomerPlan.ACTIVE
                new_plan.save(update_fields=["status"])
                self.do_change_plan_type(tier=new_plan.tier, background_update=True)
                return None, LicenseLedger.objects.create(
                    plan=new_plan,
                    is_renewal=True,
                    event_time=next_billing_cycle,
                    licenses=licenses_at_next_renewal,
                    licenses_at_next_renewal=licenses_at_next_renewal,
                )

            if plan.status == CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE:
                self.validate_plan_license_management(plan, licenses_at_next_renewal)
                if plan.fixed_price is not None:
                    raise NotImplementedError("Can't switch fixed priced monthly plan to annual.")

                plan.status = CustomerPlan.ENDED
                plan.save(update_fields=["status"])

                price_per_license, discount_for_current_plan = get_price_per_license_and_discount(
                    plan.tier, CustomerPlan.BILLING_SCHEDULE_ANNUAL, plan.customer
                )

                new_plan = CustomerPlan.objects.create(
                    customer=plan.customer,
                    billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
                    automanage_licenses=plan.automanage_licenses,
                    charge_automatically=plan.charge_automatically,
                    price_per_license=price_per_license,
                    discount=discount_for_current_plan,
                    billing_cycle_anchor=next_billing_cycle,
                    tier=plan.tier,
                    status=CustomerPlan.ACTIVE,
                    next_invoice_date=next_billing_cycle,
                    invoiced_through=None,
                    invoicing_status=CustomerPlan.INVOICING_STATUS_INITIAL_INVOICE_TO_BE_SENT,
                )

                new_plan_ledger_entry = LicenseLedger.objects.create(
                    plan=new_plan,
                    is_renewal=True,
                    event_time=next_billing_cycle,
                    licenses=licenses_at_next_renewal,
                    licenses_at_next_renewal=licenses_at_next_renewal,
                )

                self.write_to_audit_log(
                    event_type=BillingSessionEventType.CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN,
                    event_time=event_time,
                    extra_data={"monthly_plan_id": plan.id, "annual_plan_id": new_plan.id},
                    background_update=True,
                )
                return new_plan, new_plan_ledger_entry

            if plan.status == CustomerPlan.SWITCH_TO_MONTHLY_AT_END_OF_CYCLE:
                self.validate_plan_license_management(plan, licenses_at_next_renewal)
                if plan.fixed_price is not None:
                    raise BillingError("Customer is already on monthly fixed plan.")

                plan.status = CustomerPlan.ENDED
                plan.save(update_fields=["status"])

                price_per_license, discount_for_current_plan = get_price_per_license_and_discount(
                    plan.tier, CustomerPlan.BILLING_SCHEDULE_MONTHLY, plan.customer
                )

                new_plan = CustomerPlan.objects.create(
                    customer=plan.customer,
                    billing_schedule=CustomerPlan.BILLING_SCHEDULE_MONTHLY,
                    automanage_licenses=plan.automanage_licenses,
                    charge_automatically=plan.charge_automatically,
                    price_per_license=price_per_license,
                    discount=discount_for_current_plan,
                    billing_cycle_anchor=next_billing_cycle,
                    tier=plan.tier,
                    status=CustomerPlan.ACTIVE,
                    next_invoice_date=next_billing_cycle,
                    invoiced_through=None,
                    invoicing_status=CustomerPlan.INVOICING_STATUS_INITIAL_INVOICE_TO_BE_SENT,
                )

                new_plan_ledger_entry = LicenseLedger.objects.create(
                    plan=new_plan,
                    is_renewal=True,
                    event_time=next_billing_cycle,
                    licenses=licenses_at_next_renewal,
                    licenses_at_next_renewal=licenses_at_next_renewal,
                )

                self.write_to_audit_log(
                    event_type=BillingSessionEventType.CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_PLAN,
                    event_time=event_time,
                    extra_data={"annual_plan_id": plan.id, "monthly_plan_id": new_plan.id},
                    background_update=True,
                )
                return new_plan, new_plan_ledger_entry

            if plan.status == CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL:
                self.downgrade_now_without_creating_additional_invoices(plan, background_update=True)

            if plan.status == CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE:
                self.process_downgrade(plan, background_update=True)

            return None, last_ledger_entry
        return None, last_ledger_entry

    def get_next_plan(self, plan: "CustomerPlan") -> Optional["CustomerPlan"]:
        if plan.status == CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END:
            assert plan.end_date is not None
            return CustomerPlan.objects.filter(
                customer=plan.customer,
                billing_cycle_anchor=plan.end_date,
                status=CustomerPlan.NEVER_STARTED,
            ).first()
        return None

    def get_annual_recurring_revenue_for_support_data(
        self, plan: "CustomerPlan", last_ledger_entry: "LicenseLedger"
    ) -> int:
        if plan.fixed_price is not None:
            return plan.fixed_price
        revenue = self.get_customer_plan_renewal_amount(plan, last_ledger_entry)
        if plan.billing_schedule == CustomerPlan.BILLING_SCHEDULE_MONTHLY:
            revenue *= 12
        return revenue

    def get_customer_plan_renewal_amount(
        self, plan: "CustomerPlan", last_ledger_entry: "LicenseLedger"
    ) -> int:
        if plan.fixed_price is not None:
            if plan.end_date == self.get_next_billing_cycle(plan):
                return 0
            return get_amount_due_fixed_price_plan(plan.fixed_price, plan.billing_schedule)
        if last_ledger_entry.licenses_at_next_renewal is None:
            return 0
        assert plan.price_per_license is not None
        return plan.price_per_license * last_ledger_entry.licenses_at_next_renewal

    def get_billing_context_from_plan(
        self, customer: "Customer", plan: "CustomerPlan", last_ledger_entry: "LicenseLedger", now: datetime
    ) -> Dict[str, Any]:
        is_self_hosted_billing = not isinstance(self, RealmBillingSession)
        downgrade_at_end_of_cycle = plan.status == CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE
        downgrade_at_end_of_free_trial = plan.status == CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL
        switch_to_annual_at_end_of_cycle = plan.status == CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE
        switch_to_monthly_at_end_of_cycle = plan.status == CustomerPlan.SWITCH_TO_MONTHLY_AT_END_OF_CYCLE
        licenses = last_ledger_entry.licenses
        licenses_at_next_renewal = last_ledger_entry.licenses_at_next_renewal
        assert licenses_at_next_renewal is not None
        min_licenses_for_plan = self.min_licenses_for_plan(plan.tier)
        seat_count = self.current_count_for_billed_licenses()
        using_min_licenses_for_plan = (
            min_licenses_for_plan == licenses_at_next_renewal and licenses_at_next_renewal > seat_count
        )

        if plan.is_free_trial() or downgrade_at_end_of_free_trial:
            assert plan.next_invoice_date is not None
            renewal_date = f"{plan.next_invoice_date:%B} {plan.next_invoice_date.day}, {plan.next_invoice_date.year}"
        else:
            renewal_date = "{dt:%B} {dt.day}, {dt.year}".format(dt=start_of_next_billing_cycle(plan, now))

        has_paid_invoice_for_free_trial = False
        free_trial_next_renewal_date_after_invoice_paid: Optional[str] = None
        if plan.is_free_trial() and not plan.charge_automatically:
            last_sent_invoice = Invoice.objects.filter(plan=plan).order_by("-id").first()
            assert last_sent_invoice is not None
            has_paid_invoice_for_free_trial = last_sent_invoice.status == Invoice.PAID

            if has_paid_invoice_for_free_trial:
                assert plan.next_invoice_date is not None
                free_trial_days = get_free_trial_days(is_self_hosted_billing, plan.tier)
                assert free_trial_days is not None
                free_trial_next_renewal_date_after_invoice_paid = (
                    "{dt:%B} {dt.day}, {dt.year}".format(
                        dt=(start_of_next_billing_cycle(plan, plan.next_invoice_date) + timedelta(days=free_trial_days))
                    )
                )

        billing_frequency = CustomerPlan.BILLING_SCHEDULES[plan.billing_schedule]

        if switch_to_annual_at_end_of_cycle:
            num_months_next_cycle = 12
            annual_price_per_license = get_price_per_license(plan.tier, CustomerPlan.BILLING_SCHEDULE_ANNUAL, customer)
            renewal_cents = annual_price_per_license * licenses_at_next_renewal
            price_per_license = format_money(annual_price_per_license / 12)
        elif switch_to_monthly_at_end_of_cycle:
            num_months_next_cycle = 1
            monthly_price_per_license = get_price_per_license(plan.tier, CustomerPlan.BILLING_SCHEDULE_MONTHLY, customer)
            renewal_cents = monthly_price_per_license * licenses_at_next_renewal
            price_per_license = format_money(monthly_price_per_license)
        else:
            num_months_next_cycle = 12 if plan.billing_schedule == CustomerPlan.BILLING_SCHEDULE_ANNUAL else 1
            renewal_cents = self.get_customer_plan_renewal_amount(plan, last_ledger_entry)

            if plan.price_per_license is None:
                price_per_license = ""
            elif billing_frequency == "Annual":
                price_per_license = format_money(plan.price_per_license / 12)
            else:
                price_per_license = format_money(plan.price_per_license)

        pre_discount_renewal_cents = renewal_cents
        flat_discount, flat_discounted_months = self.get_flat_discount_info(plan.customer)
        if plan.fixed_price is None and flat_discounted_months > 0:
            flat_discounted_months = min(flat_discounted_months, num_months_next_cycle)
            discount = flat_discount * flat_discounted_months
            renewal_cents -= discount

        charge_automatically = plan.charge_automatically
        if customer.stripe_customer_id is not None:
            stripe_customer = stripe_get_customer(customer.stripe_customer_id)
            stripe_email = stripe_customer.email
            if charge_automatically:
                payment_method = payment_method_string(stripe_customer)
            else:
                payment_method = "Invoice"
        elif settings.DEVELOPMENT:
            payment_method = "Payment method not populated"
            stripe_email = "not_populated@zulip.com"
        else:
            raise BillingError(f"stripe_customer_id is None for {customer}")

        complimentary_access_plan_end_date = self.get_formatted_complimentary_access_plan_end_date(
            customer, status=CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END
        )
        complimentary_access_next_plan_name = self.get_complimentary_access_next_plan_name(customer)
        context: Dict[str, Any] = {
            "plan_name": plan.name,
            "has_active_plan": True,
            "free_trial": plan.is_free_trial(),
            "downgrade_at_end_of_cycle": downgrade_at_end_of_cycle,
            "downgrade_at_end_of_free_trial": downgrade_at_end_of_free_trial,
            "automanage_licenses": plan.automanage_licenses,
            "switch_to_annual_at_end_of_cycle": switch_to_annual_at_end_of_cycle,
            "switch_to_monthly_at_end_of_cycle": switch_to_monthly_at_end_of_cycle,
            "licenses": licenses,
            "licenses_at_next_renewal": licenses_at_next_renewal,
            "seat_count": seat_count,
            "exempt_from_license_number_check": customer.exempt_from_license_number_check,
            "renewal_date": renewal_date,
            "renewal_amount": cents_to_dollar_string(renewal_cents) if renewal_cents != 0 else None,
            "payment_method": payment_method,
            "charge_automatically": charge_automatically,
            "stripe_email": stripe_email,
            "CustomerPlan": CustomerPlan,
            "billing_frequency": billing_frequency,
            "fixed_price_plan": plan.fixed_price is not None,
            "price_per_license": price_per_license,
            "is_sponsorship_pending": customer.sponsorship_pending,
            "sponsorship_plan_name": self.get_sponsorship_plan_name(customer, is_self_hosted_billing),
            "discount_percent": plan.discount,
            "is_self_hosted_billing": is_self_hosted_billing,
            "complimentary_access_plan": complimentary_access_plan_end_date is not None,
            "complimentary_access_plan_end_date": complimentary_access_plan_end_date,
            "complimentary_access_next_plan_name": complimentary_access_next_plan_name,
            "using_min_licenses_for_plan": using_min_licenses_for_plan,
            "min_licenses_for_plan": min_licenses_for_plan,
            "pre_discount_renewal_cents": cents_to_dollar_string(pre_discount_renewal_cents),
            "flat_discount": format_money(customer.flat_discount),
            "discounted_months_left": customer.flat_discounted_months,
            "has_paid_invoice_for_free_trial": has_paid_invoice_for_free_trial,
            "free_trial_next_renewal_date_after_invoice_paid": free_trial_next_renewal_date_after_invoice_paid,
        }
        return context

    def get_billing_page_context(self) -> Dict[str, Any]:
        now: datetime = timezone_now()

        customer = self.get_customer()
        assert customer is not None

        plan = get_current_plan_by_customer(customer)
        assert plan is not None

        new_plan, last_ledger_entry = self.make_end_of_cycle_updates_if_needed(plan, now)
        if last_ledger_entry is None:
            return {"current_plan_downgraded": True}
        plan = new_plan if new_plan is not None else plan

        context = self.get_billing_context_from_plan(customer, plan, last_ledger_entry, now)

        next_plan = self.get_next_plan(plan)
        if next_plan is not None:
            next_plan_context = self.get_billing_context_from_plan(customer, next_plan, last_ledger_entry, now)
            keys = [
                "renewal_amount",
                "payment_method",
                "charge_automatically",
                "billing_frequency",
                "fixed_price_plan",
                "price_per_license",
                "discount_percent",
                "using_min_licenses_for_plan",
                "min_licenses_for_plan",
                "pre_discount_renewal_cents",
            ]

            for key in keys:
                context[key] = next_plan_context[key]
        return context

    def get_flat_discount_info(self, customer: Optional["Customer"] = None) -> Tuple[int, int]:
        is_self_hosted_billing = not isinstance(self, RealmBillingSession)
        flat_discount = 0
        flat_discounted_months = 0
        if is_self_hosted_billing and (customer is None or customer.flat_discounted_months > 0):
            if customer is None:
                temp_customer = Customer()
                flat_discount = temp_customer.flat_discount
                flat_discounted_months = 12
            else:
                flat_discount = customer.flat_discount
                flat_discounted_months = customer.flat_discounted_months
            assert isinstance(flat_discount, int)
            assert isinstance(flat_discounted_months, int)
        return flat_discount, flat_discounted_months

    def min_licenses_for_flat_discount_to_self_hosted_basic_plan(
        self, customer: Optional["Customer"], is_plan_free_trial_with_invoice_payment: bool = False
    ) -> int:
        price_per_license = get_price_per_license(CustomerPlan.TIER_SELF_HOSTED_BASIC, CustomerPlan.BILLING_SCHEDULE_MONTHLY)
        if customer is None or is_plan_free_trial_with_invoice_payment:
            return (Customer._meta.get_field("flat_discount").get_default() // price_per_license) + 1
        elif customer.flat_discounted_months > 0:
            return (customer.flat_discount // price_per_license) + 1
        return 1

    def min_licenses_for_plan(self, tier: int, is_plan_free_trial_with_invoice_payment: bool = False) -> int:
        customer = self.get_customer()
        if customer is not None and customer.minimum_licenses:
            assert customer.monthly_discounted_price or customer.annual_discounted_price
            return customer.minimum_licenses

        if tier == CustomerPlan.TIER_SELF_HOSTED_BASIC:
            return min(self.min_licenses_for_flat_discount_to_self_hosted_basic_plan(customer, is_plan_free_trial_with_invoice_payment), 10)
        if tier == CustomerPlan.TIER_SELF_HOSTED_BUSINESS:
            return 25

        if tier == CustomerPlan.TIER_CLOUD_PLUS:
            return 10
        return 1

    def downgrade_at_the_end_of_billing_cycle(self, plan: Optional["CustomerPlan"] = None) -> None:
        if plan is None:
            customer = self.get_customer()
            assert customer is not None
            plan = get_current_plan_by_customer(customer)
        assert plan is not None
        do_change_plan_status(plan, CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE)

    def void_all_open_invoices(self) -> int:
        customer = self.get_customer()
        if customer is None:
            return 0
        invoices = get_all_invoices_for_customer(customer)
        voided_invoices_count = 0
        for invoice in invoices:
            if invoice.status == "open":
                assert invoice.id is not None
                stripe.Invoice.void_invoice(invoice.id)
                voided_invoices_count += 1
        return voided_invoices_count

    def downgrade_now_without_creating_additional_invoices(
        self, plan: Optional["CustomerPlan"] = None, background_update: bool = False
    ) -> None:
        if plan is None:
            customer = self.get_customer()
            assert customer is not None
            plan = get_current_plan_by_customer(customer)
            if plan is None:
                return
        self.process_downgrade(plan, background_update=background_update)
        plan.invoiced_through = LicenseLedger.objects.filter(plan=plan).order_by("id").last()
        plan.next_invoice_date = next_invoice_date(plan)
        plan.save(update_fields=["invoiced_through", "next_invoice_date"])

    def do_update_plan(self, update_plan_request: UpdatePlanRequest) -> None:
        customer = self.get_customer()
        assert customer is not None
        plan = get_current_plan_by_customer(customer)
        assert plan is not None

        new_plan, last_ledger_entry = self.make_end_of_cycle_updates_if_needed(plan, timezone_now())
        if new_plan is not None:
            raise JsonableError(
                "Unable to update the plan. The plan has been expired and replaced with a new plan."
            )

        if last_ledger_entry is None:
            raise JsonableError("Unable to update the plan. The plan has ended.")

        if update_plan_request.toggle_license_management:
            assert update_plan_request.status is None
            assert update_plan_request.licenses is None
            assert update_plan_request.licenses_at_next_renewal is None
            assert update_plan_request.schedule is None

            plan.automanage_licenses = not plan.automanage_licenses
            plan.save(update_fields=["automanage_licenses"])
            return

        status = update_plan_request.status
        if status is not None:
            if status == CustomerPlan.ACTIVE:
                assert plan.status < CustomerPlan.LIVE_STATUS_THRESHOLD
                with transaction.atomic(durable=True):
                    if plan.status == CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END:
                        next_plan = self.get_next_plan(plan)
                        assert next_plan is not None
                        do_change_plan_status(next_plan, CustomerPlan.ENDED)
                    do_change_plan_status(plan, status)
            elif status == CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE:
                assert not plan.is_free_trial()
                assert plan.status < CustomerPlan.LIVE_STATUS_THRESHOLD
                self.downgrade_at_the_end_of_billing_cycle(plan=plan)
            elif status == CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE:
                assert plan.billing_schedule == CustomerPlan.BILLING_SCHEDULE_MONTHLY
                assert plan.status < CustomerPlan.LIVE_STATUS_THRESHOLD
                assert plan.status != CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE
                assert not plan.is_free_trial()
                assert plan.fixed_price is None
                do_change_plan_status(plan, status)
            elif status == CustomerPlan.SWITCH_TO_MONTHLY_AT_END_OF_CYCLE:
                assert plan.billing_schedule == CustomerPlan.BILLING_SCHEDULE_ANNUAL
                assert plan.status < CustomerPlan.LIVE_STATUS_THRESHOLD
                assert plan.status != CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE
                assert not plan.is_free_trial()
                assert plan.fixed_price is None
                do_change_plan_status(plan, status)
            elif status == CustomerPlan.ENDED:
                assert plan.is_free_trial()
                self.downgrade_now_without_creating_additional_invoices(plan=plan)
            elif status == CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL:
                assert plan.is_free_trial()
                assert plan.charge_automatically
                do_change_plan_status(plan, status)
            elif status == CustomerPlan.FREE_TRIAL:
                assert plan.charge_automatically
                if update_plan_request.schedule is not None:
                    self.do_change_schedule_after_free_trial(plan, update_plan_request.schedule)
                else:
                    assert plan.status == CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL
                    do_change_plan_status(plan, status)
            return

        licenses = update_plan_request.licenses
        if licenses is not None:
            if plan.is_free_trial():
                raise JsonableError("Cannot update licenses in the current billing period for free trial plan.")
            if plan.automanage_licenses:
                raise JsonableError("Unable to update licenses manually. Your plan is on automatic license management.")
            if last_ledger_entry.licenses == licenses:
                raise JsonableError("Your plan is already on {licenses} licenses in the current billing period.".format(licenses=licenses))
            if last_ledger_entry.licenses > licenses:
                raise JsonableError("You cannot decrease the licenses in the current billing period.")
            validate_licenses(
                plan.charge_automatically,
                licenses,
                self.current_count_for_billed_licenses(),
                plan.customer.exempt_from_license_number_check,
                self.min_licenses_for_plan(plan.tier),
            )
            self.update_license_ledger_for_manual_plan(plan, timezone_now(), licenses=licenses)
            return

        licenses_at_next_renewal = update_plan_request.licenses_at_next_renewal
        if licenses_at_next_renewal is not None:
            if plan.automanage_licenses:
                raise JsonableError("Unable to update licenses manually. Your plan is on automatic license management.")
            if plan.status in (CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE, CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL):
                raise JsonableError("Cannot change the licenses for next billing cycle for a plan that is being downgraded.")
            if last_ledger_entry.licenses_at_next_renewal == licenses_at_next_renewal:
                raise JsonableError("Your plan is already scheduled to renew with {licenses_at_next_renewal} licenses.".format(licenses_at_next_renewal=licenses_at_next_renewal))
            is_plan_free_trial_with_invoice_payment = plan.is_free_trial() and not plan.charge_automatically
            validate_licenses(
                plan.charge_automatically,
                licenses_at_next_renewal,
                self.current_count_for_billed_licenses(),
                plan.customer.exempt_from_license_number_check,
                self.min_licenses_for_plan(plan.tier, is_plan_free_trial_with_invoice_payment),
            )

            if is_plan_free_trial_with_invoice_payment:
                invoice = Invoice.objects.filter(plan=plan).order_by("-id").first()
                assert invoice is not None
                if invoice.status == Invoice.PAID:
                    assert last_ledger_entry.licenses_at_next_renewal is not None
                    if last_ledger_entry.licenses_at_next_renewal > licenses_at_next_renewal:
                        raise JsonableError("You’ve already purchased {licenses_at_next_renewal} licenses for the next billing period.".format(licenses_at_next_renewal=last_ledger_entry.licenses_at_next_renewal))
                    else:
                        self.update_license_ledger_for_manual_plan(plan, timezone_now(), licenses_at_next_renewal=licenses_at_next_renewal)
                else:
                    self.update_free_trial_invoice_with_licenses(plan, timezone_now(), licenses_at_next_renewal)
            else:
                self.update_license_ledger_for_manual_plan(plan, timezone_now(), licenses_at_next_renewal=licenses_at_next_renewal)
            return

        raise JsonableError("Nothing to change.")

    def switch_plan_tier(self, current_plan: "CustomerPlan", new_plan_tier: int) -> None:
        assert current_plan.status == CustomerPlan.SWITCH_PLAN_TIER_NOW
        assert current_plan.next_invoice_date is not None
        next_billing_cycle = current_plan.next_invoice_date

        current_plan.end_date = next_billing_cycle
        current_plan.status = CustomerPlan.ENDED
        current_plan.save(update_fields=["status", "end_date"])

        new_price_per_license, discount_for_new_plan_tier = get_price_per_license_and_discount(
            new_plan_tier, current_plan.billing_schedule, current_plan.customer
        )

        new_plan_billing_cycle_anchor = current_plan.end_date.replace(microsecond=0)

        new_plan = CustomerPlan.objects.create(
            customer=current_plan.customer,
            status=CustomerPlan.ACTIVE,
            automanage_licenses=current_plan.automanage_licenses,
            charge_automatically=current_plan.charge_automatically,
            price_per_license=new_price_per_license,
            discount=discount_for_new_plan_tier,
            billing_schedule=current_plan.billing_schedule,
            tier=new_plan_tier,
            billing_cycle_anchor=new_plan_billing_cycle_anchor,
            next_invoice_date=new_plan_billing_cycle_anchor,
        )

        current_plan_last_ledger = LicenseLedger.objects.filter(plan=current_plan).order_by("id").last()
        assert current_plan_last_ledger is not None

        old_plan_licenses_at_next_renewal = current_plan_last_ledger.licenses_at_next_renewal
        assert old_plan_licenses_at_next_renewal is not None
        licenses_for_new_plan = self.get_billable_licenses_for_customer(
            current_plan.customer, new_plan_tier, old_plan_licenses_at_next_renewal
        )
        if not new_plan.automanage_licenses:
            licenses_for_new_plan = max(old_plan_licenses_at_next_renewal, licenses_for_new_plan)

        assert licenses_for_new_plan is not None
        LicenseLedger.objects.create(
            plan=new_plan,
            is_renewal=True,
            event_time=new_plan_billing_cycle_anchor,
            licenses=licenses_for_new_plan,
            licenses_at_next_renewal=licenses_for_new_plan,
        )

    def invoice_plan(self, plan: "CustomerPlan", event_time: datetime) -> None:
        if plan.invoicing_status == CustomerPlan.INVOICING_STATUS_STARTED:
            raise NotImplementedError("Plan with invoicing_status==STARTED needs manual resolution.")
        if (plan.tier != CustomerPlan.TIER_SELF_HOSTED_LEGACY and not plan.customer.stripe_customer_id):
            raise BillingError(f"Customer has a paid plan without a Stripe customer ID: {plan.customer!s}")

        if plan.status is not CustomerPlan.SWITCH_PLAN_TIER_NOW:
            self.make_end_of_cycle_updates_if_needed(plan, event_time)

        if plan.is_a_paid_plan():
            assert plan.customer.stripe_customer_id is not None
            if plan.invoicing_status == CustomerPlan.INVOICING_STATUS_INITIAL_INVOICE_TO_BE_SENT:
                invoiced_through_id = -1
                licenses_base: Optional[int] = None
            else:
                assert plan.invoiced_through is not None
                licenses_base = plan.invoiced_through.licenses
                invoiced_through_id = plan.invoiced_through.id

            invoice_item_created = False
            invoice_period: Optional[Dict[str, Union[int, float]]] = None
            for ledger_entry in LicenseLedger.objects.filter(
                plan=plan, id__gt=invoiced_through_id, event_time__lte=event_time
            ).order_by("id"):
                price_args: Dict[str, Union[int, float]] = {}
                if ledger_entry.is_renewal:
                    if plan.fixed_price is not None:
                        amount_due = get_amount_due_fixed_price_plan(plan.fixed_price, plan.billing_schedule)
                        price_args = {"amount": amount_due}
                    else:
                        assert plan.price_per_license is not None
                        price_args = {"unit_amount": plan.price_per_license, "quantity": ledger_entry.licenses}
                    description = f"{plan.name} - renewal"
                elif (plan.fixed_price is None and licenses_base is not None and ledger_entry.licenses != licenses_base):
                    assert plan.price_per_license is not None
                    last_ledger_entry_renewal = LicenseLedger.objects.filter(plan=plan, is_renewal=True, event_time__lte=ledger_entry.event_time).order_by("-id").first()
                    assert last_ledger_entry_renewal is not None
                    last_renewal = last_ledger_entry_renewal.event_time
                    billing_period_end = start_of_next_billing_cycle(plan, ledger_entry.event_time)
                    plan_renewal_or_end_date = get_plan_renewal_or_end_date(plan, ledger_entry.event_time)
                    unit_amount = plan.price_per_license
                    if not plan.is_free_trial():
                        proration_fraction = (plan_renewal_or_end_date - ledger_entry.event_time) / (billing_period_end - last_renewal)
                        unit_amount = int(plan.price_per_license * proration_fraction + 0.5)
                    price_args = {"unit_amount": unit_amount, "quantity": ledger_entry.licenses - licenses_base}
                    description = "Additional license ({} - {})".format(
                        ledger_entry.event_time.strftime("%b %-d, %Y"), plan_renewal_or_end_date.strftime("%b %-d, %Y")
                    )

                if price_args:
                    plan.invoiced_through = ledger_entry
                    plan.invoicing_status = CustomerPlan.INVOICING_STATUS_STARTED
                    plan.save(update_fields=["invoicing_status", "invoiced_through"])
                    invoice_period = {
                        "start": datetime_to_timestamp(ledger_entry.event_time),
                        "end": datetime_to_timestamp(get_plan_renewal_or_end_date(plan, ledger_entry.event_time)),
                    }
                    stripe.InvoiceItem.create(
                        currency="usd",
                        customer=plan.customer.stripe_customer_id,
                        description=description,
                        discountable=False,
                        period=invoice_period,
                        idempotency_key=get_idempotency_key(ledger_entry),
                        **price_args,
                    )
                    invoice_item_created = True
                plan.invoiced_through = ledger_entry
                plan.invoicing_status = CustomerPlan.INVOICING_STATUS_DONE
                plan.save(update_fields=["invoicing_status", "invoiced_through"])
                licenses_base = ledger_entry.licenses

            if invoice_item_created:
                assert invoice_period is not None
                flat_discount, flat_discounted_months = self.get_flat_discount_info(plan.customer)
                if plan.fixed_price is None and flat_discounted_months > 0:
                    num_months = 12 if plan.billing_schedule == CustomerPlan.BILLING_SCHEDULE_ANNUAL else 1
                    flat_discounted_months = min(flat_discounted_months, num_months)
                    discount = flat_discount * flat_discounted_months
                    plan.customer.flat_discounted_months -= flat_discounted_months
                    plan.customer.save(update_fields=["flat_discounted_months"])
                    stripe.InvoiceItem.create(
                        currency="usd",
                        customer=plan.customer.stripe_customer_id,
                        description=f"${cents_to_dollar_string(flat_discount)}/month new customer discount",
                        amount=(-1 * discount),
                        period=invoice_period,
                    )

                if plan.charge_automatically:
                    collection_method: str = "charge_automatically"
                    days_until_due: Optional[int] = None
                else:
                    collection_method = "send_invoice"
                    days_until_due = DEFAULT_INVOICE_DAYS_UNTIL_DUE
                invoice_params = stripe.Invoice.CreateParams(
                    auto_advance=True,
                    collection_method=collection_method,
                    customer=plan.customer.stripe_customer_id,
                    statement_descriptor=plan.name,
                )
                if days_until_due is not None:
                    invoice_params["days_until_due"] = days_until_due
                stripe_invoice = stripe.Invoice.create(**invoice_params)
                stripe.Invoice.finalize_invoice(stripe_invoice)

        plan.next_invoice_date = next_invoice_date(plan)
        plan.invoice_overdue_email_sent = False
        plan.save(update_fields=["next_invoice_date", "invoice_overdue_email_sent"])

    def do_change_plan_to_new_tier(self, new_plan_tier: int) -> str:
        customer = self.get_customer()
        assert customer is not None
        current_plan = get_current_plan_by_customer(customer)

        if not current_plan or current_plan.status != CustomerPlan.ACTIVE:
            raise BillingError("Organization does not have an active plan")

        if not current_plan.customer.stripe_customer_id:
            raise BillingError("Organization missing Stripe customer.")

        type_of_tier_change = self.get_type_of_plan_tier_change(current_plan.tier, new_plan_tier)

        if type_of_tier_change == PlanTierChangeType.INVALID:
            raise BillingError("Invalid change of customer plan tier.")

        if type_of_tier_change == PlanTierChangeType.UPGRADE:
            plan_switch_time = timezone_now()
            current_plan.status = CustomerPlan.SWITCH_PLAN_TIER_NOW
            current_plan.next_invoice_date = plan_switch_time
            current_plan.save(update_fields=["status", "next_invoice_date"])

            self.do_change_plan_type(tier=new_plan_tier)

            amount_to_credit_for_early_termination = get_amount_to_credit_for_plan_tier_change(
                current_plan, plan_switch_time
            )
            stripe.Customer.create_balance_transaction(
                current_plan.customer.stripe_customer_id,
                amount=-1 * amount_to_credit_for_early_termination,
                currency="usd",
                description="Credit from early termination of active plan",
            )
            self.switch_plan_tier(current_plan, new_plan_tier)
            self.invoice_plan(current_plan, plan_switch_time)
            new_plan = get_current_plan_by_customer(customer)
            assert new_plan is not None
            self.invoice_plan(new_plan, plan_switch_time)
            return f"{self.billing_entity_display_name} upgraded to {new_plan.name}"

        assert type_of_tier_change == PlanTierChangeType.DOWNGRADE
        return ""

    def get_event_status(self, event_status_request: EventStatusRequest) -> Dict[str, Any]:
        customer = self.get_customer()

        if customer is None:
            raise JsonableError("No customer for this organization!")

        stripe_session_id = event_status_request.stripe_session_id
        if stripe_session_id is not None:
            try:
                session = Session.objects.get(stripe_session_id=stripe_session_id, customer=customer)
            except Session.DoesNotExist:
                raise JsonableError("Session not found")

            if (session.type == Session.CARD_UPDATE_FROM_BILLING_PAGE and not self.has_billing_access()):
                raise JsonableError("Must be a billing administrator or an organization owner")
            return {"session": session.to_dict()}

        stripe_invoice_id = event_status_request.stripe_invoice_id
        if stripe_invoice_id is not None:
            stripe_invoice = Invoice.objects.filter(stripe_invoice_id=stripe_invoice_id, customer=customer).last()

            if stripe_invoice is None:
                raise JsonableError("Payment intent not found")
            return {"stripe_invoice": stripe_invoice.to_dict()}

        raise JsonableError("Pass stripe_session_id or stripe_invoice_id")

    def get_sponsorship_plan_name(self, customer: Optional["Customer"], is_remotely_hosted: bool) -> str:
        if customer is not None and customer.sponsorship_pending:
            sponsorship_request = ZulipSponsorshipRequest.objects.filter(customer=customer).order_by("-id").first()
            if sponsorship_request is not None and sponsorship_request.requested_plan not in (None, SponsoredPlanTypes.UNSPECIFIED.value):
                return sponsorship_request.requested_plan

        sponsored_plan_name = CustomerPlan.name_from_tier(CustomerPlan.TIER_CLOUD_STANDARD)
        if is_remotely_hosted:
            sponsored_plan_name = CustomerPlan.name_from_tier(CustomerPlan.TIER_SELF_HOSTED_COMMUNITY)

        return sponsored_plan_name

    def get_sponsorship_request_context(self) -> Optional[Dict[str, Any]]:
        customer = self.get_customer()

        if customer is not None and customer.sponsorship_pending and self.on_paid_plan():
            return None

        is_remotely_hosted = isinstance(self, (RemoteRealmBillingSession, RemoteServerBillingSession))
        plan_name = "Free" if is_remotely_hosted else "Zulip Cloud Free"

        context: Dict[str, Any] = {
            "billing_base_url": self.billing_base_url,
            "is_remotely_hosted": is_remotely_hosted,
            "sponsorship_plan_name": self.get_sponsorship_plan_name(customer, is_remotely_hosted),
            "plan_name": plan_name,
            "org_name": self.org_name(),
        }

        if self.is_sponsored():
            context["is_sponsored"] = True

        if customer is not None:
            context["is_sponsorship_pending"] = customer.sponsorship_pending
            plan = get_current_plan_by_customer(customer)
            if plan is not None:
                context["plan_name"] = plan.name
                context["complimentary_access"] = plan.is_complimentary_access_plan()

        self.add_org_type_data_to_sponsorship_context(context)
        return context

    def request_sponsorship(self, form: "SponsorshipRequestForm") -> None:
        if not form.is_valid():
            message = " ".join(
                error["message"]
                for error_list in form.errors.get_json_data().values()
                for error in error_list
            )
            raise BillingError("Form validation error", message=message)

        request_context = self.get_sponsorship_request_session_specific_context()
        with transaction.atomic(durable=True):
            self.update_customer_sponsorship_status(True)
            sponsorship_request = ZulipSponsorshipRequest(
                customer=self.get_customer(),
                requested_by=request_context["realm_user"],
                org_website=form.cleaned_data["website"],
                org_description=form.cleaned_data["description"],
                org_type=form.cleaned_data["organization_type"],
                expected_total_users=form.cleaned_data["expected_total_users"],
                plan_to_use_zulip=form.cleaned_data["plan_to_use_zulip"],
                paid_users_count=form.cleaned_data["paid_users_count"],
                paid_users_description=form.cleaned_data["paid_users_description"],
                requested_plan=form.cleaned_data["requested_plan"],
            )
            sponsorship_request.save()

            org_type = form.cleaned_data["organization_type"]
            self.save_org_type_from_request_sponsorship_session(org_type)

            if request_context["realm_user"] is not None:
                from zerver.actions.users import do_change_is_billing_admin

                do_change_is_billing_admin(request_context["realm_user"], True)

            org_type_display_name = get_org_type_display_name(org_type)

        user_info = request_context["user_info"]
        support_url = self.support_url()
        context = {
            "requested_by": user_info["name"],
            "user_role": user_info["role"],
            "billing_entity": self.billing_entity_display_name,
            "support_url": support_url,
            "organization_type": org_type_display_name,
            "website": sponsorship_request.org_website,
            "description": sponsorship_request.org_description,
            "expected_total_users": sponsorship_request.expected_total_users,
            "plan_to_use_zulip": sponsorship_request.plan_to_use_zulip,
            "paid_users_count": sponsorship_request.paid_users_count,
            "paid_users_description": sponsorship_request.paid_users_description,
            "requested_plan": sponsorship_request.requested_plan,
            "is_cloud_organization": isinstance(self, RealmBillingSession),
        }
        send_email(
            "zerver/emails/sponsorship_request",
            to_emails=[BILLING_SUPPORT_EMAIL],
            from_name="Zulip sponsorship request",
            from_address=FromAddress.tokenized_no_reply_address(),
            reply_to_email=user_info["email"],
            context=context,
        )

    def process_support_view_request(self, support_request: Dict[str, Any]) -> str:
        support_type = support_request["support_type"]
        success_message = ""

        if support_type == SupportType.approve_sponsorship:
            success_message = self.approve_sponsorship()
        elif support_type == SupportType.update_sponsorship_status:
            assert support_request["sponsorship_status"] is not None
            sponsorship_status = support_request["sponsorship_status"]
            success_message = self.update_customer_sponsorship_status(sponsorship_status)
        elif support_type == SupportType.attach_discount:
            monthly_discounted_price = support_request["monthly_discounted_price"]
            annual_discounted_price = support_request["annual_discounted_price"]
            assert monthly_discounted_price is not None
            assert annual_discounted_price is not None
            success_message = self.attach_discount_to_customer(monthly_discounted_price, annual_discounted_price)
        elif support_type == SupportType.update_minimum_licenses:
            assert support_request["minimum_licenses"] is not None
            new_minimum_license_count = support_request["minimum_licenses"]
            success_message = self.update_customer_minimum_licenses(new_minimum_license_count)
        elif support_type == SupportType.update_required_plan_tier:
            required_plan_tier = support_request.get("required_plan_tier")
            assert required_plan_tier is not None
            success_message = self.set_required_plan_tier(required_plan_tier)
        elif support_type == SupportType.configure_fixed_price_plan:
            assert support_request["fixed_price"] is not None
            new_fixed_price = support_request["fixed_price"]
            sent_invoice_id = support_request["sent_invoice_id"]
            success_message = self.configure_fixed_price_plan(new_fixed_price, sent_invoice_id)
        elif support_type == SupportType.configure_complimentary_access_plan:
            assert support_request["plan_end_date"] is not None
            temporary_plan_end_date = support_request["plan_end_date"]
            success_message = self.configure_complimentary_access_plan(temporary_plan_end_date)
        elif support_type == SupportType.update_billing_modality:
            assert support_request["billing_modality"] is not None
            charge_automatically = support_request["billing_modality"] == "charge_automatically"
            success_message = self.update_billing_modality_of_current_plan(charge_automatically)
        elif support_type == SupportType.update_plan_end_date:
            assert support_request["plan_end_date"] is not None
            new_plan_end_date = support_request["plan_end_date"]
            success_message = self.update_end_date_of_current_plan(new_plan_end_date)
        elif support_type == SupportType.modify_plan:
            assert support_request["plan_modification"] is not None
            plan_modification = support_request["plan_modification"]
            if plan_modification == "downgrade_at_billing_cycle_end":
                self.downgrade_at_the_end_of_billing_cycle()
                success_message = f"{self.billing_entity_display_name} marked for downgrade at the end of billing cycle"
            elif plan_modification == "downgrade_now_without_additional_licenses":
                self.downgrade_now_without_creating_additional_invoices()
                success_message = f"{self.billing_entity_display_name} downgraded without creating additional invoices"
            elif plan_modification == "downgrade_now_void_open_invoices":
                self.downgrade_now_without_creating_additional_invoices()
                voided_invoices_count = self.void_all_open_invoices()
                success_message = f"{self.billing_entity_display_name} downgraded and voided {voided_invoices_count} open invoices"
            else:
                assert plan_modification == "upgrade_plan_tier"
                assert support_request["new_plan_tier"] is not None
                new_plan_tier = support_request["new_plan_tier"]
                success_message = self.do_change_plan_to_new_tier(new_plan_tier)
        elif support_type == SupportType.delete_fixed_price_next_plan:
            success_message = self.delete_fixed_price_plan()

        return success_message

    def update_free_trial_invoice_with_licenses(self, plan: "CustomerPlan", event_time: datetime, licenses: int) -> None:
        assert self.get_billable_licenses_for_customer(plan.customer, plan.tier, licenses) <= licenses
        last_sent_invoice = Invoice.objects.filter(plan=plan).order_by("-id").first()
        assert last_sent_invoice is not None
        assert last_sent_invoice.status == Invoice.SENT

        assert not plan.automanage_licenses
        assert not plan.charge_automatically
        assert plan.fixed_price is None
        assert plan.is_free_trial()

        LicenseLedger.objects.create(
            plan=plan,
            is_renewal=True,
            event_time=event_time,
            licenses=licenses,
            licenses_at_next_renewal=licenses,
        )

        stripe_invoice = stripe.Invoice.retrieve(last_sent_invoice.stripe_invoice_id)
        assert stripe_invoice.status == "open"
        assert isinstance(stripe_invoice.customer, str)
        assert stripe_invoice.statement_descriptor is not None
        assert stripe_invoice.metadata is not None
        invoice_items = stripe_invoice.lines.data
        invoice_items.reverse()
        for invoice_item in invoice_items:
            assert invoice_item.description is not None
            price_args: Dict[str, Union[int, float]] = {}
            if invoice_item.amount > 0:
                assert invoice_item.price is not None
                assert invoice_item.price.unit_amount is not None
                price_args = {"quantity": licenses, "unit_amount": invoice_item.price.unit_amount}
            else:
                price_args = {"amount": invoice_item.amount}
            stripe.InvoiceItem.create(
                currency=invoice_item.currency,
                customer=stripe_invoice.customer,
                description=invoice_item.description,
                period={"start": invoice_item.period.start, "end": invoice_item.period.end},
                **price_args,
            )

        assert plan.next_invoice_date is not None
        days_until_due = (plan.next_invoice_date - event_time).days

        new_stripe_invoice = stripe.Invoice.create(
            auto_advance=False,
            collection_method="send_invoice",
            customer=stripe_invoice.customer,
            days_until_due=days_until_due,
            statement_descriptor=stripe_invoice.statement_descriptor,
            metadata=stripe_invoice.metadata,
        )
        new_stripe_invoice = stripe.Invoice.finalize_invoice(new_stripe_invoice)
        last_sent_invoice.stripe_invoice_id = str(new_stripe_invoice.id)
        last_sent_invoice.save(update_fields=["stripe_invoice_id"])

        assert stripe_invoice.id is not None
        stripe.Invoice.void_invoice(stripe_invoice.id)

    def update_license_ledger_for_manual_plan(
        self, plan: "CustomerPlan", event_time: datetime, licenses: Optional[int] = None, licenses_at_next_renewal: Optional[int] = None
    ) -> None:
        if licenses is not None:
            if not plan.customer.exempt_from_license_number_check:
                assert self.current_count_for_billed_licenses() <= licenses
            assert licenses > plan.licenses()
            LicenseLedger.objects.create(
                plan=plan,
                event_time=event_time,
                licenses=licenses,
                licenses_at_next_renewal=licenses,
            )
        elif licenses_at_next_renewal is not None:
            assert self.get_billable_licenses_for_customer(plan.customer, plan.tier, licenses_at_next_renewal) <= licenses_at_next_renewal
            LicenseLedger.objects.create(
                plan=plan,
                event_time=event_time,
                licenses=plan.licenses(),
                licenses_at_next_renewal=licenses_at_next_renewal,
            )
        else:
            raise AssertionError("Pass licenses or licenses_at_next_renewal")

    def get_billable_licenses_for_customer(
        self, customer: "Customer", tier: int, licenses: Optional[int] = None, event_time: Optional[datetime] = None
    ) -> int:
        if licenses is not None and customer.exempt_from_license_number_check:
            return licenses

        current_licenses_count = self.current_count_for_billed_licenses(event_time)
        min_licenses_for_plan = self.min_licenses_for_plan(tier)
        if customer.exempt_from_license_number_check:
            billed_licenses = current_licenses_count
        else:
            billed_licenses = max(current_licenses_count, min_licenses_for_plan)
        return billed_licenses

    def update_license_ledger_for_automanaged_plan(self, plan: "CustomerPlan", event_time: datetime) -> Optional["CustomerPlan"]:
        new_plan, last_ledger_entry = self.make_end_of_cycle_updates_if_needed(plan, event_time)
        if last_ledger_entry is None:
            return None
        if new_plan is not None:
            plan = new_plan

        if plan.status == CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END:
            next_plan = self.get_next_plan(plan)
            assert next_plan is not None
            licenses_at_next_renewal = self.get_billable_licenses_for_customer(plan.customer, next_plan.tier, event_time=event_time)
            current_plan_licenses_at_next_renewal = self.get_billable_licenses_for_customer(plan.customer, plan.tier, event_time=event_time)
            licenses = max(current_plan_licenses_at_next_renewal, last_ledger_entry.licenses)
        else:
            licenses_at_next_renewal = self.get_billable_licenses_for_customer(plan.customer, plan.tier, event_time=event_time)
            licenses = max(licenses_at_next_renewal, last_ledger_entry.licenses)

        LicenseLedger.objects.create(
            plan=plan,
            event_time=event_time,
            licenses=licenses,
            licenses_at_next_renewal=licenses_at_next_renewal,
        )

        return plan

    def create_complimentary_access_plan(self, renewal_date: datetime, end_date: datetime) -> None:
        plan_tier = CustomerPlan.TIER_SELF_HOSTED_LEGACY
        if isinstance(self, RealmBillingSession):
            raise BillingError(f"Cannot currently configure a complimentary access plan for {self.billing_entity_display_name}.")
        customer = self.update_or_create_customer()

        complimentary_access_plan_anchor = renewal_date
        complimentary_access_plan_params: Dict[str, Any] = {
            "billing_cycle_anchor": complimentary_access_plan_anchor,
            "status": CustomerPlan.ACTIVE,
            "tier": plan_tier,
            "end_date": end_date,
            "next_invoice_date": end_date,
            "price_per_license": 0,
            "billing_schedule": CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            "automanage_licenses": True,
        }
        complimentary_access_plan = CustomerPlan.objects.create(
            customer=customer,
            **complimentary_access_plan_params,
        )

        try:
            billed_licenses = self.get_billable_licenses_for_customer(customer, complimentary_access_plan.tier)
        except MissingDataError:
            billed_licenses = 0

        ledger_entry = LicenseLedger.objects.create(
            plan=complimentary_access_plan,
            is_renewal=True,
            event_time=complimentary_access_plan_anchor,
            licenses=billed_licenses,
            licenses_at_next_renewal=billed_licenses,
        )
        complimentary_access_plan.invoiced_through = ledger_entry
        complimentary_access_plan.save(update_fields=["invoiced_through"])
        self.write_to_audit_log(
            event_type=BillingSessionEventType.CUSTOMER_PLAN_CREATED,
            event_time=complimentary_access_plan_anchor,
            extra_data=complimentary_access_plan_params,
        )

        self.do_change_plan_type(tier=CustomerPlan.TIER_SELF_HOSTED_LEGACY, is_sponsored=False)

    def add_customer_to_community_plan(self) -> None:
        assert not isinstance(self, RealmBillingSession)

        customer = self.update_or_create_customer()
        plan = get_current_plan_by_customer(customer)
        assert plan is None
        now = timezone_now()
        community_plan_params: Dict[str, Any] = {
            "billing_cycle_anchor": now,
            "status": CustomerPlan.ACTIVE,
            "tier": CustomerPlan.TIER_SELF_HOSTED_COMMUNITY,
            "next_invoice_date": None,
            "price_per_license": 0,
            "billing_schedule": CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            "automanage_licenses": True,
        }
        community_plan = CustomerPlan.objects.create(
            customer=customer,
            **community_plan_params,
        )

        try:
            billed_licenses = self.get_billable_licenses_for_customer(customer, community_plan.tier)
        except MissingDataError:
            billed_licenses = 0

        ledger_entry = LicenseLedger.objects.create(
            plan=community_plan,
            is_renewal=True,
            event_time=now,
            licenses=billed_licenses,
            licenses_at_next_renewal=billed_licenses,
        )
        community_plan.invoiced_through = ledger_entry
        community_plan.save(update_fields=["invoiced_through"])
        self.write_to_audit_log(
            event_type=BillingSessionEventType.CUSTOMER_PLAN_CREATED,
            event_time=now,
            extra_data=community_plan_params,
        )

    def get_last_ledger_for_automanaged_plan_if_exists(self) -> Optional["LicenseLedger"]:
        customer = self.get_customer()
        if customer is None:
            return None
        plan = get_current_plan_by_customer(customer)
        if plan is None:
            return None
        if not plan.automanage_licenses:
            return None

        last_ledger = LicenseLedger.objects.filter(plan=plan).order_by("id").last()
        assert last_ledger is not None
        return last_ledger

    def send_realm_created_internal_admin_message(self) -> None:
        channel: str = "signups"
        topic: str = "new organizations"
        support_url: str = self.support_url()
        organization_type: str = get_org_type_display_name(self.realm.org_type)
        message: str = f"[{self.realm.name}]({support_url}) ([{self.realm.display_subdomain}]({self.realm.url})). Organization type: {organization_type}"
        self.send_support_admin_realm_internal_message(channel, topic, message)


class RealmBillingSession(BillingSession):
    def __init__(self, user: Optional["UserProfile"] = None, realm: Optional["Realm"] = None, *, support_session: bool = False) -> None:
        self.user = user
        assert user is not None or realm is not None
        if support_session:
            assert user is not None and user.is_staff
        self.support_session = support_session

        if user is not None and realm is not None:
            assert user.is_staff or user.realm == realm
            self.realm = realm
        elif user is not None:
            self.realm = user.realm
        else:
            assert realm is not None
            self.realm = realm

    PAID_PLANS = [Realm.PLAN_TYPE_STANDARD, Realm.PLAN_TYPE_PLUS]

    @property
    def billing_entity_display_name(self) -> str:
        return self.realm.string_id

    @property
    def billing_session_url(self) -> str:
        return self.realm.url

    @property
    def billing_base_url(self) -> str:
        return ""

    def support_url(self) -> str:
        return build_support_url("support", self.realm.string_id)

    def get_customer(self) -> Optional["Customer"]:
        return get_customer_by_realm(self.realm)

    def get_email(self) -> str:
        assert self.user is not None
        return self.user.delivery_email

    def current_count_for_billed_licenses(self, event_time: Optional[datetime] = None) -> int:
        return get_latest_seat_count(self.realm)

    def get_audit_log_event(self, event_type: Any) -> int:
        if event_type is BillingSessionEventType.STRIPE_CUSTOMER_CREATED:
            return AuditLogEventType.STRIPE_CUSTOMER_CREATED
        elif event_type is BillingSessionEventType.STRIPE_CARD_CHANGED:
            return AuditLogEventType.STRIPE_CARD_CHANGED
        elif event_type is BillingSessionEventType.CUSTOMER_PLAN_CREATED:
            return AuditLogEventType.CUSTOMER_PLAN_CREATED
        elif event_type is BillingSessionEventType.DISCOUNT_CHANGED:
            return AuditLogEventType.REALM_DISCOUNT_CHANGED
        elif event_type is BillingSessionEventType.CUSTOMER_PROPERTY_CHANGED:
            return AuditLogEventType.CUSTOMER_PROPERTY_CHANGED
        elif event_type is BillingSessionEventType.SPONSORSHIP_APPROVED:
            return AuditLogEventType.REALM_SPONSORSHIP_APPROVED
        elif event_type is BillingSessionEventType.SPONSORSHIP_PENDING_STATUS_CHANGED:
            return AuditLogEventType.REALM_SPONSORSHIP_PENDING_STATUS_CHANGED
        elif event_type is BillingSessionEventType.BILLING_MODALITY_CHANGED:
            return AuditLogEventType.REALM_BILLING_MODALITY_CHANGED
        elif event_type is BillingSessionEventType.CUSTOMER_PLAN_PROPERTY_CHANGED:
            return AuditLogEventType.CUSTOMER_PLAN_PROPERTY_CHANGED
        elif event_type is BillingSessionEventType.CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN:
            return AuditLogEventType.CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN
        elif event_type is BillingSessionEventType.CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_PLAN:
            return AuditLogEventType.CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_PLAN
        else:
            raise BillingSessionAuditLogEventError(event_type)

    def write_to_audit_log(
        self,
        event_type: Any,
        event_time: datetime,
        *,
        background_update: bool = False,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        audit_log_event = self.get_audit_log_event(event_type)
        audit_log_data: Dict[str, Any] = {
            "realm": self.realm,
            "event_type": audit_log_event,
            "event_time": event_time,
        }

        if extra_data:
            audit_log_data["extra_data"] = extra_data

        if self.user is not None and not background_update:
            audit_log_data["acting_user"] = self.user

        RealmAuditLog.objects.create(**audit_log_data)

    def get_data_for_stripe_customer(self) -> StripeCustomerData:
        assert self.support_session is False
        metadata: Dict[str, Any] = {}
        metadata["realm_id"] = self.realm.id
        metadata["realm_str"] = self.realm.string_id
        realm_stripe_customer_data = StripeCustomerData(
            description=f"{self.realm.string_id} ({self.realm.name})",
            email=self.get_email(),
            metadata=metadata,
        )
        return realm_stripe_customer_data

    def update_data_for_checkout_session_and_invoice_payment(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        assert self.user is not None
        updated_metadata = dict(
            user_email=self.get_email(),
            realm_id=self.realm.id,
            realm_str=self.realm.string_id,
            user_id=self.user.id,
            **metadata,
        )
        return updated_metadata

    def update_or_create_customer(self, stripe_customer_id: Optional[str] = None, *, defaults: Optional[Dict[str, Any]] = None) -> "Customer":
        if stripe_customer_id is not None:
            assert self.support_session is False
            customer, created = Customer.objects.update_or_create(realm=self.realm, defaults={"stripe_customer_id": stripe_customer_id})
            from zerver.actions.users import do_change_is_billing_admin

            assert self.user is not None
            do_change_is_billing_admin(self.user, True)
            return customer
        else:
            customer, created = Customer.objects.update_or_create(realm=self.realm, defaults=defaults)
            return customer

    def do_change_plan_type(self, *, tier: Optional[int], is_sponsored: bool = False, background_update: bool = False) -> None:
        from zerver.actions.realm_settings import do_change_realm_plan_type

        if is_sponsored:
            plan_type = Realm.PLAN_TYPE_STANDARD_FREE
        elif tier == CustomerPlan.TIER_CLOUD_STANDARD:
            plan_type = Realm.PLAN_TYPE_STANDARD
        elif tier == CustomerPlan.TIER_CLOUD_PLUS:
            plan_type = Realm.PLAN_TYPE_PLUS
        else:
            raise AssertionError("Unexpected tier")

        acting_user = None
        if not background_update:
            acting_user = self.user

        do_change_realm_plan_type(self.realm, plan_type, acting_user=acting_user)

    def process_downgrade(self, plan: "CustomerPlan", background_update: bool = False) -> None:
        from zerver.actions.realm_settings import do_change_realm_plan_type

        acting_user = None
        if not background_update:
            acting_user = self.user

        assert plan.customer.realm is not None
        do_change_realm_plan_type(plan.customer.realm, Realm.PLAN_TYPE_LIMITED, acting_user=acting_user)
        plan.status = CustomerPlan.ENDED
        plan.save(update_fields=["status"])

    def approve_sponsorship(self) -> str:
        assert self.support_session
        customer = self.get_customer()
        if customer is not None:
            error_message = self.check_customer_not_on_paid_plan(customer)
            if error_message != "":
                raise SupportRequestError(error_message)

        from zerver.actions.message_send import internal_send_private_message

        if self.realm.deactivated:
            raise SupportRequestError("Realm has been deactivated")

        self.do_change_plan_type(tier=None, is_sponsored=True)
        if customer is not None and customer.sponsorship_pending:
            customer.sponsorship_pending = False
            customer.save(update_fields=["sponsorship_pending"])
            self.write_to_audit_log(event_type=BillingSessionEventType.SPONSORSHIP_APPROVED, event_time=timezone_now())
        notification_bot = get_system_bot(settings.NOTIFICATION_BOT, self.realm.id)
        for user in self.realm.get_human_billing_admin_and_realm_owner_users():
            with override_language(user.default_language):
                message = (
                    "Your organization's request for sponsored hosting has been approved! "
                    f"You have been upgraded to {CustomerPlan.name_from_tier(CustomerPlan.TIER_CLOUD_STANDARD)}, free of charge. :tada:\n\n"
                    "If you could [list Zulip as a sponsor on your website](/help/linking-to-zulip-website), "
                    "we would really appreciate it!"
                )
                internal_send_private_message(notification_bot, user, message)
        return f"Sponsorship approved for {self.billing_entity_display_name}; Emailed organization owners and billing admins."

    def is_sponsored(self) -> bool:
        return self.realm.plan_type == self.realm.PLAN_TYPE_STANDARD_FREE

    def get_metadata_for_stripe_update_card(self) -> Dict[str, str]:
        assert self.user is not None
        return {"type": "card_update", "user_id": str(self.user.id)}

    def get_upgrade_page_session_type_specific_context(self) -> Dict[str, Any]:
        assert self.user is not None
        return {
            "customer_name": self.realm.name,
            "email": self.get_email(),
            "is_demo_organization": self.realm.demo_organization_scheduled_deletion_date is not None,
            "demo_organization_scheduled_deletion_date": self.realm.demo_organization_scheduled_deletion_date,
            "is_self_hosting": False,
        }

    def check_plan_tier_is_billable(self, plan_tier: int) -> bool:
        implemented_plan_tiers = [CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.TIER_CLOUD_PLUS]
        return plan_tier in implemented_plan_tiers

    def get_type_of_plan_tier_change(self, current_plan_tier: int, new_plan_tier: int) -> "PlanTierChangeType":
        valid_plan_tiers = [CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.TIER_CLOUD_PLUS]
        if current_plan_tier not in valid_plan_tiers or new_plan_tier not in valid_plan_tiers or current_plan_tier == new_plan_tier:
            return PlanTierChangeType.INVALID
        if current_plan_tier == CustomerPlan.TIER_CLOUD_STANDARD and new_plan_tier == CustomerPlan.TIER_CLOUD_PLUS:
            return PlanTierChangeType.UPGRADE
        else:
            return PlanTierChangeType.DOWNGRADE

    def has_billing_access(self) -> bool:
        assert self.user is not None
        return self.user.has_billing_access

    def on_paid_plan(self) -> bool:
        return self.realm.plan_type in self.PAID_PLANS

    def org_name(self) -> str:
        return self.realm.name

    def add_org_type_data_to_sponsorship_context(self, context: Dict[str, Any]) -> None:
        context.update(
            realm_org_type=self.realm.org_type,
            sorted_org_types=sorted(
                ([org_type_name, org_type] for (org_type_name, org_type) in Realm.ORG_TYPES.items() if not org_type.get("hidden")),
                key=sponsorship_org_type_key_helper,
            ),
        )

    def get_sponsorship_request_session_specific_context(self) -> Dict[str, Any]:
        assert self.user is not None
        return {
            "realm_user": self.user,
            "user_info": {
                "name": self.user.full_name,
                "email": self.get_email(),
                "role": self.user.get_role_name(),
            },
            "realm_string_id": self.realm.string_id,
        }

    def save_org_type_from_request_sponsorship_session(self, org_type: int) -> None:
        if self.realm.org_type != org_type:
            self.realm.org_type = org_type
            self.realm.save(update_fields=["org_type"])

    def sync_license_ledger_if_needed(self) -> None:
        pass

    def send_realm_created_internal_admin_message(self) -> None:
        channel = "signups"
        topic = "new organizations"
        support_url = self.support_url()
        organization_type = get_org_type_display_name(self.realm.org_type)
        message = f"[{self.realm.name}]({support_url}) ([{self.realm.display_subdomain}]({self.realm.url})). Organization type: {organization_type}"
        self.send_support_admin_realm_internal_message(channel, topic, message)


class RemoteRealmBillingSession(BillingSession):
    def __init__(self, remote_realm: "RemoteRealm", remote_billing_user: Optional["RemoteRealmBillingUser"] = None, support_staff: Optional["UserProfile"] = None) -> None:
        self.remote_realm = remote_realm
        self.remote_billing_user = remote_billing_user
        self.support_staff = support_staff
        if support_staff is not None:
            assert support_staff.is_staff
            self.support_session = True
        else:
            self.support_session = False

    @property
    def billing_entity_display_name(self) -> str:
        return self.remote_realm.name

    @property
    def billing_session_url(self) -> str:
        return f"{settings.EXTERNAL_URI_SCHEME}{settings.SELF_HOSTING_MANAGEMENT_SUBDOMAIN}.{settings.EXTERNAL_HOST}/realm/{self.remote_realm.uuid}"

    @property
    def billing_base_url(self) -> str:
        return f"/realm/{self.remote_realm.uuid}"

    def support_url(self) -> str:
        return build_support_url("remote_servers_support", str(self.remote_realm.uuid))

    def get_customer(self) -> Optional["Customer"]:
        return get_customer_by_remote_realm(self.remote_realm)

    def get_email(self) -> str:
        assert self.remote_billing_user is not None
        return self.remote_billing_user.email

    def current_count_for_billed_licenses(self, event_time: Optional[datetime] = None) -> int:
        if has_stale_audit_log(self.remote_realm.server):
            raise MissingDataError
        remote_realm_counts = get_remote_realm_guest_and_non_guest_count(self.remote_realm, event_time)
        return remote_realm_counts.non_guest_user_count + remote_realm_counts.guest_user_count

    def missing_data_error_page(self, request: HttpRequest) -> HttpResponse:
        missing_data_context: Dict[str, Any] = {
            "remote_realm_session": True,
            "supports_remote_realms": self.remote_realm.server.last_api_feature_level is not None,
        }
        return render(request, "corporate/billing/server_not_uploading_data.html", context=missing_data_context)

    def get_audit_log_event(self, event_type: Any) -> int:
        if event_type is BillingSessionEventType.STRIPE_CUSTOMER_CREATED:
            return AuditLogEventType.STRIPE_CUSTOMER_CREATED
        elif event_type is BillingSessionEventType.STRIPE_CARD_CHANGED:
            return AuditLogEventType.STRIPE_CARD_CHANGED
        elif event_type is BillingSessionEventType.CUSTOMER_PLAN_CREATED:
            return AuditLogEventType.CUSTOMER_PLAN_CREATED
        elif event_type is BillingSessionEventType.DISCOUNT_CHANGED:
            return AuditLogEventType.REMOTE_SERVER_DISCOUNT_CHANGED
        elif event_type is BillingSessionEventType.CUSTOMER_PROPERTY_CHANGED:
            return AuditLogEventType.CUSTOMER_PROPERTY_CHANGED
        elif event_type is BillingSessionEventType.SPONSORSHIP_APPROVED:
            return AuditLogEventType.REMOTE_SERVER_SPONSORSHIP_APPROVED
        elif event_type is BillingSessionEventType.SPONSORSHIP_PENDING_STATUS_CHANGED:
            return AuditLogEventType.REMOTE_SERVER_SPONSORSHIP_PENDING_STATUS_CHANGED
        elif event_type is BillingSessionEventType.BILLING_MODALITY_CHANGED:
            return AuditLogEventType.REMOTE_SERVER_BILLING_MODALITY_CHANGED
        elif event_type is BillingSessionEventType.CUSTOMER_PLAN_PROPERTY_CHANGED:
            return AuditLogEventType.CUSTOMER_PLAN_PROPERTY_CHANGED
        elif event_type is BillingSessionEventType.BILLING_ENTITY_PLAN_TYPE_CHANGED:
            return AuditLogEventType.REMOTE_SERVER_PLAN_TYPE_CHANGED
        elif event_type is BillingSessionEventType.CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN:
            return AuditLogEventType.CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN
        elif event_type is BillingSessionEventType.CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_PLAN:
            return AuditLogEventType.CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_PLAN
        else:
            raise BillingSessionAuditLogEventError(event_type)

    def write_to_audit_log(self, event_type: Any, event_time: datetime, *, background_update: bool = False, extra_data: Optional[Dict[str, Any]] = None) -> None:
        audit_log_event = self.get_audit_log_event(event_type)
        log_data: Dict[str, Any] = {
            "server": self.remote_realm.server,
            "remote_realm": self.remote_realm,
            "event_type": audit_log_event,
            "event_time": event_time,
        }

        if not background_update:
            log_data.update({
                "acting_support_user": self.support_staff,
                "acting_remote_user": self.remote_billing_user,
            })

        RemoteRealmAuditLog.objects.create(**log_data)

    def get_data_for_stripe_customer(self) -> StripeCustomerData:
        assert self.support_session is False
        metadata: Dict[str, Any] = {}
        metadata["remote_realm_uuid"] = self.remote_realm.uuid
        metadata["remote_realm_host"] = str(self.remote_realm.host)
        realm_stripe_customer_data = StripeCustomerData(
            description=str(self.remote_realm),
            email=self.get_email(),
            metadata=metadata,
        )
        return realm_stripe_customer_data

    def update_data_for_checkout_session_and_invoice_payment(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        assert self.remote_billing_user is not None
        updated_metadata = dict(
            remote_realm_user_id=self.remote_billing_user.id,
            remote_realm_user_email=self.get_email(),
            remote_realm_host=self.remote_realm.host,
            **metadata,
        )
        return updated_metadata

    def update_or_create_customer(self, stripe_customer_id: Optional[str] = None, *, defaults: Optional[Dict[str, Any]] = None) -> "Customer":
        if stripe_customer_id is not None:
            assert self.support_session is False
            customer, created = Customer.objects.update_or_create(
                remote_realm=self.remote_realm, defaults={"stripe_customer_id": stripe_customer_id}
            )
        else:
            customer, created = Customer.objects.update_or_create(
                remote_realm=self.remote_realm, defaults=defaults
            )

        if created and not customer.annual_discounted_price and not customer.monthly_discounted_price:
            customer.flat_discounted_months = 12
            customer.save(update_fields=["flat_discounted_months"])

        return customer

    def do_change_plan_type(self, *, tier: Optional[int], is_sponsored: bool = False, background_update: bool = False) -> None:
        if is_sponsored:
            plan_type = RemoteRealm.PLAN_TYPE_COMMUNITY
            self.add_customer_to_community_plan()
        elif tier == CustomerPlan.TIER_SELF_HOSTED_BASIC:
            plan_type = RemoteRealm.PLAN_TYPE_BASIC
        elif tier == CustomerPlan.TIER_SELF_HOSTED_BUSINESS:
            plan_type = RemoteRealm.PLAN_TYPE_BUSINESS
        elif tier == CustomerPlan.TIER_SELF_HOSTED_LEGACY:
            plan_type = RemoteRealm.PLAN_TYPE_SELF_MANAGED_LEGACY
        else:
            raise AssertionError("Unexpected tier")

        old_plan_type = self.remote_realm.plan_type
        self.remote_realm.plan_type = plan_type
        self.remote_realm.save(update_fields=["plan_type"])
        self.write_to_audit_log(
            event_type=BillingSessionEventType.BILLING_ENTITY_PLAN_TYPE_CHANGED,
            event_time=timezone_now(),
            extra_data={"old_value": old_plan_type, "new_value": plan_type},
            background_update=background_update,
        )

    def approve_sponsorship(self) -> str:
        assert self.support_session
        customer = self.get_customer()
        if customer is not None:
            error_message = self.check_customer_not_on_paid_plan(customer)
            if error_message != "":
                raise SupportRequestError(error_message)

        if self.remote_realm.plan_type == RemoteRealm.PLAN_TYPE_SELF_MANAGED_LEGACY:
            plan = get_current_plan_by_customer(customer)
            if plan is not None:
                assert self.get_next_plan(plan) is None
                assert plan.tier == CustomerPlan.TIER_SELF_HOSTED_LEGACY
                plan.status = CustomerPlan.ENDED
                plan.save(update_fields=["status"])

        self.do_change_plan_type(tier=None, is_sponsored=True)
        if customer is not None and customer.sponsorship_pending:
            customer.sponsorship_pending = False
            customer.save(update_fields=["sponsorship_pending"])
            self.write_to_audit_log(event_type=BillingSessionEventType.SPONSORSHIP_APPROVED, event_time=timezone_now())
        emailed_string = ""
        billing_emails = list(
            RemoteRealmBillingUser.objects.filter(remote_realm_id=self.remote_realm.id).values_list("email", flat=True)
        )
        if len(billing_emails) > 0:
            send_email(
                "zerver/emails/sponsorship_approved_community_plan",
                to_emails=billing_emails,
                from_address=BILLING_SUPPORT_EMAIL,
                context={
                    "billing_entity": self.billing_entity_display_name,
                    "plans_link": "https://zulip.com/plans/#self-hosted",
                    "link_to_zulip": "https://zulip.com/help/linking-to-zulip-website",
                },
            )
            emailed_string = "Emailed existing billing users."
        else:
            emailed_string = "No billing users exist to email."

        return f"Sponsorship approved for {self.billing_entity_display_name}; " + emailed_string

    def is_sponsored(self) -> bool:
        return self.remote_realm.plan_type == self.remote_realm.PLAN_TYPE_COMMUNITY

    def get_metadata_for_stripe_update_card(self) -> Dict[str, str]:
        assert self.remote_billing_user is not None
        return {"type": "card_update", "remote_realm_user_id": str(self.remote_billing_user.id)}

    def get_upgrade_page_session_type_specific_context(self) -> Dict[str, Any]:
        return {
            "customer_name": self.remote_realm.host,
            "email": self.get_email(),
            "is_demo_organization": False,
            "demo_organization_scheduled_deletion_date": None,
            "is_self_hosting": True,
        }

    def process_downgrade(self, plan: "CustomerPlan", background_update: bool = False) -> None:
        with transaction.atomic(savepoint=False):
            old_plan_type = self.remote_realm.plan_type
            new_plan_type = RemoteRealm.PLAN_TYPE_SELF_MANAGED
            self.remote_realm.plan_type = new_plan_type
            self.remote_realm.save(update_fields=["plan_type"])
            self.write_to_audit_log(
                event_type=BillingSessionEventType.BILLING_ENTITY_PLAN_TYPE_CHANGED,
                event_time=timezone_now(),
                extra_data={"old_value": old_plan_type, "new_value": new_plan_type},
                background_update=background_update,
            )
        plan.status = CustomerPlan.ENDED
        plan.save(update_fields=["status"])

    def check_plan_tier_is_billable(self, plan_tier: int) -> bool:
        implemented_plan_tiers = [CustomerPlan.TIER_SELF_HOSTED_BASIC, CustomerPlan.TIER_SELF_HOSTED_BUSINESS]
        return plan_tier in implemented_plan_tiers

    def get_type_of_plan_tier_change(self, current_plan_tier: int, new_plan_tier: int) -> "PlanTierChangeType":
        valid_plan_tiers = [CustomerPlan.TIER_SELF_HOSTED_LEGACY, CustomerPlan.TIER_SELF_HOSTED_BASIC, CustomerPlan.TIER_SELF_HOSTED_BUSINESS]
        if current_plan_tier not in valid_plan_tiers or new_plan_tier not in valid_plan_tiers or current_plan_tier == new_plan_tier:
            return PlanTierChangeType.INVALID
        if current_plan_tier == CustomerPlan.TIER_SELF_HOSTED_BASIC and new_plan_tier == CustomerPlan.TIER_SELF_HOSTED_BUSINESS:
            return PlanTierChangeType.UPGRADE
        elif current_plan_tier == CustomerPlan.TIER_SELF_HOSTED_LEGACY and new_plan_tier in (CustomerPlan.TIER_SELF_HOSTED_BASIC, CustomerPlan.TIER_SELF_HOSTED_BUSINESS):
            return PlanTierChangeType.UPGRADE
        elif current_plan_tier == CustomerPlan.TIER_SELF_HOSTED_BASIC and new_plan_tier == CustomerPlan.TIER_SELF_HOSTED_LEGACY:
            return PlanTierChangeType.DOWNGRADE
        elif current_plan_tier == CustomerPlan.TIER_SELF_HOSTED_BUSINESS and new_plan_tier == CustomerPlan.TIER_SELF_HOSTED_LEGACY:
            return PlanTierChangeType.DOWNGRADE
        else:
            assert current_plan_tier == CustomerPlan.TIER_SELF_HOSTED_BUSINESS
            assert new_plan_tier == CustomerPlan.TIER_SELF_HOSTED_BASIC
            return PlanTierChangeType.DOWNGRADE

    def has_billing_access(self) -> bool:
        return True

    PAID_PLANS = [RemoteRealm.PLAN_TYPE_BASIC, RemoteRealm.PLAN_TYPE_BUSINESS, RemoteRealm.PLAN_TYPE_ENTERPRISE]

    def on_paid_plan(self) -> bool:
        return self.remote_realm.plan_type in self.PAID_PLANS

    def org_name(self) -> str:
        return self.remote_realm.host

    def add_org_type_data_to_sponsorship_context(self, context: Dict[str, Any]) -> None:
        context.update(
            realm_org_type=self.remote_realm.org_type,
            sorted_org_types=sorted(
                ([org_type_name, org_type] for (org_type_name, org_type) in Realm.ORG_TYPES.items() if not org_type.get("hidden")),
                key=sponsorship_org_type_key_helper,
            ),
        )

    def get_sponsorship_request_session_specific_context(self) -> Dict[str, Any]:
        assert self.remote_billing_user is not None
        return {
            "realm_user": None,
            "user_info": {
                "name": self.remote_billing_user.full_name,
                "email": self.get_email(),
                "role": "Remote realm administrator",
            },
            "realm_string_id": self.remote_realm.host,
        }

    def save_org_type_from_request_sponsorship_session(self, org_type: int) -> None:
        if self.remote_realm.org_type != org_type:
            self.remote_realm.org_type = org_type
            self.remote_realm.save(update_fields=["org_type"])

    def sync_license_ledger_if_needed(self) -> None:
        last_ledger = self.get_last_ledger_for_automanaged_plan_if_exists()
        if last_ledger is None:
            return
        new_audit_logs = RemoteRealmAuditLog.objects.filter(
            remote_realm=self.remote_realm,
            event_time__gt=last_ledger.event_time,
            event_type__in=RemoteRealmAuditLog.SYNCED_BILLING_EVENTS,
        ).exclude(extra_data={}).order_by("event_time")
        current_plan = last_ledger.plan
        for audit_log in new_audit_logs:
            end_of_cycle_plan = self.update_license_ledger_for_automanaged_plan(current_plan, audit_log.event_time)
            if end_of_cycle_plan is None:
                return
            current_plan = end_of_cycle_plan


class RemoteServerBillingSession(BillingSession):
    def __init__(self, remote_server: "RemoteZulipServer", remote_billing_user: Optional["RemoteServerBillingUser"] = None, support_staff: Optional["UserProfile"] = None) -> None:
        self.remote_server = remote_server
        self.remote_billing_user = remote_billing_user
        self.support_staff = support_staff
        if support_staff is not None:
            assert support_staff.is_staff
            self.support_session = True
        else:
            self.support_session = False

    @property
    def billing_entity_display_name(self) -> str:
        return self.remote_server.hostname

    @property
    def billing_session_url(self) -> str:
        return f"{settings.EXTERNAL_URI_SCHEME}{settings.SELF_HOSTING_MANAGEMENT_SUBDOMAIN}.{settings.EXTERNAL_HOST}/server/{self.remote_server.uuid}"

    @property
    def billing_base_url(self) -> str:
        return f"/server/{self.remote_server.uuid}"

    def support_url(self) -> str:
        return build_support_url("remote_servers_support", str(self.remote_server.uuid))

    def get_customer(self) -> Optional["Customer"]:
        return get_customer_by_remote_server(self.remote_server)

    def get_email(self) -> str:
        assert self.remote_billing_user is not None
        return self.remote_billing_user.email

    def current_count_for_billed_licenses(self, event_time: Optional[datetime] = None) -> int:
        if has_stale_audit_log(self.remote_server):
            raise MissingDataError
        remote_server_counts = get_remote_server_guest_and_non_guest_count(self.remote_server.id, event_time)
        return remote_server_counts.non_guest_user_count + remote_server_counts.guest_user_count

    def missing_data_error_page(self, request: HttpRequest) -> HttpResponse:
        missing_data_context: Dict[str, Any] = {
            "remote_realm_session": False,
            "supports_remote_realms": self.remote_server.last_api_feature_level is not None,
        }
        return render(request, "corporate/billing/server_not_uploading_data.html", context=missing_data_context)

    def get_audit_log_event(self, event_type: Any) -> int:
        if event_type is BillingSessionEventType.STRIPE_CUSTOMER_CREATED:
            return AuditLogEventType.STRIPE_CUSTOMER_CREATED
        elif event_type is BillingSessionEventType.STRIPE_CARD_CHANGED:
            return AuditLogEventType.STRIPE_CARD_CHANGED
        elif event_type is BillingSessionEventType.CUSTOMER_PLAN_CREATED:
            return AuditLogEventType.CUSTOMER_PLAN_CREATED
        elif event_type is BillingSessionEventType.DISCOUNT_CHANGED:
            return AuditLogEventType.REMOTE_SERVER_DISCOUNT_CHANGED
        elif event_type is BillingSessionEventType.CUSTOMER_PROPERTY_CHANGED:
            return AuditLogEventType.CUSTOMER_PROPERTY_CHANGED
        elif event_type is BillingSessionEventType.SPONSORSHIP_APPROVED:
            return AuditLogEventType.REMOTE_SERVER_SPONSORSHIP_APPROVED
        elif event_type is BillingSessionEventType.SPONSORSHIP_PENDING_STATUS_CHANGED:
            return AuditLogEventType.REMOTE_SERVER_SPONSORSHIP_PENDING_STATUS_CHANGED
        elif event_type is BillingSessionEventType.BILLING_MODALITY_CHANGED:
            return AuditLogEventType.REMOTE_SERVER_BILLING_MODALITY_CHANGED
        elif event_type is BillingSessionEventType.CUSTOMER_PLAN_PROPERTY_CHANGED:
            return AuditLogEventType.CUSTOMER_PLAN_PROPERTY_CHANGED
        elif event_type is BillingSessionEventType.BILLING_ENTITY_PLAN_TYPE_CHANGED:
            return AuditLogEventType.REMOTE_SERVER_PLAN_TYPE_CHANGED
        elif event_type is BillingSessionEventType.CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN:
            return AuditLogEventType.CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN
        elif event_type is BillingSessionEventType.CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_PLAN:
            return AuditLogEventType.CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_PLAN
        else:
            raise BillingSessionAuditLogEventError(event_type)

    def write_to_audit_log(self, event_type: Any, event_time: datetime, *, background_update: bool = False, extra_data: Optional[Dict[str, Any]] = None) -> None:
        audit_log_event = self.get_audit_log_event(event_type)
        log_data: Dict[str, Any] = {
            "server": self.remote_server,
            "event_type": audit_log_event,
            "event_time": event_time,
        }

        if not background_update:
            log_data.update({
                "acting_support_user": self.support_staff,
                "acting_remote_user": self.remote_billing_user,
            })

        RemoteZulipServerAuditLog.objects.create(**log_data)

    def get_data_for_stripe_customer(self) -> StripeCustomerData:
        assert self.support_session is False
        metadata: Dict[str, Any] = {}
        metadata["remote_server_uuid"] = self.remote_server.uuid
        metadata["remote_server_str"] = str(self.remote_server)
        realm_stripe_customer_data = StripeCustomerData(
            description=str(self.remote_server),
            email=self.get_email(),
            metadata=metadata,
        )
        return realm_stripe_customer_data

    def update_data_for_checkout_session_and_invoice_payment(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        assert self.remote_billing_user is not None
        updated_metadata = dict(
            remote_server_user_id=self.remote_billing_user.id,
            remote_server_user_email=self.get_email(),
            remote_server_host=self.remote_server.hostname,
            **metadata,
        )
        return updated_metadata

    def update_or_create_customer(self, stripe_customer_id: Optional[str] = None, *, defaults: Optional[Dict[str, Any]] = None) -> "Customer":
        if stripe_customer_id is not None:
            assert self.support_session is False
            customer, created = Customer.objects.update_or_create(
                remote_server=self.remote_server, defaults={"stripe_customer_id": stripe_customer_id}
            )
        else:
            customer, created = Customer.objects.update_or_create(
                remote_server=self.remote_server, defaults=defaults
            )

        if created and not customer.annual_discounted_price and not customer.monthly_discounted_price:
            customer.flat_discounted_months = 12
            customer.save(update_fields=["flat_discounted_months"])

        return customer

    def do_change_plan_type(self, *, tier: Optional[int], is_sponsored: bool = False, background_update: bool = False) -> None:
        if is_sponsored:
            plan_type = RemoteZulipServer.PLAN_TYPE_COMMUNITY
            self.add_customer_to_community_plan()
        elif tier == CustomerPlan.TIER_SELF_HOSTED_BASIC:
            plan_type = RemoteZulipServer.PLAN_TYPE_BASIC
        elif tier == CustomerPlan.TIER_SELF_HOSTED_BUSINESS:
            plan_type = RemoteZulipServer.PLAN_TYPE_BUSINESS
        elif tier == CustomerPlan.TIER_SELF_HOSTED_LEGACY:
            plan_type = RemoteZulipServer.PLAN_TYPE_SELF_MANAGED_LEGACY
        else:
            raise AssertionError("Unexpected tier")

        old_plan_type = self.remote_server.plan_type
        self.remote_server.plan_type = plan_type
        self.remote_server.save(update_fields=["plan_type"])
        self.write_to_audit_log(
            event_type=BillingSessionEventType.BILLING_ENTITY_PLAN_TYPE_CHANGED,
            event_time=timezone_now(),
            extra_data={"old_value": old_plan_type, "new_value": plan_type},
            background_update=background_update,
        )

    def approve_sponsorship(self) -> str:
        assert self.support_session
        customer = self.get_customer()
        if customer is not None:
            error_message = self.check_customer_not_on_paid_plan(customer)
            if error_message != "":
                raise SupportRequestError(error_message)

        if self.remote_server.plan_type == RemoteZulipServer.PLAN_TYPE_SELF_MANAGED_LEGACY:
            plan = get_current_plan_by_customer(customer)
            if plan is not None:
                assert self.get_next_plan(plan) is None
                assert plan.tier == CustomerPlan.TIER_SELF_HOSTED_LEGACY
                plan.status = CustomerPlan.ENDED
                plan.save(update_fields=["status"])
        self.do_change_plan_type(tier=None, is_sponsored=True)
        if customer is not None and customer.sponsorship_pending:
            customer.sponsorship_pending = False
            customer.save(update_fields=["sponsorship_pending"])
            self.write_to_audit_log(event_type=BillingSessionEventType.SPONSORSHIP_APPROVED, event_time=timezone_now())
        billing_emails = list(
            RemoteServerBillingUser.objects.filter(remote_server=self.remote_server).values_list("email", flat=True)
        )
        if len(billing_emails) > 0:
            send_email(
                "zerver/emails/sponsorship_approved_community_plan",
                to_emails=billing_emails,
                from_address=BILLING_SUPPORT_EMAIL,
                context={
                    "billing_entity": self.billing_entity_display_name,
                    "plans_link": "https://zulip.com/plans/#self-hosted",
                    "link_to_zulip": "https://zulip.com/help/linking-to-zulip-website",
                },
            )
            emailed_string = "Emailed existing billing users."
        else:
            emailed_string = "No billing users exist to email."
        return f"Sponsorship approved for {self.billing_entity_display_name}; " + emailed_string

    def is_sponsored(self) -> bool:
        return self.remote_server.plan_type == self.remote_server.PLAN_TYPE_COMMUNITY

    def get_metadata_for_stripe_update_card(self) -> Dict[str, str]:
        assert self.remote_billing_user is not None
        return {"type": "card_update", "remote_server_user_id": str(self.remote_billing_user.id)}

    def get_upgrade_page_session_type_specific_context(self) -> Dict[str, Any]:
        return {
            "customer_name": self.remote_server.hostname,
            "email": self.get_email(),
            "is_demo_organization": False,
            "demo_organization_scheduled_deletion_date": None,
            "is_self_hosting": True,
        }

    def process_downgrade(self, plan: "CustomerPlan", background_update: bool = False) -> None:
        with transaction.atomic(savepoint=False):
            old_plan_type = self.remote_server.plan_type
            new_plan_type = RemoteZulipServer.PLAN_TYPE_SELF_MANAGED
            self.remote_server.plan_type = new_plan_type
            self.remote_server.save(update_fields=["plan_type"])
            self.write_to_audit_log(
                event_type=BillingSessionEventType.BILLING_ENTITY_PLAN_TYPE_CHANGED,
                event_time=timezone_now(),
                extra_data={"old_value": old_plan_type, "new_value": new_plan_type},
                background_update=background_update,
            )
        plan.status = CustomerPlan.ENDED
        plan.save(update_fields=["status"])

    def check_plan_tier_is_billable(self, plan_tier: int) -> bool:
        implemented_plan_tiers = [CustomerPlan.TIER_SELF_HOSTED_BASIC, CustomerPlan.TIER_SELF_HOSTED_BUSINESS]
        return plan_tier in implemented_plan_tiers

    def get_type_of_plan_tier_change(self, current_plan_tier: int, new_plan_tier: int) -> "PlanTierChangeType":
        valid_plan_tiers = [CustomerPlan.TIER_SELF_HOSTED_LEGACY, CustomerPlan.TIER_SELF_HOSTED_BASIC, CustomerPlan.TIER_SELF_HOSTED_BUSINESS]
        if current_plan_tier not in valid_plan_tiers or new_plan_tier not in valid_plan_tiers or current_plan_tier == new_plan_tier:
            return PlanTierChangeType.INVALID
        if current_plan_tier == CustomerPlan.TIER_SELF_HOSTED_LEGACY and new_plan_tier in (CustomerPlan.TIER_SELF_HOSTED_BASIC, CustomerPlan.TIER_SELF_HOSTED_BUSINESS):
            return PlanTierChangeType.UPGRADE
        elif current_plan_tier == CustomerPlan.TIER_SELF_HOSTED_BASIC and new_plan_tier == CustomerPlan.TIER_SELF_HOSTED_BUSINESS:
            return PlanTierChangeType.UPGRADE
        elif current_plan_tier == CustomerPlan.TIER_SELF_HOSTED_BASIC and new_plan_tier == CustomerPlan.TIER_SELF_HOSTED_LEGACY:
            return PlanTierChangeType.DOWNGRADE
        elif current_plan_tier == CustomerPlan.TIER_SELF_HOSTED_BUSINESS and new_plan_tier == CustomerPlan.TIER_SELF_HOSTED_LEGACY:
            return PlanTierChangeType.DOWNGRADE
        else:
            assert current_plan_tier == CustomerPlan.TIER_SELF_HOSTED_BUSINESS
            assert new_plan_tier == CustomerPlan.TIER_SELF_HOSTED_BASIC
            return PlanTierChangeType.DOWNGRADE

    def has_billing_access(self) -> bool:
        return True

    PAID_PLANS = [RemoteZulipServer.PLAN_TYPE_BASIC, RemoteZulipServer.PLAN_TYPE_BUSINESS, RemoteZulipServer.PLAN_TYPE_ENTERPRISE]

    def on_paid_plan(self) -> bool:
        return self.remote_server.plan_type in self.PAID_PLANS

    def org_name(self) -> str:
        return self.remote_server.hostname

    def add_org_type_data_to_sponsorship_context(self, context: Dict[str, Any]) -> None:
        context.update(
            realm_org_type=self.remote_server.org_type,
            sorted_org_types=sorted(
                ([org_type_name, org_type] for (org_type_name, org_type) in Realm.ORG_TYPES.items() if not org_type.get("hidden")),
                key=sponsorship_org_type_key_helper,
            ),
        )

    def get_sponsorship_request_session_specific_context(self) -> Dict[str, Any]:
        assert self.remote_billing_user is not None
        return {
            "realm_user": None,
            "user_info": {
                "name": self.remote_billing_user.full_name,
                "email": self.get_email(),
                "role": "Remote server administrator",
            },
            "realm_string_id": self.remote_server.hostname,
        }

    def save_org_type_from_request_sponsorship_session(self, org_type: int) -> None:
        if self.remote_server.org_type != org_type:
            self.remote_server.org_type = org_type
            self.remote_server.save(update_fields=["org_type"])

    def sync_license_ledger_if_needed(self) -> None:
        last_ledger = self.get_last_ledger_for_automanaged_plan_if_exists()
        if last_ledger is None:
            return
        new_audit_logs = RemoteRealmAuditLog.objects.filter(
            server=self.remote_server,
            event_time__gt=last_ledger.event_time,
            event_type__in=RemoteRealmAuditLog.SYNCED_BILLING_EVENTS,
        ).exclude(extra_data={}).order_by("event_time")
        current_plan = last_ledger.plan
        for audit_log in new_audit_logs:
            end_of_cycle_plan = self.update_license_ledger_for_automanaged_plan(current_plan, audit_log.event_time)
            if end_of_cycle_plan is None:
                return
            current_plan = end_of_cycle_plan


def stripe_customer_has_credit_card_as_default_payment_method(stripe_customer: stripe.Customer) -> bool:
    assert stripe_customer.invoice_settings is not None
    if not stripe_customer.invoice_settings.default_payment_method:
        return False
    assert isinstance(stripe_customer.invoice_settings.default_payment_method, stripe.PaymentMethod)
    return stripe_customer.invoice_settings.default_payment_method.type == "card"


def customer_has_credit_card_as_default_payment_method(customer: "Customer") -> bool:
    if not customer.stripe_customer_id:
        return False
    stripe_customer = stripe_get_customer(customer.stripe_customer_id)
    return stripe_customer_has_credit_card_as_default_payment_method(stripe_customer)


def get_price_per_license(tier: int, billing_schedule: int, customer: Optional["Customer"] = None) -> int:
    if customer is not None:
        price_per_license = customer.get_discounted_price_for_plan(tier, billing_schedule)
        if price_per_license:
            return price_per_license

    price_map: Dict[int, Dict[str, int]] = {
        CustomerPlan.TIER_CLOUD_STANDARD: {"Annual": 8000, "Monthly": 800},
        CustomerPlan.TIER_CLOUD_PLUS: {"Annual": 12000, "Monthly": 1200},
        CustomerPlan.TIER_SELF_HOSTED_BASIC: {"Annual": 4200, "Monthly": 350},
        CustomerPlan.TIER_SELF_HOSTED_BUSINESS: {"Annual": 8000, "Monthly": 800},
        CustomerPlan.TIER_SELF_HOSTED_LEGACY: {"Annual": 0, "Monthly": 0},
    }

    try:
        price_per_license = price_map[tier][CustomerPlan.BILLING_SCHEDULES[billing_schedule]]
    except KeyError:
        if tier not in price_map:
            raise InvalidTierError(tier)
        else:
            raise InvalidBillingScheduleError(billing_schedule)

    return price_per_license


def get_price_per_license_and_discount(tier: int, billing_schedule: int, customer: Optional["Customer"]) -> Tuple[int, Optional[str]]:
    original_price_per_license = get_price_per_license(tier, billing_schedule)
    if customer is None:
        return original_price_per_license, None

    price_per_license = get_price_per_license(tier, billing_schedule, customer)
    if price_per_license == original_price_per_license:
        return price_per_license, None

    discount = format_discount_percentage(
        Decimal((original_price_per_license - price_per_license) / original_price_per_license * 100)
    )
    return price_per_license, discount


def compute_plan_parameters(
    tier: int,
    billing_schedule: int,
    customer: Optional["Customer"],
    free_trial: bool = False,
    billing_cycle_anchor: Optional[datetime] = None,
    is_self_hosted_billing: bool = False,
    upgrade_when_complimentary_access_plan_ends: bool = False,
) -> Tuple[datetime, datetime, datetime, int]:
    if billing_cycle_anchor is None:
        billing_cycle_anchor = timezone_now().replace(microsecond=0)

    if billing_schedule == CustomerPlan.BILLING_SCHEDULE_ANNUAL:
        period_end = add_months(billing_cycle_anchor, 12)
    elif billing_schedule == CustomerPlan.BILLING_SCHEDULE_MONTHLY:
        period_end = add_months(billing_cycle_anchor, 1)
    else:
        raise InvalidBillingScheduleError(billing_schedule)

    price_per_license = get_price_per_license(tier, billing_schedule, customer)
    next_invoice_date = add_months(billing_cycle_anchor, 1)
    if free_trial:
        period_end = billing_cycle_anchor + timedelta(days=assert_is_not_none(get_free_trial_days(is_self_hosted_billing, tier)))
        next_invoice_date = period_end
    if upgrade_when_complimentary_access_plan_ends:
        next_invoice_date = billing_cycle_anchor
    return billing_cycle_anchor, next_invoice_date, period_end, price_per_license


def get_free_trial_days(is_self_hosted_billing: bool = False, tier: Optional[int] = None) -> Optional[int]:
    if is_self_hosted_billing:
        if tier is not None and tier != CustomerPlan.TIER_SELF_HOSTED_BASIC:
            return None
        return settings.SELF_HOSTING_FREE_TRIAL_DAYS

    return settings.CLOUD_FREE_TRIAL_DAYS


def is_free_trial_offer_enabled(is_self_hosted_billing: bool, tier: Optional[int] = None) -> bool:
    return get_free_trial_days(is_self_hosted_billing, tier) not in (None, 0)


def ensure_customer_does_not_have_active_plan(customer: "Customer") -> None:
    if get_current_plan_by_customer(customer) is not None:
        billing_logger.warning("Upgrade of %s failed because of existing active plan.", str(customer))
        raise UpgradeWithExistingPlanError


def do_reactivate_remote_server(remote_server: "RemoteZulipServer") -> None:
    if not remote_server.deactivated:
        billing_logger.warning("Cannot reactivate remote server with ID %d, server is already active.", remote_server.id)
        return

    remote_server.deactivated = False
    remote_server.save(update_fields=["deactivated"])
    RemoteZulipServerAuditLog.objects.create(
        event_type=AuditLogEventType.REMOTE_SERVER_REACTIVATED,
        server=remote_server,
        event_time=timezone_now(),
    )


def do_deactivate_remote_server(remote_server: "RemoteZulipServer", billing_session: RemoteServerBillingSession) -> None:
    if remote_server.deactivated:
        billing_logger.warning("Cannot deactivate remote server with ID %d, server has already been deactivated.", remote_server.id)
        return

    server_plans_to_consider = CustomerPlan.objects.filter(customer__remote_server=remote_server).exclude(status=CustomerPlan.ENDED)
    realm_plans_to_consider = CustomerPlan.objects.filter(customer__remote_realm__server=remote_server).exclude(status=CustomerPlan.ENDED)

    for possible_plan in list(server_plans_to_consider) + list(realm_plans_to_consider):
        if possible_plan.tier in [CustomerPlan.TIER_SELF_HOSTED_BASE, CustomerPlan.TIER_SELF_HOSTED_LEGACY, CustomerPlan.TIER_SELF_HOSTED_COMMUNITY]:
            continue

        if possible_plan.status in [CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL, CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE]:
            continue

        raise ServerDeactivateWithExistingPlanError

    remote_server.deactivated = True
    remote_server.save(update_fields=["deactivated"])
    RemoteZulipServerAuditLog.objects.create(
        event_type=AuditLogEventType.REMOTE_SERVER_DEACTIVATED,
        server=remote_server,
        event_time=timezone_now(),
    )


def get_plan_renewal_or_end_date(plan: "CustomerPlan", event_time: datetime) -> datetime:
    billing_period_end = start_of_next_billing_cycle(plan, event_time)
    if plan.end_date is not None and plan.end_date < billing_period_end:
        return plan.end_date
    return billing_period_end


def invoice_plans_as_needed(event_time: Optional[datetime] = None) -> None:
    if event_time is None:
        event_time = timezone_now()
    for plan in CustomerPlan.objects.filter(next_invoice_date__lte=event_time).order_by("id"):
        remote_server: Optional[RemoteZulipServer] = None
        if plan.customer.realm is not None:
            billing_session: BillingSession = RealmBillingSession(realm=plan.customer.realm)
        elif plan.customer.remote_realm is not None:
            remote_realm = plan.customer.remote_realm
            remote_server = remote_realm.server
            billing_session = RemoteRealmBillingSession(remote_realm=remote_realm)
        elif plan.customer.remote_server is not None:
            remote_server = plan.customer.remote_server
            billing_session = RemoteServerBillingSession(remote_server=remote_server)
        assert plan.next_invoice_date is not None

        if (plan.fixed_price is not None and not plan.reminder_to_review_plan_email_sent and plan.end_date is not None and plan.end_date - plan.next_invoice_date <= timedelta(days=62)):
            context: Dict[str, Any] = {
                "billing_entity": billing_session.billing_entity_display_name,
                "end_date": plan.end_date.strftime("%Y-%m-%d"),
                "support_url": billing_session.support_url(),
                "notice_reason": "fixed_price_plan_ends_soon",
            }
            send_email("zerver/emails/internal_billing_notice", to_emails=[BILLING_SUPPORT_EMAIL], from_address=FromAddress.tokenized_no_reply_address(), context=context)
            plan.reminder_to_review_plan_email_sent = True
            plan.save(update_fields=["reminder_to_review_plan_email_sent"])

        if remote_server:
            free_plan_with_no_next_plan = (not plan.is_a_paid_plan() and plan.status == CustomerPlan.ACTIVE)
            free_trial_pay_by_invoice_plan = plan.is_free_trial() and not plan.charge_automatically
            last_audit_log_update = remote_server.last_audit_log_update
            if not free_plan_with_no_next_plan and (last_audit_log_update is None or plan.next_invoice_date > last_audit_log_update):
                if (last_audit_log_update is None or plan.next_invoice_date - last_audit_log_update >= timedelta(days=1)) and not plan.invoice_overdue_email_sent:
                    last_audit_log_update_string = "Never uploaded" if last_audit_log_update is None else last_audit_log_update.strftime("%Y-%m-%d")
                    context = {
                        "billing_entity": billing_session.billing_entity_display_name,
                        "support_url": billing_session.support_url(),
                        "last_audit_log_update": last_audit_log_update_string,
                        "notice_reason": "invoice_overdue",
                    }
                    send_email("zerver/emails/internal_billing_notice", to_emails=[BILLING_SUPPORT_EMAIL], from_address=FromAddress.tokenized_no_reply_address(), context=context)
                    plan.invoice_overdue_email_sent = True
                    plan.save(update_fields=["invoice_overdue_email_sent"])

                if not free_trial_pay_by_invoice_plan:
                    continue

        while plan.next_invoice_date is not None and plan.next_invoice_date <= event_time:
            billing_session.invoice_plan(plan, plan.next_invoice_date)
            plan.refresh_from_db()


def is_realm_on_free_trial(realm: "Realm") -> bool:
    plan = get_current_plan_by_realm(realm)
    return plan is not None and plan.is_free_trial()


def do_change_plan_status(plan: "CustomerPlan", status: int) -> None:
    plan.status = status
    plan.save(update_fields=["status"])
    billing_logger.info("Change plan status: Customer.id: %s, CustomerPlan.id: %s, status: %s", plan.customer.id, plan.id, status)


def get_all_invoices_for_customer(customer: "Customer") -> Generator[stripe.Invoice, None, None]:
    if customer.stripe_customer_id is None:
        return
    invoices = stripe.Invoice.list(customer=customer.stripe_customer_id, limit=100)
    while len(invoices):
        for invoice in invoices:
            yield invoice
            last_invoice = invoice
        assert last_invoice.id is not None
        invoices = stripe.Invoice.list(customer=customer.stripe_customer_id, starting_after=last_invoice.id, limit=100)


def customer_has_last_n_invoices_open(customer: "Customer", n: int) -> bool:
    if customer.stripe_customer_id is None:
        return False
    open_invoice_count = 0
    for invoice in stripe.Invoice.list(customer=customer.stripe_customer_id, limit=n):
        if invoice.status == "open":
            open_invoice_count += 1
    return open_invoice_count == n


def downgrade_small_realms_behind_on_payments_as_needed() -> None:
    customers = Customer.objects.all().exclude(stripe_customer_id=None).exclude(realm=None)
    for customer in customers:
        realm = customer.realm
        assert realm is not None
        if get_latest_seat_count(realm) >= 5:
            continue
        if get_current_plan_by_customer(customer) is not None:
            if not customer_has_last_n_invoices_open(customer, 2):
                continue
            billing_session: BillingSession = RealmBillingSession(user=None, realm=realm)
            billing_session.downgrade_now_without_creating_additional_invoices()
            billing_session.void_all_open_invoices()
            context: Dict[str, Union[str, "Realm"]] = {
                "upgrade_url": f"{realm.url}{reverse('upgrade_page')}",
                "realm": realm,
            }
            send_email_to_billing_admins_and_realm_owners("zerver/emails/realm_auto_downgraded", realm, from_name=FromAddress.security_email_from_name(language=realm.default_language), from_address=FromAddress.tokenized_no_reply_address(), language=realm.default_language, context=context)
        else:
            if customer_has_last_n_invoices_open(customer, 1):
                billing_session = RealmBillingSession(user=None, realm=realm)
                billing_session.void_all_open_invoices()


@dataclass
class PushNotificationsEnabledStatus:
    can_push: bool
    expected_end_timestamp: Optional[int]
    message: str


MAX_USERS_WITHOUT_PLAN = 10


def get_push_status_for_remote_request(remote_server: "RemoteZulipServer", remote_realm: Optional["RemoteRealm"]) -> PushNotificationsEnabledStatus:
    customer = None
    current_plan = None
    realm_billing_session: Optional[BillingSession] = None
    server_billing_session: Optional[RemoteServerBillingSession] = None

    if remote_realm is not None:
        realm_billing_session = RemoteRealmBillingSession(remote_realm)
        if realm_billing_session.is_sponsored():
            return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message="Community plan")
        customer = realm_billing_session.get_customer()
        if customer is not None:
            current_plan = get_current_plan_by_customer(customer)

    if customer is None or current_plan is None:
        server_billing_session = RemoteServerBillingSession(remote_server)
        if server_billing_session.is_sponsored():
            return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message="Community plan")
        customer = server_billing_session.get_customer()
        if customer is not None:
            current_plan = get_current_plan_by_customer(customer)

    if realm_billing_session is not None:
        user_count_billing_session: BillingSession = realm_billing_session
    else:
        assert server_billing_session is not None
        user_count_billing_session = server_billing_session

    user_count: Optional[int] = None
    if current_plan is None:
        try:
            user_count = user_count_billing_session.current_count_for_billed_licenses()
        except MissingDataError:
            return PushNotificationsEnabledStatus(can_push=False, expected_end_timestamp=None, message="Missing data")

        if user_count > MAX_USERS_WITHOUT_PLAN:
            return PushNotificationsEnabledStatus(
                can_push=False,
                expected_end_timestamp=None,
                message="Push notifications access with 10+ users requires signing up for a plan. https://zulip.com/plans/",
            )

        return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message="No plan few users")

    if current_plan.status not in [CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE, CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL]:
        return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message="Active plan")

    try:
        user_count = user_count_billing_session.current_count_for_billed_licenses()
    except MissingDataError:
        user_count = None

    if user_count is not None and user_count <= MAX_USERS_WITHOUT_PLAN:
        return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message="Expiring plan few users")

    expected_end_timestamp = datetime_to_timestamp(user_count_billing_session.get_next_billing_cycle(current_plan))
    return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=expected_end_timestamp, message="Scheduled end")
