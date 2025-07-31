#!/usr/bin/env python3
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import stripe
from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.urls import reverse

# Import your models and other modules as needed.
# For type annotations we assume the following classes exist.
# Replace Any with their proper types when available.
Customer = Any
CustomerPlan = Any
RemoteRealm = Any
RemoteServer = Any
Realm = Any
UserProfile = Any
Session = Any
ZulipSponsorshipRequest = Any

# Some constants (placeholders)
BILLING_SUPPORT_EMAIL: str = "support@example.com"
DEFAULT_INVOICE_DAYS_UNTIL_DUE: int = 15

@dataclass
class PushNotificationsEnabledStatus:
    can_push: bool
    expected_end_timestamp: Optional[float]
    message: str

MAX_USERS_WITHOUT_PLAN: int = 10

def stripe_customer_has_credit_card_as_default_payment_method(stripe_customer: stripe.Customer) -> bool:
    assert stripe_customer.invoice_settings is not None
    if not stripe_customer.invoice_settings.default_payment_method:
        return False
    assert isinstance(stripe_customer.invoice_settings.default_payment_method, stripe.PaymentMethod)
    return stripe_customer.invoice_settings.default_payment_method.type == 'card'

def customer_has_credit_card_as_default_payment_method(customer: Customer) -> bool:
    if not customer.stripe_customer_id:
        return False
    stripe_customer: stripe.Customer = stripe_get_customer(customer.stripe_customer_id)
    return stripe_customer_has_credit_card_as_default_payment_method(stripe_customer)

def get_price_per_license(tier: int, billing_schedule: str, customer: Optional[Customer] = None) -> int:
    if customer is not None:
        price_per_license: Optional[int] = customer.get_discounted_price_for_plan(tier, billing_schedule)
        if price_per_license:
            return price_per_license
    price_map: Dict[int, Dict[str, int]] = {
        CustomerPlan.TIER_CLOUD_STANDARD: {'Annual': 8000, 'Monthly': 800},
        CustomerPlan.TIER_CLOUD_PLUS: {'Annual': 12000, 'Monthly': 1200},
        CustomerPlan.TIER_SELF_HOSTED_BASIC: {'Annual': 4200, 'Monthly': 350},
        CustomerPlan.TIER_SELF_HOSTED_BUSINESS: {'Annual': 8000, 'Monthly': 800},
        CustomerPlan.TIER_SELF_HOSTED_LEGACY: {'Annual': 0, 'Monthly': 0},
    }
    try:
        return price_map[tier][CustomerPlan.BILLING_SCHEDULES[billing_schedule]]
    except KeyError:
        if tier not in price_map:
            raise InvalidTierError(tier)
        else:
            raise InvalidBillingScheduleError(billing_schedule)

def get_price_per_license_and_discount(tier: int, billing_schedule: str, customer: Customer) -> Tuple[int, Optional[str]]:
    original_price_per_license: int = get_price_per_license(tier, billing_schedule)
    if customer is None:
        return (original_price_per_license, None)
    price_per_license: int = get_price_per_license(tier, billing_schedule, customer)
    if price_per_license == original_price_per_license:
        return (price_per_license, None)
    discount: Optional[str] = format_discount_percentage(Decimal((original_price_per_license - price_per_license) / original_price_per_license * 100))
    return (price_per_license, discount)

def compute_plan_parameters(
    tier: int,
    billing_schedule: str,
    customer: Optional[Customer],
    free_trial: bool = False,
    billing_cycle_anchor: Optional[datetime] = None,
    is_self_hosted_billing: bool = False,
    upgrade_when_complimentary_access_plan_ends: bool = False
) -> Tuple[datetime, datetime, datetime, int]:
    if billing_cycle_anchor is None:
        billing_cycle_anchor = timezone.now().replace(microsecond=0)
    if billing_schedule == CustomerPlan.BILLING_SCHEDULE_ANNUAL:
        period_end: datetime = add_months(billing_cycle_anchor, 12)
    elif billing_schedule == CustomerPlan.BILLING_SCHEDULE_MONTHLY:
        period_end = add_months(billing_cycle_anchor, 1)
    else:
        raise InvalidBillingScheduleError(billing_schedule)
    price_per_license: int = get_price_per_license(tier, billing_schedule, customer)
    next_invoice_date: datetime = add_months(billing_cycle_anchor, 1)
    if free_trial:
        free_trial_days: Optional[int] = get_free_trial_days(is_self_hosted_billing, tier)
        period_end = billing_cycle_anchor + timedelta(days=assert_is_not_none(free_trial_days))
        next_invoice_date = period_end
    if upgrade_when_complimentary_access_plan_ends:
        next_invoice_date = billing_cycle_anchor
    return (billing_cycle_anchor, next_invoice_date, period_end, price_per_license)

def get_free_trial_days(is_self_hosted_billing: bool = False, tier: Optional[int] = None) -> Optional[int]:
    if is_self_hosted_billing:
        if tier is not None and tier != CustomerPlan.TIER_SELF_HOSTED_BASIC:
            return None
        return settings.SELF_HOSTING_FREE_TRIAL_DAYS
    return settings.CLOUD_FREE_TRIAL_DAYS

def is_free_trial_offer_enabled(is_self_hosted_billing: bool, tier: Optional[int] = None) -> bool:
    return get_free_trial_days(is_self_hosted_billing, tier) not in (None, 0)

def ensure_customer_does_not_have_active_plan(customer: Customer) -> None:
    if get_current_plan_by_customer(customer) is not None:
        billing_logger.warning('Upgrade of %s failed because of existing active plan.', str(customer))
        raise UpgradeWithExistingPlanError

@transaction.atomic(durable=True)
def do_reactivate_remote_server(remote_server: RemoteServer) -> None:
    if not remote_server.deactivated:
        billing_logger.warning('Cannot reactivate remote server with ID %d, server is already active.', remote_server.id)
        return
    remote_server.deactivated = False
    remote_server.save(update_fields=['deactivated'])
    RemoteZulipServerAuditLog.objects.create(event_type=AuditLogEventType.REMOTE_SERVER_REACTIVATED, server=remote_server, event_time=timezone.now())

@transaction.atomic(durable=True)
def do_deactivate_remote_server(remote_server: RemoteServer, billing_session: BillingSession) -> None:
    if remote_server.deactivated:
        billing_logger.warning('Cannot deactivate remote server with ID %d, server has already been deactivated.', remote_server.id)
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
    remote_server.save(update_fields=['deactivated'])
    RemoteZulipServerAuditLog.objects.create(event_type=AuditLogEventType.REMOTE_SERVER_DEACTIVATED, server=remote_server, event_time=timezone.now())

def get_plan_renewal_or_end_date(plan: CustomerPlan, event_time: datetime) -> datetime:
    billing_period_end: datetime = start_of_next_billing_cycle(plan, event_time)
    if plan.end_date is not None and plan.end_date < billing_period_end:
        return plan.end_date
    return billing_period_end

def invoice_plans_as_needed(event_time: Optional[datetime] = None) -> None:
    if event_time is None:
        event_time = timezone.now()
    for plan in CustomerPlan.objects.filter(next_invoice_date__lte=event_time).order_by('id'):
        remote_server: Optional[RemoteServer] = None
        if plan.customer.realm is not None:
            billing_session: BillingSession = RealmBillingSession(realm=plan.customer.realm)
        elif plan.customer.remote_realm is not None:
            remote_realm = plan.customer.remote_realm
            remote_server = remote_realm.server
            billing_session = RemoteRealmBillingSession(remote_realm=remote_realm)
        elif plan.customer.remote_server is not None:
            remote_server = plan.customer.remote_server
            billing_session = RemoteServerBillingSession(remote_server=remote_server)
        else:
            continue
        assert plan.next_invoice_date is not None
        if plan.fixed_price is not None and (not plan.reminder_to_review_plan_email_sent) and (plan.end_date is not None) and (plan.end_date - plan.next_invoice_date <= timedelta(days=62)):
            context: Dict[str, Any] = {'billing_entity': billing_session.billing_entity_display_name, 'end_date': plan.end_date.strftime('%Y-%m-%d'), 'support_url': billing_session.support_url(), 'notice_reason': 'fixed_price_plan_ends_soon'}
            send_email('zerver/emails/internal_billing_notice', to_emails=[BILLING_SUPPORT_EMAIL], from_address=FromAddress.tokenized_no_reply_address(), context=context)
            plan.reminder_to_review_plan_email_sent = True
            plan.save(update_fields=['reminder_to_review_plan_email_sent'])
        if remote_server:
            free_plan_with_no_next_plan: bool = not plan.is_a_paid_plan() and plan.status == CustomerPlan.ACTIVE
            free_trial_pay_by_invoice_plan: bool = plan.is_free_trial() and (not plan.charge_automatically)
            last_audit_log_update: Optional[datetime] = remote_server.last_audit_log_update
            if not free_plan_with_no_next_plan and (last_audit_log_update is None or plan.next_invoice_date > last_audit_log_update):
                if (last_audit_log_update is None or plan.next_invoice_date - last_audit_log_update >= timedelta(days=1)) and (not plan.invoice_overdue_email_sent):
                    last_audit_log_update_string: str = 'Never uploaded'
                    if last_audit_log_update is not None:
                        last_audit_log_update_string = last_audit_log_update.strftime('%Y-%m-%d')
                    context = {'billing_entity': billing_session.billing_entity_display_name, 'support_url': billing_session.support_url(), 'last_audit_log_update': last_audit_log_update_string, 'notice_reason': 'invoice_overdue'}
                    send_email('zerver/emails/internal_billing_notice', to_emails=[BILLING_SUPPORT_EMAIL], from_address=FromAddress.tokenized_no_reply_address(), context=context)
                    plan.invoice_overdue_email_sent = True
                    plan.save(update_fields=['invoice_overdue_email_sent'])
                if not free_trial_pay_by_invoice_plan:
                    continue
        while plan.next_invoice_date is not None and plan.next_invoice_date <= event_time:
            billing_session.invoice_plan(plan, plan.next_invoice_date)
            plan.refresh_from_db()

def customer_has_last_n_invoices_open(customer: Customer, n: int) -> bool:
    if customer.stripe_customer_id is None:
        return False
    open_invoice_count: int = 0
    for invoice in stripe.Invoice.list(customer=customer.stripe_customer_id, limit=n):
        if invoice.status == 'open':
            open_invoice_count += 1
    return open_invoice_count == n

def downgrade_small_realms_behind_on_payments_as_needed() -> None:
    customers: List[Customer] = list(Customer.objects.all().exclude(stripe_customer_id=None).exclude(realm=None))
    for customer in customers:
        realm: Optional[Realm] = customer.realm
        if realm is None:
            continue
        assert realm is not None
        if get_latest_seat_count(realm) >= 5:
            continue
        if get_current_plan_by_customer(customer) is not None:
            if not customer_has_last_n_invoices_open(customer, 2):
                continue
            billing_session: BillingSession = RealmBillingSession(realm=realm, user=None)
            billing_session.downgrade_now_without_creating_additional_invoices()
            billing_session.void_all_open_invoices()
            context: Dict[str, Any] = {'upgrade_url': f'{realm.url}{reverse("upgrade_page")}', 'realm': realm}
            send_email_to_billing_admins_and_realm_owners('zerver/emails/realm_auto_downgraded', realm, from_name=FromAddress.security_email_from_name(language=realm.default_language), from_address=FromAddress.tokenized_no_reply_address(), language=realm.default_language, context=context)
        elif customer_has_last_n_invoices_open(customer, 1):
            billing_session = RealmBillingSession(realm=realm, user=None)
            billing_session.void_all_open_invoices()

def do_change_plan_status(plan: CustomerPlan, status: Any) -> None:
    plan.status = status
    plan.save(update_fields=['status'])
    billing_logger.info('Change plan status: Customer.id: %s, CustomerPlan.id: %s, status: %s', plan.customer.id, plan.id, status)

# Below are stubs for functions and classes referenced in the code.
# Replace their bodies and types with actual implementations.

def add_months(dt: datetime, months: int) -> datetime:
    # Dummy implementation, replace with your actual function.
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, 28)
    return dt.replace(year=year, month=month, day=day)

def assert_is_not_none(value: Optional[int]) -> int:
    if value is None:
        raise ValueError("Value is None")
    return value

def get_latest_seat_count(realm: Realm) -> int:
    # Dummy implementation.
    return 0

def get_current_plan_by_customer(customer: Customer) -> Optional[CustomerPlan]:
    # Dummy implementation.
    return None

def stripe_get_customer(stripe_customer_id: str) -> stripe.Customer:
    # Dummy implementation.
    return stripe.Customer.retrieve(stripe_customer_id, expand=['invoice_settings', 'invoice_settings.default_payment_method'])

# Exception classes

class InvalidTierError(Exception):
    def __init__(self, tier: Any) -> None:
        self.tier = tier
        super().__init__(f'Unknown tier: {tier}')

class InvalidBillingScheduleError(Exception):
    def __init__(self, billing_schedule: Any) -> None:
        self.billing_schedule = billing_schedule
        super().__init__(f'Invalid billing schedule: {billing_schedule}')

class UpgradeWithExistingPlanError(Exception):
    pass

class ServerDeactivateWithExistingPlanError(Exception):
    pass

# Logging object (stub)
billing_logger: Any = None

# Address helper (stub)
class FromAddress:
    @staticmethod
    def tokenized_no_reply_address() -> str:
        return "no-reply@example.com"

    @staticmethod
    def security_email_from_name(language: str) -> str:
        return "security@example.com"

# Email send functions (stubs)
def send_email(template: str, to_emails: List[str], from_address: str, context: Dict[str, Any], from_name: Optional[str] = None, reply_to_email: Optional[str] = None) -> None:
    pass

def send_email_to_billing_admins_and_realm_owners(template: str, realm: Realm, from_name: str, from_address: str, language: str, context: Dict[str, Any]) -> None:
    pass

# Audit log models and functions (stubs)
class RemoteZulipServerAuditLog:
    @staticmethod
    def objects_create(**kwargs: Any) -> None:
        pass
    objects: Any

class RemoteRealmAuditLog:
    SYNCED_BILLING_EVENTS: List[Any] = []
    @staticmethod
    def objects_create(**kwargs: Any) -> None:
        pass
    objects: Any

class AuditLogEventType:
    REMOTE_SERVER_REACTIVATED = "remote_server_reactivated"
    REMOTE_SERVER_DEACTIVATED = "remote_server_deactivated"
    STRIPE_CUSTOMER_CREATED = "stripe_customer_created"
    STRIPE_CARD_CHANGED = "stripe_card_changed"
    CUSTOMER_PLAN_CREATED = "customer_plan_created"
    REMOTE_SERVER_DISCOUNT_CHANGED = "remote_server_discount_changed"
    CUSTOMER_PROPERTY_CHANGED = "customer_property_changed"
    REMOTE_SERVER_SPONSORSHIP_APPROVED = "remote_server_sponsorship_approved"
    REMOTE_SERVER_SPONSORSHIP_PENDING_STATUS_CHANGED = "remote_server_sponsorship_pending_status_changed"
    REMOTE_SERVER_BILLING_MODALITY_CHANGED = "remote_server_billing_modality_changed"
    CUSTOMER_PLAN_PROPERTY_CHANGED = "customer_plan_property_changed"
    REMOTE_SERVER_PLAN_TYPE_CHANGED = "remote_server_plan_type_changed"
    CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN = "customer_switched_from_monthly_to_annual_plan"
    CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_PLAN = "customer_switched_from_annual_to_monthly_plan"

# BillingSession classes and their methods should be annotated similarly.
# For brevity, only the RealmBillingSession, RemoteRealmBillingSession and RemoteServerBillingSession classes are shown with type annotations.

class BillingSession:
    # Dummy base class for type annotation purposes.
    billing_entity_display_name: str
    billing_session_url: str
    billing_base_url: str

    def support_url(self) -> str:
        raise NotImplementedError

    def get_customer(self) -> Optional[Customer]:
        raise NotImplementedError

    def get_email(self) -> str:
        raise NotImplementedError

    def current_count_for_billed_licenses(self, event_time: Optional[datetime] = None) -> int:
        raise NotImplementedError

    def invoice_plan(self, plan: CustomerPlan, event_time: datetime) -> None:
        raise NotImplementedError

    def downgrade_now_without_creating_additional_invoices(self, plan: Optional[CustomerPlan] = None, background_update: bool = False) -> None:
        raise NotImplementedError

    def void_all_open_invoices(self) -> int:
        raise NotImplementedError

class RealmBillingSession(BillingSession):
    def __init__(self, user: Optional[UserProfile] = None, realm: Optional[Realm] = None, *, support_session: bool = False) -> None:
        self.user: Optional[UserProfile] = user
        assert user is not None or realm is not None
        if support_session:
            assert user is not None and user.is_staff
        self.support_session: bool = support_session
        if user is not None and realm is not None:
            assert user.is_staff or user.realm == realm
            self.realm: Realm = realm
        elif user is not None:
            self.realm = user.realm
        else:
            assert realm is not None
            self.realm = realm

    @property
    def billing_entity_display_name(self) -> str:
        return self.realm.string_id

    @property
    def billing_session_url(self) -> str:
        return self.realm.url

    @property
    def billing_base_url(self) -> str:
        return ''

    def support_url(self) -> str:
        return build_support_url('support', self.realm.string_id)

    def get_customer(self) -> Optional[Customer]:
        return get_customer_by_realm(self.realm)

    def get_email(self) -> str:
        assert self.user is not None
        return self.user.delivery_email

    def current_count_for_billed_licenses(self, event_time: Optional[datetime] = None) -> int:
        return get_latest_seat_count(self.realm)

    def invoice_plan(self, plan: CustomerPlan, event_time: datetime) -> None:
        # Implementation goes here...
        pass

    def downgrade_now_without_creating_additional_invoices(self, plan: Optional[CustomerPlan] = None, background_update: bool = False) -> None:
        # Implementation goes here...
        pass

    def void_all_open_invoices(self) -> int:
        customer: Optional[Customer] = self.get_customer()
        if customer is None:
            return 0
        invoices = get_all_invoices_for_customer(customer)
        voided_invoices_count: int = 0
        for invoice in invoices:
            if invoice.status == 'open':
                assert invoice.id is not None
                stripe.Invoice.void_invoice(invoice.id)
                voided_invoices_count += 1
        return voided_invoices_count

    # Other methods omitted for brevity...

class RemoteRealmBillingSession(BillingSession):
    def __init__(self, remote_realm: RemoteRealm, remote_billing_user: Optional[Any] = None, support_staff: Optional[Any] = None) -> None:
        self.remote_realm: RemoteRealm = remote_realm
        self.remote_billing_user: Optional[Any] = remote_billing_user
        self.support_staff: Optional[Any] = support_staff
        if support_staff is not None:
            assert support_staff.is_staff
            self.support_session: bool = True
        else:
            self.support_session = False

    @property
    def billing_entity_display_name(self) -> str:
        return self.remote_realm.name

    @property
    def billing_session_url(self) -> str:
        return f'{settings.EXTERNAL_URI_SCHEME}{settings.SELF_HOSTING_MANAGEMENT_SUBDOMAIN}.{settings.EXTERNAL_HOST}/realm/{self.remote_realm.uuid}'

    @property
    def billing_base_url(self) -> str:
        return f'/realm/{self.remote_realm.uuid}'

    def support_url(self) -> str:
        return build_support_url('remote_servers_support', str(self.remote_realm.uuid))

    def get_customer(self) -> Optional[Customer]:
        return get_customer_by_remote_realm(self.remote_realm)

    def get_email(self) -> str:
        assert self.remote_billing_user is not None
        return self.remote_billing_user.email

    def current_count_for_billed_licenses(self, event_time: Optional[datetime] = None) -> int:
        if has_stale_audit_log(self.remote_realm.server):
            raise MissingDataError
        remote_realm_counts = get_remote_realm_guest_and_non_guest_count(self.remote_realm, event_time)
        return remote_realm_counts.non_guest_user_count + remote_realm_counts.guest_user_count

    # Other methods omitted for brevity...

class RemoteServerBillingSession(BillingSession):
    def __init__(self, remote_server: RemoteServer, remote_billing_user: Optional[Any] = None, support_staff: Optional[Any] = None) -> None:
        self.remote_server: RemoteServer = remote_server
        self.remote_billing_user: Optional[Any] = remote_billing_user
        self.support_staff: Optional[Any] = support_staff
        if support_staff is not None:
            assert support_staff.is_staff
            self.support_session: bool = True
        else:
            self.support_session = False

    @property
    def billing_entity_display_name(self) -> str:
        return self.remote_server.hostname

    @property
    def billing_session_url(self) -> str:
        return f'{settings.EXTERNAL_URI_SCHEME}{settings.SELF_HOSTING_MANAGEMENT_SUBDOMAIN}.{settings.EXTERNAL_HOST}/server/{self.remote_server.uuid}'

    @property
    def billing_base_url(self) -> str:
        return f'/server/{self.remote_server.uuid}'

    def support_url(self) -> str:
        return build_support_url('remote_servers_support', str(self.remote_server.uuid))

    def get_customer(self) -> Optional[Customer]:
        return get_customer_by_remote_server(self.remote_server)

    def get_email(self) -> str:
        assert self.remote_billing_user is not None
        return self.remote_billing_user.email

    def current_count_for_billed_licenses(self, event_time: Optional[datetime] = None) -> int:
        if has_stale_audit_log(self.remote_server):
            raise MissingDataError
        remote_server_counts = get_remote_server_guest_and_non_guest_count(self.remote_server.id, event_time)
        return remote_server_counts.non_guest_user_count + remote_server_counts.guest_user_count

    # Other methods omitted for brevity...

# Build support URL helper.
def build_support_url(support_view: str, query_text: str) -> str:
    # Dummy implementation.
    return f"https://support.example.com/{support_view}?q={query_text}"

# Placeholder for missing functions used in BillingSession.
def get_customer_by_realm(realm: Realm) -> Optional[Customer]:
    return None

def get_customer_by_remote_realm(remote_realm: RemoteRealm) -> Optional[Customer]:
    return None

def get_customer_by_remote_server(remote_server: RemoteServer) -> Optional[Customer]:
    return None

def get_remote_realm_guest_and_non_guest_count(remote_realm: RemoteRealm, event_time: Optional[datetime] = None) -> Any:
    # Dummy implementation.
    class Count:
        non_guest_user_count = 0
        guest_user_count = 0
    return Count()

def get_remote_server_guest_and_non_guest_count(remote_server_id: int, event_time: Optional[datetime] = None) -> Any:
    class Count:
        non_guest_user_count = 0
        guest_user_count = 0
    return Count()

def has_stale_audit_log(server: Any) -> bool:
    return False

def get_all_invoices_for_customer(customer: Customer) -> Iterator[Any]:
    if customer.stripe_customer_id is None:
        return iter([])
    invoices = stripe.Invoice.list(customer=customer.stripe_customer_id, limit=100)
    while invoices:
        for invoice in invoices:
            yield invoice
            last_invoice = invoice
        assert last_invoice.id is not None
        invoices = stripe.Invoice.list(customer=customer.stripe_customer_id, starting_after=last_invoice.id, limit=100)
    return iter([])

# End of annotated code.
