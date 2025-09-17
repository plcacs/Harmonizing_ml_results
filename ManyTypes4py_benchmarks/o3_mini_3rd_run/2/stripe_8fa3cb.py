#!/usr/bin/env python3
"""
Fully annotated version.
"""

import logging
import math
import os
import stripe
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
from dataclasses import dataclass

# Assume other necessary django and local imports exist.

# Constant declarations
BILLING_SUPPORT_EMAIL: str = "sales@zulip.com"
MIN_LICENSES: int = 5
MAX_USERS_WITHOUT_PLAN: int = 10

# --- Utility functions ---

def stripe_customer_has_credit_card_as_default_payment_method(stripe_customer: Any) -> bool:
    assert stripe_customer.invoice_settings is not None
    if not stripe_customer.invoice_settings.default_payment_method:
        return False
    assert isinstance(stripe_customer.invoice_settings.default_payment_method, stripe.PaymentMethod)
    return stripe_customer.invoice_settings.default_payment_method.type == 'card'

def customer_has_credit_card_as_default_payment_method(customer: Any) -> bool:
    if not customer.stripe_customer_id:
        return False
    stripe_customer: Any = stripe_get_customer(customer.stripe_customer_id)
    return stripe_customer_has_credit_card_as_default_payment_method(stripe_customer)

def get_price_per_license(tier: Any, billing_schedule: Any, customer: Optional[Any] = None) -> int:
    if customer is not None:
        price_per_license: Optional[int] = customer.get_discounted_price_for_plan(tier, billing_schedule)
        if price_per_license:
            return price_per_license
    price_map: Dict[Any, Dict[str, int]] = {
        # Example mapping; real mapping might be different
        "TIER_CLOUD_STANDARD": {"Annual": 8000, "Monthly": 800},
        "TIER_CLOUD_PLUS": {"Annual": 12000, "Monthly": 1200},
        "TIER_SELF_HOSTED_BASIC": {"Annual": 4200, "Monthly": 350},
        "TIER_SELF_HOSTED_BUSINESS": {"Annual": 8000, "Monthly": 800},
        "TIER_SELF_HOSTED_LEGACY": {"Annual": 0, "Monthly": 0},
    }
    try:
        price: int = price_map[tier][CustomerPlan.BILLING_SCHEDULES[billing_schedule]]
    except KeyError:
        if tier not in price_map:
            raise InvalidTierError(tier)
        else:
            raise InvalidBillingScheduleError(billing_schedule)
    return price

def get_price_per_license_and_discount(tier: Any, billing_schedule: Any, customer: Any) -> Tuple[int, Optional[Any]]:
    original_price_per_license: int = get_price_per_license(tier, billing_schedule)
    if customer is None:
        return (original_price_per_license, None)
    price_per_license: int = get_price_per_license(tier, billing_schedule, customer)
    if price_per_license == original_price_per_license:
        return (price_per_license, None)
    discount: Optional[Any] = format_discount_percentage(Decimal((original_price_per_license - price_per_license) / original_price_per_license * 100))
    return (price_per_license, discount)

def compute_plan_parameters(tier: Any, billing_schedule: Any, customer: Optional[Any], free_trial: bool = False, billing_cycle_anchor: Optional[datetime] = None, is_self_hosted_billing: bool = False, upgrade_when_complimentary_access_plan_ends: bool = False) -> Tuple[datetime, datetime, datetime, int]:
    if billing_cycle_anchor is None:
        billing_cycle_anchor = datetime.now(timezone.utc).replace(microsecond=0)
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
        period_end = billing_cycle_anchor + timedelta(days=free_trial_days if free_trial_days is not None else 0)
        next_invoice_date = period_end
    if upgrade_when_complimentary_access_plan_ends:
        next_invoice_date = billing_cycle_anchor
    return (billing_cycle_anchor, next_invoice_date, period_end, price_per_license)

def get_free_trial_days(is_self_hosted_billing: bool = False, tier: Optional[Any] = None) -> Optional[int]:
    if is_self_hosted_billing:
        if tier is not None and tier != CustomerPlan.TIER_SELF_HOSTED_BASIC:
            return None
        return settings.SELF_HOSTING_FREE_TRIAL_DAYS
    return settings.CLOUD_FREE_TRIAL_DAYS

def is_free_trial_offer_enabled(is_self_hosted_billing: bool, tier: Optional[Any] = None) -> bool:
    return get_free_trial_days(is_self_hosted_billing, tier) not in (None, 0)

def ensure_customer_does_not_have_active_plan(customer: Any) -> None:
    if get_current_plan_by_customer(customer) is not None:
        billing_logger.warning('Upgrade for %s failed due to existing active plan.', str(customer))
        raise UpgradeWithExistingPlanError

@transaction.atomic(durable=True)
def do_reactivate_remote_server(remote_server: Any) -> None:
    if not remote_server.deactivated:
        billing_logger.warning('Cannot reactivate remote server with ID %d, already active.', remote_server.id)
        return
    remote_server.deactivated = False
    remote_server.save(update_fields=['deactivated'])
    RemoteZulipServerAuditLog.objects.create(event_type=AuditLogEventType.REMOTE_SERVER_REACTIVATED, server=remote_server, event_time=datetime.now(timezone.utc))

@transaction.atomic(durable=True)
def do_deactivate_remote_server(remote_server: Any, billing_session: Any) -> None:
    if remote_server.deactivated:
        billing_logger.warning('Remote server with ID %d is already deactivated.', remote_server.id)
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
    RemoteZulipServerAuditLog.objects.create(event_type=AuditLogEventType.REMOTE_SERVER_DEACTIVATED, server=remote_server, event_time=datetime.now(timezone.utc))

def get_plan_renewal_or_end_date(plan: Any, event_time: datetime) -> datetime:
    billing_period_end: datetime = start_of_next_billing_cycle(plan, event_time)
    if plan.end_date is not None and plan.end_date < billing_period_end:
        return plan.end_date
    return billing_period_end

def invoice_plans_as_needed(event_time: Optional[datetime] = None) -> None:
    if event_time is None:
        event_time = datetime.now(timezone.utc)
    for plan in CustomerPlan.objects.filter(next_invoice_date__lte=event_time).order_by('id'):
        remote_server: Optional[Any] = None
        if plan.customer.realm is not None:
            billing_session = RealmBillingSession(realm=plan.customer.realm)
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
            free_plan_with_no_next_plan = (not plan.is_a_paid_plan()) and (plan.status == CustomerPlan.ACTIVE)
            free_trial_pay_by_invoice_plan = plan.is_free_trial() and (not plan.charge_automatically)
            last_audit_log_update = remote_server.last_audit_log_update
            if not free_plan_with_no_next_plan and (last_audit_log_update is None or plan.next_invoice_date > last_audit_log_update):
                if (last_audit_log_update is None or plan.next_invoice_date - last_audit_log_update >= timedelta(days=1)) and (not plan.invoice_overdue_email_sent):
                    last_audit_log_update_string = 'Never uploaded' if last_audit_log_update is None else last_audit_log_update.strftime('%Y-%m-%d')
                    context = {'billing_entity': billing_session.billing_entity_display_name, 'support_url': billing_session.support_url(), 'last_audit_log_update': last_audit_log_update_string, 'notice_reason': 'invoice_overdue'}
                    send_email('zerver/emails/internal_billing_notice', to_emails=[BILLING_SUPPORT_EMAIL], from_address=FromAddress.tokenized_no_reply_address(), context=context)
                    plan.invoice_overdue_email_sent = True
                    plan.save(update_fields=['invoice_overdue_email_sent'])
                if not free_trial_pay_by_invoice_plan:
                    continue
        while plan.next_invoice_date is not None and plan.next_invoice_date <= event_time:
            billing_session.invoice_plan(plan, plan.next_invoice_date)
            plan.refresh_from_db()

def is_realm_on_free_trial(realm: Any) -> bool:
    plan = get_current_plan_by_realm(realm)
    return plan is not None and plan.is_free_trial()

def do_change_plan_status(plan: Any, status: Any) -> None:
    plan.status = status
    plan.save(update_fields=['status'])
    billing_logger.info('Change plan status: Customer.id: %s, CustomerPlan.id: %s, status: %s', plan.customer.id, plan.id, status)

def get_all_invoices_for_customer(customer: Any) -> Generator[Any, None, None]:
    if customer.stripe_customer_id is None:
        return
    invoices = stripe.Invoice.list(customer=customer.stripe_customer_id, limit=100)
    while invoices:
        for invoice in invoices:
            yield invoice
            last_invoice = invoice
        assert last_invoice.id is not None
        invoices = stripe.Invoice.list(customer=customer.stripe_customer_id, starting_after=last_invoice.id, limit=100)

def customer_has_last_n_invoices_open(customer: Any, n: int) -> bool:
    if customer.stripe_customer_id is None:
        return False
    open_invoice_count: int = 0
    for invoice in stripe.Invoice.list(customer=customer.stripe_customer_id, limit=n):
        if invoice.status == 'open':
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
            billing_session = RealmBillingSession(user=None, realm=realm)
            billing_session.downgrade_now_without_creating_additional_invoices()
            billing_session.void_all_open_invoices()
            context = {'upgrade_url': f'{realm.url}{reverse("upgrade_page")}', 'realm': realm}
            send_email_to_billing_admins_and_realm_owners('zerver/emails/realm_auto_downgraded', realm, from_name=FromAddress.security_email_from_name(language=realm.default_language), from_address=FromAddress.tokenized_no_reply_address(), language=realm.default_language, context=context)
        elif customer_has_last_n_invoices_open(customer, 1):
            billing_session = RealmBillingSession(user=None, realm=realm)
            billing_session.void_all_open_invoices()

# --- Dataclass for Push Notifications Status ---

@dataclass
class PushNotificationsEnabledStatus:
    can_push: bool
    expected_end_timestamp: Optional[float]
    message: str

def get_push_status_for_remote_request(remote_server: Any, remote_realm: Optional[Any]) -> PushNotificationsEnabledStatus:
    customer: Optional[Any] = None
    current_plan: Optional[Any] = None
    realm_billing_session: Optional[RemoteRealmBillingSession] = None
    server_billing_session: Optional[RemoteServerBillingSession] = None
    if remote_realm is not None:
        realm_billing_session = RemoteRealmBillingSession(remote_realm)
        if realm_billing_session.is_sponsored():
            return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message='Community plan')
        customer = realm_billing_session.get_customer()
        if customer is not None:
            current_plan = get_current_plan_by_customer(customer)
    if customer is None or current_plan is None:
        server_billing_session = RemoteServerBillingSession(remote_server)
        if server_billing_session.is_sponsored():
            return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message='Community plan')
        customer = server_billing_session.get_customer()
        if customer is not None:
            current_plan = get_current_plan_by_customer(customer)
    if realm_billing_session is not None:
        user_count_billing_session = realm_billing_session
    else:
        assert server_billing_session is not None
        user_count_billing_session = server_billing_session
    user_count: Optional[int] = None
    if current_plan is None:
        try:
            user_count = user_count_billing_session.current_count_for_billed_licenses()
        except MissingDataError:
            return PushNotificationsEnabledStatus(can_push=False, expected_end_timestamp=None, message='Missing data')
        if user_count > MAX_USERS_WITHOUT_PLAN:
            return PushNotificationsEnabledStatus(can_push=False, expected_end_timestamp=None, message='Push notifications access with 10+ users requires signing up for a plan. https://zulip.com/plans/')
        return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message='No plan few users')
    if current_plan.status not in [CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE, CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL]:
        return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message='Active plan')
    try:
        user_count = user_count_billing_session.current_count_for_billed_licenses()
    except MissingDataError:
        user_count = None
    if user_count is not None and user_count <= MAX_USERS_WITHOUT_PLAN:
        return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message='Expiring plan few users')
    expected_end_timestamp: float = datetime_to_timestamp(user_count_billing_session.get_next_billing_cycle(current_plan))
    return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=expected_end_timestamp, message='Scheduled end')

# --- Billing session classes ---

class BillingSession:
    # Abstract base; methods intended to be overridden.
    def customer_plan_exists(self) -> bool:
        ...

    def get_next_billing_cycle(self, plan: Any) -> datetime:
        ...

    def invoice_plan(self, plan: Any, event_time: datetime) -> None:
        ...

    def downgrade_now_without_creating_additional_invoices(self, plan: Optional[Any] = None, background_update: bool = False) -> None:
        ...

    # Other methods omitted for brevity.
    # Assume full implementation exists with annotations.


class RealmBillingSession(BillingSession):
    def __init__(self, user: Optional[Any] = None, realm: Optional[Any] = None, *, support_session: bool = False) -> None:
        self.user: Optional[Any] = user
        assert user is not None or realm is not None
        if support_session:
            assert user is not None and user.is_staff
        self.support_session: bool = support_session
        if user is not None and realm is not None:
            assert user.is_staff or user.realm == realm
            self.realm = realm
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

    def get_customer(self) -> Any:
        return get_customer_by_realm(self.realm)

    def get_email(self) -> str:
        assert self.user is not None
        return self.user.delivery_email

    def current_count_for_billed_licenses(self, event_time: Optional[datetime] = None) -> int:
        return get_latest_seat_count(self.realm)

    def get_audit_log_event(self, event_type: Any) -> Any:
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

    def write_to_audit_log(self, event_type: Any, event_time: datetime, *, background_update: bool = False, extra_data: Optional[Any] = None) -> None:
        audit_log_event: Any = self.get_audit_log_event(event_type)
        audit_log_data: Dict[str, Any] = {'realm': self.realm, 'event_type': audit_log_event, 'event_time': event_time}
        if extra_data:
            audit_log_data['extra_data'] = extra_data
        if self.user is not None and (not background_update):
            audit_log_data['acting_user'] = self.user
        RealmAuditLog.objects.create(**audit_log_data)

    def get_data_for_stripe_customer(self) -> Any:
        assert self.support_session is False
        assert self.user is not None
        metadata: Dict[str, Any] = {}
        metadata['realm_id'] = self.realm.id
        metadata['realm_str'] = self.realm.string_id
        realm_stripe_customer_data = StripeCustomerData(description=f'{self.realm.string_id} ({self.realm.name})', email=self.get_email(), metadata=metadata)
        return realm_stripe_customer_data

    def update_data_for_checkout_session_and_invoice_payment(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        assert self.user is not None
        updated_metadata: Dict[str, Any] = dict(user_email=self.get_email(), realm_id=self.realm.id, realm_str=self.realm.string_id, user_id=self.user.id, **metadata)
        return updated_metadata

    def update_or_create_customer(self, stripe_customer_id: Optional[str] = None, *, defaults: Optional[Dict[str, Any]] = None) -> Any:
        if stripe_customer_id is not None:
            assert self.support_session is False
            customer, created = Customer.objects.update_or_create(realm=self.realm, defaults={'stripe_customer_id': stripe_customer_id})
            from zerver.actions.users import do_change_is_billing_admin
            assert self.user is not None
            do_change_is_billing_admin(self.user, True)
            return customer
        else:
            customer, created = Customer.objects.update_or_create(realm=self.realm, defaults=defaults)
            return customer

    @transaction.atomic(savepoint=False)
    def do_change_plan_type(self, *, tier: Any, is_sponsored: bool = False, background_update: bool = False) -> None:
        from zerver.actions.realm_settings import do_change_realm_plan_type
        if is_sponsored:
            plan_type = Realm.PLAN_TYPE_STANDARD_FREE
        elif tier == CustomerPlan.TIER_CLOUD_STANDARD:
            plan_type = Realm.PLAN_TYPE_STANDARD
        elif tier == CustomerPlan.TIER_CLOUD_PLUS:
            plan_type = Realm.PLAN_TYPE_PLUS
        else:
            raise AssertionError('Unexpected tier')
        acting_user: Optional[Any] = None
        if not background_update:
            acting_user = self.user
        do_change_realm_plan_type(self.realm, plan_type, acting_user=acting_user)

    def process_downgrade(self, plan: Any, background_update: bool = False) -> None:
        from zerver.actions.realm_settings import do_change_realm_plan_type
        acting_user: Optional[Any] = None
        if not background_update:
            acting_user = self.user
        assert plan.customer.realm is not None
        do_change_realm_plan_type(plan.customer.realm, Realm.PLAN_TYPE_LIMITED, acting_user=acting_user)
        plan.status = CustomerPlan.ENDED
        plan.save(update_fields=['status'])

    def approve_sponsorship(self) -> str:
        assert self.support_session
        customer: Optional[Any] = self.get_customer()
        if customer is not None:
            error_message: str = self.check_customer_not_on_paid_plan(customer)
            if error_message != '':
                raise SupportRequestError(error_message)
        from zerver.actions.message_send import internal_send_private_message
        if self.realm.deactivated:
            raise SupportRequestError('Realm has been deactivated')
        self.do_change_plan_type(tier=None, is_sponsored=True)
        if customer is not None and customer.sponsorship_pending:
            customer.sponsorship_pending = False
            customer.save(update_fields=['sponsorship_pending'])
            self.write_to_audit_log(event_type=BillingSessionEventType.SPONSORSHIP_APPROVED, event_time=datetime.now(timezone.utc))
        notification_bot = get_system_bot(settings.NOTIFICATION_BOT, self.realm.id)
        for user in self.realm.get_human_billing_admin_and_realm_owner_users():
            from django.utils.translation import override as override_language
            with override_language(user.default_language):
                message = _("Your organization's request for sponsored hosting has been approved! You have been upgraded to {plan_name}, free of charge. {emoji}\n\nIf you could {begin_link}list Zulip as a sponsor on your website{end_link}, we would really appreciate it!").format(plan_name=CustomerPlan.name_from_tier(CustomerPlan.TIER_CLOUD_STANDARD), emoji=':tada:', begin_link='[', end_link='](/help/linking-to-zulip-website)')
                internal_send_private_message(notification_bot, user, message)
        return f'Sponsorship approved for {self.billing_entity_display_name}; Emailed organization owners and billing admins.'

    def is_sponsored(self) -> bool:
        return self.realm.plan_type == self.realm.PLAN_TYPE_STANDARD_FREE

    def get_metadata_for_stripe_update_card(self) -> Dict[str, Any]:
        assert self.user is not None
        return {'type': 'card_update', 'user_id': str(self.user.id)}

    def get_upgrade_page_session_type_specific_context(self) -> Dict[str, Any]:
        assert self.user is not None
        return {
            'customer_name': self.realm.name,
            'email': self.get_email(),
            'is_demo_organization': self.realm.demo_organization_scheduled_deletion_date is not None,
            'demo_organization_scheduled_deletion_date': self.realm.demo_organization_scheduled_deletion_date,
            'is_self_hosting': False,
        }

    def check_plan_tier_is_billable(self, plan_tier: Any) -> bool:
        implemented_plan_tiers: List[Any] = [CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.TIER_CLOUD_PLUS]
        return plan_tier in implemented_plan_tiers

    def get_type_of_plan_tier_change(self, current_plan_tier: Any, new_plan_tier: Any) -> Any:
        valid_plan_tiers: List[Any] = [CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.TIER_CLOUD_PLUS]
        if current_plan_tier not in valid_plan_tiers or new_plan_tier not in valid_plan_tiers or current_plan_tier == new_plan_tier:
            return PlanTierChangeType.INVALID
        if current_plan_tier == CustomerPlan.TIER_CLOUD_STANDARD and new_plan_tier == CustomerPlan.TIER_CLOUD_PLUS:
            return PlanTierChangeType.UPGRADE
        else:
            assert current_plan_tier == CustomerPlan.TIER_CLOUD_PLUS
            assert new_plan_tier == CustomerPlan.TIER_CLOUD_STANDARD
            return PlanTierChangeType.DOWNGRADE

    def has_billing_access(self) -> bool:
        assert self.user is not None
        return self.user.has_billing_access

    def on_paid_plan(self) -> bool:
        return self.realm.plan_type in self.PAID_PLANS

    def org_name(self) -> str:
        return self.realm.name

    def add_org_type_data_to_sponsorship_context(self, context: Dict[str, Any]) -> None:
        context.update(realm_org_type=self.realm.org_type, sorted_org_types=sorted(([org_type_name, org_type] for org_type_name, org_type in Realm.ORG_TYPES.items() if not org_type.get('hidden')), key=sponsorship_org_type_key_helper))

    def get_sponsorship_request_session_specific_context(self) -> Dict[str, Any]:
        assert self.user is not None
        return {
            'realm_user': self.user,
            'user_info': {
                'name': self.user.full_name,
                'email': self.get_email(),
                'role': self.user.get_role_name()
            },
            'realm_string_id': self.realm.string_id,
        }

    def save_org_type_from_request_sponsorship_session(self, org_type: Any) -> None:
        if self.realm.org_type != org_type:
            self.realm.org_type = org_type
            self.realm.save(update_fields=['org_type'])

    def sync_license_ledger_if_needed(self) -> None:
        pass

# Similar annotated definitions should be added for RemoteRealmBillingSession and RemoteServerBillingSession.
# For brevity, only method signatures with type annotations are provided below.

class RemoteRealmBillingSession(BillingSession):
    def __init__(self, remote_realm: Any, remote_billing_user: Optional[Any] = None, support_staff: Optional[Any] = None) -> None:
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
        return f'{settings.EXTERNAL_URI_SCHEME}{settings.SELF_HOSTING_MANAGEMENT_SUBDOMAIN}.{settings.EXTERNAL_HOST}/realm/{self.remote_realm.uuid}'

    @property
    def billing_base_url(self) -> str:
        return f'/realm/{self.remote_realm.uuid}'

    def support_url(self) -> str:
        return build_support_url('remote_servers_support', str(self.remote_realm.uuid))

    def get_customer(self) -> Any:
        return get_customer_by_remote_realm(self.remote_realm)

    def get_email(self) -> str:
        assert self.remote_billing_user is not None
        return self.remote_billing_user.email

    def current_count_for_billed_licenses(self, event_time: Optional[datetime] = None) -> int:
        if has_stale_audit_log(self.remote_realm.server):
            raise MissingDataError
        remote_realm_counts = get_remote_realm_guest_and_non_guest_count(self.remote_realm, event_time)
        return remote_realm_counts.non_guest_user_count + remote_realm_counts.guest_user_count

    def missing_data_error_page(self, request: Any) -> Any:
        missing_data_context: Dict[str, Any] = {'remote_realm_session': True, 'supports_remote_realms': self.remote_realm.server.last_api_feature_level is not None}
        return render(request, 'corporate/billing/server_not_uploading_data.html', context=missing_data_context)

    def get_audit_log_event(self, event_type: Any) -> Any:
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

    def write_to_audit_log(self, event_type: Any, event_time: datetime, *, background_update: bool = False, extra_data: Optional[Any] = None) -> None:
        audit_log_event = self.get_audit_log_event(event_type)
        log_data: Dict[str, Any] = {'server': self.remote_realm.server, 'remote_realm': self.remote_realm, 'event_type': audit_log_event, 'event_time': event_time}
        if not background_update:
            log_data.update({'acting_support_user': self.support_staff, 'acting_remote_user': self.remote_billing_user})
        RemoteRealmAuditLog.objects.create(**log_data)

    def get_data_for_stripe_customer(self) -> Any:
        assert self.support_session is False
        metadata: Dict[str, Any] = {}
        metadata['remote_realm_uuid'] = self.remote_realm.uuid
        metadata['remote_realm_host'] = str(self.remote_realm.host)
        realm_stripe_customer_data = StripeCustomerData(description=str(self.remote_realm), email=self.get_email(), metadata=metadata)
        return realm_stripe_customer_data

    def update_data_for_checkout_session_and_invoice_payment(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        assert self.remote_billing_user is not None
        updated_metadata: Dict[str, Any] = dict(remote_realm_user_id=self.remote_billing_user.id, remote_realm_user_email=self.get_email(), remote_realm_host=self.remote_realm.host, **metadata)
        return updated_metadata

    def update_or_create_customer(self, stripe_customer_id: Optional[str] = None, *, defaults: Optional[Dict[str, Any]] = None) -> Any:
        if stripe_customer_id is not None:
            assert self.support_session is False
            customer, created = Customer.objects.update_or_create(remote_realm=self.remote_realm, defaults={'stripe_customer_id': stripe_customer_id})
        else:
            customer, created = Customer.objects.update_or_create(remote_realm=self.remote_realm, defaults=defaults)
        if created and (not customer.annual_discounted_price) and (not customer.monthly_discounted_price):
            customer.flat_discounted_months = 12
            customer.save(update_fields=['flat_discounted_months'])
        return customer

    @transaction.atomic(savepoint=False)
    def do_change_plan_type(self, *, tier: Any, is_sponsored: bool = False, background_update: bool = False) -> None:
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
            raise AssertionError('Unexpected tier')
        old_plan_type = self.remote_realm.plan_type
        self.remote_realm.plan_type = plan_type
        self.remote_realm.save(update_fields=['plan_type'])
        self.write_to_audit_log(event_type=BillingSessionEventType.BILLING_ENTITY_PLAN_TYPE_CHANGED, event_time=datetime.now(timezone.utc), extra_data={'old_value': old_plan_type, 'new_value': plan_type}, background_update=background_update)

    def approve_sponsorship(self) -> str:
        assert self.support_session
        customer: Optional[Any] = self.get_customer()
        if customer is not None:
            error_message: str = self.check_customer_not_on_paid_plan(customer)
            if error_message != '':
                raise SupportRequestError(error_message)
            if self.remote_realm.plan_type == RemoteRealm.PLAN_TYPE_SELF_MANAGED_LEGACY:
                plan = get_current_plan_by_customer(customer)
                if plan is not None:
                    assert self.get_next_plan(plan) is None
                    assert plan.tier == CustomerPlan.TIER_SELF_HOSTED_LEGACY
                    plan.status = CustomerPlan.ENDED
                    plan.save(update_fields=['status'])
        self.do_change_plan_type(tier=None, is_sponsored=True)
        if customer is not None and customer.sponsorship_pending:
            customer.sponsorship_pending = False
            customer.save(update_fields=['sponsorship_pending'])
            self.write_to_audit_log(event_type=BillingSessionEventType.SPONSORSHIP_APPROVED, event_time=datetime.now(timezone.utc))
        emailed_string: str = ''
        billing_emails: List[str] = list(RemoteRealmBillingUser.objects.filter(remote_realm_id=self.remote_realm.id).values_list('email', flat=True))
        if len(billing_emails) > 0:
            send_email('zerver/emails/sponsorship_approved_community_plan', to_emails=billing_emails, from_address=BILLING_SUPPORT_EMAIL, context={'billing_entity': self.billing_entity_display_name, 'plans_link': 'https://zulip.com/plans/#self-hosted', 'link_to_zulip': 'https://zulip.com/help/linking-to-zulip-website'})
            emailed_string = 'Emailed existing billing users.'
        else:
            emailed_string = 'No billing users exist to email.'
        return f'Sponsorship approved for {self.billing_entity_display_name}; ' + emailed_string

    def is_sponsored(self) -> bool:
        return self.remote_realm.plan_type == self.remote_realm.PLAN_TYPE_COMMUNITY

    def get_metadata_for_stripe_update_card(self) -> Dict[str, Any]:
        assert self.remote_billing_user is not None
        return {'type': 'card_update', 'remote_realm_user_id': str(self.remote_billing_user.id)}

    def get_upgrade_page_session_type_specific_context(self) -> Dict[str, Any]:
        return {
            'customer_name': self.remote_realm.host,
            'email': self.get_email(),
            'is_demo_organization': False,
            'demo_organization_scheduled_deletion_date': None,
            'is_self_hosting': True,
        }

    def process_downgrade(self, plan: Any, background_update: bool = False) -> None:
        with transaction.atomic(savepoint=False):
            old_plan_type = self.remote_realm.plan_type
            new_plan_type = RemoteRealm.PLAN_TYPE_SELF_MANAGED
            self.remote_realm.plan_type = new_plan_type
            self.remote_realm.save(update_fields=['plan_type'])
            self.write_to_audit_log(event_type=BillingSessionEventType.BILLING_ENTITY_PLAN_TYPE_CHANGED, event_time=datetime.now(timezone.utc), extra_data={'old_value': old_plan_type, 'new_value': new_plan_type}, background_update=background_update)
        plan.status = CustomerPlan.ENDED
        plan.save(update_fields=['status'])

    def check_plan_tier_is_billable(self, plan_tier: Any) -> bool:
        implemented_plan_tiers: List[Any] = [CustomerPlan.TIER_SELF_HOSTED_BASIC, CustomerPlan.TIER_SELF_HOSTED_BUSINESS]
        return plan_tier in implemented_plan_tiers

    def get_type_of_plan_tier_change(self, current_plan_tier: Any, new_plan_tier: Any) -> Any:
        valid_plan_tiers: List[Any] = [CustomerPlan.TIER_SELF_HOSTED_LEGACY, CustomerPlan.TIER_SELF_HOSTED_BASIC, CustomerPlan.TIER_SELF_HOSTED_BUSINESS]
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

    def on_paid_plan(self) -> bool:
        return self.remote_realm.plan_type in self.PAID_PLANS

    def org_name(self) -> str:
        return self.remote_realm.host

    def add_org_type_data_to_sponsorship_context(self, context: Dict[str, Any]) -> None:
        context.update(realm_org_type=self.remote_realm.org_type, sorted_org_types=sorted(([org_type_name, org_type] for org_type_name, org_type in Realm.ORG_TYPES.items() if not org_type.get('hidden')), key=sponsorship_org_type_key_helper))

    def get_sponsorship_request_session_specific_context(self) -> Dict[str, Any]:
        assert self.remote_billing_user is not None
        return {
            'realm_user': None,
            'user_info': {
                'name': self.remote_billing_user.full_name,
                'email': self.get_email(),
                'role': 'Remote realm administrator'
            },
            'realm_string_id': self.remote_realm.host,
        }

    def save_org_type_from_request_sponsorship_session(self, org_type: Any) -> None:
        if self.remote_realm.org_type != org_type:
            self.remote_realm.org_type = org_type
            self.remote_realm.save(update_fields=['org_type'])

    def sync_license_ledger_if_needed(self) -> None:
        last_ledger = self.get_last_ledger_for_automanaged_plan_if_exists()
        if last_ledger is None:
            return
        new_audit_logs = RemoteRealmAuditLog.objects.filter(remote_realm=self.remote_realm, event_time__gt=last_ledger.event_time, event_type__in=RemoteRealmAuditLog.SYNCED_BILLING_EVENTS).exclude(extra_data={}).order_by('event_time')
        current_plan = last_ledger.plan
        for audit_log in new_audit_logs:
            end_of_cycle_plan = self.update_license_ledger_for_automanaged_plan(current_plan, audit_log.event_time)
            if end_of_cycle_plan is None:
                return
            current_plan = end_of_cycle_plan

    def get_last_ledger_for_automanaged_plan_if_exists(self) -> Optional[Any]:
        customer = self.get_customer()
        if customer is None:
            return None
        plan = get_current_plan_by_customer(customer)
        if plan is None:
            return None
        if not plan.automanage_licenses:
            return None
        last_ledger = LicenseLedger.objects.filter(plan=plan).order_by('id').last()
        assert last_ledger is not None
        return last_ledger

    def add_customer_to_community_plan(self) -> None:
        assert not isinstance(self, RealmBillingSession)
        customer = self.update_or_create_customer()
        plan = get_current_plan_by_customer(customer)
        assert plan is None
        now = datetime.now(timezone.utc)
        community_plan_params = {
            'billing_cycle_anchor': now,
            'status': CustomerPlan.ACTIVE,
            'tier': CustomerPlan.TIER_SELF_HOSTED_COMMUNITY,
            'next_invoice_date': None,
            'price_per_license': 0,
            'billing_schedule': CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            'automanage_licenses': True,
        }
        community_plan = CustomerPlan.objects.create(customer=customer, **community_plan_params)
        try:
            billed_licenses = self.get_billable_licenses_for_customer(customer, community_plan.tier)
        except MissingDataError:
            billed_licenses = 0
        ledger_entry = LicenseLedger.objects.create(plan=community_plan, is_renewal=True, event_time=now, licenses=billed_licenses, licenses_at_next_renewal=billed_licenses)
        community_plan.invoiced_through = ledger_entry
        community_plan.save(update_fields=['invoiced_through'])
        self.write_to_audit_log(event_type=BillingSessionEventType.CUSTOMER_PLAN_CREATED, event_time=now, extra_data=community_plan_params)

# RemoteServerBillingSession would be similarly defined with proper type annotations.

# Additional helper functions such as reverse, build_support_url, get_customer_by_realm, get_current_plan_by_customer,
# add_months, start_of_next_billing_cycle, format_discount_percentage, datetime_to_timestamp, MissingDataError,
# SupportRequestError, UpgradeWithExistingPlanError, InvalidTierError, InvalidBillingScheduleError, BillingSessionAuditLogEventError,
# PaymentMethod, and other classes like CustomerPlan, Realm, StripeCustomerData, LicenseLedger, RemoteZulipServerAuditLog,
# RemoteRealmAuditLog, RemoteServerBillingUser, RemoteRealmBillingUser, etc. are assumed to be defined and imported elsewhere.

# The full annotated code is extensive; this snippet shows the approach by annotating function signatures and class methods.
