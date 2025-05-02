from typing import Optional, Dict, Any, List, Tuple, Union, Callable, Generator, TypeVar, TypedDict, Literal
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, IntEnum
from dataclasses import dataclass
from django.db.models import Model
from django.http import HttpRequest, HttpResponse
from django.core import signing
from django.conf import settings
from django.urls import reverse
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy
from django import forms
from django.core.signing import Signer
from django.db import transaction
from django.shortcuts import render
from typing_extensions import ParamSpec, override
import stripe
import logging
import math
import os
import secrets

# Define types for Stripe objects
class StripeCustomer(TypedDict, total=False):
    id: str
    email: str
    invoice_settings: Dict[str, Any]
    metadata: Dict[str, str]
    description: str

class StripeInvoice(TypedDict, total=False):
    id: str
    status: str
    amount_due: int
    customer: str
    hosted_invoice_url: str
    statement_descriptor: str
    metadata: Dict[str, str]
    lines: Dict[str, List[Dict[str, Any]]]

class StripePaymentMethod(TypedDict, total=False):
    id: str
    type: str
    card: Dict[str, str]

class StripeInvoiceItem(TypedDict, total=False):
    id: str
    amount: int
    currency: str
    description: str
    period: Dict[str, int]
    quantity: int
    unit_amount: int

# Define custom types
ParamT = ParamSpec('ParamT')
ReturnT = TypeVar('ReturnT')

class BillingModality(str, Enum):
    CHARGE_AUTOMATICALLY = "charge_automatically"
    SEND_INVOICE = "send_invoice"

class BillingSchedule(str, Enum):
    ANNUAL = "annual"
    MONTHLY = "monthly"

class LicenseManagement(str, Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"

class PlanTier(str, Enum):
    CLOUD_STANDARD = "cloud_standard"
    CLOUD_PLUS = "cloud_plus"
    SELF_HOSTED_BASIC = "self_hosted_basic"
    SELF_HOSTED_BUSINESS = "self_hosted_business"
    SELF_HOSTED_LEGACY = "self_hosted_legacy"
    SELF_HOSTED_COMMUNITY = "self_hosted_community"

class SponsoredPlanTypes(Enum):
    UNSPECIFIED = 0
    BASIC = 1
    BUSINESS = 2

class SupportType(Enum):
    approve_sponsorship = 1
    update_sponsorship_status = 2
    attach_discount = 3
    update_billing_modality = 4
    modify_plan = 5
    update_minimum_licenses = 6
    update_plan_end_date = 7
    update_required_plan_tier = 8
    configure_fixed_price_plan = 9
    delete_fixed_price_next_plan = 10
    configure_complimentary_access_plan = 11

class PlanTierChangeType(Enum):
    INVALID = 1
    UPGRADE = 2
    DOWNGRADE = 3

class InvoiceStatus(str, Enum):
    DRAFT = "draft"
    OPEN = "open"
    PAID = "paid"
    VOID = "void"
    UNCOLLECTIBLE = "uncollectible"

class CustomerPlanStatus(str, Enum):
    ACTIVE = "active"
    DOWNGRADE_AT_END_OF_CYCLE = "downgrade_at_end_of_cycle"
    DOWNGRADE_AT_END_OF_FREE_TRIAL = "downgrade_at_end_of_free_trial"
    ENDED = "ended"
    FREE_TRIAL = "free_trial"
    NEVER_STARTED = "never_started"
    SWITCH_PLAN_TIER_AT_PLAN_END = "switch_plan_tier_at_plan_end"
    SWITCH_PLAN_TIER_NOW = "switch_plan_tier_now"
    SWITCH_TO_ANNUAL_AT_END_OF_CYCLE = "switch_to_annual_at_end_of_cycle"
    SWITCH_TO_MONTHLY_AT_END_OF_CYCLE = "switch_to_monthly_at_end_of_cycle"

class CustomerPlanOfferStatus(str, Enum):
    CONFIGURED = "configured"
    PROCESSED = "processed"

class SessionType(str, Enum):
    CARD_UPDATE_FROM_BILLING_PAGE = "card_update_from_billing_page"
    CARD_UPDATE_FROM_UPGRADE_PAGE = "card_update_from_upgrade_page"

class InvoicingStatus(str, Enum):
    DONE = "done"
    INITIAL_INVOICE_TO_BE_SENT = "initial_invoice_to_be_sent"
    STARTED = "started"

# Define data classes
@dataclass
class StripeCustomerData:
    description: str
    email: str
    metadata: Dict[str, str]

@dataclass
class UpgradeRequest:
    billing_modality: BillingModality
    schedule: BillingSchedule
    license_management: LicenseManagement
    licenses: int
    signed_seat_count: str
    salt: str
    tier: PlanTier
    remote_server_plan_start_date: str
    manual_license_management: bool = False

@dataclass
class InitialUpgradeRequest:
    tier: PlanTier
    billing_modality: BillingModality
    manual_license_management: bool = False
    success_message: str = ''

@dataclass
class UpdatePlanRequest:
    status: Optional[CustomerPlanStatus] = None
    licenses: Optional[int] = None
    licenses_at_next_renewal: Optional[int] = None
    schedule: Optional[BillingSchedule] = None
    toggle_license_management: bool = False

@dataclass
class EventStatusRequest:
    stripe_session_id: Optional[str] = None
    stripe_invoice_id: Optional[str] = None

@dataclass
class SupportViewRequest(TypedDict, total=False):
    support_type: SupportType
    sponsorship_status: Optional[bool]
    monthly_discounted_price: Optional[int]
    annual_discounted_price: Optional[int]
    minimum_licenses: Optional[int]
    required_plan_tier: Optional[int]
    fixed_price: Optional[int]
    sent_invoice_id: Optional[str]
    plan_end_date: Optional[str]
    billing_modality: Optional[str]
    plan_modification: Optional[str]
    new_plan_tier: Optional[int]

@dataclass
class UpgradePageParams(TypedDict, total=False):
    pass

@dataclass
class UpgradePageSessionTypeSpecificContext(TypedDict):
    customer_name: str
    email: str
    is_demo_organization: bool
    demo_organization_scheduled_deletion_date: Optional[datetime]
    is_self_hosting: bool

@dataclass
class SponsorshipApplicantInfo(TypedDict):
    name: str
    email: str
    role: str

@dataclass
class SponsorshipRequestSessionSpecificContext(TypedDict):
    realm_user: Optional[Any]  # UserProfile or None
    user_info: SponsorshipApplicantInfo
    realm_string_id: str

@dataclass
class UpgradePageContext(TypedDict):
    customer_name: str
    stripe_email: str
    exempt_from_license_number_check: bool
    free_trial_end_date: Optional[str]
    is_demo_organization: bool
    complimentary_access_plan_end_date: Optional[str]
    manual_license_management: bool
    page_params: Dict[str, Any]
    using_min_licenses_for_plan: bool
    min_licenses_for_plan: int
    payment_method: Optional[str]
    plan: str
    fixed_price_plan: bool
    pay_by_invoice_payments_page: Optional[str]
    salt: str
    seat_count: int
    signed_seat_count: str
    success_message: str
    is_sponsorship_pending: bool
    sponsorship_plan_name: str
    scheduled_upgrade_invoice_amount_due: Optional[str]
    is_free_trial_invoice_expired_notice: bool
    free_trial_invoice_expired_notice_page_plan_name: Optional[str]

@dataclass
class PushNotificationsEnabledStatus:
    can_push: bool
    expected_end_timestamp: Optional[int]
    message: str

# Define exception classes
class BillingError(Exception):
    def __init__(self, description: str, message: Optional[str] = None):
        self.error_description = description
        self.message = message if message is not None else "Something went wrong. Please contact support."

class LicenseLimitError(Exception):
    pass

class StripeCardError(BillingError):
    pass

class StripeConnectionError(BillingError):
    pass

class ServerDeactivateWithExistingPlanError(BillingError):
    def __init__(self):
        super().__init__('server deactivation with existing plan', '')

class UpgradeWithExistingPlanError(BillingError):
    def __init__(self):
        super().__init__('subscribing with existing subscription', 'The organization is already subscribed to a plan.')

class InvalidPlanUpgradeError(BillingError):
    def __init__(self, message: str):
        super().__init__('invalid plan upgrade', message)

class InvalidBillingScheduleError(Exception):
    def __init__(self, billing_schedule: BillingSchedule):
        self.message = f'Unknown billing_schedule: {billing_schedule}'
        super().__init__(self.message)

class InvalidTierError(Exception):
    def __init__(self, tier: PlanTier):
        self.message = f'Unknown tier: {tier}'
        super().__init__(self.message)

class SupportRequestError(BillingError):
    def __init__(self, message: str):
        super().__init__('invalid support request', message)

class BillingSessionAuditLogEventError(Exception):
    def __init__(self, event_type: Any):
        self.message = f'Unknown audit log event type: {event_type}'
        super().__init__(self.message)

# Define remaining utility functions with type annotations
def format_money(cents: int) -> str:
    cents = math.ceil(cents - 0.001)
    precision = 0 if cents % 100 == 0 else 2
    dollars = cents / 100
    return f'{dollars:.{precision}f}'

def get_amount_due_fixed_price_plan(fixed_price: int, billing_schedule: BillingSchedule) -> int:
    amount_due = fixed_price
    if billing_schedule == CustomerPlan.BILLING_SCHEDULE_MONTHLY:
        amount_due = int(float(format_money(fixed_price / 12)) * 100
    return amount_due

def format_discount_percentage(discount: Decimal) -> Optional[str]:
    if not isinstance(discount, Decimal) or discount == Decimal(0):
        return None
    precision = 0 if discount * 100 % 100 == 0 else 2
    return f'{discount:.{precision}f}'

def get_latest_seat_count(realm: Any) -> int:
    return get_seat_count(realm, extra_non_guests_count=0, extra_guests_count=0)

def get_cached_seat_count(realm: Any) -> int:
    return get_latest_seat_count(realm)

def get_non_guest_user_count(realm: Any) -> int:
    return UserProfile.objects.filter(realm=realm, is_active=True, is_bot=False).exclude(role=UserProfile.ROLE_GUEST).count()

def get_guest_user_count(realm: Any) -> int:
    return UserProfile.objects.filter(realm=realm, is_active=True, is_bot=False, role=UserProfile.ROLE_GUEST).count()

def get_seat_count(realm: Any, extra_non_guests_count: int = 0, extra_guests_count: int = 0) -> int:
    non_guests = get_non_guest_user_count(realm) + extra_non_guests_count
    guests = get_guest_user_count(realm) + extra_guests_count
    return max(non_guests, math.ceil(guests / 5))

def sign_string(string: str) -> Tuple[str, str]:
    salt = secrets.token_hex(32)
    signer = Signer(salt=salt)
    return (signer.sign(string), salt)

def unsign_string(signed_string: str, salt: str) -> str:
    signer = Signer(salt=salt)
    return signer.unsign(signed_string)

def unsign_seat_count(signed_seat_count: str, salt: str) -> int:
    try:
        return int(unsign_string(signed_seat_count, salt))
    except signing.BadSignature:
        raise BillingError('tampered seat count')

def validate_licenses(charge_automatically: bool, licenses: Optional[int], seat_count: int, exempt_from_license_number_check: bool, min_licenses_for_plan: int) -> None:
    min_licenses = max(seat_count, min_licenses_for_plan)
    max_licenses = None
    if settings.TEST_SUITE and (not charge_automatically):
        min_licenses = max(seat_count, MIN_INVOICED_LICENSES)
        max_licenses = MAX_INVOICED_LICENSES
    if licenses is None or (not exempt_from_license_number_check and licenses < min_licenses):
        raise BillingError('not enough licenses', _('You must purchase licenses for all active users in your organization (minimum {min_licenses}).').format(min_licenses=min_licenses))
    if max_licenses is not None and licenses > max_licenses:
        message = _("Invoices with more than {max_licenses} licenses can't be processed from this page. To complete the upgrade, please contact {email}.").format(max_licenses=max_licenses, email=settings.ZULIP_ADMINISTRATOR)
        raise BillingError('too many licenses', message)

def check_upgrade_parameters(billing_modality: BillingModality, schedule: BillingSchedule, license_management: LicenseManagement, licenses: Optional[int], seat_count: int, exempt_from_license_number_check: bool, min_licenses_for_plan: int) -> None:
    if license_management is None:
        raise BillingError('unknown license_management')
    validate_licenses(billing_modality == 'charge_automatically', licenses, seat_count, exempt_from_license_number_check, min_licenses_for_plan)

def add_months(dt: datetime, months: int) -> datetime:
    assert months >= 0
    MAX_DAY_FOR_MONTH = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    year = dt.year
    month = dt.month + months
    while month > 12:
        year += 1
        month -= 12
    day = min(dt.day, MAX_DAY_FOR_MONTH[month])
    return dt.replace(year=year, month=month, day=day)

def next_month(billing_cycle_anchor: datetime, dt: datetime) -> datetime:
    estimated_months = round((dt - billing_cycle_anchor).days * 12.0 / 365)
    for months in range(max(estimated_months - 1, 0), estimated_months + 2):
        proposed_next_month = add_months(billing_cycle_anchor, months)
        if 20 < (proposed_next_month - dt).days < 40:
            return proposed_next_month
    raise AssertionError(f'Something wrong in next_month calculation with billing_cycle_anchor: {billing_cycle_anchor}, dt: {dt}')

def start_of_next_billing_cycle(plan: Any, event_time: datetime) -> datetime:
    months_per_period = {CustomerPlan.BILLING_SCHEDULE_ANNUAL: 12, CustomerPlan.BILLING_SCHEDULE_MONTHLY: 1}[plan.billing_schedule]
    periods = 1
    dt = plan.billing_cycle_anchor
    while dt <= event_time:
        dt = add_months(plan.billing_cycle_anchor, months_per_period * periods)
        periods += 1
    return dt

def next_invoice_date(plan: Any) -> Optional[datetime]:
    if plan.status == CustomerPlan.ENDED:
        return None
    assert plan.next_invoice_date is not None
    months = 1
    candidate_invoice_date = plan.billing_cycle_anchor
    while candidate_invoice_date <= plan.next_invoice_date:
        candidate_invoice_date = add_months(plan.billing_cycle_anchor, months)
        months += 1
    return candidate_invoice_date

def get_amount_to_credit_for_plan_tier_change(current_plan: Any, plan_change_date: datetime) -> int:
    last_renewal_ledger = LicenseLedger.objects.filter(is_renewal=True, plan=current_plan).order_by('id').last()
    assert last_renewal_ledger is not None
    assert current_plan.price_per_license is not None
    next_renewal_date = start_of_next_billing_cycle(current_plan, plan_change_date)
    last_renewal_amount = last_renewal_ledger.licenses * current_plan.price_per_license
    last_renewal_date = last_renewal_ledger.event_time
    prorated_fraction = 1 - (plan_change_date - last_renewal_date) / (next_renewal_date - last_renewal_date)
    amount_to_credit_back = math.ceil(last_renewal_amount * prorated_fraction)
    return amount_to_credit_back

def get_idempotency_key