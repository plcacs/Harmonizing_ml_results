import logging
import math
import os
import secrets
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum, IntEnum
from functools import wraps
from typing import Any, Literal, Optional, TypedDict, TypeVar, Union, cast
from urllib.parse import urlencode, urljoin
import stripe
from django import forms
from django.conf import settings
from django.core import signing
from django.core.signing import Signer
from django.db import transaction
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.urls import reverse
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy
from django.utils.translation import override as override_language
from typing_extensions import ParamSpec, override
from corporate.lib.billing_types import BillingModality, BillingSchedule, LicenseManagement
from corporate.models import (
    Customer,
    CustomerPlan,
    CustomerPlanOffer,
    Invoice,
    LicenseLedger,
    Session,
    SponsoredPlanTypes,
    ZulipSponsorshipRequest,
    get_current_plan_by_customer,
    get_current_plan_by_realm,
    get_customer_by_realm,
    get_customer_by_remote_realm,
    get_customer_by_remote_server,
)
from zerver.lib.cache import cache_with_key, get_realm_seat_count_cache_key
from zerver.lib.exceptions import JsonableError
from zerver.lib.logging_util import log_to_file
from zerver.lib.send_email import FromAddress, send_email, send_email_to_billing_admins_and_realm_owners
from zerver.lib.timestamp import datetime_to_timestamp, timestamp_to_datetime
from zerver.lib.url_encoding import append_url_query_string
from zerver.lib.utils import assert_is_not_none
from zerver.models import Realm, RealmAuditLog, Stream, UserProfile
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.realms import get_org_type_display_name, get_realm
from zerver.models.streams import get_stream
from zerver.models.users import get_system_bot
from zilencer.lib.remote_counts import MissingDataError
from zilencer.models import (
    RemoteRealm,
    RemoteRealmAuditLog,
    RemoteRealmBillingUser,
    RemoteServerBillingUser,
    RemoteZulipServer,
    RemoteZulipServerAuditLog,
    get_remote_realm_guest_and_non_guest_count,
    get_remote_server_guest_and_non_guest_count,
    has_stale_audit_log,
)
from zproject.config import get_secret

stripe.api_key = get_secret('stripe_secret_key')
BILLING_LOG_PATH = os.path.join(
    '/var/log/zulip' if not settings.DEVELOPMENT else settings.DEVELOPMENT_LOG_DIRECTORY,
    'billing.log',
)
billing_logger = logging.getLogger('corporate.stripe')
log_to_file(billing_logger, BILLING_LOG_PATH)
log_to_file(logging.getLogger('stripe'), BILLING_LOG_PATH)

ParamT = ParamSpec('ParamT')
ReturnT = TypeVar('ReturnT')
BILLING_SUPPORT_EMAIL = 'sales@zulip.com'
MIN_INVOICED_LICENSES = 30
MAX_INVOICED_LICENSES = 1000
DEFAULT_INVOICE_DAYS_UNTIL_DUE = 15
CARD_CAPITALIZATION = {
    'amex': 'American Express',
    'diners': 'Diners Club',
    'discover': 'Discover',
    'jcb': 'JCB',
    'mastercard': 'Mastercard',
    'unionpay': 'UnionPay',
    'visa': 'Visa',
}
STRIPE_API_VERSION = '2020-08-27'
stripe.api_version = STRIPE_API_VERSION

def format_money(cents: float) -> str:
    cents = math.ceil(cents - 0.001)
    if cents % 100 == 0:
        precision = 0
    else:
        precision = 2
    dollars = cents / 100
    return f'{dollars:.{precision}f}'

def get_amount_due_fixed_price_plan(fixed_price: int, billing_schedule: int) -> int:
    amount_due = fixed_price
    if billing_schedule == CustomerPlan.BILLING_SCHEDULE_MONTHLY:
        amount_due = int(float(format_money(fixed_price / 12)) * 100
    return amount_due

def format_discount_percentage(discount: Decimal) -> Optional[str]:
    if type(discount) is not Decimal or discount == Decimal(0):
        return None
    if discount * 100 % 100 == 0:
        precision = 0
    else:
        precision = 2
    return f'{discount:.{precision}f}'

def get_latest_seat_count(realm: Realm) -> int:
    return get_seat_count(realm, extra_non_guests_count=0, extra_guests_count=0)

@cache_with_key(lambda realm: get_realm_seat_count_cache_key(realm.id), timeout=3600 * 24)
def get_cached_seat_count(realm: Realm) -> int:
    return get_latest_seat_count(realm)

def get_non_guest_user_count(realm: Realm) -> int:
    return UserProfile.objects.filter(
        realm=realm, is_active=True, is_bot=False
    ).exclude(role=UserProfile.ROLE_GUEST).count()

def get_guest_user_count(realm: Realm) -> int:
    return UserProfile.objects.filter(
        realm=realm, is_active=True, is_bot=False, role=UserProfile.ROLE_GUEST
    ).count()

def get_seat_count(
    realm: Realm,
    extra_non_guests_count: int = 0,
    extra_guests_count: int = 0,
) -> int:
    non_guests = get_non_guest_user_count(realm) + extra_non_guests_count
    guests = get_guest_user_count(realm) + extra_guests_count
    return max(non_guests, math.ceil(guests / 5))

def sign_string(string: str) -> tuple[str, str]:
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

def validate_licenses(
    charge_automatically: bool,
    licenses: Optional[int],
    seat_count: int,
    exempt_from_license_number_check: bool,
    min_licenses_for_plan: int,
) -> None:
    min_licenses = max(seat_count, min_licenses_for_plan)
    max_licenses = None
    if settings.TEST_SUITE and (not charge_automatically):
        min_licenses = max(seat_count, MIN_INVOICED_LICENSES)
        max_licenses = MAX_INVOICED_LICENSES
    if licenses is None or (not exempt_from_license_number_check and licenses < min_licenses):
        raise BillingError(
            'not enough licenses',
            _('You must purchase licenses for all active users in your organization (minimum {min_licenses}).').format(
                min_licenses=min_licenses
            ),
        )
    if max_licenses is not None and licenses > max_licenses:
        message = _(
            "Invoices with more than {max_licenses} licenses can't be processed from this page. "
            "To complete the upgrade, please contact {email}."
        ).format(
            max_licenses=max_licenses,
            email=settings.ZULIP_ADMINISTRATOR,
        )
        raise BillingError('too many licenses', message)

def check_upgrade_parameters(
    billing_modality: str,
    schedule: str,
    license_management: Optional[str],
    licenses: Optional[int],
    seat_count: int,
    exempt_from_license_number_check: bool,
    min_licenses_for_plan: int,
) -> None:
    if license_management is None:
        raise BillingError('unknown license_management')
    validate_licenses(
        billing_modality == 'charge_automatically',
        licenses,
        seat_count,
        exempt_from_license_number_check,
        min_licenses_for_plan,
    )

def add_months(dt: datetime, months: int) -> datetime:
    assert months >= 0
    MAX_DAY_FOR_MONTH = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }
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
    raise AssertionError(
        f'Something wrong in next_month calculation with billing_cycle_anchor: {billing_cycle_anchor}, dt: {dt}'
    )

def start_of_next_billing_cycle(plan: CustomerPlan, event_time: datetime) -> datetime:
    months_per_period = {
        CustomerPlan.BILLING_SCHEDULE_ANNUAL: 12,
        CustomerPlan.BILLING_SCHEDULE_MONTHLY: 1,
    }[plan.billing_schedule]
    periods = 1
    dt = plan.billing_cycle_anchor
    while dt <= event_time:
        dt = add_months(plan.billing_cycle_anchor, months_per_period * periods)
        periods += 1
    return dt

def next_invoice_date(plan: CustomerPlan) -> Optional[datetime]:
    if plan.status == CustomerPlan.ENDED:
        return None
    assert plan.next_invoice_date is not None
    months = 1
    candidate_invoice_date = plan.billing_cycle_anchor
    while candidate_invoice_date <= plan.next_invoice_date:
        candidate_invoice_date = add_months(plan.billing_cycle_anchor, months)
        months += 1
    return candidate_invoice_date

def get_amount_to_credit_for_plan_tier_change(
    current_plan: CustomerPlan,
    plan_change_date: datetime,
) -> int:
    last_renewal_ledger = (
        LicenseLedger.objects.filter(is_renewal=True, plan=current_plan)
        .order_by('id')
        .last()
    )
    assert last_renewal_ledger is not None
    assert current_plan.price_per_license is not None
    next_renewal_date = start_of_next_billing_cycle(current_plan, plan_change_date)
    last_renewal_amount = last_renewal_ledger.licenses * current_plan.price_per_license
    last_renewal_date = last_renewal_ledger.event_time
    prorated_fraction = 1 - (plan_change_date - last_renewal_date) / (
        next_renewal_date - last_renewal_date
    )
    amount_to_credit_back = math.ceil(last_renewal_amount * prorated_fraction)
    return amount_to_credit_back

def get_idempotency_key(ledger_entry: LicenseLedger) -> Optional[str]:
    if settings.TEST_SUITE:
        return None
    return f'ledger_entry:{ledger_entry.id}'

def cents_to_dollar_string(cents: int) -> str:
    return f'{cents / 100.0:,.2f}'

def payment_method_string(stripe_customer: stripe.Customer) -> str:
    assert stripe_customer.invoice_settings is not None
    default_payment_method = stripe_customer.invoice_settings.default_payment_method
    if default_payment_method is None:
        return _('No payment method on file.')
    assert isinstance(default_payment_method, stripe.PaymentMethod)
    if default_payment_method.type == 'card':
        assert default_payment_method.card is not None
        brand_name = default_payment_method.card.brand
        if brand_name in CARD_CAPITALIZATION:
            brand_name = CARD_CAPITALIZATION[default_payment_method.card.brand]
        return _('{brand} ending in {last4}').format(
            brand=brand_name,
            last4=default_payment_method.card.last4,
        )
    return _('Unknown payment method. Please contact {email}.').format(
        email=settings.ZULIP_ADMINISTRATOR
    )

def build_support_url(support_view: str, query_text: str) -> str:
    support_realm_url = get_realm(settings.STAFF_SUBDOMAIN).url
    support_url = urljoin(support_realm_url, reverse(support_view))
    query = urlencode({'q': query_text})
    support_url = append_url_query_string(support_url, query)
    return support_url

def get_configured_fixed_price_plan_offer(
    customer: Customer,
    plan_tier: int,
) -> Optional[CustomerPlanOffer]:
    """
    Fixed price plan offer configured via /support which the
    customer is yet to buy or schedule a purchase.
    """
    if plan_tier == customer.required_plan_tier:
        return CustomerPlanOffer.objects.filter(
            customer=customer,
            tier=plan_tier,
            fixed_price__isnull=False,
            status=CustomerPlanOffer.CONFIGURED,
        ).first()
    return None

class BillingError(JsonableError):
    data_fields = ['error_description']
    CONTACT_SUPPORT = gettext_lazy('Something went wrong. Please contact {email}.')
    TRY_RELOADING = gettext_lazy('Something went wrong. Please reload the page.')

    def __init__(self, description: str, message: Optional[str] = None) -> None:
        self.error_description = description
        if message is None:
            message = BillingError.CONTACT_SUPPORT.format(email=settings.ZULIP_ADMINISTRATOR)
        super().__init__(message)

class LicenseLimitError(Exception):
    pass

class StripeCardError(BillingError):
    pass

class StripeConnectionError(BillingError):
    pass

class ServerDeactivateWithExistingPlanError(BillingError):
    def __init__(self) -> None:
        super().__init__('server deactivation with existing plan', '')

class UpgradeWithExistingPlanError(BillingError):
    def __init__(self) -> None:
        super().__init__(
            'subscribing with existing subscription',
            'The organization is already subscribed to a plan. Please reload the billing page.',
        )

class InvalidPlanUpgradeError(BillingError):
    def __init__(self, message: str) -> None:
        super().__init__('invalid plan upgrade', message)

class InvalidBillingScheduleError(Exception):
    def __init__(self, billing_schedule: int) -> None:
        self.message = f'Unknown billing_schedule: {billing_schedule}'
        super().__init__(self.message)

class InvalidTierError(Exception):
    def __init__(self, tier: int) -> None:
        self.message = f'Unknown tier: {tier}'
        super().__init__(self.message)

class SupportRequestError(BillingError):
    def __init__(self, message: str) -> None:
        super().__init__('invalid support request', message)

def catch_stripe_errors(func: Callable[ParamT, ReturnT]) -> Callable[ParamT, ReturnT]:
    @wraps(func)
    def wrapped(*args: ParamT.args, **kwargs: ParamT.kwargs) -> ReturnT:
        try:
            return func(*args, **kwargs)
        except stripe.StripeError as e:
            assert isinstance(e.json_body, dict)
            err = e.json_body.get('error', {})
            if isinstance(e, stripe.CardError):
                billing_logger.info(
                    'Stripe card error: %s %s %s %s',
                    e.http_status,
                    err.get('type'),
                    err.get('code'),
                    err.get('param'),
                )
                raise StripeCardError('card error', err.get('message'))
            billing_logger.error(
                'Stripe error: %s %s %s %s',
                e.http_status,
                err.get('type'),
                err.get('code'),
                err.get('param'),
            )
            if isinstance(e, stripe.RateLimitError | stripe.APIConnectionError):
                raise StripeConnectionError(
                    'stripe connection error',
                    _('Something went wrong. Please wait a few seconds and try again.'),
                )
            raise BillingError('other stripe error')
    return wrapped

@catch_stripe_errors
def stripe_get_customer(stripe_customer_id: str) -> stripe.Customer:
    return stripe.Customer.retrieve(
        stripe_customer_id,
        expand=['invoice_settings', 'invoice_settings.default_payment_method'],
    )

def sponsorship_org_type_key_helper(d: tuple[str, dict[str, Any]]) -> int:
    return d[1]['display_order']

class PriceArgs(TypedDict, total=False):
    pass

@dataclass
class