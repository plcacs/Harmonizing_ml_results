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
from typing import Any, Literal, TypedDict, TypeVar, Optional, Union, Dict, List, Tuple
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
from zerver.lib.send_email import (
    FromAddress,
    send_email,
    send_email_to_billing_admins_and_realm_owners,
)
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

stripe.api_key = get_secret("stripe_secret_key")

BILLING_LOG_PATH = os.path.join(
    "/var/log/zulip" if not settings.DEVELOPMENT else settings.DEVELOPMENT_LOG_DIRECTORY,
    "billing.log",
)
billing_logger = logging.getLogger("corporate.stripe")
log_to_file(billing_logger, BILLING_LOG_PATH)
log_to_file(logging.getLogger("stripe"), BILLING_LOG_PATH)

ParamT = ParamSpec("ParamT")
ReturnT = TypeVar("ReturnT")

BILLING_SUPPORT_EMAIL = "sales@zulip.com"

MIN_INVOICED_LICENSES = 30
MAX_INVOICED_LICENSES = 1000
DEFAULT_INVOICE_DAYS_UNTIL_DUE = 15

CARD_CAPITALIZATION = {
    "amex": "American Express",
    "diners": "Diners Club",
    "discover": "Discover",
    "jcb": "JCB",
    "mastercard": "Mastercard",
    "unionpay": "UnionPay",
    "visa": "Visa",
}

# The version of Stripe API the billing system supports.
STRIPE_API_VERSION = "2020-08-27"

stripe.api_version = STRIPE_API_VERSION


def format_money(cents: float) -> str:
    cents = math.ceil(cents - 0.001)
    if cents % 100 == 0:
        precision = 0
    else:
        precision = 2

    dollars = cents / 100
    return f"{dollars:.{precision}f}"


def get_amount_due_fixed_price_plan(fixed_price: int, billing_schedule: int) -> int:
    amount_due = fixed_price
    if billing_schedule == CustomerPlan.BILLING_SCHEDULE_MONTHLY:
        amount_due = int(float(format_money(fixed_price / 12)) * 100)
    return amount_due


def format_discount_percentage(discount: Optional[Decimal]) -> Optional[str]:
    if type(discount) is not Decimal or discount == Decimal(0):
        return None

    if discount * 100 % 100 == 0:
        precision = 0
    else:
        precision = 2
    return f"{discount:.{precision}f}"


def get_latest_seat_count(realm: Realm) -> int:
    return get_seat_count(realm, extra_non_guests_count=0, extra_guests_count=0)


@cache_with_key(lambda realm: get_realm_seat_count_cache_key(realm.id), timeout=3600 * 24)
def get_cached_seat_count(realm: Realm) -> int:
    return get_latest_seat_count(realm)


def get_non_guest_user_count(realm: Realm) -> int:
    return (
        UserProfile.objects.filter(realm=realm, is_active=True, is_bot=False)
        .exclude(role=UserProfile.ROLE_GUEST)
        .count()
    )


def get_guest_user_count(realm: Realm) -> int:
    return UserProfile.objects.filter(
        realm=realm, is_active=True, is_bot=False, role=UserProfile.ROLE_GUEST
    ).count()


def get_seat_count(
    realm: Realm, extra_non_guests_count: int = 0, extra_guests_count: int = 0
) -> int:
    non_guests = get_non_guest_user_count(realm) + extra_non_guests_count
    guests = get_guest_user_count(realm) + extra_guests_count
    return max(non_guests, math.ceil(guests / 5))


def sign_string(string: str) -> Tuple[str, str]:
    salt = secrets.token_hex(32)
    signer = Signer(salt=salt)
    return signer.sign(string), salt


def unsign_string(signed_string: str, salt: str) -> str:
    signer = Signer(salt=salt)
    return signer.unsign(signed_string)


def unsign_seat_count(signed_seat_count: str, salt: str) -> int:
    try:
        return int(unsign_string(signed_seat_count, salt))
    except signing.BadSignature:
        raise BillingError("tampered seat count")


def validate_licenses(
    charge_automatically: bool,
    licenses: Optional[int],
    seat_count: int,
    exempt_from_license_number_check: bool,
    min_licenses_for_plan: int,
) -> None:
    min_licenses = max(seat_count, min_licenses_for_plan)
    max_licenses = None
    if settings.TEST_SUITE and not charge_automatically:
        min_licenses = max(seat_count, MIN_INVOICED_LICENSES)
        max_licenses = MAX_INVOICED_LICENSES

    if licenses is None or (not exempt_from_license_number_check and licenses < min_licenses):
        raise BillingError(
            "not enough licenses",
            _(
                "You must purchase licenses for all active users in your organization (minimum {min_licenses})."
            ).format(min_licenses=min_licenses),
        )

    if max_licenses is not None and licenses > max_licenses:
        message = _(
            "Invoices with more than {max_licenses} licenses can't be processed from this page. To"
            " complete the upgrade, please contact {email}."
        ).format(max_licenses=max_licenses, email=settings.ZULIP_ADMINISTRATOR)
        raise BillingError("too many licenses", message)


def check_upgrade_parameters(
    billing_modality: BillingModality,
    schedule: BillingSchedule,
    license_management: Optional[LicenseManagement],
    licenses: Optional[int],
    seat_count: int,
    exempt_from_license_number_check: bool,
    min_licenses_for_plan: int,
) -> None:
    if license_management is None:
        raise BillingError("unknown license_management")
    validate_licenses(
        billing_modality == "charge_automatically",
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
        "Something wrong in next_month calculation with "
        f"billing_cycle_anchor: {billing_cycle_anchor}, dt: {dt}"
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
    current_plan: CustomerPlan, plan_change_date: datetime
) -> int:
    last_renewal_ledger = (
        LicenseLedger.objects.filter(is_renewal=True, plan=current_plan).order_by("id").last()
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
    return f"ledger_entry:{ledger_entry.id}"


def cents_to_dollar_string(cents: int) -> str:
    return f"{cents / 100.0:,.2f}"


def payment_method_string(stripe_customer: stripe.Customer) -> str:
    assert stripe_customer.invoice_settings is not None
    default_payment_method = stripe_customer.invoice_settings.default_payment_method
    if default_payment_method is None:
        return _("No payment method on file.")

    assert isinstance(default_payment_method, stripe.PaymentMethod)
    if default_payment_method.type == "card":
        assert default_payment_method.card is not None
        brand_name = default_payment_method.card.brand
        if brand_name in CARD_CAPITALIZATION:
            brand_name = CARD_CAPITALIZATION[default_payment_method.card.brand]
        return _("{brand} ending in {last4}").format(
            brand=brand_name,
            last4=default_payment_method.card.last4,
        )
    return _("Unknown payment method. Please contact {email}.").format(
        email=settings.ZULIP_ADMINISTRATOR,
    )


def build_support_url(support_view: str, query_text: str) -> str:
    support_realm_url = get_realm(settings.STAFF_SUBDOMAIN).url
    support_url = urljoin(support_realm_url, reverse(support_view))
    query = urlencode({"q": query_text})
    support_url = append_url_query_string(support_url, query)
    return support_url


def get_configured_fixed_price_plan_offer(
    customer: Customer, plan_tier: int
) -> Optional[CustomerPlanOffer]:
    if plan_tier == customer.required_plan_tier:
        return CustomerPlanOffer.objects.filter(
            customer=customer,
            tier=plan_tier,
            fixed_price__isnull=False,
            status=CustomerPlanOffer.CONFIGURED,
        ).first()
    return None


class BillingError(JsonableError):
    data_fields = ["error_description"]
    CONTACT_SUPPORT = gettext_lazy("Something went wrong. Please contact {email}.")
    TRY_RELOADING = gettext_lazy("Something went wrong. Please reload the page.")

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
        super().__init__(
            "server deactivation with existing plan",
            "",
        )


class UpgradeWithExistingPlanError(BillingError):
    def __init__(self) -> None:
        super().__init__(
            "subscribing with existing subscription",
            "The organization is already subscribed to a plan. Please reload the billing page.",
        )


class InvalidPlanUpgradeError(BillingError):
    def __init__(self, message: str) -> None:
        super().__init__(
            "invalid plan upgrade",
            message,
        )


class InvalidBillingScheduleError(Exception):
    def __init__(self, billing_schedule: int) -> None:
        self.message = f"Unknown billing_schedule: {billing_schedule}"
        super().__init__(self.message)


class InvalidTierError(Exception):
    def __init__(self, tier: int) -> None:
        self.message = f"Unknown tier: {tier}"
        super().__init__(self.message)


class SupportRequestError(BillingError):
    def __init__(self, message: str) -> None:
        super().__init__(
            "invalid support request",
            message,
        )


def catch_stripe_errors(func: Callable[ParamT, ReturnT]) -> Callable[ParamT, ReturnT]:
    @wraps(func)
    def wrapped(*args: ParamT.args, **kwargs: ParamT.kwargs) -> ReturnT:
        try:
            return func(*args, **kwargs)
        except stripe.StripeError as e:
            assert isinstance(e.json_body, dict)
            err = e.json_body.get("error", {})
            if isinstance(e, stripe.CardError):
                billing_logger.info(
                    "Stripe card error: %s %s %s %s",
                    e.http_status,
                    err.get("type"),
                    err.get("code"),
                    err.get("param"),
                )
                raise StripeCardError("card error", err.get("message"))
            billing_logger.error(
                "Stripe error: %s %s %s %s",
                e.http_status,
                err.get("type"),
                err.get("code"),
                err.get("param"),
            )
            if isinstance(e, stripe.RateLimitError | stripe.APIConnectionError):
                raise StripeConnectionError(
                    "stripe connection error",
                    _("Something went wrong. Please wait a few seconds and try again."),
                )
            raise BillingError("other stripe error")

    return wrapped


@catch_stripe_errors
def stripe_get_customer(stripe_customer_id: str) -> stripe.Customer:
    return stripe.Customer.retrieve(
        stripe_customer_id, expand=["invoice_settings", "invoice_settings.default_payment_method"]
    )


def sponsorship_org_type_key_helper(d: Any) -> int:
    return d[1]["display_order"]


class PriceArgs(TypedDict, total=False):
    amount: int
    unit_amount: int
    quantity: int


@dataclass
class StripeCustomerData:
    description: str
    email: str
    metadata: Dict[str, Any]


@dataclass
class UpgradeRequest:
    billing_modality: BillingModality
    schedule: BillingSchedule
    signed_seat_count: str
    salt: str
    license_management: Optional[LicenseManagement]
    licenses: Optional[int]
    tier: int
    remote_server_plan_start_date: Optional[str]


@dataclass
class InitialUpgradeRequest:
    manual_license_management: bool
    tier: int
    billing_modality: str
    success_message: str = ""


@dataclass
class UpdatePlanRequest:
    status: Optional[int]
    licenses: Optional[int]
    licenses_at_next_renewal: Optional[int]
    schedule: Optional[int]
    toggle_license_management: bool


@dataclass
class EventStatusRequest:
    stripe_session_id: Optional[str]
    stripe_invoice_id: Optional[str]


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


class SupportViewRequest(TypedDict, total=False):
    support_type: SupportType
    sponsorship_status: Optional[bool]
    monthly_discounted_price: Optional[int]
    annual_discounted_price: Optional[int]
    billing_modality: Optional[BillingModality]
    plan_modification: Optional[str]
    new_plan_tier: Optional[int]
    minimum_licenses: Optional[int]
    plan_end_date: Optional[str]
    required_plan_tier: Optional[int]
    fixed_price: Optional[int]
    sent_invoice_id: Optional[str]


class BillingSessionEventType(IntEnum):
    STRIPE_CUSTOMER_CREATED = 1
    STRIPE_CARD_CHANGED = 2
    CUSTOMER_PLAN_CREATED = 3
    DISCOUNT_CHANGED = 4
    SPONSORSHIP_APPROVED = 5
    SPONSORSHIP_PENDING_STATUS_CHANGED = 6
    BILLING_MODALITY_CHANGED = 7
    CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN = 8
    CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_PLAN = 9
    BILLING_ENTITY_PLAN_TYPE_CHANGED = 10
    CUSTOMER_PROPERTY_CHANGED = 11
    CUSTOMER_PLAN_PROPERTY_CHANGED = 12


class PlanTierChangeType(Enum):
    INVALID = 1
    UPGRADE = 2
    DOWNGRADE = 3


class BillingSessionAuditLogEventError(Exception):
    def __init__(self, event_type: BillingSessionEventType) -> None:
        self.message = f"Unknown audit log event type: {event_type}"
        super().__init__(self.message)


class UpgradePageParams(TypedDict):
    page_type: Literal["upgrade"]
    annual_price: int
    demo_organization_scheduled_deletion_date: Optional[datetime]
    monthly_price: int
    seat_count: int
    billing_base_url: str
    tier: int
    flat_discount: int
    flat_discounted_months: int
    fixed_price: Optional[int]
    setup_payment_by_invoice: bool
    free_trial_days: Optional[int]
    percent_off_annual_price: Optional[str]
    percent_off_monthly_price: Optional[str]


class UpgradePageSessionTypeSpecificContext(TypedDict):
    customer_name: str
    email: str
    is_demo_organization: bool
    demo_organization_scheduled_deletion_date: Optional[datetime]
    is_self_hosting: bool


class SponsorshipApplicantInfo(TypedDict):
    name: str
    role: str
    email: str


class SponsorshipRequestSessionSpecificContext(TypedDict):
    realm_user: Optional[UserProfile]
    user_info: SponsorshipApplicantInfo
    realm_string_id: str


class UpgradePageContext(TypedDict):
    customer_name: str
    stripe_email: str
    exempt_from_license_number_check: bool
    free_trial_end_date: Optional[str]
    is_demo_organization: bool
    manual_license_management: bool
    using_min_licenses_for_plan: bool
    min_licenses_for_plan: int
    page_params: UpgradePageParams
    payment_method: Optional[str]
    plan: str
    fixed_price_plan: bool
    pay_by_invoice_payments_page: Optional[str]
    complimentary_access_plan_end_date: Optional[str]
    salt: str
    seat_count: int
    signed_seat_count: str
    success_message: str
    is_sponsorship_pending: bool
    sponsorship_plan_name: str
    scheduled_upgrade_invoice_amount_due: Optional[str]
    is_free_trial_invoice_expired_notice: bool
    free_trial_invoice_expired_notice_page_plan_name: Optional[str]


class SponsorshipRequestForm(forms.Form):
    website = forms.URLField(
        max_length=ZulipSponsorshipRequest.MAX_ORG_URL_LENGTH, required=False, assume_scheme="https"
    )
    organization_type = forms.IntegerField()
    description = forms.CharField(widget=forms.Textarea)
    expected_total_users = forms.CharField(widget=forms.Textarea)
    plan_to_use_zulip = forms.CharField(widget=forms.Textarea)
    paid_users_count = forms.CharField(widget=forms.Textarea)
    paid_users_description = forms.CharField(widget=forms.Textarea, required=False)
    requested_plan = forms.ChoiceField(
        choices=[(plan.value, plan.name) for plan in SponsoredPlanTypes], required=False
    )


class BillingSession(ABC):
    @property
    @abstractmethod
    def billing_entity_display_name(self) -> str:
        pass

    @property
    @abstractmethod
    def billing_session_url(self) -> str:
        pass

    @property
    @abstractmethod
    def billing_base_url(self) -> str:
        pass

    @abstractmethod
    def support_url(self) -> str:
        pass

    @abstractmethod
    def get_customer(self) -> Optional[Customer]:
        pass

    @abstractmethod
    def get_email(self) -> str:
        pass

    @abstractmethod
    def current_count_for_billed_licenses(self, event_time: Optional[datetime] = None) -> int:
        pass

    @abstractmethod
    def get_audit_log_event(self, event_type: BillingSessionEventType) -> int:
        pass

    @abstractmethod
    def write_to_audit_log(
        self,
        event_type: BillingSessionEventType,
        event_time: datetime,
        *,
        background_update: bool = False,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    @abstractmethod
    def get_data_for_stripe_customer(self) -> StripeCustomerData:
        pass

    @abstractmethod
    def update_data_for_checkout_session_and_invoice_payment(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def org_name(self) -> str:
        pass

    def customer_plan_exists(self) -> bool:
        customer = self.get_customer()

        if customer is not None and CustomerPlan.objects.filter(customer=customer).exists():
            return True

        if isinstance(self, RemoteRealmBillingSession):
            return CustomerPlan.objects.filter(
                customer=get_customer_by_remote_server(self.remote_realm.server)
            ).exists()

        return False

    def get_past_invoices_session_url(self) -> str:
        headline = "List of past invoices"
        customer = self.get_customer()
        assert customer is not None and customer.stripe_customer_id is not None

        list_params = stripe.Invoice.ListParams(
            customer=customer.stripe_customer_id,
            limit=1,
            status="paid",
        )
        list_params["total"] = 0
        if stripe.Invoice.list(**list_params).data:
            headline += " ($0 invoices include payment)"

        configuration = stripe.billing_portal.Configuration.create(
            business_profile={
                "headline": headline,
            },
            features={
                "invoice_history": {"enabled": True},
            },
        )

        return stripe.billing_portal.Session.create(
            customer=customer.stripe_customer_id,
            configuration=configuration.id,
            return_url=f"{self.billing_session_url}/billing/",
        ).url

    def get_stripe_customer_portal_url(
        self,
        return_to_billing_page: bool,
        manual_license_management: bool,
        tier: Optional[int] = None,
        setup_payment_by_invoice: bool = False,
    ) -> str:
        customer = self.get_customer()
        if customer is None or customer.stripe_customer_id is None:
            customer = self.create_stripe_customer()

        assert customer.stripe_customer_id is not None

        if return_to_billing_page or tier is None:
            return_url = f"{self.billing_session_url}/billing/"
        else:
            base_return_url = f"{self.billing_session_url}/upgrade/"
            params = {
                "manual_license_management": str(manual_license_management).lower(),
                "tier": str(tier),
                "setup_payment_by_invoice": str(setup_payment_by_invoice).lower(),
            }
            return_url = f"{base_return_url}?{urlencode(params)}"

        configuration = stripe.billing_portal.Configuration.create(
            business_profile={
                "headline": "Invoice and receipt billing information",
            },
            features={
                "customer_update": {
                    "enabled": True,
                    "allowed_updates": ["address", "name", "email"],
                }
            },
        )

        return stripe.billing_portal.Session.create(
            customer=customer.stripe_customer_id,
            configuration=configuration.id,
            return_url=return_url,
        ).url

    def generate_invoice_for_upgrade(
        self,
        customer: Customer,
        price_per_license: Optional[int],
        fixed_price: Optional[int],
        licenses: int,
        plan_tier: int,
        billing_schedule: int,
        charge_automatically: bool,
        invoice_period: stripe.InvoiceItem.CreateParamsPeriod,
        license_management: Optional[str] = None,
        days_until_due: Optional[int] = None,
        on_free_trial: bool = False,
        current_plan_id: Optional[int] = None,
    ) -> stripe.Invoice:
        assert customer.stripe_customer_id is not None
        plan_name = CustomerPlan.name_from_tier(plan_tier)
        assert price_per_license is None or fixed_price is None
        price_args: PriceArgs = {}
        if fixed_price is None:
            assert price_per_license is not None
            price_args = {
                "quantity": licenses,
                "unit_amount": price_per_license,
            }
        else:
            assert fixed_price is not None
            amount_due = get_amount_due_fixed_price_plan(fixed_price, billing_schedule)
            price_args = {"amount": amount_due}

        stripe.InvoiceItem.create(
            currency="usd",
            customer=customer.stripe_customer_id,
            description=plan_name,
            discountable=False,
            period=invoice_period,
            **price_args,
        )

        if fixed_price is None and customer.flat_discounted_months > 0:
            num_months = 12 if billing_schedule == CustomerPlan.BILLING_SCHEDULE_ANNUAL else 1
            flat_discounted_months = min(customer.flat_discounted_months, num_months)
            discount = customer.flat_discount * flat_discounted_months
            customer.flat_discounted_months -= flat_discounted_months
            customer.save(update_fields=["flat_discounted_months"])

            stripe.InvoiceItem.create(
                currency="usd",
                customer=customer.stripe_customer_id,
                description=f"${cents_to_dollar_string(customer.flat_discount)}/month new customer discount",
                amount=(-1 * discount),
                period=invoice_period,
            )

        if charge_automatically:
            collection_method: Literal["charge_automatically", "send_invoice"] = (
                "charge_automatically"
            )
        else:
            collection_method = "send_invoice"
            if days_until_due is None:
                days_until_due = 1

        metadata = {
            "plan_tier": str(plan_tier),
            "billing_schedule": str(billing_schedule),
            "licenses": str(licenses),
            "license_management": str(license_management),
            "on_free_trial": str(on_free_trial),
            "current_plan_id": str(current_plan_id),
        }

        if hasattr(self, "user"):
            metadata["user_id"] = self.user.id

        auto_advance = not charge_automatically
        invoice_params = stripe.Invoice.CreateParams(
            auto_advance=auto_advance,
            collection_method=collection_method,
            customer=customer.stripe_customer_id,
            statement_descriptor=plan_name,
            metadata=metadata,
        )
        if days_until_due is not None:
            invoice_params["days_until_due"] = days_until_due
        stripe_invoice = stripe.Invoice.create(**invoice_params)
        stripe.Invoice.finalize_invoice(stripe_invoice)
        return stripe_invoice

    @abstractmethod
    def update_or_create_customer(
        self, stripe_customer_id: Optional[str] = None, *, defaults: Optional[Dict[str, Any]] = None
    ) -> Customer:
        pass

    @abstractmethod
    def do_change_plan_type(
        self, *, tier: Optional[int], is_sponsored: bool = False, background_update: bool = False
    ) -> None:
        pass

    @abstractmethod
    def process_downgrade(self, plan: CustomerPlan, background_update: bool = False) -> None:
        pass

    @abstractmethod
    def approve_sponsorship(self) -> str:
        pass

    @abstractmethod
    def is_sponsored(self) -> bool:
        pass

    @abstractmethod
    def get_sponsorship_request_session_specific_context(
        self,
    ) -> SponsorshipRequestSessionSpecificContext:
        pass

    @abstractmethod
    def save_org_type_from_request_sponsorship_session(self, org_type: int) -> None:
        pass

    @abstractmethod
    def get_upgrade_page_session_type_specific_context(
        self,
    ) -> UpgradePageSessionTypeSpecificContext:
        pass

    @abstractmethod
    def check_plan_tier_is_billable(self, plan_tier: int) -> bool:
        pass

    @abstractmethod
    def get_type_of_plan_tier_change(
        self, current_plan_tier: int, new_plan_tier: int
    ) -> PlanTierChangeType:
        pass

    @abstractmethod
    def has_billing_access(self) -> bool:
        pass

    @abstractmethod
    def on_paid_plan(self) -> bool:
        pass

    @abstractmethod
    def add_org_type_data_to_sponsorship_context(self, context: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_metadata_for_stripe_update_card(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def sync_license_ledger_if_needed(self) -> None:
        pass

    def is_sponsored_or_pending(self, customer: Optional[Customer]) -> bool:
        if (customer is not None and customer.sponsorship_pending) or self.is_sponsored():
            return True
        return False

    def get_complimentary_access_plan(
        self, customer: Optional[Customer], status: int = CustomerPlan.ACTIVE
    ) -> Optional[CustomerPlan]:
        if customer is None:
            return None

        plan_tier = CustomerPlan.TIER_SELF_HOSTED_LEGACY
        if isinstance(self, RealmBillingSession):
            return None

        return CustomerPlan.objects.filter(
            customer=customer,
            tier=plan_tier,
            status=status,
        ).first()

    def get_formatted_complimentary_access_plan_end_date(
        self, customer: Optional[Customer], status: int = CustomerPlan.ACTIVE
    ) -> Optional[str]:
        complimentary_access_plan = self.get_complimentary_access_plan(customer, status)
        if complimentary_access_plan is None:
            return None

        assert complimentary_access_plan.end_date is not None
        return complimentary_access_plan.end_date.strftime("%B %d, %Y")

    def get_complimentary_access_next_plan(self, customer: Customer) -> Optional[CustomerPlan]:
        complimentary_access_plan = self.get_complimentary_access_plan(
            customer, CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END
        )
        if complimentary_access_plan is None:
            return None

        assert complimentary_access_plan.end_date is not None
        return CustomerPlan.objects.get