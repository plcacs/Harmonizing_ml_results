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
from typing import Any, Dict, List, Literal, Optional, TypedDict, TypeVar, Union
from urllib.parse import urlencode, urljoin

import stripe
from django import forms
from django.conf import settings
from django.core import signing
from django.core.signing import BadSignature, Signer
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
BILLING_LOG_PATH: str = os.path.join(
    "/var/log/zulip" if not settings.DEVELOPMENT else settings.DEVELOPMENT_LOG_DIRECTORY, "billing.log"
)
billing_logger: logging.Logger = logging.getLogger("corporate.stripe")
log_to_file(billing_logger, BILLING_LOG_PATH)
log_to_file(logging.getLogger("stripe"), BILLING_LOG_PATH)
ParamT = ParamSpec("ParamT")
ReturnT = TypeVar("ReturnT")
BILLING_SUPPORT_EMAIL: str = "sales@zulip.com"
MIN_INVOICED_LICENSES: int = 30
MAX_INVOICED_LICENSES: int = 1000
DEFAULT_INVOICE_DAYS_UNTIL_DUE: int = 15
CARD_CAPITALIZATION: Dict[str, str] = {
    "amex": "American Express",
    "diners": "Diners Club",
    "discover": "Discover",
    "jcb": "JCB",
    "mastercard": "Mastercard",
    "unionpay": "UnionPay",
    "visa": "Visa",
}
STRIPE_API_VERSION: str = "2020-08-27"
stripe.api_version = STRIPE_API_VERSION


def format_money(cents: Union[int, float]) -> str:
    cents = math.ceil(cents - 0.001)
    if cents % 100 == 0:
        precision = 0
    else:
        precision = 2
    dollars = cents / 100
    return f"{dollars:.{precision}f}"


def get_amount_due_fixed_price_plan(
    fixed_price: int, billing_schedule: BillingSchedule
) -> int:
    amount_due = fixed_price
    if billing_schedule == CustomerPlan.BILLING_SCHEDULE_MONTHLY:
        amount_due = int(float(format_money(fixed_price / 12)) * 100)
    return amount_due


def format_discount_percentage(discount: Optional[Decimal]) -> Optional[str]:
    if type(discount) is not Decimal or discount == Decimal(0):
        return None
    if (discount * 100) % 100 == 0:
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
    return (
        UserProfile.objects.filter(
            realm=realm, is_active=True, is_bot=False, role=UserProfile.ROLE_GUEST
        ).count()
    )


def get_seat_count(
    realm: Realm, extra_non_guests_count: int = 0, extra_guests_count: int = 0
) -> int:
    non_guests = get_non_guest_user_count(realm) + extra_non_guests_count
    guests = get_guest_user_count(realm) + extra_guests_count
    return max(non_guests, math.ceil(guests / 5))


def sign_string(string: str) -> Tuple[str, str]:
    salt: str = secrets.token_hex(32)
    signer: Signer = Signer(salt=salt)
    return signer.sign(string), salt


def unsign_string(signed_string: str, salt: str) -> str:
    signer: Signer = Signer(salt=salt)
    return signer.unsign(signed_string)


def unsign_seat_count(signed_seat_count: str, salt: str) -> int:
    try:
        return int(unsign_string(signed_seat_count, salt))
    except BadSignature:
        raise BillingError("tampered seat count")


def validate_licenses(
    charge_automatically: bool,
    licenses: Optional[int],
    seat_count: int,
    exempt_from_license_number_check: bool,
    min_licenses_for_plan: int,
) -> None:
    min_licenses = max(seat_count, min_licenses_for_plan)
    max_licenses: Optional[int] = None
    if settings.TEST_SUITE and (not charge_automatically):
        min_licenses = max(seat_count, MIN_INVOICED_LICENSES)
        max_licenses = MAX_INVOICED_LICENSES
    if licenses is None or (
        not exempt_from_license_number_check and licenses < min_licenses
    ):
        raise BillingError(
            "not enough licenses",
            _(
                "You must purchase licenses for all active users in your organization (minimum {min_licenses})."
            ).format(min_licenses=min_licenses),
        )
    if max_licenses is not None and licenses > max_licenses:
        message = _(
            "Invoices with more than {max_licenses} licenses can't be processed from this page. To complete the upgrade, please contact {email}."
        ).format(
            max_licenses=max_licenses, email=settings.ZULIP_ADMINISTRATOR
        )
        raise BillingError("too many licenses", message)


def check_upgrade_parameters(
    billing_modality: BillingModality,
    schedule: BillingSchedule,
    license_management: LicenseManagement,
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
    MAX_DAY_FOR_MONTH: Dict[int, int] = {
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
        f"Something wrong in next_month calculation with billing_cycle_anchor: {billing_cycle_anchor}, dt: {dt}"
    )


def start_of_next_billing_cycle(plan: CustomerPlan, event_time: datetime) -> datetime:
    months_per_period: Dict[str, int] = {
        CustomerPlan.BILLING_SCHEDULE_ANNUAL: 12,
        CustomerPlan.BILLING_SCHEDULE_MONTHLY: 1,
    }
    months = months_per_period[plan.billing_schedule]
    periods = 1
    dt = plan.billing_cycle_anchor
    while dt <= event_time:
        dt = add_months(plan.billing_cycle_anchor, months * periods)
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
        LicenseLedger.objects.filter(is_renewal=True, plan=current_plan)
        .order_by("id")
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
            brand=brand_name, last4=default_payment_method.card.last4
        )
    return _(
        "Unknown payment method. Please contact {email}."
    ).format(email=settings.ZULIP_ADMINISTRATOR)


def build_support_url(support_view: str, query_text: str) -> str:
    support_realm_url: str = get_realm(settings.STAFF_SUBDOMAIN).url
    support_url: str = urljoin(support_realm_url, reverse(support_view))
    query: str = urlencode({"q": query_text})
    support_url = append_url_query_string(support_url, query)
    return support_url


def get_configured_fixed_price_plan_offer(
    customer: Customer, plan_tier: int
) -> Optional[CustomerPlanOffer]:
    """
    Fixed price plan offer configured via /support which the
    customer is yet to buy or schedule a purchase.
    """
    if plan_tier == customer.required_plan_tier:
        return (
            CustomerPlanOffer.objects.filter(
                customer=customer,
                tier=plan_tier,
                fixed_price__isnull=False,
                status=CustomerPlanOffer.CONFIGURED,
            )
            .first()
        )
    return None


class BillingError(JsonableError):
    data_fields: List[str] = ["error_description"]
    CONTACT_SUPPORT: "gettext_lazy" = gettext_lazy(
        "Something went wrong. Please contact {email}."
    )
    TRY_RELOADING: "gettext_lazy" = gettext_lazy(
        "Something went wrong. Please reload the page."
    )

    def __init__(self, description: str, message: Optional[str] = None) -> None:
        self.error_description: str = description
        if message is None:
            message = BillingError.CONTACT_SUPPORT.format(
                email=settings.ZULIP_ADMINISTRATOR
            )
        super().__init__(message)


class LicenseLimitError(Exception):
    pass


class StripeCardError(BillingError):
    pass


class StripeConnectionError(BillingError):
    pass


class ServerDeactivateWithExistingPlanError(BillingError):
    def __init__(self) -> None:
        super().__init__("server deactivation with existing plan", "")


class UpgradeWithExistingPlanError(BillingError):
    def __init__(self) -> None:
        super().__init__(
            "subscribing with existing subscription",
            "The organization is already subscribed to a plan. Please reload the billing page.",
        )


class InvalidPlanUpgradeError(BillingError):
    def __init__(self, message: str) -> None:
        super().__init__("invalid plan upgrade", message)


class InvalidBillingScheduleError(Exception):
    def __init__(self, billing_schedule: Any) -> None:
        self.message: str = f"Unknown billing_schedule: {billing_schedule}"
        super().__init__(self.message)


class InvalidTierError(Exception):
    def __init__(self, tier: Any) -> None:
        self.message: str = f"Unknown tier: {tier}"
        super().__init__(self.message)


class SupportRequestError(BillingError):
    def __init__(self, message: str) -> None:
        super().__init__("invalid support request", message)


def catch_stripe_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except stripe.StripeError as e:
            assert isinstance(e.json_body, dict)
            err: Dict[str, Any] = e.json_body.get("error", {})
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
            if isinstance(e, (stripe.RateLimitError, stripe.APIConnectionError)):
                raise StripeConnectionError(
                    "stripe connection error",
                    _("Something went wrong. Please wait a few seconds and try again."),
                )
            raise BillingError("other stripe error")

    return wrapped


@catch_stripe_errors
def stripe_get_customer(stripe_customer_id: str) -> stripe.Customer:
    return stripe.Customer.retrieve(
        stripe_customer_id,
        expand=["invoice_settings", "invoice_settings.default_payment_method"],
    )


class PriceArgs(TypedDict, total=False):
    pass


@dataclass
class StripeCustomerData:
    description: str
    email: str
    metadata: Dict[str, Any]


@dataclass
class UpgradeRequest:
    tier: int
    billing_modality: str
    schedule: str
    license_management: str
    licenses: Optional[int]
    signed_seat_count: str
    salt: str
    remote_server_plan_start_date: Optional[str] = None


@dataclass
class InitialUpgradeRequest:
    tier: int
    billing_modality: str
    schedule: str
    license_management: str
    licenses: Optional[int]
    signed_seat_count: str
    salt: str
    success_message: str = ""
    manual_license_management: bool = False


@dataclass
class UpdatePlanRequest:
    toggle_license_management: Optional[bool] = None
    status: Optional[str] = None
    licenses: Optional[int] = None
    licenses_at_next_renewal: Optional[int] = None
    schedule: Optional[str] = None


@dataclass
class EventStatusRequest:
    stripe_session_id: Optional[str] = None
    stripe_invoice_id: Optional[str] = None


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
    sponsorship_status: Optional[bool] = None
    monthly_discounted_price: Optional[int] = None
    annual_discounted_price: Optional[int] = None
    minimum_licenses: Optional[int] = None
    required_plan_tier: Optional[int] = None
    fixed_price: Optional[int] = None
    sent_invoice_id: Optional[str] = None
    plan_end_date: Optional[str] = None
    billing_modality: Optional[str] = None
    plan_modification: Optional[str] = None
    new_plan_tier: Optional[int] = None


class BillingSessionEventType(IntEnum):
    STRIPE_CUSTOMER_CREATED = 1
    STRIPE_CARD_CHANGED = 2
    CUSTOMER_PLAN_CREATED = 3
    DISCOUNT_CHANGED = 4
    SPONSORSHIP_APPROVED = 5
    SPONSORSHIP_PENDING_STATUS_CHANGED = 6
    BILLING_MODALITY_CHANGED = 7
    CUSTOMER_SWITCHED_FROM_MONTHLY_TO_ANNUAL_PLAN = 8
    CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_plan = 9
    BILLING_ENTITY_PLAN_TYPE_CHANGED = 10
    CUSTOMER_PROPERTY_CHANGED = 11
    CUSTOMER_PLAN_PROPERTY_CHANGED = 12


class PlanTierChangeType(Enum):
    INVALID = 1
    UPGRADE = 2
    DOWNGRADE = 3


class BillingSessionAuditLogEventError(Exception):
    def __init__(self, event_type: BillingSessionEventType) -> None:
        self.message: str = f"Unknown audit log event type: {event_type}"
        super().__init__(self.message)


class UpgradePageParams(TypedDict):
    pass


class UpgradePageSessionTypeSpecificContext(TypedDict):
    customer_name: str
    email: str
    is_demo_organization: bool
    demo_organization_scheduled_deletion_date: Optional[datetime]
    is_self_hosting: bool


class SponsorshipApplicantInfo(TypedDict):
    name: str
    email: str
    role: str


class SponsorshipRequestSessionSpecificContext(TypedDict):
    realm_user: Optional[UserProfile]
    user_info: SponsorshipApplicantInfo
    realm_string_id: str


class UpgradePageContext(TypedDict):
    pass


class SponsorshipRequestForm(forms.Form):
    website = forms.URLField(
        max_length=ZulipSponsorshipRequest.MAX_ORG_URL_LENGTH,
        required=False,
        assume_scheme="https",
    )
    organization_type = forms.IntegerField()
    description = forms.CharField(widget=forms.Textarea)
    expected_total_users = forms.CharField(widget=forms.Textarea)
    plan_to_use_zulip = forms.CharField(widget=forms.Textarea)
    paid_users_count = forms.CharField(widget=forms.Textarea)
    paid_users_description = forms.CharField(
        widget=forms.Textarea, required=False
    )
    requested_plan = forms.ChoiceField(
        choices=[(plan.value, plan.name) for plan in SponsoredPlanTypes],
        required=False,
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
    def get_audit_log_event(
        self, event_type: BillingSessionEventType
    ) -> AuditLogEventType:
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
        headline: str = "List of past invoices"
        customer = self.get_customer()
        assert customer is not None and customer.stripe_customer_id is not None
        list_params: stripe.Invoice.ListParams = stripe.Invoice.ListParams(
            customer=customer.stripe_customer_id,
            limit=1,
            status="paid",
        )
        list_params["total"] = 0
        if stripe.Invoice.list(**list_params).data:
            headline += " ($0 invoices include payment)"
        configuration: stripe.billing_portal.Configuration = stripe.billing_portal.Configuration.create(
            business_profile={"headline": headline},
            features={"invoice_history": {"enabled": True}},
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
            return_url: str = f"{self.billing_session_url}/billing/"
        else:
            base_return_url: str = f"{self.billing_session_url}/upgrade/"
            params: Dict[str, str] = {
                "manual_license_management": str(manual_license_management).lower(),
                "tier": str(tier),
                "setup_payment_by_invoice": str(setup_payment_by_invoice).lower(),
            }
            return_url = f"{base_return_url}?{urlencode(params)}"
        configuration: stripe.billing_portal.Configuration = stripe.billing_portal.Configuration.create(
            business_profile={"headline": "Invoice and receipt billing information"},
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
        billing_schedule: BillingSchedule,
        charge_automatically: bool,
        invoice_period: Dict[str, int],
        license_management: Optional[str] = None,
        days_until_due: Optional[int] = None,
        on_free_trial: bool = False,
        current_plan_id: Optional[int] = None,
    ) -> str:
        assert customer.stripe_customer_id is not None
        plan_name: str = CustomerPlan.name_from_tier(plan_tier)
        assert price_per_license is None or fixed_price is None
        price_args: Dict[str, Any] = {}
        if fixed_price is None:
            assert price_per_license is not None
            price_args = {
                "quantity": licenses,
                "unit_amount": price_per_license,
            }
        else:
            assert fixed_price is not None
            amount_due: int = get_amount_due_fixed_price_plan(fixed_price, billing_schedule)
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
            num_months: int = 12 if billing_schedule == CustomerPlan.BILLING_SCHEDULE_ANNUAL else 1
            flat_discounted_months: int = min(customer.flat_discounted_months, num_months)
            discount: int = customer.flat_discount * flat_discounted_months
            customer.flat_discounted_months -= flat_discounted_months
            customer.save(update_fields=["flat_discounted_months"])
            stripe.InvoiceItem.create(
                currency="usd",
                customer=customer.stripe_customer_id,
                description=f"${cents_to_dollar_string(customer.flat_discount)}/month new customer discount",
                amount=-1 * discount,
                period=invoice_period,
            )
        if charge_automatically:
            collection_method: str = "charge_automatically"
        else:
            collection_method = "send_invoice"
            if days_until_due is None:
                days_until_due = 1
        metadata: Dict[str, Any] = {
            "plan_tier": str(plan_tier),
            "billing_schedule": str(billing_schedule),
            "licenses": str(licenses),
            "license_management": str(license_management),
            "on_free_trial": str(on_free_trial),
            "current_plan_id": str(current_plan_id),
        }
        if hasattr(self, "user") and self.user is not None:
            metadata["user_id"] = self.user.id
        auto_advance: bool = not charge_automatically
        invoice_params: stripe.Invoice.CreateParams = stripe.Invoice.CreateParams(
            auto_advance=auto_advance,
            collection_method=collection_method,
            customer=customer.stripe_customer_id,
            statement_descriptor=plan_name,
            metadata=metadata,
        )
        if days_until_due is not None:
            invoice_params["days_until_due"] = days_until_due
        stripe_invoice: stripe.Invoice = stripe.Invoice.create(**invoice_params)
        stripe.Invoice.finalize_invoice(stripe_invoice)
        return stripe_invoice.id  # type: ignore


    @abstractmethod
    def update_or_create_customer(
        self,
        stripe_customer_id: Optional[str] = None,
        *,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Customer:
        pass

    @abstractmethod
    def do_change_plan_type(
        self,
        *,
        tier: int,
        is_sponsored: bool = False,
        background_update: bool = False,
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
    def get_sponsorship_request_session_specific_context(self) -> SponsorshipRequestSessionSpecificContext:
        pass

    @abstractmethod
    def save_org_type_from_request_sponsorship_session(self, org_type: Any) -> None:
        pass

    @abstractmethod
    def get_upgrade_page_session_type_specific_context(self) -> UpgradePageSessionTypeSpecificContext:
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
    def get_metadata_for_stripe_update_card(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def sync_license_ledger_if_needed(self) -> None:
        pass

    def is_sponsored_or_pending(self, customer: Optional[Customer]) -> bool:
        if customer is not None and customer.sponsorship_pending or self.is_sponsored():
            return True
        return False

    def get_complimentary_access_plan(
        self, customer: Optional[Customer], status: str = CustomerPlan.ACTIVE
    ) -> Optional[CustomerPlan]:
        if customer is None:
            return None
        plan_tier: int = CustomerPlan.TIER_SELF_HOSTED_LEGACY
        if isinstance(self, RealmBillingSession):
            return None
        return CustomerPlan.objects.filter(
            customer=customer, tier=plan_tier, status=status
        ).first()

    def get_formatted_complimentary_access_plan_end_date(
        self, customer: Optional[Customer], status: str = CustomerPlan.ACTIVE
    ) -> Optional[str]:
        complimentary_access_plan: Optional[CustomerPlan] = self.get_complimentary_access_plan(
            customer, status
        )
        if complimentary_access_plan is None:
            return None
        assert complimentary_access_plan.end_date is not None
        return complimentary_access_plan.end_date.strftime("%B %d, %Y")

    def get_complimentary_access_next_plan(
        self, customer: Optional[Customer]
    ) -> Optional[CustomerPlan]:
        complimentary_access_plan: Optional[CustomerPlan] = self.get_complimentary_access_plan(
            customer, CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END
        )
        if complimentary_access_plan is None:
            return None
        assert complimentary_access_plan.end_date is not None
        return CustomerPlan.objects.get(
            customer=customer,
            billing_cycle_anchor=complimentary_access_plan.end_date,
            status=CustomerPlan.NEVER_STARTED,
        )

    def get_complimentary_access_next_plan_name(
        self, customer: Optional[Customer]
    ) -> Optional[str]:
        next_plan: Optional[CustomerPlan] = self.get_complimentary_access_next_plan(
            customer
        )
        if next_plan is None:
            return None
        return next_plan.name

    @catch_stripe_errors
    def create_stripe_customer(self) -> Customer:
        stripe_customer_data: StripeCustomerData = self.get_data_for_stripe_customer()
        stripe_customer: stripe.Customer = stripe.Customer.create(
            description=stripe_customer_data.description,
            email=stripe_customer_data.email,
            metadata=stripe_customer_data.metadata,
        )
        event_time: datetime = timestamp_to_datetime(stripe_customer.created)
        with transaction.atomic(durable=True):
            self.write_to_audit_log(
                BillingSessionEventType.STRIPE_CUSTOMER_CREATED, event_time
            )
            customer: Customer = self.update_or_create_customer(stripe_customer.id)
        return customer

    @catch_stripe_errors
    def replace_payment_method(
        self, stripe_customer_id: str, payment_method: str, pay_invoices: bool = False
    ) -> None:
        stripe.Customer.modify(
            stripe_customer_id,
            invoice_settings={"default_payment_method": payment_method},
        )
        self.write_to_audit_log(
            BillingSessionEventType.STRIPE_CARD_CHANGED, timezone_now()
        )
        if pay_invoices:
            for stripe_invoice in stripe.Invoice.list(
                collection_method="charge_automatically",
                customer=stripe_customer_id,
                status="open",
            ).auto_paging_iter():
                stripe.Invoice.pay(stripe_invoice)

    @catch_stripe_errors
    def update_or_create_stripe_customer(
        self, payment_method: Optional[str] = None
    ) -> Customer:
        customer: Customer
        if self.get_customer() is None or self.get_customer().stripe_customer_id is None:
            assert payment_method is None
            return self.create_stripe_customer()
        if payment_method is not None:
            self.replace_payment_method(
                self.get_customer().stripe_customer_id, payment_method, True
            )
        return self.get_customer()

    def create_stripe_invoice_and_charge(self, metadata: Dict[str, Any]) -> str:
        """
        Charge customer based on `billing_modality`. If `billing_modality` is `charge_automatically`,
        charge customer immediately. If the charge fails, the invoice will be voided.
        If `billing_modality` is `send_invoice`, create an invoice and send it to the customer.
        """
        customer: Customer = self.get_customer()
        assert customer is not None and customer.stripe_customer_id is not None
        charge_automatically: bool = metadata["billing_modality"] == "charge_automatically"
        stripe_customer: stripe.Customer = stripe_get_customer(customer.stripe_customer_id)
        if charge_automatically and (not stripe_customer_has_credit_card_as_default_payment_method(stripe_customer)):
            raise BillingError("no payment method", _("Please add a credit card before upgrading."))
        if charge_automatically:
            assert stripe_customer.invoice_settings is not None
            assert stripe_customer.invoice_settings.default_payment_method is not None
        stripe_invoice: Optional[stripe.Invoice] = None
        invoice: Optional[Invoice] = None
        try:
            current_plan_id: Optional[int] = metadata.get("current_plan_id")
            on_free_trial: bool = bool(metadata.get("on_free_trial"))
            stripe_invoice_id: str = self.generate_invoice_for_upgrade(
                customer=customer,
                price_per_license=metadata["price_per_license"],
                fixed_price=metadata["fixed_price"],
                licenses=metadata["licenses"],
                plan_tier=metadata["plan_tier"],
                billing_schedule=metadata["billing_schedule"],
                charge_automatically=charge_automatically,
                license_management=metadata["license_management"],
                invoice_period=metadata["invoice_period"],
                days_until_due=metadata.get("days_until_due"),
                on_free_trial=on_free_trial,
                current_plan_id=current_plan_id,
            )
            stripe_invoice_id = stripe_invoice_id  # type: ignore
            invoice = Invoice.objects.create(
                stripe_invoice_id=stripe_invoice_id,
                customer=customer,
                status=Invoice.SENT,
                plan_id=current_plan_id,
                is_created_for_free_trial_upgrade=(current_plan_id is not None and on_free_trial),
            )
            if charge_automatically:
                stripe_invoice = stripe.Invoice.pay(stripe_invoice_id)
        except Exception as e:
            if stripe_invoice is not None:
                assert stripe_invoice.id is not None
                stripe.Invoice.void_invoice(stripe_invoice.id)
                if invoice is not None:
                    invoice.status = Invoice.VOID
                    invoice.save(update_fields=["status"])
            if isinstance(e, stripe.CardError):
                raise StripeCardError("card error", e.user_message)
            else:
                raise e
        assert stripe_invoice.id is not None
        return stripe_invoice.id

    @abstractmethod
    def do_upgrade(self, upgrade_request: UpgradeRequest) -> Dict[str, Any]:
        pass

    @abstractmethod
    def do_change_schedule_after_free_trial(self, plan: CustomerPlan, schedule: str) -> None:
        pass

    def generate_stripe_invoice(
        self,
        plan_tier: int,
        licenses: int,
        license_management: str,
        billing_schedule: BillingSchedule,
        billing_modality: str,
        on_free_trial: bool = False,
        days_until_due: Optional[int] = None,
        current_plan_id: Optional[int] = None,
    ) -> str:
        customer: Customer = self.update_or_create_stripe_customer()
        assert customer is not None
        fixed_price_plan_offer: Optional[CustomerPlanOffer] = get_configured_fixed_price_plan_offer(
            customer, plan_tier
        )
        general_metadata: Dict[str, Any] = {
            "billing_modality": billing_modality,
            "billing_schedule": billing_schedule,
            "licenses": licenses,
            "license_management": license_management,
            "price_per_license": None,
            "fixed_price": None,
            "type": "upgrade",
            "plan_tier": plan_tier,
            "on_free_trial": on_free_trial,
            "days_until_due": days_until_due,
            "current_plan_id": current_plan_id,
        }
        invoice_period_start: datetime
        invoice_period_end: datetime
        price_per_license: Optional[int]
        invoice_period_start, _, invoice_period_end, price_per_license = compute_plan_parameters(
            plan_tier=plan_tier,
            billing_schedule=billing_schedule,
            customer=customer,
            free_trial=on_free_trial,
            upgrade_when_complimentary_access_plan_ends=False,
        )
        if fixed_price_plan_offer is None:
            general_metadata["price_per_license"] = price_per_license
        else:
            general_metadata["fixed_price"] = fixed_price_plan_offer.fixed_price
            invoice_period_end = add_months(
                invoice_period_start, CustomerPlan.FIXED_PRICE_PLAN_DURATION_MONTHS
            )
        if on_free_trial and billing_modality == "send_invoice":
            invoice_period_start = invoice_period_end
            purchased_months: int = 1
            if billing_schedule == CustomerPlan.BILLING_SCHEDULE_ANNUAL:
                purchased_months = 12
            invoice_period_end = add_months(invoice_period_end, purchased_months)
        general_metadata["invoice_period"] = {
            "start": datetime_to_timestamp(invoice_period_start),
            "end": datetime_to_timestamp(invoice_period_end),
        }
        updated_metadata: Dict[str, Any] = self.update_data_for_checkout_session_and_invoice_payment(
            general_metadata
        )
        return self.create_stripe_invoice_and_charge(updated_metadata)

    @abstractmethod
    def do_change_plan_type(
        self,
        *,
        tier: int,
        is_sponsored: bool = False,
        background_update: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def process_downgrade(
        self, plan: CustomerPlan, background_update: bool = False
    ) -> None:
        pass

    @abstractmethod
    def approve_sponsorship(self) -> str:
        pass

    @abstractmethod
    def is_sponsored(self) -> bool:
        pass

    @abstractmethod
    def get_sponsorship_request_session_specific_context(
        self
    ) -> SponsorshipRequestSessionSpecificContext:
        pass

    @abstractmethod
    def save_org_type_from_request_sponsorship_session(self, org_type: Any) -> None:
        pass

    @abstractmethod
    def get_upgrade_page_session_type_specific_context(
        self
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
    def get_metadata_for_stripe_update_card(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def sync_license_ledger_if_needed(self) -> None:
        pass

    def is_sponsored_or_pending(self, customer: Optional[Customer]) -> bool:
        if customer is not None and customer.sponsorship_pending or self.is_sponsored():
            return True
        return False

    def get_complimentary_access_plan(
        self, customer: Optional[Customer], status: str = CustomerPlan.ACTIVE
    ) -> Optional[CustomerPlan]:
        if customer is None:
            return None
        plan_tier: int = CustomerPlan.TIER_SELF_HOSTED_LEGACY
        if isinstance(self, RealmBillingSession):
            return None
        return CustomerPlan.objects.filter(
            customer=customer, tier=plan_tier, status=status
        ).first()

    def get_formatted_complimentary_access_plan_end_date(
        self, customer: Optional[Customer], status: str = CustomerPlan.ACTIVE
    ) -> Optional[str]:
        complimentary_access_plan: Optional[CustomerPlan] = self.get_complimentary_access_plan(
            customer, status
        )
        if complimentary_access_plan is None:
            return None
        assert complimentary_access_plan.end_date is not None
        return complimentary_access_plan.end_date.strftime("%B %d, %Y")

    def get_complimentary_access_next_plan(
        self, customer: Optional[Customer]
    ) -> Optional[CustomerPlan]:
        complimentary_access_plan: Optional[CustomerPlan] = self.get_complimentary_access_plan(
            customer, CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END
        )
        if complimentary_access_plan is None:
            return None
        assert complimentary_access_plan.end_date is not None
        return CustomerPlan.objects.get(
            customer=customer,
            billing_cycle_anchor=complimentary_access_plan.end_date,
            status=CustomerPlan.NEVER_STARTED,
        )

    def get_complimentary_access_next_plan_name(
        self, customer: Optional[Customer]
    ) -> Optional[str]:
        next_plan: Optional[CustomerPlan] = self.get_complimentary_access_next_plan(
            customer
        )
        if next_plan is None:
            return None
        return next_plan.name

    @catch_stripe_errors
    def create_stripe_customer(self) -> Customer:
        stripe_customer_data: StripeCustomerData = self.get_data_for_stripe_customer()
        stripe_customer: stripe.Customer = stripe.Customer.create(
            description=stripe_customer_data.description,
            email=stripe_customer_data.email,
            metadata=stripe_customer_data.metadata,
        )
        event_time: datetime = timestamp_to_datetime(stripe_customer.created)
        with transaction.atomic(durable=True):
            self.write_to_audit_log(
                BillingSessionEventType.STRIPE_CUSTOMER_CREATED, event_time
            )
            customer: Customer = self.update_or_create_customer(stripe_customer.id)
        return customer

    @catch_stripe_errors
    def replace_payment_method(
        self, stripe_customer_id: str, payment_method: str, pay_invoices: bool = False
    ) -> None:
        stripe.Customer.modify(
            stripe_customer_id,
            invoice_settings={"default_payment_method": payment_method},
        )
        self.write_to_audit_log(
            BillingSessionEventType.STRIPE_CARD_CHANGED, timezone_now()
        )
        if pay_invoices:
            for stripe_invoice in stripe.Invoice.list(
                collection_method="charge_automatically",
                customer=stripe_customer_id,
                status="open",
            ).auto_paging_iter():
                stripe.Invoice.pay(stripe_invoice)

    @catch_stripe_errors
    def update_or_create_stripe_customer(
        self, payment_method: Optional[str] = None
    ) -> Customer:
        customer: Customer = self.get_customer()
        if customer is None or customer.stripe_customer_id is None:
            assert payment_method is None
            return self.create_stripe_customer()
        if payment_method is not None:
            self.replace_payment_method(
                customer.stripe_customer_id, payment_method, True
            )
        return customer

    def create_stripe_invoice_and_charge(self, metadata: Dict[str, Any]) -> str:
        """
        Charge customer based on `billing_modality`. If `billing_modality` is `charge_automatically`,
        charge customer immediately. If the charge fails, the invoice will be voided.
        If `billing_modality` is `send_invoice`, create an invoice and send it to the customer.
        """
        customer: Customer = self.get_customer()
        assert customer is not None and customer.stripe_customer_id is not None
        charge_automatically: bool = metadata["billing_modality"] == "charge_automatically"
        stripe_customer: stripe.Customer = stripe_get_customer(customer.stripe_customer_id)
        if charge_automatically and (
            not stripe_customer_has_credit_card_as_default_payment_method(stripe_customer)
        ):
            raise BillingError("no payment method", _("Please add a credit card before upgrading."))
        if charge_automatically:
            assert stripe_customer.invoice_settings is not None
            assert stripe_customer.invoice_settings.default_payment_method is not None
        stripe_invoice: Optional[stripe.Invoice] = None
        invoice: Optional[Invoice] = None
        try:
            current_plan_id: Optional[int] = metadata.get("current_plan_id")
            on_free_trial: bool = bool(metadata.get("on_free_trial"))
            stripe_invoice_id: str = self.generate_invoice_for_upgrade(
                customer=customer,
                price_per_license=metadata["price_per_license"],
                fixed_price=metadata["fixed_price"],
                licenses=metadata["licenses"],
                plan_tier=metadata["plan_tier"],
                billing_schedule=metadata["billing_schedule"],
                charge_automatically=charge_automatically,
                license_management=metadata["license_management"],
                invoice_period=metadata["invoice_period"],
                days_until_due=metadata.get("days_until_due"),
                on_free_trial=on_free_trial,
                current_plan_id=current_plan_id,
            )
            invoice = Invoice.objects.create(
                stripe_invoice_id=stripe_invoice_id,
                customer=customer,
                status=Invoice.SENT,
                plan_id=current_plan_id,
                is_created_for_free_trial_upgrade=(current_plan_id is not None and on_free_trial),
            )
            if charge_automatically:
                stripe_invoice = stripe.Invoice.pay(stripe_invoice_id)
        except Exception as e:
            if stripe_invoice is not None:
                assert stripe_invoice.id is not None
                stripe.Invoice.void_invoice(stripe_invoice.id)
                if invoice is not None:
                    invoice.status = Invoice.VOID
                    invoice.save(update_fields=["status"])
            if isinstance(e, stripe.CardError):
                raise StripeCardError("card error", e.user_message)
            else:
                raise e
        assert stripe_invoice.id is not None
        return stripe_invoice.id

    @abstractmethod
    def do_upgrade(self, upgrade_request: UpgradeRequest) -> Dict[str, Any]:
        pass

    @abstractmethod
    def do_change_schedule_after_free_trial(
        self, plan: CustomerPlan, schedule: str
    ) -> None:
        pass

    def generate_stripe_invoice(
        self,
        plan_tier: int,
        licenses: int,
        license_management: str,
        billing_schedule: BillingSchedule,
        billing_modality: str,
        on_free_trial: bool = False,
        days_until_due: Optional[int] = None,
        current_plan_id: Optional[int] = None,
    ) -> None:
        customer: Customer = self.update_or_create_stripe_customer()
        assert customer is not None
        fixed_price_plan_offer: Optional[CustomerPlanOffer] = get_configured_fixed_price_plan_offer(
            customer, plan_tier
        )
        general_metadata: Dict[str, Any] = {
            "billing_modality": billing_modality,
            "billing_schedule": billing_schedule,
            "licenses": licenses,
            "license_management": license_management,
            "price_per_license": None,
            "fixed_price": None,
            "type": "upgrade",
            "plan_tier": plan_tier,
            "on_free_trial": on_free_trial,
            "days_until_due": days_until_due,
            "current_plan_id": current_plan_id,
        }
        invoice_period_start: datetime
        invoice_period_end: datetime
        price_per_license: Optional[int]
        invoice_period_start, _, invoice_period_end, price_per_license = compute_plan_parameters(
            plan_tier=plan_tier,
            billing_schedule=billing_schedule,
            customer=customer,
            free_trial=on_free_trial,
            upgrade_when_complimentary_access_plan_ends=False,
        )
        if fixed_price_plan_offer is None:
            general_metadata["price_per_license"] = price_per_license
        else:
            general_metadata["fixed_price"] = fixed_price_plan_offer.fixed_price
            invoice_period_end = add_months(
                invoice_period_start, CustomerPlan.FIXED_PRICE_PLAN_DURATION_MONTHS
            )
        if on_free_trial and billing_modality == "send_invoice":
            invoice_period_start = invoice_period_end
            purchased_months: int = 1
            if billing_schedule == CustomerPlan.BILLING_SCHEDULE_ANNUAL:
                purchased_months = 12
            invoice_period_end = add_months(invoice_period_end, purchased_months)
        general_metadata["invoice_period"] = {
            "start": datetime_to_timestamp(invoice_period_start),
            "end": datetime_to_timestamp(invoice_period_end),
        }
        updated_metadata: Dict[str, Any] = self.update_data_for_checkout_session_and_invoice_payment(
            general_metadata
        )
        stripe_invoice_id: str = self.create_stripe_invoice_and_charge(updated_metadata)
        # Further processing if needed


class RealmBillingSession(BillingSession):
    def __init__(
        self,
        user: Optional[UserProfile] = None,
        realm: Optional[Realm] = None,
        *,
        support_session: bool = False,
    ) -> None:
        self.user: Optional[UserProfile] = user
        assert user is not None or realm is not None
        if support_session:
            assert user is not None and user.is_staff
        self.support_session: bool = support_session
        if user is not None and realm is not None:
            assert user.is_staff or user.realm == realm
            self.realm: Realm = realm
        elif user is not None:
            self.realm: Realm = user.realm
        else:
            assert realm is not None
            self.realm = realm

    PAID_PLANS: List[str] = [
        Realm.PLAN_TYPE_STANDARD,
        Realm.PLAN_TYPE_PLUS,
    ]

    @override
    @property
    def billing_entity_display_name(self) -> str:
        return self.realm.string_id

    @override
    @property
    def billing_session_url(self) -> str:
        return self.realm.url

    @override
    @property
    def billing_base_url(self) -> str:
        return ""

    @override
    def support_url(self) -> str:
        return build_support_url("support", self.realm.string_id)

    @override
    def get_customer(self) -> Optional[Customer]:
        return get_customer_by_realm(self.realm)

    @override
    def get_email(self) -> str:
        assert self.user is not None
        return self.user.delivery_email

    @override
    def current_count_for_billed_licenses(self, event_time: Optional[datetime] = None) -> int:
        return get_latest_seat_count(self.realm)

    @override
    def get_audit_log_event(
        self, event_type: BillingSessionEventType
    ) -> AuditLogEventType:
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
        elif event_type is BillingSessionEventType.CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_plan:
            return AuditLogEventType.CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_PLAN
        else:
            raise BillingSessionAuditLogEventError(event_type)

    @override
    def write_to_audit_log(
        self,
        event_type: BillingSessionEventType,
        event_time: datetime,
        *,
        background_update: bool = False,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        audit_log_event: AuditLogEventType = self.get_audit_log_event(event_type)
        audit_log_data: Dict[str, Any] = {
            "realm": self.realm,
            "event_type": audit_log_event,
            "event_time": event_time,
        }
        if extra_data:
            audit_log_data["extra_data"] = extra_data
        if self.user is not None and (not background_update):
            audit_log_data["acting_user"] = self.user
        RealmAuditLog.objects.create(**audit_log_data)

    @override
    def get_data_for_stripe_customer(self) -> StripeCustomerData:
        assert self.support_session is False
        assert self.user is not None
        metadata: Dict[str, Any] = {}
        metadata["realm_id"] = self.realm.id
        metadata["realm_str"] = self.realm.string_id
        realm_stripe_customer_data: StripeCustomerData = StripeCustomerData(
            description=f"{self.realm.string_id} ({self.realm.name})",
            email=self.get_email(),
            metadata=metadata,
        )
        return realm_stripe_customer_data

    @override
    def update_data_for_checkout_session_and_invoice_payment(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        assert self.user is not None
        updated_metadata: Dict[str, Any] = {
            "user_email": self.get_email(),
            "realm_id": self.realm.id,
            "realm_str": self.realm.string_id,
            "user_id": self.user.id,
            **metadata,
        }
        return updated_metadata

    @override
    def update_or_create_customer(
        self,
        stripe_customer_id: Optional[str] = None,
        *,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Customer:
        if stripe_customer_id is not None:
            assert self.support_session is False
            customer, created = Customer.objects.update_or_create(
                realm=self.realm,
                defaults={"stripe_customer_id": stripe_customer_id},
            )
            from zerver.actions.users import do_change_is_billing_admin

            assert self.user is not None
            do_change_is_billing_admin(self.user, True)
            return customer
        else:
            customer, created = Customer.objects.update_or_create(
                realm=self.realm, defaults=defaults
            )
            return customer

    @override
    def do_change_plan_type(
        self, *, tier: int, is_sponsored: bool = False, background_update: bool = False
    ) -> None:
        from zerver.actions.realm_settings import do_change_realm_plan_type

        if is_sponsored:
            plan_type: str = Realm.PLAN_TYPE_STANDARD_FREE
        elif tier == CustomerPlan.TIER_CLOUD_STANDARD:
            plan_type = Realm.PLAN_TYPE_STANDARD
        elif tier == CustomerPlan.TIER_CLOUD_PLUS:
            plan_type = Realm.PLAN_TYPE_PLUS
        else:
            raise AssertionError("Unexpected tier")
        acting_user: Optional[UserProfile] = None
        if not background_update:
            acting_user = self.user
        do_change_realm_plan_type(
            self.realm, plan_type, acting_user=acting_user
        )

    @override
    def process_downgrade(
        self, plan: CustomerPlan, background_update: bool = False
    ) -> None:
        from zerver.actions.realm_settings import do_change_realm_plan_type

        acting_user: Optional[UserProfile] = None
        if not background_update:
            acting_user = self.user
        assert plan.customer.realm is not None
        do_change_realm_plan_type(
            plan.customer.realm, Realm.PLAN_TYPE_LIMITED, acting_user=acting_user
        )
        plan.status = CustomerPlan.ENDED
        plan.save(update_fields=["status"])

    @override
    def approve_sponsorship(self) -> str:
        assert self.support_session
        customer: Optional[Customer] = self.get_customer()
        if customer is not None:
            error_message: str = self.check_customer_not_on_paid_plan(customer)
            if error_message != "":
                raise SupportRequestError(error_message)
        from zerver.actions.message_send import internal_send_private_message

        if self.realm.deactivated:
            raise SupportRequestError("Realm has been deactivated")
        self.do_change_plan_type(tier=None, is_sponsored=True)
        if customer is not None and customer.sponsorship_pending:
            customer.sponsorship_pending = False
            customer.save(update_fields=["sponsorship_pending"])
            self.write_to_audit_log(
                BillingSessionEventType.SPONSORSHIP_APPROVED, timezone_now()
            )
        notification_bot: UserProfile = get_system_bot(
            settings.NOTIFICATION_BOT, self.realm.id
        )
        for user in self.realm.get_human_admin_users():
            with override_language(user.default_language):
                message: str = _(
                    "Your organization's request for sponsored hosting has been approved! You have been upgraded to {plan_name}, free of charge. {emoji}\n\nIf you could {begin_link}list Zulip as a sponsor on your website{end_link}, we would really appreciate it!"
                ).format(
                    plan_name=CustomerPlan.name_from_tier(CustomerPlan.TIER_CLOUD_STANDARD),
                    emoji=":tada:",
                    begin_link="[",
                    end_link="](/help/linking-to-zulip-website)",
                )
                internal_send_private_message(notification_bot, user, message)
        return f"Sponsorship approved for {self.billing_entity_display_name}; Emailed organization owners and billing admins."

    @override
    def is_sponsored(self) -> bool:
        return self.realm.plan_type == self.realm.PLAN_TYPE_STANDARD_FREE

    @override
    def get_metadata_for_stripe_update_card(self) -> Dict[str, Any]:
        assert self.user is not None
        return {"type": "card_update", "user_id": str(self.user.id)}

    @override
    def get_upgrade_page_session_type_specific_context(
        self
    ) -> UpgradePageSessionTypeSpecificContext:
        assert self.user is not None
        return UpgradePageSessionTypeSpecificContext(
            customer_name=self.realm.name,
            email=self.get_email(),
            is_demo_organization=self.realm.demo_organization_scheduled_deletion_date is not None,
            demo_organization_scheduled_deletion_date=self.realm.demo_organization_scheduled_deletion_date,
            is_self_hosting=False,
        )

    @override
    def check_plan_tier_is_billable(self, plan_tier: int) -> bool:
        implemented_plan_tiers: List[int] = [
            CustomerPlan.TIER_CLOUD_STANDARD,
            CustomerPlan.TIER_CLOUD_PLUS,
        ]
        if plan_tier in implemented_plan_tiers:
            return True
        return False

    @override
    def get_type_of_plan_tier_change(
        self, current_plan_tier: int, new_plan_tier: int
    ) -> PlanTierChangeType:
        valid_plan_tiers: List[int] = [
            CustomerPlan.TIER_CLOUD_STANDARD,
            CustomerPlan.TIER_CLOUD_PLUS,
        ]
        if (
            current_plan_tier not in valid_plan_tiers
            or new_plan_tier not in valid_plan_tiers
            or current_plan_tier == new_plan_tier
        ):
            return PlanTierChangeType.INVALID
        if (
            current_plan_tier == CustomerPlan.TIER_CLOUD_STANDARD
            and new_plan_tier == CustomerPlan.TIER_CLOUD_PLUS
        ):
            return PlanTierChangeType.UPGRADE
        else:
            assert current_plan_tier == CustomerPlan.TIER_CLOUD_PLUS
            assert new_plan_tier == CustomerPlan.TIER_CLOUD_STANDARD
            return PlanTierChangeType.DOWNGRADE

    @override
    def has_billing_access(self) -> bool:
        assert self.user is not None
        return self.user.has_billing_access

    @override
    def on_paid_plan(self) -> bool:
        return self.realm.plan_type in self.PAID_PLANS

    @override
    def org_name(self) -> str:
        return self.realm.name

    @override
    def add_org_type_data_to_sponsorship_context(
        self, context: Dict[str, Any]
    ) -> None:
        context.update(
            realm_org_type=self.realm.org_type,
            sorted_org_types=sorted(
                (
                    [org_type_name, org_type]
                    for org_type_name, org_type in Realm.ORG_TYPES.items()
                    if not org_type.get("hidden")
                ),
                key=sponsorship_org_type_key_helper,
            ),
        )

    @override
    def get_sponsorship_request_session_specific_context(
        self
    ) -> SponsorshipRequestSessionSpecificContext:
        assert self.user is not None
        return SponsorshipRequestSessionSpecificContext(
            realm_user=self.user,
            user_info=SponsorshipApplicantInfo(
                name=self.user.full_name, email=self.get_email(), role=self.user.get_role_name()
            ),
            realm_string_id=self.realm.string_id,
        )

    @override
    def save_org_type_from_request_sponsorship_session(self, org_type: Any) -> None:
        if self.realm.org_type != org_type:
            self.realm.org_type = org_type
            self.realm.save(update_fields=["org_type"])

    def update_license_ledger_if_needed(self, event_time: datetime) -> Optional[CustomerPlan]:
        customer: Optional[Customer] = self.get_customer()
        if customer is None:
            return None
        plan: Optional[CustomerPlan] = get_current_plan_by_customer(customer)
        if plan is None:
            return None
        if not plan.automanage_licenses:
            return None
        return self.update_license_ledger_for_automanaged_plan(plan, event_time)

    @override
    def sync_license_ledger_if_needed(self) -> None:
        pass

    def is_sponsored_or_pending(self, customer: Optional[Customer]) -> bool:
        if (customer is not None and customer.sponsorship_pending) or self.is_sponsored():
            return True
        return False

    def get_complimentary_access_plan(
        self, customer: Optional[Customer], status: str = CustomerPlan.ACTIVE
    ) -> Optional[CustomerPlan]:
        if customer is None:
            return None
        plan_tier: int = CustomerPlan.TIER_SELF_HOSTED_LEGACY
        if isinstance(self, RealmBillingSession):
            return None
        return CustomerPlan.objects.filter(
            customer=customer, tier=plan_tier, status=status
        ).first()

    def get_formatted_complimentary_access_plan_end_date(
        self, customer: Optional[Customer], status: str = CustomerPlan.ACTIVE
    ) -> Optional[str]:
        complimentary_access_plan: Optional[CustomerPlan] = self.get_complimentary_access_plan(
            customer, status
        )
        if complimentary_access_plan is None:
            return None
        assert complimentary_access_plan.end_date is not None
        return complimentary_access_plan.end_date.strftime("%B %d, %Y")

    def get_complimentary_access_next_plan(
        self, customer: Optional[Customer]
    ) -> Optional[CustomerPlan]:
        complimentary_access_plan: Optional[CustomerPlan] = self.get_complimentary_access_plan(
            customer, CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END
        )
        if complimentary_access_plan is None:
            return None
        assert complimentary_access_plan.end_date is not None
        return CustomerPlan.objects.get(
            customer=customer,
            billing_cycle_anchor=complimentary_access_plan.end_date,
            status=CustomerPlan.NEVER_STARTED,
        )

    def get_complimentary_access_next_plan_name(
        self, customer: Optional[Customer]
    ) -> Optional[str]:
        next_plan: Optional[CustomerPlan] = self.get_complimentary_access_next_plan(
            customer
        )
        if next_plan is None:
            return None
        return next_plan.name

    @catch_stripe_errors
    def create_stripe_customer(self) -> Customer:
        stripe_customer_data: StripeCustomerData = self.get_data_for_stripe_customer()
        stripe_customer: stripe.Customer = stripe.Customer.create(
            description=stripe_customer_data.description,
            email=stripe_customer_data.email,
            metadata=stripe_customer_data.metadata,
        )
        event_time: datetime = timestamp_to_datetime(stripe_customer.created)
        with transaction.atomic(durable=True):
            self.write_to_audit_log(
                BillingSessionEventType.STRIPE_CUSTOMER_CREATED, event_time
            )
            customer: Customer = self.update_or_create_customer(stripe_customer.id)
        return customer

    @catch_stripe_errors
    def replace_payment_method(
        self, stripe_customer_id: str, payment_method: str, pay_invoices: bool = False
    ) -> None:
        stripe.Customer.modify(
            stripe_customer_id,
            invoice_settings={"default_payment_method": payment_method},
        )
        self.write_to_audit_log(
            BillingSessionEventType.STRIPE_CARD_CHANGED, timezone_now()
        )
        if pay_invoices:
            for stripe_invoice in stripe.Invoice.list(
                collection_method="charge_automatically",
                customer=stripe_customer_id,
                status="open",
            ).auto_paging_iter():
                stripe.Invoice.pay(stripe_invoice)

    @catch_stripe_errors
    def update_or_create_stripe_customer(
        self, payment_method: Optional[str] = None
    ) -> Customer:
        customer: Customer = self.get_customer()
        if customer is None or customer.stripe_customer_id is None:
            assert payment_method is None
            return self.create_stripe_customer()
        if payment_method is not None:
            self.replace_payment_method(
                customer.stripe_customer_id, payment_method, True
            )
        return customer

    def create_stripe_invoice_and_charge(self, metadata: Dict[str, Any]) -> str:
        """
        Charge customer based on `billing_modality`. If `billing_modality` is `charge_automatically`,
        charge customer immediately. If the charge fails, the invoice will be voided.
        If `billing_modality` is `send_invoice`, create an invoice and send it to the customer.
        """
        customer: Customer = self.get_customer()
        assert customer is not None and customer.stripe_customer_id is not None
        charge_automatically: bool = metadata["billing_modality"] == "charge_automatically"
        stripe_customer: stripe.Customer = stripe_get_customer(customer.stripe_customer_id)
        if charge_automatically and (
            not stripe_customer_has_credit_card_as_default_payment_method(stripe_customer)
        ):
            raise BillingError("no payment method", _("Please add a credit card before upgrading."))
        if charge_automatically:
            assert stripe_customer.invoice_settings is not None
            assert stripe_customer.invoice_settings.default_payment_method is not None
        stripe_invoice: Optional[stripe.Invoice] = None
        invoice: Optional[Invoice] = None
        try:
            current_plan_id: Optional[int] = metadata.get("current_plan_id")
            on_free_trial: bool = bool(metadata.get("on_free_trial"))
            stripe_invoice_id: str = self.generate_invoice_for_upgrade(
                customer=customer,
                price_per_license=metadata["price_per_license"],
                fixed_price=metadata["fixed_price"],
                licenses=metadata["licenses"],
                plan_tier=metadata["plan_tier"],
                billing_schedule=metadata["billing_schedule"],
                charge_automatically=charge_automatically,
                license_management=metadata["license_management"],
                invoice_period=metadata["invoice_period"],
                days_until_due=metadata.get("days_until_due"),
                on_free_trial=on_free_trial,
                current_plan_id=current_plan_id,
            )
            invoice = Invoice.objects.create(
                stripe_invoice_id=stripe_invoice_id,
                customer=customer,
                status=Invoice.SENT,
                plan_id=current_plan_id,
                is_created_for_free_trial_upgrade=(current_plan_id is not None and on_free_trial),
            )
            if charge_automatically:
                stripe_invoice = stripe.Invoice.pay(stripe_invoice_id)
        except Exception as e:
            if stripe_invoice is not None:
                assert stripe_invoice.id is not None
                stripe.Invoice.void_invoice(stripe_invoice.id)
                if invoice is not None:
                    invoice.status = Invoice.VOID
                    invoice.save(update_fields=["status"])
            if isinstance(e, stripe.CardError):
                raise StripeCardError("card error", e.user_message)
            else:
                raise e
        assert stripe_invoice.id is not None
        return stripe_invoice.id

    def add_months(self, dt: datetime, months: int) -> datetime:
        return add_months(dt, months)

    def start_of_next_billing_cycle(self, plan: CustomerPlan, event_time: datetime) -> datetime:
        return start_of_next_billing_cycle(plan, event_time)


@dataclass
class PushNotificationsEnabledStatus:
    can_push: bool
    expected_end_timestamp: Optional[int]
    message: str


def sponsorship_org_type_key_helper(d: Tuple[str, Dict[str, Any]]) -> Any:
    return d[1]["display_order"]


class RealmBillingSession(BillingSession):
    def __init__(
        self,
        user: Optional[UserProfile] = None,
        realm: Optional[Realm] = None,
        *,
        support_session: bool = False,
    ) -> None:
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

    PAID_PLANS: List[str] = [
        Realm.PLAN_TYPE_STANDARD,
        Realm.PLAN_TYPE_PLUS,
    ]

    @override
    @property
    def billing_entity_display_name(self) -> str:
        return self.realm.string_id

    @override
    @property
    def billing_session_url(self) -> str:
        return self.realm.url

    @override
    @property
    def billing_base_url(self) -> str:
        return ""

    @override
    def support_url(self) -> str:
        return build_support_url("support", self.realm.string_id)

    @override
    def get_customer(self) -> Optional[Customer]:
        return get_customer_by_realm(self.realm)

    @override
    def get_email(self) -> str:
        assert self.user is not None
        return self.user.delivery_email

    @override
    def current_count_for_billed_licenses(
        self, event_time: Optional[datetime] = None
    ) -> int:
        return get_latest_seat_count(self.realm)

    @override
    def get_audit_log_event(
        self, event_type: BillingSessionEventType
    ) -> AuditLogEventType:
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
        elif event_type is BillingSessionEventType.CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_plan:
            return AuditLogEventType.CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_PLAN
        else:
            raise BillingSessionAuditLogEventError(event_type)

    @override
    def write_to_audit_log(
        self,
        event_type: BillingSessionEventType,
        event_time: datetime,
        *,
        background_update: bool = False,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        audit_log_event: AuditLogEventType = self.get_audit_log_event(event_type)
        audit_log_data: Dict[str, Any] = {
            "realm": self.realm,
            "event_type": audit_log_event,
            "event_time": event_time,
        }
        if extra_data:
            audit_log_data["extra_data"] = extra_data
        if self.user is not None and (not background_update):
            audit_log_data["acting_user"] = self.user
        RealmAuditLog.objects.create(**audit_log_data)

    @override
    def get_data_for_stripe_customer(self) -> StripeCustomerData:
        assert self.support_session is False
        assert self.user is not None
        metadata: Dict[str, Any] = {}
        metadata["realm_id"] = self.realm.id
        metadata["realm_str"] = self.realm.string_id
        realm_stripe_customer_data: StripeCustomerData = StripeCustomerData(
            description=f"{self.realm.string_id} ({self.realm.name})",
            email=self.get_email(),
            metadata=metadata,
        )
        return realm_stripe_customer_data

    @override
    def update_data_for_checkout_session_and_invoice_payment(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        assert self.user is not None
        updated_metadata: Dict[str, Any] = {
            "user_email": self.get_email(),
            "realm_id": self.realm.id,
            "realm_str": self.realm.string_id,
            "user_id": self.user.id,
            **metadata,
        }
        return updated_metadata

    @override
    def update_or_create_customer(
        self,
        stripe_customer_id: Optional[str] = None,
        *,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Customer:
        if stripe_customer_id is not None:
            assert self.support_session is False
            customer, created = Customer.objects.update_or_create(
                realm=self.realm,
                defaults={"stripe_customer_id": stripe_customer_id},
            )
            from zerver.actions.users import do_change_is_billing_admin

            assert self.user is not None
            do_change_is_billing_admin(self.user, True)
            return customer
        else:
            customer, created = Customer.objects.update_or_create(
                realm=self.realm, defaults=defaults
            )
            return customer

    @override
    def do_change_plan_type(
        self, *, tier: int, is_sponsored: bool = False, background_update: bool = False
    ) -> None:
        from zerver.actions.realm_settings import do_change_realm_plan_type

        if is_sponsored:
            plan_type: str = Realm.PLAN_TYPE_STANDARD_FREE
            self.add_customer_to_community_plan()
        elif tier == CustomerPlan.TIER_CLOUD_STANDARD:
            plan_type = Realm.PLAN_TYPE_STANDARD
        elif tier == CustomerPlan.TIER_CLOUD_PLUS:
            plan_type = Realm.PLAN_TYPE_PLUS
        else:
            raise AssertionError("Unexpected tier")
        acting_user: Optional[UserProfile] = None
        if not background_update:
            acting_user = self.user
        do_change_realm_plan_type(
            self.realm, plan_type, acting_user=acting_user
        )

    @override
    def process_downgrade(
        self, plan: CustomerPlan, background_update: bool = False
    ) -> None:
        from zerver.actions.realm_settings import do_change_realm_plan_type

        acting_user: Optional[UserProfile] = None
        if not background_update:
            acting_user = self.user
        do_change_realm_plan_type(
            plan.customer.realm, Realm.PLAN_TYPE_LIMITED, acting_user=acting_user
        )
        plan.status = CustomerPlan.ENDED
        plan.save(update_fields=["status"])

    @override
    def approve_sponsorship(self) -> str:
        assert self.support_session
        customer: Optional[Customer] = self.get_customer()
        if customer is not None:
            error_message: str = self.check_customer_not_on_paid_plan(customer)
            if error_message != "":
                raise SupportRequestError(error_message)
        from zerver.actions.message_send import internal_send_private_message

        if self.realm.deactivated:
            raise SupportRequestError("Realm has been deactivated")
        self.do_change_plan_type(tier=None, is_sponsored=True)
        if customer is not None and customer.sponsorship_pending:
            customer.sponsorship_pending = False
            customer.save(update_fields=["sponsorship_pending"])
            self.write_to_audit_log(
                BillingSessionEventType.SPONSORSHIP_APPROVED, timezone_now()
            )
        notification_bot: UserProfile = get_system_bot(
            settings.NOTIFICATION_BOT, self.realm.id
        )
        for user in self.realm.get_human_admin_users():
            with override_language(user.default_language):
                message: str = _(
                    "Your organization's request for sponsored hosting has been approved! You have been upgraded to {plan_name}, free of charge. {emoji}\n\nIf you could {begin_link}list Zulip as a sponsor on your website{end_link}, we would really appreciate it!"
                ).format(
                    plan_name=CustomerPlan.name_from_tier(CustomerPlan.TIER_CLOUD_STANDARD),
                    emoji=":tada:",
                    begin_link="[",
                    end_link="](/help/linking-to-zulip-website)",
                )
                internal_send_private_message(notification_bot, user, message)
        return f"Sponsorship approved for {self.billing_entity_display_name}; Emailed organization owners and billing admins."

    @override
    def is_sponsored(self) -> bool:
        return self.realm.plan_type == self.realm.PLAN_TYPE_STANDARD_FREE

    @override
    def get_metadata_for_stripe_update_card(self) -> Dict[str, Any]:
        assert self.user is not None
        return {"type": "card_update", "user_id": str(self.user.id)}

    @override
    def get_upgrade_page_session_type_specific_context(
        self
    ) -> UpgradePageSessionTypeSpecificContext:
        assert self.user is not None
        return UpgradePageSessionTypeSpecificContext(
            customer_name=self.realm.name,
            email=self.get_email(),
            is_demo_organization=self.realm.demo_organization_scheduled_deletion_date is not None,
            demo_organization_scheduled_deletion_date=self.realm.demo_organization_scheduled_deletion_date,
            is_self_hosting=False,
        )

    @override
    def check_plan_tier_is_billable(self, plan_tier: int) -> bool:
        implemented_plan_tiers: List[int] = [
            CustomerPlan.TIER_CLOUD_STANDARD,
            CustomerPlan.TIER_CLOUD_PLUS,
        ]
        if plan_tier in implemented_plan_tiers:
            return True
        return False

    @override
    def get_type_of_plan_tier_change(
        self, current_plan_tier: int, new_plan_tier: int
    ) -> PlanTierChangeType:
        valid_plan_tiers: List[int] = [
            CustomerPlan.TIER_CLOUD_STANDARD,
            CustomerPlan.TIER_CLOUD_PLUS,
        ]
        if (
            current_plan_tier not in valid_plan_tiers
            or new_plan_tier not in valid_plan_tiers
            or current_plan_tier == new_plan_tier
        ):
            return PlanTierChangeType.INVALID
        if (
            current_plan_tier == CustomerPlan.TIER_CLOUD_STANDARD
            and new_plan_tier == CustomerPlan.TIER_CLOUD_PLUS
        ):
            return PlanTierChangeType.UPGRADE
        else:
            assert current_plan_tier == CustomerPlan.TIER_CLOUD_PLUS
            assert new_plan_tier == CustomerPlan.TIER_CLOUD_STANDARD
            return PlanTierChangeType.DOWNGRADE

    @override
    def has_billing_access(self) -> bool:
        assert self.user is not None
        return self.user.has_billing_access

    @override
    def on_paid_plan(self) -> bool:
        return self.realm.plan_type in self.PAID_PLANS

    @override
    def org_name(self) -> str:
        return self.realm.name

    @override
    def add_org_type_data_to_sponsorship_context(
        self, context: Dict[str, Any]
    ) -> None:
        context.update(
            realm_org_type=self.realm.org_type,
            sorted_org_types=sorted(
                (
                    [org_type_name, org_type]
                    for org_type_name, org_type in Realm.ORG_TYPES.items()
                    if not org_type.get("hidden")
                ),
                key=sponsorship_org_type_key_helper,
            ),
        )

    @override
    def get_sponsorship_request_session_specific_context(
        self
    ) -> SponsorshipRequestSessionSpecificContext:
        assert self.user is not None
        return SponsorshipRequestSessionSpecificContext(
            realm_user=self.user,
            user_info=SponsorshipApplicantInfo(
                name=self.user.full_name, email=self.get_email(), role=self.user.get_role_name()
            ),
            realm_string_id=self.realm.string_id,
        )

    @override
    def save_org_type_from_request_sponsorship_session(self, org_type: Any) -> None:
        if self.realm.org_type != org_type:
            self.realm.org_type = org_type
            self.realm.save(update_fields=["org_type"])

    def add_months(self, dt: datetime, months: int) -> datetime:
        return add_months(dt, months)

    def process_initial_upgrade(
        self,
        plan_tier: int,
        licenses: int,
        automanage_licenses: bool,
        billing_schedule: BillingSchedule,
        charge_automatically: bool,
        free_trial: bool,
        complimentary_access_plan: Optional[CustomerPlan] = None,
        upgrade_when_complimentary_access_plan_ends: bool = False,
        stripe_invoice_paid: bool = False,
    ) -> None:
        is_self_hosted_billing: bool = not isinstance(self, RealmBillingSession)
        if stripe_invoice_paid:
            customer: Customer = self.update_or_create_customer()
        else:
            customer = self.update_or_create_stripe_customer()
        self.ensure_current_plan_is_upgradable(customer, plan_tier)
        billing_cycle_anchor: Optional[datetime] = None
        if complimentary_access_plan is not None:
            free_trial = False
        if upgrade_when_complimentary_access_plan_ends:
            assert complimentary_access_plan is not None
            assert complimentary_access_plan.end_date is not None
            billing_cycle_anchor = complimentary_access_plan.end_date
        fixed_price_plan_offer: Optional[CustomerPlanOffer] = get_configured_fixed_price_plan_offer(
            customer, plan_tier
        )
        if fixed_price_plan_offer is not None:
            assert automanage_licenses is True
        billing_cycle_anchor, next_invoice_date, period_end, price_per_license = compute_plan_parameters(
            plan_tier=plan_tier,
            billing_schedule=billing_schedule,
            customer=customer,
            free_trial=free_trial,
            billing_cycle_anchor=billing_cycle_anchor,
            is_self_hosted_billing=is_self_hosted_billing,
            upgrade_when_complimentary_access_plan_ends=upgrade_when_complimentary_access_plan_ends,
        )
        with transaction.atomic(durable=True):
            current_licenses_count: int = self.get_billable_licenses_for_customer(
                customer, plan_tier, licenses
            )
            if current_licenses_count != licenses and (not automanage_licenses):
                billable_licenses: int = max(current_licenses_count, licenses)
            else:
                billable_licenses = current_licenses_count
            plan_params: Dict[str, Any] = {
                "automanage_licenses": automanage_licenses,
                "charge_automatically": charge_automatically,
                "billing_cycle_anchor": billing_cycle_anchor,
                "billing_schedule": billing_schedule,
                "tier": plan_tier,
            }
            if fixed_price_plan_offer is None:
                plan_params["price_per_license"] = price_per_license
                _price_per_license, percent_off = get_price_per_license_and_discount(
                    plan_tier, billing_schedule, customer
                )
                plan_params["discount"] = percent_off
                assert price_per_license == _price_per_license
            if free_trial:
                plan_params["status"] = CustomerPlan.FREE_TRIAL
                if charge_automatically:
                    assert customer.stripe_customer_id is not None
                    stripe_customer: stripe.Customer = stripe_get_customer(customer.stripe_customer_id)
                    if not stripe_customer_has_credit_card_as_default_payment_method(stripe_customer):
                        raise BillingError("no payment method", _("Please add a credit card before starting your free trial."))
            event_time: datetime = billing_cycle_anchor
            if upgrade_when_complimentary_access_plan_ends:
                assert complimentary_access_plan is not None
                if charge_automatically:
                    assert customer.stripe_customer_id is not None
                    stripe_customer = stripe_get_customer(customer.stripe_customer_id)
                    if not stripe_customer_has_credit_card_as_default_payment_method(stripe_customer):
                        raise BillingError(
                            "no payment method", _("Please add a credit card to schedule upgrade.")
                        )
                plan_params["status"] = CustomerPlan.NEVER_STARTED
                plan_params["invoicing_status"] = CustomerPlan.INVOICING_STATUS_INITIAL_INVOICE_TO_BE_SENT
                event_time = timezone_now().replace(microsecond=0)
                assert complimentary_access_plan.end_date == billing_cycle_anchor
                last_ledger_entry: Optional[LicenseLedger] = LicenseLedger.objects.filter(
                    plan=complimentary_access_plan
                ).order_by("-id").first()
                assert last_ledger_entry is not None
                last_ledger_entry.licenses_at_next_renewal = billable_licenses
                last_ledger_entry.save(update_fields=["licenses_at_next_renewal"])
                complimentary_access_plan.status = CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END
                complimentary_access_plan.save(update_fields=["status"])
            elif complimentary_access_plan is not None:
                complimentary_access_plan.status = CustomerPlan.ENDED
                complimentary_access_plan.save(update_fields=["status"])
            if fixed_price_plan_offer is not None:
                assert automanage_licenses is True
                plan_params["fixed_price"] = fixed_price_plan_offer.fixed_price
                period_end = add_months(
                    billing_cycle_anchor, CustomerPlan.FIXED_PRICE_PLAN_DURATION_MONTHS
                )
                plan_params["end_date"] = period_end
                plan_params["status"] = CustomerPlan.NEVER_STARTED
                plan_params["billing_schedule"] = current_plan.billing_schedule
                plan_params["next_invoice_date"] = current_plan.end_date
                plan_params["invoicing_status"] = CustomerPlan.INVOICING_STATUS_INITIAL_INVOICE_TO_BE_SENT
                plan_params["automanage_licenses"] = True
                CustomerPlan.objects.create(customer=customer, **plan_params)
                self.write_to_audit_log(
                    event_type=BillingSessionEventType.CUSTOMER_PLAN_CREATED,
                    event_time=timezone_now(),
                    extra_data=plan_params,
                )
                current_plan.status = CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END
                current_plan.next_invoice_date = current_plan.end_date
                current_plan.save(update_fields=["status", "next_invoice_date"])
                return
            plan: CustomerPlan = CustomerPlan.objects.create(
                customer=customer, next_invoice_date=next_invoice_date, **plan_params
            )
            self.write_to_audit_log(
                event_type=BillingSessionEventType.CUSTOMER_PLAN_CREATED,
                event_time=event_time,
                extra_data=plan_params,
            )
            if plan.status < CustomerPlan.LIVE_STATUS_THRESHOLD:
                self.do_change_plan_type(tier=plan_tier)
                ledger_entry: LicenseLedger = LicenseLedger.objects.create(
                    plan=plan,
                    is_renewal=True,
                    event_time=event_time,
                    licenses=licenses,
                    licenses_at_next_renewal=licenses,
                )
                plan.invoiced_through = ledger_entry
                plan.save(update_fields=["invoiced_through"])
                if (
                    stripe_invoice_paid
                    and billable_licenses != licenses
                    and (not customer.exempt_from_license_number_check)
                    and (not fixed_price_plan_offer)
                ):
                    if billable_licenses > licenses:
                        LicenseLedger.objects.create(
                            plan=plan,
                            is_renewal=False,
                            event_time=event_time,
                            licenses=billable_licenses,
                            licenses_at_next_renewal=billable_licenses,
                        )
                        self.invoice_plan(plan, event_time)
                    else:
                        LicenseLedger.objects.create(
                            plan=plan,
                            is_renewal=False,
                            event_time=event_time,
                            licenses=licenses,
                            licenses_at_next_renewal=billable_licenses,
                        )
                        context: Dict[str, Any] = {
                            "billing_entity": self.billing_entity_display_name,
                            "support_url": self.support_url(),
                            "paid_licenses": licenses,
                            "current_licenses": billable_licenses,
                            "notice_reason": "license_discrepancy",
                        }
                        send_email(
                            "zerver/emails/internal_billing_notice",
                            to_emails=[BILLING_SUPPORT_EMAIL],
                            from_address=FromAddress.tokenized_no_reply_address(),
                            context=context,
                        )
        if not stripe_invoice_paid and (not (free_trial or upgrade_when_complimentary_access_plan_ends)):
            assert plan is not None
            self.generate_invoice_for_upgrade(
                customer=customer,
                price_per_license=price_per_license,
                fixed_price=plan.fixed_price,
                licenses=billable_licenses,
                plan_tier=plan.tier,
                billing_schedule=billing_schedule,
                charge_automatically=False,
                invoice_period={"start": datetime_to_timestamp(billing_cycle_anchor), "end": datetime_to_timestamp(period_end)},
            )
        elif free_trial and (not charge_automatically):
            assert stripe_invoice_paid is False
            assert plan is not None
            assert plan.next_invoice_date is not None
            self.generate_stripe_invoice(
                plan_tier=plan.tier,
                licenses=billable_licenses,
                license_management="automatic" if automanage_licenses else "manual",
                billing_schedule=billing_schedule,
                billing_modality="send_invoice",
                on_free_trial=True,
                days_until_due=(plan.next_invoice_date - event_time).days,
                current_plan_id=plan.id,
            )

    def do_upgrade(self, upgrade_request: UpgradeRequest) -> Dict[str, Any]:
        customer: Optional[Customer] = self.get_customer()
        if customer is not None:
            self.ensure_current_plan_is_upgradable(customer, upgrade_request.tier)
        billing_modality: str = upgrade_request.billing_modality
        schedule: str = upgrade_request.schedule
        license_management: str = upgrade_request.license_management
        if billing_modality == "send_invoice":
            license_management = "manual"
        licenses: Optional[int] = upgrade_request.licenses
        request_seat_count: int = unsign_seat_count(
            upgrade_request.signed_seat_count, upgrade_request.salt
        )
        seat_count: int = self.stale_seat_count_check(request_seat_count, upgrade_request.tier)
        if billing_modality == "charge_automatically" and license_management == "automatic":
            licenses = seat_count
        exempt_from_license_number_check: bool = (
            customer is not None and customer.exempt_from_license_number_check
        )
        check_upgrade_parameters(
            billing_modality=billing_modality,
            schedule=schedule,
            license_management=license_management,
            licenses=licenses,
            seat_count=seat_count,
            exempt_from_license_number_check=exempt_from_license_number_check,
            min_licenses_for_plan=self.min_licenses_for_plan(upgrade_request.tier),
        )
        assert licenses is not None and license_management is not None
        automanage_licenses: bool = license_management == "automatic"
        charge_automatically: bool = billing_modality == "charge_automatically"
        billing_schedule: BillingSchedule = {
            "annual": CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            "monthly": CustomerPlan.BILLING_SCHEDULE_MONTHLY,
        }[schedule]
        data: Dict[str, Any] = {}
        is_self_hosted_billing: bool = not isinstance(self, RealmBillingSession)
        free_trial: bool = is_free_trial_offer_enabled(is_self_hosted_billing, upgrade_request.tier)
        if customer is not None:
            fixed_price_plan_offer: Optional[CustomerPlanOffer] = get_configured_fixed_price_plan_offer(
                customer, upgrade_request.tier
            )
            if fixed_price_plan_offer is not None:
                free_trial = False
        if self.customer_plan_exists():
            free_trial = False
        complimentary_access_plan: Optional[CustomerPlan] = self.get_complimentary_access_plan(customer)
        upgrade_when_complimentary_access_plan_ends: bool = (
            complimentary_access_plan is not None
            and upgrade_request.remote_server_plan_start_date == "billing_cycle_end_date"
        )
        if upgrade_when_complimentary_access_plan_ends or free_trial:
            self.process_initial_upgrade(
                plan_tier=upgrade_request.tier,
                licenses=licenses,
                automanage_licenses=automanage_licenses,
                billing_schedule=billing_schedule,
                charge_automatically=charge_automatically,
                free_trial=free_trial,
                complimentary_access_plan=complimentary_access_plan,
                upgrade_when_complimentary_access_plan_ends=upgrade_when_complimentary_access_plan_ends,
            )
            data["organization_upgrade_successful"] = True
        else:
            stripe_invoice_id: str = self.generate_stripe_invoice(
                plan_tier=upgrade_request.tier,
                licenses=licenses,
                license_management=license_management,
                billing_schedule=billing_schedule,
                billing_modality=billing_modality,
            )
            data["stripe_invoice_id"] = stripe_invoice_id
        return data

    def do_change_schedule_after_free_trial(
        self, plan: CustomerPlan, schedule: str
    ) -> None:
        assert plan.charge_automatically
        assert schedule in (
            CustomerPlan.BILLING_SCHEDULE_MONTHLY,
            CustomerPlan.BILLING_SCHEDULE_ANNUAL,
        )
        last_ledger_entry: Optional[LicenseLedger] = LicenseLedger.objects.filter(plan=plan).order_by("-id").first()
        assert last_ledger_entry is not None
        licenses_at_next_renewal: Optional[int] = last_ledger_entry.licenses_at_next_renewal
        assert licenses_at_next_renewal is not None
        next_billing_cycle: datetime = start_of_next_billing_cycle(plan, last_ledger_entry.event_time)
        if plan.fixed_price is not None:
            raise BillingError("Customer is already on monthly fixed plan.")
        plan.status = CustomerPlan.ENDED
        plan.next_invoice_date = None
        plan.save(update_fields=["status", "next_invoice_date"])
        price_per_license, discount_for_current_plan = get_price_per_license_and_discount(
            plan.tier, schedule, plan.customer
        )
        new_plan: CustomerPlan = CustomerPlan.objects.create(
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
        ledger_entry: LicenseLedger = LicenseLedger.objects.create(
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

    def get_next_billing_cycle(
        self, plan: CustomerPlan
    ) -> datetime:
        if plan.status in (
            CustomerPlan.FREE_TRIAL,
            CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL,
            CustomerPlan.NEVER_STARTED,
        ):
            assert plan.next_invoice_date is not None
            next_billing_cycle: datetime = plan.next_invoice_date
        elif plan.status == CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END:
            assert plan.end_date is not None
            next_billing_cycle = plan.end_date
        else:
            last_ledger_renewal: Optional[LicenseLedger] = LicenseLedger.objects.filter(
                plan=plan, is_renewal=True
            ).order_by("-id").first()
            assert last_ledger_renewal is not None
            last_renewal: datetime = last_ledger_renewal.event_time
            next_billing_cycle = start_of_next_billing_cycle(plan, last_renewal)
        if plan.end_date is not None:
            next_billing_cycle = min(next_billing_cycle, plan.end_date)
        return next_billing_cycle

    def validate_plan_license_management(
        self, plan: CustomerPlan, renewal_license_count: int
    ) -> None:
        if plan.customer.exempt_from_license_number_check:
            return
        if plan.tier not in [
            CustomerPlan.TIER_CLOUD_STANDARD,
            CustomerPlan.TIER_CLOUD_PLUS,
        ]:
            return
        min_licenses: int = self.min_licenses_for_plan(plan.tier)
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

    @transaction.atomic(durable=True)
    def make_end_of_cycle_updates_if_needed(
        self, plan: CustomerPlan, event_time: datetime
    ) -> Tuple[Optional[CustomerPlan], Optional[LicenseLedger]]:
        last_ledger_entry: Optional[LicenseLedger] = LicenseLedger.objects.filter(
            plan=plan, event_time__lte=event_time
        ).order_by("-id").first()
        next_billing_cycle: datetime = self.get_next_billing_cycle(plan)
        event_in_next_billing_cycle: bool = next_billing_cycle <= event_time
        if event_in_next_billing_cycle and last_ledger_entry is not None:
            licenses_at_next_renewal: Optional[int] = last_ledger_entry.licenses_at_next_renewal
            assert licenses_at_next_renewal is not None
            if plan.end_date == next_billing_cycle and plan.status == CustomerPlan.ACTIVE:
                self.process_downgrade(plan, True)
                return (None, None)
            if plan.status == CustomerPlan.ACTIVE:
                self.validate_plan_license_management(plan, licenses_at_next_renewal)
                return (None, LicenseLedger.objects.create(
                    plan=plan,
                    is_renewal=True,
                    event_time=next_billing_cycle,
                    licenses=licenses_at_next_renewal,
                    licenses_at_next_renewal=licenses_at_next_renewal,
                ))
            if plan.is_free_trial():
                is_renewal: bool = True
                if not plan.charge_automatically:
                    last_sent_invoice: Optional[Invoice] = Invoice.objects.filter(
                        plan=plan
                    ).order_by("-id").first()
                    if last_sent_invoice and last_sent_invoice.status == Invoice.PAID:
                        is_renewal = False
                    else:
                        plan.status = CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL
                        plan.save(update_fields=["status"])
                        self.make_end_of_cycle_updates_if_needed(plan, event_time)
                        return (None, None)
                plan.invoiced_through = last_ledger_entry
                plan.billing_cycle_anchor = next_billing_cycle.replace(microsecond=0)
                plan.status = CustomerPlan.ACTIVE
                plan.save(update_fields=["invoiced_through", "billing_cycle_anchor", "status"])
                return (
                    None,
                    LicenseLedger.objects.create(
                        plan=plan,
                        is_renewal=is_renewal,
                        event_time=next_billing_cycle,
                        licenses=licenses_at_next_renewal,
                        licenses_at_next_renewal=licenses_at_next_renewal,
                    ),
                )
            if plan.status == CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END:
                plan.status = CustomerPlan.ENDED
                plan.save(update_fields=["status"])
                new_plan: CustomerPlan = CustomerPlan.objects.get(
                    customer=plan.customer,
                    billing_cycle_anchor=plan.end_date,
                    status=CustomerPlan.NEVER_STARTED,
                )
                self.validate_plan_license_management(new_plan, licenses_at_next_renewal)
                new_plan.status = CustomerPlan.ACTIVE
                new_plan.save(update_fields=["status"])
                self.do_change_plan_type(tier=new_plan.tier, background_update=True)
                return (
                    None,
                    LicenseLedger.objects.create(
                        plan=new_plan,
                        is_renewal=True,
                        event_time=next_billing_cycle,
                        licenses=licenses_at_next_renewal,
                        licenses_at_next_renewal=licenses_at_next_renewal,
                    ),
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
                new_plan: CustomerPlan = CustomerPlan.objects.create(
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
                new_plan_ledger_entry: LicenseLedger = LicenseLedger.objects.create(
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
                return (new_plan, new_plan_ledger_entry)
            if plan.status == CustomerPlan.SWITCH_TO_MONTHLY_AT_END_OF_CYCLE:
                self.validate_plan_license_management(plan, licenses_at_next_renewal)
                if plan.fixed_price is not None:
                    raise BillingError("Customer is already on monthly fixed plan.")
                plan.status = CustomerPlan.ENDED
                plan.save(update_fields=["status"])
                price_per_license, discount_for_current_plan = get_price_per_license_and_discount(
                    plan.tier, CustomerPlan.BILLING_SCHEDULE_MONTHLY, plan.customer
                )
                new_plan: CustomerPlan = CustomerPlan.objects.create(
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
                new_plan_ledger_entry: LicenseLedger = LicenseLedger.objects.create(
                    plan=new_plan,
                    is_renewal=True,
                    event_time=next_billing_cycle,
                    licenses=licenses_at_next_renewal,
                    licenses_at_next_renewal=licenses_at_next_renewal,
                )
                self.write_to_audit_log(
                    event_type=BillingSessionEventType.CUSTOMER_SWITCHED_FROM_ANNUAL_TO_MONTHLY_plan,
                    event_time=event_time,
                    extra_data={"annual_plan_id": plan.id, "monthly_plan_id": new_plan.id},
                    background_update=True,
                )
                return (new_plan, new_plan_ledger_entry)
            if plan.status == CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL:
                self.downgrade_now_without_creating_additional_invoices(plan, background_update=True)
            if plan.status == CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE:
                self.process_downgrade(plan, background_update=True)
            return (None, None)
        return (None, last_ledger_entry)

    def get_amount_to_credit_for_plan_tier_change(
        self, current_plan: CustomerPlan, plan_change_date: datetime
    ) -> int:
        return get_amount_to_credit_for_plan_tier_change(current_plan, plan_change_date)

    def invoice_plan(self, plan: CustomerPlan, event_time: datetime) -> None:
        # Implementation needed based on the context of the code
        pass

    def get_billable_licenses_for_customer(
        self, customer: Customer, tier: int, licenses: int, event_time: Optional[datetime] = None
    ) -> int:
        return get_billable_licenses_for_customer(self, customer, tier, licenses, event_time)

    def downgrade_now_without_creating_additional_invoices(
        self, plan: Optional[CustomerPlan] = None, background_update: bool = False
    ) -> None:
        if plan is None:
            customer: Optional[Customer] = self.get_customer()
            if customer is None:
                return
            plan = get_current_plan_by_customer(customer)
            if plan is None:
                return
        self.process_downgrade(plan, background_update=background_update)
        plan.invoiced_through = LicenseLedger.objects.filter(plan=plan).order_by("-id").first()
        plan.next_invoice_date = next_invoice_date(plan)
        plan.save(update_fields=["invoiced_through", "next_invoice_date"])

    def void_all_open_invoices(self) -> int:
        customer: Optional[Customer] = self.get_customer()
        if customer is None:
            return 0
        voided_invoices_count: int = 0
        for invoice in get_all_invoices_for_customer(customer):
            if invoice.status == "open":
                if invoice.id is not None:
                    stripe.Invoice.void_invoice(invoice.id)
                    voided_invoices_count += 1
        return voided_invoices_count

    def get_flat_discount_info(
        self, customer: Optional[Customer] = None
    ) -> Tuple[int, int]:
        return get_flat_discount_info(self, customer)

    def get_upgrade_page_context(
        self, initial_upgrade_request: InitialUpgradeRequest
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        return get_upgrade_page_context(self, initial_upgrade_request)

    def ensure_current_plan_is_upgradable(
        self, customer: Customer, new_plan_tier: int
    ) -> None:
        ensure_customer_does_not_have_active_plan(customer)
