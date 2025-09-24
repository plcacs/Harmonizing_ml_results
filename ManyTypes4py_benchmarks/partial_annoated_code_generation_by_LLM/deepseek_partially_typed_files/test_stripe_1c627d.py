import itertools
import json
import operator
import os
import re
import sys
import typing
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar, cast
from unittest import mock
from unittest.mock import MagicMock, Mock, patch
import orjson
import responses
import stripe
import time_machine
from django.conf import settings
from django.core import signing
from django.urls.resolvers import get_resolver
from django.utils.crypto import get_random_string
from django.utils.timezone import now as timezone_now
from typing_extensions import ParamSpec, override
from corporate.lib.stripe import (
    DEFAULT_INVOICE_DAYS_UNTIL_DUE,
    MAX_INVOICED_LICENSES,
    MIN_INVOICED_LICENSES,
    STRIPE_API_VERSION,
    BillingError,
    BillingSessionAuditLogEventError,
    BillingSessionEventType,
    InitialUpgradeRequest,
    InvalidBillingScheduleError,
    InvalidTierError,
    RealmBillingSession,
    RemoteRealmBillingSession,
    RemoteServerBillingSession,
    StripeCardError,
    SupportRequestError,
    SupportType,
    SupportViewRequest,
    UpdatePlanRequest,
    add_months,
    catch_stripe_errors,
    compute_plan_parameters,
    customer_has_credit_card_as_default_payment_method,
    customer_has_last_n_invoices_open,
    do_deactivate_remote_server,
    do_reactivate_remote_server,
    downgrade_small_realms_behind_on_payments_as_needed,
    get_latest_seat_count,
    get_plan_renewal_or_end_date,
    get_price_per_license,
    invoice_plans_as_needed,
    is_free_trial_offer_enabled,
    is_realm_on_free_trial,
    next_month,
    sign_string,
    stripe_customer_has_credit_card_as_default_payment_method,
    stripe_get_customer,
    unsign_string,
)
from corporate.models import (
    Customer,
    CustomerPlan,
    CustomerPlanOffer,
    Event,
    Invoice,
    LicenseLedger,
    ZulipSponsorshipRequest,
    get_current_plan_by_customer,
    get_current_plan_by_realm,
    get_customer_by_realm,
    get_customer_by_remote_realm,
)
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
from zilencer.models import (
    RemoteRealm,
    RemoteRealmAuditLog,
    RemoteRealmBillingUser,
    RemoteServerBillingUser,
    RemoteZulipServer,
    RemoteZulipServerAuditLog,
)

if TYPE_CHECKING:
    from django.test.client import _MonkeyPatchedWSGIResponse as TestHttpResponse

CallableT = TypeVar("CallableT", bound=Callable[..., Any])
ParamT = ParamSpec("ParamT")
ReturnT = TypeVar("ReturnT")
STRIPE_FIXTURES_DIR = "corporate/tests/stripe_fixtures"


def stripe_fixture_path(decorated_function_name: str, mocked_function_name: str, call_count: int) -> str:
    decorated_function_name = decorated_function_name.removeprefix("test_")
    mocked_function_name = mocked_function_name.removeprefix("stripe.")
    return f"{STRIPE_FIXTURES_DIR}/{decorated_function_name}--{mocked_function_name}.{call_count}.json"


def fixture_files_for_function(decorated_function: Callable[..., Any]) -> list[str]:
    decorated_function_name = decorated_function.__name__
    decorated_function_name = decorated_function_name.removeprefix("test_")
    return sorted(
        (
            f"{STRIPE_FIXTURES_DIR}/{f}"
            for f in os.listdir(STRIPE_FIXTURES_DIR)
            if f.startswith(decorated_function_name + "--")
        )
    )


def generate_and_save_stripe_fixture(
    decorated_function_name: str, mocked_function_name: str, mocked_function: Callable[..., Any]
) -> Callable[..., Any]:
    def _generate_and_save_stripe_fixture(*args: Any, **kwargs: Any) -> Any:
        mock = operator.attrgetter(mocked_function_name)(sys.modules[__name__])
        fixture_path = stripe_fixture_path(decorated_function_name, mocked_function_name, mock.call_count)
        try:
            with responses.RequestsMock() as request_mock:
                request_mock.add_passthru("https://api.stripe.com")
                stripe_object = mocked_function(*args, **kwargs)
        except stripe.StripeError as e:
            with open(fixture_path, "w") as f:
                assert e.headers is not None
                error_dict = {**vars(e), "headers": dict(e.headers)}
                if e.http_body is None:
                    assert e.json_body is not None
                    error_dict["http_body"] = json.dumps(e.json_body)
                f.write(json.dumps(error_dict, indent=2, separators=(",", ": "), sort_keys=True) + "\n")
            raise
        with open(fixture_path, "w") as f:
            if stripe_object is not None:
                f.write(str(stripe_object) + "\n")
            else:
                f.write("{}\n")
        return stripe_object

    return _generate_and_save_stripe_fixture


def read_stripe_fixture(decorated_function_name: str, mocked_function_name: str) -> Callable[..., Any]:
    def _read_stripe_fixture(*args: Any, **kwargs: Any) -> Any:
        mock = operator.attrgetter(mocked_function_name)(sys.modules[__name__])
        fixture_path = stripe_fixture_path(decorated_function_name, mocked_function_name, mock.call_count)
        with open(fixture_path, "rb") as f:
            fixture = orjson.loads(f.read())
        if "json_body" in fixture:
            requester = stripe._api_requestor._APIRequestor()
            requester._interpret_response(
                fixture["http_body"], fixture["http_status"], fixture["headers"], "V1"
            )
        return stripe.convert_to_stripe_object(fixture)

    return _read_stripe_fixture


def delete_fixture_data(decorated_function: Callable[..., Any]) -> None:
    for fixture_file in fixture_files_for_function(decorated_function):
        os.remove(fixture_file)


def normalize_fixture_data(
    decorated_function: Callable[..., Any], tested_timestamp_fields: list[str] = []
) -> None:
    id_lengths = [
        ("test", 12),
        ("cus", 14),
        ("prod", 14),
        ("req", 14),
        ("si", 14),
        ("sli", 14),
        ("sub", 14),
        ("acct", 16),
        ("card", 24),
        ("ch", 24),
        ("ii", 24),
        ("il", 24),
        ("in", 24),
        ("pi", 24),
        ("price", 24),
        ("src", 24),
        ("src_client_secret", 24),
        ("tok", 24),
        ("txn", 24),
        ("invst", 26),
        ("rcpt", 31),
        ("seti", 24),
        ("pm", 24),
        ("setatt", 24),
        ("bpc", 24),
        ("bps", 24),
    ]
    pattern_translations = {
        '"exp_month": ([0-9]+)': "1",
        '"exp_year": ([0-9]+)': "9999",
        '"postal_code": "([0-9]+)"': "12345",
        '"invoice_prefix": "([A-Za-z0-9]{7,8})"': "NORMALIZED",
        '"fingerprint": "([A-Za-z0-9]{16})"': "NORMALIZED",
        '"number": "([A-Za-z0-9]{7,8}-[A-Za-z0-9]{4})"': "NORMALIZED",
        '"address": "([A-Za-z0-9]{9}-test_[A-Za-z0-9]{12})"': "000000000-test_NORMALIZED",
        '"client_secret": "([\\w]+)"': "NORMALIZED",
        '"url": "https://billing.stripe.com/p/session/test_([\\w]+)"': "NORMALIZED",
        '"url": "https://checkout.stripe.com/c/pay/cs_test_([\\w#%]+)"': "NORMALIZED",
        '"receipt_url": "https://pay.stripe.com/receipts/invoices/([\\w-]+)\\?s=[\\w]+"': "NORMALIZED",
        '"hosted_invoice_url": "https://invoice.stripe.com/i/acct_[\\w]+/test_[\\w,]+\\?s=[\\w]+"': '"hosted_invoice_url": "https://invoice.stripe.com/i/acct_NORMALIZED/test_NORMALIZED?s=ap"',
        '"invoice_pdf": "https://pay.stripe.com/invoice/acct_[\\w]+/test_[\\w,]+/pdf\\?s=[\\w]+"': '"invoice_pdf": "https://pay.stripe.com/invoice/acct_NORMALIZED/test_NORMALIZED/pdf?s=ap"',
        '"id": "([\\w]+)"': "FILE_NAME",
        '"realm_id": "[0-9]+"': '"realm_id": "1"',
        '"account_name": "[\\w\\s]+"': '"account_name": "NORMALIZED"',
    }
    pattern_translations.update(
        {
            f"{prefix}_[A-Za-z0-9]{{{length}}}": f"{prefix}_NORMALIZED"
            for (prefix, length) in id_lengths
        }
    )
    for (i, timestamp_field) in enumerate(tested_timestamp_fields):
        pattern_translations[
            f'"{timestamp_field}": 1[5-9][0-9]{{8}}(?![0-9-])'
        ] = f'"{timestamp_field}": {1000000000 + i}'
    normalized_values: dict[str, dict[str, str]] = {
        pattern: {} for pattern in pattern_translations
    }
    for fixture_file in fixture_files_for_function(decorated_function):
        with open(fixture_file) as f:
            file_content = f.read()
        for (pattern, translation) in pattern_translations.items():
            for match in re.findall(pattern, file_content):
                if match not in normalized_values[pattern]:
                    if pattern.startswith('"id": "'):
                        normalized_values[pattern][match] = fixture_file.split("/")[-1]
                    else:
                        normalized_values[pattern][match] = translation
                file_content = file_content.replace(match, normalized_values[pattern][match])
        file_content = re.sub('(?<="risk_score": )(\\d+)', "0", file_content)
        file_content = re.sub('(?<="times_redeemed": )(\\d+)', "0", file_content)
        file_content = re.sub(
            '(?<="idempotency_key": )"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"',
            '"00000000-0000-0000-0000-000000000000"',
            file_content,
        )
        file_content = re.sub('(?<="Date": )"(.* GMT)"', '"NORMALIZED DATETIME"', file_content)
        file_content = re.sub("[0-3]\\d [A-Z][a-z]{2} 20[1-2]\\d", "NORMALIZED DATE", file_content)
        file_content = re.sub('"\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}"', '"0.0.0.0"', file_content)
        file_content = re.sub(": (1[5-9][0-9]{8})(?![0-9-])", ": 1000000000", file_content)
        with open(fixture_file, "w") as f:
            f.write(file_content)


MOCKED_STRIPE_FUNCTION_NAMES = [
    f"stripe.{name}"
    for name in [
        "billing_portal.Configuration.create",
        "billing_portal.Session.create",
        "checkout.Session.create",
        "checkout.Session.list",
        "Charge.create",
        "Charge.list",
        "Coupon.create",
        "Customer.create",
        "Customer.create_balance_transaction",
        "Customer.list_balance_transactions",
        "Customer.retrieve",
        "Customer.save",
        "Customer.list",
        "Customer.modify",
        "Event.list",
        "Invoice.create",
        "Invoice.finalize_invoice",
        "Invoice.list",
        "Invoice.pay",
        "Invoice.refresh",
        "Invoice.retrieve",
        "Invoice.upcoming",
        "Invoice.void_invoice",
        "InvoiceItem.create",
        "InvoiceItem.list",
        "PaymentMethod.attach",
        "PaymentMethod.create",
        "PaymentMethod.detach",
        "PaymentMethod.list",
        "Plan.create",
        "Product.create",
        "SetupIntent.create",
        "SetupIntent.list",
        "SetupIntent.retrieve",
        "Subscription.create",
        "Subscription.delete",
        "Subscription.retrieve",
        "Subscription.save",
        "Token.create",
    ]
]


def mock_stripe(
    tested_timestamp_fields: list[str] = [], generate: bool = settings.GENERATE_STRIPE_FIXTURES
) -> Callable[..., Any]:
    def _mock_stripe(decorated_function: Callable[..., Any]) -> Callable[..., Any]:
        generate_fixture = generate
        if generate_fixture:
            assert stripe.api_key
        for mocked_function_name in MOCKED_STRIPE_FUNCTION_NAMES:
            mocked_function = operator.attrgetter(mocked_function_name)(sys.modules[__name__])
            if generate_fixture:
                side_effect = generate_and_save_stripe_fixture(
                    decorated_function.__name__, mocked_function_name, mocked_function
                )
            else:
                side_effect = read_stripe_fixture(decorated_function.__name__, mocked_function_name)
            decorated_function = patch(
                mocked_function_name,
                side_effect=side_effect,
                autospec=mocked_function_name.endswith(".refresh"),
            )(decorated_function)

        @wraps(decorated_function)
        def wrapped(*args: ParamT.args, **kwargs: ParamT.kwargs) -> ReturnT:
            if generate_fixture:
                delete_fixture_data(decorated_function)
                val = decorated_function(*args, **kwargs)
                normalize_fixture_data(decorated_function, tested_timestamp_fields)
                return val
            else:
                return decorated_function(*args, **kwargs)

        return wrapped

    return _mock_stripe


class StripeTestCase(ZulipTestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        realm = get_realm("zulip")
        active_emails = [
            self.example_email("AARON"),
            self.example_email("cordelia"),
            self.example_email("hamlet"),
            self.example_email("iago"),
            self.example_email("othello"),
            self.example_email("desdemona"),
            self.example_email("polonius"),
            self.example_email("default_bot"),
        ]
        for user_profile in UserProfile.objects.filter(realm_id=realm.id).exclude(
            delivery_email__in=active_emails
        ):
            do_deactivate_user(user_profile, acting_user=None)
        self.assertEqual(UserProfile.objects.filter(realm=realm, is_active=True).count(), 8)
        self.assertEqual(UserProfile.objects.exclude(realm=realm).filter(is_active=True).count(), 10)
        self.assertEqual(get_latest_seat_count(realm), 6)
        self.seat_count = 6
        (self.signed_seat_count, self.salt) = sign_string(str(self.seat_count))
        self.now = datetime(2012, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        self.next_month = datetime(2012, 2, 2, 3, 4, 5, tzinfo=timezone.utc)
        self.next_year = datetime(2013, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        hamlet = self.example_user("hamlet")
        hamlet.is_billing_admin = True
        hamlet.save(update_fields=["is_billing_admin"])
        self.billing_session: RealmBillingSession | RemoteRealmBillingSession | RemoteServerBillingSession = RealmBillingSession(
            user=hamlet, realm=realm
        )

    def get_signed_seat_count_from_response(self, response: Any) -> Optional[str]:
        match = re