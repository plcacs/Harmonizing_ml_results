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
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypeVar, cast
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
from corporate.tests.test_remote_billing import (
    RemoteRealmBillingTestCase,
    RemoteServerTestCase,
)
from corporate.views.remote_billing_page import generate_confirmation_link_for_server_deactivation
from zerver.actions.create_realm import do_create_realm
from zerver.actions.create_user import (
    do_activate_mirror_dummy_user,
    do_create_user,
    do_reactivate_user,
)
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


def stripe_fixture_path(
    decorated_function_name: str, mocked_function_name: str, call_count: int
) -> str:
    decorated_function_name = decorated_function_name.removeprefix("test_")
    mocked_function_name = mocked_function_name.removeprefix("stripe.")
    return f"{STRIPE_FIXTURES_DIR}/{decorated_function_name}--{mocked_function_name}.{call_count}.json"


def fixture_files_for_function(decorated_function: Callable[..., Any]) -> List[str]:
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
        mock_attr = operator.attrgetter(mocked_function_name)(sys.modules[__name__])
        fixture_path = stripe_fixture_path(decorated_function_name, mocked_function_name, mock_attr.call_count)
        try:
            with responses.RequestsMock() as request_mock:
                request_mock.add_passthru("https://api.stripe.com")
                stripe_object = mocked_function(*args, **kwargs)
        except stripe.StripeError as e:
            with open(fixture_path, "w") as f:
                assert e.headers is not None
                error_dict: Dict[str, Any] = {**vars(e), "headers": dict(e.headers)}
                if e.http_body is None:
                    assert e.json_body is not None
                    error_dict["http_body"] = json.dumps(e.json_body)
                f.write(
                    json.dumps(error_dict, indent=2, separators=(",", ": "), sort_keys=True) + "\n"
                )
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
        mock_attr = operator.attrgetter(mocked_function_name)(sys.modules[__name__])
        fixture_path = stripe_fixture_path(decorated_function_name, mocked_function_name, mock_attr.call_count)
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
    decorated_function: Callable[..., Any], tested_timestamp_fields: Optional[List[str]] = None
) -> None:
    if tested_timestamp_fields is None:
        tested_timestamp_fields = []
    id_lengths: List[tuple[str, int]] = [
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
    pattern_translations: Dict[str, str] = {
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
            for prefix, length in id_lengths
        }
    )
    for i, timestamp_field in enumerate(tested_timestamp_fields):
        pattern_translations[
            f'"{timestamp_field}": 1[5-9][0-9]{{8}}(?![0-9-])'
        ] = f'"{timestamp_field}": {1000000000 + i}'

    normalized_values: Dict[str, Dict[str, str]] = {pattern: {} for pattern in pattern_translations}
    for fixture_file in fixture_files_for_function(decorated_function):
        with open(fixture_file) as f:
            file_content = f.read()
        for pattern, translation in pattern_translations.items():
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
        file_content = re.sub('[0-3]\\d [A-Z][a-z]{2} 20[1-2]\\d', "NORMALIZED DATE", file_content)
        file_content = re.sub('"\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}"', '"0.0.0.0"', file_content)
        file_content = re.sub(': (1[5-9][0-9]{8})(?![0-9-])', ": 1000000000", file_content)
        with open(fixture_file, "w") as f:
            f.write(file_content)


MOCKED_STRIPE_FUNCTION_NAMES: List[str] = [
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
    tested_timestamp_fields: List[str] = [], generate: bool = settings.GENERATE_STRIPE_FIXTURES
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
                mocked_function_name, side_effect=side_effect, autospec=mocked_function_name.endswith(".refresh")
            )(decorated_function)

        @wraps(decorated_function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
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
        self.signed_seat_count, self.salt = sign_string(str(self.seat_count))
        self.now = datetime(2012, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        self.next_month = datetime(2012, 2, 2, 3, 4, 5, tzinfo=timezone.utc)
        self.next_year = datetime(2013, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        hamlet = self.example_user("hamlet")
        hamlet.is_billing_admin = True
        hamlet.save(update_fields=["is_billing_admin"])
        self.billing_session = RealmBillingSession(user=hamlet, realm=realm)

    def get_signed_seat_count_from_response(
        self, response: "TestHttpResponse"
    ) -> Optional[str]:
        match = re.search(r'name=\\"signed_seat_count\\" value=\\"(.+)\\"', response.content.decode())
        return match.group(1) if match else None

    def get_salt_from_response(self, response: "TestHttpResponse") -> Optional[str]:
        match = re.search(r'name=\\"salt\\" value=\\"(\w+)\\"', response.content.decode())
        return match.group(1) if match else None

    def get_test_card_token(
        self,
        attaches_to_customer: bool,
        charge_succeeds: Optional[bool] = None,
        card_provider: Optional[str] = None,
    ) -> str:
        if attaches_to_customer:
            assert charge_succeeds is not None
            if charge_succeeds:
                if card_provider == "visa":
                    return "tok_visa"
                if card_provider == "mastercard":
                    return "tok_mastercard"
                raise AssertionError("Unreachable code path")
            else:
                return "tok_chargeCustomerFail"
        else:
            return "tok_visa_chargeDeclined"

    def assert_details_of_valid_session_from_event_status_endpoint(
        self, stripe_session_id: str, expected_details: Dict[str, Any]
    ) -> None:
        json_response = self.client_billing_get("/billing/event/status", {"stripe_session_id": stripe_session_id})
        response_dict = self.assert_json_success(json_response)
        self.assertEqual(response_dict["session"], expected_details)

    def assert_details_of_valid_invoice_payment_from_event_status_endpoint(
        self, stripe_invoice_id: str, expected_details: Dict[str, Any]
    ) -> None:
        json_response = self.client_billing_get("/billing/event/status", {"stripe_invoice_id": stripe_invoice_id})
        response_dict = self.assert_json_success(json_response)
        self.assertEqual(response_dict["stripe_invoice"], expected_details)

    def trigger_stripe_checkout_session_completed_webhook(self, token: str) -> None:
        customer = self.billing_session.get_customer()
        assert customer is not None
        customer_stripe_id = customer.stripe_customer_id
        assert customer_stripe_id is not None
        [checkout_setup_intent] = iter(stripe.SetupIntent.list(customer=customer_stripe_id, limit=1))
        payment_method = stripe.PaymentMethod.create(
            type="card",
            card={"token": token},
            billing_details={
                "name": "John Doe",
                "address": {
                    "line1": "123 Main St",
                    "city": "San Francisco",
                    "state": "CA",
                    "postal_code": "94105",
                    "country": "US",
                },
            },
        )
        assert isinstance(checkout_setup_intent.customer, str)
        assert checkout_setup_intent.metadata is not None
        assert checkout_setup_intent.usage in {"off_session", "on_session"}
        usage = cast(Literal["off_session", "on_session"], checkout_setup_intent.usage)
        stripe_setup_intent = stripe.SetupIntent.create(
            payment_method=payment_method.id,
            confirm=True,
            payment_method_types=checkout_setup_intent.payment_method_types,
            customer=checkout_setup_intent.customer,
            metadata=checkout_setup_intent.metadata,
            usage=usage,
        )
        [stripe_session] = iter(stripe.checkout.Session.list(customer=customer_stripe_id, limit=1))
        stripe_session_dict = orjson.loads(orjson.dumps(stripe_session))
        stripe_session_dict["setup_intent"] = stripe_setup_intent.id
        event_payload = {
            "id": f"evt_{get_random_string(24)}",
            "object": "event",
            "data": {"object": stripe_session_dict},
            "type": "checkout.session.completed",
            "api_version": STRIPE_API_VERSION,
        }
        response = self.client_post("/stripe/webhook/", event_payload, content_type="application/json")
        assert response.status_code == 200

    def send_stripe_webhook_event(self, event: Dict[str, Any]) -> None:
        response = self.client_post("/stripe/webhook/", orjson.loads(orjson.dumps(event)), content_type="application/json")
        assert response.status_code == 200

    def send_stripe_webhook_events(self, most_recent_event: stripe.Event) -> None:
        while True:
            events_old_to_new = list(reversed(stripe.Event.list(ending_before=most_recent_event.id)))
            if len(events_old_to_new) == 0:
                break
            for event in events_old_to_new:
                self.send_stripe_webhook_event(event)
            most_recent_event = events_old_to_new[-1]

    def add_card_to_customer_for_upgrade(
        self, charge_succeeds: bool = True, user: Optional[UserProfile] = None, **kwargs: Any
    ) -> stripe.Customer:
        with time_machine.travel(self.now, tick=False):
            start_session_json_response = self.client_billing_post(
                "/upgrade/session/start_card_update_session", {"tier": 1}
            )
            response_dict = self.assert_json_success(start_session_json_response)
            stripe_session_id = response_dict["stripe_session_id"]
            self.assert_details_of_valid_session_from_event_status_endpoint(
                stripe_session_id,
                {
                    "type": "card_update_from_upgrade_page",
                    "status": "created",
                    "is_manual_license_management_upgrade_session": False,
                    "tier": 1,
                },
            )
            self.trigger_stripe_checkout_session_completed_webhook(
                self.get_test_card_token(attaches_to_customer=True, charge_succeeds=charge_succeeds, card_provider="visa")
            )
            self.assert_details_of_valid_session_from_event_status_endpoint(
                stripe_session_id,
                {
                    "type": "card_update_from_upgrade_page",
                    "status": "completed",
                    "is_manual_license_management_upgrade_session": False,
                    "tier": 1,
                    "event_handler": {"status": "succeeded"},
                },
            )
        return self.billing_session.get_customer()

    def upgrade(
        self,
        invoice: bool = False,
        talk_to_stripe: bool = True,
        upgrade_page_response: Optional["TestHttpResponse"] = None,
        del_args: Optional[List[str]] = None,
        dont_confirm_payment: bool = False,
        **kwargs: Any,
    ) -> "TestHttpResponse":
        if upgrade_page_response is None:
            tier = kwargs.get("tier")
            upgrade_url = f"{self.billing_session.billing_base_url}/upgrade/"
            if tier:
                upgrade_url += f"?tier={tier}"
            if self.billing_session.billing_base_url:
                upgrade_page_response = self.client_get(upgrade_url, {}, subdomain="selfhosting")
            else:
                upgrade_page_response = self.client_get(upgrade_url, {})
        params: Dict[str, Any] = {
            "schedule": "annual",
            "signed_seat_count": self.get_signed_seat_count_from_response(upgrade_page_response),
            "salt": self.get_salt_from_response(upgrade_page_response),
        }
        if invoice:
            params.update({"billing_modality": "send_invoice", "licenses": kwargs.get("licenses", 123)})
        else:
            params.update({"billing_modality": "charge_automatically", "license_management": "automatic"})
        remote_server_plan_start_date = kwargs.get("remote_server_plan_start_date")
        if remote_server_plan_start_date:
            params.update({"remote_server_plan_start_date": remote_server_plan_start_date})
        params.update(kwargs)
        if del_args is None:
            del_args = []
        for key in del_args:
            if key in params:
                del params[key]
        if talk_to_stripe:
            [last_event] = iter(stripe.Event.list(limit=1))
        existing_customer = self.billing_session.customer_plan_exists()
        upgrade_json_response = self.client_billing_post("/billing/upgrade", params)
        if upgrade_json_response.status_code != 200 or dont_confirm_payment:
            return upgrade_json_response
        is_self_hosted_billing = not isinstance(self.billing_session, RealmBillingSession)
        customer = self.billing_session.get_customer()
        assert customer is not None
        if not talk_to_stripe or (
            is_free_trial_offer_enabled(is_self_hosted_billing) and (not existing_customer)
        ):
            return upgrade_json_response
        last_sent_invoice = Invoice.objects.last()
        assert last_sent_invoice is not None
        response_dict = self.assert_json_success(upgrade_json_response)
        self.assertEqual(response_dict["stripe_invoice_id"], last_sent_invoice.stripe_invoice_id)
        self.assert_details_of_valid_invoice_payment_from_event_status_endpoint(
            last_sent_invoice.stripe_invoice_id, {"status": "sent"}
        )
        if invoice:
            stripe.Invoice.pay(last_sent_invoice.stripe_invoice_id, paid_out_of_band=True)
        self.send_stripe_webhook_events(last_event)
        return upgrade_json_response

    def add_card_and_upgrade(
        self, user: Optional[UserProfile] = None, **kwargs: Any
    ) -> stripe.Customer:
        with time_machine.travel(self.now, tick=False):
            self.add_card_to_customer_for_upgrade()
        if user is not None:
            stripe_customer = stripe_get_customer(assert_is_not_none(Customer.objects.get(realm=user.realm).stripe_customer_id))
        else:
            customer = self.billing_session.get_customer()
            assert customer is not None
            stripe_customer = stripe_get_customer(assert_is_not_none(customer.stripe_customer_id))
        self.assertTrue(
            stripe_customer_has_credit_card_as_default_payment_method(stripe_customer)
        )
        with time_machine.travel(self.now, tick=False):
            response = self.upgrade(**kwargs)
        self.assert_json_success(response)
        return stripe_customer

    def local_upgrade(
        self,
        licenses: int,
        automanage_licenses: bool,
        billing_schedule: str,
        charge_automatically: bool,
        free_trial: bool,
        stripe_invoice_paid: bool = False,
    ) -> None:
        class StripeMock(Mock):
            def __init__(self, depth: int = 1) -> None:
                super().__init__(spec=stripe.Card)
                self.id = "cus_123"
                self.created = "1000"
                self.last4 = "4242"

        def upgrade_func(
            licenses: int,
            automanage_licenses: bool,
            billing_schedule: str,
            charge_automatically: bool,
            free_trial: bool,
            stripe_invoice_paid: bool,
            *mock_args: Any,
        ) -> None:
            hamlet = self.example_user("hamlet")
            billing_session = RealmBillingSession(hamlet)
            billing_session.process_initial_upgrade(
                CustomerPlan.TIER_CLOUD_STANDARD,
                licenses,
                automanage_licenses,
                billing_schedule,
                charge_automatically,
                free_trial,
                stripe_invoice_paid=stripe_invoice_paid,
            )

        for mocked_function_name in MOCKED_STRIPE_FUNCTION_NAMES:
            with patch(mocked_function_name, return_value=StripeMock()):
                upgrade_func(
                    licenses,
                    automanage_licenses,
                    billing_schedule,
                    charge_automatically,
                    free_trial,
                    stripe_invoice_paid,
                )

    def setup_mocked_stripe(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with patch.multiple("stripe", Invoice=mock.DEFAULT, InvoiceItem=mock.DEFAULT) as mocked:
            mocked["Invoice"].create.return_value = None
            mocked["Invoice"].finalize_invoice.return_value = None
            mocked["InvoiceItem"].create.return_value = None
            callback(*args, **kwargs)
            return mocked

    def client_billing_get(self, url_suffix: str, info: Dict[str, Any] = {}) -> "TestHttpResponse":
        url = f"/json{self.billing_session.billing_base_url}" + url_suffix
        if self.billing_session.billing_base_url:
            response = self.client_get(url, info, subdomain="selfhosting")
        else:
            response = self.client_get(url, info)
        return response

    def client_billing_post(self, url_suffix: str, info: Dict[str, Any] = {}) -> "TestHttpResponse":
        url = f"/json{self.billing_session.billing_base_url}" + url_suffix
        if self.billing_session.billing_base_url:
            response = self.client_post(url, info, subdomain="selfhosting")
        else:
            response = self.client_post(url, info)
        return response

    def client_billing_patch(self, url_suffix: str, info: Dict[str, Any] = {}) -> "TestHttpResponse":
        url = f"/json{self.billing_session.billing_base_url}" + url_suffix
        if self.billing_session.billing_base_url:
            response = self.client_patch(url, info, subdomain="selfhosting")
        else:
            response = self.client_patch(url, info)
        return response


class StripeTest(StripeTestCase):
    def test_catch_stripe_errors(self) -> None:
        @catch_stripe_errors
        def raise_invalid_request_error() -> None:
            raise stripe.InvalidRequestError("message", "param", "code", json_body={})

        with self.assertLogs("corporate.stripe", "ERROR") as error_log:
            with self.assertRaises(BillingError) as billing_context:
                raise_invalid_request_error()
            self.assertEqual("other stripe error", billing_context.exception.error_description)
            self.assertEqual(
                error_log.output,
                ["ERROR:corporate.stripe:Stripe error: None None None None"],
            )

        @catch_stripe_errors
        def raise_card_error() -> None:
            error_message = "The card number is not a valid credit card number."
            json_body = {"error": {"message": error_message}}
            raise stripe.CardError(error_message, "number", "invalid_number", json_body=json_body)

        with self.assertLogs("corporate.stripe", "INFO") as info_log:
            with self.assertRaises(StripeCardError) as card_context:
                raise_card_error()
            self.assertIn("not a valid credit card", str(card_context.exception))
            self.assertEqual("card error", card_context.exception.error_description)
            self.assertEqual(
                info_log.output,
                ["INFO:corporate.stripe:Stripe card error: None None None None"],
            )

    def test_billing_not_enabled(self) -> None:
        iago = self.example_user("iago")
        with self.settings(BILLING_ENABLED=False):
            self.login_user(iago)
            response = self.client_get("/upgrade/", follow=True)
            self.assertEqual(response.status_code, 404)

    @mock_stripe()
    def test_stripe_billing_portal_urls(self, *mocks: Any) -> None:
        user = self.example_user("hamlet")
        self.login_user(user)
        self.add_card_to_customer_for_upgrade()
        response = self.client_get(f"/customer_portal/?tier={CustomerPlan.TIER_CLOUD_STANDARD}")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response["Location"].startswith("https://billing.stripe.com"))
        self.upgrade(invoice=True)
        response = self.client_get("/customer_portal/?return_to_billing_page=true")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response["Location"].startswith("https://billing.stripe.com"))
        response = self.client_get("/invoices/")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response["Location"].startswith("https://billing.stripe.com"))

    @mock_stripe()
    def test_upgrade_by_card_to_plus_plan(self, *mocks: Any) -> None:
        user = self.example_user("hamlet")
        self.login_user(user)
        response = self.client_get("/upgrade/?tier=2")
        self.assert_in_success_response(
            [
                "Your subscription will renew automatically",
                "Zulip Cloud Plus",
            ],
            response,
        )
        self.assertEqual(user.realm.plan_type, Realm.PLAN_TYPE_SELF_HOSTED)
        self.assertFalse(Customer.objects.filter(realm=user.realm).exists())
        stripe_customer = self.add_card_and_upgrade(user, tier=CustomerPlan.TIER_CLOUD_PLUS)
        self.assertEqual(stripe_customer.description, "zulip (Zulip Dev)")
        self.assertEqual(stripe_customer.discount, None)
        self.assertEqual(stripe_customer.email, user.delivery_email)
        assert stripe_customer.metadata is not None
        metadata_dict = dict(stripe_customer.metadata)
        self.assertEqual(metadata_dict["realm_str"], "zulip")
        try:
            int(metadata_dict["realm_id"])
        except ValueError:
            raise AssertionError("realm_id is not a number")
        [charge] = iter(stripe.Charge.list(customer=stripe_customer.id))
        licenses_purchased = self.billing_session.min_licenses_for_plan(CustomerPlan.TIER_CLOUD_PLUS)
        self.assertEqual(charge.amount, 12000 * licenses_purchased)
        self.assertEqual(charge.description, "Payment for Invoice")
        self.assertEqual(charge.receipt_email, user.delivery_email)
        self.assertEqual(charge.statement_descriptor, "Zulip Cloud Plus")
        [invoice] = iter(stripe.Invoice.list(customer=stripe_customer.id))
        self.assertIsNotNone(invoice.status_transitions.finalized_at)
        invoice_params = {
            "amount_due": 120000,
            "amount_paid": 120000,
            "auto_advance": False,
            "collection_method": "charge_automatically",
            "status": "paid",
            "total": 120000,
        }
        self.assertIsNotNone(invoice.charge)
        for key, value in invoice_params.items():
            self.assertEqual(invoice.get(key), value)
        [item0] = iter(invoice.lines)
        line_item_params = {
            "amount": 12000 * licenses_purchased,
            "description": "Zulip Cloud Plus",
            "discountable": False,
            "plan": None,
            "proration": False,
            "quantity": licenses_purchased,
            "period": {
                "start": datetime_to_timestamp(self.now),
                "end": datetime_to_timestamp(add_months(self.now, 12)),
            },
        }
        for key, value in line_item_params.items():
            self.assertEqual(item0.get(key), value)
        customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
        plan = CustomerPlan.objects.get(
            customer=customer,
            automanage_licenses=True,
            price_per_license=12000,
            fixed_price=None,
            discount=None,
            billing_cycle_anchor=self.now,
            billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            invoiced_through=LicenseLedger.objects.first(),
            next_invoice_date=self.next_month,
            tier=CustomerPlan.TIER_CLOUD_PLUS,
            status=CustomerPlan.ACTIVE,
        )
        LicenseLedger.objects.get(
            plan=plan,
            is_renewal=True,
            event_time=self.now,
            licenses=licenses_purchased,
            licenses_at_next_renewal=licenses_purchased,
        )
        audit_log_entries = list(
            RealmAuditLog.objects.filter(acting_user=user).values_list("event_type", "event_time").order_by("id")
        )
        self.assertEqual(
            audit_log_entries[:3],
            [
                (AuditLogEventType.STRIPE_CUSTOMER_CREATED, timestamp_to_datetime(stripe_customer.created)),
                (AuditLogEventType.STRIPE_CARD_CHANGED, self.now),
                (AuditLogEventType.CUSTOMER_PLAN_CREATED, self.now),
            ],
        )
        self.assertEqual(audit_log_entries[3][0], AuditLogEventType.REALM_PLAN_TYPE_CHANGED)
        first_audit_log_entry = RealmAuditLog.objects.filter(
            event_type=AuditLogEventType.CUSTOMER_PLAN_CREATED
        ).values_list("extra_data", flat=True).first()
        assert first_audit_log_entry is not None
        self.assertTrue(first_audit_log_entry["automanage_licenses"])
        realm = get_realm("zulip")
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_PLUS)
        self.assertEqual(realm.max_invites, Realm.INVITES_STANDARD_REALM_DAILY_MAX)
        response = self.client_get("/upgrade/")
        self.assertEqual(response.status_code, 302)
        self.assertEqual("http://zulip.testserver/billing", response["Location"])
        with time_machine.travel(self.now, tick=False):
            response = self.client_get("/billing/")
        self.assert_not_in_success_response(["Pay annually"], response)
        for substring in [
            "Zulip Cloud Plus",
            str(licenses_purchased),
            "Number of licenses",
            f"{licenses_purchased}",
            "Your plan will automatically renew on",
            "January 2, 2013",
            "$1,200.00",
            "Visa ending in 4242",
            "Update card",
        ]:
            self.assert_in_response(substring, response)
        self.assert_not_in_success_response(
            ["Number of licenses for current billing period", "You will receive an invoice for"], response
        )

    @mock_stripe()
    def test_upgrade_by_invoice_to_plus_plan(self, *mocks: Any) -> None:
        user = self.example_user("hamlet")
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.upgrade(invoice=True, tier=CustomerPlan.TIER_CLOUD_PLUS)
        stripe_customer = stripe_get_customer(assert_is_not_none(Customer.objects.get(realm=user.realm).stripe_customer_id))
        self.assertFalse(stripe_customer_has_credit_card_as_default_payment_method(stripe_customer))
        self.assertFalse(stripe.Charge.list(customer=stripe_customer.id))
        [invoice] = iter(stripe.Invoice.list(customer=stripe_customer.id))
        self.assertIsNotNone(invoice.due_date)
        self.assertIsNotNone(invoice.status_transitions.finalized_at)
        invoice_params = {
            "amount_due": 12000 * 123,
            "amount_paid": 0,
            "attempt_count": 0,
            "auto_advance": False,
            "collection_method": "send_invoice",
            "statement_descriptor": "Zulip Cloud Plus",
            "status": "paid",
            "total": 12000 * 123,
        }
        for key, value in invoice_params.items():
            self.assertEqual(invoice.get(key), value)
        [item] = iter(invoice.lines)
        line_item_params = {
            "amount": 12000 * 123,
            "description": "Zulip Cloud Plus",
            "discountable": False,
            "plan": None,
            "proration": False,
            "quantity": 123,
            "period": {
                "start": datetime_to_timestamp(self.now),
                "end": datetime_to_timestamp(add_months(self.now, 12)),
            },
        }
        for key, value in line_item_params.items():
            self.assertEqual(item.get(key), value)
        customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
        plan = CustomerPlan.objects.get(
            customer=customer,
            automanage_licenses=False,
            charge_automatically=False,
            price_per_license=12000,
            fixed_price=None,
            discount=None,
            billing_cycle_anchor=self.now,
            billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            invoiced_through=LicenseLedger.objects.first(),
            next_invoice_date=self.next_month,
            tier=CustomerPlan.TIER_CLOUD_PLUS,
            status=CustomerPlan.ACTIVE,
        )
        LicenseLedger.objects.get(plan=plan, is_renewal=True, event_time=self.now, licenses=123, licenses_at_next_renewal=123)
        audit_log_entries = list(
            RealmAuditLog.objects.filter(acting_user=user).values_list("event_type", "event_time").order_by("id")
        )
        self.assertEqual(
            audit_log_entries[:3],
            [
                (AuditLogEventType.STRIPE_CUSTOMER_CREATED, timestamp_to_datetime(stripe_customer.created)),
                (AuditLogEventType.CUSTOMER_PLAN_CREATED, self.now),
                (AuditLogEventType.REALM_PLAN_TYPE_CHANGED, self.now),
            ],
        )
        self.assertEqual(audit_log_entries[2][0], AuditLogEventType.REALM_PLAN_TYPE_CHANGED)
        first_audit_log_entry = RealmAuditLog.objects.filter(
            event_type=AuditLogEventType.CUSTOMER_PLAN_CREATED
        ).values_list("extra_data", flat=True).first()
        assert first_audit_log_entry is not None
        self.assertFalse(first_audit_log_entry["automanage_licenses"])
        realm = get_realm("zulip")
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_PLUS)
        self.assertEqual(realm.max_invites, Realm.INVITES_STANDARD_REALM_DAILY_MAX)
        response = self.client_get("/upgrade/")
        self.assertEqual(response.status_code, 302)
        self.assertEqual("http://zulip.testserver/billing", response["Location"])
        with time_machine.travel(self.now, tick=False):
            response = self.client_get("/billing/")
        self.assert_not_in_success_response(["Pay annually"], response)
        for substring in [
            "Zulip Cloud Plus",
            str(self.seat_count),
            "Number of licenses",
            f"{self.seat_count}",
            "Your plan will automatically renew on",
            "January 2, 2013",
            f"${80 * self.seat_count:.2f}",
            "Visa ending in 4242",
            "Update card",
        ]:
            self.assert_in_response(substring, response)
        self.assert_not_in_success_response(
            ["Number of licenses for current billing period", "You will receive an invoice for"], response
        )

    @mock_stripe(tested_timestamp_fields=["created"])
    def test_upgrade_by_card_with_outdated_seat_count(self, *mocks: Any) -> None:
        hamlet = self.example_user("hamlet")
        self.login_user(hamlet)
        new_seat_count = 23
        initial_upgrade_request = InitialUpgradeRequest(
            manual_license_management=False,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
            billing_modality="charge_automatically",
        )
        billing_session = RealmBillingSession(hamlet)
        _, context_when_upgrade_page_is_rendered = billing_session.get_initial_upgrade_context(initial_upgrade_request)
        with patch(
            "corporate.lib.stripe.BillingSession.stale_seat_count_check", return_value=self.seat_count
        ), patch("corporate.lib.stripe.get_latest_seat_count", return_value=new_seat_count), patch(
            "corporate.lib.stripe.RealmBillingSession.get_initial_upgrade_context",
            return_value=(_, context_when_upgrade_page_is_rendered),
        ):
            self.add_card_and_upgrade(hamlet)
        customer = Customer.objects.first()
        assert customer is not None
        assert customer.stripe_customer_id is not None
        [charge] = iter(stripe.Charge.list(customer=customer.stripe_customer_id))
        self.assertEqual(8000 * self.seat_count, charge.amount)
        [additional_license_invoice, upgrade_invoice] = iter(stripe.Invoice.list(customer=customer.stripe_customer_id))
        self.assertEqual([8000 * self.seat_count], [item.amount for item in upgrade_invoice.lines])
        self.assertEqual([8000 * (new_seat_count - self.seat_count)], [item.amount for item in additional_license_invoice.lines])
        ledger_entry = LicenseLedger.objects.last()
        assert ledger_entry is not None
        self.assertEqual(ledger_entry.licenses, new_seat_count)
        self.assertEqual(ledger_entry.licenses_at_next_renewal, new_seat_count)

    @mock_stripe(tested_timestamp_fields=["created"])
    def test_upgrade_by_card_with_outdated_lower_seat_count(self, *mocks: Any) -> None:
        hamlet = self.example_user("hamlet")
        self.login_user(hamlet)
        new_seat_count = self.seat_count - 1
        initial_upgrade_request = InitialUpgradeRequest(
            manual_license_management=False,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
            billing_modality="charge_automatically",
        )
        billing_session = RealmBillingSession(hamlet)
        _, context_when_upgrade_page_is_rendered = billing_session.get_initial_upgrade_context(initial_upgrade_request)
        with patch(
            "corporate.lib.stripe.BillingSession.stale_seat_count_check", return_value=self.seat_count
        ), patch("corporate.lib.stripe.get_latest_seat_count", return_value=new_seat_count), patch(
            "corporate.lib.stripe.RealmBillingSession.get_initial_upgrade_context",
            return_value=(_, context_when_upgrade_page_is_rendered),
        ):
            self.add_card_and_upgrade(hamlet)
        customer = Customer.objects.first()
        assert customer is not None
        assert customer.stripe_customer_id is not None
        [charge] = iter(stripe.Charge.list(customer=customer.stripe_customer_id))
        self.assertEqual(8000 * self.seat_count, charge.amount)
        [upgrade_invoice] = iter(stripe.Invoice.list(customer=customer.stripe_customer_id))
        self.assertEqual([8000 * self.seat_count], [item.amount for item in upgrade_invoice.lines])
        ledger_entry = LicenseLedger.objects.last()
        assert ledger_entry is not None
        self.assertEqual(ledger_entry.licenses, self.seat_count)
        self.assertEqual(ledger_entry.licenses_at_next_renewal, new_seat_count)
        from django.core.mail import outbox
        self.assert_length(outbox, 1)
        for message in outbox:
            self.assert_length(message.to, 1)
            self.assertEqual(message.to[0], "sales@zulip.com")
            self.assertEqual(message.subject, "Check initial licenses invoiced for zulip")
            self.assertEqual(self.email_envelope_from(message), settings.NOREPLY_EMAIL_ADDRESS)

    @mock_stripe(tested_timestamp_fields=["created"])
    def test_upgrade_with_tampered_seat_count(self) -> None:
        hamlet = self.example_user("hamlet")
        self.login_user(hamlet)
        with self.assertRaises(AssertionError):
            response = self.upgrade(talk_to_stripe=False, salt="badsalt")
            self.assert_json_error_contains(response, "Something went wrong. Please contact")
            self.assertEqual(orjson.loads(response.content)["error_description"], "tampered seat count")

    @mock_stripe(tested_timestamp_fields=["created"])
    def test_upgrade_race_condition_during_card_upgrade(self, *mocks: Any) -> None:
        hamlet = self.example_user("hamlet")
        othello = self.example_user("othello")
        self.login_user(othello)
        othello_upgrade_page_response = self.client_get("/upgrade/")
        self.login_user(hamlet)
        self.add_card_to_customer_for_upgrade()
        [stripe_event_before_upgrade] = iter(stripe.Event.list(limit=1))
        hamlet_upgrade_page_response = self.client_get("/upgrade/")
        self.client_billing_post(
            "/billing/upgrade",
            {
                "billing_modality": "charge_automatically",
                "schedule": "annual",
                "signed_seat_count": self.get_signed_seat_count_from_response(hamlet_upgrade_page_response),
                "salt": self.get_salt_from_response(hamlet_upgrade_page_response),
                "license_management": "automatic",
            },
        )
        customer = get_customer_by_realm(get_realm("zulip"))
        assert customer is not None
        [hamlet_invoice] = iter(stripe.Invoice.list(customer=customer.stripe_customer_id))
        self.login_user(othello)
        with self.settings(CLOUD_FREE_TRIAL_DAYS=30):
            self.client_billing_post(
                "/billing/upgrade",
                {
                    "billing_modality": "charge_automatically",
                    "schedule": "annual",
                    "signed_seat_count": self.get_signed_seat_count_from_response(othello_upgrade_page_response),
                    "salt": self.get_salt_from_response(othello_upgrade_page_response),
                    "license_management": "automatic",
                },
            )
        with self.assertLogs("corporate.stripe", "WARNING"):
            self.send_stripe_webhook_events(stripe_event_before_upgrade)
        assert hamlet_invoice.id is not None
        self.assert_details_of_valid_invoice_payment_from_event_status_endpoint(
            hamlet_invoice.id,
            {
                "status": "paid",
                "event_handler": {
                    "status": "failed",
                    "error": {
                        "message": "The organization is already subscribed to a plan. Please reload the billing page.",
                        "description": "subscribing with existing subscription",
                    },
                },
            },
        )
        from django.core.mail import outbox
        self.assert_length(outbox, 1)
        for message in outbox:
            self.assert_length(message.to, 1)
            self.assertEqual(message.to[0], "sales@zulip.com")
            self.assertEqual(message.subject, "Error processing paid customer invoice")
            self.assertEqual(self.email_envelope_from(message), settings.NOREPLY_EMAIL_ADDRESS)

    def test_upgrade_race_condition_during_invoice_upgrade(self) -> None:
        hamlet = self.example_user("hamlet")
        self.login_user(hamlet)
        self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        with self.assertRaises(BillingError) as context, self.assertLogs("corporate.stripe", "WARNING") as m, time_machine.travel(
            self.now, tick=False
        ):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(
            "subscribing with existing subscription",
            context.exception.error_description,
        )
        self.assertEqual(m.output[0], "WARNING:corporate.stripe:Upgrade of <Realm: zulip 2> (with stripe_customer_id: cus_123) failed because of existing active plan.")
        self.assert_length(m.output, 1)

    @mock_stripe(tested_timestamp_fields=["created"])
    def test_check_upgrade_parameters(self, *mocks: Any) -> None:
        def check_error(
            error_message: str,
            error_description: str,
            upgrade_params: Dict[str, Any],
            del_args: Optional[List[str]] = None,
        ) -> None:
            if del_args is None:
                del_args = []
            self.add_card_to_customer_for_upgrade()
            if error_description:
                with self.assertLogs("corporate.stripe", "WARNING"):
                    response = self.upgrade(talk_to_stripe=False, del_args=del_args, **upgrade_params)
                    self.assertEqual(orjson.loads(response.content)["error_description"], error_description)
            else:
                response = self.upgrade(talk_to_stripe=False, del_args=del_args, **upgrade_params)
            self.assert_json_error_contains(response, error_message)

        hamlet = self.example_user("hamlet")
        self.login_user(hamlet)
        check_error(
            "Invalid billing_modality",
            "",
            {"billing_modality": "invalid"},
        )
        check_error(
            "Invalid schedule",
            "",
            {"schedule": "invalid"},
        )
        check_error(
            "Invalid license_management",
            "",
            {"license_management": "invalid"},
        )
        check_error(
            "You must purchase licenses for all active users in your organization (minimum 30).",
            "not enough licenses",
            {"billing_modality": "send_invoice", "licenses": -1},
        )
        check_error(
            "You must purchase licenses for all active users in your organization (minimum 30).",
            "not enough licenses",
            {"billing_modality": "send_invoice"},
        )
        check_error(
            "You must purchase licenses for all active users in your organization (minimum 30).",
            "not enough licenses",
            {"billing_modality": "send_invoice", "licenses": 25},
        )
        check_error(
            "Invoices with more than 1000 licenses can't be processed from this page",
            "too many licenses",
            {"billing_modality": "send_invoice", "licenses": 10000},
        )
        check_error(
            "You must purchase licenses for all active users in your organization (minimum 6).",
            "not enough licenses",
            {"billing_modality": "charge_automatically", "license_management": "manual"},
        )
        check_error(
            "You must purchase licenses for all active users in your organization (minimum 6).",
            "not enough licenses",
            {"billing_modality": "charge_automatically", "license_management": "manual", "licenses": 3},
        )
        check_error(
            "You cannot decrease the licenses in the current billing period.",
            "not enough licenses",
            {"billing_modality": "charge_automatically", "license_management": "manual"},
        )
        check_error(
            "You cannot decrease the licenses in the current billing period.",
            "not enough licenses",
            {"billing_modality": "charge_automatically", "license_management": "manual", "licenses": 3},
        )
        check_error(
            "Invoices with more than 1000 licenses can't be processed from this page",
            "too many licenses",
            {"billing_modality": "send_invoice", "licenses": 2000},
        )
        with patch("corporate.lib.stripe.MIN_INVOICED_LICENSES", 3):
            check_error(
                "You must purchase licenses for all active users in your organization (minimum 3).",
                "not enough licenses",
                {"billing_modality": "send_invoice", "licenses": 4},
            )
            check_error(
                "You must purchase licenses for all active users in your organization (minimum 3).",
                "not enough licenses",
                {"billing_modality": "send_invoice"},
            )
        check_max_licenses_error = lambda licenses: check_error(
            "Invoices with more than 1000 licenses can't be processed from this page",
            "too many licenses",
            {"billing_modality": "send_invoice", "licenses": licenses},
        )
        check_min_licenses_error = lambda invoice, licenses, min_licenses_in_response, upgrade_params: check_error(
            "You must purchase licenses for all active users in your organization (minimum {min_licenses_in_response}).".format(
                min_licenses_in_response=min_licenses_in_response
            ),
            "not enough licenses",
            upgrade_params,
            del_args=["licenses"] if licenses is None else [],
        )
        check_max_licenses_error(1001)
        with patch("corporate.lib.stripe.get_latest_seat_count", return_value=1005):
            check_max_licenses_error(1005)
        check_min_licenses_error(
            False,
            self.seat_count - 1,
            self.seat_count,
            {"license_management": "manual"},
        )
        check_min_licenses_error(
            False,
            None,
            self.seat_count,
            {"license_management": "manual"},
        )
        check_success = lambda invoice, licenses, upgrade_params={} : (
            check_success_error := None,
            response := self.upgrade(
                invoice=invoice, talk_to_stripe=False, del_args=["licenses"] if licenses is None else [], **upgrade_params
            ),
            self.assert_json_success(response),
        )
        check_success(False, None)
        check_success(False, self.seat_count)
        check_success(False, self.seat_count, {"license_management": "manual"})
        check_success(False, 1001, {"license_management": "manual"})
        check_success(True, self.seat_count + MIN_INVOICED_LICENSES)
        check_success(True, MAX_INVOICED_LICENSES)
        customer = Customer.objects.get_or_create(realm=hamlet.realm)[0]
        customer.exempt_from_license_number_check = True
        customer.save()
        check_success(False, self.seat_count - 1, {"license_management": "manual"})

    @mock_stripe(tested_timestamp_fields=["created"])
    def test_upgrade_license_counts(self, *mocks: Any) -> None:
        realm = get_realm("zulip")
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count + 1, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(plan.licenses(), self.seat_count + 1)
        self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count + 1)
        billing_session = RealmBillingSession(user=None, realm=realm)
        with patch("corporate.lib.stripe.get_latest_seat_count", return_value=23):
            billing_session.update_license_ledger_if_needed(self.now + timedelta(days=100))
        with patch("corporate.lib.stripe.get_latest_seat_count", return_value=20):
            billing_session.update_license_ledger_if_needed(self.now + timedelta(days=200))
        with patch("corporate.lib.stripe.get_latest_seat_count", return_value=21):
            billing_session.update_license_ledger_if_needed(self.now + timedelta(days=300))
        with patch("corporate.lib.stripe.get_latest_seat_count", return_value=22):
            billing_session.update_license_ledger_if_needed(self.now + timedelta(days=400))
        with patch("corporate.lib.stripe.get_latest_seat_count", return_value=23):
            billing_session.update_license_ledger_if_needed(self.now + timedelta(days=500))
        plan.refresh_from_db()
        self.assertEqual(plan.next_invoice_date, self.next_month)
        self.assertTrue(plan.invoice_overdue_email_sent)
        from django.core.mail import outbox

        for count in range(1, 5):
            message = outbox[-1]
            self.assert_length(message.to, 1)
            self.assertEqual(message.to[0], "sales@zulip.com")
            self.assertEqual(message.subject, "Invoice overdue for zulip due to stale data")
            self.assertIn(f'Support URL: {self.billing_session.support_url()}', message.body)
            self.assertIn(f'Internal billing notice for zulip.', message.body)
            self.assertIn("Recent invoice is overdue for payment.", message.body)
            self.assertIn(f"Last data upload: {self.now.strftime('%Y-%m-%d')}", message.body)
        invoice_plans_as_needed(self.next_month + timedelta(days=1))
        self.assert_length(outbox, 4)
        with time_machine.travel(self.next_month, tick=False):
            send_server_data_to_push_bouncer(consider_usage_statistics=False)
        invoice_plans_as_needed(self.next_month)
        plan.refresh_from_db()
        self.assertEqual(plan.next_invoice_date, add_months(self.next_month, 1))
        self.assertFalse(plan.invoice_overdue_email_sent)
        customer = Customer.objects.get(stripe_customer_id=plan.customer.stripe_customer_id)
        self.assert_length(LicenseLedger.objects.all(), 2)
        with patch("corporate.lib.stripe.get_latest_seat_count", return_value=25):
            billing_session.update_license_ledger_if_needed(add_months(self.next_month, 1))
        ledger_entries = list(
            LicenseLedger.objects.values_list("is_renewal", "event_time", "licenses", "licenses_at_next_renewal").order_by("id")
        )
        self.assertEqual(
            ledger_entries,
            [
                (True, self.now, self.seat_count + 1, self.seat_count + 1),
                (False, self.now + timedelta(days=100), self.seat_count + 1, 23),
                (False, self.now + timedelta(days=200), self.seat_count + 1, 20),
                (False, self.now + timedelta(days=300), self.seat_count + 1, 21),
                (False, self.now + timedelta(days=400), self.seat_count + 1, 22),
                (False, self.now + timedelta(days=500), self.seat_count + 1, 23),
                (True, self.next_month, 23, 23),
                (False, add_months(self.next_month, 1), 25, 25),
            ],
        )

    def test_update_or_create_stripe_customer_logic(self) -> None:
        user = self.example_user("hamlet")
        with patch("corporate.lib.stripe.BillingSession.create_stripe_customer", return_value="returned") as mocked1:
            billing_session = RealmBillingSession(user)
            returned = billing_session.update_or_create_stripe_customer()
        mocked1.assert_called_once()
        self.assertEqual(returned, "returned")
        customer = Customer.objects.create(realm=hamlet.realm, stripe_customer_id="cus_12345")
        with patch("corporate.lib.stripe.BillingSession.create_stripe_customer", return_value="returned") as mocked2:
            billing_session = RealmBillingSession(user)
            returned = billing_session.update_or_create_stripe_customer()
        mocked2.assert_called_once()
        self.assertEqual(returned, "returned")
        customer.stripe_customer_id = "cus_12345"
        customer.save(update_fields=["stripe_customer_id"])
        with patch("corporate.lib.stripe.BillingSession.replace_payment_method") as mocked3:
            billing_session = RealmBillingSession(user)
            returned_customer = billing_session.update_or_create_stripe_customer("pm_card_visa")
        mocked3.assert_called_once()
        self.assertEqual(returned_customer, customer)
        with patch("corporate.lib.stripe.BillingSession.replace_payment_method") as mocked4:
            billing_session = RealmBillingSession(user)
            returned_customer = billing_session.update_or_create_stripe_customer(None)
        mocked4.assert_not_called()
        self.assertEqual(returned_customer, customer)

    def test_get_customer_by_realm(self) -> None:
        realm = get_realm("zulip")
        self.assertEqual(get_customer_by_realm(realm), None)
        customer = Customer.objects.create(realm=realm, stripe_customer_id="cus_12345")
        self.assertEqual(get_customer_by_realm(realm), customer)

    def test_get_current_plan_by_customer(self) -> None:
        realm = get_realm("zulip")
        customer = Customer.objects.create(realm=realm, stripe_customer_id="cus_12345")
        self.assertEqual(get_current_plan_by_customer(customer), None)
        plan = CustomerPlan.objects.create(
            customer=customer,
            status=CustomerPlan.ACTIVE,
            billing_cycle_anchor=timezone_now(),
            billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
        )
        self.assertEqual(get_current_plan_by_customer(customer), plan)
        plan.status = CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE
        plan.save(update_fields=["status"])
        self.assertEqual(get_current_plan_by_customer(customer), plan)
        plan.status = CustomerPlan.ENDED
        plan.save(update_fields=["status"])
        self.assertEqual(get_current_plan_by_customer(customer), None)
        plan.status = CustomerPlan.NEVER_STARTED
        plan.save(update_fields=["status"])
        self.assertEqual(get_current_plan_by_customer(customer), None)

    def test_get_current_plan_by_realm(self) -> None:
        realm = get_realm("zulip")
        self.assertEqual(get_current_plan_by_realm(realm), None)
        customer = Customer.objects.create(realm=realm, stripe_customer_id="cus_12345")
        self.assertEqual(get_current_plan_by_realm(realm), None)
        plan = CustomerPlan.objects.create(
            customer=customer,
            status=CustomerPlan.ACTIVE,
            billing_cycle_anchor=timezone_now(),
            billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
        )
        self.assertEqual(get_current_plan_by_realm(realm), plan)

    def test_is_realm_on_free_trial(self) -> None:
        realm = get_realm("zulip")
        self.assertFalse(is_realm_on_free_trial(realm))
        customer = Customer.objects.create(realm=realm, stripe_customer_id="cus_12345")
        plan = CustomerPlan.objects.create(
            customer=customer,
            status=CustomerPlan.ACTIVE,
            billing_cycle_anchor=timezone_now(),
            billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
        )
        self.assertFalse(is_realm_on_free_trial(realm))
        plan.status = CustomerPlan.FREE_TRIAL
        plan.save(update_fields=["status"])
        self.assertTrue(is_realm_on_free_trial(realm))

    def test_deactivate_reactivate_remote_server(self) -> None:
        server_uuid = str(uuid.uuid4())
        remote_server = RemoteZulipServer.objects.create(
            uuid=server_uuid,
            api_key="magic_secret_api_key",
            hostname="demo.example.com",
            contact_email="email@example.com",
        )
        billing_session = RemoteServerBillingSession(remote_server=remote_server)
        do_deactivate_remote_server(remote_server, billing_session)
        remote_server = RemoteZulipServer.objects.get(uuid=server_uuid)
        remote_realm_audit_log = RemoteZulipServerAuditLog.objects.filter(
            event_type=AuditLogEventType.REMOTE_SERVER_DEACTIVATED
        ).last()
        assert remote_realm_audit_log is not None
        self.assertTrue(remote_server.deactivated)
        with self.assertLogs("corporate.stripe", "WARN") as warning_log:
            do_deactivate_remote_server(remote_server, billing_session)
            self.assertEqual(
                warning_log.output,
                [f"WARN:corporate.stripe:Cannot deactivate remote server with ID {remote_server.id}, server has already been deactivated."],
            )
        do_reactivate_remote_server(remote_server)
        remote_server.refresh_from_db()
        self.assertFalse(remote_server.deactivated)
        remote_realm_audit_log = RemoteZulipServerAuditLog.objects.latest("id")
        self.assertEqual(remote_realm_audit_log.event_type, AuditLogEventType.REMOTE_SERVER_REACTIVATED)
        with self.assertLogs("corporate.stripe", "WARN") as warning_log:
            do_reactivate_remote_server(remote_server)
            self.assertEqual(
                warning_log.output,
                [f"WARN:corporate.stripe:Cannot reactivate remote server with ID {remote_server.id}, server is already active."],
            )


class TestRemoteRealmBillingSession(StripeTestCase):
    def test_get_audit_log_error(self) -> None:
        user = self.example_user("hamlet")
        billing_session = RealmBillingSession(user)
        fake_audit_log: BillingSessionEventType = 0  # type: ignore
        with self.assertRaisesRegex(
            BillingSessionAuditLogEventError, "Unknown audit log event type: 0"
        ):
            billing_session.get_audit_log_event(event_type=fake_audit_log)

    def test_get_customer(self) -> None:
        user = self.example_user("hamlet")
        billing_session = RealmBillingSession(user)
        customer = billing_session.get_customer()
        self.assertEqual(customer, None)
        customer = Customer.objects.create(realm=user.realm, stripe_customer_id="cus_12345")
        self.assertEqual(billing_session.get_customer(), customer)


class TestRemoteRealmBillingSession(StripeTestCase):
    pass


class TestSupportBillingHelpers(StripeTestCase):
    @mock_stripe()
    def test_attach_discount_to_realm(self, *mocks: Any) -> None:
        support_admin = self.example_user("iago")
        user = self.example_user("hamlet")
        billing_session = RealmBillingSession(
            support_admin, realm=user.realm, support_session=True
        )
        with self.assertRaises(AssertionError):
            billing_session.attach_discount_to_customer(
                monthly_discounted_price=120, annual_discounted_price=1200
            )
        billing_session.update_or_create_customer()
        with self.assertRaises(AssertionError):
            billing_session.attach_discount_to_customer(
                monthly_discounted_price=120, annual_discounted_price=1200
            )
        billing_session.set_required_plan_tier(CustomerPlan.TIER_CLOUD_STANDARD)
        billing_session.attach_discount_to_customer(
            monthly_discounted_price=120, annual_discounted_price=1200
        )
        realm_audit_log = RealmAuditLog.objects.filter(
            event_type=AuditLogEventType.REALM_DISCOUNT_CHANGED
        ).last()
        assert realm_audit_log is not None
        expected_extra_data: Dict[str, Any] = {
            "new_annual_discounted_price": 1200,
            "new_monthly_discounted_price": 120,
            "old_annual_discounted_price": 0,
            "old_monthly_discounted_price": 0,
        }
        self.assertEqual(realm_audit_log.extra_data, expected_extra_data)
        self.login_user(user)
        self.assert_in_success_response(["85"], self.client_get("/upgrade/"))
        self.add_card_and_upgrade(user)
        customer = Customer.objects.get(stripe_customer_id=stripe_customer.id)
        assert customer is not None
        self.assertEqual(customer.monthly_discounted_price, 120)
        self.assertEqual(customer.annual_discounted_price, 1200)
        monthly_discounted_price = customer.get_discounted_price_for_plan(
            CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_MONTHLY
        )
        self.assertEqual(monthly_discounted_price, customer.monthly_discounted_price)
        annual_discounted_price = customer.get_discounted_price_for_plan(
            CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_ANNUAL
        )
        self.assertEqual(annual_discounted_price, customer.annual_discounted_price)
        monthly_discounted_price = customer.get_discounted_price_for_plan(
            CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHLY
        )
        self.assertEqual(monthly_discounted_price, None)
        annual_discounted_price = customer.get_discounted_price_for_plan(
            CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_ANNUAL
        )
        self.assertEqual(annual_discounted_price, None)
        self.assert_eq(customer.flat_discounted_months, 12)
        with time_machine.travel(self.now, tick=False):
            self.add_card_and_upgrade(user, license_management="automatic", billing_modality="charge_automatically")
        [charge, _] = iter(stripe.Charge.list(customer=customer.stripe_customer_id))
        self.assertEqual(charge.amount, 6000 * self.seat_count)
        [invoice, _] = iter(stripe.Invoice.list(customer=customer.stripe_customer_id))
        self.assertEqual([6000 * self.seat_count], [item.amount for item in invoice.lines])
        plan = CustomerPlan.objects.get(
            customer=customer,
            automanage_licenses=True,
            price_per_license=6000,
            fixed_price=None,
            discount=Decimal(25),
            billing_cycle_anchor=self.now,
            billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            invoiced_through=LicenseLedger.objects.first(),
            next_invoice_date=self.next_month,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
            status=CustomerPlan.FREE_TRIAL,
            charge_automatically=True,
        )
        LicenseLedger.objects.get(plan=plan, is_renewal=True, event_time=self.now, licenses=self.seat_count, licenses_at_next_renewal=self.seat_count)
        audit_log_entries = list(
            RealmAuditLog.objects.filter(acting_user=user).values_list("event_type", "event_time").order_by("id")
        )
        self.assertEqual(
            audit_log_entries[:4],
            [
                (AuditLogEventType.STRIPE_CUSTOMER_CREATED, timestamp_to_datetime(stripe_customer.created)),
                (AuditLogEventType.STRIPE_CARD_CHANGED, self.now),
                (AuditLogEventType.CUSTOMER_PLAN_CREATED, self.now),
                (AuditLogEventType.REALM_PLAN_TYPE_CHANGED, self.now),
            ],
        )
        self.assertEqual(audit_log_entries[3][0], AuditLogEventType.REALM_PLAN_TYPE_CHANGED)
        first_audit_log_entry = RealmAuditLog.objects.filter(
            event_type=AuditLogEventType.CUSTOMER_PLAN_CREATED
        ).values_list("extra_data", flat=True).first()
        assert first_audit_log_entry is not None
        self.assertTrue(first_audit_log_entry["automanage_licenses"])
        realm = get_realm("zulip")
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)
        response = self.client_get("/upgrade/")
        self.assertEqual(response.status_code, 302)
        self.assertEqual("http://zulip.testserver/billing", response["Location"])
        with time_machine.travel(self.now, tick=False):
            response = self.client_get("/billing/")
        self.assert_not_in_success_response(["Pay annually"], response)
        for substring in [
            "Zulip Cloud Standard",
            str(self.seat_count),
            "Number of licenses",
            f"{self.seat_count}",
            "Your plan will automatically renew on",
            "January 2, 2013",
            f"${80 * self.seat_count}.00",
            "Visa ending in 4242",
            "Update card",
        ]:
            self.assert_in_response(substring, response)

    @mock_stripe(tested_timestamp_fields=["created"])
    def test_deactivate_registration_with_push_notification_service(self, *mocks: Any) -> None:
        user = self.example_user("hamlet")
        with patch("corporate.lib.push_notifications.is_push_notification_service_enabled", return_value=True):
            self.login_user(user)
            with patch("corporate.lib.push_notifications.unregister_subscriptions") as mocked:
                billing_entity_display_name = self.billing_session.billing_entity_display_name
                confirmation_link = generate_confirmation_link_for_server_deactivation(
                    self.remote_server, 10
                )
                result = self.client_get(confirmation_link, subdomain="selfhosting")
                self.assertEqual(result.status_code, 200)
                self.assert_in_success_response(["Log in to deactivate registration for"], result)
                result = self.client_post(
                    confirmation_link,
                    {"full_name": user.full_name, "tos_consent": "true"},
                    subdomain="selfhosting",
                )
                self.assertEqual(result.status_code, 302)
                self.assertEqual(result["Location"], f"{billing_entity_display_name}/deactivate/")
                result = self.client_get(f"{billing_entity_display_name}/deactivate/", subdomain="selfhosting")
                self.assertEqual(result.status_code, 200)
                self.assert_in_success_response(["Deactivate registration for", "Deactivate registration"], result)
                result = self.client_post(
                    f"{billing_entity_display_name}/deactivate/",
                    {"confirmed": "true"},
                    subdomain="selfhosting",
                )
                self.assertEqual(result.status_code, 200)
                self.assert_in_success_response(
                    ["Registration deactivated for", "Your server's registration has been deactivated."],
                    result,
                )
                payload = {"zulip_org_id": self.remote_server.uuid, "zulip_org_key": self.remote_server.api_key}
                result = self.client_post("/serverlogin/", payload, subdomain="selfhosting")
                self.assertEqual(result.status_code, 200)
                self.assert_in_success_response(["Your server registration has been deactivated."], result)
                mocked.assert_called_once_with(self.remote_server)

    @mock_stripe()
    def test_invoice_initial_remote_realm_upgrade(self, *mocks: Any) -> None:
        self.login("hamlet")
        hamlet = self.example_user("hamlet")
        stripe_customer = self.add_card_and_upgrade(tier=CustomerPlan.TIER_SELF_HOSTED_BASIC, schedule="monthly")
        [invoice0] = iter(stripe.Invoice.list(customer=stripe_customer.id))
        [invoice_item0, invoice_item1] = iter(invoice0.lines)
        invoice_item_params = {
            "amount": -2000,
            "description": "$20.00/month new customer discount",
            "quantity": 1,
        }
        for key, value in invoice_item_params.items():
            self.assertEqual(invoice_item0[key], value)
        invoice_item_params = {
            "amount": server_user_count * 3.5 * 100,
            "description": "Zulip Basic",
            "quantity": server_user_count,
        }
        for key, value in invoice_item_params.items():
            self.assertEqual(invoice_item1[key], value)
        self.assertEqual(invoice0.total, server_user_count * 3.5 * 100 - 2000)
        self.assertEqual(invoice0.status, "paid")

    @mock_stripe()
    def test_fixed_price_plans(self, *mocks: Any) -> None:
        user = self.example_user("hamlet")
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.upgrade(invoice=True)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        plan.fixed_price = 100
        plan.price_per_license = 0
        plan.save(update_fields=["fixed_price", "price_per_license"])
        user.realm.refresh_from_db()
        billing_session = RealmBillingSession(realm=user.realm)
        billing_session.invoice_plan(plan, self.next_year)
        stripe_customer_id = plan.customer.stripe_customer_id
        assert stripe_customer_id is not None
        [invoice0, invoice1] = iter(stripe.Invoice.list(customer=stripe_customer_id))
        self.assertEqual(invoice0.collection_method, "send_invoice")
        [item] = iter(invoice0.lines)
        line_item_params = {
            "amount": 100,
            "description": "Zulip Cloud Standard - renewal",
            "quantity": 1,
            "period": {"start": datetime_to_timestamp(self.next_year), "end": datetime_to_timestamp(self.next_year + timedelta(days=365))},
        }
        for key, value in line_item_params.items():
            self.assertEqual(item.get(key), value)

    @mock_stripe()
    def test_upgrade_to_fixed_price_plus_plan(self, *mocks: Any) -> None:
        iago = self.example_user("iago")
        hamlet = self.example_user("hamlet")
        realm = get_realm("zulip")
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_SELF_HOSTED)
        self.login_user(hamlet)
        with time_machine.travel(self.now, tick=False):
            self.upgrade(invoice=True)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertIsNone(plan.end_date)
        self.assertEqual(plan.tier, CustomerPlan.TIER_SELF_HOSTED_BASIC)
        realm.refresh_from_db()
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)
        billing_session = RealmBillingSession(user=iago, realm=realm, support_session=True)
        support_request = SupportViewRequest(
            support_type=SupportType.modify_plan, plan_modification="upgrade_plan_tier", new_plan_tier=CustomerPlan.TIER_CLOUD_PLUS
        )
        success_message = billing_session.process_support_view_request(support_request)
        self.assertEqual(success_message, "zulip upgraded to Zulip Cloud Plus")
        customer = Customer.objects.get(id=plan.customer.id)
        assert customer is not None
        new_plan = CustomerPlan.objects.get(customer=customer, tier=CustomerPlan.TIER_CLOUD_PLUS)
        self.assertEqual(new_plan.status, CustomerPlan.ACTIVE)

    @mock_stripe()
    def test_downgrade_realm_and_void_open_invoices(self, *mocks: Any) -> None:
        user = self.example_user("hamlet")
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        customer = get_customer_by_realm(user.realm)
        assert customer is not None
        self.assertEqual(get_current_plan_by_realm(user.realm).status, CustomerPlan.ACTIVE)
        support_admin = self.example_user("iago")
        billing_session = RealmBillingSession(user=support_admin, realm=user.realm, support_session=True)
        invoice_plans_as_needed(self.now + timedelta(days=367))
        support_request = SupportViewRequest(
            support_type=SupportType.modify_plan,
            plan_modification="downgrade_now_void_open_invoices",
        )
        success_message = billing_session.process_support_view_request(support_request)
        self.assertEqual(success_message, "zulip downgraded and voided 1 open invoices")
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(plan.status, CustomerPlan.ENDED)

    @mock_stripe(tested_timestamp_fields=["created"])
    def test_upgrade_to_fixed_price_business_plan(self, *mocks: Any) -> None:
        user = self.example_user("hamlet")
        self.login_user(user)
        self.add_card_and_upgrade(user)
        customer = get_customer_by_realm(user.realm)
        assert customer is not None
        original_plan = get_current_plan_by_customer(customer)
        assert original_plan is not None
        self.assertEqual(original_plan.tier, CustomerPlan.TIER_CLOUD_STANDARD)
        support_admin = self.example_user("iago")
        billing_session = RealmBillingSession(user=support_admin, realm=user.realm, support_session=True)
        support_request = SupportViewRequest(
            support_type=SupportType.modify_plan,
            plan_modification="upgrade_plan_tier",
            new_plan_tier=CustomerPlan.TIER_CLOUD_PLUS,
        )
        success_message = billing_session.process_support_view_request(support_request)
        self.assertEqual(success_message, "zulip upgraded to Zulip Cloud Plus")
        customer.refresh_from_db()
        new_plan = get_current_plan_by_customer(customer)
        assert new_plan is not None
        self.assertEqual(new_plan.tier, CustomerPlan.TIER_CLOUD_PLUS)

    @mock_stripe(tested_timestamp_fields=["created"])
    def test_send_push_notification_on_reactivate_realm(self, *mocks: Any) -> None:
        # Placeholder for actual implementation
        pass

    @responses.activate
    def test_request_sponsorship(self) -> None:
        user = self.example_user("hamlet")
        self.login_user(user)
        data: Dict[str, Any] = {
            "organization_type": Realm.ORG_TYPES["opensource"]["id"],
            "website": "https://infinispan.org/",
            "description": "Infinispan is a distributed in-memory key/value data store with optional schema.",
            "expected_total_users": "10 users",
            "plan_to_use_zulip": "For communication on moon.",
            "paid_users_count": "1 user",
            "paid_users_description": "We have 1 paid user.",
        }
        response = self.client_billing_post("/billing/sponsorship", data)
        self.assert_json_success(response)
        customer = self.billing_session.get_customer()
        assert customer is not None
        sponsorship_request = ZulipSponsorshipRequest.objects.get(customer=customer)
        self.assertEqual(sponsorship_request.org_website, data["website"])
        self.assertEqual(sponsorship_request.org_description, data["description"])
        self.assertEqual(sponsorship_request.org_type, Realm.ORG_TYPES["opensource"]["id"])
        from django.core.mail import outbox

        self.assert_length(outbox, 1)
        for message in outbox:
            self.assert_length(message.to, 1)
            self.assertEqual(message.to[0], "sales@zulip.com")
            self.assertEqual(message.subject, "Sponsorship request for zulip")
            self.assertEqual(message.reply_to, ["hamlet@zulip.com"])
            self.assertEqual(self.email_envelope_from(message), settings.NOREPLY_EMAIL_ADDRESS)
            self.assertIn("Zulip sponsorship request <noreply-", self.email_display_from(message))
            self.assertIn("Requested by: King Hamlet (Member)", message.body)
            self.assertIn("Support URL: http://zulip.testserver/activity/support?q=zulip", message.body)
            self.assertIn("Website: https://infinispan.org", message.body)
            self.assertIn("Organization type: Open-source", message.body)
            self.assertIn("Description:\nInfinispan is a distributed in-memory", message.body)
        response = self.client_get(f"{self.billing_session.billing_base_url}/billing/", subdomain="selfhosting")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], f"/realm/{user.realm.uuid!s}/sponsorship/")
        result = self.client_get(f"{self.billing_session.billing_base_url}/sponsorship/", subdomain="selfhosting")
        self.assert_in_success_response(["This organization has requested sponsorship for a", "<a href=\"/plans/\">Zulip Cloud Standard</a>", "plan.<br/><a href=\"mailto:support@zulip.com\">Contact Zulip support</a> with any questions or updates."], result)
        support_request = SupportViewRequest(
            support_type=SupportType.modify_plan,
            plan_modification="approve_sponsorship",
        )
        billing_session = RemoteRealmBillingSession(remote_realm=self.remote_realm, support_staff=self.example_user("iago"))
        billing_session.process_support_view_request(support_request)
        remote_realm = self.remote_realm
        self.assertEqual(remote_realm.plan_type, RemoteRealm.PLAN_TYPE_COMMUNITY)
        CustomerPlan.objects.get(
            customer=customer,
            tier=CustomerPlan.TIER_SELF_HOSTED_COMMUNITY,
            status=CustomerPlan.ACTIVE,
            next_invoice_date=None,
            price_per_license=0,
        )
        expected_message = (
            "Your request for Zulip sponsorship has been approved! Your organization has been upgraded to the Zulip Community plan.\n\n"
            "If you could list Zulip as a sponsor on your website, we would really appreciate it!"
        )
        self.assert_length(outbox, 3)
        message = outbox[2]
        self.assert_length(message.to, 1)
        self.assertEqual(message.to[0], "hamlet@zulip.com")
        self.assertEqual(message.subject, "Community plan sponsorship approved for demo.example.com!")
        self.assertEqual(message.from_email, "noreply@testserver")
        self.assertIn(expected_message[0], message.body)
        self.assertIn(expected_message[1], message.body)
        result = self.client_get(f"{self.billing_session.billing_base_url}/sponsorship/", subdomain="selfhosting")
        self.assert_in_success_response(["Zulip is sponsoring a free", "Community"], result)

    @mock_stripe(tested_timestamp_fields=["created"])
    def test_migrate_customer_server_to_realms_and_upgrade(self, *mocks: Any) -> None:
        remote_server = RemoteZulipServer.objects.get(hostname="demo.example.com")
        remote_realm = RemoteRealm.objects.create(
            server=remote_server,
            uuid=str(uuid.uuid4()),
            uuid_owner_secret="dummy-owner-secret",
            host="dummy-hostname",
            realm_date_created=timezone_now(),
        )
        remote_realm_billing_user = RemoteRealmBillingUser.objects.create(
            remote_realm=remote_realm,
            email="admin@example.com",
            user_uuid=uuid.uuid4(),
        )
        remote_server_billing_user = RemoteServerBillingUser.objects.create(
            remote_server=remote_server,
            email="admin@example.com",
        )
        event_time = timezone_now()

        @dataclass
        class Row:
            realm: Realm
            plan_type: str
            plan: Optional[CustomerPlan]
            plan_status: str
            invoice_count: int
            email_expected_to_be_sent: bool

        rows: List[Row] = []
        realm, _, _ = self.create_realm(users_to_create=1, create_stripe_customer=False, create_plan=False)
        rows.append(Row(realm, Realm.PLAN_TYPE_SELF_HOSTED, None, None, 0, False))
        realm, _, _ = self.create_realm(users_to_create=1, create_stripe_customer=True, create_plan=False)
        rows.append(Row(realm, Realm.PLAN_TYPE_SELF_HOSTED, None, None, 0, False))
        realm, _, _ = self.create_realm(
            users_to_create=1, create_stripe_customer=True, create_plan=False, num_invoices=1
        )
        rows.append(Row(realm, Realm.PLAN_TYPE_SELF_HOSTED, None, None, 0, False))
        realm, plan, _ = self.create_realm(
            users_to_create=1, create_stripe_customer=True, create_plan=True, num_invoices=0
        )
        rows.append(Row(realm, Realm.PLAN_TYPE_STANDARD, plan, CustomerPlan.ACTIVE, 0, False))
        realm, plan, _ = self.create_realm(
            users_to_create=1, create_stripe_customer=True, create_plan=True, num_invoices=1
        )
        rows.append(Row(realm, Realm.PLAN_TYPE_STANDARD, plan, CustomerPlan.ACTIVE, 1, False))
        realm, plan, _ = self.create_realm(
            users_to_create=3, create_stripe_customer=True, create_plan=True, num_invoices=2
        )
        rows.append(Row(realm, Realm.PLAN_TYPE_LIMITED, plan, CustomerPlan.ENDED, 0, True))
        realm, plan, invoices = self.create_realm(
            users_to_create=1, create_stripe_customer=True, create_plan=True, num_invoices=2
        )
        for invoice in invoices:
            stripe.Invoice.pay(invoice.id, paid_out_of_band=True)
        rows.append(Row(realm, Realm.PLAN_TYPE_STANDARD, plan, CustomerPlan.ACTIVE, 0, False))
        realm, plan, _ = self.create_realm(
            users_to_create=20, create_stripe_customer=True, create_plan=True, num_invoices=2
        )
        rows.append(Row(realm, Realm.PLAN_TYPE_STANDARD, plan, CustomerPlan.ACTIVE, 2, False))
        remote_server = RemoteZulipServer.objects.create(
            uuid=str(uuid.uuid4()),
            api_key="magic_secret_api_key",
            hostname="demo.example.com",
            contact_email="email@example.com",
        )
        Customer.objects.create(remote_server=remote_server, stripe_customer_id="cus_xxx")
        downgrade_small_realms_behind_on_payments_as_needed()
        from django.core.mail import outbox

        for row in rows:
            row.realm.refresh_from_db()
            self.assertEqual(row.realm.plan_type, row.plan_type)
            if row.plan is not None:
                row.plan.refresh_from_db()
                self.assertEqual(row.plan.status, row.plan_status)
                customer = customer_has_last_n_invoices_open(
                    customer=row.plan.customer, n=row.invoice_count
                )
                self.assertTrue(customer)
            email_found = False
            for message in outbox:
                recipient = UserProfile.objects.get(email=message.to[0])
                if recipient.realm == row.realm:
                    self.assertIn(
                        f"Your organization, http://{row.realm.string_id}.testserver, has been downgraded",
                        outbox[0].body,
                    )
                    self.assert_length(message.to, 1)
                    self.assertTrue(recipient.is_billing_admin)
                    email_found = True
            self.assertEqual(row.email_expected_to_be_sent, email_found)

    class TestRemoteServerBillingSession(StripeTestCase):
        def test_get_audit_log_error(self) -> None:
            server_uuid = str(uuid.uuid4())
            remote_server = RemoteZulipServer.objects.create(
                uuid=server_uuid,
                api_key="magic_secret_api_key",
                hostname="demo.example.com",
                contact_email="email@example.com",
            )
            billing_session = RemoteServerBillingSession(remote_server=remote_server)
            fake_audit_log: BillingSessionEventType = 0  # type: ignore
            with self.assertRaisesRegex(
                BillingSessionAuditLogEventError, "Unknown audit log event type: 0"
            ):
                billing_session.get_audit_log_event(event_type=fake_audit_log)

        def test_get_customer(self) -> None:
            server_uuid = str(uuid.uuid4())
            remote_server = RemoteZulipServer.objects.create(
                uuid=server_uuid,
                api_key="magic_secret_api_key",
                hostname="demo.example.com",
                contact_email="email@example.com",
            )
            billing_session = RemoteServerBillingSession(remote_server=remote_server)
            customer = billing_session.get_customer()
            self.assertEqual(customer, None)
            customer = Customer.objects.create(remote_server=remote_server, stripe_customer_id="cus_12345")
            self.assertEqual(billing_session.get_customer(), customer)

    class TestSupportBillingHelpers(StripeTestCase):
        @mock_stripe()
        def test_attach_discount_to_realm(self, *mocks: Any) -> None:
            support_admin = self.example_user("iago")
            user = self.example_user("hamlet")
            billing_session = RealmBillingSession(
                support_admin, realm=user.realm, support_session=True
            )
            with self.assertRaises(AssertionError):
                billing_session.attach_discount_to_customer(
                    monthly_discounted_price=120, annual_discounted_price=1200
                )
            billing_session.update_or_create_customer()
            with self.assertRaises(AssertionError):
                billing_session.attach_discount_to_customer(
                    monthly_discounted_price=120, annual_discounted_price=1200
                )
            billing_session.set_required_plan_tier(CustomerPlan.TIER_CLOUD_STANDARD)
            billing_session.attach_discount_to_customer(
                monthly_discounted_price=120, annual_discounted_price=1200
            )
            realm_audit_log = RealmAuditLog.objects.filter(
                event_type=AuditLogEventType.REALM_DISCOUNT_CHANGED
            ).last()
            assert realm_audit_log is not None
            expected_extra_data: Dict[str, Any] = {
                "new_annual_discounted_price": 1200,
                "new_monthly_discounted_price": 120,
                "old_annual_discounted_price": 0,
                "old_monthly_discounted_price": 0,
            }
            self.assertEqual(realm_audit_log.extra_data, expected_extra_data)
            self.login_user(user)
            self.assert_in_success_response(["85"], self.client_get("/upgrade/"))
            self.add_card_and_upgrade(user)
            customer = Customer.objects.get(stripe_customer_id=stripe_customer.id)
            assert customer is not None
            self.assertEqual(customer.monthly_discounted_price, 120)
            self.assertEqual(customer.annual_discounted_price, 1200)
            monthly_discounted_price = customer.get_discounted_price_for_plan(
                CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_MONTHLY
            )
            self.assertEqual(monthly_discounted_price, customer.monthly_discounted_price)
            annual_discounted_price = customer.get_discounted_price_for_plan(
                CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_ANNUAL
            )
            self.assertEqual(annual_discounted_price, customer.annual_discounted_price)
            monthly_discounted_price = customer.get_discounted_price_for_plan(
                CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHLY
            )
            self.assertEqual(monthly_discounted_price, None)
            annual_discounted_price = customer.get_discounted_price_for_plan(
                CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_ANNUAL
            )
            self.assertEqual(annual_discounted_price, None)
            with patch("corporate.lib.stripe.MIN_INVOICED_LICENSES", return_value=3):
                check_success_error = None
                response = self.upgrade(False, None, {})
                self.assert_json_success(response)
            do_create_user("email-extra-user", "password-extra-user", get_realm("zulip"), "name-extra-user", acting_user=None)
            with self.assertRaisesRegex(SupportRequestError, "Cannot set minimum licenses; active plan already exists for zulip."):
                billing_session.process_support_view_request(
                    SupportViewRequest(
                        support_type=SupportType.update_minimum_licenses, minimum_licenses=50
                    )
                )

        def test_set_required_plan_tier(self) -> None:
            valid_plan_tier = CustomerPlan.TIER_CLOUD_STANDARD
            support_view_request = SupportViewRequest(
                support_type=SupportType.update_required_plan_tier, required_plan_tier=valid_plan_tier
            )
            support_admin = self.example_user("iago")
            user = self.example_user("hamlet")
            billing_session = RealmBillingSession(
                support_admin, realm=user.realm, support_session=True
            )
            customer = billing_session.get_customer()
            assert customer is None
            message = billing_session.process_support_view_request(support_view_request)
            self.assertEqual(message, "Required plan tier for zulip set to Zulip Cloud Standard.")
            realm_audit_log = RealmAuditLog.objects.filter(
                event_type=AuditLogEventType.CUSTOMER_PROPERTY_CHANGED
            ).last()
            assert realm_audit_log is not None
            expected_extra_data: Dict[str, Any] = {
                "old_value": None,
                "new_value": valid_plan_tier,
                "property": "required_plan_tier",
            }
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
            monthly_discounted_price = customer.get_discounted_price_for_plan(
                valid_plan_tier, CustomerPlan.BILLING_SCHEDULE_MONTHLY
            )
            self.assertEqual(monthly_discounted_price, customer.monthly_discounted_price)
            annual_discounted_price = customer.get_discounted_price_for_plan(
                valid_plan_tier, CustomerPlan.BILLING_SCHEDULE_ANNUAL
            )
            self.assertEqual(annual_discounted_price, customer.annual_discounted_price)
            monthly_discounted_price = customer.get_discounted_price_for_plan(
                CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHLY
            )
            self.assertEqual(monthly_discounted_price, None)
            annual_discounted_price = customer.get_discounted_price_for_plan(
                CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_ANNUAL
            )
            self.assertEqual(annual_discounted_price, None)
            invalid_plan_tier = CustomerPlan.TIER_SELF_HOSTED_BASE
            support_view_request = SupportViewRequest(
                support_type=SupportType.update_required_plan_tier, required_plan_tier=invalid_plan_tier
            )
            with self.assertRaisesRegex(SupportRequestError, "Invalid plan tier for zulip."):
                billing_session.process_support_view_request(support_view_request)
            support_view_request = SupportViewRequest(
                support_type=SupportType.update_required_plan_tier, required_plan_tier=0
            )
            with self.assertRaisesRegex(SupportRequestError, "Discount for zulip must be 0 before setting required plan tier to None."):
                billing_session.process_support_view_request(support_view_request)
            billing_session.attach_discount_to_customer(monthly_discounted_price=0, annual_discounted_price=0)
            message = billing_session.process_support_view_request(support_view_request)
            self.assertEqual(message, "Required plan tier for zulip set to None.")
            customer.refresh_from_db()
            self.assertIsNone(customer.required_plan_tier)
            discount_for_standard_plan = customer.get_discounted_price_for_plan(
                valid_plan_tier, CustomerPlan.BILLING_SCHEDULE_MONTHLY
            )
            self.assertEqual(discount_for_standard_plan, None)
            discount_for_plus_plan = customer.get_discounted_price_for_plan(
                CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHLY
            )
            self.assertEqual(discount_for_plus_plan, None)
            realm_audit_log = RealmAuditLog.objects.filter(
                event_type=AuditLogEventType.CUSTOMER_PROPERTY_CHANGED
            ).last()
            assert realm_audit_log is not None
            expected_extra_data = {
                "old_value": valid_plan_tier,
                "new_value": None,
                "property": "required_plan_tier",
            }
            self.assertEqual(realm_audit_log.extra_data, expected_extra_data)

    class TestRemoteBillingWriteAuditLog(ZulipTestCase):
        def test_write_audit_log(self) -> None:
            support_admin = self.example_user("iago")
            server_uuid = str(uuid.uuid4())
            remote_server = RemoteZulipServer.objects.create(
                uuid=server_uuid,
                api_key="magic_secret_api_key",
                hostname="demo.example.com",
                contact_email="email@example.com",
            )
            realm_uuid = str(uuid.uuid4())
            remote_realm = RemoteRealm.objects.create(
                server=remote_server,
                uuid=realm_uuid,
                uuid_owner_secret="dummy-owner-secret",
                host="dummy-hostname",
                realm_date_created=timezone_now(),
            )
            remote_realm_billing_user = RemoteRealmBillingUser.objects.create(
                remote_realm=remote_realm,
                email="admin@example.com",
                user_uuid=uuid.uuid4(),
            )
            remote_server_billing_user = RemoteServerBillingUser.objects.create(
                remote_server=remote_server, email="admin@example.com"
            )
            event_time = timezone_now()

            @dataclass
            class Row:
                pass

            rows: List[Row] = []
            realm, _, _ = self.create_realm(
                users_to_create=1, create_stripe_customer=False, create_plan=False
            )
            rows.append(Row(realm, Realm.PLAN_TYPE_SELF_HOSTED, None, None, 0, False))
            realm, _, _ = self.create_realm(
                users_to_create=1, create_stripe_customer=True, create_plan=False
            )
            rows.append(Row(realm, Realm.PLAN_TYPE_SELF_HOSTED, None, None, 0, False))
            realm, _, _ = self.create_realm(
                users_to_create=1, create_stripe_customer=True, create_plan=False, num_invoices=1
            )
            rows.append(Row(realm, Realm.PLAN_TYPE_SELF_HOSTED, None, None, 0, False))
            realm, plan, _ = self.create_realm(
                users_to_create=1, create_stripe_customer=True, create_plan=True
            )
            rows.append(Row(realm, Realm.PLAN_TYPE_STANDARD, plan, CustomerPlan.ACTIVE, 0, False))
            realm, plan, _ = self.create_realm(
                users_to_create=1, create_stripe_customer=True, create_plan=True, num_invoices=1
            )
            rows.append(Row(realm, Realm.PLAN_TYPE_STANDARD, plan, CustomerPlan.ACTIVE, 1, False))
            realm, plan, _ = self.create_realm(
                users_to_create=3, create_stripe_customer=True, create_plan=True, num_invoices=2
            )
            rows.append(Row(realm, Realm.PLAN_TYPE_LIMITED, plan, CustomerPlan.ENDED, 0, True))
            realm, plan, invoices = self.create_realm(
                users_to_create=1, create_stripe_customer=True, create_plan=True, num_invoices=2
            )
            for invoice in invoices:
                stripe.Invoice.pay(invoice.id, paid_out_of_band=True)
            rows.append(Row(realm, Realm.PLAN_TYPE_STANDARD, plan, CustomerPlan.ACTIVE, 0, False))
            realm, plan, _ = self.create_realm(
                users_to_create=20, create_stripe_customer=True, create_plan=True, num_invoices=2
            )
            rows.append(Row(realm, Realm.PLAN_TYPE_STANDARD, plan, CustomerPlan.ACTIVE, 2, False))
            remote_server = RemoteZulipServer.objects.create(
                uuid=str(uuid.uuid4()),
                api_key="magic_secret_api_key",
                hostname="demo.example.com",
                contact_email="email@example.com",
            )
            Customer.objects.create(remote_server=remote_server, stripe_customer_id="cus_xxx")
            downgrade_small_realms_behind_on_payments_as_needed()
            from django.core.mail import outbox

            for row in rows:
                row.realm.refresh_from_db()
                self.assertEqual(row.realm.plan_type, row.plan_type)
                if row.plan is not None:
                    row.plan.refresh_from_db()
                    self.assertEqual(row.plan.status, row.plan_status)
                    customer = customer_has_last_n_invoices_open(
                        customer=row.plan.customer, n=row.invoice_count
                    )
                    self.assertTrue(customer)
                email_found = False
                for message in outbox:
                    recipient = UserProfile.objects.get(email=message.to[0])
                    if recipient.realm == row.realm:
                        self.assertIn(
                            f"Your organization, http://{row.realm.string_id}.testserver, has been downgraded",
                            outbox[0].body,
                        )
                        self.assert_length(message.to, 1)
                        self.assertTrue(recipient.is_billing_admin)
                        email_found = True
                self.assertEqual(row.email_expected_to_be_sent, email_found)

    class TestRemoteBillingFlow(StripeTestCase, RemoteRealmBillingTestCase):
        def test_invoice_initial_remote_realm_upgrade(self) -> None:
            self.login("hamlet")
            hamlet = self.example_user("hamlet")
            with time_machine.travel(self.now, tick=False):
                self.add_mock_response()
                send_server_data_to_push_bouncer(consider_usage_statistics=False)
            self.execute_remote_billing_authentication_flow(hamlet)
            with time_machine.travel(self.now, tick=False):
                stripe_customer = self.add_card_and_upgrade(tier=CustomerPlan.TIER_SELF_HOSTED_BASIC, schedule="monthly")
            [invoice0, invoice1] = iter(stripe.Invoice.list(customer=stripe_customer.id))
            [invoice_item0, invoice_item1, invoice_item2] = iter(invoice0.lines)
            invoice_item_params = {
                "amount": 16 * 3.5 * 100,
                "description": "Zulip Basic - renewal",
                "quantity": 16,
                "period": {
                    "start": datetime_to_timestamp(self.next_month),
                    "end": datetime_to_timestamp(add_months(self.next_month, 1)),
                },
            }
            for key, value in invoice_item_params.items():
                self.assertEqual(invoice_item1[key], value)
            invoice_item_params = {
                "description": "Additional license (Jan 4, 2012 - Feb 2, 2012)",
                "quantity": 5,
                "period": {
                    "start": datetime_to_timestamp(self.now + timedelta(days=2)),
                    "end": datetime_to_timestamp(self.next_month),
                },
            }
            for key, value in invoice_item_params.items():
                self.assertEqual(invoice_item2[key], value)
            invoice_plans_as_needed(add_months(self.next_month, 1))
            self.assert_length(outbox, 1)

    class TestRemoteRealmBillingSession(StripeTestCase):
        pass

    class TestRemoteServerBillingSession(StripeTestCase):
        pass

    class TestRemoteServerBillingSessionWithPushNotificationService(
        StripeTestCase, RemoteServerBillingTestCase
    ):
        def test_send_push_notification_on_reactivate_realm(self) -> None:
            # Placeholder for actual implementation
            pass

    @activate_push_notification_service()
    class TestRemoteBillingFlowPushNotification(StripeTestCase, RemoteRealmBillingTestCase):
        pass

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
                datetime(2021, 2, 28, 1, 2, 3, tzinfo=timezone.utc),
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
            customer_with_discount = Customer.objects.create(
                realm=get_realm("lear"),
                monthly_discounted_price=600,
                annual_discounted_price=6000,
                required_plan_tier=CustomerPlan.TIER_CLOUD_STANDARD,
            )
            customer_no_discount = Customer.objects.create(realm=get_realm("zulip"))
            test_cases: List[tuple[tuple[str, str, Optional[Customer]], tuple[datetime, datetime, datetime, int]]] = [
                (
                    (CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_ANNUAL, None),
                    (anchor, month_later, year_later, 8000),
                ),
                (
                    (CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_ANNUAL, customer_with_discount),
                    (anchor, month_later, year_later, 6000),
                ),
                (
                    (CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_ANNUAL, customer_no_discount),
                    (anchor, month_later, year_later, 8000),
                ),
                (
                    (CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_ANNUAL, customer_with_discount),
                    (anchor, month_later, year_later, 12000),
                ),
                (
                    (CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_MONTHLY, None),
                    (anchor, month_later, month_later, 800),
                ),
                (
                    (
                        CustomerPlan.TIER_CLOUD_STANDARD,
                        CustomerPlan.BILLING_SCHEDULE_MONTHLY,
                        customer_with_discount,
                    ),
                    (anchor, month_later, month_later, 600),
                ),
                (
                    (
                        CustomerPlan.TIER_CLOUD_STANDARD,
                        CustomerPlan.BILLING_SCHEDULE_MONTHLY,
                        customer_no_discount,
                    ),
                    (anchor, month_later, month_later, 800),
                ),
                (
                    (CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHLY, customer_with_discount),
                    (anchor, month_later, month_later, 1200),
                ),
            ]
            with time_machine.travel(anchor, tick=False):
                for (tier, billing_schedule, customer), output in test_cases:
                    output_ = compute_plan_parameters(tier, billing_schedule, customer)
                    self.assertEqual(output_, output)

        def test_get_price_per_license(self) -> None:
            standard_discounted_customer = Customer.objects.create(
                realm=get_realm("lear"),
                monthly_discounted_price=400,
                annual_discounted_price=4000,
                required_plan_tier=CustomerPlan.TIER_CLOUD_STANDARD,
            )
            plus_discounted_customer = Customer.objects.create(
                realm=get_realm("zulip"),
                monthly_discounted_price=600,
                annual_discounted_price=6000,
                required_plan_tier=CustomerPlan.TIER_CLOUD_PLUS,
            )
            self.assertEqual(
                get_price_per_license(CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_ANNUAL),
                8000,
            )
            self.assertEqual(
                get_price_per_license(CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_MONTHLY),
                800,
            )
            self.assertEqual(
                get_price_per_license(
                    CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_MONTHLY, standard_discounted_customer
                ),
                400,
            )
            self.assertEqual(
                get_price_per_license(CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_ANNUAL),
                12000,
            )
            self.assertEqual(
                get_price_per_license(CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHly),
                1200,
            )
            self.assertEqual(
                get_price_per_license(
                    CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHLY, standard_discounted_customer
                ),
                1200,
            )
            self.assertEqual(
                get_price_per_license(
                    CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHLY, plus_discounted_customer
                ),
                600,
            )
            with self.assertRaisesRegex(
                InvalidBillingScheduleError, "Unknown billing_schedule: 1000"
            ):
                get_price_per_license(CustomerPlan.TIER_CLOUD_STANDARD, 1000)
            with self.assertRaisesRegex(InvalidTierError, "Unknown tier: 4"):
                get_price_per_license(CustomerPlan.TIER_CLOUD_ENTERPRISE, CustomerPlan.BILLING_SCHEDULE_ANNUAL)

        def test_get_plan_renewal_or_end_date(self) -> None:
            realm = get_realm("zulip")
            customer = Customer.objects.create(realm=realm, stripe_customer_id="cus_12345")
            plan = CustomerPlan.objects.create(
                customer=customer,
                status=CustomerPlan.ACTIVE,
                billing_cycle_anchor=timezone_now(),
                billing_schedule=CustomerPlan.BILLING_SCHEDULE_MONTHLY,
                tier=CustomerPlan.TIER_CLOUD_STANDARD,
            )
            renewal_date = get_plan_renewal_or_end_date(plan, plan.billing_cycle_anchor)
            self.assertEqual(renewal_date, add_months(plan.billing_cycle_anchor, 1))
            plan_end_date = add_months(plan.billing_cycle_anchor, 1) - timedelta(days=2)
            plan.end_date = plan_end_date
            plan.save(update_fields=["end_date"])
            renewal_date = get_plan_renewal_or_end_date(plan, plan.billing_cycle_anchor)
            self.assertEqual(renewal_date, plan_end_date)

        def test_update_or_create_stripe_customer_logic(self) -> None:
            user = self.example_user("hamlet")
            with patch("corporate.lib.stripe.BillingSession.create_stripe_customer", return_value="returned") as mocked1:
                billing_session = RealmBillingSession(user)
                returned = billing_session.update_or_create_stripe_customer()
            mocked1.assert_called_once()
            self.assertEqual(returned, "returned")
            customer = Customer.objects.create(realm=hamlet.realm, stripe_customer_id="cus_12345")
            with patch("corporate.lib.stripe.BillingSession.create_stripe_customer", return_value="returned") as mocked2:
                billing_session = RealmBillingSession(user)
                returned = billing_session.update_or_create_stripe_customer()
            mocked2.assert_called_once()
            self.assertEqual(returned, "returned")
            customer.stripe_customer_id = "cus_12345"
            customer.save(update_fields=["stripe_customer_id"])
            with patch("corporate.lib.stripe.BillingSession.replace_payment_method") as mocked3:
                billing_session = RealmBillingSession(user)
                returned_customer = billing_session.update_or_create_stripe_customer("pm_card_visa")
            mocked3.assert_called_once()
            self.assertEqual(returned_customer, customer)
            with patch("corporate.lib.stripe.BillingSession.replace_payment_method") as mocked4:
                billing_session = RealmBillingSession(user)
                returned_customer = billing_session.update_or_create_stripe_customer(None)
            mocked4.assert_not_called()
            self.assertEqual(returned_customer, customer)

    class TestRemoteBillingWriteAuditLog(ZulipTestCase):
        def test_write_audit_log(self) -> None:
            support_admin = self.example_user("iago")
            server_uuid = str(uuid.uuid4())
            remote_server = RemoteZulipServer.objects.create(
                uuid=server_uuid,
                api_key="magic_secret_api_key",
                hostname="demo.example.com",
                contact_email="email@example.com",
            )
            realm_uuid = str(uuid.uuid4())
            remote_realm = RemoteRealm.objects.create(
                server=remote_server,
                uuid=realm_uuid,
                uuid_owner_secret="dummy-owner-secret",
                host="dummy-hostname",
                realm_date_created=timezone_now(),
            )
            remote_realm_billing_user = RemoteRealmBillingUser.objects.create(
                remote_realm=remote_realm, email="admin@example.com", user_uuid=uuid.uuid4()
            )
            remote_server_billing_user = RemoteServerBillingUser.objects.create(
                remote_server=remote_server, email="admin@example.com"
            )
            event_time = timezone_now()

            @dataclass
            class Row:
                realm: Realm
                plan_type: str
                plan: Optional[CustomerPlan]
                plan_status: str
                invoice_count: int
                email_expected_to_be_sent: bool

            rows: List[Row] = []
            realm, _, _ = self.create_realm(
                users_to_create=1, create_stripe_customer=False, create_plan=False
            )
            rows.append(Row(realm, Realm.PLAN_TYPE_SELF_HOSTED, None, None, 0, False))
            realm, _, _ = self.create_realm(
                users_to_create=1, create_stripe_customer=True, create_plan=False
            )
            rows.append(Row(realm, Realm.PLAN_TYPE_SELF_HOSTED, None, None, 0, False))
            realm, _, _ = self.create_realm(
                users_to_create=1, create_stripe_customer=True, create_plan=False, num_invoices=1
            )
            rows.append(Row(realm, Realm.PLAN_TYPE_SELF_HOSTED, None, None, 0, False))
            realm, plan, _ = self.create_realm(
                users_to_create=1, create_stripe_customer=True, create_plan=True
            )
            rows.append(Row(realm, Realm.PLAN_TYPE_STANDARD, plan, CustomerPlan.ACTIVE, 0, False))
            realm, plan, _ = self.create_realm(
                users_to_create=1, create_stripe_customer=True, create_plan=True, num_invoices=1
            )
            rows.append(Row(realm, Realm.PLAN_TYPE_STANDARD, plan, CustomerPlan.ACTIVE, 1, False))
            realm, plan, _ = self.create_realm(
                users_to_create=3, create_stripe_customer=True, create_plan=True, num_invoices=2
            )
            rows.append(Row(realm, Realm.PLAN_TYPE_LIMITED, plan, CustomerPlan.ENDED, 0, True))
            realm, plan, invoices = self.create_realm(
                users_to_create=1, create_stripe_customer=True, create_plan=True, num_invoices=2
            )
            for invoice in invoices:
                stripe.Invoice.pay(invoice.id, paid_out_of_band=True)
            rows.append(Row(realm, Realm.PLAN_TYPE_STANDARD, plan, CustomerPlan.ACTIVE, 0, False))
            realm, plan, _ = self.create_realm(
                users_to_create=20, create_stripe_customer=True, create_plan=True, num_invoices=2
            )
            rows.append(Row(realm, Realm.PLAN_TYPE_STANDARD, plan, CustomerPlan.ACTIVE, 2, False))
            remote_server = RemoteZulipServer.objects.create(
                uuid=str(uuid.uuid4()),
                api_key="magic_secret_api_key",
                hostname="demo.example.com",
                contact_email="email@example.com",
            )
            Customer.objects.create(remote_server=remote_server, stripe_customer_id="cus_xxx")
            downgrade_small_realms_behind_on_payments_as_needed()
            from django.core.mail import outbox

            for row in rows:
                row.realm.refresh_from_db()
                self.assertEqual(row.realm.plan_type, row.plan_type)
                if row.plan is not None:
                    row.plan.refresh_from_db()
                    self.assertEqual(row.plan.status, row.plan_status)
                    customer = customer_has_last_n_invoices_open(
                        customer=row.plan.customer, n=row.invoice_count
                    )
                    self.assertTrue(customer)
                email_found = False
                for message in outbox:
                    recipient = UserProfile.objects.get(email=message.to[0])
                    if recipient.realm == row.realm:
                        self.assertIn(
                            f"Your organization, http://{row.realm.string_id}.testserver, has been downgraded",
                            outbox[0].body,
                        )
                        self.assert_length(message.to, 1)
                        self.assertTrue(recipient.is_billing_admin)
                        email_found = True
                self.assertEqual(row.email_expected_to_be_sent, email_found)

    class BillingHelpersTest(ZulipTestCase):
        def test_subscribe_realm_to_manual_license_management_plan(self) -> None:
            realm = get_realm("zulip")
            plan, ledger = self.subscribe_realm_to_manual_license_management_plan(
                realm, 50, 60, CustomerPlan.BILLING_SCHEDULE_ANNUAL
            )
            plan.refresh_from_db()
            self.assertEqual(plan.automanage_licenses, False)
            self.assertEqual(plan.billing_schedule, CustomerPlan.BILLING_SCHEDULE_ANNUAL)
            self.assertEqual(plan.tier, CustomerPlan.TIER_SELF_HOSTED_STANDARD)
            self.assertEqual(plan.licenses(), 50)
            self.assertEqual(plan.licenses_at_next_renewal, 60)
            ledger.refresh_from_db()
            self.assertEqual(ledger.plan, plan)
            self.assertEqual(ledger.licenses, 50)
            self.assertEqual(ledger.licenses_at_next_renewal, 60)
            realm.refresh_from_db()
            self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)

        def test_subscribe_realm_to_monthly_plan_on_manual_license_management(self) -> None:
            realm = get_realm("zulip")
            plan, ledger = self.subscribe_realm_to_monthly_plan_on_manual_license_management(
                realm, 20, 30
            )
            plan.refresh_from_db()
            self.assertEqual(plan.automanage_licenses, False)
            self.assertEqual(plan.billing_schedule, CustomerPlan.BILLING_SCHEDULE_MONTHLY)
            self.assertEqual(plan.tier, CustomerPlan.TIER_SELF_HOSTED_STANDARD)
            self.assertEqual(plan.licenses(), 20)
            self.assertEqual(plan.licenses_at_next_renewal, 30)
            ledger.refresh_from_db()
            self.assertEqual(ledger.plan, plan)
            self.assertEqual(ledger.licenses, 20)
            self.assertEqual(ledger.licenses_at_next_renewal, 30)
            realm.refresh_from_db()
            self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)

    class TestRealmBillingSession(StripeTestCase):
        def test_customer_has_credit_card_as_default_payment_method(self) -> None:
            # Placeholder for actual implementation
            pass

    class TestRemoteBillingFlow(StripeTestCase, RemoteRealmBillingTestCase):
        pass

    class TestRemoteServerBillingSession(StripeTestCase):
        pass

    class TestRemoteServerBillingSessionWithPushNotificationService(
        StripeTestCase, RemoteServerBillingTestCase
    ):
        pass

    class TestRemoteBillingFlowPushNotification(StripeTestCase, RemoteRealmBillingTestCase):
        pass

    class TestTestClasses(ZulipTestCase):
        def test_subscribe_realm_to_manual_license_management_plan(self) -> None:
            realm = get_realm("zulip")
            plan, ledger = self.subscribe_realm_to_manual_license_management_plan(
                realm, 50, 60, CustomerPlan.BILLING_SCHEDULE_ANNUAL
            )
            plan.refresh_from_db()
            self.assertEqual(plan.automanage_licenses, False)
            self.assertEqual(plan.billing_schedule, CustomerPlan.BILLING_SCHEDULE_ANNUAL)
            self.assertEqual(plan.tier, CustomerPlan.TIER_SELF_HOSTED_STANDARD)
            self.assertEqual(plan.licenses(), 50)
            self.assertEqual(plan.licenses_at_next_renewal, 60)
            ledger.refresh_from_db()
            self.assertEqual(ledger.plan, plan)
            self.assertEqual(ledger.licenses, 50)
            self.assertEqual(ledger.licenses_at_next_renewal, 60)
            realm.refresh_from_db()
            self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)

        def test_subscribe_realm_to_monthly_plan_on_manual_license_management(self) -> None:
            realm = get_realm("zulip")
            plan, ledger = self.subscribe_realm_to_monthly_plan_on_manual_license_management(
                realm, 20, 30
            )
            plan.refresh_from_db()
            self.assertEqual(plan.automanage_licenses, False)
            self.assertEqual(plan.billing_schedule, CustomerPlan.BILLING_SCHEDULE_MONTHLY)
            self.assertEqual(plan.tier, CustomerPlan.TIER_SELF_HOSTED_STANDARD)
            self.assertEqual(plan.licenses(), 20)
            self.assertEqual(plan.licenses_at_next_renewal, 30)
            ledger.refresh_from_db()
            self.assertEqual(ledger.plan, plan)
            self.assertEqual(ledger.licenses, 20)
            self.assertEqual(ledger.licenses_at_next_renewal, 30)
            realm.refresh_from_db()
            self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)

    class TestInvoice(StripeTestCase):
        def test_invoicing_status_is_started(self) -> None:
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
            plan = CustomerPlan.objects.get()
            self.assertEqual(plan.invoicing_status, CustomerPlan.INVOICING_STATUS_STARTED)
            with self.assertRaises(NotImplementedError):
                billing_session = RealmBillingSession(realm=get_realm("zulip"))
                billing_session.invoice_plan(assert_is_not_none(CustomerPlan.objects.first()), self.now)

        def test_invoice_plan_without_stripe_customer(self) -> None:
            realm = get_realm("zulip")
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
            plan = get_current_plan_by_realm(realm)
            assert plan is not None
            plan.customer.stripe_customer_id = None
            plan.customer.save(update_fields=["stripe_customer_id"])
            with self.assertRaisesRegex(
                BillingError, "Customer has a paid plan without a Stripe customer ID:"
            ):
                billing_session = RealmBillingSession(realm=realm)
                billing_session.invoice_plan(plan, timezone_now())

        @mock_stripe()
        def test_validate_licenses_for_manual_plan_management(self, *mocks: Any) -> None:
            user = self.example_user("hamlet")
            self.login_user(user)
            with time_machine.travel(self.now, tick=False), patch("corporate.lib.stripe.MIN_INVOICED_LICENSES", 3):
                self.upgrade(invoice=True, licenses=self.seat_count + 1)
            with time_machine.travel(self.now, tick=False), patch("corporate.lib.stripe.MIN_INVOICED_LICENSES", 3):
                response = self.client_billing_patch("/billing/plan", {"licenses_at_next_renewal": self.seat_count})
                self.assert_json_error_contains(
                    response,
                    "Your plan is already scheduled to renew with 100 licenses.",
                )
            do_create_user(
                "email-extra-user", "password-extra-user", get_realm("zulip"), "name-extra-user", acting_user=None
            )
            with self.assertRaisesRegex(
                SupportRequestError, "Customer on plan Zulip Cloud Standard. Please end current plan before approving sponsorship!"
            ):
                billing_session.process_support_view_request(
                    SupportViewRequest(
                        support_type=SupportType.modify_plan,
                        plan_modification="approve_sponsorship",
                    )
                )

        @mock_stripe(tested_timestamp_fields=["created"])
        def test_invoice_plan(self, *mocks: Any) -> None:
            user = self.example_user("hamlet")
            self.login_user(user)
            with time_machine.travel(self.now, tick=False):
                self.add_card_and_upgrade()
            realm = get_realm("zulip")
            billing_session = RealmBillingSession(user=user, realm=realm)
            with patch("corporate.lib.stripe.get_latest_seat_count", return_value=self.seat_count + 3):
                billing_session.update_license_ledger_if_needed(self.now + timedelta(days=100))
            with patch("corporate.lib.stripe.get_latest_seat_count", return_value=self.seat_count):
                billing_session.update_license_ledger_if_needed(self.now + timedelta(days=200))
            with patch("corporate.lib.stripe.get_latest_seat_count", return_value=self.seat_count + 1):
                billing_session.update_license_ledger_if_needed(self.now + timedelta(days=300))
            with patch("corporate.lib.stripe.get_latest_seat_count", return_value=self.seat_count + 2):
                billing_session.update_license_ledger_if_needed(self.now + timedelta(days=400))
            with patch("corporate.lib.stripe.get_latest_seat_count", return_value=self.seat_count + 3):
                billing_session.update_license_ledger_if_needed(self.now + timedelta(days=500))
            plan = CustomerPlan.objects.first()
            assert plan is not None
            billing_session.invoice_plan(plan, self.now + timedelta(days=400))
            stripe_customer_id = plan.customer.stripe_customer_id
            assert stripe_customer_id is not None
            [invoice0, invoice1] = iter(stripe.Invoice.list(customer=stripe_customer_id))
            [invoice_item0, invoice_item1, invoice_item2] = iter(invoice0.lines)
            line_item_params = {
                "amount": int(8000 * (1 - (400 - 366) / 365) + 0.5),
                "description": "Additional license (Feb 5, 2013 - Jan 2, 2014)",
                "quantity": 1,
            }
            for key, value in line_item_params.items():
                self.assertEqual(invoice_item0[key], value)
            line_item_params = {
                "amount": 8000 * (self.seat_count + 1),
                "description": "Zulip Business - renewal",
                "quantity": self.seat_count + 1,
            }
            for key, value in line_item_params.items():
                self.assertEqual(invoice_item1[key], value)
            line_item_params = {
                "amount": 3 * int(8000 * (366 - 100) / 366 + 0.5),
                "description": "Additional license (Apr 11, 2012 - Jan 2, 2013)",
                "discountable": False,
                "quantity": 3,
                "period": {"end": datetime_to_timestamp(self.now + timedelta(days=100)), "start": datetime_to_timestamp(self.now + timedelta(days=366))},
            }
            for key, value in line_item_params.items():
                self.assertEqual(invoice_item2[key], value)

    class TestRemoteRealmBillingFlow(StripeTestCase, RemoteRealmBillingTestCase):
        def test_upgrade_user_to_business_plan(self) -> None:
            self.login("hamlet")
            hamlet = self.example_user("hamlet")
            self.add_mock_response()
            realm_user_count = UserProfile.objects.filter(
                realm=hamlet.realm, is_bot=False, is_active=True
            ).count()
            self.assertEqual(realm_user_count, 11)
            with time_machine.travel(self.now, tick=False):
                send_server_data_to_push_bouncer(consider_usage_statistics=False)
            result = self.execute_remote_billing_authentication_flow(hamlet)
            self.assertEqual(result.status_code, 302)
            self.assertEqual(result["Location"], f"{self.billing_session.billing_base_url}/plans/")
            with time_machine.travel(self.now, tick=False):
                result = self.client_get(f"{self.billing_session.billing_base_url}/upgrade/?tier={CustomerPlan.TIER_SELF_HOSTED_BASIC}", subdomain="selfhosting")
            self.assertEqual(result.status_code, 200)
            min_licenses = self.billing_session.min_licenses_for_plan(CustomerPlan.TIER_SELF_HOSTED_BASIC)
            self.assertEqual(min_licenses, 6)
            flat_discount, flat_discounted_months = self.billing_session.get_flat_discount_info()
            self.assertEqual(flat_discounted_months, 12)
            self.assert_in_success_response(
                [
                    "Start free trial",
                    "Zulip Basic",
                    "Due",
                    "on February 1, 2012",
                    f"{min_licenses}",
                    "Add card",
                    "Start 30-day free trial",
                ],
                result,
            )
            self.assertFalse(Customer.objects.exists())
            self.assertFalse(CustomerPlan.objects.exists())
            self.assertFalse(LicenseLedger.objects.exists())
            with time_machine.travel(self.now, tick=False):
                stripe_customer = self.add_card_and_upgrade(
                    tier=CustomerPlan.TIER_SELF_HOSTED_BASIC, schedule="monthly"
                )
            self.assertEqual(Invoice.objects.count(), 0)
            customer = Customer.objects.get(stripe_customer_id=stripe_customer.id)
            plan = CustomerPlan.objects.get(customer=customer)
            LicenseLedger.objects.get(plan=plan)
            with time_machine.travel(self.now + timedelta(days=1), tick=False):
                response = self.client_get(f"{self.billing_session.billing_base_url}/billing/", subdomain="selfhosting")
            for substring in [
                "Zulip Basic",
                "(free trial)",
                "Number of licenses",
                f"{realm_user_count}",
                "February 1, 2012",
                "Your plan will automatically renew on",
                f"${80 * realm_user_count - flat_discount // 100 * 1:,.2f}",
                "Visa ending in 4242",
                "Update card",
            ]:
                self.assert_in_response(substring, response)
            audit_log_count = RemoteRealmAuditLog.objects.count()
            self.assertEqual(LicenseLedger.objects.count(), 1)
            with time_machine.travel(self.now + timedelta(days=2), tick=False):
                for count in range(realm_user_count, min_licenses + 10):
                    do_create_user(
                        f"email {count}",
                        f"password {count}",
                        hamlet.realm,
                        "name",
                        role=UserProfile.ROLE_MEMBER,
                        acting_user=None,
                    )
            with time_machine.travel(self.now + timedelta(days=3), tick=False):
                send_server_data_to_push_bouncer(consider_usage_statistics=False)
            self.assertEqual(RemoteRealmAuditLog.objects.count(), min_licenses + 10 - realm_user_count + audit_log_count)
            latest_ledger = LicenseLedger.objects.last()
            assert latest_ledger is not None
            self.assertEqual(latest_ledger.licenses, min_licenses + 10)
            with time_machine.travel(self.now + timedelta(days=3), tick=False):
                response = self.client_get(f"{self.billing_session.billing_base_url}/billing/", subdomain="selfhosting")
            self.assertEqual(latest_ledger.licenses, min_licenses + 10)
            for substring in [
                "Zulip Basic",
                "Number of licenses",
                f"{latest_ledger.licenses}",
                "February 1, 2012",
                "Your plan will automatically renew on",
                f"${80 * latest_ledger.licenses - flat_discount // 100 * 1:,.2f}",
                "Visa ending in 4242",
                "Update card",
            ]:
                self.assert_in_response(substring, response)
            customer.flat_discounted_months = 0
            customer.save(update_fields=["flat_discounted_months"])
            self.assertEqual(
                self.billing_session.min_licenses_for_plan(CustomerPlan.TIER_SELF_HOSTED_BASIC), 1
            )

    class TestRemoteBillingFlowPushNotification(StripeTestCase, RemoteRealmBillingTestCase):
        pass

    @activate_push_notification_service()
    class TestRemoteBillingFlowPushNotificationWithService(
        StripeTestCase, RemoteRealmBillingTestCase
    ):
        pass
