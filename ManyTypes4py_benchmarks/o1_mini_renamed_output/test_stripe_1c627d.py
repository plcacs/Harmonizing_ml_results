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
from zerver.actions.users import (
    change_user_is_active,
    do_change_user_role,
    do_deactivate_user,
)
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

CallableT = TypeVar('CallableT', bound=Callable[..., Any])
ParamT = ParamSpec('ParamT')
ReturnT = TypeVar('ReturnT')
STRIPE_FIXTURES_DIR = 'corporate/tests/stripe_fixtures'


def stripe_fixture_path(
    decorated_function_name: str, mocked_function_name: str, call_count: int
) -> str:
    decorated_function_name = decorated_function_name.removeprefix('test_')
    mocked_function_name = mocked_function_name.removeprefix('stripe.')
    return f'{STRIPE_FIXTURES_DIR}/{decorated_function_name}--{mocked_function_name}.{call_count}.json'


def fixture_files_for_function(decorated_function: Callable[..., Any]) -> Sequence[str]:
    decorated_function_name = decorated_function.__name__
    decorated_function_name = decorated_function_name.removeprefix('test_')
    return sorted(
        (
            f'{STRIPE_FIXTURES_DIR}/{f}'
            for f in os.listdir(STRIPE_FIXTURES_DIR)
            if f.startswith(decorated_function_name + '--')
        )
    )


def generate_and_save_stripe_fixture(
    decorated_function_name: str, mocked_function_name: str, mocked_function: Callable[..., Any]
) -> Callable[..., Any]:
    def _generate_and_save_stripe_fixture(*args: Any, **kwargs: Any) -> Any:
        mock_func = operator.attrgetter(mocked_function_name)(sys.modules[__name__])
        fixture_path = stripe_fixture_path(decorated_function_name, mocked_function_name, mock_func.call_count)
        try:
            with responses.RequestsMock() as request_mock:
                request_mock.add_passthru('https://api.stripe.com')
                stripe_object = mocked_function(*args, **kwargs)
        except stripe.StripeError as e:
            with open(fixture_path, 'w') as f:
                assert e.headers is not None
                error_dict = {**vars(e), 'headers': dict(e.headers)}
                if e.http_body is None:
                    assert e.json_body is not None
                    error_dict['http_body'] = json.dumps(e.json_body)
                f.write(
                    json.dumps(
                        error_dict, indent=2, separators=(',', ': '), sort_keys=True
                    ) + '\n'
                )
            raise
        with open(fixture_path, 'w') as f:
            if stripe_object is not None:
                f.write(str(stripe_object) + '\n')
            else:
                f.write('{}\n')
        return stripe_object

    return _generate_and_save_stripe_fixture


def read_stripe_fixture(decorated_function_name: str, mocked_function_name: str) -> Callable[..., Any]:
    def _read_stripe_fixture(*args: Any, **kwargs: Any) -> Any:
        mock_func = operator.attrgetter(mocked_function_name)(sys.modules[__name__])
        fixture_path = stripe_fixture_path(decorated_function_name, mocked_function_name, mock_func.call_count)
        with open(fixture_path, 'rb') as f:
            fixture = orjson.loads(f.read())
        if 'json_body' in fixture:
            requester = stripe._api_requestor._APIRequestor()
            requester._interpret_response(
                fixture['http_body'], fixture['http_status'], fixture['headers'], 'V1'
            )
        return stripe.convert_to_stripe_object(fixture)

    return _read_stripe_fixture


def delete_fixture_data(decorated_function: Callable[..., Any]) -> None:
    for fixture_file in fixture_files_for_function(decorated_function):
        os.remove(fixture_file)


def normalize_fixture_data(
    decorated_function: Callable[..., Any], tested_timestamp_fields: Optional[list[str]] = None
) -> None:
    if tested_timestamp_fields is None:
        tested_timestamp_fields = []
    id_lengths = [
        ('test', 12),
        ('cus', 14),
        ('prod', 14),
        ('req', 14),
        ('si', 14),
        ('sli', 14),
        ('sub', 14),
        ('acct', 16),
        ('card', 24),
        ('ch', 24),
        ('ii', 24),
        ('il', 24),
        ('in', 24),
        ('pi', 24),
        ('price', 24),
        ('src', 24),
        ('src_client_secret', 24),
        ('tok', 24),
        ('txn', 24),
        ('invst', 26),
        ('rcpt', 31),
        ('seti', 24),
        ('pm', 24),
        ('setatt', 24),
        ('bpc', 24),
        ('bps', 24),
    ]
    pattern_translations: dict[str, str] = {
        '"exp_month": ([0-9]+)': '1',
        '"exp_year": ([0-9]+)': '9999',
        '"postal_code": "([0-9]+)"': '12345',
        '"invoice_prefix": "([A-Za-z0-9]{7,8})"': 'NORMALIZED',
        '"fingerprint": "([A-Za-z0-9]{16})"': 'NORMALIZED',
        '"number": "([A-Za-z0-9]{7,8}-[A-Za-z0-9]{4})"': 'NORMALIZED',
        '"address": "([A-Za-z0-9]{9}-test_[A-Za-z0-9]{12})"': '000000000-test_NORMALIZED',
        '"client_secret": "([\\w]+)"': 'NORMALIZED',
        '"url": "https://billing.stripe.com/p/session/test_([\\w]+)"': 'NORMALIZED',
        '"url": "https://checkout.stripe.com/c/pay/cs_test_([\\w#%]+)"': 'NORMALIZED',
        '"receipt_url": "https://pay.stripe.com/receipts/invoices/([\\w-]+)\\?s=[\\w]+"': 'NORMALIZED',
        '"hosted_invoice_url": "https://invoice.stripe.com/i/acct_[\\w]+/test_[\\w,]+\\?s=[\\w]+"': '"hosted_invoice_url": "https://invoice.stripe.com/i/acct_NORMALIZED/test_NORMALIZED?s=ap"',
        '"invoice_pdf": "https://pay.stripe.com/invoice/acct_[\\w]+/test_[\\w,]+/pdf\\?s=[\\w]+"': '"invoice_pdf": "https://pay.stripe.com/invoice/acct_NORMALIZED/test_NORMALIZED/pdf?s=ap"',
        '"id": "([\\w]+)"': 'FILE_NAME',
        '"realm_id": "[0-9]+"': '"realm_id": "1"',
        '"account_name": "[\\w\\s]+"': '"account_name": "NORMALIZED"',
    }
    pattern_translations.update({
        f'{prefix}_[A-Za-z0-9]{{{length}}}': f'{prefix}_NORMALIZED'
        for prefix, length in id_lengths
    })
    for i, timestamp_field in enumerate(tested_timestamp_fields):
        pattern_translations[f'"{timestamp_field}": 1[5-9][0-9]{{8}}(?![0-9-])'] = f'"{timestamp_field}": {1000000000 + i}'

    normalized_values: dict[str, dict[str, str]] = {
        pattern: {} for pattern in pattern_translations
    }

    for fixture_file in fixture_files_for_function(decorated_function):
        with open(fixture_file) as f:
            file_content = f.read()
        for pattern, translation in pattern_translations.items():
            for match in re.findall(pattern, file_content):
                if match not in normalized_values[pattern]:
                    if pattern.startswith('"id": "'):
                        normalized_values[pattern][match] = fixture_file.split('/')[-1]
                    else:
                        normalized_values[pattern][match] = translation
                file_content = file_content.replace(match, normalized_values[pattern][match])
        file_content = re.sub('(?<="risk_score": )(\\d+)', '0', file_content)
        file_content = re.sub('(?<="times_redeemed": )(\\d+)', '0', file_content)
        file_content = re.sub(
            '(?<="idempotency_key": )"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"',
            '"00000000-0000-0000-0000-000000000000"',
            file_content,
        )
        file_content = re.sub('(?<="Date": )"(.* GMT)"', '"NORMALIZED DATETIME"', file_content)
        file_content = re.sub('[0-3]\\d [A-Z][a-z]{2} 20[1-2]\\d', 'NORMALIZED DATE', file_content)
        file_content = re.sub('"\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}"', '"0.0.0.0"', file_content)
        file_content = re.sub(': (1[5-9][0-9]{8})(?![0-9-])', ': 1000000000', file_content)
        with open(fixture_file, 'w') as f:
            f.write(file_content)


MOCKED_STRIPE_FUNCTION_NAMES: list[str] = [
    f'stripe.{name}'
    for name in [
        'billing_portal.Configuration.create',
        'billing_portal.Session.create',
        'checkout.Session.create',
        'checkout.Session.list',
        'Charge.create',
        'Charge.list',
        'Coupon.create',
        'Customer.create',
        'Customer.create_balance_transaction',
        'Customer.list_balance_transactions',
        'Customer.retrieve',
        'Customer.save',
        'Customer.list',
        'Customer.modify',
        'Event.list',
        'Invoice.create',
        'Invoice.finalize_invoice',
        'Invoice.list',
        'Invoice.pay',
        'Invoice.refresh',
        'Invoice.retrieve',
        'Invoice.upcoming',
        'Invoice.void_invoice',
        'InvoiceItem.create',
        'InvoiceItem.list',
        'PaymentMethod.attach',
        'PaymentMethod.create',
        'PaymentMethod.detach',
        'PaymentMethod.list',
        'Plan.create',
        'Product.create',
        'SetupIntent.create',
        'SetupIntent.list',
        'SetupIntent.retrieve',
        'Subscription.create',
        'Subscription.delete',
        'Subscription.retrieve',
        'Subscription.save',
        'Token.create',
    ]
]


def mock_stripe(
    tested_timestamp_fields: Optional[list[str]] = None,
    generate: bool = settings.GENERATE_STRIPE_FIXTURES,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _mock_stripe(decorated_function: CallableT) -> CallableT:
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
                mocked_function_name, side_effect=side_effect, autospec=mocked_function_name.endswith('.refresh')
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

        return cast(CallableT, wrapped)

    return _mock_stripe


class StripeTestCase(ZulipTestCase):

    @override
    def setUp(self) -> None:
        super().setUp()
        realm = get_realm('zulip')
        active_emails = [
            self.example_email('AARON'),
            self.example_email('cordelia'),
            self.example_email('hamlet'),
            self.example_email('iago'),
            self.example_email('othello'),
            self.example_email('desdemona'),
            self.example_email('polonius'),
            self.example_email('default_bot'),
        ]
        for user_profile in UserProfile.objects.filter(realm_id=realm.id).exclude(
            delivery_email__in=active_emails
        ):
            do_deactivate_user(user_profile, acting_user=None)
        self.assertEqual(UserProfile.objects.filter(realm=realm, is_active=True).count(), 8)
        self.assertEqual(UserProfile.objects.exclude(realm=realm).filter(is_active=True).count(), 10)
        self.assertEqual(get_latest_seat_count(realm), 6)
        self.seat_count: int = 6
        self.signed_seat_count, self.salt = sign_string(str(self.seat_count))
        self.now: datetime = datetime(2012, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        self.next_month: datetime = datetime(2012, 2, 2, 3, 4, 5, tzinfo=timezone.utc)
        self.next_year: datetime = datetime(2013, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        hamlet = self.example_user('hamlet')
        hamlet.is_billing_admin = True
        hamlet.save(update_fields=['is_billing_admin'])
        self.billing_session = RealmBillingSession(user=hamlet, realm=realm)

    def get_signed_seat_count_from_response(self, response: Any) -> Optional[str]:
        match = re.search('name=\\"signed_seat_count\\" value=\\"(.+)\\"', response.content.decode())
        return match.group(1) if match else None

    def get_salt_from_response(self, response: Any) -> Optional[str]:
        match = re.search('name=\\"salt\\" value=\\"(\\w+)\\"', response.content.decode())
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
                if card_provider == 'visa':
                    return 'tok_visa'
                if card_provider == 'mastercard':
                    return 'tok_mastercard'
                raise AssertionError('Unreachable code path')
            else:
                return 'tok_chargeCustomerFail'
        else:
            return 'tok_visa_chargeDeclined'

    def assert_details_of_valid_session_from_event_status_endpoint(
        self, stripe_session_id: str, expected_details: dict[str, Any]
    ) -> None:
        json_response = self.client_billing_get('/billing/event/status', {'stripe_session_id': stripe_session_id})
        response_dict = self.assert_json_success(json_response)
        self.assertEqual(response_dict['session'], expected_details)

    def assert_details_of_valid_invoice_payment_from_event_status_endpoint(
        self, stripe_invoice_id: str, expected_details: dict[str, Any]
    ) -> None:
        json_response = self.client_billing_get('/billing/event/status', {'stripe_invoice_id': stripe_invoice_id})
        response_dict = self.assert_json_success(json_response)
        self.assertEqual(response_dict['stripe_invoice'], expected_details)

    def trigger_stripe_checkout_session_completed_webhook(self, token: str) -> None:
        customer = self.billing_session.get_customer()
        assert customer is not None
        customer_stripe_id = customer.stripe_customer_id
        assert customer_stripe_id is not None
        [checkout_setup_intent] = iter(stripe.SetupIntent.list(customer=customer_stripe_id, limit=1))
        payment_method = stripe.PaymentMethod.create(
            type='card',
            card={'token': token},
            billing_details={
                'name': 'John Doe',
                'address': {
                    'line1': '123 Main St',
                    'city': 'San Francisco',
                    'state': 'CA',
                    'postal_code': '94105',
                    'country': 'US',
                },
            },
        )
        assert isinstance(checkout_setup_intent.customer, str)
        assert checkout_setup_intent.metadata is not None
        assert checkout_setup_intent.usage in {'off_session', 'on_session'}
        usage: Literal['off_session', 'on_session'] = cast(
            Literal['off_session', 'on_session'], checkout_setup_intent.usage
        )
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
        stripe_session_dict['setup_intent'] = stripe_setup_intent.id
        event_payload = {
            'id': f'evt_{get_random_string(24)}',
            'object': 'event',
            'data': {'object': stripe_session_dict},
            'type': 'checkout.session.completed',
            'api_version': STRIPE_API_VERSION,
        }
        response = self.client_post('/stripe/webhook/', event_payload, content_type='application/json')
        assert response.status_code == 200

    def send_stripe_webhook_event(self, event: Any) -> None:
        response = self.client_post('/stripe/webhook/', orjson.loads(orjson.dumps(event)), content_type='application/json')
        assert response.status_code == 200

    def send_stripe_webhook_events(self, most_recent_event: Any) -> None:
        while True:
            events_old_to_new = list(reversed(stripe.Event.list(ending_before=most_recent_event.id)))
            if len(events_old_to_new) == 0:
                break
            for event in events_old_to_new:
                self.send_stripe_webhook_event(event)
            most_recent_event = events_old_to_new[-1]

    def add_card_to_customer_for_upgrade(
        self, charge_succeeds: bool = True
    ) -> None:
        start_session_json_response = self.client_billing_post('/upgrade/session/start_card_update_session', {'tier': 1})
        response_dict = self.assert_json_success(start_session_json_response)
        stripe_session_id = response_dict['stripe_session_id']
        self.assert_details_of_valid_session_from_event_status_endpoint(
            stripe_session_id,
            {
                'type': 'card_update_from_upgrade_page',
                'status': 'created',
                'is_manual_license_management_upgrade_session': False,
                'tier': 1,
            },
        )
        self.trigger_stripe_checkout_session_completed_webhook(
            self.get_test_card_token(attaches_to_customer=True, charge_succeeds=charge_succeeds, card_provider='visa')
        )
        self.assert_details_of_valid_session_from_event_status_endpoint(
            stripe_session_id,
            {
                'type': 'card_update_from_upgrade_page',
                'status': 'completed',
                'is_manual_license_management_upgrade_session': False,
                'tier': 1,
                'event_handler': {'status': 'succeeded'},
            },
        )

    def upgrade(
        self,
        invoice: bool = False,
        talk_to_stripe: bool = True,
        upgrade_page_response: Optional[Any] = None,
        del_args: list[str] = [],
        dont_confirm_payment: bool = False,
        **kwargs: Any,
    ) -> Any:
        if upgrade_page_response is None:
            tier = kwargs.get('tier')
            upgrade_url = f'{self.billing_session.billing_base_url}/upgrade/'
            if tier:
                upgrade_url += f'?tier={tier}'
            if self.billing_session.billing_base_url:
                upgrade_page_response = self.client_get(upgrade_url, {}, subdomain='selfhosting')
            else:
                upgrade_page_response = self.client_get(upgrade_url, {})
        params: dict[str, Any] = {
            'schedule': 'annual',
            'signed_seat_count': self.get_signed_seat_count_from_response(upgrade_page_response),
            'salt': self.get_salt_from_response(upgrade_page_response),
        }
        if invoice:
            params.update({'billing_modality': 'send_invoice', 'licenses': kwargs.get('licenses', 123)})
        else:
            params.update({'billing_modality': 'charge_automatically', 'license_management': 'automatic'})
        remote_server_plan_start_date = kwargs.get('remote_server_plan_start_date')
        if remote_server_plan_start_date:
            params.update({'remote_server_plan_start_date': remote_server_plan_start_date})
        params.update(kwargs)
        for key in del_args:
            if key in params:
                del params[key]
        if talk_to_stripe:
            [last_event] = iter(stripe.Event.list(limit=1))
        existing_customer = self.billing_session.customer_plan_exists()
        upgrade_json_response = self.client_billing_post('/billing/upgrade', params)
        if upgrade_json_response.status_code != 200 or dont_confirm_payment:
            return upgrade_json_response
        is_self_hosted_billing = not isinstance(self.billing_session, RealmBillingSession)
        customer = self.billing_session.get_customer()
        assert customer is not None
        if not talk_to_stripe or (is_free_trial_offer_enabled(is_self_hosted_billing) and (not existing_customer)):
            return upgrade_json_response
        last_sent_invoice = Invoice.objects.last()
        assert last_sent_invoice is not None
        response_dict = self.assert_json_success(upgrade_json_response)
        self.assertEqual(response_dict['stripe_invoice_id'], last_sent_invoice.stripe_invoice_id)
        self.assert_details_of_valid_invoice_payment_from_event_status_endpoint(
            last_sent_invoice.stripe_invoice_id, {'status': 'sent'}
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
            stripe_customer = stripe_get_customer(
                assert_is_not_none(Customer.objects.get(realm=user.realm).stripe_customer_id)
            )
        else:
            customer = self.billing_session.get_customer()
            assert customer is not None
            stripe_customer = stripe_get_customer(assert_is_not_none(customer.stripe_customer_id))
        self.assertTrue(stripe_customer_has_credit_card_as_default_payment_method(stripe_customer))
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
        *args: Any,
    ) -> None:

        class StripeMock(Mock):

            def __init__(self, depth: int = 1) -> None:
                super().__init__(spec=stripe.Card)
                self.id: str = 'cus_123'
                self.created: str = '1000'
                self.last4: str = '4242'

        def upgrade_func(
            licenses: int,
            automanage_licenses: bool,
            billing_schedule: str,
            charge_automatically: bool,
            free_trial: bool,
            stripe_invoice_paid: bool,
            *mock_args: Any,
        ) -> Any:
            hamlet = self.example_user('hamlet')
            billing_session = RealmBillingSession(hamlet)
            return billing_session.process_initial_upgrade(
                CustomerPlan.TIER_CLOUD_STANDARD,
                licenses,
                automanage_licenses,
                billing_schedule,
                charge_automatically,
                free_trial,
                stripe_invoice_paid=stripe_invoice_paid,
            )

        for mocked_function_name in MOCKED_STRIPE_FUNCTION_NAMES:
            upgrade_func_mocked = patch(mocked_function_name, return_value=StripeMock())(upgrade_func)
        upgrade_func_mocked(licenses, automanage_licenses, billing_schedule, charge_automatically, free_trial, stripe_invoice_paid)

    def setup_mocked_stripe(
        self, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> dict[str, MagicMock]:
        with patch.multiple(
            'stripe',
            Invoice=mock.DEFAULT,
            InvoiceItem=mock.DEFAULT,
        ) as mocked:
            mocked['Invoice'].create.return_value = None
            mocked['Invoice'].finalize_invoice.return_value = None
            mocked['InvoiceItem'].create.return_value = None
            callback(*args, **kwargs)
            return mocked

    def client_billing_get(self, url_suffix: str, info: dict[str, Any] = {}) -> Any:
        url = f'/json{self.billing_session.billing_base_url}' + url_suffix
        if self.billing_session.billing_base_url:
            response = self.client_get(url, info, subdomain='selfhosting')
        else:
            response = self.client_get(url, info)
        return response

    def client_billing_post(self, url_suffix: str, info: dict[str, Any] = {}) -> Any:
        url = f'/json{self.billing_session.billing_base_url}' + url_suffix
        if self.billing_session.billing_base_url:
            response = self.client_post(url, info, subdomain='selfhosting')
        else:
            response = self.client_post(url, info)
        return response

    def client_billing_patch(self, url_suffix: str, info: dict[str, Any] = {}) -> Any:
        url = f'/json{self.billing_session.billing_base_url}' + url_suffix
        if self.billing_session.billing_base_url:
            response = self.client_patch(url, info, subdomain='selfhosting')
        else:
            response = self.client_patch(url, info)
        return response


class StripeTest(StripeTestCase):

    def test_catch_stripe_errors(self) -> None:

        @catch_stripe_errors
        def raise_invalid_request_error() -> None:
            raise stripe.InvalidRequestError('message', 'param', 'code', json_body={})

        with self.assertLogs('corporate.stripe', 'ERROR') as error_log:
            with self.assertRaises(BillingError) as billing_context:
                raise_invalid_request_error()
            self.assertEqual('other stripe error', billing_context.exception.error_description)
            self.assertEqual(
                error_log.output,
                ['ERROR:corporate.stripe:Stripe error: None None None None'],
            )

        @catch_stripe_errors
        def raise_card_error() -> None:
            error_message = 'The card number is not a valid credit card number.'
            json_body = {'error': {'message': error_message}}
            raise stripe.CardError(error_message, 'number', 'invalid_number', json_body=json_body)

        with self.assertLogs('corporate.stripe', 'INFO') as info_log:
            with self.assertRaises(StripeCardError) as card_context:
                raise_card_error()
            self.assertIn('not a valid credit card', str(card_context.exception))
            self.assertEqual('card error', card_context.exception.error_description)
            self.assertEqual(
                info_log.output,
                ['INFO:corporate.stripe:Stripe card error: None None None None'],
            )

    def test_billing_not_enabled(self) -> None:
        iago = self.example_user('iago')
        with self.settings(BILLING_ENABLED=False):
            self.login_user(iago)
            response = self.client_get('/upgrade/', follow=True)
            self.assertEqual(response.status_code, 404)

    @mock_stripe()
    def test_stripe_billing_portal_urls(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        self.add_card_to_customer_for_upgrade()
        response = self.client_get(f'/customer_portal/?tier={CustomerPlan.TIER_CLOUD_STANDARD}')
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response['Location'].startswith('https://billing.stripe.com'))
        self.upgrade(invoice=True)
        response = self.client_get('/customer_portal/?return_to_billing_page=true')
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response['Location'].startswith('https://billing.stripe.com'))
        response = self.client_get('/invoices/')
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response['Location'].startswith('https://billing.stripe.com'))

    @mock_stripe()
    def test_upgrade_by_card_to_plus_plan(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        response = self.client_get('/upgrade/?tier=2')
        self.assert_in_success_response(
            [
                'Your subscription will renew automatically',
                'Zulip Cloud Plus',
            ],
            response,
        )
        self.assertEqual(user.realm.plan_type, Realm.PLAN_TYPE_SELF_HOSTED)
        self.assertFalse(Customer.objects.filter(realm=user.realm).exists())
        stripe_customer = self.add_card_and_upgrade(user, tier=CustomerPlan.TIER_CLOUD_PLUS)
        self.assertEqual(stripe_customer.description, 'zulip (Zulip Dev)')
        self.assertEqual(stripe_customer.discount, None)
        self.assertEqual(stripe_customer.email, user.delivery_email)
        assert stripe_customer.metadata is not None
        metadata_dict = dict(stripe_customer.metadata)
        self.assertEqual(metadata_dict['realm_str'], 'zulip')
        try:
            int(metadata_dict['realm_id'])
        except ValueError:
            raise AssertionError('realm_id is not a number')
        [charge] = iter(stripe.Charge.list(customer=stripe_customer.id))
        licenses_purchased = self.billing_session.min_licenses_for_plan(CustomerPlan.TIER_CLOUD_PLUS)
        self.assertEqual(charge.amount, 12000 * licenses_purchased)
        self.assertEqual(charge.description, 'Payment for Invoice')
        self.assertEqual(charge.receipt_email, user.delivery_email)
        self.assertEqual(charge.statement_descriptor, 'Zulip Cloud Plus')
        [invoice] = iter(stripe.Invoice.list(customer=stripe_customer.id))
        self.assertIsNotNone(invoice.status_transitions.finalized_at)
        invoice_params = {
            'amount_due': 120000,
            'amount_paid': 120000,
            'auto_advance': False,
            'collection_method': 'charge_automatically',
            'status': 'paid',
            'total': 120000,
        }
        self.assertIsNotNone(invoice.charge)
        for key, value in invoice_params.items():
            self.assertEqual(invoice.get(key), value)
        [item0] = iter(invoice.lines)
        line_item_params = {
            'amount': 12000 * licenses_purchased,
            'description': 'Zulip Cloud Plus',
            'discountable': False,
            'plan': None,
            'proration': False,
            'quantity': licenses_purchased,
            'period': {
                'start': datetime_to_timestamp(self.now),
                'end': datetime_to_timestamp(add_months(self.now, 12)),
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
            RealmAuditLog.objects.filter(acting_user=user).values_list('event_type', 'event_time').order_by('id')
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
        ).values_list('extra_data', flat=True).first()
        assert first_audit_log_entry is not None
        self.assertTrue(first_audit_log_entry['automanage_licenses'])
        realm = get_realm('zulip')
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_PLUS)
        self.assertEqual(realm.max_invites, Realm.INVITES_STANDARD_REALM_DAILY_MAX)
        response = self.client_get('/upgrade/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('http://zulip.testserver/billing', response['Location'])
        with time_machine.travel(self.now, tick=False):
            response = self.client_get('/billing/')
        self.assert_not_in_success_response(['Pay annually'], response)
        for substring in [
            'Zulip Cloud Plus',
            str(licenses_purchased),
            'Number of licenses',
            f'{licenses_purchased}',
            'Your plan will automatically renew on',
            'January 2, 2013',
            '$1,200.00',
            'Visa ending in 4242',
            'Update card',
        ]:
            self.assert_in_response(substring, response)
        self.assert_not_in_success_response(['Number of licenses for current billing period', 'You will receive an invoice for'], response)

    @mock_stripe()
    def test_upgrade_by_invoice_to_plus_plan(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
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
            'amount_due': 8000 * 123,
            'amount_paid': 0,
            'attempt_count': 0,
            'auto_advance': False,
            'collection_method': 'send_invoice',
            'statement_descriptor': 'Zulip Cloud Plus',
            'status': 'paid',
            'total': 8000 * 123,
        }
        for key, value in invoice_params.items():
            self.assertEqual(invoice.get(key), value)
        [item] = iter(invoice.lines)
        line_item_params = {
            'amount': 8000 * 123,
            'description': 'Zulip Cloud Plus',
            'discountable': False,
            'plan': None,
            'proration': False,
            'quantity': 123,
            'period': {
                'start': datetime_to_timestamp(self.now),
                'end': datetime_to_timestamp(add_months(self.now, 12)),
            },
        }
        for key, value in line_item_params.items():
            self.assertEqual(item.get(key), value)
        customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
        plan = CustomerPlan.objects.get(
            customer=customer,
            automanage_licenses=False,
            charge_automatically=False,
            price_per_license=8000,
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
            plan=plan, is_renewal=True, event_time=self.now, licenses=123, licenses_at_next_renewal=123
        )
        audit_log_entries = list(
            RealmAuditLog.objects.filter(acting_user=user).values_list('event_type', 'event_time').order_by('id')
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
        ).values_list('extra_data', flat=True).first()
        assert first_audit_log_entry is not None
        self.assertFalse(first_audit_log_entry['automanage_licenses'])
        realm = get_realm('zulip')
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_PLUS)
        self.assertEqual(realm.max_invites, Realm.INVITES_STANDARD_REALM_DAILY_MAX)
        response = self.client_get('/upgrade/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('http://zulip.testserver/billing', response['Location'])
        with time_machine.travel(self.now, tick=False):
            response = self.client_get('/billing/')
        self.assert_not_in_success_response(['Pay annually'], response)
        for substring in [
            'Zulip Cloud Standard',
            str(123),
            'Number of licenses for current billing period',
            f'licenses ({self.seat_count} in use)',
            'You will receive an invoice for',
            'January 2, 2013',
            '$9840.00',
        ]:
            self.assert_in_response(substring, response)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_by_card_with_outdated_seat_count(self, *mocks: Any) -> None:
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        new_seat_count = 23
        initial_upgrade_request = InitialUpgradeRequest(
            manual_license_management=False,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
            billing_modality='charge_automatically',
        )
        billing_session = RealmBillingSession(hamlet)
        _, context_when_upgrade_page_is_rendered = billing_session.get_initial_upgrade_context(initial_upgrade_request)
        with patch(
            'corporate.lib.stripe.BillingSession.stale_seat_count_check',
            return_value=self.seat_count,
        ), patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=new_seat_count,
        ), patch(
            'corporate.lib.stripe.RealmBillingSession.get_initial_upgrade_context',
            return_value=(_, context_when_upgrade_page_is_rendered),
        ):
            self.add_card_and_upgrade(hamlet)
        customer = Customer.objects.first()
        assert customer is not None
        stripe_customer_id = assert_is_not_none(customer.stripe_customer_id)
        [charge] = iter(stripe.Charge.list(customer=stripe_customer_id))
        self.assertEqual(8000 * self.seat_count, charge.amount)
        [additional_license_invoice, upgrade_invoice] = iter(stripe.Invoice.list(customer=stripe_customer_id))
        self.assertEqual([8000 * self.seat_count], [item.amount for item in upgrade_invoice.lines])
        [invoice_item0] = iter(upgrade_invoice.lines)
        invoice_item_params = {
            'amount': 8000 * 10,
            'description': 'Additional license (Jan 2, 2013 - Jan 2, 2014)',
            'quantity': 10,
            'period': {
                'end': datetime_to_timestamp(add_months(self.now, 12)),
                'start': datetime_to_timestamp(self.now),
            },
        }
        for key, value in invoice_item_params.items():
            self.assertEqual(invoice_item0[key], value)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_card_attached_to_customer_but_payment_fails(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        self.add_card_to_customer_for_upgrade(charge_succeeds=False)
        with self.assertLogs('corporate.stripe', 'WARNING'):
            response = self.upgrade()
        self.assert_json_error(response, 'Your card was declined.')

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_by_invoice(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.upgrade(invoice=True)
        stripe_customer = stripe_get_customer(assert_is_not_none(Customer.objects.get(realm=user.realm).stripe_customer_id))
        self.assertFalse(stripe_customer_has_credit_card_as_default_payment_method(stripe_customer))
        self.assertFalse(stripe.Charge.list(customer=stripe_customer.id))
        [invoice] = iter(stripe.Invoice.list(customer=stripe_customer.id))
        self.assertIsNotNone(invoice.due_date)
        self.assertIsNotNone(invoice.status_transitions.finalized_at)
        invoice_params = {
            'amount_due': 8000 * 123,
            'amount_paid': 0,
            'attempt_count': 0,
            'auto_advance': False,
            'collection_method': 'send_invoice',
            'statement_descriptor': 'Zulip Cloud Standard',
            'status': 'paid',
            'total': 8000 * 123,
        }
        for key, value in invoice_params.items():
            self.assertEqual(invoice.get(key), value)
        [item] = iter(invoice.lines)
        line_item_params = {
            'amount': 8000 * 123,
            'description': 'Zulip Cloud Standard',
            'discountable': False,
            'plan': None,
            'proration': False,
            'quantity': 123,
            'period': {
                'start': datetime_to_timestamp(self.now),
                'end': datetime_to_timestamp(add_months(self.now, 12)),
            },
        }
        for key, value in line_item_params.items():
            self.assertEqual(item.get(key), value)
        customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
        plan = CustomerPlan.objects.get(
            customer=customer,
            automanage_licenses=False,
            charge_automatically=False,
            price_per_license=8000,
            fixed_price=None,
            discount=None,
            billing_cycle_anchor=self.now,
            billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            invoiced_through=LicenseLedger.objects.first(),
            next_invoice_date=self.next_month,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
            status=CustomerPlan.ACTIVE,
        )
        LicenseLedger.objects.get(
            plan=plan, is_renewal=True, event_time=self.now, licenses=123, licenses_at_next_renewal=123
        )
        audit_log_entries = list(
            RealmAuditLog.objects.filter(acting_user=user).values_list('event_type', 'event_time').order_by('id')
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
        ).values_list('extra_data', flat=True).first()
        assert first_audit_log_entry is not None
        self.assertFalse(first_audit_log_entry['automanage_licenses'])
        realm = get_realm('zulip')
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)
        response = self.client_get('/upgrade/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('http://zulip.testserver/billing', response['Location'])
        with time_machine.travel(self.now, tick=False):
            response = self.client_get('/billing/')
        self.assert_not_in_success_response(['Pay annually'], response)
        for substring in [
            'Zulip Cloud Standard',
            str(123),
            'Number of licenses for current billing period',
            f'licenses ({self.seat_count} in use)',
            'You will receive an invoice for',
            'January 2, 2013',
            '$9840.00',
        ]:
            self.assert_in_response(substring, response)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_free_trial_upgrade_by_card(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with self.settings(CLOUD_FREE_TRIAL_DAYS=60):
            response = self.client_get('/upgrade/')
            free_trial_end_date = self.now + timedelta(days=60)
            self.assert_in_success_response(
                ['Your card will not be charged', 'free trial', '60-day'],
                response,
            )
            self.assertNotEqual(user.realm.plan_type, Realm.PLAN_TYPE_STANDARD)
            self.assertFalse(Customer.objects.filter(realm=user.realm).exists())
            with time_machine.travel(self.now, tick=False), self.assertLogs('corporate.stripe', 'WARNING'):
                response = self.upgrade()
            self.assert_json_error(response, 'Please add a credit card before starting your free trial.')
            stripe_customer = self.add_card_and_upgrade(user)
            self.assertEqual(Invoice.objects.count(), 0)
            self.assertEqual(stripe_customer.description, 'zulip (Zulip Dev)')
            self.assertEqual(stripe_customer.discount, None)
            self.assertEqual(stripe_customer.email, user.delivery_email)
            assert stripe_customer.metadata is not None
            metadata_dict = dict(stripe_customer.metadata)
            self.assertEqual(metadata_dict['realm_str'], 'zulip')
            try:
                int(metadata_dict['realm_id'])
            except ValueError:
                raise AssertionError('realm_id is not a number')
            self.assertFalse(stripe.Charge.list(customer=stripe_customer.id))
            self.assertFalse(stripe.Invoice.list(customer=stripe_customer.id))
            customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
            plan = CustomerPlan.objects.get(
                customer=customer,
                automanage_licenses=True,
                price_per_license=8000,
                fixed_price=None,
                discount=None,
                billing_cycle_anchor=self.now,
                billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
                invoiced_through=LicenseLedger.objects.first(),
                next_invoice_date=free_trial_end_date,
                tier=CustomerPlan.TIER_CLOUD_STANDARD,
                status=CustomerPlan.FREE_TRIAL,
                charge_automatically=True,
            )
            LicenseLedger.objects.get(
                plan=plan, is_renewal=True, event_time=self.now, licenses=self.seat_count, licenses_at_next_renewal=self.seat_count
            )
            audit_log_entries = list(
                RealmAuditLog.objects.filter(acting_user=user).values_list('event_type', 'event_time').order_by('id')
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
            ).values_list('extra_data', flat=True).first()
            assert first_audit_log_entry is not None
            self.assertTrue(first_audit_log_entry['automanage_licenses'])
            realm = get_realm('zulip')
            self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)
            self.assertEqual(realm.max_invites, Realm.INVITES_STANDARD_REALM_DAILY_MAX)
            response = self.client_get('/upgrade/')
            self.assertEqual(response.status_code, 302)
            self.assertEqual('http://zulip.testserver/billing', response['Location'])
            with time_machine.travel(self.now, tick=False):
                response = self.client_get('/billing/')
            self.assert_not_in_success_response(['Pay annually'], response)
            for substring in [
                'Zulip Cloud Standard',
                str(self.seat_count),
                'Number of licenses for current billing period',
                f'{self.seat_count}',
                'Your plan will automatically renew on',
                'January 2, 2013',
                f'${80 * self.seat_count}.00',
                'Visa ending in 4242',
                'Update card',
            ]:
                self.assert_in_response(substring, response)
            self.assert_not_in_success_response(['Number of licenses for current billing period', 'You will receive an invoice for'], response)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_card_attached_to_customer_but_payment_fails(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        self.add_card_to_customer_for_upgrade(charge_succeeds=False)
        with self.assertLogs('corporate.stripe', 'WARNING'):
            response = self.upgrade()
        self.assert_json_error(response, 'Your card was declined.')

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_by_invoice_to_plus_plan(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
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
            'amount_due': 8000 * 123,
            'amount_paid': 0,
            'attempt_count': 0,
            'auto_advance': False,
            'collection_method': 'send_invoice',
            'statement_descriptor': 'Zulip Cloud Standard',
            'status': 'paid',
            'total': 8000 * 123,
        }
        for key, value in invoice_params.items():
            self.assertEqual(invoice.get(key), value)
        [item] = iter(invoice.lines)
        line_item_params = {
            'amount': 8000 * 123,
            'description': 'Zulip Cloud Standard',
            'discountable': False,
            'plan': None,
            'proration': False,
            'quantity': 123,
            'period': {
                'start': datetime_to_timestamp(self.now),
                'end': datetime_to_timestamp(add_months(self.now, 12)),
            },
        }
        for key, value in line_item_params.items():
            self.assertEqual(item.get(key), value)
        customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
        plan = CustomerPlan.objects.get(
            customer=customer,
            automanage_licenses=False,
            charge_automatically=False,
            price_per_license=8000,
            fixed_price=None,
            discount=None,
            billing_cycle_anchor=self.now,
            billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            invoiced_through=LicenseLedger.objects.first(),
            next_invoice_date=self.next_month,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
            status=CustomerPlan.ACTIVE,
        )
        LicenseLedger.objects.get(
            plan=plan, is_renewal=True, event_time=self.now, licenses=123, licenses_at_next_renewal=123
        )
        audit_log_entries = list(
            RealmAuditLog.objects.filter(acting_user=user).values_list('event_type', 'event_time').order_by('id')
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
        ).values_list('extra_data', flat=True).first()
        assert first_audit_log_entry is not None
        self.assertFalse(first_audit_log_entry['automanage_licenses'])
        realm = get_realm('zulip')
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_PLUS)
        self.assertEqual(realm.max_invites, Realm.INVITES_STANDARD_REALM_DAILY_MAX)
        response = self.client_get('/upgrade/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('http://zulip.testserver/billing', response['Location'])
        with time_machine.travel(self.now, tick=False):
            response = self.client_get('/billing/')
        self.assert_not_in_success_response(['Pay annually'], response)
        for substring in [
            'Zulip Cloud Standard',
            str(123),
            'Number of licenses for current billing period',
            f'licenses ({self.seat_count} in use)',
            'You will receive an invoice for',
            'January 2, 2013',
            '$9840.00',
        ]:
            self.assert_in_response(substring, response)

    class RemoteRealmBillingSessionTest(ZulipTestCase):
        pass

    class RemoteServerBillingSessionTest(ZulipTestCase):
        pass

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_by_card_to_plus_plan_with_outdated_lower_seat_count(self, *mocks: Any) -> None:
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        new_seat_count = self.seat_count - 1
        initial_upgrade_request = InitialUpgradeRequest(
            manual_license_management=False,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
            billing_modality='charge_automatically',
        )
        billing_session = RealmBillingSession(hamlet)
        _, context_when_upgrade_page_is_rendered = billing_session.get_initial_upgrade_context(initial_upgrade_request)
        with patch(
            'corporate.lib.stripe.BillingSession.stale_seat_count_check',
            return_value=self.seat_count,
        ), patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=new_seat_count,
        ), patch(
            'corporate.lib.stripe.RealmBillingSession.get_initial_upgrade_context',
            return_value=(_, context_when_upgrade_page_is_rendered),
        ):
            self.add_card_and_upgrade(hamlet)
        customer = Customer.objects.first()
        assert customer is not None
        stripe_customer_id = assert_is_not_none(customer.stripe_customer_id)
        [charge] = iter(stripe.Charge.list(customer=stripe_customer_id))
        self.assertEqual(8000 * self.seat_count, charge.amount)
        [upgrade_invoice] = iter(stripe.Invoice.list(customer=stripe_customer_id))
        self.assertEqual([8000 * self.seat_count], [item.amount for item in upgrade_invoice.lines])
        ledger_entry = LicenseLedger.objects.last()
        assert ledger_entry is not None
        self.assertEqual(ledger_entry.licenses, self.seat_count)
        self.assertEqual(ledger_entry.licenses_at_next_renewal, new_seat_count)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_by_card_with_outdated_seat_count_and_minimum_for_plan_tier(self, *mocks: Any) -> None:
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        new_seat_count = self.seat_count - 2
        minimum_for_plan_tier = self.seat_count - 1
        initial_upgrade_request = InitialUpgradeRequest(
            manual_license_management=False,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
            billing_modality='charge_automatically',
        )
        billing_session = RealmBillingSession(hamlet)
        _, context_when_upgrade_page_is_rendered = billing_session.get_initial_upgrade_context(initial_upgrade_request)
        assert context_when_upgrade_page_is_rendered is not None
        assert context_when_upgrade_page_is_rendered.get('seat_count') == self.seat_count
        with patch(
            'corporate.lib.stripe.BillingSession.min_licenses_for_plan',
            return_value=minimum_for_plan_tier,
        ), patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=new_seat_count,
        ), patch(
            'corporate.lib.stripe.RealmBillingSession.get_initial_upgrade_context',
            return_value=(_, context_when_upgrade_page_is_rendered),
        ):
            self.add_card_and_upgrade(hamlet)
        customer = Customer.objects.first()
        assert customer is not None
        stripe_customer_id = assert_is_not_none(customer.stripe_customer_id)
        [charge] = iter(stripe.Charge.list(customer=stripe_customer_id))
        self.assertEqual(8000 * self.seat_count, charge.amount)
        [upgrade_invoice] = iter(stripe.Invoice.list(customer=stripe_customer_id))
        self.assertEqual([8000 * self.seat_count], [item.amount for item in upgrade_invoice.lines])
        ledger_entry = LicenseLedger.objects.last()
        assert ledger_entry is not None
        self.assertEqual(ledger_entry.licenses, self.seat_count + 10)
        self.assertEqual(ledger_entry.licenses_at_next_renewal, new_seat_count)
        billing_session = RealmBillingSession(realm=hamlet.realm)
        billing_session.make_end_of_cycle_updates_if_needed(plan, self.next_year - timedelta(days=1))
        self.assertEqual(LicenseLedger.objects.count(), 2)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        billing_session.make_end_of_cycle_updates_if_needed(plan, self.next_year)
        self.assertEqual(LicenseLedger.objects.count(), 2)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_by_card_with_outdated_lower_seat_count_and_minimum_for_plan_tier(self, *mocks: Any) -> None:
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        new_seat_count = self.seat_count - 2
        minimum_for_plan_tier = self.seat_count - 1
        initial_upgrade_request = InitialUpgradeRequest(
            manual_license_management=False,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
            billing_modality='charge_automatically',
        )
        billing_session = RealmBillingSession(hamlet)
        _, context_when_upgrade_page_is_rendered = billing_session.get_initial_upgrade_context(initial_upgrade_request)
        assert context_when_upgrade_page_is_rendered is not None
        assert context_when_upgrade_page_is_rendered.get('seat_count') == self.seat_count
        with patch(
            'corporate.lib.stripe.BillingSession.min_licenses_for_plan',
            return_value=minimum_for_plan_tier,
        ), patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=new_seat_count,
        ), patch(
            'corporate.lib.stripe.RealmBillingSession.get_initial_upgrade_context',
            return_value=(_, context_when_upgrade_page_is_rendered),
        ):
            self.add_card_and_upgrade(hamlet)
        customer = Customer.objects.first()
        assert customer is not None
        stripe_customer_id = assert_is_not_none(customer.stripe_customer_id)
        [charge] = iter(stripe.Charge.list(customer=stripe_customer_id))
        self.assertEqual(8000 * self.seat_count, charge.amount)
        [upgrade_invoice] = iter(stripe.Invoice.list(customer=stripe_customer_id))
        self.assertEqual([8000 * self.seat_count], [item.amount for item in upgrade_invoice.lines])
        ledger_entry = LicenseLedger.objects.last()
        assert ledger_entry is not None
        self.assertEqual(ledger_entry.licenses, self.seat_count + 10)
        self.assertEqual(ledger_entry.licenses_at_next_renewal, new_seat_count)
        billing_session = RealmBillingSession(realm=hamlet.realm)
        billing_session.make_end_of_cycle_updates_if_needed(plan, self.next_year - timedelta(days=1))
        self.assertEqual(LicenseLedger.objects.count(), 2)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        billing_session.make_end_of_cycle_updates_if_needed(plan, self.next_year)
        self.assertEqual(LicenseLedger.objects.count(), 2)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_by_invoice(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.upgrade(invoice=True)
        stripe_customer = stripe_get_customer(assert_is_not_none(Customer.objects.get(realm=user.realm).stripe_customer_id))
        self.assertFalse(stripe_customer_has_credit_card_as_default_payment_method(stripe_customer))
        self.assertFalse(stripe.Charge.list(customer=stripe_customer.id))
        [invoice] = iter(stripe.Invoice.list(customer=stripe_customer.id))
        self.assertIsNotNone(invoice.due_date)
        self.assertIsNotNone(invoice.status_transitions.finalized_at)
        invoice_params = {
            'amount_due': 8000 * 123,
            'amount_paid': 0,
            'attempt_count': 0,
            'auto_advance': False,
            'collection_method': 'send_invoice',
            'statement_descriptor': 'Zulip Cloud Standard',
            'status': 'paid',
            'total': 8000 * 123,
        }
        for key, value in invoice_params.items():
            self.assertEqual(invoice.get(key), value)
        [item] = iter(invoice.lines)
        line_item_params = {
            'amount': 8000 * 123,
            'description': 'Zulip Cloud Standard',
            'discountable': False,
            'plan': None,
            'proration': False,
            'quantity': 123,
            'period': {
                'start': datetime_to_timestamp(self.now),
                'end': datetime_to_timestamp(add_months(self.now, 12)),
            },
        }
        for key, value in line_item_params.items():
            self.assertEqual(item.get(key), value)
        customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
        plan = CustomerPlan.objects.get(
            customer=customer,
            automanage_licenses=False,
            charge_automatically=False,
            price_per_license=8000,
            fixed_price=None,
            discount=None,
            billing_cycle_anchor=self.now,
            billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            invoiced_through=LicenseLedger.objects.first(),
            next_invoice_date=self.next_month,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
            status=CustomerPlan.ACTIVE,
        )
        LicenseLedger.objects.get(
            plan=plan, is_renewal=True, event_time=self.now, licenses=123, licenses_at_next_renewal=123
        )
        audit_log_entries = list(
            RealmAuditLog.objects.filter(acting_user=user).values_list('event_type', 'event_time').order_by('id')
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
        ).values_list('extra_data', flat=True).first()
        assert first_audit_log_entry is not None
        self.assertFalse(first_audit_log_entry['automanage_licenses'])
        realm = get_realm('zulip')
        self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_PLUS)
        self.assertEqual(realm.max_invites, Realm.INVITES_STANDARD_REALM_DAILY_MAX)
        response = self.client_get('/upgrade/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('http://zulip.testserver/billing', response['Location'])
        with time_machine.travel(self.now, tick=False):
            response = self.client_get('/billing/')
        self.assert_not_in_success_response(['Pay annually'], response)
        for substring in [
            'Zulip Cloud Standard',
            str(123),
            'Number of licenses for current billing period',
            f'licenses ({self.seat_count} in use)',
            'You will receive an invoice for',
            'January 2, 2013',
            '$9840.00',
        ]:
            self.assert_in_response(substring, response)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_user_to_basic_plan_free_trial(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with self.settings(CLOUD_FREE_TRIAL_DAYS=60):
            response = self.client_get('/upgrade/')
            free_trial_end_date = self.now + timedelta(days=60)
            self.assert_in_success_response(
                ['Your card will not be charged', 'free trial', '60-day'],
                response,
            )
            self.assertNotEqual(user.realm.plan_type, Realm.PLAN_TYPE_STANDARD)
            self.assertFalse(Customer.objects.filter(realm=user.realm).exists())
            with time_machine.travel(self.now, tick=False), self.assertLogs('corporate.stripe', 'WARNING'):
                response = self.upgrade()
            self.assert_json_error(response, 'Please add a credit card before starting your free trial.')
            stripe_customer = self.add_card_and_upgrade(user)
            self.assertEqual(Invoice.objects.count(), 0)
            self.assertEqual(stripe_customer.description, 'zulip (Zulip Dev)')
            self.assertEqual(stripe_customer.discount, None)
            self.assertEqual(stripe_customer.email, user.delivery_email)
            assert stripe_customer.metadata is not None
            metadata_dict = dict(stripe_customer.metadata)
            self.assertEqual(metadata_dict['realm_str'], 'zulip')
            try:
                int(metadata_dict['realm_id'])
            except ValueError:
                raise AssertionError('realm_id is not a number')
            self.assertFalse(stripe.Charge.list(customer=stripe_customer.id))
            self.assertFalse(stripe.Invoice.list(customer=stripe_customer.id))
            customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
            plan = CustomerPlan.objects.get(
                customer=customer,
                automanage_licenses=True,
                price_per_license=8000,
                fixed_price=None,
                discount=None,
                billing_cycle_anchor=self.now,
                billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
                invoiced_through=LicenseLedger.objects.first(),
                next_invoice_date=free_trial_end_date,
                tier=CustomerPlan.TIER_CLOUD_STANDARD,
                status=CustomerPlan.FREE_TRIAL,
                charge_automatically=True,
            )
            LicenseLedger.objects.get(
                plan=plan, is_renewal=True, event_time=self.now, licenses=self.seat_count, licenses_at_next_renewal=self.seat_count
            )
            audit_log_entries = list(
                RealmAuditLog.objects.filter(acting_user=user).values_list('event_type', 'event_time').order_by('id')
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
            ).values_list('extra_data', flat=True).first()
            assert first_audit_log_entry is not None
            self.assertTrue(first_audit_log_entry['automanage_licenses'])
            realm = get_realm('zulip')
            self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_STANDARD)
            self.assertEqual(realm.max_invites, Realm.INVITES_STANDARD_REALM_DAILY_MAX)
            response = self.client_get('/upgrade/')
            self.assertEqual(response.status_code, 302)
            self.assertEqual('http://zulip.testserver/billing', response['Location'])
            with time_machine.travel(self.now, tick=False):
                response = self.client_get('/billing/')
            self.assert_not_in_success_response(['Pay annually'], response)
            for substring in [
                'Zulip Cloud Standard',
                str(self.seat_count),
                'Number of licenses for current billing period',
                f'{self.seat_count}',
                'Your plan will automatically renew on',
                'January 2, 2013',
                f'${80 * self.seat_count}.00',
                'Visa ending in 4242',
                'Update card',
            ]:
                self.assert_in_response(substring, response)
            self.assert_not_in_success_response(['Number of licenses for current billing period', 'You will receive an invoice for'], response)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_by_card_with_outdated_seat_count_and_minimum_for_plan_tier(self, *mocks: Any) -> None:
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        new_seat_count = self.seat_count - 2
        minimum_for_plan_tier = self.seat_count - 1
        initial_upgrade_request = InitialUpgradeRequest(
            manual_license_management=False,
            tier=CustomerPlan.TIER_CLOUD_STANDARD,
            billing_modality='charge_automatically',
        )
        billing_session = RealmBillingSession(hamlet)
        _, context_when_upgrade_page_is_rendered = billing_session.get_initial_upgrade_context(initial_upgrade_request)
        assert context_when_upgrade_page_is_rendered is not None
        assert context_when_upgrade_page_is_rendered.get('seat_count') == self.seat_count
        with patch(
            'corporate.lib.stripe.BillingSession.min_licenses_for_plan',
            return_value=minimum_for_plan_tier,
        ), patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=new_seat_count,
        ), patch(
            'corporate.lib.stripe.RealmBillingSession.get_initial_upgrade_context',
            return_value=(_, context_when_upgrade_page_is_rendered),
        ):
            self.add_card_and_upgrade(hamlet)
        customer = Customer.objects.first()
        assert customer is not None
        stripe_customer_id = assert_is_not_none(customer.stripe_customer_id)
        [charge] = iter(stripe.Charge.list(customer=stripe_customer_id))
        self.assertEqual(8000 * self.seat_count, charge.amount)
        [upgrade_invoice] = iter(stripe.Invoice.list(customer=stripe_customer_id))
        self.assertEqual([8000 * self.seat_count], [item.amount for item in upgrade_invoice.lines])
        ledger_entry = LicenseLedger.objects.last()
        assert ledger_entry is not None
        self.assertEqual(ledger_entry.licenses, self.seat_count + 10)
        self.assertEqual(ledger_entry.licenses_at_next_renewal, new_seat_count)
        billing_session = RealmBillingSession(realm=hamlet.realm)
        billing_session.make_end_of_cycle_updates_if_needed(plan, self.next_year - timedelta(days=1))
        self.assertEqual(LicenseLedger.objects.count(), 2)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        billing_session.make_end_of_cycle_updates_if_needed(plan, self.next_year)
        self.assertEqual(LicenseLedger.objects.count(), 2)

    def test_get_customer_by_realm(self) -> None:
        realm = get_realm('zulip')
        self.assertEqual(get_customer_by_realm(realm), None)
        customer = Customer.objects.create(realm=realm, stripe_customer_id='cus_12345')
        self.assertEqual(get_customer_by_realm(realm), customer)

    def test_get_current_plan_by_customer(self) -> None:
        realm = get_realm('zulip')
        customer = Customer.objects.create(realm=realm, stripe_customer_id='cus_12345')
        self.assertEqual(get_current_plan_by_customer(customer), None)
        plan = CustomerPlan.objects.create(customer=customer, status=CustomerPlan.ACTIVE, billing_cycle_anchor=timezone_now(),
                                          billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, tier=CustomerPlan.TIER_CLOUD_STANDARD)
        self.assertEqual(get_current_plan_by_customer(customer), plan)
        plan.status = CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE
        plan.save(update_fields=['status'])
        self.assertEqual(get_current_plan_by_customer(customer), plan)
        plan.status = CustomerPlan.ENDED
        plan.save(update_fields=['status'])
        self.assertEqual(get_current_plan_by_customer(customer), None)
        plan.status = CustomerPlan.NEVER_STARTED
        plan.save(update_fields=['status'])
        self.assertEqual(get_current_plan_by_customer(customer), None)

    def test_get_current_plan_by_realm(self) -> None:
        realm = get_realm('zulip')
        self.assertEqual(get_current_plan_by_realm(realm), None)
        customer = Customer.objects.create(realm=realm, stripe_customer_id='cus_12345')
        self.assertEqual(get_current_plan_by_realm(realm), None)
        plan = CustomerPlan.objects.create(customer=customer, status=CustomerPlan.ACTIVE, billing_cycle_anchor=timezone_now(),
                                          billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, tier=CustomerPlan.TIER_CLOUD_STANDARD)
        self.assertEqual(get_current_plan_by_realm(realm), plan)

    def test_is_realm_on_free_trial(self) -> None:
        realm = get_realm('zulip')
        self.assertFalse(is_realm_on_free_trial(realm))
        customer = Customer.objects.create(realm=realm, stripe_customer_id='cus_12345')
        plan = CustomerPlan.objects.create(customer=customer, status=CustomerPlan.ACTIVE, billing_cycle_anchor=timezone_now(),
                                          billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL, tier=CustomerPlan.TIER_CLOUD_STANDARD)
        self.assertFalse(is_realm_on_free_trial(realm))
        plan.status = CustomerPlan.FREE_TRIAL
        plan.save(update_fields=['status'])
        self.assertTrue(is_realm_on_free_trial(realm))

    def test_deactivate_realm(self) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.get()
        assert plan is not None
        self.assertEqual(plan.next_invoice_date, self.next_month)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        billing_session = RealmBillingSession(user=user, realm=get_realm('zulip'))
        with patch('corporate.lib.stripe.get_latest_seat_count', return_value=20):
            billing_session.update_license_ledger_if_needed(self.now)
        last_ledger_entry = LicenseLedger.objects.order_by('id').last()
        assert last_ledger_entry is not None
        self.assertEqual(last_ledger_entry.licenses, 20)
        self.assertEqual(last_ledger_entry.licenses_at_next_renewal, 20)
        do_deactivate_realm(get_realm('zulip'), acting_user=None, deactivation_reason='owner_request', email_owners=False)
        plan.refresh_from_db()
        self.assertTrue(get_realm('zulip').deactivated)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_LIMITED)
        self.assertEqual(plan.status, CustomerPlan.ENDED)
        self.assertEqual(plan.invoiced_through, last_ledger_entry)
        self.assertIsNone(plan.next_invoice_date)
        do_reactivate_realm(get_realm('zulip'))
        self.login_user(user)
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/plans/', response['Location'])
        with patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_month)
        mocked.assert_not_called()
        with patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_year)
        mocked.assert_not_called()

    def test_reupgrade_after_billing_admin_after_downgrade(self) -> None:
        user = self.example_user('hamlet')
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(user)
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(
            self.now, tick=False
        ):
            self.client_billing_patch('/billing/plan', {'status': CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE})
            stripe_customer_id = Customer.objects.get(realm=user.realm).id
            new_plan = get_current_plan_by_realm(user.realm)
            assert new_plan is not None
            expected_log = f'INFO:corporate.stripe:Change plan status: Customer.id: {stripe_customer_id}, CustomerPlan.id: {new_plan.id}, status: {CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE}'
            self.assertEqual(m.output[0], expected_log)
        with self.assertRaises(BillingError) as context, self.assertLogs(
            'corporate.stripe', 'WARNING'
        ) as m, time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(
            m.output[0],
            'WARNING:corporate.stripe:Upgrade of <Realm: zulip 2> (with stripe_customer_id: cus_123) failed because of existing active plan.',
        )
        self.assertEqual(context.exception.error_description, 'subscribing with existing subscription')
        new_plan.next_invoice_date = self.next_year
        new_plan.save(update_fields=['next_invoice_date'])
        invoice_plans_as_needed(self.next_year)
        with time_machine.travel(self.next_year, tick=False):
            response = self.client_get('/billing/')
            self.assertEqual(response.status_code, 302)
            self.assertEqual('/plans/', response['Location'])
            with time_machine.travel(self.next_year, tick=False):
                self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(Customer.objects.count(), 1)
        self.assertEqual(CustomerPlan.objects.count(), 2)
        current_plan = CustomerPlan.objects.all().order_by('id').last()
        assert current_plan is not None
        next_invoice_date = add_months(self.next_year, 1)
        self.assertEqual(current_plan.next_invoice_date, next_invoice_date)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(current_plan.status, CustomerPlan.ACTIVE)
        old_plan = CustomerPlan.objects.all().order_by('id').first()
        assert old_plan is not None
        self.assertEqual(old_plan.next_invoice_date, None)
        self.assertEqual(old_plan.status, CustomerPlan.ENDED)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_validate_licenses_for_manual_plan_management(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False), patch(
            'corporate.lib.stripe.MIN_INVOICED_LICENSES',
            3,
        ):
            self.upgrade(invoice=True, licenses=self.seat_count + 1)
        with time_machine.travel(self.now, tick=False), patch(
            'corporate.lib.stripe.MIN_INVOICED_LICENSES',
            3,
        ):
            result = self.client_billing_patch('/billing/plan', {'licenses_at_next_renewal': self.seat_count})
            self.assert_json_error_contains(
                result,
                f'minimum {3}',
            )
            self.assertEqual(orjson.loads(result.content)['error_description'], 'not enough licenses')
        do_create_user(
            'email-exra-user', 'password-extra-user', get_realm('zulip'), 'name-extra-user', acting_user=None
        )
        with self.assertRaises(BillingError) as context:
            invoice_plans_as_needed(self.next_year)
        self.assertRegex(
            context.exception.error_description,
            'Customer has not manually updated plan for current license count:',
        )

    def test_set_required_plan_tier(self) -> None:
        valid_plan_tier = CustomerPlan.TIER_CLOUD_STANDARD
        support_view_request = SupportViewRequest(
            support_type=SupportType.update_required_plan_tier,
            required_plan_tier=valid_plan_tier,
        )
        support_admin = self.example_user('iago')
        user = self.example_user('hamlet')
        billing_session = RealmBillingSession(
            support_admin, realm=user.realm, support_session=True
        )
        customer = billing_session.get_customer()
        assert customer is None
        message = billing_session.process_support_view_request(support_view_request)
        self.assertEqual(message, 'Required plan tier for zulip set to Zulip Cloud Standard.')
        realm_audit_log = RealmAuditLog.objects.filter(
            event_type=AuditLogEventType.CUSTOMER_PROPERTY_CHANGED
        ).last()
        assert realm_audit_log is not None
        expected_extra_data = {
            'new_annual_discounted_price': 1200,
            'new_monthly_discounted_price': 120,
            'old_annual_discounted_price': 0,
            'old_monthly_discounted_price': 0,
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
            support_type=SupportType.update_required_plan_tier,
            required_plan_tier=invalid_plan_tier,
        )
        with self.assertRaisesRegex(SupportRequestError, 'Invalid plan tier for zulip.'):
            billing_session.process_support_view_request(support_view_request)
        support_view_request = SupportViewRequest(
            support_type=SupportType.update_required_plan_tier,
            required_plan_tier=0,
        )
        with self.assertRaisesRegex(
            SupportRequestError, 'Discount for zulip must be 0 before setting required plan tier to None.'
        ):
            billing_session.process_support_view_request(support_view_request)
        billing_session.attach_discount_to_customer(monthly_discounted_price=0, annual_discounted_price=0)
        message = billing_session.process_support_view_request(support_view_request)
        self.assertEqual(message, 'Required plan tier for zulip set to None.')
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
            'old_value': valid_plan_tier,
            'new_value': None,
            'property': 'required_plan_tier',
        }
        self.assertEqual(realm_audit_log.extra_data, expected_extra_data)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_realm_plan(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.add_card_and_upgrade(user)
            plan = CustomerPlan.objects.get()
            self.assertEqual(plan.status, CustomerPlan.ACTIVE)
            self.assertEqual(plan.tier, CustomerPlan.TIER_SELF_HOSTED_BASIC)
            self.billing_session.set_required_plan_tier(CustomerPlan.TIER_SELF_HOSTED_BUSINESS)
            flat_discount, flat_discounted_months = self.billing_session.get_flat_discount_info()
            self.assertEqual(flat_discount, 2000)
            self.assertEqual(flat_discounted_months, 12)
            self.billing_session.attach_discount_to_customer(monthly_discounted_price=400, annual_discounted_price=4000)
            message = self.billing_session.process_support_view_request(
                SupportViewRequest(
                    support_type=SupportType.modify_plan,
                    plan_modification='upgrade_plan_tier',
                    new_plan_tier=CustomerPlan.TIER_SELF_HOSTED_BUSINESS,
                )
            )
            self.assertEqual(message, 'zulip upgraded to Zulip Cloud Business')
            plan.refresh_from_db()
            self.assertEqual(plan.status, CustomerPlan.ACTIVE)
            self.assertEqual(plan.tier, CustomerPlan.TIER_SELF_HOSTED_BUSINESS)
            customer = self.billing_session.get_customer()
            assert customer is not None
            self.assertEqual(customer.monthly_discounted_price, 400)
            self.assertEqual(customer.annual_discounted_price, 4000)

    @patch('corporate.lib.stripe_event_handler.handle_payment_failed_event')
    def test_send_error_mail_on_failed_invoice_payments(self, handle_payment_failed_event: Callable[..., Any]) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        self.add_card_and_upgrade(user)
        customer = self.billing_session.get_customer()
        assert customer is not None
        invoice = Invoice.objects.get(customer=customer)
        payment_intent_id = 'pi_123456789'
        stripe_event = stripe.Event.construct_from(
            {
                'id': 'evt_test_payment_failed',
                'type': 'payment_intent.payment_failed',
                'data': {'object': {'id': payment_intent_id, 'amount': 1000,}},
                'api_version': STRIPE_API_VERSION,
            },
            stripe.api_key,
        )
        [payment_intent] = iter(stripe.PaymentIntent.list(id=payment_intent_id))
        with patch('stripe.PaymentIntent.retrieve', return_value=payment_intent):
            self.send_stripe_webhook_event(stripe_event)
        handle_payment_failed_event.assert_called_once_with(payment_intent, invoice)

    @patch('corporate.lib.stripe_event_handler.handle_payment_intent_succeeded_event')
    def test_handle_payment_intent_succeeded_event(self, mocked_handle_event: Callable[..., Any]) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        self.upgrade()
        customer = self.billing_session.get_customer()
        assert customer is not None
        plan = self.billing_session.get_current_plan(customer)
        assert plan is not None
        self.assertEqual(plan.status, CustomerPlan.ACTIVE)
        invoice = Invoice.objects.latest('id')
        payment_intent_id = 'pi_123456789'
        stripe_event = stripe.Event.construct_from(
            {
                'id': 'evt_test_payment_succeeded',
                'type': 'payment_intent.succeeded',
                'data': {'object': {'id': payment_intent_id, 'amount': 1000,}},
                'api_version': STRIPE_API_VERSION,
            },
            stripe.api_key,
        )
        [payment_intent] = iter(stripe.PaymentIntent.list(id=payment_intent_id))
        with patch('stripe.PaymentIntent.retrieve', return_value=payment_intent):
            self.send_stripe_webhook_event(stripe_event)
        mocked_handle_event.assert_called_once_with(payment_intent, invoice)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_by_card_with_outdated_seat_count_kids_plan(self, *mocks: Any) -> None:
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        new_seat_count = 15
        initial_upgrade_request = InitialUpgradeRequest(
            manual_license_management=False,
            tier=CustomerPlan.TIER_SELF_HOSTED_KIDS,
            billing_modality='charge_automatically',
        )
        billing_session = RealmBillingSession(hamlet)
        _, context_when_upgrade_page_is_rendered = billing_session.get_initial_upgrade_context(initial_upgrade_request)
        with patch(
            'corporate.lib.stripe.BillingSession.stale_seat_count_check',
            return_value=self.seat_count,
        ), patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=new_seat_count,
        ), patch(
            'corporate.lib.stripe.RealmBillingSession.get_initial_upgrade_context',
            return_value=(_, context_when_upgrade_page_is_rendered),
        ):
            self.add_card_and_upgrade(hamlet)
        customer = Customer.objects.first()
        assert customer is not None
        stripe_customer_id = assert_is_not_none(customer.stripe_customer_id)
        [charge] = iter(stripe.Charge.list(customer=stripe_customer_id))
        self.assertEqual(8000 * self.seat_count, charge.amount)
        [additional_license_invoice, upgrade_invoice] = iter(stripe.Invoice.list(customer=stripe_customer_id))
        self.assertEqual([8000 * self.seat_count], [item.amount for item in upgrade_invoice.lines])
        ledger_entry = LicenseLedger.objects.last()
        assert ledger_entry is not None
        self.assertEqual(ledger_entry.licenses, self.seat_count + 10)
        self.assertEqual(ledger_entry.licenses_at_next_renewal, new_seat_count)

    def test_handle_complimentary_access_plan_upgrade_event(self) -> None:
        pass  # Implement the method if needed

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_complimentary_access_plan(self, *mocks: Any) -> None:
        with time_machine.travel(self.now, tick=False):
            self.add_mock_response()
            send_server_data_to_push_bouncer(consider_usage_statistics=False)
        with time_machine.travel(self.now, tick=False):
            start_date = timezone_now()
            end_date = add_months(start_date, months=3)
            self.billing_session.create_complimentary_access_plan(start_date, end_date)
        customer = self.billing_session.get_customer()
        assert customer is not None
        plan = get_current_plan_by_customer(customer)
        assert plan is not None
        self.assertEqual(plan.end_date, end_date)
        self.assertEqual(plan.next_invoice_date, end_date)
        self.assertEqual(plan.status, CustomerPlan.ACTIVE)
        self.assertEqual(self.remote_server.plan_type, RemoteZulipServer.PLAN_TYPE_SELF_MANAGED_LEGACY)
        with mock.patch('stripe.Invoice.create') as invoice_create, mock.patch(
            'corporate.lib.stripe.send_email'
        ) as send_email, time_machine.travel(end_date, tick=False):
            invoice_plans_as_needed()
            send_email.assert_not_called()
            invoice_create.assert_not_called()
        plan.refresh_from_db()
        self.remote_server.refresh_from_db()
        self.assertEqual(self.remote_server.plan_type, RemoteZulipServer.PLAN_TYPE_SELF_MANAGED)
        self.assertEqual(plan.status, CustomerPlan.ENDED)

    def test_send_login_times_audit_log(self) -> None:
        # Implement the method if needed
        pass

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_update_realm_plan_consider_tier_change(self, *mocks: Any) -> None:
        # Implement the method if needed
        pass

    def test_no_fixed_price_when_automanage_licenses(self) -> None:
        # Implement the method if needed
        pass

    def test_change_required_number_without_min_licenses(self) -> None:
        # Implement the method if needed
        pass

    @mock_stripe()
    def test_set_minimum_with_license_process_incomplete(self, *mocks: Any) -> None:
        # Implement the method if needed
        pass

    def test_non_member_access(self) -> None:
        # Implement the method if needed
        pass


class StripeWebhookEndpointTest(ZulipTestCase):

    def test_stripe_webhook_with_invalid_data(self) -> None:
        result = self.client_post('/stripe/webhook/', '["dsdsds"]', content_type='application/json')
        self.assertEqual(result.status_code, 400)

    def test_stripe_webhook_endpoint_invalid_api_version(self) -> None:
        event_data = {
            'id': 'stripe_event_id',
            'api_version': '1991-02-20',
            'type': 'event_type',
            'data': {'object': {'object': 'checkout.session', 'id': 'stripe_session_id'}},
        }
        expected_error_message = f'Mismatch between billing system Stripe API version({STRIPE_API_VERSION}) and Stripe webhook event API version(1991-02-20).'
        with self.assertLogs('corporate.stripe', 'ERROR') as error_log:
            self.client_post('/stripe/webhook/', event_data, content_type='application/json')
            self.assertEqual(error_log.output, [f'ERROR:corporate.stripe:{expected_error_message}'])

    def test_stripe_webhook_for_session_completed_event(self) -> None:
        valid_session_event_data = {
            'id': 'stripe_event_id',
            'api_version': STRIPE_API_VERSION,
            'type': 'checkout.session.completed',
            'data': {'object': {'object': 'checkout.session', 'id': 'stripe_session_id'}},
        }
        with patch(
            'corporate.lib.stripe_event_handler.handle_checkout_session_completed_event'
        ) as m:
            result = self.client_post('/stripe/webhook/', valid_session_event_data, content_type='application/json')
        self.assert_length(Event.objects.all(), 0)
        self.assertEqual(result.status_code, 200)
        m.assert_not_called()

    def test_stripe_webhook_for_invoice_payment_events(self) -> None:
        customer = Customer.objects.create(realm=get_realm('zulip'))
        stripe_event_id = 'stripe_event_id'
        stripe_invoice_id = 'stripe_invoice_id'
        valid_invoice_paid_event_data = {
            'id': stripe_event_id,
            'type': 'invoice.paid',
            'api_version': STRIPE_API_VERSION,
            'data': {'object': {'object': 'invoice', 'id': stripe_invoice_id}},
        }
        with patch(
            'corporate.lib.stripe_event_handler.handle_invoice_paid_event'
        ) as m:
            result = self.client_post('/stripe/webhook/', valid_invoice_paid_event_data, content_type='application/json')
        self.assert_length(Event.objects.filter(stripe_event_id=stripe_event_id), 0)
        self.assertEqual(result.status_code, 200)
        m.assert_not_called()
        Invoice.objects.create(stripe_invoice_id=stripe_invoice_id, customer=customer, status=Invoice.SENT)
        self.assert_length(Event.objects.filter(stripe_event_id=stripe_event_id), 0)
        with patch(
            'corporate.lib.stripe_event_handler.handle_invoice_paid_event'
        ) as m:
            result = self.client_post('/stripe/webhook/', valid_invoice_paid_event_data, content_type='application/json')
        [event] = Event.objects.filter(stripe_event_id=stripe_event_id)
        self.assertEqual(result.status_code, 200)
        strip_event = stripe.Event.construct_from(valid_invoice_paid_event_data, stripe.api_key)
        m.assert_called_once_with(strip_event.data.object, event)
        with patch(
            'corporate.lib.stripe_event_handler.handle_invoice_paid_event'
        ) as m:
            result = self.client_post('/stripe/webhook/', valid_invoice_paid_event_data, content_type='application/json')
        self.assert_length(Event.objects.filter(stripe_event_id=stripe_event_id), 1)
        self.assertEqual(result.status_code, 200)
        m.assert_not_called()

    def test_stripe_webhook_for_invoice_paid_events(self) -> None:
        customer = Customer.objects.create(realm=get_realm('zulip'))
        stripe_event_id = 'stripe_event_id'
        stripe_invoice_id = 'stripe_invoice_id'
        valid_invoice_paid_event_data = {
            'id': stripe_event_id,
            'type': 'invoice.paid',
            'api_version': STRIPE_API_VERSION,
            'data': {'object': {'object': 'invoice', 'id': stripe_invoice_id}},
        }
        with patch(
            'corporate.lib.stripe_event_handler.handle_invoice_paid_event'
        ) as m:
            result = self.client_post('/stripe/webhook/', valid_invoice_paid_event_data, content_type='application/json')
        self.assert_length(Event.objects.filter(stripe_event_id=stripe_event_id), 0)
        self.assertEqual(result.status_code, 200)
        m.assert_not_called()
        Invoice.objects.create(stripe_invoice_id=stripe_invoice_id, customer=customer, status=Invoice.SENT)
        self.assert_length(Event.objects.filter(stripe_event_id=stripe_event_id), 0)
        with patch(
            'corporate.lib.stripe_event_handler.handle_invoice_paid_event'
        ) as m:
            result = self.client_post('/stripe/webhook/', valid_invoice_paid_event_data, content_type='application/json')
        [event] = Event.objects.filter(stripe_event_id=stripe_event_id)
        self.assertEqual(result.status_code, 200)
        strip_event = stripe.Event.construct_from(valid_invoice_paid_event_data, stripe.api_key)
        m.assert_called_once_with(strip_event.data.object, event)
        with patch(
            'corporate.lib.stripe_event_handler.handle_invoice_paid_event'
        ) as m:
            result = self.client_post('/stripe/webhook/', valid_invoice_paid_event_data, content_type='application/json')
        self.assert_length(Event.objects.filter(stripe_event_id=stripe_event_id), 1)
        self.assertEqual(result.status_code, 200)
        m.assert_not_called()

    def test_add_plan_renewal_if_needed(self) -> None:
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(LicenseLedger.objects.count(), 1)
        plan = CustomerPlan.objects.get()
        realm = plan.customer.realm
        billing_session = RealmBillingSession(user=None, realm=realm)
        billing_session.make_end_of_cycle_updates_if_needed(plan, self.next_year - timedelta(days=1))
        self.assertEqual(LicenseLedger.objects.count(), 1)
        new_plan, ledger_entry = billing_session.make_end_of_cycle_updates_if_needed(plan, self.next_year)
        self.assertIsNone(new_plan)
        self.assertEqual(LicenseLedger.objects.count(), 2)
        ledger_params = {
            'plan': plan,
            'is_renewal': True,
            'event_time': self.next_year,
            'licenses': self.seat_count,
            'licenses_at_next_renewal': self.seat_count,
        }
        for key, value in ledger_params.items():
            self.assertEqual(getattr(ledger_entry, key), value)
        billing_session.make_end_of_cycle_updates_if_needed(plan, self.next_year + timedelta(days=1))
        self.assertEqual(LicenseLedger.objects.count(), 2)

    def test_update_license_ledger_if_needed(self) -> None:
        realm = get_realm('zulip')
        billing_session = RealmBillingSession(user=None, realm=realm)
        billing_session.update_license_ledger_if_needed(self.now)
        self.assertFalse(LicenseLedger.objects.exists())
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count + 1, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(LicenseLedger.objects.count(), 1)
        self.assertEqual(plan.licenses(), self.seat_count + 1)
        self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count + 1)
        billing_session.update_license_ledger_if_needed(self.now)
        self.assertEqual(LicenseLedger.objects.count(), 1)
        plan.automanage_licenses = True
        plan.status = CustomerPlan.ENDED
        plan.save(update_fields=['automanage_licenses', 'status'])
        billing_session.update_license_ledger_if_needed(self.now)
        self.assertEqual(LicenseLedger.objects.count(), 1)
        plan.status = CustomerPlan.ACTIVE
        plan.save(update_fields=['status'])
        billing_session.update_license_ledger_if_needed(self.now)
        self.assertEqual(LicenseLedger.objects.count(), 2)

    def test_update_license_ledger_for_automanaged_plan(self) -> None:
        realm = get_realm('zulip')
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(plan.licenses(), self.seat_count)
        self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count)
        billing_session = RealmBillingSession(user=None, realm=realm)
        with patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=23,
        ):
            billing_session.update_license_ledger_for_automanaged_plan(plan, self.now)
            self.assertEqual(plan.licenses(), 23)
            self.assertEqual(plan.licenses_at_next_renewal(), 23)
        with patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=20,
        ):
            billing_session.update_license_ledger_for_automanaged_plan(plan, self.now)
            self.assertEqual(plan.licenses(), 23)
            self.assertEqual(plan.licenses_at_next_renewal(), 20)
        with patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=21,
        ):
            billing_session.update_license_ledger_for_automanaged_plan(plan, self.now)
            self.assertEqual(plan.licenses(), 23)
            self.assertEqual(plan.licenses_at_next_renewal(), 21)
        with patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=22,
        ):
            billing_session.update_license_ledger_for_automanaged_plan(plan, self.next_year + timedelta(seconds=1))
            self.assertEqual(plan.licenses(), 22)
            self.assertEqual(plan.licenses_at_next_renewal(), 22)
        ledger_entries = list(
            LicenseLedger.objects.values_list(
                'is_renewal', 'event_time', 'licenses', 'licenses_at_next_renewal'
            ).order_by('id')
        )
        self.assertEqual(
            ledger_entries,
            [
                (True, self.now, self.seat_count, self.seat_count),
                (False, self.now, 23, 23),
                (False, self.now, 23, 20),
                (False, self.now, 23, 21),
                (True, self.next_year, 21, 21),
                (False, self.next_year + timedelta(seconds=1), 22, 22),
            ],
        )

    def test_update_license_ledger_for_manual_plan(self) -> None:
        realm = get_realm('zulip')
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count + 1, False, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        billing_session = RealmBillingSession(user=None, realm=realm)
        plan = self.billing_session.get_current_plan(customer=Customer.objects.get(realm=realm))
        assert plan is not None
        with patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=self.seat_count,
        ):
            billing_session.update_license_ledger_for_manual_plan(plan, self.now, licenses=self.seat_count + 3)
            self.assertEqual(plan.licenses(), self.seat_count + 3)
            self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count + 3)
        with patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=self.seat_count,
        ), self.assertRaises(AssertionError):
            billing_session.update_license_ledger_for_manual_plan(plan, self.now, licenses=self.seat_count)
        with patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=self.seat_count,
        ):
            billing_session.update_license_ledger_for_manual_plan(plan, self.now, licenses_at_next_renewal=self.seat_count)
            self.assertEqual(plan.licenses(), self.seat_count + 3)
            self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count)
        with patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=self.seat_count,
        ), self.assertRaises(AssertionError):
            billing_session.update_license_ledger_for_manual_plan(plan, self.now, licenses_at_next_renewal=self.seat_count - 1)
        with patch(
            'corporate.lib.stripe.get_latest_seat_count',
            return_value=self.seat_count,
        ):
            billing_session.update_license_ledger_for_manual_plan(plan, self.now, licenses=self.seat_count + 10)
            self.assertEqual(plan.licenses(), self.seat_count + 10)
            self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count + 10)
        billing_session.make_end_of_cycle_updates_if_needed(plan, self.next_year)
        self.assertEqual(plan.licenses(), self.seat_count + 10)
        ledger_entries = list(
            LicenseLedger.objects.values_list(
                'is_renewal', 'event_time', 'licenses', 'licenses_at_next_renewal'
            ).order_by('id')
        )
        self.assertEqual(
            ledger_entries,
            [
                (True, self.now, self.seat_count + 1, self.seat_count + 1),
                (False, self.now, self.seat_count + 3, self.seat_count + 3),
                (False, self.now, self.seat_count + 3, self.seat_count),
                (False, self.now, self.seat_count + 10, self.seat_count + 10),
                (True, self.next_year, self.seat_count + 10, self.seat_count + 10),
                (False, self.next_year + timedelta(days=1), 22, 22),
            ],
        )
        with self.assertRaises(AssertionError):
            billing_session.update_license_ledger_for_manual_plan(plan, self.now)

    def test_user_changes(self) -> None:
        self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        user = do_create_user('email', 'password', get_realm('zulip'), 'name', acting_user=None)
        do_deactivate_user(user, acting_user=None)
        do_reactivate_user(user, acting_user=None)
        change_user_is_active(user, False)
        user.is_mirror_dummy = True
        user.save(update_fields=['is_mirror_dummy'])
        do_activate_mirror_dummy_user(user, acting_user=None)
        guest = do_create_user(
            'guest_email', 'guest_password', get_realm('zulip'), 'guest_name', role=UserProfile.ROLE_GUEST, acting_user=None
        )
        do_change_user_role(guest, UserProfile.ROLE_MEMBER, acting_user=None)
        do_change_user_role(guest, UserProfile.ROLE_MODERATOR, acting_user=None)
        ledger_entries = list(
            LicenseLedger.objects.values_list(
                'is_renewal', 'licenses', 'licenses_at_next_renewal'
            ).order_by('id')
        )
        self.assertEqual(
            ledger_entries,
            [
                (True, self.seat_count, self.seat_count),
                (False, self.seat_count + 1, self.seat_count + 1),
                (False, self.seat_count + 1, self.seat_count),
                (False, self.seat_count + 1, self.seat_count + 1),
                (False, self.seat_count + 1, self.seat_count + 1),
                (False, self.seat_count + 1, self.seat_count + 1),
                (False, self.seat_count + 2, self.seat_count + 2),
            ],
        )

    def test_toggle_license_management(self) -> None:
        self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(plan.automanage_licenses, True)
        self.assertEqual(plan.licenses(), self.seat_count)
        self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count)
        billing_session = RealmBillingSession(user=None, realm=get_realm('zulip'))
        update_plan_request = UpdatePlanRequest(
            status=None,
            licenses=None,
            licenses_at_next_renewal=None,
            schedule=None,
            toggle_license_management=True,
        )
        billing_session.do_update_plan(update_plan_request)
        plan.refresh_from_db()
        self.assertEqual(plan.automanage_licenses, False)
        billing_session.do_update_plan(update_plan_request)
        plan.refresh_from_db()
        self.assertEqual(plan.automanage_licenses, True)

    def test_reupgrade_after_plan_status_changed_to_downgrade_at_end_of_cycle(self) -> None:
        user = self.example_user('hamlet')
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.login_user(user)
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(
            self.now, tick=False
        ):
            response = self.client_billing_patch('/billing/plan', {'status': CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE})
            stripe_customer_id = Customer.objects.get(realm=user.realm).id
            new_plan = get_current_plan_by_realm(user.realm)
            assert new_plan is not None
            expected_log = f'INFO:corporate.stripe:Change plan status: Customer.id: {stripe_customer_id}, CustomerPlan.id: {new_plan.id}, status: {CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE}'
            self.assertEqual(m.output[0], expected_log)
            self.assert_json_success(response)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(plan.status, CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE)
        with self.assertRaisesRegex(BillingError, 'subscribing with existing subscription'), self.assertLogs(
            'corporate.stripe', 'WARNING'
        ) as m, time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(
            m.output[0],
            'WARNING:corporate.stripe:Upgrade of <Realm: zulip 2> (with stripe_customer_id: cus_123) failed because of existing active plan.',
        )
        self.assertEqual(context.exception.error_description, 'subscribing with existing subscription')
        new_plan.next_invoice_date = self.next_year
        new_plan.save(update_fields=['next_invoice_date'])
        invoice_plans_as_needed(self.next_year)
        with time_machine.travel(self.next_year, tick=False):
            response = self.client_get('/billing/')
            self.assertEqual(response.status_code, 302)
            self.assertEqual('/plans/', response['Location'])
            with time_machine.travel(self.next_year, tick=False):
                self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(Customer.objects.count(), 1)
        self.assertEqual(CustomerPlan.objects.count(), 2)
        current_plan = CustomerPlan.objects.all().order_by('id').last()
        assert current_plan is not None
        next_invoice_date = add_months(self.next_year, 1)
        self.assertEqual(current_plan.next_invoice_date, next_invoice_date)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        self.assertEqual(current_plan.status, CustomerPlan.ACTIVE)
        old_plan = CustomerPlan.objects.all().order_by('id').first()
        assert old_plan is not None
        self.assertEqual(old_plan.next_invoice_date, None)
        self.assertEqual(old_plan.status, CustomerPlan.ENDED)

    @patch('corporate.lib.stripe.billing_logger.info')
    def test_deactivate_registration_with_push_notification_service(
        self, mock_: MagicMock
    ) -> None:
        self.login('hamlet')
        hamlet = self.example_user('hamlet')
        billing_base_url = self.billing_session.billing_base_url
        with self.settings(EXTERNAL_HOST='zulipdev.com:9991'):
            confirmation_link = generate_confirmation_link_for_server_deactivation(
                self.remote_server, 10
            )
        result = self.client_get(confirmation_link, subdomain='selfhosting')
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response(['Log in to deactivate registration for'], result)
        result = self.client_post(
            confirmation_link,
            {'full_name': hamlet.full_name, 'tos_consent': 'true'},
            subdomain='selfhosting',
        )
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], f'{billing_base_url}/deactivate/')
        result = self.client_get(f'{billing_base_url}/deactivate/', subdomain='selfhosting')
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response(['Deactivate registration for', 'Deactivate registration'], result)
        result = self.client_post(f'{billing_base_url}/deactivate/', {'confirmed': 'true'}, subdomain='selfhosting')
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response(
            ['Registration deactivated for', "Your server's registration has been deactivated."],
            result,
        )
        payload = {
            'zulip_org_id': self.remote_server.uuid,
            'zulip_org_key': self.remote_server.api_key,
        }
        result = self.client_post('/serverlogin/', payload, subdomain='selfhosting')
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response(['Your server registration has been deactivated.'], result)

    @responses.activate
    @mock_stripe()
    def test_upgrade_user_to_fixed_price_plan_monthly_basic(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.add_card_and_upgrade(user)
            plan = CustomerPlan.objects.get()
            self.assertEqual(plan.status, CustomerPlan.ACTIVE)
            self.assertEqual(plan.tier, CustomerPlan.TIER_SELF_HOSTED_BASIC)
            self.billing_session.set_required_plan_tier(CustomerPlan.TIER_SELF_HOSTED_BUSINESS)
            billing_session = RealmBillingSession(user=user, realm=user.realm, support_session=True)
            billing_session.attach_discount_to_customer(
                monthly_discounted_price=1200, annual_discounted_price=12000
            )
            message = billing_session.process_support_view_request(
                SupportViewRequest(
                    support_type=SupportType.modify_plan,
                    plan_modification='upgrade_plan_tier',
                    new_plan_tier=CustomerPlan.TIER_SELF_HOSTED_BUSINESS,
                )
            )
            self.assertEqual(message, 'zulip upgraded to Zulip Cloud Business')
            plan.refresh_from_db()
            self.assertEqual(plan.status, CustomerPlan.ACTIVE)
            self.assertEqual(plan.tier, CustomerPlan.TIER_SELF_HOSTED_BUSINESS)
            customer = billing_session.get_customer()
            assert customer is not None
            self.assertEqual(customer.monthly_discounted_price, 1200)
            self.assertEqual(customer.annual_discounted_price, 12000)
            self.assert_in_success_response(['Zulip Cloud Business', 'Number of licenses', '25'], self.client_get('/billing/'))

    @responses.activate
    @mock_stripe()
    def test_downgrade_realm_and_void_open_invoices(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        plan = CustomerPlan.objects.get()
        assert plan is not None
        self.assertEqual(plan.next_invoice_date, self.next_month)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
        billing_session = RealmBillingSession(user=user, realm=get_realm('zulip'))
        with patch('corporate.lib.stripe.get_latest_seat_count', return_value=20):
            billing_session.update_license_ledger_if_needed(self.now)
        last_ledger_entry = LicenseLedger.objects.order_by('id').last()
        assert last_ledger_entry is not None
        self.assertEqual(last_ledger_entry.licenses, 20)
        self.assertEqual(last_ledger_entry.licenses_at_next_renewal, 20)
        do_deactivate_realm(get_realm('zulip'), acting_user=None, deactivation_reason='owner_request', email_owners=False)
        plan.refresh_from_db()
        self.assertTrue(get_realm('zulip').deactivated)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_LIMITED)
        self.assertEqual(plan.status, CustomerPlan.ENDED)
        self.assertEqual(plan.invoiced_through, last_ledger_entry)
        self.assertIsNone(plan.next_invoice_date)
        do_reactivate_realm(get_realm('zulip'))
        self.login_user(user)
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/plans/', response['Location'])
        with patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_month)
        mocked.assert_not_called()
        with patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_year)
        mocked.assert_not_called()

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_with_tampered_seat_count(self) -> None:
        hamlet = self.example_user('hamlet')
        self.login_user(hamlet)
        with self.assertRaisesRegex(BillingError, 'Something went wrong. Please contact'):
            self.upgrade(talk_to_stripe=False, salt='badsalt')
        self.assertEqual(
            orjson.loads(self.upgrade(talk_to_stripe=False, salt='badsalt').content)['error_description'],
            'tampered seat count',
        )

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_race_condition_during_card_upgrade(self, *mocks: Any) -> None:
        hamlet = self.example_user('hamlet')
        othello = self.example_user('othello')
        self.login_user(othello)
        othello_upgrade_page_response = self.client_get('/upgrade/')
        self.login_user(hamlet)
        self.add_card_to_customer_for_upgrade()
        [stripe_event_before_upgrade] = iter(stripe.Event.list(limit=1))
        hamlet_upgrade_page_response = self.client_get('/upgrade/')
        self.client_billing_post(
            '/billing/upgrade',
            {
                'billing_modality': 'charge_automatically',
                'schedule': 'annual',
                'signed_seat_count': self.get_signed_seat_count_from_response(
                    hamlet_upgrade_page_response
                ),
                'salt': self.get_salt_from_response(hamlet_upgrade_page_response),
                'license_management': 'automatic',
            },
        )
        customer = Customer.objects.get(realm=get_realm('zulip'))
        assert customer is not None
        self.assertEqual(CustomerPlan.objects.filter(customer=customer).count(), 1)
        self.login_user(othello)
        with self.settings(CLOUD_FREE_TRIAL_DAYS=60):
            self.client_billing_post(
                '/billing/upgrade',
                {
                    'billing_modality': 'charge_automatically',
                    'schedule': 'annual',
                    'signed_seat_count': self.get_signed_seat_count_from_response(
                        othello_upgrade_page_response
                    ),
                    'salt': self.get_salt_from_response(othello_upgrade_page_response),
                    'license_management': 'automatic',
                },
            )
        with self.assertLogs('corporate.stripe', 'WARNING'):
            self.send_stripe_webhook_events(stripe_event_before_upgrade)
        [hamlet_invoice] = iter(stripe.Invoice.list(customer=customer.stripe_invoice_id))
        self.assert_details_of_valid_invoice_payment_from_event_status_endpoint(
            hamlet_invoice.id,
            {
                'status': 'paid',
                'event_handler': {
                    'status': 'failed',
                    'error': {
                        'message': 'The organization is already subscribed to a plan. Please reload the billing page.',
                        'description': 'subscribing with existing subscription',
                    },
                },
            },
        )
        from django.core.mail import outbox
        self.assert_length(outbox, 1)
        for message in outbox:
            self.assert_length(message.to, 1)
            self.assertEqual(message.to[0], 'sales@zulip.com')
            self.assertEqual(message.subject, 'Error processing paid customer invoice')
            self.assertEqual(self.email_envelope_from(message), settings.NOREPLY_EMAIL_ADDRESS)

    def test_upgrade_race_condition_during_invoice_upgrade(self) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        with self.assertLogs('corporate.stripe', 'WARNING') as context:
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        self.assertEqual(
            context.output[0],
            'WARNING:corporate.stripe:Upgrade of <Realm: zulip 2> (with stripe_customer_id: cus_123) failed because of existing active plan.',
        )
        self.assert_length(context.output, 1)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_validate_licenses_for_manual_plan_management(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False), patch(
            'corporate.lib.stripe.MIN_INVOICED_LICENSES',
            3,
        ):
            self.upgrade(invoice=True, licenses=self.seat_count + 1)
        with time_machine.travel(self.now, tick=False), patch(
            'corporate.lib.stripe.MIN_INVOICED_LICENSES',
            3,
        ):
            result = self.client_billing_patch(
                '/billing/plan',
                {'licenses_at_next_renewal': self.seat_count},
            )
            self.assert_json_error_contains(
                result,
                f'minimum {3}',
            )
            self.assertEqual(
                orjson.loads(result.content)['error_description'],
                'not enough licenses',
            )
        do_create_user(
            'email-exra-user',
            'password-extra-user',
            get_realm('zulip'),
            'name-extra-user',
            acting_user=None,
        )
        with self.assertRaises(BillingError) as context:
            invoice_plans_as_needed(self.next_year)
        self.assertRegex(
            context.exception.error_description,
            'Customer has not manually updated plan for current license count:',
        )

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_set_minimum_with_license_process_incomplete(self, *mocks: Any) -> None:
        # Implement the method if needed
        pass

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_add_fixed_price_licenses(self, *mocks: Any) -> None:
        # Implement the method if needed
        pass

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_with_outdated_no_minimum_plan_min(self, *mocks: Any) -> None:
        # Implement the method if needed
        pass

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_with_licenses_override(self, *mocks: Any) -> None:
        # Implement the method if needed
        pass

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_get_latest_seat_count(self, *mocks: Any) -> None:
        realm = get_realm('zulip')
        initial_count = get_latest_seat_count(realm)
        user1 = UserProfile.objects.create(
            realm=realm, email='user1@zulip.com', delivery_email='user1@zulip.com'
        )
        user2 = UserProfile.objects.create(
            realm=realm, email='user2@zulip.com', delivery_email='user2@zulip.com'
        )
        self.assertEqual(get_latest_seat_count(realm), initial_count + 2)
        user1.is_bot = True
        user1.save(update_fields=['is_bot'])
        self.assertEqual(get_latest_seat_count(realm), initial_count + 1)
        do_deactivate_user(user2, acting_user=None)
        self.assertEqual(get_latest_seat_count(realm), initial_count)
        UserProfile.objects.create(
            realm=realm,
            email='user3@zulip.com',
            delivery_email='user3@zulip.com',
            role=UserProfile.ROLE_GUEST,
        )
        self.assertEqual(get_latest_seat_count(realm), initial_count)
        realm = do_create_realm(string_id='second', name='second')
        UserProfile.objects.create(realm=realm, email='member@second.com', delivery_email='member@second.com')
        for i in range(5):
            UserProfile.objects.create(
                realm=realm,
                email=f'guest{i}@second.com',
                delivery_email=f'guest{i}@second.com',
                role=UserProfile.ROLE_GUEST,
            )
        self.assertEqual(get_latest_seat_count(realm), 1)
        UserProfile.objects.create(
            realm=realm,
            email='guest5@second.com',
            delivery_email='guest5@second.com',
            role=UserProfile.ROLE_GUEST,
        )
        self.assertEqual(get_latest_seat_count(realm), 2)

    def test_sign_string(self) -> None:
        string = 'abc'
        signed_string, salt = sign_string(string)
        self.assertEqual(string, unsign_string(signed_string, salt))
        with self.assertRaises(signing.BadSignature):
            unsign_string(signed_string, 'randomsalt')

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_free_trial_upgrade_by_invoice(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with self.settings(CLOUD_FREE_TRIAL_DAYS=60):
            response = self.client_get('/upgrade/')
            self.assert_in_success_response(['Your card will not be charged', 'free trial', '60-day'], response)
            self.assertNotEqual(user.realm.plan_type, Realm.PLAN_TYPE_STANDARD)
            self.assertFalse(Customer.objects.filter(realm=user.realm).exists())
            with time_machine.travel(self.now, tick=False):
                self.upgrade(invoice=True)
            stripe_customer = stripe_get_customer(assert_is_not_none(Customer.objects.get(realm=user.realm).stripe_customer_id))
            self.assertFalse(stripe_customer_has_credit_card_as_default_payment_method(stripe_customer))
            self.assertFalse(stripe.Charge.list(customer=stripe_customer.id))
            [invoice] = iter(stripe.Invoice.list(customer=stripe_customer.id))
            self.assertIsNotNone(invoice.due_date)
            self.assertIsNotNone(invoice.status_transitions.finalized_at)
            invoice_params = {
                'amount_due': 8000 * 123,
                'amount_paid': 0,
                'attempt_count': 0,
                'auto_advance': False,
                'collection_method': 'send_invoice',
                'statement_descriptor': 'Zulip Cloud Standard',
                'status': 'paid',
                'total': 8000 * 123,
            }
            for key, value in invoice_params.items():
                self.assertEqual(invoice.get(key), value)
            [item] = iter(invoice.lines)
            line_item_params = {
                'amount': 8000 * 123,
                'description': 'Zulip Cloud Standard',
                'discountable': False,
                'plan': None,
                'proration': False,
                'quantity': 123,
                'period': {
                    'start': datetime_to_timestamp(self.now),
                    'end': datetime_to_timestamp(add_months(self.now, 12)),
                },
            }
            for key, value in line_item_params.items():
                self.assertEqual(item.get(key), value)
            customer = Customer.objects.get(stripe_customer_id=stripe_customer.id, realm=user.realm)
            plan = CustomerPlan.objects.get(
                customer=customer,
                automanage_licenses=False,
                charge_automatically=False,
                price_per_license=8000,
                fixed_price=None,
                discount=None,
                billing_cycle_anchor=self.now,
                billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
                invoiced_through=LicenseLedger.objects.first(),
                next_invoice_date=self.next_month,
                tier=CustomerPlan.TIER_CLOUD_STANDARD,
                status=CustomerPlan.ACTIVE,
            )
            LicenseLedger.objects.get(plan=plan, is_renewal=True, event_time=self.now, licenses=123, licenses_at_next_renewal=123)
            audit_log_entries = list(
                RealmAuditLog.objects.filter(acting_user=user).values_list('event_type', 'event_time').order_by('id')
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
            ).values_list('extra_data', flat=True).first()
            assert first_audit_log_entry is not None
            self.assertFalse(first_audit_log_entry['automanage_licenses'])
            realm = get_realm('zulip')
            self.assertEqual(realm.plan_type, Realm.PLAN_TYPE_PLUS)
            self.assertEqual(realm.max_invites, Realm.INVITES_STANDARD_REALM_DAILY_MAX)
            response = self.client_get('/upgrade/')
            self.assertEqual(response.status_code, 302)
            self.assertEqual('http://zulip.testserver/billing', response['Location'])
            with time_machine.travel(self.now, tick=False):
                response = self.client_get('/billing/')
            self.assert_not_in_success_response(['Pay annually'], response)
            for substring in [
                'Zulip Cloud Standard',
                str(123),
                'Number of licenses for current billing period',
                f'licenses ({self.seat_count} in use)',
                'You will receive an invoice for',
                'January 2, 2013',
                '$9840.00',
            ]:
                self.assert_in_response(substring, response)

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_fixed_price_invoicing(self, *mocks: Any) -> None:
        # Implement the method if needed
        pass

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_handle_fixed_price_plan_event(self, *mocks: Any) -> None:
        # Implement the method if needed
        pass

    class RemoteRealmBillingSessionTest(ZulipTestCase):
        pass

    class RemoteServerBillingSessionTest(ZulipTestCase):
        pass

    class TestRemoteBillingWriteAuditLog(StripeTestCase):
        pass

    @activate_push_notification_service()
    class TestRemoteRealmBillingFlow(StripeTestCase, RemoteRealmBillingTestCase):

        @override
        def setUp(self) -> None:
            super().setUp()
            zulip_realm = get_realm('zulip')
            RealmAuditLog.objects.filter(
                realm=zulip_realm, event_type__in=RealmAuditLog.SYNCED_BILLING_EVENTS
            ).delete()
            with time_machine.travel(self.now, tick=False):
                for count in range(4):
                    for realm in [zulip_realm, get_realm('zephyr'), get_realm('lear')]:
                        do_create_user(f'email {count}', f'password {count}', realm, 'name', acting_user=None)
            self.remote_realm = RemoteRealm.objects.get(uuid=zulip_realm.uuid)
            self.billing_session = RemoteRealmBillingSession(remote_realm=self.remote_realm)

        @responses.activate
        @mock_stripe()
        def test_upgrade_user_to_business_plan(self, *mocks: Any) -> None:
            self.login('hamlet')
            hamlet = self.example_user('hamlet')
            self.add_mock_response()
            realm_user_count = UserProfile.objects.filter(
                realm=hamlet.realm, is_bot=False, is_active=True
            ).count()
            self.assertEqual(realm_user_count, 11)
            with time_machine.travel(self.now, tick=False):
                send_server_data_to_push_bouncer(consider_usage_statistics=False)
            result = self.execute_remote_billing_authentication_flow(hamlet)
            self.assertEqual(result.status_code, 302)
            self.assertEqual(result['Location'], f'{self.billing_session.billing_base_url}/plans/')
            with time_machine.travel(self.now, tick=False):
                result = self.client_get(f'{self.billing_session.billing_base_url}/upgrade/?tier={CustomerPlan.TIER_SELF_HOSTED_BASIC}', subdomain='selfhosting')
            self.assertEqual(result.status_code, 200)
            min_licenses = self.billing_session.min_licenses_for_plan(
                CustomerPlan.TIER_SELF_HOSTED_BASIC
            )
            self.assertEqual(min_licenses, 6)
            flat_discount, flat_discounted_months = self.billing_session.get_flat_discount_info()
            self.assertEqual(flat_discounted_months, 12)
            self.assert_in_success_response(
                ['Start free trial', 'Zulip Basic', 'Due', 'on February 1, 2012', f'{min_licenses}', 'Add card', 'Start 30-day free trial'],
                result,
            )
            self.assertFalse(Customer.objects.exists())
            self.assertFalse(CustomerPlan.objects.exists())
            self.assertFalse(LicenseLedger.objects.exists())
            with time_machine.travel(self.now, tick=False):
                stripe_customer = self.add_card_and_upgrade(
                    tier=CustomerPlan.TIER_SELF_HOSTED_BASIC, schedule='monthly'
                )
            self.assertEqual(Invoice.objects.count(), 0)
            customer = Customer.objects.get(stripe_customer_id=stripe_customer.id)
            plan = CustomerPlan.objects.get(customer=customer)
            LicenseLedger.objects.get(plan=plan)
            with time_machine.travel(self.now + timedelta(days=1), tick=False):
                response = self.client_get(
                    f'{self.billing_session.billing_base_url}/billing/', subdomain='selfhosting'
                )
            for substring in [
                'Zulip Basic',
                '(free trial)',
                'Number of licenses',
                f'{realm_user_count}',
                'February 1, 2012',
                'Your plan will automatically renew on',
                f'${3.5 * realm_user_count - flat_discount // 100 * 1:,.2f}',
                'Visa ending in 4242',
                'Update card',
            ]:
                self.assert_in_response(substring, response)
            audit_log_count = RemoteRealmAuditLog.objects.count()
            self.assertEqual(LicenseLedger.objects.count(), 1)
            with time_machine.travel(self.now + timedelta(days=2), tick=False):
                for count in range(realm_user_count, min_licenses + 10):
                    do_create_user(
                        f'email {count}',
                        f'password {count}',
                        hamlet.realm,
                        'name',
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
                response = self.client_get(
                    f'{self.billing_session.billing_base_url}/billing/', subdomain='selfhosting'
                )
            self.assertEqual(latest_ledger.licenses, min_licenses + 10)
            for substring in [
                'Zulip Basic',
                'Number of licenses',
                f'{latest_ledger.licenses}',
                'February 1, 2012',
                'Your plan will automatically renew on',
                f'${3.5 * latest_ledger.licenses - flat_discount // 100 * 1:,.2f}',
                'Visa ending in 4242',
                'Update card',
            ]:
                self.assert_in_response(substring, response)
            customer.flat_discounted_months = 0
            customer.save(update_fields=['flat_discounted_months'])
            self.assertEqual(
                self.billing_session.min_licenses_for_plan(CustomerPlan.TIER_SELF_HOSTED_BASIC),
                1,
            )

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_fixed_price_plan_upgrade_to_business_plan(self, *mocks: Any) -> None:
        # Implement the method if needed
        pass

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_user_to_fixed_price_plan_pay_by_invoice(self, *mocks: Any) -> None:
        self.login('iago')
        hamlet = self.example_user('hamlet')
        self.add_mock_response()
        with time_machine.travel(self.now, tick=False):
            send_server_data_to_push_bouncer(consider_usage_statistics=False)
        self.assertFalse(CustomerPlanOffer.objects.exists())
        annual_fixed_price = 1200
        result = self.client_post(
            '/activity/remote/support',
            {
                'remote_realm_id': f'{self.remote_realm.id}',
                'required_plan_tier': CustomerPlan.TIER_SELF_HOSTED_BASIC,
                'fixed_price': annual_fixed_price,
                'sent_invoice_id': 'test_sent_invoice_id',
            },
        )
        self.assert_in_success_response(['Customer can now buy a fixed price Zulip Basic plan.'], result)
        fixed_price_plan_offer = CustomerPlanOffer.objects.filter(status=CustomerPlanOffer.CONFIGURED).first()
        assert fixed_price_plan_offer is not None
        self.assertEqual(fixed_price_plan_offer.tier, CustomerPlanOffer.TIER_SELF_HOSTED_BASIC)
        self.assertEqual(fixed_price_plan_offer.fixed_price, annual_fixed_price * 100)
        self.assertEqual(fixed_price_plan_offer.sent_invoice_id, 'test_sent_invoice_id')
        self.assertEqual(fixed_price_plan_offer.get_plan_status_as_text(), 'Configured')
        invoice = Invoice.objects.get(stripe_invoice_id='test_sent_invoice_id')
        self.assertEqual(invoice.status, Invoice.SENT)
        stripe_customer_id = 'cus_123'
        assert stripe_customer_id is not None
        [invoice0, invoice1] = iter(stripe.Invoice.list(customer=stripe_customer_id))
        self.assertIsNotNone(invoice0.status_transitions.finalized_at)
        [invoice_item0] = iter(invoice0.lines)
        invoice_item_params = {
            'amount': 25 * 80 * 100,
            'description': 'Zulip Business - renewal',
            'quantity': 25,
            'period': {
                'start': datetime_to_timestamp(self.next_year),
                'end': datetime_to_timestamp(add_months(self.next_year, 1)),
            },
        }
        for key, value in invoice_item_params.items():
            self.assertEqual(invoice_item0[key], value)
        self.execute_remote_billing_authentication_flow(hamlet, expect_tos=False)
        with time_machine.travel(self.now + timedelta(days=1), tick=False), patch(
            'corporate.lib.stripe.customer_has_credit_card_as_default_payment_method',
            return_value=False,
        ), patch(
            'stripe.Customer.retrieve',
            return_value=Mock(id=stripe_customer_id, email=hamlet.delivery_email),
        ), patch(
            'stripe.Invoice.retrieve',
            return_value=MagicMock(),
        ):
            response = self.client_get(f'{self.billing_session.billing_base_url}/billing/', subdomain='selfhosting')
        for substring in ['Zulip Basic', hamlet.delivery_email, 'Annual', 'This is a fixed-price plan', 'You will be contacted by Zulip Sales']:
            self.assert_in_response(substring, response)

    @responses.activate
    @mock_stripe(tested_timestamp_fields=['created'])
    def test_switch_from_annual_plan_to_monthly_plan_for_automatic_license_management(
        self, *mocks: Any
    ) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        self.add_card_and_upgrade(user, schedule='annual')
        annual_plan = CustomerPlan.objects.first()
        assert annual_plan is not None
        self.assertEqual(annual_plan.automanage_licenses, True)
        self.assertEqual(annual_plan.billing_schedule, CustomerPlan.BILLING_SCHEDULE_ANNUAL)
        stripe_customer_id = Customer.objects.get(realm=user.realm).id
        new_plan = CustomerPlan.objects.first()
        assert new_plan is not None
        with self.assertLogs('corporate.stripe', 'INFO') as m, time_machine.travel(self.now, tick=False):
            response = self.client_billing_patch(
                '/billing/plan', {'status': CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE}
            )
            expected_log = (
                f'INFO:corporate.stripe:Change plan status: Customer.id: {stripe_customer_id}, '
                f'CustomerPlan.id: {new_plan.id}, status: {CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE}'
            )
            self.assertEqual(m.output[0], expected_log)
            self.assert_json_success(response)
        annual_plan.refresh_from_db()
        self.assertEqual(annual_plan.status, CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE)
        self.assertEqual(annual_plan.next_invoice_date, annual_plan_end_date)
        billing_session = RealmBillingSession(user=user, realm=user.realm)
        with patch(
            'corporate.lib.stripe.get_latest_seat_count', return_value=20
        ):
            billing_session.update_license_ledger_if_needed(self.now)
        self.assertEqual(LicenseLedger.objects.filter(plan=annual_plan).count(), 2)
        with time_machine.travel(self.next_month, tick=False):
            send_server_data_to_push_bouncer(consider_usage_statistics=False)
        invoice_plans_as_needed(self.next_month + timedelta(days=1))
        self.assertEqual(LicenseLedger.objects.filter(plan=annual_plan).count(), 2)
        billing_session = RealmBillingSession(user=user, realm=user.realm)
        with patch(
            'corporate.lib.stripe.get_latest_seat_count', return_value=30
        ):
            billing_session.update_license_ledger_if_needed(self.next_month + timedelta(days=1))
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertEqual(plan.invoicing_status, CustomerPlan.INVOICING_STATUS_DONE)
        self.assertEqual(plan.invoiced_through, LicenseLedger.objects.filter(plan=plan).last())
        self.assertEqual(plan.next_invoice_date, add_months(self.next_month, 1))
        self.assertEqual(plan.billing_cycle_anchor, self.next_month)
        plan.refresh_from_db()
        self.assertEqual(plan.invoicing_status, CustomerPlan.INVOICING_STATUS_DONE)
        self.assertEqual(plan.next_invoice_date, add_months(self.next_month, 1))
        self.assertEqual(plan.billing_cycle_anchor, self.next_month)

    @patch('corporate.lib.stripe.billing_logger.info')
    def test_deactivate_realm_with_push_notification_service(
        self, mock_: MagicMock
    ) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        with time_machine.travel(self.now, tick=False):
            self.local_upgrade(self.seat_count, True, CustomerPlan.BILLING_SCHEDULE_ANNUAL, True, False)
        do_deactivate_realm(get_realm('zulip'), acting_user=None, deactivation_reason='owner_request', email_owners=False)
        plan = CustomerPlan.objects.first()
        assert plan is not None
        self.assertTrue(get_realm('zulip').deactivated)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_LIMITED)
        self.assertEqual(plan.status, CustomerPlan.ENDED)
        self.assertEqual(plan.invoiced_through, LicenseLedger.objects.first())
        self.assertIsNone(plan.next_invoice_date)
        do_reactivate_realm(get_realm('zulip'))
        self.login_user(user)
        response = self.client_get('/billing/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual('/plans/', response['Location'])
        with patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_month)
        mocked.assert_not_called()
        with patch('corporate.lib.stripe.BillingSession.invoice_plan') as mocked:
            invoice_plans_as_needed(self.next_year)
        mocked.assert_not_called()

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_upgrade_to_fixed_price_plus_plan(self, *mocks: Any) -> None:
        iago = self.example_user('iago')
        hamlet = self.example_user('hamlet')
        realm = get_realm('zulip')
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
        billing_session = RealmBillingSession(user=hamlet, realm=realm)
        next_billing_cycle = billing_session.get_next_billing_cycle(plan)
        plan_end_date_string = next_billing_cycle.strftime('%Y-%m-%d')
        plan_end_date = datetime.strptime(plan_end_date_string, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        self.logout()
        self.login_user(iago)
        result = self.client_post(
            '/activity/remote/support',
            {'remote_realm_id': f'{self.remote_realm.id}', 'fixed_price': 360},
        )
        self.assert_in_success_response(
            [f'Fixed price Zulip Cloud Plus plan scheduled to start on {plan_end_date_string}.'],
            result,
        )
        plan.refresh_from_db()
        self.assertEqual(plan.status, CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END)
        self.assertEqual(plan.next_invoice_date, plan_end_date)
        new_plan = CustomerPlan.objects.filter(fixed_price__isnull=False).first()
        assert new_plan is not None
        self.assertEqual(new_plan.next_invoice_date, plan_end_date)
        self.assertEqual(new_plan.invoicing_status, CustomerPlan.INVOICING_STATUS_INITIAL_INVOICE_TO_BE_SENT)
        self.assertEqual(new_plan.billing_cycle_anchor, plan_end_date)
        self.assertEqual(new_plan.next_invoice_date, plan_end_date)
        self.assertEqual(new_plan.invoicing_status, CustomerPlan.INVOICING_STATUS_INITIAL_INVOICE_TO_BE_SENT)
        self.assertEqual(new_plan.billing_cycle_anchor, plan_end_date)
        self.assertEqual(new_plan.next_invoice_date, plan_end_date)
        self.assertEqual(new_plan.invoicing_status, CustomerPlan.INVOICING_STATUS_INITIAL_INVOICE_TO_BE_SENT)
        self.assertEqual(new_plan.tier, CustomerPlan.TIER_SELF_HOSTED_BUSINESS)
        self.assertEqual(new_plan.status, CustomerPlan.NEVER_STARTED)
        self.assertEqual(
            new_plan.fixed_price, 360 * 100
        )  # assuming fixed_price is in cents
        self.assertIsNone(new_plan.price_per_license)
        self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_PLUS)
        [invoice0] = iter(stripe.Invoice.list(customer=plan.customer.stripe_customer_id))
        [invoice_item0, invoice_item1] = iter(
            invoice0.lines
        )
        invoice_item_params = {
            'amount': 5 * 7366,
            'description': 'Additional license (Feb 1, 2013 - Feb 1, 2014)',
            'quantity': 5,
            'period': {
                'start': datetime_to_timestamp(planned_cycle_start),
                'end': datetime_to_timestamp(planned_cycle_end),
            },
        }
        for key, value in invoice_item_params.items():
            self.assertEqual(invoice_item0[key], value)
        invoice_plans_as_needed(plan_end_date)
        self.assert_length(outbox, messages_count + 1)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_downgrade_at_end_of_free_trial(self, *mocks: Any) -> None:
        user = self.example_user('hamlet')
        self.login_user(user)
        free_trial_end_date = self.now + timedelta(days=60)
        with self.settings(CLOUD_FREE_TRIAL_DAYS=60):
            with time_machine.travel(self.now, tick=False):
                self.upgrade(invoice=True)
            plan = CustomerPlan.objects.get()
            assert plan is not None
            self.assertEqual(plan.next_invoice_date, free_trial_end_date)
            self.assertEqual(get_realm('zulip').plan_type, Realm.PLAN_TYPE_STANDARD)
            self.assertEqual(plan.status, CustomerPlan.FREE_TRIAL)
            self.assertEqual(plan.licenses(), self.seat_count)
            self.assertEqual(plan.licenses_at_next_renewal(), self.seat_count)
            billing_session = RealmBillingSession(user=user, realm=get_realm('zulip'))
            with patch('corporate.lib.stripe.get_latest_seat_count', return_value=20):
                billing_session.update_license_ledger_if_needed(self.now)
            self.assertEqual(LicenseLedger.objects.order_by('-id').values_list('licenses', 'licenses_at_next_renewal').first(), (20, 20))
            self.login_user(user)
            with patch('corporate.lib.stripe.get_latest_seat_count', return_value=20), patch('corporate.lib.stripe.some_other_method'):
                # Implement appropriately
                pass
            billing_session.set_required_plan_tier(CustomerPlan.TIER_SELF_HOSTED_STANDARD)
            billing_session.attach_discount_to_customer(monthly_discounted_price=0, annual_discounted_price=0)
            billing_session.update_license_ledger_if_needed(plan, self.now)
            billing_session.write_to_audit_log()
            # Additional test assertions can be added here

    @patch('some.module.function')
    def test_some_other_case(self, mocked_function: Callable[..., Any]) -> None:
        # Implement the method if needed
        pass

    # Additional test methods can be added here


class TestRemoteBillingWriteAuditLog(StripeTestCase):
    @dataclass
    class Row:
        pass

    def test_write_audit_log(self) -> None:
        support_admin = self.example_user('iago')
        server_uuid = str(uuid.uuid4())
        remote_server = RemoteZulipServer.objects.create(
            uuid=server_uuid,
            api_key='magic_secret_api_key',
            hostname='demo.example.com',
            contact_email='email@example.com',
        )
        realm_uuid = str(uuid.uuid4())
        remote_realm = RemoteRealm.objects.create(
            server=remote_server,
            uuid=realm_uuid,
            uuid_owner_secret='dummy-owner-secret',
            host='dummy-hostname',
            realm_date_created=timezone_now(),
        )
        remote_realm_billing_user = RemoteRealmBillingUser.objects.create(
            remote_realm=remote_realm, email='admin@example.com', user_uuid=uuid.uuid4()
        )
        remote_server_billing_user = RemoteServerBillingUser.objects.create(
            remote_server=remote_server, email='admin@example.com'
        )
        event_time = timezone_now()

        def assert_audit_log(
            audit_log: Any,
            acting_remote_user: Optional[Any],
            acting_support_user: Optional[Any],
            event_type: Any,
            event_time: datetime,
        ) -> None:
            self.assertEqual(audit_log.event_type, event_type)
            self.assertEqual(audit_log.event_time, event_time)
            self.assertEqual(audit_log.acting_remote_user, acting_remote_user)
            self.assertEqual(audit_log.acting_support_user, acting_support_user)

        for session_class, audit_log_class, remote_object, remote_user in [
            (RemoteRealmBillingSession, RemoteRealmAuditLog, remote_realm, remote_realm_billing_user),
            (RemoteServerBillingSession, RemoteZulipServerAuditLog, remote_server, remote_server_billing_user),
        ]:
            audit_log_model = cast(
                type[RemoteRealmAuditLog] | type[RemoteZulipServerAuditLog], audit_log_class
            )
            assert isinstance(remote_user, RemoteRealmBillingUser | RemoteServerBillingUser)
            session = session_class(remote_object)
            session.write_to_audit_log(event_type=BillingSessionEventType.CUSTOMER_PLAN_CREATED, event_time=event_time)
            audit_log = audit_log_model.objects.latest('id')
            assert_audit_log(
                audit_log,
                None,
                None,
                AuditLogEventType.CUSTOMER_PLAN_CREATED,
                event_time,
            )
            session = session_class(remote_object, remote_billing_user=remote_user)
            session.write_to_audit_log(event_type=BillingSessionEventType.CUSTOMER_PLAN_CREATED, event_time=event_time)
            audit_log = audit_log_model.objects.latest('id')
            assert_audit_log(
                audit_log,
                remote_user,
                None,
                AuditLogEventType.CUSTOMER_PLAN_CREATED,
                event_time,
            )
            session = session_class(remote_object, remote_billing_user=None, support_staff=support_admin)
            session.write_to_audit_log(event_type=BillingSessionEventType.CUSTOMER_PLAN_CREATED, event_time=event_time)
            audit_log = audit_log_model.objects.latest('id')
            assert_audit_log(
                audit_log,
                None,
                support_admin,
                AuditLogEventType.CUSTOMER_PLAN_CREATED,
                event_time,
            )

    # Additional test methods can be added here


class TestRemoteServerBillingSession(StripeTestCase):
    def test_get_audit_log_error(self) -> None:
        server_uuid = str(uuid.uuid4())
        remote_server = RemoteZulipServer.objects.create(
            uuid=server_uuid,
            api_key='magic_secret_api_key',
            hostname='demo.example.com',
            contact_email='email@example.com',
        )
        billing_session = RemoteServerBillingSession(remote_server=remote_server)
        fake_audit_log = cast(BillingSessionEventType, 0)
        with self.assertRaisesRegex(
            BillingSessionAuditLogEventError, 'Unknown audit log event type: 0'
        ):
            billing_session.get_audit_log_event(event_type=fake_audit_log)

    def test_get_customer(self) -> None:
        server_uuid = str(uuid.uuid4())
        remote_server = RemoteZulipServer.objects.create(
            uuid=server_uuid,
            api_key='magic_secret_api_key',
            hostname='demo.example.com',
            contact_email='email@example.com',
        )
        billing_session = RemoteServerBillingSession(remote_server=remote_server)
        customer = billing_session.get_customer()
        self.assertEqual(customer, None)
        customer = Customer.objects.create(remote_server=remote_server, stripe_customer_id='cus_12345')
        self.assertEqual(billing_session.get_customer(), customer)


class TestRemoteBillingWriteAuditLog(StripeTestCase):
    pass


class TestRemoteBillingFlow(StripeTestCase, RemoteRealmBillingTestCase):
    pass


class TestRemoteServerBillingFlow(StripeTestCase, RemoteServerTestCase):
    pass


class InvoiceTest(StripeTestCase):
    pass

    def test_invoice_initial_remote_realm_upgrade(self) -> None:
        self.login('hamlet')
        hamlet = self.example_user('hamlet')
        self.add_mock_response()
        with time_machine.travel(self.now, tick=False):
            send_server_data_to_push_bouncer(consider_usage_statistics=False)
        self.execute_remote_billing_authentication_flow(hamlet)
        with time_machine.travel(self.now, tick=False):
            stripe_customer = self.add_card_and_upgrade(tier=CustomerPlan.TIER_SELF_HOSTED_BASIC, schedule='monthly')
        [invoice0] = iter(stripe.Invoice.list(customer=stripe_customer.id))
        [invoice_item0, invoice_item1] = iter(invoice0.lines)
        invoice_item_params = {
            'amount': -2000,
            'description': '$20.00/month new customer discount',
            'quantity': 1,
        }
        for key, value in invoice_item_params.items():
            self.assertEqual(invoice_item0[key], value)
        invoice_item_params = {
            'amount': 25 * 3.5 * 100,
            'description': 'Zulip Basic',
            'quantity': 25,
        }
        for key, value in invoice_item_params.items():
            self.assertEqual(invoice_item1[key], value)
        self.assertEqual(invoice0.total, 25 * 3.5 * 100 - 2000)
        self.assertEqual(invoice0.status, 'paid')

    def test_upgrade_realm_plan_for_automanage_licenses(self) -> None:
        # Implement the method if needed
        pass


class RequiresBillingAccessTest(StripeTestCase):

    @override
    def setUp(self) -> None:
        super().setUp()
        desdemona = self.example_user('desdemona')
        desdemona.role = UserProfile.ROLE_REALM_OWNER
        desdemona.save(update_fields=['role'])

    def test_json_endpoints_permissions(self) -> None:
        guest = self.example_user('polonius')
        member = self.example_user('othello')
        realm_admin = self.example_user('iago')
        billing_admin = self.example_user('hamlet')
        billing_admin.is_billing_admin = True
        billing_admin.save(update_fields=['is_billing_admin'])
        tested_endpoints: set[str] = set()

        def check_users_cant_access(
            users: Sequence[UserProfile],
            error_message: str,
            url: str,
            method: Literal['POST', 'GET', 'PATCH'],
            data: dict[str, Any],
        ) -> None:
            tested_endpoints.add(url)
            for user in users:
                self.login_user(user)
                if method == 'POST':
                    client_func = self.client_post
                elif method == 'GET':
                    client_func = self.client_get
                else:
                    client_func = self.client_patch
                result = client_func(url, data, content_type='application/json')
                self.assert_json_error_contains(result, error_message)

        check_users_cant_access(
            [guest],
            'Must be an organization member',
            '/json/billing/upgrade',
            'POST',
            {},
        )
        check_users_cant_access(
            [guest],
            'Must be an organization member',
            '/json/billing/sponsorship',
            'POST',
            {},
        )
        check_users_cant_access(
            [guest, member, realm_admin],
            'Must be a billing administrator or an organization owner',
            '/json/billing/plan',
            'PATCH',
            {},
        )
        check_users_cant_access(
            [guest, member, realm_admin],
            'Must be a billing administrator or an organization owner',
            '/json/billing/session/start_card_update_session',
            'POST',
            {},
        )
        check_users_cant_access(
            [guest],
            'Must be an organization member',
            '/json/upgrade/session/start_card_update_session',
            'POST',
            {},
        )
        check_users_cant_access(
            [guest],
            'Must be an organization member',
            '/json/billing/event/status',
            'GET',
            {},
        )
        reverse_dict = get_resolver('corporate.urls').reverse_dict
        json_endpoints = {
            pat
            for name in reverse_dict
            for matches, pat, defaults, converters in reverse_dict.getlist(name)
            if pat.startswith('json/') and not pat.startswith(('json/realm/', 'json/server/'))
        }
        self.assertEqual(json_endpoints, tested_endpoints)

    @mock_stripe(tested_timestamp_fields=['created'])
    def test_billing_page_permissions(self, *mocks: Any) -> None:
        self.login_user(self.example_user('polonius'))
        response = self.client_get('/upgrade/', follow=True)
        self.assertEqual(response.status_code, 404)
        non_owner_non_billing_admin = self.example_user('othello')
        self.login_user(non_owner_non_billing_admin)
        response = self.client_get('/billing/')
        self.assert_in_success_response(['You must be an organization owner or a billing administrator to view this page.'], response)
        self.add_card_and_upgrade(non_owner_non_billing_admin)
        response = self.client_get('/billing/')
        self.assert_in_success_response(['Zulip Cloud Standard'], response)
        self.assert_not_in_success_response(['You must be an organization owner or a billing administrator to view this page.'], response)
        desdemona = self.example_user('desdemona')
        desdemona.role = UserProfile.ROLE_REALM_OWNER
        desdemona.save(update_fields=['role'])
        self.login_user(self.example_user('desdemona'))
        response = self.client_get('/billing/')
        self.assert_in_success_response(['Zulip Cloud Standard'], response)
        self.login_user(self.example_user('cordelia'))
        response = self.client_get('/billing/')
        self.assert_in_success_response(['You must be an organization owner or a billing administrator'], response)

    class BillingHelpersTest(ZulipTestCase):
        pass

    @activate_push_notification_service()
    class TestRemoteServerBillingFlow(StripeTestCase, RemoteServerTestCase):
        pass

    class BillingHelpersTest(ZulipTestCase):
        def test_compute_plan_parameters(self) -> None:
            realm = get_realm('zulip')
            customer_with_discount = Customer.objects.create(
                realm=get_realm('lear'),
                monthly_discounted_price=600,
                annual_discounted_price=6000,
                required_plan_tier=CustomerPlan.TIER_CLOUD_STANDARD,
            )
            customer_no_discount = Customer.objects.create(realm=get_realm('zulip'))
            test_cases: list[
                tuple[
                    tuple[str, str, Optional[Customer]],
                    tuple[datetime, datetime, datetime, int],
                ]
            ] = [
                (
                    (CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_ANNUAL, None),
                    (self.now, self.next_month, self.next_year, 8000),
                ),
                (
                    (
                        CustomerPlan.TIER_CLOUD_STANDARD,
                        CustomerPlan.BILLING_SCHEDULE_ANNUAL,
                        customer_with_discount,
                    ),
                    (self.now, self.next_month, self.next_year, 6000),
                ),
                (
                    (
                        CustomerPlan.TIER_CLOUD_STANDARD,
                        CustomerPlan.BILLING_SCHEDULE_ANNUAL,
                        customer_no_discount,
                    ),
                    (self.now, self.next_month, self.next_year, 8000),
                ),
                (
                    (
                        CustomerPlan.TIER_CLOUD_PLUS,
                        CustomerPlan.BILLING_SCHEDULE_ANNUAL,
                        customer_with_discount,
                    ),
                    (self.now, self.next_month, self.next_year, 12000),
                ),
                (
                    (CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_MONTHLY, None),
                    (self.now, self.next_month, self.next_month, 800),
                ),
                (
                    (
                        CustomerPlan.TIER_CLOUD_STANDARD,
                        CustomerPlan.BILLING_SCHEDULE_MONTHLY,
                        customer_with_discount,
                    ),
                    (self.now, self.next_month, self.next_month, 600),
                ),
                (
                    (
                        CustomerPlan.TIER_CLOUD_STANDARD,
                        CustomerPlan.BILLING_SCHEDULE_MONTHLY,
                        customer_no_discount,
                    ),
                    (self.now, self.next_month, self.next_month, 800),
                ),
                (
                    (
                        CustomerPlan.TIER_CLOUD_PLUS,
                        CustomerPlan.BILLING_SCHEDULE_MONTHly,
                        customer_with_discount,
                    ),
                    (self.now, self.next_month, self.next_month, 1200),
                ),
            ]
            with time_machine.travel(anchor, tick=False):
                for (tier, billing_schedule, customer), output in test_cases:
                    output_ = compute_plan_parameters(tier, billing_schedule, customer)
                    self.assertEqual(output_, output)

        def test_get_price_per_license(self) -> None:
            standard_discounted_customer = Customer.objects.create(
                realm=get_realm('lear'),
                monthly_discounted_price=400,
                annual_discounted_price=4000,
                required_plan_tier=CustomerPlan.TIER_CLOUD_STANDARD,
            )
            plus_discounted_customer = Customer.objects.create(
                realm=get_realm('zulip'),
                monthly_discounted_price=600,
                annual_discounted_price=6000,
                required_plan_tier=CustomerPlan.TIER_SELF_HOSTED_PLUS,
            )
            self.assertEqual(
                get_price_per_license(
                    CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_ANNUAL
                ),
                8000,
            )
            self.assertEqual(
                get_price_per_license(
                    CustomerPlan.TIER_CLOUD_STANDARD, CustomerPlan.BILLING_SCHEDULE_MONTHLY
                ),
                800,
            )
            self.assertEqual(
                get_price_per_license(
                    CustomerPlan.TIER_CLOUD_STANDARD,
                    CustomerPlan.BILLING_SCHEDULE_MONTHLY,
                    standard_discounted_customer,
                ),
                400,
            )
            self.assertEqual(
                get_price_per_license(
                    CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_ANNUAL
                ),
                12000,
            )
            self.assertEqual(
                get_price_per_license(
                    CustomerPlan.TIER_CLOUD_PLUS, CustomerPlan.BILLING_SCHEDULE_MONTHLY
                ),
                1200,
            )
            self.assertEqual(
                get_price_per_license(
                    CustomerPlan.TIER_CLOUD_PLUS,
                    CustomerPlan.BILLING_SCHEDULE_MONTHLY,
                    standard_discounted_customer,
                ),
                1200,
            )
            self.assertEqual(
                get_price_per_license(
                    CustomerPlan.TIER_SELF_HOSTED_PLUS,
                    CustomerPlan.BILLING_SCHEDULE_MONTHLY,
                    plus_discounted_customer,
                ),
                600,
            )
            with self.assertRaisesRegex(
                InvalidBillingScheduleError, 'Unknown billing_schedule: 1000'
            ):
                get_price_per_license(
                    CustomerPlan.TIER_CLOUD_STANDARD, 1000
                )
            with self.assertRaisesRegex(InvalidTierError, 'Unknown tier: 4'):
                get_price_per_license(
                    CustomerPlan.TIER_SELF_HOSTED_ENTERPRISE, CustomerPlan.BILLING_SCHEDULE_ANNUAL
                )

        def test_get_plan_renewal_or_end_date(self) -> None:
            realm = get_realm('zulip')
            customer = Customer.objects.create(realm=realm, stripe_customer_id='cus_12345')
            plan = CustomerPlan.objects.create(
                customer=customer,
                status=CustomerPlan.ACTIVE,
                billing_cycle_anchor=timezone_now(),
                billing_schedule=CustomerPlan.BILLING_SCHEDULE_MONTHLY,
                tier=CustomerPlan.TIER_CLOUD_STANDARD,
            )
            renewal_date = get_plan_renewal_or_end_date(plan, timezone_now())
            self.assertEqual(renewal_date, add_months(timezone_now(), 1))
            plan_end_date = add_months(timezone_now(), 1) - timedelta(days=2)
            plan.end_date = plan_end_date
            plan.save(update_fields=['end_date'])
            renewal_date = get_plan_renewal_or_end_date(plan, timezone_now())
            self.assertEqual(renewal_date, plan_end_date)

        def test_deactivate_remote_server(self) -> None:
            realm = get_realm('zulip')
            user = self.example_user('hamlet')
            self.login_user(user)
            remote_server = RemoteZulipServer.objects.create(
                uuid=str(uuid.uuid4()),
                api_key='magic_secret_api_key',
                hostname='demo.example.com',
                contact_email='email@example.com',
            )
            remote_realm = RemoteRealm.objects.create(
                server=remote_server,
                uuid=str(uuid.uuid4()),
                uuid_owner_secret='dummy-owner-secret',
                host='dummy-hostname',
                realm_date_created=timezone_now(),
            )
            customer = Customer.objects.create(remote_realm=remote_realm, stripe_customer_id='cus_12345')
            billing_session = RemoteRealmBillingSession(remote_realm=remote_realm, support_staff=user)
            billing_session.write_to_audit_log(event_type=BillingSessionEventType.CUSTOMER_PLAN_CREATED, event_time=self.now)
            do_deactivate_remote_server(remote_server, billing_session)
            remote_server.refresh_from_db()
            remote_realm.refresh_from_db()
            self.assertTrue(remote_server.deactivated)
            self.assertEqual(remote_realm.plan_type, RemoteRealm.PLAN_TYPE_SELF_MANAGED_LEGACY)
            # Add additional assertions as needed

    class TestSupportBillingHelpers(StripeTestCase):
        pass
