#!/usr/bin/env python3
from datetime import timedelta
from typing import Optional, Any, TYPE_CHECKING
from unittest import mock
import responses
import time_machine
from django.conf import settings
from django.utils.timezone import now as timezone_now
from typing_extensions import override
from corporate.lib.remote_billing_util import (
    REMOTE_BILLING_SESSION_VALIDITY_SECONDS,
    LegacyServerIdentityDict,
    RemoteBillingIdentityDict,
    RemoteBillingUserDict,
)
from corporate.lib.stripe import RemoteRealmBillingSession, RemoteServerBillingSession, add_months
from corporate.models import (
    CustomerPlan,
    LicenseLedger,
    get_current_plan_by_customer,
    get_customer_by_remote_realm,
    get_customer_by_remote_server,
)
from corporate.views.remote_billing_page import generate_confirmation_link_for_server_deactivation
from zerver.lib.exceptions import RemoteRealmServerMismatchError
from zerver.lib.rate_limiter import RateLimitedIPAddr
from zerver.lib.remote_server import send_server_data_to_push_bouncer
from zerver.lib.send_email import FromAddress
from zerver.lib.test_classes import BouncerTestCase
from zerver.lib.test_helpers import activate_push_notification_service, ratelimit_rule
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.models import Realm, UserProfile
from zerver.models.realms import get_realm
from zilencer.models import (
    PreregistrationRemoteRealmBillingUser,
    PreregistrationRemoteServerBillingUser,
    RateLimitedRemoteZulipServer,
    RemoteRealm,
    RemoteRealmBillingUser,
    RemoteServerBillingUser,
    RemoteZulipServer,
)

if TYPE_CHECKING:
    from django.test.client import _MonkeyPatchedWSGIResponse as TestHttpResponse

@activate_push_notification_service()
class RemoteRealmBillingTestCase(BouncerTestCase):
    def execute_remote_billing_authentication_flow(
        self,
        user: UserProfile,
        next_page: Optional[str] = None,
        expect_tos: bool = True,
        confirm_tos: bool = True,
        first_time_login: bool = True,
        return_without_clicking_confirmation_link: bool = False,
        return_from_auth_url: bool = False,
    ) -> "TestHttpResponse":
        now = timezone_now()
        self_hosted_billing_url: str = '/self-hosted-billing/'
        if next_page is not None:
            self_hosted_billing_url += f'?next_page={next_page}'
        with time_machine.travel(now, tick=False):
            result = self.client_get(self_hosted_billing_url)
        self.assertEqual(result.status_code, 302)
        self.assertIn('http://selfhosting.testserver/remote-billing-login/', result['Location'])
        signed_auth_url: str = result['Location']
        signed_access_token: str = signed_auth_url.split('/')[-1]
        with time_machine.travel(now, tick=False):
            result = self.client_get(signed_auth_url, subdomain='selfhosting')
        if return_from_auth_url:
            return result
        if first_time_login:
            self.assertFalse(RemoteRealmBillingUser.objects.filter(user_uuid=user.uuid).exists())
            self.assertEqual(result.status_code, 200)
            self.assert_in_success_response(['Enter email'], result)
            self.assert_in_success_response([user.realm.host], result)
            self.assert_in_success_response([f'action="/remote-billing-login/{signed_access_token}/confirm/"'], result)
            with time_machine.travel(now, tick=False):
                result = self.client_post(
                    f'/remote-billing-login/{signed_access_token}/confirm/', {'email': user.delivery_email}, subdomain='selfhosting'
                )
            if result.status_code == 429:
                return result
            self.assertEqual(result.status_code, 200)
            self.assert_in_success_response(
                ['To finish logging in, check your email account (', ') for a confirmation email from Zulip.', user.delivery_email], result
            )
            confirmation_url: str = self.get_confirmation_url_from_outbox(
                user.delivery_email,
                url_pattern=f'{settings.SELF_HOSTING_MANAGEMENT_SUBDOMAIN}.{settings.EXTERNAL_HOST}(\\S+)',
                email_body_contains='confirm your email and log in to Zulip plan management',
            )
            if return_without_clicking_confirmation_link:
                return result
            with time_machine.travel(now, tick=False):
                result = self.client_get(confirmation_url, subdomain='selfhosting')
            remote_billing_user = RemoteRealmBillingUser.objects.latest('id')
            self.assertEqual(remote_billing_user.user_uuid, user.uuid)
            self.assertEqual(remote_billing_user.email, user.delivery_email)
            prereg_user = PreregistrationRemoteRealmBillingUser.objects.latest('id')
            self.assertEqual(prereg_user.created_user, remote_billing_user)
            self.assertEqual(remote_billing_user.date_joined, now)
            self.assertEqual(result.status_code, 302)
            self.assertTrue(result['Location'].startswith('/remote-billing-login/'))
            result = self.client_get(result['Location'], subdomain='selfhosting')
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response(['Log in to Zulip plan management'], result)
        self.assert_in_success_response([user.realm.host], result)
        params: dict[str, Any] = {}
        if expect_tos:
            self.assert_in_success_response(['I agree', 'Terms of Service'], result)
        if confirm_tos:
            params = {'tos_consent': 'true'}
        with time_machine.travel(now, tick=False):
            result = self.client_post(signed_auth_url, params, subdomain='selfhosting')
        if result.status_code >= 400:
            return result
        remote_billing_user = RemoteRealmBillingUser.objects.get(user_uuid=user.uuid)
        identity_dict: RemoteBillingIdentityDict = RemoteBillingIdentityDict(
            user=RemoteBillingUserDict(
                user_email=user.delivery_email,
                user_uuid=str(user.uuid),
                user_full_name=user.full_name,
            ),
            remote_server_uuid=str(self.server.uuid),
            remote_realm_uuid=str(user.realm.uuid),
            remote_billing_user_id=remote_billing_user.id,
            authenticated_at=datetime_to_timestamp(now),
            uri_scheme='http://',
            next_page=next_page,
        )
        self.assertEqual(self.client.session['remote_billing_identities'][f'remote_realm:{user.realm.uuid!s}'], identity_dict)
        self.assertEqual(remote_billing_user.last_login, now)
        return result

@activate_push_notification_service()
class SelfHostedBillingEndpointBasicTest(RemoteRealmBillingTestCase):
    @responses.activate
    def test_self_hosted_billing_endpoints(self) -> None:
        self.login('hamlet')
        for url in ['/self-hosted-billing/', '/json/self-hosted-billing', '/self-hosted-billing/not-configured/']:
            result = self.client_get(url)
            self.assert_json_error(result, 'Must be an organization owner')
        self.login('desdemona')
        self.add_mock_response()
        self_hosted_billing_url: str = '/self-hosted-billing/'
        self_hosted_billing_json_url: str = '/json/self-hosted-billing'
        with self.settings(ZULIP_SERVICE_PUSH_NOTIFICATIONS=False):
            with self.settings(CORPORATE_ENABLED=True):
                result = self.client_get(self_hosted_billing_url)
                self.assertEqual(result.status_code, 404)
                self.assert_in_response('Page not found (404)', result)
            with self.settings(CORPORATE_ENABLED=False):
                result = self.client_get(self_hosted_billing_url)
                self.assertEqual(result.status_code, 302)
                redirect_url: str = result['Location']
                self.assertEqual(redirect_url, '/self-hosted-billing/not-configured/')
                with self.assertLogs('django.request'):
                    result = self.client_get(redirect_url)
                    self.assert_in_response('This server is not configured to use push notifications.', result)
            with self.settings(CORPORATE_ENABLED=True):
                result = self.client_get(self_hosted_billing_json_url)
                self.assert_json_error(result, "Server doesn't use the push notification service", 404)
            with self.settings(CORPORATE_ENABLED=False):
                result = self.client_get(self_hosted_billing_json_url)
                self.assert_json_success(result)
                redirect_url = result.json()['billing_access_url']
                self.assertEqual(redirect_url, '/self-hosted-billing/not-configured/')
                with self.assertLogs('django.request'):
                    result = self.client_get(redirect_url)
                    self.assert_in_response('This server is not configured to use push notifications.', result)
        with mock.patch('zerver.views.push_notifications.send_to_push_bouncer', side_effect=RemoteRealmServerMismatchError):
            result = self.client_get(self_hosted_billing_url)
            self.assertEqual(result.status_code, 403)
            self.assert_in_response('Unexpected Zulip server registration', result)
            result = self.client_get(self_hosted_billing_json_url)
            self.assert_json_error(
                result,
                'Your organization is registered to a different Zulip server. Please contact Zulip support for assistance in resolving this issue.',
                403,
            )
        result = self.client_get(self_hosted_billing_url)
        self.assertEqual(result.status_code, 302)
        self.assertIn('http://selfhosting.testserver/remote-billing-login/', result['Location'])
        result = self.client_get(self_hosted_billing_json_url)
        self.assert_json_success(result)
        data: dict[str, Any] = result.json()
        self.assertEqual(sorted(data.keys()), ['billing_access_url', 'msg', 'result'])
        self.assertIn('http://selfhosting.testserver/remote-billing-login/', data['billing_access_url'])

@activate_push_notification_service()
class RemoteBillingAuthenticationTest(RemoteRealmBillingTestCase):
    def test_self_hosted_config_error_page(self) -> None:
        self.login('desdemona')
        with self.settings(CORPORATE_ENABLED=False, ZULIP_SERVICE_PUSH_NOTIFICATIONS=False), self.assertLogs('django.request'):
            result = self.client_get('/self-hosted-billing/not-configured/')
            self.assertEqual(result.status_code, 500)
            self.assert_in_response('This server is not configured to use push notifications.', result)
        with self.settings(CORPORATE_ENABLED=False):
            result = self.client_get('/self-hosted-billing/not-configured/')
            self.assertEqual(result.status_code, 404)
        with self.settings(CORPORATE_ENABLED=True, ZULIP_SERVICE_PUSH_NOTIFICATIONS=False):
            result = self.client_get('/self-hosted-billing/not-configured/')
            self.assertEqual(result.status_code, 404)

    @responses.activate
    def test_remote_billing_authentication_flow(self) -> None:
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        realm: Realm = desdemona.realm
        self.add_mock_response()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        result = self.execute_remote_billing_authentication_flow(desdemona)
        self.assertEqual(result['Location'], f'/realm/{realm.uuid!s}/plans/')
        result = self.client_get(result['Location'], subdomain='selfhosting')
        self.assert_in_success_response(['showing-self-hosted', 'Retain full control'], result)

    @ratelimit_rule(10, 3, domain='sends_email_by_remote_server')
    @ratelimit_rule(10, 2, domain='sends_email_by_ip')
    @responses.activate
    def test_remote_billing_authentication_flow_rate_limited(self) -> None:
        RateLimitedIPAddr('127.0.0.1', domain='sends_email_by_ip').clear_history()
        RateLimitedRemoteZulipServer(self.server, domain='sends_email_by_remote_server').clear_history()
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        self.add_mock_response()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        for i in range(2):
            result = self.execute_remote_billing_authentication_flow(desdemona, return_without_clicking_confirmation_link=True)
            self.assertEqual(result.status_code, 200)
        result = self.execute_remote_billing_authentication_flow(desdemona, return_without_clicking_confirmation_link=True)
        self.assertEqual(result.status_code, 429)
        self.assert_in_response('You have exceeded the limit', result)
        RateLimitedIPAddr('127.0.0.1', domain='sends_email_by_ip').clear_history()
        result = self.execute_remote_billing_authentication_flow(desdemona, return_without_clicking_confirmation_link=True)
        self.assertEqual(result.status_code, 200)
        with self.assertLogs('zilencer.auth', 'WARN') as mock_log:
            result = self.execute_remote_billing_authentication_flow(desdemona, return_without_clicking_confirmation_link=True)
            self.assertEqual(result.status_code, 429)
            self.assert_in_response('Your server has exceeded the limit', result)
        self.assertEqual(
            mock_log.output,
            [f'WARNING:zilencer.auth:Remote server {self.server.hostname} {str(self.server.uuid)[:12]} exceeded rate limits on domain sends_email_by_remote_server'],
        )

    @responses.activate
    def test_remote_billing_authentication_flow_realm_not_registered(self) -> None:
        RemoteRealm.objects.all().delete()
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        realm: Realm = desdemona.realm
        self.add_mock_response()
        self.assertFalse(RemoteRealm.objects.filter(uuid=realm.uuid).exists())
        with mock.patch('zerver.views.push_notifications.send_server_data_to_push_bouncer', side_effect=send_server_data_to_push_bouncer) as m:
            result = self.execute_remote_billing_authentication_flow(desdemona)
        m.assert_called_once()
        self.assertTrue(RemoteRealm.objects.filter(uuid=realm.uuid).exists())
        self.assertEqual(result['Location'], f'/realm/{realm.uuid!s}/plans/')
        result = self.client_get(result['Location'], subdomain='selfhosting')
        self.assert_in_success_response(['showing-self-hosted', 'Retain full control'], result)

    @responses.activate
    def test_remote_billing_authentication_flow_tos_consent_failure(self) -> None:
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        self.add_mock_response()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        result = self.execute_remote_billing_authentication_flow(desdemona, expect_tos=True, confirm_tos=False)
        self.assert_json_error(result, 'You must accept the Terms of Service to proceed.')

    @responses.activate
    def test_remote_billing_authentication_flow_tos_consent_update(self) -> None:
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        self.add_mock_response()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        with self.settings(TERMS_OF_SERVICE_VERSION='1.0'):
            result = self.execute_remote_billing_authentication_flow(desdemona, expect_tos=True, confirm_tos=True)
        self.assertEqual(result.status_code, 302)
        remote_billing_user = RemoteRealmBillingUser.objects.last()
        assert remote_billing_user is not None
        self.assertEqual(remote_billing_user.user_uuid, desdemona.uuid)
        self.assertEqual(remote_billing_user.tos_version, '1.0')
        with self.settings(TERMS_OF_SERVICE_VERSION='2.0'):
            result = self.execute_remote_billing_authentication_flow(desdemona, expect_tos=True, confirm_tos=False, first_time_login=False)
            self.assert_json_error(result, 'You must accept the Terms of Service to proceed.')
            result = self.execute_remote_billing_authentication_flow(desdemona, expect_tos=True, confirm_tos=True, first_time_login=False)
        remote_billing_user.refresh_from_db()
        self.assertEqual(remote_billing_user.user_uuid, desdemona.uuid)
        self.assertEqual(remote_billing_user.tos_version, '2.0')

    @responses.activate
    def test_remote_billing_authentication_flow_expired_session(self) -> None:
        now = timezone_now()
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        realm: Realm = desdemona.realm
        self.add_mock_response()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        with time_machine.travel(now, tick=False):
            result = self.execute_remote_billing_authentication_flow(desdemona)
        self.assertEqual(result['Location'], f'/realm/{realm.uuid!s}/plans/')
        final_url: str = result['Location']
        with time_machine.travel(now + timedelta(seconds=1), tick=False):
            result = self.client_get(final_url, subdomain='selfhosting')
        self.assert_in_success_response(['showing-self-hosted', 'Retain full control'], result)
        with time_machine.travel(now + timedelta(seconds=REMOTE_BILLING_SESSION_VALIDITY_SECONDS + 1), tick=False):
            result = self.client_get(final_url, subdomain='selfhosting')
            self.assertEqual(result.status_code, 302)
            self.assertEqual(result['Location'], f'http://{desdemona.realm.host}/self-hosted-billing/?next_page=plans')
            result = self.execute_remote_billing_authentication_flow(
                desdemona,
                next_page='plans',
                expect_tos=False,
                confirm_tos=False,
                first_time_login=False,
            )
            self.assertEqual(result['Location'], f'/realm/{realm.uuid!s}/plans/')
            result = self.client_get(result['Location'], subdomain='selfhosting')
            self.assert_in_success_response(['showing-self-hosted', 'Retain full control'], result)

    @responses.activate
    def test_remote_billing_unauthed_access(self) -> None:
        now = timezone_now()
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        realm: Realm = desdemona.realm
        self.add_mock_response()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        result = self.client_get(f'/realm/{realm.uuid!s}/plans/', subdomain='selfhosting')
        self.assert_json_error(result, 'User not authenticated', 401)
        result = self.execute_remote_billing_authentication_flow(desdemona)
        self.assertEqual(result['Location'], f'/realm/{realm.uuid!s}/plans/')
        final_url: str = result['Location']
        result = self.client_get(final_url, subdomain='selfhosting')
        self.assertEqual(result.status_code, 200)
        RemoteRealm.objects.filter(uuid=realm.uuid).delete()
        with self.assertLogs('django.request', 'ERROR') as m, self.assertRaises(AssertionError):
            self.client_get(final_url, subdomain='selfhosting')
        self.assertIn('The remote realm is missing despite being in the RemoteBillingIdentityDict', m.output[0])
        with time_machine.travel(now + timedelta(seconds=REMOTE_BILLING_SESSION_VALIDITY_SECONDS + 30), tick=False), self.assertLogs('django.request', 'ERROR') as m, self.assertRaises(AssertionError):
            self.client_get(final_url, subdomain='selfhosting')
        self.assertIn('RemoteBillingIdentityExpiredError', m.output[0])
        self.assertIn('AssertionError', m.output[0])

    @responses.activate
    def test_remote_billing_authentication_flow_to_sponsorship_page(self) -> None:
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        realm: Realm = desdemona.realm
        self.add_mock_response()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        result = self.execute_remote_billing_authentication_flow(desdemona, 'sponsorship')
        self.assertEqual(result['Location'], f'/realm/{realm.uuid!s}/sponsorship/')
        result = self.client_get(result['Location'], subdomain='selfhosting')
        self.assert_in_success_response(['Request Zulip', 'sponsorship', 'Description of your organization'], result)

    @responses.activate
    def test_remote_billing_authentication_flow_to_upgrade_page(self) -> None:
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        realm: Realm = desdemona.realm
        self.add_mock_response()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        result = self.execute_remote_billing_authentication_flow(desdemona, 'upgrade')
        self.assertEqual(result['Location'], f'/realm/{realm.uuid!s}/upgrade/')
        with mock.patch('corporate.lib.stripe.has_stale_audit_log', return_value=False):
            result = self.client_get(result['Location'], subdomain='selfhosting')
            self.assert_in_success_response(['Upgrade', 'Purchase Zulip', 'Your subscription will renew automatically.'], result)

    @responses.activate
    def test_remote_billing_authentication_flow_cant_access_billing_without_finishing_confirmation(self) -> None:
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        realm: Realm = desdemona.realm
        self.add_mock_response()
        result = self.execute_remote_billing_authentication_flow(
            desdemona, expect_tos=True, confirm_tos=False, first_time_login=True, return_without_clicking_confirmation_link=True
        )
        result = self.client_get(f'/realm/{realm.uuid!s}/billing/', subdomain='selfhosting')
        self.assertEqual(result.status_code, 401)

    @responses.activate
    def test_remote_billing_authentication_flow_generate_two_confirmation_links_before_confirming(self) -> None:
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        self.add_mock_response()
        result = self.execute_remote_billing_authentication_flow(
            desdemona, expect_tos=True, confirm_tos=False, first_time_login=True, return_without_clicking_confirmation_link=True
        )
        self.assertEqual(result.status_code, 200)
        first_confirmation_url: str = self.get_confirmation_url_from_outbox(
            desdemona.delivery_email,
            url_pattern=f'{settings.SELF_HOSTING_MANAGEMENT_SUBDOMAIN}.{settings.EXTERNAL_HOST}' + '(\\S+)',
        )
        first_prereg_user = PreregistrationRemoteRealmBillingUser.objects.latest('id')
        result = self.execute_remote_billing_authentication_flow(
            desdemona, expect_tos=True, confirm_tos=False, first_time_login=True, return_without_clicking_confirmation_link=True
        )
        self.assertEqual(result.status_code, 200)
        second_confirmation_url: str = self.get_confirmation_url_from_outbox(
            desdemona.delivery_email,
            url_pattern=f'{settings.SELF_HOSTING_MANAGEMENT_SUBDOMAIN}.{settings.EXTERNAL_HOST}' + '(\\S+)',
        )
        second_prereg_user = PreregistrationRemoteRealmBillingUser.objects.latest('id')
        self.assertNotEqual(first_confirmation_url, second_confirmation_url)
        self.assertNotEqual(first_prereg_user.id, second_prereg_user.id)
        now = timezone_now()
        with time_machine.travel(now, tick=False):
            result = self.client_get(first_confirmation_url, subdomain='selfhosting')
        self.assertEqual(result.status_code, 302)
        self.assertTrue(result['Location'].startswith('/remote-billing-login/'))
        remote_billing_user = RemoteRealmBillingUser.objects.latest('id')
        self.assertEqual(remote_billing_user.user_uuid, desdemona.uuid)
        self.assertEqual(remote_billing_user.email, desdemona.delivery_email)
        first_prereg_user.refresh_from_db()
        self.assertEqual(first_prereg_user.created_user, remote_billing_user)
        with time_machine.travel(now + timedelta(seconds=1), tick=False), self.assertLogs('corporate.stripe', 'INFO') as mock_logger:
            result = self.client_get(second_confirmation_url, subdomain='selfhosting')
        self.assertEqual(result.status_code, 302)
        self.assertTrue(result['Location'].startswith('/remote-billing-login/'))
        self.assertEqual(RemoteRealmBillingUser.objects.latest('id'), remote_billing_user)
        self.assertEqual(second_prereg_user.created_user, None)
        self.assertEqual(
            mock_logger.output,
            [f'INFO:corporate.stripe:Matching RemoteRealmBillingUser already exists for PreregistrationRemoteRealmBillingUser {second_prereg_user.id}'],
        )

    @responses.activate
    def test_transfer_complimentary_access_plan_scheduled_for_upgrade_from_server_to_realm(self) -> None:
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        self.assertIsNone(get_customer_by_remote_server(self.server))
        start_date = timezone_now()
        end_date = add_months(timezone_now(), 10)
        server_billing_session = RemoteServerBillingSession(self.server)
        server_billing_session.create_complimentary_access_plan(start_date, end_date)
        server_customer = server_billing_session.get_customer()
        assert server_customer is not None
        server_plan = get_current_plan_by_customer(server_customer)
        assert server_plan is not None
        self.assertEqual(self.server.plan_type, RemoteZulipServer.PLAN_TYPE_SELF_MANAGED_LEGACY)
        self.assertEqual(server_plan.tier, CustomerPlan.TIER_SELF_HOSTED_LEGACY)
        self.assertEqual(server_plan.status, CustomerPlan.ACTIVE)
        server_plan.status = CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END
        server_plan.save(update_fields=['status'])
        server_next_plan = CustomerPlan.objects.create(
            customer=server_customer,
            billing_cycle_anchor=end_date,
            billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            tier=CustomerPlan.TIER_SELF_HOSTED_BUSINESS,
            status=CustomerPlan.NEVER_STARTED,
        )
        self.assert_length(Realm.objects.all(), 4)
        RemoteRealm.objects.all().delete()
        self.add_mock_response()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        result = self.execute_remote_billing_authentication_flow(desdemona, return_from_auth_url=True)
        self.assertEqual(result.status_code, 200)
        self.assert_in_response('Plan management not available', result)
        self.server.refresh_from_db()
        self.assertEqual(self.server.plan_type, RemoteZulipServer.PLAN_TYPE_SELF_MANAGED_LEGACY)
        self.assert_length(RemoteRealm.objects.all(), 4)
        for remote_realm in RemoteRealm.objects.all():
            self.assertIsNone(get_customer_by_remote_realm(remote_realm))
        server_plan.refresh_from_db()
        self.assertEqual(get_current_plan_by_customer(server_customer), server_plan)
        self.assertEqual(server_plan.customer, server_customer)
        Realm.objects.exclude(string_id__in=['zulip', 'zulipinternal']).update(deactivated=True)
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        result = self.execute_remote_billing_authentication_flow(desdemona, return_from_auth_url=False)
        self.assertEqual(result.status_code, 302)
        self.server.refresh_from_db()
        self.assertEqual(self.server.plan_type, RemoteZulipServer.PLAN_TYPE_SELF_MANAGED)
        self.assertCountEqual(
            RemoteRealm.objects.filter(realm_deactivated=True).values_list('host', flat=True), ['zephyr.testserver', 'lear.testserver']
        )
        self.assertEqual(RemoteRealm.objects.filter(realm_deactivated=False).count(), 2)
        remote_realm_with_plan = RemoteRealm.objects.get(realm_deactivated=False, is_system_bot_realm=False)
        system_bot_remote_realm = RemoteRealm.objects.get(realm_deactivated=False, is_system_bot_realm=True)
        self.assertIsNone(get_customer_by_remote_realm(system_bot_remote_realm))
        self.assertEqual(remote_realm_with_plan.host, 'zulip.testserver')
        customer = get_customer_by_remote_realm(remote_realm_with_plan)
        assert customer is not None
        self.assertEqual(customer, server_customer)
        plan = get_current_plan_by_customer(customer)
        assert plan is not None
        self.assertEqual(remote_realm_with_plan.plan_type, RemoteRealm.PLAN_TYPE_SELF_MANAGED_LEGACY)
        self.assertEqual(plan.tier, CustomerPlan.TIER_SELF_HOSTED_LEGACY)
        self.assertEqual(plan.status, CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END)
        self.assertEqual(plan.billing_cycle_anchor, start_date)
        self.assertEqual(plan.end_date, end_date)
        self.assertEqual(RemoteRealmBillingSession(remote_realm_with_plan).get_next_plan(plan), server_next_plan)

    @responses.activate
    def test_transfer_plan_from_server_to_realm_when_realm_has_customer(self) -> None:
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        zulip_realm = get_realm('zulip')
        server_billing_session = RemoteServerBillingSession(self.server)
        server_customer = server_billing_session.update_or_create_customer(stripe_customer_id='cus_123server')
        server_plan = CustomerPlan.objects.create(
            customer=server_customer,
            billing_cycle_anchor=timezone_now(),
            billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            tier=CustomerPlan.TIER_SELF_HOSTED_COMMUNITY,
            status=CustomerPlan.ACTIVE,
        )
        self.server.plan_type = RemoteZulipServer.PLAN_TYPE_COMMUNITY
        self.server.save(update_fields=['plan_type'])
        RemoteRealm.objects.all().delete()
        Realm.objects.exclude(string_id__in=['zulip', 'zulipinternal']).update(deactivated=True)
        self.add_mock_response()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        remote_realm = RemoteRealm.objects.get(uuid=zulip_realm.uuid)
        realm_billing_session = RemoteRealmBillingSession(remote_realm)
        realm_customer = realm_billing_session.update_or_create_customer(stripe_customer_id='cus_123realm')
        realm_plan = CustomerPlan.objects.create(
            customer=realm_customer,
            billing_cycle_anchor=timezone_now(),
            billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            tier=CustomerPlan.TIER_SELF_MANAGED_LEGACY,
            status=CustomerPlan.ACTIVE,
        )
        remote_realm.plan_type = RemoteRealm.PLAN_TYPE_SELF_MANAGED_LEGACY
        remote_realm.save(update_fields=['plan_type'])
        with self.assertLogs('zilencer.views', 'WARN') as mock_warn:
            result = self.execute_remote_billing_authentication_flow(desdemona, return_from_auth_url=True)
        self.assertEqual(
            mock_warn.output,
            [f"WARNING:zilencer.views:Failed to migrate customer from server (id: {remote_realm.server.id}) to realm (id: {remote_realm.id}): RemoteRealm customer already exists and plans can't be migrated automatically."],
        )
        self.assert_json_error(result, f"Couldn't reconcile billing data between server and realm. Please contact {FromAddress.SUPPORT}")
        realm_plan.status = CustomerPlan.ENDED
        realm_plan.save(update_fields=['status'])
        server_plan.status = CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END
        server_plan.save(update_fields=['status'])
        with self.assertLogs('zilencer.views', 'WARN') as mock_warn:
            result = self.execute_remote_billing_authentication_flow(desdemona, return_from_auth_url=True)
        self.assertEqual(
            mock_warn.output,
            [f"WARNING:zilencer.views:Failed to migrate customer from server (id: {remote_realm.server.id}) to realm (id: {remote_realm.id}): RemoteRealm customer already exists and plans can't be migrated automatically."],
        )
        self.assert_json_error(result, f"Couldn't reconcile billing data between server and realm. Please contact {FromAddress.SUPPORT}")
        server_plan.status = CustomerPlan.ACTIVE
        server_plan.save(update_fields=['status'])
        realm_customer.refresh_from_db()
        self.assertEqual(realm_customer.stripe_customer_id, 'cus_123realm')
        with self.assertLogs('zilencer.views', 'WARN') as mock_warn:
            result = self.execute_remote_billing_authentication_flow(desdemona, return_from_auth_url=True)
        self.assertEqual(
            mock_warn.output,
            [f"WARNING:zilencer.views:Failed to migrate customer from server (id: {remote_realm.server.id}) to realm (id: {remote_realm.id}): RemoteRealm customer already exists and plans can't be migrated automatically."],
        )
        self.assert_json_error(result, f"Couldn't reconcile billing data between server and realm. Please contact {FromAddress.SUPPORT}")
        realm_customer.stripe_customer_id = None
        realm_customer.save(update_fields=['stripe_customer_id'])
        result = self.execute_remote_billing_authentication_flow(desdemona, return_from_auth_url=False)
        self.assertEqual(result.status_code, 302)
        self.server.refresh_from_db()
        self.assertEqual(self.server.plan_type, RemoteZulipServer.PLAN_TYPE_SELF_MANAGED)
        self.assertEqual(get_customer_by_remote_realm(remote_realm), realm_customer)
        self.assertEqual(get_customer_by_remote_server(self.server), server_customer)
        self.assertEqual(get_current_plan_by_customer(server_customer), None)
        self.assertEqual(get_current_plan_by_customer(realm_customer), server_plan)
        remote_realm.refresh_from_db()
        self.assertEqual(remote_realm.plan_type, RemoteRealm.PLAN_TYPE_COMMUNITY)
        realm_customer.refresh_from_db()
        self.assertEqual(realm_customer.stripe_customer_id, 'cus_123server')
        server_customer.refresh_from_db()
        self.assertEqual(server_customer.stripe_customer_id, None)

    @responses.activate
    def test_transfer_business_plan_from_server_to_realm(self) -> None:
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        self.assertIsNone(get_customer_by_remote_server(self.server))
        self.assertEqual(self.server.plan_type, RemoteZulipServer.PLAN_TYPE_SELF_MANAGED)
        server_billing_session = RemoteServerBillingSession(self.server)
        server_customer = server_billing_session.update_or_create_customer(stripe_customer_id=None)
        assert server_customer is not None
        server_plan = CustomerPlan.objects.create(
            customer=server_customer,
            billing_cycle_anchor=timezone_now(),
            billing_schedule=CustomerPlan.BILLING_SCHEDULE_ANNUAL,
            tier=CustomerPlan.TIER_SELF_HOSTED_BUSINESS,
            status=CustomerPlan.ACTIVE,
            automanage_licenses=True,
        )
        initial_license_count: int = 100
        LicenseLedger.objects.create(
            plan=server_plan,
            is_renewal=True,
            event_time=timezone_now(),
            licenses=initial_license_count,
            licenses_at_next_renewal=initial_license_count,
        )
        self.server.plan_type = RemoteZulipServer.PLAN_TYPE_BUSINESS
        self.server.save(update_fields=['plan_type'])
        self.assert_length(Realm.objects.all(), 4)
        RemoteRealm.objects.all().delete()
        self.add_mock_response()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        result = self.execute_remote_billing_authentication_flow(desdemona, return_from_auth_url=True)
        self.assertEqual(result.status_code, 200)
        self.assert_in_response('Plan management not available', result)
        self.server.refresh_from_db()
        self.assertEqual(self.server.plan_type, RemoteZulipServer.PLAN_TYPE_BUSINESS)
        self.assert_length(RemoteRealm.objects.all(), 4)
        for remote_realm in RemoteRealm.objects.all():
            self.assertIsNone(get_customer_by_remote_realm(remote_realm))
        server_plan.refresh_from_db()
        self.assertEqual(get_current_plan_by_customer(server_customer), server_plan)
        self.assertEqual(server_plan.customer, server_customer)
        Realm.objects.exclude(string_id__in=['zulip', 'zulipinternal']).update(deactivated=True)
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        result = self.execute_remote_billing_authentication_flow(desdemona, return_from_auth_url=False)
        self.assertEqual(result.status_code, 302)
        self.server.refresh_from_db()
        self.assertEqual(self.server.plan_type, RemoteZulipServer.PLAN_TYPE_SELF_MANAGED)
        self.assertEqual(RemoteRealm.objects.filter(realm_deactivated=False).count(), 2)
        remote_realm_with_plan = RemoteRealm.objects.get(realm_deactivated=False, is_system_bot_realm=False)
        system_bot_remote_realm = RemoteRealm.objects.get(realm_deactivated=False, is_system_bot_realm=True)
        self.assertIsNone(get_customer_by_remote_realm(system_bot_remote_realm))
        self.assertEqual(remote_realm_with_plan.host, 'zulip.testserver')
        customer = get_customer_by_remote_realm(remote_realm_with_plan)
        assert customer is not None
        self.assertEqual(customer, server_customer)
        plan = get_current_plan_by_customer(customer)
        assert plan is not None
        self.assertEqual(remote_realm_with_plan.plan_type, RemoteRealm.PLAN_TYPE_BUSINESS)
        self.assertEqual(plan.tier, CustomerPlan.TIER_SELF_HOSTED_BUSINESS)
        self.assertEqual(plan.status, CustomerPlan.ACTIVE)
        billing_session = RemoteRealmBillingSession(remote_realm=remote_realm_with_plan)
        license_ledger = billing_session.get_last_ledger_for_automanaged_plan_if_exists()
        billable_licenses = billing_session.get_billable_licenses_for_customer(customer, plan.tier)
        assert license_ledger is not None
        self.assertNotEqual(initial_license_count, billable_licenses)
        self.assertEqual(license_ledger.licenses, initial_license_count)
        self.assertEqual(license_ledger.licenses_at_next_renewal, billable_licenses)
        self.assertFalse(license_ledger.is_renewal)

    @responses.activate
    def test_transfer_plan_from_server_to_realm_edge_cases(self) -> None:
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        self.assertIsNone(get_customer_by_remote_server(self.server))
        self.add_mock_response()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        result = self.execute_remote_billing_authentication_flow(desdemona)
        self.assertEqual(result.status_code, 302)
        self.assertIsNone(get_customer_by_remote_server(self.server))
        server_billing_session = RemoteServerBillingSession(self.server)
        server_customer = server_billing_session.update_or_create_customer(stripe_customer_id=None)
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        result = self.execute_remote_billing_authentication_flow(desdemona, first_time_login=False, expect_tos=False)
        self.assertEqual(result.status_code, 302)
        self.assertIsNone(get_current_plan_by_customer(server_customer))
        start_date = timezone_now()
        end_date = add_months(timezone_now(), 10)
        server_billing_session = RemoteServerBillingSession(self.server)
        server_billing_session.create_complimentary_access_plan(start_date, end_date)
        Realm.objects.all().update(deactivated=True)
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        result = self.execute_remote_billing_authentication_flow(desdemona, return_from_auth_url=True)
        self.assertEqual(result.status_code, 200)
        self.assert_in_response('Plan management not available', result)
        server_plan = get_current_plan_by_customer(server_customer)
        assert server_plan is not None
        self.assertEqual(self.server.plan_type, RemoteZulipServer.PLAN_TYPE_SELF_MANAGED_LEGACY)
        self.assertEqual(server_plan.tier, CustomerPlan.TIER_SELF_HOSTED_LEGACY)
        self.assertEqual(server_plan.status, CustomerPlan.ACTIVE)
        server_plan.tier = CustomerPlan.TIER_SELF_HOSTED_BUSINESS
        server_plan.save(update_fields=['tier'])
        self.server.plan_type = RemoteZulipServer.PLAN_TYPE_BUSINESS
        self.server.save(update_fields=['plan_type'])
        result = self.execute_remote_billing_authentication_flow(desdemona, return_from_auth_url=True)
        self.assertEqual(result.status_code, 200)
        self.assert_in_response('Plan management not available', result)
        server_customer.refresh_from_db()
        server_plan.refresh_from_db()
        self.assertEqual(self.server.plan_type, RemoteZulipServer.PLAN_TYPE_BUSINESS)
        self.assertEqual(server_plan.tier, CustomerPlan.TIER_SELF_HOSTED_BUSINESS)
        Realm.objects.all().delete()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        server_customer.refresh_from_db()
        server_plan.refresh_from_db()
        self.assertEqual(self.server.plan_type, RemoteZulipServer.PLAN_TYPE_BUSINESS)
        self.assertEqual(server_plan.tier, CustomerPlan.TIER_SELF_HOSTED_BUSINESS)
        self.assertEqual(server_plan.status, CustomerPlan.ACTIVE)

class RemoteServerTestCase(BouncerTestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        self.uuid = self.server.uuid
        self.secret = self.server.api_key

    def execute_remote_billing_authentication_flow(
        self,
        email: str,
        full_name: str,
        next_page: Optional[str] = None,
        expect_tos: bool = True,
        confirm_tos: bool = True,
        return_without_clicking_confirmation_link: bool = False,
    ) -> "TestHttpResponse":
        now = timezone_now()
        with time_machine.travel(now, tick=False):
            payload: dict[str, Any] = {'zulip_org_id': self.uuid, 'zulip_org_key': self.secret}
            if next_page is not None:
                payload['next_page'] = next_page
            result = self.client_post('/serverlogin/', payload, subdomain='selfhosting')
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response(['Enter log in email'], result)
        if next_page is not None:
            self.assert_in_success_response([f'<input type="hidden" name="next_page" value="{next_page}" />'], result)
        self.assert_in_success_response([f'action="/serverlogin/{self.uuid!s}/confirm/"'], result)
        identity_dict: LegacyServerIdentityDict = LegacyServerIdentityDict(
            remote_server_uuid=str(self.server.uuid), authenticated_at=datetime_to_timestamp(now), remote_billing_user_id=None
        )
        self.assertEqual(self.client.session['remote_billing_identities'][f'remote_server:{self.uuid!s}'], identity_dict)
        payload = {'email': email}
        if next_page is not None:
            payload['next_page'] = next_page
        with time_machine.travel(now, tick=False):
            result = self.client_post(f'/serverlogin/{self.uuid!s}/confirm/', payload, subdomain='selfhosting')
        if result.status_code == 429:
            return result
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response(['We have sent', 'a log in', 'link will expire in', email], result)
        confirmation_url: str = self.get_confirmation_url_from_outbox(
            email,
            url_pattern=f'{settings.SELF_HOSTING_MANAGEMENT_SUBDOMAIN}.{settings.EXTERNAL_HOST}' + '(\\S+)',
            email_body_contains='This link will expire in 24 hours',
        )
        if return_without_clicking_confirmation_link:
            return result
        with time_machine.travel(now, tick=False):
            result = self.client_get(confirmation_url, subdomain='selfhosting')
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response([f'Log in to Zulip plan management for {self.server.hostname}', email], result)
        self.assert_in_success_response([f'action="{confirmation_url}"'], result)
        if expect_tos:
            self.assert_in_success_response(['I agree', 'Terms of Service'], result)
        payload = {'full_name': full_name}
        if confirm_tos:
            payload['tos_consent'] = 'true'
        with time_machine.travel(now, tick=False):
            result = self.client_post(confirmation_url, payload, subdomain='selfhosting')
        if result.status_code >= 400:
            return result
        remote_billing_user = RemoteServerBillingUser.objects.get(remote_server=self.server, email=email)
        identity_dict = LegacyServerIdentityDict(
            remote_server_uuid=str(self.server.uuid), authenticated_at=datetime_to_timestamp(now), remote_billing_user_id=remote_billing_user.id
        )
        self.assertEqual(self.client.session['remote_billing_identities'][f'remote_server:{self.uuid!s}'], identity_dict)
        self.assertEqual(remote_billing_user.last_login, now)
        return result

class LegacyServerLoginTest(RemoteServerTestCase):
    @ratelimit_rule(10, 3, domain='sends_email_by_remote_server')
    @ratelimit_rule(10, 2, domain='sends_email_by_ip')
    def test_remote_billing_authentication_flow_rate_limited(self) -> None:
        RateLimitedIPAddr('127.0.0.1', domain='sends_email_by_ip').clear_history()
        RateLimitedRemoteZulipServer(self.server, domain='sends_email_by_remote_server').clear_history()
        self.login('desdemona')
        desdemona: UserProfile = self.example_user('desdemona')
        for i in range(2):
            result = self.execute_remote_billing_authentication_flow(
                desdemona.delivery_email, desdemona.full_name, return_without_clicking_confirmation_link=True
            )
            self.assertEqual(result.status_code, 200)
        result = self.execute_remote_billing_authentication_flow(
            desdemona.delivery_email, desdemona.full_name, return_without_clicking_confirmation_link=True
        )
        self.assertEqual(result.status_code, 429)
        self.assert_in_response('You have exceeded the limit', result)
        RateLimitedIPAddr('127.0.0.1', domain='sends_email_by_ip').clear_history()
        result = self.execute_remote_billing_authentication_flow(
            desdemona.delivery_email, desdemona.full_name, return_without_clicking_confirmation_link=True
        )
        self.assertEqual(result.status_code, 200)
        with self.assertLogs('zilencer.auth', 'WARN') as mock_log:
            result = self.execute_remote_billing_authentication_flow(
                desdemona.delivery_email, desdemona.full_name, return_without_clicking_confirmation_link=True
            )
            self.assertEqual(result.status_code, 429)
            self.assert_in_response('Your server has exceeded the limit', result)
        self.assertEqual(
            mock_log.output,
            [f'WARNING:zilencer.auth:Remote server {self.server.hostname} {str(self.server.uuid)[:12]} exceeded rate limits on domain sends_email_by_remote_server'],
        )

    def test_server_login_get(self) -> None:
        result = self.client_get('/serverlogin/', subdomain='selfhosting')
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response(['Authenticate server for Zulip plan management'], result)

    def test_server_login_invalid_zulip_org_id(self) -> None:
        result = self.client_post('/serverlogin/', {'zulip_org_id': 'invalid', 'zulip_org_key': 'secret'}, subdomain='selfhosting')
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response(['This zulip_org_id is not registered with Zulip&#39;s billing management system.'], result)

    def test_server_login_invalid_zulip_org_key(self) -> None:
        result = self.client_post('/serverlogin/', {'zulip_org_id': self.uuid, 'zulip_org_key': 'invalid'}, subdomain='selfhosting')
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response(['Invalid zulip_org_key for this zulip_org_id.'], result)

    def test_server_login_deactivated_server(self) -> None:
        self.server.deactivated = True
        self.server.save(update_fields=['deactivated'])
        result = self.client_post('/serverlogin/', {'zulip_org_id': self.uuid, 'zulip_org_key': self.secret}, subdomain='selfhosting')
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response(['Your server registration has been deactivated.'], result)

    def test_server_login_success_with_no_plan(self) -> None:
        hamlet: UserProfile = self.example_user('hamlet')
        now = timezone_now()
        with time_machine.travel(now, tick=False):
            result = self.execute_remote_billing_authentication_flow(hamlet.delivery_email, hamlet.full_name, expect_tos=True, confirm_tos=True)
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], f'/server/{self.uuid}/plans/')
        result = self.client_get(f'/server/{self.uuid}/billing/', subdomain='selfhosting')
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], f'/server/{self.uuid}/upgrade/')
        with mock.patch('corporate.lib.stripe.has_stale_audit_log', return_value=False):
            result = self.client_get(result['Location'], subdomain='selfhosting')
            self.assert_in_success_response([f'Upgrade {self.server.hostname}'], result)
        remote_billing_user = RemoteServerBillingUser.objects.latest('id')
        self.assertEqual(remote_billing_user.email, hamlet.delivery_email)
        prereg_user = PreregistrationRemoteServerBillingUser.objects.latest('id')
        self.assertEqual(prereg_user.created_user, remote_billing_user)
        self.assertEqual(remote_billing_user.date_joined, now)

    def test_server_login_success_consent_is_not_re_asked(self) -> None:
        hamlet: UserProfile = self.example_user('hamlet')
        result = self.execute_remote_billing_authentication_flow(hamlet.delivery_email, hamlet.full_name, expect_tos=True, confirm_tos=True)
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], f'/server/{self.uuid}/plans/')
        result = self.execute_remote_billing_authentication_flow(hamlet.delivery_email, hamlet.full_name, expect_tos=False, confirm_tos=False)
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], f'/server/{self.uuid}/plans/')

    def test_server_login_success_with_next_page(self) -> None:
        hamlet: UserProfile = self.example_user('hamlet')
        result = self.client_post('/serverlogin/', {'zulip_org_id': self.uuid, 'zulip_org_key': self.secret, 'next_page': 'invalid'}, subdomain='selfhosting')
        self.assert_json_error(result, 'Invalid next_page', 400)
        result = self.execute_remote_billing_authentication_flow(hamlet.delivery_email, hamlet.full_name, next_page='sponsorship')
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], f'/server/{self.uuid}/sponsorship/')
        result = self.client_get(result['Location'], subdomain='selfhosting')
        self.assert_in_success_response(['Request Zulip', 'sponsorship', 'Community'], result)

    def test_server_login_next_page_in_form_persists(self) -> None:
        result = self.client_get('/serverlogin/?next_page=billing', subdomain='selfhosting')
        self.assert_in_success_response(['<input type="hidden" name="next_page" value="billing" />'], result)
        result = self.client_post('/serverlogin/', {'zulip_org_id': self.uuid, 'zulip_org_key': 'invalid', 'next_page': 'billing'}, subdomain='selfhosting')
        self.assertEqual(result.status_code, 200)
        self.assert_in_success_response(['Invalid zulip_org_key for this zulip_org_id.'], result)
        self.assert_in_success_response(['<input type="hidden" name="next_page" value="billing" />'], result)

    def test_server_billing_unauthed(self) -> None:
        hamlet: UserProfile = self.example_user('hamlet')
        now = timezone_now()
        result = self.client_get(f'/server/{self.uuid}/billing/', subdomain='selfhosting')
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/serverlogin/?next_page=billing')
        result = self.client_get(result['Location'], subdomain='selfhosting')
        self.assert_in_success_response(['<input type="hidden" name="next_page" value="billing" />'], result)
        with time_machine.travel(now, tick=False):
            self.execute_remote_billing_authentication_flow(hamlet.delivery_email, hamlet.full_name, next_page='upgrade', return_without_clicking_confirmation_link=True)
        result = self.client_get(f'/server/{self.uuid}/billing/', subdomain='selfhosting')
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/serverlogin/?next_page=billing')
        with time_machine.travel(now, tick=False):
            result = self.execute_remote_billing_authentication_flow(hamlet.delivery_email, hamlet.full_name, next_page='upgrade')
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], f'/server/{self.uuid}/upgrade/')
        with mock.patch('corporate.lib.stripe.has_stale_audit_log', return_value=False):
            result = self.client_get(result['Location'], subdomain='selfhosting')
            self.assert_in_success_response([f'Upgrade {self.server.hostname}'], result)
        with time_machine.travel(now + timedelta(seconds=REMOTE_BILLING_SESSION_VALIDITY_SECONDS + 30), tick=False):
            result = self.client_get(f'/server/{self.uuid}/upgrade/', subdomain='selfhosting')
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], '/serverlogin/?next_page=upgrade')

    def test_remote_billing_authentication_flow_tos_consent_failure(self) -> None:
        hamlet: UserProfile = self.example_user('hamlet')
        result = self.execute_remote_billing_authentication_flow(hamlet.email, hamlet.full_name, expect_tos=True, confirm_tos=False)
        self.assert_json_error(result, 'You must accept the Terms of Service to proceed.')

class TestGenerateDeactivationLink(BouncerTestCase):
    def test_generate_deactivation_link(self) -> None:
        server = self.server
        confirmation_url: str = generate_confirmation_link_for_server_deactivation(server, validity_in_minutes=60)
        result = self.client_get(confirmation_url, subdomain='selfhosting')
        self.assert_in_success_response(['Log in to deactivate registration for', server.contact_email], result)
        payload = {'full_name': 'test', 'tos_consent': 'true'}
        result = self.client_post(confirmation_url, payload, subdomain='selfhosting')
        self.assertEqual(result.status_code, 302)
        self.assertEqual(result['Location'], f'/server/{server.uuid!s}/deactivate/')
        result = self.client_get(result['Location'], subdomain='selfhosting')
        self.assert_in_success_response(["You are about to deactivate this server's", server.hostname, f'action="/server/{server.uuid!s}/deactivate/"'], result)
        result = self.client_post(f'/server/{server.uuid!s}/deactivate/', {'confirmed': 'true'}, subdomain='selfhosting')
        self.assert_in_success_response([f'Registration deactivated for<br />{server.hostname}'], result)
        server.refresh_from_db()
        self.assertEqual(server.deactivated, True)