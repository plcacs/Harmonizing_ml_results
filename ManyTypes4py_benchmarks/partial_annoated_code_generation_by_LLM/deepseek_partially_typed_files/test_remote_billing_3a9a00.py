from datetime import timedelta
from typing import TYPE_CHECKING, Optional, Dict, Any, Union
from unittest import mock
import responses
import time_machine
from django.conf import settings
from django.utils.timezone import now as timezone_now
from typing_extensions import override
from corporate.lib.remote_billing_util import REMOTE_BILLING_SESSION_VALIDITY_SECONDS, LegacyServerIdentityDict, RemoteBillingIdentityDict, RemoteBillingUserDict
from corporate.lib.stripe import RemoteRealmBillingSession, RemoteServerBillingSession, add_months
from corporate.models import CustomerPlan, LicenseLedger, get_current_plan_by_customer, get_customer_by_remote_realm, get_customer_by_remote_server
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
from zilencer.models import PreregistrationRemoteRealmBillingUser, PreregistrationRemoteServerBillingUser, RateLimitedRemoteZulipServer, RemoteRealm, RemoteRealmBillingUser, RemoteServerBillingUser, RemoteZulipServer
if TYPE_CHECKING:
    from django.test.client import _MonkeyPatchedWSGIResponse as TestHttpResponse

class RemoteRealmBillingTestCase(BouncerTestCase):

    def execute_remote_billing_authentication_flow(self, user: UserProfile, next_page: Optional[str] = None, expect_tos: bool = True, confirm_tos: bool = True, first_time_login: bool = True, return_without_clicking_confirmation_link: bool = False, return_from_auth_url: bool = False) -> 'TestHttpResponse':
        now = timezone_now()
        self_hosted_billing_url = '/self-hosted-billing/'
        if next_page is not None:
            self_hosted_billing_url += f'?next_page={next_page}'
        with time_machine.travel(now, tick=False):
            result = self.client_get(self_hosted_billing_url)
        self.assertEqual(result.status_code, 302)
        self.assertIn('http://selfhosting.testserver/remote-billing-login/', result['Location'])
        signed_auth_url = result['Location']
        signed_access_token = signed_auth_url.split('/')[-1]
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
                result = self.client_post(f'/remote-billing-login/{signed_access_token}/confirm/', {'email': user.delivery_email}, subdomain='selfhosting')
            if result.status_code == 429:
                return result
            self.assertEqual(result.status_code, 200)
            self.assert_in_success_response(['To finish logging in, check your email account (', ') for a confirmation email from Zulip.', user.delivery_email], result)
            confirmation_url = self.get_confirmation_url_from_outbox(user.delivery_email, url_pattern=f'{settings.SELF_HOSTING_MANAGEMENT_SUBDOMAIN}.{settings.EXTERNAL_HOST}(\\S+)', email_body_contains='confirm your email and log in to Zulip plan management')
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
        params: Dict[str, Any] = {}
        if expect_tos:
            self.assert_in_success_response(['I agree', 'Terms of Service'], result)
        if confirm_tos:
            params = {'tos_consent': 'true'}
        with time_machine.travel(now, tick=False):
            result = self.client_post(signed_auth_url, params, subdomain='selfhosting')
        if result.status_code >= 400:
            return result
        remote_billing_user = RemoteRealmBillingUser.objects.get(user_uuid=user.uuid)
        identity_dict = RemoteBillingIdentityDict(user=RemoteBillingUserDict(user_email=user.delivery_email, user_uuid=str(user.uuid), user_full_name=user.full_name), remote_server_uuid=str(self.server.uuid), remote_realm_uuid=str(user.realm.uuid), remote_billing_user_id=remote_billing_user.id, authenticated_at=datetime_to_timestamp(now), uri_scheme='http://', next_page=next_page)
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
        self_hosted_billing_url = '/self-hosted-billing/'
        self_hosted_billing_json_url = '/json/self-hosted-billing'
        with self.settings(ZULIP_SERVICE_PUSH_NOTIFICATIONS=False):
            with self.settings(CORPORATE_ENABLED=True):
                result = self.client_get(self_hosted_billing_url)
                self.assertEqual(result.status_code, 404)
                self.assert_in_response('Page not found (404)', result)
            with self.settings(CORPORATE_ENABLED=False):
                result = self.client_get(self_hosted_billing_url)
                self.assertEqual(result.status_code, 302)
                redirect_url = result['Location']
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
            self.assert_json_error(result, 'Your organization is registered to a different Zulip server. Please contact Zulip support for assistance in resolving this issue.', 403)
        result = self.client_get(self_hosted_billing_url)
        self.assertEqual(result.status_code, 302)
        self.assertIn('http://selfhosting.testserver/remote-billing-login/', result['Location'])
        result = self.client_get(self_hosted_billing_json_url)
        self.assert_json_success(result)
        data = result.json()
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
        desdemona = self.example_user('desdemona')
        realm = desdemona.realm
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
        desdemona = self.example_user('desdemona')
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
        self.assertEqual(mock_log.output, [f'WARNING:zilencer.auth:Remote server {self.server.hostname} {str(self.server.uuid)[:12]} exceeded rate limits on domain sends_email_by_remote_server'])

    @responses.activate
    def test_remote_billing_authentication_flow_realm_not_registered(self) -> None:
        RemoteRealm.objects.all().delete()
        self.login('desdemona')
        desdemona = self.example_user('desdemona')
        realm = desdemona.realm
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
        desdemona = self.example_user('desdemona')
        self.add_mock_response()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        result = self.execute_remote_billing_authentication_flow(desdemona, expect_tos=True, confirm_tos=False)
        self.assert_json_error(result, 'You must accept the Terms of Service to proceed.')

    @responses.activate
    def test_remote_billing_authentication_flow_tos_consent_update(self) -> None:
        self.login('desdemona')
        desdemona = self.example_user('desdemona')
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
        desdemona = self.example_user('desdemona')
        realm = desdemona.realm
        self.add_mock_response()
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        with time_machine.travel(now, tick=False):
            result = self.execute_remote_billing_authentication_flow(desdemona)
        self.assertEqual(result['Location'], f'/realm/{realm.uuid!s}/plans/')
        final_url = result['Location']
        with time_machine.travel(now + timedelta(seconds=1), tick=False):
            result = self.client_get(final_url, subdomain='selfhosting')
        self.assert_in_success_response(['showing-self-hosted', 'Retain full control'], result)
        with time_machine.travel(now + timedelta(seconds=REMOTE_BILLING_SESSION_VALIDITY_SECONDS + 1), tick=False):
            result = self.client_get(final_url, subdomain='selfhosting')
            self.assertEqual(result.status_code, 302)
            self.assertEqual(result['Location'], f'http://{desdemona.realm.host}/self-hosted-billing/?next_page=plans')
            result = self.execute_remote_billing_authentication_flow(desdemona, next_page='plans', expect_tos=False, confirm_tos=False, first_time_login=False)
            self.assertEqual(result['Location'], f'/realm/{realm.uuid!s}/plans/')
            result = self.client_get(result['Location'], subdomain='selfhosting')
            self.assert_in_success_response(['showing-self-hosted',