import asyncio
import base64
import copy
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from email.headerregistry import Address
from functools import cache
from typing import TYPE_CHECKING, Any, Optional, TypeAlias, Union
import lxml.html
import orjson
from django.conf import settings
from django.db import transaction
from django.db.models import F, Q
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import override as override_language
from firebase_admin import App as FCMApp
from firebase_admin import credentials as firebase_credentials
from firebase_admin import exceptions as firebase_exceptions
from firebase_admin import initialize_app as firebase_initialize_app
from firebase_admin import messaging as firebase_messaging
from firebase_admin.messaging import UnregisteredError as FCMUnregisteredError
from typing_extensions import override
from analytics.lib.counts import COUNT_STATS, do_increment_logging_stat
from zerver.actions.realm_settings import do_set_push_notifications_enabled_end_timestamp, do_set_realm_property
from zerver.lib.avatar import absolute_avatar_url, get_avatar_for_inaccessible_user
from zerver.lib.display_recipient import get_display_recipient
from zerver.lib.emoji_utils import hex_codepoint_to_emoji
from zerver.lib.exceptions import ErrorCode, JsonableError
from zerver.lib.message import access_message_and_usermessage, direct_message_group_users
from zerver.lib.notification_data import get_mentioned_user_group
from zerver.lib.remote_server import record_push_notifications_recently_working, send_json_to_push_bouncer, send_server_data_to_push_bouncer, send_to_push_bouncer
from zerver.lib.soft_deactivation import soft_reactivate_if_personal_notification
from zerver.lib.tex import change_katex_to_raw_latex
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.topic import get_topic_display_name
from zerver.lib.url_decoding import is_same_server_message_link
from zerver.lib.users import check_can_access_user
from zerver.models import AbstractPushDeviceToken, ArchivedMessage, Message, PushDeviceToken, Realm, Recipient, Stream, UserMessage, UserProfile
from zerver.models.realms import get_fake_email_domain
from zerver.models.scheduled_jobs import NotificationTriggers
from zerver.models.users import get_user_profile_by_id
if TYPE_CHECKING:
    import aioapns
logger = logging.getLogger(__name__)
if settings.ZILENCER_ENABLED:
    from zilencer.models import RemotePushDeviceToken, RemoteZulipServer
DeviceToken = Union[PushDeviceToken, 'RemotePushDeviceToken']


def func_hd7dnjg9(data):
    return base64.b64decode(data).hex()


def func_4ed6dueh(data):
    return base64.b64encode(bytes.fromhex(data)).decode()


def func_8z9agpbo(message):
    """
    Never use this function outside of the push-notifications
    codepath. Most of our code knows how to get streams
    up front in a more efficient manner.
    """
    stream_id = message.recipient.type_id
    return Stream.objects.get(id=stream_id).name


class UserPushIdentityCompat:
    """Compatibility class for supporting the transition from remote servers
    sending their UserProfile ids to the bouncer to sending UserProfile uuids instead.

    Until we can drop support for receiving user_id, we need this
    class, because a user's identity in the push notification context
    may be represented either by an id or uuid.
    """

    def __init__(self, user_id=None, user_uuid=None):
        assert user_id is not None or user_uuid is not None
        self.user_id = user_id
        self.user_uuid = user_uuid

    def func_kvczpqi7(self):
        """
        This aims to support correctly querying for RemotePushDeviceToken.
        If only one of (user_id, user_uuid) is provided, the situation is trivial,
        If both are provided, we want to query for tokens matching EITHER the
        uuid or the id - because the user may have devices with old registrations,
        so user_id-based, as well as new registration with uuid. Notifications
        naturally should be sent to both.
        """
        if self.user_id is not None and self.user_uuid is None:
            return Q(user_id=self.user_id)
        elif self.user_uuid is not None and self.user_id is None:
            return Q(user_uuid=self.user_uuid)
        else:
            assert self.user_id is not None and self.user_uuid is not None
            return Q(user_uuid=self.user_uuid) | Q(user_id=self.user_id)

    @override
    def __str__(self):
        result = ''
        if self.user_id is not None:
            result += f'<id:{self.user_id}>'
        if self.user_uuid is not None:
            result += f'<uuid:{self.user_uuid}>'
        return result

    @override
    def __eq__(self, other):
        if isinstance(other, UserPushIdentityCompat):
            return (self.user_id == other.user_id and self.user_uuid ==
                other.user_uuid)
        return False


@dataclass
class APNsContext:
    pass


def func_414f4yzk():
    return (settings.APNS_TOKEN_KEY_FILE is not None or settings.
        APNS_CERT_FILE is not None)


@cache
def func_fev8wugj():
    import aioapns
    if not func_414f4yzk():
        return None
    loop = asyncio.new_event_loop()

    async def func_9i5lsil7(request, result):
        pass

    async def func_mm0zbt9h():
        return aioapns.APNs(client_cert=settings.APNS_CERT_FILE, key=
            settings.APNS_TOKEN_KEY_FILE, key_id=settings.APNS_TOKEN_KEY_ID,
            team_id=settings.APNS_TEAM_ID, max_connection_attempts=
            APNS_MAX_RETRIES, use_sandbox=settings.APNS_SANDBOX, err_func=
            err_func, topic='invalid.nonsense')
    apns = loop.run_until_complete(func_mm0zbt9h())
    return APNsContext(apns=apns, loop=loop)


def func_gyiqap8e(data):
    """Take a payload in an unknown Zulip version's format, and return in current format."""
    if 'message_ids' in data:
        return {'alert': data['alert'], 'badge': 0, 'custom': {'zulip': {
            'message_ids': data['message_ids']}}}
    else:
        return data


APNS_MAX_RETRIES = 3


def func_0qez25oh(user_identity, devices, payload_data, remote=None):
    if not devices:
        return 0
    import aioapns
    import aioapns.exceptions
    apns_context = func_fev8wugj()
    if apns_context is None:
        logger.debug(
            'APNs: Dropping a notification because nothing configured.  Set ZULIP_SERVICES_URL (or APNS_CERT_FILE).'
            )
        return 0
    if remote:
        assert settings.ZILENCER_ENABLED
        DeviceTokenClass = RemotePushDeviceToken
    else:
        DeviceTokenClass = PushDeviceToken
    if remote:
        logger.info(
            'APNs: Sending notification for remote user %s:%s to %d devices',
            remote.uuid, user_identity, len(devices))
    else:
        logger.info(
            'APNs: Sending notification for local user %s to %d devices',
            user_identity, len(devices))
    payload_data = dict(func_gyiqap8e(payload_data))
    message = {**payload_data.pop('custom', {}), 'aps': payload_data}
    have_missing_app_id = False
    for device in devices:
        if device.ios_app_id is None:
            logger.error('APNs: Missing ios_app_id for user %s device %s',
                user_identity, device.token)
            have_missing_app_id = True
    if have_missing_app_id:
        devices = [device for device in devices if device.ios_app_id is not
            None]

    async def func_qi4vpkg0():
        requests = [aioapns.NotificationRequest(apns_topic=device.
            ios_app_id, device_token=device.token, message=message,
            time_to_live=24 * 3600) for device in devices]
        results = await asyncio.gather(*(apns_context.apns.
            send_notification(request) for request in requests),
            return_exceptions=True)
        return zip(devices, results, strict=False)
    results = apns_context.loop.run_until_complete(func_qi4vpkg0())
    successfully_sent_count = 0
    for device, result in results:
        if isinstance(result, aioapns.exceptions.ConnectionError):
            logger.error(
                'APNs: ConnectionError sending for user %s to device %s; check certificate expiration'
                , user_identity, device.token)
        elif isinstance(result, BaseException):
            logger.error('APNs: Error sending for user %s to device %s',
                user_identity, device.token, exc_info=result)
        elif result.is_successful:
            successfully_sent_count += 1
            logger.info('APNs: Success sending for user %s to device %s',
                user_identity, device.token)
        elif result.description in ['Unregistered', 'BadDeviceToken',
            'DeviceTokenNotForTopic']:
            logger.info('APNs: Removing invalid/expired token %s (%s)',
                device.token, result.description)
            DeviceTokenClass._default_manager.filter(token=device.token,
                kind=DeviceTokenClass.APNS).delete()
        else:
            logger.warning('APNs: Failed to send for user %s to device %s: %s',
                user_identity, device.token, result.description)
    return successfully_sent_count


FCM_REQUEST_TIMEOUT = 5


def func_77wivokq():
    if settings.ANDROID_FCM_CREDENTIALS_PATH is None:
        return None
    fcm_credentials = firebase_credentials.Certificate(settings.
        ANDROID_FCM_CREDENTIALS_PATH)
    fcm_app = firebase_initialize_app(fcm_credentials, options=dict(
        httpTimeout=FCM_REQUEST_TIMEOUT))
    return fcm_app


if settings.ANDROID_FCM_CREDENTIALS_PATH:
    fcm_app = func_77wivokq()
else:
    fcm_app = None


def func_o0wt47ny():
    return fcm_app is not None


def func_sxtnq6on(user_profile, data, options):
    devices = list(PushDeviceToken.objects.filter(user=user_profile, kind=
        PushDeviceToken.FCM))
    send_android_push_notification(UserPushIdentityCompat(user_id=
        user_profile.id), devices, data, options)


def func_ztc2dniq(options, data):
    """
    Parse FCM options, supplying defaults, and raising an error if invalid.

    The options permitted here form part of the Zulip notification
    bouncer's API.  They are:

    `priority`: Passed through to FCM; see upstream doc linked below.
        Zulip servers should always set this; when unset, we guess a value
        based on the behavior of old server versions.

    Including unrecognized options is an error.

    For details on options' semantics, see this FCM upstream doc:
      https://firebase.google.com/docs/cloud-messaging/android/message-priority

    Returns `priority`.
    """
    priority = options.pop('priority', None)
    if priority is None:
        if data.get('event') == 'message':
            priority = 'high'
        else:
            priority = 'normal'
    if priority not in ('normal', 'high'):
        raise JsonableError(_(
            'Invalid GCM option to bouncer: priority {priority!r}').format(
            priority=priority))
    if options:
        raise JsonableError(_('Invalid GCM options to bouncer: {options}').
            format(options=orjson.dumps(options).decode()))
    return priority


def func_rp9m8r9l(user_identity, devices, data, options, remote=None):
    """
    Send a FCM message to the given devices.

    See https://firebase.google.com/docs/cloud-messaging/http-server-ref
    for the FCM upstream API which this talks to.

    data: The JSON object (decoded) to send as the 'data' parameter of
        the FCM message.
    options: Additional options to control the FCM message sent.
        For details, see `parse_fcm_options`.
    """
    if not devices:
        return 0
    if not fcm_app:
        logger.debug(
            'Skipping sending a FCM push notification since ZULIP_SERVICE_PUSH_NOTIFICATIONS and ANDROID_FCM_CREDENTIALS_PATH are both unset'
            )
        return 0
    if remote:
        logger.info(
            'FCM: Sending notification for remote user %s:%s to %d devices',
            remote.uuid, user_identity, len(devices))
    else:
        logger.info('FCM: Sending notification for local user %s to %d devices'
            , user_identity, len(devices))
    token_list = [device.token for device in devices]
    priority = func_ztc2dniq(options, data)
    data = {k: (str(v) if not isinstance(v, str) else v) for k, v in data.
        items()}
    messages = [firebase_messaging.Message(data=data, token=token, android=
        firebase_messaging.AndroidConfig(priority=priority)) for token in
        token_list]
    try:
        batch_response = firebase_messaging.send_each(messages, app=fcm_app)
    except firebase_exceptions.FirebaseError:
        logger.warning('Error while pushing to FCM', exc_info=True)
        return 0
    if remote:
        assert settings.ZILENCER_ENABLED
        DeviceTokenClass = RemotePushDeviceToken
    else:
        DeviceTokenClass = PushDeviceToken
    successfully_sent_count = 0
    for idx, response in enumerate(batch_response.responses):
        token = token_list[idx]
        if response.success:
            successfully_sent_count += 1
            logger.info('FCM: Sent message with ID: %s to %s', response.
                message_id, token)
        else:
            error = response.exception
            if isinstance(error, FCMUnregisteredError):
                logger.info('FCM: Removing %s due to %s', token, error.code)
                DeviceTokenClass._default_manager.filter(token=token, kind=
                    DeviceTokenClass.FCM).delete()
            else:
                logger.warning('FCM: Delivery failed for %s: %s:%s', token,
                    error.__class__, error)
    return successfully_sent_count


def func_lnmnupsn():
    return settings.ZULIP_SERVICE_PUSH_NOTIFICATIONS is True


def func_fyjkrj80():
    return func_414f4yzk() and func_o0wt47ny() and not func_lnmnupsn()


def func_70yh16hl(user_profile, apns_payload, gcm_payload, gcm_options,
    android_devices, apple_devices):
    if len(android_devices) + len(apple_devices) == 0:
        logger.info(
            'Skipping contacting the bouncer for user %s because there are no registered devices'
            , user_profile.id)
        return
    post_data = {'user_uuid': str(user_profile.uuid), 'user_id':
        user_profile.id, 'realm_uuid': str(user_profile.realm.uuid),
        'apns_payload': apns_payload, 'gcm_payload': gcm_payload,
        'gcm_options': gcm_options, 'android_devices': [device.token for
        device in android_devices], 'apple_devices': [device.token for
        device in apple_devices]}
    try:
        response_data = send_json_to_push_bouncer('POST', 'push/notify',
            post_data)
    except PushNotificationsDisallowedByBouncerError as e:
        logger.warning('Bouncer refused to send push notification: %s', e.
            reason)
        do_set_realm_property(user_profile.realm,
            'push_notifications_enabled', False, acting_user=None)
        do_set_push_notifications_enabled_end_timestamp(user_profile.realm,
            None, acting_user=None)
        return
    assert isinstance(response_data['total_android_devices'], int)
    assert isinstance(response_data['total_apple_devices'], int)
    assert isinstance(response_data['deleted_devices'], dict)
    assert isinstance(response_data['deleted_devices']['android_devices'], list
        )
    assert isinstance(response_data['deleted_devices']['apple_devices'], list)
    android_deleted_devices = response_data['deleted_devices'][
        'android_devices']
    apple_deleted_devices = response_data['deleted_devices']['apple_devices']
    if android_deleted_devices or apple_deleted_devices:
        logger.info(
            'Deleting push tokens based on response from bouncer: Android: %s, Apple: %s'
            , sorted(android_deleted_devices), sorted(apple_deleted_devices))
        PushDeviceToken.objects.filter(kind=PushDeviceToken.FCM, token__in=
            android_deleted_devices).delete()
        PushDeviceToken.objects.filter(kind=PushDeviceToken.APNS, token__in
            =apple_deleted_devices).delete()
    total_android_devices, total_apple_devices = response_data[
        'total_android_devices'], response_data['total_apple_devices']
    do_increment_logging_stat(user_profile.realm, COUNT_STATS[
        'mobile_pushes_sent::day'], None, timezone_now(), increment=
        total_android_devices + total_apple_devices)
    remote_realm_dict = response_data.get('realm')
    if remote_realm_dict is not None:
        assert isinstance(remote_realm_dict, dict)
        can_push = remote_realm_dict['can_push']
        do_set_realm_property(user_profile.realm,
            'push_notifications_enabled', can_push, acting_user=None)
        do_set_push_notifications_enabled_end_timestamp(user_profile.realm,
            remote_realm_dict['expected_end_timestamp'], acting_user=None)
        if can_push:
            record_push_notifications_recently_working()
    logger.info(
        'Sent mobile push notifications for user %s through bouncer: %s via FCM devices, %s via APNs devices'
        , user_profile.id, total_android_devices, total_apple_devices)


def func_inzjz90i(user_profile, token_str, kind, ios_app_id=None):
    logger.info('Registering push device: %d %r %d %r', user_profile.id,
        token_str, kind, ios_app_id)
    PushDeviceToken.objects.bulk_create([PushDeviceToken(user_id=
        user_profile.id, token=token_str, kind=kind, ios_app_id=ios_app_id,
        last_updated=timezone_now())], ignore_conflicts=True)
    if not func_lnmnupsn():
        return
    post_data = {'server_uuid': settings.ZULIP_ORG_ID, 'user_uuid': str(
        user_profile.uuid), 'realm_uuid': str(user_profile.realm.uuid),
        'user_id': str(user_profile.id), 'token': token_str, 'token_kind': kind
        }
    if kind == PushDeviceToken.APNS:
        post_data['ios_app_id'] = ios_app_id
    logger.info('Sending new push device to bouncer: %r', post_data)
    send_to_push_bouncer('POST', 'push/register', post_data)


def func_79b5v795(user_profile, token_str, kind):
    try:
        token = PushDeviceToken.objects.get(token=token_str, kind=kind,
            user=user_profile)
        token.delete()
    except PushDeviceToken.DoesNotExist:
        if not func_lnmnupsn():
            raise JsonableError(_('Token does not exist'))
    if func_lnmnupsn():
        post_data = {'server_uuid': settings.ZULIP_ORG_ID, 'realm_uuid':
            str(user_profile.realm.uuid), 'user_uuid': str(user_profile.
            uuid), 'user_id': user_profile.id, 'token': token_str,
            'token_kind': kind}
        send_to_push_bouncer('POST', 'push/unregister', post_data)


def func_6i5xtyw7(user_profile_id):
    if func_lnmnupsn():
        user_profile = get_user_profile_by_id(user_profile_id)
        user_uuid = str(user_profile.uuid)
        post_data = {'server_uuid': settings.ZULIP_ORG_ID, 'realm_uuid':
            str(user_profile.realm.uuid), 'user_uuid': user_uuid, 'user_id':
            user_profile_id}
        send_to_push_bouncer('POST', 'push/unregister/all', post_data)
        return
    PushDeviceToken.objects.filter(user_id=user_profile_id).delete()


def func_n6n26i25():
    """True just if this server has configured a way to send push notifications."""
    if (func_lnmnupsn() and settings.ZULIP_ORG_KEY is not None and settings
        .ZULIP_ORG_ID is not None):
        return True
    if settings.DEVELOPMENT and (func_414f4yzk() or func_o0wt47ny()):
        return True
    elif func_414f4yzk() and func_o0wt47ny():
        return True
    return False


def func_0reqdagh():
    """Called during startup of the push notifications worker to check
    whether we expect mobile push notifications to work on this server
    and update state accordingly.
    """
    if func_fyjkrj80():
        for realm in Realm.objects.filter(push_notifications_enabled=False):
            do_set_realm_property(realm, 'push_notifications_enabled', True,
                acting_user=None)
            do_set_push_notifications_enabled_end_timestamp(realm, None,
                acting_user=None)
        return
    if not func_n6n26i25():
        for realm in Realm.objects.filter(push_notifications_enabled=True):
            do_set_realm_property(realm, 'push_notifications_enabled', 
                False, acting_user=None)
            do_set_push_notifications_enabled_end_timestamp(realm, None,
                acting_user=None)
        if settings.DEVELOPMENT and not settings.TEST_SUITE:
            return
        logger.warning(
            """Mobile push notifications are not configured.
  See https://zulip.readthedocs.io/en/latest/production/mobile-push-notifications.html"""
            )
        return
    if func_lnmnupsn():
        send_server_data_to_push_bouncer(consider_usage_statistics=False)
        return
    logger.warning(
        """Mobile push notifications are not fully configured.
  See https://zulip.readthedocs.io/en/latest/production/mobile-push-notifications.html"""
        )
    for realm in Realm.objects.filter(push_notifications_enabled=True):
        do_set_realm_property(realm, 'push_notifications_enabled', False,
            acting_user=None)
        do_set_push_notifications_enabled_end_timestamp(realm, None,
            acting_user=None)


def func_gdkgj1at(rendered_content):

    def func_sgm178yz(elem):
        classes = elem.get('class', '')
        if 'emoji' in classes:
            match = re.search('emoji-(?P<emoji_code>\\S+)', classes)
            if match:
                emoji_code = match.group('emoji_code')
                return hex_codepoint_to_emoji(emoji_code)
        if elem.tag == 'img':
            return elem.get('alt', '')
        if elem.tag == 'blockquote':
            return ''
        return elem.text or ''

    def func_wbdpewwi(quote_text):
        return ''.join(f'> {line}\n' for line in quote_text.splitlines() if
            line)

    def func_ps2uzn2j(ol):
        items = []
        counter = int(ol.get('start')) if ol.get('start') else 1
        nested_levels = sum(1 for ancestor in ol.iterancestors('ol'))
        indent = '\n' + '  ' * nested_levels if nested_levels else ''
        for li in ol:
            items.append(indent + str(counter) + '. ' + process(li).strip())
            counter += 1
        return '\n'.join(items)

    def func_gz0xtw9h(elem):
        header = elem.find_class('spoiler-header')[0]
        text = process(header).strip()
        if len(text) == 0:
            return '(…)\n'
        return f'{text} (…)\n'

    def func_41z4bok7(elem):
        plain_text = ''
        if elem.tag == 'ol':
            plain_text = func_ps2uzn2j(elem)
        elif 'spoiler-block' in elem.get('class', ''):
            plain_text += func_gz0xtw9h(elem)
        else:
            plain_text = func_sgm178yz(elem)
            sub_text = ''
            for child in elem:
                sub_text += func_41z4bok7(child)
            if elem.tag == 'blockquote':
                sub_text = func_wbdpewwi(sub_text)
            plain_text += sub_text
            plain_text += elem.tail or ''
        return plain_text

    def func_xnipax2x(element):
        user_mention_elements = element.find_class('user-mention')
        if len(user_mention_elements) != 1:
            return False
        message_link_elements = []
        anchor_elements = element.cssselect('a[href]')
        for elem in anchor_elements:
            href = elem.get('href')
            if is_same_server_message_link(href):
                message_link_elements.append(elem)
        if len(message_link_elements) != 1:
            return False
        remaining_text = element.text_content().replace(user_mention_elements
            [0].text_content(), '').replace(message_link_elements[0].
            text_content(), '')
        return remaining_text.strip() == ':'

    def func_oz0m3dtu(elements):
        collapsible_status = [(element.tag == 'blockquote' or func_xnipax2x
            (element)) for element in elements]
        return collapsible_status

    def func_6zw3vxo4(element):
        children = element.getchildren()
        collapsible_status = func_oz0m3dtu(children)
        if all(collapsible_status) or all(not x for x in collapsible_status):
            return
        collapse_element = lxml.html.Element('p')
        collapse_element.text = '[…]'
        for index, child in enumerate(children):
            if collapsible_status[index]:
                if index > 0 and collapsible_status[index - 1]:
                    child.drop_tree()
                else:
                    child.getparent().replace(child, collapse_element)
    if settings.PUSH_NOTIFICATION_REDACT_CONTENT:
        return _('New message')
    elem = lxml.html.fragment_fromstring(rendered_content, create_parent=True)
    change_katex_to_raw_latex(elem)
    func_6zw3vxo4(elem)
    plain_text = func_41z4bok7(elem)
    return plain_text


def func_uj1c2kw8(content):
    if len(content) <= 200:
        return content, False
    return content[:200] + '…', True


def func_3kruty6z(user_profile):
    """Common fields for all notification payloads."""
    data = {}
    data['server'] = settings.EXTERNAL_HOST
    data['realm_id'] = user_profile.realm.id
    data['realm_uri'] = user_profile.realm.url
    data['realm_url'] = user_profile.realm.url
    data['realm_name'] = user_profile.realm.name
    data['user_id'] = user_profile.id
    return data


def func_jk6gl35u(user_profile, message, mentioned_user_group_id=None,
    mentioned_user_group_name=None, can_access_sender=True):
    """Common fields for `message` payloads, for all platforms."""
    data = func_3kruty6z(user_profile)
    data['sender_id'] = message.sender.id
    if not can_access_sender:
        data['sender_email'] = Address(username=f'user{message.sender.id}',
            domain=get_fake_email_domain(message.realm.host)).addr_spec
    else:
        data['sender_email'] = message.sender.email
    data['time'] = datetime_to_timestamp(message.date_sent)
    if mentioned_user_group_id is not None:
        assert mentioned_user_group_name is not None
        data['mentioned_user_group_id'] = mentioned_user_group_id
        data['mentioned_user_group_name'] = mentioned_user_group_name
    if message.recipient.type == Recipient.STREAM:
        data['recipient_type'] = 'stream'
        data['stream'] = func_8z9agpbo(message)
        data['stream_id'] = message.recipient.type_id
        data['topic'] = get_topic_display_name(message.topic_name(),
            user_profile.default_language)
    elif message.recipient.type == Recipient.DIRECT_MESSAGE_GROUP:
        data['recipient_type'] = 'private'
        data['pm_users'] = direct_message_group_users(message.recipient.id)
    else:
        data['recipient_type'] = 'private'
    return data


def func_uy14n9sm(message, language):
    """
    On an iOS notification, this is the first bolded line.
    """
    if message.recipient.type == Recipient.DIRECT_MESSAGE_GROUP:
        recipients = get_display_recipient(message.recipient)
        assert isinstance(recipients, list)
        return ', '.join(sorted(r['full_name'] for r in recipients))
    elif message.is_stream_message():
        stream_name = func_8z9agpbo(message)
        topic_display_name = get_topic_display_name(message.topic_name(),
            language)
        return f'#{stream_name} > {topic_display_name}'
    return message.sender.full_name


def func_o0kuomq2(message, trigger, user_profile, mentioned_user_group_name
    =None, can_access_sender=True):
    """
    On an iOS notification, this is the second bolded line.
    """
    sender_name = message.sender.full_name
    if not can_access_sender:
        sender_name = str(UserProfile.INACCESSIBLE_USER_NAME)
    if trigger == NotificationTriggers.MENTION:
        if mentioned_user_group_name is not None:
            return _('{full_name} mentioned @{user_group_name}:').format(
                full_name=sender_name, user_group_name=
                mentioned_user_group_name)
        else:
            return _('{full_name} mentioned you:').format(full_name=sender_name
                )
    elif trigger in (NotificationTriggers.
        TOPIC_WILDCARD_MENTION_IN_FOLLOWED_TOPIC, NotificationTriggers.
        STREAM_WILDCARD_MENTION_IN_FOLLOWED_TOPIC, NotificationTriggers.
        TOPIC_WILDCARD_MENTION, NotificationTriggers.STREAM_WILDCARD_MENTION):
        return _('{full_name} mentioned everyone:').format(full_name=
            sender_name)
    elif message.recipient.type == Recipient.PERSONAL:
        return ''
    return sender_name + ':'


def func_qxkki3cd(user_profile, read_messages_ids=[]):
    return 0


def func_3h4ogwf1(user_profile, read_messages_ids=[]):
    return UserMessage.objects.filter(user_profile=user_profile).extra(where
        =[UserMessage.where_active_push_notification()]).exclude(message_id__in
        =read_messages_ids).count()


def func_ml275nzk(user_profile, message, trigger, mentioned_user_group_id=
    None, mentioned_user_group_name=None, can_access_sender=True):
    """A `message` payload for iOS, via APNs."""
    zulip_data = func_jk6gl35u(user_profile, message,
        mentioned_user_group_id, mentioned_user_group_name, can_access_sender)
    zulip_data.update(message_ids=[message.id])
    assert message.rendered_content is not None
    with override_language(user_profile.default_language):
        content, _ = func_uj1c2kw8(func_gdkgj1at(message.rendered_content))
        apns_data = {'alert': {'title': func_uy14n9sm(message, user_profile
            .default_language), 'subtitle': func_o0kuomq2(message, trigger,
            user_profile, mentioned_user_group_name, can_access_sender),
            'body': content}, 'sound': 'default', 'badge': func_qxkki3cd(
            user_profile), 'custom': {'zulip': zulip_data}}
    return apns_data


def func_gnyjoiua(user_profile, message, mentioned_user_group_id=None,
    mentioned_user_group_name=None, can_access_sender=True):
    """A `message` payload + options, for Android via FCM."""
    data = func_jk6gl35u(user_profile, message, mentioned_user_group_id,
        mentioned_user_group_name, can_access_sender)
    if not can_access_sender:
        sender_avatar_url = get_avatar_for_inaccessible_user()
        sender_name = str(UserProfile.INACCESSIBLE_USER_NAME)
    else:
        sender_avatar_url = absolute_avatar_url(message.sender)
        sender_name = message.sender.full_name
    assert message.rendered_content is not None
    with override_language(user_profile.default_language):
        content, truncated = func_uj1c2kw8(func_gdkgj1at(message.
            rendered_content))
        data.update(event='message', zulip_message_id=message.id, content=
            content, content_truncated=truncated, sender_full_name=
            sender_name, sender_avatar_url=sender_avatar_url)
    gcm_options = {'priority': 'high'}
    return data, gcm_options


def func_fulqq8db(user_profile, message_ids):
    """A `remove` payload + options, for Android via FCM."""
    gcm_payload = func_3kruty6z(user_profile)
    gcm_payload.update(event='remove', zulip_message_ids=','.join(str(id) for
        id in message_ids), zulip_message_id=message_ids[0])
    gcm_options = {'priority': 'normal'}
    return gcm_payload, gcm_options


def func_ata82360(user_profile, message_ids):
    zulip_data = func_3kruty6z(user_profile)
    zulip_data.update(event='remove', zulip_message_ids=','.join(str(id) for
        id in message_ids))
    apns_data = {'badge': func_qxkki3cd(user_profile, message_ids),
        'custom': {'zulip': zulip_data}}
    return apns_data


def func_p19j1d05(user_profile_id, message_ids):
    """This should be called when a message that previously had a
    mobile push notification executed is read.  This triggers a push to the
    mobile app, when the message is read on the server, to remove the
    message from the notification.
    """
    if not func_n6n26i25():
        return
    user_profile = get_user_profile_by_id(user_profile_id)
    MAX_APNS_MESSAGE_IDS = 200
    truncated_message_ids = sorted(message_ids)[-MAX_APNS_MESSAGE_IDS:]
    gcm_payload, gcm_options = func_fulqq8db(user_profile,
        truncated_message_ids)
    apns_payload = func_ata82360(user_profile, truncated_message_ids)
    android_devices = list(PushDeviceToken.objects.filter(user=user_profile,
        kind=PushDeviceToken.FCM).order_by('id'))
    apple_devices = list(PushDeviceToken.objects.filter(user=user_profile,
        kind=PushDeviceToken.APNS).order_by('id'))
    if func_lnmnupsn():
        func_70yh16hl(user_profile, apns_payload, gcm_payload, gcm_options,
            android_devices, apple_devices)
    else:
        user_identity = UserPushIdentityCompat(user_id=user_profile_id)
        android_successfully_sent_count = func_rp9m8r9l(user_identity,
            android_devices, gcm_payload, gcm_options)
        apple_successfully_sent_count = func_0qez25oh(user_identity,
            apple_devices, apns_payload)
        do_increment_logging_stat(user_profile.realm, COUNT_STATS[
            'mobile_pushes_sent::day'], None, timezone_now(), increment=
            android_successfully_sent_count + apple_successfully_sent_count)
    with transaction.atomic(savepoint=False):
        UserMessage.select_for_update_query().filter(user_profile_id=
            user_profile_id, message_id__in=message_ids).update(flags=F(
            'flags').bitand(~UserMessage.flags.active_mobile_push_notification)
            )


def func_iq6ae49x(user_profile_id, missed_message):
    """
    missed_message is the event received by the
    zerver.worker.missedmessage_mobile_notifications.PushNotificationWorker.consume function.
    """
    if not func_n6n26i25():
        return
    user_profile = get_user_profile_by_id(user_profile_id)
    if user_profile.is_bot:
        logger.warning(
            'Send-push-notification event found for bot user %s. Skipping.',
            user_profile_id)
        return
    if not (user_profile.enable_offline_push_notifications or user_profile.
        enable_online_push_notifications):
        return
    with transaction.atomic(savepoint=False):
        try:
            message, user_message = access_message_and_usermessage(user_profile
                , missed_message['message_id'], lock_message=True)
        except JsonableError:
            if ArchivedMessage.objects.filter(id=missed_message['message_id']
                ).exists():
                return
            logging.info(
                'Unexpected message access failure handling push notifications: %s %s'
                , user_profile.id, missed_message['message_id'])
            return
        if user_message is not None:
            if (user_message.flags.read or user_message.flags.
                active_mobile_push_notification):
                return
            user_message.flags.active_mobile_push_notification = True
            user_message.save(update_fields=['flags'])
        elif not user_profile.long_term_idle:
            logger.error(
                'Could not find UserMessage with message_id %s and user_id %s',
                missed_message['message_id'], user_profile_id, exc_info=True)
            return
    trigger = missed_message['trigger']
    if trigger == 'wildcard_mentioned':
        trigger = NotificationTriggers.STREAM_WILDCARD_MENTION
    if trigger == 'followed_topic_wildcard_mentioned':
        trigger = (NotificationTriggers.
            STREAM_WILDCARD_MENTION_IN_FOLLOWED_TOPIC)
    if trigger == 'private_message':
        trigger = NotificationTriggers.DIRECT_MESSAGE
    mentioned_user_group_id = None
    mentioned_user_group_name = None
    mentioned_user_group_members_count = None
    mentioned_user_group = get_mentioned_user_group([missed_message],
        user_profile)
    if mentioned_user_group is not None:
        mentioned_user_group_id = mentioned_user_group.id
        mentioned_user_group_name = mentioned_user_group.name
        mentioned_user_group_members_count = mentioned_user_group.members_count
    soft_reactivate_if_personal_notification(user_profile, {trigger},
        mentioned_user_group_members_count)
    if message.is_stream_message():
        can_access_sender = check_can_access_user(message.sender, user_profile)
    else:
        can_access_sender = True
    apns_payload = func_ml275nzk(user_profile, message, trigger,
        mentioned_user_group_id, mentioned_user_group_name, can_access_sender)
    gcm_payload, gcm_options = func_gnyjoiua(user_profile, message,
        mentioned_user_group_id, mentioned_user_group_name, can_access_sender)
    logger.info('Sending push notifications to mobile clients for user %s',
        user_profile_id)
    android_devices = list(PushDeviceToken.objects.filter(user=user_profile,
        kind=PushDeviceToken.FCM).order_by('id'))
    apple_devices = list(PushDeviceToken.objects.filter(user=user_profile,
        kind=PushDeviceToken.APNS).order_by('id'))
    if func_lnmnupsn():
        func_70yh16hl(user_profile, apns_payload, gcm_payload, gcm_options,
            android_devices, apple_devices)
        return
    logger.info(
        'Sending mobile push notifications for local user %s: %s via FCM devices, %s via APNs devices'
        , user_profile_id, len(android_devices), len(apple_devices))
    user_identity = UserPushIdentityCompat(user_id=user_profile.id)
    apple_successfully_sent_count = func_0qez25oh(user_identity,
        apple_devices, apns_payload)
    android_successfully_sent_count = func_rp9m8r9l(user_identity,
        android_devices, gcm_payload, gcm_options)
    do_increment_logging_stat(user_profile.realm, COUNT_STATS[
        'mobile_pushes_sent::day'], None, timezone_now(), increment=
        apple_successfully_sent_count + android_successfully_sent_count)


def func_k1yhheky(user_identity, devices, base_payload, remote=None):
    payload = copy.deepcopy(base_payload)
    payload['event'] = 'test'
    apple_devices = [device for device in devices if device.kind ==
        PushDeviceToken.APNS]
    android_devices = [device for device in devices if device.kind ==
        PushDeviceToken.FCM]
    apple_payload = copy.deepcopy(payload)
    android_payload = copy.deepcopy(payload)
    realm_url = base_payload.get('realm_url', base_payload['realm_uri'])
    realm_name = base_payload['realm_name']
    apns_data = {'alert': {'title': _('Test notification'), 'body': _(
        'This is a test notification from {realm_name} ({realm_url}).').
        format(realm_name=realm_name, realm_url=realm_url)}, 'sound':
        'default', 'custom': {'zulip': apple_payload}}
    func_0qez25oh(user_identity, apple_devices, apns_data, remote=remote)
    android_payload['time'] = datetime_to_timestamp(timezone_now())
    gcm_options = {'priority': 'high'}
    func_rp9m8r9l(user_identity, android_devices, android_payload,
        gcm_options, remote=remote)


def func_91q5jso7(user_profile, devices):
    base_payload = func_3kruty6z(user_profile)
    if func_lnmnupsn():
        for device in devices:
            post_data = {'realm_uuid': str(user_profile.realm.uuid),
                'user_uuid': str(user_profile.uuid), 'user_id':
                user_profile.id, 'token': device.token, 'token_kind':
                device.kind, 'base_payload': base_payload}
            logger.info('Sending test push notification to bouncer: %r',
                post_data)
            send_json_to_push_bouncer('POST', 'push/test_notification',
                post_data)
        return
    user_identity = UserPushIdentityCompat(user_id=user_profile.id,
        user_uuid=str(user_profile.uuid))
    func_k1yhheky(user_identity, devices, base_payload, remote=None)


class InvalidPushDeviceTokenError(JsonableError):
    code = ErrorCode.INVALID_PUSH_DEVICE_TOKEN

    def __init__(self):
        pass

    @staticmethod
    @override
    def func_qs14phos():
        return _('Device not recognized')


class InvalidRemotePushDeviceTokenError(JsonableError):
    code = ErrorCode.INVALID_REMOTE_PUSH_DEVICE_TOKEN

    def __init__(self):
        pass

    @staticmethod
    @override
    def func_qs14phos():
        return _('Device not recognized by the push bouncer')


class PushNotificationsDisallowedByBouncerError(Exception):

    def __init__(self, reason):
        self.reason = reason


class HostnameAlreadyInUseBouncerError(JsonableError):
    code = ErrorCode.HOSTNAME_ALREADY_IN_USE_BOUNCER_ERROR
    data_fields = ['hostname']

    def __init__(self, hostname):
        self.hostname = hostname

    @staticmethod
    @override
    def func_qs14phos():
        return 'A server with hostname {hostname} already exists'
