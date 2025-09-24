import asyncio
import base64
import copy
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from email.headerregistry import Address
from functools import cache
from typing import TYPE_CHECKING, Any, Optional, TypeAlias, Union, cast
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
    from zilencer.models import RemotePushDeviceToken, RemoteZulipServer
logger = logging.getLogger(__name__)
if settings.ZILENCER_ENABLED:
    from zilencer.models import RemotePushDeviceToken, RemoteZulipServer
DeviceToken: TypeAlias = Union[PushDeviceToken, 'RemotePushDeviceToken']

def b64_to_hex(data: str) -> str:
    return base64.b64decode(data).hex()

def hex_to_b64(data: str) -> str:
    return base64.b64encode(bytes.fromhex(data)).decode()

def get_message_stream_name_from_database(message: Message) -> str:
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

    def __init__(self, user_id: int | None = None, user_uuid: str | None = None) -> None:
        assert user_id is not None or user_uuid is not None
        self.user_id = user_id
        self.user_uuid = user_uuid

    def filter_q(self) -> Q:
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
    def __str__(self) -> str:
        result = ''
        if self.user_id is not None:
            result += f'<id:{self.user_id}>'
        if self.user_uuid is not None:
            result += f'<uuid:{self.user_uuid}>'
        return result

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, UserPushIdentityCompat):
            return self.user_id == other.user_id and self.user_uuid == other.user_uuid
        return False

@dataclass
class APNsContext:
    apns: 'aioapns.APNs'
    loop: asyncio.AbstractEventLoop

def has_apns_credentials() -> bool:
    return settings.APNS_TOKEN_KEY_FILE is not None or settings.APNS_CERT_FILE is not None

@cache
def get_apns_context() -> Optional[APNsContext]:
    import aioapns
    if not has_apns_credentials():
        return None
    loop = asyncio.new_event_loop()

    async def err_func(request: aioapns.NotificationRequest, result: aioapns.common.NotificationResult) -> None:
        pass

    async def make_apns() -> aioapns.APNs:
        return aioapns.APNs(client_cert=settings.APNS_CERT_FILE, key=settings.APNS_TOKEN_KEY_FILE, key_id=settings.APNS_TOKEN_KEY_ID, team_id=settings.APNS_TEAM_ID, max_connection_attempts=APNS_MAX_RETRIES, use_sandbox=settings.APNS_SANDBOX, err_func=err_func, topic='invalid.nonsense')
    apns = loop.run_until_complete(make_apns())
    return APNsContext(apns=apns, loop=loop)

def modernize_apns_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Take a payload in an unknown Zulip version's format, and return in current format."""
    if 'message_ids' in data:
        return {'alert': data['alert'], 'badge': 0, 'custom': {'zulip': {'message_ids': data['message_ids']}}}
    else:
        return data
APNS_MAX_RETRIES: int = 3

def send_apple_push_notification(user_identity: UserPushIdentityCompat, devices: Sequence[DeviceToken], payload_data: dict[str, Any], remote: Optional['RemoteZulipServer'] = None) -> int:
    if not devices:
        return 0
    import aioapns
    import aioapns.exceptions
    apns_context = get_apns_context()
    if apns_context is None:
        logger.debug('APNs: Dropping a notification because nothing configured.  Set ZULIP_SERVICES_URL (or APNS_CERT_FILE).')
        return 0
    if remote:
        assert settings.ZILENCER_ENABLED
        DeviceTokenClass: type[AbstractPushDeviceToken] = RemotePushDeviceToken
    else:
        DeviceTokenClass = PushDeviceToken
    if remote:
        logger.info('APNs: Sending notification for remote user %s:%s to %d devices', remote.uuid, user_identity, len(devices))
    else:
        logger.info('APNs: Sending notification for local user %s to %d devices', user_identity, len(devices))
    payload_data = dict(modernize_apns_payload(payload_data))
    message: dict[str, Any] = {**payload_data.pop('custom', {}), 'aps': payload_data}
    have_missing_app_id = False
    for device in devices:
        if device.ios_app_id is None:
            logger.error('APNs: Missing ios_app_id for user %s device %s', user_identity, device.token)
            have_missing_app_id = True
    if have_missing_app_id:
        devices = [device for device in devices if device.ios_app_id is not None]

    async def send_all_notifications() -> Iterable[tuple[DeviceToken, Union[aioapns.common.NotificationResult, BaseException]]]:
        requests = [aioapns.NotificationRequest(apns_topic=device.ios_app_id, device_token=device.token, message=message, time_to_live=24 * 3600) for device in devices]
        results = await asyncio.gather(*(apns_context.apns.send_notification(request) for request in requests), return_exceptions=True)
        return zip(devices, results, strict=False)
    results = apns_context.loop.run_until_complete(send_all_notifications())
    successfully_sent_count = 0
    for (device, result) in results:
        if isinstance(result, aioapns.exceptions.ConnectionError):
            logger.error('APNs: ConnectionError sending for user %s to device %s; check certificate expiration', user_identity, device.token)
        elif isinstance(result, BaseException):
            logger.error('APNs: Error sending for user %s to device %s', user_identity, device.token, exc_info=result)
        elif result.is_successful:
            successfully_sent_count += 1
            logger.info('APNs: Success sending for user %s to device %s', user_identity, device.token)
        elif result.description in ['Unregistered', 'BadDeviceToken', 'DeviceTokenNotForTopic']:
            logger.info('APNs: Removing invalid/expired token %s (%s)', device.token, result.description)
            DeviceTokenClass._default_manager.filter(token=device.token, kind=DeviceTokenClass.APNS).delete()
        else:
            logger.warning('APNs: Failed to send for user %s to device %s: %s', user_identity, device.token, result.description)
    return successfully_sent_count
FCM_REQUEST_TIMEOUT: int = 5

def make_fcm_app() -> Optional[FCMApp]:
    if settings.ANDROID_FCM_CREDENTIALS_PATH is None:
        return None
    fcm_credentials = firebase_credentials.Certificate(settings.ANDROID_FCM_CREDENTIALS_PATH)
    fcm_app = firebase_initialize_app(fcm_credentials, options=dict(httpTimeout=FCM_REQUEST_TIMEOUT))
    return fcm_app
if settings.ANDROID_FCM_CREDENTIALS_PATH:
    fcm_app: Optional[FCMApp] = make_fcm_app()
else:
    fcm_app = None

def has_fcm_credentials() -> bool:
    return fcm_app is not None

def send_android_push_notification_to_user(user_profile: UserProfile, data: dict[str, Any], options: dict[str, Any]) -> None:
    devices = list(PushDeviceToken.objects.filter(user=user_profile, kind=PushDeviceToken.FCM))
    send_android_push_notification(UserPushIdentityCompat(user_id=user_profile.id), devices, data, options)

def parse_fcm_options(options: dict[str, Any], data: dict[str, Any]) -> str:
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
        raise JsonableError(_('Invalid GCM option to bouncer: priority {priority!r}').format(priority=priority))
    if options:
        raise JsonableError(_('Invalid GCM options to bouncer: {options}').format(options=orjson.dumps(options).decode()))
    return priority

def send_android_push_notification(user_identity: UserPushIdentityCompat, devices: Sequence[DeviceToken], data: dict[str, Any], options: dict[str, Any], remote: Optional['RemoteZulipServer'] = None) -> int:
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
        logger.debug('Skipping sending a FCM push notification since ZULIP_SERVICE_PUSH_NOTIFICATIONS and ANDROID_FCM_CREDENTIALS_PATH are both unset')
        return 0
    if remote:
        logger.info('FCM: Sending notification for remote user %s:%s to %d devices', remote.uuid, user_identity, len(devices))
    else:
        logger.info('FCM: Sending notification for local user %s to %d devices', user_identity, len(devices))
    token_list = [device.token for device in devices]
    priority = parse_fcm_options(options, data)
    data = {k: str(v) if not isinstance(v, str) else v for (k, v) in data.items()}
    messages = [firebase_messaging.Message(data=data, token=token, android=firebase_messaging.AndroidConfig(priority=priority)) for token in token_list]
    try:
        batch_response = firebase_messaging.send_each(messages, app=fcm_app)
    except firebase_exceptions.FirebaseError:
        logger.warning('Error while pushing to FCM', exc_info=True)
        return 0
    if remote:
        assert settings.ZILENCER_ENABLED
        DeviceTokenClass: type[AbstractPushDeviceToken] = RemotePushDeviceToken
    else:
        DeviceTokenClass = PushDeviceToken
    successfully_sent_count = 0
    for (idx, response) in enumerate(batch_response.responses):
        token = token_list[idx]
        if response.success:
            successfully_sent_count += 1
            logger.info('FCM: Sent message with ID: %s to %s', response.message_id, token)
        else:
            error = response.exception
            if isinstance(error, FCMUnregisteredError):
                logger.info('FCM: Removing %s due to %s', token, error.code)
                DeviceTokenClass._default_manager.filter(token=token, kind=DeviceTokenClass.FCM).delete()
            else:
                logger.warning('FCM: Delivery failed for %s: %s:%s', token, error.__class__, error)
    return successfully_sent_count

def uses_notification_bouncer() -> bool:
    return settings.ZULIP_SERVICE_PUSH_NOTIFICATIONS is True

def sends_notifications_directly() -> bool:
    return has_apns_credentials() and has_fcm_credentials() and (not uses_notification_bouncer())

def send_notifications_to_bouncer(user_profile: UserProfile, apns_payload: dict[str, Any], gcm_payload: dict[str, Any], gcm_options: dict[str, Any], android_devices: Sequence[PushDeviceToken], apple_devices: Sequence[PushDeviceToken]) -> None:
    if len(android_devices) + len(apple_devices) == 0:
        logger.info('Skipping contacting the bouncer for user %s because there are no registered devices', user_profile.id)
        return
    post_data = {'user_uuid': str(user_profile.uuid), 'user_id': user_profile.id, 'realm_uuid': str(user_profile.realm.uuid), 'apns_payload': apns_payload, 'gcm_payload': gcm_payload, 'gcm_options': gcm_options, 'android_devices': [device.token for device in android_devices], 'apple_devices': [device.token for device in apple_devices]}
    try:
        response_data = send_json_to_push_bouncer('POST', 'push/notify', post_data)
    except PushNotificationsDisallowedByBouncerError as e:
        logger.warning('Bouncer refused to send push notification: %s', e.reason)
        do_set_realm_property(user_profile.realm, 'push_notifications_enabled', False, acting_user=None)
        do_set_push_notifications_enabled_end_timestamp(user_profile.realm, None, acting_user=None)
        return
    assert isinstance(response_data['total_android_devices'], int)
    assert isinstance(response_data['total_apple_devices'], int)
    assert isinstance(response_data['deleted_devices'], dict)
    assert isinstance(response_data['deleted_devices']['android_devices'], list)
    assert isinstance(response_data['deleted_devices']['apple_devices'], list)
    android_deleted_devices = response_data['deleted_devices']['android_devices']
    apple_deleted_devices = response_data['deleted_devices']['apple_devices']
    if android_deleted_devices or apple