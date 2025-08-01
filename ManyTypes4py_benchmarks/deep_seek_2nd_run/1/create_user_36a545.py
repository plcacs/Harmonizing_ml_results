from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import suppress
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from django.conf import settings
from django.db import IntegrityError, transaction
from django.db.models import F, QuerySet
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import override as override_language
from confirmation import settings as confirmation_settings
from zerver.actions.message_send import internal_send_group_direct_message, internal_send_private_message, internal_send_stream_message
from zerver.actions.streams import bulk_add_subscriptions, send_peer_subscriber_events
from zerver.actions.user_groups import bulk_add_members_to_user_groups, do_send_user_group_members_update_event
from zerver.actions.users import change_user_is_active, get_service_dicts_for_bot, send_update_events_for_anonymous_group_settings
from zerver.lib.avatar import avatar_url
from zerver.lib.create_user import create_user
from zerver.lib.default_streams import get_slim_realm_default_streams
from zerver.lib.email_notifications import enqueue_welcome_emails, send_account_registered_email
from zerver.lib.exceptions import JsonableError
from zerver.lib.invites import notify_invites_changed
from zerver.lib.mention import silent_mention_syntax_for_user
from zerver.lib.remote_server import maybe_enqueue_audit_log_upload
from zerver.lib.send_email import clear_scheduled_invitation_emails
from zerver.lib.streams import can_access_stream_history
from zerver.lib.subscription_info import bulk_get_subscriber_peer_info
from zerver.lib.user_counts import realm_user_count, realm_user_count_by_role
from zerver.lib.user_groups import get_system_user_group_for_user
from zerver.lib.users import can_access_delivery_email, format_user_row, get_data_for_inaccessible_user, get_user_ids_who_can_access_user, user_access_restricted_in_realm, user_profile_to_user_row
from zerver.models import DefaultStreamGroup, Message, NamedUserGroup, OnboardingStep, OnboardingUserMessage, PreregistrationRealm, PreregistrationUser, Realm, RealmAuditLog, Recipient, Stream, Subscription, UserGroupMembership, UserMessage, UserProfile
from zerver.models.groups import SystemGroups
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.users import active_user_ids, bot_owner_user_ids, get_system_bot

MAX_NUM_RECENT_MESSAGES: int = 1000
MAX_NUM_RECENT_UNREAD_MESSAGES: int = 20

def send_message_to_signup_notification_stream(sender: UserProfile, realm: Realm, message: str) -> None:
    signup_announcements_stream: Optional[Stream] = realm.signup_announcements_stream
    if signup_announcements_stream is None:
        return
    with override_language(realm.default_language):
        topic_name: str = _('signups')
    internal_send_stream_message(sender, signup_announcements_stream, topic_name, message)

def send_group_direct_message_to_admins(sender: UserProfile, realm: Realm, content: str) -> None:
    administrators: List[UserProfile] = list(realm.get_human_admin_users())
    internal_send_group_direct_message(realm, sender, content, recipient_users=administrators)

def notify_new_user(user_profile: UserProfile) -> None:
    user_count: int = realm_user_count(user_profile.realm)
    sender_email: str = settings.NOTIFICATION_BOT
    sender: UserProfile = get_system_bot(sender_email, user_profile.realm_id)
    is_first_user: bool = user_count == 1
    if not is_first_user:
        with override_language(user_profile.realm.default_language):
            message: str = _('{user} joined this organization.').format(user=silent_mention_syntax_for_user(user_profile), user_count=user_count)
            send_message_to_signup_notification_stream(sender, user_profile.realm, message)
        if settings.BILLING_ENABLED:
            from corporate.lib.registration import generate_licenses_low_warning_message_if_required
            licenses_low_warning_message: Optional[str] = generate_licenses_low_warning_message_if_required(user_profile.realm)
            if licenses_low_warning_message is not None:
                message += '\n'
                message += licenses_low_warning_message
                send_group_direct_message_to_admins(sender, user_profile.realm, message)

def set_up_streams_and_groups_for_new_human_user(*, user_profile: UserProfile, prereg_user: Optional[PreregistrationUser]=None, default_stream_groups: List[DefaultStreamGroup]=[], add_initial_stream_subscriptions: bool=True, realm_creation: bool=False) -> None:
    realm: Realm = user_profile.realm
    streams: List[Stream] = []
    user_groups: List[NamedUserGroup] = []
    acting_user: Optional[UserProfile] = None
    if prereg_user is not None:
        streams = list(prereg_user.streams.all())
        user_groups = list(prereg_user.groups.all())
        acting_user = prereg_user.referred_by
        assert prereg_user.created_user is None, 'PregistrationUser should not be reused'
    
    if add_initial_stream_subscriptions:
        if prereg_user is None or prereg_user.include_realm_default_subscriptions:
            default_streams: QuerySet[Stream] = get_slim_realm_default_streams(realm.id)
            streams = list(set(streams) | set(default_streams))
        for default_stream_group in default_stream_groups:
            default_stream_group_streams: QuerySet[Stream] = default_stream_group.streams.all()
            for stream in default_stream_group_streams:
                if stream not in streams:
                    streams.append(stream)
    else:
        streams = []
    bulk_add_subscriptions(realm, streams, [user_profile], from_user_creation=True, acting_user=acting_user)
    bulk_add_members_to_user_groups(user_groups, [user_profile.id], acting_user=acting_user)
    add_new_user_history(user_profile, streams, realm_creation=realm_creation)

def add_new_user_history(user_profile: UserProfile, streams: List[Stream], *, realm_creation: bool=False) -> None:
    realm: Realm = user_profile.realm
    recipient_ids: List[int] = [stream.recipient_id for stream in streams if can_access_stream_history(user_profile, stream)]
    recent_message_ids: Set[int] = set(Message.objects.filter(realm_id=realm.id, recipient_id__in=recipient_ids).order_by('-id').values_list('id', flat=True)[0:MAX_NUM_RECENT_MESSAGES])
    tracked_onboarding_message_ids: Set[int] = set()
    message_id_to_onboarding_user_message: Dict[int, OnboardingUserMessage] = {}
    onboarding_user_messages_queryset: QuerySet[OnboardingUserMessage] = OnboardingUserMessage.objects.filter(realm_id=realm.id)
    for onboarding_user_message in onboarding_user_messages_queryset:
        tracked_onboarding_message_ids.add(onboarding_user_message.message_id)
        message_id_to_onboarding_user_message[onboarding_user_message.message_id] = onboarding_user_message
    tracked_onboarding_messages_exist: bool = len(tracked_onboarding_message_ids) > 0
    message_history_ids: Set[int] = recent_message_ids.union(tracked_onboarding_message_ids)
    if len(message_history_ids) > 0:
        already_used_ids: Set[int] = set(UserMessage.objects.filter(message_id__in=recent_message_ids, user_profile=user_profile).values_list('message_id', flat=True))
        backfill_message_ids: List[int] = sorted(message_history_ids - already_used_ids)
        older_message_ids: Set[int] = set()
        if not tracked_onboarding_messages_exist:
            older_message_ids = set(backfill_message_ids[:-MAX_NUM_RECENT_UNREAD_MESSAGES])
        ums_to_create: List[UserMessage] = []
        for message_id in backfill_message_ids:
            um: UserMessage = UserMessage(user_profile=user_profile, message_id=message_id)
            if not realm_creation:
                um.flags = UserMessage.flags.historical
            if tracked_onboarding_messages_exist:
                if message_id not in tracked_onboarding_message_ids:
                    um.flags |= UserMessage.flags.read
                elif message_id_to_onboarding_user_message[message_id].flags.starred.is_set:
                    um.flags |= UserMessage.flags.starred
            elif message_id in older_message_ids:
                um.flags |= UserMessage.flags.read
            ums_to_create.append(um)
        UserMessage.objects.bulk_create(ums_to_create)

def process_new_human_user(user_profile: UserProfile, prereg_user: Optional[PreregistrationUser]=None, default_stream_groups: List[DefaultStreamGroup]=[], realm_creation: bool=False, add_initial_stream_subscriptions: bool=True) -> None:
    set_up_streams_and_groups_for_new_human_user(user_profile=user_profile, prereg_user=prereg_user, default_stream_groups=default_stream_groups, add_initial_stream_subscriptions=add_initial_stream_subscriptions, realm_creation=realm_creation)
    realm: Realm = user_profile.realm
    mit_beta_user: bool = realm.is_zephyr_mirror_realm
    if not mit_beta_user and prereg_user is not None and (prereg_user.referred_by is not None) and prereg_user.referred_by.is_active and prereg_user.notify_referrer_on_join:
        with override_language(prereg_user.referred_by.default_language):
            internal_send_private_message(get_system_bot(settings.NOTIFICATION_BOT, prereg_user.referred_by.realm_id), prereg_user.referred_by, _('{user} accepted your invitation to join Zulip!').format(user=silent_mention_syntax_for_user(user_profile)))
    if prereg_user is not None:
        prereg_user.status = confirmation_settings.STATUS_USED
        prereg_user.created_user = user_profile
        prereg_user.save(update_fields=['status', 'created_user'])
    if prereg_user is not None:
        PreregistrationUser.objects.filter(email__iexact=user_profile.delivery_email, realm=user_profile.realm).exclude(id=prereg_user.id).update(status=confirmation_settings.STATUS_REVOKED)
    else:
        PreregistrationUser.objects.filter(email__iexact=user_profile.delivery_email, realm=user_profile.realm).update(status=confirmation_settings.STATUS_REVOKED)
    if prereg_user is not None and prereg_user.referred_by is not None:
        notify_invites_changed(user_profile.realm, changed_invite_referrer=prereg_user.referred_by)
    notify_new_user(user_profile)
    clear_scheduled_invitation_emails(user_profile.delivery_email)
    if realm.send_welcome_emails:
        enqueue_welcome_emails(user_profile, realm_creation)
    send_account_registered_email(user_profile, realm_creation)
    from zerver.lib.onboarding import send_initial_direct_message
    message_id: int = send_initial_direct_message(user_profile)
    UserMessage.objects.filter(user_profile=user_profile, message_id=message_id).update(flags=F('flags').bitor(UserMessage.flags.starred))
    with suppress(IntegrityError), transaction.atomic(savepoint=True):
        OnboardingStep.objects.create(user=user_profile, onboarding_step='visibility_policy_banner')

def notify_created_user(user_profile: UserProfile, notify_user_ids: List[int]) -> None:
    user_row: Dict[str, Any] = user_profile_to_user_row(user_profile)
    format_user_row_kwargs: Dict[str, Any] = {'realm_id': user_profile.realm_id, 'row': user_row, 'client_gravatar': False, 'user_avatar_url_field_optional': False, 'custom_profile_field_data': {}}
    user_ids_without_access_to_created_user: List[int] = []
    users_with_access_to_created_users: List[UserProfile] = []
    if notify_user_ids:
        users_with_access_to_created_users = list(user_profile.realm.get_active_users().filter(id__in=notify_user_ids))
    else:
        active_realm_users: List[UserProfile] = list(user_profile.realm.get_active_users())
        if user_access_restricted_in_realm(user_profile):
            for user in active_realm_users:
                if user.is_guest:
                    user_ids_without_access_to_created_user.append(user.id)
                else:
                    users_with_access_to_created_users.append(user)
        else:
            users_with_access_to_created_users = active_realm_users
    user_ids_with_real_email_access: List[int] = []
    user_ids_without_real_email_access: List[int] = []
    person_for_real_email_access_users: Optional[Dict[str, Any]] = None
    person_for_without_real_email_access_users: Optional[Dict[str, Any]] = None
    for recipient_user in users_with_access_to_created_users:
        if can_access_delivery_email(recipient_user, user_profile.id, user_row['email_address_visibility']):
            user_ids_with_real_email_access.append(recipient_user.id)
            if person_for_real_email_access_users is None:
                person_for_real_email_access_users = format_user_row(**format_user_row_kwargs, acting_user=recipient_user)
        else:
            user_ids_without_real_email_access.append(recipient_user.id)
            if person_for_without_real_email_access_users is None:
                person_for_without_real_email_access_users = format_user_row(**format_user_row_kwargs, acting_user=recipient_user)
    if user_ids_with_real_email_access:
        assert person_for_real_email_access_users is not None
        event: Dict[str, Any] = dict(type='realm_user', op='add', person=person_for_real_email_access_users)
        send_event_on_commit(user_profile.realm, event, user_ids_with_real_email_access)
    if user_ids_without_real_email_access:
        assert person_for_without_real_email_access_users is not None
        event = dict(type='realm_user', op='add', person=person_for_without_real_email_access_users)
        send_event_on_commit(user_profile.realm, event, user_ids_without_real_email_access)
    if user_ids_without_access_to_created_user:
        event = dict(type='realm_user', op='add', person=get_data_for_inaccessible_user(user_profile.realm, user_profile.id), inaccessible_user=True)
        send_event_on_commit(user_profile.realm, event, user_ids_without_access_to_created_user)

def created_bot_event(user_profile: UserProfile) -> Dict[str, Any]:
    def stream_name(stream: Optional[Stream]) -> Optional[str]:
        if not stream:
            return None
        return stream.name
    default_sending_stream_name: Optional[str] = stream_name(user_profile.default_sending_stream)
    default_events_register_stream_name: Optional[str] = stream_name(user_profile.default_events_register_stream)
    bot: Dict[str, Any] = dict(
        email=user_profile.email,
        user_id=user_profile.id,
        full_name=user_profile.full_name,
        bot_type=user_profile.bot_type,
        is_active=user_profile.is_active,
        api_key=user_profile.api_key,
        default_sending_stream=default_sending_stream_name,
        default_events_register_stream=default_events_register_stream_name,
        default_all_public_streams=user_profile.default_all_public_streams,
        avatar_url=avatar_url(user_profile),
        services=get_service_dicts_for_bot(user_profile.id)
    )
    if user_profile.bot_owner_id is not None:
        bot['owner_id'] = user_profile.bot_owner_id
    return dict(type='realm_bot', op='add', bot=bot)

def notify_created_bot(user_profile: UserProfile) -> None:
    event: Dict[str, Any] = created_bot_event(user_profile)
    send_event_on_commit(user_profile.realm, event, bot_owner_user_ids(user_profile))

@transaction.atomic(durable=True)
def do_create_user(
    email: str,
    password: Optional[str],
    realm: Realm,
    full_name: str,
    bot_type: Optional[int]=None,
    role: Optional[int]=None,
    bot_owner: Optional[UserProfile]=None,
    tos_version: Optional[str]=None,
    timezone: str='',
    avatar_source: str=UserProfile.AVATAR_FROM_GRAVATAR,
    default_language: Optional[str]=None,
    default_sending_stream: Optional[Stream]=None,
    default_events_register_stream: Optional[Stream]=None,
    default_all_public_streams: Optional[bool]=None,
    prereg_user: Optional[PreregistrationUser]=None,
    prereg_realm: Optional[PreregistrationRealm]=None,
    default_stream_groups: List[DefaultStreamGroup]=[],
    source_profile: Optional[UserProfile]=None,
    realm_creation: bool=False,
    *,
    acting_user: Optional[UserProfile],
    enable_marketing_emails: bool=True,
    email_address_visibility: Optional[int]=None,
    add_initial_stream_subscriptions: bool=True
) -> UserProfile:
    if settings.BILLING_ENABLED:
        from corporate.lib.stripe import RealmBillingSession
    user_profile: UserProfile = create_user(
        email=email,
        password=password,
        realm=realm,
        full_name=full_name,
        role=role,
        bot_type=bot_type,
        bot_owner=bot_owner,
        tos_version=tos_version,
        timezone=timezone,
        avatar_source=avatar_source,
        default_language=default_language,
        default_sending_stream=default_sending_stream,
        default_events_register_stream=default_events_register_stream,
        default_all_public_streams=default_all_public_streams,
        source_profile=source_profile,
        enable_marketing_emails=enable_marketing_emails,
        email_address_visibility=email_address_visibility
    )
    event_time = user_profile.date_joined
    if not acting_user:
        acting_user = user_profile
