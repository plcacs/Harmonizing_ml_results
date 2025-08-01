from typing import Dict, Any, List, Set
from django.conf import settings
from django.utils.translation import gettext as _
from zerver.lib.exceptions import JsonableError
from zerver.lib.stream_subscription import get_active_subscriptions_for_stream_id
from zerver.models import Realm, Stream, UserProfile
from zerver.models.users import get_user_by_id_in_realm_including_cross_realm
from zerver.tornado.django_api import send_event_rollback_unsafe


def do_send_typing_notification(
    realm: Realm,
    sender: UserProfile,
    recipient_user_profiles: List[UserProfile],
    operator: str,
) -> None:
    sender_dict: Dict[str, Any] = {"user_id": sender.id, "email": sender.email}
    recipient_dicts: List[Dict[str, Any]] = [
        {"user_id": profile.id, "email": profile.email} for profile in recipient_user_profiles
    ]
    event: Dict[str, Any] = {
        "type": "typing",
        "message_type": "direct",
        "op": operator,
        "sender": sender_dict,
        "recipients": recipient_dicts,
    }
    user_ids_to_notify: List[int] = [
        user.id for user in recipient_user_profiles if user.is_active and user.receives_typing_notifications
    ]
    send_event_rollback_unsafe(realm, event, user_ids_to_notify)


def check_send_typing_notification(
    sender: UserProfile, user_ids: List[int], operator: str
) -> None:
    realm: Realm = sender.realm
    if sender.id not in user_ids:
        user_ids.append(sender.id)
    user_profiles: List[UserProfile] = []
    for user_id in user_ids:
        try:
            user_profile: UserProfile = get_user_by_id_in_realm_including_cross_realm(
                user_id, sender.realm
            )
        except UserProfile.DoesNotExist:
            raise JsonableError(_("Invalid user ID {user_id}").format(user_id=user_id))
        user_profiles.append(user_profile)
    do_send_typing_notification(realm=realm, sender=sender, recipient_user_profiles=user_profiles, operator=operator)


def do_send_stream_typing_notification(
    sender: UserProfile, operator: str, stream: Stream, topic_name: str
) -> None:
    sender_dict: Dict[str, Any] = {"user_id": sender.id, "email": sender.email}
    event: Dict[str, Any] = {
        "type": "typing",
        "message_type": "stream",
        "op": operator,
        "sender": sender_dict,
        "stream_id": stream.id,
        "topic": topic_name,
    }
    subscriptions_query = get_active_subscriptions_for_stream_id(stream.id, include_deactivated_users=False)
    total_subscriptions: int = subscriptions_query.count()
    if total_subscriptions > settings.MAX_STREAM_SIZE_FOR_TYPING_NOTIFICATIONS:
        return
    user_ids_to_notify: Set[int] = set(
        subscriptions_query.exclude(user_profile__long_term_idle=True)
        .exclude(user_profile__receives_typing_notifications=False)
        .values_list("user_profile_id", flat=True)
    )
    send_event_rollback_unsafe(sender.realm, event, user_ids_to_notify)


def do_send_stream_message_edit_typing_notification(
    sender: UserProfile, channel_id: int, message_id: int, operator: str, topic_name: str
) -> None:
    event: Dict[str, Any] = {
        "type": "typing_edit_message",
        "op": operator,
        "sender_id": sender.id,
        "message_id": message_id,
        "recipient": {"type": "channel", "channel_id": channel_id, "topic": topic_name},
    }
    subscriptions_query = get_active_subscriptions_for_stream_id(channel_id, include_deactivated_users=False)
    total_subscriptions: int = subscriptions_query.count()
    if total_subscriptions > settings.MAX_STREAM_SIZE_FOR_TYPING_NOTIFICATIONS:
        return
    user_ids_to_notify: Set[int] = set(
        subscriptions_query.exclude(user_profile__long_term_idle=True)
        .exclude(user_profile__receives_typing_notifications=False)
        .values_list("user_profile_id", flat=True)
    )
    send_event_rollback_unsafe(sender.realm, event, user_ids_to_notify)


def do_send_direct_message_edit_typing_notification(
    sender: UserProfile, user_ids: List[int], message_id: int, operator: str
) -> None:
    recipient_user_profiles: List[UserProfile] = []
    for user_id in user_ids:
        user_profile: UserProfile = get_user_by_id_in_realm_including_cross_realm(user_id, sender.realm)
        recipient_user_profiles.append(user_profile)
    user_ids_to_notify: List[int] = [
        user.id for user in recipient_user_profiles if user.is_active and user.receives_typing_notifications
    ]
    event: Dict[str, Any] = {
        "type": "typing_edit_message",
        "op": operator,
        "sender_id": sender.id,
        "message_id": message_id,
        "recipient": {"type": "direct", "user_ids": user_ids_to_notify},
    }
    send_event_rollback_unsafe(sender.realm, event, user_ids_to_notify)