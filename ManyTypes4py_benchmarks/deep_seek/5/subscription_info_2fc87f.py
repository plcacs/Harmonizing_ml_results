import itertools
from collections.abc import Callable, Collection, Iterable, Mapping
from operator import itemgetter
from typing import Any, Dict, List, Optional, Set, Tuple, cast
from django.core.exceptions import ValidationError
from django.db import connection
from django.db.models import QuerySet
from django.utils.translation import gettext as _
from psycopg2.sql import SQL
from zerver.lib.exceptions import JsonableError
from zerver.lib.stream_color import STREAM_ASSIGNMENT_COLORS
from zerver.lib.stream_subscription import SubscriberPeerInfo, get_active_subscriptions_for_stream_id, get_stream_subscriptions_for_user, get_user_ids_for_streams
from zerver.lib.stream_traffic import get_average_weekly_stream_traffic, get_streams_traffic
from zerver.lib.streams import UserGroupMembershipDetails, get_group_setting_value_dict_for_streams, get_setting_values_for_group_settings, get_stream_post_policy_value_based_on_group_setting, get_user_ids_with_metadata_access_via_permission_groups, get_web_public_streams_queryset, has_metadata_access_to_channel_via_groups, subscribed_to_stream
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.types import AnonymousSettingGroupDict, APIStreamDict, NeverSubscribedStreamDict, RawStreamDict, RawSubscriptionDict, SubscriptionInfo, SubscriptionStreamDict
from zerver.lib.user_groups import get_recursive_membership_groups
from zerver.models import Realm, Stream, Subscription, UserProfile
from zerver.models.streams import get_all_streams

def get_web_public_subs(realm: Realm) -> SubscriptionInfo:
    color_idx = 0

    def get_next_color() -> str:
        nonlocal color_idx
        color = STREAM_ASSIGNMENT_COLORS[color_idx]
        color_idx = (color_idx + 1) % len(STREAM_ASSIGNMENT_COLORS)
        return color
    
    subscribed: List[SubscriptionStreamDict] = []
    streams = get_web_public_streams_queryset(realm)
    setting_groups_dict = get_group_setting_value_dict_for_streams(list(streams))
    for stream in streams:
        is_archived = stream.deactivated
        can_add_subscribers_group = setting_groups_dict[stream.can_add_subscribers_group_id]
        can_administer_channel_group = setting_groups_dict[stream.can_administer_channel_group_id]
        can_send_message_group = setting_groups_dict[stream.can_send_message_group_id]
        can_remove_subscribers_group = setting_groups_dict[stream.can_remove_subscribers_group_id]
        creator_id = stream.creator_id
        date_created = datetime_to_timestamp(stream.date_created)
        description = stream.description
        first_message_id = stream.first_message_id
        is_recently_active = stream.is_recently_active
        history_public_to_subscribers = stream.history_public_to_subscribers
        invite_only = stream.invite_only
        is_web_public = stream.is_web_public
        message_retention_days = stream.message_retention_days
        name = stream.name
        rendered_description = stream.rendered_description
        stream_id = stream.id
        stream_post_policy = get_stream_post_policy_value_based_on_group_setting(stream.can_send_message_group)
        is_announcement_only = stream_post_policy == Stream.STREAM_POST_POLICY_ADMINS
        audible_notifications = True
        color = get_next_color()
        desktop_notifications = True
        email_notifications = True
        in_home_view = True
        is_muted = False
        pin_to_top = False
        push_notifications = True
        stream_weekly_traffic = get_average_weekly_stream_traffic(stream.id, stream.date_created, {})
        wildcard_mentions_notify = True
        sub = SubscriptionStreamDict(
            is_archived=is_archived, audible_notifications=audible_notifications,
            can_add_subscribers_group=can_add_subscribers_group,
            can_administer_channel_group=can_administer_channel_group,
            can_send_message_group=can_send_message_group,
            can_remove_subscribers_group=can_remove_subscribers_group,
            color=color, creator_id=creator_id, date_created=date_created,
            description=description, desktop_notifications=desktop_notifications,
            email_notifications=email_notifications, first_message_id=first_message_id,
            is_recently_active=is_recently_active,
            history_public_to_subscribers=history_public_to_subscribers,
            in_home_view=in_home_view, invite_only=invite_only,
            is_announcement_only=is_announcement_only, is_muted=is_muted,
            is_web_public=is_web_public, message_retention_days=message_retention_days,
            name=name, pin_to_top=pin_to_top, push_notifications=push_notifications,
            rendered_description=rendered_description, stream_id=stream_id,
            stream_post_policy=stream_post_policy, stream_weekly_traffic=stream_weekly_traffic,
            wildcard_mentions_notify=wildcard_mentions_notify
        )
        subscribed.append(sub)
    return SubscriptionInfo(subscriptions=subscribed, unsubscribed=[], never_subscribed=[])

def build_unsubscribed_sub_from_stream_dict(
    user: UserProfile,
    sub_dict: RawSubscriptionDict,
    stream_dict: APIStreamDict
) -> SubscriptionStreamDict:
    subscription_stream_dict = build_stream_dict_for_sub(user, sub_dict, stream_dict)
    return subscription_stream_dict

def build_stream_api_dict(
    raw_stream_dict: RawStreamDict,
    recent_traffic: Optional[Dict[int, int]],
    setting_groups_dict: Dict[int, AnonymousSettingGroupDict]
) -> APIStreamDict:
    if recent_traffic is not None:
        stream_weekly_traffic = get_average_weekly_stream_traffic(
            raw_stream_dict['id'], raw_stream_dict['date_created'], recent_traffic
        )
    else:
        stream_weekly_traffic = None
    is_announcement_only = raw_stream_dict['stream_post_policy'] == Stream.STREAM_POST_POLICY_ADMINS
    can_add_subscribers_group = setting_groups_dict[raw_stream_dict['can_add_subscribers_group_id']]
    can_administer_channel_group = setting_groups_dict[raw_stream_dict['can_administer_channel_group_id']]
    can_send_message_group = setting_groups_dict[raw_stream_dict['can_send_message_group_id']]
    can_remove_subscribers_group = setting_groups_dict[raw_stream_dict['can_remove_subscribers_group_id']]
    return APIStreamDict(
        is_archived=raw_stream_dict['deactivated'],
        can_add_subscribers_group=can_add_subscribers_group,
        can_administer_channel_group=can_administer_channel_group,
        can_send_message_group=can_send_message_group,
        can_remove_subscribers_group=can_remove_subscribers_group,
        creator_id=raw_stream_dict['creator_id'],
        date_created=datetime_to_timestamp(raw_stream_dict['date_created']),
        description=raw_stream_dict['description'],
        first_message_id=raw_stream_dict['first_message_id'],
        history_public_to_subscribers=raw_stream_dict['history_public_to_subscribers'],
        invite_only=raw_stream_dict['invite_only'],
        is_web_public=raw_stream_dict['is_web_public'],
        message_retention_days=raw_stream_dict['message_retention_days'],
        name=raw_stream_dict['name'],
        rendered_description=raw_stream_dict['rendered_description'],
        stream_id=raw_stream_dict['id'],
        stream_post_policy=raw_stream_dict['stream_post_policy'],
        stream_weekly_traffic=stream_weekly_traffic,
        is_announcement_only=is_announcement_only,
        is_recently_active=raw_stream_dict['is_recently_active']
    )

def build_stream_dict_for_sub(
    user: UserProfile,
    sub_dict: RawSubscriptionDict,
    stream_dict: APIStreamDict
) -> SubscriptionStreamDict:
    is_archived = stream_dict['is_archived']
    can_add_subscribers_group = stream_dict['can_add_subscribers_group']
    can_administer_channel_group = stream_dict['can_administer_channel_group']
    can_send_message_group = stream_dict['can_send_message_group']
    can_remove_subscribers_group = stream_dict['can_remove_subscribers_group']
    creator_id = stream_dict['creator_id']
    date_created = stream_dict['date_created']
    description = stream_dict['description']
    first_message_id = stream_dict['first_message_id']
    history_public_to_subscribers = stream_dict['history_public_to_subscribers']
    invite_only = stream_dict['invite_only']
    is_web_public = stream_dict['is_web_public']
    message_retention_days = stream_dict['message_retention_days']
    name = stream_dict['name']
    rendered_description = stream_dict['rendered_description']
    stream_id = stream_dict['stream_id']
    stream_post_policy = stream_dict['stream_post_policy']
    stream_weekly_traffic = stream_dict['stream_weekly_traffic']
    is_announcement_only = stream_dict['is_announcement_only']
    is_recently_active = stream_dict['is_recently_active']
    color = sub_dict['color']
    is_muted = sub_dict['is_muted']
    pin_to_top = sub_dict['pin_to_top']
    audible_notifications = sub_dict['audible_notifications']
    desktop_notifications = sub_dict['desktop_notifications']
    email_notifications = sub_dict['email_notifications']
    push_notifications = sub_dict['push_notifications']
    wildcard_mentions_notify = sub_dict['wildcard_mentions_notify']
    in_home_view = not is_muted
    return SubscriptionStreamDict(
        is_archived=is_archived, audible_notifications=audible_notifications,
        can_add_subscribers_group=can_add_subscribers_group,
        can_administer_channel_group=can_administer_channel_group,
        can_send_message_group=can_send_message_group,
        can_remove_subscribers_group=can_remove_subscribers_group,
        color=color, creator_id=creator_id, date_created=date_created,
        description=description, desktop_notifications=desktop_notifications,
        email_notifications=email_notifications, first_message_id=first_message_id,
        is_recently_active=is_recently_active,
        history_public_to_subscribers=history_public_to_subscribers,
        in_home_view=in_home_view, invite_only=invite_only,
        is_announcement_only=is_announcement_only, is_muted=is_muted,
        is_web_public=is_web_public, message_retention_days=message_retention_days,
        name=name, pin_to_top=pin_to_top, push_notifications=push_notifications,
        rendered_description=rendered_description, stream_id=stream_id,
        stream_post_policy=stream_post_policy, stream_weekly_traffic=stream_weekly_traffic,
        wildcard_mentions_notify=wildcard_mentions_notify
    )

def build_stream_dict_for_never_sub(
    raw_stream_dict: RawStreamDict,
    recent_traffic: Optional[Dict[int, int]],
    setting_groups_dict: Dict[int, AnonymousSettingGroupDict]
) -> NeverSubscribedStreamDict:
    is_archived = raw_stream_dict['deactivated']
    creator_id = raw_stream_dict['creator_id']
    date_created = datetime_to_timestamp(raw_stream_dict['date_created'])
    description = raw_stream_dict['description']
    first_message_id = raw_stream_dict['first_message_id']
    is_recently_active = raw_stream_dict['is_recently_active']
    history_public_to_subscribers = raw_stream_dict['history_public_to_subscribers']
    invite_only = raw_stream_dict['invite_only']
    is_web_public = raw_stream_dict['is_web_public']
    message_retention_days = raw_stream_dict['message_retention_days']
    name = raw_stream_dict['name']
    rendered_description = raw_stream_dict['rendered_description']
    stream_id = raw_stream_dict['id']
    stream_post_policy = raw_stream_dict['stream_post_policy']
    if recent_traffic is not None:
        stream_weekly_traffic = get_average_weekly_stream_traffic(
            raw_stream_dict['id'], raw_stream_dict['date_created'], recent_traffic
        )
    else:
        stream_weekly_traffic = None
    can_add_subscribers_group_value = setting_groups_dict[raw_stream_dict['can_add_subscribers_group_id']]
    can_administer_channel_group_value = setting_groups_dict[raw_stream_dict['can_administer_channel_group_id']]
    can_send_message_group_value = setting_groups_dict[raw_stream_dict['can_send_message_group_id']]
    can_remove_subscribers_group_value = setting_groups_dict[raw_stream_dict['can_remove_subscribers_group_id']]
    is_announcement_only = raw_stream_dict['stream_post_policy'] == Stream.STREAM_POST_POLICY_ADMINS
    return NeverSubscribedStreamDict(
        is_archived=is_archived,
        can_add_subscribers_group=can_add_subscribers_group_value,
        can_administer_channel_group=can_administer_channel_group_value,
        can_send_message_group=can_send_message_group_value,
        can_remove_subscribers_group=can_remove_subscribers_group_value,
        creator_id=creator_id,
        date_created=date_created,
        description=description,
        first_message_id=first_message_id,
        is_recently_active=is_recently_active,
        history_public_to_subscribers=history_public_to_subscribers,
        invite_only=invite_only,
        is_announcement_only=is_announcement_only,
        is_web_public=is_web_public,
        message_retention_days=message_retention_days,
        name=name,
        rendered_description=rendered_description,
        stream_id=stream_id,
        stream_post_policy=stream_post_policy,
        stream_weekly_traffic=stream_weekly_traffic
    )

def validate_user_access_to_subscribers(user_profile: UserProfile, stream: Stream) -> None:
    user_group_membership_details = UserGroupMembershipDetails(user_recursive_group_ids=None)
    validate_user_access_to_subscribers_helper(
        user_profile,
        {
            'realm_id': stream.realm_id,
            'is_web_public': stream.is_web_public,
            'invite_only': stream.invite_only,
            'can_administer_channel_group_id': stream.can_administer_channel_group_id,
            'can_add_subscribers_group_id': stream.can_add_subscribers_group_id
        },
        lambda user_profile: subscribed_to_stream(user_profile, stream.id),
        user_group_membership_details=user_group_membership_details
    )

def validate_user_access_to_subscribers_helper(
    user_profile: UserProfile,
    stream_dict: Dict[str, Any],
    check_user_subscribed: Callable[[UserProfile], bool],
    user_group_membership_details: UserGroupMembershipDetails
) -> None:
    if user_profile is None:
        raise ValidationError('Missing user to validate access for')
    if user_profile.realm_id != stream_dict['realm_id']:
        raise ValidationError('Requesting user not in given realm')
    if stream_dict['is_web_public']:
        return
    if user_profile.is_guest and check_user_subscribed(user_profile):
        return
    if not user_profile.can_access_public_streams() and (not stream_dict['invite_only']):
        raise JsonableError(_('Subscriber data is not available for this channel'))
    if user_profile.is_realm_admin:
        return
    if user_group_membership_details.user_recursive_group_ids is None:
        user_group_membership_details.user_recursive_group_ids = set(
            get_recursive_membership_groups(user_profile).values_list('id', flat=True)
    if has_metadata_access_to_channel_via_groups(
        user_profile,
        user_group_membership_details.user_recursive_group_ids,
        stream_dict['can_administer_channel_group_id'],
        stream_dict['can_add_subscribers_group_id']
    ):
        return
    if stream_dict['invite_only'] and (not check_user_subscribed(user_profile)):
        raise JsonableError(_('Unable to retrieve subscribers for private channel'))

def bulk_get_subscriber_user_ids(
    stream_dicts: List[RawStreamDict],
    user_profile: UserProfile,
    subscribed_stream_ids: Set[int]
) -> Dict[int, List[int]]:
    target_stream_dicts: List[RawStreamDict] = []
    check_user_subscribed = lambda user_profile: is_subscribed
    user_group_membership_details = UserGroupMembershipDetails(user_recursive_group_ids=None)
    for stream_dict in stream_dicts:
        stream_id = stream_dict['id']
        is_subscribed = stream_id in subscribed_stream_ids
        try:
            validate_user_access_to_subscribers_helper(
                user_profile, stream_dict, check_user_subscribed, user_group_membership_details
            )
        except JsonableError:
            continue
        target_stream_dicts.append(stream_dict)
    recip_to_stream_id = {stream['recipient_id']: stream['id'] for stream in target_stream_dicts}
    recipient_ids = sorted((stream['recipient_id'] for stream in target_stream_dicts))
    result = {stream['id']: [] for stream in stream_dicts}
    if not recipient_ids:
        return result
    query = SQL('\n        SELECT\n            zerver_subscription.recipient_id,\n            zerver_subscription.user_profile_id\n        FROM\n            zerver_subscription\n        WHERE\n            zerver_subscription.recipient_id in %(recipient_ids)s AND\n            zerver_subscription.active AND\n            zerver_subscription.is_user_active\n        ORDER BY\n            zerver_subscription.recipient_id,\n            zerver_subscription.user_profile_id\n        ')
    cursor = connection.cursor()
    cursor.execute(query, {'recipient_ids': tuple(recipient_ids)})
    rows = cursor.fetchall()
    cursor.close()
    for recip