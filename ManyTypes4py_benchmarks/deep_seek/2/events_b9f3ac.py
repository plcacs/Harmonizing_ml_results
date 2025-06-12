import copy
import logging
import time
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
from django.conf import settings
from django.utils.translation import gettext as _
from typing_extensions import NotRequired, TypedDict
from version import API_FEATURE_LEVEL, ZULIP_MERGE_BASE, ZULIP_VERSION
from zerver.actions.default_streams import default_stream_groups_to_dicts_sorted
from zerver.actions.realm_settings import get_realm_authentication_methods_for_page_params_api
from zerver.actions.saved_snippets import do_get_saved_snippets
from zerver.actions.users import get_owned_bot_dicts
from zerver.lib import emoji
from zerver.lib.alert_words import user_alert_words
from zerver.lib.avatar import avatar_url
from zerver.lib.bot_config import load_bot_config_template
from zerver.lib.compatibility import is_outdated_server
from zerver.lib.default_streams import get_default_stream_ids_for_realm
from zerver.lib.exceptions import JsonableError
from zerver.lib.external_accounts import get_default_external_accounts
from zerver.lib.integrations import EMBEDDED_BOTS, WEBHOOK_INTEGRATIONS, get_all_event_types_for_integration
from zerver.lib.message import add_message_to_unread_msgs, aggregate_unread_data, apply_unread_message_event, extract_unread_data_from_um_rows, get_raw_unread_data, get_recent_conversations_recipient_id, get_recent_private_conversations, get_starred_message_ids, remove_message_id_from_unread_mgs
from zerver.lib.muted_users import get_user_mutes
from zerver.lib.narrow_helpers import NarrowTerm, read_stop_words
from zerver.lib.narrow_predicate import check_narrow_for_events
from zerver.lib.onboarding_steps import get_next_onboarding_steps
from zerver.lib.presence import get_presence_for_user, get_presences_for_realm
from zerver.lib.realm_icon import realm_icon_url
from zerver.lib.realm_logo import get_realm_logo_source, get_realm_logo_url
from zerver.lib.scheduled_messages import get_undelivered_scheduled_messages
from zerver.lib.soft_deactivation import reactivate_user_if_soft_deactivated
from zerver.lib.sounds import get_available_notification_sounds
from zerver.lib.stream_subscription import handle_stream_notifications_compatibility
from zerver.lib.streams import do_get_streams, get_web_public_streams
from zerver.lib.subscription_info import build_unsubscribed_sub_from_stream_dict, gather_subscriptions_helper, get_web_public_subs
from zerver.lib.thumbnail import THUMBNAIL_OUTPUT_FORMATS
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.timezone import canonicalize_timezone
from zerver.lib.topic import TOPIC_NAME, maybe_rename_general_chat_to_empty_topic
from zerver.lib.user_groups import get_group_setting_value_for_api, get_recursive_membership_groups, get_server_supported_permission_settings, user_groups_in_realm_serialized
from zerver.lib.user_status import get_all_users_status_dict
from zerver.lib.user_topics import get_topic_mutes, get_user_topics
from zerver.lib.users import get_cross_realm_dicts, get_data_for_inaccessible_user, get_users_for_api, is_administrator_role, max_message_id_for_user
from zerver.lib.utils import optional_bytes_to_mib
from zerver.models import Client, CustomProfileField, Draft, Message, NamedUserGroup, Realm, RealmUserDefault, Recipient, Stream, Subscription, UserProfile, UserStatus, UserTopic
from zerver.models.constants import MAX_TOPIC_NAME_LENGTH
from zerver.models.custom_profile_fields import custom_profile_fields_for_realm
from zerver.models.linkifiers import linkifiers_for_realm
from zerver.models.realm_emoji import get_all_custom_emoji_for_realm
from zerver.models.realm_playgrounds import get_realm_playgrounds
from zerver.models.realms import get_corresponding_policy_value_for_group_setting, get_realm_domains, get_realm_with_settings
from zerver.models.streams import get_default_stream_groups
from zerver.tornado.django_api import get_user_events, request_event_queue
from zproject.backends import email_auth_enabled, password_auth_enabled

class StateDict(TypedDict, total=False):
    queue_id: str
    zulip_version: str
    zulip_feature_level: int
    zulip_merge_base: str
    alert_words: List[str]
    custom_profile_fields: List[Dict[str, Any]]
    custom_profile_field_types: Dict[str, Dict[str, Any]]
    onboarding_steps: List[Dict[str, Any]]
    max_message_id: int
    saved_snippets: List[Dict[str, Any]]
    drafts: List[Dict[str, Any]]
    scheduled_messages: List[Dict[str, Any]]
    muted_topics: List[List[Union[str, int]]]
    muted_users: List[List[int]]
    presences: Dict[str, Any]
    presence_last_update_id: int
    server_timestamp: float
    # ... (other fields would be added similarly)

def add_realm_logo_fields(state: StateDict, realm: Realm) -> None:
    state['realm_logo_url'] = get_realm_logo_url(realm, night=False)
    state['realm_logo_source'] = get_realm_logo_source(realm, night=False)
    state['realm_night_logo_url'] = get_realm_logo_url(realm, night=True)
    state['realm_night_logo_source'] = get_realm_logo_source(realm, night=True)
    state['max_logo_file_size_mib'] = settings.MAX_LOGO_FILE_SIZE_MIB

def always_want(msg_type: str) -> bool:
    return True

def fetch_initial_state_data(
    user_profile: Optional[UserProfile],
    *,
    realm: Realm,
    event_types: Optional[Set[str]] = None,
    queue_id: str = '',
    client_gravatar: bool = False,
    user_avatar_url_field_optional: bool = False,
    user_settings_object: bool = False,
    slim_presence: bool = False,
    presence_last_update_id_fetched_by_client: Optional[int] = None,
    presence_history_limit_days: Optional[int] = None,
    include_subscribers: bool = True,
    include_streams: bool = True,
    spectator_requested_language: Optional[str] = None,
    pronouns_field_type_supported: bool = True,
    linkifier_url_template: bool = False,
    user_list_incomplete: bool = False,
    include_deactivated_groups: bool = False,
    archived_channels: bool = False,
) -> StateDict:
    state: StateDict = {'queue_id': queue_id}
    want: Callable[[str], bool]
    if event_types is None:
        want = always_want
    else:
        want = set(event_types).__contains__
    
    # ... (rest of the function implementation remains the same)
    
    return state

def apply_events(
    user_profile: UserProfile,
    *,
    state: StateDict,
    events: List[Dict[str, Any]],
    fetch_event_types: Optional[Set[str]],
    client_gravatar: bool,
    slim_presence: bool,
    include_subscribers: bool,
    linkifier_url_template: bool,
    user_list_incomplete: bool,
    include_deactivated_groups: bool,
    archived_channels: bool = False,
) -> None:
    for event in events:
        if fetch_event_types is not None and event['type'] not in fetch_event_types:
            continue
        apply_event(
            user_profile,
            state=state,
            event=event,
            client_gravatar=client_gravatar,
            slim_presence=slim_presence,
            include_subscribers=include_subscribers,
            linkifier_url_template=linkifier_url_template,
            user_list_incomplete=user_list_incomplete,
            include_deactivated_groups=include_deactivated_groups,
            archived_channels=archived_channels,
        )

def apply_event(
    user_profile: UserProfile,
    *,
    state: StateDict,
    event: Dict[str, Any],
    client_gravatar: bool,
    slim_presence: bool,
    include_subscribers: bool,
    linkifier_url_template: bool,
    user_list_incomplete: bool,
    include_deactivated_groups: bool,
    archived_channels: bool = False,
) -> None:
    # ... (implementation remains the same)

class ClientCapabilities(TypedDict):
    notification_settings_null: NotRequired[bool]
    bulk_message_deletion: NotRequired[bool]
    user_avatar_url_field_optional: NotRequired[bool]
    stream_typing_notifications: NotRequired[bool]
    user_settings_object: NotRequired[bool]
    linkifier_url_template: NotRequired[bool]
    user_list_incomplete: NotRequired[bool]
    include_deactivated_groups: NotRequired[bool]
    archived_channels: NotRequired[bool]
    empty_topic_name: NotRequired[bool]

DEFAULT_CLIENT_CAPABILITIES: ClientCapabilities = ClientCapabilities(
    notification_settings_null=False
)

def do_events_register(
    user_profile: Optional[UserProfile],
    realm: Realm,
    user_client: Client,
    apply_markdown: bool = True,
    client_gravatar: bool = False,
    slim_presence: bool = False,
    presence_last_update_id_fetched_by_client: Optional[int] = None,
    presence_history_limit_days: Optional[int] = None,
    event_types: Optional[Iterable[str]] = None,
    queue_lifespan_secs: int = 0,
    all_public_streams: bool = False,
    include_subscribers: bool = True,
    include_streams: bool = True,
    client_capabilities: ClientCapabilities = DEFAULT_CLIENT_CAPABILITIES,
    narrow: List[NarrowTerm] = [],
    fetch_event_types: Optional[Iterable[str]] = None,
    spectator_requested_language: Optional[str] = None,
    pronouns_field_type_supported: bool = True,
) -> StateDict:
    # ... (implementation remains the same)

def post_process_state(
    user_profile: Optional[UserProfile],
    ret: StateDict,
    notification_settings_null: bool,
    allow_empty_topic_name: bool,
) -> None:
    # ... (implementation remains the same)
