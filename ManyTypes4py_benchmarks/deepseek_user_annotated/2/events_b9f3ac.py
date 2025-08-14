# See https://zulip.readthedocs.io/en/latest/subsystems/events-system.html for
# high-level documentation on how this system works.
import copy
import logging
import time
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
from zerver.lib.integrations import (
    EMBEDDED_BOTS,
    WEBHOOK_INTEGRATIONS,
    get_all_event_types_for_integration,
)
from zerver.lib.message import (
    add_message_to_unread_msgs,
    aggregate_unread_data,
    apply_unread_message_event,
    extract_unread_data_from_um_rows,
    get_raw_unread_data,
    get_recent_conversations_recipient_id,
    get_recent_private_conversations,
    get_starred_message_ids,
    remove_message_id_from_unread_mgs,
)
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
from zerver.lib.subscription_info import (
    build_unsubscribed_sub_from_stream_dict,
    gather_subscriptions_helper,
    get_web_public_subs,
)
from zerver.lib.thumbnail import THUMBNAIL_OUTPUT_FORMATS
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.timezone import canonicalize_timezone
from zerver.lib.topic import TOPIC_NAME, maybe_rename_general_chat_to_empty_topic
from zerver.lib.user_groups import (
    get_group_setting_value_for_api,
    get_recursive_membership_groups,
    get_server_supported_permission_settings,
    user_groups_in_realm_serialized,
)
from zerver.lib.user_status import get_all_users_status_dict
from zerver.lib.user_topics import get_topic_mutes, get_user_topics
from zerver.lib.users import (
    get_cross_realm_dicts,
    get_data_for_inaccessible_user,
    get_users_for_api,
    is_administrator_role,
    max_message_id_for_user,
)
from zerver.lib.utils import optional_bytes_to_mib
from zerver.models import (
    Client,
    CustomProfileField,
    Draft,
    Message,
    NamedUserGroup,
    Realm,
    RealmUserDefault,
    Recipient,
    Stream,
    Subscription,
    UserProfile,
    UserStatus,
    UserTopic,
)
from zerver.models.constants import MAX_TOPIC_NAME_LENGTH
from zerver.models.custom_profile_fields import custom_profile_fields_for_realm
from zerver.models.linkifiers import linkifiers_for_realm
from zerver.models.realm_emoji import get_all_custom_emoji_for_realm
from zerver.models.realm_playgrounds import get_realm_playgrounds
from zerver.models.realms import (
    get_corresponding_policy_value_for_group_setting,
    get_realm_domains,
    get_realm_with_settings,
)
from zerver.models.streams import get_default_stream_groups
from zerver.tornado.django_api import get_user_events, request_event_queue
from zproject.backends import email_auth_enabled, password_auth_enabled


def add_realm_logo_fields(state: Dict[str, Any], realm: Realm) -> None:
    state["realm_logo_url"] = get_realm_logo_url(realm, night=False)
    state["realm_logo_source"] = get_realm_logo_source(realm, night=False)
    state["realm_night_logo_url"] = get_realm_logo_url(realm, night=True)
    state["realm_night_logo_source"] = get_realm_logo_source(realm, night=True)
    state["max_logo_file_size_mib"] = settings.MAX_LOGO_FILE_SIZE_MIB


def always_want(msg_type: str) -> bool:
    """
    This function is used as a helper in
    fetch_initial_state_data, when the user passes
    in None for event_types, and we want to fetch
    info for every event type.  Defining this at module
    level makes it easier to mock.
    """
    return True


def fetch_initial_state_data(
    user_profile: Optional[UserProfile],
    *,
    realm: Realm,
    event_types: Optional[Iterable[str]] = None,
    queue_id: Optional[str] = "",
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
) -> Dict[str, Any]:
    """When `event_types` is None, fetches the core data powering the
    web app's `page_params` and `/api/v1/register` (for mobile/terminal
    apps).  Can also fetch a subset as determined by `event_types`.

    The user_profile=None code path is used for logged-out public
    access to streams with is_web_public=True.

    Whenever you add new code to this function, you should also add
    corresponding events for changes in the data structures and new
    code to apply_events (and add a test in test_events.py).
    """
    state: Dict[str, Any] = {"queue_id": queue_id}

    if event_types is None:
        # return True always
        want: Callable[[str], bool] = always_want
    else:
        want = set(event_types).__contains__

    # Show the version info unconditionally.
    state["zulip_version"] = ZULIP_VERSION
    state["zulip_feature_level"] = API_FEATURE_LEVEL
    state["zulip_merge_base"] = ZULIP_MERGE_BASE

    if want("alert_words"):
        state["alert_words"] = [] if user_profile is None else user_alert_words(user_profile)

    if want("custom_profile_fields"):
        if user_profile is None:
            # Spectators can't access full user profiles or
            # personal settings, so we send an empty list.
            state["custom_profile_fields"] = []
        else:
            fields = custom_profile_fields_for_realm(realm.id)
            state["custom_profile_fields"] = [f.as_dict() for f in fields]
        state["custom_profile_field_types"] = {
            item[4]: {"id": item[0], "name": str(item[1])}
            for item in CustomProfileField.ALL_FIELD_TYPES
        }

        if not pronouns_field_type_supported:
            for field in state["custom_profile_fields"]:
                if field["type"] == CustomProfileField.PRONOUNS:
                    field["type"] = CustomProfileField.SHORT_TEXT

            del state["custom_profile_field_types"]["PRONOUNS"]

    if want("onboarding_steps"):
        # Even if we offered special onboarding steps for guests without an
        # account, we'd maybe need to store their state using cookies
        # or local storage, rather than in the database.
        state["onboarding_steps"] = (
            [] if user_profile is None else get_next_onboarding_steps(user_profile)
        )

    if want("message"):
        # Since the introduction of `anchor="latest"` in the API,
        # `max_message_id` is primarily used for generating `local_id`
        # values that are higher than this.  We likely can eventually
        # remove this parameter from the API.
        state["max_message_id"] = max_message_id_for_user(user_profile)

    if want("saved_snippets"):
        if user_profile is None:
            state["saved_snippets"] = []
        else:
            state["saved_snippets"] = do_get_saved_snippets(user_profile)

    if want("drafts"):
        if user_profile is None:
            state["drafts"] = []
        else:
            # Note: if a user ever disables syncing drafts then all of
            # their old drafts stored on the server will be deleted and
            # simply retained in local storage. In which case user_drafts
            # would just be an empty queryset.
            user_draft_objects = Draft.objects.filter(user_profile=user_profile).order_by(
                "-last_edit_time"
            )[: settings.MAX_DRAFTS_IN_REGISTER_RESPONSE]
            user_draft_dicts = [draft.to_dict() for draft in user_draft_objects]
            state["drafts"] = user_draft_dicts

    if want("scheduled_messages"):
        state["scheduled_messages"] = (
            [] if user_profile is None else get_undelivered_scheduled_messages(user_profile)
        )

    if want("muted_topics") and (
        # Suppress muted_topics data for clients that explicitly
        # support user_topic. This allows clients to request both the
        # user_topic and muted_topics, and receive the duplicate
        # muted_topics data only from older servers that don't yet
        # support user_topic.
        event_types is None or not want("user_topic")
    ):
        state["muted_topics"] = [] if user_profile is None else get_topic_mutes(user_profile)

    if want("muted_users"):
        state["muted_users"] = [] if user_profile is None else get_user_mutes(user_profile)

    if want("presence"):
        if presence_last_update_id_fetched_by_client is not None:
            # This param being submitted by the client, means they want to use
            # the modern API.
            slim_presence = True

        if user_profile is not None:
            presences, presence_last_update_id_fetched_by_server = get_presences_for_realm(
                realm,
                slim_presence,
                last_update_id_fetched_by_client=presence_last_update_id_fetched_by_client,
                history_limit_days=presence_history_limit_days,
                requesting_user_profile=user_profile,
            )
            state["presences"] = presences
            state["presence_last_update_id"] = presence_last_update_id_fetched_by_server
        else:
            state["presences"] = {}

        # Send server_timestamp, to match the format of `GET /presence` requests.
        state["server_timestamp"] = time.time()

    if want("realm"):
        # The realm bundle includes both realm properties and server
        # properties, since it's rare that one would want one and not
        # the other. We expect most clients to want it.
        #
        # A note on naming: For some settings, one could imagine
        # having a server-level value and a realm-level value (with
        # the server value serving as the default for the realm
        # value). For such settings, we prefer the following naming
        # scheme:
        #
        # * realm_inline_image_preview (current realm setting)
        # * server_inline_image_preview (server-level default)
        #
        # In situations where for backwards-compatibility reasons we
        # have an unadorned name, we should arrange that clients using
        # that unadorned name work correctly (i.e. that should be the
        # currently active setting, not a server-level default).
        #
        # Other settings, which are just server-level settings or data
        # about the version of Zulip, can be named without prefixes,
        # e.g. giphy_rating_options or development_environment.
        for property_name in Realm.property_types:
            state["realm_" + property_name] = getattr(realm, property_name)

        for setting_name in Realm.REALM_PERMISSION_GROUP_SETTINGS:
            setting_value = getattr(realm, setting_name)
            state["realm_" + setting_name] = get_group_setting_value_for_api(setting_value)

        state["realm_create_public_stream_policy"] = (
            get_corresponding_policy_value_for_group_setting(
                realm, "can_create_public_channel_group", Realm.COMMON_POLICY_TYPES
            )
        )
        state["realm_create_private_stream_policy"] = (
            get_corresponding_policy_value_for_group_setting(
                realm, "can_create_private_channel_group", Realm.COMMON_POLICY_TYPES
            )
        )
        state["realm_create_web_public_stream_policy"] = (
            get_corresponding_policy_value_for_group_setting(
                realm,
                "can_create_web_public_channel_group",
                Realm.CREATE_WEB_PUBLIC_STREAM_POLICY_TYPES,
            )
        )
        state["realm_wildcard_mention_policy"] = get_corresponding_policy_value_for_group_setting(
            realm,
            "can_mention_many_users_group",
            Realm.WILDCARD_MENTION_POLICY_TYPES,
        )

        # Most state is handled via the property_types framework;
        # these manual entries are for those realm settings that don't
        # fit into that framework.
        realm_authentication_methods_dict = realm.authentication_methods_dict()
        state["realm_authentication_methods"] = (
            get_realm_authentication_methods_for_page_params_api(
                realm, realm_authentication_methods_dict
            )
        )

        # We pretend these features are disabled because anonymous
        # users can't access them.  In the future, we may want to move
        # this logic to the frontends, so that we can correctly
        # display what these fields are in the settings.
        state["realm_allow_message_editing"] = (
            False if user_profile is None else realm.allow_message_editing
        )

        # This setting determines whether to send presence and also
        # whether to display of users list in the right sidebar; we
        # want both behaviors for logged-out users.  We may in the
        # future choose to move this logic to the frontend.
        state["realm_presence_disabled"] = True if user_profile is None else realm.presence_disabled

        # Important: Encode units in the client-facing API name.
        state["max_avatar_file_size_mib"] = settings.MAX_AVATAR_FILE_SIZE_MIB
        state["max_file_upload_size_mib"] = realm.get_max_file_upload_size_mebibytes()
        state["max_icon_file_size_mib"] = settings.MAX_ICON_FILE_SIZE_MIB
        upload_quota_bytes = realm.upload_quota_bytes()
        state["realm_upload_quota_mib"] = optional_bytes_to_mib(upload_quota_bytes)

        state["realm_icon_url"] = realm_icon_url(realm)
        state["realm_icon_source"] = realm.icon_source
        add_realm_logo_fields(state, realm)

        # TODO/compatibility: realm_uri is a deprecated alias for realm_url that
        # can be removed once there are no longer clients relying on it.
        state["realm_url"] = state["realm_uri"] = realm.url
        state["realm_bot_domain"] = realm.get_bot_domain()
        state["realm_available_video_chat_providers"] = realm.get_enabled_video_chat_providers()
        state["settings_send_digest_emails"] = settings.SEND_DIGEST_EMAILS

        state["realm_digest_emails_enabled"] = (
            realm.digest_emails_enabled and settings.SEND_DIGEST_EMAILS
        )
        state["realm_email_auth_enabled"] = email_auth_enabled(
            realm, realm_authentication_methods_dict
        )
        state["realm_password_auth_enabled"] = password_auth_enabled(
            realm, realm_authentication_methods_dict
        )

        state["server_generation"] = settings.SERVER_GENERATION
        state["realm_is_zephyr_mirror_realm"] = realm.is_zephyr_mirror_realm
        state["development_environment"] = settings.DEVELOPMENT
        state["realm_org_type"] = realm.org_type
        state["realm_plan_type"] = realm.plan_type
        state["zulip_plan_is_not_limited"] = realm.plan_type != Realm.PLAN_TYPE_LIMITED
        state["upgrade_text_for_wide_organization_logo"] = str