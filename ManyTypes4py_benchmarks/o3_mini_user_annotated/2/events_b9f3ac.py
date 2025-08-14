#!/usr/bin/env python3
import copy
import logging
import time
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any, Dict, List, Optional, Set

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
    """
    state: Dict[str, Any] = {"queue_id": queue_id}

    if event_types is None:
        want: Callable[[str], bool] = always_want
    else:
        want = set(event_types).__contains__  # type: ignore

    state["zulip_version"] = ZULIP_VERSION
    state["zulip_feature_level"] = API_FEATURE_LEVEL
    state["zulip_merge_base"] = ZULIP_MERGE_BASE

    if want("alert_words"):
        state["alert_words"] = [] if user_profile is None else user_alert_words(user_profile)

    if want("custom_profile_fields"):
        if user_profile is None:
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
        state["onboarding_steps"] = [] if user_profile is None else get_next_onboarding_steps(user_profile)

    if want("message"):
        state["max_message_id"] = max_message_id_for_user(user_profile)

    if want("saved_snippets"):
        state["saved_snippets"] = [] if user_profile is None else do_get_saved_snippets(user_profile)

    if want("drafts"):
        if user_profile is None:
            state["drafts"] = []
        else:
            user_draft_objects = Draft.objects.filter(user_profile=user_profile).order_by("-last_edit_time")[: settings.MAX_DRAFTS_IN_REGISTER_RESPONSE]
            user_draft_dicts = [draft.to_dict() for draft in user_draft_objects]
            state["drafts"] = user_draft_dicts

    if want("scheduled_messages"):
        state["scheduled_messages"] = [] if user_profile is None else get_undelivered_scheduled_messages(user_profile)

    if want("muted_topics") and (event_types is None or not want("user_topic")):
        state["muted_topics"] = [] if user_profile is None else get_topic_mutes(user_profile)

    if want("muted_users"):
        state["muted_users"] = [] if user_profile is None else get_user_mutes(user_profile)

    if want("presence"):
        if presence_last_update_id_fetched_by_client is not None:
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

        state["server_timestamp"] = time.time()

    if want("realm"):
        for property_name in Realm.property_types:
            state["realm_" + property_name] = getattr(realm, property_name)

        for setting_name in Realm.REALM_PERMISSION_GROUP_SETTINGS:
            setting_value = getattr(realm, setting_name)
            state["realm_" + setting_name] = get_group_setting_value_for_api(setting_value)

        state["realm_create_public_stream_policy"] = get_corresponding_policy_value_for_group_setting(
            realm, "can_create_public_channel_group", Realm.COMMON_POLICY_TYPES
        )
        state["realm_create_private_stream_policy"] = get_corresponding_policy_value_for_group_setting(
            realm, "can_create_private_channel_group", Realm.COMMON_POLICY_TYPES
        )
        state["realm_create_web_public_stream_policy"] = get_corresponding_policy_value_for_group_setting(
            realm,
            "can_create_web_public_channel_group",
            Realm.CREATE_WEB_PUBLIC_STREAM_POLICY_TYPES,
        )
        state["realm_wildcard_mention_policy"] = get_corresponding_policy_value_for_group_setting(
            realm,
            "can_mention_many_users_group",
            Realm.WILDCARD_MENTION_POLICY_TYPES,
        )

        realm_authentication_methods_dict = realm.authentication_methods_dict()
        state["realm_authentication_methods"] = get_realm_authentication_methods_for_page_params_api(
            realm, realm_authentication_methods_dict
        )

        state["realm_allow_message_editing"] = False if user_profile is None else realm.allow_message_editing
        state["realm_presence_disabled"] = True if user_profile is None else realm.presence_disabled
        state["max_avatar_file_size_mib"] = settings.MAX_AVATAR_FILE_SIZE_MIB
        state["max_file_upload_size_mib"] = realm.get_max_file_upload_size_mebibytes()
        state["max_icon_file_size_mib"] = settings.MAX_ICON_FILE_SIZE_MIB
        upload_quota_bytes = realm.upload_quota_bytes()
        state["realm_upload_quota_mib"] = optional_bytes_to_mib(upload_quota_bytes)

        state["realm_icon_url"] = realm_icon_url(realm)
        state["realm_icon_source"] = realm.icon_source
        add_realm_logo_fields(state, realm)

        state["realm_url"] = state["realm_uri"] = realm.url
        state["realm_bot_domain"] = realm.get_bot_domain()
        state["realm_available_video_chat_providers"] = realm.get_enabled_video_chat_providers()
        state["settings_send_digest_emails"] = settings.SEND_DIGEST_EMAILS

        state["realm_digest_emails_enabled"] = realm.digest_emails_enabled and settings.SEND_DIGEST_EMAILS
        state["realm_email_auth_enabled"] = email_auth_enabled(realm, realm_authentication_methods_dict)
        state["realm_password_auth_enabled"] = password_auth_enabled(realm, realm_authentication_methods_dict)

        state["server_generation"] = settings.SERVER_GENERATION
        state["realm_is_zephyr_mirror_realm"] = realm.is_zephyr_mirror_realm
        state["development_environment"] = settings.DEVELOPMENT
        state["realm_org_type"] = realm.org_type
        state["realm_plan_type"] = realm.plan_type
        state["zulip_plan_is_not_limited"] = realm.plan_type != Realm.PLAN_TYPE_LIMITED
        state["upgrade_text_for_wide_organization_logo"] = str(Realm.UPGRADE_TEXT_STANDARD)

        if realm.push_notifications_enabled_end_timestamp is not None:
            state["realm_push_notifications_enabled_end_timestamp"] = datetime_to_timestamp(
                realm.push_notifications_enabled_end_timestamp
            )
        else:
            state["realm_push_notifications_enabled_end_timestamp"] = None

        state["password_min_length"] = settings.PASSWORD_MIN_LENGTH
        state["password_max_length"] = settings.PASSWORD_MAX_LENGTH
        state["password_min_guesses"] = settings.PASSWORD_MIN_GUESSES
        state["server_inline_image_preview"] = settings.INLINE_IMAGE_PREVIEW
        state["server_inline_url_embed_preview"] = settings.INLINE_URL_EMBED_PREVIEW
        state["server_thumbnail_formats"] = [
            {
                "name": str(thumbnail_format),
                "max_width": thumbnail_format.max_width,
                "max_height": thumbnail_format.max_height,
                "format": thumbnail_format.extension,
                "animated": thumbnail_format.animated,
            }
            for thumbnail_format in THUMBNAIL_OUTPUT_FORMATS
        ]
        state["server_avatar_changes_disabled"] = settings.AVATAR_CHANGES_DISABLED
        state["server_name_changes_disabled"] = settings.NAME_CHANGES_DISABLED
        state["server_web_public_streams_enabled"] = settings.WEB_PUBLIC_STREAMS_ENABLED
        state["giphy_rating_options"] = realm.get_giphy_rating_options()

        state["server_emoji_data_url"] = emoji.data_url()
        state["server_needs_upgrade"] = is_outdated_server(user_profile)
        state["event_queue_longpoll_timeout_seconds"] = settings.EVENT_QUEUE_LONGPOLL_TIMEOUT_SECONDS
        state["realm_default_external_accounts"] = get_default_external_accounts()

        server_default_jitsi_server_url: Optional[str] = (
            settings.JITSI_SERVER_URL.rstrip("/") if settings.JITSI_SERVER_URL is not None else None
        )
        state["server_jitsi_server_url"] = server_default_jitsi_server_url
        state["jitsi_server_url"] = (
            realm.jitsi_server_url if realm.jitsi_server_url is not None else server_default_jitsi_server_url
        )

        state["server_can_summarize_topics"] = settings.TOPIC_SUMMARIZATION_MODEL is not None

        moderation_request_channel = realm.moderation_request_channel
        if moderation_request_channel:
            state["realm_moderation_request_channel_id"] = moderation_request_channel.id
        else:
            state["realm_moderation_request_channel_id"] = -1

        new_stream_announcements_stream = realm.new_stream_announcements_stream
        if new_stream_announcements_stream:
            state["realm_new_stream_announcements_stream_id"] = new_stream_announcements_stream.id
        else:
            state["realm_new_stream_announcements_stream_id"] = -1

        signup_announcements_stream = realm.signup_announcements_stream
        if signup_announcements_stream:
            state["realm_signup_announcements_stream_id"] = signup_announcements_stream.id
        else:
            state["realm_signup_announcements_stream_id"] = -1

        zulip_update_announcements_stream = realm.zulip_update_announcements_stream
        if zulip_update_announcements_stream:
            state["realm_zulip_update_announcements_stream_id"] = zulip_update_announcements_stream.id
        else:
            state["realm_zulip_update_announcements_stream_id"] = -1

        state["max_stream_name_length"] = Stream.MAX_NAME_LENGTH
        state["max_stream_description_length"] = Stream.MAX_DESCRIPTION_LENGTH
        state["max_topic_length"] = MAX_TOPIC_NAME_LENGTH
        state["max_message_length"] = settings.MAX_MESSAGE_LENGTH
        if realm.demo_organization_scheduled_deletion_date is not None:
            state["demo_organization_scheduled_deletion_date"] = datetime_to_timestamp(
                realm.demo_organization_scheduled_deletion_date
            )
        state["realm_date_created"] = datetime_to_timestamp(realm.date_created)

        state["server_presence_ping_interval_seconds"] = settings.PRESENCE_PING_INTERVAL_SECS
        state["server_presence_offline_threshold_seconds"] = settings.OFFLINE_THRESHOLD_SECS
        state["server_typing_started_expiry_period_milliseconds"] = settings.TYPING_STARTED_EXPIRY_PERIOD_MILLISECONDS
        state["server_typing_stopped_wait_period_milliseconds"] = settings.TYPING_STOPPED_WAIT_PERIOD_MILLISECONDS
        state["server_typing_started_wait_period_milliseconds"] = settings.TYPING_STARTED_WAIT_PERIOD_MILLISECONDS
        state["server_supported_permission_settings"] = get_server_supported_permission_settings()
        state["server_min_deactivated_realm_deletion_days"] = settings.MIN_DEACTIVATED_REALM_DELETION_DAYS
        state["server_max_deactivated_realm_deletion_days"] = settings.MAX_DEACTIVATED_REALM_DELETION_DAYS
        state["realm_empty_topic_display_name"] = Message.EMPTY_TOPIC_FALLBACK_NAME

    if want("realm_user_settings_defaults"):
        realm_user_default = RealmUserDefault.objects.get(realm=realm)
        state["realm_user_settings_defaults"] = {}
        for property_name in RealmUserDefault.property_types:
            state["realm_user_settings_defaults"][property_name] = getattr(realm_user_default, property_name)
        state["realm_user_settings_defaults"]["emojiset_choices"] = RealmUserDefault.emojiset_choices()
        state["realm_user_settings_defaults"]["available_notification_sounds"] = get_available_notification_sounds()

    if want("realm_domains"):
        state["realm_domains"] = get_realm_domains(realm)

    if want("realm_emoji"):
        state["realm_emoji"] = get_all_custom_emoji_for_realm(realm.id)

    if want("realm_linkifiers"):
        if linkifier_url_template:
            state["realm_linkifiers"] = linkifiers_for_realm(realm.id)
        else:
            state["realm_linkifiers"] = []

    if want("realm_filters"):
        state["realm_filters"] = []

    if want("realm_playgrounds"):
        state["realm_playgrounds"] = get_realm_playgrounds(realm)

    if want("realm_user_groups"):
        state["realm_user_groups"] = user_groups_in_realm_serialized(
            realm, include_deactivated_groups=include_deactivated_groups
        )

    if user_profile is not None:
        settings_user: UserProfile = user_profile
    else:
        assert spectator_requested_language is not None
        settings_user = UserProfile(
            full_name="Anonymous User",
            email="username@example.com",
            delivery_email="username@example.com",
            realm=realm,
            role=UserProfile.ROLE_GUEST,
            is_billing_admin=False,
            avatar_source=UserProfile.AVATAR_FROM_GRAVATAR,
            id=0,
            default_language=spectator_requested_language,
            web_home_view="recent_topics",
        )
    if want("realm_user"):
        state["raw_users"] = get_users_for_api(
            realm,
            user_profile,
            client_gravatar=client_gravatar,
            user_avatar_url_field_optional=user_avatar_url_field_optional,
            include_custom_profile_fields=user_profile is not None,
            user_list_incomplete=user_list_incomplete,
        )
        state["cross_realm_bots"] = list(get_cross_realm_dicts())
        state["avatar_source"] = settings_user.avatar_source
        state["avatar_url_medium"] = avatar_url(settings_user, medium=True, client_gravatar=False)
        state["avatar_url"] = avatar_url(settings_user, medium=False, client_gravatar=False)
        settings_user_recursive_group_ids: Set[int] = set(get_recursive_membership_groups(settings_user).values_list("id", flat=True))
        state["can_create_private_streams"] = (realm.can_create_private_channel_group_id in settings_user_recursive_group_ids)
        state["can_create_public_streams"] = (realm.can_create_public_channel_group_id in settings_user_recursive_group_ids)
        state["can_create_web_public_streams"] = (realm.can_create_web_public_channel_group_id in settings_user_recursive_group_ids)
        state["can_create_streams"] = (
            state["can_create_private_streams"]
            or state["can_create_public_streams"]
            or state["can_create_web_public_streams"]
        )
        state["can_invite_others_to_realm"] = (realm.can_invite_users_group_id in settings_user_recursive_group_ids)
        state["is_admin"] = settings_user.is_realm_admin
        state["is_owner"] = settings_user.is_realm_owner
        state["is_moderator"] = settings_user.is_moderator
        state["is_guest"] = settings_user.is_guest
        state["is_billing_admin"] = settings_user.is_billing_admin
        state["user_id"] = settings_user.id
        state["email"] = settings_user.email
        state["delivery_email"] = settings_user.delivery_email
        state["full_name"] = settings_user.full_name

    if want("realm_bot"):
        state["realm_bots"] = [] if user_profile is None else get_owned_bot_dicts(user_profile)

    if want("realm_embedded_bots"):
        state["realm_embedded_bots"] = [
            {"name": bot.name, "config": load_bot_config_template(bot.name)}
            for bot in EMBEDDED_BOTS
        ]

    if want("realm_incoming_webhook_bots"):
        state["realm_incoming_webhook_bots"] = [
            {
                "name": integration.name,
                "display_name": integration.display_name,
                "all_event_types": get_all_event_types_for_integration(integration),
                "config_options": [
                    {
                        "key": c.name,
                        "label": c.description,
                        "validator": c.validator.__name__,
                    }
                    for c in integration.config_options
                ] if integration.config_options else [],
            }
            for integration in WEBHOOK_INTEGRATIONS
            if integration.legacy is False
        ]

    if want("recent_private_conversations"):
        state["raw_recent_private_conversations"] = {} if user_profile is None else get_recent_private_conversations(user_profile)

    if want("subscription"):
        if user_profile is not None:
            sub_info = gather_subscriptions_helper(
                user_profile,
                include_subscribers=include_subscribers,
                include_archived_channels=archived_channels,
            )
        else:
            sub_info = get_web_public_subs(realm)
        state["subscriptions"] = sub_info.subscriptions
        state["unsubscribed"] = sub_info.unsubscribed
        state["never_subscribed"] = sub_info.never_subscribed

    if want("update_message_flags") and want("message"):
        if user_profile is not None:
            state["raw_unread_msgs"] = get_raw_unread_data(user_profile)
        else:
            state["raw_unread_msgs"] = extract_unread_data_from_um_rows([], user_profile)

    if want("starred_messages"):
        state["starred_messages"] = [] if user_profile is None else get_starred_message_ids(user_profile)

    if want("stream") and include_streams:
        if user_profile is not None:
            state["streams"] = do_get_streams(
                user_profile,
                include_web_public=True,
                include_all_active=user_profile.is_realm_admin,
            )
        else:
            state["streams"] = get_web_public_streams(realm)
    if want("default_streams"):
        if settings_user.is_guest:
            state["realm_default_streams"] = []
        else:
            state["realm_default_streams"] = list(get_default_stream_ids_for_realm(realm.id))
    if want("default_stream_groups"):
        if settings_user.is_guest:
            state["realm_default_stream_groups"] = []
        else:
            state["realm_default_stream_groups"] = default_stream_groups_to_dicts_sorted(get_default_stream_groups(realm))
    if want("stop_words"):
        state["stop_words"] = read_stop_words()
    if want("update_display_settings") and not user_settings_object:
        for prop in UserProfile.display_settings_legacy:
            state[prop] = getattr(settings_user, prop)
        state["emojiset_choices"] = UserProfile.emojiset_choices()
        state["timezone"] = canonicalize_timezone(settings_user.timezone)
    if want("update_global_notifications") and not user_settings_object:
        for notification in UserProfile.notification_settings_legacy:
            state[notification] = getattr(settings_user, notification)
        state["available_notification_sounds"] = get_available_notification_sounds()
    if want("user_settings"):
        state["user_settings"] = {}
        for prop in UserProfile.property_types:
            state["user_settings"][prop] = getattr(settings_user, prop)
        state["user_settings"]["emojiset_choices"] = UserProfile.emojiset_choices()
        state["user_settings"]["timezone"] = canonicalize_timezone(settings_user.timezone)
        state["user_settings"]["available_notification_sounds"] = get_available_notification_sounds()
    if want("user_status"):
        state["user_status"] = {} if user_profile is None else get_all_users_status_dict(realm=realm, user_profile=user_profile)
    if want("user_topic"):
        state["user_topics"] = [] if user_profile is None else get_user_topics(user_profile)
    if want("video_calls"):
        state["has_zoom_token"] = settings_user.zoom_token is not None
    if want("giphy"):
        state["giphy_api_key"] = settings.GIPHY_API_KEY if settings.GIPHY_API_KEY else ""
    if user_profile is None:
        assert state["is_admin"] is False
        assert state["is_owner"] is False
        assert state["is_guest"] is True

    return state


def apply_events(
    user_profile: UserProfile,
    *,
    state: Dict[str, Any],
    events: Iterable[Dict[str, Any]],
    fetch_event_types: Optional[Collection[str]],
    client_gravatar: bool,
    slim_presence: bool,
    include_subscribers: bool,
    linkifier_url_template: bool,
    user_list_incomplete: bool,
    include_deactivated_groups: bool,
    archived_channels: bool = False,
) -> None:
    for event in events:
        if fetch_event_types is not None and event["type"] not in fetch_event_types:
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
    state: Dict[str, Any],
    event: Dict[str, Any],
    client_gravatar: bool,
    slim_presence: bool,
    include_subscribers: bool,
    linkifier_url_template: bool,
    user_list_incomplete: bool,
    include_deactivated_groups: bool,
    archived_channels: bool = False,
) -> None:
    if event["type"] == "message":
        state["max_message_id"] = max(state["max_message_id"], event["message"]["id"])
        if "raw_unread_msgs" in state and "read" not in event["flags"]:
            apply_unread_message_event(
                user_profile,
                state["raw_unread_msgs"],
                event["message"],
                event["flags"],
            )
        if event["message"]["type"] != "stream":
            if "raw_recent_private_conversations" in state:
                conversations: Dict[Any, Any] = state["raw_recent_private_conversations"]
                recipient_id = get_recent_conversations_recipient_id(
                    user_profile, event["message"]["recipient_id"], event["message"]["sender_id"]
                )
                if recipient_id not in conversations:
                    conversations[recipient_id] = dict(
                        user_ids=sorted(
                            user_dict["id"]
                            for user_dict in event["message"]["display_recipient"]
                            if user_dict["id"] != user_profile.id
                        ),
                    )
                conversations[recipient_id]["max_message_id"] = event["message"]["id"]
            return
        for sub_dict in state.get("subscriptions", []):
            if (event["message"]["stream_id"] == sub_dict["stream_id"] and sub_dict["first_message_id"] is None):
                sub_dict["first_message_id"] = event["message"]["id"]
        for stream_dict in state.get("streams", []):
            if (event["message"]["stream_id"] == stream_dict["stream_id"] and stream_dict["first_message_id"] is None):
                stream_dict["first_message_id"] = event["message"]["id"]

    elif event["type"] == "heartbeat":
        pass

    elif event["type"] == "saved_snippets":
        if event["op"] == "add":
            state["saved_snippets"].append(event["saved_snippet"])
        elif event["op"] == "remove":
            for idx, saved_snippet in enumerate(state["saved_snippets"]):
                if saved_snippet["id"] == event["saved_snippet_id"]:
                    del state["saved_snippets"][idx]
                    break

    elif event["type"] == "drafts":
        if event["op"] == "add":
            state["drafts"].extend(event["drafts"])
        else:
            if event["op"] == "update":
                event_draft_idx = event["draft"]["id"]
                def _draft_update_action(i: int) -> None:
                    state["drafts"][i] = event["draft"]
            elif event["op"] == "remove":
                event_draft_idx = event["draft_id"]
                def _draft_update_action(i: int) -> None:
                    del state["drafts"][i]
            state_draft_idx: Optional[int] = None
            for idx, draft in enumerate(state["drafts"]):
                if draft["id"] == event_draft_idx:
                    state_draft_idx = idx
                    break
            assert state_draft_idx is not None
            _draft_update_action(state_draft_idx)

    elif event["type"] == "scheduled_messages":
        if event["op"] == "add":
            assert len(event["scheduled_messages"]) == 1
            state["scheduled_messages"].append(event["scheduled_messages"][0])
            state["scheduled_messages"].sort(key=lambda scheduled_message: scheduled_message["scheduled_delivery_timestamp"])
        if event["op"] == "update":
            for idx, scheduled_message in enumerate(state["scheduled_messages"]):
                if scheduled_message["scheduled_message_id"] == event["scheduled_message"]["scheduled_message_id"]:
                    state["scheduled_messages"][idx] = event["scheduled_message"]
                    if (scheduled_message["scheduled_delivery_timestamp"]
                        != event["scheduled_message"]["scheduled_delivery_timestamp"]):
                        state["scheduled_messages"].sort(key=lambda scheduled_message: scheduled_message["scheduled_delivery_timestamp"])
                    break
        if event["op"] == "remove":
            for idx, scheduled_message in enumerate(state["scheduled_messages"]):
                if scheduled_message["scheduled_message_id"] == event["scheduled_message_id"]:
                    del state["scheduled_messages"][idx]

    elif event["type"] == "onboarding_steps":
        state["onboarding_steps"] = event["onboarding_steps"]

    elif event["type"] == "custom_profile_fields":
        state["custom_profile_fields"] = event["fields"]
        custom_profile_field_ids: Set[int] = {field["id"] for field in state["custom_profile_fields"]}
        if "raw_users" in state:
            for user_dict in state["raw_users"].values():
                if "profile_data" not in user_dict:
                    continue
                profile_data = user_dict["profile_data"]
                for field_id in list(profile_data.keys()):
                    if int(field_id) not in custom_profile_field_ids:
                        del profile_data[field_id]

    elif event["type"] == "realm_user":
        person = event["person"]
        person_user_id = person["user_id"]
        if event["op"] == "add":
            person = copy.deepcopy(person)
            if client_gravatar:
                email_address_visibility = UserProfile.objects.get(id=person_user_id).email_address_visibility
                if email_address_visibility != UserProfile.EMAIL_ADDRESS_VISIBILITY_EVERYONE:
                    client_gravatar = False
            if client_gravatar and person["avatar_url"].startswith("https://secure.gravatar.com/"):
                person["avatar_url"] = None
            person["is_active"] = True
            if not person["is_bot"]:
                person["profile_data"] = {}
            state["raw_users"][person_user_id] = person
        elif event["op"] == "update":
            is_me = person_user_id == user_profile.id
            if is_me:
                if "avatar_url" in person and "avatar_url" in state:
                    state["avatar_source"] = person["avatar_source"]
                    state["avatar_url"] = person["avatar_url"]
                    state["avatar_url_medium"] = person["avatar_url_medium"]
                if "role" in person:
                    state["is_admin"] = is_administrator_role(person["role"])
                    state["is_owner"] = person["role"] == UserProfile.ROLE_REALM_OWNER
                    state["is_moderator"] = person["role"] == UserProfile.ROLE_MODERATOR
                    state["is_guest"] = person["role"] == UserProfile.ROLE_GUEST
                    state["can_create_private_streams"] = user_profile.can_create_private_streams()
                    state["can_create_public_streams"] = user_profile.can_create_public_streams()
                    state["can_create_web_public_streams"] = user_profile.can_create_web_public_streams()
                    state["can_create_streams"] = (
                        state["can_create_private_streams"]
                        or state["can_create_public_streams"]
                        or state["can_create_web_public_streams"]
                    )
                    state["can_invite_others_to_realm"] = user_profile.can_invite_users_by_email()
                    if state["is_guest"]:
                        state["realm_default_streams"] = []
                    else:
                        state["realm_default_streams"] = list(get_default_stream_ids_for_realm(user_profile.realm_id))
                for field in ["delivery_email", "email", "full_name", "is_billing_admin"]:
                    if field in person and field in state:
                        state[field] = person[field]
                if "new_email" in person:
                    state["email"] = person["new_email"]
                if "role" in person and "realm_bots" in state:
                    prev_state = state["raw_users"][user_profile.id]
                    was_admin = prev_state["is_admin"]
                    now_admin = is_administrator_role(person["role"])
                    if was_admin and not now_admin:
                        state["realm_bots"] = []
                    if not was_admin and now_admin:
                        state["realm_bots"] = get_owned_bot_dicts(user_profile)
            if person_user_id in state["raw_users"]:
                p = state["raw_users"][person_user_id]
                if "avatar_url" in person:
                    if client_gravatar:
                        email_address_visibility = UserProfile.objects.get(id=person_user_id).email_address_visibility
                        if email_address_visibility != UserProfile.EMAIL_ADDRESS_VISIBILITY_EVERYONE:
                            client_gravatar = False
                    if client_gravatar and person["avatar_url"].startswith("https://secure.gravatar.com/"):
                        person["avatar_url"] = None
                        person["avatar_url_medium"] = None
                for field in p:
                    if field in person:
                        p[field] = person[field]
                if "role" in person:
                    p["is_admin"] = is_administrator_role(person["role"])
                    p["is_owner"] = person["role"] == UserProfile.ROLE_REALM_OWNER
                    p["is_guest"] = person["role"] == UserProfile.ROLE_GUEST
                if "is_billing_admin" in person:
                    p["is_billing_admin"] = person["is_billing_admin"]
                if "custom_profile_field" in person:
                    custom_field_id = str(person["custom_profile_field"]["id"])
                    custom_field_new_value = person["custom_profile_field"]["value"]
                    if custom_field_new_value is None and "profile_data" in p:
                        p["profile_data"].pop(custom_field_id, None)
                    elif "rendered_value" in person["custom_profile_field"]:
                        p["profile_data"][custom_field_id] = {
                            "value": custom_field_new_value,
                            "rendered_value": person["custom_profile_field"]["rendered_value"],
                        }
                    else:
                        p["profile_data"][custom_field_id] = {"value": custom_field_new_value}
                if "new_email" in person:
                    p["email"] = person["new_email"]
                if "is_active" in person and not person["is_active"]:
                    if include_subscribers:
                        for sub_dict in [state["subscriptions"], state["unsubscribed"], state["never_subscribed"]]:
                            sub_dict[:] = [user_id for user_id in sub_dict["subscribers"] if user_id != person_user_id]  # type: ignore
                    for user_group in state["realm_user_groups"]:
                        user_group["members"] = [user_id for user_id in user_group["members"] if user_id != person_user_id]
                    for setting_name in Realm.REALM_PERMISSION_GROUP_SETTINGS:
                        if not isinstance(state["realm_" + setting_name], int):
                            state["realm_" + setting_name].direct_members = [
                                user_id for user_id in state["realm_" + setting_name].direct_members if user_id != person_user_id
                            ]
                    for group in state["realm_user_groups"]:
                        for setting_name in NamedUserGroup.GROUP_PERMISSION_SETTINGS:
                            if not isinstance(group[setting_name], int):
                                group[setting_name].direct_members = [
                                    user_id for user_id in group[setting_name].direct_members if user_id != person_user_id
                                ]
        elif event["op"] == "remove":
            if person_user_id in state["raw_users"]:
                if user_list_incomplete:
                    del state["raw_users"][person_user_id]
                else:
                    inaccessible_user_dict = get_data_for_inaccessible_user(user_profile.realm, person_user_id)
                    state["raw_users"][person_user_id] = inaccessible_user_dict
            if include_subscribers:
                for sub_dict in [state["subscriptions"], state["unsubscribed"], state["never_subscribed"]]:
                    for sub in sub_dict:
                        sub["subscribers"] = [user_id for user_id in sub["subscribers"] if user_id != person_user_id]
        else:
            raise AssertionError("Unexpected event type {type}/{op}".format(**event))
    elif event["type"] == "realm_bot":
        if event["op"] == "add":
            state["realm_bots"].append(event["bot"])
        elif event["op"] == "delete":
            state["realm_bots"] = [item for item in state["realm_bots"] if item["user_id"] != event["bot"]["user_id"]]
        elif event["op"] == "update":
            for bot in state["realm_bots"]:
                if bot["user_id"] == event["bot"]["user_id"]:
                    if "owner_id" in event["bot"]:
                        bot["owner_id"] = event["bot"]["owner_id"]
                    else:
                        bot.update(event["bot"])
        else:
            raise AssertionError("Unexpected event type {type}/{op}".format(**event))
    elif event["type"] == "stream":
        if event["op"] == "create":
            for stream in event["streams"]:
                stream_data = copy.deepcopy(stream)
                if include_subscribers:
                    stream_data["subscribers"] = []
                unsubscribed_stream_sub = Subscription.objects.filter(
                    user_profile=user_profile,
                    recipient__type_id=stream["stream_id"],
                    recipient__type=Recipient.STREAM,
                ).values(*Subscription.API_FIELDS, "recipient_id", "active")
                if len(unsubscribed_stream_sub) == 1:
                    unsubscribed_stream_dict = build_unsubscribed_sub_from_stream_dict(
                        user_profile, unsubscribed_stream_sub[0], stream_data
                    )
                    if include_subscribers:
                        unsubscribed_stream_dict["subscribers"] = []
                    state["unsubscribed"].append(unsubscribed_stream_dict)
                else:
                    assert len(unsubscribed_stream_sub) == 0
                    state["never_subscribed"].append(stream_data)
                if "streams" in state:
                    state["streams"].append(stream)
            state["unsubscribed"].sort(key=lambda elt: elt["name"])
            state["never_subscribed"].sort(key=lambda elt: elt["name"])
            if "streams" in state:
                state["streams"].sort(key=lambda elt: elt["name"])
        if event["op"] == "delete":
            deleted_stream_ids: Set[int] = {stream["stream_id"] for stream in event["streams"]}
            if "streams" in state:
                state["streams"] = [s for s in state["streams"] if s["stream_id"] not in deleted_stream_ids]
            if archived_channels:
                for stream in state["subscriptions"]:
                    if stream["stream_id"] in deleted_stream_ids:
                        stream["is_archived"] = True
                for stream in state["unsubscribed"]:
                    if stream["stream_id"] in deleted_stream_ids:
                        stream["is_archived"] = True
                        stream["first_message_id"] = Stream.objects.get(id=stream["stream_id"]).first_message_id
            else:
                state["subscriptions"] = [stream for stream in state["subscriptions"] if stream["stream_id"] not in deleted_stream_ids]
                state["unsubscribed"] = [stream for stream in state["unsubscribed"] if stream["stream_id"] not in deleted_stream_ids]
            state["never_subscribed"] = [stream for stream in state["never_subscribed"] if stream["stream_id"] not in deleted_stream_ids]
        if event["op"] == "update":
            for sub_list in [state["subscriptions"], state["unsubscribed"], state["never_subscribed"]]:
                for obj in sub_list:
                    if obj["name"].lower() == event["name"].lower():
                        obj[event["property"]] = event["value"]
                        if event["property"] == "description":
                            obj["rendered_description"] = event["rendered_description"]
                        if event.get("history_public_to_subscribers") is not None:
                            obj["history_public_to_subscribers"] = event["history_public_to_subscribers"]
                        if event.get("is_web_public") is not None:
                            obj["is_web_public"] = event["is_web_public"]
            if "streams" in state:
                for stream in state["streams"]:
                    if stream["name"].lower() == event["name"].lower():
                        prop = event["property"]
                        if prop in stream:
                            stream[prop] = event["value"]
                            if prop == "description":
                                stream["rendered_description"] = event["rendered_description"]
                            if event.get("history_public_to_subscribers") is not None:
                                stream["history_public_to_subscribers"] = event["history_public_to_subscribers"]
                            if event.get("is_web_public") is not None:
                                stream["is_web_public"] = event["is_web_public"]
    elif event["type"] == "default_streams":
        state["realm_default_streams"] = event["default_streams"]
    elif event["type"] == "default_stream_groups":
        state["realm_default_stream_groups"] = event["default_stream_groups"]
    elif event["type"] == "realm":
        if event["op"] == "update":
            field = "realm_" + event["property"]
            state[field] = event["value"]
            if field == "realm_jitsi_server_url":
                state["jitsi_server_url"] = state["realm_jitsi_server_url"] if state["realm_jitsi_server_url"] is not None else state["server_jitsi_server_url"]
        elif event["op"] == "update_dict":
            for key, value in event["data"].items():
                if key == "max_file_upload_size_mib":
                    state["max_file_upload_size_mib"] = value
                    continue
                state["realm_" + key] = value
                if key == "authentication_methods":
                    state["realm_password_auth_enabled"] = value["Email"]["enabled"] or value["LDAP"]["enabled"]
                    state["realm_email_auth_enabled"] = value["Email"]["enabled"]
                if key in ["can_create_public_channel_group", "can_create_private_channel_group", "can_create_web_public_channel_group"]:
                    if key == "can_create_public_channel_group":
                        state["realm_create_public_stream_policy"] = get_corresponding_policy_value_for_group_setting(
                            user_profile.realm, "can_create_public_channel_group", Realm.COMMON_POLICY_TYPES
                        )
                        state["can_create_public_streams"] = user_profile.has_permission(key)
                    elif key == "can_create_private_channel_group":
                        state["realm_create_private_stream_policy"] = get_corresponding_policy_value_for_group_setting(
                            user_profile.realm, "can_create_private_channel_group", Realm.COMMON_POLICY_TYPES
                        )
                        state["can_create_private_streams"] = user_profile.has_permission(key)
                    else:
                        state["realm_create_web_public_stream_policy"] = get_corresponding_policy_value_for_group_setting(
                            user_profile.realm, "can_create_web_public_channel_group", Realm.CREATE_WEB_PUBLIC_STREAM_POLICY_TYPES
                        )
                        state["can_create_web_public_streams"] = user_profile.has_permission(key)
                    state["can_create_streams"] = (
                        state["can_create_private_streams"] or state["can_create_public_streams"] or state["can_create_web_public_streams"]
                    )
                if key == "can_invite_users_group" and "can_invite_others_to_realm" in state:
                    state["can_invite_others_to_realm"] = user_profile.has_permission("can_invite_users_group")
                if key == "can_mention_many_users_group":
                    state["realm_wildcard_mention_policy"] = get_corresponding_policy_value_for_group_setting(
                        user_profile.realm, "can_mention_many_users_group", Realm.WILDCARD_MENTION_POLICY_TYPES
                    )
                if key == "plan_type":
                    state["zulip_plan_is_not_limited"] = value != Realm.PLAN_TYPE_LIMITED
        elif event["op"] == "deactivated":
            pass
        else:
            raise AssertionError("Unexpected event type {type}/{op}".format(**event))
    elif event["type"] == "realm_user_settings_defaults":
        if event["op"] == "update":
            state["realm_user_settings_defaults"][event["property"]] = event["value"]
        else:
            raise AssertionError("Unexpected event type {type}/{op}".format(**event))
    elif event["type"] == "subscription":
        if event["op"] == "add":
            added_stream_ids: Set[int] = {sub["stream_id"] for sub in event["subscriptions"]}
            was_added = lambda s: s["stream_id"] in added_stream_ids
            existing_stream_ids: Set[int] = {sub["stream_id"] for sub in state["subscriptions"]}
            for sub in event["subscriptions"]:
                if sub["stream_id"] not in existing_stream_ids:
                    if "subscribers" in sub and not include_subscribers:
                        sub = copy.deepcopy(sub)
                        del sub["subscribers"]
                    state["subscriptions"].append(sub)
            state["unsubscribed"] = [s for s in state["unsubscribed"] if not was_added(s)]
            state["never_subscribed"] = [s for s in state["never_subscribed"] if not was_added(s)]
        elif event["op"] == "remove":
            removed_stream_ids: Set[int] = {sub["stream_id"] for sub in event["subscriptions"]}
            was_removed = lambda s: s["stream_id"] in removed_stream_ids
            removed_subs = list(filter(was_removed, state["subscriptions"]))
            if include_subscribers:
                for sub in removed_subs:
                    sub["subscribers"].remove(user_profile.id)
            state["unsubscribed"] += removed_subs
            state["subscriptions"] = [s for s in state["subscriptions"] if not was_removed(s)]
        elif event["op"] == "update":
            for sub in state["subscriptions"]:
                if sub["stream_id"] == event["stream_id"]:
                    sub[event["property"]] = event["value"]
        elif event["op"] == "peer_add":
            if include_subscribers:
                stream_ids = set(event["stream_ids"])
                user_ids = set(event["user_ids"])
                for sub_dict in [state["subscriptions"], state["unsubscribed"], state["never_subscribed"]]:
                    for sub in sub_dict:
                        if sub["stream_id"] in stream_ids:
                            subscribers = set(sub["subscribers"]) | user_ids
                            sub["subscribers"] = sorted(subscribers)
        elif event["op"] == "peer_remove":
            if include_subscribers:
                stream_ids = set(event["stream_ids"])
                user_ids = set(event["user_ids"])
                for sub_dict in [state["subscriptions"], state["unsubscribed"], state["never_subscribed"]]:
                    for sub in sub_dict:
                        if sub["stream_id"] in stream_ids:
                            subscribers = set(sub["subscribers"]) - user_ids
                            sub["subscribers"] = sorted(subscribers)
        else:
            raise AssertionError("Unexpected event type {type}/{op}".format(**event))
    elif event["type"] == "presence":
        if slim_presence:
            user_key = str(event["user_id"])
        else:
            user_key = event["email"]
        state["presences"][user_key] = get_presence_for_user(event["user_id"], slim_presence)[user_key]
    elif event["type"] == "update_message":
        if "raw_unread_msgs" in state and "new_stream_id" in event:
            stream_dict = state["raw_unread_msgs"]["stream_dict"]
            stream_id = event["new_stream_id"]
            for message_id in event["message_ids"]:
                if message_id in stream_dict:
                    stream_dict[message_id]["stream_id"] = stream_id
        if "raw_unread_msgs" in state and TOPIC_NAME in event:
            stream_dict = state["raw_unread_msgs"]["stream_dict"]
            topic_name = event[TOPIC_NAME]
            for message_id in event["message_ids"]:
                if message_id in stream_dict:
                    stream_dict[message_id]["topic"] = topic_name
    elif event["type"] == "delete_message":
        message_ids: List[int]
        if "message_id" in event:
            message_ids = [event["message_id"]]
        else:
            message_ids = event["message_ids"]
        state["max_message_id"] = max_message_id_for_user(user_profile)
        if "raw_unread_msgs" in state:
            for remove_id in message_ids:
                remove_message_id_from_unread_mgs(state["raw_unread_msgs"], remove_id)
        if "raw_recent_private_conversations" not in state or event["message_type"] != "private":
            return
        state["raw_recent_private_conversations"] = get_recent_private_conversations(user_profile)
    elif event["type"] == "reaction":
        pass
    elif event["type"] == "submessage":
        pass
    elif event["type"] == "typing":
        pass
    elif event["type"] == "typing_edit_message":
        pass
    elif event["type"] == "attachment":
        pass
    elif event["type"] == "update_message_flags":
        if "raw_unread_msgs" in state and event["flag"] == "read" and event["op"] == "add":
            for remove_id in event["messages"]:
                remove_message_id_from_unread_mgs(state["raw_unread_msgs"], remove_id)
        if "raw_unread_msgs" in state and event["flag"] == "read" and event["op"] == "remove":
            for message_id_str, message_details in event["message_details"].items():
                add_message_to_unread_msgs(user_profile.id, state["raw_unread_msgs"], int(message_id_str), message_details)
        if event["flag"] == "starred" and "starred_messages" in state:
            if event["op"] == "add":
                state["starred_messages"] += event["messages"]
            if event["op"] == "remove":
                state["starred_messages"] = [message for message in state["starred_messages"] if message not in event["messages"]]
    elif event["type"] == "realm_domains":
        if event["op"] == "add":
            state["realm_domains"].append(event["realm_domain"])
        elif event["op"] == "change":
            for realm_domain in state["realm_domains"]:
                if realm_domain["domain"] == event["realm_domain"]["domain"]:
                    realm_domain["allow_subdomains"] = event["realm_domain"]["allow_subdomains"]
        elif event["op"] == "remove":
            state["realm_domains"] = [realm_domain for realm_domain in state["realm_domains"] if realm_domain["domain"] != event["domain"]]
        else:
            raise AssertionError("Unexpected event type {type}/{op}".format(**event))
    elif event["type"] == "realm_emoji":
        state["realm_emoji"] = event["realm_emoji"]
    elif event["type"] == "realm_export":
        pass
    elif event["type"] == "realm_export_consent":
        pass
    elif event["type"] == "alert_words":
        state["alert_words"] = event["alert_words"]
    elif event["type"] == "muted_topics":
        state["muted_topics"] = event["muted_topics"]
    elif event["type"] == "muted_users":
        state["muted_users"] = event["muted_users"]
    elif event["type"] == "realm_linkifiers":
        if linkifier_url_template:
            state["realm_linkifiers"] = event["realm_linkifiers"]
    elif event["type"] == "realm_playgrounds":
        state["realm_playgrounds"] = event["realm_playgrounds"]
    elif event["type"] == "update_display_settings":
        if event["setting_name"] != "timezone":
            assert event["setting_name"] in UserProfile.display_settings_legacy
        state[event["setting_name"]] = event["setting"]
    elif event["type"] == "update_global_notifications":
        assert event["notification_name"] in UserProfile.notification_settings_legacy
        state[event["notification_name"]] = event["setting"]
    elif event["type"] == "user_settings":
        if event["property"] != "timezone":
            assert event["property"] in UserProfile.property_types
        if event["property"] in {**UserProfile.display_settings_legacy, **UserProfile.notification_settings_legacy}:
            state[event["property"]] = event["value"]
        state["user_settings"][event["property"]] = event["value"]
    elif event["type"] == "invites_changed":
        pass
    elif event["type"] == "user_group":
        if event["op"] == "add":
            state["realm_user_groups"].append(event["group"])
            state["realm_user_groups"].sort(key=lambda group: group["id"])
        elif event["op"] == "update":
            for user_group in state["realm_user_groups"]:
                if user_group["id"] == event["group_id"]:
                    user_group.update(event["data"])
        elif event["op"] == "add_members":
            for user_group in state["realm_user_groups"]:
                if user_group["id"] == event["group_id"]:
                    user_group["members"].extend(event["user_ids"])
                    user_group["members"].sort()
        elif event["op"] == "remove_members":
            for user_group in state["realm_user_groups"]:
                if user_group["id"] == event["group_id"]:
                    members = set(user_group["members"])
                    user_group["members"] = sorted(members - set(event["user_ids"]))
        elif event["op"] == "add_subgroups":
            for user_group in state["realm_user_groups"]:
                if user_group["id"] == event["group_id"]:
                    user_group["direct_subgroup_ids"].extend(event["direct_subgroup_ids"])
                    user_group["direct_subgroup_ids"].sort()
        elif event["op"] == "remove_subgroups":
            for user_group in state["realm_user_groups"]:
                if user_group["id"] == event["group_id"]:
                    subgroups = set(user_group["direct_subgroup_ids"])
                    user_group["direct_subgroup_ids"] = sorted(subgroups - set(event["direct_subgroup_ids"]))
        elif event["op"] == "remove":
            state["realm_user_groups"] = [ug for ug in state["realm_user_groups"] if ug["id"] != event["group_id"]]
        else:
            raise AssertionError("Unexpected event type {type}/{op}".format(**event))
    elif event["type"] == "user_status":
        user_id_str = str(event["user_id"])
        user_status = state["user_status"]
        away = event.get("away")
        status_text = event.get("status_text")
        emoji_name = event.get("emoji_name")
        emoji_code = event.get("emoji_code")
        reaction_type = event.get("reaction_type")
        if user_id_str not in user_status:
            user_status[user_id_str] = {}
        if away is not None:
            if away:
                user_status[user_id_str]["away"] = True
            else:
                user_status[user_id_str].pop("away", None)
        if status_text is not None:
            if status_text == "":
                user_status[user_id_str].pop("status_text", None)
            else:
                user_status[user_id_str]["status_text"] = status_text
            if emoji_name is not None:
                if emoji_name == "":
                    user_status[user_id_str].pop("emoji_name", None)
                else:
                    user_status[user_id_str]["emoji_name"] = emoji_name
                if emoji_code is not None:
                    if emoji_code == "":
                        user_status[user_id_str].pop("emoji_code", None)
                    else:
                        user_status[user_id_str]["emoji_code"] = emoji_code
                if reaction_type is not None:
                    if reaction_type == UserStatus.UNICODE_EMOJI and emoji_name == "":
                        user_status[user_id_str].pop("reaction_type", None)
                    else:
                        user_status[user_id_str]["reaction_type"] = reaction_type
        if not user_status[user_id_str]:
            user_status.pop(user_id_str, None)
        state["user_status"] = user_status
    elif event["type"] == "user_topic":
        if event["visibility_policy"] == UserTopic.VisibilityPolicy.INHERIT:
            user_topics_state = state["user_topics"]
            for i in range(len(user_topics_state)):
                topic_name = maybe_rename_general_chat_to_empty_topic(event["topic_name"])
                if (user_topics_state[i]["stream_id"] == event["stream_id"]
                    and user_topics_state[i]["topic_name"] == topic_name):
                    del user_topics_state[i]
                    break
        else:
            fields = ["stream_id", "topic_name", "visibility_policy", "last_updated"]
            state["user_topics"].append({x: event[x] for x in fields})
    elif event["type"] == "has_zoom_token":
        state["has_zoom_token"] = event["value"]
    elif event["type"] == "web_reload_client":
        logging.warning("Got a web_reload_client event during apply_events")
    elif event["type"] == "restart":
        pass
    else:
        raise AssertionError("Unexpected event type {}".format(event["type"]))


class ClientCapabilities(TypedDict):
    notification_settings_null: bool
    bulk_message_deletion: NotRequired[bool]
    user_avatar_url_field_optional: NotRequired[bool]
    stream_typing_notifications: NotRequired[bool]
    user_settings_object: NotRequired[bool]
    linkifier_url_template: NotRequired[bool]
    user_list_incomplete: NotRequired[bool]
    include_deactivated_groups: NotRequired[bool]
    archived_channels: NotRequired[bool]
    empty_topic_name: NotRequired[bool]


DEFAULT_CLIENT_CAPABILITIES: ClientCapabilities = {"notification_settings_null": False}


def do_events_register(
    user_profile: Optional[UserProfile],
    realm: Realm,
    user_client: Client,
    apply_markdown: bool = True,
    client_gravatar: bool = False,
    slim_presence: bool = False,
    presence_last_update_id_fetched_by_client: Optional[int] = None,
    presence_history_limit_days: Optional[int] = None,
    event_types: Optional[Sequence[str]] = None,
    queue_lifespan_secs: int = 0,
    all_public_streams: bool = False,
    include_subscribers: bool = True,
    include_streams: bool = True,
    client_capabilities: ClientCapabilities = DEFAULT_CLIENT_CAPABILITIES,
    narrow: Collection[NarrowTerm] = [],
    fetch_event_types: Optional[Collection[str]] = None,
    spectator_requested_language: Optional[str] = None,
    pronouns_field_type_supported: bool = True,
) -> Dict[str, Any]:
    check_narrow_for_events(narrow)
    notification_settings_null: bool = client_capabilities.get("notification_settings_null", False)
    bulk_message_deletion: bool = client_capabilities.get("bulk_message_deletion", False)
    user_avatar_url_field_optional: bool = client_capabilities.get("user_avatar_url_field_optional", False)
    stream_typing_notifications: bool = client_capabilities.get("stream_typing_notifications", False)
    user_settings_object: bool = client_capabilities.get("user_settings_object", False)
    linkifier_url_template: bool = client_capabilities.get("linkifier_url_template", False)
    user_list_incomplete: bool = client_capabilities.get("user_list_incomplete", False)
    include_deactivated_groups: bool = client_capabilities.get("include_deactivated_groups", False)
    archived_channels: bool = client_capabilities.get("archived_channels", False)
    empty_topic_name: bool = client_capabilities.get("empty_topic_name", False)
    event_types_set: Optional[Set[str]]
    if fetch_event_types is not None:
        event_types_set = set(fetch_event_types)
    elif event_types is not None:
        event_types_set = set(event_types)
    else:
        event_types_set = None
    realm = get_realm_with_settings(realm_id=realm.id)
    if user_profile is None:
        assert client_gravatar is False
        assert include_subscribers is False
        assert include_streams is False
        ret: Dict[str, Any] = fetch_initial_state_data(
            user_profile,
            realm=realm,
            event_types=event_types_set,
            queue_id=None,
            client_gravatar=client_gravatar,
            linkifier_url_template=linkifier_url_template,
            user_avatar_url_field_optional=user_avatar_url_field_optional,
            user_settings_object=user_settings_object,
            user_list_incomplete=user_list_incomplete,
            archived_channels=archived_channels,
            slim_presence=True,
            presence_last_update_id_fetched_by_client=None,
            presence_history_limit_days=None,
            include_subscribers=include_subscribers,
            include_streams=include_streams,
            spectator_requested_language=spectator_requested_language,
            include_deactivated_groups=include_deactivated_groups,
        )
        post_process_state(user_profile, ret, notification_settings_null=False, allow_empty_topic_name=empty_topic_name)
        return ret
    reactivate_user_if_soft_deactivated(user_profile)
    legacy_narrow: List[List[Any]] = [[nt.operator, nt.operand] for nt in narrow]
    queue_id: Optional[str] = request_event_queue(
        user_profile,
        user_client,
        apply_markdown,
        client_gravatar,
        slim_presence,
        queue_lifespan_secs,
        event_types,
        all_public_streams,
        narrow=legacy_narrow,
        bulk_message_deletion=bulk_message_deletion,
        stream_typing_notifications=stream_typing_notifications,
        user_settings_object=user_settings_object,
        pronouns_field_type_supported=pronouns_field_type_supported,
        linkifier_url_template=linkifier_url_template,
        user_list_incomplete=user_list_incomplete,
        include_deactivated_groups=include_deactivated_groups,
        archived_channels=archived_channels,
        empty_topic_name=empty_topic_name,
    )
    if queue_id is None:
        raise JsonableError(_("Could not allocate event queue"))
    ret = fetch_initial_state_data(
        user_profile,
        realm=realm,
        event_types=event_types_set,
        queue_id=queue_id,
        client_gravatar=client_gravatar,
        user_avatar_url_field_optional=user_avatar_url_field_optional,
        user_settings_object=user_settings_object,
        slim_presence=slim_presence,
        presence_last_update_id_fetched_by_client=presence_last_update_id_fetched_by_client,
        presence_history_limit_days=presence_history_limit_days,
        include_subscribers=include_subscribers,
        include_streams=include_streams,
        pronouns_field_type_supported=pronouns_field_type_supported,
        linkifier_url_template=linkifier_url_template,
        user_list_incomplete=user_list_incomplete,
        include_deactivated_groups=include_deactivated_groups,
        archived_channels=archived_channels,
    )
    events: List[Dict[str, Any]] = get_user_events(user_profile, queue_id, -1)
    apply_events(
        user_profile,
        state=ret,
        events=events,
        fetch_event_types=fetch_event_types,
        client_gravatar=client_gravatar,
        slim_presence=slim_presence,
        include_subscribers=include_subscribers,
        linkifier_url_template=linkifier_url_template,
        user_list_incomplete=user_list_incomplete,
        include_deactivated_groups=include_deactivated_groups,
    )
    post_process_state(user_profile, ret, notification_settings_null, allow_empty_topic_name=empty_topic_name)
    if len(events) > 0:
        ret["last_event_id"] = events[-1]["id"]
    else:
        ret["last_event_id"] = -1
    return ret


def post_process_state(
    user_profile: Optional[UserProfile],
    ret: Dict[str, Any],
    notification_settings_null: bool,
    allow_empty_topic_name: bool,
) -> None:
    if "raw_unread_msgs" in ret:
        ret["unread_msgs"] = aggregate_unread_data(ret["raw_unread_msgs"], allow_empty_topic_name)
        del ret["raw_unread_msgs"]
    if "raw_users" in ret:
        user_dicts: List[Dict[str, Any]] = sorted(ret["raw_users"].values(), key=lambda x: x["user_id"])
        ret["realm_users"] = [d for d in user_dicts if d["is_active"]]
        ret["realm_non_active_users"] = [d for d in user_dicts if not d["is_active"]]
        for d in user_dicts:
            d.pop("is_active")
        del ret["raw_users"]
    if "raw_recent_private_conversations" in ret:
        ret["recent_private_conversations"] = sorted(
            (dict(**value) for (recipient_id, value) in ret["raw_recent_private_conversations"].items()),
            key=lambda x: -x["max_message_id"],
        )
        del ret["raw_recent_private_conversations"]
    if not notification_settings_null and "subscriptions" in ret:
        for stream_dict in ret["subscriptions"] + ret["unsubscribed"]:
            handle_stream_notifications_compatibility(user_profile, stream_dict, notification_settings_null)
    if not allow_empty_topic_name and "user_topics" in ret:
        for user_topic in ret["user_topics"]:
            if user_topic["topic_name"] == "":
                user_topic["topic_name"] = Message.EMPTY_TOPIC_FALLBACK_NAME