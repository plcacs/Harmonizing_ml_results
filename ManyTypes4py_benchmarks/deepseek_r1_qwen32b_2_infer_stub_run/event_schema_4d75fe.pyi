from collections.abc import Callable
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
    get_origin,
    get_args,
)
from pydantic import BaseModel
from zerver.lib.event_types import (
    AllowMessageEditingData,
    AuthenticationData,
    BaseEvent,
    BotServicesEmbedded,
    BotServicesOutgoing,
    EventAlertWords,
    EventAttachmentAdd,
    EventAttachmentRemove,
    EventAttachmentUpdate,
    EventCustomProfileFields,
    EventDefaultStreamGroups,
    EventDefaultStreams,
    EventDeleteMessage,
    EventDirectMessage,
    EventDraftsAdd,
    EventDraftsRemove,
    EventDraftsUpdate,
    EventHasZoomToken,
    EventHeartbeat,
    EventInvitesChanged,
    EventMessage,
    EventMutedTopics,
    EventMutedUsers,
    EventOnboardingSteps,
    EventPresence,
    EventReactionAdd,
    EventReactionRemove,
    EventRealmBotAdd,
    EventRealmBotDelete,
    EventRealmBotUpdate,
    EventRealmDeactivated,
    EventRealmDomainsAdd,
    EventRealmDomainsChange,
    EventRealmDomainsRemove,
    EventRealmEmojiUpdate,
    EventRealmExport,
    EventRealmExportConsent,
    EventRealmLinkifiers,
    EventRealmPlaygrounds,
    EventRealmUpdate,
    EventRealmUpdateDict,
    EventRealmUserAdd,
    EventRealmUserRemove,
    EventRealmUserSettingsDefaultsUpdate,
    EventRealmUserUpdate,
    EventRestart,
    EventSavedSnippetsAdd,
    EventSavedSnippetsRemove,
    EventScheduledMessagesAdd,
    EventScheduledMessagesRemove,
    EventScheduledMessagesUpdate,
    EventStreamCreate,
    EventStreamDelete,
    EventStreamUpdate,
    EventSubmessage,
    EventSubscriptionAdd,
    EventSubscriptionPeerAdd,
    EventSubscriptionPeerRemove,
    EventSubscriptionRemove,
    EventSubscriptionUpdate,
    EventTypingEditChannelMessageStart,
    EventTypingEditChannelMessageStop,
    EventTypingEditDirectMessageStart,
    EventTypingEditDirectMessageStop,
    EventTypingStart,
    EventTypingStop,
    EventUpdateDisplaySettings,
    EventUpdateGlobalNotifications,
    EventUpdateMessage,
    EventUpdateMessageFlagsAdd,
    EventUpdateMessageFlagsRemove,
    EventUserGroupAdd,
    EventUserGroupAddMembers,
    EventUserGroupAddSubgroups,
    EventUserGroupRemove,
    EventUserGroupRemoveMembers,
    EventUserGroupRemoveSubgroups,
    EventUserGroupUpdate,
    EventUserSettingsUpdate,
    EventUserStatus,
    EventUserTopic,
    EventWebReloadClient,
    GroupSettingUpdateData,
    IconData,
    LogoData,
    MessageContentEditLimitSecondsData,
    NightLogoData,
    PersonAvatarFields,
    PersonBotOwnerId,
    PersonCustomProfileField,
    PersonDeliveryEmail,
    PersonEmail,
    PersonFullName,
    PersonIsActive,
    PersonIsBillingAdmin,
    PersonRole,
    PersonTimezone,
    PlanTypeData,
)
from zerver.models import (
    Realm,
    RealmUserDefault,
    Stream,
    UserProfile,
)

def validate_with_model(data: Dict[str, Any], model: type[BaseModel]) -> None:
    ...

def make_checker(base_model: type[BaseModel]) -> Callable[[str, Dict[str, Any]], None]:
    ...

check_alert_words: Callable[[str, Dict[str, Any]], None] = ...
check_attachment_add: Callable[[str, Dict[str, Any]], None] = ...
check_attachment_remove: Callable[[str, Dict[str, Any]], None] = ...
check_attachment_update: Callable[[str, Dict[str, Any]], None] = ...
check_custom_profile_fields: Callable[[str, Dict[str, Any]], None] = ...
check_default_stream_groups: Callable[[str, Dict[str, Any]], None] = ...
check_default_streams: Callable[[str, Dict[str, Any]], None] = ...
check_direct_message: Callable[[str, Dict[str, Any]], None] = ...
check_draft_add: Callable[[str, Dict[str, Any]], None] = ...
check_draft_remove: Callable[[str, Dict[str, Any]], None] = ...
check_draft_update: Callable[[str, Dict[str, Any]], None] = ...
check_heartbeat: Callable[[str, Dict[str, Any]], None] = ...
check_invites_changed: Callable[[str, Dict[str, Any]], None] = ...
check_message: Callable[[str, Dict[str, Any]], None] = ...
check_muted_topics: Callable[[str, Dict[str, Any]], None] = ...
check_muted_users: Callable[[str, Dict[str, Any]], None] = ...
check_onboarding_steps: Callable[[str, Dict[str, Any]], None] = ...
check_reaction_add: Callable[[str, Dict[str, Any]], None] = ...
check_reaction_remove: Callable[[str, Dict[str, Any]], None] = ...
check_realm_bot_delete: Callable[[str, Dict[str, Any]], None] = ...
check_realm_deactivated: Callable[[str, Dict[str, Any]], None] = ...
check_realm_domains_add: Callable[[str, Dict[str, Any]], None] = ...
check_realm_domains_change: Callable[[str, Dict[str, Any]], None] = ...
check_realm_domains_remove: Callable[[str, Dict[str, Any]], None] = ...
check_realm_export_consent: Callable[[str, Dict[str, Any]], None] = ...
check_realm_linkifiers: Callable[[str, Dict[str, Any]], None] = ...
check_realm_playgrounds: Callable[[str, Dict[str, Any]], None] = ...
check_realm_user_add: Callable[[str, Dict[str, Any]], None] = ...
check_realm_user_remove: Callable[[str, Dict[str, Any]], None] = ...
check_restart: Callable[[str, Dict[str, Any]], None] = ...
check_saved_snippets_add: Callable[[str, Dict[str, Any]], None] = ...
check_saved_snippets_remove: Callable[[str, Dict[str, Any]], None] = ...
check_scheduled_message_add: Callable[[str, Dict[str, Any]], None] = ...
check_scheduled_message_remove: Callable[[str, Dict[str, Any]], None] = ...
check_scheduled_message_update: Callable[[str, Dict[str, Any]], None] = ...
check_stream_create: Callable[[str, Dict[str, Any]], None] = ...
check_stream_delete: Callable[[str, Dict[str, Any]], None] = ...
check_submessage: Callable[[str, Dict[str, Any]], None] = ...
check_subscription_add: Callable[[str, Dict[str, Any]], None] = ...
check_subscription_peer_add: Callable[[str, Dict[str, Any]], None] = ...
check_subscription_peer_remove: Callable[[str, Dict[str, Any]], None] = ...
check_subscription_remove: Callable[[str, Dict[str, Any]], None] = ...
check_typing_start: Callable[[str, Dict[str, Any]], None] = ...
check_typing_stop: Callable[[str, Dict[str, Any]], None] = ...
check_typing_edit_channel_message_start: Callable[[str, Dict[str, Any]], None] = ...
check_typing_edit_direct_message_start: Callable[[str, Dict[str, Any]], None] = ...
check_typing_edit_channel_message_stop: Callable[[str, Dict[str, Any]], None] = ...
check_typing_edit_direct_message_stop: Callable[[str, Dict[str, Any]], None] = ...
check_update_message_flags_add: Callable[[str, Dict[str, Any]], None] = ...
check_update_message_flags_remove: Callable[[str, Dict[str, Any]], None] = ...
check_user_group_add: Callable[[str, Dict[str, Any]], None] = ...
check_user_group_add_members: Callable[[str, Dict[str, Any]], None] = ...
check_user_group_add_subgroups: Callable[[str, Dict[str, Any]], None] = ...
check_user_group_remove: Callable[[str, Dict[str, Any]], None] = ...
check_user_group_remove_members: Callable[[str, Dict[str, Any]], None] = ...
check_user_group_remove_subgroups: Callable[[str, Dict[str, Any]], None] = ...
check_user_topic: Callable[[str, Dict[str, Any]], None] = ...
check_web_reload_client_event: Callable[[str, Dict[str, Any]], None] = ...
_check_delete_message: Callable[[str, Dict[str, Any]], None] = ...
_check_has_zoom_token: Callable[[str, Dict[str, Any]], None] = ...
_check_presence: Callable[[str, Dict[str, Any]], None] = ...
_check_realm_bot_add: Callable[[str, Dict[str, Any]], None] = ...
_check_realm_bot_update: Callable[[str, Dict[str, Any]], None] = ...
_check_realm_default_update: Callable[[str, Dict[str, Any]], None] = ...
_check_realm_emoji_update: Callable[[str, Dict[str, Any]], None] = ...
_check_realm_export: Callable[[str, Dict[str, Any]], None] = ...
_check_realm_update: Callable[[str, Dict[str, Any]], None] = ...
_check_realm_update_dict: Callable[[str, Dict[str, Any]], None] = ...
_check_realm_user_update: Callable[[str, Dict[str, Any]], None] = ...
_check_stream_update: Callable[[str, Dict[str, Any]], None] = ...
_check_subscription_update: Callable[[str, Dict[str, Any]], None] = ...
_check_update_display_settings: Callable[[str, Dict[str, Any]], None] = ...
_check_update_global_notifications: Callable[[str, Dict[str, Any]], None] = ...
_check_update_message: Callable[[str, Dict[str, Any]], None] = ...
_check_user_group_update: Callable[[str, Dict[str, Any]], None] = ...
_check_user_settings_update: Callable[[str, Dict[str, Any]], None] = ...
_check_user_status: Callable[[str, Dict[str, Any]], None] = ...

PERSON_TYPES: Dict[str, Union[
    type[PersonAvatarFields],
    type[PersonBotOwnerId],
    type[PersonCustomProfileField],
    type[PersonDeliveryEmail],
    type[PersonEmail],
    type[PersonFullName],
    type[PersonIsActive],
    type[PersonIsBillingAdmin],
    type[PersonRole],
    type[PersonTimezone],
]] = ...

def check_delete_message(
    var_name: str,
    event: Dict[str, Any],
    message_type: str,
    num_message_ids: int,
    is_legacy: bool,
) -> None:
    ...

def check_has_zoom_token(
    var_name: str,
    event: Dict[str, Any],
    value: bool,
) -> None:
    ...

def check_presence(
    var_name: str,
    event: Dict[str, Any],
    has_email: bool,
    presence_key: str,
    status: str,
) -> None:
    ...

def check_realm_bot_add(var_name: str, event: Dict[str, Any]) -> None:
    ...

def check_realm_bot_update(
    var_name: str,
    event: Dict[str, Any],
    field: str,
) -> None:
    ...

def check_realm_emoji_update(var_name: str, event: Dict[str, Any]) -> None:
    ...

def check_realm_export(
    var_name: str,
    event: Dict[str, Any],
    has_export_url: bool,
    has_deleted_timestamp: bool,
    has_failed_timestamp: bool,
) -> None:
    ...

def check_realm_update(
    var_name: str,
    event: Dict[str, Any],
    prop: str,
) -> None:
    ...

def check_realm_default_update(
    var_name: str,
    event: Dict[str, Any],
    prop: str,
) -> None:
    ...

def check_realm_update_dict(var_name: str, event: Dict[str, Any]) -> None:
    ...

def check_realm_user_update(
    var_name: str,
    event: Dict[str, Any],
    person_flavor: str,
) -> None:
    ...

def check_stream_update(var_name: str, event: Dict[str, Any]) -> None:
    ...

def check_subscription_update(
    var_name: str,
    event: Dict[str, Any],
    property: str,
    value: Any,
) -> None:
    ...

def check_update_display_settings(
    var_name: str,
    event: Dict[str, Any],
) -> None:
    ...

def check_user_settings_update(
    var_name: str,
    event: Dict[str, Any],
) -> None:
    ...

def check_update_global_notifications(
    var_name: str,
    event: Dict[str, Any],
    desired_val: Any,
) -> None:
    ...

def check_update_message(
    var_name: str,
    event: Dict[str, Any],
    is_stream_message: bool,
    has_content: bool,
    has_topic: bool,
    has_new_stream_id: bool,
    is_embedded_update_only: bool,
) -> None:
    ...

def check_user_group_update(
    var_name: str,
    event: Dict[str, Any],
    fields: Set[str],
) -> None:
    ...

def check_user_status(
    var_name: str,
    event: Dict[str, Any],
    fields: Set[str],
) -> None:
    ...