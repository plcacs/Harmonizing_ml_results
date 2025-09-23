from collections.abc import Callable
from pprint import PrettyPrinter
from typing import cast, Any, Type
from pydantic import BaseModel
from zerver.lib.event_types import AllowMessageEditingData, AuthenticationData, BaseEvent, BotServicesEmbedded, BotServicesOutgoing, EventAlertWords, EventAttachmentAdd, EventAttachmentRemove, EventAttachmentUpdate, EventCustomProfileFields, EventDefaultStreamGroups, EventDefaultStreams, EventDeleteMessage, EventDirectMessage, EventDraftsAdd, EventDraftsRemove, EventDraftsUpdate, EventHasZoomToken, EventHeartbeat, EventInvitesChanged, EventMessage, EventMutedTopics, EventMutedUsers, EventOnboardingSteps, EventPresence, EventReactionAdd, EventReactionRemove, EventRealmBotAdd, EventRealmBotDelete, EventRealmBotUpdate, EventRealmDeactivated, EventRealmDomainsAdd, EventRealmDomainsChange, EventRealmDomainsRemove, EventRealmEmojiUpdate, EventRealmExport, EventRealmExportConsent, EventRealmLinkifiers, EventRealmPlaygrounds, EventRealmUpdate, EventRealmUpdateDict, EventRealmUserAdd, EventRealmUserRemove, EventRealmUserSettingsDefaultsUpdate, EventRealmUserUpdate, EventRestart, EventSavedSnippetsAdd, EventSavedSnippetsRemove, EventScheduledMessagesAdd, EventScheduledMessagesRemove, EventScheduledMessagesUpdate, EventStreamCreate, EventStreamDelete, EventStreamUpdate, EventSubmessage, EventSubscriptionAdd, EventSubscriptionPeerAdd, EventSubscriptionPeerRemove, EventSubscriptionRemove, EventSubscriptionUpdate, EventTypingEditChannelMessageStart, EventTypingEditChannelMessageStop, EventTypingEditDirectMessageStart, EventTypingEditDirectMessageStop, EventTypingStart, EventTypingStop, EventUpdateDisplaySettings, EventUpdateGlobalNotifications, EventUpdateMessage, EventUpdateMessageFlagsAdd, EventUpdateMessageFlagsRemove, EventUserGroupAdd, EventUserGroupAddMembers, EventUserGroupAddSubgroups, EventUserGroupRemove, EventUserGroupRemoveMembers, EventUserGroupRemoveSubgroups, EventUserGroupUpdate, EventUserSettingsUpdate, EventUserStatus, EventWebReloadClient, GroupSettingUpdateData, IconData, LogoData, MessageContentEditLimitSecondsData, NightLogoData, PersonAvatarFields, PersonBotOwnerId, PersonCustomProfileField, PersonDeliveryEmail, PersonEmail, PersonFullName, PersonIsActive, PersonIsBillingAdmin, PersonRole, PersonTimezone, PlanTypeData
from zerver.lib.topic import ORIG_TOPIC, TOPIC_NAME
from zerver.lib.types import AnonymousSettingGroupDict
from zerver.models import Realm, RealmUserDefault, Stream, UserProfile

def validate_with_model(data: dict[str, Any], model: Type[BaseModel]) -> None:
    allowed_fields = set(model.model_fields.keys())
    if not set(data.keys()).issubset(allowed_fields):
        raise ValueError(f'Extra fields not allowed: {set(data.keys()) - allowed_fields}')
    model.model_validate(data, strict=True)

def make_checker(base_model: Type[BaseModel]) -> Callable[[str, dict[str, Any]], None]:
    def f(label: str, event: dict[str, Any]) -> None:
        try:
            validate_with_model(event, base_model)
        except Exception as e:
            print(f'\nFAILURE:\n\nThe event below fails the check to make sure it has the\ncorrect "shape" of data:\n\n    {label}\n\nOften this is a symptom that the following type definition\nis either broken or needs to be updated due to other\nchanges that you have made:\n\n    {base_model}\n\nA traceback should follow to help you debug this problem.\n\nHere is the event:\n')
            PrettyPrinter(indent=4).pprint(event)
            raise e
    return f

check_alert_words: Callable[[str, dict[str, Any]], None] = make_checker(EventAlertWords)
check_attachment_add: Callable[[str, dict[str, Any]], None] = make_checker(EventAttachmentAdd)
check_attachment_remove: Callable[[str, dict[str, Any]], None] = make_checker(EventAttachmentRemove)
check_attachment_update: Callable[[str, dict[str, Any]], None] = make_checker(EventAttachmentUpdate)
check_custom_profile_fields: Callable[[str, dict[str, Any]], None] = make_checker(EventCustomProfileFields)
check_default_stream_groups: Callable[[str, dict[str, Any]], None] = make_checker(EventDefaultStreamGroups)
check_default_streams: Callable[[str, dict[str, Any]], None] = make_checker(EventDefaultStreams)
check_direct_message: Callable[[str, dict[str, Any]], None] = make_checker(EventDirectMessage)
check_draft_add: Callable[[str, dict[str, Any]], None] = make_checker(EventDraftsAdd)
check_draft_remove: Callable[[str, dict[str, Any]], None] = make_checker(EventDraftsRemove)
check_draft_update: Callable[[str, dict[str, Any]], None] = make_checker(EventDraftsUpdate)
check_heartbeat: Callable[[str, dict[str, Any]], None] = make_checker(EventHeartbeat)
check_invites_changed: Callable[[str, dict[str, Any]], None] = make_checker(EventInvitesChanged)
check_message: Callable[[str, dict[str, Any]], None] = make_checker(EventMessage)
check_muted_topics: Callable[[str, dict[str, Any]], None] = make_checker(EventMutedTopics)
check_muted_users: Callable[[str, dict[str, Any]], None] = make_checker(EventMutedUsers)
check_onboarding_steps: Callable[[str, dict[str, Any]], None] = make_checker(EventOnboardingSteps)
check_reaction_add: Callable[[str, dict[str, Any]], None] = make_checker(EventReactionAdd)
check_reaction_remove: Callable[[str, dict[str, Any]], None] = make_checker(EventReactionRemove)
check_realm_bot_delete: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmBotDelete)
check_realm_deactivated: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmDeactivated)
check_realm_domains_add: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmDomainsAdd)
check_realm_domains_change: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmDomainsChange)
check_realm_domains_remove: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmDomainsRemove)
check_realm_export_consent: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmExportConsent)
check_realm_linkifiers: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmLinkifiers)
check_realm_playgrounds: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmPlaygrounds)
check_realm_user_add: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmUserAdd)
check_realm_user_remove: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmUserRemove)
check_restart: Callable[[str, dict[str, Any]], None] = make_checker(EventRestart)
check_saved_snippets_add: Callable[[str, dict[str, Any]], None] = make_checker(EventSavedSnippetsAdd)
check_saved_snippets_remove: Callable[[str, dict[str, Any]], None] = make_checker(EventSavedSnippetsRemove)
check_scheduled_message_add: Callable[[str, dict[str, Any]], None] = make_checker(EventScheduledMessagesAdd)
check_scheduled_message_remove: Callable[[str, dict[str, Any]], None] = make_checker(EventScheduledMessagesRemove)
check_scheduled_message_update: Callable[[str, dict[str, Any]], None] = make_checker(EventScheduledMessagesUpdate)
check_stream_create: Callable[[str, dict[str, Any]], None] = make_checker(EventStreamCreate)
check_stream_delete: Callable[[str, dict[str, Any]], None] = make_checker(EventStreamDelete)
check_submessage: Callable[[str, dict[str, Any]], None] = make_checker(EventSubmessage)
check_subscription_add: Callable[[str, dict[str, Any]], None] = make_checker(EventSubscriptionAdd)
check_subscription_peer_add: Callable[[str, dict[str, Any]], None] = make_checker(EventSubscriptionPeerAdd)
check_subscription_peer_remove: Callable[[str, dict[str, Any]], None] = make_checker(EventSubscriptionPeerRemove)
check_subscription_remove: Callable[[str, dict[str, Any]], None] = make_checker(EventSubscriptionRemove)
check_typing_start: Callable[[str, dict[str, Any]], None] = make_checker(EventTypingStart)
check_typing_stop: Callable[[str, dict[str, Any]], None] = make_checker(EventTypingStop)
check_typing_edit_channel_message_start: Callable[[str, dict[str, Any]], None] = make_checker(EventTypingEditChannelMessageStart)
check_typing_edit_direct_message_start: Callable[[str, dict[str, Any]], None] = make_checker(EventTypingEditDirectMessageStart)
check_typing_edit_channel_message_stop: Callable[[str, dict[str, Any]], None] = make_checker(EventTypingEditChannelMessageStop)
check_typing_edit_direct_message_stop: Callable[[str, dict[str, Any]], None] = make_checker(EventTypingEditDirectMessageStop)
check_update_message_flags_add: Callable[[str, dict[str, Any]], None] = make_checker(EventUpdateMessageFlagsAdd)
check_update_message_flags_remove: Callable[[str, dict[str, Any]], None] = make_checker(EventUpdateMessageFlagsRemove)
check_user_group_add: Callable[[str, dict[str, Any]], None] = make_checker(EventUserGroupAdd)
check_user_group_add_members: Callable[[str, dict[str, Any]], None] = make_checker(EventUserGroupAddMembers)
check_user_group_add_subgroups: Callable[[str, dict[str, Any]], None] = make_checker(EventUserGroupAddSubgroups)
check_user_group_remove: Callable[[str, dict[str, Any]], None] = make_checker(EventUserGroupRemove)
check_user_group_remove_members: Callable[[str, dict[str, Any]], None] = make_checker(EventUserGroupRemoveMembers)
check_user_group_remove_subgroups: Callable[[str, dict[str, Any]], None] = make_checker(EventUserGroupRemoveSubgroups)
check_user_topic: Callable[[str, dict[str, Any]], None] = make_checker(EventUserTopic)
check_web_reload_client_event: Callable[[str, dict[str, Any]], None] = make_checker(EventWebReloadClient)
_check_delete_message: Callable[[str, dict[str, Any]], None] = make_checker(EventDeleteMessage)
_check_has_zoom_token: Callable[[str, dict[str, Any]], None] = make_checker(EventHasZoomToken)
_check_presence: Callable[[str, dict[str, Any]], None] = make_checker(EventPresence)
_check_realm_bot_add: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmBotAdd)
_check_realm_bot_update: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmBotUpdate)
_check_realm_default_update: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmUserSettingsDefaultsUpdate)
_check_realm_emoji_update: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmEmojiUpdate)
_check_realm_export: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmExport)
_check_realm_update: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmUpdate)
_check_realm_update_dict: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmUpdateDict)
_check_realm_user_update: Callable[[str, dict[str, Any]], None] = make_checker(EventRealmUserUpdate)
_check_stream_update: Callable[[str, dict[str, Any]], None] = make_checker(EventStreamUpdate)
_check_subscription_update: Callable[[str, dict[str, Any]], None] = make_checker(EventSubscriptionUpdate)
_check_update_display_settings: Callable[[str, dict[str, Any]], None] = make_checker(EventUpdateDisplaySettings)
_check_update_global_notifications: Callable[[str, dict[str, Any]], None] = make_checker(EventUpdateGlobalNotifications)
_check_update_message: Callable[[str, dict[str, Any]], None] = make_checker(EventUpdateMessage)
_check_user_group_update: Callable[[str, dict[str, Any]], None] = make_checker(EventUserGroupUpdate)
_check_user_settings_update: Callable[[str, dict[str, Any]], None] = make_checker(EventUserSettingsUpdate)
_check_user_status: Callable[[str, dict[str, Any]], None] = make_checker(EventUserStatus)
PERSON_TYPES: dict[str, Type[BaseModel]] = dict(avatar_fields=PersonAvatarFields, bot_owner_id=PersonBotOwnerId, custom_profile_field=PersonCustomProfileField, delivery_email=PersonDeliveryEmail, email=PersonEmail, full_name=PersonFullName, is_billing_admin=PersonIsBillingAdmin, role=PersonRole, timezone=PersonTimezone, is_active=PersonIsActive)

def check_delete_message(var_name: str, event: dict[str, Any], message_type: str, num_message_ids: int, is_legacy: bool) -> None:
    _check_delete_message(var_name, event)
    keys = {'id', 'type', 'message_type'}
    assert event['message_type'] == message_type
    if message_type == 'stream':
        keys |= {'stream_id', 'topic'}
    elif message_type == 'private':
        pass
    else:
        raise AssertionError('unexpected message_type')
    if is_legacy:
        assert num_message_ids == 1
        keys.add('message_id')
    else:
        assert isinstance(event['message_ids'], list)
        assert num_message_ids == len(event['message_ids'])
        keys.add('message_ids')
    assert set(event.keys()) == keys

def check_has_zoom_token(var_name: str, event: dict[str, Any], value: bool) -> None:
    _check_has_zoom_token(var_name, event)
    assert event['value'] == value

def check_presence(var_name: str, event: dict[str, Any], has_email: bool, presence_key: str, status: str) -> None:
    _check_presence(var_name, event)
    assert ('email' in event) == has_email
    assert isinstance(event['presence'], dict)
    [(event_presence_key, event_presence_value)] = event['presence'].items()
    assert event_presence_key == presence_key
    assert event_presence_value['status'] == status

def check_realm_bot_add(var_name: str, event: dict[str, Any]) -> None:
    _check_realm_bot_add(var_name, event)
    assert isinstance(event['bot'], dict)
    bot_type = event['bot']['bot_type']
    services = event['bot']['services']
    if bot_type == UserProfile.DEFAULT_BOT:
        assert services == []
    elif bot_type == UserProfile.OUTGOING_WEBHOOK_BOT:
        assert len(services) == 1
        validate_with_model(services[0], BotServicesOutgoing)
    elif bot_type == UserProfile.EMBEDDED_BOT:
        assert len(services) == 1
        validate_with_model(services[0], BotServicesEmbedded)
    else:
        raise AssertionError(f'Unknown bot_type: {bot_type}')

def check_realm_bot_update(var_name: str, event: dict[str, Any], field: str) -> None:
    _check_realm_bot_update(var_name, event)
    assert isinstance(event['bot'], dict)
    assert {'user_id', field} == set(event['bot'].keys())

def check_realm_emoji_update(var_name: str, event: dict[str, Any]) -> None:
    """
    The way we send realm emojis is kinda clumsy--we
    send a dict mapping the emoji id to a sub_dict with
    the fields (including the id).  Ideally we can streamline
    this and just send a list of dicts.  The clients can make
    a Map as needed.
    """
    _check_realm_emoji_update(var_name, event)
    assert isinstance(event['realm_emoji'], dict)
    for (k, v) in event['realm_emoji'].items():
        assert v['id'] == k

def check_realm_export(var_name: str, event: dict[str, Any], has_export_url: bool, has_deleted_timestamp: bool, has_failed_timestamp: bool) -> None:
    _check_realm_export(var_name, event)
    assert isinstance(event['exports'], list)
    assert len(event['exports']) == 1
    export = event['exports'][0]
    assert has_export_url == (export['export_url'] is not None)
    assert has_deleted_timestamp == (export['deleted_timestamp'] is not None)
    assert has_failed_timestamp == (export['failed_timestamp'] is not None)

def check_realm_update(var_name: str, event: dict[str, Any], prop: str) -> None:
    """
    Realm updates have these two fields:

        property
        value

    We check not only the basic schema, but also that
    the value people actually matches the type from
    Realm.property_types that we have configured
    for the property.
   