from collections.abc import Callable
from pprint import PrettyPrinter
from typing import Any, Dict, Set, cast
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
from zerver.lib.topic import ORIG_TOPIC, TOPIC_NAME
from zerver.lib.types import AnonymousSettingGroupDict
from zerver.models import Realm, RealmUserDefault, Stream, UserProfile


def validate_with_model(data: Dict[str, Any], model: type[BaseModel]) -> None:
    allowed_fields: Set[str] = set(model.model_fields.keys())
    if not set(data.keys()).issubset(allowed_fields):
        raise ValueError(f'Extra fields not allowed: {set(data.keys()) - allowed_fields}')
    model.model_validate(data, strict=True)


def make_checker(base_model: type[BaseModel]) -> Callable[[str, Dict[str, Any]], None]:
    def f(label: str, event: Dict[str, Any]) -> None:
        try:
            validate_with_model(event, base_model)
        except Exception as e:
            print(
                f'\nFAILURE:\n\nThe event below fails the check to make sure it has the\ncorrect "shape" of data:\n\n    {label}\n\nOften this is a symptom that the following type definition\nis either broken or needs to be updated due to other\nchanges that you have made:\n\n    {base_model}\n\nA traceback should follow to help you debug this problem.\n\nHere is the event:\n'
            )
            PrettyPrinter(indent=4).pprint(event)
            raise e

    return f


check_alert_words = make_checker(EventAlertWords)
check_attachment_add = make_checker(EventAttachmentAdd)
check_attachment_remove = make_checker(EventAttachmentRemove)
check_attachment_update = make_checker(EventAttachmentUpdate)
check_custom_profile_fields = make_checker(EventCustomProfileFields)
check_default_stream_groups = make_checker(EventDefaultStreamGroups)
check_default_streams = make_checker(EventDefaultStreams)
check_direct_message = make_checker(EventDirectMessage)
check_draft_add = make_checker(EventDraftsAdd)
check_draft_remove = make_checker(EventDraftsRemove)
check_draft_update = make_checker(EventDraftsUpdate)
check_heartbeat = make_checker(EventHeartbeat)
check_invites_changed = make_checker(EventInvitesChanged)
check_message = make_checker(EventMessage)
check_muted_topics = make_checker(EventMutedTopics)
check_muted_users = make_checker(EventMutedUsers)
check_onboarding_steps = make_checker(EventOnboardingSteps)
check_reaction_add = make_checker(EventReactionAdd)
check_reaction_remove = make_checker(EventReactionRemove)
check_realm_bot_delete = make_checker(EventRealmBotDelete)
check_realm_deactivated = make_checker(EventRealmDeactivated)
check_realm_domains_add = make_checker(EventRealmDomainsAdd)
check_realm_domains_change = make_checker(EventRealmDomainsChange)
check_realm_domains_remove = make_checker(EventRealmDomainsRemove)
check_realm_export_consent = make_checker(EventRealmExportConsent)
check_realm_linkifiers = make_checker(EventRealmLinkifiers)
check_realm_playgrounds = make_checker(EventRealmPlaygrounds)
check_realm_user_add = make_checker(EventRealmUserAdd)
check_realm_user_remove = make_checker(EventRealmUserRemove)
check_restart = make_checker(EventRestart)
check_saved_snippets_add = make_checker(EventSavedSnippetsAdd)
check_saved_snippets_remove = make_checker(EventSavedSnippetsRemove)
check_scheduled_message_add = make_checker(EventScheduledMessagesAdd)
check_scheduled_message_remove = make_checker(EventScheduledMessagesRemove)
check_scheduled_message_update = make_checker(EventScheduledMessagesUpdate)
check_stream_create = make_checker(EventStreamCreate)
check_stream_delete = make_checker(EventStreamDelete)
check_submessage = make_checker(EventSubmessage)
check_subscription_add = make_checker(EventSubscriptionAdd)
check_subscription_peer_add = make_checker(EventSubscriptionPeerAdd)
check_subscription_peer_remove = make_checker(EventSubscriptionPeerRemove)
check_subscription_remove = make_checker(EventSubscriptionRemove)
check_typing_start = make_checker(EventTypingStart)
check_typing_stop = make_checker(EventTypingStop)
check_typing_edit_channel_message_start = make_checker(EventTypingEditChannelMessageStart)
check_typing_edit_direct_message_start = make_checker(EventTypingEditDirectMessageStart)
check_typing_edit_channel_message_stop = make_checker(EventTypingEditChannelMessageStop)
check_typing_edit_direct_message_stop = make_checker(EventTypingEditDirectMessageStop)
check_update_message_flags_add = make_checker(EventUpdateMessageFlagsAdd)
check_update_message_flags_remove = make_checker(EventUpdateMessageFlagsRemove)
check_user_group_add = make_checker(EventUserGroupAdd)
check_user_group_add_members = make_checker(EventUserGroupAddMembers)
check_user_group_add_subgroups = make_checker(EventUserGroupAddSubgroups)
check_user_group_remove = make_checker(EventUserGroupRemove)
check_user_group_remove_members = make_checker(EventUserGroupRemoveMembers)
check_user_group_remove_subgroups = make_checker(EventUserGroupRemoveSubgroups)
check_user_topic = make_checker(EventUserTopic)
check_web_reload_client_event = make_checker(EventWebReloadClient)
_check_delete_message = make_checker(EventDeleteMessage)
_check_has_zoom_token = make_checker(EventHasZoomToken)
_check_presence = make_checker(EventPresence)
_check_realm_bot_add = make_checker(EventRealmBotAdd)
_check_realm_bot_update = make_checker(EventRealmBotUpdate)
_check_realm_default_update = make_checker(EventRealmUserSettingsDefaultsUpdate)
_check_realm_emoji_update = make_checker(EventRealmEmojiUpdate)
_check_realm_export = make_checker(EventRealmExport)
_check_realm_update = make_checker(EventRealmUpdate)
_check_realm_update_dict = make_checker(EventRealmUpdateDict)
_check_realm_user_update = make_checker(EventRealmUserUpdate)
_check_stream_update = make_checker(EventStreamUpdate)
_check_subscription_update = make_checker(EventSubscriptionUpdate)
_check_update_display_settings = make_checker(EventUpdateDisplaySettings)
_check_update_global_notifications = make_checker(EventUpdateGlobalNotifications)
_check_update_message = make_checker(EventUpdateMessage)
_check_user_group_update = make_checker(EventUserGroupUpdate)
_check_user_settings_update = make_checker(EventUserSettingsUpdate)
_check_user_status = make_checker(EventUserStatus)
PERSON_TYPES: Dict[str, Any] = dict(
    avatar_fields=PersonAvatarFields,
    bot_owner_id=PersonBotOwnerId,
    custom_profile_field=PersonCustomProfileField,
    delivery_email=PersonDeliveryEmail,
    email=PersonEmail,
    full_name=PersonFullName,
    is_billing_admin=PersonIsBillingAdmin,
    role=PersonRole,
    timezone=PersonTimezone,
    is_active=PersonIsActive,
)


def check_delete_message(
    var_name: str,
    event: Dict[str, Any],
    message_type: str,
    num_message_ids: int,
    is_legacy: bool,
) -> None:
    _check_delete_message(var_name, event)
    keys: Set[str] = {'id', 'type', 'message_type'}
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


def check_has_zoom_token(var_name: str, event: Dict[str, Any], value: Any) -> None:
    _check_has_zoom_token(var_name, event)
    assert event['value'] == value


def check_presence(
    var_name: str, event: Dict[str, Any], has_email: bool, presence_key: str, status: Any
) -> None:
    _check_presence(var_name, event)
    assert ('email' in event) == has_email
    assert isinstance(event['presence'], dict)
    [(event_presence_key, event_presence_value)] = event['presence'].items()
    assert event_presence_key == presence_key
    assert event_presence_value['status'] == status


def check_realm_bot_add(var_name: str, event: Dict[str, Any]) -> None:
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


def check_realm_bot_update(var_name: str, event: Dict[str, Any], field: str) -> None:
    _check_realm_bot_update(var_name, event)
    assert isinstance(event['bot'], dict)
    assert {'user_id', field} == set(event['bot'].keys())


def check_realm_emoji_update(var_name: str, event: Dict[str, Any]) -> None:
    """
    The way we send realm emojis is kinda clumsy--we
    send a dict mapping the emoji id to a sub_dict with
    the fields (including the id).  Ideally we can streamline
    this and just send a list of dicts.  The clients can make
    a Map as needed.
    """
    _check_realm_emoji_update(var_name, event)
    assert isinstance(event['realm_emoji'], dict)
    for k, v in event['realm_emoji'].items():
        assert v['id'] == k


def check_realm_export(
    var_name: str,
    event: Dict[str, Any],
    has_export_url: bool,
    has_deleted_timestamp: bool,
    has_failed_timestamp: bool,
) -> None:
    _check_realm_export(var_name, event)
    assert isinstance(event['exports'], list)
    assert len(event['exports']) == 1
    export = event['exports'][0]
    assert has_export_url == (export['export_url'] is not None)
    assert has_deleted_timestamp == (export['deleted_timestamp'] is not None)
    assert has_failed_timestamp == (export['failed_timestamp'] is not None)


def check_realm_update(var_name: str, event: Dict[str, Any], prop: str) -> None:
    """
    Realm updates have these two fields:

        property
        value

    We check not only the basic schema, but also that
    the value people actually matches the type from
    Realm.property_types that we have configured
    for the property.
    """
    _check_realm_update(var_name, event)
    assert prop == event['property']
    value = event['value']
    if prop in [
        'moderation_request_channel_id',
        'new_stream_announcements_stream_id',
        'signup_announcements_stream_id',
        'zulip_update_announcements_stream_id',
        'org_type',
    ]:
        assert isinstance(value, int)
        return
    property_type = Realm.property_types[prop]
    assert isinstance(value, property_type)


def check_realm_default_update(var_name: str, event: Dict[str, Any], prop: str) -> None:
    _check_realm_default_update(var_name, event)
    assert prop == event['property']
    assert prop != 'default_language'
    assert prop in RealmUserDefault.property_types
    prop_type = RealmUserDefault.property_types[prop]
    assert isinstance(event['value'], prop_type)


def check_realm_update_dict(var_name: str, event: Dict[str, Any]) -> None:
    _check_realm_update_dict(var_name, event)
    if event['property'] == 'default':
        assert isinstance(event['data'], dict)
        if 'allow_message_editing' in event['data']:
            sub_type = AllowMessageEditingData
        elif 'message_content_edit_limit_seconds' in event['data']:
            sub_type = MessageContentEditLimitSecondsData
        elif 'authentication_methods' in event['data']:
            sub_type = AuthenticationData
        elif any((setting_name in event['data'] for setting_name in Realm.REALM_PERMISSION_GROUP_SETTINGS)):
            sub_type = GroupSettingUpdateData
        elif 'plan_type' in event['data']:
            sub_type = PlanTypeData
        else:
            raise AssertionError('unhandled fields in data')
    elif event['property'] == 'icon':
        sub_type = IconData
    elif event['property'] == 'logo':
        sub_type = LogoData
    elif event['property'] == 'night_logo':
        sub_type = NightLogoData
    else:
        raise AssertionError(f"unhandled property: {event['property']}")
    validate_with_model(cast(Dict[str, Any], event['data']), sub_type)


def check_realm_user_update(var_name: str, event: Dict[str, Any], person_flavor: str) -> None:
    _check_realm_user_update(var_name, event)
    sub_type = PERSON_TYPES[person_flavor]
    validate_with_model(cast(Dict[str, Any], event['person']), sub_type)


def check_stream_update(var_name: str, event: Dict[str, Any]) -> None:
    _check_stream_update(var_name, event)
    prop = event['property']
    value = event['value']
    extra_keys: Set[str] = set(event.keys()) - {'id', 'type', 'op', 'property', 'value', 'name', 'stream_id', 'first_message_id'}
    if prop == 'description':
        assert extra_keys == {'rendered_description'}
        assert isinstance(value, str)
    elif prop == 'invite_only':
        assert extra_keys == {'history_public_to_subscribers', 'is_web_public'}
        assert isinstance(value, bool)
    elif prop == 'message_retention_days':
        assert extra_keys == set()
        if value is not None:
            assert isinstance(value, int)
    elif prop == 'name':
        assert extra_keys == set()
        assert isinstance(value, str)
    elif prop == 'stream_post_policy':
        assert extra_keys == set()
        assert value in Stream.STREAM_POST_POLICY_TYPES
    elif prop in Stream.stream_permission_group_settings:
        assert extra_keys == set()
        assert isinstance(value, int | AnonymousSettingGroupDict)
    elif prop == 'first_message_id':
        assert extra_keys == set()
        assert isinstance(value, int)
    elif prop == 'is_recently_active':
        assert extra_keys == set()
        assert isinstance(value, bool)
    elif prop == 'is_announcement_only':
        assert extra_keys == set()
        assert isinstance(value, bool)
    else:
        raise AssertionError(f'Unknown property: {prop}')


def check_subscription_update(var_name: str, event: Dict[str, Any], property: str, value: Any) -> None:
    _check_subscription_update(var_name, event)
    assert event['property'] == property
    assert event['value'] == value


def check_update_display_settings(var_name: str, event: Dict[str, Any]) -> None:
    """
    Display setting events have a "setting" field that
    is more specifically typed according to the
    UserProfile.property_types dictionary.
    """
    _check_update_display_settings(var_name, event)
    setting_name: str = event['setting_name']
    setting = event['setting']
    assert isinstance(setting_name, str)
    if setting_name == 'timezone':
        assert isinstance(setting, str)
    else:
        setting_type = UserProfile.property_types[setting_name]
        assert isinstance(setting, setting_type)
    if setting_name == 'default_language':
        assert 'language_name' in event
    else:
        assert 'language_name' not in event


def check_user_settings_update(var_name: str, event: Dict[str, Any]) -> None:
    _check_user_settings_update(var_name, event)
    setting_name: str = event['property']
    value = event['value']
    assert isinstance(setting_name, str)
    if setting_name == 'timezone':
        assert isinstance(value, str)
    else:
        setting_type = UserProfile.property_types[setting_name]
        assert isinstance(value, setting_type)
    if setting_name == 'default_language':
        assert 'language_name' in event
    else:
        assert 'language_name' not in event


def check_update_global_notifications(var_name: str, event: Dict[str, Any], desired_val: Any) -> None:
    """
    See UserProfile.notification_settings_legacy for
    more details.
    """
    _check_update_global_notifications(var_name, event)
    setting_name: str = event['notification_name']
    setting = event['setting']
    assert setting == desired_val
    assert isinstance(setting_name, str)
    setting_type = UserProfile.notification_settings_legacy[setting_name]
    assert isinstance(setting, setting_type)


def check_update_message(
    var_name: str,
    event: Dict[str, Any],
    is_stream_message: bool,
    has_content: bool,
    has_topic: bool,
    has_new_stream_id: bool,
    is_embedded_update_only: bool,
) -> None:
    _check_update_message(var_name, event)
    actual_keys: Set[str] = set(event.keys())
    expected_keys: Set[str] = {'id', 'type', 'user_id', 'edit_timestamp', 'message_id', 'flags', 'message_ids', 'rendering_only'}
    if is_stream_message:
        expected_keys |= {'stream_id', 'stream_name'}
    if has_content:
        expected_keys |= {'is_me_message', 'orig_content', 'orig_rendered_content', 'content', 'rendered_content'}
    if has_topic:
        expected_keys |= {'topic_links', ORIG_TOPIC, TOPIC_NAME, 'propagate_mode'}
    if has_new_stream_id:
        expected_keys |= {'new_stream_id', ORIG_TOPIC, 'propagate_mode'}
    if is_embedded_update_only:
        expected_keys |= {'content', 'rendered_content'}
        assert event['user_id'] is None
    else:
        assert isinstance(event['user_id'], int)
    assert event['rendering_only'] == is_embedded_update_only
    assert expected_keys == actual_keys


def check_user_group_update(var_name: str, event: Dict[str, Any], fields: Set[str]) -> None:
    _check_user_group_update(var_name, event)
    assert isinstance(event['data'], dict)
    assert set(event['data'].keys()) == fields


def check_user_status(var_name: str, event: Dict[str, Any], fields: Set[str]) -> None:
    _check_user_status(var_name, event)
    assert set(event.keys()) == {'id', 'type', 'user_id'} | fields