from typing import Any, Dict, List, Literal, Optional, Set, Type, Union

def validate_with_model(data: Dict[str, object], model: Type[BaseModel]) -> None:
    allowed_fields = set(model.model_fields.keys())
    if not set(data.keys()).issubset(allowed_fields):  # nocoverage
        raise ValueError(f"Extra fields not allowed: {set(data.keys()) - allowed_fields}")

    model.model_validate(data, strict=True)

def make_checker(base_model: Type[BaseEvent]) -> Callable[[str, Dict[str, object]], None]:
    def f(label: str, event: Dict[str, object]) -> None:
        try:
            validate_with_model(event, base_model)
        except Exception as e:  # nocoverage
            print(f"""
FAILURE:

The event below fails the check to make sure it has the
correct "shape" of data:

    {label}

Often this is a symptom that the following type definition
is either broken or needs to be updated due to other
changes that you have made:

    {base_model}

A traceback should follow to help you debug this problem.

Here is the event:
""")

            PrettyPrinter(indent=4).pprint(event)
            raise e

    return f

def check_delete_message(
    var_name: str,
    event: Dict[str, object],
    message_type: Literal["stream", "private"],
    num_message_ids: int,
    is_legacy: bool,
) -> None:
    _check_delete_message(var_name, event)

    keys = {"id", "type", "message_type"}

    assert event["message_type"] == message_type

    if message_type == "stream":
        keys |= {"stream_id", "topic"}
    elif message_type == "private":
        pass
    else:
        raise AssertionError("unexpected message_type")

    if is_legacy:
        assert num_message_ids == 1
        keys.add("message_id")
    else:
        assert isinstance(event["message_ids"], list)
        assert num_message_ids == len(event["message_ids"])
        keys.add("message_ids")

    assert set(event.keys()) == keys

def check_has_zoom_token(
    var_name: str,
    event: Dict[str, object],
    value: bool,
) -> None:
    _check_has_zoom_token(var_name, event)
    assert event["value"] == value

def check_presence(
    var_name: str,
    event: Dict[str, object],
    has_email: bool,
    presence_key: str,
    status: str,
) -> None:
    _check_presence(var_name, event)

    assert ("email" in event) == has_email

    assert isinstance(event["presence"], dict)

    # Our tests only have one presence value.
    [(event_presence_key, event_presence_value)] = event["presence"].items()
    assert event_presence_key == presence_key
    assert event_presence_value["status"] == status

def check_realm_bot_add(
    var_name: str,
    event: Dict[str, object],
) -> None:
    _check_realm_bot_add(var_name, event)

    assert isinstance(event["bot"], dict)
    bot_type = event["bot"]["bot_type"]

    services = event["bot"]["services"]

    if bot_type == UserProfile.DEFAULT_BOT:
        assert services == []
    elif bot_type == UserProfile.OUTGOING_WEBHOOK_BOT:
        assert len(services) == 1
        validate_with_model(services[0], BotServicesOutgoing)
    elif bot_type == UserProfile.EMBEDDED_BOT:
        assert len(services) == 1
        validate_with_model(services[0], BotServicesEmbedded)
    else:
        raise AssertionError(f"Unknown bot_type: {bot_type}")

def check_realm_bot_update(
    var_name: str,
    event: Dict[str, object],
    field: str,
) -> None:
    _check_realm_bot_update(var_name, event)

    assert isinstance(event["bot"], dict)
    assert {"user_id", field} == set(event["bot"].keys())

def check_realm_emoji_update(var_name: str, event: Dict[str, object]) -> None:
    _check_realm_emoji_update(var_name, event)

    assert isinstance(event["realm_emoji"], dict)
    for k, v in event["realm_emoji"].items():
        assert v["id"] == k

def check_realm_export(
    var_name: str,
    event: Dict[str, object],
    has_export_url: bool,
    has_deleted_timestamp: bool,
    has_failed_timestamp: bool,
) -> None:
    _check_realm_export(var_name, event)

    assert isinstance(event["exports"], list)
    assert len(event["exports"]) == 1
    export = event["exports"][0]

    assert has_export_url == (export["export_url"] is not None)
    assert has_deleted_timestamp == (export["deleted_timestamp"] is not None)
    assert has_failed_timestamp == (export["failed_timestamp"] is not None)

def check_realm_update(
    var_name: str,
    event: Dict[str, object],
    prop: str,
) -> None:
    _check_realm_update(var_name, event)

    assert prop == event["property"]
    value = event["value"]

    if prop in [
        "moderation_request_channel_id",
        "new_stream_announcements_stream_id",
        "signup_announcements_stream_id",
        "zulip_update_announcements_stream_id",
        "org_type",
    ]:
        assert isinstance(value, int)
        return

    property_type = Realm.property_types[prop]
    assert isinstance(value, property_type)

def check_realm_default_update(
    var_name: str,
    event: Dict[str, object],
    prop: str,
) -> None:
    _check_realm_default_update(var_name, event)

    assert prop == event["property"]
    assert prop != "default_language"
    assert prop in RealmUserDefault.property_types

    prop_type = RealmUserDefault.property_types[prop]
    assert isinstance(event["value"], prop_type)

def check_realm_update_dict(
    var_name: str,
    event: Dict[str, object],
) -> None:
    _check_realm_update_dict(var_name, event)

    if event["property"] == "default":
        assert isinstance(event["data"], dict)

        if "allow_message_editing" in event["data"]:
            sub_type: Type[BaseModel] = AllowMessageEditingData
        elif "message_content_edit_limit_seconds" in event["data"]:
            sub_type = MessageContentEditLimitSecondsData
        elif "authentication_methods" in event["data"]:
            sub_type = AuthenticationData
        elif any(
            setting_name in event["data"] for setting_name in Realm.REALM_PERMISSION_GROUP_SETTINGS
        ):
            sub_type = GroupSettingUpdateData
        elif "plan_type" in event["data"]:
            sub_type = PlanTypeData
        else:
            raise AssertionError("unhandled fields in data")

    elif event["property"] == "icon":
        sub_type = IconData
    elif event["property"] == "logo":
        sub_type = LogoData
    elif event["property"] == "night_logo":
        sub_type = NightLogoData
    else:
        raise AssertionError("unhandled property: {event['property']}")

    validate_with_model(cast(Dict[str, object], event["data"]), sub_type)

def check_realm_user_update(
    var_name: str,
    event: Dict[str, object],
    person_flavor: str,
) -> None:
    _check_realm_user_update(var_name, event)

    sub_type = PERSON_TYPES[person_flavor]
    validate_with_model(cast(Dict[str, object], event["person"]), sub_type)

def check_stream_update(
    var_name: str,
    event: Dict[str, object],
) -> None:
    _check_stream_update(var_name, event)
    prop = event["property"]
    value = event["value"]

    extra_keys = set(event.keys()) - {
        "id",
        "type",
        "op",
        "property",
        "value",
        "name",
        "stream_id",
        "first_message_id",
    }

    if prop == "description":
        assert extra_keys == {"rendered_description"}
        assert isinstance(value, str)
    elif prop == "invite_only":
        assert extra_keys == {"history_public_to_subscribers", "is_web_public"}
        assert isinstance(value, bool)
    elif prop == "message_retention_days":
        assert extra_keys == set()
        if value is not None:
            assert isinstance(value, int)
    elif prop == "name":
        assert extra_keys == set()
        assert isinstance(value, str)
    elif prop == "stream_post_policy":
        assert extra_keys == set()
        assert value in Stream.STREAM_POST_POLICY_TYPES
    elif prop in Stream.stream_permission_group_settings:
        assert extra_keys == set()
        assert isinstance(value, Union[int, AnonymousSettingGroupDict])
    elif prop == "first_message_id":
        assert extra_keys == set()
        assert isinstance(value, int)
    elif prop == "is_recently_active":
        assert extra_keys == set()
        assert isinstance(value, bool)
    elif prop == "is_announcement_only":
        assert extra_keys == set()
        assert isinstance(value, bool)
    else:
        raise AssertionError(f"Unknown property: {prop}")

def check_subscription_update(
    var_name: str, event: Dict[str, object], property: str, value: bool
) -> None:
    _check_subscription_update(var_name, event)
    assert event["property"] == property
    assert event["value"] == value

def check_update_display_settings(
    var_name: str,
    event: Dict[str, object],
) -> None:
    _check_update_display_settings(var_name, event)
    setting_name = event["setting_name"]
    setting = event["setting"]

    assert isinstance(setting_name, str)
    if setting_name == "timezone":
        assert isinstance(setting, str)
    else:
        setting_type = UserProfile.property_types[setting_name]
        assert isinstance(setting, setting_type)

    if setting_name == "default_language":
        assert "language_name" in event
    else:
        assert "language_name" not in event

def check_user_settings_update(
    var_name: str,
    event: Dict[str, object],
) -> None:
    _check_user_settings_update(var_name, event)
    setting_name = event["property"]
    value = event["value"]

    assert isinstance(setting_name, str)
    if setting_name == "timezone":
        assert isinstance(value, str)
    else:
        setting_type = UserProfile.property_types[setting_name]
        assert isinstance(value, setting_type)

    if setting_name == "default_language":
        assert "language_name" in event
    else:
        assert "language_name" not in event

def check_update_global_notifications(
    var_name: str,
    event: Dict[str, object],
    desired_val: Union[bool, int, str],
) -> None:
    _check_update_global_notifications(var_name, event)
    setting_name = event["notification_name"]
    setting = event["setting"]
    assert setting == desired_val

    assert isinstance(setting_name, str)
    setting_type = UserProfile.notification_settings_legacy[setting_name]
    assert isinstance(setting, setting_type)

def check_update_message(
    var_name: str,
    event: Dict[str, object],
    is_stream_message: bool,
    has_content: bool,
    has_topic: bool,
    has_new_stream_id: bool,
    is_embedded_update_only: bool,
) -> None:
    _check_update_message(var_name, event)

    actual_keys = set(event.keys())
    expected_keys = {
        "id",
        "type",
        "user_id",
        "edit_timestamp",
        "message_id",
        "flags",
        "message_ids",
        "rendering_only",
    }

    if is_stream_message:
        expected_keys |= {
            "stream_id",
            "stream_name",
        }

    if has_content:
        expected_keys |= {
            "is_me_message",
            "orig_content",
            "orig_rendered_content",
            "content",
            "rendered_content",
        }

    if has_topic:
        expected_keys |= {
            "topic_links",
            ORIG_TOPIC,
            TOPIC_NAME,
            "propagate_mode",
        }

    if has_new_stream_id:
        expected_keys |= {
            "new_stream_id",
            ORIG_TOPIC,
            "propagate_mode",
        }

    if is_embedded_update_only:
        expected_keys |= {
            "content",
            "rendered_content",
        }
        assert event["user_id"] is None
    else:
        assert isinstance(event["user_id"], int)

    assert event["rendering_only"] == is_embedded_update_only
    assert expected_keys == actual_keys

def check_user_group_update(var_name: str, event: Dict[str, object], fields: Set[str]) -> None:
    _check_user_group_update(var_name, event)

    assert isinstance(event["data"], dict)

    assert set(event["data"].keys()) == fields

def check_user_status(var_name: str, event: Dict[str, object], fields: Set[str]) -> None:
    _check_user_status(var_name, event)

    assert set(event.keys()) == {"id", "type", "user_id"} | fields
