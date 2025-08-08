def validate_with_model(data: dict, model: BaseModel) -> None:
    allowed_fields = set(model.__fields__.keys())
    if not set(data.keys()).issubset(allowed_fields):
        raise ValueError(f'Extra fields not allowed: {set(data.keys()) - allowed_fields}')
    model.validate(data, strict=True)

def make_checker(base_model: BaseModel) -> Callable:
    def f(label: str, event: dict) -> None:
        try:
            validate_with_model(event, base_model)
        except Exception as e:
            print(f'\nFAILURE:\n\nThe event below fails the check to make sure it has the\ncorrect "shape" of data:\n\n    {label}\n\nOften this is a symptom that the following type definition\nis either broken or needs to be updated due to other\nchanges that you have made:\n\n    {base_model}\n\nA traceback should follow to help you debug this problem.\n\nHere is the event:\n')
            PrettyPrinter(indent=4).pprint(event)
            raise e
    return f

def check_delete_message(var_name: str, event: dict, message_type: str, num_message_ids: int, is_legacy: bool) -> None:
    ...

def check_has_zoom_token(var_name: str, event: dict, value: bool) -> None:
    ...

def check_presence(var_name: str, event: dict, has_email: bool, presence_key: str, status: str) -> None:
    ...

def check_realm_bot_add(var_name: str, event: dict) -> None:
    ...

def check_realm_bot_update(var_name: str, event: dict, field: str) -> None:
    ...

def check_realm_emoji_update(var_name: str, event: dict) -> None:
    ...

def check_realm_export(var_name: str, event: dict, has_export_url: bool, has_deleted_timestamp: bool, has_failed_timestamp: bool) -> None:
    ...

def check_realm_update(var_name: str, event: dict, prop: str) -> None:
    ...

def check_realm_default_update(var_name: str, event: dict, prop: str) -> None:
    ...

def check_realm_update_dict(var_name: str, event: dict) -> None:
    ...

def check_realm_user_update(var_name: str, event: dict, person_flavor: str) -> None:
    ...

def check_stream_update(var_name: str, event: dict) -> None:
    ...

def check_subscription_update(var_name: str, event: dict, property: str, value: bool) -> None:
    ...

def check_update_display_settings(var_name: str, event: dict) -> None:
    ...

def check_user_settings_update(var_name: str, event: dict) -> None:
    ...

def check_update_global_notifications(var_name: str, event: dict, desired_val: bool) -> None:
    ...

def check_update_message(var_name: str, event: dict, is_stream_message: bool, has_content: bool, has_topic: bool, has_new_stream_id: bool, is_embedded_update_only: bool) -> None:
    ...

def check_user_group_update(var_name: str, event: dict, fields: set) -> None:
    ...

def check_user_status(var_name: str, event: dict, fields: set) -> None:
    ...
