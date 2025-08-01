from collections.abc import Iterable
import orjson
from django.db import transaction
from django.utils.translation import gettext as _
from zerver.lib.exceptions import JsonableError
from zerver.lib.external_accounts import DEFAULT_EXTERNAL_ACCOUNTS
from zerver.lib.streams import render_stream_description
from zerver.lib.types import ProfileDataElementUpdateDict, ProfileFieldData
from zerver.lib.users import get_user_ids_who_can_access_user
from zerver.models import CustomProfileField, CustomProfileFieldValue, Realm, UserProfile
from zerver.models.custom_profile_fields import custom_profile_fields_for_realm
from zerver.models.users import active_user_ids
from zerver.tornado.django_api import send_event_on_commit
from typing import Any, Dict, List, Optional, Set, Union

def notify_realm_custom_profile_fields(realm: Realm) -> None:
    fields = custom_profile_fields_for_realm(realm.id)
    event: Dict[str, Any] = dict(type='custom_profile_fields', fields=[f.as_dict() for f in fields])
    send_event_on_commit(realm, event, active_user_ids(realm.id))

@transaction.atomic(durable=True)
def try_add_realm_default_custom_profile_field(realm: Realm, field_subtype: str, display_in_profile_summary: bool = False, required: bool = False, editable_by_user: bool = True) -> CustomProfileField:
    field_data = DEFAULT_EXTERNAL_ACCOUNTS[field_subtype]
    custom_profile_field = CustomProfileField(realm=realm, name=str(field_data.name), field_type=CustomProfileField.EXTERNAL_ACCOUNT, hint=field_data.hint, field_data=orjson.dumps(dict(subtype=field_subtype)).decode(), display_in_profile_summary=display_in_profile_summary, required=required, editable_by_user=editable_by_user)
    custom_profile_field.save()
    custom_profile_field.order = custom_profile_field.id
    custom_profile_field.save(update_fields=['order'])
    notify_realm_custom_profile_fields(realm)
    return custom_profile_field

@transaction.atomic(durable=True)
def try_add_realm_custom_profile_field(realm: Realm, name: str, field_type: int, hint: str = '', field_data: Optional[Dict[str, Any]] = None, display_in_profile_summary: bool = False, required: bool = False, editable_by_user: bool = True) -> CustomProfileField:
    custom_profile_field = CustomProfileField(realm=realm, name=name, field_type=field_type, display_in_profile_summary=display_in_profile_summary, required=required, editable_by_user=editable_by_user)
    custom_profile_field.hint = hint
    if custom_profile_field.field_type in (CustomProfileField.SELECT, CustomProfileField.EXTERNAL_ACCOUNT):
        custom_profile_field.field_data = orjson.dumps(field_data or {}).decode()
    custom_profile_field.save()
    custom_profile_field.order = custom_profile_field.id
    custom_profile_field.save(update_fields=['order'])
    notify_realm_custom_profile_fields(realm)
    return custom_profile_field

@transaction.atomic(durable=True)
def do_remove_realm_custom_profile_field(realm: Realm, field: CustomProfileField) -> None:
    """
    Deleting a field will also delete the user profile data
    associated with it in CustomProfileFieldValue model.
    """
    field.delete()
    notify_realm_custom_profile_fields(realm)

def do_remove_realm_custom_profile_fields(realm: Realm) -> None:
    CustomProfileField.objects.filter(realm=realm).delete()

def remove_custom_profile_field_value_if_required(field: CustomProfileField, field_data: Dict[str, Any]) -> None:
    old_values: Set[str] = set(orjson.loads(field.field_data).keys())
    new_values: Set[str] = set(field_data.keys())
    removed_values = old_values - new_values
    if removed_values:
        CustomProfileFieldValue.objects.filter(field=field, value__in=removed_values).delete()

@transaction.atomic(durable=True)
def try_update_realm_custom_profile_field(realm: Realm, field: CustomProfileField, name: Optional[str] = None, hint: Optional[str] = None, field_data: Optional[Dict[str, Any]] = None, display_in_profile_summary: Optional[bool] = None, required: Optional[bool] = None, editable_by_user: Optional[bool] = None) -> None:
    if name is not None:
        field.name = name
    if hint is not None:
        field.hint = hint
    if required is not None:
        field.required = required
    if editable_by_user is not None:
        field.editable_by_user = editable_by_user
    if display_in_profile_summary is not None:
        field.display_in_profile_summary = display_in_profile_summary
    if field.field_type in (CustomProfileField.SELECT, CustomProfileField.EXTERNAL_ACCOUNT):
        if field_data is not None and field.field_type == CustomProfileField.SELECT:
            remove_custom_profile_field_value_if_required(field, field_data)
        if field_data is not None or field.field_data == '':
            field.field_data = orjson.dumps(field_data or {}).decode()
    field.save()
    notify_realm_custom_profile_fields(realm)

@transaction.atomic(durable=True)
def try_reorder_realm_custom_profile_fields(realm: Realm, order: List[int]) -> None:
    order_mapping: Dict[int, int] = {_[1]: _[0] for _ in enumerate(order)}
    custom_profile_fields = CustomProfileField.objects.filter(realm=realm)
    for custom_profile_field in custom_profile_fields:
        if custom_profile_field.id not in order_mapping:
            raise JsonableError(_('Invalid order mapping.'))
    for custom_profile_field in custom_profile_fields:
        custom_profile_field.order = order_mapping[custom_profile_field.id]
        custom_profile_field.save(update_fields=['order'])
    notify_realm_custom_profile_fields(realm)

def notify_user_update_custom_profile_data(user_profile: UserProfile, field: Dict[str, Any]) -> None:
    data: Dict[str, Any] = dict(id=field['id'], value=field['value'])
    if field['rendered_value']:
        data['rendered_value'] = field['rendered_value']
    payload: Dict[str, Any] = dict(user_id=user_profile.id, custom_profile_field=data)
    event: Dict[str, Any] = dict(type='realm_user', op='update', person=payload)
    send_event_on_commit(user_profile.realm, event, get_user_ids_who_can_access_user(user_profile))

@transaction.atomic(savepoint=False)
def do_update_user_custom_profile_data_if_changed(user_profile: UserProfile, data: List[ProfileDataElementUpdateDict]) -> None:
    for custom_profile_field in data:
        field_value, created = CustomProfileFieldValue.objects.get_or_create(user_profile=user_profile, field_id=custom_profile_field['id'])
        if isinstance(custom_profile_field['value'], str):
            custom_profile_field_value_string = custom_profile_field['value']
        else:
            custom_profile_field_value_string = orjson.dumps(custom_profile_field['value']).decode()
        if not created and field_value.value == custom_profile_field_value_string:
            continue
        field_value.value = custom_profile_field_value_string
        if field_value.field.is_renderable():
            field_value.rendered_value = render_stream_description(custom_profile_field_value_string, user_profile.realm)
            field_value.save(update_fields=['value', 'rendered_value'])
        else:
            field_value.save(update_fields=['value'])
        notify_user_update_custom_profile_data(user_profile, {'id': field_value.field_id, 'value': field_value.value, 'rendered_value': field_value.rendered_value, 'type': field_value.field.field_type})

@transaction.atomic(savepoint=False)
def check_remove_custom_profile_field_value(user_profile: UserProfile, field_id: int, acting_user: UserProfile) -> None:
    try:
        custom_profile_field = CustomProfileField.objects.get(realm=user_profile.realm, id=field_id)
        if not acting_user.is_realm_admin and (not custom_profile_field.editable_by_user):
            raise JsonableError(_('You are not allowed to change this field. Contact an administrator to update it.'))
        field_value = CustomProfileFieldValue.objects.get(field=custom_profile_field, user_profile=user_profile)
        field_value.delete()
        notify_user_update_custom_profile_data(user_profile, {'id': field_id, 'value': None, 'rendered_value': None, 'type': custom_profile_field.field_type})
    except CustomProfileField.DoesNotExist:
        raise JsonableError(_('Field id {id} not found.').format(id=field_id))
    except CustomProfileFieldValue.DoesNotExist:
        pass
