def do_set_realm_property(realm: Realm, name: str, value: Any, *, acting_user: UserProfile) -> None:
def do_set_push_notifications_enabled_end_timestamp(realm: Realm, value: Any, *, acting_user: UserProfile) -> None:
def do_change_realm_permission_group_setting(realm: Realm, setting_name: str, user_group: UserGroup, old_setting_api_value: AnonymousSettingGroupDict = None, *, acting_user: UserProfile) -> None:
def parse_and_set_setting_value_if_required(realm: Realm, setting_name: str, value: Any, *, acting_user: UserProfile) -> Tuple[Any, bool]:
def get_realm_authentication_methods_for_page_params_api(realm: Realm, authentication_methods: dict) -> dict:
def validate_authentication_methods_dict_from_api(realm: Realm, authentication_methods: dict) -> None:
def validate_plan_for_authentication_methods(realm: Realm, authentication_methods: dict) -> None:
def do_set_realm_authentication_methods(realm: Realm, authentication_methods: dict, *, acting_user: UserProfile) -> None:
def do_set_realm_stream(realm: Realm, field: str, stream: Stream, stream_id: int, *, acting_user: UserProfile) -> None:
def do_set_realm_moderation_request_channel(realm: Realm, stream: Stream, stream_id: int, *, acting_user: UserProfile) -> None:
def do_set_realm_new_stream_announcements_stream(realm: Realm, stream: Stream, stream_id: int, *, acting_user: UserProfile) -> None:
def do_set_realm_signup_announcements_stream(realm: Realm, stream: Stream, stream_id: int, *, acting_user: UserProfile) -> None:
def do_set_realm_zulip_update_announcements_stream(realm: Realm, stream: Stream, stream_id: int, *, acting_user: UserProfile) -> None:
def do_set_realm_user_default_setting(realm_user_default: RealmUserDefault, name: str, value: Any, *, acting_user: UserProfile) -> None:
def do_deactivate_realm(realm: Realm, *, acting_user: UserProfile, deactivation_reason: RealmDeactivationReasonType, deletion_delay_days: Optional[int] = None, email_owners: bool) -> None:
def do_reactivate_realm(realm: Realm) -> None:
def do_add_deactivated_redirect(realm: Realm, redirect_url: str) -> None:
def do_delete_all_realm_attachments(realm: Realm, *, batch_size: int = 1000) -> None:
def do_scrub_realm(realm: Realm, *, acting_user: UserProfile) -> None:
def scrub_deactivated_realm(realm_to_scrub: Realm) -> None:
def clean_deactivated_realm_data() -> None:
def do_change_realm_org_type(realm: Realm, org_type: str, acting_user: UserProfile) -> None:
def do_change_realm_max_invites(realm: Realm, max_invites: int, acting_user: UserProfile) -> None:
def do_change_realm_plan_type(realm: Realm, plan_type: str, *, acting_user: UserProfile) -> None:
def do_send_realm_reactivation_email(realm: Realm, *, acting_user: UserProfile) -> None:
def do_send_realm_deactivation_email(realm: Realm, acting_user: UserProfile, deletion_delay_days: Optional[int]) -> None:
