from typing import Union, List, Dict, Any

def b64_to_hex(data: str) -> str:
    return base64.b64decode(data).hex()

def hex_to_b64(data: str) -> str:
    return base64.b64encode(bytes.fromhex(data)).decode()

def get_message_stream_name_from_database(message: Message) -> str:
    stream_id = message.recipient.type_id
    return Stream.objects.get(id=stream_id).name

def has_apns_credentials() -> bool:
    return settings.APNS_TOKEN_KEY_FILE is not None or settings.APNS_CERT_FILE is not None

def get_apns_context() -> Optional[APNsContext]:
    ...

def modernize_apns_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    ...

def send_apple_push_notification(user_identity: UserPushIdentityCompat, devices: List[DeviceToken], payload_data: Dict[str, Any], remote: Optional[bool] = None) -> int:
    ...

def make_fcm_app() -> Optional[FCMApp]:
    ...

def has_fcm_credentials() -> bool:
    ...

def send_android_push_notification_to_user(user_profile: UserProfile, data: Dict[str, Any], options: Dict[str, Any]) -> None:
    ...

def parse_fcm_options(options: Dict[str, Any], data: Dict[str, Any]) -> str:
    ...

def send_android_push_notification(user_identity: UserPushIdentityCompat, devices: List[DeviceToken], data: Dict[str, Any], options: Dict[str, Any], remote: Optional[bool] = None) -> int:
    ...

def uses_notification_bouncer() -> bool:
    ...

def sends_notifications_directly() -> bool:
    ...

def send_notifications_to_bouncer(user_profile: UserProfile, apns_payload: Dict[str, Any], gcm_payload: Dict[str, Any], gcm_options: Dict[str, Any], android_devices: List[DeviceToken], apple_devices: List[DeviceToken]) -> None:
    ...

def add_push_device_token(user_profile: UserProfile, token_str: str, kind: int, ios_app_id: Optional[str] = None) -> None:
    ...

def remove_push_device_token(user_profile: UserProfile, token_str: str, kind: int) -> None:
    ...

def clear_push_device_tokens(user_profile_id: int) -> None:
    ...

def push_notifications_configured() -> bool:
    ...

def initialize_push_notifications() -> None:
    ...

def get_mobile_push_content(rendered_content: str) -> str:
    ...

def truncate_content(content: str) -> Tuple[str, bool]:
    ...

def get_base_payload(user_profile: UserProfile) -> Dict[str, Any]:
    ...

def get_message_payload(user_profile: UserProfile, message: Message, mentioned_user_group_id: Optional[int] = None, mentioned_user_group_name: Optional[str] = None, can_access_sender: bool = True) -> Dict[str, Any]:
    ...

def get_apns_alert_title(message: Message, language: str) -> str:
    ...

def get_apns_alert_subtitle(message: Message, trigger: str, user_profile: UserProfile, mentioned_user_group_name: Optional[str] = None, can_access_sender: bool = True) -> str:
    ...

def get_apns_badge_count(user_profile: UserProfile, read_messages_ids: List[int] = []) -> int:
    ...

def get_apns_badge_count_future(user_profile: UserProfile, read_messages_ids: List[int] = []) -> int:
    ...

def get_message_payload_apns(user_profile: UserProfile, message: Message, trigger: str, mentioned_user_group_id: Optional[int] = None, mentioned_user_group_name: Optional[str] = None, can_access_sender: bool = True) -> Dict[str, Any]:
    ...

def get_message_payload_gcm(user_profile: UserProfile, message: Message, mentioned_user_group_id: Optional[int] = None, mentioned_user_group_name: Optional[str] = None, can_access_sender: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ...

def get_remove_payload_gcm(user_profile: UserProfile, message_ids: List[int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ...

def get_remove_payload_apns(user_profile: UserProfile, message_ids: List[int]) -> Dict[str, Any]:
    ...

def handle_remove_push_notification(user_profile_id: int, message_ids: List[int]) -> None:
    ...

def handle_push_notification(user_profile_id: int, missed_message: Dict[str, Any]) -> None:
    ...

def send_test_push_notification_directly_to_devices(user_identity: UserPushIdentityCompat, devices: List[DeviceToken], base_payload: Dict[str, Any], remote: Optional[bool] = None) -> None:
    ...

def send_test_push_notification(user_profile: UserProfile, devices: List[DeviceToken]) -> None:
    ...

class InvalidPushDeviceTokenError(JsonableError):
    ...

class InvalidRemotePushDeviceTokenError(JsonableError):
    ...

class PushNotificationsDisallowedByBouncerError(Exception):
    ...

class HostnameAlreadyInUseBouncerError(JsonableError):
    ...
