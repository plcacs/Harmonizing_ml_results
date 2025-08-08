from typing import Any, Protocol, TypeAlias, TypeVar
from zerver.models import Attachment, DirectMessageGroup, Message, Realm, RealmEmoji, Recipient, Stream, Subscription, UserProfile
from zerver.data_import.sequencer import NEXT_ID
from zerver.lib.avatar_hash import user_avatar_base_path_from_ids
from zerver.lib.message import normalize_body_for_import
from zerver.lib.mime_types import INLINE_MIME_TYPES, guess_extension
from zerver.lib.partial import partial
from zerver.lib.stream_color import STREAM_ASSIGNMENT_COLORS as STREAM_COLORS
from zproject.backends import all_default_backend_names

ZerverFieldsT = dict[str, Any]

class SubscriberHandler:
    stream_info: dict[int, list[int]]
    direct_message_group_info: dict[int, list[int]]

    def __init__(self) -> None:
        self.stream_info = {}
        self.direct_message_group_info = {}

    def set_info(self, users: list[int], stream_id: int = None, direct_message_group_id: int = None) -> None:
        if stream_id is not None:
            self.stream_info[stream_id] = users
        elif direct_message_group_id is not None:
            self.direct_message_group_info[direct_message_group_id] = users
        else:
            raise AssertionError('stream_id or direct_message_group_id is required')

    def get_users(self, stream_id: int = None, direct_message_group_id: int = None) -> list[int]:
        if stream_id is not None:
            return self.stream_info[stream_id]
        elif direct_message_group_id is not None:
            return self.direct_message_group_info[direct_message_group_id]
        else:
            raise AssertionError('stream_id or direct_message_group_id is required')

def build_zerver_realm(realm_id: int, realm_subdomain: str, time: Any, other_product: str) -> list[ZerverFieldsT]:
    ...

def build_user_profile(avatar_source: str, date_joined: Any, delivery_email: str, email: str, full_name: str, id: int, is_active: bool, role: int, is_mirror_dummy: bool, realm_id: int, short_name: str, timezone: str, is_bot: bool = False, bot_type: str = None) -> ZerverFieldsT:
    ...

def build_avatar(zulip_user_id: int, realm_id: int, email: str, avatar_url: str, timestamp: Any, avatar_list: list[ZerverFieldsT]) -> None:
    ...

def make_subscriber_map(zerver_subscription: list[ZerverFieldsT]) -> dict[int, set[int]]:
    ...

def make_user_messages(zerver_message: list[ZerverFieldsT], subscriber_map: dict[int, set[int]], is_pm_data: bool, mention_map: dict[int, set[int]], wildcard_mention_map: dict[int, bool] = {}) -> list[ZerverFieldsT]:
    ...

def build_subscription(recipient_id: int, user_id: int, subscription_id: int) -> ZerverFieldsT:
    ...

class GetUsers(Protocol):
    def __call__(self, stream_id: int = ..., direct_message_group_id: int = ...) -> None:
        ...

def build_stream_subscriptions(get_users: GetUsers, zerver_recipient: list[ZerverFieldsT], zerver_stream: list[ZerverFieldsT]) -> list[ZerverFieldsT]:
    ...

def build_direct_message_group_subscriptions(get_users: GetUsers, zerver_recipient: list[ZerverFieldsT], zerver_direct_message_group: list[ZerverFieldsT]) -> list[ZerverFieldsT]:
    ...

def build_personal_subscriptions(zerver_recipient: list[ZerverFieldsT]) -> list[ZerverFieldsT]:
    ...

def build_recipient(type_id: int, recipient_id: int, type: int) -> ZerverFieldsT:
    ...

def build_recipients(zerver_userprofile: list[ZerverFieldsT], zerver_stream: list[ZerverFieldsT], zerver_direct_message_group: list[ZerverFieldsT] = []) -> list[ZerverFieldsT]:
    ...

def build_realm(zerver_realm: ZerverFieldsT, realm_id: int, domain_name: str) -> ZerverFieldsT:
    ...

def build_usermessages(zerver_usermessage: list[ZerverFieldsT], subscriber_map: dict[int, set[int]], recipient_id: int, mentioned_user_ids: set[int], message_id: int, is_private: bool, long_term_idle: set[int] = set()) -> tuple[int, int]:
    ...

def build_user_message(user_id: int, message_id: int, is_private: bool, is_mentioned: bool, wildcard_mention: bool = False) -> ZerverFieldsT:
    ...

def build_defaultstream(realm_id: int, stream_id: int, defaultstream_id: int) -> ZerverFieldsT:
    ...

def build_stream(date_created: Any, realm_id: int, name: str, description: str, stream_id: int, deactivated: bool = False, invite_only: bool = False, stream_post_policy: int = 1) -> ZerverFieldsT:
    ...

def build_direct_message_group(direct_message_group_id: int, group_size: int) -> ZerverFieldsT:
    ...

def build_message(*, topic_name: str, date_sent: Any, message_id: int, content: str, rendered_content: str, user_id: int, recipient_id: int, realm_id: int, has_image: bool = False, has_link: bool = False, has_attachment: bool = True) -> ZerverFieldsT:
    ...

def build_attachment(realm_id: int, message_ids: list[int], user_id: int, fileinfo: dict[str, Any], s3_path: str, zerver_attachment: list[ZerverFieldsT]) -> None:
    ...

def get_avatar(avatar_dir: str, size_url_suffix: str, avatar_upload_item: tuple[str, str, str]) -> None:
    ...

def process_avatars(avatar_list: list[ZerverFieldsT], avatar_dir: str, realm_id: int, threads: int, size_url_suffix: str = '') -> list[ZerverFieldsT]:
    ...

def wrapping_function(f: Callable, item: Any) -> None:
    ...

def run_parallel_wrapper(f: Callable, full_items: ListJobData, threads: int = 6) -> None:
    ...

def get_uploads(upload_dir: str, upload: tuple[str, str]) -> None:
    ...

def process_uploads(upload_list: list[ZerverFieldsT], upload_dir: str, threads: int) -> list[ZerverFieldsT]:
    ...

def build_realm_emoji(realm_id: int, name: str, id: int, file_name: str) -> ZerverFieldsT:
    ...

def get_emojis(emoji_dir: str, emoji_url: str, emoji_path: str) -> str:
    ...

def process_emojis(zerver_realmemoji: list[ZerverFieldsT], emoji_dir: str, emoji_url_map: dict[str, str], threads: int) -> list[ZerverFieldsT]:
    ...

def create_converted_data_files(data: Any, output_dir: str, file_path: str) -> None:
    ...

def long_term_idle_helper(message_iterator: Iterable[Any], user_from_message: Callable[[Any], int], timestamp_from_message: Callable[[Any], Any], zulip_user_id_from_user: Callable[[int], int], all_user_ids_iterator: Iterable[int], zerver_userprofile: list[ZerverFieldsT]) -> set[int]:
    ...

def validate_user_emails_for_import(user_emails: list[str]) -> None:
    ...
