```python
import io
import logging
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import IO, Any
from django.core.files.uploadedfile import UploadedFile
from django.db.transaction import Atomic
from zerver.lib.exceptions import ErrorCode, JsonableError
from zerver.lib.upload.base import StreamingSourceWithSize, ZulipUploadBackend
from zerver.models import Attachment, Message, Realm, RealmEmoji, ScheduledMessage, UserProfile

class RealmUploadQuotaError(JsonableError):
    code: ErrorCode = ...

def check_upload_within_quota(realm: Realm, uploaded_file_size: int) -> None: ...

def create_attachment(
    file_name: str,
    path_id: str,
    content_type: str,
    file_data: bytes | UploadedFile,
    user_profile: UserProfile,
    realm: Realm,
) -> None: ...

def get_file_info(user_file: UploadedFile) -> tuple[str, str]: ...

upload_backend: ZulipUploadBackend = ...

def get_public_upload_root_url() -> str: ...

def sanitize_name(value: str, *, strict: bool = False) -> str: ...

def upload_message_attachment(
    uploaded_file_name: str,
    content_type: str,
    file_data: bytes | UploadedFile,
    user_profile: UserProfile,
    target_realm: Realm | None = None,
) -> tuple[str, str]: ...

def claim_attachment(
    path_id: str,
    message: Message | ScheduledMessage,
    is_message_realm_public: bool,
    is_message_web_public: bool = False,
) -> Attachment: ...

def upload_message_attachment_from_request(
    user_file: UploadedFile,
    user_profile: UserProfile,
) -> tuple[str, str]: ...

def attachment_vips_source(path_id: str) -> pyvips.Source: ...

def save_attachment_contents(path_id: str, filehandle: IO[bytes]) -> None: ...

def delete_message_attachment(path_id: str) -> bool: ...

def delete_message_attachments(path_ids: list[str]) -> None: ...

def all_message_attachments(
    *, include_thumbnails: bool = False, prefix: str = ''
) -> Iterator[tuple[str, datetime]]: ...

def get_avatar_url(hash_key: str, medium: bool = False) -> str: ...

def write_avatar_images(
    file_path: str,
    user_profile: UserProfile,
    image_data: bytes,
    *,
    content_type: str,
    backend: ZulipUploadBackend | None = None,
    future: bool = True,
) -> None: ...

def upload_avatar_image(
    user_file: UploadedFile,
    user_profile: UserProfile,
    content_type: str | None = None,
    backend: ZulipUploadBackend | None = None,
    future: bool = True,
) -> None: ...

def copy_avatar(source_profile: UserProfile, target_profile: UserProfile) -> None: ...

def ensure_avatar_image(user_profile: UserProfile, medium: bool = False) -> None: ...

def delete_avatar_image(user_profile: UserProfile, avatar_version: int) -> None: ...

def upload_icon_image(
    user_file: UploadedFile, user_profile: UserProfile, content_type: str
) -> None: ...

def upload_logo_image(
    user_file: UploadedFile,
    user_profile: UserProfile,
    night: bool,
    content_type: str,
) -> None: ...

def upload_emoji_image(
    emoji_file: UploadedFile,
    emoji_file_name: str,
    user_profile: UserProfile,
    content_type: str,
    backend: ZulipUploadBackend | None = None,
) -> bool: ...

def get_emoji_file_content(
    session: OutgoingSession,
    emoji_url: str,
    emoji_id: int,
    logger: logging.Logger,
) -> tuple[bytes, str]: ...

def handle_reupload_emojis_event(realm: Realm, logger: logging.Logger) -> None: ...

def upload_export_tarball(
    realm: Realm,
    tarball_path: str,
    percent_callback: Callable[[int], None] | None = None,
) -> str: ...

def delete_export_tarball(export_path: str) -> None: ...
```