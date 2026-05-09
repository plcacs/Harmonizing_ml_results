import io
import logging
import os
import re
import unicodedata
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import (
    Any,
    AnyStr,
    IO,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    Dict,
    Set,
    FrozenSet,
    TypeVar,
    overload,
)
from urllib.parse import unquote, urljoin
import pyvips
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.db import transaction
from django.utils.translation import gettext as _
from zerver.lib.avatar_hash import user_avatar_base_path_from_ids, user_avatar_path
from zerver.lib.exceptions import ErrorCode, JsonableError
from zerver.lib.mime_types import INLINE_MIME_TYPES, guess_type
from zerver.lib.outgoing_http import OutgoingSession
from zerver.lib.thumbnail import (
    MAX_EMOJI_GIF_FILE_SIZE_BYTES,
    MEDIUM_AVATAR_SIZE,
    THUMBNAIL_ACCEPT_IMAGE_TYPES,
    BadImageError,
    maybe_thumbnail,
    resize_avatar,
    resize_emoji,
)
from zerver.lib.upload.base import StreamingSourceWithSize, ZulipUploadBackend
from zerver.models import (
    Attachment,
    Message,
    Realm,
    RealmEmoji,
    ScheduledMessage,
    UserProfile,
)
from zerver.models.users import is_cross_realm_bot_email

class RealmUploadQuotaError(JsonableError):
    code: ErrorCode.REALM_UPLOAD_QUOTA

def check_upload_within_quota(realm: Realm, uploaded_file_size: int) -> None:
    ...

def create_attachment(
    file_name: str,
    path_id: str,
    content_type: str,
    file_data: Union[bytes, StreamingSourceWithSize],
    user_profile: UserProfile,
    realm: Realm,
) -> None:
    ...

def get_file_info(user_file: UploadedFile) -> Tuple[str, str]:
    ...

upload_backend: Any = ...

def get_public_upload_root_url() -> str:
    ...

def sanitize_name(value: str, *, strict: bool = False) -> str:
    ...

def upload_message_attachment(
    uploaded_file_name: str,
    content_type: str,
    file_data: Union[bytes, StreamingSourceWithSize],
    user_profile: UserProfile,
    target_realm: Optional[Realm] = None,
) -> Tuple[str, str]:
    ...

def claim_attachment(
    path_id: str,
    message: Union[Message, ScheduledMessage],
    is_message_realm_public: bool,
    is_message_web_public: bool = False,
) -> Attachment:
    ...

def upload_message_attachment_from_request(
    user_file: UploadedFile,
    user_profile: UserProfile,
) -> Tuple[str, str]:
    ...

def attachment_vips_source(path_id: str) -> Optional[pyvips.Image]:
    ...

def save_attachment_contents(path_id: str, filehandle: IO[Any]) -> None:
    ...

def delete_message_attachment(path_id: str) -> None:
    ...

def delete_message_attachments(path_ids: Iterable[str]) -> None:
    ...

def all_message_attachments(
    *, include_thumbnails: bool = False, prefix: str = ''
) -> Iterator[str]:
    ...

def get_avatar_url(hash_key: str, medium: bool = False) -> str:
    ...

def write_avatar_images(
    file_path: str,
    user_profile: UserProfile,
    image_data: bytes,
    content_type: str,
    backend: Optional[ZulipUploadBackend] = None,
    future: bool = True,
) -> None:
    ...

def upload_avatar_image(
    user_file: UploadedFile,
    user_profile: UserProfile,
    content_type: Optional[str] = None,
    backend: Optional[ZulipUploadBackend] = None,
    future: bool = True,
) -> None:
    ...

def copy_avatar(source_profile: UserProfile, target_profile: UserProfile) -> None:
    ...

def ensure_avatar_image(user_profile: UserProfile, medium: bool = False) -> None:
    ...

def delete_avatar_image(user_profile: UserProfile, avatar_version: int) -> None:
    ...

def upload_icon_image(
    user_file: UploadedFile,
    user_profile: UserProfile,
    content_type: str,
) -> None:
    ...

def upload_logo_image(
    user_file: UploadedFile,
    user_profile: UserProfile,
    night: bool,
    content_type: str,
) -> None:
    ...

def upload_emoji_image(
    emoji_file: IO[Any],
    emoji_file_name: str,
    user_profile: UserProfile,
    content_type: str,
    backend: Optional[ZulipUploadBackend] = None,
) -> bool:
    ...

def get_emoji_file_content(
    session: OutgoingSession,
    emoji_url: str,
    emoji_id: int,
    logger: logging.Logger,
) -> Tuple[bytes, str]:
    ...

def handle_reupload_emojis_event(realm: Realm, logger: logging.Logger) -> None:
    ...

def upload_export_tarball(
    realm: Realm,
    tarball_path: str,
    percent_callback: Optional[Callable[[float], None]] = None,
) -> None:
    ...

def delete_export_tarball(export_path: str) -> None:
    ...