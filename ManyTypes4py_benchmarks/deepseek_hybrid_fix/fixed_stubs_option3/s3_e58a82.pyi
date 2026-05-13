import logging
import os
import secrets
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import IO, TYPE_CHECKING, Any, Literal, Optional, Union
from urllib.parse import SplitResult

import botocore
import pyvips
from botocore.client import Config
from botocore.response import StreamingBody
from django.conf import settings
from django.utils.http import content_disposition_header

from zerver.lib.mime_types import INLINE_MIME_TYPES
from zerver.lib.partial import partial
from zerver.lib.thumbnail import resize_logo, resize_realm_icon
from zerver.lib.upload.base import StreamingSourceWithSize, ZulipUploadBackend
from zerver.models import Realm, RealmEmoji, UserProfile

import io

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_s3.service_resource import Bucket, Object

SIGNED_UPLOAD_URL_DURATION: int = 60

def get_bucket(bucket_name: str, authed: bool = True) -> 'Bucket': ...
def upload_content_to_s3(
    bucket: 'Bucket',
    path: str,
    content_type: Optional[str],
    user_profile: Optional[UserProfile],
    contents: Union[bytes, io.IOBase],
    *,
    storage_class: str = 'STANDARD',
    cache_control: Optional[str] = None,
    extra_metadata: Optional[dict[str, str]] = None,
    filename: Optional[str] = None,
) -> None: ...

BOTO_CLIENT: Optional['S3Client'] = None

def get_boto_client() -> 'S3Client': ...
def get_signed_upload_url(path: str, filename: str, force_download: bool = False) -> str: ...

class S3UploadBackend(ZulipUploadBackend):
    avatar_bucket: 'Bucket'
    uploads_bucket: 'Bucket'
    export_bucket: Optional['Bucket']
    public_upload_url_base: str

    def __init__(self) -> None: ...
    def delete_file_from_s3(self, path_id: str, bucket: 'Bucket') -> bool: ...
    def construct_public_upload_url_base(self) -> str: ...
    @override
    def get_public_upload_root_url(self) -> str: ...
    def get_public_upload_url(self, key: str) -> str: ...
    @override
    def generate_message_upload_path(self, realm_id: str, sanitized_file_name: str) -> str: ...
    @override
    def upload_message_attachment(self, path_id: str, filename: str, content_type: str, file_data: Union[bytes, io.IOBase], user_profile: UserProfile) -> None: ...
    @override
    def save_attachment_contents(self, path_id: str, filehandle: io.IOBase) -> None: ...
    @override
    def attachment_vips_source(self, path_id: str) -> StreamingSourceWithSize: ...
    @override
    def delete_message_attachment(self, path_id: str) -> bool: ...
    @override
    def delete_message_attachments(self, path_ids: list[str]) -> None: ...
    @override
    def all_message_attachments(self, include_thumbnails: bool = False, prefix: str = '') -> Iterator[tuple[str, datetime]]: ...
    @override
    def get_avatar_url(self, hash_key: str, medium: bool = False) -> str: ...
    @override
    def get_avatar_contents(self, file_path: str) -> tuple[bytes, str]: ...
    @override
    def upload_single_avatar_image(self, file_path: str, *, user_profile: UserProfile, image_data: bytes, content_type: str, future: bool = True) -> None: ...
    @override
    def delete_avatar_image(self, path_id: str) -> None: ...
    @override
    def get_realm_icon_url(self, realm_id: int, version: int) -> str: ...
    @override
    def upload_realm_icon_image(self, icon_file: io.IOBase, user_profile: UserProfile, content_type: str) -> None: ...
    @override
    def get_realm_logo_url(self, realm_id: int, version: int, night: bool) -> str: ...
    @override
    def upload_realm_logo_image(self, logo_file: io.IOBase, user_profile: UserProfile, night: bool, content_type: str) -> None: ...
    @override
    def get_emoji_url(self, emoji_file_name: str, realm_id: int, still: bool = False) -> str: ...
    @override
    def upload_single_emoji_image(self, path: str, content_type: str, user_profile: UserProfile, image_data: bytes) -> None: ...
    @override
    def get_export_tarball_url(self, realm: Realm, export_path: str) -> str: ...
    def export_object(self, tarball_path: str) -> 'Object': ...
    @override
    def upload_export_tarball(self, realm: Realm, tarball_path: str, percent_callback: Optional[Callable[[int], None]] = None) -> str: ...
    @override
    def delete_export_tarball(self, export_path: str) -> Optional[str]: ...