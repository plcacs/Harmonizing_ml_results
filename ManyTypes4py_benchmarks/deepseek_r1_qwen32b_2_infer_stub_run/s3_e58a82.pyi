import datetime
import os
import typing
from collections.abc import Callable, Iterator
from typing import IO, Any, Optional, Union, List, Tuple, Dict, Literal, Optional, AnyStr, Iterator, Iterable, Generator, Optional, Any, TypeVar, overload
from urllib.parse import urljoin, urlsplit, urlunsplit
from botocore.client import Config
from botocore.response import StreamingBody
from django.conf import settings
from django.utils.http import content_disposition_header
from pyvips import SourceCustom
from zerver.lib.thumbnail import resize_logo, resize_realm_icon
from zerver.lib.upload.base import StreamingSourceWithSize, ZulipUploadBackend
from zerver.models import Realm, RealmEmoji, UserProfile
from mypy_boto3_s3.client import S3Client
from mypy_boto3_s3.service_resource import Bucket, Object

SIGNED_UPLOAD_URL_DURATION: int
BOTO_CLIENT: Optional[S3Client]

def get_bucket(bucket_name: str, authed: bool = True) -> Bucket:
    ...

def upload_content_to_s3(bucket: Bucket, path: str, content_type: Optional[str], user_profile: Optional[UserProfile], contents: Union[IO[bytes], bytes], *, storage_class: str = 'STANDARD', cache_control: Optional[str] = None, extra_metadata: Optional[Dict[str, str]] = None, filename: Optional[str] = None) -> None:
    ...

def get_boto_client() -> S3Client:
    ...

def get_signed_upload_url(path: str, filename: str, force_download: bool = False) -> str:
    ...

class S3UploadBackend(ZulipUploadBackend):
    def __init__(self) -> None:
        ...

    def delete_file_from_s3(self, path_id: str, bucket: Bucket) -> bool:
        ...

    def construct_public_upload_url_base(self) -> str:
        ...

    def get_public_upload_root_url(self) -> str:
        ...

    def get_public_upload_url(self, key: str) -> str:
        ...

    def generate_message_upload_path(self, realm_id: str, sanitized_file_name: str) -> str:
        ...

    def upload_message_attachment(self, path_id: str, filename: str, content_type: str, file_data: Union[IO[bytes], bytes], user_profile: UserProfile) -> None:
        ...

    def save_attachment_contents(self, path_id: str, filehandle: IO[bytes]) -> None:
        ...

    def attachment_vips_source(self, path_id: str) -> StreamingSourceWithSize:
        ...

    def delete_message_attachment(self, path_id: str) -> bool:
        ...

    def delete_message_attachments(self, path_ids: List[str]) -> None:
        ...

    def all_message_attachments(self, include_thumbnails: bool = False, prefix: str = '') -> Generator[Tuple[str, datetime.datetime], None, None]:
        ...

    def get_avatar_url(self, hash_key: str, medium: bool = False) -> str:
        ...

    def get_avatar_contents(self, file_path: str) -> Tuple[bytes, str]:
        ...

    def upload_single_avatar_image(self, file_path: str, *, user_profile: UserProfile, image_data: Union[IO[bytes], bytes], content_type: str, future: bool = True) -> None:
        ...

    def delete_avatar_image(self, path_id: str) -> None:
        ...

    def get_realm_icon_url(self, realm_id: str, version: int) -> str:
        ...

    def upload_realm_icon_image(self, icon_file: IO[bytes], user_profile: UserProfile, content_type: str) -> None:
        ...

    def get_realm_logo_url(self, realm_id: str, version: int, night: bool) -> str:
        ...

    def upload_realm_logo_image(self, logo_file: IO[bytes], user_profile: UserProfile, night: bool, content_type: str) -> None:
        ...

    def get_emoji_url(self, emoji_file_name: str, realm_id: str, still: bool = False) -> str:
        ...

    def upload_single_emoji_image(self, path: str, content_type: str, user_profile: UserProfile, image_data: Union[IO[bytes], bytes]) -> None:
        ...

    def get_export_tarball_url(self, realm: Realm, export_path: str) -> str:
        ...

    def export_object(self, tarball_path: str) -> Object:
        ...

    def upload_export_tarball(self, realm: Realm, tarball_path: str, percent_callback: Optional[Callable[[int], None]] = None) -> str:
        ...

    def delete_export_tarball(self, export_path: str) -> Optional[str]:
        ...