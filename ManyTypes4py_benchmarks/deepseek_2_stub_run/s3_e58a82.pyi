```python
import io
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import IO, TYPE_CHECKING, Any, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit
import botocore
import pyvips
from botocore.client import Config
from botocore.response import StreamingBody
from django.conf import settings
from typing_extensions import override
from zerver.lib.upload.base import StreamingSourceWithSize, ZulipUploadBackend
from zerver.models import Realm, RealmEmoji, UserProfile

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_s3.service_resource import Bucket, Object

SIGNED_UPLOAD_URL_DURATION: int = ...

def get_bucket(bucket_name: str, authed: bool = ...) -> Any: ...

def upload_content_to_s3(
    bucket: Any,
    path: str,
    content_type: str | None,
    user_profile: UserProfile | None,
    contents: Any,
    *,
    storage_class: str = ...,
    cache_control: str | None = ...,
    extra_metadata: dict[str, str] | None = ...,
    filename: str | None = ...
) -> None: ...

BOTO_CLIENT: Any = ...

def get_boto_client() -> Any: ...

def get_signed_upload_url(path: str, filename: str, force_download: bool = ...) -> str: ...

class S3UploadBackend(ZulipUploadBackend):
    avatar_bucket: Any = ...
    uploads_bucket: Any = ...
    export_bucket: Any | None = ...
    public_upload_url_base: str = ...

    def __init__(self) -> None: ...
    def delete_file_from_s3(self, path_id: str, bucket: Any) -> bool: ...
    def construct_public_upload_url_base(self) -> str: ...
    @override
    def get_public_upload_root_url(self) -> str: ...
    def get_public_upload_url(self, key: str) -> str: ...
    @override
    def generate_message_upload_path(self, realm_id: str, sanitized_file_name: str) -> str: ...
    @override
    def upload_message_attachment(
        self,
        path_id: str,
        filename: str,
        content_type: str,
        file_data: Any,
        user_profile: UserProfile
    ) -> None: ...
    @override
    def save_attachment_contents(self, path_id: str, filehandle: IO[bytes]) -> None: ...
    @override
    def attachment_vips_source(self, path_id: str) -> StreamingSourceWithSize: ...
    @override
    def delete_message_attachment(self, path_id: str) -> bool: ...
    @override
    def delete_message_attachments(self, path_ids: list[str]) -> None: ...
    @override
    def all_message_attachments(
        self,
        include_thumbnails: bool = ...,
        prefix: str = ...
    ) -> Iterator[tuple[str, datetime]]: ...
    @override
    def get_avatar_url(self, hash_key: str, medium: bool = ...) -> str: ...
    @override
    def get_avatar_contents(self, file_path: str) -> tuple[bytes, str]: ...
    @override
    def upload_single_avatar_image(
        self,
        file_path: str,
        *,
        user_profile: UserProfile,
        image_data: bytes,
        content_type: str,
        future: bool = ...
    ) -> None: ...
    @override
    def delete_avatar_image(self, path_id: str) -> None: ...
    @override
    def get_realm_icon_url(self, realm_id: int, version: int) -> str: ...
    @override
    def upload_realm_icon_image(
        self,
        icon_file: IO[bytes],
        user_profile: UserProfile,
        content_type: str
    ) -> None: ...
    @override
    def get_realm_logo_url(self, realm_id: int, version: int, night: bool) -> str: ...
    @override
    def upload_realm_logo_image(
        self,
        logo_file: IO[bytes],
        user_profile: UserProfile,
        night: bool,
        content_type: str
    ) -> None: ...
    @override
    def get_emoji_url(self, emoji_file_name: str, realm_id: int, still: bool = ...) -> str: ...
    @override
    def upload_single_emoji_image(
        self,
        path: str,
        content_type: str,
        user_profile: UserProfile,
        image_data: bytes
    ) -> None: ...
    @override
    def get_export_tarball_url(self, realm: Realm, export_path: str) -> str: ...
    def export_object(self, tarball_path: str) -> Any: ...
    @override
    def upload_export_tarball(
        self,
        realm: Realm,
        tarball_path: str,
        percent_callback: Callable[[int], None] | None = ...
    ) -> str: ...
    @override
    def delete_export_tarball(self, export_path: str) -> str | None: ...
```