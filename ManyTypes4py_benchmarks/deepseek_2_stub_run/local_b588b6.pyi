```python
import os
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import IO, Any, Literal
from typing_extensions import override
from django.conf import settings
import pyvips
from zerver.lib.upload.base import StreamingSourceWithSize, ZulipUploadBackend
from zerver.models import Realm, RealmEmoji, UserProfile

def assert_is_local_storage_path(type: str, full_path: str) -> None: ...

def write_local_file(type: str, path: str, file_data: bytes) -> None: ...

def read_local_file(type: str, path: str) -> Iterator[bytes]: ...

def delete_local_file(type: str, path: str) -> bool: ...

class LocalUploadBackend(ZulipUploadBackend):
    @override
    def get_public_upload_root_url(self) -> str: ...
    
    @override
    def generate_message_upload_path(self, realm_id: str, sanitized_file_name: str) -> str: ...
    
    @override
    def upload_message_attachment(self, path_id: str, filename: str, content_type: str, file_data: bytes, user_profile: UserProfile) -> None: ...
    
    @override
    def save_attachment_contents(self, path_id: str, filehandle: IO[bytes]) -> None: ...
    
    @override
    def attachment_vips_source(self, path_id: str) -> StreamingSourceWithSize: ...
    
    @override
    def delete_message_attachment(self, path_id: str) -> bool: ...
    
    @override
    def all_message_attachments(self, include_thumbnails: bool = ..., prefix: str = ...) -> Iterator[tuple[str, datetime]]: ...
    
    @override
    def get_avatar_url(self, hash_key: str, medium: bool = ...) -> str: ...
    
    @override
    def get_avatar_contents(self, file_path: str) -> tuple[bytes, str]: ...
    
    @override
    def upload_single_avatar_image(self, file_path: str, *, user_profile: UserProfile, image_data: bytes, content_type: str, future: bool = ...) -> None: ...
    
    @override
    def delete_avatar_image(self, path_id: str) -> None: ...
    
    @override
    def get_realm_icon_url(self, realm_id: int, version: int) -> str: ...
    
    @override
    def upload_realm_icon_image(self, icon_file: IO[bytes], user_profile: UserProfile, content_type: str) -> None: ...
    
    @override
    def get_realm_logo_url(self, realm_id: int, version: int, night: bool) -> str: ...
    
    @override
    def upload_realm_logo_image(self, logo_file: IO[bytes], user_profile: UserProfile, night: bool, content_type: str) -> None: ...
    
    @override
    def get_emoji_url(self, emoji_file_name: str, realm_id: int, still: bool = ...) -> str: ...
    
    @override
    def upload_single_emoji_image(self, path: str, content_type: str, user_profile: UserProfile, image_data: bytes) -> None: ...
    
    @override
    def get_export_tarball_url(self, realm: Realm, export_path: str) -> str: ...
    
    @override
    def upload_export_tarball(self, realm: Realm, tarball_path: str, percent_callback: Callable[[int], None] | None = ...) -> str: ...
    
    @override
    def delete_export_tarball(self, export_path: str) -> str | None: ...
```