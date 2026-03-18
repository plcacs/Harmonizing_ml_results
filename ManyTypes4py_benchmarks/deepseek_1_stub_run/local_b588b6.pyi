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

def assert_is_local_storage_path(type: Any, full_path: Any) -> None: ...

def write_local_file(type: Any, path: Any, file_data: Any) -> None: ...

def read_local_file(type: Any, path: Any) -> Iterator[bytes]: ...

def delete_local_file(type: Any, path: Any) -> bool: ...

class LocalUploadBackend(ZulipUploadBackend):
    @override
    def get_public_upload_root_url(self) -> str: ...
    
    @override
    def generate_message_upload_path(self, realm_id: Any, sanitized_file_name: Any) -> str: ...
    
    @override
    def upload_message_attachment(self, path_id: Any, filename: Any, content_type: Any, file_data: Any, user_profile: Any) -> None: ...
    
    @override
    def save_attachment_contents(self, path_id: Any, filehandle: Any) -> None: ...
    
    @override
    def attachment_vips_source(self, path_id: Any) -> StreamingSourceWithSize: ...
    
    @override
    def delete_message_attachment(self, path_id: Any) -> bool: ...
    
    @override
    def all_message_attachments(self, include_thumbnails: bool = ..., prefix: str = ...) -> Iterator[tuple[str, datetime]]: ...
    
    @override
    def get_avatar_url(self, hash_key: Any, medium: bool = ...) -> str: ...
    
    @override
    def get_avatar_contents(self, file_path: Any) -> tuple[bytes, str]: ...
    
    @override
    def upload_single_avatar_image(self, file_path: Any, *, user_profile: Any, image_data: Any, content_type: Any, future: bool = ...) -> None: ...
    
    @override
    def delete_avatar_image(self, path_id: Any) -> None: ...
    
    @override
    def get_realm_icon_url(self, realm_id: Any, version: Any) -> str: ...
    
    @override
    def upload_realm_icon_image(self, icon_file: Any, user_profile: Any, content_type: Any) -> None: ...
    
    @override
    def get_realm_logo_url(self, realm_id: Any, version: Any, night: Any) -> str: ...
    
    @override
    def upload_realm_logo_image(self, logo_file: Any, user_profile: Any, night: Any, content_type: Any) -> None: ...
    
    @override
    def get_emoji_url(self, emoji_file_name: Any, realm_id: Any, still: bool = ...) -> str: ...
    
    @override
    def upload_single_emoji_image(self, path: Any, content_type: Any, user_profile: Any, image_data: Any) -> None: ...
    
    @override
    def get_export_tarball_url(self, realm: Any, export_path: Any) -> str: ...
    
    @override
    def upload_export_tarball(self, realm: Any, tarball_path: Any, percent_callback: Any = ...) -> str: ...
    
    @override
    def delete_export_tarball(self, export_path: Any) -> str | None: ...
```