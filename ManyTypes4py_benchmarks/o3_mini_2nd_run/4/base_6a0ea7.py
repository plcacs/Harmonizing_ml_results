import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import IO, Any, Optional
import pyvips
from zerver.models import Realm, UserProfile

@dataclass
class StreamingSourceWithSize:
    # You can add the specific fields as required
    pass

class ZulipUploadBackend:

    def get_public_upload_root_url(self) -> str:
        raise NotImplementedError

    def generate_message_upload_path(self, realm_id: int, uploaded_file_name: str) -> str:
        raise NotImplementedError

    def upload_message_attachment(
        self,
        path_id: str,
        filename: str,
        content_type: str,
        file_data: IO[Any],
        user_profile: UserProfile,
    ) -> Any:
        raise NotImplementedError

    def save_attachment_contents(self, path_id: str, filehandle: IO[Any]) -> None:
        raise NotImplementedError

    def attachment_vips_source(self, path_id: str) -> StreamingSourceWithSize:
        raise NotImplementedError

    def delete_message_attachment(self, path_id: str) -> None:
        raise NotImplementedError

    def delete_message_attachments(self, path_ids: Iterator[str]) -> None:
        for path_id in path_ids:
            self.delete_message_attachment(path_id)

    def all_message_attachments(self, include_thumbnails: bool = False, prefix: str = '') -> Iterator[Any]:
        raise NotImplementedError

    def get_avatar_url(self, hash_key: str, medium: bool = False) -> str:
        raise NotImplementedError

    def get_avatar_contents(self, file_path: str) -> bytes:
        raise NotImplementedError

    def get_avatar_path(self, hash_key: str, medium: bool = False) -> str:
        if medium:
            return f'{hash_key}-medium.png'
        else:
            return f'{hash_key}.png'

    def upload_single_avatar_image(
        self,
        file_path: str,
        *,
        user_profile: UserProfile,
        image_data: bytes,
        content_type: str,
        future: bool = True
    ) -> None:
        raise NotImplementedError

    def delete_avatar_image(self, path_id: str) -> None:
        raise NotImplementedError

    def realm_avatar_and_logo_path(self, realm: Realm) -> str:
        return os.path.join(str(realm.id), 'realm')

    def get_realm_icon_url(self, realm_id: int, version: str) -> str:
        raise NotImplementedError

    def upload_realm_icon_image(
        self,
        icon_file: IO[Any],
        user_profile: UserProfile,
        content_type: str,
    ) -> None:
        raise NotImplementedError

    def get_realm_logo_url(self, realm_id: int, version: str, night: bool) -> str:
        raise NotImplementedError

    def upload_realm_logo_image(
        self,
        logo_file: IO[Any],
        user_profile: UserProfile,
        night: bool,
        content_type: str,
    ) -> None:
        raise NotImplementedError

    def get_emoji_url(self, emoji_file_name: str, realm_id: int, still: bool = False) -> str:
        raise NotImplementedError

    def upload_single_emoji_image(
        self,
        path: str,
        content_type: str,
        user_profile: UserProfile,
        image_data: bytes,
    ) -> None:
        raise NotImplementedError

    def get_export_tarball_url(self, realm: Realm, export_path: str) -> str:
        raise NotImplementedError

    def upload_export_tarball(
        self,
        realm: Realm,
        tarball_path: str,
        percent_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        raise NotImplementedError

    def delete_export_tarball(self, export_path: str) -> None:
        raise NotImplementedError