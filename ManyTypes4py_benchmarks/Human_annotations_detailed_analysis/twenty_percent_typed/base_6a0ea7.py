import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import IO, Any

import pyvips

from zerver.models import Realm, UserProfile


@dataclass
class StreamingSourceWithSize:
    size: int
    source: pyvips.Source


class ZulipUploadBackend:
    # Message attachment uploads
    def get_public_upload_root_url(self) -> str:
        raise NotImplementedError

    def generate_message_upload_path(self, realm_id, uploaded_file_name) -> str:
        raise NotImplementedError

    def upload_message_attachment(
        self,
        path_id,
        filename,
        content_type,
        file_data,
        user_profile,
    ) -> None:
        raise NotImplementedError

    def save_attachment_contents(self, path_id, filehandle: IO[bytes]) -> None:
        raise NotImplementedError

    def attachment_vips_source(self, path_id: str) -> StreamingSourceWithSize:
        raise NotImplementedError

    def delete_message_attachment(self, path_id) -> bool:
        raise NotImplementedError

    def delete_message_attachments(self, path_ids) -> None:
        for path_id in path_ids:
            self.delete_message_attachment(path_id)

    def all_message_attachments(
        self,
        include_thumbnails = False,
        prefix = "",
    ) -> Iterator[tuple[str, datetime]]:
        raise NotImplementedError

    # Avatar image uploads
    def get_avatar_url(self, hash_key: str, medium = False) -> str:
        raise NotImplementedError

    def get_avatar_contents(self, file_path: str) -> tuple[bytes, str]:
        raise NotImplementedError

    def get_avatar_path(self, hash_key: str, medium = False) -> str:
        if medium:
            return f"{hash_key}-medium.png"
        else:
            return f"{hash_key}.png"

    def upload_single_avatar_image(
        self,
        file_path,
        *,
        user_profile: UserProfile,
        image_data: bytes,
        content_type,
        future = True,
    ) -> None:
        raise NotImplementedError

    def delete_avatar_image(self, path_id) -> None:
        raise NotImplementedError

    # Realm icon and logo uploads
    def realm_avatar_and_logo_path(self, realm) -> str:
        return os.path.join(str(realm.id), "realm")

    def get_realm_icon_url(self, realm_id, version) -> str:
        raise NotImplementedError

    def upload_realm_icon_image(
        self, icon_file, user_profile, content_type
    ) -> None:
        raise NotImplementedError

    def get_realm_logo_url(self, realm_id, version, night) -> str:
        raise NotImplementedError

    def upload_realm_logo_image(
        self, logo_file, user_profile, night, content_type
    ) -> None:
        raise NotImplementedError

    # Realm emoji uploads
    def get_emoji_url(self, emoji_file_name: str, realm_id, still = False) -> str:
        raise NotImplementedError

    def upload_single_emoji_image(
        self,
        path: str,
        content_type,
        user_profile: UserProfile,
        image_data,
    ) -> None:
        raise NotImplementedError

    # Export tarballs
    def get_export_tarball_url(self, realm, export_path) -> str:
        raise NotImplementedError

    def upload_export_tarball(
        self,
        realm,
        tarball_path,
        percent_callback = None,
    ) -> str:
        raise NotImplementedError

    def delete_export_tarball(self, export_path) -> str | None:
        raise NotImplementedError
