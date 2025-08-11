import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import IO, Any
import pyvips
from zerver.models import Realm, UserProfile

@dataclass
class StreamingSourceWithSize:
    pass

class ZulipUploadBackend:

    def get_public_upload_root_url(self) -> None:
        raise NotImplementedError

    def generate_message_upload_path(self, realm_id: Union[str, int, None], uploaded_file_name: Union[str, int, None]) -> None:
        raise NotImplementedError

    def upload_message_attachment(self, path_id: Union[zerver.data_imporsequencer.IdMapper, int, typing.AbstractSet], filename: Union[zerver.data_imporsequencer.IdMapper, int, typing.AbstractSet], content_type: Union[zerver.data_imporsequencer.IdMapper, int, typing.AbstractSet], file_data: Union[zerver.data_imporsequencer.IdMapper, int, typing.AbstractSet], user_profile: Union[zerver.data_imporsequencer.IdMapper, int, typing.AbstractSet]) -> None:
        raise NotImplementedError

    def save_attachment_contents(self, path_id: Union[str, bool, list[str]], filehandle: Union[str, bool, list[str]]) -> None:
        raise NotImplementedError

    def attachment_vips_source(self, path_id: Union[str, bool, list[str]]) -> None:
        raise NotImplementedError

    def delete_message_attachment(self, path_id: Union[str, bool, list[str]]) -> None:
        raise NotImplementedError

    def delete_message_attachments(self, path_ids: Union[str, list[str]]) -> None:
        for path_id in path_ids:
            self.delete_message_attachment(path_id)

    def all_message_attachments(self, include_thumbnails: bool=False, prefix: typing.Text='') -> None:
        raise NotImplementedError

    def get_avatar_url(self, hash_key: Union[bool, str], medium: bool=False) -> None:
        raise NotImplementedError

    def get_avatar_contents(self, file_path: Union[str, list[str]]) -> None:
        raise NotImplementedError

    def get_avatar_path(self, hash_key: Union[str, bool], medium: bool=False) -> typing.Text:
        if medium:
            return f'{hash_key}-medium.png'
        else:
            return f'{hash_key}.png'

    def upload_single_avatar_image(self, file_path: Union[bool, str, None], *, user_profile: Union[bool, str, None], image_data: Union[bool, str, None], content_type: Union[bool, str, None], future: bool=True) -> None:
        raise NotImplementedError

    def delete_avatar_image(self, path_id: Union[str, int]) -> None:
        raise NotImplementedError

    def realm_avatar_and_logo_path(self, realm: Union[zerver.models.Realm, str, models.Profile]) -> str:
        return os.path.join(str(realm.id), 'realm')

    def get_realm_icon_url(self, realm_id: int, version: int) -> None:
        raise NotImplementedError

    def upload_realm_icon_image(self, icon_file: Union[str, zerver.models.Client, zerver.models.UserProfile], user_profile: Union[str, zerver.models.Client, zerver.models.UserProfile], content_type: Union[str, zerver.models.Client, zerver.models.UserProfile]) -> None:
        raise NotImplementedError

    def get_realm_logo_url(self, realm_id: bool, version: bool, night: bool) -> None:
        raise NotImplementedError

    def upload_realm_logo_image(self, logo_file: Union[bool, zerver.models.Realm, str], user_profile: Union[bool, zerver.models.Realm, str], night: Union[bool, zerver.models.Realm, str], content_type: Union[bool, zerver.models.Realm, str]) -> None:
        raise NotImplementedError

    def get_emoji_url(self, emoji_file_name: Union[int, str, None], realm_id: Union[int, str, None], still: bool=False) -> None:
        raise NotImplementedError

    def upload_single_emoji_image(self, path: Union[str, None, zerver.models.Realm], content_type: Union[str, None, zerver.models.Realm], user_profile: Union[str, None, zerver.models.Realm], image_data: Union[str, None, zerver.models.Realm]) -> None:
        raise NotImplementedError

    def get_export_tarball_url(self, realm: Union[zerver.models.Realm, str, zerver.models.UserProfile], export_path: Union[zerver.models.Realm, str, zerver.models.UserProfile]) -> None:
        raise NotImplementedError

    def upload_export_tarball(self, realm: Union[typing.Callable[typing.Any, None], None, str, zerver.models.Realm], tarball_path: Union[typing.Callable[typing.Any, None], None, str, zerver.models.Realm], percent_callback: Union[None, typing.Callable[typing.Any, None], str, zerver.models.Realm]=None) -> None:
        raise NotImplementedError

    def delete_export_tarball(self, export_path: Union[str, list[str], typing.Collection, None]) -> None:
        raise NotImplementedError