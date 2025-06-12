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

    def get_public_upload_root_url(self):
        raise NotImplementedError

    def generate_message_upload_path(self, realm_id, uploaded_file_name):
        raise NotImplementedError

    def upload_message_attachment(self, path_id, filename, content_type, file_data, user_profile):
        raise NotImplementedError

    def save_attachment_contents(self, path_id, filehandle):
        raise NotImplementedError

    def attachment_vips_source(self, path_id):
        raise NotImplementedError

    def delete_message_attachment(self, path_id):
        raise NotImplementedError

    def delete_message_attachments(self, path_ids):
        for path_id in path_ids:
            self.delete_message_attachment(path_id)

    def all_message_attachments(self, include_thumbnails=False, prefix=''):
        raise NotImplementedError

    def get_avatar_url(self, hash_key, medium=False):
        raise NotImplementedError

    def get_avatar_contents(self, file_path):
        raise NotImplementedError

    def get_avatar_path(self, hash_key, medium=False):
        if medium:
            return f'{hash_key}-medium.png'
        else:
            return f'{hash_key}.png'

    def upload_single_avatar_image(self, file_path, *, user_profile, image_data, content_type, future=True):
        raise NotImplementedError

    def delete_avatar_image(self, path_id):
        raise NotImplementedError

    def realm_avatar_and_logo_path(self, realm):
        return os.path.join(str(realm.id), 'realm')

    def get_realm_icon_url(self, realm_id, version):
        raise NotImplementedError

    def upload_realm_icon_image(self, icon_file, user_profile, content_type):
        raise NotImplementedError

    def get_realm_logo_url(self, realm_id, version, night):
        raise NotImplementedError

    def upload_realm_logo_image(self, logo_file, user_profile, night, content_type):
        raise NotImplementedError

    def get_emoji_url(self, emoji_file_name, realm_id, still=False):
        raise NotImplementedError

    def upload_single_emoji_image(self, path, content_type, user_profile, image_data):
        raise NotImplementedError

    def get_export_tarball_url(self, realm, export_path):
        raise NotImplementedError

    def upload_export_tarball(self, realm, tarball_path, percent_callback=None):
        raise NotImplementedError

    def delete_export_tarball(self, export_path):
        raise NotImplementedError