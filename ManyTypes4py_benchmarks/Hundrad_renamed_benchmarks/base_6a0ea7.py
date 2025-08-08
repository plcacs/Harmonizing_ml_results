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

    def func_9iio7n0i(self):
        raise NotImplementedError

    def func_joq72n2k(self, realm_id, uploaded_file_name):
        raise NotImplementedError

    def func_oab8ina8(self, path_id, filename, content_type, file_data,
        user_profile):
        raise NotImplementedError

    def func_sb8qlz07(self, path_id, filehandle):
        raise NotImplementedError

    def func_18f3zdln(self, path_id):
        raise NotImplementedError

    def func_z952cnqw(self, path_id):
        raise NotImplementedError

    def func_3ij4t1jg(self, path_ids):
        for path_id in path_ids:
            self.delete_message_attachment(path_id)

    def func_8jpktchw(self, include_thumbnails=False, prefix=''):
        raise NotImplementedError

    def func_79ej4379(self, hash_key, medium=False):
        raise NotImplementedError

    def func_put9u5oh(self, file_path):
        raise NotImplementedError

    def func_poh150hj(self, hash_key, medium=False):
        if medium:
            return f'{hash_key}-medium.png'
        else:
            return f'{hash_key}.png'

    def func_brh7sikx(self, file_path, *, user_profile, image_data,
        content_type, future=True):
        raise NotImplementedError

    def func_3wu7xld5(self, path_id):
        raise NotImplementedError

    def func_rtznmluk(self, realm):
        return os.path.join(str(realm.id), 'realm')

    def func_5yfw9iir(self, realm_id, version):
        raise NotImplementedError

    def func_uyp8d6tw(self, icon_file, user_profile, content_type):
        raise NotImplementedError

    def func_b4azc9cw(self, realm_id, version, night):
        raise NotImplementedError

    def func_tzii0bhi(self, logo_file, user_profile, night, content_type):
        raise NotImplementedError

    def func_up1plnom(self, emoji_file_name, realm_id, still=False):
        raise NotImplementedError

    def func_lq3df4v9(self, path, content_type, user_profile, image_data):
        raise NotImplementedError

    def func_jgeaeinn(self, realm, export_path):
        raise NotImplementedError

    def func_6ngoswz3(self, realm, tarball_path, percent_callback=None):
        raise NotImplementedError

    def func_zowz53xw(self, export_path):
        raise NotImplementedError
