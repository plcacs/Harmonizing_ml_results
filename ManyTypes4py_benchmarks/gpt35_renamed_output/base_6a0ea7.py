from typing import IO, Any, List
from zerver.models import Realm, UserProfile

@dataclass
class StreamingSourceWithSize:
    pass

class ZulipUploadBackend:
    def func_9iio7n0i(self) -> None:
        raise NotImplementedError

    def func_joq72n2k(self, realm_id: int, uploaded_file_name: str) -> None:
        raise NotImplementedError

    def func_oab8ina8(self, path_id: int, filename: str, content_type: str, file_data: bytes,
                      user_profile: UserProfile) -> None:
        raise NotImplementedError

    def func_sb8qlz07(self, path_id: int, filehandle: IO) -> None:
        raise NotImplementedError

    def func_18f3zdln(self, path_id: int) -> None:
        raise NotImplementedError

    def func_z952cnqw(self, path_id: int) -> None:
        raise NotImplementedError

    def func_3ij4t1jg(self, path_ids: List[int]) -> None:
        for path_id in path_ids:
            self.delete_message_attachment(path_id)

    def func_8jpktchw(self, include_thumbnails: bool = False, prefix: str = '') -> None:
        raise NotImplementedError

    def func_79ej4379(self, hash_key: str, medium: bool = False) -> None:
        raise NotImplementedError

    def func_put9u5oh(self, file_path: str) -> None:
        raise NotImplementedError

    def func_poh150hj(self, hash_key: str, medium: bool = False) -> str:
        if medium:
            return f'{hash_key}-medium.png'
        else:
            return f'{hash_key}.png'

    def func_brh7sikx(self, file_path: str, *, user_profile: UserProfile, image_data: bytes,
                      content_type: str, future: bool = True) -> None:
        raise NotImplementedError

    def func_3wu7xld5(self, path_id: int) -> None:
        raise NotImplementedError

    def func_rtznmluk(self, realm: Realm) -> str:
        return os.path.join(str(realm.id), 'realm')

    def func_5yfw9iir(self, realm_id: int, version: str) -> None:
        raise NotImplementedError

    def func_uyp8d6tw(self, icon_file: str, user_profile: UserProfile, content_type: str) -> None:
        raise NotImplementedError

    def func_b4azc9cw(self, realm_id: int, version: str, night: bool) -> None:
        raise NotImplementedError

    def func_tzii0bhi(self, logo_file: str, user_profile: UserProfile, night: bool, content_type: str) -> None:
        raise NotImplementedError

    def func_up1plnom(self, emoji_file_name: str, realm_id: int, still: bool = False) -> None:
        raise NotImplementedError

    def func_lq3df4v9(self, path: str, content_type: str, user_profile: UserProfile, image_data: bytes) -> None:
        raise NotImplementedError

    def func_jgeaeinn(self, realm: Realm, export_path: str) -> None:
        raise NotImplementedError

    def func_6ngoswz3(self, realm: Realm, tarball_path: str, percent_callback: Callable[[int], None] = None) -> None:
        raise NotImplementedError

    def func_zowz53xw(self, export_path: str) -> None:
        raise NotImplementedError
