def assert_is_local_storage_path(type: str, full_path: str) -> None:

def write_local_file(type: str, path: str, file_data: bytes) -> None:

def read_local_file(type: str, path: str) -> Iterator[bytes]:

def delete_local_file(type: str, path: str) -> bool:

class LocalUploadBackend(ZulipUploadBackend):

    def get_public_upload_root_url(self) -> str:

    def generate_message_upload_path(self, realm_id: str, sanitized_file_name: str) -> str:

    def upload_message_attachment(self, path_id: str, filename: str, content_type: str, file_data: bytes, user_profile: UserProfile) -> None:

    def save_attachment_contents(self, path_id: str, filehandle: IO[Any]) -> None:

    def attachment_vips_source(self, path_id: str) -> StreamingSourceWithSize:

    def delete_message_attachment(self, path_id: str) -> bool:

    def all_message_attachments(self, include_thumbnails: bool = False, prefix: str = '') -> Iterator[tuple[str, datetime]]:

    def get_avatar_url(self, hash_key: str, medium: bool = False) -> str:

    def get_avatar_contents(self, file_path: str) -> tuple[bytes, str]:

    def upload_single_avatar_image(self, file_path: str, user_profile: UserProfile, image_data: bytes, content_type: str, future: bool = True) -> None:

    def delete_avatar_image(self, path_id: str) -> None:

    def get_realm_icon_url(self, realm_id: str, version: str) -> str:

    def upload_realm_icon_image(self, icon_file: IO[Any], user_profile: UserProfile, content_type: str) -> None:

    def get_realm_logo_url(self, realm_id: str, version: str, night: bool) -> str:

    def upload_realm_logo_image(self, logo_file: IO[Any], user_profile: UserProfile, night: bool, content_type: str) -> None:

    def get_emoji_url(self, emoji_file_name: str, realm_id: str, still: bool = False) -> str:

    def upload_single_emoji_image(self, path: str, content_type: str, user_profile: UserProfile, image_data: bytes) -> None:

    def get_export_tarball_url(self, realm: Realm, export_path: str) -> str:

    def upload_export_tarball(self, realm: Realm, tarball_path: str, percent_callback: Callable[[int], None] = None) -> str:

    def delete_export_tarball(self, export_path: str) -> Optional[str]:
