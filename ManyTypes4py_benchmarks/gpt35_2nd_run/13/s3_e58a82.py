def get_bucket(bucket_name: str, authed: bool = True) -> 'Bucket':
def upload_content_to_s3(bucket: 'Bucket', path: str, content_type: str, user_profile: 'UserProfile', contents: bytes, *, storage_class: str = 'STANDARD', cache_control: str = None, extra_metadata: dict = None, filename: str = None) -> None:
def get_boto_client() -> 'S3Client':
def get_signed_upload_url(path: str, filename: str, force_download: bool = False) -> str:
class S3UploadBackend(ZulipUploadBackend):
    def __init__(self) -> None:
    def delete_file_from_s3(self, path_id: str, bucket: 'Bucket') -> bool:
    def construct_public_upload_url_base(self) -> str:
    def get_public_upload_root_url(self) -> str:
    def get_public_upload_url(self, key: str) -> str:
    def generate_message_upload_path(self, realm_id: str, sanitized_file_name: str) -> str:
    def upload_message_attachment(self, path_id: str, filename: str, content_type: str, file_data: bytes, user_profile: 'UserProfile') -> None:
    def save_attachment_contents(self, path_id: str, filehandle: 'IO[bytes]') -> None:
    def attachment_vips_source(self, path_id: str) -> 'StreamingSourceWithSize':
    def delete_message_attachment(self, path_id: str) -> bool:
    def delete_message_attachments(self, path_ids: list[str]) -> None:
    def all_message_attachments(self, include_thumbnails: bool = False, prefix: str = '') -> Iterator[tuple[str, datetime]]:
    def get_avatar_url(self, hash_key: str, medium: bool = False) -> str:
    def get_avatar_contents(self, file_path: str) -> tuple[bytes, str]:
    def upload_single_avatar_image(self, file_path: str, *, user_profile: 'UserProfile', image_data: bytes, content_type: str, future: bool = True) -> None:
    def delete_avatar_image(self, path_id: str) -> bool:
    def get_realm_icon_url(self, realm_id: str, version: int) -> str:
    def upload_realm_icon_image(self, icon_file: 'IO[bytes]', user_profile: 'UserProfile', content_type: str) -> None:
    def get_realm_logo_url(self, realm_id: str, version: int, night: bool) -> str:
    def upload_realm_logo_image(self, logo_file: 'IO[bytes]', user_profile: 'UserProfile', night: bool, content_type: str) -> None:
    def get_emoji_url(self, emoji_file_name: str, realm_id: str, still: bool = False) -> str:
    def upload_single_emoji_image(self, path: str, content_type: str, user_profile: 'UserProfile', image_data: bytes) -> None:
    def get_export_tarball_url(self, realm: 'Realm', export_path: str) -> str:
    def export_object(self, tarball_path: str) -> 'Object':
    def upload_export_tarball(self, realm: 'Realm', tarball_path: str, percent_callback: Callable[[int], Any] = None) -> str:
    def delete_export_tarball(self, export_path: str) -> str:
