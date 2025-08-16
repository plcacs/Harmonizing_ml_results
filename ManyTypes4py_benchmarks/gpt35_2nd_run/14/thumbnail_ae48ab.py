from typing import TypeVar, List, Tuple, Dict, Any

class BaseThumbnailFormat:
    max_width: int
    max_height: int
    animated: bool
    extension: str

    def __eq__(self, other: Any) -> bool:
        ...

    def __str__(self) -> str:
        ...

    @classmethod
    def from_string(cls, format_string: str) -> 'BaseThumbnailFormat':
        ...


class ThumbnailFormat(BaseThumbnailFormat):
    opts: str


class StoredThumbnailFormat(BaseThumbnailFormat):
    ...


def libvips_check_image(image_data: bytes, truncated_animation: bool = False) -> Iterator:
    ...


def resize_avatar(image_data: bytes, size: int = DEFAULT_AVATAR_SIZE) -> bytes:
    ...


def resize_realm_icon(image_data: bytes) -> bytes:
    ...


def resize_logo(image_data: bytes) -> bytes:
    ...


def resize_emoji(image_data: bytes, emoji_file_name: str, size: int = DEFAULT_EMOJI_SIZE) -> Tuple[bytes, bytes]:
    ...


def missing_thumbnails(image_attachment: Any) -> List[BaseThumbnailFormat]:
    ...


def maybe_thumbnail(content: bytes, content_type: str, path_id: str, realm_id: int, skip_events: bool = False) -> Any:
    ...


def get_image_thumbnail_path(image_attachment: Any, thumbnail_format: BaseThumbnailFormat) -> str:
    ...


def split_thumbnail_path(file_path: str) -> Tuple[str, BaseThumbnailFormat]:
    ...


class MarkdownImageMetadata:
    transcoded_image: Any


def get_user_upload_previews(realm_id: int, content: str, lock: bool = False, enqueue: bool = True, path_ids: List[str] = None) -> Dict[str, MarkdownImageMetadata]:
    ...


def get_default_thumbnail_url(image_attachment: Any) -> Tuple[str, bool]:
    ...


def get_transcoded_format(image_attachment: Any) -> Any:
    ...


def rewrite_thumbnailed_images(rendered_content: str, images: Dict[str, MarkdownImageMetadata], to_delete: List[str] = None) -> Tuple[str, set]:
    ...
