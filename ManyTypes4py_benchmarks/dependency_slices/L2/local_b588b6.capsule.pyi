# === Third-party dependency: django.conf ===
settings: LazySettings

# === Third-party dependency: pyvips ===
# Used symbols: Source

# === Internal dependency: zerver.lib.mime_types ===
# re-export: from mimetypes import guess_type as guess_type

# === Internal dependency: zerver.lib.thumbnail ===
def resize_realm_icon(image_data: bytes) -> bytes: ...
def resize_logo(image_data: bytes) -> bytes: ...

# === Internal dependency: zerver.lib.timestamp ===
def timestamp_to_datetime(timestamp: float) -> datetime: ...

# === Internal dependency: zerver.lib.upload.base ===
class StreamingSourceWithSize: ...
class ZulipUploadBackend:
    ...

# === Internal dependency: zerver.lib.utils ===
def assert_is_not_none(value: T | None) -> T: ...

# === Internal dependency: zerver.models ===
# re-export: from zerver.models.realm_emoji import RealmEmoji as RealmEmoji