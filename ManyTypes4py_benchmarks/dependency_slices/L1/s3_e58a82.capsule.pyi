from typing import Any

# === Third-party dependency: boto3 ===
def resource(*args, **kwargs) -> Any: ...

# === Third-party dependency: botocore ===
class UNSIGNED: ...
UNSIGNED: Any

# === Third-party dependency: botocore.client ===
# Used symbols: Config

# === Third-party dependency: django.conf ===
settings: LazySettings

# === Third-party dependency: django.utils.http ===
def content_disposition_header(as_attachment, filename) -> Any: ...

# === Third-party dependency: mypy_boto3_s3.service_resource ===
class Bucket(ServiceResource): ...

# === Third-party dependency: pyvips ===
# Used symbols: Source, SourceCustom

# === Internal dependency: zerver.lib.mime_types ===
INLINE_MIME_TYPES = ['application/pdf', 'audio/aac', 'audio/flac', 'audio/mp4', 'audio/mpeg', 'audio/wav', 'audio/webm', 'image/apng', ...]

# === Internal dependency: zerver.lib.partial ===
partial: Any

# === Internal dependency: zerver.lib.thumbnail ===
def resize_realm_icon(image_data): ...
def resize_logo(image_data): ...

# === Internal dependency: zerver.lib.upload.base ===
class StreamingSourceWithSize: ...
class ZulipUploadBackend:
    ...

# === Internal dependency: zerver.models ===
from zerver.models.realm_emoji import RealmEmoji as RealmEmoji