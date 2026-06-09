from typing import Any

# === Internal dependency: mimesis.datasets ===
CONTENT_ENCODING_DIRECTIVES: Any
CORS_OPENER_POLICIES: Any
CORS_RESOURCE_POLICIES: Any
HTTP_METHODS: Any
HTTP_SERVERS: Any
HTTP_STATUS_CODES: Any
HTTP_STATUS_MSGS: Any
PUBLIC_DNS: Any
TLD: Any
USERNAMES: Any
USER_AGENTS: Any

# === Internal dependency: mimesis.enums ===
class IPv4Purpose(Enum): ...
class Locale(Enum): ...
class PortRange(Enum): ...
class TLDType(Enum): ...
class URLScheme(Enum): ...
class DSNType(Enum): ...

# === Internal dependency: mimesis.providers.base ===
class BaseProvider:
    ...

# === Internal dependency: mimesis.providers.code ===
class Code(BaseProvider):
    ...

# === Internal dependency: mimesis.providers.date ===
class Datetime(BaseDataProvider):
    ...

# === Internal dependency: mimesis.providers.file ===
class File(BaseProvider):
    ...

# === Internal dependency: mimesis.providers.text ===
class Text(BaseDataProvider):
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None: ...