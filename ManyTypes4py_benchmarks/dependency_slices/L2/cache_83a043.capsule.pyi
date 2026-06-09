# === Third-party dependency: django.conf ===
settings: LazySettings

# === Third-party dependency: django.core.cache ===
caches: CacheHandler

# === Third-party dependency: django.core.cache.backends.base ===
class BaseCache: ...

# === Third-party dependency: django.db.models ===
# Used symbols: Q, QuerySet

# === Internal dependency: zerver.models ===
# re-export: from zerver.models.streams import Subscription as Subscription
# re-export: from zerver.models.users import UserProfile as UserProfile

# === Internal dependency: zerver.models.users ===
def is_cross_realm_bot_email(email: str) -> bool: ...