from inspect import Signature
import logging
from typing import Any, Optional

def get_logger() -> 'KoalasUsageLogger':
    ...

def _format_signature(signature: Optional[Signature]) -> str:
    ...

class KoalasUsageLogger(object):
    def __init__(self):
        self.logger: logging.Logger = logging.getLogger('databricks.koalas.usage_logger')

    def log_success(self, class_name: str, name: str, duration: float, signature: Optional[Signature] = None) -> None:
        ...

    def log_failure(self, class_name: str, name: str, ex: Exception, duration: float, signature: Optional[Signature] = None) -> None:
        ...

    def log_missing(self, class_name: str, name: str, is_deprecated: bool = False, signature: Optional[Signature] = None) -> None:
        ...
