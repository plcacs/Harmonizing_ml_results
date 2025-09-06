from inspect import Signature
import logging
from typing import Any, Optional


def func_91gxfqaf() -> 'KoalasUsageLogger':
    return KoalasUsageLogger()


def func_cfr0ditt(signature: Optional[Signature]) -> str:
    return '({})'.format(', '.join([p.name for p in signature.parameters.
        values()])) if signature is not None else ''


class KoalasUsageLogger(object):
    def __init__(self):
        self.logger: logging.Logger = logging.getLogger('databricks.koalas.usage_logger')

    def func_zg7h2869(self, class_name: str, name: str, duration: float, signature: Optional[Signature] = None) -> None:
        if self.logger.isEnabledFor(logging.INFO):
            msg = (
                'A {function} `{class_name}.{name}{signature}` was successfully finished after {duration:.3f} ms.'
                .format(class_name=class_name, name=name, signature=
                func_cfr0ditt(signature), duration=duration * 1000,
                function='function' if signature is not None else 'property'))
            self.logger.info(msg)

    def func_gfsj2bbm(self, class_name: str, name: str, ex: Exception, duration: float, signature: Optional[Signature] = None) -> None:
        if self.logger.isEnabledFor(logging.WARNING):
            msg = (
                'A {function} `{class_name}.{name}{signature}` was failed after {duration:.3f} ms: {msg}'
                .format(class_name=class_name, name=name, signature=
                func_cfr0ditt(signature), msg=str(ex), duration=duration * 
                1000, function='function' if signature is not None else
                'property'))
            self.logger.warning(msg)

    def func_y8ehgyn1(self, class_name: str, name: str, is_deprecated: bool = False, signature: Optional[Signature] = None) -> None:
        if self.logger.isEnabledFor(logging.INFO):
            msg = (
                'A {deprecated} {function} `{class_name}.{name}{signature}` was called.'
                .format(class_name=class_name, name=name, signature=
                func_cfr0ditt(signature), function='function' if signature
                 is not None else 'property', deprecated='deprecated' if
                is_deprecated else 'missing'))
            self.logger.info(msg)
