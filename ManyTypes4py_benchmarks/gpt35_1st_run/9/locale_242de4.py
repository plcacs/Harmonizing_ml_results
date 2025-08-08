from typing import Iterable, Any, Union, Dict, Optional

_default_locale: str = 'en_US'
_translations: Dict[str, Dict[str, Dict[str, str]]] = {}
_supported_locales: frozenset = frozenset([_default_locale])
_use_gettext: bool = False
CONTEXT_SEPARATOR: str = '\x04'

def get(*locale_codes: str) -> 'Locale':
    ...

def set_default_locale(code: str) -> None:
    ...

def load_translations(directory: str, encoding: Optional[str] = None) -> None:
    ...

def load_gettext_translations(directory: str, domain: str) -> None:
    ...

def get_supported_locales() -> frozenset:
    ...

class Locale:
    _cache: Dict[str, 'Locale'] = {}

    @classmethod
    def get_closest(cls, *locale_codes: str) -> 'Locale':
        ...

    @classmethod
    def get(cls, code: str) -> 'Locale':
        ...

    def __init__(self, code: str) -> None:
        ...

    def translate(self, message: str, plural_message: Optional[str] = None, count: Optional[int] = None) -> str:
        ...

    def pgettext(self, context: str, message: str, plural_message: Optional[str] = None, count: Optional[int] = None) -> str:
        ...

    def format_date(self, date: Union[int, float, datetime.datetime], gmt_offset: int = 0, relative: bool = True, shorter: bool = False, full_format: bool = False) -> str:
        ...

    def format_day(self, date: datetime.datetime, gmt_offset: int = 0, dow: bool = True) -> str:
        ...

    def list(self, parts: Iterable[str]) -> str:
        ...

    def friendly_number(self, value: int) -> str:
        ...

class CSVLocale(Locale):
    def __init__(self, code: str, translations: Dict[str, Dict[str, str]]) -> None:
        ...

    def translate(self, message: str, plural_message: Optional[str] = None, count: Optional[int] = None) -> str:
        ...

    def pgettext(self, context: str, message: str, plural_message: Optional[str] = None, count: Optional[int] = None) -> str:
        ...

class GettextLocale(Locale):
    def __init__(self, code: str, translations: gettext.GNUTranslations) -> None:
        ...

    def translate(self, message: str, plural_message: Optional[str] = None, count: Optional[int] = None) -> str:
        ...

    def pgettext(self, context: str, message: str, plural_message: Optional[str] = None, count: Optional[int] = None) -> str:
        ...
