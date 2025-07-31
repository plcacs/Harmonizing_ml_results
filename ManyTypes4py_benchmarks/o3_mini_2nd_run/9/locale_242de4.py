import codecs
import csv
import datetime
import gettext
import glob
import os
import re
from typing import Any, Dict, FrozenSet, List, Optional, Union
from tornado import escape
from tornado.log import gen_log
from tornado._locale_data import LOCALE_NAMES

_default_locale: str = 'en_US'
_translations: Dict[str, Any] = {}
_supported_locales: FrozenSet[str] = frozenset([_default_locale])
_use_gettext: bool = False
CONTEXT_SEPARATOR: str = '\x04'


def get(*locale_codes: Optional[str]) -> "Locale":
    return Locale.get_closest(*locale_codes)


def set_default_locale(code: str) -> None:
    global _default_locale
    global _supported_locales
    _default_locale = code
    _supported_locales = frozenset(list(_translations.keys()) + [_default_locale])


def load_translations(directory: str, encoding: Optional[str] = None) -> None:
    global _translations
    global _supported_locales
    _translations = {}
    for path in os.listdir(directory):
        if not path.endswith('.csv'):
            continue
        locale, extension = path.split('.')
        if not re.match('[a-z]+(_[A-Z]+)?$', locale):
            gen_log.error('Unrecognized locale %r (path: %s)', locale, os.path.join(directory, path))
            continue
        full_path: str = os.path.join(directory, path)
        current_encoding: Optional[str] = encoding
        if current_encoding is None:
            with open(full_path, 'rb') as bf:
                data: bytes = bf.read(len(codecs.BOM_UTF16_LE))
            if data in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
                current_encoding = 'utf-16'
            else:
                current_encoding = 'utf-8-sig'
        with open(full_path, encoding=current_encoding) as f:
            _translations[locale] = {}
            for i, row in enumerate(csv.reader(f)):
                if not row or len(row) < 2:
                    continue
                row = [escape.to_unicode(c).strip() for c in row]
                english: str = row[0]
                translation: str = row[1]
                if len(row) > 2:
                    plural: str = row[2] or 'unknown'
                else:
                    plural = 'unknown'
                if plural not in ('plural', 'singular', 'unknown'):
                    gen_log.error('Unrecognized plural indicator %r in %s line %d', plural, path, i + 1)
                    continue
                _translations[locale].setdefault(plural, {})[english] = translation
    _supported_locales = frozenset(list(_translations.keys()) + [_default_locale])
    gen_log.debug('Supported locales: %s', sorted(_supported_locales))


def load_gettext_translations(directory: str, domain: str) -> None:
    global _translations
    global _supported_locales
    global _use_gettext
    _translations = {}
    for filename in glob.glob(os.path.join(directory, '*', 'LC_MESSAGES', domain + '.mo')):
        lang: str = os.path.basename(os.path.dirname(os.path.dirname(filename)))
        try:
            _translations[lang] = gettext.translation(domain, directory, languages=[lang])
        except Exception as e:
            gen_log.error("Cannot load translation for '%s': %s", lang, str(e))
            continue
    _supported_locales = frozenset(list(_translations.keys()) + [_default_locale])
    _use_gettext = True
    gen_log.debug('Supported locales: %s', sorted(_supported_locales))


def get_supported_locales() -> FrozenSet[str]:
    return _supported_locales


class Locale:
    _cache: Dict[str, "Locale"] = {}

    @classmethod
    def get_closest(cls, *locale_codes: Optional[str]) -> "Locale":
        for code in locale_codes:
            if not code:
                continue
            code = code.replace('-', '_')
            parts = code.split('_')
            if len(parts) > 2:
                continue
            elif len(parts) == 2:
                code = parts[0].lower() + '_' + parts[1].upper()
            if code in _supported_locales:
                return cls.get(code)
            if parts[0].lower() in _supported_locales:
                return cls.get(parts[0].lower())
        return cls.get(_default_locale)

    @classmethod
    def get(cls, code: str) -> "Locale":
        if code not in cls._cache:
            assert code in _supported_locales
            translations: Any = _translations.get(code, None)
            if translations is None:
                locale_obj: Locale = CSVLocale(code, {})
            elif _use_gettext:
                locale_obj = GettextLocale(code, translations)
            else:
                locale_obj = CSVLocale(code, translations)
            cls._cache[code] = locale_obj
        return cls._cache[code]

    def __init__(self, code: str) -> None:
        self.code: str = code
        self.name: str = LOCALE_NAMES.get(code, {}).get('name', 'Unknown')
        self.rtl: bool = False
        for prefix in ['fa', 'ar', 'he']:
            if self.code.startswith(prefix):
                self.rtl = True
                break
        _ = self.translate
        self._months: List[str] = [_('January'), _('February'), _('March'), _('April'), _('May'),
                                     _('June'), _('July'), _('August'), _('September'), _('October'),
                                     _('November'), _('December')]
        self._weekdays: List[str] = [_('Monday'), _('Tuesday'), _('Wednesday'), _('Thursday'),
                                       _('Friday'), _('Saturday'), _('Sunday')]

    def translate(self, message: str, plural_message: Optional[str] = None, count: Optional[int] = None) -> str:
        raise NotImplementedError()

    def pgettext(self, context: str, message: str, plural_message: Optional[str] = None, count: Optional[int] = None) -> str:
        raise NotImplementedError()

    def format_date(self, date: Union[datetime.datetime, int, float], gmt_offset: int = 0, relative: bool = True,
                    shorter: bool = False, full_format: bool = False) -> str:
        if isinstance(date, (int, float)):
            date = datetime.datetime.fromtimestamp(date, datetime.timezone.utc)
        if date.tzinfo is None:
            date = date.replace(tzinfo=datetime.timezone.utc)
        now: datetime.datetime = datetime.datetime.now(datetime.timezone.utc)
        if date > now:
            if relative and (date - now).seconds < 60:
                date = now
            else:
                full_format = True
        local_date: datetime.datetime = date - datetime.timedelta(minutes=gmt_offset)
        local_now: datetime.datetime = now - datetime.timedelta(minutes=gmt_offset)
        local_yesterday: datetime.datetime = local_now - datetime.timedelta(hours=24)
        difference: datetime.timedelta = now - date
        seconds: int = difference.seconds
        days: int = difference.days
        _ = self.translate
        format_str: Optional[str] = None
        if not full_format:
            if relative and days == 0:
                if seconds < 50:
                    return _('1 second ago', '%(seconds)d seconds ago', seconds) % {'seconds': seconds}
                if seconds < 50 * 60:
                    minutes: int = round(seconds / 60.0)
                    return _('1 minute ago', '%(minutes)d minutes ago', minutes) % {'minutes': minutes}
                hours: int = round(seconds / (60.0 * 60))
                return _('1 hour ago', '%(hours)d hours ago', hours) % {'hours': hours}
            if days == 0:
                format_str = _('%(time)s')
            elif days == 1 and local_date.day == local_yesterday.day and relative:
                format_str = _('yesterday') if shorter else _('yesterday at %(time)s')
            elif days < 5:
                format_str = _('%(weekday)s') if shorter else _('%(weekday)s at %(time)s')
            elif days < 334:
                format_str = _('%(month_name)s %(day)s') if shorter else _('%(month_name)s %(day)s at %(time)s')
        if format_str is None:
            format_str = _('%(month_name)s %(day)s, %(year)s') if shorter else _('%(month_name)s %(day)s, %(year)s at %(time)s')
        tfhour_clock: bool = self.code not in ('en', 'en_US', 'zh_CN')
        if tfhour_clock:
            str_time: str = '%d:%02d' % (local_date.hour, local_date.minute)
        elif self.code == 'zh_CN':
            str_time = '%s%d:%02d' % (('上午', '下午')[local_date.hour >= 12],
                                       local_date.hour % 12 or 12, local_date.minute)
        else:
            str_time = '%d:%02d %s' % (local_date.hour % 12 or 12, local_date.minute, ('am', 'pm')[local_date.hour >= 12])
        return format_str % {
            'month_name': self._months[local_date.month - 1],
            'weekday': self._weekdays[local_date.weekday()],
            'day': str(local_date.day),
            'year': str(local_date.year),
            'time': str_time
        }

    def format_day(self, date: datetime.datetime, gmt_offset: int = 0, dow: bool = True) -> str:
        local_date: datetime.datetime = date - datetime.timedelta(minutes=gmt_offset)
        _ = self.translate
        if dow:
            return _('%(weekday)s, %(month_name)s %(day)s') % {
                'month_name': self._months[local_date.month - 1],
                'weekday': self._weekdays[local_date.weekday()],
                'day': str(local_date.day)
            }
        else:
            return _('%(month_name)s %(day)s') % {
                'month_name': self._months[local_date.month - 1],
                'day': str(local_date.day)
            }

    def list(self, parts: List[str]) -> str:
        _ = self.translate
        if len(parts) == 0:
            return ''
        if len(parts) == 1:
            return parts[0]
        comma: str = ' و ' if self.code.startswith('fa') else ', '
        return _('%(commas)s and %(last)s') % {'commas': comma.join(parts[:-1]), 'last': parts[-1]}

    def friendly_number(self, value: int) -> str:
        if self.code not in ('en', 'en_US'):
            return str(value)
        s: str = str(value)
        parts: List[str] = []
        while s:
            parts.append(s[-3:])
            s = s[:-3]
        return ','.join(reversed(parts))


class CSVLocale(Locale):
    def __init__(self, code: str, translations: Dict[str, Dict[str, str]]) -> None:
        self.translations: Dict[str, Dict[str, str]] = translations
        super().__init__(code)

    def translate(self, message: str, plural_message: Optional[str] = None, count: Optional[int] = None) -> str:
        if plural_message is not None:
            assert count is not None
            if count != 1:
                message = plural_message
                message_dict: Dict[str, str] = self.translations.get('plural', {})
            else:
                message_dict = self.translations.get('singular', {})
        else:
            message_dict = self.translations.get('unknown', {})
        return message_dict.get(message, message)

    def pgettext(self, context: str, message: str, plural_message: Optional[str] = None, count: Optional[int] = None) -> str:
        if self.translations:
            gen_log.warning('pgettext is not supported by CSVLocale')
        return self.translate(message, plural_message, count)


class GettextLocale(Locale):
    def __init__(self, code: str, translations: gettext.GNUTranslations) -> None:
        self.ngettext = translations.ngettext
        self.gettext = translations.gettext
        super().__init__(code)

    def translate(self, message: str, plural_message: Optional[str] = None, count: Optional[int] = None) -> str:
        if plural_message is not None:
            assert count is not None
            return self.ngettext(message, plural_message, count)
        else:
            return self.gettext(message)

    def pgettext(self, context: str, message: str, plural_message: Optional[str] = None, count: Optional[int] = None) -> str:
        if plural_message is not None:
            assert count is not None
            msgs_with_ctxt = (f'{context}{CONTEXT_SEPARATOR}{message}', f'{context}{CONTEXT_SEPARATOR}{plural_message}', count)
            result: str = self.ngettext(*msgs_with_ctxt)
            if CONTEXT_SEPARATOR in result:
                result = self.ngettext(message, plural_message, count)
            return result
        else:
            msg_with_ctxt: str = f'{context}{CONTEXT_SEPARATOR}{message}'
            result = self.gettext(msg_with_ctxt)
            if CONTEXT_SEPARATOR in result:
                result = message
            return result