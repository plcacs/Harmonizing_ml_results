#!/usr/bin/env python3
"""Translation methods for generating localized strings.

To load a locale and generate a translated string::

    user_locale = tornado.locale.get("es_LA")
    print(user_locale.translate("Sign out"))

`tornado.locale.get()` returns the closest matching locale, not necessarily the
specific locale you requested. You can support pluralization with
additional arguments to `~Locale.translate()`, e.g.::

    people = [...]
    message = user_locale.translate(
        "%(list)s is online", "%(list)s are online", len(people))
    print(message % {"list": user_locale.list(people)})

The first string is chosen if ``len(people) == 1``, otherwise the second
string is chosen.

Applications should call one of `load_translations` (which uses a simple
CSV format) or `load_gettext_translations` (which uses the ``.mo`` format
supported by `gettext` and related tools).  If neither method is called,
the `Locale.translate` method will simply return the original string.
"""
import codecs
import csv
import datetime
import gettext
import glob
import os
import re
from typing import Any, Dict, Iterable, Optional, Union, FrozenSet
from tornado import escape
from tornado.log import gen_log
from tornado._locale_data import LOCALE_NAMES

_default_locale: str = 'en_US'
_translations: Dict[str, Any] = {}
_supported_locales: FrozenSet[str] = frozenset([_default_locale])
_use_gettext: bool = False
CONTEXT_SEPARATOR: str = '\x04'


def get(*locale_codes: str) -> "Locale":
    """Returns the closest match for the given locale codes.

    We iterate over all given locale codes in order. If we have a tight
    or a loose match for the code (e.g., "en" for "en_US"), we return
    the locale. Otherwise we move to the next code in the list.

    By default we return ``en_US`` if no translations are found for any of
    the specified locales. You can change the default locale with
    `set_default_locale()`.
    """
    return Locale.get_closest(*locale_codes)


def set_default_locale(code: str) -> None:
    """Sets the default locale.

    The default locale is assumed to be the language used for all strings
    in the system. The translations loaded from disk are mappings from
    the default locale to the destination locale. Consequently, you don't
    need to create a translation file for the default locale.
    """
    global _default_locale
    global _supported_locales
    _default_locale = code
    _supported_locales = frozenset(list(_translations.keys()) + [_default_locale])


def load_translations(directory: str, encoding: Optional[str] = None) -> None:
    """Loads translations from CSV files in a directory.

    Translations are strings with optional Python-style named placeholders
    (e.g., ``My name is %(name)s``) and their associated translations.

    The directory should have translation files of the form ``LOCALE.csv``,
    e.g. ``es_GT.csv``. The CSV files should have two or three columns: string,
    translation, and an optional plural indicator. Plural indicators should
    be one of "plural" or "singular". A given string can have both singular
    and plural forms. For example ``%(name)s liked this`` may have a
    different verb conjugation depending on whether %(name)s is one
    name or a list of names. There should be two rows in the CSV file for
    that string, one with plural indicator "singular", and one "plural".
    For strings with no verbs that would change on translation, simply
    use "unknown" or the empty string (or don't include the column at all).

    The file is read using the `csv` module in the default "excel" dialect.
    In this format there should not be spaces after the commas.

    If no ``encoding`` parameter is given, the encoding will be
    detected automatically (among UTF-8 and UTF-16) if the file
    contains a byte-order marker (BOM), defaulting to UTF-8 if no BOM
    is present.

    Example translation ``es_LA.csv``::

        "I love you","Te amo"
        "%(name)s liked this","A %(name)s les gustó esto","plural"
        "%(name)s liked this","A %(name)s le gustó esto","singular"

    .. versionchanged:: 4.3
       Added ``encoding`` parameter. Added support for BOM-based encoding
       detection, UTF-16, and UTF-8-with-BOM.
    """
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
        curr_encoding: Optional[str] = encoding
        if curr_encoding is None:
            with open(full_path, 'rb') as bf:
                data: bytes = bf.read(len(codecs.BOM_UTF16_LE))
            if data in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
                curr_encoding = 'utf-16'
            else:
                curr_encoding = 'utf-8-sig'
        with open(full_path, encoding=curr_encoding) as f:
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
    """Loads translations from `gettext`'s locale tree

    Locale tree is similar to system's ``/usr/share/locale``, like::

        {directory}/{lang}/LC_MESSAGES/{domain}.mo

    Three steps are required to have your app translated:

    1. Generate POT translation file::

        xgettext --language=Python --keyword=_:1,2 -d mydomain file1.py file2.html etc

    2. Merge against existing POT file::

        msgmerge old.po mydomain.po > new.po

    3. Compile::

        msgfmt mydomain.po -o {directory}/pt_BR/LC_MESSAGES/mydomain.mo
    """
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
    """Returns a list of all the supported locale codes."""
    return _supported_locales


class Locale:
    """Object representing a locale.

    After calling one of `load_translations` or `load_gettext_translations`,
    call `get` or `get_closest` to get a Locale object.
    """
    _cache: Dict[str, "Locale"] = {}

    @classmethod
    def get_closest(cls, *locale_codes: str) -> "Locale":
        """Returns the closest match for the given locale code."""
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
        """Returns the Locale for the given locale code.

        If it is not supported, we raise an exception.
        """
        if code not in cls._cache:
            assert code in _supported_locales
            translations: Optional[Any] = _translations.get(code, None)
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
        self._months: Iterable[str] = [_('January'), _('February'), _('March'), _('April'),
                                         _('May'), _('June'), _('July'), _('August'),
                                         _('September'), _('October'), _('November'), _('December')]
        self._weekdays: Iterable[str] = [_('Monday'), _('Tuesday'), _('Wednesday'),
                                         _('Thursday'), _('Friday'), _('Saturday'), _('Sunday')]

    def translate(self, message: str, plural_message: Optional[str] = None, count: Optional[int] = None) -> str:
        """Returns the translation for the given message for this locale.

        If ``plural_message`` is given, you must also provide
        ``count``. We return ``plural_message`` when ``count != 1``,
        and we return the singular form for the given message when
        ``count == 1``.
        """
        raise NotImplementedError()

    def pgettext(self, context: str, message: str, plural_message: Optional[str] = None, count: Optional[int] = None) -> str:
        raise NotImplementedError()

    def format_date(self, date: Union[datetime.datetime, int, float], gmt_offset: int = 0, relative: bool = True,
                    shorter: bool = False, full_format: bool = False) -> str:
        """Formats the given date.

        By default, we return a relative time (e.g., "2 minutes ago"). You
        can return an absolute date string with ``relative=False``.

        You can force a full format date ("July 10, 1980") with
        ``full_format=True``.

        This method is primarily intended for dates in the past.
        For dates in the future, we fall back to full format.

        .. versionchanged:: 6.4
           Aware `datetime.datetime` objects are now supported (naive
           datetimes are still assumed to be UTC).
        """
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
        fmt: Optional[str] = None
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
                fmt = _('%(time)s')
            elif days == 1 and local_date.day == local_yesterday.day and relative:
                fmt = _('yesterday') if shorter else _('yesterday at %(time)s')
            elif days < 5:
                fmt = _('%(weekday)s') if shorter else _('%(weekday)s at %(time)s')
            elif days < 334:
                fmt = _('%(month_name)s %(day)s') if shorter else _('%(month_name)s %(day)s at %(time)s')
        if fmt is None:
            fmt = _('%(month_name)s %(day)s, %(year)s') if shorter else _('%(month_name)s %(day)s, %(year)s at %(time)s')
        tfhour_clock: bool = self.code not in ('en', 'en_US', 'zh_CN')
        if tfhour_clock:
            str_time: str = '%d:%02d' % (local_date.hour, local_date.minute)
        elif self.code == 'zh_CN':
            str_time = '%s%d:%02d' % (('上午', '下午')[local_date.hour >= 12], local_date.hour % 12 or 12, local_date.minute)
        else:
            str_time = '%d:%02d %s' % (local_date.hour % 12 or 12, local_date.minute, ('am', 'pm')[local_date.hour >= 12])
        return fmt % {'month_name': list(self._months)[local_date.month - 1],
                      'weekday': list(self._weekdays)[local_date.weekday()],
                      'day': str(local_date.day),
                      'year': str(local_date.year),
                      'time': str_time}

    def format_day(self, date: datetime.datetime, gmt_offset: int = 0, dow: bool = True) -> str:
        """Formats the given date as a day of week.

        Example: "Monday, January 22". You can remove the day of week with
        ``dow=False``.
        """
        local_date: datetime.datetime = date - datetime.timedelta(minutes=gmt_offset)
        _ = self.translate
        if dow:
            return _('%(weekday)s, %(month_name)s %(day)s') % {'month_name': list(self._months)[local_date.month - 1],
                                                                 'weekday': list(self._weekdays)[local_date.weekday()],
                                                                 'day': str(local_date.day)}
        else:
            return _('%(month_name)s %(day)s') % {'month_name': list(self._months)[local_date.month - 1],
                                                  'day': str(local_date.day)}

    def list(self, parts: Iterable[str]) -> str:
        """Returns a comma-separated list for the given list of parts.

        The format is, e.g., "A, B and C", "A and B" or just "A" for lists
        of size 1.
        """
        _ = self.translate
        parts_list = list(parts)
        if len(parts_list) == 0:
            return ''
        if len(parts_list) == 1:
            return parts_list[0]
        comma: str = ' و ' if self.code.startswith('fa') else ', '
        return _('%(commas)s and %(last)s') % {'commas': comma.join(parts_list[:-1]), 'last': parts_list[-1]}

    def friendly_number(self, value: int) -> str:
        """Returns a comma-separated number for the given integer."""
        if self.code not in ('en', 'en_US'):
            return str(value)
        s: str = str(value)
        parts: list[str] = []
        while s:
            parts.append(s[-3:])
            s = s[:-3]
        return ','.join(reversed(parts))


class CSVLocale(Locale):
    """Locale implementation using tornado's CSV translation format."""

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
    """Locale implementation using the `gettext` module."""

    def __init__(self, code: str, translations: Any) -> None:
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
        """Allows to set context for translation, accepts plural forms.

        Usage example::

            pgettext("law", "right")
            pgettext("good", "right")

        Plural message example::

            pgettext("organization", "club", "clubs", len(clubs))
            pgettext("stick", "club", "clubs", len(clubs))

        To generate POT file with context, add following options to step 1
        of `load_gettext_translations` sequence::

            xgettext [basic options] --keyword=pgettext:1c,2 --keyword=pgettext:1c,2,3

        .. versionadded:: 4.2
        """
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
