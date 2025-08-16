from __future__ import unicode_literals
from copy import deepcopy
from pypinyin.compat import text_type, callable_check
from pypinyin.constants import PHRASES_DICT, PINYIN_DICT, RE_HANS
from pypinyin.contrib.uv import V2UMixin
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.contrib.tone_sandhi import ToneSandhiMixin
from pypinyin.utils import _remove_dup_and_empty
from pypinyin.style import auto_discover
from pypinyin.style import convert as convert_style
auto_discover()

class Converter(object):

    def convert(self, words: str, style, heteronym: bool, errors, strict: bool, **kwargs) -> list:
        raise NotImplementedError

    def pre_convert_style(self, han: str, orig_pinyin: str, style, strict: bool, **kwargs):
        pass

    def convert_style(self, han: str, orig_pinyin: str, style, strict: bool, **kwargs) -> str

    def post_convert_style(self, han: str, orig_pinyin: str, converted_pinyin: str, style, strict: bool, **kwargs):
        pass

    def pre_handle_nopinyin(self, chars: str, style, heteronym: bool, errors, strict: bool, **kwargs):
        pass

    def handle_nopinyin(self, chars: str, style, heteronym: bool, errors, strict: bool, **kwargs) -> list

    def post_handle_nopinyin(self, chars: str, style, heteronym: bool, errors, strict: bool, pinyin: list, **kwargs):
        pass

    def post_pinyin(self, han: str, heteronym: bool, pinyin: list, **kwargs):
        pass

    def _phrase_pinyin(self, phrase: str, style, heteronym: bool, errors, strict: bool) -> list

    def convert_styles(self, pinyin_list: list, phrase: str, style, heteronym: bool, errors, strict: bool, **kwargs) -> list

    def _single_pinyin(self, han: str, style, heteronym: bool, errors, strict: bool) -> list

    def _convert_style(self, han: str, pinyin: str, style, strict: bool, default: str, **kwargs) -> str

    def _convert_nopinyin_chars(self, chars: str, style, heteronym: bool, errors, strict: bool) -> str

class _v2UConverter(V2UMixin, DefaultConverter):
    pass

class _neutralToneWith5Converter(NeutralToneWith5Mixin, DefaultConverter):
    pass

class _toneSandhiConverter(ToneSandhiMixin, DefaultConverter):
    pass

class UltimateConverter(DefaultConverter):

    def __init__(self, v_to_u: bool = False, neutral_tone_with_five: bool = False, tone_sandhi: bool = False, **kwargs):
        super(UltimateConverter, self).__init__(**kwargs)
        self._v_to_u = v_to_u
        self._neutral_tone_with_five = neutral_tone_with_five
        self._tone_sandhi = tone_sandhi

    def post_convert_style(self, han: str, orig_pinyin: str, converted_pinyin: str, style, strict: bool, **kwargs) -> str

    def post_pinyin(self, han: str, heteronym: bool, pinyin: list, **kwargs) -> list

_mixConverter = UltimateConverter
