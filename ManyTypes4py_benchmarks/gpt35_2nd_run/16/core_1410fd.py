from __future__ import unicode_literals
from itertools import chain
from pypinyin.compat import text_type
from pypinyin.constants import PHRASES_DICT, PINYIN_DICT, Style, RE_HANS
from pypinyin.converter import DefaultConverter, UltimateConverter
from pypinyin.contrib.tone_sandhi import ToneSandhiMixin
from pypinyin.contrib.tone_convert import tone2_to_tone
from pypinyin.seg import mmseg
from pypinyin.seg.simpleseg import seg

def load_single_dict(pinyin_dict: dict, style: str = 'default') -> None:
    ...

def load_phrases_dict(phrases_dict: dict, style: str = 'default') -> None:
    ...

class Pinyin:
    def __init__(self, converter=None, **kwargs):
        ...

    def pinyin(self, hans: text_type, style: Style = Style.TONE, heteronym: bool = False, errors: str = 'default', strict: bool = True, **kwargs) -> list:
        ...

    def lazy_pinyin(self, hans: text_type, style: Style = Style.NORMAL, errors: str = 'default', strict: bool = True, **kwargs) -> list:
        ...

    def pre_seg(self, hans: text_type, **kwargs) -> list:
        ...

    def seg(self, hans: text_type, **kwargs) -> list:
        ...

    def get_seg(self, **kwargs) -> callable:
        ...

    def post_seg(self, hans: text_type, seg_data: list, **kwargs) -> list:
        ...

def to_fixed(pinyin: str, style: Style, strict: bool = True) -> str:
    ...

def handle_nopinyin(chars: text_type, errors: str = 'default', heteronym: bool = True) -> list:
    ...

def single_pinyin(han: text_type, style: Style, heteronym: bool, errors: str = 'default', strict: bool = True) -> list:
    ...

def phrase_pinyin(phrase: text_type, style: Style, heteronym: bool, errors: str = 'default', strict: bool = True) -> list:
    ...

def pinyin(hans: text_type, style: Style = Style.TONE, heteronym: bool = False, errors: str = 'default', strict: bool = True, v_to_u: bool = False, neutral_tone_with_five: bool = False) -> list:
    ...

def slug(hans: text_type, style: Style = Style.NORMAL, heteronym: bool = False, separator: str = '-', errors: str = 'default', strict: bool = True) -> str:
    ...

def lazy_pinyin(hans: text_type, style: Style = Style.NORMAL, errors: str = 'default', strict: bool = True, v_to_u: bool = False, neutral_tone_with_five: bool = False, tone_sandhi: bool = False) -> list:
    ...
