from __future__ import unicode_literals
from itertools import chain
from typing import Callable, Dict, List, Optional, Union
from pypinyin.compat import text_type
from pypinyin.constants import PHRASES_DICT, PINYIN_DICT, Style, RE_HANS
from pypinyin.converter import DefaultConverter, UltimateConverter
from pypinyin.contrib.tone_sandhi import ToneSandhiMixin
from pypinyin.contrib.tone_convert import tone2_to_tone
from pypinyin.seg import mmseg
from pypinyin.seg.simpleseg import seg

def load_single_dict(pinyin_dict: Dict[int, str], style: str = 'default') -> None:
    if style == 'tone2':
        for k, v in pinyin_dict.items():
            v = tone2_to_tone(v)
            PINYIN_DICT[k] = v
    else:
        PINYIN_DICT.update(pinyin_dict)
    mmseg.retrain(mmseg.seg)

def load_phrases_dict(phrases_dict: Dict[str, List[List[str]]], style: str = 'default') -> None:
    if style == 'tone2':
        for k, value in phrases_dict.items():
            v = [list(map(tone2_to_tone, pys)) for pys in value]
            PHRASES_DICT[k] = v
    else:
        PHRASES_DICT.update(phrases_dict)
    mmseg.retrain(mmseg.seg)

class Pinyin(object):

    def __init__(self, converter: Optional[DefaultConverter] = None, **kwargs) -> None:
        self._converter = converter or DefaultConverter()

    def pinyin(self, hans: Union[str, List[str]], style: Style = Style.TONE, heteronym: bool = False, errors: Union[str, Callable] = 'default', strict: bool = True, **kwargs) -> List[List[str]]:
        if isinstance(hans, text_type):
            han_list = self.seg(hans)
        elif isinstance(self._converter, UltimateConverter) or isinstance(self._converter, ToneSandhiMixin):
            han_list = []
            for h in hans:
                if not RE_HANS.match(h):
                    han_list.extend(self.seg(h))
                else:
                    han_list.append(h)
        else:
            han_list = chain(*(self.seg(x) for x in hans))
        pys = []
        for words in han_list:
            pys.extend(self._converter.convert(words, style, heteronym, errors, strict=strict))
        return pys

    def lazy_pinyin(self, hans: Union[str, List[str]], style: Style = Style.NORMAL, errors: Union[str, Callable] = 'default', strict: bool = True, **kwargs) -> List[str]:
        return list(chain(*self.pinyin(hans, style=style, heteronym=False, errors=errors, strict=strict)))

    def pre_seg(self, hans: str, **kwargs) -> Optional[List[str]]:
        pass

    def seg(self, hans: str, **kwargs) -> List[str]:
        pre_data = self.pre_seg(hans)
        if isinstance(pre_data, list):
            seg_data = pre_data
        else:
            seg_data = self.get_seg()(hans)
        post_data = self.post_seg(hans, seg_data)
        if isinstance(post_data, list):
            return post_data
        return seg_data

    def get_seg(self, **kwargs) -> Callable[[str], List[str]]:
        return seg

    def post_seg(self, hans: str, seg_data: List[str], **kwargs) -> Optional[List[str]]:
        pass

_default_convert = DefaultConverter()
_default_pinyin = Pinyin(_default_convert)

def to_fixed(pinyin: str, style: Style, strict: bool = True) -> str:
    return _default_convert.convert_style('', pinyin, style=style, strict=strict, default=pinyin)

_to_fixed = to_fixed

def handle_nopinyin(chars: str, errors: Union[str, Callable] = 'default', heteronym: bool = True) -> List[str]:
    return _default_convert.handle_nopinyin(chars, style=None, errors=errors, heteronym=heteronym, strict=True)

def single_pinyin(han: str, style: Style, heteronym: bool, errors: Union[str, Callable] = 'default', strict: bool = True) -> List[str]:
    return _default_convert._single_pinyin(han, style, heteronym, errors=errors, strict=strict)

def phrase_pinyin(phrase: str, style: Style, heteronym: bool, errors: Union[str, Callable] = 'default', strict: bool = True) -> List[List[str]]:
    return _default_convert._phrase_pinyin(phrase, style, heteronym, errors=errors, strict=strict)

def pinyin(hans: Union[str, List[str]], style: Style = Style.TONE, heteronym: bool = False, errors: Union[str, Callable] = 'default', strict: bool = True, v_to_u: bool = False, neutral_tone_with_five: bool = False) -> List[List[str]]:
    _pinyin = Pinyin(UltimateConverter(v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five))
    return _pinyin.pinyin(hans, style=style, heteronym=heteronym, errors=errors, strict=strict)

def slug(hans: Union[str, List[str]], style: Style = Style.NORMAL, heteronym: bool = False, separator: str = '-', errors: Union[str, Callable] = 'default', strict: bool = True) -> str:
    return separator.join(chain(*_default_pinyin.pinyin(hans, style=style, heteronym=heteronym, errors=errors, strict=strict)))

def lazy_pinyin(hans: Union[str, List[str]], style: Style = Style.NORMAL, errors: Union[str, Callable] = 'default', strict: bool = True, v_to_u: bool = False, neutral_tone_with_five: bool = False, tone_sandhi: bool = False) -> List[str]:
    _pinyin = Pinyin(UltimateConverter(v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five, tone_sandhi=tone_sandhi))
    return _pinyin.lazy_pinyin(hans, style=style, errors=errors, strict=strict)
