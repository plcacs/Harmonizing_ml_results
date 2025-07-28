#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Union
from pypinyin.compat import text_type
from pypinyin.constants import PHRASES_DICT, PINYIN_DICT, Style, RE_HANS
from pypinyin.converter import DefaultConverter, UltimateConverter
from pypinyin.contrib.tone_sandhi import ToneSandhiMixin
from pypinyin.contrib.tone_convert import tone2_to_tone
from pypinyin.seg import mmseg
from pypinyin.seg.simpleseg import seg


def load_single_dict(pinyin_dict: Dict[int, str], style: str = 'default') -> None:
    """载入用户自定义的单字拼音库

    :param pinyin_dict: 单字拼音库。比如： {0x963F: u"ā,ē"}
    :param style: pinyin_dict 参数值的拼音库风格. 支持 'default', 'tone2'
    """
    if style == 'tone2':
        for k, v in pinyin_dict.items():
            v = tone2_to_tone(v)
            PINYIN_DICT[k] = v
    else:
        PINYIN_DICT.update(pinyin_dict)
    mmseg.retrain(mmseg.seg)


def load_phrases_dict(phrases_dict: Dict[str, List[List[str]]], style: str = 'default') -> None:
    """载入用户自定义的词语拼音库

    :param phrases_dict: 词语拼音库。比如： {u"阿爸": [[u"ā"], [u"bà"]]}
    :param style: phrases_dict 参数值的拼音库风格. 支持 'default', 'tone2'
    """
    if style == 'tone2':
        for k, value in phrases_dict.items():
            v = [list(map(tone2_to_tone, pys)) for pys in value]
            PHRASES_DICT[k] = v
    else:
        PHRASES_DICT.update(phrases_dict)
    mmseg.retrain(mmseg.seg)


class Pinyin(object):
    def __init__(self, converter: Optional[Union[DefaultConverter, UltimateConverter]] = None, **kwargs: Any) -> None:
        self._converter: Union[DefaultConverter, UltimateConverter] = converter or DefaultConverter()

    def pinyin(
        self,
        hans: Union[str, List[str]],
        style: int = Style.TONE,
        heteronym: bool = False,
        errors: Union[str, Callable[[Any], Any]] = 'default',
        strict: bool = True,
        **kwargs: Any,
    ) -> List[List[str]]:
        """将汉字转换为拼音，返回汉字的拼音列表。"""
        if isinstance(hans, text_type):
            han_list: Any = self.seg(hans)
        elif isinstance(self._converter, UltimateConverter) or isinstance(self._converter, ToneSandhiMixin):
            han_list = []
            for h in hans:
                if not RE_HANS.match(h):
                    han_list.extend(self.seg(h))
                else:
                    han_list.append(h)
        else:
            han_list = chain(*(self.seg(x) for x in hans))
        pys: List[List[str]] = []
        for words in han_list:
            pys.extend(self._converter.convert(words, style, heteronym, errors, strict=strict))
        return pys

    def lazy_pinyin(
        self,
        hans: Union[str, List[str]],
        style: int = Style.NORMAL,
        errors: Union[str, Callable[[Any], Any]] = 'default',
        strict: bool = True,
        **kwargs: Any,
    ) -> List[str]:
        """将汉字转换为拼音，返回不包含多音字结果的拼音列表."""
        return list(chain(*self.pinyin(hans, style=style, heteronym=False, errors=errors, strict=strict)))

    def pre_seg(self, hans: str, **kwargs: Any) -> Optional[List[str]]:
        """对字符串进行分词前的预处理。"""
        pass

    def seg(self, hans: str, **kwargs: Any) -> List[str]:
        """对汉字进行分词。"""
        pre_data: Optional[List[str]] = self.pre_seg(hans)
        if isinstance(pre_data, list):
            seg_data: List[str] = pre_data
        else:
            seg_data = self.get_seg()(hans)
        post_data: Optional[List[str]] = self.post_seg(hans, seg_data)
        if isinstance(post_data, list):
            return post_data
        return seg_data

    def get_seg(self, **kwargs: Any) -> Callable[[str], List[str]]:
        """获取分词函数。"""
        return seg

    def post_seg(self, hans: str, seg_data: List[str], **kwargs: Any) -> Optional[List[str]]:
        """对分词后的结果进行处理。"""
        pass


_default_convert: DefaultConverter = DefaultConverter()
_default_pinyin: Pinyin = Pinyin(_default_convert)


def to_fixed(pinyin: str, style: int, strict: bool = True) -> str:
    return _default_convert.convert_style('', pinyin, style=style, strict=strict, default=pinyin)


_to_fixed = to_fixed


def handle_nopinyin(chars: str, errors: Union[str, Callable[[Any], Any]] = 'default', heteronym: bool = True) -> List[str]:
    return _default_convert.handle_nopinyin(chars, style=None, errors=errors, heteronym=heteronym, strict=True)


def single_pinyin(han: str, style: int, heteronym: bool, errors: Union[str, Callable[[Any], Any]] = 'default', strict: bool = True) -> List[str]:
    return _default_convert._single_pinyin(han, style, heteronym, errors=errors, strict=strict)


def phrase_pinyin(phrase: str, style: int, heteronym: bool, errors: Union[str, Callable[[Any], Any]] = 'default', strict: bool = True) -> List[List[str]]:
    return _default_convert._phrase_pinyin(phrase, style, heteronym, errors=errors, strict=strict)


def pinyin(
    hans: Union[str, List[str]],
    style: int = Style.TONE,
    heteronym: bool = False,
    errors: Union[str, Callable[[Any], Any]] = 'default',
    strict: bool = True,
    v_to_u: bool = False,
    neutral_tone_with_five: bool = False,
) -> List[List[str]]:
    """将汉字转换为拼音，返回汉字的拼音列表。"""
    _pinyin: Pinyin = Pinyin(UltimateConverter(v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five))
    return _pinyin.pinyin(hans, style=style, heteronym=heteronym, errors=errors, strict=strict)


def slug(
    hans: Union[str, List[str]],
    style: int = Style.NORMAL,
    heteronym: bool = False,
    separator: str = '-',
    errors: Union[str, Callable[[Any], Any]] = 'default',
    strict: bool = True,
) -> str:
    """将汉字转换为拼音，然后生成 slug 字符串."""
    return separator.join(chain(*_default_pinyin.pinyin(hans, style=style, heteronym=heteronym, errors=errors, strict=strict)))


def lazy_pinyin(
    hans: Union[str, List[str]],
    style: int = Style.NORMAL,
    errors: Union[str, Callable[[Any], Any]] = 'default',
    strict: bool = True,
    v_to_u: bool = False,
    neutral_tone_with_five: bool = False,
    tone_sandhi: bool = False,
) -> List[str]:
    """将汉字转换为拼音，返回不包含多音字结果的拼音列表."""
    _pinyin: Pinyin = Pinyin(
        UltimateConverter(v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five, tone_sandhi=tone_sandhi)
    )
    return _pinyin.lazy_pinyin(hans, style=style, errors=errors, strict=strict)