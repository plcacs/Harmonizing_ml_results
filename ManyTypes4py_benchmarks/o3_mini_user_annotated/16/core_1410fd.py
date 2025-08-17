#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Union

from pypinyin.compat import text_type
from pypinyin.constants import (
    PHRASES_DICT, PINYIN_DICT, Style, RE_HANS
)
from pypinyin.converter import DefaultConverter, UltimateConverter
from pypinyin.contrib.tone_sandhi import ToneSandhiMixin
from pypinyin.contrib.tone_convert import tone2_to_tone
from pypinyin.seg import mmseg
from pypinyin.seg.simpleseg import seg


def load_single_dict(pinyin_dict: Dict[int, str], style: str = 'default') -> None:
    """载入用户自定义的单字拼音库

    :param pinyin_dict: 单字拼音库。比如： {0x963F: "ā,ē"}
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

    :param phrases_dict: 词语拼音库。比如： { "阿爸": [["ā"], ["bà"]] }
    :param style: phrases_dict 参数值的拼音库风格. 支持 'default', 'tone2'
    """
    if style == 'tone2':
        for k, value in phrases_dict.items():
            v = [
                list(map(tone2_to_tone, pys))
                for pys in value
            ]
            PHRASES_DICT[k] = v
    else:
        PHRASES_DICT.update(phrases_dict)

    mmseg.retrain(mmseg.seg)


class Pinyin(object):
    def __init__(self, converter: Optional[Any] = None, **kwargs: Any) -> None:
        self._converter = converter or DefaultConverter()

    def pinyin(self,
               hans: Union[text_type, List[text_type]],
               style: Style = Style.TONE,
               heteronym: bool = False,
               errors: Union[str, Callable[[Any], Any]] = 'default',
               strict: bool = True,
               **kwargs: Any) -> List[Any]:
        """将汉字转换为拼音，返回汉字的拼音列表。

        :param hans: 汉字字符串( '你好吗' )或列表( ['你好', '吗'] ).
        :param style: 指定拼音风格，默认是 Style.TONE 风格。
        :param errors: 指定如何处理没有拼音的字符。
        :param heteronym: 是否启用多音字
        :param strict: 是否严格遵照《汉语拼音方案》来处理声母和韵母
        :return: 拼音列表
        """
        # 对字符串进行分词处理
        if isinstance(hans, text_type):
            han_list: List[Any] = self.seg(hans)
        else:
            if isinstance(self._converter, UltimateConverter) or \
               isinstance(self._converter, ToneSandhiMixin):
                han_list = []  # type: List[Any]
                for h in hans:
                    if not RE_HANS.match(h):
                        han_list.extend(self.seg(h))
                    else:
                        han_list.append(h)
            else:
                han_list = list(chain(*(self.seg(x) for x in hans)))  # type: List[Any]

        pys: List[Any] = []
        for words in han_list:
            pys.extend(
                self._converter.convert(
                    words, style, heteronym, errors, strict=strict))
        return pys

    def lazy_pinyin(self,
                    hans: Union[text_type, List[text_type]],
                    style: Style = Style.NORMAL,
                    errors: Union[str, Callable[[Any], Any]] = 'default',
                    strict: bool = True,
                    **kwargs: Any) -> List[str]:
        """将汉字转换为拼音，返回不包含多音字结果的拼音列表.

        :param hans: 汉字字符串( '你好吗' )或列表( ['你好', '吗'] ).
        :param style: 指定拼音风格，默认是 Style.NORMAL 风格。
        :param errors: 指定如何处理没有拼音的字符
        :param strict: 是否严格遵照《汉语拼音方案》来处理声母和韵母
        :return: 拼音列表，如 ['zhong', 'guo', 'ren']
        """
        return list(
            chain(
                *self.pinyin(
                    hans, style=style, heteronym=False,
                    errors=errors, strict=strict)))

    def pre_seg(self, hans: text_type, **kwargs: Any) -> Optional[List[Any]]:
        """对字符串进行分词前的预处理。

        :param hans: 分词前的字符串
        :return: None 或 分词后的列表
        """
        pass

    def seg(self, hans: text_type, **kwargs: Any) -> List[Any]:
        """对汉字进行分词。

        :param hans: 待分词的字符串
        :return: 分词后的结果列表
        """
        pre_data: Optional[List[Any]] = self.pre_seg(hans)
        if isinstance(pre_data, list):
            seg_data: List[Any] = pre_data
        else:
            seg_data = self.get_seg()(hans)

        post_data = self.post_seg(hans, seg_data)
        if isinstance(post_data, list):
            return post_data

        return seg_data

    def get_seg(self, **kwargs: Any) -> Callable[[text_type], List[text_type]]:
        """获取分词函数。

        :return: 分词函数
        """
        return seg

    def post_seg(self, hans: text_type, seg_data: List[Any], **kwargs: Any) -> Optional[List[Any]]:
        """对分词后的结果进行处理。

        :param hans: 分词前的字符串
        :param seg_data: 分词后的结果列表
        :return: 处理后的分词结果或 None
        """
        pass


_default_convert: DefaultConverter = DefaultConverter()
_default_pinyin: Pinyin = Pinyin(_default_convert)


def to_fixed(pinyin: str, style: Any, strict: bool = True) -> str:
    # 用于向后兼容，TODO: 废弃
    return _default_convert.convert_style(
        '', pinyin, style=style, strict=strict, default=pinyin)


_to_fixed = to_fixed


def handle_nopinyin(chars: Any,
                      errors: Union[str, Callable[[Any], Any]] = 'default',
                      heteronym: bool = True) -> Any:
    # 用于向后兼容，TODO: 废弃
    return _default_convert.handle_nopinyin(
        chars, style=None, errors=errors, heteronym=heteronym, strict=True)


def single_pinyin(han: Any,
                  style: Any,
                  heteronym: bool,
                  errors: Union[str, Callable[[Any], Any]] = 'default',
                  strict: bool = True) -> Any:
    # 用于向后兼容，TODO: 废弃
    return _default_convert._single_pinyin(
        han, style, heteronym, errors=errors, strict=strict)


def phrase_pinyin(phrase: Any,
                  style: Any,
                  heteronym: bool,
                  errors: Union[str, Callable[[Any], Any]] = 'default',
                  strict: bool = True) -> Any:
    # 用于向后兼容，TODO: 废弃
    return _default_convert._phrase_pinyin(
        phrase, style, heteronym, errors=errors, strict=strict)


def pinyin(hans: Union[text_type, List[text_type]],
           style: Style = Style.TONE,
           heteronym: bool = False,
           errors: Union[str, Callable[[Any], Any]] = 'default',
           strict: bool = True,
           v_to_u: bool = False,
           neutral_tone_with_five: bool = False) -> List[Any]:
    """将汉字转换为拼音，返回汉字的拼音列表。

    :param hans: 汉字字符串或字符串列表
    :param style: 指定拼音风格，默认是 Style.TONE 风格。
    :param heteronym: 是否启用多音字
    :param errors: 如何处理没有拼音的字符
    :param strict: 是否严格遵照《汉语拼音方案》
    :param v_to_u: 是否将 'v' 转换为 'ü'
    :param neutral_tone_with_five: 是否使用 5 标识轻声
    :return: 拼音列表
    """
    _pinyin = Pinyin(UltimateConverter(
        v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five))
    return _pinyin.pinyin(
        hans, style=style, heteronym=heteronym, errors=errors, strict=strict)


def slug(hans: Union[text_type, List[text_type]],
         style: Style = Style.NORMAL,
         heteronym: bool = False,
         separator: str = '-',
         errors: Union[str, Callable[[Any], Any]] = 'default',
         strict: bool = True) -> str:
    """将汉字转换为拼音，然后生成 slug 字符串.

    :param hans: 汉字字符串或字符串列表
    :param style: 指定拼音风格，默认是 Style.NORMAL 风格。
    :param heteronym: 是否启用多音字
    :param separator: 拼音之间的分隔符
    :param errors: 如何处理没有拼音的字符
    :param strict: 是否严格遵照《汉语拼音方案》
    :return: 生成的 slug 字符串
    """
    return separator.join(
        chain(
            *_default_pinyin.pinyin(
                hans, style=style, heteronym=heteronym,
                errors=errors, strict=strict
            )
        )
    )


def lazy_pinyin(hans: Union[text_type, List[text_type]],
                style: Style = Style.NORMAL,
                errors: Union[str, Callable[[Any], Any]] = 'default',
                strict: bool = True,
                v_to_u: bool = False,
                neutral_tone_with_five: bool = False,
                tone_sandhi: bool = False) -> List[str]:
    """将汉字转换为拼音，返回不包含多音字结果的拼音列表.

    :param hans: 汉字字符串或字符串列表
    :param style: 指定拼音风格，默认是 Style.NORMAL 风格。
    :param errors: 如何处理没有拼音的字符
    :param strict: 是否严格遵照《汉语拼音方案》
    :param v_to_u: 是否将 'v' 转换为 'ü'
    :param neutral_tone_with_five: 是否使用 5 标识轻声
    :param tone_sandhi: 是否按照声调变调规则处理拼音
    :return: 拼音列表，如 ['zhong', 'guo', 'ren']
    """
    _pinyin = Pinyin(UltimateConverter(
        v_to_u=v_to_u,
        neutral_tone_with_five=neutral_tone_with_five,
        tone_sandhi=tone_sandhi))
    return _pinyin.lazy_pinyin(
        hans, style=style, errors=errors, strict=strict)