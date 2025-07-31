#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from itertools import chain
from typing import List, Dict, Union, Callable, Any, Optional
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
    def __init__(self, converter: Optional[Any] = None, **kwargs: Any) -> None:
        self._converter = converter or DefaultConverter()

    def pinyin(self,
               hans: Union[str, List[str]],
               style: str = Style.TONE,
               heteronym: bool = False,
               errors: Union[str, Callable[[Any], Any]] = 'default',
               strict: bool = True,
               **kwargs: Any) -> List[List[str]]:
        """将汉字转换为拼音，返回汉字的拼音列表。

        :param hans: 汉字字符串( '你好吗' )或列表( ['你好', '吗'] ).
        :param style: 指定拼音风格，默认是 Style.TONE 风格。
        :param errors: 指定如何处理没有拼音的字符。
        :param heteronym: 是否启用多音字
        :param strict: 是否严格遵照《汉语拼音方案》来处理声母和韵母
        :return: 拼音列表
        """
        if isinstance(hans, text_type):
            han_list = self.seg(hans)
        elif isinstance(self._converter, UltimateConverter) or isinstance(self._converter, ToneSandhiMixin):
            han_list: List[str] = []
            for h in hans:
                if not RE_HANS.match(h):
                    han_list.extend(self.seg(h))
                else:
                    han_list.append(h)
        else:
            han_list = list(chain(*(self.seg(x) for x in hans)))
        pys: List[List[str]] = []
        for words in han_list:
            pys.extend(self._converter.convert(words, style, heteronym, errors, strict=strict))
        return pys

    def lazy_pinyin(self,
                    hans: Union[str, List[str]],
                    style: str = Style.NORMAL,
                    errors: Union[str, Callable[[Any], Any]] = 'default',
                    strict: bool = True,
                    **kwargs: Any) -> List[str]:
        """将汉字转换为拼音，返回不包含多音字结果的拼音列表.

        :param hans: 汉字字符串( '你好吗' )或列表( ['你好', '吗'] ).
        :param style: 指定拼音风格，默认是 Style.NORMAL 风格。
        :param errors: 指定如何处理没有拼音的字符。
        :param strict: 是否严格遵照《汉语拼音方案》来处理声母和韵母
        :return: 拼音列表 (例如 ['zhong', 'guo', 'ren'])
        """
        return list(chain(*self.pinyin(hans, style=style, heteronym=False, errors=errors, strict=strict)))

    def pre_seg(self, hans: str, **kwargs: Any) -> Optional[List[str]]:
        """对字符串进行分词前的预处理。

        :param hans: 分词前的字符串
        :return: 如果返回列表则表示已经分词；否则返回 None
        """
        pass

    def seg(self, hans: str, **kwargs: Any) -> List[str]:
        """对汉字进行分词。

        :param hans: 汉字字符串
        :return: 分词结果列表
        """
        pre_data = self.pre_seg(hans)
        if isinstance(pre_data, list):
            seg_data = pre_data
        else:
            seg_data = self.get_seg()(hans)
        post_data = self.post_seg(hans, seg_data)
        if isinstance(post_data, list):
            return post_data
        return seg_data

    def get_seg(self, **kwargs: Any) -> Callable[[str], List[str]]:
        """获取分词函数。

        :return: 分词函数
        """
        return seg

    def post_seg(self, hans: str, seg_data: List[str], **kwargs: Any) -> Optional[List[str]]:
        """对分词后的结果进行后处理。

        :param hans: 分词前的字符串
        :param seg_data: 分词后的结果
        :return: 如果返回列表则使用该列表，否则返回原始 seg_data
        """
        pass

_default_convert: DefaultConverter = DefaultConverter()
_default_pinyin: Pinyin = Pinyin(_default_convert)

def to_fixed(pinyin: str, style: str, strict: bool = True) -> str:
    return _default_convert.convert_style('', pinyin, style=style, strict=strict, default=pinyin)

_to_fixed = to_fixed

def handle_nopinyin(chars: str,
                     errors: Union[str, Callable[[Any], Any]] = 'default',
                     heteronym: bool = True) -> List[List[str]]:
    return _default_convert.handle_nopinyin(chars, style=None, errors=errors, heteronym=heteronym, strict=True)

def single_pinyin(han: str,
                  style: str,
                  heteronym: bool,
                  errors: Union[str, Callable[[Any], Any]] = 'default',
                  strict: bool = True) -> List[str]:
    return _default_convert._single_pinyin(han, style, heteronym, errors=errors, strict=strict)

def phrase_pinyin(phrase: str,
                  style: str,
                  heteronym: bool,
                  errors: Union[str, Callable[[Any], Any]] = 'default',
                  strict: bool = True) -> List[str]:
    return _default_convert._phrase_pinyin(phrase, style, heteronym, errors=errors, strict=strict)

def pinyin(hans: Union[str, List[str]],
           style: str = Style.TONE,
           heteronym: bool = False,
           errors: Union[str, Callable[[Any], Any]] = 'default',
           strict: bool = True,
           v_to_u: bool = False,
           neutral_tone_with_five: bool = False) -> List[List[str]]:
    """将汉字转换为拼音，返回汉字的拼音列表。

    :param hans: 汉字字符串( '你好吗' )或列表( ['你好', '吗'] ).
    :param style: 指定拼音风格，默认是 Style.TONE 风格。
    :param errors: 指定如何处理没有拼音的字符。
    :param heteronym: 是否启用多音字
    :param strict: 是否严格遵照《汉语拼音方案》来处理声母和韵母
    :param v_to_u: 无声调拼音是否将 'v' 转换为 'ü'
    :param neutral_tone_with_five: 数字调拼音中是否用 '5' 表示轻声
    :return: 拼音列表
    """
    _pinyin = Pinyin(UltimateConverter(v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five))
    return _pinyin.pinyin(hans, style=style, heteronym=heteronym, errors=errors, strict=strict)

def slug(hans: Union[str, List[str]],
         style: str = Style.NORMAL,
         heteronym: bool = False,
         separator: str = '-',
         errors: Union[str, Callable[[Any], Any]] = 'default',
         strict: bool = True) -> str:
    """将汉字转换为拼音，然后生成 slug 字符串.

    :param hans: 汉字字符串( '你好吗' )或列表( ['你好', '吗'] ).
    :param style: 指定拼音风格，默认是 Style.NORMAL 风格。
    :param heteronym: 是否启用多音字
    :param separator: 分隔符
    :param errors: 指定如何处理没有拼音的字符。
    :param strict: 是否严格遵照《汉语拼音方案》来处理声母和韵母
    :return: slug 字符串
    """
    return separator.join(chain(*_default_pinyin.pinyin(hans, style=style, heteronym=heteronym, errors=errors, strict=strict)))

def lazy_pinyin(hans: Union[str, List[str]],
                style: str = Style.NORMAL,
                errors: Union[str, Callable[[Any], Any]] = 'default',
                strict: bool = True,
                v_to_u: bool = False,
                neutral_tone_with_five: bool = False,
                tone_sandhi: bool = False) -> List[str]:
    """将汉字转换为拼音，返回不包含多音字结果的拼音列表.

    :param hans: 汉字字符串( '你好吗' )或列表( ['你好', '吗'] ).
    :param style: 指定拼音风格，默认是 Style.NORMAL 风格。
    :param errors: 指定如何处理没有拼音的字符。
    :param strict: 是否严格遵照《汉语拼音方案》来处理声母和韵母
    :param v_to_u: 无声调拼音是否将 'v' 转换为 'ü'
    :param neutral_tone_with_five: 数字调拼音中是否用 '5' 表示轻声
    :param tone_sandhi: 是否按照声调变调规则处理拼音
    :return: 拼音列表 (例如 ['zhong', 'guo', 'ren'])
    """
    _pinyin = Pinyin(UltimateConverter(v_to_u=v_to_u,
                                        neutral_tone_with_five=neutral_tone_with_five,
                                        tone_sandhi=tone_sandhi))
    return _pinyin.lazy_pinyin(hans, style=style, errors=errors, strict=strict)
