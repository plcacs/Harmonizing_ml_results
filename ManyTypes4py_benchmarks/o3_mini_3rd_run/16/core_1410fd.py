#!/usr/bin/env python
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
    def __init__(self, converter: Optional[DefaultConverter] = None, **kwargs: Any) -> None:
        self._converter: DefaultConverter = converter or DefaultConverter()

    def pinyin(
        self,
        hans: Union[text_type, List[text_type]],
        style: int = Style.TONE,
        heteronym: bool = False,
        errors: Union[str, Callable[[str, Any], Union[str, List[str]]]] = 'default',
        strict: bool = True,
        **kwargs: Any,
    ) -> List[List[str]]:
        """将汉字转换为拼音，返回汉字的拼音列表。

        :param hans: 汉字字符串( '你好吗' )或列表( ['你好', '吗'] ).
        :param style: 指定拼音风格，默认是 Style.TONE 风格。
        :param errors: 指定如何处理没有拼音的字符。
        :param heteronym: 是否启用多音字
        :param strict: 是否严格遵照《汉语拼音方案》来处理声母和韵母
        :return: 拼音列表
        """
        if isinstance(hans, text_type):
            han_list: List[str] = self.seg(hans)
        elif isinstance(self._converter, UltimateConverter) or isinstance(self._converter, ToneSandhiMixin):
            han_list = []  # type: List[str]
            for h in hans:
                if not RE_HANS.match(h):
                    han_list.extend(self.seg(h))
                else:
                    han_list.append(h)
        else:
            han_list = list(chain(*(self.seg(x) for x in hans)))
        pys: List[List[str]] = []
        for words in han_list:
            # Each convert call returns a list of pinyin(s) for the input.
            converted = self._converter.convert(words, style, heteronym, errors, strict=strict)
            pys.extend(converted)
        return pys

    def lazy_pinyin(
        self,
        hans: Union[text_type, List[text_type]],
        style: int = Style.NORMAL,
        errors: Union[str, Callable[[str, Any], Union[str, List[str]]]] = 'default',
        strict: bool = True,
        **kwargs: Any,
    ) -> List[str]:
        """将汉字转换为拼音，返回不包含多音字结果的拼音列表.

        :param hans: 汉字字符串( '你好吗' )或列表( ['你好', '吗'] ).
        :param style: 指定拼音风格，默认是 Style.NORMAL 风格。
        :param errors: 指定如何处理没有拼音的字符
        :param strict: 是否严格遵照《汉语拼音方案》来处理声母和韵母
        :return: 拼音列表
        """
        return list(chain(*self.pinyin(hans, style=style, heteronym=False, errors=errors, strict=strict)))

    def pre_seg(self, hans: text_type, **kwargs: Any) -> Optional[List[str]]:
        """对字符串进行分词前的预处理。

        :param hans: 分词前的字符串
        :return: None 或 分词后的列表
        """
        return None

    def seg(self, hans: text_type, **kwargs: Any) -> List[str]:
        """对汉字进行分词。

        :param hans: 输入的汉字字符串
        :return: 分词后的结果列表
        """
        pre_data: Optional[List[str]] = self.pre_seg(hans, **kwargs)
        if isinstance(pre_data, list):
            seg_data: List[str] = pre_data
        else:
            seg_data = self.get_seg()(hans)
        post_data: Optional[List[str]] = self.post_seg(hans, seg_data, **kwargs)
        if isinstance(post_data, list):
            return post_data
        return seg_data

    def get_seg(self, **kwargs: Any) -> Callable[[text_type], List[str]]:
        """获取分词函数。

        :return: 分词函数
        """
        return seg

    def post_seg(self, hans: text_type, seg_data: List[str], **kwargs: Any) -> Optional[List[str]]:
        """对分词后的结果做处理。

        :param hans: 分词前的字符串
        :param seg_data: 分词后的结果列表
        :return: None 或 处理后的分词列表
        """
        return None


_default_convert: DefaultConverter = DefaultConverter()
_default_pinyin: Pinyin = Pinyin(_default_convert)


def to_fixed(pinyin: str, style: int, strict: bool = True) -> Any:
    return _default_convert.convert_style('', pinyin, style=style, strict=strict, default=pinyin)


_to_fixed = to_fixed


def handle_nopinyin(
    chars: Any, errors: Union[str, Callable[[str, Any], Union[str, List[str]]]] = 'default', heteronym: bool = True
) -> Any:
    return _default_convert.handle_nopinyin(chars, style=None, errors=errors, heteronym=heteronym, strict=True)


def single_pinyin(
    han: str,
    style: int,
    heteronym: bool,
    errors: Union[str, Callable[[str, Any], Union[str, List[str]]]] = 'default',
    strict: bool = True,
) -> List[str]:
    return _default_convert._single_pinyin(han, style, heteronym, errors=errors, strict=strict)


def phrase_pinyin(
    phrase: str,
    style: int,
    heteronym: bool,
    errors: Union[str, Callable[[str, Any], Union[str, List[str]]]] = 'default',
    strict: bool = True,
) -> Any:
    return _default_convert._phrase_pinyin(phrase, style, heteronym, errors=errors, strict=strict)


def pinyin(
    hans: Union[text_type, List[text_type]],
    style: int = Style.TONE,
    heteronym: bool = False,
    errors: Union[str, Callable[[str, Any], Union[str, List[str]]]] = 'default',
    strict: bool = True,
    v_to_u: bool = False,
    neutral_tone_with_five: bool = False,
) -> List[List[str]]:
    """将汉字转换为拼音，返回汉字的拼音列表。

    :param hans: 汉字字符串( '你好吗' )或列表( ['你好', '吗'] ).
    :param style: 指定拼音风格，默认是 Style.TONE 风格。
    :param heteronym: 是否启用多音字
    :param errors: 指定如何处理没有拼音的字符。
    :param strict: 是否严格遵照《汉语拼音方案》来处理声母和韵母
    :param v_to_u: 是否用 'ü' 代替 'v'
    :param neutral_tone_with_five: 是否使用 5 标识轻声
    :return: 拼音列表
    """
    _pinyin = Pinyin(
        UltimateConverter(v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five)
    )
    return _pinyin.pinyin(hans, style=style, heteronym=heteronym, errors=errors, strict=strict)


def slug(
    hans: Union[text_type, List[text_type]],
    style: int = Style.NORMAL,
    heteronym: bool = False,
    separator: str = '-',
    errors: Union[str, Callable[..., Any]] = 'default',
    strict: bool = True,
) -> str:
    """将汉字转换为拼音，然后生成 slug 字符串.

    :param hans: 汉字字符串( '你好吗' )或列表( ['你好', '吗'] ).
    :param style: 指定拼音风格，默认是 Style.NORMAL 风格。
    :param heteronym: 是否启用多音字
    :param separator: 两个拼音间的分隔符/连接符
    :param errors: 指定如何处理没有拼音的字符
    :param strict: 是否严格遵照《汉语拼音方案》来处理声母和韵母
    :return: slug 字符串
    """
    return separator.join(
        chain(*_default_pinyin.pinyin(hans, style=style, heteronym=heteronym, errors=errors, strict=strict))
    )


def lazy_pinyin(
    hans: Union[text_type, List[text_type]],
    style: int = Style.NORMAL,
    errors: Union[str, Callable[[str, Any], Union[str, List[str]]]] = 'default',
    strict: bool = True,
    v_to_u: bool = False,
    neutral_tone_with_five: bool = False,
    tone_sandhi: bool = False,
) -> List[str]:
    """将汉字转换为拼音，返回不包含多音字结果的拼音列表.

    :param hans: 汉字字符串( '你好吗' )或列表( ['你好', '吗'] ).
    :param style: 指定拼音风格，默认是 Style.NORMAL 风格。
    :param errors: 指定如何处理没有拼音的字符
    :param strict: 是否严格遵照《汉语拼音方案》来处理声母和韵母
    :param v_to_u: 是否用 'ü' 代替 'v'
    :param neutral_tone_with_five: 是否使用 5 标识轻声
    :param tone_sandhi: 是否对拼音进行变调处理
    :return: 拼音列表
    """
    _pinyin = Pinyin(
        UltimateConverter(
            v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five, tone_sandhi=tone_sandhi
        )
    )
    return _pinyin.lazy_pinyin(hans, style=style, errors=errors, strict=strict)