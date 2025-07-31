from __future__ import unicode_literals
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union
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
    def convert(
        self,
        words: str,
        style: Any,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any
    ) -> List[List[str]]:
        raise NotImplementedError

class DefaultConverter(Converter):
    def __init__(self, **kwargs: Any) -> None:
        pass

    def convert(
        self,
        words: str,
        style: Any,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any
    ) -> List[List[str]]:
        """根据参数把汉字转成相应风格的拼音结果。

        :param words: 汉字字符串
        :type words: unicode
        :param style: 拼音风格
        :param heteronym: 是否启用多音字
        :type heteronym: bool
        :param errors: 如果处理没有拼音的字符
        :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                       是否严格遵照《汉语拼音方案》来处理声母和韵母，
                       详见 :ref:`strict`
        :type strict: bool
        :return: 按风格转换后的拼音结果
        :rtype: list
        """
        pys: List[List[str]] = []
        if RE_HANS.match(words):
            pys = self._phrase_pinyin(words, style=style, heteronym=heteronym, errors=errors, strict=strict)
            post_data = self.post_pinyin(words, heteronym, pys)
            if post_data is not None:
                pys = post_data
            pys = self.convert_styles(pys, words, style, heteronym, errors, strict)
        else:
            py = self.handle_nopinyin(words, style=style, errors=errors, heteronym=heteronym, strict=strict)
            if py:
                pys.extend(py)
        return _remove_dup_and_empty(pys)

    def pre_convert_style(
        self,
        han: str,
        orig_pinyin: str,
        style: Any,
        strict: bool,
        **kwargs: Any
    ) -> Optional[str]:
        """在把原始带声调的拼音按拼音风格转换前会调用 ``pre_convert_style`` 方法。

        如果返回值不为 ``None`` 会使用返回的结果代替 ``orig_pinyin``
        来进行后面的风格转换。
        """
        pass

    def convert_style(
        self,
        han: str,
        orig_pinyin: str,
        style: Any,
        strict: bool,
        **kwargs: Any
    ) -> str:
        """按 ``style`` 的值对 ``orig_pinyin`` 进行处理，返回处理后的拼音

        转换风格前会调用 ``pre_convert_style`` 方法，
        转换后会调用 ``post_convert_style`` 方法。
        """
        pre_data: Optional[str] = self.pre_convert_style(han, orig_pinyin, style=style, strict=strict)
        if pre_data is not None:
            pinyin: str = pre_data
        else:
            pinyin = orig_pinyin
        converted_pinyin: str = self._convert_style(han, pinyin, style=style, strict=strict, default=pinyin)
        post_data: Optional[str] = self.post_convert_style(han, pinyin, converted_pinyin, style=style, strict=strict)
        if post_data is None:
            post_data = converted_pinyin
        return post_data

    def post_convert_style(
        self,
        han: str,
        orig_pinyin: str,
        converted_pinyin: str,
        style: Any,
        strict: bool,
        **kwargs: Any
    ) -> Optional[str]:
        """在把原始带声调的拼音按拼音风格转换前会调用 ``pre_convert_style`` 方法。

        如果返回值不为 ``None`` 会使用返回的结果代替 ``converted_pinyin``
        作为拼音风格转换后的最终拼音结果。
        """
        pass

    def pre_handle_nopinyin(
        self,
        chars: str,
        style: Any,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any
    ) -> Optional[Union[str, List[List[str]]]]:
        """处理没有拼音的字符串前会调用 ``pre_handle_nopinyin`` 方法。

        如果返回值不为 ``None`` 会使用返回的结果作为处理没有拼音字符串的结果，
        不再使用内置方法进行处理。
        """
        pass

    def handle_nopinyin(
        self,
        chars: str,
        style: Any,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any
    ) -> List[List[str]]:
        """处理没有拼音的字符串。

        处理前会调用 ``pre_handle_nopinyin`` 方法，
        处理后会调用 ``post_handle_nopinyin`` 方法。
        """
        pre_data: Optional[Union[str, List[List[str]]]] = self.pre_handle_nopinyin(chars, style, heteronym, errors=errors, strict=strict)
        if pre_data is not None:
            py: Union[str, List[List[str]]] = pre_data
        else:
            pre_data = chars
            py = self._convert_nopinyin_chars(pre_data, style, errors=errors, heteronym=heteronym, strict=strict)
        post_data: Optional[List[List[str]]] = self.post_handle_nopinyin(chars, style, errors=errors, heteronym=heteronym, strict=strict, pinyin=py)
        if post_data is not None:
            py = post_data
        if not py:
            return []
        if isinstance(py, list):
            if isinstance(py[0], list):
                if heteronym:
                    return py
                return [[x[0]] for x in py]
            return [[i] for i in py]
        else:
            return [[py]]

    def post_handle_nopinyin(
        self,
        chars: str,
        style: Any,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        pinyin: Union[str, List[List[str]]],
        **kwargs: Any
    ) -> Optional[List[List[str]]]:
        """处理完没有拼音的字符串后会调用 ``post_handle_nopinyin`` 方法。

        如果返回值不为 ``None`` 会使用返回的结果作为处理没有拼音的字符串的结果。
        """
        pass

    def post_pinyin(
        self,
        han: str,
        heteronym: bool,
        pinyin: List[Any],
        **kwargs: Any
    ) -> Optional[List[Any]]:
        """找到汉字对应的拼音后，会调用 ``post_pinyin`` 方法。

        如果返回值不为 ``None`` 会使用返回的结果作为 han 的拼音数据。
        """
        pass

    def _phrase_pinyin(
        self,
        phrase: str,
        style: Any,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool
    ) -> List[List[str]]:
        """词语拼音转换.
        """
        pinyin_list: List[List[str]] = []
        if phrase in PHRASES_DICT:
            pinyin_list = deepcopy(PHRASES_DICT[phrase])
        else:
            for han in phrase:
                py: List[List[str]] = self._single_pinyin(han, style, heteronym, errors, strict)
                pinyin_list.extend(py)
        return pinyin_list

    def convert_styles(
        self,
        pinyin_list: List[List[str]],
        phrase: str,
        style: Any,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any
    ) -> List[List[str]]:
        """转换多个汉字的拼音结果的风格"""
        for idx, item in enumerate(pinyin_list):
            han: str = phrase[idx]
            if heteronym:
                pinyin_list[idx] = [self.convert_style(han, orig_pinyin=x, style=style, strict=strict) for x in item]  # type: ignore
            else:
                orig_pinyin: str = item[0]
                pinyin_list[idx] = [self.convert_style(han, orig_pinyin=orig_pinyin, style=style, strict=strict)]
        return pinyin_list

    def _single_pinyin(
        self,
        han: str,
        style: Any,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool
    ) -> List[List[str]]:
        """单字拼音转换.
        """
        num: int = ord(han)
        if num not in PINYIN_DICT:
            return self.handle_nopinyin(han, style=style, errors=errors, heteronym=heteronym, strict=strict)
        pys: List[str] = PINYIN_DICT[num].split(',')
        return [pys]

    def _convert_style(
        self,
        han: str,
        pinyin: str,
        style: Any,
        strict: bool,
        default: str,
        **kwargs: Any
    ) -> str:
        kwargs['han'] = han
        return convert_style(pinyin, style, strict, default=default, **kwargs)

    def _convert_nopinyin_chars(
        self,
        chars: str,
        style: Any,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool
    ) -> Any:
        """转换没有拼音的字符。
        """
        if callable_check(errors):
            return errors(chars)
        if errors == 'default':
            return chars
        elif errors == 'ignore':
            return None
        elif errors == 'replace':
            if len(chars) > 1:
                return ''.join((text_type('%x' % ord(x)) for x in chars))
            else:
                return text_type('%x' % ord(chars))

class _v2UConverter(V2UMixin, DefaultConverter):
    pass

class _neutralToneWith5Converter(NeutralToneWith5Mixin, DefaultConverter):
    pass

class _toneSandhiConverter(ToneSandhiMixin, DefaultConverter):
    pass

class UltimateConverter(DefaultConverter):
    def __init__(
        self,
        v_to_u: bool = False,
        neutral_tone_with_five: bool = False,
        tone_sandhi: bool = False,
        **kwargs: Any
    ) -> None:
        super(UltimateConverter, self).__init__(**kwargs)
        self._v_to_u: bool = v_to_u
        self._neutral_tone_with_five: bool = neutral_tone_with_five
        self._tone_sandhi: bool = tone_sandhi

    def post_convert_style(
        self,
        han: str,
        orig_pinyin: str,
        converted_pinyin: str,
        style: Any,
        strict: bool,
        **kwargs: Any
    ) -> str:
        post_data: Optional[str] = super(UltimateConverter, self).post_convert_style(
            han, orig_pinyin, converted_pinyin, style, strict, **kwargs
        )
        if post_data is not None:
            converted_pinyin = post_data
        if self._v_to_u:
            post_data = _v2UConverter().post_convert_style(han, orig_pinyin, converted_pinyin, style, strict, **kwargs)
            if post_data is not None:
                converted_pinyin = post_data
        if self._neutral_tone_with_five:
            post_data = _neutralToneWith5Converter().post_convert_style(han, orig_pinyin, converted_pinyin, style, strict, **kwargs)
            if post_data is not None:
                converted_pinyin = post_data
        return converted_pinyin

    def post_pinyin(
        self,
        han: str,
        heteronym: bool,
        pinyin: List[Any],
        **kwargs: Any
    ) -> List[Any]:
        post_data: Optional[List[Any]] = super(UltimateConverter, self).post_pinyin(han, heteronym, pinyin, **kwargs)
        if post_data is not None:
            pinyin = post_data
        if self._tone_sandhi:
            post_data = _toneSandhiConverter().post_pinyin(han, heteronym, pinyin, **kwargs)
            if post_data is not None:
                pinyin = post_data
        return pinyin

_mixConverter = UltimateConverter