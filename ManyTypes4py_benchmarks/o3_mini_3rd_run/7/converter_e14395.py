from __future__ import unicode_literals
from copy import deepcopy
from typing import List, Optional, Any, Union, Callable, Dict
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
    def convert(self, words: str, style: Any, heteronym: bool, errors: Union[str, Callable[[str], Any]], strict: bool, **kwargs: Any) -> List[Any]:
        raise NotImplementedError


class DefaultConverter(Converter):
    def __init__(self, **kwargs: Any) -> None:
        pass

    def convert(self, words: str, style: Any, heteronym: bool, errors: Union[str, Callable[[str], Any]], strict: bool, **kwargs: Any) -> List[List[str]]:
        pys: List[Any] = []
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

    def pre_convert_style(self, han: str, orig_pinyin: str, style: Any, strict: bool, **kwargs: Any) -> Optional[str]:
        # Custom implementation may return a modified pinyin string.
        pass

    def convert_style(self, han: str, orig_pinyin: str, style: Any, strict: bool, **kwargs: Any) -> str:
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

    def post_convert_style(self, han: str, orig_pinyin: str, converted_pinyin: str, style: Any, strict: bool, **kwargs: Any) -> Optional[str]:
        # Custom implementation may return a modified pinyin string.
        pass

    def pre_handle_nopinyin(self, chars: str, style: Any, heteronym: bool, errors: Union[str, Callable[[str], Any]], strict: bool, **kwargs: Any) -> Optional[Any]:
        # Custom implementation may return a modified value to handle non-pinyin characters.
        pass

    def handle_nopinyin(self, chars: str, style: Any, heteronym: bool, errors: Union[str, Callable[[str], Any]], strict: bool, **kwargs: Any) -> List[List[str]]:
        pre_data: Optional[Any] = self.pre_handle_nopinyin(chars, style, errors=errors, heteronym=heteronym, strict=strict)
        if pre_data is not None:
            py: Any = pre_data
        else:
            pre_data = chars
            py = self._convert_nopinyin_chars(pre_data, style, errors=errors, heteronym=heteronym, strict=strict)
        post_data: Optional[Any] = self.post_handle_nopinyin(chars, style, errors=errors, heteronym=heteronym, strict=strict, pinyin=py)
        if post_data is not None:
            py = post_data
        if not py:
            return []
        if isinstance(py, list):
            if py and isinstance(py[0], list):
                if heteronym:
                    return py
                return [[x[0]] for x in py]
            return [[i] for i in py]
        else:
            return [[py]]

    def post_handle_nopinyin(self, chars: str, style: Any, heteronym: bool, errors: Union[str, Callable[[str], Any]], strict: bool, pinyin: Any, **kwargs: Any) -> Optional[Any]:
        # Custom implementation may return a modified value for non-pinyin characters.
        pass

    def post_pinyin(self, han: str, heteronym: bool, pinyin: List[List[str]], **kwargs: Any) -> Optional[List[List[str]]]:
        # Custom implementation may return a modified pinyin list.
        pass

    def _phrase_pinyin(self, phrase: str, style: Any, heteronym: bool, errors: Union[str, Callable[[str], Any]], strict: bool) -> List[List[str]]:
        pinyin_list: List[Any] = []
        if phrase in PHRASES_DICT:
            pinyin_list = deepcopy(PHRASES_DICT[phrase])
        else:
            for han in phrase:
                py: List[List[str]] = self._single_pinyin(han, style, heteronym, errors, strict)
                pinyin_list.extend(py)
        return pinyin_list

    def convert_styles(self, pinyin_list: List[List[str]], phrase: str, style: Any, heteronym: bool, errors: Union[str, Callable[[str], Any]], strict: bool, **kwargs: Any) -> List[List[str]]:
        for idx, item in enumerate(pinyin_list):
            han: str = phrase[idx]
            if heteronym:
                pinyin_list[idx] = [self.convert_style(han, orig_pinyin=x, style=style, strict=strict) for x in item]
            else:
                orig_pinyin: str = item[0]
                pinyin_list[idx] = [self.convert_style(han, orig_pinyin=orig_pinyin, style=style, strict=strict)]
        return pinyin_list

    def _single_pinyin(self, han: str, style: Any, heteronym: bool, errors: Union[str, Callable[[str], Any]], strict: bool) -> List[List[str]]:
        num: int = ord(han)
        if num not in PINYIN_DICT:
            return self.handle_nopinyin(han, style=style, errors=errors, heteronym=heteronym, strict=strict)
        pys: List[str] = PINYIN_DICT[num].split(',')
        return [pys]

    def _convert_style(self, han: str, pinyin: str, style: Any, strict: bool, default: str, **kwargs: Any) -> str:
        kwargs['han'] = han
        return convert_style(pinyin, style, strict, default=default, **kwargs)

    def _convert_nopinyin_chars(self, chars: str, style: Any, heteronym: bool, errors: Union[str, Callable[[str], Any]], strict: bool) -> Optional[Union[str, List[str]]]:
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
        return None


class _v2UConverter(V2UMixin, DefaultConverter):
    pass


class _neutralToneWith5Converter(NeutralToneWith5Mixin, DefaultConverter):
    pass


class _toneSandhiConverter(ToneSandhiMixin, DefaultConverter):
    pass


class UltimateConverter(DefaultConverter):
    def __init__(self, v_to_u: bool = False, neutral_tone_with_five: bool = False, tone_sandhi: bool = False, **kwargs: Any) -> None:
        super(UltimateConverter, self).__init__(**kwargs)
        self._v_to_u: bool = v_to_u
        self._neutral_tone_with_five: bool = neutral_tone_with_five
        self._tone_sandhi: bool = tone_sandhi

    def post_convert_style(self, han: str, orig_pinyin: str, converted_pinyin: str, style: Any, strict: bool, **kwargs: Any) -> str:
        post_data: Optional[str] = super(UltimateConverter, self).post_convert_style(han, orig_pinyin, converted_pinyin, style, strict, **kwargs)
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

    def post_pinyin(self, han: str, heteronym: bool, pinyin: List[List[str]], **kwargs: Any) -> List[List[str]]:
        post_data: Optional[List[List[str]]] = super(UltimateConverter, self).post_pinyin(han, heteronym, pinyin, **kwargs)
        if post_data is not None:
            pinyin = post_data
        if self._tone_sandhi:
            post_data = _toneSandhiConverter().post_pinyin(han, heteronym, pinyin, **kwargs)
            if post_data is not None:
                pinyin = post_data
        return pinyin

_mixConverter = UltimateConverter
