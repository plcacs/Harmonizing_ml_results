from __future__ import unicode_literals
from typing import Any, Callable, List, Optional, Union

from pypinyin.contrib.uv import V2UMixin
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.contrib.tone_sandhi import ToneSandhiMixin


class Converter(object):
    def convert(
        self,
        words: str,
        style: int,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any,
    ) -> List[List[str]]: ...


class DefaultConverter(Converter):
    def __init__(self, **kwargs: Any) -> None: ...

    def convert(
        self,
        words: str,
        style: int,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any,
    ) -> List[List[str]]: ...

    def pre_convert_style(
        self,
        han: str,
        orig_pinyin: str,
        style: int,
        strict: bool,
        **kwargs: Any,
    ) -> Optional[str]: ...

    def convert_style(
        self,
        han: str,
        orig_pinyin: str,
        style: int,
        strict: bool,
        **kwargs: Any,
    ) -> str: ...

    def post_convert_style(
        self,
        han: str,
        orig_pinyin: str,
        converted_pinyin: str,
        style: int,
        strict: bool,
        **kwargs: Any,
    ) -> Optional[str]: ...

    def pre_handle_nopinyin(
        self,
        chars: str,
        style: int,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any,
    ) -> Optional[Union[str, List[Any]]]: ...

    def handle_nopinyin(
        self,
        chars: str,
        style: int,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any,
    ) -> List[List[str]]: ...

    def post_handle_nopinyin(
        self,
        chars: str,
        style: int,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        pinyin: Union[List[Any], str, None],
        **kwargs: Any,
    ) -> Optional[Union[str, List[Any]]]: ...

    def post_pinyin(
        self,
        han: str,
        heteronym: bool,
        pinyin: List[List[str]],
        **kwargs: Any,
    ) -> Optional[List[List[str]]]: ...

    def _phrase_pinyin(
        self,
        phrase: str,
        style: int,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
    ) -> List[List[str]]: ...

    def convert_styles(
        self,
        pinyin_list: List[List[str]],
        phrase: str,
        style: int,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any,
    ) -> List[List[str]]: ...

    def _single_pinyin(
        self,
        han: str,
        style: int,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
    ) -> List[List[str]]: ...

    def _convert_style(
        self,
        han: str,
        pinyin: str,
        style: int,
        strict: bool,
        default: str,
        **kwargs: Any,
    ) -> str: ...

    def _convert_nopinyin_chars(
        self,
        chars: str,
        style: int,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
    ) -> Optional[str]: ...


class _v2UConverter(V2UMixin, DefaultConverter): ...

class _neutralToneWith5Converter(NeutralToneWith5Mixin, DefaultConverter): ...

class _toneSandhiConverter(ToneSandhiMixin, DefaultConverter): ...


class UltimateConverter(DefaultConverter):
    _v_to_u: bool
    _neutral_tone_with_five: bool
    _tone_sandhi: bool

    def __init__(
        self,
        v_to_u: bool = ...,
        neutral_tone_with_five: bool = ...,
        tone_sandhi: bool = ...,
        **kwargs: Any,
    ) -> None: ...

    def post_convert_style(
        self,
        han: str,
        orig_pinyin: str,
        converted_pinyin: str,
        style: int,
        strict: bool,
        **kwargs: Any,
    ) -> Optional[str]: ...

    def post_pinyin(
        self,
        han: str,
        heteronym: bool,
        pinyin: List[List[str]],
        **kwargs: Any,
    ) -> Optional[List[List[str]]]: ...

_mixConverter = UltimateConverter