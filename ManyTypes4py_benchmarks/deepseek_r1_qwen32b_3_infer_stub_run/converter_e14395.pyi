from __future__ import unicode_literals
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pypinyin.compat import text_type
from pypinyin.style import convert as convert_style

class Converter:
    def convert(self, words: str, style: Any, heteronym: bool, errors: str, strict: bool, **kwargs: Any) -> List[str]:
        ...

class DefaultConverter(Converter):
    def __init__(self, **kwargs: Any) -> None:
        ...

    def convert(self, words: str, style: Any, heteronym: bool, errors: str, strict: bool, **kwargs: Any) -> List[str]:
        ...

    def pre_convert_style(self, han: str, orig_pinyin: str, style: Any, strict: bool, **kwargs: Any) -> Optional[str]:
        ...

    def convert_style(self, han: str, orig_pinyin: str, style: Any, strict: bool, **kwargs: Any) -> str:
        ...

    def post_convert_style(self, han: str, orig_pinyin: str, converted_pinyin: str, style: Any, strict: bool, **kwargs: Any) -> Optional[str]:
        ...

    def pre_handle_nopinyin(self, chars: str, style: Any, heteronym: bool, errors: str, strict: bool, **kwargs: Any) -> Optional[Any]:
        ...

    def handle_nopinyin(self, chars: str, style: Any, heteronym: bool, errors: str, strict: bool, **kwargs: Any) -> List[List[str]]:
        ...

    def post_handle_nopinyin(self, chars: str, style: Any, heteronym: bool, errors: str, strict: bool, pinyin: Any, **kwargs: Any) -> Optional[Any]:
        ...

    def post_pinyin(self, han: Union[str, Sequence[str]], heteronym: bool, pinyin: List[str], **kwargs: Any) -> Optional[List[str]]:
        ...

    def _phrase_pinyin(self, phrase: str, style: Any, heteronym: bool, errors: str, strict: bool) -> List[str]:
        ...

    def convert_styles(self, pinyin_list: List[List[str]], phrase: str, style: Any, heteronym: bool, errors: str, strict: bool, **kwargs: Any) -> List[List[str]]:
        ...

    def _single_pinyin(self, han: str, style: Any, heteronym: bool, errors: str, strict: bool) -> List[List[str]]:
        ...

    def _convert_style(self, han: str, pinyin: str, style: Any, strict: bool, default: str, **kwargs: Any) -> str:
        ...

    def _convert_nopinyin_chars(self, chars: str, style: Any, heteronym: bool, errors: str, strict: bool) -> Optional[Union[str, List[str]]]:
        ...

class V2UMixin:
    ...

class NeutralToneWith5Mixin:
    ...

class ToneSandhiMixin:
    ...

class UltimateConverter(DefaultConverter):
    def __init__(self, v_to_u: bool = ..., neutral_tone_with_five: bool = ..., tone_sandhi: bool = ..., **kwargs: Any) -> None:
        ...

    def post_convert_style(self, han: str, orig_pinyin: str, converted_pinyin: str, style: Any, strict: bool, **kwargs: Any) -> str:
        ...

    def post_pinyin(self, han: Union[str, Sequence[str]], heteronym: bool, pinyin: List[str], **kwargs: Any) -> Optional[List[str]]:
        ...