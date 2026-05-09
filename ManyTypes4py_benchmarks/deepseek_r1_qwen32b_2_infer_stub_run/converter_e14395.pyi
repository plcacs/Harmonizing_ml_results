from typing import Any, Dict, Iterable, List, Optional, Union
from pypinyin.compat import text_type
from pypinyin.contrib.uv import V2UMixin
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.contrib.tone_sandhi import ToneSandhiMixin

class Converter:
    def convert(self, words: str, style: int, heteronym: bool, errors: str, strict: bool, **kwargs: Any) -> List[str]:
        ...

class DefaultConverter(Converter):
    def __init__(self, **kwargs: Any) -> None:
        ...

    def convert(self, words: str, style: int, heteronym: bool, errors: str, strict: bool, **kwargs: Any) -> List[str]:
        ...

    def pre_convert_style(self, han: str, orig_pinyin: str, style: int, strict: bool, **kwargs: Any) -> Optional[str]:
        ...

    def convert_style(self, han: str, orig_pinyin: str, style: int, strict: bool, **kwargs: Any) -> str:
        ...

    def post_convert_style(self, han: str, orig_pinyin: str, converted_pinyin: str, style: int, strict: bool, **kwargs: Any) -> Optional[str]:
        ...

    def pre_handle_nopinyin(self, chars: str, style: int, heteronym: bool, errors: str, strict: bool, **kwargs: Any) -> Optional[Union[List[str], str]]:
        ...

    def handle_nopinyin(self, chars: str, style: int, heteronym: bool, errors: str, strict: bool, **kwargs: Any) -> List[List[str]]:
        ...

    def post_handle_nopinyin(self, chars: str, style: int, heteronym: bool, errors: str, strict: bool, pinyin: List[List[str]], **kwargs: Any) -> Optional[List[List[str]]]:
        ...

    def post_pinyin(self, han: Union[str, Iterable[str]], heteronym: bool, pinyin: List[str], **kwargs: Any) -> Optional[List[str]]:
        ...

    def _phrase_pinyin(self, phrase: str, style: int, heteronym: bool, errors: str, strict: bool) -> List[List[str]]:
        ...

    def convert_styles(self, pinyin_list: List[List[str]], phrase: str, style: int, heteronym: bool, errors: str, strict: bool, **kwargs: Any) -> List[List[str]]:
        ...

    def _single_pinyin(self, han: str, style: int, heteronym: bool, errors: str, strict: bool) -> List[str]:
        ...

    def _convert_style(self, han: str, pinyin: str, style: int, strict: bool, default: str, **kwargs: Any) -> str:
        ...

    def _convert_nopinyin_chars(self, chars: str, style: int, heteronym: bool, errors: str, strict: bool) -> Optional[Union[List[str], str]]:
        ...

class _v2UConverter(V2UMixin, DefaultConverter):
    ...

class _neutralToneWith5Converter(NeutralToneWith5Mixin, DefaultConverter):
    ...

class _toneSandhiConverter(ToneSandhiMixin, DefaultConverter):
    ...

class UltimateConverter(DefaultConverter):
    def __init__(self, v_to_u: bool = False, neutral_tone_with_five: bool = False, tone_sandhi: bool = False, **kwargs: Any) -> None:
        ...

    def post_convert_style(self, han: str, orig_pinyin: str, converted_pinyin: str, style: int, strict: bool, **kwargs: Any) -> Optional[str]:
        ...

    def post_pinyin(self, han: Union[str, Iterable[str]], heteronym: bool, pinyin: List[str], **kwargs: Any) -> Optional[List[str]]:
        ...

_mixConverter = UltimateConverter