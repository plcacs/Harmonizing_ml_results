from typing import Any, Callable, List, Optional, Union, overload

class Converter:
    def convert(
        self,
        words: str,
        style: str,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any,
    ) -> List[List[str]]:
        ...

class DefaultConverter(Converter):
    def __init__(self, **kwargs: Any) -> None:
        ...

    def convert(
        self,
        words: str,
        style: str,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any,
    ) -> List[List[str]]:
        ...

    def pre_convert_style(
        self,
        han: str,
        orig_pinyin: str,
        style: str,
        strict: bool,
        **kwargs: Any,
    ) -> Optional[str]:
        ...

    def convert_style(
        self,
        han: str,
        orig_pinyin: str,
        style: str,
        strict: bool,
        **kwargs: Any,
    ) -> str:
        ...

    def post_convert_style(
        self,
        han: str,
        orig_pinyin: str,
        converted_pinyin: str,
        style: str,
        strict: bool,
        **kwargs: Any,
    ) -> Optional[str]:
        ...

    def pre_handle_nopinyin(
        self,
        chars: str,
        style: str,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any,
    ) -> Optional[Union[str, List[Any]]]:
        ...

    def handle_nopinyin(
        self,
        chars: str,
        style: str,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any,
    ) -> List[List[str]]:
        ...

    def post_handle_nopinyin(
        self,
        chars: str,
        style: str,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        pinyin: Any,
        **kwargs: Any,
    ) -> Optional[Any]:
        ...

    def post_pinyin(
        self,
        han: str,
        heteronym: bool,
        pinyin: List[str],
        **kwargs: Any,
    ) -> Optional[List[str]]:
        ...

    def _phrase_pinyin(
        self,
        phrase: str,
        style: str,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
    ) -> List[List[str]]:
        ...

    def convert_styles(
        self,
        pinyin_list: List[List[str]],
        phrase: str,
        style: str,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
        **kwargs: Any,
    ) -> List[List[str]]:
        ...

    def _single_pinyin(
        self,
        han: str,
        style: str,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
    ) -> List[List[str]]:
        ...

    def _convert_style(
        self,
        han: str,
        pinyin: str,
        style: str,
        strict: bool,
        default: str,
        **kwargs: Any,
    ) -> str:
        ...

    def _convert_nopinyin_chars(
        self,
        chars: str,
        style: str,
        heteronym: bool,
        errors: Union[str, Callable[[str], Any]],
        strict: bool,
    ) -> Optional[str]:
        ...

class _v2UConverter(DefaultConverter):
    ...

class _neutralToneWith5Converter(DefaultConverter):
    ...

class _toneSandhiConverter(DefaultConverter):
    ...

class UltimateConverter(DefaultConverter):
    def __init__(
        self,
        v_to_u: bool = False,
        neutral_tone_with_five: bool = False,
        tone_sandhi: bool = False,
        **kwargs: Any,
    ) -> None:
        ...

    def post_convert_style(
        self,
        han: str,
        orig_pinyin: str,
        converted_pinyin: str,
        style: str,
        strict: bool,
        **kwargs: Any,
    ) -> str:
        ...

    def post_pinyin(
        self,
        han: str,
        heteronym: bool,
        pinyin: List[str],
        **kwargs: Any,
    ) -> List[str]:
        ...

_mixConverter: type[UltimateConverter] = UltimateConverter