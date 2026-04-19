from typing import Any, Callable, List, Optional, Union, Literal

PinyinList = List[List[str]]
NopinyinResult = Union[str, List[str], PinyinList]
ErrorsHandler = Callable[[str], Optional[NopinyinResult]]
ErrorsType = Union[Literal['default', 'ignore', 'replace'], ErrorsHandler]


class Converter:
    def convert(
        self,
        words: str,
        style: int,
        heteronym: bool,
        errors: ErrorsType,
        strict: bool,
        **kwargs: Any
    ) -> PinyinList: ...


class DefaultConverter(Converter):
    def __init__(self, **kwargs: Any) -> None: ...
    def convert(
        self,
        words: str,
        style: int,
        heteronym: bool,
        errors: ErrorsType,
        strict: bool,
        **kwargs: Any
    ) -> PinyinList: ...
    def pre_convert_style(
        self,
        han: str,
        orig_pinyin: str,
        style: int,
        strict: bool,
        **kwargs: Any
    ) -> Optional[str]: ...
    def convert_style(
        self,
        han: str,
        orig_pinyin: str,
        style: int,
        strict: bool,
        **kwargs: Any
    ) -> str: ...
    def post_convert_style(
        self,
        han: str,
        orig_pinyin: str,
        converted_pinyin: str,
        style: int,
        strict: bool,
        **kwargs: Any
    ) -> Optional[str]: ...
    def pre_handle_nopinyin(
        self,
        chars: str,
        style: int,
        heteronym: bool,
        errors: ErrorsType,
        strict: bool,
        **kwargs: Any
    ) -> Optional[NopinyinResult]: ...
    def handle_nopinyin(
        self,
        chars: str,
        style: int,
        heteronym: bool,
        errors: ErrorsType,
        strict: bool,
        **kwargs: Any
    ) -> PinyinList: ...
    def post_handle_nopinyin(
        self,
        chars: str,
        style: int,
        heteronym: bool,
        errors: ErrorsType,
        strict: bool,
        pinyin: NopinyinResult,
        **kwargs: Any
    ) -> Optional[NopinyinResult]: ...
    def post_pinyin(
        self,
        han: str,
        heteronym: bool,
        pinyin: PinyinList,
        **kwargs: Any
    ) -> Optional[PinyinList]: ...
    def convert_styles(
        self,
        pinyin_list: PinyinList,
        phrase: str,
        style: int,
        heteronym: bool,
        errors: ErrorsType,
        strict: bool,
        **kwargs: Any
    ) -> PinyinList: ...


class UltimateConverter(DefaultConverter):
    _v_to_u: bool
    _neutral_tone_with_five: bool
    _tone_sandhi: bool

    def __init__(
        self,
        v_to_u: bool = False,
        neutral_tone_with_five: bool = False,
        tone_sandhi: bool = False,
        **kwargs: Any
    ) -> None: ...
    def post_convert_style(
        self,
        han: str,
        orig_pinyin: str,
        converted_pinyin: str,
        style: int,
        strict: bool,
        **kwargs: Any
    ) -> str: ...
    def post_pinyin(
        self,
        han: str,
        heteronym: bool,
        pinyin: PinyinList,
        **kwargs: Any
    ) -> PinyinList: ...