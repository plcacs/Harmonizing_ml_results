from typing import Any

# === Internal dependency: pypinyin.compat ===
def callable_check(obj: Any) -> bool: ...
text_type: Any

# === Internal dependency: pypinyin.constants ===
PHRASES_DICT: Any
PINYIN_DICT: Any
RE_HANS: Any

# === Internal dependency: pypinyin.contrib.neutral_tone ===
class NeutralToneWith5Mixin(object):
    def post_convert_style(self, han: Text, orig_pinyin: Text, converted_pinyin: Text, style: TStyle, strict: bool, **kwargs: Any) -> Optional[Text]: ...

# === Internal dependency: pypinyin.contrib.tone_sandhi ===
class ToneSandhiMixin(object):
    def post_pinyin(self, han: Text, heteronym: bool, pinyin: TPinyinResult, **kwargs: Any) -> Union[TPinyinResult, None]: ...

# === Internal dependency: pypinyin.contrib.uv ===
class V2UMixin(object):
    def post_convert_style(self, han: Text, orig_pinyin: Text, converted_pinyin: Text, style: TStyle, strict: bool, **kwargs: Any) -> Optional[Text]: ...

# === Internal dependency: pypinyin.style ===
def auto_discover() -> None: ...

# === Internal dependency: pypinyin.utils ===
def _remove_dup_and_empty(lst_list: List[List[Text]]) -> List[List[Text]]: ...