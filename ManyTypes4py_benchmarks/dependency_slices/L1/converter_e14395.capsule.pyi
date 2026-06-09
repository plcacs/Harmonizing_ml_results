# === Internal dependency: pypinyin.compat ===
def callable_check(obj): ...
text_type = Ellipsis

# === Internal dependency: pypinyin.constants ===
PHRASES_DICT = Ellipsis
PINYIN_DICT = Ellipsis
RE_HANS = Ellipsis

# === Internal dependency: pypinyin.contrib.neutral_tone ===
class NeutralToneWith5Mixin(object):
    def post_convert_style(self, han, orig_pinyin, converted_pinyin, style, strict, **kwargs): ...

# === Internal dependency: pypinyin.contrib.tone_sandhi ===
class ToneSandhiMixin(object):
    def post_pinyin(self, han, heteronym, pinyin, **kwargs): ...

# === Internal dependency: pypinyin.contrib.uv ===
class V2UMixin(object):
    def post_convert_style(self, han, orig_pinyin, converted_pinyin, style, strict, **kwargs): ...

# === Internal dependency: pypinyin.style ===
def auto_discover(): ...

# === Internal dependency: pypinyin.utils ===
def _remove_dup_and_empty(lst_list): ...