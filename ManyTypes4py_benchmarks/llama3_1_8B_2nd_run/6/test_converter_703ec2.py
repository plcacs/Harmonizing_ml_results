from __future__ import unicode_literals
from pypinyin.constants import Style
from pypinyin.converter import DefaultConverter

def test_pre_convert_style_return_value() -> None:
    class A(DefaultConverter):
        def pre_convert_style(self, han: str, orig_pinyin: list[str], style: Style, strict: bool, **kwargs: dict[str, str]) -> str:
            return 'test'
    han = '测试'
    assert DefaultConverter().convert(han, Style.TONE2, False, 'ignore', True) == [['ce4'], ['shi4']]
    assert A().convert(han, Style.TONE2, False, 'ignore', True) == [['test'], ['test']]

def test_post_convert_style_return_value() -> None:
    class A(DefaultConverter):
        def post_convert_style(self, han: str, orig_pinyin: list[str], converted_pinyin: list[list[str]], style: Style, strict: bool, **kwargs: dict[str, str]) -> list[list[str]]:
            return ['test']
    han = '测试'
    assert DefaultConverter().convert(han, Style.TONE2, False, 'ignore', True) == [['ce4'], ['shi4']]
    assert A().convert(han, Style.TONE2, False, 'ignore', True) == [['test'], ['test']]

def test_pre_handle_nopinyin_return_value() -> None:
    class A(DefaultConverter):
        def pre_handle_nopinyin(self, chars: str, style: Style, heteronym: bool, errors: str, strict: bool, **kwargs: dict[str, str]) -> str:
            return 'abc'
    han = 'test'
    assert DefaultConverter().convert(han, Style.TONE2, False, 'default', True) == [['test']]
    assert A().convert(han, Style.TONE2, False, 'default', True) == [['abc']]

def test_post_handle_nopinyin_return_value() -> None:
    class A(DefaultConverter):
        def post_handle_nopinyin(self, chars: str, style: Style, heteronym: bool, errors: str, strict: bool, pinyin: list[list[str]], **kwargs: dict[str, str]) -> list[list[str]]:
            return [['abc']]
    han = 'test'
    assert DefaultConverter().convert(han, Style.TONE2, False, 'default', True) == [['test']]
    assert A().convert(han, Style.TONE2, False, 'default', True) == [['abc']]

def test_post_pinyin_return_value_single_pinyin() -> None:
    class A(DefaultConverter):
        def post_pinyin(self, han: str, heteronym: bool, pinyin: list[list[str]], **kwargs: dict[str, str]) -> list[list[str]]:
            return {'测': [['zhāo']], '试': [['yáng']], '测试': [['zhāo'], ['yáng']]}[han]
    han = '测试'
    assert DefaultConverter().convert(han, Style.TONE3, False, 'ignore', True) == [['ce4'], ['shi4']]
    assert A().convert(han, Style.TONE3, False, 'ignore', True) == [['zhao1'], ['yang2']]

def test_post_pinyin_return_value_phrase_pinyin() -> None:
    class A(DefaultConverter):
        def post_pinyin(self, han: str, heteronym: bool, pinyin: list[list[str]], **kwargs: dict[str, str]) -> list[list[str]]:
            return {'北': [['zhāo']], '京': [['yáng']], '北京': [['zhāo'], ['yáng']]}[han]
    han = '北京'
    assert DefaultConverter().convert(han, Style.TONE3, False, 'ignore', True) == [['bei3'], ['jing1']]
    assert A().convert(han, Style.TONE3, False, 'ignore', True) == [['zhao1'], ['yang2']]
