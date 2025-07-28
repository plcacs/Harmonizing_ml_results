from __future__ import unicode_literals
from typing import Any, List, Dict
from pypinyin.constants import Style
from pypinyin.converter import DefaultConverter

def test_pre_convert_style_return_value() -> None:
    class A(DefaultConverter):
        def pre_convert_style(self, han: str, orig_pinyin: List[List[str]], style: int, strict: bool, **kwargs: Any) -> Any:
            return 'test'
    han: str = '测试'
    assert DefaultConverter().convert(han, Style.TONE2, False, 'ignore', True) == [['ce4'], ['shi4']]
    assert A().convert(han, Style.TONE2, False, 'ignore', True) == [['test'], ['test']]

def test_post_convert_style_return_value() -> None:
    class A(DefaultConverter):
        def post_convert_style(self, han: str, orig_pinyin: List[List[str]], converted_pinyin: List[List[str]], style: int, strict: bool, **kwargs: Any) -> Any:
            return 'test'
    han: str = '测试'
    assert DefaultConverter().convert(han, Style.TONE2, False, 'ignore', True) == [['ce4'], ['shi4']]
    assert A().convert(han, Style.TONE2, False, 'ignore', True) == [['test'], ['test']]

def test_pre_handle_nopinyin_return_value() -> None:
    class A(DefaultConverter):
        def pre_handle_nopinyin(self, chars: str, style: int, heteronym: bool, errors: str, strict: bool, **kwargs: Any) -> Any:
            return 'abc'
    han: str = 'test'
    assert DefaultConverter().convert(han, Style.TONE2, False, 'default', True) == [['test']]
    assert A().convert(han, Style.TONE2, False, 'default', True) == [['abc']]

def test_post_handle_nopinyin_return_value() -> None:
    class A(DefaultConverter):
        def post_handle_nopinyin(self, chars: str, style: int, heteronym: bool, errors: str, strict: bool, pinyin: List[List[str]], **kwargs: Any) -> Any:
            return 'abc'
    han: str = 'test'
    assert DefaultConverter().convert(han, Style.TONE2, False, 'default', True) == [['test']]
    assert A().convert(han, Style.TONE2, False, 'default', True) == [['abc']]

def test_post_pinyin_return_value_single_pinyin() -> None:
    class A(DefaultConverter):
        def post_pinyin(self, han: str, heteronym: bool, pinyin: List[List[str]], **kwargs: Any) -> List[List[str]]:
            mapping: Dict[str, List[List[str]]] = {
                '测': [['zhāo']],
                '试': [['yáng']],
                '测试': [['zhāo'], ['yáng']]
            }
            return mapping[han]
    han: str = '测试'
    assert DefaultConverter().convert(han, Style.TONE3, False, 'ignore', True) == [['ce4'], ['shi4']]
    assert A().convert(han, Style.TONE3, False, 'ignore', True) == [['zhao1'], ['yang2']]

def test_post_pinyin_return_value_phrase_pinyin() -> None:
    class A(DefaultConverter):
        def post_pinyin(self, han: str, heteronym: bool, pinyin: List[List[str]], **kwargs: Any) -> List[List[str]]:
            mapping: Dict[str, List[List[str]]] = {
                '北': [['zhāo']],
                '京': [['yáng']],
                '北京': [['zhāo'], ['yáng']]
            }
            return mapping[han]
    han: str = '北京'
    assert DefaultConverter().convert(han, Style.TONE3, False, 'ignore', True) == [['bei3'], ['jing1']]
    assert A().convert(han, Style.TONE3, False, 'ignore', True) == [['zhao1'], ['yang2']]