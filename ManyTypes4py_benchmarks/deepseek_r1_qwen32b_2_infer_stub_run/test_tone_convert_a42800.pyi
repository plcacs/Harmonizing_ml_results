from __future__ import unicode_literals
from pytest import mark
from pypinyin import pinyin_dict
from pypinyin.contrib.tone_convert import (
    tone_to_normal,
    tone_to_tone2,
    tone2_to_tone,
    tone_to_tone3,
    tone3_to_tone,
    tone2_to_normal,
    tone2_to_tone3,
    tone3_to_tone2,
    tone3_to_normal,
    to_normal,
    to_tone,
    to_tone2,
    to_tone3,
    to_initials,
    to_finals,
    to_finals_tone,
    to_finals_tone2,
    to_finals_tone3,
)

@mark.parametrize('pinyin,result', [['zhōng', 'zhong'], ['ān', 'an'], ['yuè', 'yue'], ['er', 'er'], ['nǚ', 'nv'], ['nv', 'nv'], ['ā', 'a'], ['a', 'a']])
def test_tone_to_normal(pinyin: str, result: str) -> None:
    ...

@mark.parametrize('pinyin,v_to_u,result', [['nǚ', False, 'nv'], ['nv', False, 'nv'], ['nǚ', True, 'nü'], ['nv', True, 'nü']])
def test_tone_to_normal_with_v_to_u(pinyin: str, v_to_u: bool, result: str) -> None:
    ...

@mark.parametrize('pinyin,result', [['zhōng', 'zho1ng'], ['ān', 'a1n'], ['yuè', 'yue4'], ['er', 'er'], ['nǚ', 'nv3'], ['nv', 'nv'], ['ā', 'a1'], ['a', 'a'], ['shang', 'shang']])
def test_tone_tone2(pinyin: str, result: str) -> None:
    ...

@mark.parametrize('pinyin,neutral_tone_with_five,result', [['shang', False, 'shang'], ['shang', True, 'sha5ng']])
def test_tone_tone2_with_neutral_tone_with_five(pinyin: str, neutral_tone_with_five: bool, result: str) -> None:
    ...

@mark.parametrize('pinyin,v_to_u,result', [['nǚ', False, 'nv3'], ['nv', False, 'nv'], ['nǚ', True, 'nü3'], ['nv', True, 'nü']])
def test_tone_tone2_with_v_to_u(pinyin: str, v_to_u: bool, result: str) -> None:
    ...

@mark.parametrize('pinyin,result', [['zhōng', 'zhong1'], ['ān', 'an1'], ['yuè', 'yue4'], ['er', 'er'], ['nǚ', 'nv3'], ['nv', 'nv'], ['ā', 'a1'], ['a', 'a'], ['shang', 'shang']])
def test_tone_tone3(pinyin: str, result: str) -> None:
    ...

@mark.parametrize('pinyin,neutral_tone_with_five,result', [['shang', False, 'shang'], ['shang', True, 'shang5'], ['', False, ''], ['', True, '']])
def test_tone_tone3_with_neutral_tone_with_five(pinyin: str, neutral_tone_with_five: bool, result: str) -> None:
    ...

@mark.parametrize('pinyin,v_to_u,result', [['nǚ', False, 'nv3'], ['nǚ', True, 'nü3'], ['nv', True, 'nü']])
def test_tone_tone3_with_v_to_u(pinyin: str, v_to_u: bool, result: str) -> None:
    ...

@mark.parametrize('pinyin,result', [['zho1ng', 'zhong1'], ['a1n', 'an1'], ['yue4', 'yue4'], ['er', 'er'], ['nv3', 'nv3'], ['nü3', 'nv3'], ['a1', 'a1'], ['a', 'a'], ['shang', 'shang']])
def test_tone2_tone3(pinyin: str, result: str) -> None:
    ...

@mark.parametrize('pinyin,v_to_u,result', [['lüe3', False, 'lve3'], ['lüe3', True, 'lüe3']])
def test_tone2_tone3_with_v_to_u(pinyin: str, v_to_u: bool, result: str) -> None:
    ...

@mark.parametrize('pinyin,result', [['zho1ng', 'zhong'], ['a1n', 'an'], ['yue4', 'yue'], ['er', 'er'], ['nv3', 'nv'], ['nü3', 'nv'], ['a1', 'a'], ['a', 'a'], ['shang', 'shang'], ['sha5ng', 'shang']])
def test_tone2_to_normal(pinyin: str, result: str) -> None:
    ...

@mark.parametrize('pinyin,v_to_u,result', [['nv3', False, 'nv'], ['nv3', True, 'nü'], ['nü3', False, 'nv'], ['nü3', True, 'nü']])
def test_tone2_to_normal_with_v_to_u(pinyin: str, v_to_u: bool, result: str) -> None:
    ...

@mark.parametrize('pinyin,result', [['zhong1', 'zhong'], ['an1', 'an'], ['yue4', 'yue'], ['er', 'er'], ['nv3', 'nv'], ['nü3', 'nv'], ['a1', 'a'], ['a', 'a'], ['shang', 'shang'], ['shang5', 'shang']])
def test_tone3_to_normal(pinyin: str, result: str) -> None:
    ...

@mark.parametrize('pinyin,v_to_u,result', [['nv3', False, 'nv'], ['nv3', True, 'nü'], ['nü3', False, 'nv'], ['nü3', True, 'nü']])
def test_tone3_to_normal_with_v_to_u(pinyin: str, v_to_u: bool, result: str) -> None:
    ...

@mark.parametrize('pinyin,result', [['zhong1', 'zho1ng'], ['lüe4', 'lve4']])
def test_tone3_to_tone2(pinyin: str, result: str) -> None:
    ...

@mark.parametrize('pinyin,v_to_u,result', [['lüe4', False, 'lve4'], ['lüe4', True, 'lüe4']])
def test_tone3_to_tone2_with_v_to_u(pinyin: str, v_to_u: bool, result: str) -> None:
    ...

@mark.parametrize('pinyin,strict,result', [['zhōng', True, 'zh'], ['zhōng', False, 'zh'], ['zho1ng', True, 'zh'], ['zho1ng', False, 'zh'], ['zhong1', True, 'zh'], ['zhong1', False, 'zh'], ['zhong', True, 'zh'], ['zhong', False, 'zh'], ['yu', True, ''], ['yu', False, 'y']])
def test_to_initials(pinyin: str, strict: bool, result: str) -> None:
    ...

@mark.parametrize('pinyin,strict,v_to_u,result', [['zhōng', True, False, 'ong'], ['zhōng', False, False, 'ong'], ['zho1ng', True, False, 'ong'], ['zho1ng', False, False, 'ong'], ['zhong1', True, False, 'ong'], ['zhong1', False, False, 'ong'], ['zhong', True, False, 'ong'], ['zhong', False, False, 'ong'], ['nǚ', True, False, 'v'], ['nv', True, False, 'v'], ['nü', True, False, 'v'], ['nǚ', True, True, 'ü'], ['nü', True, True, 'ü'], ['nv', True, True, 'ü'], ['gui', True, False, 'uei'], ['gui', False, False, 'ui']])
def test_to_finals(pinyin: str, strict: bool, v_to_u: bool, result: str) -> None:
    ...

@mark.parametrize('pinyin,strict,result', [['zhōng', True, 'ōng'], ['zho1ng', True, 'ōng'], ['zhong1', True, 'ōng'], ['zhōng', False, 'ōng'], ['yū', True, 'ǖ'], ['yu1', True, 'ǖ'], ['yū', False, 'ū']])
def test_to_finals_tone(pinyin: str, strict: bool, result: str) -> None:
    ...

@mark.parametrize('pinyin,strict,v_to_u,neutral_tone_with_five,result', [['zhōng', True, False, False, 'o1ng'], ['zhong1', True, False, False, 'o1ng'], ['zho1ng', True, False, False, 'o1ng'], ['zhōng', False, False, False, 'o1ng'], ['zhong', False, False, True, 'o5ng'], ['yū', True, False, False, 'v1'], ['yu1', True, False, False, 'v1'], ['yū', True, True, False, 'ü1'], ['yū', False, False, False, 'u1'], ['yū', False, True, False, 'u1']])
def test_to_finals_tone2(pinyin: str, strict: bool, v_to_u: bool, neutral_tone_with_five: bool, result: str) -> None:
    ...

@mark.parametrize('pinyin,strict,v_to_u,neutral_tone_with_five,result', [['zhōng', True, False, False, 'ong1'], ['zhong1', True, False, False, 'ong1'], ['zho1ng', True, False, False, 'ong1'], ['zhōng', False, False, False, 'ong1'], ['zhong', False, False, True, 'ong5'], ['yū', True, False, False, 'v1'], ['yu1', True, False, False, 'v1'], ['yū', True, True, False, 'ü1'], ['yū', False, False, False, 'u1'], ['yū', False, True, False, 'u1']])
def test_to_finals_tone3(pinyin: str, strict: bool, v_to_u: bool, neutral_tone_with_five: bool, result: str) -> None:
    ...

def test_tone_to_tone2_tone3_to_tone() -> None:
    ...

@mark.parametrize('input', ['lün', 'lvn', 'lü5n', 'lün5', 'lv5n', 'lvn5'])
def test_issue_290_1(input: str) -> None:
    ...

@mark.parametrize('input', ['lǘ', 'lü2', 'lv2'])
def test_issue_290_2(input: str) -> None:
    ...

@mark.parametrize('input', ['lǘn', 'lü2n', 'lün2', 'lv2n', 'lvn2'])
def test_issue_290_3(input: str) -> None:
    ...

@mark.parametrize('input', ['shang', 'sha5ng', 'shang5'])
def test_issue_290_4(input: str) -> None:
    ...