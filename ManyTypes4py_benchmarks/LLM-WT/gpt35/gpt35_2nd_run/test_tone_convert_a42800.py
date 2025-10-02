from __future__ import unicode_literals
from pytest import mark
from pypinyin import pinyin_dict
from pypinyin.contrib.tone_convert import tone_to_normal, tone_to_tone2, tone2_to_tone, tone_to_tone3, tone3_to_tone, tone2_to_normal, tone2_to_tone3, tone3_to_tone2, tone3_to_normal, to_normal, to_tone, to_tone2, to_tone3, to_initials, to_finals, to_finals_tone, to_finals_tone2, to_finals_tone3

@mark.parametrize('pinyin: str, result: str', [['zhōng', 'zhong'], ['ān', 'an'], ['yuè', 'yue'], ['er', 'er'], ['nǚ', 'nv'], ['nv', 'nv'], ['ā', 'a'], ['a', 'a']])
def test_tone_to_normal(pinyin, result):
    assert tone_to_normal(pinyin) == result
    assert to_normal(pinyin) == result
    assert to_normal(result) == result

@mark.parametrize('pinyin: str, v_to_u: bool, result: str', [['nǚ', False, 'nv'], ['nv', False, 'nv'], ['nǚ', True, 'nü'], ['nv', True, 'nü']])
def test_tone_to_normal_with_v_to_u(pinyin, v_to_u, result):
    assert tone_to_normal(pinyin, v_to_u=v_to_u) == result
    assert to_normal(pinyin, v_to_u=v_to_u) == result

@mark.parametrize('pinyin: str, result: str', [['zhōng', 'zho1ng'], ['ān', 'a1n'], ['yuè', 'yue4'], ['er', 'er'], ['nǚ', 'nv3'], ['nv', 'nv'], ['ā', 'a1'], ['a', 'a'], ['shang', 'shang']])
def test_tone_tone2(pinyin, result):
    assert tone_to_tone2(pinyin) == result
    assert to_tone2(pinyin) == result

@mark.parametrize('pinyin: str, neutral_tone_with_five: bool, result: str', [['shang', False, 'shang'], ['shang', True, 'sha5ng']])
def test_tone_tone2_with_neutral_tone_with_five(pinyin, neutral_tone_with_five, result):
    assert tone_to_tone2(pinyin, neutral_tone_with_five=neutral_tone_with_five) == result
    assert tone_to_tone2(pinyin, neutral_tone_with_5=neutral_tone_with_five) == result
    assert to_tone2(pinyin, neutral_tone_with_five=neutral_tone_with_five) == result
    assert to_tone2(pinyin, neutral_tone_with_5=neutral_tone_with_five) == result
    assert tone2_to_tone(result) == pinyin
    assert to_tone(result) == pinyin

@mark.parametrize('pinyin: str, v_to_u: bool, result: str', [['nǚ', False, 'nv3'], ['nv', False, 'nv'], ['nǚ', True, 'nü3'], ['nv', True, 'nü']])
def test_tone_tone2_with_v_to_u(pinyin, v_to_u, result):
    assert tone_to_tone2(pinyin, v_to_u=v_to_u) == result
    assert to_tone2(pinyin, v_to_u=v_to_u) == result

@mark.parametrize('pinyin: str, result: str', [['zhōng', 'zhong1'], ['ān', 'an1'], ['yuè', 'yue4'], ['er', 'er'], ['nǚ', 'nv3'], ['nv', 'nv'], ['ā', 'a1'], ['a', 'a'], ['shang', 'shang']])
def test_tone_tone3(pinyin, result):
    assert tone_to_tone3(pinyin) == result
    assert to_tone3(pinyin) == result

@mark.parametrize('pinyin: str, neutral_tone_with_five: bool, result: str', [['shang', False, 'shang'], ['shang', True, 'shang5'], ['', False, ''], ['', True, '']])
def test_tone_tone3_with_neutral_tone_with_five(pinyin, neutral_tone_with_five, result):
    assert tone_to_tone3(pinyin, neutral_tone_with_five=neutral_tone_with_five) == result
    assert tone_to_tone3(pinyin, neutral_tone_with_5=neutral_tone_with_five) == result
    assert to_tone3(pinyin, neutral_tone_with_five=neutral_tone_with_five) == result
    assert to_tone3(pinyin, neutral_tone_with_5=neutral_tone_with_five) == result
    assert tone3_to_tone(result) == pinyin
    assert to_tone(result) == pinyin

@mark.parametrize('pinyin: str, v_to_u: bool, result: str', [['nǚ', False, 'nv3'], ['nǚ', True, 'nü3'], ['nv', True, 'nü']])
def test_tone_tone3_with_v_to_u(pinyin, v_to_u, result):
    assert tone_to_tone3(pinyin, v_to_u=v_to_u) == result
    assert to_tone3(pinyin, v_to_u=v_to_u) == result

@mark.parametrize('pinyin: str, result: str', [['zho1ng', 'zhong1'], ['a1n', 'an1'], ['yue4', 'yue4'], ['er', 'er'], ['nv3', 'nv3'], ['nü3', 'nv3'], ['a1', 'a1'], ['a', 'a'], ['shang', 'shang']])
def test_tone2_tone3(pinyin, result):
    assert tone2_to_tone3(pinyin) == result
    assert to_tone3(pinyin) == result

@mark.parametrize('pinyin: str, v_to_u: bool, result: str', [['lüe3', False, 'lve3'], ['lüe3', True, 'lüe3']])
def test_tone2_tone3_with_v_to_u(pinyin, v_to_u, result):
    assert tone2_to_tone3(pinyin, v_to_u=v_to_u) == result

@mark.parametrize('pinyin: str, result: str', [['zho1ng', 'zhong'], ['a1n', 'an'], ['yue4', 'yue'], ['er', 'er'], ['nv3', 'nv'], ['nü3', 'nv'], ['a1', 'a'], ['a', 'a'], ['shang', 'shang'], ['sha5ng', 'shang']])
def test_tone2_to_normal(pinyin, result):
    assert tone2_to_normal(pinyin) == result
    assert to_normal(pinyin) == result
    assert to_normal(result) == result

@mark.parametrize('pinyin: str, v_to_u: bool, result: str', [['nv3', False, 'nv'], ['nv3', True, 'nü'], ['nü3', False, 'nv'], ['nü3', True, 'nü']])
def test_tone2_to_normal_with_v_to_u(pinyin, v_to_u, result):
    assert tone2_to_normal(pinyin, v_to_u=v_to_u) == result
    assert to_normal(pinyin, v_to_u=v_to_u) == result
    assert to_normal(result, v_to_u=v_to_u) == result

@mark.parametrize('pinyin: str, result: str', [['zhong1', 'zhong'], ['an1', 'an'], ['yue4', 'yue'], ['er', 'er'], ['nv3', 'nv'], ['nü3', 'nv'], ['a1', 'a'], ['a', 'a'], ['shang', 'shang'], ['shang5', 'shang']])
def test_tone3_to_normal(pinyin, result):
    assert tone3_to_normal(pinyin) == result
    assert to_normal(pinyin) == result

@mark.parametrize('pinyin: str, v_to_u: bool, result: str', [['nv3', False, 'nv'], ['nv3', True, 'nü'], ['nü3', False, 'nv'], ['nü3', True, 'nü']])
def test_tone3_to_normal_with_v_to_u(pinyin, v_to_u, result):
    assert tone3_to_normal(pinyin, v_to_u=v_to_u) == result
    assert to_normal(pinyin, v_to_u=v_to_u) == result

@mark.parametrize('pinyin: str, result: str', [['zhong1', 'zho1ng'], ['lüe4', 'lve4']])
def test_tone3_to_tone2(pinyin, result):
    assert tone3_to_tone2(pinyin) == result

@mark.parametrize('pinyin: str, v_to_u: bool, result: str', [['lüe4', False, 'lve4'], ['lüe4', True, 'lüe4']])
def test_tone3_to_tone2_with_v_to_u(pinyin, v_to_u, result):
    assert tone3_to_tone2(pinyin, v_to_u=v_to_u) == result

@mark.parametrize('pinyin: str', ['lün', 'lvn', 'lü5n', 'lün5', 'lv5n', 'lvn5'])
def test_issue_290_1(input):
    assert to_normal(input) == 'lvn'
    assert to_normal(input, v_to_u=True) == 'lün'
    assert to_tone(input) == 'lün'
    assert to_tone2(input) == 'lvn'
    assert to_tone2(input, neutral_tone_with_five=True) == 'lv5n'
    assert to_tone2(input, v_to_u=True) == 'lün'
    assert to_tone2(input, v_to_u=True, neutral_tone_with_five=True) == 'lü5n'
    assert to_tone3(input) == 'lvn'
    assert to_tone3(input, neutral_tone_with_five=True) == 'lvn5'
    assert to_tone3(input, v_to_u=True) == 'lün'
    assert to_tone3(input, v_to_u=True, neutral_tone_with_five=True) == 'lün5'
    assert to_finals(input) == 'vn'
    assert to_finals(input, v_to_u=True) == 'ün'
    assert to_finals_tone(input) == 'vn'
    assert to_finals_tone2(input) == 'vn'
    assert to_finals_tone2(input, neutral_tone_with_five=True) == 'v5n'
    assert to_finals_tone2(input, v_to_u=True) == 'ün'
    assert to_finals_tone2(input, v_to_u=True, neutral_tone_with_five=True) == 'ü5n'
    assert to_finals_tone3(input) == 'vn'
    assert to_finals_tone3(input, neutral_tone_with_five=True) == 'vn5'
    assert to_finals_tone3(input, v_to_u=True) == 'ün'
    assert to_finals_tone3(input, v_to_u=True, neutral_tone_with_five=True) == 'ün5'

@mark.parametrize('pinyin: str', ['lǘ', 'lü2', 'lv2'])
def test_issue_290_2(input):
    assert to_normal(input) == 'lv'
    assert to_normal(input, v_to_u=True) == 'lü'
    assert to_tone(input) == 'lǘ'
    assert to_tone2(input) == 'lv2'
    assert to_tone2(input, v_to_u=True) == 'lü2'
    assert to_tone3(input) == 'lv2'
    assert to_tone3(input, v_to_u=True) == 'lü2'
    assert to_finals(input) == 'v'
    assert to_finals(input, v_to_u=True) == 'ü'
    assert to_finals_tone(input) == 'ǘ'
    assert to_finals_tone2(input) == 'v2'
    assert to_finals_tone2(input, v_to_u=True) == 'ü2'
    assert to_finals_tone3(input) == 'v2'
    assert to_finals_tone3(input, v_to_u=True) == 'ü2'

@mark.parametrize('pinyin: str', ['lǘn', 'lü2n', 'lün2', 'lv2n', 'lvn2'])
def test_issue_290_3(input):
    assert to_normal(input) == 'lvn'
    assert to_normal(input, v_to_u=True) == 'lün'
    assert to_tone(input) == 'lǘn'
    assert to_tone2(input) == 'lv2n'
    assert to_tone2(input, v_to_u=True) == 'lü2n'
    assert to_tone3(input) == 'lvn2'
    assert to_tone3(input, v_to_u=True) == 'lün2'
    assert to_finals(input) == 'vn'
    assert to_finals(input, v_to_u=True) == 'ün'
    assert to_finals_tone(input) == 'ǘn'
    assert to_finals_tone2(input) == 'v2n'
    assert to_finals_tone2(input, v_to_u=True) == 'ü2n'
    assert to_finals_tone3(input) == 'vn2'
    assert to_finals_tone3(input, v_to_u=True) == 'ün2'

@mark.parametrize('pinyin: str', ['shang', 'sha5ng', 'shang5'])
def test_issue_290_4(input):
    assert to_normal(input) == 'shang'
    assert to_normal(input, v_to_u=True) == 'shang'
    assert to_tone(input) == 'shang'
    assert to_tone2(input) == 'shang'
    assert to_tone2(input, neutral_tone_with_five=True) == 'sha5ng'
    assert to_tone2(input, v_to_u=True) == 'shang'
    assert to_tone2(input, v_to_u=True, neutral_tone_with_five=True) == 'sha5ng'
    assert to_tone3(input) == 'shang'
    assert to_tone3(input, neutral_tone_with_five=True) == 'shang5'
    assert to_tone3(input, v_to_u=True) == 'shang'
    assert to_tone3(input, v_to_u=True, neutral_tone_with_five=True) == 'shang5'
    assert to_finals(input) == 'ang'
    assert to_finals(input, v_to_u=True) == 'ang'
    assert to_finals_tone(input) == 'ang'
    assert to_finals_tone2(input) == 'ang'
    assert to_finals_tone2(input, neutral_tone_with_five=True) == 'a5ng'
    assert to_finals_tone2(input, v_to_u=True) == 'ang'
    assert to_finals_tone2(input, v_to_u=True, neutral_tone_with_five=True) == 'a5ng'
    assert to_finals_tone3(input) == 'ang'
    assert to_finals_tone3(input, neutral_tone_with_five=True) == 'ang5'
    assert to_finals_tone3(input, v_to_u=True) == 'ang'
    assert to_finals_tone3(input, v_to_u=True, neutral_tone_with_five=True) == 'ang5'

def test_tone_to_tone2_tone3_to_tone():
    pinyin_set = set()
    for py in pinyin_dict.pinyin_dict.values():
        pinyin_set.update(py.split(','))
    for py in pinyin_set:
        tone2 = tone_to_tone2(py)
        assert tone2_to_tone(tone2) == py
        assert to_tone(tone2) == py
        tone2_3 = tone2_to_tone3(tone2)
        assert tone3_to_tone(tone2_3) == py
        assert to_tone(tone2_3) == py
        tone3 = tone_to_tone3(py)
        assert tone3_to_tone(tone3) == py
        assert to_tone(tone3) == py
        tone3_2 = tone3_to_tone2(tone3)
        assert tone2_to_tone(tone3_2) == py
        assert to_tone(tone3_2) == py

@mark.parametrize('input: str', ['lün', 'lvn', 'lü5n', 'lün5', 'lv5n', 'lvn5'])
def test_issue_290_1(input):
    assert to_normal(input) == 'lvn'
    assert to_normal(input, v_to_u=True) == 'lün'
    assert to_tone(input) == 'lün'
    assert to_tone2(input) == 'lvn'
    assert to_tone2(input, neutral_tone_with_five=True) == 'lv5n'
    assert to_tone2(input, v_to_u=True) == 'lün'
    assert to_tone2(input, v_to_u=True, neutral_tone_with_five=True) == 'lü5n'
    assert to_tone3(input) == 'lvn'
    assert to_tone3(input, neutral_tone_with_five=True) == 'lvn5'
    assert to_tone3(input, v_to_u=True) == 'lün'
    assert to_tone3(input, v_to_u=True, neutral_tone_with_five=True) == 'lün5'
    assert to_finals(input) == 'vn'
    assert to_finals(input, v_to_u=True) == 'ün'
    assert to_finals_tone(input) == 'vn'
    assert to_finals_tone2(input) == 'vn'
    assert to_finals_tone2(input, neutral_tone_with_five=True) == 'v5n'
    assert to_finals_tone2(input, v_to_u=True) == 'ün'
    assert to_finals_tone2(input, v_to_u=True, neutral_tone_with_five=True) == 'ü5n'
    assert to_finals_tone3(input) == 'vn'
    assert to_finals_tone3(input, neutral_tone_with_five=True) == 'vn5'
