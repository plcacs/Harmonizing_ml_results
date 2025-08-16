from pandas import Series
import numpy as np
import re

def test_translate(index_or_series: Series, any_string_dtype: str, infer_string: bool) -> Series:
    obj = index_or_series(['abcdefg', 'abcc', 'cdddfg', 'cdefggg'], dtype=any_string_dtype)
    table = str.maketrans('abc', 'cde')
    result = obj.str.translate(table)
    expected = index_or_series(['cdedefg', 'cdee', 'edddfg', 'edefggg'], dtype=any_string_dtype)
    return result

def test_translate_mixed_object(s: Series) -> Series:
    table = str.maketrans('abc', 'cde')
    expected = Series(['c', 'd', 'e', np.nan], dtype=object)
    result = s.str.translate(table)
    return result

def test_flags_kwarg(data: Series, any_string_dtype: str) -> Series:
    pat = '([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\\.([A-Z]{2,4})'
    result = data.str.extract(pat, flags=re.IGNORECASE, expand=True)
    result = data.str.match(pat, flags=re.IGNORECASE)
    result = data.str.fullmatch(pat, flags=re.IGNORECASE)
    result = data.str.findall(pat, flags=re.IGNORECASE)
    result = data.str.count(pat, flags=re.IGNORECASE)
    result = data.str.contains(pat, flags=re.IGNORECASE)
    return result
