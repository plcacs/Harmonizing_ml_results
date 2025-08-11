from __future__ import annotations
from functools import partial
import re
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from pandas._libs import lib
from pandas.compat import pa_version_under10p1, pa_version_under11p0, pa_version_under13p0, pa_version_under17p0
if not pa_version_under10p1:
    import pyarrow as pa
    import pyarrow.compute as pc
if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas._typing import Scalar, Self

class ArrowStringArrayMixin:

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def _convert_bool_result(self, result: Union[str, typing.Iterator], na: Any=lib.no_default, method_name: Union[None, str, typing.Iterator]=None) -> None:
        raise NotImplementedError

    def _convert_int_result(self, result: Union[list[str], dict, str, None]) -> None:
        raise NotImplementedError

    def _apply_elementwise(self, func: Union[typing.Callable, zerver.lib.types.ViewFuncT]) -> None:
        raise NotImplementedError

    def _str_len(self):
        result = pc.utf8_length(self._pa_array)
        return self._convert_int_result(result)

    def _str_lower(self) -> Union[str, list[str], typing.BinaryIO]:
        return type(self)(pc.utf8_lower(self._pa_array))

    def _str_upper(self) -> Union[str, bytes, typing.AnyStr]:
        return type(self)(pc.utf8_upper(self._pa_array))

    def _str_strip(self, to_strip: Union[None, int, str]=None) -> Union[str, set[str], dict[str, typing.Any]]:
        if to_strip is None:
            result = pc.utf8_trim_whitespace(self._pa_array)
        else:
            result = pc.utf8_trim(self._pa_array, characters=to_strip)
        return type(self)(result)

    def _str_lstrip(self, to_strip: Union[None, str, tuple[int]]=None) -> Union[str, set[str], list[str]]:
        if to_strip is None:
            result = pc.utf8_ltrim_whitespace(self._pa_array)
        else:
            result = pc.utf8_ltrim(self._pa_array, characters=to_strip)
        return type(self)(result)

    def _str_rstrip(self, to_strip: Union[None, str, bool]=None) -> Union[str, set[str], typing.Callable[str,str, str]]:
        if to_strip is None:
            result = pc.utf8_rtrim_whitespace(self._pa_array)
        else:
            result = pc.utf8_rtrim(self._pa_array, characters=to_strip)
        return type(self)(result)

    def _str_pad(self, width: str, side: typing.Text='left', fillchar: typing.Text=' ') -> Union[str, typing.AbstractSet]:
        if side == 'left':
            pa_pad = pc.utf8_lpad
        elif side == 'right':
            pa_pad = pc.utf8_rpad
        elif side == 'both':
            if pa_version_under17p0:
                from pandas import array
                obj_arr = self.astype(object, copy=False)
                obj = array(obj_arr, dtype=object)
                result = obj._str_pad(width, side, fillchar)
                return type(self)._from_sequence(result, dtype=self.dtype)
            else:
                lean_left = width % 2 == 0
                pa_pad = partial(pc.utf8_center, lean_left_on_odd_padding=lean_left)
        else:
            raise ValueError(f"Invalid side: {side}. Side must be one of 'left', 'right', 'both'")
        return type(self)(pa_pad(self._pa_array, width=width, padding=fillchar))

    def _str_get(self, i: Union[int, float]) -> Union[set[str], str, int]:
        lengths = pc.utf8_length(self._pa_array)
        if i >= 0:
            out_of_bounds = pc.greater_equal(i, lengths)
            start = i
            stop = i + 1
            step = 1
        else:
            out_of_bounds = pc.greater(-i, lengths)
            start = i
            stop = i - 1
            step = -1
        not_out_of_bounds = pc.invert(out_of_bounds.fill_null(True))
        selected = pc.utf8_slice_codeunits(self._pa_array, start=start, stop=stop, step=step)
        null_value = pa.scalar(None, type=self._pa_array.type)
        result = pc.if_else(not_out_of_bounds, selected, null_value)
        return type(self)(result)

    def _str_slice(self, start: Union[None, float, int]=None, stop: Union[None, float, int]=None, step: Union[float, int]=None) -> Union[dict, T]:
        if pa_version_under11p0:
            result = self._apply_elementwise(lambda val: val[start:stop:step])
            return type(self)(pa.chunked_array(result, type=self._pa_array.type))
        if start is None:
            if step is not None and step < 0:
                start = -1
            else:
                start = 0
        if step is None:
            step = 1
        return type(self)(pc.utf8_slice_codeunits(self._pa_array, start=start, stop=stop, step=step))

    def _str_slice_replace(self, start: Union[None, int, T, float]=None, stop: Union[None, int, float]=None, repl: Union[None, int, float, tuple[int]]=None) -> Union[str, bytes]:
        if repl is None:
            repl = ''
        if start is None:
            start = 0
        if stop is None:
            stop = np.iinfo(np.int64).max
        return type(self)(pc.utf8_replace_slice(self._pa_array, start, stop, repl))

    def _str_replace(self, pat: Union[str, None, bool, typing.Callable[..., T]], repl: Union[str, None, bool, typing.Callable[..., T]], n: int=-1, case: bool=True, flags: int=0, regex: bool=True) -> Union[str, set[str], list[str]]:
        if isinstance(pat, re.Pattern) or callable(repl) or (not case) or flags:
            raise NotImplementedError('replace is not supported with a re.Pattern, callable repl, case=False, or flags!=0')
        func = pc.replace_substring_regex if regex else pc.replace_substring
        pa_max_replacements = None if n < 0 else n
        result = func(self._pa_array, pattern=pat, replacement=repl, max_replacements=pa_max_replacements)
        return type(self)(result)

    def _str_capitalize(self) -> str:
        return type(self)(pc.utf8_capitalize(self._pa_array))

    def _str_title(self) -> Union[str, dict, list]:
        return type(self)(pc.utf8_title(self._pa_array))

    def _str_swapcase(self) -> Union[str, bytes, typing.AnyStr]:
        return type(self)(pc.utf8_swapcase(self._pa_array))

    def _str_removeprefix(self, prefix: str) -> Union[T, typing.Mapping, typing.Type]:
        if not pa_version_under13p0:
            starts_with = pc.starts_with(self._pa_array, pattern=prefix)
            removed = pc.utf8_slice_codeunits(self._pa_array, len(prefix))
            result = pc.if_else(starts_with, removed, self._pa_array)
            return type(self)(result)
        predicate = lambda val: val.removeprefix(prefix)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_removesuffix(self, suffix: Union[str, list[str]]) -> Union[str, typing.Iterable[T], typing.Callable[str,str, str]]:
        ends_with = pc.ends_with(self._pa_array, pattern=suffix)
        removed = pc.utf8_slice_codeunits(self._pa_array, 0, stop=-len(suffix))
        result = pc.if_else(ends_with, removed, self._pa_array)
        return type(self)(result)

    def _str_startswith(self, pat: Union[str, typing.Pattern, int], na: Any=lib.no_default):
        if isinstance(pat, str):
            result = pc.starts_with(self._pa_array, pattern=pat)
        elif len(pat) == 0:
            result = pc.if_else(pc.is_null(self._pa_array), None, False)
        else:
            result = pc.starts_with(self._pa_array, pattern=pat[0])
            for p in pat[1:]:
                result = pc.or_(result, pc.starts_with(self._pa_array, pattern=p))
        return self._convert_bool_result(result, na=na, method_name='startswith')

    def _str_endswith(self, pat: Union[str, int], na: Any=lib.no_default):
        if isinstance(pat, str):
            result = pc.ends_with(self._pa_array, pattern=pat)
        elif len(pat) == 0:
            result = pc.if_else(pc.is_null(self._pa_array), None, False)
        else:
            result = pc.ends_with(self._pa_array, pattern=pat[0])
            for p in pat[1:]:
                result = pc.or_(result, pc.ends_with(self._pa_array, pattern=p))
        return self._convert_bool_result(result, na=na, method_name='endswith')

    def _str_isalnum(self):
        result = pc.utf8_is_alnum(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isalpha(self):
        result = pc.utf8_is_alpha(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isascii(self):
        result = pc.string_is_ascii(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isdecimal(self):
        result = pc.utf8_is_decimal(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isdigit(self):
        result = pc.utf8_is_digit(self._pa_array)
        return self._convert_bool_result(result)

    def _str_islower(self):
        result = pc.utf8_is_lower(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isnumeric(self):
        result = pc.utf8_is_numeric(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isspace(self):
        result = pc.utf8_is_space(self._pa_array)
        return self._convert_bool_result(result)

    def _str_istitle(self):
        result = pc.utf8_is_title(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isupper(self):
        result = pc.utf8_is_upper(self._pa_array)
        return self._convert_bool_result(result)

    def _str_contains(self, pat: Union[bool, dict[str, dict[str, int]], set[str]], case: bool=True, flags: int=0, na: Any=lib.no_default, regex: bool=True):
        if flags:
            raise NotImplementedError(f'contains not implemented with flags={flags!r}')
        if regex:
            pa_contains = pc.match_substring_regex
        else:
            pa_contains = pc.match_substring
        result = pa_contains(self._pa_array, pat, ignore_case=not case)
        return self._convert_bool_result(result, na=na, method_name='contains')

    def _str_match(self, pat: str, case: bool=True, flags: int=0, na: Any=lib.no_default):
        if not pat.startswith('^'):
            pat = f'^{pat}'
        return self._str_contains(pat, case, flags, na, regex=True)

    def _str_fullmatch(self, pat: str, case: bool=True, flags: int=0, na: Any=lib.no_default):
        if not pat.endswith('$') or pat.endswith('\\$'):
            pat = f'{pat}$'
        return self._str_match(pat, case, flags, na)

    def _str_find(self, sub: Union[int, None, str], start: int=0, end: Union[None, int, str]=None):
        if pa_version_under13p0 and (not (start != 0 and end is not None)) and (not (start == 0 and end is None)):
            res_list = self._apply_elementwise(lambda val: val.find(sub, start, end))
            return self._convert_int_result(pa.chunked_array(res_list))
        if (start == 0 or start is None) and end is None:
            result = pc.find_substring(self._pa_array, sub)
        else:
            if sub == '':
                res_list = self._apply_elementwise(lambda val: val.find(sub, start, end))
                return self._convert_int_result(pa.chunked_array(res_list))
            if start is None:
                start_offset = 0
                start = 0
            elif start < 0:
                start_offset = pc.add(start, pc.utf8_length(self._pa_array))
                start_offset = pc.if_else(pc.less(start_offset, 0), 0, start_offset)
            else:
                start_offset = start
            slices = pc.utf8_slice_codeunits(self._pa_array, start, stop=end)
            result = pc.find_substring(slices, sub)
            found = pc.not_equal(result, pa.scalar(-1, type=result.type))
            offset_result = pc.add(result, start_offset)
            result = pc.if_else(found, offset_result, -1)
        return self._convert_int_result(result)