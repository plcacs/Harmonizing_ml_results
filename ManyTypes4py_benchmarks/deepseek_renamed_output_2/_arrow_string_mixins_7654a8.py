from __future__ import annotations
from functools import partial
import re
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, Pattern, Sequence
import numpy as np
from pandas._libs import lib
from pandas.compat import pa_version_under10p1, pa_version_under11p0, pa_version_under13p0, pa_version_under17p0
if not pa_version_under10p1:
    import pyarrow as pa
    import pyarrow.compute as pc
if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pandas._typing import Scalar, Self


class ArrowStringArrayMixin:

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def func_jccdkkis(self, result: Any, na: Any = lib.no_default, method_name: Optional[str] = None) -> Any:
        raise NotImplementedError

    def func_yl5ftohi(self, result: Any) -> Any:
        raise NotImplementedError

    def func_etlamf88(self, func: Callable[[Any], Any]) -> Any:
        raise NotImplementedError

    def func_78l621e1(self) -> Any:
        result = pc.utf8_length(self._pa_array)
        return self._convert_int_result(result)

    def func_0i6k0wk3(self) -> Self:
        return type(self)(pc.utf8_lower(self._pa_array))

    def func_bsdscxhz(self) -> Self:
        return type(self)(pc.utf8_upper(self._pa_array))

    def func_snelh9z8(self, to_strip: Optional[str] = None) -> Self:
        if to_strip is None:
            result = pc.utf8_trim_whitespace(self._pa_array)
        else:
            result = pc.utf8_trim(self._pa_array, characters=to_strip)
        return type(self)(result)

    def func_cydf8fiy(self, to_strip: Optional[str] = None) -> Self:
        if to_strip is None:
            result = pc.utf8_ltrim_whitespace(self._pa_array)
        else:
            result = pc.utf8_ltrim(self._pa_array, characters=to_strip)
        return type(self)(result)

    def func_c04ms2px(self, to_strip: Optional[str] = None) -> Self:
        if to_strip is None:
            result = pc.utf8_rtrim_whitespace(self._pa_array)
        else:
            result = pc.utf8_rtrim(self._pa_array, characters=to_strip)
        return type(self)(result)

    def func_2k6tquiy(self, width: int, side: Literal['left', 'right', 'both'] = 'left', fillchar: str = ' ') -> Self:
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
                pa_pad = partial(pc.utf8_center, lean_left_on_odd_padding=
                    lean_left)
        else:
            raise ValueError(
                f"Invalid side: {side}. Side must be one of 'left', 'right', 'both'"
                )
        return type(self)(pa_pad(self._pa_array, width=width, padding=fillchar)
            )

    def func_ci9u6fbk(self, i: int) -> Self:
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
        selected = pc.utf8_slice_codeunits(self._pa_array, start=start,
            stop=stop, step=step)
        null_value = pa.scalar(None, type=self._pa_array.type)
        result = pc.if_else(not_out_of_bounds, selected, null_value)
        return type(self)(result)

    def func_jevzmtv2(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> Self:
        if pa_version_under11p0:
            result = self._apply_elementwise(lambda val: val[start:stop:step])
            return type(self)(pa.chunked_array(result, type=self._pa_array.
                type))
        if start is None:
            if step is not None and step < 0:
                start = -1
            else:
                start = 0
        if step is None:
            step = 1
        return type(self)(pc.utf8_slice_codeunits(self._pa_array, start=
            start, stop=stop, step=step))

    def func_2jha3nt0(self, start: Optional[int] = None, stop: Optional[int] = None, repl: Optional[str] = None) -> Self:
        if repl is None:
            repl = ''
        if start is None:
            start = 0
        if stop is None:
            stop = np.iinfo(np.int64).max
        return type(self)(pc.utf8_replace_slice(self._pa_array, start, stop,
            repl))

    def func_uf8fnxc9(self, pat: Union[str, Pattern[str]], repl: Union[str, Callable[[str], str]], n: int = -1, case: bool = True, flags: int = 0, regex: bool = True) -> Self:
        if isinstance(pat, re.Pattern) or callable(repl) or not case or flags:
            raise NotImplementedError(
                'replace is not supported with a re.Pattern, callable repl, case=False, or flags!=0'
                )
        func = pc.replace_substring_regex if regex else pc.replace_substring
        pa_max_replacements = None if n < 0 else n
        result = func(self._pa_array, pattern=pat, replacement=repl,
            max_replacements=pa_max_replacements)
        return type(self)(result)

    def func_gpu63ozs(self) -> Self:
        return type(self)(pc.utf8_capitalize(self._pa_array))

    def func_o5dpmzu8(self) -> Self:
        return type(self)(pc.utf8_title(self._pa_array))

    def func_nmlk0k68(self) -> Self:
        return type(self)(pc.utf8_swapcase(self._pa_array))

    def func_02qrkash(self, prefix: str) -> Self:
        if not pa_version_under13p0:
            starts_with = pc.starts_with(self._pa_array, pattern=prefix)
            removed = pc.utf8_slice_codeunits(self._pa_array, len(prefix))
            result = pc.if_else(starts_with, removed, self._pa_array)
            return type(self)(result)
        predicate = lambda val: val.removeprefix(prefix)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def func_2gzgqetn(self, suffix: str) -> Self:
        ends_with = pc.ends_with(self._pa_array, pattern=suffix)
        removed = pc.utf8_slice_codeunits(self._pa_array, 0, stop=-len(suffix))
        result = pc.if_else(ends_with, removed, self._pa_array)
        return type(self)(result)

    def func_oe1w03om(self, pat: Union[str, Sequence[str]], na: Any = lib.no_default) -> Any:
        if isinstance(pat, str):
            result = pc.starts_with(self._pa_array, pattern=pat)
        elif len(pat) == 0:
            result = pc.if_else(pc.is_null(self._pa_array), None, False)
        else:
            result = pc.starts_with(self._pa_array, pattern=pat[0])
            for p in pat[1:]:
                result = pc.or_(result, pc.starts_with(self._pa_array,
                    pattern=p))
        return self._convert_bool_result(result, na=na, method_name=
            'startswith')

    def func_7berzn3m(self, pat: Union[str, Sequence[str]], na: Any = lib.no_default) -> Any:
        if isinstance(pat, str):
            result = pc.ends_with(self._pa_array, pattern=pat)
        elif len(pat) == 0:
            result = pc.if_else(pc.is_null(self._pa_array), None, False)
        else:
            result = pc.ends_with(self._pa_array, pattern=pat[0])
            for p in pat[1:]:
                result = pc.or_(result, pc.ends_with(self._pa_array, pattern=p)
                    )
        return self._convert_bool_result(result, na=na, method_name='endswith')

    def func_db48kci9(self) -> Any:
        result = pc.utf8_is_alnum(self._pa_array)
        return self._convert_bool_result(result)

    def func_0xcplj4q(self) -> Any:
        result = pc.utf8_is_alpha(self._pa_array)
        return self._convert_bool_result(result)

    def func_1yskerle(self) -> Any:
        result = pc.string_is_ascii(self._pa_array)
        return self._convert_bool_result(result)

    def func_w7nqbb61(self) -> Any:
        result = pc.utf8_is_decimal(self._pa_array)
        return self._convert_bool_result(result)

    def func_mx57koys(self) -> Any:
        result = pc.utf8_is_digit(self._pa_array)
        return self._convert_bool_result(result)

    def func_0lds0czf(self) -> Any:
        result = pc.utf8_is_lower(self._pa_array)
        return self._convert_bool_result(result)

    def func_ye8i9799(self) -> Any:
        result = pc.utf8_is_numeric(self._pa_array)
        return self._convert_bool_result(result)

    def func_8tr4zou4(self) -> Any:
        result = pc.utf8_is_space(self._pa_array)
        return self._convert_bool_result(result)

    def func_v6r6vl0l(self) -> Any:
        result = pc.utf8_is_title(self._pa_array)
        return self._convert_bool_result(result)

    def func_8mkggzhf(self) -> Any:
        result = pc.utf8_is_upper(self._pa_array)
        return self._convert_bool_result(result)

    def func_15w3c7lp(self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default,
        regex: bool = True) -> Any:
        if flags:
            raise NotImplementedError(
                f'contains not implemented with flags={flags!r}')
        if regex:
            pa_contains = pc.match_substring_regex
        else:
            pa_contains = pc.match_substring
        result = pa_contains(self._pa_array, pat, ignore_case=not case)
        return self._convert_bool_result(result, na=na, method_name='contains')

    def func_ag0m32rd(self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default) -> Any:
        if not pat.startswith('^'):
            pat = f'^{pat}'
        return self._str_contains(pat, case, flags, na, regex=True)

    def func_8h2so8wn(self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default) -> Any:
        if not pat.endswith('$') or pat.endswith('\\$'):
            pat = f'{pat}$'
        return self._str_match(pat, case, flags, na)

    def func_sm0rtkui(self, sub: str, start: int = 0, end: Optional[int] = None) -> Any:
        if pa_version_under13p0 and not (start != 0 and end is not None
            ) and not (start == 0 and end is None):
            res_list = self._apply_elementwise(lambda val: val.find(sub,
                start, end))
            return self._convert_int_result(pa.chunked_array(res_list))
        if (start == 0 or start is None) and end is None:
            result = pc.find_substring(self._pa_array, sub)
        else:
            if sub == '':
                res_list = self._apply_elementwise(lambda val: val.find(sub,
                    start, end))
                return self._convert_int_result(pa.chunked_array(res_list))
            if start is None:
                start_offset = 0
                start = 0
            elif start < 0:
                start_offset = pc.add(start, pc.utf8_length(self._pa_array))
                start_offset = pc.if_else(pc.less(start_offset, 0), 0,
                    start_offset)
            else:
                start_offset = start
            slices = pc.utf8_slice_codeunits(self._pa_array, start, stop=end)
            result = pc.find_substring(slices, sub)
            found = pc.not_equal(result, pa.scalar(-1, type=result.type))
            offset_result = pc.add(result, start_offset)
            result = pc.if_else(found, offset_result, -1)
        return self._convert_int_result(result)
