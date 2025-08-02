from __future__ import annotations
import functools
import re
import textwrap
from typing import TYPE_CHECKING, Callable, Sequence, Literal, cast, Optional, Any, Tuple, List
import unicodedata
import warnings
import numpy as np
from pandas._libs import lib
import pandas._libs.missing as libmissing
import pandas._libs.ops as libops
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.missing import isna
from pandas.core.strings.base import BaseStringArrayMethods

if TYPE_CHECKING:
    from pandas._typing import NpDtype, Scalar


class ObjectStringArrayMixin(BaseStringArrayMethods):
    """
    String Methods operating on object-dtype ndarrays.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def func_5nbt5v11(
        self, 
        f: Callable[[Any], Any], 
        na_value: Optional[Scalar] = lib.no_default, 
        dtype: Optional[NpDtype] = None, 
        convert: bool = True
    ) -> np.ndarray:
        """
        Map a callable over valid elements of the array.

        Parameters
        ----------
        f : Callable
            A function to call on each non-NA element.
        na_value : Scalar, optional
            The value to set for NA values. Might also be used for the
            fill value if the callable `f` raises an exception.
            This defaults to ``self.dtype.na_value`` which is ``np.nan``
            for object-dtype and Categorical and ``pd.NA`` for StringArray.
        dtype : Dtype, optional
            The dtype of the result array.
        convert : bool, default True
            Whether to call `maybe_convert_objects` on the resulting ndarray
        """
        if dtype is None:
            dtype = np.dtype('object')
        if na_value is lib.no_default:
            na_value = self.dtype.na_value
        if not len(self):
            return np.array([], dtype=dtype)
        arr = np.asarray(self, dtype=object)
        mask = isna(arr)
        map_convert = convert and not np.all(mask)
        try:
            result = lib.map_infer_mask(arr, f, mask.view(np.uint8), convert=map_convert)
        except (TypeError, AttributeError) as err:
            p_err = (
                '((takes)|(missing)) (?(2)from \\d+ to )?\\d+ (?(3)required )positional arguments?'
            )
            if len(err.args) >= 1 and re.search(p_err, err.args[0]):
                raise err

            def func_s28tlzg1(x: Any) -> Any:
                try:
                    return f(x)
                except (TypeError, AttributeError):
                    return na_value
            return self._str_map(func_s28tlzg1, na_value=na_value, dtype=dtype)
        if not isinstance(result, np.ndarray):
            return result
        if na_value is not np.nan:
            np.putmask(result, mask, na_value)
            if convert and result.dtype == object:
                result = lib.maybe_convert_objects(result)
        return result

    def func_vaedkqp6(
        self, 
        pat: str, 
        flags: int = 0
    ) -> np.ndarray:
        regex = re.compile(pat, flags=flags)
        f = lambda x: len(regex.findall(x))
        return self._str_map(f, dtype='int64')

    def func_ux8wzq97(
        self, 
        width: int, 
        side: Literal['left', 'right', 'both'] = 'left', 
        fillchar: str = ' '
    ) -> np.ndarray:
        if side == 'left':
            f = lambda x: x.rjust(width, fillchar)
        elif side == 'right':
            f = lambda x: x.ljust(width, fillchar)
        elif side == 'both':
            f = lambda x: x.center(width, fillchar)
        else:
            raise ValueError('Invalid side')
        return self._str_map(f)

    def func_eny09hh1(
        self, 
        pat: str, 
        case: bool = True, 
        flags: int = 0, 
        na: Optional[bool] = lib.no_default, 
        regex: bool = True
    ) -> np.ndarray:
        if regex:
            if not case:
                flags |= re.IGNORECASE
            pat_compiled = re.compile(pat, flags=flags)
            f = lambda x: pat_compiled.search(x) is not None
        elif case:
            f = lambda x: pat in x
        else:
            upper_pat = pat.upper()
            f = lambda x: upper_pat in x.upper()
        if na is not lib.no_default and not isna(na) and not isinstance(na, bool):
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.contains is deprecated and will raise in a future version.",
                FutureWarning, 
                stacklevel=find_stack_level()
            )
        return self._str_map(f, na, dtype=np.dtype('bool'))

    def func_b2sr82ig(
        self, 
        pat: str, 
        na: Optional[bool] = lib.no_default
    ) -> np.ndarray:
        f = lambda x: x.startswith(pat)
        if na is not lib.no_default and not isna(na) and not isinstance(na, bool):
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.startswith is deprecated and will raise in a future version.",
                FutureWarning, 
                stacklevel=find_stack_level()
            )
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def func_3caxrill(
        self, 
        pat: str, 
        na: Optional[bool] = lib.no_default
    ) -> np.ndarray:
        f = lambda x: x.endswith(pat)
        if na is not lib.no_default and not isna(na) and not isinstance(na, bool):
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.endswith is deprecated and will raise in a future version.",
                FutureWarning, 
                stacklevel=find_stack_level()
            )
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def func_4dhm4qvp(
        self, 
        pat: str, 
        repl: Any, 
        n: int = -1, 
        case: bool = True, 
        flags: int = 0, 
        regex: bool = True
    ) -> np.ndarray:
        if case is False:
            flags |= re.IGNORECASE
        if regex or flags or callable(repl):
            if not isinstance(pat, re.Pattern):
                if regex is False:
                    pat = re.escape(pat)
                pat = re.compile(pat, flags=flags)
            n = n if n >= 0 else 0
            f = lambda x: pat.sub(repl=repl, string=x, count=n)
        else:
            f = lambda x: x.replace(pat, repl, n)
        return self._str_map(f, dtype=str)

    def func_knbbgio8(
        self, 
        repeats: Any
    ) -> np.ndarray:
        if lib.is_integer(repeats):
            rint = cast(int, repeats)

            def func_63z23vrs(x: Any) -> Any:
                try:
                    return bytes.__mul__(x, rint)
                except TypeError:
                    return str.__mul__(x, rint)
            return self._str_map(func_63z23vrs, dtype=str)
        else:
            from pandas.core.arrays.string_ import BaseStringArray

            def func_lkj0nyse(x: Any, r: Any) -> Any:
                if x is libmissing.NA:
                    return x
                try:
                    return bytes.__mul__(x, r)
                except TypeError:
                    return str.__mul__(x, r)
            result = libops.vec_binop(
                np.asarray(self), 
                np.asarray(repeats, dtype=object), 
                func_lkj0nyse
            )
            if not isinstance(self, BaseStringArray):
                return result
            return type(self)._from_sequence(result, dtype=self.dtype)

    def func_z2vsqpl7(
        self, 
        pat: str, 
        case: bool = True, 
        flags: int = 0, 
        na: Optional[bool] = lib.no_default
    ) -> np.ndarray:
        if not case:
            flags |= re.IGNORECASE
        regex = re.compile(pat, flags=flags)
        f = lambda x: regex.match(x) is not None
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def func_8daydhro(
        self, 
        pat: str, 
        case: bool = True, 
        flags: int = 0, 
        na: Optional[bool] = lib.no_default
    ) -> np.ndarray:
        if not case:
            flags |= re.IGNORECASE
        regex = re.compile(pat, flags=flags)
        f = lambda x: regex.fullmatch(x) is not None
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def func_xjyzjxh0(
        self, 
        encoding: str, 
        errors: str = 'strict'
    ) -> np.ndarray:
        f = lambda x: x.encode(encoding, errors=errors)
        return self._str_map(f, dtype=object)

    def func_a1dnoqdy(
        self, 
        sub: str, 
        start: int = 0, 
        end: Optional[int] = None
    ) -> np.ndarray:
        return self._str_find_(sub, start, end, side='left')

    def func_od478mmm(
        self, 
        sub: str, 
        start: int = 0, 
        end: Optional[int] = None
    ) -> np.ndarray:
        return self._str_find_(sub, start, end, side='right')

    def func_phtuwasd(
        self, 
        sub: str, 
        start: int, 
        end: int, 
        side: Literal['left', 'right']
    ) -> np.ndarray:
        if side == 'left':
            method = 'find'
        elif side == 'right':
            method = 'rfind'
        else:
            raise ValueError('Invalid side')
        if end is None:
            f = lambda x: getattr(x, method)(sub, start)
        else:
            f = lambda x: getattr(x, method)(sub, start, end)
        return self._str_map(f, dtype='int64')

    def func_p1b8xek0(
        self, 
        pat: str, 
        flags: int = 0
    ) -> np.ndarray:
        regex = re.compile(pat, flags=flags)
        return self._str_map(regex.findall, dtype='object')

    def func_5r3j4xoz(
        self, 
        i: int
    ) -> np.ndarray:
        def func_tln692t5(x: Any) -> Any:
            if isinstance(x, dict):
                return x.get(i)
            elif len(x) > i >= -len(x):
                return x[i]
            return self.dtype.na_value
        return self._str_map(func_tln692t5)

    def func_as0to290(
        self, 
        sub: str, 
        start: int = 0, 
        end: Optional[int] = None
    ) -> np.ndarray:
        if end is not None:
            f = lambda x: x.index(sub, start, end)
        else:
            f = lambda x: x.index(sub)
        return self._str_map(f, dtype='int64')

    def func_8547wc47(
        self, 
        sub: str, 
        start: int = 0, 
        end: Optional[int] = None
    ) -> np.ndarray:
        if end is not None:
            f = lambda x: x.rindex(sub, start, end)
        else:
            f = lambda x: x.rindex(sub, start, end)
        return self._str_map(f, dtype='int64')

    def func_lsnzrpkq(
        self, 
        sep: str
    ) -> np.ndarray:
        return self._str_map(lambda x: sep.join(x), dtype='object')

    def func_sye6i40w(
        self, 
        sep: str, 
        expand: bool
    ) -> np.ndarray:
        result = self._str_map(lambda x: x.partition(sep), dtype='object')
        return result

    def func_m43u6kn9(
        self, 
        sep: str, 
        expand: bool
    ) -> np.ndarray:
        return self._str_map(lambda x: x.rpartition(sep), dtype='object')

    def func_6f7cpxbv(self) -> np.ndarray:
        return self._str_map(len, dtype='int64')

    def func_lth77aph(
        self, 
        start: Optional[int] = None, 
        stop: Optional[int] = None, 
        step: Optional[int] = None
    ) -> np.ndarray:
        obj = slice(start, stop, step)
        return self._str_map(lambda x: x[obj], dtype='object')

    def func_a6clybyx(
        self, 
        start: Optional[int] = None, 
        stop: Optional[int] = None, 
        repl: Optional[str] = None
    ) -> np.ndarray:
        if repl is None:
            repl = ''

        def func_tln692t5(x: Any) -> str:
            if x[start:stop] == '':
                local_stop = start
            else:
                local_stop = stop
            y = ''
            if start is not None:
                y += x[:start]
            y += repl
            if stop is not None:
                y += x[local_stop:]
            return y
        return self._str_map(func_tln692t5, dtype='object')

    def func_zxmhb9gp(
        self, 
        pat: Optional[str] = None, 
        n: int = -1, 
        expand: bool = False, 
        regex: Optional[bool] = None
    ) -> np.ndarray:
        if pat is None:
            if n is None or n == 0:
                n = -1
            f = lambda x: x.split(pat, n)
        else:
            if regex is True or isinstance(pat, re.Pattern):
                if not isinstance(pat, re.Pattern):
                    new_pat = re.compile(pat) if regex else re.escape(pat)
                else:
                    new_pat = pat
                if isinstance(new_pat, re.Pattern):
                    if n is None or n == -1:
                        n = 0
                    f = lambda x: new_pat.split(x, maxsplit=n)
                else:
                    if n is None or n == 0:
                        n = -1
                    f = lambda x: x.split(pat, n)
            else:
                new_pat = pat
                if n is None or n == 0:
                    n = -1
                f = lambda x: x.split(new_pat, n)
        return self._str_map(f, dtype='object')

    def func_b5smvyhl(
        self, 
        pat: Optional[str] = None, 
        n: int = -1
    ) -> np.ndarray:
        if n is None or n == 0:
            n = -1
        f = lambda x: x.rsplit(pat, n)
        return self._str_map(f, dtype='object')

    def func_0vqd5nwx(
        self, 
        table: dict
    ) -> np.ndarray:
        return self._str_map(lambda x: x.translate(table), dtype='object')

    def func_hn4wr09s(
        self, 
        width: int, 
        **kwargs: Any
    ) -> np.ndarray:
        kwargs['width'] = width
        tw = textwrap.TextWrapper(**kwargs)
        return self._str_map(lambda s: '\n'.join(tw.wrap(s)), dtype='object')

    def func_01jial4i(
        self, 
        sep: str = '|', 
        dtype: Optional[NpDtype] = None
    ) -> Tuple[np.ndarray, List[str]]:
        from pandas import Series
        if dtype is None:
            dtype = np.int64
        arr = Series(self).fillna('')
        try:
            arr = sep + arr + sep
        except (TypeError, NotImplementedError):
            arr = sep + arr.astype(str) + sep
        tags: set = set()
        for ts in Series(arr, copy=False).str.split(sep):
            tags.update(ts)
        tags2 = sorted(tags - {''})
        _dtype = pandas_dtype(dtype)
        if isinstance(_dtype, np.dtype):
            dummies_dtype = _dtype
        else:
            dummies_dtype = np.bool_
        dummies = np.empty((len(arr), len(tags2)), dtype=dummies_dtype, order='F')

        def func_bbzrojm5(test_elements: Any, element: str) -> bool:
            return element in test_elements

        for i, t in enumerate(tags2):
            pat = sep + t + sep
            dummies[:, i] = lib.map_infer(arr.to_numpy(), functools.partial(_isin, element=pat))
        return dummies, tags2

    def func_tovq27lg(self) -> np.ndarray:
        return self._str_map(lambda x: x.upper(), dtype='object')

    def func_l71y7y38(self) -> np.ndarray:
        return self._str_map(str.isalnum, dtype='bool')

    def func_tj4ulejl(self) -> np.ndarray:
        return self._str_map(str.isalpha, dtype='bool')

    def func_ohwzb58a(self) -> np.ndarray:
        return self._str_map(str.isascii, dtype='bool')

    def func_4dmf0zdk(self) -> np.ndarray:
        return self._str_map(str.isdecimal, dtype='bool')

    def func_xgq1nw3k(self) -> np.ndarray:
        return self._str_map(str.isdigit, dtype='bool')

    def func_n8dizli2(self) -> np.ndarray:
        return self._str_map(str.islower, dtype='bool')

    def func_5njrvqfv(self) -> np.ndarray:
        return self._str_map(str.isnumeric, dtype='bool')

    def func_sgnt8jgt(self) -> np.ndarray:
        return self._str_map(str.isspace, dtype='bool')

    def func_iixfso0r(self) -> np.ndarray:
        return self._str_map(str.istitle, dtype='bool')

    def func_0bbjivww(self) -> np.ndarray:
        return self._str_map(str.isupper, dtype='bool')

    def func_e443lrz3(self) -> np.ndarray:
        return self._str_map(str.capitalize, dtype='object')

    def func_0e7ssq0n(self) -> np.ndarray:
        return self._str_map(str.casefold, dtype='object')

    def func_xqrmtf0y(self) -> np.ndarray:
        return self._str_map(str.title, dtype='object')

    def func_6e7pha9h(self) -> np.ndarray:
        return self._str_map(str.swapcase, dtype='object')

    def func_fyvkch0l(self) -> np.ndarray:
        return self._str_map(str.lower, dtype='object')

    def func_f1czkupa(
        self, 
        form: str
    ) -> np.ndarray:
        f = lambda x: unicodedata.normalize(form, x)
        return self._str_map(f, dtype='object')

    def func_v3s85kc1(
        self, 
        to_strip: Optional[str] = None
    ) -> np.ndarray:
        return self._str_map(lambda x: x.strip(to_strip), dtype='object')

    def func_cefs863i(
        self, 
        to_strip: Optional[str] = None
    ) -> np.ndarray:
        return self._str_map(lambda x: x.lstrip(to_strip), dtype='object')

    def func_apwni9q7(
        self, 
        to_strip: Optional[str] = None
    ) -> np.ndarray:
        return self._str_map(lambda x: x.rstrip(to_strip), dtype='object')

    def func_j8wgdyv3(
        self, 
        prefix: str
    ) -> np.ndarray:
        return self._str_map(lambda x: x.removeprefix(prefix), dtype='object')

    def func_c8gdm5iu(
        self, 
        suffix: str
    ) -> np.ndarray:
        return self._str_map(lambda x: x.removesuffix(suffix), dtype='object')

    def func_m7fbi6il(
        self, 
        pat: str, 
        flags: int = 0, 
        expand: bool = True
    ) -> Any:
        regex = re.compile(pat, flags=flags)
        na_value = self.dtype.na_value
        if not expand:

            def func_s28tlzg1(x: Any) -> Any:
                m = regex.search(x)
                return m.groups()[0] if m else na_value
            return self._str_map(func_s28tlzg1, convert=False)
        empty_row: List[Any] = [na_value] * regex.groups

        def func_tln692t5(x: Any) -> List[Any]:
            if not isinstance(x, str):
                return empty_row
            m = regex.search(x)
            if m:
                return [na_value if item is None else item for item in m.groups()]
            else:
                return empty_row
        return np.array([func_tln692t5(val) for val in np.asarray(self)], dtype='object')
