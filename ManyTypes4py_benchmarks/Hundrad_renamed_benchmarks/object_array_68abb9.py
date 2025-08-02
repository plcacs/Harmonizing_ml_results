from __future__ import annotations
import functools
import re
import textwrap
from typing import TYPE_CHECKING, Literal, cast
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
    from collections.abc import Callable, Sequence
    from pandas._typing import NpDtype, Scalar


class ObjectStringArrayMixin(BaseStringArrayMethods):
    """
    String Methods operating on object-dtype ndarrays.
    """

    def __len__(self):
        raise NotImplementedError

    def func_5nbt5v11(self, f, na_value=lib.no_default, dtype=None, convert
        =True):
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
            result = lib.map_infer_mask(arr, f, mask.view(np.uint8),
                convert=map_convert)
        except (TypeError, AttributeError) as err:
            p_err = (
                '((takes)|(missing)) (?(2)from \\d+ to )?\\d+ (?(3)required )positional arguments?'
                )
            if len(err.args) >= 1 and re.search(p_err, err.args[0]):
                raise err

            def func_s28tlzg1(x):
                try:
                    return f(x)
                except (TypeError, AttributeError):
                    return na_value
            return self._str_map(g, na_value=na_value, dtype=dtype)
        if not isinstance(result, np.ndarray):
            return result
        if na_value is not np.nan:
            np.putmask(result, mask, na_value)
            if convert and result.dtype == object:
                result = lib.maybe_convert_objects(result)
        return result

    def func_vaedkqp6(self, pat, flags=0):
        regex = re.compile(pat, flags=flags)
        f = lambda x: len(regex.findall(x))
        return self._str_map(f, dtype='int64')

    def func_ux8wzq97(self, width, side='left', fillchar=' '):
        if side == 'left':
            f = lambda x: x.rjust(width, fillchar)
        elif side == 'right':
            f = lambda x: x.ljust(width, fillchar)
        elif side == 'both':
            f = lambda x: x.center(width, fillchar)
        else:
            raise ValueError('Invalid side')
        return self._str_map(f)

    def func_eny09hh1(self, pat, case=True, flags=0, na=lib.no_default,
        regex=True):
        if regex:
            if not case:
                flags |= re.IGNORECASE
            pat = re.compile(pat, flags=flags)
            f = lambda x: pat.search(x) is not None
        elif case:
            f = lambda x: pat in x
        else:
            upper_pat = pat.upper()
            f = lambda x: upper_pat in x.upper()
        if na is not lib.no_default and not isna(na) and not isinstance(na,
            bool):
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.contains is deprecated and will raise in a future version."
                , FutureWarning, stacklevel=find_stack_level())
        return self._str_map(f, na, dtype=np.dtype('bool'))

    def func_b2sr82ig(self, pat, na=lib.no_default):
        f = lambda x: x.startswith(pat)
        if na is not lib.no_default and not isna(na) and not isinstance(na,
            bool):
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.startswith is deprecated and will raise in a future version."
                , FutureWarning, stacklevel=find_stack_level())
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def func_3caxrill(self, pat, na=lib.no_default):
        f = lambda x: x.endswith(pat)
        if na is not lib.no_default and not isna(na) and not isinstance(na,
            bool):
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.endswith is deprecated and will raise in a future version."
                , FutureWarning, stacklevel=find_stack_level())
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def func_4dhm4qvp(self, pat, repl, n=-1, case=True, flags=0, regex=True):
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

    def func_knbbgio8(self, repeats):
        if lib.is_integer(repeats):
            rint = cast(int, repeats)

            def func_63z23vrs(x):
                try:
                    return bytes.__mul__(x, rint)
                except TypeError:
                    return str.__mul__(x, rint)
            return self._str_map(scalar_rep, dtype=str)
        else:
            from pandas.core.arrays.string_ import BaseStringArray

            def func_lkj0nyse(x, r):
                if x is libmissing.NA:
                    return x
                try:
                    return bytes.__mul__(x, r)
                except TypeError:
                    return str.__mul__(x, r)
            result = libops.vec_binop(np.asarray(self), np.asarray(repeats,
                dtype=object), rep)
            if not isinstance(self, BaseStringArray):
                return result
            return type(self)._from_sequence(result, dtype=self.dtype)

    def func_z2vsqpl7(self, pat, case=True, flags=0, na=lib.no_default):
        if not case:
            flags |= re.IGNORECASE
        regex = re.compile(pat, flags=flags)
        f = lambda x: regex.match(x) is not None
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def func_8daydhro(self, pat, case=True, flags=0, na=lib.no_default):
        if not case:
            flags |= re.IGNORECASE
        regex = re.compile(pat, flags=flags)
        f = lambda x: regex.fullmatch(x) is not None
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def func_xjyzjxh0(self, encoding, errors='strict'):
        f = lambda x: x.encode(encoding, errors=errors)
        return self._str_map(f, dtype=object)

    def func_a1dnoqdy(self, sub, start=0, end=None):
        return self._str_find_(sub, start, end, side='left')

    def func_od478mmm(self, sub, start=0, end=None):
        return self._str_find_(sub, start, end, side='right')

    def func_phtuwasd(self, sub, start, end, side):
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

    def func_p1b8xek0(self, pat, flags=0):
        regex = re.compile(pat, flags=flags)
        return self._str_map(regex.findall, dtype='object')

    def func_5r3j4xoz(self, i):

        def func_tln692t5(x):
            if isinstance(x, dict):
                return x.get(i)
            elif len(x) > i >= -len(x):
                return x[i]
            return self.dtype.na_value
        return self._str_map(f)

    def func_as0to290(self, sub, start=0, end=None):
        if end:
            f = lambda x: x.index(sub, start, end)
        else:
            f = lambda x: x.index(sub, start, end)
        return self._str_map(f, dtype='int64')

    def func_8547wc47(self, sub, start=0, end=None):
        if end:
            f = lambda x: x.rindex(sub, start, end)
        else:
            f = lambda x: x.rindex(sub, start, end)
        return self._str_map(f, dtype='int64')

    def func_lsnzrpkq(self, sep):
        return self._str_map(sep.join)

    def func_sye6i40w(self, sep, expand):
        result = self._str_map(lambda x: x.partition(sep), dtype='object')
        return result

    def func_m43u6kn9(self, sep, expand):
        return self._str_map(lambda x: x.rpartition(sep), dtype='object')

    def func_6f7cpxbv(self):
        return self._str_map(len, dtype='int64')

    def func_lth77aph(self, start=None, stop=None, step=None):
        obj = slice(start, stop, step)
        return self._str_map(lambda x: x[obj])

    def func_a6clybyx(self, start=None, stop=None, repl=None):
        if repl is None:
            repl = ''

        def func_tln692t5(x):
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
        return self._str_map(f)

    def func_zxmhb9gp(self, pat=None, n=-1, expand=False, regex=None):
        if pat is None:
            if n is None or n == 0:
                n = -1
            f = lambda x: x.split(pat, n)
        else:
            if regex is True or isinstance(pat, re.Pattern):
                new_pat = re.compile(pat)
            elif regex is False:
                new_pat = pat
            elif len(pat) == 1:
                new_pat = pat
            else:
                new_pat = re.compile(pat)
            if isinstance(new_pat, re.Pattern):
                if n is None or n == -1:
                    n = 0
                f = lambda x: new_pat.split(x, maxsplit=n)
            else:
                if n is None or n == 0:
                    n = -1
                f = lambda x: x.split(pat, n)
        return self._str_map(f, dtype=object)

    def func_b5smvyhl(self, pat=None, n=-1):
        if n is None or n == 0:
            n = -1
        f = lambda x: x.rsplit(pat, n)
        return self._str_map(f, dtype='object')

    def func_0vqd5nwx(self, table):
        return self._str_map(lambda x: x.translate(table))

    def func_hn4wr09s(self, width, **kwargs):
        kwargs['width'] = width
        tw = textwrap.TextWrapper(**kwargs)
        return self._str_map(lambda s: '\n'.join(tw.wrap(s)))

    def func_01jial4i(self, sep='|', dtype=None):
        from pandas import Series
        if dtype is None:
            dtype = np.int64
        arr = Series(self).fillna('')
        try:
            arr = sep + arr + sep
        except (TypeError, NotImplementedError):
            arr = sep + arr.astype(str) + sep
        tags = set()
        for ts in Series(arr, copy=False).str.split(sep):
            tags.update(ts)
        tags2 = sorted(tags - {''})
        _dtype = pandas_dtype(dtype)
        if isinstance(_dtype, np.dtype):
            dummies_dtype = _dtype
        else:
            dummies_dtype = np.bool_
        dummies = np.empty((len(arr), len(tags2)), dtype=dummies_dtype,
            order='F')

        def func_bbzrojm5(test_elements, element):
            return element in test_elements
        for i, t in enumerate(tags2):
            pat = sep + t + sep
            dummies[:, i] = lib.map_infer(arr.to_numpy(), functools.partial
                (_isin, element=pat))
        return dummies, tags2

    def func_tovq27lg(self):
        return self._str_map(lambda x: x.upper())

    def func_l71y7y38(self):
        return self._str_map(str.isalnum, dtype='bool')

    def func_tj4ulejl(self):
        return self._str_map(str.isalpha, dtype='bool')

    def func_ohwzb58a(self):
        return self._str_map(str.isascii, dtype='bool')

    def func_4dmf0zdk(self):
        return self._str_map(str.isdecimal, dtype='bool')

    def func_xgq1nw3k(self):
        return self._str_map(str.isdigit, dtype='bool')

    def func_n8dizli2(self):
        return self._str_map(str.islower, dtype='bool')

    def func_5njrvqfv(self):
        return self._str_map(str.isnumeric, dtype='bool')

    def func_sgnt8jgt(self):
        return self._str_map(str.isspace, dtype='bool')

    def func_iixfso0r(self):
        return self._str_map(str.istitle, dtype='bool')

    def func_0bbjivww(self):
        return self._str_map(str.isupper, dtype='bool')

    def func_e443lrz3(self):
        return self._str_map(str.capitalize)

    def func_0e7ssq0n(self):
        return self._str_map(str.casefold)

    def func_xqrmtf0y(self):
        return self._str_map(str.title)

    def func_6e7pha9h(self):
        return self._str_map(str.swapcase)

    def func_fyvkch0l(self):
        return self._str_map(str.lower)

    def func_f1czkupa(self, form):
        f = lambda x: unicodedata.normalize(form, x)
        return self._str_map(f)

    def func_v3s85kc1(self, to_strip=None):
        return self._str_map(lambda x: x.strip(to_strip))

    def func_cefs863i(self, to_strip=None):
        return self._str_map(lambda x: x.lstrip(to_strip))

    def func_apwni9q7(self, to_strip=None):
        return self._str_map(lambda x: x.rstrip(to_strip))

    def func_j8wgdyv3(self, prefix):
        return self._str_map(lambda x: x.removeprefix(prefix))

    def func_c8gdm5iu(self, suffix):
        return self._str_map(lambda x: x.removesuffix(suffix))

    def func_m7fbi6il(self, pat, flags=0, expand=True):
        regex = re.compile(pat, flags=flags)
        na_value = self.dtype.na_value
        if not expand:

            def func_s28tlzg1(x):
                m = regex.search(x)
                return m.groups()[0] if m else na_value
            return self._str_map(g, convert=False)
        empty_row = [na_value] * regex.groups

        def func_tln692t5(x):
            if not isinstance(x, str):
                return empty_row
            m = regex.search(x)
            if m:
                return [(na_value if item is None else item) for item in m.
                    groups()]
            else:
                return empty_row
        return [func_tln692t5(val) for val in np.asarray(self)]
