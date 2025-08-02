from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Literal
from pandas._libs import lib
if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    import re
    from pandas._typing import NpDtype, Scalar, Self


class BaseStringArrayMethods(abc.ABC):
    """
    Base class for extension arrays implementing string methods.

    This is where our ExtensionArrays can override the implementation of
    Series.str.<method>. We don't expect this to work with
    3rd-party extension arrays.

    * User calls Series.str.<method>
    * pandas extracts the extension array from the Series
    * pandas calls ``extension_array._str_<method>(*args, **kwargs)``
    * pandas wraps the result, to return to the user.

    See :ref:`Series.str` for the docstring of each method.
    """

    def func_whv4lrx3(self, key):
        if isinstance(key, slice):
            return self._str_slice(start=key.start, stop=key.stop, step=key
                .step)
        else:
            return self._str_get(key)

    @abc.abstractmethod
    def func_4u5magyw(self, pat, flags=0):
        pass

    @abc.abstractmethod
    def func_9ypd9wix(self, width, side='left', fillchar=' '):
        pass

    @abc.abstractmethod
    def func_u0lcsy3c(self, pat, case=True, flags=0, na=None, regex=True):
        pass

    @abc.abstractmethod
    def func_4718j2mr(self, pat, na=None):
        pass

    @abc.abstractmethod
    def func_07o39u4w(self, pat, na=None):
        pass

    @abc.abstractmethod
    def func_4z5qv2ay(self, pat, repl, n=-1, case=True, flags=0, regex=True):
        pass

    @abc.abstractmethod
    def func_s2cxok9z(self, repeats):
        pass

    @abc.abstractmethod
    def func_fdpjb21o(self, pat, case=True, flags=0, na=lib.no_default):
        pass

    @abc.abstractmethod
    def func_glhlddxz(self, pat, case=True, flags=0, na=lib.no_default):
        pass

    @abc.abstractmethod
    def func_oq9h1j27(self, encoding, errors='strict'):
        pass

    @abc.abstractmethod
    def func_x0qycg30(self, sub, start=0, end=None):
        pass

    @abc.abstractmethod
    def func_e68ar6jh(self, sub, start=0, end=None):
        pass

    @abc.abstractmethod
    def func_sjjxeo5g(self, pat, flags=0):
        pass

    @abc.abstractmethod
    def func_xv97tw1a(self, i):
        pass

    @abc.abstractmethod
    def func_g1fjhk6b(self, sub, start=0, end=None):
        pass

    @abc.abstractmethod
    def func_obpq70xp(self, sub, start=0, end=None):
        pass

    @abc.abstractmethod
    def func_ooh2yi79(self, sep):
        pass

    @abc.abstractmethod
    def func_1nywyju6(self, sep, expand):
        pass

    @abc.abstractmethod
    def func_r9q5c5lk(self, sep, expand):
        pass

    @abc.abstractmethod
    def func_5899boco(self):
        pass

    @abc.abstractmethod
    def func_lcioc9hf(self, start=None, stop=None, step=None):
        pass

    @abc.abstractmethod
    def func_zu8lwg8z(self, start=None, stop=None, repl=None):
        pass

    @abc.abstractmethod
    def func_ib5v6btm(self, table):
        pass

    @abc.abstractmethod
    def func_tgnfhwux(self, width, **kwargs):
        pass

    @abc.abstractmethod
    def func_37qbtq12(self, sep='|', dtype=None):
        pass

    @abc.abstractmethod
    def func_qf1xc97s(self):
        pass

    @abc.abstractmethod
    def func_ehc2frkn(self):
        pass

    @abc.abstractmethod
    def func_wbsnu7pw(self):
        pass

    @abc.abstractmethod
    def func_x92krww1(self):
        pass

    @abc.abstractmethod
    def func_o5mcuz6l(self):
        pass

    @abc.abstractmethod
    def func_v8hrcgcp(self):
        pass

    @abc.abstractmethod
    def func_c8blbw2z(self):
        pass

    @abc.abstractmethod
    def func_f14o5jnj(self):
        pass

    @abc.abstractmethod
    def func_8s5d18gy(self):
        pass

    @abc.abstractmethod
    def func_7pbab0b9(self):
        pass

    @abc.abstractmethod
    def func_620mi5s1(self):
        pass

    @abc.abstractmethod
    def func_u31a4aod(self):
        pass

    @abc.abstractmethod
    def func_7csr67gv(self):
        pass

    @abc.abstractmethod
    def func_p1yp23es(self):
        pass

    @abc.abstractmethod
    def func_gmoed4nf(self):
        pass

    @abc.abstractmethod
    def func_1fdnf04t(self):
        pass

    @abc.abstractmethod
    def func_66fv3zo3(self, form):
        pass

    @abc.abstractmethod
    def func_vvv2sfe3(self, to_strip=None):
        pass

    @abc.abstractmethod
    def func_766gnkqd(self, to_strip=None):
        pass

    @abc.abstractmethod
    def func_r6zpucbc(self, to_strip=None):
        pass

    @abc.abstractmethod
    def func_1t3lwq14(self, prefix):
        pass

    @abc.abstractmethod
    def func_lrmc93uc(self, suffix):
        pass

    @abc.abstractmethod
    def func_t2i8evxm(self, pat=None, n=-1, expand=False, regex=None):
        pass

    @abc.abstractmethod
    def func_h3qf6mw7(self, pat=None, n=-1):
        pass

    @abc.abstractmethod
    def func_bhe08751(self, pat, flags=0, expand=True):
        pass
