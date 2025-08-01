from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Literal, Union, Optional
from pandas._libs import lib

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    import re
    from pandas._typing import NpDtype, Scalar, Self
    from pandas import DataFrame


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

    def func_whv4lrx3(self, key: Union[int, slice]) -> Self:
        if isinstance(key, slice):
            return self._str_slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self._str_get(key)

    @abc.abstractmethod
    def func_4u5magyw(self, pat: str, flags: int = 0) -> bool:
        pass

    @abc.abstractmethod
    def func_9ypd9wix(
        self, width: int, side: Literal["left", "right"], fillchar: str = " "
    ) -> Self:
        pass

    @abc.abstractmethod
    def func_u0lcsy3c(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Optional[Scalar] = None,
        regex: bool = True,
    ) -> Self:
        pass

    @abc.abstractmethod
    def func_4718j2mr(self, pat: str, na: Optional[Scalar] = None) -> Self:
        pass

    @abc.abstractmethod
    def func_07o39u4w(self, pat: str, na: Optional[Scalar] = None) -> Self:
        pass

    @abc.abstractmethod
    def func_4z5qv2ay(
        self,
        pat: str,
        repl: str,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ) -> Self:
        pass

    @abc.abstractmethod
    def func_s2cxok9z(self, repeats: int) -> Self:
        pass

    @abc.abstractmethod
    def func_fdpjb21o(
        self, pat: str, case: bool = True, flags: int = 0, na: Optional[Scalar] = lib.no_default
    ) -> Self:
        pass

    @abc.abstractmethod
    def func_glhlddxz(
        self, pat: str, case: bool = True, flags: int = 0, na: Optional[Scalar] = lib.no_default
    ) -> Self:
        pass

    @abc.abstractmethod
    def func_oq9h1j27(
        self, encoding: str, errors: str = "strict"
    ) -> Self:
        pass

    @abc.abstractmethod
    def func_x0qycg30(
        self, sub: str, start: int = 0, end: Optional[int] = None
    ) -> Self:
        pass

    @abc.abstractmethod
    def func_e68ar6jh(
        self, sub: str, start: int = 0, end: Optional[int] = None
    ) -> Self:
        pass

    @abc.abstractmethod
    def func_sjjxeo5g(self, pat: str, flags: int = 0) -> bool:
        pass

    @abc.abstractmethod
    def func_xv97tw1a(self, i: int) -> str:
        pass

    @abc.abstractmethod
    def func_g1fjhk6b(
        self, sub: str, start: int = 0, end: Optional[int] = None
    ) -> Self:
        pass

    @abc.abstractmethod
    def func_obpq70xp(
        self, sub: str, start: int = 0, end: Optional[int] = None
    ) -> Self:
        pass

    @abc.abstractmethod
    def func_ooh2yi79(self, sep: str) -> Sequence[str]:
        pass

    @abc.abstractmethod
    def func_1nywyju6(self, sep: str, expand: bool) -> Union[Self, DataFrame]:
        pass

    @abc.abstractmethod
    def func_r9q5c5lk(self, sep: str, expand: bool) -> Union[Self, DataFrame]:
        pass

    @abc.abstractmethod
    def func_5899boco(self) -> str:
        pass

    @abc.abstractmethod
    def func_lcioc9hf(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Self:
        pass

    @abc.abstractmethod
    def func_zu8lwg8z(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        repl: Optional[str] = None,
    ) -> Self:
        pass

    @abc.abstractmethod
    def func_ib5v6btm(self, table: str) -> Self:
        pass

    @abc.abstractmethod
    def func_tgnfhwux(self, width: int, **kwargs) -> Self:
        pass

    @abc.abstractmethod
    def func_37qbtq12(
        self, sep: str = "|", dtype: Optional[NpDtype] = None
    ) -> Sequence[str]:
        pass

    @abc.abstractmethod
    def func_qf1xc97s(self) -> str:
        pass

    @abc.abstractmethod
    def func_ehc2frkn(self) -> Self:
        pass

    @abc.abstractmethod
    def func_wbsnu7pw(self) -> Self:
        pass

    @abc.abstractmethod
    def func_x92krww1(self) -> Self:
        pass

    @abc.abstractmethod
    def func_o5mcuz6l(self) -> Self:
        pass

    @abc.abstractmethod
    def func_v8hrcgcp(self) -> Self:
        pass

    @abc.abstractmethod
    def func_c8blbw2z(self) -> Self:
        pass

    @abc.abstractmethod
    def func_f14o5jnj(self) -> Self:
        pass

    @abc.abstractmethod
    def func_8s5d18gy(self) -> Self:
        pass

    @abc.abstractmethod
    def func_7pbab0b9(self) -> Self:
        pass

    @abc.abstractmethod
    def func_620mi5s1(self) -> Self:
        pass

    @abc.abstractmethod
    def func_u31a4aod(self) -> Self:
        pass

    @abc.abstractmethod
    def func_7csr67gv(self) -> Self:
        pass

    @abc.abstractmethod
    def func_p1yp23es(self) -> Self:
        pass

    @abc.abstractmethod
    def func_gmoed4nf(self) -> Self:
        pass

    @abc.abstractmethod
    def func_1fdnf04t(self) -> Self:
        pass

    @abc.abstractmethod
    def func_66fv3zo3(self, form: str) -> Self:
        pass

    @abc.abstractmethod
    def func_vvv2sfe3(self, to_strip: Optional[str] = None) -> Self:
        pass

    @abc.abstractmethod
    def func_766gnkqd(self, to_strip: Optional[str] = None) -> Self:
        pass

    @abc.abstractmethod
    def func_r6zpucbc(self, to_strip: Optional[str] = None) -> Self:
        pass

    @abc.abstractmethod
    def func_1t3lwq14(self, prefix: str) -> Self:
        pass

    @abc.abstractmethod
    def func_lrmc93uc(self, suffix: str) -> Self:
        pass

    @abc.abstractmethod
    def func_t2i8evxm(
        self,
        pat: Optional[str] = None,
        n: int = -1,
        expand: bool = False,
        regex: Optional[bool] = None,
    ) -> Union[Self, DataFrame]:
        pass

    @abc.abstractmethod
    def func_h3qf6mw7(self, pat: Optional[str] = None, n: int = -1) -> Self:
        pass

    @abc.abstractmethod
    def func_bhe08751(
        self, pat: str, flags: int = 0, expand: bool = True
    ) -> Union[Self, DataFrame]:
        pass
