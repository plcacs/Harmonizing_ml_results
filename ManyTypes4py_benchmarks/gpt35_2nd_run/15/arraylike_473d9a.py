    def _cmp_method(self, other, op) -> Any:
    def __eq__(self, other) -> Any:
    def __ne__(self, other) -> Any:
    def __lt__(self, other) -> Any:
    def __le__(self, other) -> Any:
    def __gt__(self, other) -> Any:
    def __ge__(self, other) -> Any:
    def _logical_method(self, other, op) -> Any:
    def __and__(self, other) -> Any:
    def __rand__(self, other) -> Any:
    def __or__(self, other) -> Any:
    def __ror__(self, other) -> Any:
    def __xor__(self, other) -> Any:
    def __rxor__(self, other) -> Any:
    def _arith_method(self, other, op) -> Any:
    def __add__(self, other) -> Any:
    def __radd__(self, other) -> Any:
    def __sub__(self, other) -> Any:
    def __rsub__(self, other) -> Any:
    def __mul__(self, other) -> Any:
    def __rmul__(self, other) -> Any:
    def __truediv__(self, other) -> Any:
    def __rtruediv__(self, other) -> Any:
    def __floordiv__(self, other) -> Any:
    def __rfloordiv(self, other) -> Any:
    def __mod__(self, other) -> Any:
    def __rmod__(self, other) -> Any:
    def __divmod__(self, other) -> Any:
    def __rdivmod__(self, other) -> Any:
    def __pow__(self, other) -> Any:
    def __rpow__(self, other) -> Any:
def array_ufunc(self, ufunc, method, *inputs, **kwargs) -> Any:
def _standardize_out_kwarg(**kwargs) -> dict:
def dispatch_ufunc_with_out(self, ufunc, method, *inputs, **kwargs) -> Any:
def _assign_where(out, result, where) -> None:
def default_array_ufunc(self, ufunc, method, *inputs, **kwargs) -> Any:
def dispatch_reduction_ufunc(self, ufunc, method, *inputs, **kwargs) -> Any:
