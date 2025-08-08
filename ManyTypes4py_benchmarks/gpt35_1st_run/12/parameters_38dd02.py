from typing import Any, Union, List
from freqtrade.enums import HyperoptState
from freqtrade.optimize.hyperopt_tools import HyperoptStateContainer
from freqtrade.exceptions import OperationalException

class BaseParameter(ABC):
    def __init__(self, *, default: Any, space: str = None, optimize: bool = True, load: bool = True, **kwargs: Any) -> None:
    def get_space(self, name: str) -> Any:
    def can_optimize(self) -> bool:

class NumericParameter(BaseParameter):
    float_or_int: Union[int, float]
    def __init__(self, low: Union[int, List[int]], high: Union[int, List[int], None], *, default: Any, space: str = None, optimize: bool = True, load: bool = True, **kwargs: Any) -> None:
    def get_space(self, name: str) -> Any:
    @property
    def range(self) -> List[int]:

class IntParameter(NumericParameter):
    def __init__(self, low: Union[int, List[int]], high: Union[int, List[int], None], *, default: Any, space: str = None, optimize: bool = True, load: bool = True, **kwargs: Any) -> None:
    def get_space(self, name: str) -> Any:

class RealParameter(NumericParameter):
    def __init__(self, low: Union[int, List[int]], high: Union[int, List[int], None], *, default: Any, space: str = None, optimize: bool = True, load: bool = True, **kwargs: Any) -> None:
    def get_space(self, name: str) -> Any:

class DecimalParameter(NumericParameter):
    def __init__(self, low: Union[int, List[int]], high: Union[int, List[int], None], *, default: Any, decimals: int = 3, space: str = None, optimize: bool = True, load: bool = True, **kwargs: Any) -> None:
    def get_space(self, name: str) -> Any:
    @property
    def range(self) -> List[float]:

class CategoricalParameter(BaseParameter):
    def __init__(self, categories: List[Any], *, default: Any = None, space: str = None, optimize: bool = True, load: bool = True, **kwargs: Any) -> None:
    def get_space(self, name: str) -> Any:
    @property
    def range(self) -> List[Any]:

class BooleanParameter(CategoricalParameter):
    def __init__(self, *, default: Any = None, space: str = None, optimize: bool = True, load: bool = True, **kwargs: Any) -> None:
