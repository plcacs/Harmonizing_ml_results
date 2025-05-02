import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, Union, List, Optional
from freqtrade.enums import HyperoptState
from freqtrade.optimize.hyperopt_tools import HyperoptStateContainer
with suppress(ImportError):
    from skopt.space import Categorical, Integer, Real
    from freqtrade.optimize.space import SKDecimal
from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)

class BaseParameter(ABC):
    """
    Defines a parameter that can be optimized by hyperopt.
    """
    in_space: bool = False

    def __init__(self, *, default: Any, space: Optional[str] = None, optimize: bool = True, load: bool = True, **kwargs: Any) -> None:
        if 'name' in kwargs:
            raise OperationalException('Name is determined by parameter field name and can not be specified manually.')
        self.category: Optional[str] = space
        self._space_params: dict = kwargs
        self.value: Any = default
        self.optimize: bool = optimize
        self.load: bool = load

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value})'

    @abstractmethod
    def get_space(self, name: str) -> Any:
        pass

    def can_optimize(self) -> bool:
        return self.in_space and self.optimize and (HyperoptStateContainer.state != HyperoptState.OPTIMIZE)

class NumericParameter(BaseParameter):
    """Internal parameter used for Numeric purposes"""
    float_or_int = int | float

    def __init__(self, low: Union[int, float, Sequence], high: Optional[Union[int, float]] = None, *, default: Union[int, float], space: Optional[str] = None, optimize: bool = True, load: bool = True, **kwargs: Any) -> None:
        if high is not None and isinstance(low, Sequence):
            raise OperationalException(f'{self.__class__.__name__} space invalid.')
        if high is None or isinstance(low, Sequence):
            if not isinstance(low, Sequence) or len(low) != 2:
                raise OperationalException(f'{self.__class__.__name__} space must be [low, high]')
            self.low, self.high = low
        else:
            self.low = low
            self.high = high
        super().__init__(default=default, space=space, optimize=optimize, load=load, **kwargs)

class IntParameter(NumericParameter):

    def __init__(self, low: int, high: Optional[int] = None, *, default: int, space: Optional[str] = None, optimize: bool = True, load: bool = True, **kwargs: Any) -> None:
        super().__init__(low=low, high=high, default=default, space=space, optimize=optimize, load=load, **kwargs)

    def get_space(self, name: str) -> Integer:
        return Integer(low=self.low, high=self.high, name=name, **self._space_params)

    @property
    def range(self) -> range:
        if self.can_optimize():
            return range(self.low, self.high + 1)
        else:
            return range(self.value, self.value + 1)

class RealParameter(NumericParameter):

    def __init__(self, low: float, high: Optional[float] = None, *, default: float, space: Optional[str] = None, optimize: bool = True, load: bool = True, **kwargs: Any) -> None:
        super().__init__(low=low, high=high, default=default, space=space, optimize=optimize, load=load, **kwargs)

    def get_space(self, name: str) -> Real:
        return Real(low=self.low, high=self.high, name=name, **self._space_params)

class DecimalParameter(NumericParameter):

    def __init__(self, low: float, high: Optional[float] = None, *, default: float, decimals: int = 3, space: Optional[str] = None, optimize: bool = True, load: bool = True, **kwargs: Any) -> None:
        self._decimals: int = decimals
        default = round(default, self._decimals)
        super().__init__(low=low, high=high, default=default, space=space, optimize=optimize, load=load, **kwargs)

    def get_space(self, name: str) -> SKDecimal:
        return SKDecimal(low=self.low, high=self.high, decimals=self._decimals, name=name, **self._space_params)

    @property
    def range(self) -> List[float]:
        if self.can_optimize():
            low = int(self.low * pow(10, self._decimals))
            high = int(self.high * pow(10, self._decimals)) + 1
            return [round(n * pow(0.1, self._decimals), self._decimals) for n in range(low, high)]
        else:
            return [self.value]

class CategoricalParameter(BaseParameter):

    def __init__(self, categories: List[Any], *, default: Optional[Any] = None, space: Optional[str] = None, optimize: bool = True, load: bool = True, **kwargs: Any) -> None:
        if len(categories) < 2:
            raise OperationalException('CategoricalParameter space must be [a, b, ...] (at least two parameters)')
        self.opt_range: List[Any] = categories
        super().__init__(default=default, space=space, optimize=optimize, load=load, **kwargs)

    def get_space(self, name: str) -> Categorical:
        return Categorical(self.opt_range, name=name, **self._space_params)

    @property
    def range(self) -> List[Any]:
        if self.can_optimize():
            return self.opt_range
        else:
            return [self.value]

class BooleanParameter(CategoricalParameter):

    def __init__(self, *, default: Optional[bool] = None, space: Optional[str] = None, optimize: bool = True, load: bool = True, **kwargs: Any) -> None:
        categories = [True, False]
        super().__init__(categories=categories, default=default, space=space, optimize=optimize, load=load, **kwargs)
