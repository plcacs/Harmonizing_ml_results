"""
IHyperStrategy interface, hyperoptable Parameter class.
This module defines a base class for auto-hyperoptable strategies.
"""
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, List, Optional, Union
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

    def __init__(
        self,
        *,
        default: Any,
        space: Optional[str] = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize hyperopt-optimizable parameter.
        :param space: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
         parameter field
         name is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to skopt.space.(Integer|Real|Categorical).
        """
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
        """
        Get-space - will be used by Hyperopt to get the hyperopt Space
        """
        ...

    def can_optimize(self) -> bool:
        return self.in_space and self.optimize and (HyperoptStateContainer.state != HyperoptState.OPTIMIZE)

class NumericParameter(BaseParameter):
    """Internal parameter used for Numeric purposes"""
    float_or_int = Union[int, float]

    def __init__(
        self,
        low: Union[float, int, Sequence[Union[float, int]]],
        high: Optional[Union[float, int]] = None,
        *,
        default: Union[float, int],
        space: Optional[str] = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize hyperopt-optimizable numeric parameter.
        Cannot be instantiated, but provides the validation for other numeric parameters
        :param low: Lower end (inclusive) of optimization space or [low, high].
        :param high: Upper end (inclusive) of optimization space.
                     Must be none of entire range is passed first parameter.
        :param default: A default value.
        :param space: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
                      parameter fieldname is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to skopt.space.*.
        """
        if high is not None and isinstance(low, Sequence):
            raise OperationalException(f'{self.__class__.__name__} space invalid.')
        if high is None or isinstance(low, Sequence):
            if not isinstance(low, Sequence) or len(low) != 2:
                raise OperationalException(f'{self.__class__.__name__} space must be [low, high]')
            self.low: Union[float, int] = low[0]
            self.high: Union[float, int] = low[1]
        else:
            self.low: Union[float, int] = low
            self.high: Union[float, int] = high
        super().__init__(default=default, space=space, optimize=optimize, load=load, **kwargs)

class IntParameter(NumericParameter):

    def __init__(
        self,
        low: Union[int, Sequence[int]],
        high: Optional[int] = None,
        *,
        default: int,
        space: Optional[str] = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize hyperopt-optimizable integer parameter.
        :param low: Lower end (inclusive) of optimization space or [low, high].
        :param high: Upper end (inclusive) of optimization space.
                     Must be none of entire range is passed first parameter.
        :param default: A default value.
        :param space: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
                      parameter fieldname is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to skopt.space.Integer.
        """
        super().__init__(low=low, high=high, default=default, space=space, optimize=optimize, load=load, **kwargs)

    def get_space(self, name: str) -> Integer:
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        """
        return Integer(low=self.low, high=self.high, name=name, **self._space_params)

    @property
    def range(self) -> List[int]:
        """
        Get each value in this space as list.
        Returns a List from low to high (inclusive) in Hyperopt mode.
        Returns a List with 1 item (`value`) in "non-hyperopt" mode, to avoid
        calculating 100ds of indicators.
        """
        if self.can_optimize():
            return list(range(int(self.low), int(self.high) + 1))
        else:
            return list(range(int(self.value), int(self.value) + 1))

class RealParameter(NumericParameter):

    def __init__(
        self,
        low: Union[float, int, Sequence[Union[float, int]]],
        high: Optional[Union[float, int]] = None,
        *,
        default: float,
        space: Optional[str] = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize hyperopt-optimizable floating point parameter with unlimited precision.
        :param low: Lower end (inclusive) of optimization space or [low, high].
        :param high: Upper end (inclusive) of optimization space.
                     Must be none if entire range is passed first parameter.
        :param default: A default value.
        :param space: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
                      parameter fieldname is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to skopt.space.Real.
        """
        super().__init__(low=low, high=high, default=default, space=space, optimize=optimize, load=load, **kwargs)

    def get_space(self, name: str) -> Real:
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        """
        return Real(low=self.low, high=self.high, name=name, **self._space_params)

class DecimalParameter(NumericParameter):

    def __init__(
        self,
        low: Union[float, int, Sequence[Union[float, int]]],
        high: Optional[Union[float, int]] = None,
        *,
        default: float,
        decimals: int = 3,
        space: Optional[str] = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize hyperopt-optimizable decimal parameter with a limited precision.
        :param low: Lower end (inclusive) of optimization space or [low, high].
        :param high: Upper end (inclusive) of optimization space.
                     Must be none if entire range is passed first parameter.
        :param default: A default value.
        :param decimals: A number of decimals after floating point to be included in testing.
        :param space: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
                      parameter fieldname is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to skopt.space.Integer.
        """
        self._decimals: int = decimals
        default = round(default, self._decimals)
        super().__init__(low=low, high=high, default=default, space=space, optimize=optimize, load=load, **kwargs)

    def get_space(self, name: str) -> SKDecimal:
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        """
        return SKDecimal(low=self.low, high=self.high, decimals=self._decimals, name=name, **self._space_params)

    @property
    def range(self) -> List[float]:
        """
        Get each value in this space as list.
        Returns a List from low to high (inclusive) in Hyperopt mode.
        Returns a List with 1 item (`value`) in "non-hyperopt" mode, to avoid
        calculating 100ds of indicators.
        """
        if self.can_optimize():
            low = int(self.low * pow(10, self._decimals))
            high = int(self.high * pow(10, self._decimals)) + 1
            return [round(n * pow(0.1, self._decimals), self._decimals) for n in range(low, high)]
        else:
            return [self.value]

class CategoricalParameter(BaseParameter):

    def __init__(
        self,
        categories: List[Any],
        *,
        default: Optional[Any] = None,
        space: Optional[str] = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize hyperopt-optimizable parameter.
        :param categories: Optimization space, [a, b, ...].
        :param default: A default value. If not specified, first item from specified space will be
         used.
        :param space: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
         parameter field
         name is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to skopt.space.Categorical.
        """
        if len(categories) < 2:
            raise OperationalException('CategoricalParameter space must be [a, b, ...] (at least two parameters)')
        self.opt_range: List[Any] = categories
        super().__init__(default=default, space=space, optimize=optimize, load=load, **kwargs)

    def get_space(self, name: str) -> Categorical:
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        """
        return Categorical(self.opt_range, name=name, **self._space_params)

    @property
    def range(self) -> List[Any]:
        """
        Get each value in this space as list.
        Returns a List of categories in Hyperopt mode.
        Returns a List with 1 item (`value`) in "non-hyperopt" mode, to avoid
        calculating 100ds of indicators.
        """
        if self.can_optimize():
            return self.opt_range
        else:
            return [self.value]

class BooleanParameter(CategoricalParameter):

    def __init__(
        self,
        *,
        default: Optional[bool] = None,
        space: Optional[str] = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize hyperopt-optimizable Boolean Parameter.
        It's a shortcut to `CategoricalParameter([True, False])`.
        :param default: A default value. If not specified, first item from specified space will be
         used.
        :param space: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
         parameter field
         name is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to skopt.space.Categorical.
        """
        categories = [True, False]
        super().__init__(categories=categories, default=default, space=space, optimize=optimize, load=load, **kwargs)
