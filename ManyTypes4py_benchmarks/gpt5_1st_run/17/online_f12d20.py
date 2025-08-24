from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Tuple, Optional, Any
import numpy as np
from numpy.typing import NDArray
from pandas.compat._optional import import_optional_dependency

EWMFuncType = Callable[
    [
        NDArray[np.floating],
        NDArray[np.floating],
        int,
        float,
        float,
        NDArray[np.floating],
        bool,
        bool,
    ],
    Tuple[NDArray[np.floating], NDArray[np.floating]],
]


def generate_online_numba_ewma_func(nopython: bool, nogil: bool, parallel: bool) -> EWMFuncType:
    """
    Generate a numba jitted groupby ewma function specified by values
    from engine_kwargs.

    Parameters
    ----------
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit

    Returns
    -------
    Numba function
    """
    if TYPE_CHECKING:
        import numba  # type: ignore
    else:
        numba: Any = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def online_ewma(
        values: NDArray[np.floating],
        deltas: NDArray[np.floating],
        minimum_periods: int,
        old_wt_factor: float,
        new_wt: float,
        old_wt: NDArray[np.floating],
        adjust: bool,
        ignore_na: bool,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute online exponentially weighted mean per column over 2D values.

        Takes the first observation as is, then computes the subsequent
        exponentially weighted mean accounting minimum periods.
        """
        result: NDArray[np.floating] = np.empty(values.shape)
        weighted_avg: NDArray[np.floating] = values[0].copy()
        nobs: NDArray[np.int64] = (~np.isnan(weighted_avg)).astype(np.int64)
        result[0] = np.where(nobs >= minimum_periods, weighted_avg, np.nan)
        for i in range(1, len(values)):
            cur: NDArray[np.floating] = values[i]
            is_observations: NDArray[np.bool_] = ~np.isnan(cur)
            nobs += is_observations.astype(np.int64)
            for j in numba.prange(len(cur)):
                if not np.isnan(weighted_avg[j]):
                    if is_observations[j] or not ignore_na:
                        old_wt[j] *= old_wt_factor ** deltas[j - 1]
                        if is_observations[j]:
                            if weighted_avg[j] != cur[j]:
                                weighted_avg[j] = (old_wt[j] * weighted_avg[j] + new_wt * cur[j]) / (old_wt[j] + new_wt)
                            if adjust:
                                old_wt[j] += new_wt
                            else:
                                old_wt[j] = 1.0
                elif is_observations[j]:
                    weighted_avg[j] = cur[j]
            result[i] = np.where(nobs >= minimum_periods, weighted_avg, np.nan)
        return (result, old_wt)
    return online_ewma


class EWMMeanState:
    shape: tuple[int, ...]
    adjust: bool
    ignore_na: bool
    new_wt: float
    old_wt_factor: float
    old_wt: NDArray[np.floating]
    last_ewm: Optional[NDArray[np.floating]]

    def __init__(self, com: float, adjust: bool, ignore_na: bool, shape: tuple[int, ...]) -> None:
        alpha: float = 1.0 / (1.0 + com)
        self.shape = shape
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.new_wt = 1.0 if adjust else alpha
        self.old_wt_factor = 1.0 - alpha
        self.old_wt = np.ones(self.shape[-1])
        self.last_ewm = None

    def run_ewm(
        self,
        weighted_avg: NDArray[np.floating],
        deltas: NDArray[np.floating],
        min_periods: int,
        ewm_func: EWMFuncType,
    ) -> NDArray[np.floating]:
        result, old_wt = ewm_func(
            weighted_avg,
            deltas,
            min_periods,
            self.old_wt_factor,
            self.new_wt,
            self.old_wt,
            self.adjust,
            self.ignore_na,
        )
        self.old_wt = old_wt
        self.last_ewm = result[-1]
        return result

    def reset(self) -> None:
        self.old_wt = np.ones(self.shape[-1])
        self.last_ewm = None