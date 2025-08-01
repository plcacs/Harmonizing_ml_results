from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Tuple
import numpy as np
from pandas.compat._optional import import_optional_dependency

def generate_online_numba_ewma_func(nopython: bool, nogil: bool, parallel: bool) -> Callable[
    [np.ndarray, np.ndarray, int, float, float, np.ndarray, bool, bool],
    Tuple[np.ndarray, np.ndarray]
]:
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
        numba = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def online_ewma(
        values: np.ndarray,
        deltas: np.ndarray,
        minimum_periods: int,
        old_wt_factor: float,
        new_wt: float,
        old_wt: np.ndarray,
        adjust: bool,
        ignore_na: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute online exponentially weighted mean per column over 2D values.

        Takes the first observation as is, then computes the subsequent
        exponentially weighted mean accounting minimum periods.
        """
        result = np.empty(values.shape)
        weighted_avg = values[0].copy()
        nobs = (~np.isnan(weighted_avg)).astype(np.int64)
        result[0] = np.where(nobs >= minimum_periods, weighted_avg, np.nan)
        for i in range(1, len(values)):
            cur = values[i]
            is_observations = ~np.isnan(cur)
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
        return result, old_wt

    return online_ewma

class EWMMeanState:
    def __init__(self, com: float, adjust: bool, ignore_na: bool, shape: Tuple[int, ...]) -> None:
        alpha = 1.0 / (1.0 + com)
        self.shape: Tuple[int, ...] = shape
        self.adjust: bool = adjust
        self.ignore_na: bool = ignore_na
        self.new_wt: float = 1.0 if adjust else alpha
        self.old_wt_factor: float = 1.0 - alpha
        self.old_wt: np.ndarray = np.ones(self.shape[-1])
        self.last_ewm: np.ndarray | None = None

    def run_ewm(
        self,
        weighted_avg: np.ndarray,
        deltas: np.ndarray,
        min_periods: int,
        ewm_func: Callable[
            [np.ndarray, np.ndarray, int, float, float, np.ndarray, bool, bool],
            Tuple[np.ndarray, np.ndarray]
        ]
    ) -> np.ndarray:
        result, old_wt = ewm_func(weighted_avg, deltas, min_periods, self.old_wt_factor, self.new_wt, self.old_wt, self.adjust, self.ignore_na)
        self.old_wt = old_wt
        self.last_ewm = result[-1]
        return result

    def reset(self) -> None:
        self.old_wt = np.ones(self.shape[-1])
        self.last_ewm = None