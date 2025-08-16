import typing as tp
import numpy as np
import nevergrad as ng
from nevergrad.ops import mutations
from . import photonics
from .. import base

# Type alias for function taking a numpy array and returning a float
FunctionType = tp.Callable[[np.ndarray], float]
InstrumentParameter = ng.p.Parameter

ceviche: tp.Any = photonics.ceviche


def _make_parametrization(
    name: str,
    dimension: int,
    bounding_method: str = "bouncing",
    rolling: bool = False,
    as_tuple: bool = False,
) -> InstrumentParameter:
    """Creates appropriate parametrization for a Photonics problem

    Parameters
    ----------
    name: str
        problem name, among bragg, chirped, cf_photosic_realistic, cf_photosic_reference and morpho
    dimension: int
        size of the problem among 16, 40 and 60 (morpho) or 80 (bragg and chirped)
    bounding_method: str
        transform type for the bounding ("arctan", "tanh", "bouncing" or "clipping", see `Array.bounded`)
    rolling: bool
        whether rolling mutation should be applied.
    as_tuple: bool
        whether we should use a Tuple of Array instead of a 2D-array.

    Returns
    -------
    InstrumentParameter
        the parametrization for the problem
    """
    if name == "bragg":
        shape: tp.Tuple[int, int] = (2, dimension // 2)
        bounds: tp.List[tp.Tuple[int, int]] = [(2, 3), (30, 180)]
    elif name == "cf_photosic_realistic":
        shape = (2, dimension // 2)
        bounds = [(1, 9), (30, 180)]
    elif name == "cf_photosic_reference":
        shape = (1, dimension)
        bounds = [(30, 180)]
    elif name == "chirped":
        shape = (1, dimension)
        bounds = [(30, 180)]
    elif name == "morpho":
        shape = (4, dimension // 4)
        bounds = [(0, 300), (0, 600), (30, 600), (0, 300)]
    else:
        raise NotImplementedError(f"Transform for {name} is not implemented")
    divisor: int = max(2, len(bounds))
    assert not dimension % divisor, f"points length should be a multiple of {divisor}, got {dimension}"
    assert (
        shape[0] * shape[1] == dimension
    ), f"Cannot work with dimension {dimension} for {name}: not divisible by {shape[0]}."
    b_array: np.ndarray = np.array(bounds)
    assert b_array.shape[0] == shape[0]  # pylint: disable=unsubscriptable-object
    ones: np.ndarray = np.ones((1, int(shape[1])))
    init: np.ndarray = np.sum(b_array, axis=1, keepdims=True).dot(ones) / 2  # type: ignore
    if as_tuple:
        instrum: InstrumentParameter = ng.p.Instrumentation(
            *[
                ng.p.Array(init=init[:, i]).set_bounds(
                    b_array[:, 0], b_array[:, 1], method=bounding_method, full_range_sampling=True
                )
                for i in range(init.shape[1])  # type: ignore
            ]
        ).set_name("as_tuple")
        assert instrum.dimension == dimension, instrum
        return instrum
    array: InstrumentParameter = ng.p.Array(init=init)  # type: ignore
    if bounding_method not in ("arctan", "tanh"):
        # sigma must be adapted for clipping and constraint methods
        sigma_init: tp.Union[tp.List[tp.List[float]], tp.List[tp.List[float]]] = [[10.0]] if name != "bragg" else [[0.03], [10.0]]
        sigma: InstrumentParameter = ng.p.Array(init=sigma_init).set_mutation(exponent=2.0)  # type: ignore
        array.set_mutation(sigma=sigma)
    if rolling:
        mutation = mutations.MutationChoice([mutations.Cauchy(), mutations.Translation(axis=1)])
        array = mutation(array)
    array.set_bounds(b_array[:, [0]], b_array[:, [1]], method=bounding_method, full_range_sampling=True)
    array = ng.ops.mutations.Crossover(axis=1)(array).set_name("")
    assert array.dimension == dimension, f"Unexpected {array} for dimension {dimension}"
    return array


class Photonics(base.ExperimentFunction):
    """Function calling photonics code

    Parameters
    ----------
    name: str
        problem name, among bragg, chirped, cf_photosic_realistic, cf_photosic_reference and morpho
    dimension: int
        size of the problem among 16, 40 and 60 (morpho) or 80 (bragg and chirped)
    bounding_method: str
        transform type for the bounding ("arctan", "tanh", "bouncing" or "clipping", see `Array.bounded`)
    rolling: bool
        whether rolling mutation should be applied.
    as_tuple: bool
        whether we use a Tuple of Array instead of a 2D-array.
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        bounding_method: str = "clipping",
        rolling: bool = False,
        as_tuple: bool = False,
    ) -> None:
        assert name in [
            "bragg",
            "morpho",
            "chirped",
            "cf_photosic_reference",
            "cf_photosic_realistic",
        ], f"Unknown {name}"
        self.name: str = name + ("_as_tuple" if as_tuple else "")
        self._as_tuple: bool = as_tuple
        self._base_func: FunctionType = getattr(photonics, name)
        param: InstrumentParameter = _make_parametrization(
            name=name,
            dimension=dimension,
            bounding_method=bounding_method,
            rolling=rolling,
            as_tuple=as_tuple,
        )
        super().__init__(self._compute, param)

    def to_array(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        data: np.ndarray = np.concatenate(args).T if self._as_tuple else args[0]
        assert data.size == self.dimension
        return np.asarray(data).ravel()

    def evaluation_function(self, *recommendations: ng.p.Parameter) -> float:
        assert len(recommendations) == 1, "Should not be a pareto set for a singleobjective function"
        recom: ng.p.Parameter = recommendations[0]
        x: np.ndarray = self.to_array(*recom.args, **recom.kwargs)
        loss: float = self.function(x)
        assert isinstance(loss, float)
        base.update_leaderboard(f"{self.name},{self.parametrization.dimension}", loss, x, verbose=True)
        return loss

    def _compute(self, *args: tp.Any, **kwargs: tp.Any) -> float:
        x: np.ndarray = self.to_array(*args, **kwargs)
        try:
            output: float = self._base_func(x)
        except Exception:
            output = float("inf")
        if np.isnan(output):
            output = float("inf")
        return output
