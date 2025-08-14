#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from . import metamodel
from . import base
from . import oneshot


class Crossover:
    def __init__(
        self,
        random_state: np.random.RandomState,
        crossover: tp.Union[str, float],
        parameter: tp.Optional[p.Parameter] = None,
    ) -> None:
        self.CR: float = 0.5
        self.crossover: tp.Union[str, float] = crossover
        self.random_state: np.random.RandomState = random_state
        if isinstance(crossover, float):
            self.CR = crossover
        elif crossover == "random":
            self.CR = self.random_state.uniform(0.0, 1.0)
        elif crossover not in ["twopoints", "onepoint", "rotated_twopoints", "voronoi"]:
            raise ValueError(f'Unknown crossover "{crossover}"')
        self.shape: tp.Optional[np.ndarray] = (
            np.array(parameter.value).shape if (crossover == "voronoi" and parameter is not None) else None
        )

    def apply(self, donor: np.ndarray, individual: np.ndarray) -> None:
        dim: int = donor.size
        if self.crossover == "twopoints" and dim >= 4:
            self.twopoints(donor, individual)
        elif self.crossover == "rotated_twopoints" and dim >= 4:
            self.rotated_twopoints(donor, individual)
        elif self.crossover == "onepoint" and dim >= 3:
            self.onepoint(donor, individual)
        elif self.crossover == "voronoi":
            self.voronoi(donor, individual)
        else:
            self.variablewise(donor, individual)

    def variablewise(self, donor: np.ndarray, individual: np.ndarray) -> None:
        R: int = self.random_state.randint(donor.size)
        transfer: np.ndarray = np.array(
            [idx != R and self.random_state.uniform(0, 1) > self.CR for idx in range(donor.size)]
        )
        donor[transfer] = individual[transfer]

    def onepoint(self, donor: np.ndarray, individual: np.ndarray) -> None:
        R: int = self.random_state.randint(1, donor.size)
        if self.random_state.choice([True, False]):
            donor[R:] = individual[R:]
        else:
            donor[:R] = individual[:R]

    def twopoints(self, donor: np.ndarray, individual: np.ndarray) -> None:
        bounds: list[int] = sorted(self.random_state.choice(donor.size + 1, size=2, replace=False).tolist())
        if bounds[1] == donor.size and not bounds[0]:
            bounds[self.random_state.randint(2)] = self.random_state.randint(1, donor.size)
        if self.random_state.choice([True, False]):
            donor[bounds[0] : bounds[1]] = individual[bounds[0] : bounds[1]]
        else:
            donor[: bounds[0]] = individual[: bounds[0]]
            donor[bounds[1] :] = individual[bounds[1] :]

    def rotated_twopoints(self, donor: np.ndarray, individual: np.ndarray) -> None:
        bounds: list[int] = sorted(self.random_state.choice(donor.size + 1, size=2, replace=False).tolist())
        if bounds[1] == donor.size and not bounds[0]:
            bounds[self.random_state.randint(2)] = self.random_state.randint(1, donor.size)
        bounds2: list[int] = [self.random_state.choice(donor.size + 1 - bounds[1] + bounds[0])]
        bounds2.append(bounds2[0] + bounds[1] - bounds[0])
        assert bounds[1] < donor.size + 1
        donor[bounds[0] : bounds[1]] = individual[bounds2[0] : bounds2[1]]

    def voronoi(self, donor: np.ndarray, individual: np.ndarray) -> None:
        shape: tp.Optional[np.ndarray] = self.shape
        if shape is None or len(shape) < 2:
            warnings.warn("Voronoi DE needs a shape.")
            self.twopoints(donor, individual)
            return
        local_donor: np.ndarray = donor.reshape(shape)
        local_individual: np.ndarray = individual.reshape(shape)
        x1: np.ndarray = np.array([np.random.randint(shape[i]) for i in range(len(shape))])
        x2: np.ndarray = np.array([np.random.randint(shape[i]) for i in range(len(shape))])
        x3: np.ndarray = np.array([np.random.randint(shape[i]) for i in range(len(shape))])
        x4: np.ndarray = np.array([np.random.randint(shape[i]) for i in range(len(shape))])
        it = np.nditer(local_donor, flags=["multi_index"])
        for _ in it:
            current_index: tuple[int, ...] = it.multi_index
            d1: float = np.linalg.norm(np.array(current_index) - x1)
            d2: float = np.linalg.norm(np.array(current_index) - x2)
            d3: float = np.linalg.norm(np.array(current_index) - x3)
            d4: float = np.linalg.norm(np.array(current_index) - x4)
            if min([d1, d2, d3]) > d4:
                local_donor[current_index] = local_individual[current_index]
        donor[:] = local_donor.flatten()[:]
        individual[:] = local_individual.flatten()[:]


class _DE(base.Optimizer):
    """Differential evolution.

    Default pop size equal to 30
    We return the mean of the individuals with fitness better than median, which might be stupid sometimes.
    CR =.5, F1=.8, F2=.8, curr-to-best.
    Initial population: pure random.
    """

    def __init__(
        self,
        parametrization: base.IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        config: tp.Optional["DifferentialEvolution"] = None,
        weights: tp.Any = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.objective_weights: tp.Any = weights
        self._config: DifferentialEvolution = DifferentialEvolution() if config is None else config
        self.scale: float = (
            float(1.0 / np.sqrt(self.dimension))
            if isinstance(self._config.scale, str)
            else self._config.scale
        )
        pop_choice: dict[str, tp.Any] = {
            "standard": 0,
            "small": 1 + int(np.sqrt(np.log(self.dimension + 3))),
            "dimension": self.dimension + 1,
            "large": 7 * self.dimension,
        }
        if isinstance(self._config.popsize, int):
            self.llambda: int = self._config.popsize
        else:
            self.llambda = max(30, self.num_workers, pop_choice[self._config.popsize])
        self._MULTIOBJECTIVE_AUTO_BOUND: int = max(getattr(self, "_MULTIOBJECTIVE_AUTO_BOUND", 0), self.llambda)  # type: ignore
        self._penalize_cheap_violations: bool = True
        self._uid_queue: base.utils.UidQueue = base.utils.UidQueue()
        self.population: dict[str, p.Parameter] = {}
        self.sampler: tp.Optional[base.Optimizer] = None
        self._no_hypervolume: bool = self._config.multiobjective_adaptation

    def set_objective_weights(self, weights: tp.Any) -> None:
        self.objective_weights = weights

    def recommend(self) -> p.Parameter:
        sample_size: int = int((self.dimension * (self.dimension - 1)) / 2 + 2 * self.dimension + 1)
        if self._config.high_speed and len(self.archive) >= sample_size:
            try:
                meta_data: tp.Any = metamodel.learn_on_k_best(self.archive, sample_size)
                return self.parametrization.spawn_child().set_standardized_data(meta_data)
            except metamodel.MetaModelFailure:
                pass
        if self._config.recommendation != "noisy":
            return self.current_bests[self._config.recommendation].parameter
        med_fitness: float = np.median([p_elem.loss for p_elem in self.population.values() if p_elem.loss is not None])
        good_guys: list[p.Parameter] = [p_elem for p_elem in self.population.values() if p_elem.loss is not None and p_elem.loss < med_fitness]
        if not good_guys:
            return self.current_bests["pessimistic"].parameter
        data: tp.Any = sum(
            [g.get_standardized_data(reference=self.parametrization) for g in good_guys]
        ) / len(good_guys)
        out: p.Parameter = self.parametrization.spawn_child()
        with p.helpers.deterministic_sampling(out):
            out.set_standardized_data(data)
        return out

    def _internal_ask_candidate(self) -> p.Parameter:
        if len(self.population) < self.llambda:  # initialization phase
            init: str = self._config.initialization
            if self.sampler is None and init == "QO":
                self.sampler = oneshot.SamplingSearch(
                    sampler="Hammersley", scrambled=True, opposition_mode="quasi"
                )(self.parametrization, budget=self.llambda)
            if self.sampler is None and init == "SO":
                self.sampler = oneshot.SamplingSearch(
                    sampler="Hammersley", scrambled=True, opposition_mode="special"
                )(self.parametrization, budget=self.llambda)
            if self.sampler is None and init not in ["gaussian", "parametrization"]:
                assert init in ["LHS", "QR"]
                self.sampler = oneshot.SamplingSearch(
                    sampler=init if init == "LHS" else "Hammersley", scrambled=init == "QR", scale=self.scale
                )(self.parametrization, budget=self.llambda)
            if init == "parametrization":
                candidate: p.Parameter = self.parametrization.sample()
            elif self.sampler is not None:
                candidate = self.sampler.ask()
            elif self._config.crossover == "voronoi":
                new_guy: np.ndarray = (
                    self.scale * self._rng.normal(0, 1, self.dimension)
                    if len(self.population) > self.llambda / 6
                    else self.scale * self._rng.normal() * np.ones(self.dimension)
                )
                candidate = self.parametrization.spawn_child().set_standardized_data(new_guy)
            else:
                new_guy = self.scale * self._rng.normal(0, 1, self.dimension)
                candidate = self.parametrization.spawn_child().set_standardized_data(new_guy)
            candidate.heritage["lineage"] = candidate.uid
            self.population[candidate.uid] = candidate
            self._uid_queue.asked.add(candidate.uid)
            return candidate

        lineage: str = self._uid_queue.ask()
        parent: p.Parameter = self.population[lineage]
        candidate: p.Parameter = parent.spawn_child()
        candidate.heritage["lineage"] = lineage
        data: np.ndarray = candidate.get_standardized_data(reference=self.parametrization)
        uids: list[str] = list(self.population)
        a: p.Parameter
        b: p.Parameter
        a, b = (self.population[uids[self._rng.randint(self.llambda)]] for _ in range(2))
        best: p.Parameter = self.current_bests["pessimistic"].parameter
        if self._config.multiobjective_adaptation and self.num_objectives > 1:
            pareto: list[p.Parameter] = self.pareto_front()
            if pareto:
                best = parent if parent in pareto else pareto[self._rng.choice(len(pareto))]
            if len(pareto) > 2:
                idxs: np.ndarray = self._rng.choice(len(pareto), size=2, replace=False)
                a, b = (pareto[int(idxs[i])] for i in range(2))
        data_a: np.ndarray
        data_b: np.ndarray
        data_best: np.ndarray
        data_a, data_b, data_best = (
            indiv.get_standardized_data(reference=self.parametrization) for indiv in (a, b, best)
        )
        donor: np.ndarray = data + self._config.F1 * (data_a - data_b) + self._config.F2 * (data_best - data)
        candidate.parents_uids.extend([a.uid, b.uid])
        co: tp.Union[str, float] = self._config.crossover
        if co == "parametrization":
            candidate.recombine(self.parametrization.spawn_child().set_standardized_data(donor))
        else:
            crossovers: Crossover = Crossover(
                self._rng, 1.0 / self.dimension if co == "dimension" else co, self.parametrization
            )
            crossovers.apply(donor, data)
            candidate.set_standardized_data(donor, reference=self.parametrization)
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        uid: str = candidate.heritage["lineage"]
        if uid not in self.population:
            self._internal_tell_not_asked(candidate, loss)
            return
        self._uid_queue.tell(uid)
        parent: p.Parameter = self.population[uid]
        mo_adapt: bool = self._config.multiobjective_adaptation and self.num_objectives > 1
        if mo_adapt:
            if self.objective_weights is None:
                self.objective_weights = np.ones(self.num_objectives)
            else:
                assert len(self.objective_weights) == self.num_objectives
        mo_adapt &= candidate._losses is not None
        if not mo_adapt and loss <= base._loss(parent):
            self.population[uid] = candidate
        elif mo_adapt and (
            parent._losses is None
            or np.average(candidate.losses < parent.losses, weights=self.objective_weights) > self._rng.rand()
        ):
            self.population[uid] = candidate
        elif self._config.propagate_heritage and loss <= float("inf"):
            self.population[uid].heritage.update(candidate.heritage)

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        discardable: tp.Optional[str] = None
        if len(self.population) >= self.llambda:
            if self.num_objectives == 1:
                uid_worst, worst = max(self.population.items(), key=lambda p_item: base._loss(p_item[1]))
                if loss < base._loss(worst):
                    discardable = uid_worst
            else:
                pareto_uids: set[str] = {c.uid for c in self.pareto_front()}
                if candidate.uid in pareto_uids:
                    non_pareto_pop: set[str] = {c.uid for c in self.population.values()} - pareto_uids
                    if non_pareto_pop:
                        nonpareto: p.Parameter = {c.uid: c for c in self.population.values()}[list(non_pareto_pop)[0]]
                        discardable = nonpareto.heritage["lineage"]
        if discardable is not None:
            del self.population[discardable]
            self._uid_queue.discard(discardable)
        if len(self.population) < self.llambda:
            self.population[candidate.uid] = candidate
            self._uid_queue.tell(candidate.uid)


class DifferentialEvolution(base.ConfiguredOptimizer):
    """Differential evolution is typically used for continuous optimization.
    It uses differences between points in the population for doing mutations in fruitful directions;
    it is therefore a kind of covariance adaptation without any explicit covariance,
    making it super fast in high dimension. This class implements several variants of differential
    evolution, some of them adapted to genetic mutations as in
    `Holland’s work <https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Two-point_and_k-point_crossover>`_),
    (this combination is termed :code:`TwoPointsDE` in Nevergrad, corresponding to :code:`crossover="twopoints"`),
    or to the noisy setting (coined :code:`NoisyDE`, corresponding to :code:`recommendation="noisy"`).
    In that last case, the optimizer returns the mean of the individuals with fitness better than median,
    which might be stupid sometimes though.

    Default settings are CR =.5, F1=.8, F2=.8, curr-to-best, pop size is 30
    Initial population: pure random.
    """

    def __init__(
        self,
        *,
        initialization: str = "parametrization",
        scale: tp.Union[str, float] = 1.0,
        recommendation: str = "optimistic",
        crossover: tp.Union[str, float] = 0.5,
        F1: float = 0.8,
        F2: float = 0.8,
        popsize: tp.Union[str, int] = "standard",
        propagate_heritage: bool = False,
        multiobjective_adaptation: bool = True,
        high_speed: bool = False,
    ) -> None:
        super().__init__(_DE, locals(), as_config=True)
        assert recommendation in ["optimistic", "pessimistic", "noisy", "mean"]
        assert initialization in ["gaussian", "LHS", "QO", "SO", "QR", "parametrization"]
        assert isinstance(scale, float) or scale == "mini"
        if not isinstance(popsize, int):
            assert popsize in ["large", "dimension", "standard", "small"]
        assert isinstance(crossover, float) or crossover in [
            "onepoint",
            "twopoints",
            "rotated_twopoints",
            "dimension",
            "random",
            "parametrization",
            "voronoi",
        ]
        self.initialization: str = initialization
        self.scale: tp.Union[str, float] = scale
        self.high_speed: bool = high_speed
        self.recommendation: str = recommendation
        self.propagate_heritage: bool = propagate_heritage
        self.F1: float = F1
        self.F2: float = F2
        self.crossover: tp.Union[str, float] = crossover
        self.popsize: tp.Union[str, int] = popsize
        self.multiobjective_adaptation: bool = multiobjective_adaptation


DE: DifferentialEvolution = DifferentialEvolution().set_name("DE", register=True)
LPSDE: DifferentialEvolution = DifferentialEvolution(popsize="large").set_name("LPSDE", register=True)
TwoPointsDE: DifferentialEvolution = DifferentialEvolution(crossover="twopoints").set_name("TwoPointsDE", register=True)
VoronoiDE: DifferentialEvolution = DifferentialEvolution(crossover="voronoi").set_name("VoronoiDE", register=True)
RotatedTwoPointsDE: DifferentialEvolution = DifferentialEvolution(crossover="rotated_twopoints").set_name(
    "RotatedTwoPointsDE", register=True
)

LhsDE: DifferentialEvolution = DifferentialEvolution(initialization="LHS").set_name("LhsDE", register=True)
QrDE: DifferentialEvolution = DifferentialEvolution(initialization="QR").set_name("QrDE", register=True)
QODE: DifferentialEvolution = DifferentialEvolution(initialization="QO").set_name("QODE", register=True)
SPQODE: DifferentialEvolution = DifferentialEvolution(initialization="QO", popsize="small").set_name("SPQODE", register=True)
QOTPDE: DifferentialEvolution = DifferentialEvolution(initialization="QO", crossover="twopoints").set_name("QOTPDE", register=True)
LQOTPDE: DifferentialEvolution = DifferentialEvolution(initialization="QO", scale=10.0, crossover="twopoints").set_name(
    "LQOTPDE", register=True
)
LQODE: DifferentialEvolution = DifferentialEvolution(initialization="QO", scale=10.0).set_name("LQODE", register=True)
SODE: DifferentialEvolution = DifferentialEvolution(initialization="SO").set_name("SODE", register=True)
NoisyDE: DifferentialEvolution = DifferentialEvolution(recommendation="noisy").set_name("NoisyDE", register=True)
AlmostRotationInvariantDE: DifferentialEvolution = DifferentialEvolution(crossover=0.9).set_name(
    "AlmostRotationInvariantDE", register=True
)
RotationInvariantDE: DifferentialEvolution = DifferentialEvolution(crossover=1.0, popsize="dimension").set_name(
    "RotationInvariantDE", register=True
)

DiscreteDE: DifferentialEvolution = DifferentialEvolution(crossover="dimension").set_name("DiscreteDE", register=True)