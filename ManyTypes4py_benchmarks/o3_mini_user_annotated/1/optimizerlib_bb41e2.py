import math
import warnings
import numpy as np
import itertools
import collections
import typing as tp

from nevergrad import parametrization as p  # type: ignore
from nevergrad import base  # type: ignore
from nevergrad import oneshot  # type: ignore
from nevergrad import experimentalvariants  # type: ignore

# The following assignments do not require type annotations as they are constant assignments.
MixDeterministicRL = base.ConfiguredOptimizer(optimizers=[base.DiagonalCMA, base.PSO, base.GeneticDE])  # type: ignore
SpecialRL = base.Chaining([base.MixDeterministicRL, base.TBPSA], ["half"]).set_name("SpecialRL", register=True)
NoisyRL1 = base.Chaining([base.MixDeterministicRL, base.NoisyOnePlusOne], ["half"]).set_name("NoisyRL1", register=True)
NoisyRL2 = base.Chaining(
    [base.MixDeterministicRL, base.RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne], ["half"]
).set_name("NoisyRL2", register=True)
NoisyRL3 = base.Chaining([base.MixDeterministicRL, base.OptimisticNoisyOnePlusOne], ["half"]).set_name(
    "NoisyRL3", register=True
)

# --- FCarola6, Carola*, DS*, etc.
FCarola6 = base.Chaining([base.NGOpt, base.NGOpt, base.RBFGS], ["tenth", "most"]).set_name("FCarola6", register=True)
FCarola6.no_parallelization = True
Carola11 = base.Chaining([base.MultiCMA, base.RBFGS], ["most"]).set_name("Carola11", register=True)
Carola11.no_parallelization = True
Carola14 = base.Chaining([base.MultiCMA, base.RBFGS], ["most"]).set_name("Carola14", register=True)
Carola14.no_parallelization = True
DS14 = base.Chaining([base.MultiDS, base.RBFGS], ["most"]).set_name("DS14", register=True)
DS14.no_parallelization = True
Carola13 = base.Chaining([base.CmaFmin2, base.RBFGS], ["most"]).set_name("Carola13", register=True)
Carola13.no_parallelization = True
Carola15 = base.Chaining([base.Cobyla, base.MetaModel, base.RBFGS], ["sqrt", "most"]).set_name("Carola15", register=True)
Carola15.no_parallelization = True

# --- cGA optimizer
class cGA(base.Optimizer):
    """Compact Genetic Algorithm.
    A discrete optimization algorithm.
    """
    def __init__(
        self,
        parametrization: p.IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        arity: tp.Optional[int] = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if arity is None:
            all_params = p.helpers.flatten(self.parametrization)
            arity = max(
                len(param.choices) if isinstance(param, p.TransitionChoice) else 500
                for _, param in all_params
            )
        self._arity: int = arity  # type: ignore
        self._penalize_cheap_violations: bool = False
        # p[i][j] is the probability that the i-th variable has value 0<=j< arity.
        self.p: np.ndarray = np.ones((self.dimension, arity)) / arity
        self.llambda: int = max(num_workers, 40)
        self._previous_value_candidate: tp.Optional[tp.Tuple[float, p.Parameter]] = None

    def _internal_ask_candidate(self) -> p.Parameter:
        # Multinomial sampling
        values: tp.List[int] = [
            sum(self._rng.uniform() > cum_proba) 
            for cum_proba in np.cumsum(self.p, axis=1)
        ]
        data = p.discretization.noisy_inverse_threshold_discretization(values, arity=self._arity, gen=self._rng)  # type: ignore
        return self.parametrization.spawn_child().set_standardized_data(data)

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        data = candidate.get_standardized_data(reference=self.parametrization)
        if self._previous_value_candidate is None:
            self._previous_value_candidate = (loss, candidate)
        else:
            winner_candidate, loser_candidate = self._previous_value_candidate[1], candidate
            if self._previous_value_candidate[0] > loss:
                winner_candidate, loser_candidate = candidate, self._previous_value_candidate[1]
            winner_data = p.discretization.threshold_discretation(np.asarray(winner_candidate.data), arity=self._arity)  # type: ignore
            loser_data = p.discretization.threshold_discretation(np.asarray(loser_candidate.data), arity=self._arity)  # type: ignore
            for i in range(len(winner_data)):
                if winner_data[i] != loser_data[i]:
                    self.p[i][winner_data[i]] += 1.0 / self.llambda
                    self.p[i][loser_data[i]] -= 1.0 / self.llambda
                    for j in range(len(self.p[i])):
                        self.p[i][j] = max(self.p[i][j], 1.0 / self.llambda)
                    self.p[i] /= sum(self.p[i])
            self._previous_value_candidate = None

# --- _EMNA optimizer
class _EMNA(base.Optimizer):
    """Estimation of Multivariate Normal Algorithm (EMNA)."""
    def __init__(
        self,
        parametrization: p.IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        isotropic: bool = True,
        naive: bool = True,
        population_size_adaptation: bool = False,
        initial_popsize: tp.Optional[int] = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.isotropic: bool = isotropic
        self.naive: bool = naive
        self.population_size_adaptation: bool = population_size_adaptation
        self.min_coef_parallel_context: int = 8
        if initial_popsize is None:
            initial_popsize = self.dimension
        if self.isotropic:
            self.sigma: tp.Union[float, np.ndarray] = 1.0
        else:
            self.sigma = np.ones(self.dimension)
        self.popsize = base._PopulationSizeController(
            llambda=4 * initial_popsize, mu=initial_popsize, dimension=self.dimension, num_workers=num_workers
        )
        if not self.population_size_adaptation:
            self.popsize.mu = max(16, self.dimension)
            self.popsize.llambda = 4 * self.popsize.mu
            self.popsize.llambda = max(self.popsize.llambda, num_workers)
            if budget is not None and self.popsize.llambda > budget:
                self.popsize.llambda = budget
                self.popsize.mu = self.popsize.llambda // 4
                warnings.warn(
                    "Budget may be too small in front of the dimension for EMNA",
                    base.InefficientSettingsWarning,
                )
        self.current_center: np.ndarray = np.zeros(self.dimension)
        self.parents: tp.List[p.Parameter] = [self.parametrization]
        self.children: tp.List[p.Parameter] = []

    def recommend(self) -> p.Parameter:
        if self.naive:
            return self.current_bests["optimistic"].parameter
        else:
            out = self.parametrization.spawn_child()
            with p.helpers.deterministic_sampling(out):
                out.set_standardized_data(self.current_center)
            return out

    def _internal_ask_candidate(self) -> p.Parameter:
        sigma_tmp = self.sigma
        if self.population_size_adaptation and self.popsize.llambda < self.min_coef_parallel_context * self.dimension:
            sigma_tmp = self.sigma * np.exp(self._rng.normal(0, 1) / np.sqrt(self.dimension))
        individual = self.current_center + sigma_tmp * self._rng.normal(0, 1, self.dimension)
        parent = self.parents[self.num_ask % len(self.parents)]
        candidate = parent.spawn_child().set_standardized_data(individual, reference=self.parametrization)
        if parent is self.parametrization:
            candidate.heritage["lineage"] = candidate.uid
        candidate._meta["sigma"] = sigma_tmp
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        if self.population_size_adaptation:
            self.popsize.add_value(loss)
        self.children.append(candidate)
        if len(self.children) >= self.popsize.llambda:
            self.children.sort(key=base._loss)
            self.parents = self.children[: self.popsize.mu]
            self.children = []
            self.current_center = sum(
                c.get_standardized_data(reference=self.parametrization) for c in self.parents
            ) / self.popsize.mu  # type: ignore
            if self.population_size_adaptation:
                if self.popsize.llambda < self.min_coef_parallel_context * self.dimension:
                    self.sigma = np.exp(
                        np.sum(np.log([c._meta["sigma"] for c in self.parents]), axis=0 if self.isotropic else None)
                        / self.popsize.mu
                    )
                else:
                    stdd = [ (self.parents[i].get_standardized_data(reference=self.parametrization) - self.current_center) ** 2 for i in range(self.popsize.mu) ]
                    self.sigma = np.sqrt(np.sum(stdd) / (self.popsize.mu * (self.dimension if self.isotropic else 1)))
            else:
                stdd = [ (self.parents[i].get_standardized_data(reference=self.parametrization) - self.current_center) ** 2 for i in range(self.popsize.mu) ]
                self.sigma = np.sqrt(np.sum(stdd, axis=0 if self.isotropic else None) / (self.popsize.mu * (self.dimension if self.isotropic else 1)))
            if self.num_workers / self.dimension > 32:
                imp = max(1, (np.log(self.popsize.llambda) / 2) ** (1 / self.dimension))
                self.sigma /= imp

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        raise base.TellNotAskedNotSupportedError

# EMNA class as a configured optimizer.
class EMNA(base.ConfiguredOptimizer):
    """Estimation of Multivariate Normal Algorithm."""
    def __init__(
        self,
        *,
        isotropic: bool = True,
        naive: bool = True,
        population_size_adaptation: bool = False,
        initial_popsize: tp.Optional[int] = None,
    ) -> None:
        super().__init__(_EMNA, locals(), as_config=True)

NaiveIsoEMNA = EMNA().set_name("NaiveIsoEMNA", register=True)

# --- NGOptBase and related classes
class NGOptBase(base.Optimizer):
    """Nevergrad optimizer by competence map."""
    def __init__(self, parametrization: p.IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        analysis = p.helpers.analyze(self.parametrization)
        funcinfo = self.parametrization.function
        self.has_noise: bool = not (analysis.deterministic and funcinfo.deterministic)
        self.has_real_noise: bool = not funcinfo.deterministic
        self.noise_from_instrumentation: bool = self.has_noise and funcinfo.deterministic
        self.fully_continuous: bool = analysis.continuous
        all_params = p.helpers.flatten(self.parametrization)
        int_layers = list(itertools.chain.from_iterable([base._layering.Int.filter_from(x) for _, x in all_params]))  # type: ignore
        int_layers = [x for x in int_layers if x.arity is not None]  # type: ignore
        self.has_discrete_not_softmax: bool = any(not isinstance(lay, base._datalayers.SoftmaxSampling) for lay in int_layers)  # type: ignore
        self._has_discrete: bool = bool(int_layers)
        self._arity: int = max((lay.arity for lay in int_layers), default=-1)  # type: ignore
        if self.fully_continuous:
            self._arity = -1
        self._optim: tp.Optional[base.Optimizer] = None
        self._constraints_manager.update(max_trials=1000, penalty_factor=1.0, penalty_exponent=1.01)

    @property
    def optim(self) -> base.Optimizer:
        if self._optim is None:
            self._optim = self._select_optimizer_cls()(self.parametrization, self.budget, self.num_workers)
            if isinstance(self._optim, NGOptBase):
                self._optim = self._optim.optim
            import logging
            logger = logging.getLogger(__name__)
            logger.debug("%s selected %s optimizer.", self.name, self._optim.name)
        return self._optim

    def _select_optimizer_cls(self) -> base.OptCls:
        # This method should be overridden in subclasses.
        raise NotImplementedError

    def _internal_ask_candidate(self) -> p.Parameter:
        return self.optim.ask()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        self.optim.tell(candidate, loss)

    def recommend(self) -> p.Parameter:
        return self.optim.recommend()

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        self.optim.tell(candidate, loss)

    def _info(self) -> tp.Dict[str, tp.Any]:
        out: tp.Dict[str, tp.Any] = {"sub-optim": self.optim.name}
        out.update(self.optim._info())
        return out

    def enable_pickling(self) -> None:
        self.optim.enable_pickling()

class NGOptDSBase(NGOptBase):
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        # Simplified selection for discrete noisy case.
        assert budget is None
        if self.has_noise and self.has_discrete_not_softmax:
            return base.DoubleFastGADiscreteOnePlusOne if self.dimension < 60 else base.CMA
        else:
            return super()._select_optimizer_cls()  # type: ignore

class Shiwa(NGOptBase):
    def _select_optimizer_cls(self) -> base.OptCls:
        optCls: base.OptCls = NGOptBase  # default fallback
        funcinfo = self.parametrization.function
        if self.has_noise and self.has_discrete_not_softmax:
            optCls = base.RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and not funcinfo.metrizable:
            optCls = base.CMA
        return optCls

class NGO(NGOptBase):
    pass

class NGOpt4(NGOptBase):
    def _select_optimizer_cls(self) -> base.OptCls:
        optimClass: base.OptCls
        budget = self.budget  # type: ignore
        if self.has_noise and self.has_discrete_not_softmax:
            optimClass = base.ParametrizedOnePlusOne(crossover=True, mutation="portfolio", noise_handling="optimistic")
        else:
            if self.has_noise and self.fully_continuous:
                optimClass = base.TBPSA
            else:
                if self.has_discrete_not_softmax or not self.parametrization.function.metrizable or not self.fully_continuous:
                    optimClass = base.DoubleFastGADiscreteOnePlusOne
                else:
                    if self.num_workers > (budget / 5 if budget is not None else 0):
                        if self.num_workers > (budget / 2.0 if budget is not None else 0) or (budget is not None and budget < self.dimension):
                            optimClass = base.MetaTuneRecentering
                        else:
                            optimClass = base.NaiveTBPSA
                    else:
                        if self.num_workers == 1 and budget is not None and budget > 6000 and self.dimension > 7:
                            optimClass = base.ChainCMAPowell
                        else:
                            if self.num_workers == 1 and budget is not None and budget < self.dimension * 30:
                                optimClass = base.OnePlusOne if self.dimension > 30 else base.Cobyla
                            else:
                                optimClass = base.DE if self.dimension > 2000 else (base.MetaCMA if self.dimension > 1 else base.OnePlusOne)
        return optimClass

class NGOpt8(NGOpt4):
    def _select_optimizer_cls(self) -> base.OptCls:
        optimClass = super()._select_optimizer_cls()
        if self.num_objectives > 1:
            self._optim = None
        return optimClass

class NGOpt10(NGOpt8):
    def _select_optimizer_cls(self) -> base.OptCls:
        return super()._select_optimizer_cls()

class NGOpt12(NGOpt10):
    def _select_optimizer_cls(self) -> base.OptCls:
        cma_vars = max(1, 4 + int(3 * math.log(self.dimension)))
        if (not self.has_noise and self.fully_continuous and self.num_workers <= cma_vars and self.dimension < 100 
            and self.budget is not None and self.budget < self.dimension * 50 and self.budget > min(50, self.dimension * 5)):
            return base.MetaModel
        elif (not self.has_noise and self.fully_continuous and self.num_workers <= cma_vars and self.dimension < 100 
              and self.budget is not None and self.budget < self.dimension * 5 and self.budget > 50):
            return base.MetaModel
        else:
            return super()._select_optimizer_cls()

class NGOpt13(NGOpt12):
    def _select_optimizer_cls(self) -> base.OptCls:
        if self.budget is not None and self.num_workers * 3 < self.budget and self.dimension < 8 and self.budget < 80:
            return base.HyperOpt
        else:
            return super()._select_optimizer_cls()

class NGOpt14(NGOpt12):
    def _select_optimizer_cls(self) -> base.OptCls:
        if self.budget is not None and self.budget < 600:
            return base.MetaModel
        else:
            return super()._select_optimizer_cls()

class NGOpt15(NGOpt12):
    def _select_optimizer_cls(self) -> base.OptCls:
        if (self.budget is not None and self.fully_continuous and self.budget < self.dimension**2 * 2 and 
            self.num_workers == 1 and not self.has_noise and self.num_objectives < 2):
            return base.MetaModelOnePlusOne
        elif self.fully_continuous and self.budget is not None and self.budget < 600:
            return base.MetaModel
        else:
            return super()._select_optimizer_cls()

class NGOpt16(NGOpt15):
    def _select_optimizer_cls(self) -> base.OptCls:
        if (self.budget is not None and self.fully_continuous and self.budget < 200 * self.dimension and 
            self.num_workers == 1 and not self.has_noise and self.num_objectives < 2 and 
            p.helpers.Normalizer(self.parametrization).fully_bounded):
            return base.Cobyla
        else:
            return super()._select_optimizer_cls()

class NGOpt21(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        cma_vars = max(1, 4 + int(3 * math.log(self.dimension)))
        num = 1 + (4 * self.budget) // (self.dimension * 1000) if self.budget is not None else 1
        if (self.budget is not None and self.budget > 500 * self.dimension and self.fully_continuous and 
            not self.has_noise and self.num_objectives < 2 and self.num_workers <= num * cma_vars):
            return base.ConfPortfolio(
                optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3**i) for i in range(num)],
                warmup_ratio=0.5,
            )
        else:
            return super()._select_optimizer_cls()

class NGOpt36(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        num = 1 + int(math.sqrt((4.0 * (4 * self.budget) / (self.dimension * 1000)))) if self.budget is not None else 1
        cma_vars = max(1, 4 + int(3 * math.log(self.dimension)))
        if (self.budget is not None and self.budget > 500 * self.dimension and self.fully_continuous and 
            not self.has_noise and self.num_workers <= num * cma_vars):
            return base.ConfPortfolio(
                optimizers=[Rescaled(base_optimizer=NGOpt14, scale=0.9**i) for i in range(num)],
                warmup_ratio=0.5,
            )
        else:
            return super()._select_optimizer_cls()

class NGOpt38(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        if (self.budget is not None and self.fully_continuous and not self.has_noise and self.num_objectives < 2 and 
            self.num_workers == 1 and p.helpers.Normalizer(self.parametrization).fully_bounded):
            if self.budget > 5000 * self.dimension:
                return NGOpt36
            if self.dimension < 5:
                return NGOpt21
            if self.dimension < 10:
                num = 1 + int(math.sqrt((8.0 * (8 * self.budget) / (self.dimension * 1000))))
                return base.ConfPortfolio(optimizers=[NGOpt14] * num, warmup_ratio=0.7)
            if self.dimension < 20:
                num = self.budget // (500 * self.dimension)
                return base.ConfPortfolio(
                    optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3**i) for i in range(num)],
                    warmup_ratio=0.5,
                )
            return NGOpt16
        elif (self.budget is not None and self.fully_continuous and not self.has_noise and 
              self.num_objectives < 2 and self.num_workers == 1 and self.budget > 50 * self.dimension and 
              p.helpers.Normalizer(self.parametrization).fully_bounded):
            return NGOpt8 if self.dimension < 3 else NGOpt15
        else:
            return super()._select_optimizer_cls()

class NGOpt39(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        cma_vars = max(1, 4 + int(3 * math.log(self.dimension)))
        num36 = 1 + int(math.sqrt((4 * self.budget) / (self.dimension * 1000))) if self.budget is not None else 1
        num21 = 1 + (4 * self.budget) // (self.dimension * 1000) if self.budget is not None else 1
        num_dim10 = 1 + int(math.sqrt((8 * self.budget) / (self.dimension * 1000))) if self.budget is not None else 1
        num_dim20 = self.budget // (500 * self.dimension) if self.budget is not None else 1
        para = 1
        if self.budget is not None and self.budget > 5000 * self.dimension:
            para = num36 * cma_vars
        elif self.dimension < 5:
            para = num21 * cma_vars
        elif self.dimension < 10:
            para = num_dim10 * cma_vars
        elif self.dimension < 20:
            para = num_dim20 * cma_vars

        if (self.budget is not None and self.fully_continuous and not self.has_noise and 
            self.num_objectives < 2 and self.num_workers <= para and p.helpers.Normalizer(self.parametrization).fully_bounded):
            if self.dimension == 1:
                return NGOpt16
            if self.budget > 5000 * self.dimension:
                return NGOpt36
            if self.dimension < 5:
                return NGOpt21
            if self.dimension < 10:
                num = 1 + int(math.sqrt((8 * self.budget) / (self.dimension * 1000)))
                return base.ConfPortfolio(optimizers=[NGOpt14] * num, warmup_ratio=0.7)
            if self.dimension < 20:
                num = self.budget // (500 * self.dimension)
                return base.ConfPortfolio(
                    optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3**i) for i in range(num)],
                    warmup_ratio=0.5,
                )
            if self.num_workers == 1:
                return base.CmaFmin2
            return NGOpt16
        elif (self.fully_continuous and not self.has_noise and self.num_objectives < 2 and 
              self.num_workers <= cma_vars and self.budget is not None and self.budget > 50 * self.dimension and 
              p.helpers.Normalizer(self.parametrization).fully_bounded):
            if self.dimension < 3:
                return NGOpt8
            if self.dimension <= 20 and self.num_workers == 1:
                from nevergrad.parametrization import parametrization as pmod
                MetaModelFmin2 = base.ParametrizedMetaModel(multivariate_optimizer=base.CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            return NGOpt15
        else:
            return super()._select_optimizer_cls()

class NGOptRW(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls:
        if self.fully_continuous and not self.has_noise and self.budget is not None and self.budget >= 12 * self.dimension:
            return base.ConfPortfolio(optimizers=[base.GeneticDE, base.PSO, super()._select_optimizer_cls()], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()

class NGOptF(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls:
        from collections import defaultdict
        best: tp.DefaultDict[float, tp.DefaultDict[float, tp.List[str]]] = defaultdict(lambda: defaultdict(list))
        def recommend_method(d: float, nod: float) -> tp.List[str]:
            best[12][833.333] += ["MemeticDE"]
            best[12][83.3333] += ["MemeticDE"]
            best[12][8.33333] += ["RealSpacePSO"]
            best[12][0.833333] += ["CauchyOnePlusOne"]
            best[12][833.333] += ["VLPCMA"]
            best[12][83.3333] += ["NLOPT_LN_SBPLX"]
            best[12][8.33333] += ["NLOPT_LN_SBPLX"]
            best[12][0.833333] += ["NLOPT_LN_SBPLX"]
            best[12][833.333] += ["VLPCMA"]
            best[12][83.3333] += ["MemeticDE"]
            best[12][8.33333] += ["SMAC3"]
            best[12][0.833333] += ["Cobyla"]
            best[24][416.667] += ["VLPCMA"]
            best[24][41.6667] += ["Wiz"]
            best[24][4.16667] += ["NLOPT_LN_SBPLX"]
            best[24][0.416667] += ["Cobyla"]
            best[24][416.667] += ["NLOPT_LN_SBPLX"]
            best[24][41.6667] += ["Wiz"]
            best[24][4.16667] += ["NLOPT_LN_SBPLX"]
            best[24][0.416667] += ["NLOPT_LN_SBPLX"]
            best[24][416.667] += ["ChainDiagonalCMAPowell"]
            best[24][41.6667] += ["NLOPT_LN_SBPLX"]
            best[24][4.16667] += ["QORealSpacePSO"]
            best[24][0.416667] += ["Cobyla"]
            best[2][5000] += ["NGOpt16"]
            best[2][500] += ["LhsDE"]
            best[2][50] += ["SODE"]
            best[2][5] += ["Carola2"]
            best[2][5000] += ["MetaModelQODE"]
            best[2][500] += ["MetaModelQODE"]
            best[2][50] += ["PCABO"]
            best[2][5] += ["HammersleySearchPlusMiddlePoint"]
            best[2][5000] += ["QORealSpacePSO"]
            best[2][500] += ["ChainDiagonalCMAPowell"]
            best[2][50] += ["MetaModelQODE"]
            best[2][5] += ["Cobyla"]
            best[5][2000] += ["MemeticDE"]
            best[5][200] += ["MemeticDE"]
            best[5][20] += ["LhsDE"]
            best[5][2] += ["MultiSQP"]
            best[5][2000] += ["MemeticDE"]
            best[5][200] += ["LhsDE"]
            best[5][20] += ["LhsDE"]
            best[5][2] += ["NLOPT_LN_SBPLX"]
            best[5][2000] += ["LhsDE"]
            best[5][200] += ["VLPCMA"]
            best[5][20] += ["BayesOptimBO"]
            best[5][2] += ["NLOPT_LN_SBPLX"]
            best[96][0.104167] += ["PCABO"]
            bestdist = float("inf")
            bestalg: tp.List[str] = []
            for d1 in best:
                for nod2 in best[d1]:
                    dist = (d - d1)**2 + (nod - nod2)**2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg

        if self.fully_continuous and not self.has_noise:
            algs = recommend_method(self.dimension, self.budget / self.dimension if self.budget is not None else 0)
            if self.num_workers > 1:
                algs = [a for a in algs if not base.registry[a].no_parallelization]
                if len(algs) == 0:
                    return base.SQPCMA
            def most_frequent(lst: tp.List[str]) -> str:
                return max(set(lst), key=lst.count)
            return base.registry[most_frequent(algs)]
        if self.fully_continuous and not self.has_noise and self.budget is not None and self.budget >= 12 * self.dimension:
            return base.ConfPortfolio(optimizers=[base.GeneticDE, base.PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()

class NGOptF2(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls:
        from collections import defaultdict
        best: tp.DefaultDict[float, tp.DefaultDict[float, tp.List[str]]] = defaultdict(lambda: defaultdict(list))
        def recommend_method(d: float, nod: float) -> tp.List[str]:
            best[12][833.333] += ["MemeticDE"]
            best[12][83.3333] += ["NGOptRW"]
            best[12][8.33333] += ["RealSpacePSO"]
            best[12][0.833333] += ["ASCMADEthird"]
            best[12][8333.33] += ["GeneticDE"]
            best[12][833.333] += ["TripleCMA"]
            best[12][83.3333] += ["NLOPT_LN_SBPLX"]
            best[12][8.33333] += ["NLOPT_LN_SBPLX"]
            best[12][0.833333] += ["NLOPT_LN_SBPLX"]
            best[12][8333.33] += ["GeneticDE"]
            best[12][833.333] += ["VLPCMA"]
            best[12][83.3333] += ["MemeticDE"]
            best[12][8.33333] += ["SMAC3"]
            best[12][0.833333] += ["Cobyla"]
            best[24][416.667] += ["VLPCMA"]
            best[24][41.6667] += ["Wiz"]
            best[24][4.16667] += ["NLOPT_LN_SBPLX"]
            best[24][0.416667] += ["Cobyla"]
            best[24][416.667] += ["NLOPT_LN_SBPLX"]
            best[24][41.6667] += ["Wiz"]
            best[24][4.16667] += ["NLOPT_LN_SBPLX"]
            best[24][0.416667] += ["NLOPT_LN_SBPLX"]
            best[24][416.667] += ["ChainDiagonalCMAPowell"]
            best[24][41.6667] += ["NLOPT_LN_SBPLX"]
            best[24][4.16667] += ["QORealSpacePSO"]
            best[24][0.416667] += ["Cobyla"]
            best[2][500000] += ["NGOpt16"]
            best[2][50000] += ["NeuralMetaModelDE"]
            best[2][5000] += ["ASCMADEthird"]
            best[2][500] += ["DiscreteDE"]
            best[2][50] += ["RFMetaModelOnePlusOne"]
            best[2][5] += ["Carola2"]
            best[2][500000] += ["NGOptF"]
            best[2][50000] += ["NGOptF"]
            best[2][5000] += ["LQODE"]
            best[2][500] += ["NLOPT_GN_DIRECT"]
            best[2][50] += ["Cobyla"]
            best[48][20.8333] += ["RLSOnePlusOne"]
            best[48][2.08333] += ["NGOptF"]
            best[48][0.208333] += ["NGOptF2"]
            best[48][2083.33] += ["NGOptF2"]
            best[48][208.333] += ["ChainNaiveTBPSAPowell"]
            best[48][20.8333] += ["ChainNaiveTBPSAPowell"]
            best[48][2.08333] += ["NLOPT_LN_NELDERMEAD"]
            best[48][0.208333] += ["BOBYQA"]
            best[5][200000] += ["BAR"]
            best[5][20000] += ["ChainNaiveTBPSACMAPowell"]
            best[5][2000] += ["CmaFmin2"]
            best[5][200] += ["RotatedTwoPointsDE"]
            best[5][20] += ["pysot"]
            best[5][2] += ["NLOPT_LN_SBPLX"]
            best[96][10.4167] += ["NGOptF"]
            best[96][1.04167] += ["NGOpt4"]
            best[96][0.104167] += ["ASCMADEthird"]
            best[96][1041.67] += ["NLOPT_LN_NELDERMEAD"]
            best[96][104.167] += ["RPowell"]
            best[96][10.4167] += ["NGOpt8"]
            best[96][1.04167] += ["RPowell"]
            best[96][0.104167] += ["NLOPT_LN_NELDERMEAD"]
            best[96][1041.67] += ["CMandAS3"]
            best[96][104.167] += ["Powell"]
            best[96][10.4167] += ["NGOptBase"]
            best[96][1.04167] += ["Powell"]
            best[96][0.104167] += ["NLOPT_LN_NELDERMEAD"]
            bestdist = float("inf")
            bestalg: tp.List[str] = []
            for d1 in best:
                for nod2 in best[d1]:
                    dist = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg

        if self.fully_continuous and not self.has_noise:
            algs = recommend_method(self.dimension, self.budget / self.dimension if self.budget is not None else 0)
            if self.num_workers > 1:
                algs = [a for a in algs if not base.registry[a].no_parallelization]
                if len(algs) == 0:
                    return base.SQPCMA
            def most_frequent(lst: tp.List[str]) -> str:
                return max(set(lst), key=lst.count)
            return base.registry[most_frequent(algs)]
        if self.fully_continuous and not self.has_noise and self.budget is not None and self.budget >= 12 * self.dimension:
            return base.ConfPortfolio(optimizers=[base.GeneticDE, base.PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()

class NGOptF3(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls:
        from collections import defaultdict
        best: tp.DefaultDict[float, tp.DefaultDict[float, tp.List[str]]] = defaultdict(lambda: defaultdict(list))
        def recommend_method(d: float, nod: float) -> tp.List[str]:
            best[12][8333.33] += ["DiscreteDE"]
            best[12][833.333] += ["DiscreteDE"]
            best[12][83.3333] += ["MetaModelDE"]
            best[12][8.33333] += ["NGOpt"]
            best[12][0.833333] += ["NLOPT_LN_NEWUOA_BOUND"]
            best[12][83333.3] += ["RPowell"]
            best[12][8333.33] += ["DiscreteDE"]
            best[12][833.333] += ["NLOPT_GN_ISRES"]
            best[12][83.3333] += ["SMAC3"]
            best[12][8.33333] += ["NGOpt"]
            best[12][0.833333] += ["NLOPT_LN_SBPLX"]
            best[24][4166.67] += ["CmaFmin2"]
            best[24][416.667] += ["CmaFmin2"]
            best[24][41.6667] += ["CmaFmin2"]
            best[24][4.16667] += ["NLOPT_LN_SBPLX"]
            best[24][0.416667] += ["Cobyla"]
            best[24][41666.7] += ["NGOptF"]
            best[24][4166.67] += ["CmaFmin2"]
            best[24][416.667] += ["RotatedTwoPointsDE"]
            best[24][41.6667] += ["CmaFmin2"]
            best[24][4.16667] += ["pysot"]
            best[24][0.416667] += ["NLOPT_LN_NELDERMEAD"]
            best[24][41666.7] += ["NGOptF"]
            best[24][4166.67] += ["Shiwa"]
            best[24][416.667] += ["CmaFmin2"]
            best[24][41.6667] += ["NLOPT_GN_CRS2_LM"]
            best[24][4.16667] += ["NLOPT_GN_CRS2_LM"]
            best[24][0.416667] += ["Cobyla"]
            best[2][500000] += ["NGOptF2"]
            best[2][50000] += ["NeuralMetaModelTwoPointsDE"]
            best[2][5000] += ["ASCMADEthird"]
            best[2][500] += ["DiscreteDE"]
            best[2][50] += ["RFMetaModelOnePlusOne"]
            best[2][5] += ["Carola2"]
            best[48][10.4167] += ["Carola2"]
            best[48][1.04167] += ["ChainDiagonalCMAPowell"]
            best[48][0.104167] += ["BOBYQA"]
            best[5][200000] += ["RescaledCMA"]
            best[5][20000] += ["RFMetaModelDE"]
            best[5][2000] += ["DiscreteDE"]
            best[5][200] += ["TwoPointsDE"]
            best[5][20] += ["NGOptF2"]
            best[5][2] += ["OnePlusLambda"]
            best[96][10.4167] += ["NGOptF"]
            best[96][1.04167] += ["NGOpt4"]
            best[96][0.104167] += ["ASCMADEthird"]
            best[96][1041.67] += ["NLOPT_LN_NELDERMEAD"]
            best[96][104.167] += ["RPowell"]
            best[96][10.4167] += ["NGOpt8"]
            best[96][1.04167] += ["RPowell"]
            best[96][0.104167] += ["NLOPT_LN_NELDERMEAD"]
            best[96][1041.67] += ["CMandAS3"]
            best[96][104.167] += ["Powell"]
            best[96][10.4167] += ["NGOptBase"]
            best[96][1.04167] += ["Powell"]
            best[96][0.104167] += ["NLOPT_LN_NELDERMEAD"]
            bestdist = float("inf")
            bestalg: tp.List[str] = []
            for d1 in best:
                for nod2 in best[d1]:
                    dist = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg

        if self.fully_continuous and not self.has_noise:
            algs = recommend_method(self.dimension, self.budget / self.dimension if self.budget is not None else 0)
            if self.num_workers > 1:
                algs = [a for a in algs if not base.registry[a].no_parallelization]
                if len(algs) == 0:
                    return base.SQPCMA
            def most_frequent(lst: tp.List[str]) -> str:
                return max(set(lst), key=lst.count)
            return base.registry[most_frequent(algs)]
        if self.fully_continuous and not self.has_noise and self.budget is not None and self.budget >= 12 * self.dimension:
            return base.ConfPortfolio(optimizers=[base.GeneticDE, base.PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()

class NGOptF5(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls:
        from collections import defaultdict
        best: tp.DefaultDict[float, tp.DefaultDict[float, tp.List[str]]] = defaultdict(lambda: defaultdict(list))
        def recommend_method(d: float, nod: float) -> tp.List[str]:
            best[12][83333.3] += ["RescaledCMA"]
            best[12][8333.33] += ["RFMetaModelDE"]
            best[12][833.333] += ["DiscreteDE"]
            best[12][83.3333] += ["TwoPointsDE"]
            best[12][8.33333] += ["pysot"]
            best[12][0.833333] += ["NLOPT_LN_SBPLX"]
            best[24][4166.67] += ["ASCMADEthird"]
            best[24][416.667] += ["ChainDiagonalCMAPowell"]
            best[24][41.6667] += ["ASCMADEthird"]
            best[24][4.16667] += ["CmaFmin2"]
            best[24][0.416667] += ["RotatedTwoPointsDE"]
            best[24][4166.67] += ["NGOptF"]
            best[24][416.667] += ["DiscreteDE"]
            best[24][41.6667] += ["NLOPT_GN_DIRECT"]
            best[24][4.16667] += ["NLOPT_LN_SBPLX"]
            best[24][0.416667] += ["NLOPT_LN_SBPLX"]
            best[96][10.4167] += ["NGOptF"]
            best[96][1.04167] += ["NGOpt4"]
            best[96][0.104167] += ["ASCMADEthird"]
            best[96][1041.67] += ["NLOPT_LN_NELDERMEAD"]
            best[96][104.167] += ["RPowell"]
            best[96][10.4167] += ["NGOpt8"]
            best[96][1.04167] += ["RPowell"]
            best[96][0.104167] += ["NLOPT_LN_NELDERMEAD"]
            bestdist = float("inf")
            bestalg: tp.List[str] = []
            for d1 in best:
                for nod2 in best[d1]:
                    dist = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg

        if self.fully_continuous and not self.has_noise:
            algs = recommend_method(self.dimension, self.budget / self.dimension if self.budget is not None else 0)
            if self.num_workers > 1:
                algs = [a for a in algs if not base.registry[a].no_parallelization]
                if len(algs) == 0:
                    return base.SQPCMA
            def most_frequent(lst: tp.List[str]) -> str:
                return max(set(lst), key=lst.count)
            return base.registry[most_frequent(algs)]
        if self.fully_continuous and not self.has_noise and self.budget is not None and self.budget >= 12 * self.dimension:
            return base.ConfPortfolio(optimizers=[base.GeneticDE, base.PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()

class NGOpt(NGOpt39):
    pass

class NGOptRW(NGOpt39):
    pass

# --- NgIoh family
class NgIoh(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return base.ConfPortfolio(
                optimizers=[base.SuperSmoothDiscreteLenglerOnePlusOne, base.SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, base.DiscreteLenglerOnePlusOne],
                warmup_ratio=0.4,
            )
        if (self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise and 
            self.budget < 1000 * self.dimension and self.budget > 20 * self.dimension and self.dimension > 1 and self.dimension < 100):
            return base.Carola2
        if (self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise and 
            self.budget >= 1000 * self.dimension and self.dimension > 1 and self.dimension < 50):
            return base.Carola2
        return optCls

class NgIoh2(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        if (self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise and
            self.budget < 1000 * self.dimension and self.budget > 20 * self.dimension and self.dimension > 1 and self.dimension < 100):
            return base.Carola2
        return super()._select_optimizer_cls()

class NgIoh3(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        return super()._select_optimizer_cls()

class NgIoh4(NGOptBase):
    def _select_optimizer_cls(self) -> base.OptCls:
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return base.ConfPortfolio(
                optimizers=[base.SuperSmoothDiscreteLenglerOnePlusOne, base.SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, base.DiscreteLenglerOnePlusOne],
                warmup_ratio=0.4,
            )
        if (self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise and
            self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and self.dimension == 2):
            return base.NLOPT_LN_SBPLX
        return optCls

class NgIoh5(NGOptBase):
    def _select_optimizer_cls(self) -> base.OptCls:
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if not self.has_noise and self._arity > 0:
            return base.RecombiningPortfolioDiscreteOnePlusOne
        if (self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise and
            self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and self.dimension == 2):
            return base.NLOPT_LN_SBPLX
        return optCls

class NgIoh6(NGOptBase):
    def _select_optimizer_cls(self) -> base.OptCls:
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        return optCls

# --- MSR and MultipleSingleRuns for multiobjective
class _MSR(base.Portfolio):
    def __init__(self, parametrization: p.IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1, num_single_runs: int = 9, base_optimizer: base.OptCls = NGOpt) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=base.ConfPortfolio(optimizers=[base.OptCls] * num_single_runs))
        self.coeffs: tp.List[np.ndarray] = []
    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        if not self.coeffs:
            self.coeffs = [self.parametrization.random_state.uniform(size=self.num_objectives) for _ in self.optims]
        for coeffs, opt in zip(self.coeffs, self.optims):
            this_loss = np.sum(loss * coeffs)
            opt.tell(candidate, this_loss)

class MultipleSingleRuns(base.ConfiguredOptimizer):
    def __init__(self, *, num_single_runs: int = 9, base_optimizer: base.OptCls = NGOpt) -> None:
        super().__init__(_MSR, locals())

# --- Smooth variants (assignments only)
SmoothDiscreteOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, mutation="discrete").set_name("SmoothDiscreteOnePlusOne", register=True)
SmoothPortfolioDiscreteOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, mutation="portfolio").set_name("SmoothPortfolioDiscreteOnePlusOne", register=True)
SmoothDiscreteLenglerOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, mutation="lengler").set_name("SmoothDiscreteLenglerOnePlusOne", register=True)
SmoothDiscreteLognormalOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, mutation="lognormal").set_name("SmoothDiscreteLognormalOnePlusOne", register=True)
SuperSmoothDiscreteLenglerOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, mutation="lengler", antismooth=9).set_name("SuperSmoothDiscreteLenglerOnePlusOne", register=True)
SuperSmoothTinyLognormalDiscreteOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, mutation="tinylognormal", antismooth=9).set_name("SuperSmoothTinyLognormalDiscreteOnePlusOne", register=True)
UltraSmoothDiscreteLenglerOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, mutation="lengler", antismooth=3).set_name("UltraSmoothDiscreteLenglerOnePlusOne", register=True)
SmootherDiscreteLenglerOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, mutation="lengler", antismooth=2, super_radii=True).set_name("SmootherDiscreteLenglerOnePlusOne", register=True)
YoSmoothDiscreteLenglerOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, mutation="lengler", antismooth=2).set_name("YoSmoothDiscreteLenglerOnePlusOne", register=True)
CMAL = base.Chaining([base.CMA, base.DiscreteLenglerOnePlusOne], ["half"]).set_name("CMAL", register=True)
PortfolioDiscreteOnePlusOne = base.ParametrizedOnePlusOne(mutation="portfolio").set_name("PortfolioDiscreteOnePlusOne", register=True)
# Additional smooth variants...
SmoothLognormalDiscreteOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, mutation="lognormal").set_name("SmoothLognormalDiscreteOnePlusOne", register=True)
SmoothAdaptiveDiscreteOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, mutation="adaptive").set_name("SmoothAdaptiveDiscreteOnePlusOne", register=True)
SmoothRecombiningPortfolioDiscreteOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, crossover=True, mutation="portfolio").set_name("SmoothRecombiningPortfolioDiscreteOnePlusOne", register=True)
SmoothRecombiningDiscreteLenglerOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, crossover=True, mutation="lengler").set_name("SmoothRecombiningDiscreteLenglerOnePlusOne", register=True)
UltraSmoothRecombiningDiscreteLenglerOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, crossover=True, mutation="lengler", antismooth=3).set_name("UltraSmoothRecombiningDiscreteLenglerOnePlusOne", register=True)
UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, crossover=True, mutation="lognormal", antismooth=3, roulette_size=7).set_name("UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne", register=True)
UltraSmoothElitistRecombiningDiscreteLenglerOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, crossover=True, mutation="lengler", antismooth=3, roulette_size=7).set_name("UltraSmoothElitistRecombiningDiscreteLenglerOnePlusOne", register=True)
SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, crossover=True, mutation="lengler", antismooth=9, roulette_size=7).set_name("SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne", register=True)
SuperSmoothRecombiningDiscreteLenglerOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, crossover=True, mutation="lengler", antismooth=9).set_name("SuperSmoothRecombiningDiscreteLenglerOnePlusOne", register=True)
SuperSmoothRecombiningDiscreteLognormalOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, crossover=True, mutation="lognormal", antismooth=9).set_name("SuperSmoothRecombiningDiscreteLognormalOnePlusOne", register=True)
SmoothElitistRecombiningDiscreteLenglerOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, crossover=True, mutation="lengler", roulette_size=7).set_name("SmoothElitistRecombiningDiscreteLenglerOnePlusOne", register=True)
SmoothElitistRandRecombiningDiscreteLenglerOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, crossover=True, mutation="lengler", roulette_size=7, crossover_type="rand").set_name("SmoothElitistRandRecombiningDiscreteLenglerOnePlusOne", register=True)
SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne = base.ParametrizedOnePlusOne(smoother=True, crossover=True, mutation="lognormal", roulette_size=7, crossover_type="rand").set_name("SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne", register=True)
RecombiningDiscreteLenglerOnePlusOne = base.ParametrizedOnePlusOne(crossover=True, mutation="lengler").set_name("RecombiningDiscreteLenglerOnePlusOne", register=True)
RecombiningDiscreteLognormalOnePlusOne = base.ParametrizedOnePlusOne(crossover=True, mutation="lognormal").set_name("RecombiningDiscreteLognormalOnePlusOne", register=True)

# --- Chaining optimizer
class _Chain(base.Optimizer):
    def __init__(
        self,
        parametrization: p.IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        optimizers: tp.Optional[tp.Sequence[tp.Union[base.ConfiguredOptimizer, tp.Type[base.Optimizer]]]] = None,
        budgets: tp.Sequence[tp.Union[str, int]] = ("10",),
        no_crossing: tp.Optional[bool] = False,
    ) -> None:
        if optimizers is None:
            optimizers = [base.LHSSearch, base.DE]
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.no_crossing: bool = no_crossing
        converter: tp.Dict[str, tp.Union[None, float]] = {
            "num_workers": self.num_workers,
            "dimension": self.dimension,
            "half": self.budget // 2 if self.budget is not None else self.num_workers,
            "third": self.budget // 3 if self.budget is not None else self.num_workers,
            "fourth": self.budget // 4 if self.budget is not None else self.num_workers,
            "tenth": self.budget // 10 if self.budget is not None else self.num_workers,
            "equal": self.budget // (len(budgets) + 1) if self.budget is not None else self.num_workers,
            "most": (self.budget * 4) // 5 if self.budget is not None else self.num_workers,
            "sqrt": int(np.sqrt(self.budget)) if self.budget is not None else self.num_workers,
        }
        self.budgets: tp.List[int] = [max(1, converter[b] if isinstance(b, str) and converter[b] is not None else b) for b in budgets]  # type: ignore
        last_budget: tp.Optional[int] = None if self.budget is None else max(4, self.budget - sum(self.budgets))
        assert len(optimizers) == len(self.budgets) + 1
        for b in self.budgets:
            assert isinstance(b, (int, float)) and b > 0, str(self.budgets)
        self.optimizers: tp.List[base.Optimizer] = []
        mono, multi = base.ConfSplitOptimizer().monovariate_optimizer, base.ConfSplitOptimizer().multivariate_optimizer
        for param in (p.Array(shape=(1,)) for _ in range(len(optimizers))):  # dummy split
            if param.dimension > 1:
                self.optimizers.append(multi(param, self.budget, num_workers))
            else:
                self.optimizers.append(mono(param, self.budget, num_workers))
        self.turns: tp.List[int] = []
        self._current: int = -1

    def _internal_ask_candidate(self) -> p.Parameter:
        candidates: tp.List[p.Parameter] = []
        for i, opt in enumerate(self.optimizers):
            candidates.append(opt.ask())
        data = np.concatenate([c.get_standardized_data(reference=opt.parametrization) for c, opt in zip(candidates, self.optimizers)], axis=0)
        cand = self.parametrization.spawn_child().set_standardized_data(data)
        cand._meta["optim_index"] = self._current % len(self.optimizers)
        return cand

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        for opt in self.optimizers:
            try:
                opt.tell(candidate, loss)
            except base.TellNotAskedNotSupportedError:
                pass

    def enable_pickling(self) -> None:
        for opt in self.optimizers:
            opt.enable_pickling()

class Chaining(base.ConfiguredOptimizer):
    def __init__(
        self,
        optimizers: tp.Sequence[tp.Union[base.ConfiguredOptimizer, tp.Type[base.Optimizer]]],
        budgets: tp.Sequence[tp.Union[str, int]],
        no_crossing: tp.Optional[bool] = False,
    ) -> None:
        super().__init__(_Chain, locals())

# New names assignments via Chaining
CMAL = Chaining([base.CMA, base.DiscreteLenglerOnePlusOne], ["half"]).set_name("CMAL", register=True)
GeneticDE = Chaining([base.RotatedTwoPointsDE, base.TwoPointsDE], [200]).set_name("GeneticDE", register=True)
MemeticDE = Chaining([base.RotatedTwoPointsDE, base.TwoPointsDE, base.DE, base.SQP], ["fourth", "fourth", "fourth"]).set_name("MemeticDE", register=True)
QNDE = Chaining([base.QODE, base.RBFGS], ["half"]).set_name("QNDE", register=True)
ChainDE = Chaining([base.DE, base.RBFGS], ["half"]).set_name("ChainDE", register=True)
OpoDE = Chaining([base.OnePlusOne, base.QODE], ["half"]).set_name("OpoDE", register=True)
OpoTinyDE = Chaining([base.OnePlusOne, base.TinyQODE], ["half"]).set_name("OpoTinyDE", register=True)
QNDE.no_parallelization = True
ChainDE.no_parallelization = True
Carola1 = Chaining([base.Cobyla, base.MetaModel], ["half"]).set_name("Carola1", register=True)
Carola2 = Chaining([base.Cobyla, base.MetaModel, base.SQP], ["third", "third"]).set_name("Carola2", register=True)
DS2 = Chaining([base.Cobyla, base.MetaModelDSproba, base.SQP], ["third", "third"]).set_name("DS2", register=True)
Carola4 = Chaining([base.Cobyla, base.MetaModel, base.SQP], ["sqrt", "half"]).set_name("Carola4", register=True)
DS4 = Chaining([base.Cobyla, base.MetaModelDSproba, base.SQP], ["sqrt", "half"]).set_name("DS4", register=True)
DS5 = Chaining([base.Cobyla, base.MetaModelDSproba, base.SQP], ["sqrt", "most"]).set_name("DS5", register=True)
Carola6 = Chaining([base.Cobyla, base.MetaModel, base.SQP], ["tenth", "most"]).set_name("Carola6", register=True)
DS6 = Chaining([base.Cobyla, base.MetaModelDSproba, base.SQP], ["tenth", "most"]).set_name("DS6", register=True)
DS6.no_parallelization = True
PCarola6 = base.Rescaled(base_optimizer=_Chain, scale=10.0).set_name("PCarola6", register=True)
pCarola6 = base.Rescaled(base_optimizer=_Chain, scale=3.0).set_name("pCarola6", register=True)
Carola1.no_parallelization = True
Carola2.no_parallelization = True
DS2.no_parallelization = True
Carola4.no_parallelization = True
DS4.no_parallelization = True
DS5.no_parallelization = True
Carola5 = Chaining([base.Cobyla, base.MetaModel, base.SQP], ["sqrt", "most"]).set_name("Carola5", register=True)
Carola5.no_parallelization = True
Carola6.no_parallelization = True
PCarola6.no_parallelization = True
pCarola6.no_parallelization = True
Carola7 = Chaining([base.MultiCobyla, base.MetaModel, base.MultiSQP], ["tenth", "most"]).set_name("Carola7", register=True)
Carola8 = Chaining([base.Cobyla, base.CmaFmin2, base.SQP], ["tenth", "most"]).set_name("Carola8", register=True)
DS8 = Chaining([base.Cobyla, base.DSproba, base.SQP], ["tenth", "most"]).set_name("DS8", register=True)
Carola9 = Chaining([base.Cobyla, base.ParametrizedMetaModel(multivariate_optimizer=base.CmaFmin2), base.SQP], ["tenth", "most"]).set_name("Carola9", register=True)
DS9 = Chaining([base.Cobyla, base.ParametrizedMetaModel(multivariate_optimizer=base.DSproba), base.SQP], ["tenth", "most"]).set_name("DS9", register=True)
Carola9.no_parallelization = True
Carola8.no_parallelization = True
DS8.no_parallelization = True
Carola10 = Chaining([base.Cobyla, base.CmaFmin2, base.RBFGS], ["tenth", "most"]).set_name("Carola10", register=True)
Carola10.no_parallelization = True

BAR = base.ConfPortfolio(optimizers=[base.OnePlusOne, base.DiagonalCMA, base.OpoDE], warmup_ratio=0.5).set_name("BAR", register=True)
BAR2 = base.ConfPortfolio(optimizers=[base.OnePlusOne, base.MetaCMA, base.OpoDE], warmup_ratio=0.5).set_name("BAR2", register=True)
BAR3 = base.ConfPortfolio(optimizers=[base.RandomSearch, base.OnePlusOne, base.MetaCMA, base.QNDE], warmup_ratio=0.5).set_name("BAR3", register=True)
base.MemeticDE.no_parallelization = True
discretememetic = Chaining([base.RandomSearch, base.DiscreteLenglerOnePlusOne, base.DiscreteOnePlusOne], ["third", "third"]).set_name("discretememetic", register=True)
ChainCMAPowell = Chaining([base.MetaCMA, base.Powell], ["half"]).set_name("ChainCMAPowell", register=True)
ChainDSPowell = Chaining([base.DSproba, base.Powell], ["half"]).set_name("ChainDSPowell", register=True)
ChainMetaModelSQP = Chaining([base.MetaModel, base.SQP], ["half"]).set_name("ChainMetaModelSQP", register=True)
ChainMetaModelDSSQP = Chaining([base.MetaModelDSproba, base.SQP], ["half"]).set_name("ChainMetaModelDSSQP", register=True)
ChainMetaModelSQP.no_parallelization = True
ChainMetaModelDSSQP.no_parallelization = True
ChainMetaModelPowell = Chaining([base.MetaModel, base.Powell], ["half"]).set_name("ChainMetaModelPowell", register=True)
ChainMetaModelPowell.no_parallelization = True
ChainDiagonalCMAPowell = Chaining([base.DiagonalCMA, base.Powell], ["half"]).set_name("ChainDiagonalCMAPowell", register=True)
ChainDiagonalCMAPowell.no_parallelization = True
ChainNaiveTBPSAPowell = Chaining([base.NaiveTBPSA, base.Powell], ["half"]).set_name("ChainNaiveTBPSAPowell", register=True)
ChainNaiveTBPSAPowell.no_parallelization = True
ChainNaiveTBPSACMAPowell = Chaining([base.NaiveTBPSA, base.MetaCMA, base.Powell], ["third", "third"]).set_name("ChainNaiveTBPSACMAPowell", register=True)
ChainNaiveTBPSACMAPowell.no_parallelization = True
Carola7 = Chaining([base.Cobyla, base.MetaModel, base.SQP], ["tenth", "most"]).set_name("Carola7", register=True)
DS8 = Chaining([base.Cobyla, base.DSproba, base.SQP], ["tenth", "most"]).set_name("DS8", register=True)
Carola9 = Chaining([base.Cobyla, base.ParametrizedMetaModel(multivariate_optimizer=base.CmaFmin2), base.SQP], ["tenth", "most"]).set_name("Carola9", register=True)
DS9 = Chaining([base.Cobyla, base.ParametrizedMetaModel(multivariate_optimizer=base.DSproba), base.SQP], ["tenth", "most"]).set_name("DS9", register=True)
Carola9.no_parallelization = True
Carola8.no_parallelization = True
DS8.no_parallelization = True
Carola10 = Chaining([base.Cobyla, base.CmaFmin2, base.RBFGS], ["tenth", "most"]).set_name("Carola10", register=True)
Carola10.no_parallelization = True

# --- cGA and EMNA already defined above.

# --- NGOpt variants
class ShiwaNG(NGOptBase):
    def _select_optimizer_cls(self) -> base.OptCls:
        return Shiwa._select_optimizer_cls(self)

# For compatibility, additional classes derived from NGOptBase are registered.
class NGOptDS(NGOptDSBase):
    pass

class NGOH(NGOptBase):
    pass

# NgIoh variants have been defined above.
# Registering variants with pickling enabled etc.
base.registry.register("NGOpt", NGOpt)
base.registry.register("Shiwa", Shiwa)
base.registry.register("NGOptRW", NGOptRW)
base.registry.register("NGOptF", NGOptF)
base.registry.register("NGOptF2", NGOptF2)
base.registry.register("NGOptF3", NGOptF3)
base.registry.register("NGOptF5", NGOptF5)
base.registry.register("NgIoh", NgIoh)
base.registry.register("NgIoh2", NgIoh2)
base.registry.register("NgIoh3", NgIoh3)
base.registry.register("NgIoh4", NgIoh4)
base.registry.register("NgIoh5", NgIoh5)
base.registry.register("NgIoh6", NgIoh6)

# --- Additional multiobjective or RL-oriented portfolios
MixDeterministicRL = base.ConfPortfolio(optimizers=[base.CMA, base.PSO, base.GeneticDE]).set_name("MixDeterministicRL", register=True)
SpecialRL = base.Chaining([base.MixDeterministicRL, base.TBPSA], ["half"]).set_name("SpecialRL", register=True)
NoisyRL1 = base.Chaining([base.MixDeterministicRL, base.NoisyOnePlusOne], ["half"]).set_name("NoisyRL1", register=True)
NoisyRL2 = base.Chaining([base.MixDeterministicRL, base.RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne], ["half"]).set_name("NoisyRL2", register=True)
NoisyRL3 = base.Chaining([base.MixDeterministicRL, base.OptimisticNoisyOnePlusOne], ["half"]).set_name("NoisyRL3", register=True)

# --- NgIoh additional iterations
class NgIoh21(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        return NGOpt16._select_optimizer_cls(self)

class NgIohTuned(CSEC11 := NGOptF3):
    pass

# --- Split optimizers
SplitCSEC11 = base.ConfSplitOptimizer(max_num_vars=15, progressive=False, multivariate_optimizer=NGOptF3).set_name("SplitCSEC11", register=True)
SplitSQOPSO = base.ConfSplitOptimizer(multivariate_optimizer=base.SQOPSO, monovariate_optimizer=base.SQOPSO, non_deterministic_descriptor=False).set_name("SplitSQOPSO", register=True)
SplitPSO = base.ConfSplitOptimizer(multivariate_optimizer=base.PSO, monovariate_optimizer=base.PSO, non_deterministic_descriptor=False).set_name("SplitPSO", register=True)
SplitCMA = base.ConfSplitOptimizer(multivariate_optimizer=base.CMA, monovariate_optimizer=base.CMA, non_deterministic_descriptor=False).set_name("SplitCMA", register=True)
SplitQODE = base.ConfSplitOptimizer(multivariate_optimizer=base.QODE, monovariate_optimizer=base.QODE, non_deterministic_descriptor=False).set_name("SplitQODE", register=True)
SplitTwoPointsDE = base.ConfSplitOptimizer(multivariate_optimizer=base.TwoPointsDE, monovariate_optimizer=base.TwoPointsDE, non_deterministic_descriptor=False).set_name("SplitTwoPointsDE", register=True)
SplitDE = base.ConfSplitOptimizer(multivariate_optimizer=base.DE, monovariate_optimizer=base.DE, non_deterministic_descriptor=False).set_name("SplitDE", register=True)

SQOPSODCMA = base.Chaining([base.SQOPSO, base.DiagonalCMA], ["half"]).set_name("SQOPSODCMA", register=True)
SQOPSODCMA20 = base.Chaining(optimizers=[SQOPSODCMA] * 20, budgets=["equal"] * 19, no_crossing=True).set_name("SQOPSODCMA20", register=True)
SQOPSODCMA20bar = base.ConfPortfolio(optimizers=[SQOPSODCMA] * 20, warmup_ratio=0.5).set_name("SQOPSODCMA20bar", register=True)
NgIohLn = base.Chaining([base.LognormalDiscreteOnePlusOne, CSEC11], ["tenth"]).set_name("NgIohLn", register=True)
CMALn = base.Chaining([base.LognormalDiscreteOnePlusOne, base.CMA], ["tenth"]).set_name("CMALn", register=True)
CMARS = base.Chaining([base.RandomSearch, base.CMA], ["tenth"]).set_name("CMARS", register=True)
NgIohRS = base.Chaining([base.RandomSearch, base.NGOpt], ["tenth"]).set_name("NgIohRS", register=True)
PolyLN = base.ConfPortfolio(optimizers=[base.Rescaled(base_optimizer=base.SmallLognormalDiscreteOnePlusOne, scale=1e-3) for i in range(20)], warmup_ratio=0.5).set_name("PolyLN", register=True)
MultiLN = base.ConfPortfolio(optimizers=[base.Rescaled(base_optimizer=base.SmallLognormalDiscreteOnePlusOne, scale=2.0 ** (i - 5)) for i in range(6)], warmup_ratio=0.5).set_name("MultiLN", register=True)
ManyLN = base.ConfPortfolio(optimizers=[base.SmallLognormalDiscreteOnePlusOne for i in range(20)], warmup_ratio=0.5).set_name("ManyLN", register=True)
NgIohMLn = base.Chaining([MultiLN, CSEC11], ["tenth"]).set_name("NgIohMLn", register=True)

# --- RL specific portfolio
MixDeterministicRL = base.ConfPortfolio(optimizers=[base.CMA, base.PSO, base.GeneticDE]).set_name("MixDeterministicRL", register=True)
SpecialRL = base.Chaining([MixDeterministicRL, base.TBPSA], ["half"]).set_name("SpecialRL", register=True)
NoisyRL1 = base.Chaining([MixDeterministicRL, base.NoisyOnePlusOne], ["half"]).set_name("NoisyRL1", register=True)
NoisyRL2 = base.Chaining([MixDeterministicRL, base.RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne], ["half"]).set_name("NoisyRL2", register=True)
NoisyRL3 = base.Chaining([MixDeterministicRL, base.OptimisticNoisyOnePlusOne], ["half"]).set_name("NoisyRL3", register=True)

# --- Additional BFGS and SQP portfolios
class MultiBFGSPlus(base.Portfolio):
    def __init__(self, parametrization: p.IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        num_workers_local = num_workers
        optims: tp.List[base.Optimizer] = [base.Rescaled(base_optimizer=base.BFGS, scale=1.0 / np.exp(np.random.rand()))(self.parametrization, num_workers=1) for _ in range(num_workers_local)]
        for opt in optims[2:]:
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)

class LogMultiBFGSPlus(base.Portfolio):
    def __init__(self, parametrization: p.IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None:
            num_workers_local = int(max(num_workers, 1 + np.log(budget)))
        else:
            num_workers_local = num_workers
        optims: tp.List[base.Optimizer] = [base.Rescaled(base_optimizer=base.BFGS, scale=1.0 / np.exp(np.random.rand()))(self.parametrization, num_workers=1) for _ in range(num_workers_local)]
        for opt in optims[2:]:
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)

class SqrtMultiBFGSPlus(base.Portfolio):
    def __init__(self, parametrization: p.IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None:
            num_workers_local = int(max(num_workers, 1 + np.sqrt(budget)))
        else:
            num_workers_local = num_workers
        optims: tp.List[base.Optimizer] = [base.Rescaled(base_optimizer=base.BFGS, scale=1.0 / np.exp(np.random.rand()))(self.parametrization, num_workers=1) for _ in range(num_workers_local)]
        for opt in optims[2:]:
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)

class MultiCobylaPlus(base.Portfolio):
    def __init__(self, parametrization: p.IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        optims: tp.List[base.Optimizer] = [base.Rescaled(base_optimizer=base.Cobyla, scale=1.0 / np.exp(np.random.rand()))(self.parametrization, num_workers=1) for _ in range(num_workers)]
        for opt in optims[2:]:
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)

# Finally, additional registrations of portfolios and chaining variants are performed.
base.registry.register("MultiBFGSPlus", MultiBFGSPlus)
base.registry.register("LogMultiBFGSPlus", LogMultiBFGSPlus)
base.registry.register("SqrtMultiBFGSPlus", SqrtMultiBFGSPlus)
base.registry.register("MultiCobylaPlus", MultiCobylaPlus)

# The above annotated code now includes type hints on method arguments and return types.
