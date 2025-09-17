from typing import Optional, List, Dict, Any, Type
from collections import defaultdict

# The following assignments define configured optimizer instances with type hints.
SplitCSEC11: ConfSplitOptimizer = ConfSplitOptimizer(multivariate_optimizer=CSEC11, monovariate_optimizer=CSEC11, non_deterministic_descriptor=False).set_name('SplitCSEC11', register=True)
SplitSQOPSO: ConfSplitOptimizer = ConfSplitOptimizer(multivariate_optimizer=SQOPSO, monovariate_optimizer=SQOPSO, non_deterministic_descriptor=False).set_name('SplitSQOPSO', register=True)
SplitPSO: ConfSplitOptimizer = ConfSplitOptimizer(multivariate_optimizer=PSO, monovariate_optimizer=PSO, non_deterministic_descriptor=False).set_name('SplitPSO', register=True)
SplitCMA: ConfSplitOptimizer = ConfSplitOptimizer(multivariate_optimizer=CMA, monovariate_optimizer=CMA, non_deterministic_descriptor=False).set_name('SplitCMA', register=True)
SplitQODE: ConfSplitOptimizer = ConfSplitOptimizer(multivariate_optimizer=QODE, monovariate_optimizer=QODE, non_deterministic_descriptor=False).set_name('SplitQODE', register=True)
SplitTwoPointsDE: ConfSplitOptimizer = ConfSplitOptimizer(multivariate_optimizer=TwoPointsDE, monovariate_optimizer=TwoPointsDE, non_deterministic_descriptor=False).set_name('SplitTwoPointsDE', register=True)
SplitDE: ConfSplitOptimizer = ConfSplitOptimizer(multivariate_optimizer=DE, monovariate_optimizer=DE, non_deterministic_descriptor=False).set_name('SplitDE', register=True)
SQOPSODCMA: Any = Chaining([SQOPSO, DiagonalCMA], ['half']).set_name('SQOPSODCMA', register=True)
SQOPSODCMA20: Any = Chaining(optimizers=[SQOPSODCMA] * 20, budgets=['equal'] * 19, no_crossing=True).set_name('SQOPSODCMA20', register=True)
SQOPSODCMA20bar: Any = ConfPortfolio(optimizers=[SQOPSODCMA] * 20, warmup_ratio=0.5).set_name('SQOPSODCMA20bar', register=True)
NgIohLn: Any = Chaining([LognormalDiscreteOnePlusOne, CSEC11], ['tenth']).set_name('NgIohLn', register=True)
CMALn: Any = Chaining([LognormalDiscreteOnePlusOne, CMA], ['tenth']).set_name('CMALn', register=True)
CMARS: Any = Chaining([RandomSearch, CMA], ['tenth']).set_name('CMARS', register=True)
NgIohRS: Any = Chaining([oneshot.RandomSearch, CSEC11], ['tenth']).set_name('NgIohRS', register=True)
PolyLN: Any = ConfPortfolio(optimizers=[Rescaled(base_optimizer=SmallLognormalDiscreteOnePlusOne, scale=np.random.rand()) for i in range(20)], warmup_ratio=0.5).set_name('PolyLN', register=True)
MultiLN: Any = ConfPortfolio(optimizers=[Rescaled(base_optimizer=SmallLognormalDiscreteOnePlusOne, scale=2.0 ** (i - 5)) for i in range(6)], warmup_ratio=0.5).set_name('MultiLN', register=True)
ManyLN: Any = ConfPortfolio(optimizers=[SmallLognormalDiscreteOnePlusOne for i in range(20)], warmup_ratio=0.5).set_name('ManyLN', register=True)
NgIohMLn: Any = Chaining([MultiLN, CSEC11], ['tenth']).set_name('NgIohMLn', register=True)

# Begin class definitions with type annotations.
class CSEC(NGOpt39):
    def _select_optimizer_cls(self, budget: Optional[int] = None) -> Any:
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and (self.budget >= 3000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return NGOpt21._select_optimizer_cls(self, budget)
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and (self.budget >= 30 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return NGOpt4._select_optimizer_cls(self)
        if self.fully_continuous and self.budget is not None and (self.num_workers > self.budget / 5) and (self.budget >= 30 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return NGDS3
        return NGOpt4._select_optimizer_cls(self)


class NGOptF(NGOpt39):
    def _select_optimizer_cls(self) -> Any:
        best: Dict[int, Dict[float, List[str]]] = defaultdict(lambda: defaultdict(list))
        def recommend_method(d: int, nod: float) -> List[str]:
            best[12][833.333] += ['MemeticDE']
            best[12][83.3333] += ['MemeticDE']
            best[12][8.33333] += ['RealSpacePSO']
            best[12][0.833333] += ['CauchyOnePlusOne']
            best[12][833.333] += ['VLPCMA']
            best[12][83.3333] += ['NLOPT_LN_SBPLX']
            best[12][8.33333] += ['NLOPT_LN_SBPLX']
            best[12][0.833333] += ['NLOPT_LN_SBPLX']
            best[12][833.333] += ['VLPCMA']
            best[12][83.3333] += ['MemeticDE']
            best[12][8.33333] += ['SMAC3']
            best[12][0.833333] += ['Cobyla']
            best[24][416.667] += ['VLPCMA']
            best[24][41.6667] += ['Wiz']
            best[24][4.16667] += ['NLOPT_LN_SBPLX']
            best[24][0.416667] += ['Cobyla']
            best[24][416.667] += ['NLOPT_LN_SBPLX']
            best[24][41.6667] += ['Wiz']
            best[24][4.16667] += ['NLOPT_LN_SBPLX']
            best[24][0.416667] += ['NLOPT_LN_SBPLX']
            best[24][416.667] += ['ChainDiagonalCMAPowell']
            best[24][41.6667] += ['NLOPT_LN_SBPLX']
            best[24][4.16667] += ['QORealSpacePSO']
            best[24][0.416667] += ['Cobyla']
            best[2][5000] += ['NGOpt16']
            best[2][500] += ['LhsDE']
            best[2][50] += ['SODE']
            best[2][5] += ['Carola2']
            best[2][5000] += ['MetaModelQODE']
            best[2][500] += ['MetaModelQODE']
            best[2][50] += ['PCABO']
            best[2][5] += ['HammersleySearchPlusMiddlePoint']
            best[2][5000] += ['QORealSpacePSO']
            best[2][500] += ['ChainDiagonalCMAPowell']
            best[2][50] += ['MetaModelQODE']
            best[2][5] += ['Cobyla']
            best[5][2000] += ['MemeticDE']
            best[5][200] += ['MemeticDE']
            best[5][20] += ['LhsDE']
            best[5][2] += ['MultiSQP']
            best[5][2000] += ['MemeticDE']
            best[5][200] += ['LhsDE']
            best[5][20] += ['LhsDE']
            best[5][2] += ['NLOPT_LN_SBPLX']
            best[5][2000] += ['LhsDE']
            best[5][200] += ['VLPCMA']
            best[5][20] += ['BayesOptimBO']
            best[5][2] += ['NLOPT_LN_SBPLX']
            best[96][0.104167] += ['PCABO']
            bestdist: float = float('inf')
            bestalg: List[str] = []
            for d1 in best:
                for nod2 in best[d1]:
                    dist: float = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg
        if self.fully_continuous and (not self.has_noise):
            algs: List[str] = recommend_method(self.dimension, self.budget / self.dimension if self.budget else 0)
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA
            def most_frequent(List_: List[str]) -> str:
                return max(set(List_), key=List_.count)
            return registry[most_frequent(algs)]
        if self.fully_continuous and (not self.has_noise) and (self.budget is not None and self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()


class NGOptF2(NGOpt39):
    def _select_optimizer_cls(self) -> Any:
        best: Dict[int, Dict[float, List[str]]] = defaultdict(lambda: defaultdict(list))
        def recommend_method(d: int, nod: float) -> List[str]:
            best[12][833.333] += ['MemeticDE']
            best[12][83.3333] += ['NGOptF2']
            best[12][8.33333] += ['NGOpt']
            best[12][0.833333] += ['NLOPT_GN_DIRECT']
            best[12][833.333] += ['VLPCMA']
            best[12][83.3333] += ['MemeticDE']
            best[12][8.33333] += ['SMAC3']
            best[12][0.833333] += ['Cobyla']
            best[24][416.667] += ['VLPCMA']
            best[24][41.6667] += ['Wiz']
            best[24][4.16667] += ['NLOPT_LN_SBPLX']
            best[24][0.416667] += ['Cobyla']
            best[24][416.667] += ['NLOPT_LN_SBPLX']
            best[24][41.6667] += ['Wiz']
            best[24][4.16667] += ['NLOPT_LN_SBPLX']
            best[24][0.416667] += ['NLOPT_LN_SBPLX']
            best[24][416.667] += ['ChainDiagonalCMAPowell']
            best[24][41.6667] += ['NLOPT_LN_SBPLX']
            best[24][4.16667] += ['QORealSpacePSO']
            best[24][0.416667] += ['Cobyla']
            best[2][5000] += ['NGOpt16']
            best[2][500] += ['LhsDE']
            best[2][50] += ['SODE']
            best[2][5] += ['Carola2']
            best[2][5000] += ['MetaModelQODE']
            best[2][500] += ['MetaModelQODE']
            best[2][50] += ['PCABO']
            best[2][5] += ['HammersleySearchPlusMiddlePoint']
            best[2][5000] += ['NGOptF2']
            best[2][500] += ['ChainDiagonalCMAPowell']
            best[2][50] += ['MetaModelQODE']
            best[2][5] += ['Cobyla']
            best[5][2000] += ['MemeticDE']
            best[5][200] += ['MemeticDE']
            best[5][20] += ['LhsDE']
            best[5][2] += ['MultiSQP']
            best[5][2000] += ['MemeticDE']
            best[5][200] += ['LhsDE']
            best[5][20] += ['LhsDE']
            best[5][2] += ['NLOPT_LN_SBPLX']
            best[5][2000] += ['LhsDE']
            best[5][200] += ['VLPCMA']
            best[5][20] += ['BayesOptimBO']
            best[5][2] += ['NLOPT_LN_SBPLX']
            best[96][0.104167] += ['PCABO']
            bestdist: float = float('inf')
            bestalg: List[str] = []
            for d1 in best:
                for nod2 in best[d1]:
                    dist: float = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg
        if self.fully_continuous and (not self.has_noise):
            algs: List[str] = recommend_method(self.dimension, self.budget / self.dimension if self.budget else 0)
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA
            def most_frequent(List_: List[str]) -> str:
                return max(set(List_), key=List_.count)
            return registry[most_frequent(algs)]
        if self.fully_continuous and (not self.has_noise) and (self.budget is not None and self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()


class NGOptF3(NGOpt39):
    def _select_optimizer_cls(self) -> Any:
        best: Dict[int, Dict[float, List[str]]] = defaultdict(lambda: defaultdict(list))
        def recommend_method(d: int, nod: float) -> List[str]:
            best[12][8333.33] += ['DiscreteDE']
            best[12][833.333] += ['DiscreteDE']
            best[12][83.3333] += ['MetaModelDE']
            best[12][8.33333] += ['NGOpt']
            best[12][0.833333] += ['NLOPT_LN_NEWUOA_BOUND']
            best[12][83333.3] += ['NGOptBase']
            best[12][8333.33] += ['RPowell']
            best[12][833.333] += ['NLOPT_GN_ISRES']
            best[12][83.3333] += ['SMAC3']
            best[12][8.33333] += ['NGOpt']
            best[12][0.833333] += ['NLOPT_LN_SBPLX']
            best[24][4166.67] += ['CmaFmin2']
            best[24][416.667] += ['CmaFmin2']
            best[24][41.6667] += ['CmaFmin2']
            best[24][4.16667] += ['NLOPT_LN_SBPLX']
            best[24][0.416667] += ['Cobyla']
            best[24][41666.7] += ['NGOptF']
            best[24][4166.67] += ['CmaFmin2']
            best[24][416.667] += ['RotatedTwoPointsDE']
            best[24][41.6667] += ['CmaFmin2']
            best[24][4.16667] += ['pysot']
            best[24][0.416667] += ['NLOPT_GN_CRS2_LM']
            best[24][41666.7] += ['NGOptF']
            best[24][4166.67] += ['Carola1']
            best[24][416.667] += ['ChainDiagonalCMAPowell']
            best[24][41.6667] += ['NLOPT_LN_SBPLX']
            best[24][4.16667] += ['NLOPT_GN_DIRECT']
            best[24][0.416667] += ['Cobyla']
            best[2][500000] += ['NGOptF2']
            best[2][50000] += ['NeuralMetaModelDE']
            best[2][5000] += ['ASCMADEthird']
            best[2][500] += ['LQODE']
            best[2][50] += ['SODE']
            best[2][5] += ['Carola2']
            best[2][500000] += ['NGOptF']
            best[2][50000] += ['NGOptF']
            best[2][5000] += ['LQODE']
            best[2][500] += ['ChainDiagonalCMAPowell']
            best[2][50] += ['NLOPT_GN_DIRECT']
            best[2][5] += ['Cobyla']
            best[48][10.4167] += ['Carola2']
            best[48][1.04167] += ['ChainDiagonalCMAPowell']
            best[48][0.104167] += ['MetaModelQODE']
            best[48][10.4167] += ['NGOpt8']
            best[48][1.04167] += ['ChoiceBase']
            best[48][0.104167] += ['Carola1']
            best[48][104.167] += ['NGOptF']
            best[48][10.4167] += ['CMandAS3']
            best[48][1.04167] += ['ASCMADEthird']
            best[48][0.104167] += ['CMAtuning']
            bestdist: float = float('inf')
            bestalg: List[str] = []
            for d1 in best:
                for nod2 in best[d1]:
                    dist: float = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg
        if self.fully_continuous and (not self.has_noise):
            algs: List[str] = recommend_method(self.dimension, self.budget / self.dimension if self.budget else 0)
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA
            def most_frequent(List_: List[str]) -> str:
                return max(set(List_), key=List_.count)
            return registry[most_frequent(algs)]
        if self.fully_continuous and (not self.has_noise) and (self.budget is not None and self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()


class NGOpt(NGOpt39):
    pass


class Wiz(NGOpt16):
    def _select_optimizer_cls(self) -> Any:
        if self.fully_continuous and self.has_noise:
            DeterministicMix: Any = ConfPortfolio(optimizers=[DiagonalCMA, PSO, GeneticDE])
            return Chaining([DeterministicMix, OptimisticNoisyOnePlusOne], ['half'])
        cma_vars: int = max(1, 4 + int(3 * np.log(self.dimension)))
        num36: int = 1 + int(np.sqrt(4.0 * (4 * self.budget) // (self.dimension * 1000))) if self.budget is not None else 1
        num21: int = 1 + 4 * self.budget // (self.dimension * 1000) if self.budget is not None else 1
        num_dim10: int = 1 + int(np.sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000))) if self.budget is not None else 1
        num_dim20: int = self.budget // (500 * self.dimension) if self.budget is not None else 1
        para: int = 1
        if self.budget is not None and self.budget > 5000 * self.dimension:
            para = num36 * cma_vars
        elif self.dimension < 5:
            para = num21 * cma_vars
        elif self.dimension < 10:
            para = num_dim10 * cma_vars
        elif self.dimension < 20:
            para = num_dim20 * cma_vars
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1):
            return Carola2
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget > 10000 * self.dimension) and (not self.has_noise) and (self.dimension > 1):
            return Carola2
        if self.budget is not None and self.budget > 500 * self.dimension and self.fully_continuous and (not self.has_noise) and (self.num_objectives < 2) and (self.num_workers <= para) and p.helpers.Normalizer(self.parametrization).fully_bounded:
            if self.dimension == 1:
                return NGOpt16
            if self.budget > 5000 * self.dimension:
                return NGOpt36
            if self.dimension < 5:
                return NGOpt21
            if self.dimension < 10:
                num: int = 1 + int(np.sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)))
                return ConfPortfolio(optimizers=[NGOpt14] * num, warmup_ratio=0.7)
            if self.dimension < 20:
                num: int = self.budget // (500 * self.dimension)
                return ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3 ** i) for i in range(num)], warmup_ratio=0.5)
            if self.num_workers == 1:
                return CmaFmin2
            return NGOpt16
        elif self.budget is not None and self.fully_continuous and (not self.has_noise) and (self.num_objectives < 2) and (self.num_workers <= cma_vars) and (self.budget > 50 * self.dimension) and p.helpers.Normalizer(self.parametrization).fully_bounded:
            if self.dimension < 3:
                return NGOpt8
            if self.dimension <= 20 and self.num_workers == 1:
                MetaModelFmin2: Any = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            return NGOpt15
        else:
            return super()._select_optimizer_cls()


class NgIoh(NGOptBase):
    pass


class NGOpt4(NGOptBase):
    def _select_optimizer_cls(self) -> Any:
        self.fully_continuous = self.fully_continuous and (not self.has_discrete_not_softmax) and (self._arity < 0)
        budget: int = self.budget  # type: ignore
        num_workers: int = self.num_workers
        funcinfo: Any = self.parametrization.function
        assert budget is not None
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optimClass: Any = ParametrizedOnePlusOne(crossover=True, mutation='portfolio', noise_handling='optimistic')
        elif self._arity > 0:
            if self._arity == 2:
                optimClass = DiscreteOnePlusOne
            else:
                optimClass = AdaptiveDiscreteOnePlusOne if self._arity < 5 else CMandAS2
        elif self.has_noise and self.fully_continuous and (self.dimension > 100):
            optimClass = TBPSA
        elif self.num_workers > budget / 5:
            if self.num_workers > budget / 2.0 or budget < self.dimension:
                optimClass = MetaTuneRecentering
            elif self.dimension > 30:
                optimClass = OnePlusOne
            else:
                optimClass = Cobyla
        elif self.num_workers == 1 and budget > 6000 and (self.dimension > 7):
            optimClass = ChainCMAPowell
        elif self.num_workers == 1 and budget < self.dimension * 30:
            optimClass = OnePlusOne if self.dimension > 30 else Cobyla
        else:
            optimClass = DE if self.dimension > 2000 else MetaCMA if self.dimension > 1 else OnePlusOne
        return optimClass


class NGOpt8(NGOpt4):
    def _select_optimizer_cls(self) -> Any:
        if self.has_noise and (self.has_discrete_not_softmax or not self.parametrization.function.metrizable):
            if self.budget is not None and self.budget > 10000:
                optimClass = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
            else:
                optimClass = ParametrizedOnePlusOne(crossover=True, mutation='discrete', noise_handling='optimistic')
        elif self._arity > 0:
            if self.budget is not None and self.budget < 1000 and self.num_workers == 1:
                optimClass = DiscreteBSOOnePlusOne
            elif self.num_workers > 2:
                optimClass = CMandAS2
            else:
                optimClass = super()._select_optimizer_cls()
        elif not (self.has_noise and self.fully_continuous and (self.dimension > 100)) and (not (self.has_noise and self.fully_continuous)) and (not self.num_workers > self.budget / 5) and (self.num_workers == 1 and self.budget > 6000 and (self.dimension > 7)):
            optimClass = ChainMetaModelPowell
        elif self.fully_continuous and self.budget is not None and (self.budget < self.dimension * 30):
            if self.dimension > 30:
                optimClass = OnePlusOne
            elif self.dimension < 5:
                optimClass = MetaModel
            else:
                optimClass = Cobyla
        else:
            optimClass = super()._select_optimizer_cls()
        return optimClass

    def _num_objectives_set_callback(self) -> None:
        super()._num_objectives_set_callback()
        if self.num_objectives > 1:
            if self.noise_from_instrumentation or not self.has_noise:
                self._optim = DE(self.parametrization, self.budget, self.num_workers)


class NGOpt10(NGOpt8):
    def _select_optimizer_cls(self) -> Any:
        if not self.has_noise and self._arity > 0:
            return DiscreteLenglerOnePlusOne
        else:
            return super()._select_optimizer_cls()
    def recommend(self) -> Any:
        return base.Optimizer.recommend(self)


class NGOpt12(NGOpt10):
    def _select_optimizer_cls(self) -> Any:
        cma_vars: int = max(1, 4 + int(3 * np.log(self.dimension)))
        if (not self.has_noise and self.fully_continuous and (self.num_workers <= cma_vars) and (self.dimension < 100) and 
            self.budget is not None and (self.budget < self.dimension * 50) and (self.budget > min(50, self.dimension * 5))):
            return MetaModel
        elif (not self.has_noise and self.fully_continuous and (self.num_workers <= cma_vars) and (self.dimension < 100) and 
              self.budget is not None and (self.budget < self.dimension * 5) and (self.budget > 50)):
            return MetaModel
        else:
            return super()._select_optimizer_cls()


class NGOpt13(NGOpt12):
    def _select_optimizer_cls(self) -> Any:
        if self.budget is not None and self.num_workers * 3 < self.budget and self.dimension < 8 and self.budget < 80:
            return HyperOpt
        else:
            return super()._select_optimizer_cls()


class NGOpt14(NGOpt12):
    def _select_optimizer_cls(self) -> Any:
        if self.budget is not None and self.budget < 600:
            return MetaModel
        else:
            return super()._select_optimizer_cls()


@registry.register
class NGOpt15(NGOpt12):
    def _select_optimizer_cls(self) -> Any:
        if (self.budget is not None and self.fully_continuous and (self.budget < self.dimension ** 2 * 2) and 
            self.num_workers == 1 and (not self.has_noise) and (self.num_objectives < 2)):
            return MetaModelOnePlusOne
        elif self.fully_continuous and self.budget is not None and (self.budget < 600):
            return MetaModel
        else:
            return super()._select_optimizer_cls()


@registry.register
class NGOpt16(NGOpt15):
    def _select_optimizer_cls(self) -> Any:
        if (self.budget is not None and self.fully_continuous and (self.budget < 200 * self.dimension) and 
            self.num_workers == 1 and (not self.has_noise) and (self.num_objectives < 2) and 
            p.helpers.Normalizer(self.parametrization).fully_bounded):
            return Cobyla
        else:
            return super()._select_optimizer_cls()


class NGOpt21(NGOpt16):
    def _select_optimizer_cls(self) -> Any:
        cma_vars: int = max(1, 4 + int(3 * np.log(self.dimension)))
        num: int = 1 + 4 * self.budget // (self.dimension * 1000) if self.budget is not None else 1
        if (self.budget is not None and self.budget > 500 * self.dimension and self.fully_continuous and 
            (self.num_workers <= num * cma_vars)):
            return ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3 ** i) for i in range(num)], warmup_ratio=0.5)
        else:
            return super()._select_optimizer_cls()


class NGOpt36(NGOpt16):
    def _select_optimizer_cls(self) -> Any:
        num: int = 1 + int(np.sqrt(4.0 * (4 * self.budget) // (self.dimension * 1000))) if self.budget is not None else 1
        cma_vars: int = max(1, 4 + int(3 * np.log(self.dimension)))
        if (self.budget is not None and self.budget > 500 * self.dimension and self.fully_continuous and 
            (self.num_workers <= num * cma_vars)):
            return ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=0.9 ** i) for i in range(num)], warmup_ratio=0.5)
        else:
            return super()._select_optimizer_cls()


class NGOpt38(NGOpt16):
    def _select_optimizer_cls(self) -> Any:
        if (self.budget is not None and self.budget > 500 * self.dimension and self.fully_continuous and 
            (not self.has_noise) and (self.num_objectives < 2) and (self.num_workers == 1) and 
            p.helpers.Normalizer(self.parametrization).fully_bounded):
            if self.budget > 5000 * self.dimension:
                return NGOpt36
            if self.dimension < 5:
                return NGOpt21
            if self.dimension < 10:
                num: int = 1 + int(np.sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)))
                return ConfPortfolio(optimizers=[NGOpt14] * num, warmup_ratio=0.7)
            if self.dimension < 20:
                num: int = self.budget // (500 * self.dimension)
                return ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3 ** i) for i in range(num)], warmup_ratio=0.5)
            if self.num_workers == 1:
                return CmaFmin2
            return NGOpt16
        elif (self.budget is not None and self.fully_continuous and (not self.has_noise) and 
              (self.num_objectives < 2) and (self.num_workers == 1) and (self.budget > 50 * self.dimension) and 
              p.helpers.Normalizer(self.parametrization).fully_bounded):
            return NGOpt8 if self.dimension < 3 else NGOpt15
        else:
            return super()._select_optimizer_cls()


class NGOpt39(NGOpt16):
    def _select_optimizer_cls(self) -> Any:
        best: Dict[int, Dict[float, List[str]]] = defaultdict(lambda: defaultdict(list))
        def recommend_method(d: int, nod: float) -> List[str]:
            best[12][83333.3] += ['NGOptF3']
            best[12][8333.33] += ['DiscreteDE']
            best[12][833.333] += ['DiscreteDE']
            best[12][83.3333] += ['NGOptF']
            best[12][8.33333] += ['RotatedTwoPointsDE']
            best[12][0.833333] += ['NLOPT_LN_SBPLX']
            best[24][4166.67] += ['NLOPT_LN_SBPLX']
            best[24][41.6667] += ['ChainNaiveTBPSAPowell']
            best[24][4.16667] += ['NLOPT_GN_DIRECT']
            best[24][0.416667] += ['Cobyla']
            best[2][500000] += ['BAR']
            best[2][50000] += ['ASCMADEthird']
            best[2][5000] += ['ChainMetaModelPowell']
            best[2][500] += ['DiscreteDE']
            best[2][50] += ['RFMetaModelOnePlusOne']
            best[2][5] += ['Carola2']
            best[48][104.167] += ['NGOptF']
            best[48][10.4167] += ['NGOpt4']
            best[48][1.04167] += ['RPowell']
            best[48][0.104167] += ['ASCMADEthird']
            bestdist: float = float('inf')
            bestalg: List[str] = []
            for d1 in best:
                for nod2 in best[d1]:
                    dist: float = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg
        if self.fully_continuous and (not self.has_noise):
            algs: List[str] = recommend_method(self.dimension, self.budget / self.dimension if self.budget else 0)
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA
            def most_frequent(List_: List[str]) -> str:
                return max(set(List_), key=List_.count)
            return registry[most_frequent(algs)]
        elif self.fully_continuous and (not self.has_noise) and (self.budget is not None and self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()


@registry.register
class NGOptRW(NGOpt39):
    def _select_optimizer_cls(self) -> Any:
        if self.fully_continuous and (not self.has_noise) and (self.budget is not None and self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, super()._select_optimizer_cls()], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()


@registry.register
class NGOptF(NGOptF):
    pass


@registry.register
class NGOptF2(NGOptF2):
    pass


@registry.register
class NGOptF3(NGOptF3):
    pass


@registry.register
class NGOptF5(NGOptF5):
    def _select_optimizer_cls(self) -> Any:
        best: Dict[int, Dict[float, List[str]]] = defaultdict(lambda: defaultdict(list))
        def recommend_method(d: int, nod: float) -> List[str]:
            best[12][83333.3] += ['NGOptF5']
            best[12][8333.33] += ['NGOptF5']
            best[12][833.333] += ['NLOPT_GN_ISRES']
            best[12][83.3333] += ['SMAC3']
            best[12][8.33333] += ['NGOpt']
            best[12][0.833333] += ['NLOPT_LN_SBPLX']
            best[24][4166.67] += ['DiscreteDE']
            best[24][416.667] += ['RotatedTwoPointsDE']
            best[24][41.6667] += ['NLOPT_GN_DIRECT']
            best[24][4.16667] += ['NLOPT_GN_DIRECT']
            best[24][0.416667] += ['Cobyla']
            best[2][500000] += ['BAR']
            best[2][50000] += ['ChainMetaModelPowell']
            best[2][5000] += ['ASCMADEthird']
            best[2][500] += ['NeuralMetaModelTwoPointsDE']
            best[2][50] += ['CMandAS2']
            best[2][5] += ['NaiveTBPSA']
            best[48][10.4167] += ['Carola2']
            best[48][1.04167] += ['ChainDiagonalCMAPowell']
            best[48][0.104167] += ['ASCMADEthird']
            bestdist: float = float('inf')
            bestalg: List[str] = []
            for d1 in best:
                for nod2 in best[d1]:
                    dist: float = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg
        if self.fully_continuous and (not self.has_noise):
            algs: List[str] = recommend_method(self.dimension, self.budget / self.dimension if self.budget else 0)
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA
            def most_frequent(List_: List[str]) -> str:
                return max(set(List_), key=List_.count)
            return registry[most_frequent(algs)]
        elif self.fully_continuous and (not self.has_noise) and (self.budget is not None and self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()


@registry.register
class NGOpt(NGOpt):
    pass


@registry.register
class Wiz(NWiz := Wiz):  # Alias for exporting Wiz
    pass


@registry.register
class NgIoh(NgIoh := NgIoh):  
    pass


@registry.register
class NgIoh2(NgIoh2 := NgIoh2):
    pass


@registry.register
class NgIoh3(NgIoh3 := NgIoh3):
    pass


@registry.register
class NgIoh4(NgIoh4 := NgIoh4):
    pass


@registry.register
class NgIoh5(NgIoh5 := NgIoh5):
    pass


@registry.register
class NgIoh6(NgIoh6 := NgIoh6):
    pass


class _MSR(Portfolio):
    def __init__(self, parametrization: Any, budget: Optional[int] = None, num_workers: int = 1, num_single_runs: int = 9, base_optimizer: Type[NGOpt] = NGOpt) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[base_optimizer] * num_single_runs))
        self.coeffs: List[Any] = []
    def _internal_tell_candidate(self, candidate: Any, loss: float) -> None:
        if not self.coeffs:
            self.coeffs = [self.parametrization.random_state.uniform(size=self.num_objectives) for _ in self.optims]
        for coeffs, opt in zip(self.coeffs, self.optims):
            this_loss: float = float(sum(loss * coeff for loss, coeff in zip(loss if isinstance(loss, list) else [loss], coeffs)))
            opt.tell(candidate, this_loss)


class MultipleSingleRuns(base.ConfiguredOptimizer):
    def __init__(self, *, num_single_runs: int = 9, base_optimizer: Type[NGOpt] = NGOpt) -> None:
        super().__init__(_MSR, locals())

# Smooth variants of OnePlusOne with discrete and lognormal mutations.
SmoothDiscreteOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, mutation='discrete').set_name('SmoothDiscreteOnePlusOne', register=True)
SmoothPortfolioDiscreteOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, mutation='portfolio').set_name('SmoothPortfolioDiscreteOnePlusOne', register=True)
SmoothDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, mutation='lengler').set_name('SmoothDiscreteLenglerOnePlusOne', register=True)
SmoothDiscreteLognormalOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, mutation='lognormal').set_name('SmoothDiscreteLognormalOnePlusOne', register=True)
SuperSmoothDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, mutation='lengler', antismooth=9).set_name('SuperSmoothDiscreteLenglerOnePlusOne', register=True)
SuperSmoothTinyLognormalDiscreteOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, mutation='tinylognormal', antismooth=9).set_name('SuperSmoothTinyLognormalDiscreteOnePlusOne', register=True)
UltraSmoothDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, mutation='lengler', antismooth=3).set_name('UltraSmoothDiscreteLenglerOnePlusOne', register=True)
SmootherDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, mutation='lengler', antismooth=2, super_radii=True).set_name('SmootherDiscreteLenglerOnePlusOne', register=True)
YoSmoothDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, mutation='lengler', antismooth=2).set_name('YoSmoothDiscreteLenglerOnePlusOne', register=True)
CMALS: Any = Chaining([CMA, DiscreteLenglerOnePlusOne, UltraSmoothDiscreteLenglerOnePlusOne], ['third', 'third']).set_name('CMALS', register=True)
UltraSmoothDiscreteLognormalOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, mutation='lognormal', antismooth=3).set_name('UltraSmoothDiscreteLognormalOnePlusOne', register=True)
CMALYS: Any = Chaining([CMA, YoSmoothDiscreteLenglerOnePlusOne], ['tenth']).set_name('CMALYS', register=True)
CLengler: Any = Chaining([CMA, DiscreteLenglerOnePlusOne], ['tenth']).set_name('CLengler', register=True)
CMALL: Any = Chaining([CMA, DiscreteLenglerOnePlusOne, UltraSmoothDiscreteLognormalOnePlusOne], ['third', 'third']).set_name('CMALL', register=True)
CMAILL: Any = Chaining([ImageMetaModel, ImageMetaModelLengler, UltraSmoothDiscreteLognormalOnePlusOne], ['third', 'third']).set_name('CMAILL', register=True)
CMASL: Any = Chaining([CMA, SmootherDiscreteLenglerOnePlusOne], ['tenth']).set_name('CMASL', register=True)
CMASL2: Any = Chaining([CMA, SmootherDiscreteLenglerOnePlusOne], ['third']).set_name('CMASL2', register=True)
CMASL3: Any = Chaining([CMA, SmootherDiscreteLenglerOnePlusOne], ['half']).set_name('CMASL3', register=True)
CMAL2: Any = Chaining([CMA, SmootherDiscreteLenglerOnePlusOne], ['half']).set_name('CMAL2', register=True)
CMAL3: Any = Chaining([DiagonalCMA, SmootherDiscreteLenglerOnePlusOne], ['half']).set_name('CMAL3', register=True)
SmoothLognormalDiscreteOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, mutation='lognormal').set_name('SmoothLognormalDiscreteOnePlusOne', register=True)
SmoothAdaptiveDiscreteOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, mutation='adaptive').set_name('SmoothAdaptiveDiscreteOnePlusOne', register=True)
SmoothRecombiningPortfolioDiscreteOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='portfolio').set_name('SmoothRecombiningPortfolioDiscreteOnePlusOne', register=True)
SmoothRecombiningDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler').set_name('SmoothRecombiningDiscreteLenglerOnePlusOne', register=True)
UltraSmoothRecombiningDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', antismooth=3).set_name('UltraSmoothRecombiningDiscreteLenglerOnePlusOne', register=True)
UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lognormal', antismooth=3, roulette_size=7).set_name('UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne', register=True)
UltraSmoothElitistRecombiningDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', antismooth=3, roulette_size=7).set_name('UltraSmoothElitistRecombiningDiscreteLenglerOnePlusOne', register=True)
SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', antismooth=9, roulette_size=7).set_name('SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne', register=True)
SuperSmoothRecombiningDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', antismooth=9).set_name('SuperSmoothRecombiningDiscreteLenglerOnePlusOne', register=True)
SuperSmoothRecombiningDiscreteLognormalOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lognormal', antismooth=9).set_name('SuperSmoothRecombiningDiscreteLognormalOnePlusOne', register=True)
SmoothElitistRecombiningDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', roulette_size=7).set_name('SmoothElitistRecombiningDiscreteLenglerOnePlusOne', register=True)
SmoothElitistRandRecombiningDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', roulette_size=7, crossover_type='rand').set_name('SmoothElitistRandRecombiningDiscreteLenglerOnePlusOne', register=True)
SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne: Any = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lognormal', roulette_size=7, crossover_type='rand').set_name('SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne', register=True)
RecombiningDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(crossover=True, mutation='lengler').set_name('RecombiningDiscreteLenglerOnePlusOne', register=True)
RecombiningDiscreteLognormalOnePlusOne: Any = ParametrizedOnePlusOne(crossover=True, mutation='lognormal').set_name('RecombiningDiscreteLognormalOnePlusOne', register=True)
MaxRecombiningDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='max').set_name('MaxRecombiningDiscreteLenglerOnePlusOne', register=True)
MinRecombiningDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='min').set_name('MinRecombiningDiscreteLenglerOnePlusOne', register=True)
OnePtRecombiningDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='onepoint').set_name('OnePtRecombiningDiscreteLenglerOnePlusOne', register=True)
TwoPtRecombiningDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='twopoint').set_name('TwoPtRecombiningDiscreteLenglerOnePlusOne', register=True)
RandRecombiningDiscreteLenglerOnePlusOne: Any = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='rand').set_name('RandRecombiningDiscreteLenglerOnePlusOne', register=True)
RandRecombiningDiscreteLognormalOnePlusOne: Any = ParametrizedOnePlusOne(crossover=True, mutation='lognormal', crossover_type='rand').set_name('RandRecombiningDiscreteLognormalOnePlusOne', register=True)
MixDeterministicRL: Any = ConfPortfolio(optimizers=[OnePlusOne, DiagonalCMA, OpoDE], warmup_ratio=0.5).set_name('MixDeterministicRL', register=True)
SpecialRL: Any = Chaining([MixDeterministicRL, TBPSA], ['half']).set_name('SpecialRL', register=True)
NoisyRL1: Any = Chaining([MixDeterministicRL, NoisyOnePlusOne], ['half']).set_name('NoisyRL1', register=True)
NoisyRL2: Any = Chaining([MixDeterministicRL, RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne], ['half']).set_name('NoisyRL2', register=True)
NoisyRL3: Any = Chaining([MixDeterministicRL, OptimisticNoisyOnePlusOne], ['half']).set_name('NoisyRL3', register=True)
from . import experimentalvariants
FCarola6: Any = Chaining([NGOpt, NGOpt, RBFGS], ['tenth', 'most']).set_name('FCarola6', register=True)
FCarola6.no_parallelization = True
Carola11: Any = Chaining([MultiCMA, RBFGS], ['most']).set_name('Carola11', register=True)
Carola11.no_parallelization = True
Carola14: Any = Chaining([MultiCMA, RBFGS], ['most']).set_name('Carola14', register=True)
Carola14.no_parallelization = True
DS14: Any = Chaining([MultiDS, RBFGS], ['most']).set_name('DS14', register=True)
DS14.no_parallelization = True
Carola13: Any = Chaining([CmaFmin2, RBFGS], ['most']).set_name('Carola13', register=True)
Carola13.no_parallelization = True
Carola15: Any = Chaining([Cobyla, MetaModel, RBFGS], ['sqrt', 'most']).set_name('Carola15', register=True)
Carola15.no_parallelization = True

@registry.register
class cGA(base.Optimizer):
    """Compact Genetic Algorithm."""
    def __init__(self, parametrization: Any, budget: Optional[int] = None, num_workers: int = 1, arity: Optional[int] = None) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if arity is None:
            all_params = p.helpers.flatten(self.parametrization)
            arity = max((len(param.choices) if isinstance(param, p.TransitionChoice) else 500 for _, param in all_params))
        self._arity: int = arity
        self._penalize_cheap_violations: bool = False
        self.p: Any = np.ones((self.dimension, arity)) / arity
        self.llambda: int = max(num_workers, 40)
        self._previous_value_candidate: Optional[Any] = None
    def _internal_ask_candidate(self) -> Any:
        values: List[int] = [sum(self._rng.uniform() > cum_proba) for cum_proba in np.cumsum(self.p, axis=1)]
        data: Any = discretization.noisy_inverse_threshold_discretization(values, arity=self._arity, gen=self._rng)
        return self.parametrization.spawn_child().set_standardized_data(data)
    def _internal_tell_candidate(self, candidate: Any, loss: float) -> None:
        data: Any = candidate.get_standardized_data(reference=self.parametrization)
        if self._previous_value_candidate is None:
            self._previous_value_candidate = (loss, data)
        else:
            winner, loser = (self._previous_value_candidate[1], data)
            if self._previous_value_candidate[0] > loss:
                winner, loser = (loser, winner)
            winner_data = discretization.threshold_discretization(np.asarray(winner.data), arity=self._arity)
            loser_data = discretization.threshold_discretization(np.asarray(loser.data), arity=self._arity)
            for i in range(len(winner_data)):
                if winner_data[i] != loser_data[i]:
                    self.p[i][winner_data[i]] += 1.0 / self.llambda
                    self.p[i][loser_data[i]] -= 1.0 / self.llambda
                    for j in range(len(self.p[i])):
                        self.p[i][j] = max(self.p[i][j], 1.0 / self.llambda)
                    self.p[i] /= sum(self.p[i])
            self._previous_value_candidate = None


class _EMNA(base.Optimizer):
    def __init__(self, parametrization: Any, budget: Optional[int] = None, num_workers: int = 1, isotropic: bool = True, naive: bool = True, population_size_adaptation: bool = False, initial_popsize: Optional[int] = None) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.isotropic: bool = isotropic
        self.naive: bool = naive
        self.population_size_adaptation: bool = population_size_adaptation
        self.min_coef_parallel_context: int = 8
        if initial_popsize is None:
            initial_popsize = self.dimension
        if self.isotropic:
            self.sigma: Any = 1.0
        else:
            self.sigma = np.ones(self.dimension)
        self.popsize = _PopulationSizeController(llambda=4 * initial_popsize, mu=initial_popsize, dimension=self.dimension, num_workers=num_workers)
        if not self.population_size_adaptation:
            self.popsize.mu = max(16, self.dimension)
            self.popsize.llambda = 4 * self.popsize.mu
            self.popsize.llambda = max(self.popsize.llambda, num_workers)
            if budget is not None and self.popsize.llambda > budget:
                self.popsize.llambda = budget
                self.popsize.mu = self.popsize.llambda // 4
                warnings.warn('Budget may be too small in front of the dimension for EMNA', errors.InefficientSettingsWarning)
        self.current_center: Any = np.zeros(self.dimension)
        self.parents: List[Any] = [self.parametrization]
        self.children: List[Any] = []
    def recommend(self) -> Any:
        if self.naive:
            return self.current_bests['optimistic'].parameter
        else:
            out: Any = self.parametrization.spawn_child()
            with p.helpers.deterministic_sampling(out):
                out.set_standardized_data(self.current_center)
            return out
    def _internal_ask_candidate(self) -> Any:
        sigma_tmp = self.sigma
        if self.population_size_adaptation and self.popsize.llambda < self.min_coef_parallel_context * self.dimension:
            sigma_tmp = self.sigma * np.exp(self._rng.normal(0, 1) / np.sqrt(self.dimension))
        individual = self.current_center + sigma_tmp * self._rng.normal(0, 1, self.dimension)
        parent = self.parents[self.num_ask % len(self.parents)]
        candidate = parent.spawn_child().set_standardized_data(individual, reference=self.parametrization)
        if parent is self.parametrization:
            candidate.heritage['lineage'] = candidate.uid
        candidate._meta['sigma'] = sigma_tmp
        return candidate
    def _internal_tell_candidate(self, candidate: Any, loss: float) -> None:
        if self.population_size_adaptation:
            self.popsize.add_value(loss)
        self.children.append(candidate)
        if len(self.children) >= self.popsize.llambda:
            self.children.sort(key=base._loss)
            self.parents = self.children[:self.popsize.mu]
            self.children = []
            self.current_center = sum((c.get_standardized_data(reference=self.parametrization) for c in self.parents)) / self.popsize.mu
            if self.population_size_adaptation:
                if self.popsize.llambda < self.min_coef_parallel_context * self.dimension:
                    self.sigma = np.exp(np.sum(np.log([c._meta['sigma'] for c in self.parents]), axis=0 if self.isotropic else None) / self.popsize.mu)
                else:
                    stdd = [(self.parents[i].get_standardized_data(reference=self.parametrization) - self.current_center) ** 2 for i in range(self.popsize.mu)]
                    self.sigma = np.sqrt(np.sum(stdd) / (self.popsize.mu * (self.dimension if self.isotropic else 1)))
            else:
                stdd = [(self.parents[i].get_standardized_data(reference=self.parametrization) - self.current_center) ** 2 for i in range(self.popsize.mu)]
                self.sigma = np.sqrt(np.sum(stdd, axis=0 if self.isotropic else None) / (self.popsize.mu * (self.dimension if self.isotropic else 1)))
            if self.num_workers / self.dimension > 32:
                imp: int = max(1, (np.log(self.popsize.llambda) / 2) ** (1 / self.dimension))
                self.sigma /= imp
    def _internal_tell_not_asked(self, candidate: Any, loss: float) -> None:
        raise errors.TellNotAskedNotSupportedError

class EMNA(base.ConfiguredOptimizer):
    def __init__(self, *, isotropic: bool = True, naive: bool = True, population_size_adaptation: bool = False, initial_popsize: Optional[int] = None) -> None:
        super().__init__(_EMNA, locals())

NaiveIsoEMNA: EMNA = EMNA().set_name('NaiveIsoEMNA', register=True)


@registry.register
class NGOptBase(base.Optimizer):
    def __init__(self, parametrization: Any, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        analysis: Any = p.helpers.analyze(self.parametrization)
        funcinfo: Any = self.parametrization.function
        self.has_noise: bool = not (analysis.deterministic and funcinfo.deterministic)
        self.has_real_noise: bool = not funcinfo.deterministic
        self.noise_from_instrumentation: bool = self.has_noise and funcinfo.deterministic
        self.fully_continuous: bool = analysis.continuous
        all_params = p.helpers.flatten(self.parametrization)
        int_layers: List[Any] = list(itertools.chain.from_iterable([_layering.Int.filter_from(x) for _, x in all_params]))
        int_layers = [x for x in int_layers if x.arity is not None]
        self.has_discrete_not_softmax: bool = any((not isinstance(lay, _datalayers.SoftmaxSampling) for lay in int_layers))
        self._has_discrete: bool = bool(int_layers)
        self._arity: int = max((lay.arity for lay in int_layers), default=-1)
        if self.fully_continuous:
            self._arity = -1
        self._optim: Optional[Any] = None
        self._constraints_manager.update(max_trials=1000, penalty_factor=1.0, penalty_exponent=1.01)
    @property
    def optim(self) -> Any:
        if self._optim is None:
            self._optim = self._select_optimizer_cls()(self.parametrization, self.budget, self.num_workers)
            self._optim = self._optim if not isinstance(self._optim, NGOptBase) else self._optim.optim
            logger.debug('%s selected %s optimizer.', *(x.name for x in (self, self._optim)))
        return self._optim
    def _select_optimizer_cls(self) -> Any:
        return CMA  # default return; override in subclasses
    def _internal_ask_candidate(self) -> Any:
        return self.optim.ask()
    def _internal_tell_candidate(self, candidate: Any, loss: float) -> None:
        self.optim.tell(candidate, loss)
    def recommend(self) -> Any:
        return self.optim.recommend()
    def _internal_tell_not_asked(self, candidate: Any, loss: float) -> None:
        self.optim.tell(candidate, loss)
    def _info(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {'sub-optim': self.optim.name}
        out.update(self.optim._info())
        return out
    def enable_pickling(self) -> None:
        self.optim.enable_pickling()


@registry.register
class NGOptDSBase(NGOptBase):
    def _select_optimizer_cls(self) -> Any:
        assert self.budget is not None
        if self.has_noise and self.has_discrete_not_softmax:
            return DoubleFastGADiscreteOnePlusOne if self.dimension < 60 else CMA
        elif self.has_noise and self.fully_continuous:
            return Chaining([SQOPSO, OptimisticDiscreteOnePlusOne], ['half'])
        elif self.has_discrete_not_softmax or not self.parametrization.function.metrizable or (not self.fully_continuous):
            return DoubleFastGADiscreteOnePlusOne
        elif self.num_workers > self.budget / 5:
            if self.num_workers > self.budget / 2.0 or self.budget < self.dimension:
                return MetaRecentering
            else:
                return NaiveTBPSA
        elif self.num_workers == 1 and self.budget > 6000 and (self.dimension > 7):
            return ChainDSPowell
        elif self.num_workers == 1 and self.budget < self.dimension * 30:
            return OnePlusOne if self.dimension > 30 else Cobyla
        else:
            return DE if self.dimension > 2000 else MetaCMA if self.dimension > 1 else OnePlusOne


@registry.register
class Shiwa(NGOptBase):
    def _select_optimizer_cls(self) -> Any:
        optCls: Any = NGOptBase
        funcinfo: Any = self.parametrization.function
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            optCls = CMA
        return optCls


@registry.register
class NGO(NGOptBase):
    pass


@registry.register
class NGOpt8(NGOpt8):
    pass


@registry.register
class NGOpt10(NGOpt10):
    pass


@registry.register
class NGOpt12(NGOpt12):
    pass


@registry.register
class NGOpt13(NGOpt13):
    pass


@registry.register
class NGOpt14(NGOpt14):
    pass


@registry.register
class NGOpt15(NGOpt15):
    pass


@registry.register
class NGOpt16(NGOpt16):
    pass


@registry.register
class NGOpt21(NGOpt21):
    pass


@registry.register
class NGOpt36(NGOpt36):
    pass


@registry.register
class NGOpt38(NGOpt38):
    pass


@registry.register
class NGOpt39(NGOpt39):
    pass


@registry.register
class NGOptRW(NGOptRW):
    pass


@registry.register
class NGOptF2(NGOptF2):
    pass


@registry.register
class NGOptF3(NGOptF3):
    pass


@registry.register
class NGOptF5(NGOptF5):
    pass


@registry.register
class NGOpt(NGOpt):
    pass


@registry.register
class Wiz(Wiz):
    pass


@registry.register
class NgIoh(NgIoh):
    pass


@registry.register
class NgIoh2(NgIoh2):
    pass


@registry.register
class NgIoh3(NgIoh3):
    pass


@registry.register
class NgIoh4(NgIoh4):
    pass


@registry.register
class NgIoh5(NgIoh5):
    pass


@registry.register
class NgIoh6(NgIoh6):
    pass


@registry.register
class NgIoh7(NgIoh7):
    def _select_optimizer_cls(self) -> Any:
        optCls: Any = NGOptBase
        funcinfo: Any = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and (not self.has_noise):
            if self.budget < 1000 * self.dimension and self.budget > 20 * self.dimension and self.dimension > 1 and self.dimension < 100:
                return Carola2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            optCls = CMA
        return optCls


@registry.register
class NgIoh8(NgIoh8):
    pass


@registry.register
class NgIoh9(NgIoh9):
    pass


@registry.register
class NgIoh10(NgIoh10):
    pass


@registry.register
class NgIoh11(NgIoh11):
    def _select_optimizer_cls(self, budget: Optional[int] = None) -> Any:
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and (not self.has_noise):
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return Carola14
                if self.dimension < 4:
                    return Carola4
                if self.dimension < 8:
                    return Carola5
                if self.dimension < 15:
                    return Carola9
                if self.dimension < 30:
                    return Carola8
                if self.dimension < 60:
                    return Carola9
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2: Any = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and (self.dimension > 30):
                return ChainMetaModelSQP
            if self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and self.dimension == 2:
                return NLOPT_LN_SBPLX
            if self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and self.dimension < 15:
                return ChainMetaModelSQP
            if self.budget >= 300 * self.dimension and self.budget < 3000 * self.dimension and self.dimension < 30:
                return MultiCMA
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and self.dimension > 1 and self.dimension < 100:
            return Carola2
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and self.dimension > 1 and self.dimension < 50:
            return Carola2
        if self.has_noise and (self.has_discrete_not_softmax or not self.parametrization.function.metrizable):
            return RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return NGOptBase._select_optimizer_cls(self)


@registry.register
class NgIoh12(NgIoh12):
    pass


@registry.register
class NgIoh13(NgIoh13):
    pass


@registry.register
class NgIoh14(NgIoh14):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.no_crossing: bool = True


@registry.register
class NgIoh15(NgIoh15):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.no_crossing = True


@registry.register
class NgIoh16(NgIoh16):
    pass


@registry.register
class NgIoh17(NgIoh17):
    pass


@registry.register
class NgDS(NgDS11):
    pass


@registry.register
class NgIoh21(NgIoh21):
    pass


@registry.register
class NgIohTuned(CSEC11):
    pass

# End of annotated code.
