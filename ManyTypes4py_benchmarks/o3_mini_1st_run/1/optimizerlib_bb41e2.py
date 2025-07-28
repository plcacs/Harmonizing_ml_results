from typing import Any, Dict, List, Optional, Union
from collections import defaultdict
from . import experimentalvariants

class FCarola6(Any):  # Placeholder for external type
    pass

class Carola2(Any):  # Placeholder for external type
    pass

class Carola4(Any):
    pass

class Carola5(Any):
    pass

class Carola6(Any):
    pass

class Carola8(Any):
    pass

class Carola9(Any):
    pass

class Carola11(Any):
    pass

class Carola13(Any):
    pass

class Carola14(Any):
    pass

class Carola15(Any):
    pass

class ChainDiagonalCMAPowell(Any):
    pass

class ChainMetaModelPowell(Any):
    pass

class ChainMetaModelSQP(Any):
    pass

class ChainMetaModelDSSQP(Any):
    pass

class CmaFmin2(Any):
    pass

class CMA(Any):
    pass

class DE(Any):
    pass

class DiagonalCMA(Any):
    pass

class DSproba(Any):
    pass

class FCMA(Any):
    pass

class GeneticDE(Any):
    pass

class HammersleySearchPlusMiddlePoint(Any):
    pass

class LHSSearch(Any):
    pass

class LhsDE(Any):
    pass

class MetaModel(Any):
    pass

class MetaModelQODE(Any):
    pass

class MultiCMA(Any):
    pass

class MultiDS(Any):
    pass

class MultiSQP(Any):
    pass

class NGOptBase:
    def __init__(self, parametrization: Any, budget: Optional[int] = None, num_workers: int = 1) -> None:
        # ... original __init__ code ...
        pass

    def _select_optimizer_cls(self) -> Any:
        # ... original _select_optimizer_cls code ...
        pass

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

class NGOptDSBase(NGOptBase):
    def _select_optimizer_cls(self) -> Any:
        assert self.budget is not None
        if self.has_noise and self.has_discrete_not_softmax:
            cls: Any = DoubleFastGADiscreteOnePlusOne if self.dimension < 60 else CMA
        elif self.has_noise and self.fully_continuous:
            cls = TBPSA
        elif self.has_discrete_not_softmax or not self.parametrization.function.metrizable or (not self.fully_continuous):
            cls = DoubleFastGADiscreteOnePlusOne
        elif self.num_workers > self.budget / 5:
            if self.num_workers > self.budget / 2.0 or self.budget < self.dimension:
                cls = MetaTuneRecentering
            else:
                cls = NaiveTBPSA
        elif self.num_workers == 1 and self.budget > 6000 and (self.dimension > 7):
            cls = ChainCMAPowell
        elif self.num_workers == 1 and self.budget < self.dimension * 30:
            cls = OnePlusOne if self.dimension > 30 else Cobyla
        else:
            cls = DE if self.dimension > 2000 else MetaCMA if self.dimension > 1 else OnePlusOne
        return cls

class Shiwa(NGOptBase):
    def _select_optimizer_cls(self) -> Any:
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            optCls = CMA
        return optCls

class NGO(NGOptBase):
    pass

class NGOpt4(NGOptBase):
    def _select_optimizer_cls(self) -> Any:
        if self.has_noise and (self.has_discrete_not_softmax or not self.parametrization.function.metrizable):
            mutation: str = 'portfolio' if self.budget > 1000 else 'discrete'
            optimClass: Any = ParametrizedOnePlusOne(crossover=True, mutation=mutation, noise_handling='optimistic')
        elif self._arity > 0:
            if self._arity == 2:
                optimClass = DiscreteOnePlusOne
            else:
                optimClass = AdaptiveDiscreteOnePlusOne if self._arity < 5 else CMandAS2
        elif self.has_noise and self.fully_continuous and (self.dimension > 100):
            optimClass = ConfSplitOptimizer(num_optims=13, progressive=True, multivariate_optimizer=OptimisticDiscreteOnePlusOne)
        elif self.has_noise and self.fully_continuous:
            if self.budget > 100:
                optimClass = OnePlusOne if self.noise_from_instrumentation or self.num_workers > 1 else SQP
            else:
                optimClass = OnePlusOne
        elif self.has_discrete_not_softmax or not self.parametrization.function.metrizable or (not self.fully_continuous):
            optimClass = DoubleFastGADiscreteOnePlusOne
        elif self.num_workers > self.budget / 5:
            if self.num_workers > self.budget / 2.0 or self.budget < self.dimension:
                optimClass = MetaTuneRecentering
            elif self.dimension < 5 and self.budget < 100:
                optimClass = DiagonalCMA
            elif self.dimension < 5 and self.budget < 500:
                optimClass = Chaining([DiagonalCMA, MetaModel], [100])
            else:
                optimClass = NaiveTBPSA
        elif self.num_workers == 1 and self.budget > 6000 and (self.dimension > 7):
            optimClass = ChainNaiveTBPSACMAPowell
        elif self.num_workers == 1 and self.budget < self.dimension * 30:
            if self.dimension > 30:
                optimClass = OnePlusOne
            elif self.dimension < 5:
                optimClass = MetaModel
            else:
                optimClass = Cobyla
        else:
            optimClass = DE if self.dimension > 2000 else MetaCMA if self.dimension > 1 else OnePlusOne
        return optimClass

class NGOpt8(NGOpt4):
    def _select_optimizer_cls(self) -> Any:
        if self.has_noise and (self.has_discrete_not_softmax or not self.parametrization.function.metrizable):
            if self.budget > 10000:
                optimClass = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
            else:
                optimClass = ParametrizedOnePlusOne(crossover=True, mutation='discrete', noise_handling='optimistic')
        elif self._arity > 0:
            if self.budget < 1000 and self.num_workers == 1:
                optimClass = DiscreteBSOOnePlusOne
            elif self.num_workers > 2:
                optimClass = CMandAS2
            else:
                optimClass = super()._select_optimizer_cls()
        elif not (self.has_noise and self.fully_continuous and (self.dimension > 100)) and (not (self.has_noise and self.fully_continuous)) and (not self.num_workers > self.budget / 5) and (self.num_workers == 1 and self.budget > 6000 and (self.dimension > 7)) and (self.num_workers < self.budget):
            optimClass = ChainMetaModelPowell
        elif self.fully_continuous and (not self.has_noise) and (self.num_objectives < 2) and (self.num_workers == 1) and (self.budget > 50 * self.dimension) and self.parametrization.function.metrizable:
            return NGOpt8 if self.dimension < 3 else NGOpt15
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
        cma_vars: int = max(1, 4 + int(3 * __import__("math").log(self.dimension)))
        if not self.has_noise and self.fully_continuous and (self.num_workers <= cma_vars) and (self.dimension < 100) and (self.budget is not None) and (self.budget < self.dimension * 50) and (self.budget > min(50, self.dimension * 5)):
            return MetaModel
        elif not self.has_noise and self.fully_continuous and (self.num_workers <= cma_vars) and (self.dimension < 100) and (self.budget is not None) and (self.budget < self.dimension * 5) and (self.budget > 50):
            return MetaModel
        else:
            return super()._select_optimizer_cls()
        
class NGOpt13(NGOpt12):
    def _select_optimizer_cls(self) -> Any:
        if self.budget is not None and (self.num_workers * 3 < self.budget) and (self.dimension < 8) and (self.budget < 80):
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
        if self.budget is not None and self.fully_continuous and (self.budget < self.dimension ** 2 * 2) and (self.num_workers == 1) and (not self.has_noise) and (self.num_objectives < 2):
            return MetaModelOnePlusOne
        elif self.fully_continuous and self.budget is not None and (self.budget < 600):
            return MetaModel
        else:
            return super()._select_optimizer_cls()

@registry.register
class NGOpt16(NGOpt15):
    def _select_optimizer_cls(self) -> Any:
        if self.budget is not None and self.fully_continuous and (self.budget < 200 * self.dimension) and (self.num_workers == 1) and (not self.has_noise) and (self.num_objectives < 2) and __import__("p.helpers").Normalizer(self.parametrization).fully_bounded:
            return Cobyla
        else:
            return super()._select_optimizer_cls()

class NGOpt21(NGOpt16):
    def _select_optimizer_cls(self) -> Any:
        cma_vars = max(1, 4 + int(3 * __import__("math").log(self.dimension)))
        num: int = 1 + 4 * self.budget // (self.dimension * 1000) if self.budget is not None else 1
        if self.budget is not None and self.budget > 500 * self.dimension and self.fully_continuous and (self.num_workers <= num * cma_vars):
            return ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3 ** i) for i in range(num)], warmup_ratio=0.5)
        else:
            return super()._select_optimizer_cls()

class NGOpt36(NGOpt16):
    def _select_optimizer_cls(self) -> Any:
        num: int = 1 + int(__import__("numpy").sqrt(4.0 * (4 * self.budget) // (self.dimension * 1000)) ) if self.budget is not None else 1
        cma_vars = max(1, 4 + int(3 * __import__("math").log(self.dimension)))
        if self.budget is not None and self.budget > 500 * self.dimension and self.fully_continuous and (self.num_workers <= num * cma_vars):
            return ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=0.9 ** i) for i in range(num)], warmup_ratio=0.5)
        else:
            return super()._select_optimizer_cls()

class NGOpt38(NGOpt16):
    def _select_optimizer_cls(self) -> Any:
        if self.budget is not None and self.budget > 500 * self.dimension and self.fully_continuous and (not self.has_noise) and (self.num_objectives < 2) and (self.num_workers == 1) and __import__("p.helpers").Normalizer(self.parametrization).fully_bounded:
            if self.budget > 5000 * self.dimension:
                return NGOpt36
            if self.dimension < 5:
                return NGOpt21
            if self.dimension < 10:
                num = 1 + int(__import__("numpy").sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)) )
                return ConfPortfolio(optimizers=[NGOpt14] * num, warmup_ratio=0.7)
            if self.dimension < 20:
                num = self.budget // (500 * self.dimension)
                return ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3 ** i) for i in range(num)], warmup_ratio=0.5)
            return NGOpt16
        elif self.budget is not None and self.fully_continuous and (not self.has_noise) and (self.num_objectives < 2) and (self.num_workers == 1) and (self.budget > 50 * self.dimension) and __import__("p.helpers").Normalizer(self.parametrization).fully_bounded:
            return NGOpt8 if self.dimension < 3 else NGOpt15
        else:
            return super()._select_optimizer_cls()

class NGOpt39(NGOpt16):
    def _select_optimizer_cls(self) -> Any:
        best: Dict[Any, Dict[Any, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        def recommend_method(d: float, nod: float) -> List[str]:
            best[12][833.333] += ['MemeticDE']
            best[12][83.3333] += ['NGOptRW']
            best[12][8.33333] += ['RealSpacePSO']
            best[12][0.833333] += ['ASCMADEthird']
            best[12][8333.33] += ['GeneticDE']
            best[12][833.333] += ['TripleCMA']
            best[12][83.3333] += ['NLOPT_LN_SBPLX']
            best[12][8.33333] += ['NLOPT_LN_SBPLX']
            best[12][0.833333] += ['NLOPT_LN_SBPLX']
            best[12][8333.33] += ['GeneticDE']
            best[12][833.333] += ['VLPCMA']
            best[12][83.3333] += ['MemeticDE']
            best[12][8.33333] += ['SMAC3']
            best[12][0.833333] += ['Cobyla']
            best[24][416.667] += ['VLPCMA']
            best[24][41.6667] += ['Wiz']
            best[24][4.16667] += ['NLOPT_LN_SBPLX']
            best[24][0.416667] += ['Cobyla']
            best[24][4166.67] += ['NGOptF']
            best[24][416.667] += ['NLOPT_LN_SBPLX']
            best[24][41.6667] += ['ChainNaiveTBPSAPowell']
            best[24][4.16667] += ['NLOPT_LN_SBPLX']
            best[24][0.416667] += ['NLOPT_LN_NELDERMEAD']
            best[24][4166.67] += ['Carola1']
            best[24][416.667] += ['ChainDiagonalCMAPowell']
            best[24][41.6667] += ['NLOPT_LN_SBPLX']
            best[24][4.16667] += ['NLOPT_GN_CRS2_LM']
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
            best[48][20.8333] += ['RLSOnePlusOne']
            best[48][2.08333] += ['NGOptF']
            best[48][0.208333] += ['NGOptF2']
            best[48][208.333] += ['MetaModelQODE']
            best[48][20.8333] += ['DiscreteLengler2OnePlusOne']
            best[48][2.08333] += ['NGOptF']
            best[48][0.208333] += ['NGOptF2']
            best[48][2083.33] += ['NGOptF2']
            best[48][208.333] += ['ChainNaiveTBPSAPowell']
            best[48][20.8333] += ['ChainNaiveTBPSAPowell']
            best[48][2.08333] += ['NLOPT_LN_NELDERMEAD']
            best[48][0.208333] += ['BOBYQA']
            best[5][200000] += ['NGOptF']
            best[5][20000] += ['MemeticDE']
            best[5][2000] += ['DiscreteDE']
            best[5][200] += ['TwoPointsDE']
            best[5][20] += ['NGOptF2']
            best[5][2] += ['MultiSQP']
            best[5][200000] += ['LhsDE']
            best[5][20000] += ['VLPCMA']
            best[5][2000] += ['BayesOptimBO']
            best[5][200] += ['SMAC3']
            best[5][20] += ['NLOPT_LN_SBPLX']
            best[5][2] += ['NLOPT_LN_SBPLX']
            best[96][10.4167] += ['Carola2']
            best[96][1.04167] += ['ChainDiagonalCMAPowell']
            best[96][0.104167] += ['MetaModelQODE']
            best[96][10.4167] += ['NGOpt8']
            best[96][1.04167] += ['ChoiceBase']
            best[96][0.104167] += ['Carola1']
            best[96][104.167] += ['NGOptF']
            best[96][10.4167] += ['CMandAS3']
            best[96][1.04167] += ['ASCMADEthird']
            best[96][0.104167] += ['CMAtuning']
            bestdist: float = float('inf')
            for d1 in best:
                for nod2 in best[d1]:
                    dist: float = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg: List[str] = best[d1][nod2]
            return bestalg
        if self.fully_continuous and (not self.has_noise):
            algs: List[str] = recommend_method(float(self.dimension), self.budget / self.dimension if self.budget is not None else 0)
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA
            def most_frequent(List: List[str]) -> str:
                return max(set(List), key=List.count)
            return registry[most_frequent(algs)]
        if self.fully_continuous and (not self.has_noise) and (self.budget is not None and self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()
        
class NGOptRW(NGOpt39):
    def _select_optimizer_cls(self) -> Any:
        if self.fully_continuous and (not self.has_noise) and (self.budget is not None and self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, super()._select_optimizer_cls()], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()

class NGOptF(NGOpt39):
    def _select_optimizer_cls(self) -> Any:
        best: Dict[Any, Dict[Any, List[str]]] = defaultdict(lambda: defaultdict(list))
        def recommend_method(d: float, nod: float) -> List[str]:
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
            for d1 in best:
                for nod2 in best[d1]:
                    dist: float = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg: List[str] = best[d1][nod2]
            return bestalg
        if self.fully_continuous and (not self.has_noise):
            algs: List[str] = recommend_method(float(self.dimension), self.budget / self.dimension if self.budget is not None else 0)
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA
            def most_frequent(List: List[str]) -> str:
                return max(set(List), key=List.count)
            return ConfPortfolio(optimizers=[registry[most_frequent(algs)]], warmup_ratio=0.6)
        if self.fully_continuous and (not self.has_noise) and (self.budget is not None and self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()
            
class NGOptF2(NGOpt39):
    def _select_optimizer_cls(self) -> Any:
        best: Dict[Any, Dict[Any, List[str]]] = defaultdict(lambda: defaultdict(list))
        def recommend_method(d: float, nod: float) -> List[str]:
            best[12][8333.33] += ['DiscreteDE']
            best[12][833.333] += ['DiscreteDE']
            best[12][83.3333] += ['MetaModelDE']
            best[12][8.33333] += ['NGOpt']
            best[12][0.833333] += ['NLOPT_GN_DIRECT_L']
            best[12][83333.3] += ['RPowell']
            best[12][8333.33] += ['DiscreteDE']
            best[12][833.333] += ['NLOPT_GN_ISRES']
            best[12][83.3333] += ['SMAC3']
            best[12][8.33333] += ['NGOpt']
            best[12][0.833333] += ['NLOPT_LN_SBPLX']
            best[12][83333.3] += ['NGOptBase']
            best[12][8333.33] += ['RPowell']
            best[12][833.333] += ['ChainMetaModelPowell']
            best[12][83.3333] += ['RotatedTwoPointsDE']
            best[12][8.33333] += ['NGOpt39']
            best[12][0.833333] += ['NLOPT_LN_NEWUOA_BOUND']
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
            best[24][0.416667] += ['NLOPT_LN_NELDERMEAD']
            best[24][41666.7] += ['NGOptF']
            best[24][4166.67] += ['Shiwa']
            best[24][416.667] += ['CmaFmin2']
            best[24][41.6667] += ['NLOPT_GN_CRS2_LM']
            best[24][4.16667] += ['NLOPT_GN_CRS2_LM']
            best[24][0.416667] += ['Cobyla']
            best[2][500000] += ['BAR']
            best[2][50000] += ['ChainMetaModelSQP']
            best[2][5000] += ['CM']
            best[2][500] += ['ChainDiagonalCMAPowell']
            best[2][50] += ['Powell']
            best[2][5] += ['Cobyla']
            best[48][208.333] += ['DiscreteDE']
            best[48][20.8333] += ['pysot']
            best[48][2.08333] += ['MultiCobyla']
            best[48][0.208333] += ['NLOPT_LN_NELDERMEAD']
            best[48][2083.33] += ['RPowell']
            best[48][208.333] += ['DiscreteDE']
            best[48][20.8333] += ['ChainNaiveTBPSACMAPowell']
            best[48][2.08333] += ['NLOPT_LN_BOBYQA']
            best[48][0.208333] += ['BOBYQA']
            best[48][20833.3] += ['MetaModelTwoPointsDE']
            best[48][2083.33] += ['RPowell']
            best[48][208.333] += ['DiscreteDE']
            best[48][20.8333] += ['ChainNaiveTBPSACMAPowell']
            best[48][2.08333] += ['NLOPT_LN_SBPLX']
            best[48][0.208333] += ['BOBYQA']
            bestdist: float = float('inf')
            for d1 in best:
                for nod2 in best[d1]:
                    dist: float = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg: List[str] = best[d1][nod2]
            return bestalg
        if self.fully_continuous and (not self.has_noise):
            algs: List[str] = recommend_method(float(self.dimension), self.budget / self.dimension if self.budget is not None else 0)
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA
            def most_frequent(List: List[str]) -> str:
                return max(set(List), key=List.count)
            return registry[most_frequent(algs)]
        if self.fully_continuous and (not self.has_noise) and (self.budget is not None and self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()
            
class NGOptF3(NGOpt39):
    def _select_optimizer_cls(self) -> Any:
        best: Dict[Any, Dict[Any, List[str]]] = defaultdict(lambda: defaultdict(list))
        def recommend_method(d: float, nod: float) -> List[str]:
            best[12][8333.33] += ['MemeticDE']
            best[12][83.3333] += ['NGOptRW']
            best[12][8.33333] += ['RealSpacePSO']
            best[12][0.833333] += ['ASCMADEthird']
            best[12][83333.3] += ['GeneticDE']
            best[12][8333.33] += ['TripleCMA']
            best[12][83.3333] += ['NLOPT_LN_SBPLX']
            best[12][8.33333] += ['NLOPT_LN_SBPLX']
            best[12][0.833333] += ['NLOPT_LN_SBPLX']
            best[12][83333.3] += ['GeneticDE']
            best[12][8333.33] += ['VLPCMA']
            best[12][83.3333] += ['MemeticDE']
            best[12][8.33333] += ['SMAC3']
            best[12][0.833333] += ['Cobyla']
            best[24][4166.67] += ['VLPCMA']
            best[24][41.6667] += ['Wiz']
            best[24][4.16667] += ['NLOPT_LN_SBPLX']
            best[24][0.416667] += ['Cobyla']
            best[24][41666.7] += ['NGOptF']
            best[24][4166.67] += ['ChainMetaModelSQP']
            best[24][416.667] += ['NGOptF']
            best[24][41.6667] += ['NLOPT_LN_SBPLX']
            best[24][4.16667] += ['NLOPT_GN_CRS2_LM']
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
            best[2][50] += ['Powell']
            best[2][5] += ['Cobyla']
            best[48][208.333] += ['DiscreteDE']
            best[48][20.8333] += ['pysot']
            best[48][2.08333] += ['MultiCobyla']
            best[48][0.208333] += ['NLOPT_LN_NELDERMEAD']
            best[48][2083.33] += ['RPowell']
            best[48][208.333] += ['DiscreteDE']
            best[48][20.8333] += ['ChainNaiveTBPSACMAPowell']
            best[48][2.08333] += ['NLOPT_LN_BOBYQA']
            best[48][0.208333] += ['BOBYQA']
            best[48][20833.3] += ['MetaModelTwoPointsDE']
            best[48][2083.33] += ['RPowell']
            best[48][208.333] += ['DiscreteDE']
            best[48][20.8333] += ['ChainNaiveTBPSACMAPowell']
            best[48][2.08333] += ['NLOPT_LN_SBPLX']
            best[48][0.208333] += ['BOBYQA']
            bestdist: float = float('inf')
            for d1 in best:
                for nod2 in best[d1]:
                    dist: float = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg: List[str] = best[d1][nod2]
            return bestalg
        if self.fully_continuous and (not self.has_noise):
            algs: List[str] = recommend_method(float(self.dimension), self.budget / self.dimension if self.budget is not None else 0)
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA
            def most_frequent(List: List[str]) -> str:
                return max(set(List), key=List.count)
            return registry[most_frequent(algs)]
        if self.fully_continuous and (not self.has_noise) and (self.budget is not None and self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()
            
@registry.register
class NGOpt(NGOpt39):
    pass

@registry.register
class Wiz(NGOpt16):
    def _select_optimizer_cls(self) -> Any:
        if self.fully_continuous and self.has_noise:
            DeterministicMix = ConfPortfolio(optimizers=[DiagonalCMA, PSO, GeneticDE])
            return Chaining([DeterministicMix, OptimisticNoisyOnePlusOne], ['half'])
        cma_vars = max(1, 4 + int(__import__("math").log(self.dimension)))
        num36: int = 1 + int(__import__("numpy").sqrt(4.0 * (4 * self.budget) // (self.dimension * 1000)) ) if self.budget is not None else 1
        num21: int = 1 + 4 * self.budget // (self.dimension * 1000) if self.budget is not None else 1
        num_dim10: int = 1 + int(__import__("numpy").sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)) ) if self.budget is not None else 1
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
        if self.budget is not None and self.budget > 500 * self.dimension and self.fully_continuous and (not self.has_noise) and (self.num_objectives < 2) and (self.num_workers <= para) and __import__("p.helpers").Normalizer(self.parametrization).fully_bounded:
            if self.dimension == 1:
                return NGOpt16
            if self.budget > 5000 * self.dimension:
                return NGOpt36
            if self.dimension < 5:
                return NGOpt21
            if self.dimension < 10:
                num = 1 + int(__import__("numpy").sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)))
                return ConfPortfolio(optimizers=[NGOpt14] * num, warmup_ratio=0.7)
            if self.dimension < 20:
                num = self.budget // (500 * self.dimension)
                return ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3 ** i) for i in range(num)], warmup_ratio=0.5)
            if self.num_workers == 1:
                return CmaFmin2
            return NGOpt16
        elif self.budget is not None and self.fully_continuous and (not self.has_noise) and (self.num_objectives < 2) and (self.num_workers <= cma_vars) and (self.budget > 50 * self.dimension) and __import__("p.helpers").Normalizer(self.parametrization).fully_bounded:
            if self.dimension < 3:
                return NGOpt8
            if self.dimension <= 20 and self.num_workers == 1:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            return NGOpt15
        else:
            return super()._select_optimizer_cls()
            
@registry.register
class NgIoh(NGOptBase):
    pass

@registry.register
class NGOpt14(NGOptBase):
    pass

@registry.register
class NGOpt15(NGOptBase):
    pass

@registry.register
class NGOpt16(NGOptBase):
    pass

class NGOpt21(NGOptBase):
    pass

class NGOpt36(NGOptBase):
    pass

class NGOpt38(NGOptBase):
    pass

class NGOpt39(NGOptBase):
    pass

@registry.register
class NGOptRW(NGOpt39):
    pass

@registry.register
class NGOptF(NGOpt39):
    pass

@registry.register
class NGOptF2(NGOpt39):
    pass

@registry.register
class NGOptF3(NGOpt39):
    pass

@registry.register
class NGOpt(NGOpt39):
    pass

@registry.register
class Wiz(NGOpt16):
    pass

@registry.register
class NgIoh(NGOptBase):
    pass

@registry.register
class NgIoh2(NGOptBase):
    pass

@registry.register
class NgIoh3(NGOptBase):
    pass

@registry.register
class NgIoh4(NGOptBase):
    pass

@registry.register
class NgIoh5(NGOptBase):
    pass

@registry.register
class NgIoh6(NGOptBase):
    pass

class _MSR(ConfPortfolio):
    def __init__(self, parametrization: Any, budget: Optional[int] = None, num_workers: int = 1, num_single_runs: int = 9, base_optimizer: Any = NGOpt) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[base_optimizer] * num_single_runs))
        self.coeffs: List[Any] = []

    def _internal_tell_candidate(self, candidate: Any, loss: float) -> None:
        if not self.coeffs:
            self.coeffs = [self.parametrization.random_state.uniform(size=self.num_objectives) for _ in self.optims]
        for coeffs, opt in zip(self.coeffs, self.optims):
            this_loss: float = __import__("numpy").sum(loss * coeffs)
            opt.tell(candidate, this_loss)

class MultipleSingleRuns(ConfPortfolio):
    def __init__(self, *, num_single_runs: int = 9, base_optimizer: Any = NGOpt) -> None:
        super().__init__(_MSR, locals())

SmoothDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='discrete').set_name('SmoothDiscreteOnePlusOne', register=True)
SmoothPortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='portfolio').set_name('SmoothPortfolioDiscreteOnePlusOne', register=True)
SmoothDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lengler').set_name('SmoothDiscreteLenglerOnePlusOne', register=True)
SmoothDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lognormal').set_name('SmoothDiscreteLognormalOnePlusOne', register=True)
SuperSmoothDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lengler', antismooth=9).set_name('SuperSmoothDiscreteLenglerOnePlusOne', register=True)
SuperSmoothTinyLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='tinylognormal', antismooth=9).set_name('SuperSmoothTinyLognormalDiscreteOnePlusOne', register=True)
UltraSmoothDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lengler', antismooth=3).set_name('UltraSmoothDiscreteLenglerOnePlusOne', register=True)
SmootherDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lengler', antismooth=2, super_radii=True).set_name('SmootherDiscreteLenglerOnePlusOne', register=True)
YoSmoothDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lengler', antismooth=2).set_name('YoSmoothDiscreteLenglerOnePlusOne', register=True)
CMALS = Chaining([CMA, DiscreteLenglerOnePlusOne, UltraSmoothDiscreteLenglerOnePlusOne], ['third', 'third']).set_name('CMALS', register=True)
UltraSmoothDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lognormal', antismooth=3).set_name('UltraSmoothDiscreteLognormalOnePlusOne', register=True)
CMALYS = Chaining([CMA, YoSmoothDiscreteLenglerOnePlusOne], ['tenth']).set_name('CMALYS', register=True)
CLengler = Chaining([CMA, DiscreteLenglerOnePlusOne], ['tenth']).set_name('CLengler', register=True)
CMALL = Chaining([CMA, DiscreteLenglerOnePlusOne, UltraSmoothDiscreteLognormalOnePlusOne], ['third', 'third']).set_name('CMALL', register=True)
CMAILL = Chaining([ImageMetaModel, ImageMetaModelLengler, UltraSmoothDiscreteLognormalOnePlusOne], ['third', 'third']).set_name('CMAILL', register=True)
CMASL = Chaining([CMA, SmootherDiscreteLenglerOnePlusOne], ['tenth']).set_name('CMASL', register=True)
CMASL2 = Chaining([CMA, SmootherDiscreteLenglerOnePlusOne], ['third']).set_name('CMASL2', register=True)
CMASL3 = Chaining([CMA, SmootherDiscreteLenglerOnePlusOne], ['half']).set_name('CMASL3', register=True)
CMAL2 = Chaining([CMA, SmootherDiscreteLenglerOnePlusOne], ['half']).set_name('CMAL2', register=True)
CMAL3 = Chaining([DiagonalCMA, SmootherDiscreteLenglerOnePlusOne], ['half']).set_name('CMAL3', register=True)
SmoothLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lognormal').set_name('SmoothLognormalDiscreteOnePlusOne', register=True)
SmoothAdaptiveDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='adaptive').set_name('SmoothAdaptiveDiscreteOnePlusOne', register=True)
SmoothRecombiningPortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='portfolio').set_name('SmoothRecombiningPortfolioDiscreteOnePlusOne', register=True)
SmoothRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler').set_name('SmoothRecombiningDiscreteLenglerOnePlusOne', register=True)
UltraSmoothRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', antismooth=3).set_name('UltraSmoothRecombiningDiscreteLenglerOnePlusOne', register=True)
UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lognormal', antismooth=3, roulette_size=7).set_name('UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne', register=True)
UltraSmoothElitistRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', antismooth=3, roulette_size=7).set_name('UltraSmoothElitistRecombiningDiscreteLenglerOnePlusOne', register=True)
SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', antismooth=9, roulette_size=7).set_name('SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne', register=True)
SuperSmoothRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', antismooth=9).set_name('SuperSmoothRecombiningDiscreteLenglerOnePlusOne', register=True)
SuperSmoothRecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lognormal', antismooth=9).set_name('SuperSmoothRecombiningDiscreteLognormalOnePlusOne', register=True)
SmoothElitistRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', roulette_size=7).set_name('SmoothElitistRecombiningDiscreteLenglerOnePlusOne', register=True)
SmoothElitistRandRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', roulette_size=7, crossover_type='rand').set_name('SmoothElitistRandRecombiningDiscreteLenglerOnePlusOne', register=True)
SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lognormal', roulette_size=7, crossover_type='rand').set_name('SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne', register=True)
RecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lengler').set_name('RecombiningDiscreteLenglerOnePlusOne', register=True)
RecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lognormal').set_name('RecombiningDiscreteLognormalOnePlusOne', register=True)
MaxRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='max').set_name('MaxRecombiningDiscreteLenglerOnePlusOne', register=True)
MinRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='min').set_name('MinRecombiningDiscreteLenglerOnePlusOne', register=True)
OnePtRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='onepoint').set_name('OnePtRecombiningDiscreteLenglerOnePlusOne', register=True)
TwoPtRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='twopoint').set_name('TwoPtRecombiningDiscreteLenglerOnePlusOne', register=True)
RandRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='rand').set_name('RandRecombiningDiscreteLenglerOnePlusOne', register=True)
RandRecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lognormal', crossover_type='rand').set_name('RandRecombiningDiscreteLognormalOnePlusOne', register=True)

@registry.register
class NgIoh7(NGOptBase):
    def _select_optimizer_cls(self) -> Any:
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (not self.has_noise):
            return Carola2
        if self.has_noise and (self.has_discrete_not_softmax or not self.parametrization.function.metrizable):
            return RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not self.parametrization.function.metrizable):
            return CMA
        return NGOptBase._select_optimizer_cls(self)

@registry.register
class NgIoh11(NGOptBase):
    def _select_optimizer_cls(self, budget: Optional[int] = None) -> Any:
        if self.budget is not None and self.dimension < self.budget:
            return NgIoh21._select_optimizer_cls(self, budget)
        assert budget is None
        optCls: Any = NGOptBase
        funcinfo: Any = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            if self.budget > 2000 * self.dimension:
                vlpcma = ParametrizedMetaModel(multivariate_optimizer=VLPCMA)
                return vlpcma
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
                return DS6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return DS6
            if 3000 * self.dimension < self.budget:
                MetaModelDS = ParametrizedMetaModel(multivariate_optimizer=DSproba)
                MetaModelDS.no_parallelization = True
                return MetaModelDS
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                return ChainMetaModelDSSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and (self.dimension > 30):
                return ChainMetaModelDSSQP
            if self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and (self.dimension == 2):
                return NLOPT_LN_SBPLX
            if self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and (self.dimension < 15):
                return ChainMetaModelDSSQP
            if self.budget >= 300 * self.dimension and self.budget < 3000 * self.dimension and (self.dimension < 30):
                return MultiDS
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            return DS2
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return DS2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgIoh21(NgIoh11):
    def _select_optimizer_cls(self, budget: Optional[int] = None) -> Any:
        if self.budget is not None and self.dimension < self.budget:
            return NgIoh21._select_optimizer_cls(self, budget)
        assert budget is None
        optCls: Any = NGOptBase
        funcinfo: Any = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            if self.budget > 2000 * self.dimension:
                vlpcma = ParametrizedMetaModel(multivariate_optimizer=VLPCMA) if self.dimension > 4 else ParametrizedMetaModel(multivariate_optimizer=LPCMA)
                return vlpcma
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
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and (self.dimension > 30):
                return ChainMetaModelSQP
            if self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and (self.dimension == 2):
                return NLOPT_LN_SBPLX
            if self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and (self.dimension < 15):
                return ChainMetaModelSQP
            if self.budget >= 300 * self.dimension and self.budget < 3000 * self.dimension and (self.dimension < 30):
                return MultiCMA
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            return Carola2
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return Carola2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgIohTuned(ConfPortfolio):
    def __init__(self, parametrization: Any, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[NGOpt, TwoPointsDE, PSO, SQP, ScrHammersleySearch], warmup_ratio=0.5))
        
# Additional optimizer definitions with similar type annotations would follow here...
# For brevity, only the structure of some classes is shown.
# Each __init__ method and function would be annotated with appropriate types as above.
