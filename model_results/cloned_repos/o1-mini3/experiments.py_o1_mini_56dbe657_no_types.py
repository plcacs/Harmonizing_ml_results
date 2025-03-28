import os
import warnings
import typing as tp
import inspect
import itertools
import numpy as np
import nevergrad as ng
import nevergrad.functions.corefuncs as corefuncs
from nevergrad.functions import base as fbase
from nevergrad.functions import ExperimentFunction, ArtificialFunction, FarOptimumFunction
from nevergrad.functions.fishing import OptimizeFish
from nevergrad.functions.pbt import PBT
from nevergrad.functions.ml import MLTuning
from nevergrad.functions import mlda as _mlda
from nevergrad.functions.photonics import Photonics, ceviche as photonics_ceviche
from nevergrad.functions.arcoating import ARCoating
from nevergrad.functions import images as imagesxp
from nevergrad.functions.powersystems import PowerSystem
from nevergrad.functions.ac import NgAquacrop
from nevergrad.functions.stsp import STSP
from nevergrad.functions.topology_optimization import TO
from nevergrad.functions.lsgo import make_function as lsgo_makefunction
from nevergrad.functions.rocket import Rocket
from nevergrad.functions.mixsimulator import OptimizeMix
from nevergrad.functions.unitcommitment import UnitCommitmentProblem
from nevergrad.functions import control, rl
from nevergrad.functions.games import game
from nevergrad.functions import iohprofiler, helpers
from nevergrad.functions.cycling import Cycling
from .xpbase import Experiment, create_seed_generator, registry
from .optgroups import get_optimizers
from . import frozenexperiments
from . import gymexperiments

def refactor_optims(x):
    if False:
        return list(np.random.choice(['NgIohTuned', 'NGOpt', 'NGOptRW', 'ChainCMASQP', 'PymooBIPOP', 'NLOPT_LN_SBPLX', 'QNDE', 'BFGSCMAPlus', 'ChainMetaModelSQP', 'BFGSCMA', 'BAR4', 'BFGSCMAPlus', 'LBFGSB', 'LQOTPDE', 'LogSQPCMA'], 4))
    list_optims: tp.List[Any] = x
    algos: Dict[str, List[str]] = {}
    algos['aquacrop_fao'] = ['CMA', 'CMandAS2', 'DE', 'MetaModel', 'NGOpt10']
    algos['bonnans'] = ['AdaptiveDiscreteOnePlusOne', 'DiscreteBSOOnePlusOne', 'DiscreteLenglerFourthOnePlusOne', 'DiscreteLenglerHalfOnePlusOne', 'DiscreteLenglerOnePlusOne', 'MemeticDE']
    algos['double_o_seven'] = ['DiagonalCMA', 'DiscreteDE', 'MetaTuneRecentering', 'PSO', 'RecombiningOptimisticNoisyDiscreteOnePlusOne', 'TBPSA']
    algos['fishing'] = ['CMA', 'CMandAS2', 'ChainMetaModelSQP', 'DE', 'MetaModel', 'NGOpt10']
    algos['mldakmeans'] = ['DE', 'SplitCMA5', 'SplitTwoPointsDE3', 'SplitTwoPointsDE5', 'SplitTwoPointsDEAuto', 'TwoPointsDE']
    algos['mltuning'] = ['OnePlusOne', 'RandomSearch']
    algos['mono_rocket'] = ['CMA', 'CMandAS2', 'DE', 'MetaModel', 'NGOpt10']
    algos['ms_bbob'] = ['ChainMetaModelSQP', 'MetaModelOnePlusOne', 'Powell', 'QODE', 'SQP', 'TinyCMA']
    algos['multiobjective_example_hd'] = ['DiscreteLenglerOnePlusOne', 'DiscreteOnePlusOne', 'MetaNGOpt10', 'ParametrizationDE', 'RecES', 'RecMutDE']
    algos['multiobjective_example_many_hd'] = ['DiscreteLenglerOnePlusOne', 'DiscreteOnePlusOne', 'MetaNGOpt10', 'ParametrizationDE', 'RecES', 'RecMutDE']
    algos['multiobjective_example'] = ['CMA', 'DE', 'ParametrizationDE', 'RecES', 'RecMutDE']
    algos['naive_seq_keras_tuning'] = ['CMA', 'DE', 'HyperOpt', 'OnePlusOne', 'RandomSearch', 'TwoPointsDE']
    algos['nano_naive_seq_mltuning'] = ['DE', 'HyperOpt', 'OnePlusOne', 'RandomSearch', 'TwoPointsDE']
    algos['nano_seq_mltuning'] = ['DE', 'HyperOpt', 'OnePlusOne', 'RandomSearch', 'TwoPointsDE']
    algos['oneshot_mltuning'] = ['DE', 'OnePlusOne', 'RandomSearch', 'TwoPointsDE']
    algos['pbbob'] = ['CMAbounded', 'DE', 'MetaModelDE', 'MetaModelOnePlusOne', 'QODE', 'QrDE']
    algos['pbo_reduced_suite'] = ['DiscreteLenglerOnePlusOne', 'LognormalDiscreteOnePlusOne', 'DiscreteLenglerOnePlusOneT', 'DiscreteLenglerOnePlusOneT', 'SADiscreteLenglerOnePlusOneExp09', 'SADiscreteLenglerOnePlusOneExp09', 'discretememetic']
    algos['reduced_yahdlbbbob'] = ['CMA', 'DE', 'MetaModelOnePlusOne', 'OnePlusOne', 'PSO', 'RFMetaModelDE']
    algos['seq_keras_tuning'] = ['CMA', 'DE', 'HyperOpt', 'OnePlusOne', 'RandomSearch', 'TwoPointsDE']
    algos['sequential_topology_optimization'] = ['CMA', 'DE', 'GeneticDE', 'OnePlusOne', 'TwoPointsDE', 'VoronoiDE']
    algos['spsa_benchmark'] = ['CMA', 'DE', 'NaiveTBPSA', 'OnePlusOne', 'SPSA', 'TBPSA']
    algos['topology_optimization'] = ['CMA', 'DE', 'GeneticDE', 'OnePlusOne', 'TwoPointsDE', 'VoronoiDE']
    algos['yabbob'] = ['CMA', 'ChainMetaModelSQP', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne']
    algos['yabigbbob'] = ['ChainMetaModelSQP', 'MetaModel', 'MetaModelDE', 'NeuralMetaModel', 'PSO', 'TwoPointsDE']
    algos['yaboundedbbob'] = ['CMA', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yaboxbbob'] = ['CMA', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yamegapenbbob'] = ['ChainMetaModelSQP', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yamegapenboundedbbob'] = ['CMA', 'ChainMetaModelSQP', 'MetaModel', 'MetaModelOnePlusOne', 'OnePlusOne', 'RFMetaModel']
    algos['yamegapenboxbbob'] = ['ChainMetaModelSQP', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yanoisybbob'] = ['TBPSA', 'NoisyRL2', 'NoisyRL3', 'RecombiningOptimisticNoisyDiscreteOnePlusOne', 'RBFGS', 'MicroCMA', 'NoisyDiscreteOnePlusOne', 'RandomSearch', 'RecombiningOptimisticNoisyDiscreteOnePlusOne', 'SQP']
    algos['yaonepenbbob'] = ['CMandAS2', 'ChainMetaModelSQP', 'MetaModel', 'NGOpt', 'NeuralMetaModel', 'Shiwa']
    algos['yaonepenboundedbbob'] = ['CMA', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yaonepenboxbbob'] = ['CMA', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yaonepennoisybbob'] = ['NoisyDiscreteOnePlusOne', 'RandomSearch', 'SQP', 'TBPSA']
    algos['yaonepenparabbob'] = ['CMA', 'MetaModel', 'MetaModelDE', 'NeuralMetaModel', 'RFMetaModel', 'RFMetaModelDE']
    algos['yaonepensmallbbob'] = ['Cobyla', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yaparabbob'] = ['CMA', 'MetaModel', 'MetaModelDE', 'NeuralMetaModel', 'RFMetaModel', 'RFMetaModelDE']
    algos['yapenbbob'] = ['ChainMetaModelSQP', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yapenboundedbbob'] = ['CMA', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yapenboxbbob'] = ['CMA', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yapennoisybbob'] = ['NoisyDiscreteOnePlusOne', 'RandomSearch', 'SQP', 'TBPSA']
    algos['yapenparabbob'] = ['CMA', 'MetaModel', 'MetaModelDE', 'NeuralMetaModel', 'RFMetaModel', 'RFMetaModelDE']
    algos['yapensmallbbob'] = ['Cobyla', 'MetaModel', 'MetaModelOnePlusOne', 'OnePlusOne', 'RFMetaModel', 'RFMetaModelDE']
    algos['yasmallbbob'] = ['Cobyla', 'MetaModelDE', 'MetaModelOnePlusOne', 'OnePlusOne', 'PSO', 'RFMetaModelDE']
    algos['yatinybbob'] = ['Cobyla', 'DE', 'MetaModel', 'MetaModelDE', 'MetaModelOnePlusOne', 'TwoPointsDE']
    algos['yatuningbbob'] = ['Cobyla', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'RFMetaModelDE', 'RandomSearch', 'TwoPointsDE']
    benchmark = str(inspect.stack()[1].function)
    if benchmark in algos:
        list_algos: tp.List[str] = algos[benchmark][:5] + ['CSEC10', 'NGOpt', 'NLOPT_LN_SBPLX']
        return list_algos if 'eras' in benchmark or 'tial_instrum' in benchmark or 'big' in benchmark or ('lsgo' in benchmark) or ('rock' in benchmark) else list_algos
    if benchmark in algos:
        list_algos = algos[benchmark]
        return list_algos if 'eras' in benchmark or 'tial_instrum' in benchmark or 'big' in benchmark or ('lsgo' in benchmark) or ('rock' in benchmark) else list_algos
    return ['NgDS3', 'NgIoh4', 'NgIoh21', 'NGOpt', 'NGDSRW']

def skip_ci(*, reason: str):
    """Only use this if there is a good reason for not testing the xp,
    such as very slow for instance (>1min) with no way to make it faster.
    This is dangerous because it won't test reproducibility and the experiment
    may therefore be corrupted with no way to notice it automatically.
    """
    if os.environ.get('NEVERGRAD_PYTEST', False):
        raise fbase.UnsupportedExperiment('Skipping CI: ' + reason)

class _Constraint:

    def __init__(self, name, as_bool):
        self.name: str = name
        self.as_bool: bool = as_bool

    def __call__(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError(f'Unexpected inputs as np.ndarray, got {data}')
        if self.name == 'sum':
            value: float = float(np.sum(data))
        elif self.name == 'diff':
            value = float(np.sum(data[::2]) - np.sum(data[1::2]))
        elif self.name == 'second_diff':
            value = float(2 * np.sum(data[1::2]) - 3 * np.sum(data[::2]))
        elif self.name == 'ball':
            value = float(np.sum(np.square(data)) - len(data) - np.sqrt(len(data)))
        else:
            raise NotImplementedError(f'Unknown function {self.name}')
        return value > 0 if self.as_bool else value

@registry.register
def keras_tuning(seed=None, overfitter=False, seq=False, veryseq=False):
    """Machine learning hyperparameter tuning experiment. Based on Keras models."""
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    optims: tp.List[str] = ['OnePlusOne', 'RandomSearch', 'Cobyla']
    optims = ['DE', 'TwoPointsDE', 'HyperOpt', 'MetaModelOnePlusOne']
    optims = get_optimizers('oneshot', seed=next(seedg))
    optims = ['MetaTuneRecentering', 'MetaRecentering', 'HullCenterHullAvgCauchyScrHammersleySearch', 'LHSSearch', 'LHSCauchySearch']
    optims = ['NGOpt', 'NGOptRW', 'QODE']
    optims = ['NGOpt']
    optims = ['PCABO', 'NGOpt', 'QODE']
    optims = ['QOPSO']
    optims = ['SQOPSO']
    optims = refactor_optims(optims)
    datasets: tp.List[str] = ['kerasBoston', 'diabetes', 'auto-mpg', 'red-wine', 'white-wine']
    optims = refactor_optims(optims)
    for dimension in [None]:
        for dataset in datasets:
            function: MLTuning = MLTuning(regressor='keras_dense_nn', data_dimension=dimension, dataset=dataset, overfitter=overfitter)
            for budget in [150, 500]:
                for num_workers in [1, budget // 4] if seq else [budget]:
                    if veryseq and num_workers > 1:
                        continue
                    for optim in optims:
                        xp: Experiment = Experiment(function, optim, num_workers=num_workers, budget=budget, seed=next(seedg))
                        skip_ci(reason='too slow')
                        xp.function.parametrization.real_world = True
                        xp.function.parametrization.hptuning = True
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def mltuning(seed=None, overfitter=False, seq=False, veryseq=False, nano=False):
    """Machine learning hyperparameter tuning experiment. Based on scikit models."""
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    optims: tp.List[str] = ['DE', 'TwoPointsDE', 'HyperOpt', 'MetaModelOnePlusOne']
    optims = get_optimizers('oneshot', seed=next(seedg))
    optims = ['MetaTuneRecentering', 'MetaRecentering', 'HullCenterHullAvgCauchyScrHammersleySearch', 'LHSSearch', 'LHSCauchySearch']
    optims = ['NGOpt', 'NGOptRW', 'QODE']
    optims = ['NGOpt']
    optims = ['PCABO']
    optims = ['PCABO', 'NGOpt', 'QODE']
    optims = ['QOPSO']
    optims = ['SQOPSO']
    optims = refactor_optims(optims)
    for dimension in [None, 1, 2, 3]:
        if dimension is None:
            datasets = ['diabetes', 'auto-mpg', 'red-wine', 'white-wine']
        else:
            datasets = ['artificialcos', 'artificial', 'artificialsquare']
        for regressor in ['mlp', 'decision_tree', 'decision_tree_depth']:
            for dataset in datasets:
                function: MLTuning = MLTuning(regressor=regressor, data_dimension=dimension, dataset=dataset, overfitter=overfitter)
                for budget in [150, 500] if not nano else [80, 160]:
                    parallelization: tp.List[int] = [1, budget // 4] if seq else [budget]
                    for num_workers in parallelization:
                        if veryseq and num_workers > 1:
                            continue
                        for optim in optims:
                            xp: Experiment = Experiment(function, optim, num_workers=num_workers, budget=budget, seed=next(seedg))
                            skip_ci(reason='too slow')
                            xp.function.parametrization.real_world = True
                            xp.function.parametrization.hptuning = True
                            if not xp.is_incoherent:
                                yield xp

@registry.register
def naivemltuning(seed=None):
    """Counterpart of mltuning with overfitting of valid loss, i.e. train/valid/valid instead of train/valid/test."""
    return mltuning(seed, overfitter=True)

@registry.register
def veryseq_keras_tuning(seed=None):
    """Iterative counterpart of keras tuning."""
    return keras_tuning(seed, overfitter=False, seq=True, veryseq=True)

@registry.register
def seq_keras_tuning(seed=None):
    """Iterative counterpart of keras tuning."""
    return keras_tuning(seed, overfitter=False, seq=True)

@registry.register
def naive_seq_keras_tuning(seed=None):
    """Naive counterpart (no overfitting, see naivemltuning)of seq_keras_tuning."""
    return keras_tuning(seed, overfitter=True, seq=True)

@registry.register
def naive_veryseq_keras_tuning(seed=None):
    """Naive counterpart (no overfitting, see naivemltuning)of seq_keras_tuning."""
    return keras_tuning(seed, overfitter=True, seq=True, veryseq=True)

@registry.register
def oneshot_mltuning(seed=None):
    """One-shot counterpart of Scikit tuning."""
    return mltuning(seed, overfitter=False, seq=False)

@registry.register
def seq_mltuning(seed=None):
    """Iterative counterpart of mltuning."""
    return mltuning(seed, overfitter=False, seq=True)

@registry.register
def nano_seq_mltuning(seed=None):
    """Iterative counterpart of seq_mltuning with smaller budget."""
    return mltuning(seed, overfitter=False, seq=True, nano=True)

@registry.register
def nano_veryseq_mltuning(seed=None):
    """Iterative counterpart of seq_mltuning with smaller budget."""
    return mltuning(seed, overfitter=False, seq=True, nano=True, veryseq=True)

@registry.register
def nano_naive_veryseq_mltuning(seed=None):
    """Iterative counterpart of mltuning with overfitting of valid loss, i.e. train/valid/valid instead of train/valid/test,
    and with lower budget."""
    return mltuning(seed, overfitter=True, seq=True, nano=True, veryseq=True)

@registry.register
def nano_naive_seq_mltuning(seed=None):
    """Iterative counterpart of mltuning with overfitting of valid loss, i.e. train/valid/valid instead of train/valid/test,
    and with lower budget."""
    return mltuning(seed, overfitter=True, seq=True, nano=True)

@registry.register
def naive_seq_mltuning(seed=None):
    """Iterative counterpart of mltuning with overfitting of valid loss, i.e. train/valid/valid instead of train/valid/test."""
    return mltuning(seed, overfitter=True, seq=True)

@registry.register
def yawidebbob(seed=None):
    """Yet Another Wide Black-Box Optimization Benchmark.
    The goal is basically to have a very wide family of problems: continuous and discrete,
    noisy and noise-free, mono- and multi-objective,  constrained and not constrained, sequential
    and parallel.

    TODO(oteytaud): this requires a significant improvement, covering mixed problems and different types of constraints.
    """
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    total_xp_per_optim: int = 0
    functions: tp.List[ArtificialFunction] = [ArtificialFunction(name, block_dimension=50, rotation=rotation, translation_factor=tf) for name in ['cigar', 'ellipsoid'] for rotation in [True, False] for tf in [0.1, 10.0]]
    for i, func in enumerate(functions):
        func.parametrization.register_cheap_constraint(_Constraint('sum', as_bool=i % 2 == 0))
    assert len(functions) == 8
    names: tp.List[str] = ['hm', 'rastrigin', 'sphere', 'doublelinearslope', 'ellipsoid']
    functions += [ArtificialFunction(name, block_dimension=d, rotation=rotation, noise_level=nl, split=split, translation_factor=tf, num_blocks=num_blocks) for name in names for rotation in [True, False] for nl in [0.0, 100.0] for tf in [0.1, 10.0] for num_blocks in [1, 8] for d in [5, 70, 10000] for split in [True, False]][::37]
    assert len(functions) == 21, f'{len(functions)} problems instead of 21. Yawidebbob should be standard.'
    optims: tp.List[str] = ['NGOptRW', 'NGOpt', 'RandomSearch', 'CMA', 'DE', 'DiscreteLenglerOnePlusOne']
    optims = refactor_optims(optims)
    index: int = 0
    for function in functions:
        for budget in [50, 1500, 25000]:
            for nw in [1, budget // 4] + ([] if budget <= 300 else [300]):
                index += 1
                if index % 5 == 0:
                    total_xp_per_optim += 1
                    for optim in optims:
                        xp: Experiment = Experiment(function, optim, num_workers=nw, budget=budget, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp
    assert total_xp_per_optim == 33, f'We have 33 single-obj xps per optimizer (got {total_xp_per_optim}).'
    index = 0
    for nv in [200, 2000]:
        for arity in [2, 7, 37]:
            instrum: ng.p.TransitionChoice = ng.p.TransitionChoice(range(arity), repetitions=nv)
            for name in ['onemax', 'leadingones', 'jump']:
                index += 1
                if index % 4 != 0:
                    continue
                dfunc: ExperimentFunction = ExperimentFunction(corefuncs.DiscreteFunction(name, arity), instrum.set_name('transition'))
                dfunc.add_descriptors(arity=arity)
                dfunc.add_descriptors(nv=nv)
                dfunc.add_descriptors(instrum_str='transition')
                for budget in [50, 500, 5000]:
                    for nw in [1, 100]:
                        total_xp_per_optim += 1
                        for optim in optims:
                            yield Experiment(dfunc, optim, num_workers=nw, budget=budget, seed=next(seedg))
    assert total_xp_per_optim == 57, f'Including discrete, we check xps per optimizer (got {total_xp_per_optim}).'
    mofuncs: tp.List[fbase.MultiExperiment] = []
    for name1, name2 in itertools.product(['sphere', 'ellipsoid'], ['sphere', 'hm']):
        for tf in [0.25, 4.0]:
            mofuncs += [fbase.MultiExperiment([ArtificialFunction(name1, block_dimension=7), ArtificialFunction(name2, block_dimension=7, translation_factor=tf)], upper_bounds=np.array((100.0, 100.0)))]
            mofuncs[-1].add_descriptors(num_objectives=2)
    for name1, name2, name3 in itertools.product(['sphere', 'ellipsoid'], ['sphere', 'hm'], ['sphere', 'hm']):
        for tf in [0.25, 4.0]:
            mofuncs += [fbase.MultiExperiment([ArtificialFunction(name1, block_dimension=7, translation_factor=1.0 / tf), ArtificialFunction(name2, block_dimension=7, translation_factor=tf), ArtificialFunction(name3, block_dimension=7)], upper_bounds=np.array((100.0, 100.0, 100.0)))]
            mofuncs[-1].add_descriptors(num_objectives=3)
    index = 0
    for mofunc in mofuncs[::3]:
        for budget in [2000, 4000, 8000]:
            for nw in [1, 20, 100]:
                index += 1
                if index % 5 == 0:
                    total_xp_per_optim += 1
                    for optim in optims:
                        xp: Experiment = Experiment(mofunc, optim, budget=budget, num_workers=nw, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp
    assert total_xp_per_optim == 71, f'We should have 71 xps per optimizer, not {total_xp_per_optim}.'

@registry.register
def parallel_small_budget(seed=None):
    """Parallel optimization with small budgets"""
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    optims: tp.List[str] = ['DE', 'TwoPointsDE', 'CMA', 'NGOpt', 'PSO', 'OnePlusOne', 'RandomSearch']
    names: tp.List[str] = ['hm', 'rastrigin', 'griewank', 'rosenbrock', 'ackley', 'multipeak']
    names += ['sphere', 'cigar', 'ellipsoid', 'altellipsoid']
    names += ['deceptiveillcond', 'deceptivemultimodal', 'deceptivepath']
    functions: tp.List[ArtificialFunction] = [ArtificialFunction(name, block_dimension=d, rotation=rotation) for name in names for rotation in [True, False] for d in [2, 4, 8]]
    budgets: tp.List[int] = [10, 50, 100, 200, 400]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in budgets:
                for nw in [2, 8, 16]:
                    for batch in [True, False]:
                        if nw < budget / 4:
                            xp: Experiment = Experiment(function, optim, num_workers=nw, budget=budget, batch_mode=batch, seed=next(seedg))
                            if not xp.is_incoherent:
                                yield xp

@registry.register
def instrum_discrete(seed=None):
    """Comparison of optimization algorithms equipped with distinct instrumentations.
    Onemax, Leadingones, Jump function."""
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    optims: tp.List[str] = ['DiscreteOnePlusOne', 'NGOpt', 'CMA', 'TwoPointsDE', 'DiscreteLenglerOnePlusOne']
    optims = ['RFMetaModelOnePlusOne']
    optims = ['FastGADiscreteOnePlusOne']
    optims = ['DoubleFastGADiscreteOnePlusOne']
    optims = ['DiscreteOnePlusOne']
    optims = ['OnePlusOne']
    optims = ['DiscreteLenglerOnePlusOne']
    optims = ['NGOpt', 'NGOptRW']
    optims = refactor_optims(optims)
    for nv in [10, 50, 200, 1000, 5000]:
        for arity in [2, 3, 7, 30]:
            for instrum_str in ['Unordered', 'Softmax', 'Ordered']:
                if instrum_str == 'Softmax':
                    instrum: ng.p.Choice = ng.p.Choice(range(arity), repetitions=nv)
                else:
                    assert instrum_str in ('Ordered', 'Unordered')
                    instrum = ng.p.TransitionChoice(range(arity), repetitions=nv, ordered=instrum_str == 'Ordered')
                for name in ['onemax', 'leadingones', 'jump']:
                    dfunc: ExperimentFunction = ExperimentFunction(corefuncs.DiscreteFunction(name, arity), instrum.set_name(instrum_str))
                    dfunc.add_descriptors(arity=arity)
                    dfunc.add_descriptors(nv=nv)
                    dfunc.add_descriptors(instrum_str=instrum_str)
                    for optim in optims:
                        for nw in [1, 10]:
                            for budget in [50, 500, 5000]:
                                yield Experiment(dfunc, optim, num_workers=nw, budget=budget, seed=next(seedg))

@registry.register
def sequential_instrum_discrete(seed=None):
    """Sequential counterpart of instrum_discrete."""
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    optims: tp.List[str] = ['DiscreteOnePlusOne', 'NGOpt', 'CMA', 'TwoPointsDE', 'DiscreteLenglerOnePlusOne']
    optims = ['OnePlusOne']
    optims = ['DiscreteLenglerOnePlusOne']
    optims = ['NGOpt', 'NGOptRW']
    optims = [l for l in list(ng.optimizers.registry.keys()) if 'DiscreteOneP' in l and 'SA' not in l and ('Smooth' not in l) and ('Noisy' not in l) and ('Optimis' not in l) and ('T' != l[-1])] + ['cGA', 'DiscreteDE']
    optims = refactor_optims(optims)
    for nv in [10, 50, 200, 1000, 5000]:
        for arity in [2, 3, 7, 30]:
            for instrum_str in ['Unordered', 'Softmax', 'Ordered']:
                if instrum_str == 'Softmax':
                    instrum: ng.p.Choice = ng.p.Choice(range(arity), repetitions=nv)
                else:
                    instrum = ng.p.TransitionChoice(range(arity), repetitions=nv, ordered=instrum_str == 'Ordered')
                for name in ['onemax', 'leadingones', 'jump']:
                    dfunc: ExperimentFunction = ExperimentFunction(corefuncs.DiscreteFunction(name, arity), instrum.set_name(instrum_str))
                    dfunc.add_descriptors(arity=arity)
                    dfunc.add_descriptors(nv=nv)
                    dfunc.add_descriptors(instrum_str=instrum_str)
                    for optim in optims:
                        for budget in [50, 500, 5000, 50000]:
                            yield Experiment(dfunc, optim, budget=budget, seed=next(seedg))

@registry.register
def deceptive(seed=None):
    """Very difficult objective functions: one is highly multimodal (infinitely many local optima),
    one has an infinite condition number, one has an infinitely long path towards the optimum.
    Looks somehow fractal."""
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    names: tp.List[str] = ['deceptivemultimodal', 'deceptiveillcond', 'deceptivepath']
    optims: tp.List[str] = ['CMA', 'DE', 'TwoPointsDE', 'PSO', 'OnePlusOne', 'RandomSearch', 'NGOptRW']
    optims = ['RBFGS', 'LBFGSB', 'DE', 'TwoPointsDE', 'RandomSearch', 'OnePlusOne', 'PSO', 'CMA', 'ChainMetaModelSQP', 'MemeticDE', 'MetaModel', 'RFMetaModel', 'MetaModelDE', 'RFMetaModelDE']
    optims = ['NGOpt']
    functions: tp.List[ArtificialFunction] = [ArtificialFunction(name, block_dimension=2, num_blocks=n_blocks, rotation=rotation, aggregator=aggregator) for name in names for rotation in [False, True] for n_blocks in [1, 2, 8, 16] for aggregator in ['sum', 'max']]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [25, 37, 50, 75, 87, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))

@registry.register
def lowbudget(seed=None):
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    names: tp.List[str] = ['sphere', 'rastrigin', 'cigar']
    optims: tp.List[str] = ['AX', 'BOBYQA', 'Cobyla', 'RandomSearch', 'CMA', 'NGOpt', 'DE', 'PSO', 'pysot', 'negpysot']
    functions: tp.List[ArtificialFunction] = [ArtificialFunction(name, block_dimension=7, bounded=b) for name in names for b in [True, False]]
    for func in functions:
        for optim in optims:
            for budget in [10, 20, 30]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))

@registry.register
def parallel(seed=None):
    """Parallel optimization on 3 classical objective functions: sphere, rastrigin, cigar.
    The number of workers is 20 % of the budget.
    Testing both no useless variables and 5/6 of useless variables."""
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    names: tp.List[str] = ['sphere', 'rastrigin', 'cigar']
    optims: tp.List[str] = get_optimizers('parallel_basics', seed=next(seedg))
    functions: tp.List[ArtificialFunction] = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) for name in names for bd in [25] for uv_factor in [0, 5]]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=int(budget / 5), seed=next(seedg))

@registry.register
def harderparallel(seed=None):
    """Parallel optimization on 4 classical objective functions. More distinct settings than << parallel >>."""
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    names: tp.List[str] = ['sphere', 'rastrigin', 'cigar', 'ellipsoid']
    optims: tp.List[str] = ['NGOpt10'] + get_optimizers('emna_variants', seed=next(seedg))
    functions: tp.List[ArtificialFunction] = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) for name in names for bd in [5, 25] for uv_factor in [0, 5]]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000]:
                for num_workers in [int(budget / 10), int(budget / 5), int(budget / 3)]:
                    yield Experiment(func, optim, budget=budget, num_workers=num_workers, seed=next(seedg))

@registry.register
def oneshot(seed=None):
    """One shot optimization of 3 classical objective functions (sphere, rastrigin, cigar).
    0 or 5 dummy variables per real variable.
    Base dimension 3 or 25.
    budget 30, 100 or 3000."""
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    names: tp.List[str] = ['sphere', 'rastrigin', 'cigar']
    optims: tp.List[str] = get_optimizers('oneshot', seed=next(seedg))
    functions: tp.List[ArtificialFunction] = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) for name in names for bd in [3, 10, 30, 100, 300, 1000, 3000] for uv_factor in [0]]
    for func in functions:
        for optim in optims:
            for budget in [100000, 30, 100, 300, 1000, 3000, 10000]:
                if func.dimension < 3000 or budget < 100000:
                    yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))

@registry.register
def doe(seed=None):
    """One shot optimization of 3 classical objective functions (sphere, rastrigin, cigar), simplified.
    Base dimension 2000 or 20000. No rotation, no dummy variable.
    Budget 30, 100, 3000, 10000, 30000, 100000."""
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    names: tp.List[str] = ['sphere', 'rastrigin', 'cigar']
    optims: tp.List[str] = get_optimizers('oneshot', seed=next(seedg))
    functions: tp.List[ArtificialFunction] = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) for name in names for bd in [2000, 20000] for uv_factor in [0]]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000, 30000, 100000]:
                yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))

@registry.register
def newdoe(seed=None):
    """One shot optimization of 3 classical objective functions (sphere, rastrigin, cigar), simplified.
    Tested on more dimensionalities than doe, namely 20, 200, 2000, 20000. No dummy variables.
    Budgets 30, 100, 3000, 10000, 30000, 100000, 300000."""
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    names: tp.List[str] = ['sphere', 'rastrigin', 'cigar']
    optims: tp.List[str] = get_optimizers('oneshot', seed=next(seedg))
    functions: tp.List[ArtificialFunction] = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) for name in names for bd in [2000, 20, 200, 20000] for uv_factor in [0]]
    budgets: tp.List[int] = [30, 100, 3000, 10000, 30000, 100000, 300000]
    for func in functions:
        for optim in optims:
            for budget in budgets:
                yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))

@registry.register
def fiveshots(seed=None):
    """Five-shots optimization of 3 classical objective functions (sphere, rastrigin, cigar).
    Base dimension 3 or 25. 0 or 5 dummy variable per real variable. Budget 30, 100 or 3000."""
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    names: tp.List[str] = ['sphere', 'rastrigin', 'cigar']
    optims: tp.List[str] = get_optimizers('oneshot', 'basics', seed=next(seedg))
    functions: tp.List[ArtificialFunction] = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) for name in names for bd in [3, 25] for uv_factor in [0, 5]]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=budget // 5, seed=next(seedg))

@registry.register
def multimodal(seed=None, para=False):
    """Experiment on multimodal functions, namely hm, rastrigin, griewank, rosenbrock, ackley, lunacek,
    deceptivemultimodal.
    0 or 5 dummy variable per real variable.
    Base dimension 3 or 25.
    Budget in 3000, 10000, 30000, 100000.
    Sequential.
    """
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    names: tp.List[str] = ['hm', 'rastrigin', 'griewank', 'rosenbrock', 'ackley', 'lunacek', 'deceptivemultimodal']
    optims: tp.List[str] = get_optimizers('basics', seed=next(seedg))
    optims = ['RBFGS', 'LBFGSB', 'DE', 'TwoPointsDE', 'RandomSearch', 'OnePlusOne', 'PSO', 'CMA', 'ChainMetaModelSQP', 'MemeticDE', 'MetaModel', 'RFMetaModel', 'MetaModelDE', 'RFMetaModelDE']
    optims = ['NGOpt']
    optims = refactor_optims(optims)
    functions: tp.List[ArtificialFunction] = [ArtificialFunction(name, block_dimension=bd) for name in names for bd in [1000, 6000, 36000]]
    for func in functions:
        for optim in optims:
            for budget in [3000, 10000]:
                for nw in [1]:
                    yield Experiment(func, optim, budget=budget, num_workers=nw, seed=next(seedg))

@registry.register
def hdbo4d(seed=None):
    """All Bayesian optimization methods on various functions.
    Budget 25, 31, 37, 43, 50, 60.
    Dimension 20.
    Sphere, Cigar, Hm, Ellipsoid.
    """
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    names: tp.List[str] = ['hm', 'rastrigin', 'griewank', 'rosenbrock', 'ackley', 'lunacek', 'deceptivemultimodal', 'bucherastrigin', 'multipeak']
    names += ['sphere', 'doublelinearslope', 'stepdoublelinearslope']
    names += ['cigar', 'altcigar', 'ellipsoid', 'altellipsoid', 'stepellipsoid', 'discus', 'bentcigar']
    names += ['deceptiveillcond', 'deceptivemultimodal', 'deceptivepath']
    noise: bool = False
    if noise:
        noise_level: int = 100000 if False else 100
    else:
        noise_level: int = 0
    optims: tp.List[str] = ['MetaModelDE', 'NeuralMetaModelDE', 'SVMMetaModelDE', 'RFMetaModelDE', 'MetaModelTwoPointsDE', 'NeuralMetaModelTwoPointsDE', 'SVMMetaModelTwoPointsDE', 'RFMetaModelTwoPointsDE', 'GeneticDE']
    optims = ['CMA', 'NGOpt', 'NGOptRW']
    optims = ['DiagonalCMA', 'TinyQODE', 'OpoDE', 'OpoTinyDE']
    optims = ['TinyQODE', 'OpoDE', 'OpoTinyDE']
    optims = refactor_optims(optims)
    functions: tp.List[ArtificialFunction] = [ArtificialFunction(name, block_dimension=d, rotation=rotation, noise_level=noise_level, split=split, num_blocks=num_blocks, bounded=bounded or box) for name in names for rotation in [True, False] for num_blocks in ([1] if not split else [7, 12]) for d in ([100, 1000, 3000] if False else [2, 5, 10, 15] if False else [40] if False else [2, 10, 50] if not noise else [2, 10, 50])]
    assert 1 <= 1 <= 4
    functions = functions[::1]
    constraints: tp.List[tp.Any] = [_Constraint(name, as_bool) for as_bool in [False, True] for name in ['sum', 'diff', 'second_diff', 'ball']]
    if 0 > 0:
        constraints = []
        dim: int = 1000
        max_num_constraints: int = 0
        constraint_case: int = -1
        xs: np.ndarray = np.random.rand(dim)

        def make_ctr(i):
            xfail: np.ndarray = np.random.RandomState(i).rand(dim)

            def f(x):
                local_dim: int = min(dim, len(x))
                x = x[:local_dim]
                normal: float = np.exp(np.random.RandomState(i + 31721).randn() - 1.0) * np.linalg.norm((x - xs[:local_dim]) * np.random.RandomState(i + 741).randn(local_dim))
                return normal - np.sum((xs[:local_dim] - xfail[:local_dim]) * (x - (xs[:local_dim] + xfail[:local_dim]) / 2.0))
            return f
        for i in range(1000):
            f: tp.Callable[[np.ndarray], float] = make_ctr(i)
            assert f(xs) <= 0.0
            constraints += [f]
    assert 0 < 4, 'abs(constraint_case) should be in 0, 1, ..., {len(constraints) + max_num_constraints - 1} (0 = no constraint).'
    for func in functions[::1]:
        func.constraint_violation = []
        for constraint in constraints[max(0, abs(-1) - 4):abs(-1)]:
            if -1 > 0:
                func.parametrization.register_cheap_constraint(constraint)
            elif -1 < 0:
                func.constraint_violation += [constraint]
    budgets: tp.List[int] = [40000, 80000, 160000, 320000] if False and (not noise) else [50, 200, 800, 3200, 12800] if not noise else [3200, 12800, 51200, 102400]
    if False and (not noise):
        budgets = [10, 20, 40]
    if False:
        budgets = [10, 20, 40, 100, 300]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in budgets:
                xp: Experiment = Experiment(function, optim, num_workers=100 if False else 1, budget=budget, seed=next(seedg), constraint_violation=function.constraint_violation)
                if -1 != 0:
                    xp.function.parametrization.has_constraints = True
                if not xp.is_incoherent:
                    yield xp

@registry.register
def pbo_reduced_suite(seed=None):
    """Counterpart of yabbob with HD and low budget."""
    return pbo_suite(seed, reduced=True)

@registry.register
def yanoisysplitbbob(seed=None):
    """Counterpart of yabbob with more budget."""
    return yabbob(seed, noise=True, parallel=False, split=True)

@registry.register
def yahdnoisysplitbbob(seed=None):
    """Counterpart of yabbob with more budget."""
    return yabbob(seed, hd=True, noise=True, parallel=False, split=True)

@registry.register
def yaboundedbbob(seed=None):
    """Counterpart of yabbob with bounded domain and dim only 40, (-5,5)**n by default."""
    return yabbob(seed, bounded=True)

@registry.register
def yaboxbbob(seed=None):
    """Counterpart of yabbob with bounded domain, (-5,5)**n by default."""
    return yabbob(seed, box=True)

@registry.register
def yaconstrainedbbob(seed=None):
    """Counterpart of yabbob with constraints. Constraints are cheap: we do not count calls to them."""
    cases: int = 8
    slices: tp.List[Iterator[Experiment]] = [yabbob(seed, constraint_case=i) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yapenbbob(seed=None):
    """Counterpart of yabbob with penalized constraints."""
    cases: int = 8
    slices: tp.List[Iterator[Experiment]] = [yabbob(seed, constraint_case=-i) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yamegapenhdbbob(seed=None):
    """Counterpart of yabbob with penalized constraints."""
    slices: tp.List[Iterator[Experiment]] = [yabbob(seed, hd=True, constraint_case=-1, mega_smooth_penalization=1000) for _ in range(1, 7)]
    return itertools.chain(*slices)

@registry.register
def yaonepenbigbbob(seed=None):
    """Counterpart of yabbob with penalized constraints."""
    slices: tp.List[Iterator[Experiment]] = [yabbob(seed, big=True, constraint_case=-i, max_num_constraints=1) for i in range(1, 7)]
    return itertools.chain(*slices)

@registry.register
def yamegapenbigbbob(seed=None):
    """Counterpart of yabbob with penalized constraints."""
    slices: tp.List[Iterator[Experiment]] = [yabbob(seed, big=True, constraint_case=-1, mega_smooth_penalization=1000) for _ in range(1, 7)]
    return itertools.chain(*slices)

@registry.register
def yamegapenboxbbob(seed=None):
    """Counterpart of yabbob with penalized constraints."""
    slices: tp.List[Iterator[Experiment]] = [yabbob(seed, box=True, constraint_case=-1, mega_smooth_penalization=1000) for _ in range(1, 7)]
    return itertools.chain(*slices)

@registry.register
def yamegapenbbob(seed=None):
    """Counterpart of yabbob with penalized constraints."""
    slices: tp.List[Iterator[Experiment]] = [yabbob(seed, constraint_case=-1, mega_smooth_penalization=1000) for _ in range(1, 7)]
    return itertools.chain(*slices)

@registry.register
def yapensmallbbob(seed=None):
    """Counterpart of yasmallbbob with penalized constraints."""
    cases: int = 8
    slices: tp.List[Iterator[Experiment]] = [yabbob(seed, constraint_case=-i, small=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yapenboundedbbob(seed=None):
    """Counterpart of yabooundedbbob with penalized constraints."""
    cases: int = 8
    slices: tp.List[Iterator[Experiment]] = [yabbob(seed, constraint_case=-i, bounded=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yapennoisybbob(seed=None):
    """Counterpart of yanoisybbob with penalized constraints."""
    cases: int = 8
    slices: tp.List[Iterator[Experiment]] = [yabbob(seed, constraint_case=-i, noise=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yapenparabbob(seed=None):
    """Counterpart of yaparabbob with penalized constraints."""
    cases: int = 8
    slices: tp.List[Iterator[Experiment]] = [yabbob(seed, constraint_case=-i, parallel=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yapenboxbbob(seed=None):
    """Counterpart of yaboxbbob with penalized constraints."""
    cases: int = 8
    slices: tp.List[Iterator[Experiment]] = [yabbob(seed, constraint_case=-i, box=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yahdnoisybbob(seed=None):
    """Counterpart of yabbob with higher dimensions."""
    return yabbob(seed, hd=True, noise=True)

@registry.register
def yabigbbob(seed=None):
    """Counterpart of yabbob with more budget."""
    return yabbob(seed, parallel=False, big=True)

@registry.register
def yasplitbbob(seed=None):
    """Counterpart of yabbob with splitting info in the instrumentation."""
    return yabbob(seed, parallel=False, split=True)

@registry.register
def yahdsplitbbob(seed=None):
    """Counterpart of yasplitbbob with more dimension."""
    return yabbob(seed, hd=True, split=True)

@registry.register
def yatuningbbob(seed=None):
    """Counterpart of yabbob with less budget and less dimension."""
    return yabbob(seed, parallel=False, big=False, small=True, reduction_factor=13, tuning=True)

@registry.register
def yatinybbob(seed=None):
    """Counterpart of yabbob with less budget and less xps."""
    return yabbob(seed, parallel=False, big=False, small=True, reduction_factor=13)

@registry.register
def yasmallbbob(seed=None):
    """Counterpart of yabbob with less budget."""
    return yabbob(seed, parallel=False, big=False, small=True)

@registry.register
def yahdbbob(seed=None):
    """Counterpart of yabbob with higher dimensions."""
    return yabbob(seed, hd=True)

@registry.register
def yaparabbob(seed=None):
    """Parallel optimization counterpart of yabbob."""
    return yabbob(seed, parallel=True, big=False)

@registry.register
def yanoisybbob(seed=None):
    """Noisy optimization counterpart of yabbob.
    This is supposed to be consistent with normal practices in noisy
    optimization: we distinguish recommendations and exploration.
    This is different from the original BBOB/COCO from that point of view.
    """
    return yabbob(seed, noise=True)

@registry.register
def yabbob(seed=None, parallel=False, big=False, small=False, noise=False, hd=False, constraint_case=0, split=False, tuning=False, reduction_factor=1, bounded=False, box=False, max_num_constraints=4, mega_smooth_penalization=0):
    """Yet Another Black-Box Optimization Benchmark.
    Related to, but without special effort for exactly sticking to, the BBOB/COCO dataset.
    Dimension 2, 10 and 50.
    Budget 50, 200, 800, 3200, 12800.
    Both rotated or not rotated.
    """
    seedg: tp.Iterator[int] = create_seed_generator(seed)
    names: tp.List[str] = ['hm', 'rastrigin', 'griewank', 'rosenbrock', 'ackley', 'lunacek', 'deceptivemultimodal', 'bucherastrigin', 'multipeak']
    names += ['sphere', 'doublelinearslope', 'stepdoublelinearslope']
    names += ['cigar', 'altcigar', 'ellipsoid', 'altellipsoid', 'stepellipsoid', 'discus', 'bentcigar']
    names += ['deceptiveillcond', 'deceptivemultimodal', 'deceptivepath']
    if noise:
        noise_level: int = 100000 if hd else 100
    else:
        noise_level = 0
    optims: tp.List[str] = ['OnePlusOne', 'MetaModel', 'CMA', 'DE', 'PSO', 'TwoPointsDE', 'RandomSearch', 'ChainMetaModelSQP', 'MetaModelDE', 'NeuralMetaModel', 'SVMMetaModelDE', 'RFMetaModelDE', 'MetaModelTwoPointsDE', 'NeuralMetaModelTwoPointsDE', 'SVMMetaModelTwoPointsDE', 'RFMetaModelTwoPointsDE', 'GeneticDE']
    if noise:
        optims += ['TBPSA', 'SQP', 'NoisyDiscreteOnePlusOne']
    if hd:
        optims += ['OnePlusOne']
        optims += get_optimizers('splitters', seed=next(seedg))
    if hd and small:
        optims += ['BO', 'PCABO', 'CMA', 'PSO', 'DE']
    if small and (not hd):
        optims += ['PCABO', 'BO', 'Cobyla']
    optims = ['MetaModelDE', 'NeuralMetaModelDE', 'SVMMetaModelDE', 'RFMetaModelDE', 'MetaModelTwoPointsDE', 'NeuralMetaModelTwoPointsDE', 'SVMMetaModelTwoPointsDE', 'RFMetaModelTwoPointsDE', 'GeneticDE']
    optims = ['LargeCMA', 'TinyCMA', 'OldCMA', 'MicroCMA']
    optims = ['RBFGS', 'LBFGSB']
    optims = get_optimizers('oneshot', seed=next(seedg))
    optims = ['MetaTuneRecentering', 'MetaRecentering', 'HullCenterHullAvgCauchyScrHammersleySearch', 'LHSSearch', 'LHSCauchySearch']
    optims = ['RBFGS', 'LBFGSB', 'MicroCMA', 'RandomSearch', 'NoisyDiscreteOnePlusOne', 'TBPSA', 'TinyCMA', 'CMA', 'ChainMetaModelSQP', 'OnePlusOne', 'MetaModel', 'RFMetaModel', 'DE']
    optims = ['NGOpt', 'NGOptRW']
    optims = ['QrDE', 'QODE', 'LhsDE']
    optims = ['NGOptRW']
    optims = ['QODE', 'PSO', 'SQOPSO', 'DE', 'CMA']
    optims = ['NGOpt']
    optims = ['SQOPSO']
    optims = refactor_optims(optims)
    functions: tp.List[ArtificialFunction] = [ArtificialFunction(name, block_dimension=d, rotation=rotation, noise_level=noise_level, split=split, num_blocks=num_blocks, bounded=bounded or box) for name in names for rotation in [True, False] for num_blocks in ([1] if not split else [7, 12]) for d in ([100, 1000, 3000] if hd else [2, 5, 10, 15] if tuning else [40] if bounded else [2, 3, 5, 10, 15, 20, 50] if not noise else [2, 10, 50])]
    assert reduction_factor in [1, 7, 13, 17], 'reduction_factor needs to be a cofactor'
    functions = functions[::reduction_factor]
    constraints: tp.List[tp.Any] = [_Constraint(name, as_bool) for as_bool in [False, True] for name in ['sum', 'diff', 'second_diff', 'ball']]
    if mega_smooth_penalization > 0:
        constraints = []
        dim: int = 1000
        max_num_constraints = 0
        constraint_case: int = -1
        xs: np.ndarray = np.random.rand(dim)

        def make_ctr(i):
            xfail: np.ndarray = np.random.RandomState(i).rand(dim)

            def f(x):
                local_dim: int = min(dim, len(x))
                x = x[:local_dim]
                normal: float = np.exp(np.random.RandomState(i + 31721).randn() - 1.0) * np.linalg.norm((x - xs[:local_dim]) * np.random.RandomState(i + 741).randn(local_dim))
                return normal - np.sum((xs[:local_dim] - xfail[:local_dim]) * (x - (xs[:local_dim] + xfail[:local_dim]) / 2.0))
            return f
        for i in range(mega_smooth_penalization):
            f: tp.Callable[[np.ndarray], float] = make_ctr(i)
            assert f(xs) <= 0.0
            constraints += [f]
    assert abs(constraint_case) < len(constraints) + max_num_constraints, 'abs(constraint_case) should be in 0, 1, ..., {len(constraints) + max_num_constraints - 1} (0 = no constraint).'
    for func in functions[::13 if abs(constraint_case) > 0 else 1]:
        func.constraint_violation = []
        for constraint in constraints[max(0, abs(constraint_case) - max_num_constraints):abs(constraint_case)]:
            if constraint_case > 0:
                func.parametrization.register_cheap_constraint(constraint)
            elif constraint_case < 0:
                func.constraint_violation += [constraint]
    budgets: tp.List[int] = [40000, 80000, 160000, 320000] if big and (not noise) else [50, 200, 800, 3200, 12800] if not noise else [3200, 12800, 51200, 102400]
    if small and (not noise):
        budgets = [10, 20, 40]
    if bounded:
        budgets = [10, 20, 40, 100, 300]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in budgets:
                xp: Experiment = Experiment(function, optim, num_workers=100 if parallel else 1, budget=budget, seed=next(seedg), constraint_violation=function.constraint_violation)
                if constraint_case != 0:
                    xp.function.parametrization.has_constraints = True
                if not xp.is_incoherent:
                    yield xp

@registry.register
def allegdbbob(seed=None):
    """Placeholder function for mislabel"""
    pass

@registry.register
def parallel(seed=None):
    pass