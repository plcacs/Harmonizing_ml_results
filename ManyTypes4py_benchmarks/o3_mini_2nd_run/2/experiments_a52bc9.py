from typing import Optional, Iterator, Union, Iterable
import os
import warnings
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
from nevergrad.functions.photonics import Photonics
from nevergrad.functions.photonics import ceviche as photonics_ceviche
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
from nevergrad.functions import control
from nevergrad.functions import rl
from nevergrad.functions.games import game
from nevergrad.functions import iohprofiler
from nevergrad.functions import helpers
from nevergrad.functions.cycling import Cycling
from .xpbase import Experiment, create_seed_generator, registry

def lsgo() -> Iterator[Experiment]:
    # lsgo returns experiments yielding Experiment objects.
    optims = ['DiagonalCMA', 'TinyQODE', 'OpoDE', 'OpoTinyDE']
    optims = refactor_optims(optims)
    for i in range(1, 16):
        for optim in optims:
            for budget in [120000, 600000, 3000000]:
                yield Experiment(lsgo_makefunction(i).instrumented(), optim, budget=budget)

def smallbudget_lsgo() -> Iterator[Experiment]:
    optims = ['DiagonalCMA', 'TinyQODE', 'OpoDE', 'OpoTinyDE']
    optims = refactor_optims(optims)
    for i in range(1, 16):
        for optim in optims:
            for budget in [1200, 6000, 30000]:
                yield Experiment(lsgo_makefunction(i).instrumented(), optim, budget=budget)

@registry.register
def keras_tuning(seed: Optional[int] = None, overfitter: bool = False, seq: bool = False, veryseq: bool = False) -> Iterator[Experiment]:
    # Implementation remains unchanged; type annotations added to parameters and return.
    seedg = create_seed_generator(seed)
    optims = ['OnePlusOne', 'RandomSearch', 'Cobyla']
    optims = get_optimizers('oneshot', seed=next(seedg))
    optims = ['MetaTuneRecentering', 'MetaRecentering', 'HullCenterHullAvgCauchyScrHammersleySearch', 'LHSSearch', 'LHSCauchySearch']
    optims = ['NGOpt', 'NGOptRW', 'QODE']
    optims = ['NGOpt']
    optims = ['PCABO', 'NGOpt', 'QODE']
    optims = ['QOPSO']
    optims = ['SQOPSO']
    optims = refactor_optims(optims)
    datasets = ['kerasBoston', 'diabetes', 'auto-mpg', 'red-wine', 'white-wine']
    optims = refactor_optims(optims)
    for dimension in [None]:
        for dataset in datasets:
            function = MLTuning(regressor='keras_dense_nn', data_dimension=dimension, dataset=dataset, overfitter=overfitter)
            for budget in [150, 500]:
                for num_workers in ([1, budget // 4] if seq else [budget]):
                    if veryseq and num_workers > 1:
                        continue
                    for optim in optims:
                        xp = Experiment(function, optim, num_workers=num_workers, budget=budget, seed=next(seedg))
                        skip_ci(reason='too slow')
                        xp.function.parametrization.real_world = True
                        xp.function.parametrization.hptuning = True
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def mltuning(seed: Optional[int] = None, overfitter: bool = False, seq: bool = False, veryseq: bool = False, nano: bool = False) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ['OnePlusOne', 'RandomSearch', 'Cobyla']
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
                function = MLTuning(regressor=regressor, data_dimension=dimension, dataset=dataset, overfitter=overfitter)
                for budget in ([150, 500] if not nano else [80, 160]):
                    parallelization = ([1, budget // 4] if seq else [budget])
                    for num_workers in parallelization:
                        if veryseq and num_workers > 1:
                            continue
                        for optim in optims:
                            xp = Experiment(function, optim, num_workers=num_workers, budget=budget, seed=next(seedg))
                            skip_ci(reason='too slow')
                            xp.function.parametrization.real_world = True
                            xp.function.parametrization.hptuning = True
                            if not xp.is_incoherent:
                                yield xp

@registry.register
def naivemltuning(seed: Optional[int] = None) -> Iterator[Experiment]:
    return mltuning(seed, overfitter=True)

@registry.register
def veryseq_keras_tuning(seed: Optional[int] = None) -> Iterator[Experiment]:
    return keras_tuning(seed, overfitter=False, seq=True, veryseq=True)

@registry.register
def seq_keras_tuning(seed: Optional[int] = None) -> Iterator[Experiment]:
    return keras_tuning(seed, overfitter=False, seq=True)

@registry.register
def naive_seq_keras_tuning(seed: Optional[int] = None) -> Iterator[Experiment]:
    return keras_tuning(seed, overfitter=True, seq=True)

@registry.register
def naive_veryseq_keras_tuning(seed: Optional[int] = None) -> Iterator[Experiment]:
    return keras_tuning(seed, overfitter=True, seq=True, veryseq=True)

@registry.register
def oneshot_mltuning(seed: Optional[int] = None) -> Iterator[Experiment]:
    return mltuning(seed, overfitter=False, seq=False)

@registry.register
def seq_mltuning(seed: Optional[int] = None) -> Iterator[Experiment]:
    return mltuning(seed, overfitter=False, seq=True)

@registry.register
def nano_seq_mltuning(seed: Optional[int] = None) -> Iterator[Experiment]:
    return mltuning(seed, overfitter=False, seq=True, nano=True)

@registry.register
def nano_veryseq_mltuning(seed: Optional[int] = None) -> Iterator[Experiment]:
    return mltuning(seed, overfitter=False, seq=True, nano=True, veryseq=True)

@registry.register
def nano_naive_veryseq_mltuning(seed: Optional[int] = None) -> Iterator[Experiment]:
    return mltuning(seed, overfitter=True, seq=True, nano=True, veryseq=True)

@registry.register
def nano_naive_seq_mltuning(seed: Optional[int] = None) -> Iterator[Experiment]:
    return mltuning(seed, overfitter=True, seq=True, nano=True)

@registry.register
def naive_seq_mltuning(seed: Optional[int] = None) -> Iterator[Experiment]:
    return mltuning(seed, overfitter=True, seq=True)

@registry.register
def yawidebbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    total_xp_per_optim = 0
    functions = [ArtificialFunction(name, block_dimension=50, rotation=rotation, translation_factor=tf) 
                 for name in ['cigar', 'ellipsoid'] 
                 for rotation in [True, False] 
                 for tf in [0.1, 10.0]]
    for i, func in enumerate(functions):
        func.parametrization.register_cheap_constraint(_Constraint('sum', as_bool=i % 2 == 0))
    assert len(functions) == 8
    names = ['hm', 'rastrigin', 'sphere', 'doublelinearslope', 'ellipsoid']
    functions += [ArtificialFunction(name, block_dimension=d, rotation=rotation, noise_level=nl, split=split, translation_factor=tf, num_blocks=num_blocks) 
                  for name in names 
                  for rotation in [True, False] 
                  for nl in [0.0, 100.0] 
                  for tf in [0.1, 10.0] 
                  for num_blocks in [1, 8] 
                  for d in [5, 70, 10000] 
                  for split in [True, False]][::37]
    assert len(functions) == 21, f'{len(functions)} problems instead of 21. Yawidebbob should be standard.'
    optims = ['NGOptRW', 'NGOpt', 'RandomSearch', 'CMA', 'DE', 'DiscreteLenglerOnePlusOne']
    optims = refactor_optims(optims)
    index = 0
    for function in functions:
        for budget in [50, 1500, 25000]:
            for nw in [1, budget] + ([] if budget <= 300 else [300]):
                index += 1
                if index % 5 == 0:
                    total_xp_per_optim += 1
                    for optim in optims:
                        xp = Experiment(function, optim, num_workers=nw, budget=budget, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp
    assert total_xp_per_optim == 33, f'We have 33 single-obj xps per optimizer (got {total_xp_per_optim}).'
    index = 0
    for nv in [200, 2000]:
        for arity in [2, 7, 37]:
            instrum = ng.p.TransitionChoice(range(arity), repetitions=nv)
            for name in ['onemax', 'leadingones', 'jump']:
                index += 1
                if index % 4 != 0:
                    continue
                dfunc = ExperimentFunction(corefuncs.DiscreteFunction(name, arity), instrum.set_name('transition'))
                dfunc.add_descriptors(arity=arity)
                for budget in [500, 1500, 5000]:
                    for nw in [1, 100]:
                        total_xp_per_optim += 1
                        for optim in optims:
                            yield Experiment(dfunc, optim, num_workers=nw, budget=budget, seed=next(seedg))
    assert total_xp_per_optim == 57, f'Including discrete, we check xps per optimizer (got {total_xp_per_optim}).'
    mofuncs = []
    for name1 in ['sphere', 'ellipsoid']:
        for name2 in ['sphere', 'hm']:
            for tf in [0.25, 4.0]:
                mofuncs += [fbase.MultiExperiment([ArtificialFunction(name1, block_dimension=7), ArtificialFunction(name2, block_dimension=7, translation_factor=tf)], upper_bounds=np.array((100.0, 100.0)))]
                mofuncs[-1].add_descriptors(num_objectives=2)
    for name1 in ['sphere', 'ellipsoid']:
        for name2 in ['sphere', 'hm']:
            for name3 in ['sphere', 'hm']:
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
                        yield Experiment(mofunc, optim, budget=budget, num_workers=nw, seed=next(seedg))
    assert total_xp_per_optim == 71, f'We should have 71 xps per optimizer, not {total_xp_per_optim}.'

@registry.register
def parallel_small_budget(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ['DE', 'TwoPointsDE', 'CMA', 'NGOpt', 'PSO', 'OnePlusOne', 'RandomSearch']
    names = ['hm', 'rastrigin', 'griewank', 'rosenbrock', 'ackley', 'multipeak']
    names += ['sphere', 'cigar', 'ellipsoid', 'altellipsoid']
    names += ['deceptiveillcond', 'deceptivemultimodal', 'deceptivepath']
    functions = [ArtificialFunction(name, block_dimension=d, rotation=rotation) 
                 for name in names 
                 for rotation in [True, False] 
                 for d in [2, 4, 8]]
    budgets = [10, 50, 100, 200, 400]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in budgets:
                for nw in [2, 8, 16]:
                    for batch in [True, False]:
                        if nw < budget / 4:
                            xp = Experiment(function, optim, num_workers=nw, budget=budget, batch_mode=batch, seed=next(seedg))
                            if not xp.is_incoherent:
                                yield xp

@registry.register
def instrum_discrete(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ['DiscreteOnePlusOne', 'NGOpt', 'CMA', 'TwoPointsDE', 'DiscreteLenglerOnePlusOne']
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
                    instrum = ng.p.Choice(range(arity), repetitions=nv)
                else:
                    instrum = ng.p.TransitionChoice(range(arity), repetitions=nv, ordered=(instrum_str == 'Ordered'))
                for name in ['onemax', 'leadingones', 'jump']:
                    dfunc = ExperimentFunction(corefuncs.DiscreteFunction(name, arity), instrum.set_name(instrum_str))
                    dfunc.add_descriptors(arity=arity)
                    dfunc.add_descriptors(nv=nv)
                    dfunc.add_descriptors(instrum_str=instrum_str)
                    for optim in optims:
                        for nw in [1, 10]:
                            for budget in [50, 500, 5000]:
                                yield Experiment(dfunc, optim, num_workers=nw, budget=budget, seed=next(seedg))

@registry.register
def sequential_instrum_discrete(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ['DiscreteOnePlusOne', 'NGOpt', 'CMA', 'TwoPointsDE', 'DiscreteLenglerOnePlusOne']
    optims = ['OnePlusOne']
    optims = ['DiscreteLenglerOnePlusOne']
    optims = ['NGOpt', 'NGOptRW']
    optims = [l for l in list(ng.optimizers.registry.keys()) if 'DiscreteOneP' in l and 'SA' not in l and ('Smooth' not in l) and ('Noisy' not in l) and ('Optimis' not in l) and (l[-1] != 'T')] + ['cGA', 'DiscreteDE']
    optims = refactor_optims(optims)
    for nv in [10, 50, 200, 1000, 5000]:
        for arity in [2, 3, 7, 30]:
            for instrum_str in ['Unordered', 'Softmax', 'Ordered']:
                if instrum_str == 'Softmax':
                    instrum = ng.p.Choice(range(arity), repetitions=nv)
                else:
                    instrum = ng.p.TransitionChoice(range(arity), repetitions=nv, ordered=(instrum_str == 'Ordered'))
                for name in ['onemax', 'leadingones', 'jump']:
                    dfunc = ExperimentFunction(corefuncs.DiscreteFunction(name, arity), instrum.set_name(instrum_str))
                    dfunc.add_descriptors(arity=arity)
                    dfunc.add_descriptors(nv=nv)
                    dfunc.add_descriptors(instrum_str=instrum_str)
                    for optim in optims:
                        for budget in [50, 500, 5000, 50000]:
                            yield Experiment(dfunc, optim, budget=budget, seed=next(seedg))

@registry.register
def deceptive(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    names = ['deceptivemultimodal', 'deceptiveillcond', 'deceptivepath']
    optims = ['CMA', 'DE', 'TwoPointsDE', 'PSO', 'OnePlusOne', 'RandomSearch', 'NGOptRW']
    optims = ['RBFGS', 'LBFGSB', 'DE', 'TwoPointsDE', 'RandomSearch', 'OnePlusOne', 'PSO', 'CMA', 'ChainMetaModelSQP', 'MemeticDE', 'MetaModel', 'RFMetaModel', 'MetaModelDE', 'RFMetaModelDE']
    optims = ['NGOpt']
    functions = [ArtificialFunction(name, block_dimension=2, num_blocks=n_blocks, rotation=rotation, aggregator=aggregator) 
                 for name in names 
                 for rotation in [False, True] 
                 for n_blocks in [1, 2, 8, 16] 
                 for aggregator in ['sum', 'max']]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [25, 37, 50, 75, 87, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))

@registry.register
def lowbudget(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    names = ['sphere', 'rastrigin', 'cigar']
    optims = ['AX', 'BOBYQA', 'Cobyla', 'RandomSearch', 'CMA', 'NGOpt', 'DE', 'PSO', 'pysot', 'negpysot']
    functions = [ArtificialFunction(name, block_dimension=bd, bounded=b) for name in names for bd in [7] for b in [True, False]]
    for func in functions:
        for optim in optims:
            for budget in [10, 20, 30]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))

@registry.register
def parallel(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    names = ['sphere', 'rastrigin', 'cigar']
    optims = get_optimizers('parallel_basics', seed=next(seedg))
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) 
                 for name in names 
                 for bd in [25] 
                 for uv_factor in [0, 5]]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=int(budget / 5), seed=next(seedg))

@registry.register
def harderparallel(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    names = ['sphere', 'rastrigin', 'cigar', 'ellipsoid']
    optims = ['NGOpt10'] + get_optimizers('emna_variants', seed=next(seedg))
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) 
                 for name in names 
                 for bd in [5, 25] 
                 for uv_factor in [0, 5]]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000]:
                for num_workers in [int(budget / 10), int(budget / 5), int(budget / 3)]:
                    yield Experiment(func, optim, budget=budget, num_workers=num_workers, seed=next(seedg))

@registry.register
def oneshot(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    names = ['sphere', 'rastrigin', 'cigar']
    optims = get_optimizers('oneshot', seed=next(seedg))
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) 
                 for name in names 
                 for bd in [3, 10, 30, 100, 300, 1000, 3000] 
                 for uv_factor in [0]]
    for func in functions:
        for optim in optims:
            for budget in [100000, 30, 100, 300, 1000, 3000, 10000]:
                if func.dimension < 3000 or budget < 100000:
                    yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))

@registry.register
def doe(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    names = ['sphere', 'rastrigin', 'cigar']
    optims = get_optimizers('oneshot', seed=next(seedg))
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) 
                 for name in names 
                 for bd in [2000, 20000] 
                 for uv_factor in [0]]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000, 30000, 100000]:
                yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))

@registry.register
def newdoe(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    names = ['sphere', 'rastrigin', 'cigar']
    optims = get_optimizers('oneshot', seed=next(seedg))
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) 
                 for name in names 
                 for bd in [2000, 20, 200, 20000] 
                 for uv_factor in [0]]
    budgets = [30, 100, 3000, 10000, 30000, 100000, 300000]
    for func in functions:
        for optim in optims:
            for budget in budgets:
                yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))

@registry.register
def fiveshots(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    names = ['sphere', 'rastrigin', 'cigar']
    optims = get_optimizers('oneshot', 'basics', seed=next(seedg))
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) 
                 for name in names 
                 for bd in [3, 25] 
                 for uv_factor in [0, 5]]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=budget // 5, seed=next(seedg))

@registry.register
def multimodal(seed: Optional[int] = None, para: bool = False) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    names = ['hm', 'rastrigin', 'griewank', 'rosenbrock', 'ackley', 'lunacek', 'deceptivemultimodal']
    optims = get_optimizers('basics', seed=next(seedg))
    if not para:
        optims += get_optimizers('scipy', seed=next(seedg))
    optims = ['RBFGS', 'LBFGSB', 'DE', 'TwoPointsDE', 'RandomSearch', 'OnePlusOne', 'PSO', 'CMA', 'ChainMetaModelSQP', 'MemeticDE', 'MetaModel', 'RFMetaModel', 'MetaModelDE', 'RFMetaModelDE']
    optims = ['NGOpt']
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) 
                 for name in names 
                 for bd in [3, 25] 
                 for uv_factor in [0, 5]]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [3000, 10000, 30000, 100000]:
                for nw in ([1000] if para else [1]):
                    xp = Experiment(func, optim, budget=budget, num_workers=nw, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp

@registry.register
def hdmultimodal(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    names = ['hm', 'rastrigin', 'griewank', 'rosenbrock', 'ackley', 'lunacek', 'deceptivemultimodal']
    optims = get_optimizers('basics', 'multimodal', seed=next(seedg))
    optims = ['RBFGS', 'LBFGSB', 'DE', 'TwoPointsDE', 'RandomSearch', 'OnePlusOne', 'PSO', 'CMA', 'ChainMetaModelSQP', 'MemeticDE', 'MetaModel', 'RFMetaModel', 'MetaModelDE', 'RFMetaModelDE']
    functions = [ArtificialFunction(name, block_dimension=bd) for name in names for bd in [1000, 6000, 36000]]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [3000, 10000]:
                for nw in [1]:
                    yield Experiment(func, optim, budget=budget, num_workers=nw, seed=next(seedg))

@registry.register
def paramultimodal(seed: Optional[int] = None) -> Iterator[Experiment]:
    return multimodal(seed, para=True)

@registry.register
def bonnans(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    instrum = ng.p.TransitionChoice(range(2), repetitions=100, ordered=False)
    softmax_instrum = ng.p.Choice(range(2), repetitions=100)
    optims = ['RotatedTwoPointsDE', 'DiscreteLenglerOnePlusOne', 'DiscreteLengler2OnePlusOne', 'DiscreteLengler3OnePlusOne', 'DiscreteLenglerHalfOnePlusOne', 'DiscreteLenglerFourthOnePlusOne', 'PortfolioDiscreteOnePlusOne', 'FastGADiscreteOnePlusOne', 'DiscreteDoerrOnePlusOne', 'DiscreteBSOOnePlusOne', 'DiscreteOnePlusOne', 'AdaptiveDiscreteOnePlusOne', 'GeneticDE', 'DE', 'TwoPointsDE', 'DiscreteOnePlusOne', 'CMA', 'SQP', 'MetaModel', 'DiagonalCMA']
    optims = ['RFMetaModelOnePlusOne']
    optims = ['MemeticDE', 'cGA', 'DoubleFastGADiscreteOnePlusOne', 'FastGADiscreteOnePlusOne']
    optims = ['NGOpt', 'NGOptRW']
    optims = refactor_optims(optims)
    for i in range(21):
        bonnans = corefuncs.BonnansFunction(index=i)
        for optim in optims:
            instrum_str = 'TransitionChoice' if 'Discrete' in optim else 'Softmax'
            dfunc = ExperimentFunction(bonnans, instrum if instrum_str == 'TransitionChoice' else softmax_instrum)
            dfunc.add_descriptors(index=i)
            dfunc.add_descriptors(instrum_str=instrum_str)
            for budget in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                yield Experiment(dfunc, optim, num_workers=1, budget=budget, seed=next(seedg))

@registry.register
def yabbob(seed: Optional[int] = None, parallel: bool = False, big: bool = False, small: bool = False, noise: bool = False, hd: bool = False, constraint_case: int = 0, split: bool = False, tuning: bool = False, reduction_factor: int = 1, bounded: bool = False, box: bool = False, max_num_constraints: int = 4, mega_smooth_penalization: int = 0) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    names = ['hm', 'rastrigin', 'griewank', 'rosenbrock', 'ackley', 'lunacek', 'deceptivemultimodal', 'bucherastrigin', 'multipeak']
    names += ['sphere', 'doublelinearslope', 'stepdoublelinearslope']
    names += ['cigar', 'altcigar', 'ellipsoid', 'altellipsoid', 'stepellipsoid', 'discus', 'bentcigar']
    names += ['deceptiveillcond', 'deceptivemultimodal', 'deceptivepath']
    if noise:
        noise_level = 100000 if hd else 100
    else:
        noise_level = 0
    optims = ['OnePlusOne', 'MetaModel', 'CMA', 'DE', 'PSO', 'TwoPointsDE', 'RandomSearch', 'ChainMetaModelSQP', 'NeuralMetaModel', 'MetaModelDE', 'MetaModelOnePlusOne']
    if noise:
        optims += ['TBPSA', 'SQP', 'NoisyDiscreteOnePlusOne']
    if hd:
        optims += ['OnePlusOne']
        optims += get_optimizers('splitters', seed=next(seedg))
    if hd and small:
        optims += ['BO', 'PCABO', 'CMA', 'PSO', 'DE']
    if small and (not hd):
        optims += ['PCABO', 'BO', 'Cobyla']
    optims = ['MetaModelDE', 'MetaModelOnePlusOne', 'OnePlusOne', 'ChainMetaModelSQP', 'RFMetaModel', 'RFMetaModelDE']
    optims = ['MetaModelDE', 'NeuralMetaModelDE', 'SVMMetaModelDE', 'RFMetaModelDE', 'MetaModelTwoPointsDE', 'NeuralMetaModelTwoPointsDE', 'SVMMetaModelTwoPointsDE', 'RFMetaModelTwoPointsDE', 'GeneticDE']
    optims = ['LargeCMA', 'TinyCMA', 'OldCMA', 'MicroCMA']
    optims = ['RBFGS', 'LBFGSB']
    optims = get_optimizers('oneshot', seed=next(seedg))
    optims = ['MetaTuneRecentering', 'MetaRecentering', 'HullCenterHullAvgCauchyScrHammersleySearch', 'LHSSearch', 'LHSCauchySearch']
    optims = ['RBFGS', 'LBFGSB', 'MicroCMA', 'RandomSearch', 'NoisyDiscreteOnePlusOne', 'TBPSA', 'TinyCMA', 'CMA', 'ChainMetaModelSQP', 'OnePlusOne', 'MetaModel', 'RFMetaModel', 'DE']
    optims = ['NGOpt', 'NGOptRW']
    optims = ['QrDE', 'QODE', 'LhsDE']
    optims = ['NGOptRW']
    if noise:
        optims = ['NoisyOnePlusOne']
    else:
        optims = ['MetaModelPSO', 'RFMetaModelPSO', 'SVMMetaModelPSO']
    optims = ['PCABO']
    optims = ['PCABO', 'NGOpt', 'QODE']
    optims = ['QOPSO']
    optims = ['NGOpt']
    optims = ['SQOPSO']
    functions = [ArtificialFunction(name, block_dimension=d, rotation=rotation, noise_level=noise_level, split=split, num_blocks=num_blocks, bounded=bounded or box) 
                 for name in names 
                 for rotation in [True, False] 
                 for num_blocks in ([1] if not split else [7, 12]) 
                 for d in (([100, 1000, 3000] if hd else [2, 5, 10, 15] if tuning else [40] if bounded else ([2, 3, 5, 10, 15, 20, 50] if noise else [2, 10, 50])))]
    assert reduction_factor in [1, 7, 13, 17]
    functions = functions[::reduction_factor]
    constraints = [_Constraint(name, as_bool) for as_bool in [False, True] for name in ['sum', 'diff', 'second_diff', 'ball']]
    if mega_smooth_penalization > 0:
        constraints = []
        dim = 1000
        max_num_constraints = mega_smooth_penalization
        constraint_case = -abs(constraint_case)
        xs = np.random.rand(dim)
        def make_ctr(i: int):
            xfail = np.random.RandomState(i).rand(dim)
            def f(x: np.ndarray) -> float:
                local_dim = min(dim, len(x))
                x_local = x[:local_dim]
                normal = np.exp(np.random.RandomState(i + 31721).randn() - 1.0) * np.linalg.norm((x_local - xs[:local_dim]) * np.random.RandomState(i + 741).randn(local_dim))
                return normal - np.sum((xs[:local_dim] - xfail[:local_dim]) * (x_local - (xs[:local_dim] + xfail[:local_dim]) / 2.0))
            return f
        for i in range(mega_smooth_penalization):
            f = make_ctr(i)
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
    budgets = ([40000, 80000, 160000, 320000] if big and (not noise) else ([50, 200, 800, 3200, 12800] if not noise else [3200, 12800, 51200, 102400]))
    if small and (not noise):
        budgets = [10, 20, 40]
    if bounded:
        budgets = [10, 20, 40, 100, 300]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in budgets:
                xp = Experiment(function, optim, num_workers=(100 if parallel else 1), budget=budget, seed=next(seedg), constraint_violation=function.constraint_violation)
                if constraint_case != 0:
                    xp.function.parametrization.has_constraints = True
                if not xp.is_incoherent:
                    yield xp

@registry.register
def yahdlbbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, hd=True, small=True)

@registry.register
def reduced_yahdlbbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, hd=True, small=True, reduction_factor=17)

@registry.register
def yanoisysplitbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, noise=True, parallel=False, split=True)

@registry.register
def yahdnoisysplitbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, hd=True, noise=True, parallel=False, split=True)

@registry.register
def yaconstrainedbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    cases = 8
    slices = [yabbob(seed, constraint_case=i) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yapenbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    cases = 8
    slices = [yabbob(seed, constraint_case=-i) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yamegapenhdbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    slices = [yabbob(seed, hd=True, constraint_case=-1, mega_smooth_penalization=1000) for i in range(1, 7)]
    return itertools.chain(*slices)

@registry.register
def yaonepenbigbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    slices = [yabbob(seed, big=True, constraint_case=-i, max_num_constraints=1) for i in range(1, 7)]
    return itertools.chain(*slices)

@registry.register
def yamegapenbigbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    slices = [yabbob(seed, big=True, constraint_case=-1, mega_smooth_penalization=1000) for i in range(1, 7)]
    return itertools.chain(*slices)

@registry.register
def yamegapenboxbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    slices = [yabbob(seed, box=True, constraint_case=-1, mega_smooth_penalization=1000) for i in range(1, 7)]
    return itertools.chain(*slices)

@registry.register
def yamegapenbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    slices = [yabbob(seed, constraint_case=-1, mega_smooth_penalization=1000) for i in range(1, 7)]
    return itertools.chain(*slices)

@registry.register
def yamegapenboundedbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    slices = [yabbob(seed, bounded=True, constraint_case=-1, mega_smooth_penalization=1000) for i in range(1, 7)]
    return itertools.chain(*slices)

@registry.register
def yapensmallbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    cases = 8
    slices = [yabbob(seed, constraint_case=-i, small=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yapenboundedbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    cases = 8
    slices = [yabbob(seed, constraint_case=-i, bounded=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yapennoisybbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    cases = 8
    slices = [yabbob(seed, constraint_case=-i, noise=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yapenparabbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    cases = 8
    slices = [yabbob(seed, constraint_case=-i, parallel=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yapenboxbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    cases = 8
    slices = [yabbob(seed, constraint_case=-i, box=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yaonepenbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    cases = 8
    slices = [yabbob(seed, max_num_constraints=1, constraint_case=-i) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yaonepensmallbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    cases = 8
    slices = [yabbob(seed, max_num_constraints=1, constraint_case=-i, small=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yaonepenboundedbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    cases = 8
    slices = [yabbob(seed, max_num_constraints=1, constraint_case=-i, bounded=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yaonepennoisybbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    cases = 8
    slices = [yabbob(seed, max_num_constraints=1, constraint_case=-i, noise=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yaonepenparabbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    cases = 8
    slices = [yabbob(seed, max_num_constraints=1, constraint_case=-i, parallel=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yaonepenboxbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    cases = 8
    slices = [yabbob(seed, max_num_constraints=1, constraint_case=-i, box=True) for i in range(1, cases)]
    return itertools.chain(*slices)

@registry.register
def yahdnoisybbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, hd=True, noise=True)

@registry.register
def yabigbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, parallel=False, big=True)

@registry.register
def yasplitbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, parallel=False, split=True)

@registry.register
def yahdsplitbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, hd=True, split=True)

@registry.register
def yatuningbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, parallel=False, big=False, small=True, reduction_factor=13, tuning=True)

@registry.register
def yatinybbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, parallel=False, big=False, small=True, reduction_factor=13)

@registry.register
def yasmallbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, parallel=False, big=False, small=True)

@registry.register
def yahdbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, hd=True)

@registry.register
def yaparabbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, parallel=True, big=False)

@registry.register
def yanoisybbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, noise=True)

@registry.register
def yaboundedbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, bounded=True)

@registry.register
def yaboxbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    return yabbob(seed, box=True)

@registry.register
def ms_bbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ['QODE']
    optims = ['CMA', 'LargeCMA', 'OldCMA', 'DE', 'PSO', 'Powell', 'Cobyla', 'SQP']
    optims = ['QOPSO', 'QORealSpacePSO']
    optims = ['SQOPSO']
    dims = [2, 3, 5, 10, 20]
    functions = [ArtificialFunction(name, block_dimension=d, rotation=rotation, expo=expo, translation_factor=tf) 
                 for name in ['cigar', 'sphere', 'rastrigin'] 
                 for rotation in [True] 
                 for expo in [1.0, 5.0] 
                 for tf in [0.01, 0.1, 1.0, 10.0] 
                 for d in dims]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 400, 800, 1600, 3200]:
                for nw in [1]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))

@registry.register
def zp_ms_bbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ['QODE']
    optims = ['CMA', 'LargeCMA', 'OldCMA', 'DE', 'PSO', 'Powell', 'Cobyla', 'SQP']
    optims = ['QOPSO', 'QORealSpacePSO']
    optims = ['SQOPSO']
    dims = [2, 3, 5, 10, 20]
    functions = [ArtificialFunction(name, block_dimension=d, rotation=rotation, expo=expo, translation_factor=tf, zero_pen=True) 
                 for name in ['cigar', 'sphere', 'rastrigin'] 
                 for rotation in [True] 
                 for expo in [1.0, 5.0] 
                 for tf in [0.01, 0.1, 1.0, 10.0] 
                 for d in dims]
    optims = ['QODE', 'PSO', 'SQOPSO', 'DE', 'CMA']
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 300, 400, 500, 600, 700, 800]:
                for nw in [1, 10, 50]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))

def nozp_noms_bbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ['QODE']
    optims = ['CMA', 'LargeCMA', 'OldCMA', 'DE', 'PSO', 'Powell', 'Cobyla', 'SQP']
    optims = ['QOPSO', 'QORealSpacePSO']
    optims = ['SQOPSO']
    dims = [2, 3, 5, 10, 20]
    functions = [ArtificialFunction(name, block_dimension=d, rotation=rotation, expo=expo, translation_factor=tf, zero_pen=False) 
                 for name in ['cigar', 'sphere', 'rastrigin'] 
                 for rotation in [True] 
                 for expo in [1.0, 5.0] 
                 for tf in [1.0] 
                 for d in dims]
    optims = ['QODE', 'PSO', 'SQOPSO', 'DE', 'CMA']
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 400, 800, 1600, 3200]:
                for nw in [1]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))

@registry.register
def pbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ['ChainMetaModelSQP', 'MetaModelOnePlusOne', 'MetaModelDE']
    optims = ['LargeCMA', 'TinyCMA', 'OldCMA', 'MicroCMA']
    optims = ['RBFGS', 'LBFGSB', 'MemeticDE']
    optims = ['QrDE', 'QODE', 'LhsDE', 'NGOpt', 'NGOptRW']
    optims = ['TinyCMA', 'QODE', 'MetaModelOnePlusOne', 'LhsDE', 'TinyLhsDE', 'TinyQODE']
    optims = ['QOPSO', 'QORealSpacePSO']
    optims = ['SQOPSO']
    dims = [40, 20]
    functions = [ArtificialFunction(name, block_dimension=d, rotation=rotation, expo=expo) 
                 for name in ['cigar', 'sphere', 'rastrigin', 'hm', 'deceptivemultimodal'] 
                 for rotation in [True] 
                 for expo in [1.0, 3.0, 5.0, 7.0, 9.0] 
                 for d in dims]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 300, 400, 500, 600, 700, 800]:
                for nw in [1, 10, 50]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))

@registry.register
def zp_pbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ['ChainMetaModelSQP', 'MetaModelOnePlusOne', 'MetaModelDE']
    optims = ['LargeCMA', 'TinyCMA', 'OldCMA', 'MicroCMA']
    optims = ['RBFGS', 'LBFGSB', 'MemeticDE']
    optims = ['QrDE', 'QODE', 'LhsDE', 'NGOpt', 'NGOptRW']
    optims = ['TinyCMA', 'QODE', 'MetaModelOnePlusOne', 'LhsDE', 'TinyLhsDE', 'TinyQODE']
    optims = ['QOPSO', 'QORealSpacePSO']
    optims = ['SQOPSO']
    dims = [40, 20]
    functions = [ArtificialFunction(name, block_dimension=d, rotation=rotation, expo=expo, zero_pen=True) 
                 for name in ['cigar', 'sphere', 'rastrigin', 'hm', 'deceptivemultimodal'] 
                 for rotation in [True] 
                 for expo in [1.0, 3.0, 5.0, 7.0, 9.0] 
                 for d in dims]
    optims = ['QODE', 'PSO', 'SQOPSO', 'DE', 'CMA']
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 300, 400, 500, 600, 700, 800]:
                for nw in [1, 10, 50]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))

@registry.register
def illcondi(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = get_optimizers('basics', seed=next(seedg))
    functions = [ArtificialFunction(name, block_dimension=50, rotation=rotation) 
                 for name in ['cigar', 'ellipsoid'] 
                 for rotation in [True, False]]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 1000, 10000]:
                yield Experiment(function, optim, budget=budget, num_workers=1, seed=next(seedg))

@registry.register
def illcondipara(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    functions = [ArtificialFunction(name, block_dimension=50, rotation=rotation) 
                 for name in ['cigar', 'ellipsoid'] 
                 for rotation in [True, False]]
    optims = get_optimizers('competitive', seed=next(seedg))
    optims = refactor_optims(optims)
    for function in functions:
        for budget in [100, 1000, 10000]:
            for optim in optims:
                xp = Experiment(function, optim, budget=budget, num_workers=50, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp

@registry.register
def constrained_illconditioned_parallel(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    functions = [ArtificialFunction(name, block_dimension=50, rotation=rotation) 
                 for name in ['cigar', 'ellipsoid'] 
                 for rotation in [True, False]]
    for func in functions:
        func.parametrization.register_cheap_constraint(_Constraint('sum', as_bool=False))
    optims = ['DE', 'CMA', 'NGOpt']
    optims = refactor_optims(optims)
    for function in functions:
        for budget in [400, 4000, 40000]:
            optims = get_optimizers('large', seed=next(seedg))
            for optim in optims:
                yield Experiment(function, optim, budget=budget, num_workers=1, seed=next(seedg))

@registry.register
def ranknoisy(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = get_optimizers('progressive', seed=next(seedg)) + ['OptimisticNoisyOnePlusOne', 'OptimisticDiscreteOnePlusOne', 'NGOpt10']
    optims = ['SPSA', 'TinySPSA', 'TBPSA', 'NoisyOnePlusOne', 'NoisyDiscreteOnePlusOne']
    optims = get_optimizers('basics', 'noisy', 'splitters', 'progressive', seed=next(seedg))
    optims = refactor_optims(optims)
    for budget in [25000, 50000, 100000]:
        for optim in optims:
            for d in [20000, 200, 2000]:
                for name in ['cigar', 'altcigar', 'ellipsoid', 'altellipsoid']:
                    for noise_dissymmetry in [False, True]:
                        function = ArtificialFunction(name=name, rotation=False, block_dimension=d, noise_level=10, noise_dissymmetry=noise_dissymmetry, translation_factor=1.0)
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))

@registry.register
def noisy(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = get_optimizers('progressive', seed=next(seedg)) + ['OptimisticNoisyOnePlusOne', 'OptimisticDiscreteOnePlusOne']
    optims += ['NGOpt10', 'Shiwa', 'DiagonalCMA']
    optims += sorted((x for x, y in ng.optimizers.registry.items() if 'SPSA' in x or 'TBPSA' in x or 'ois' in x or ('epea' in x) or ('Random' in x)))
    optims = refactor_optims(optims)
    for budget in [25000, 50000, 100000]:
        for optim in optims:
            for d in [2, 20, 200, 2000]:
                for name in ['sphere', 'rosenbrock', 'cigar', 'hm']:
                    for noise_dissymmetry in [False, True]:
                        function = ArtificialFunction(name=name, rotation=True, block_dimension=d, noise_level=10, noise_dissymmetry=noise_dissymmetry, translation_factor=1.0)
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))

@registry.register
def paraalldes(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    for budget in [10, 100, 1000, 10000, 100000]:
        for optim in sorted((x for x, y in ng.optimizers.registry.items() if 'DE' in x and 'Tune' in x)):
            for rotation in [False]:
                for d in [5, 20, 100, 500, 2500]:
                    for name in ['sphere', 'cigar', 'hm', 'ellipsoid']:
                        for u in [0]:
                            function = ArtificialFunction(name=name, rotation=rotation, block_dimension=d, useless_variables=d * u, translation_factor=1.0)
                            yield Experiment(function, optim, budget=budget, seed=next(seedg), num_workers=max(d, budget // 6))

@registry.register
def parahdbo4d(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    for budget in [25, 31, 37, 43, 50, 60]:
        for optim in refactor_optims(sorted((x for x, y in ng.optimizers.registry.items() if 'BO' in x and 'Tune' in x))):
            for rotation in [False]:
                for d in [20, 2000]:
                    for name in ['sphere', 'cigar', 'hm', 'ellipsoid']:
                        for u in [0]:
                            function = ArtificialFunction(name=name, rotation=rotation, block_dimension=d, useless_variables=d * u, translation_factor=1.0)
                            yield Experiment(function, optim, budget=budget, seed=next(seedg), num_workers=max(d, budget // 6))

@registry.register
def alldes(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    for budget in [10, 100, 1000, 10000, 100000]:
        for optim in refactor_optims(sorted((x for x, y in ng.optimizers.registry.items() if 'DE' in x or 'Shiwa' in x))):
            for rotation in [False]:
                for d in [5, 20, 100]:
                    for name in ['sphere', 'cigar', 'hm', 'ellipsoid']:
                        for u in [0]:
                            function = ArtificialFunction(name=name, rotation=rotation, block_dimension=d, useless_variables=d * u, translation_factor=1.0)
                            yield Experiment(function, optim, budget=budget, seed=next(seedg))

@registry.register
def hdbo4d(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    for budget in [25, 31, 37, 43, 50, 60]:
        for optim in refactor_optims(get_optimizers('all_bo', seed=next(seedg))):
            for rotation in [False]:
                for d in [20]:
                    for name in ['sphere', 'cigar', 'hm', 'ellipsoid']:
                        for u in [0]:
                            function = ArtificialFunction(name=name, rotation=rotation, block_dimension=d, useless_variables=d * u, translation_factor=1.0)
                            yield Experiment(function, optim, budget=budget, seed=next(seedg))
                            
@registry.register
def spsa_benchmark(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = get_optimizers('spsa', seed=next(seedg))
    optims += ['NGOpt', 'NGOptRW']
    optims = refactor_optims(optims)
    for budget in [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        for optim in optims:
            for rotation in [True, False]:
                for name in ['sphere', 'sphere4', 'cigar']:
                    function = ArtificialFunction(name=name, rotation=rotation, block_dimension=20, noise_level=10)
                    yield Experiment(function, optim, budget=budget, seed=next(seedg))

@registry.register
def realworld(seed: Optional[int] = None) -> Iterator[Experiment]:
    funcs = [_mlda.Clustering.from_mlda(name, num, rescale) for name, num in [('Ruspini', 5), ('German towns', 10)] for rescale in [True, False]]
    funcs += [_mlda.SammonMapping.from_mlda('Virus', rescale=False), _mlda.SammonMapping.from_mlda('Virus', rescale=True)]
    funcs += [_mlda.Landscape(transform) for transform in [None, 'square', 'gaussian']]
    funcs += [ARCoating()]
    funcs += [PowerSystem(), PowerSystem(13)]
    funcs += [STSP(), STSP(500)]
    funcs += [game.Game('war')]
    funcs += [game.Game('batawaf')]
    funcs += [game.Game('flip')]
    funcs += [game.Game('guesswho')]
    funcs += [game.Game('bigguesswho')]
    base_env = rl.envs.DoubleOSeven(verbose=False)
    random_agent = rl.agents.Agent007(base_env)
    modules = {'mono': rl.agents.Perceptron, 'multi': rl.agents.DenseNet}
    agents = {a: rl.agents.TorchAgent.from_module_maker(base_env, m, deterministic=False) for a, m in modules.items()}
    env = base_env.with_agent(player_0=random_agent).as_single_agent()
    runner = rl.EnvironmentRunner(env.copy(), num_repetitions=100, max_step=50)
    for archi in ['mono', 'multi']:
        func = rl.agents.TorchAgentFunction(agents[archi], runner, reward_postprocessing=lambda x: 1 - x)
        funcs += [func]
    seedg = create_seed_generator(seed)
    optims = get_optimizers('basics', seed=next(seedg))
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def aquacrop_fao(seed: Optional[int] = None) -> Iterator[Experiment]:
    funcs = [NgAquacrop(i, 300.0 + 150.0 * np.cos(i)) for i in range(3, 7)]
    seedg = create_seed_generator(seed)
    optims = get_optimizers('basics', seed=next(seedg))
    optims = ['RBFGS', 'LBFGSB', 'MemeticDE']
    optims = ['PCABO']
    optims = ['PCABO', 'NGOpt', 'QODE']
    optims = ['QOPSO']
    optims = ['NGOpt']
    optims = ['SQOPSO']
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600]:
        for num_workers in [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        xp.function.parametrization.real_world = True
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def fishing(seed: Optional[int] = None) -> Iterator[Experiment]:
    funcs = [OptimizeFish(i) for i in [17, 35, 52, 70, 88, 105]]
    seedg = create_seed_generator(seed)
    optims = get_optimizers('basics', seed=next(seedg))
    optims += ['NGOpt', 'NGOptRW', 'ChainMetaModelSQP']
    optims = ['NGOpt']
    optims = ['PCABO']
    optims = ['PCABO', 'NGOpt', 'QODE']
    optims = ['QOPSO']
    optims = ['SQOPSO']
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600]:
        for algo in optims:
            for fu in funcs:
                xp = Experiment(fu, algo, budget, seed=next(seedg))
                xp.function.parametrization.real_world = True
                if not xp.is_incoherent:
                    yield xp

@registry.register
def rocket(seed: Optional[int] = None, seq: bool = False) -> Iterator[Experiment]:
    funcs = [Rocket(i) for i in range(17)]
    seedg = create_seed_generator(seed)
    optims = get_optimizers('basics', seed=next(seedg))
    optims += ['NGOpt', 'NGOptRW', 'ChainMetaModelSQP']
    optims = ['RBFGS', 'LBFGSB', 'MemeticDE']
    optims = ['CMA', 'PSO', 'QODE', 'QRDE', 'MetaModelPSO']
    if seq:
        optims += ['RBFGS', 'LBFGSB', 'MemeticDE']
    optims = ['NGOpt']
    optims = ['PCABO']
    optims = ['PCABO', 'NGOpt', 'QODE']
    optims = ['QOPSO']
    optims = ['SQOPSO']
    optims = ['NGOpt', 'QOPSO', 'SOPSO', 'QODE', 'SODE', 'CMA', 'DiagonalCMA', 'MetaModelOnePlusOne', 'MetaModelDE']
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600]:
        for num_workers in ([1] if seq else [1, 30]):
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        xp.function.parametrization.real_world = True
                        skip_ci(reason='Too slow')
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def mono_rocket(seed: Optional[int] = None) -> Iterator[Experiment]:
    return rocket(seed, seq=True)

@registry.register
def mixsimulator(seed: Optional[int] = None) -> Iterator[Experiment]:
    funcs = [OptimizeMix()]
    seedg = create_seed_generator(seed)
    optims = get_optimizers('basics', seed=next(seedg))
    optims = refactor_optims(optims)
    for budget in [20, 40, 80, 160]:
        for num_workers in [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def control_problem(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    num_rollouts = 1
    funcs = [Env(num_rollouts=num_rollouts, random_state=seed) for Env in [control.Swimmer, control.HalfCheetah, control.Hopper, control.Walker2d, control.Ant, control.Humanoid]]
    sigmas = [0.1, 0.1, 0.1, 0.1, 0.01, 0.001]
    funcs2 = []
    for sigma, func in zip(sigmas, funcs):
        f = func.copy()
        param = f.parametrization.copy()
        for array in param:
            array.set_mutation(sigma=sigma)
        param.set_name(f'sigma={sigma}')
        f.parametrization = param
        f.parametrization.freeze()
        funcs2.append(f)
    optims = get_optimizers('basics')
    optims = ['NGOpt', 'PSO', 'CMA']
    optims = refactor_optims(optims)
    for budget in [50, 75, 100, 150, 200, 250, 300, 400, 500, 1000, 3000, 5000, 8000, 16000, 32000, 64000]:
        for algo in optims:
            for fu in funcs2:
                xp = Experiment(fu, algo, budget, num_workers=1, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp

@registry.register
def neuro_control_problem(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    num_rollouts = 1
    funcs = [Env(num_rollouts=num_rollouts, intermediate_layer_dim=(50,), random_state=seed) for Env in [control.Swimmer, control.HalfCheetah, control.Hopper, control.Walker2d, control.Ant, control.Humanoid]]
    optims = ['CMA', 'NGOpt4', 'DiagonalCMA', 'NGOpt8', 'MetaModel', 'ChainCMAPowell']
    optims = ['NGOpt', 'CMA', 'PSO']
    optims = refactor_optims(optims)
    for budget in [50, 500, 5000, 10000, 20000, 35000, 50000, 100000, 200000]:
        for algo in optims:
            for fu in funcs:
                xp = Experiment(fu, algo, budget, num_workers=1, seed=next(seedg))
                xp.function.parametrization.real_world = True
                xp.function.parametrization.neural = True
                if not xp.is_incoherent:
                    yield xp

@registry.register
def olympus_surfaces(seed: Optional[int] = None) -> Iterator[Experiment]:
    from nevergrad.functions.olympussurfaces import OlympusSurface
    funcs = []
    for kind in OlympusSurface.SURFACE_KINDS:
        for k in range(2, 5):
            for noise in ['GaussianNoise', 'UniformNoise', 'GammaNoise']:
                for noise_scale in [0.5, 1]:
                    funcs.append(OlympusSurface(kind, 10 ** k, noise, noise_scale))
    seedg = create_seed_generator(seed)
    optims = get_optimizers('basics', 'noisy', seed=next(seedg))
    optims = ['NGOpt', 'CMA']
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def olympus_emulators(seed: Optional[int] = None) -> Iterator[Experiment]:
    from nevergrad.functions.olympussurfaces import OlympusEmulator
    funcs = []
    for dataset_kind in OlympusEmulator.DATASETS:
        for model_kind in ['BayesNeuralNet', 'NeuralNet']:
            funcs.append(OlympusEmulator(dataset_kind, model_kind))
    seedg = create_seed_generator(seed)
    optims = get_optimizers('basics', 'noisy', seed=next(seedg))
    optims = ['NGOpt', 'CMA']
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def topology_optimization(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    funcs = [TO(i) for i in [10, 20, 30, 40]]
    optims = ['CMA', 'GeneticDE', 'TwoPointsDE', 'VoronoiDE', 'DE', 'PSO', 'RandomSearch', 'OnePlusOne']
    optims = ['NGOpt']
    optims = refactor_optims(optims)
    for budget in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960]:
        for optim in optims:
            for f in funcs:
                for nw in [1, 30]:
                    yield Experiment(f, optim, budget, num_workers=nw, seed=next(seedg))

@registry.register
def sequential_topology_optimization(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    funcs = [TO(i) for i in [10, 20, 30, 40]]
    optims = ['CMA', 'GeneticDE', 'TwoPointsDE', 'VoronoiDE', 'DE', 'PSO', 'RandomSearch', 'OnePlusOne']
    optims = ['NGOpt']
    optims = refactor_optims(optims)
    for budget in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960]:
        for optim in optims:
            for f in funcs:
                for nw in [1, 30]:
                    yield Experiment(f, optim, budget, num_workers=nw, seed=next(seedg))

@registry.register
def simple_tsp(seed: Optional[int] = None, complex_tsp: bool = False) -> Iterator[Experiment]:
    funcs = [STSP(10 ** k, complex_tsp) for k in range(2, 6)]
    seedg = create_seed_generator(seed)
    optims = ['RotatedTwoPointsDE', 'DiscreteLenglerOnePlusOne', 'DiscreteDoerrOnePlusOne', 'DiscreteBSOOnePlusOne', 'AdaptiveDiscreteOnePlusOne', 'GeneticDE', 'DE', 'TwoPointsDE', 'DiscreteOnePlusOne', 'CMA', 'MetaModel', 'DiagonalCMA']
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def complex_tsp(seed: Optional[int] = None) -> Iterator[Experiment]:
    return simple_tsp(seed, complex_tsp=True)

@registry.register
def sequential_fastgames(seed: Optional[int] = None) -> Iterator[Experiment]:
    funcs = [game.Game(name) for name in ['war', 'batawaf', 'flip', 'guesswho', 'bigguesswho']]
    seedg = create_seed_generator(seed)
    optims = get_optimizers('noisy', 'splitters', 'progressive', seed=next(seedg))
    optims = refactor_optims(optims)
    for budget in [12800, 25600, 51200, 102400]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        xp.function.parametrization.real_world = True
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def powersystems(seed: Optional[int] = None) -> Iterator[Experiment]:
    funcs = []
    for dams in [3, 5, 9, 13]:
        funcs += [PowerSystem(dams, depth=2, width=3)]
    seedg = create_seed_generator(seed)
    budgets = [3200, 6400, 12800]
    optims = get_optimizers('basics', 'noisy', 'splitters', 'progressive', seed=next(seedg))
    optims = refactor_optims(optims)
    for budget in budgets:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        xp.function.parametrization.real_world = True
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def mlda(seed: Optional[int] = None) -> Iterator[Experiment]:
    funcs = [_mlda.Clustering.from_mlda(name, num, rescale) for name, num in [('Ruspini', 5), ('German towns', 10)] for rescale in [True, False]]
    funcs += [_mlda.SammonMapping.from_mlda('Virus', rescale=False), _mlda.SammonMapping.from_mlda('Virus', rescale=True)]
    funcs += [_mlda.Perceptron.from_mlda(name) for name in ['quadratic', 'sine', 'abs', 'heaviside']]
    funcs += [_mlda.Landscape(transform) for transform in [None, 'square', 'gaussian']]
    seedg = create_seed_generator(seed)
    optims = get_optimizers('basics', seed=next(seedg))
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for func in funcs:
                        xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                        xp.function.parametrization.real_world = True
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def mldakmeans(seed: Optional[int] = None) -> Iterator[Experiment]:
    funcs = [_mlda.Clustering.from_mlda(name, num, rescale) for name, num in [('Ruspini', 5), ('German towns', 10), ('Ruspini', 50), ('German towns', 100)] for rescale in [True, False]]
    seedg = create_seed_generator(seed)
    optims = get_optimizers('splitters', 'progressive', seed=next(seedg))
    optims += ['DE', 'CMA', 'PSO', 'TwoPointsDE', 'RandomSearch']
    optims = ['QODE', 'QRDE']
    optims = ['NGOpt']
    optims = refactor_optims(optims)
    for budget in [1000, 10000]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for func in funcs:
                        xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def image_similarity(seed: Optional[int] = None, with_pgan: bool = False, similarity: bool = True) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = get_optimizers('structured_moo', seed=next(seedg))
    funcs = [imagesxp.Image(loss=loss, with_pgan=with_pgan) for loss in imagesxp.imagelosses.registry.values() if loss.REQUIRES_REFERENCE == similarity]
    optims = refactor_optims(optims)
    for budget in [100 * 5 ** k for k in range(3)]:
        for func in funcs:
            for algo in optims:
                xp = Experiment(func, algo, budget, num_workers=1, seed=next(seedg))
                skip_ci(reason='too slow')
                if not xp.is_incoherent:
                    yield xp

@registry.register
def image_similarity_pgan(seed: Optional[int] = None) -> Iterator[Experiment]:
    return image_similarity(seed, with_pgan=True)

@registry.register
def image_single_quality(seed: Optional[int] = None) -> Iterator[Experiment]:
    return image_similarity(seed, with_pgan=False, similarity=False)

@registry.register
def image_single_quality_pgan(seed: Optional[int] = None) -> Iterator[Experiment]:
    return image_similarity(seed, with_pgan=True, similarity=False)

@registry.register
def image_multi_similarity(seed: Optional[int] = None, cross_valid: bool = False, with_pgan: bool = False) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = get_optimizers('structured_moo', seed=next(seedg))
    funcs = [imagesxp.Image(loss=loss, with_pgan=with_pgan) for loss in imagesxp.imagelosses.registry.values() if loss.REQUIRES_REFERENCE]
    base_values = [func(func.parametrization.sample().value) for func in funcs]
    if cross_valid:
        skip_ci(reason='Too slow')
        mofuncs = helpers.SpecialEvaluationExperiment.create_crossvalidation_experiments(funcs, pareto_size=25)
    else:
        mofuncs = [fbase.MultiExperiment(funcs, upper_bounds=base_values)]
    optims = refactor_optims(optims)
    for budget in [100 * 5 ** k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                for mofunc in mofuncs:
                    xp = Experiment(mofunc, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp

@registry.register
def image_multi_similarity_pgan(seed: Optional[int] = None) -> Iterator[Experiment]:
    return image_multi_similarity(seed, with_pgan=True)

@registry.register
def image_multi_similarity_cv(seed: Optional[int] = None) -> Iterator[Experiment]:
    return image_multi_similarity(seed, cross_valid=True)

@registry.register
def image_multi_similarity_pgan_cv(seed: Optional[int] = None) -> Iterator[Experiment]:
    return image_multi_similarity(seed, cross_valid=True, with_pgan=True)

@registry.register
def image_quality_proxy(seed: Optional[int] = None, with_pgan: bool = False) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = get_optimizers('structured_moo', seed=next(seedg))
    iqa, blur, brisque = [imagesxp.Image(loss=loss, with_pgan=with_pgan) for loss in (imagesxp.imagelosses.Koncept512, imagesxp.imagelosses.Blur, imagesxp.imagelosses.Brisque)]
    optims = refactor_optims(optims)
    for budget in [100 * 5 ** k for k in range(3)]:
        for algo in optims:
            for func in [blur, brisque]:
                sfunc = helpers.SpecialEvaluationExperiment(func, evaluation=iqa)
                sfunc.add_descriptors(non_proxy_function=False)
                xp = Experiment(sfunc, algo, budget, num_workers=1, seed=next(seedg))
                yield xp

@registry.register
def image_quality_proxy_pgan(seed: Optional[int] = None) -> Iterator[Experiment]:
    return image_quality_proxy(seed, with_pgan=True)

@registry.register
def image_quality(seed: Optional[int] = None, cross_val: bool = False, with_pgan: bool = False, num_images: int = 1) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = get_optimizers('structured_moo', seed=next(seedg))
    funcs = [imagesxp.Image(loss=loss, with_pgan=with_pgan, num_images=num_images) for loss in (imagesxp.imagelosses.Koncept512, imagesxp.imagelosses.Blur, imagesxp.imagelosses.Brisque)]
    if cross_val:
        mofuncs = helpers.SpecialEvaluationExperiment.create_crossvalidation_experiments(experiments=[funcs[0], funcs[2]], training_only_experiments=[funcs[1]], pareto_size=16)
    else:
        upper_bounds = [func(func.parametrization.value) for func in funcs]
        mofuncs = [fbase.MultiExperiment(funcs, upper_bounds=upper_bounds)]
    optims = refactor_optims(optims)
    for budget in [100 * 5 ** k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                for func in mofuncs:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp

@registry.register
def morphing_pgan_quality(seed: Optional[int] = None) -> Iterator[Experiment]:
    return image_quality(seed, with_pgan=True, num_images=2)

@registry.register
def image_quality_cv(seed: Optional[int] = None) -> Iterator[Experiment]:
    return image_quality(seed, cross_val=True)

@registry.register
def image_quality_pgan(seed: Optional[int] = None) -> Iterator[Experiment]:
    return image_quality(seed, with_pgan=True)

@registry.register
def image_quality_cv_pgan(seed: Optional[int] = None) -> Iterator[Experiment]:
    return image_quality(seed, cross_val=True, with_pgan=True)

@registry.register
def image_similarity_and_quality(seed: Optional[int] = None, cross_val: bool = False, with_pgan: bool = False) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = get_optimizers('structured_moo', seed=next(seedg))
    func_iqa = imagesxp.Image(loss=imagesxp.imagelosses.Koncept512, with_pgan=with_pgan)
    func_blur = imagesxp.Image(loss=imagesxp.imagelosses.Blur, with_pgan=with_pgan)
    base_blur_value = func_blur(func_blur.parametrization.value)
    optims = refactor_optims(optims)
    for func in [imagesxp.Image(loss=loss, with_pgan=with_pgan) for loss in imagesxp.imagelosses.registry.values() if loss.REQUIRES_REFERENCE]:
        base_value = func(func.parametrization.value)
        if cross_val:
            mofuncs = helpers.SpecialEvaluationExperiment.create_crossvalidation_experiments(training_only_experiments=[func, func_blur], experiments=[func_iqa], pareto_size=16)
        else:
            mofuncs = [fbase.MultiExperiment([func, func_blur, func_iqa], upper_bounds=[base_value, base_blur_value, 100.0])]
        for budget in [100 * 5 ** k for k in range(3)]:
            for algo in optims:
                for mofunc in mofuncs:
                    xp = Experiment(mofunc, algo, budget, num_workers=1, seed=next(seedg))
                    yield xp

@registry.register
def image_similarity_and_quality_cv(seed: Optional[int] = None) -> Iterator[Experiment]:
    return image_similarity_and_quality(seed, cross_val=True)

@registry.register
def image_similarity_and_quality_pgan(seed: Optional[int] = None) -> Iterator[Experiment]:
    return image_similarity_and_quality(seed, with_pgan=True)

@registry.register
def image_similarity_and_quality_cv_pgan(seed: Optional[int] = None) -> Iterator[Experiment]:
    return image_similarity_and_quality(seed, cross_val=True, with_pgan=True)

@registry.register
def double_o_seven(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    base_env = rl.envs.DoubleOSeven(verbose=False)
    random_agent = rl.agents.Agent007(base_env)
    modules = {'mono': rl.agents.Perceptron, 'multi': rl.agents.DenseNet}
    agents = {a: rl.agents.TorchAgent.from_module_maker(base_env, m, deterministic=False) for a, m in modules.items()}
    env = base_env.with_agent(player_0=random_agent).as_single_agent()
    dde = ng.optimizers.DifferentialEvolution(crossover='dimension').set_name('DiscreteDE')
    optimizers = ['PSO', dde, 'MetaTuneRecentering', 'DiagonalCMA', 'TBPSA', 'SPSA', 'RecombiningOptimisticNoisyDiscreteOnePlusOne', 'MetaModelPSO']
    optimizers = ['NGOpt', 'NGOptRW']
    optimizers = refactor_optims(optimizers)
    for num_repetitions in [1, 10, 100]:
        for archi in ['mono', 'multi']:
            for optim in optimizers:
                for env_budget in [5000, 10000, 20000, 40000]:
                    for num_workers in [1, 10, 100]:
                        runner = rl.EnvironmentRunner(env.copy(), num_repetitions=num_repetitions, max_step=50)
                        func = rl.agents.TorchAgentFunction(agents[archi], runner, reward_postprocessing=lambda x: 1 - x)
                        opt_budget = env_budget // num_repetitions
                        xp = Experiment(func, optim, budget=opt_budget, num_workers=num_workers, seed=next(seedg))
                        xp.function.parametrization.real_world = True
                        yield xp

@registry.register
def multiobjective_example(seed: Optional[int] = None, hd: bool = False, many: bool = False) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = get_optimizers('structure', 'structured_moo', seed=next(seedg))
    optims += [ng.families.DifferentialEvolution(multiobjective_adaptation=False).set_name('DE-noadapt'), 
               ng.families.DifferentialEvolution(crossover='twopoints', multiobjective_adaptation=False).set_name('TwoPointsDE-noadapt')]
    optims += ['DiscreteOnePlusOne', 'DiscreteLenglerOnePlusOne']
    optims = ['PymooNSGA2', 'PymooBatchNSGA2', 'LPCMA', 'VLPCMA', 'CMA']
    optims = ['LPCMA', 'VLPCMA', 'CMA']
    popsizes = [20, 40, 80]
    optims += [ng.families.EvolutionStrategy(recombination_ratio=recomb, only_offsprings=only, popsize=pop, offsprings=pop * 5) 
               for only in [True, False] for recomb in [0.1, 0.5] for pop in popsizes]
    optims = refactor_optims(optims)
    mofuncs = []
    dim = 2000 if hd else 7
    for name1, name2 in itertools.product(['sphere'], ['sphere', 'hm']):
        mofuncs.append(fbase.MultiExperiment([ArtificialFunction(name1, block_dimension=dim), ArtificialFunction(name2, block_dimension=dim)] + ([ArtificialFunction(name1, block_dimension=dim), ArtificialFunction(name2, block_dimension=dim)] if many else []), upper_bounds=[100, 100] * (2 if many else 1)))
        mofuncs.append(fbase.MultiExperiment([ArtificialFunction(name1, block_dimension=dim - 1), ArtificialFunction('sphere', block_dimension=dim - 1), ArtificialFunction(name2, block_dimension=dim - 1)] + ([ArtificialFunction(name1, block_dimension=dim - 1), ArtificialFunction('sphere', block_dimension=dim - 1), ArtificialFunction(name2, block_dimension=dim - 1)] if many else []), upper_bounds=[100, 100, 100.0] * (2 if many else 1)))
    for mofunc in mofuncs:
        for optim in optims:
            for budget in [100, 200, 400, 800, 1600, 3200]:
                for nw in [1, 100]:
                    xp = Experiment(mofunc, optim, budget=budget, num_workers=nw, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp

@registry.register
def multiobjective_example_hd(seed: Optional[int] = None) -> Iterator[Experiment]:
    return multiobjective_example(seed, hd=True)

@registry.register
def multiobjective_example_many_hd(seed: Optional[int] = None) -> Iterator[Experiment]:
    return multiobjective_example(seed, hd=True, many=True)

@registry.register
def multiobjective_example_many(seed: Optional[int] = None) -> Iterator[Experiment]:
    return multiobjective_example(seed, many=True)

@registry.register
def pbt(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optimizers = ['CMA', 'TwoPointsDE', 'Shiwa', 'OnePlusOne', 'DE', 'PSO', 'NaiveTBPSA', 'RecombiningOptimisticNoisyDiscreteOnePlusOne', 'PortfolioNoisyDiscreteOnePlusOne']
    optimizers = refactor_optims(optimizers)
    for func in PBT.itercases():
        for optim in optimizers:
            for budget in [100, 400, 1000, 4000, 10000]:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))

@registry.register
def far_optimum_es(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = get_optimizers('es', 'basics', seed=next(seedg))
    optims = refactor_optims(optims)
    for func in FarOptimumFunction.itercases():
        for optim in optims:
            for budget in [100, 400, 1000, 4000, 10000]:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))

@registry.register
def ceviche(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    instrum = ng.p.Array(shape=(40, 40), lower=0.0, upper=1.0).set_integer_casting()
    func = ExperimentFunction(photonics_ceviche, instrum.set_name('transition'))
    algos = ['DiagonalCMA', 'PSO', 'DE', 'CMA', 'OnePlusOne', 'LognormalDiscreteOnePlusOne', 'DiscreteLenglerOnePlusOne', 'MetaModel', 'MetaModelDE', 'MetaModelDSproba', 'MetaModelOnePlusOne', 'MetaModelPSO', 'MetaModelQODE', 'MetaModelTwoPointsDE', 'NeuralMetaModel', 'NeuralMetaModelDE', 'NeuralMetaModelTwoPointsDE', 'RFMetaModel', 'RFMetaModelDE', 'RFMetaModelOnePlusOne', 'RFMetaModelPSO', 'RFMetaModelTwoPointsDE', 'SVMMetaModel', 'SVMMetaModelDE', 'SVMMetaModelPSO', 'SVMMetaModelTwoPointsDE', 'RandRecombiningDiscreteLognormalOnePlusOne', 'SmoothDiscreteLognormalOnePlusOne', 'SmoothLognormalDiscreteOnePlusOne', 'UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne', 'SuperSmoothRecombiningDiscreteLognormalOnePlusOne', 'SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne', 'RecombiningDiscreteLognormalOnePlusOne', 'RandRecombiningDiscreteLognormalOnePlusOne', 'UltraSmoothDiscreteLognormalOnePlusOne', 'ZetaSmoothDiscreteLognormalOnePlusOne', 'SuperSmoothDiscreteLognormalOnePlusOne']
    for optim in algos:
        for budget in [20, 50, 100, 160, 240]:
            yield Experiment(func, optim, budget=budget, seed=next(seedg))

@registry.register
def multi_ceviche(seed: Optional[int] = None, c0: bool = False, precompute: bool = False, warmstart: bool = False) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    algos = ['DiagonalCMA', 'PSO', 'DE', 'CMA', 'OnePlusOne', 'LognormalDiscreteOnePlusOne', 'DiscreteLenglerOnePlusOne', 'MetaModel', 'MetaModelDE', 'MetaModelDSproba', 'MetaModelOnePlusOne', 'MetaModelPSO', 'MetaModelQODE', 'MetaModelTwoPointsDE', 'NeuralMetaModel', 'NeuralMetaModelDE', 'NeuralMetaModelTwoPointsDE', 'RFMetaModel', 'RFMetaModelDE', 'RFMetaModelOnePlusOne', 'RFMetaModelPSO', 'RFMetaModelTwoPointsDE', 'SVMMetaModel', 'SVMMetaModelDE', 'SVMMetaModelPSO', 'SVMMetaModelTwoPointsDE', 'RandRecombiningDiscreteLognormalOnePlusOne', 'SmoothDiscreteLognormalOnePlusOne', 'SmoothLognormalDiscreteOnePlusOne', 'UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne', 'SuperSmoothRecombiningDiscreteLognormalOnePlusOne', 'SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne', 'RecombiningDiscreteLognormalOnePlusOne', 'RandRecombiningDiscreteLognormalOnePlusOne', 'UltraSmoothDiscreteLognormalOnePlusOne', 'ZetaSmoothDiscreteLognormalOnePlusOne', 'SuperSmoothDiscreteLognormalOnePlusOne']
    assert not (precompute and (not warmstart))
    if not precompute:
        algos = ['RF1MetaModelLogNormal', 'Neural1MetaModelLogNormal', 'SVM1MetaModelLogNormal', 'CMAL']
    else:
        algos = ['UltraSmoothDiscreteLognormalOnePlusOne', 'DiscreteLenglerOnePlusOne', 'CMA', 'CMAL']
    algos = ['CMALS', 'CMALYS', 'CMALL']
    algos = ['CLengler', 'CMALS', 'CMALYS', 'CMALL', 'CMAL']
    algos = ['CMASL2', 'CMASL3']
    algos = ['DiagonalCMA', 'CMAL3', 'CMA', 'CLengler', 'CMALL', 'CMALYS', 'CMALSDiscreteLenglerOnePlusOne', 'CMASL3', 'CMASL2', 'DSproba']
    algos += ['LognormalDiscreteOnePlusOne', 'CMA', 'DiscreteLenglerOnePlusOne', 'SmoothDiscreteLognormalOnePlusOne', 'SuperSmoothDiscreteLognormalOnePlusOne', 'AnisotropicAdaptiveDiscreteOnePlusOne', 'Neural1MetaModelE', 'SVM1MetaModelE', 'Quad1MetaModelE', 'RF1MetaModelE', 'UltraSmoothDiscreteLognormalOnePlusOne', 'VoronoiDE', 'UltraSmoothDiscreteLognormalOnePlusOne', 'VoronoiDE', 'RF1MetaModelLogNormal', 'Neural1MetaModelLogNormal', 'SVM1MetaModelLogNormal', 'DSproba', 'ImageMetaModelE', 'ImageMetaModelOnePlusOne', 'ImageMetaModelDiagonalCMA', 'ImageMetaModelLengler', 'ImageMetaModelLogNormal']
    algos = [a for a in algos if a in list(ng.optimizers.registry.keys())]
    for benchmark_type in [np.random.choice([0, 1, 2, 3])]:
        if warmstart:
            try:
                suggestion = np.load(f'bestnp{benchmark_type}.npy')
            except Exception as e:
                print('Be caereful! You need warmstart data for warmstarting :-)  use scripts/plot_ceviche.sh.')
                raise e
        shape = tuple([int(p) for p in list(photonics_ceviche(None, benchmark_type))])
        name = photonics_ceviche('name', benchmark_type) + str(shape)
        instrumc0 = ng.p.Array(shape=shape, lower=0.0, upper=1.0)
        instrumc0c = ng.p.Array(shape=shape, lower=0.0, upper=1.0)
        instrumc0pen = ng.p.Array(shape=shape, lower=0.0, upper=1.0)
        instrum = ng.p.Array(shape=shape, lower=0.0, upper=1.0).set_integer_casting()
        instrum2 = ng.p.Array(shape=shape, lower=0.0, upper=1.0)
        instrum2p = ng.p.Array(shape=shape, lower=0.0, upper=1.0)
        def pc(x: np.ndarray) -> float:
            return photonics_ceviche(x, benchmark_type)
        def fpc(x: np.ndarray) -> Union[float, tuple]:
            loss, grad = photonics_ceviche(x.reshape(shape), benchmark_type, wantgrad=True)
            return (loss, grad.flatten())
        def epc(x: np.ndarray) -> float:
            return photonics_ceviche(x, benchmark_type, discretize=True)
        instrum.set_name(name)
        instrumc0.set_name(name)
        instrumc0c.set_name(name)
        instrumc0pen.set_name(name)
        instrum2.set_name(name)
        instrum2p.set_name(name)
        func = ExperimentFunction(pc, instrum)
        c0func = ExperimentFunction(pc, instrumc0)
        c0cfunc = ExperimentFunction(pc, instrumc0c)
        c0penfunc = ExperimentFunction(pc, instrumc0pen)
        eval_func = ExperimentFunction(epc, instrum2)
        import copy
        def export_numpy(name: str, array: np.ndarray, fields: Optional[np.ndarray] = None) -> None:
            from PIL import Image
            x = (255 * (1 - array)).astype('uint8')
            if fields is not None:
                np.save(name + 'fields.', fields)
                np.save(name + 'savedarray', array)
            im = Image.fromarray(x)
            im.convert('RGB').save(f'{name}_{np.average(np.abs(array.flatten() - 0.5) < 0.35)}_{np.average(np.abs(array.flatten() - 0.5) < 0.45)}.png', mode='L')
        def cv(x: np.ndarray) -> float:
            return float(np.sum(np.clip(np.abs(x - np.round(x)) - 0.001, 0.0, 50000000.0)))
        budgets = ([np.random.choice([3, 20, 50, 90, 150, 250, 400, 800, 1600, 3200, 6400]), 
                     np.random.choice([12800, 25600, 51200, 102400, 204800, 409600])] if not precompute else [np.random.choice([409600, 204800 + 102400, 204800]) - 102400])
        if benchmark_type == 3:
            budgets = ([np.random.choice([3, 20, 50, 90, 150, 250, 400, 800, 1600, 3200, 6400]), 
                         np.random.choice([12800, 25600, 51200, 102400])] if not precompute else [np.random.choice([204800 + 51200, 204800]) - 102400])
        for optim in [np.random.choice(algos)]:
            for budget in budgets:
                if (np.random.rand() < 0.05 or precompute) and (not warmstart):
                    from scipy import optimize as scipyoptimize
                    x0 = np.random.rand(np.prod(shape))
                    result = scipyoptimize.minimize(fpc, x0=x0, method='L-BFGS-B', tol=1e-09, jac=True, options={'maxiter': budget if not precompute else 102400}, bounds=[[0, 1] for _ in range(np.prod(shape))])
                    assert -1e-05 <= np.min(result.x.flatten())
                    assert np.max(result.x.flatten()) <= 1.0001
                    real_loss = epc(result.x.reshape(shape))
                    fake_loss, _ = fpc(result.x.reshape(shape))
                    if not precompute:
                        print(f'\nLOGPB{benchmark_type} LBFGSB with_budget {budget} returns {real_loss}')
                        print(f'\nLOGPB{benchmark_type} CheatingLBFGSB with_budget {budget} returns {fake_loss}')
                    initial_point = result.x.reshape(shape)
                    if budget > 100000 or np.random.rand() < 0.05:
                        export_numpy(f'pb{benchmark_type}_budget{(budget if not precompute else 102400)}_bfgs_{real_loss}_{fake_loss}', result.x.reshape(shape))
                if (c0 and np.random.choice([True, False, False, False])) and (not precompute):
                    pen = np.random.choice([True, False, False] + [False] * 20) and (not precompute)
                    pre_optim = ng.optimizers.registry[optim]
                    if pen:
                        try:
                            optim2 = type(optim, pre_optim.__bases__, dict(pre_optim.__dict__))
                        except:
                            optim2 = copy.deepcopy(pre_optim)
                        try:
                            optim2.name += 'c0p'
                        except:
                            optim2.__name__ += 'c0p'
                        sfunc = helpers.SpecialEvaluationExperiment(c0penfunc, evaluation=eval_func)
                        yield Experiment(sfunc, optim2, budget=budget, seed=next(seedg), constraint_violation=[cv], penalize_violation_at_test=False, suggestions=[suggestion] if warmstart else None)
                    else:
                        cheat = np.random.choice([False, True])
                        try:
                            optim3 = type(optim, pre_optim.__bases__, dict(pre_optim.__dict__))
                        except:
                            optim3 = copy.deepcopy(pre_optim)
                        try:
                            optim3.name += ('c0' if not cheat else 'c0c') + ('P' if precompute else '')
                        except:
                            optim3.__name__ += ('c0' if not cheat else 'c0c') + ('P' if precompute else '')
                        def plot_pc(x: np.ndarray) -> float:
                            fake_loss = photonics_ceviche(x, benchmark_type)
                            real_loss = photonics_ceviche(x, benchmark_type, discretize=True)
                            if budget > 100000 or np.random.rand() < 0.05:
                                export_numpy(f'pb{benchmark_type}_{optim}c0c_budget{budget}_{real_loss}_fl{fake_loss}', x.reshape(shape))
                            return fake_loss
                        if precompute:
                            instrum2i = ng.p.Array(init=initial_point, lower=0.0, upper=1.0)
                            instrum2i.set_name(name)
                        plot_cheat_eval_func = ExperimentFunction(plot_pc, instrum2 if not precompute else instrum2i)
                        sfunc = helpers.SpecialEvaluationExperiment(c0func if not cheat else c0cfunc, evaluation=eval_func if not cheat else plot_cheat_eval_func)
                        yield Experiment(sfunc, optim3, budget=budget, seed=next(seedg), suggestions=[suggestion] if warmstart else None)
                else:
                    def plot_epc(x: np.ndarray) -> float:
                        real_loss, fields = photonics_ceviche(x, benchmark_type, discretize=True, wantfields=True)
                        if budget > 100000 or np.random.rand() < 0.05:
                            export_numpy(f'pb{benchmark_type}_{optim}_budget{budget}_{real_loss}', x.reshape(shape), fields)
                        return real_loss
                    plot_eval_func = ExperimentFunction(plot_epc, instrum2p)
                    pfunc = helpers.SpecialEvaluationExperiment(func, evaluation=plot_eval_func)
                    yield Experiment(func if np.random.rand() < 0.0 else pfunc, optim, budget=budget, seed=next(seedg), suggestions=[suggestion] if warmstart else None)

@registry.register
def multi_ceviche_c0(seed: Optional[int] = None) -> Iterator[Experiment]:
    return multi_ceviche(seed, c0=True)

@registry.register
def multi_ceviche_c0_warmstart(seed: Optional[int] = None) -> Iterator[Experiment]:
    return multi_ceviche(seed, c0=True, warmstart=True)

@registry.register
def multi_ceviche_c0p(seed: Optional[int] = None) -> Iterator[Experiment]:
    return multi_ceviche(seed, c0=True, precompute=True)

@registry.register
def photonics(seed: Optional[int] = None, as_tuple: bool = False, small: bool = False, ultrasmall: bool = False, verysmall: bool = False) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    divider = 2 if small else 1
    if ultrasmall or verysmall:
        divider = 4
    optims = get_optimizers('es', 'basics', 'splitters', seed=next(seedg))
    optims = ['MemeticDE', 'PSO', 'DE', 'CMA', 'OnePlusOne', 'TwoPointsDE', 'GeneticDE', 'ChainMetaModelSQP', 'MetaModelDE', 'SVMMetaModelDE', 'RFMetaModelDE', 'RBFGS', 'LBFGSB']
    optims = refactor_optims(optims)
    for method in ['clipping', 'tanh']:
        for name in (['bragg'] if ultrasmall else (['cf_photosic_reference', 'cf_photosic_realistic'] if verysmall else ['bragg', 'chirped', 'morpho', 'cf_photosic_realistic', 'cf_photosic_reference'])):
            func = Photonics(name, 4 * (60 // divider // 4) if name == 'morpho' else 80 // divider, bounding_method=method, as_tuple=as_tuple)
            for budget in [10.0, 100.0, 1000.0]:
                for algo in optims:
                    xp = Experiment(func, algo, int(budget), num_workers=1, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp

@registry.register
def photonics2(seed: Optional[int] = None) -> Iterator[Experiment]:
    return photonics(seed, as_tuple=True)

@registry.register
def ultrasmall_photonics(seed: Optional[int] = None) -> Iterator[Experiment]:
    return photonics(seed, as_tuple=False, small=True, ultrasmall=True)

@registry.register
def ultrasmall_photonics2(seed: Optional[int] = None) -> Iterator[Experiment]:
    return photonics(seed, as_tuple=True, small=True, ultrasmall=True)

@registry.register
def verysmall_photonics(seed: Optional[int] = None) -> Iterator[Experiment]:
    return photonics(seed, as_tuple=False, small=True, verysmall=True)

@registry.register
def verysmall_photonics2(seed: Optional[int] = None) -> Iterator[Experiment]:
    return photonics(seed, as_tuple=True, small=True, verysmall=True)

@registry.register
def small_photonics(seed: Optional[int] = None) -> Iterator[Experiment]:
    return photonics(seed, as_tuple=False, small=True)

@registry.register
def small_photonics2(seed: Optional[int] = None) -> Iterator[Experiment]:
    return photonics(seed, as_tuple=True, small=True)

@registry.register
def adversarial_attack(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = get_optimizers('structure', 'structured_moo', seed=next(seedg))
    folder = os.environ.get('NEVERGRAD_ADVERSARIAL_EXPERIMENT_FOLDER', None)
    if folder is None:
        warnings.warn('Using random images, set variable NEVERGRAD_ADVERSARIAL_EXPERIMENT_FOLDER to specify a folder')
    optims = refactor_optims(optims)
    for func in imagesxp.ImageAdversarial.make_folder_functions(folder=folder):
        for budget in [100, 200, 300, 400, 1700]:
            for num_workers in [1]:
                for algo in optims:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp

def pbo_suite(seed: Optional[int] = None, reduced: bool = False) -> Iterator[Experiment]:
    dde = ng.optimizers.DifferentialEvolution(crossover='dimension').set_name('DiscreteDE')
    seedg = create_seed_generator(seed)
    index = 0
    list_optims = ['DiscreteOnePlusOne', 'Shiwa', 'CMA', 'PSO', 'TwoPointsDE', 'DE', 'OnePlusOne', 'AdaptiveDiscreteOnePlusOne', 'CMandAS2', 'PortfolioDiscreteOnePlusOne', 'DoubleFastGADiscreteOnePlusOne', 'MultiDiscrete', 'cGA', dde]
    if reduced:
        list_optims = [x for x in ng.optimizers.registry.keys() if 'iscre' in x and 'ois' not in x and ('ptim' not in x) and ('oerr' not in x)]
    list_optims = ['NGOpt', 'NGOptRW']
    list_optims = refactor_optims(list_optims)
    for dim in [16, 64, 100]:
        for fid in range(1, 24):
            for iid in range(1, 5):
                index += 1
                if reduced and index % 13:
                    continue
                for instrumentation in ['Softmax', 'Ordered', 'Unordered']:
                    try:
                        func = iohprofiler.PBOFunction(fid, iid, dim, instrumentation=instrumentation)
                        func.add_descriptors(instrum_str=instrumentation)
                    except ModuleNotFoundError as e:
                        raise fbase.UnsupportedExperiment('IOHexperimenter needs to be installed') from e
                    for optim in list_optims:
                        for nw in [1, 10]:
                            for budget in [100, 1000, 10000]:
                                yield Experiment(func, optim, num_workers=nw, budget=budget, seed=next(seedg))

@registry.register
def pbo_reduced_suite(seed: Optional[int] = None) -> Iterator[Experiment]:
    return pbo_suite(seed, reduced=True)

def causal_similarity(seed: Optional[int] = None) -> Iterator[Experiment]:
    from nevergrad.functions.causaldiscovery import CausalDiscovery
    seedg = create_seed_generator(seed)
    optims = ['CMA', 'NGOpt8', 'DE', 'PSO', 'RecES', 'RecMixES', 'RecMutDE', 'ParametrizationDE']
    func = CausalDiscovery()
    optims = refactor_optims(optims)
    for budget in [100 * 5 ** k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp

def unit_commitment(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ['CMA', 'NGOpt8', 'DE', 'PSO', 'RecES', 'RecMixES', 'RecMutDE', 'ParametrizationDE']
    optims = refactor_optims(optims)
    for num_timepoint in [5, 10, 20]:
        for num_generator in [3, 8]:
            func = UnitCommitmentProblem(num_timepoints=num_timepoint, num_generators=num_generator)
            for budget in [100 * 5 ** k for k in range(3)]:
                for algo in optims:
                    xp = Experiment(func, algo, budget, num_workers=1, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp

def team_cycling(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ['NGOpt10', 'CMA', 'DE']
    funcs = [Cycling(num) for num in [30, 31, 61, 22, 23, 45]]
    optims = refactor_optims(optims)
    for function in funcs:
        for budget in [3000]:
            for optim in optims:
                xp = Experiment(function, optim, budget=budget, num_workers=10, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp