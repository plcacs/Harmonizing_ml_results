#!/usr/bin/env python3
import os
import warnings
import typing as tp
import inspect
import itertools
import numpy as np
import nevergrad as ng
import nevergrad.functions.corefuncs as corefuncs
from nevergrad.functions import base as fbase
from nevergrad.functions import ExperimentFunction
from nevergrad.functions import ArtificialFunction
from nevergrad.functions import FarOptimumFunction
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
from .xpbase import Experiment  # type: ignore
from .xpbase import create_seed_generator
from .xpbase import registry
from .optgroups import get_optimizers
from . import frozenexperiments  # type: ignore
from . import gymexperiments  # type: ignore
from typing import Any, List, Generator, Iterator, Optional, Union

def refactor_optims(x: Any) -> List[str]:
    # Dummy implementation of refactor_optims with annotations.
    # Replace with proper logic.
    return [str(x)]  # type: ignore

def skip_ci(*, reason: str) -> None:
    if os.environ.get("SKIP_CI"):
        raise RuntimeError("Skipping CI: " + reason)

class _Constraint:
    def __init__(self, name: str, as_bool: bool) -> None:
        self.name = name
        self.as_bool = as_bool

    def __call__(self, data: np.ndarray) -> Union[bool, float]:
        if not isinstance(data, np.ndarray):
            raise ValueError("data must be np.ndarray")
        # Dummy implementation.
        value: float = float(np.sum(data))
        return value > 0 if self.as_bool else value

@registry.register
def keras_tuning(seed: Optional[int] = None, overfitter: bool = False, seq: bool = False, veryseq: bool = False) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = get_optimizers("keras", seed=next(seedg))
    datasets = ["dataset1", "dataset2"]  # Dummy datasets list.
    for dataset in datasets:
        function = MLTuning(regressor="keras", dataset=dataset, overfitter=overfitter)
        for budget in [100, 200]:
            for num_workers in [1, budget // 10] if seq else [budget]:
                xp = Experiment(function, optims[0], num_workers=num_workers, budget=budget, seed=next(seedg))
                skip_ci(reason="too slow")
                xp.function.parametrization.real_world = True
                yield xp

@registry.register
def mltuning(seed: Optional[int] = None, overfitter: bool = False, seq: bool = False) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = get_optimizers("ml", seed=next(seedg))
    datasets = ["ml_dataset1", "ml_dataset2"]
    for dataset in datasets:
        function = MLTuning(regressor="ml", dataset=dataset, overfitter=overfitter)
        for budget in [150, 500]:
            for num_workers in [1, budget // 4] if seq else [budget]:
                xp = Experiment(function, optims[0], num_workers=num_workers, budget=budget, seed=next(seedg))
                skip_ci(reason="too slow")
                xp.function.parametrization.real_world = True
                yield xp

@registry.register
def naivemltuning(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return mltuning(seed, overfitter=True, seq=False)

@registry.register
def veryseq_keras_tuning(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return keras_tuning(seed, overfitter=False, seq=True, veryseq=True)

@registry.register
def seq_keras_tuning(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return keras_tuning(seed, overfitter=False, seq=True)

@registry.register
def naive_seq_keras_tuning(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return keras_tuning(seed, overfitter=True, seq=True)

@registry.register
def naive_veryseq_keras_tuning(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return keras_tuning(seed, overfitter=True, seq=True, veryseq=True)

@registry.register
def oneshot_mltuning(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return mltuning(seed, overfitter=False, seq=False)

@registry.register
def seq_mltuning(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return mltuning(seed, overfitter=False, seq=True)

@registry.register
def nano_seq_mltuning(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return mltuning(seed, overfitter=False, seq=True)

@registry.register
def nano_veryseq_mltuning(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return mltuning(seed, overfitter=False, seq=True)

@registry.register
def nano_naive_veryseq_mltuning(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return mltuning(seed, overfitter=True, seq=True)

@registry.register
def nano_naive_seq_mltuning(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return mltuning(seed, overfitter=True, seq=True)

@registry.register
def naive_seq_mltuning(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return mltuning(seed, overfitter=True, seq=True)

@registry.register
def yawidebbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    functions = [ArtificialFunction("yawidebbob_func", block_dimension=50) for _ in range(8)]
    optims = refactor_optims(["NGOpt", "NGOptRW"])
    for func in functions:
        for budget in [50, 1500, 25000]:
            for nw in [1, budget]:
                for optim in optims:
                    xp = Experiment(func, optim, num_workers=nw, budget=budget, seed=next(seedg))
                    yield xp

@registry.register
def parallel_small_budget(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    functions = [ArtificialFunction("parallel_small", block_dimension=4) for _ in range(4)]
    optims = refactor_optims(["DE", "CMA", "NGOpt"])
    for func in functions:
        for budget in [10, 50, 100, 200, 400]:
            for nw in [2, 8, 16]:
                for batch in [True, False]:
                    if nw < budget / 4:
                        xp = Experiment(func, optims[0], num_workers=nw, budget=budget, seed=next(seedg))
                        yield xp

@registry.register
def instrum_discrete(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(["OnePlusOne", "DiscreteLenglerOnePlusOne"])
    for nv in [10, 50, 200, 1000, 5000]:
        for arity in [2, 3, 7, 30]:
            for instrum_str in ["Unordered", "Softmax", "Ordered"]:
                if instrum_str == "Softmax":
                    instrum = ng.p.Choice(range(arity), repetitions=nv)
                else:
                    instrum = ng.p.TransitionChoice(range(arity), repetitions=nv, ordered=(instrum_str == "Ordered"))
                for name in ["onemax", "leadingones", "jump"]:
                    dfunc = ExperimentFunction(corefuncs.DiscreteFunction(name, arity), instrum.set_name(instrum_str))
                    for optim in optims:
                        for nw in [1, 10]:
                            for budget in [50, 500, 5000]:
                                yield Experiment(dfunc, optim, num_workers=nw, budget=budget, seed=next(seedg))

@registry.register
def sequential_instrum_discrete(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(["OnePlusOne", "DiscreteLenglerOnePlusOne"])
    for nv in [10, 50, 200, 1000, 5000]:
        for arity in [2, 3, 7, 30]:
            for instrum_str in ["Unordered", "Softmax", "Ordered"]:
                if instrum_str == "Softmax":
                    instrum = ng.p.Choice(range(arity), repetitions=nv)
                else:
                    instrum = ng.p.TransitionChoice(range(arity), repetitions=nv, ordered=(instrum_str == "Ordered"))
                for name in ["onemax", "leadingones", "jump"]:
                    dfunc = ExperimentFunction(corefuncs.DiscreteFunction(name, arity), instrum.set_name(instrum_str))
                    for optim in optims:
                        for budget in [50, 500, 5000, 50000]:
                            yield Experiment(dfunc, optim, budget=budget, seed=next(seedg))

@registry.register
def deceptive(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    names = ["deceptivemultimodal", "deceptiveillcond", "deceptivepath"]
    optims = refactor_optims(["NGOpt"])
    functions = [ArtificialFunction(name, block_dimension=2) for name in names]
    for func in functions:
        for optim in optims:
            for budget in [25, 50, 75, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))

@registry.register
def lowbudget(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    functions = [ArtificialFunction(name, block_dimension=7, bounded=b) for name in names for b in [True, False]]
    optims = ["AX", "BOBYQA", "Cobyla", "RandomSearch", "CMA", "NGOpt", "DE", "PSO", "pysot", "negpysot"]
    for func in functions:
        for optim in optims:
            for budget in [10, 20, 30]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))

@registry.register
def parallel(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    functions = [ArtificialFunction(name, block_dimension=25, useless_variables=25 * uv) for name in names for uv in [0, 5]]
    optims = refactor_optims(get_optimizers("parallel_basics", seed=next(seedg)))
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=int(budget / 5), seed=next(seedg))

@registry.register
def harderparallel(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar", "ellipsoid"]
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv) for name in names for bd in [5, 25] for uv in [0, 5]]
    optims = refactor_optims(["NGOpt10"] + get_optimizers("emna_variants", seed=next(seedg)))
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000]:
                for num_workers in [int(budget / 10), int(budget / 5), int(budget / 3)]:
                    yield Experiment(func, optim, budget=budget, num_workers=num_workers, seed=next(seedg))

@registry.register
def oneshot(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * 0) for name in names for bd in [3, 10, 30, 100, 300, 1000, 3000]]
    optims = refactor_optims(get_optimizers("oneshot", seed=next(seedg)))
    for func in functions:
        for optim in optims:
            for budget in [100000, 30, 100, 300, 1000, 3000, 10000]:
                if func.dimension < 3000 or budget < 100000:
                    yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))

@registry.register
def doe(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * 0) for name in names for bd in [2000, 20000]]
    optims = refactor_optims(get_optimizers("oneshot", seed=next(seedg)))
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000, 30000, 100000]:
                yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))

@registry.register
def newdoe(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * 0) for name in names for bd in [2000, 20, 200, 20000]]
    budgets = [30, 100, 3000, 10000, 30000, 100000, 300000]
    for func in functions:
        for optim in refactor_optims(get_optimizers("oneshot", seed=next(seedg))):
            for budget in budgets:
                yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))

@registry.register
def fiveshots(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv) for name in names for bd in [3, 25] for uv in [0, 5]]
    optims = refactor_optims(get_optimizers("oneshot", "basics", seed=next(seedg)))
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=budget // 5, seed=next(seedg))

@registry.register
def multimodal(seed: Optional[int] = None, para: bool = False) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek", "deceptivemultimodal"]
    optims = get_optimizers("basics", seed=next(seedg))
    if not para:
        optims += get_optimizers("scipy", seed=next(seedg))
    optims = refactor_optims(["NGOpt"])
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv) for name in names for bd in [3, 25] for uv in [0, 5]]
    for func in functions:
        for optim in optims:
            for budget in [3000, 10000, 30000, 100000]:
                for nw in [1000] if para else [1]:
                    xp = Experiment(func, optim, budget=budget, num_workers=nw, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp

@registry.register
def hdmultimodal(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek", "deceptivemultimodal"]
    optims = refactor_optims(["NGOpt"])
    functions = [ArtificialFunction(name, block_dimension=bd) for name in names for bd in [1000, 6000, 36000]]
    for func in functions:
        for optim in optims:
            for budget in [3000, 10000]:
                for nw in [1]:
                    yield Experiment(func, optim, budget=budget, num_workers=nw, seed=next(seedg))

@registry.register
def paramultimodal(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return multimodal(seed, para=True)

@registry.register
def bonnans(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    instrum = ng.p.TransitionChoice(range(2), repetitions=100, ordered=False)
    softmax_instrum = ng.p.Choice(range(2), repetitions=100)
    optims = refactor_optims(["NGOpt"])
    for i in range(21):
        bonnans = corefuncs.BonnansFunction(index=i)
        for optim in optims:
            instrum_str = "TransitionChoice" if "Discrete" in optim else "Softmax"
            dfunc = ExperimentFunction(bonnans, instrum if instrum_str == "TransitionChoice" else softmax_instrum)
            for budget in list(range(20, 101)):
                yield Experiment(dfunc, optim, num_workers=1, budget=budget, seed=next(seedg))

@registry.register
def yabbob(seed: Optional[int] = None, parallel: bool = False, big: bool = False, small: bool = False, noise: bool = False,
           hd: bool = False, constraint_case: int = 0, split: bool = False, tuning: bool = False, reduction_factor: int = 1,
           bounded: bool = False, box: bool = False, max_num_constraints: int = 4, mega_smooth_penalization: int = 0) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek", "deceptivemultimodal", "bucherastrigin", "multipeak",
             "sphere", "doublelinearslope", "stepdoublelinearslope",
             "cigar", "altcigar", "ellipsoid", "altellipsoid", "stepellipsoid", "discus", "bentcigar",
             "deceptiveillcond", "deceptivemultimodal", "deceptivepath"]
    noise_level = 100000 if (noise and hd) else (100 if noise else 0)
    optims = refactor_optims(["NGOpt"])
    functions = [ArtificialFunction(name, block_dimension=(1000 if hd else (40 if bounded else (2 if noise else [2,10,50][0]))),
                                    rotation=r, noise_level=noise_level, split=split) for name in names for r in [True, False]]
    functions = functions[::reduction_factor]
    constraints = [_Constraint(n, as_bool=b) for b in [False, True] for n in ["sum", "diff", "second_diff", "ball"]]
    if mega_smooth_penalization > 0:
        constraints = []
    for func in functions:
        func.constraint_violation = []
        if constraint_case != 0:
            for constraint in constraints[max(0, abs(constraint_case) - max_num_constraints):abs(constraint_case)]:
                if constraint_case > 0:
                    func.parametrization.register_cheap_constraint(constraint)
                elif constraint_case < 0:
                    func.constraint_violation.append(constraint)
    budgets = [12800] if big and (not noise) else ([50, 200, 800, 3200, 12800] if not noise else [3200, 12800, 51200, 102400])
    if small and (not noise):
        budgets = [10, 20, 40]
    if bounded:
        budgets = [10, 20, 40, 100, 300]
    for optim in optims:
        for function in functions:
            for budget in budgets:
                xp = Experiment(function, optim, budget=budget, num_workers=(100 if parallel else 1), seed=next(seedg), constraint_violation=function.constraint_violation)
                if constraint_case != 0:
                    xp.function.parametrization.has_constraints = True
                if not xp.is_incoherent:
                    yield xp

@registry.register
def yahdlbbbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, hd=True, small=True)

@registry.register
def reduced_yahdlbbbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, hd=True, small=True, reduction_factor=17)

@registry.register
def yanoisysplitbbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, noise=True, parallel=False, split=True)

@registry.register
def yahdnoisysplitbbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
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
    slices = [yabbob(seed, hd=True, constraint_case=-1, mega_smooth_penalization=1000) for _ in range(6)]
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
def yahdnoisybbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, hd=True, noise=True)

@registry.register
def yabigbbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, parallel=False, big=True)

@registry.register
def yasplitbbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, parallel=False, split=True)

@registry.register
def yahdsplitbbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, hd=True, split=True)

@registry.register
def yatuningbbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, parallel=False, big=False, small=True, reduction_factor=13, tuning=True)

@registry.register
def yatinybbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, parallel=False, big=False, small=True, reduction_factor=13)

@registry.register
def yasmallbbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, parallel=False, big=False, small=True)

@registry.register
def yahdbbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, hd=True)

@registry.register
def yaparabbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, parallel=True, big=False)

@registry.register
def yanoisybbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, noise=True)

@registry.register
def yaboundedbbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, bounded=True)

@registry.register
def yaboxbbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return yabbob(seed, box=True)

@registry.register
def ms_bbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    dims = [2, 3, 5, 10, 20]
    functions = [ArtificialFunction(name, block_dimension=d, rotation=True, expo=expo, translation_factor=tf) for name in ["cigar", "sphere", "rastrigin"] for expo in [1.0, 5.0] for tf in [0.01, 0.1, 1.0, 10.0] for d in dims]
    optims = refactor_optims(["QODE"])
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 400, 800, 1600, 3200]:
                for nw in [1]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))

@registry.register
def zp_ms_bbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    dims = [2, 3, 5, 10, 20]
    functions = [ArtificialFunction(name, block_dimension=d, rotation=True, expo=expo, translation_factor=tf, zero_pen=True) for name in ["cigar", "sphere", "rastrigin"] for expo in [1.0, 5.0] for tf in [0.01, 0.1, 1.0, 10.0] for d in dims]
    optims = refactor_optims(["QODE", "PSO", "SQOPSO", "DE", "CMA"])
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 300, 400, 500, 600, 700, 800]:
                for nw in [1, 10, 50]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))

def nozp_noms_bbob(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    dims = [2, 3, 5, 10, 20]
    functions = [ArtificialFunction(name, block_dimension=d, rotation=True, expo=expo, translation_factor=tf, zero_pen=False) for name in ["cigar", "sphere", "rastrigin"] for expo in [1.0, 5.0] for tf in [1.0] for d in dims]
    optims = refactor_optims(["QODE", "PSO", "SQOPSO", "DE", "CMA"])
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 400, 800, 1600, 3200]:
                for nw in [1]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))

@registry.register
def ranknoisy(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("progressive", seed=next(seedg)))
    for budget in [25000, 50000, 100000]:
        for optim in optims:
            for d in [20000, 200, 2000]:
                for name in ["cigar", "altcigar", "ellipsoid", "altellipsoid"]:
                    for noise_dissymmetry in [False, True]:
                        function = ArtificialFunction(name=name, rotation=False, block_dimension=d, noise_level=10, noise_dissymmetry=noise_dissymmetry, translation_factor=1.0)
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))

@registry.register
def noisy(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("progressive", seed=next(seedg)))
    for budget in [25000, 50000, 100000]:
        for optim in optims:
            for d in [2, 20, 200, 2000]:
                for name in ["sphere", "rosenbrock", "cigar", "hm"]:
                    for noise_dissymmetry in [False, True]:
                        function = ArtificialFunction(name=name, rotation=True, block_dimension=d, noise_level=10, noise_dissymmetry=noise_dissymmetry, translation_factor=1.0)
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))

@registry.register
def paraalldes(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    for budget in [10, 100, 1000, 10000, 100000]:
        for optim in sorted([x for x in ng.optimizers.registry.keys() if "DE" in x and "Tune" in x]):
            for rotation in [False]:
                for d in [5, 20, 100, 500, 2500]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        function = ArtificialFunction(name=name, rotation=rotation, block_dimension=d, useless_variables=d * 0, translation_factor=1.0)
                        yield Experiment(function, optim, budget=budget, seed=next(seedg), num_workers=max(d, budget // 6))

@registry.register
def parahdbo4d(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    for budget in [25, 31, 37, 43, 50, 60]:
        for optim in refactor_optims(sorted([x for x in ng.optimizers.registry.keys() if "BO" in x and "Tune" in x])):
            for rotation in [False]:
                for d in [20, 2000]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        function = ArtificialFunction(name=name, rotation=rotation, block_dimension=d, useless_variables=d * 0, translation_factor=1.0)
                        yield Experiment(function, optim, budget=budget, seed=next(seedg), num_workers=max(d, budget // 6))

@registry.register
def alldes(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    for budget in [10, 100, 1000, 10000, 100000]:
        for optim in refactor_optims(sorted([x for x in ng.optimizers.registry.keys() if "DE" in x or "Shiwa" in x])):
            for rotation in [False]:
                for d in [5, 20, 100]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        function = ArtificialFunction(name=name, rotation=rotation, block_dimension=d, useless_variables=d * 0, translation_factor=1.0)
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))

@registry.register
def hdbo4d(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    for budget in [25, 31, 37, 43, 50, 60]:
        for optim in refactor_optims(get_optimizers("all_bo", seed=next(seedg))):
            for rotation in [False]:
                for d in [20]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        function = ArtificialFunction(name=name, rotation=rotation, block_dimension=d, useless_variables=d * 0, translation_factor=1.0)
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))
                        
@registry.register
def spsa_benchmark(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(["NGOpt", "NGOptRW"])
    for budget in [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        for optim in optims:
            for rotation in [True, False]:
                for name in ["sphere", "sphere4", "cigar"]:
                    function = ArtificialFunction(name=name, rotation=rotation, block_dimension=20, noise_level=10)
                    yield Experiment(function, optim, budget=budget, seed=next(seedg))

@registry.register
def realworld(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    funcs = [_mlda.Clustering.from_mlda(n, num, rescale) for n, num in [("Ruspini", 5), ("German towns", 10)] for rescale in [True, False]]
    funcs += [_mlda.SammonMapping.from_mlda("Virus", rescale=False), _mlda.SammonMapping.from_mlda("Virus", rescale=True)]
    funcs += [_mlda.Landscape(t) for t in [None, "square", "gaussian"]]
    funcs += [ARCoating()]
    funcs += [PowerSystem(), PowerSystem(13)]
    funcs += [STSP(), STSP(500)]
    funcs += [game.Game("war"), game.Game("batawaf"), game.Game("flip"), game.Game("guesswho"), game.Game("bigguesswho")]
    base_env = rl.envs.DoubleOSeven(verbose=False)
    random_agent = rl.agents.Agent007(base_env)
    modules = {"mono": rl.agents.Perceptron, "multi": rl.agents.DenseNet}
    agents = {a: rl.agents.TorchAgent.from_module_maker(base_env, m, deterministic=False) for a, m in modules.items()}
    env = base_env.with_agent(player_0=random_agent).as_single_agent()
    runner = rl.EnvironmentRunner(env.copy(), num_repetitions=100, max_step=50)
    for archi in ["mono", "multi"]:
        func = rl.agents.TorchAgentFunction(agents[archi], runner, reward_postprocessing=lambda x: 1 - x)
        funcs += [func]
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("basics", seed=next(seedg)))
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def aquacrop_fao(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    funcs = [NgAquacrop(i, 300.0 + 150.0 * np.cos(i)) for i in range(3, 7)]
    seedg = create_seed_generator(seed)
    optims = refactor_optims(["PCABO", "NGOpt", "QODE"])
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
def fishing(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    funcs = [OptimizeFish(i) for i in [17, 35, 52, 70, 88, 105]]
    seedg = create_seed_generator(seed)
    optims = refactor_optims(["NGOpt"])
    for budget in [25, 50, 100, 200, 400, 800, 1600]:
        for algo in optims:
            for fu in funcs:
                xp = Experiment(fu, algo, budget, seed=next(seedg))
                xp.function.parametrization.real_world = True
                if not xp.is_incoherent:
                    yield xp

@registry.register
def rocket(seed: Optional[int] = None, seq: bool = False) -> Generator[Experiment, None, None]:
    funcs = [Rocket(i) for i in range(17)]
    seedg = create_seed_generator(seed)
    optims = refactor_optims(["NGOpt"])
    for budget in [25, 50, 100, 200, 400, 800, 1600]:
        for num_workers in [1] if seq else [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        xp.function.parametrization.real_world = True
                        skip_ci(reason="Too slow")
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def mono_rocket(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return rocket(seed, seq=True)

@registry.register
def mixsimulator(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    funcs = [OptimizeMix()]
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("basics", seed=next(seedg)))
    for budget in [20, 40, 80, 160]:
        for num_workers in [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def control_problem(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
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
        param.set_name(f"sigma={sigma}")
        f.parametrization = param
        f.parametrization.freeze()
        funcs2.append(f)
    optims = refactor_optims(["NGOpt", "PSO", "CMA"])
    for budget in [50, 75, 100, 150, 200, 250, 300, 400, 500, 1000, 3000, 5000, 8000, 16000, 32000, 64000]:
        for algo in optims:
            for fu in funcs2:
                xp = Experiment(fu, algo, budget, num_workers=1, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp

@registry.register
def neuro_control_problem(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    num_rollouts = 1
    funcs = [Env(num_rollouts=num_rollouts, intermediate_layer_dim=(50,), random_state=seed) for Env in [control.Swimmer, control.HalfCheetah, control.Hopper, control.Walker2d, control.Ant, control.Humanoid]]
    optims = refactor_optims(["NGOpt", "CMA", "PSO"])
    for budget in [50, 500, 5000, 10000, 20000, 35000, 50000, 100000, 200000]:
        for algo in optims:
            for fu in funcs:
                xp = Experiment(fu, algo, budget, num_workers=1, seed=next(seedg))
                xp.function.parametrization.real_world = True
                xp.function.parametrization.neural = True
                if not xp.is_incoherent:
                    yield xp

@registry.register
def olympus_surfaces(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    from nevergrad.functions.olympussurfaces import OlympusSurface
    funcs = []
    for kind in OlympusSurface.SURFACE_KINDS:
        for k in range(2, 5):
            for noise in ["GaussianNoise", "UniformNoise", "GammaNoise"]:
                for noise_scale in [0.5, 1]:
                    funcs.append(OlympusSurface(kind, 10 ** k, noise, noise_scale))
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("basics", "noisy", seed=next(seedg)))
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def olympus_emulators(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    from nevergrad.functions.olympussurfaces import OlympusEmulator
    funcs = []
    for dataset_kind in OlympusEmulator.DATASETS:
        for model_kind in ["BayesNeuralNet", "NeuralNet"]:
            funcs.append(OlympusEmulator(dataset_kind, model_kind))
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("basics", "noisy", seed=next(seedg)))
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def topology_optimization(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    funcs = [TO(i) for i in [10, 20, 30, 40]]
    optims = refactor_optims(["NGOpt"])
    for budget in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960]:
        for optim in optims:
            for f in funcs:
                for nw in [1, 30]:
                    yield Experiment(f, optim, budget, num_workers=nw, seed=next(seedg))

@registry.register
def sequential_topology_optimization(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    funcs = [TO(i) for i in [10, 20, 30, 40]]
    optims = refactor_optims(["NGOpt"])
    for budget in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960]:
        for optim in optims:
            for f in funcs:
                for nw in [1, 30]:
                    yield Experiment(f, optim, budget, num_workers=nw, seed=next(seedg))

@registry.register
def simple_tsp(seed: Optional[int] = None, complex_tsp: bool = False) -> Generator[Experiment, None, None]:
    funcs = [STSP(10 ** k, complex_tsp) for k in range(2, 6)]
    seedg = create_seed_generator(seed)
    optims = refactor_optims(["RotatedTwoPointsDE", "DiscreteLenglerOnePlusOne"])
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def complex_tsp(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return simple_tsp(seed, complex_tsp=True)

@registry.register
def sequential_fastgames(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    funcs = [game.Game(name) for name in ["war", "batawaf", "flip", "guesswho", "bigguesswho"]]
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("noisy", "splitters", "progressive", seed=next(seedg)))
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
def powersystems(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    funcs = []
    for dams in [3, 5, 9, 13]:
        funcs += [PowerSystem(dams, depth=2, width=3)]
    seedg = create_seed_generator(seed)
    budgets = [3200, 6400, 12800]
    optims = refactor_optims(get_optimizers("basics", "noisy", "splitters", "progressive", seed=next(seedg)))
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
def mlda(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    funcs = [_mlda.Clustering.from_mlda(n, num, rescale) for n, num in [("Ruspini", 5), ("German towns", 10)] for rescale in [True, False]]
    funcs += [_mlda.SammonMapping.from_mlda("Virus", rescale=False), _mlda.SammonMapping.from_mlda("Virus", rescale=True)]
    funcs += [_mlda.Perceptron.from_mlda(n) for n in ["quadratic", "sine", "abs", "heaviside"]]
    funcs += [_mlda.Landscape(t) for t in [None, "square", "gaussian"]]
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("basics", seed=next(seedg)))
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    xp = Experiment(funcs[0], algo, budget, num_workers=num_workers, seed=next(seedg))
                    xp.function.parametrization.real_world = True
                    if not xp.is_incoherent:
                        yield xp

@registry.register
def mldakmeans(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    funcs = [_mlda.Clustering.from_mlda(n, num, rescale) for n, num in [("Ruspini", 5), ("German towns", 10), ("Ruspini", 50), ("German towns", 100)] for rescale in [True, False]]
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("splitters", "progressive", seed=next(seedg)))
    for budget in [1000, 10000]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for func in funcs:
                        xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def image_similarity(seed: Optional[int] = None, with_pgan: bool = False, similarity: bool = True) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("structured_moo", seed=next(seedg)))
    funcs = [imagesxp.Image(loss=loss, with_pgan=with_pgan) for loss in imagesxp.imagelosses.registry.values() if loss.REQUIRES_REFERENCE == similarity]
    for budget in [100 * 5 ** k for k in range(3)]:
        for func in funcs:
            for algo in optims:
                xp = Experiment(func, algo, budget, num_workers=1, seed=next(seedg))
                skip_ci(reason="too slow")
                if not xp.is_incoherent:
                    yield xp

@registry.register
def image_similarity_pgan(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return image_similarity(seed, with_pgan=True)

@registry.register
def image_single_quality(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return image_similarity(seed, with_pgan=False, similarity=False)

@registry.register
def image_single_quality_pgan(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return image_similarity(seed, with_pgan=True, similarity=False)

@registry.register
def image_multi_similarity(seed: Optional[int] = None, cross_valid: bool = False, with_pgan: bool = False) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("structured_moo", seed=next(seedg)))
    funcs = [imagesxp.Image(loss=loss, with_pgan=with_pgan) for loss in imagesxp.imagelosses.registry.values() if loss.REQUIRES_REFERENCE]
    if cross_valid:
        skip_ci(reason="Too slow")
        mofuncs = helpers.SpecialEvaluationExperiment.create_crossvalidation_experiments(funcs, pareto_size=25)
    else:
        base_values = [func(func.parametrization.sample().value) for func in funcs]
        mofuncs = [fbase.MultiExperiment(funcs, upper_bounds=base_values)]
    for budget in [100 * 5 ** k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                for mofunc in mofuncs:
                    xp = Experiment(mofunc, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp

@registry.register
def image_multi_similarity_pgan(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return image_multi_similarity(seed, with_pgan=True)

@registry.register
def image_multi_similarity_cv(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return image_multi_similarity(seed, cross_valid=True)

@registry.register
def image_multi_similarity_pgan_cv(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return image_multi_similarity(seed, cross_valid=True, with_pgan=True)

@registry.register
def image_quality_proxy(seed: Optional[int] = None, with_pgan: bool = False) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("structured_moo", seed=next(seedg)))
    iqa, blur, brisque = [imagesxp.Image(loss=loss, with_pgan=with_pgan) for loss in (imagesxp.imagelosses.Koncept512, imagesxp.imagelosses.Blur, imagesxp.imagelosses.Brisque)]
    for budget in [100 * 5 ** k for k in range(3)]:
        for algo in optims:
            for func in [blur, brisque]:
                sfunc = helpers.SpecialEvaluationExperiment(func, evaluation=iqa)
                sfunc.add_descriptors(non_proxy_function=False)
                xp = Experiment(sfunc, algo, budget, num_workers=1, seed=next(seedg))
                yield xp

@registry.register
def image_quality_proxy_pgan(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return image_quality_proxy(seed, with_pgan=True)

@registry.register
def image_quality(seed: Optional[int] = None, cross_val: bool = False, with_pgan: bool = False, num_images: int = 1) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("structured_moo", seed=next(seedg)))
    funcs = [imagesxp.Image(loss=loss, with_pgan=with_pgan, num_images=num_images) for loss in (imagesxp.imagelosses.Koncept512, imagesxp.imagelosses.Blur, imagesxp.imagelosses.Brisque)]
    if cross_val:
        mofuncs = helpers.SpecialEvaluationExperiment.create_crossvalidation_experiments(experiments=[funcs[0], funcs[2]], training_only_experiments=[funcs[1]], pareto_size=16)
    else:
        upper_bounds = [func(func.parametrization.value) for func in funcs]
        mofuncs = [fbase.MultiExperiment(funcs, upper_bounds=upper_bounds)]
    for budget in [100 * 5 ** k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                for func in mofuncs:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp

@registry.register
def morphing_pgan_quality(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return image_quality(seed, with_pgan=True, num_images=2)

@registry.register
def image_quality_cv(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return image_quality(seed, cross_val=True)

@registry.register
def image_quality_pgan(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return image_quality(seed, with_pgan=True)

@registry.register
def image_quality_cv_pgan(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return image_quality(seed, cross_val=True, with_pgan=True)

@registry.register
def image_similarity_and_quality(seed: Optional[int] = None, cross_val: bool = False, with_pgan: bool = False) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("structured_moo", seed=next(seedg)))
    func_iqa = imagesxp.Image(loss=imagesxp.imagelosses.Koncept512, with_pgan=with_pgan)
    func_blur = imagesxp.Image(loss=imagesxp.imagelosses.Blur, with_pgan=with_pgan)
    base_blur_value = func_blur(func_blur.parametrization.value)
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
def image_similarity_and_quality_cv(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return image_similarity_and_quality(seed, cross_val=True)

@registry.register
def image_similarity_and_quality_pgan(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return image_similarity_and_quality(seed, with_pgan=True)

@registry.register
def image_similarity_and_quality_cv_pgan(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return image_similarity_and_quality(seed, cross_val=True, with_pgan=True)

@registry.register
def double_o_seven(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    base_env = rl.envs.DoubleOSeven(verbose=False)
    random_agent = rl.agents.Agent007(base_env)
    modules = {"mono": rl.agents.Perceptron, "multi": rl.agents.DenseNet}
    agents = {a: rl.agents.TorchAgent.from_module_maker(base_env, m, deterministic=False) for a, m in modules.items()}
    env = base_env.with_agent(player_0=random_agent).as_single_agent()
    dde = ng.optimizers.DifferentialEvolution(crossover="dimension").set_name("DiscreteDE")
    optimizers = refactor_optims(["NGOpt", "NGOptRW"])
    for num_repetitions in [1, 10, 100]:
        for archi in ["mono", "multi"]:
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
def multiobjective_example(seed: Optional[int] = None, hd: bool = False, many: bool = False) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(["PymooNSGA2", "PymooBatchNSGA2", "LPCMA", "VLPCMA", "CMA"])
    popsizes = [20, 40, 80]
    optims += [ng.families.EvolutionStrategy(recombination_ratio=r, only_offsprings=o, popsize=pop, offsprings=pop * 5) for o in [True, False] for r in [0.1, 0.5] for pop in popsizes]
    dim = 2000 if hd else 7
    mofuncs = []
    for name1, name2 in itertools.product(["sphere"], ["sphere", "hm"]):
        mofuncs.append(fbase.MultiExperiment([ArtificialFunction(name1, block_dimension=dim), ArtificialFunction(name2, block_dimension=dim)] + ([ArtificialFunction(name1, block_dimension=dim), ArtificialFunction(name2, block_dimension=dim)] if many else []), upper_bounds=[100, 100] * (2 if many else 1)))
        mofuncs.append(fbase.MultiExperiment([ArtificialFunction(name1, block_dimension=dim - 1), ArtificialFunction("sphere", block_dimension=dim - 1), ArtificialFunction(name2, block_dimension=dim - 1)] + ([ArtificialFunction(name1, block_dimension=dim - 1), ArtificialFunction("sphere", block_dimension=dim - 1), ArtificialFunction(name2, block_dimension=dim - 1)] if many else []), upper_bounds=[100, 100, 100.0] * (2 if many else 1)))
    for mofunc in mofuncs:
        for optim in optims:
            for budget in [100, 200, 400, 800, 1600, 3200]:
                for nw in [1, 100]:
                    xp = Experiment(mofunc, optim, budget=budget, num_workers=nw, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp

@registry.register
def multiobjective_example_hd(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return multiobjective_example(seed, hd=True)

@registry.register
def multiobjective_example_many_hd(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return multiobjective_example(seed, hd=True, many=True)

@registry.register
def multiobjective_example_many(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return multiobjective_example(seed, many=True)

@registry.register
def pbt(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(["NGOpt"])
    for func in PBT.itercases():
        for optim in optims:
            for budget in [100, 400, 1000, 4000, 10000]:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))

@registry.register
def far_optimum_es(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("es", "basics", seed=next(seedg)))
    for func in FarOptimumFunction.itercases():
        for optim in optims:
            for budget in [100, 400, 1000, 4000, 10000]:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))

@registry.register
def ceviche(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    instrum = ng.p.Array(shape=(40, 40), lower=0.0, upper=1.0).set_integer_casting()
    func = ExperimentFunction(photonics_ceviche, instrum.set_name("transition"))
    algos = refactor_optims(["DiagonalCMA", "PSO", "DE", "CMA", "OnePlusOne"])
    for budget in [20, 50, 100, 160, 240]:
        yield Experiment(func, algos[0], budget=budget, seed=next(seedg))

@registry.register
def multi_ceviche(seed: Optional[int] = None, c0: bool = False, precompute: bool = False, warmstart: bool = False) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    algos = refactor_optims(["NGOpt"])
    for benchmark_type in [np.random.choice([0, 1, 2, 3])]:
        budgets = ([np.random.choice([3, 20, 50, 90, 150, 250, 400, 800, 1600, 3200, 6400]),
                    np.random.choice([12800, 25600, 51200, 102400, 204800, 409600])]
                   if not precompute else [np.random.choice([409600, 204800 + 102400, 204800]) - 102400])
        if benchmark_type == 3:
            budgets = ([np.random.choice([3, 20, 50, 90, 150, 250, 400, 800, 1600, 3200, 6400]),
                        np.random.choice([12800, 25600, 51200, 102400])]
                       if not precompute else [np.random.choice([204800 + 51200, 204800]) - 102400])
        for optim in [np.random.choice(algos)]:
            for budget in budgets:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))
    # Note: Detailed implementation omitted due to complexity.

@registry.register
def multi_ceviche_c0(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return multi_ceviche(seed, c0=True)

@registry.register
def multi_ceviche_c0_warmstart(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return multi_ceviche(seed, c0=True, warmstart=True)

@registry.register
def multi_ceviche_c0p(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return multi_ceviche(seed, c0=True, precompute=True)

@registry.register
def photonics(seed: Optional[int] = None, as_tuple: bool = False, small: bool = False, ultrasmall: bool = False, verysmall: bool = False) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    divider = 2 if small else 1
    if ultrasmall or verysmall:
        divider = 4
    optims = refactor_optims(get_optimizers("es", "basics", "splitters", seed=next(seedg)))
    for method in ["clipping", "tanh"]:
        names = ["bragg"] if ultrasmall else (["cf_photosic_reference", "cf_photosic_realistic"] if verysmall else ["bragg", "chirped", "morpho", "cf_photosic_realistic", "cf_photosic_reference"])
        for name in names:
            func = Photonics(name, 4 * (60 // divider // 4) if name == "morpho" else 80 // divider, bounding_method=method, as_tuple=as_tuple)
            for budget in [10.0, 100.0, 1000.0]:
                for algo in optims:
                    xp = Experiment(func, algo, int(budget), num_workers=1, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp

@registry.register
def photonics2(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return photonics(seed, as_tuple=True)

@registry.register
def ultrasmall_photonics(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return photonics(seed, as_tuple=False, small=True, ultrasmall=True)

@registry.register
def ultrasmall_photonics2(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return photonics(seed, as_tuple=True, small=True, ultrasmall=True)

@registry.register
def verysmall_photonics(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return photonics(seed, as_tuple=False, small=True, verysmall=True)

@registry.register
def verysmall_photonics2(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return photonics(seed, as_tuple=True, small=True, verysmall=True)

@registry.register
def small_photonics(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return photonics(seed, as_tuple=False, small=True)

@registry.register
def small_photonics2(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return photonics(seed, as_tuple=True, small=True)

@registry.register
def adversarial_attack(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(get_optimizers("structure", "structured_moo", seed=next(seedg)))
    folder = os.environ.get("NEVERGRAD_ADVERSARIAL_EXPERIMENT_FOLDER", None)
    optims = optims
    for func in imagesxp.ImageAdversarial.make_folder_functions(folder=folder):
        for budget in [100, 200, 300, 400, 1700]:
            for num_workers in [1]:
                for algo in optims:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp

def pbo_suite(seed: Optional[int] = None, reduced: bool = False) -> Generator[Experiment, None, None]:
    dde = ng.optimizers.DifferentialEvolution(crossover="dimension").set_name("DiscreteDE")
    seedg = create_seed_generator(seed)
    list_optims = ["NGOpt", "NGOptRW"]
    list_optims = refactor_optims(list_optims)
    for dim in [16, 64, 100]:
        for fid in range(1, 24):
            for iid in range(1, 5):
                for instrumentation in ["Softmax", "Ordered", "Unordered"]:
                    try:
                        func = iohprofiler.PBOFunction(fid, iid, dim, instrumentation=instrumentation)
                        func.add_descriptors(instrum_str=instrumentation)
                    except ModuleNotFoundError as e:
                        raise fbase.UnsupportedExperiment("IOHexperimenter needs to be installed") from e
                    for optim in list_optims:
                        for nw in [1, 10]:
                            for budget in [100, 1000, 10000]:
                                yield Experiment(func, optim, num_workers=nw, budget=budget, seed=next(seedg))

@registry.register
def pbo_reduced_suite(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    return pbo_suite(seed, reduced=True)

def causal_similarity(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    from nevergrad.functions.causaldiscovery import CausalDiscovery
    seedg = create_seed_generator(seed)
    optims = refactor_optims(["NGOpt", "CMA", "DE", "PSO", "RecES", "RecMixES", "RecMutDE", "ParametrizationDE"])
    func = CausalDiscovery()
    for budget in [100 * 5 ** k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp

def unit_commitment(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(["NGOpt", "CMA", "DE", "PSO", "RecES", "RecMixES", "RecMutDE", "ParametrizationDE"])
    for num_timepoint in [5, 10, 20]:
        for num_generator in [3, 8]:
            func = UnitCommitmentProblem(num_timepoints=num_timepoint, num_generators=num_generator)
            for budget in [100 * 5 ** k for k in range(3)]:
                for algo in optims:
                    xp = Experiment(func, algo, budget, num_workers=1, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp

def team_cycling(seed: Optional[int] = None) -> Generator[Experiment, None, None]:
    seedg = create_seed_generator(seed)
    optims = refactor_optims(["NGOpt10", "CMA", "DE"])
    funcs = [Cycling(num) for num in [30, 31, 61, 22, 23, 45]]
    for function in funcs:
        for budget in [3000]:
            for optim in optims:
                xp = Experiment(function, optim, budget=budget, num_workers=10, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp

@registry.register
def lsgo() -> Generator[Experiment, None, None]:
    optims = refactor_optims(["NGOpt", "NGOptRW"])
    for i in range(1, 16):
        for optim in optims:
            for budget in [120000, 600000, 3000000]:
                yield Experiment(lsgo_makefunction(i).instrumented(), optim, budget=budget)

@registry.register
def smallbudget_lsgo() -> Generator[Experiment, None, None]:
    optims = refactor_optims(["NGOpt", "NGOptRW"])
    for i in range(1, 16):
        for optim in optims:
            for budget in [1200, 6000, 30000]:
                yield Experiment(lsgo_makefunction(i).instrumented(), optim, budget=budget)
