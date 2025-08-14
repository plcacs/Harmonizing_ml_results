import os
import warnings
import typing as tp
import numpy as np
import itertools
import nevergrad as ng
from nevergrad.functions import Experiment
from nevergrad.functions import FarOptimumFunction
from nevergrad.functions import ARCoating
from nevergrad.functions import OptimizeFish
from nevergrad.functions import Rocket
from nevergrad.functions import UnitCommitmentProblem
from nevergrad.functions import Cycling
from nevergrad.functions import TO
from nevergrad.functions.lsgo import make_function as lsgo_makefunction
from nevergrad.functions import control
from nevergrad.functions import rl
from nevergrad.functions import images as imagesxp
from nevergrad.functions.olympussurfaces import OlympusSurface, OlympusEmulator
from nevergrad.functions.causaldiscovery import CausalDiscovery

# Note: Some functions might be imported from different modules. Adjust the imports if necessary.

# All decorated functions below return an iterator of Experiment.
# Type alias for convenience.
ExperimentIterator = tp.Iterator[Experiment]
OptionalInt = tp.Optional[int]


@ng.optimizers.registry.register
def pbo_suite(seed: OptionalInt = None, reduced: bool = False) -> ExperimentIterator:
    dde = ng.optimizers.DifferentialEvolution(crossover="dimension").set_name("DiscreteDE")
    seedg = ng.functions.base.create_seed_generator(seed)
    index: int = 0
    list_optims: tp.List[tp.Union[str, ng.optimizers.Optimizer]] = [
        "DiscreteOnePlusOne",
        "Shiwa",
        "CMA",
        "PSO",
        "TwoPointsDE",
        "DE",
        "OnePlusOne",
        "AdaptiveDiscreteOnePlusOne",
        "CMandAS2",
        "PortfolioDiscreteOnePlusOne",
        "DoubleFastGADiscreteOnePlusOne",
        "MultiDiscrete",
        "cGA",
        dde,
    ]
    if reduced:
        list_optims = [
            x
            for x in ng.optimizers.registry.keys()
            if "iscre" in x and "ois" not in x and "ptim" not in x and "oerr" not in x
        ]
    list_optims = ["NGOpt", "NGOptRW"]
    list_optims = refactor_optims(list_optims)
    for dim in [16, 64, 100]:
        for fid in range(1, 24):
            for iid in range(1, 5):
                index += 1
                if reduced and index % 13:
                    continue
                for instrumentation in ["Softmax", "Ordered", "Unordered"]:
                    try:
                        func = ng.optimizers.iohprofiler.PBOFunction(fid, iid, dim, instrumentation=instrumentation)
                        func.add_descriptors(instrum_str=instrumentation)
                    except ModuleNotFoundError as e:
                        raise ng.functions.base.UnsupportedExperiment("IOHexperimenter needs to be installed") from e
                    for optim in list_optims:
                        for nw in [1, 10]:
                            for budget in [100, 1000, 10000]:
                                yield Experiment(func, optim, num_workers=nw, budget=budget, seed=next(seedg))


@ng.optimizers.registry.register
def pbo_reduced_suite(seed: OptionalInt = None) -> ExperimentIterator:
    return pbo_suite(seed, reduced=True)


@ng.optimizers.registry.register
def causal_similarity(seed: OptionalInt = None) -> ExperimentIterator:
    """Finding the best causal graph"""
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = ["CMA", "NGOpt8", "DE", "PSO", "RecES", "RecMixES", "RecMutDE", "ParametrizationDE"]
    func: CausalDiscovery = CausalDiscovery()
    optims = refactor_optims(optims)
    for budget in [100 * 5**k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


@ng.optimizers.registry.register
def unit_commitment(seed: OptionalInt = None) -> ExperimentIterator:
    """Unit commitment problem."""
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = ["CMA", "NGOpt8", "DE", "PSO", "RecES", "RecMixES", "RecMutDE", "ParametrizationDE"]
    optims = refactor_optims(optims)
    for num_timepoint in [5, 10, 20]:
        for num_generator in [3, 8]:
            func = UnitCommitmentProblem(num_timepoints=num_timepoint, num_generators=num_generator)
            for budget in [100 * 5**k for k in range(3)]:
                for algo in optims:
                    xp = Experiment(func, algo, budget, num_workers=1, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp


@ng.optimizers.registry.register
def team_cycling(seed: OptionalInt = None) -> ExperimentIterator:
    """Experiment to optimise team pursuit track cycling problem."""
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = ["NGOpt10", "CMA", "DE"]
    funcs = [Cycling(num) for num in [30, 31, 61, 22, 23, 45]]
    optims = refactor_optims(optims)
    for function in funcs:
        for budget in [3000]:
            for optim in optims:
                xp = Experiment(function, optim, budget=budget, num_workers=10, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


@ng.optimizers.registry.register
def lsgo() -> ExperimentIterator:
    optims = [
        "Shiwa",
        "Cobyla",
        "Powell",
        "CMandAS2",
        "SQP",
        "DE",
        "TwoPointsDE",
        "CMA",
        "PSO",
        "OnePlusOne",
        "RBFGS",
    ]
    optims = ["PSO", "RealPSO"]
    optims = ["CMA", "PSO", "SQOPSO", "TinyCMA", "Cobyla"]
    optims = ["TwoPointsDE", "DE", "LhsDE"]
    optims = [
        "DE",
        "TwoPointsDE",
        "VoronoiDE",
        "RotatedTwoPointsDE",
        "LhsDE",
        "QrDE",
        "QODE",
        "SODE",
        "NoisyDE",
        "AlmostRotationInvariantDE",
        "RotationInvariantDE",
        "DiscreteDE",
        "RecMutDE",
        "MutDE",
        "OnePointDE",
        "ParametrizationDE",
        "MiniDE",
        "MiniLhsDE",
        "MiniQrDE",
        "BPRotationInvariantDE",
        "HSDE",
        "LhsHSDE",
        "TinyLhsDE",
        "TinyQODE",
        "MetaModelDE",
        "MetaModelQODE",
        "NeuralMetaModelDE",
        "SVMMetaModelDE",
        "RFMetaModelDE",
        "MetaModelTwoPointsDE",
        "NeuralMetaModelTwoPointsDE",
        "SVMMetaModelTwoPointsDE",
        "RFMetaModelTwoPointsDE",
        "GeneticDE",
        "MemeticDE",
        "QNDE",
    ]
    optims = ["CMA", "NGOpt", "NGOptRW"]
    optims = ["DiagonalCMA", "TinyQODE", "OpoDE", "OpoTinyDE"]
    optims = ["TinyQODE", "OpoDE", "OpoTinyDE"]
    optims = refactor_optims(optims)
    for i in range(1, 16):
        for optim in optims:
            for budget in [120000, 600000, 3000000]:
                yield Experiment(lsgo_makefunction(i).instrumented(), optim, budget=budget)


@ng.optimizers.registry.register
def smallbudget_lsgo() -> ExperimentIterator:
    optims = [
        "Shiwa",
        "Cobyla",
        "Powell",
        "CMandAS2",
        "SQP",
        "DE",
        "TwoPointsDE",
        "CMA",
        "PSO",
        "OnePlusOne",
        "RBFGS",
    ]
    optims = ["PSO", "RealPSO"]
    optims = ["CMA", "PSO", "SQOPSO", "TinyCMA", "Cobyla"]
    optims = ["TwoPointsDE", "DE", "LhsDE"]
    optims = [
        "DE",
        "TwoPointsDE",
        "VoronoiDE",
        "RotatedTwoPointsDE",
        "LhsDE",
        "QrDE",
        "QODE",
        "SODE",
        "NoisyDE",
        "AlmostRotationInvariantDE",
        "RotationInvariantDE",
        "DiscreteDE",
        "RecMutDE",
        "MutDE",
        "OnePointDE",
        "ParametrizationDE",
        "MiniDE",
        "MiniLhsDE",
        "MiniQrDE",
        "BPRotationInvariantDE",
        "HSDE",
        "LhsHSDE",
        "TinyLhsDE",
        "TinyQODE",
        "MetaModelDE",
        "MetaModelQODE",
        "NeuralMetaModelDE",
        "SVMMetaModelDE",
        "RFMetaModelDE",
        "MetaModelTwoPointsDE",
        "NeuralMetaModelTwoPointsDE",
        "SVMMetaModelTwoPointsDE",
        "RFMetaModelTwoPointsDE",
        "GeneticDE",
        "MemeticDE",
        "QNDE",
    ]
    optims = ["CMA", "NGOpt", "NGOptRW"]
    optims = ["DiagonalCMA", "TinyQODE", "OpoDE", "OpoTinyDE"]
    optims = ["TinyQODE", "OpoDE", "OpoTinyDE"]
    optims = refactor_optims(optims)
    for i in range(1, 16):
        for optim in optims:
            for budget in [1200, 6000, 30000]:
                yield Experiment(lsgo_makefunction(i).instrumented(), optim, budget=budget)


@ng.optimizers.registry.register
def pbbob(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = [
        "OldCMA",
        "CMAbounded",
        "CMAsmall",
        "CMAstd",
        "CMApara",
        "CMAtuning",
        "DiagonalCMA",
        "FCMA",
        "RescaledCMA",
        "ASCMADEthird",
        "MultiCMA",
        "TripleCMA",
        "PolyCMA",
        "MultiScaleCMA",
        "DE",
        "OnePointDE",
        "GeneticDE",
        "TwoPointsDE",
        "PSO",
        "NGOptRW",
        "NGOpt",
    ]
    optims = ["ChainMetaModelSQP", "MetaModelOnePlusOne", "MetaModelDE"]
    optims = ["LargeCMA", "TinyCMA", "OldCMA", "MicroCMA"]
    optims = ["RBFGS", "LBFGSB", "MemeticDE"]
    optims = ["QrDE", "QODE", "LhsDE", "NGOpt", "NGOptRW"]
    optims = ["TinyCMA", "QODE", "MetaModelOnePlusOne", "LhsDE", "TinyLhsDE", "TinyQODE"]
    optims = ["QOPSO", "QORealSpacePSO"]
    optims = ["SQOPSO"]
    dims = [40, 20]
    functions = [
        ng.functions.corefuncs.ArtificialFunction(name, block_dimension=d, rotation=rotation, expo=expo)
        for name in ["cigar", "sphere", "rastrigin", "hm", "deceptivemultimodal"]
        for rotation in [True]
        for expo in [1.0, 3.0, 5.0, 7.0, 9.0]
        for d in dims
    ]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 300, 400, 500, 600, 700, 800]:
                for nw in [1, 10, 50]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))


@ng.optimizers.registry.register
def zp_pbbob(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = [
        "OldCMA",
        "CMAbounded",
        "CMAsmall",
        "CMAstd",
        "CMApara",
        "CMAtuning",
        "DiagonalCMA",
        "FCMA",
        "RescaledCMA",
        "ASCMADEthird",
        "MultiCMA",
        "TripleCMA",
        "PolyCMA",
        "MultiScaleCMA",
        "DE",
        "OnePointDE",
        "GeneticDE",
        "TwoPointsDE",
        "PSO",
        "NGOptRW",
        "NGOpt",
    ]
    optims = ["ChainMetaModelSQP", "MetaModelOnePlusOne", "MetaModelDE"]
    optims = ["LargeCMA", "TinyCMA", "OldCMA", "MicroCMA"]
    optims = ["RBFGS", "LBFGSB", "MemeticDE"]
    optims = ["QrDE", "QODE", "LhsDE", "NGOpt", "NGOptRW"]
    optims = ["TinyCMA", "QODE", "MetaModelOnePlusOne", "LhsDE", "TinyLhsDE", "TinyQODE"]
    optims = ["QOPSO", "QORealSpacePSO"]
    optims = ["SQOPSO"]
    dims = [40, 20]
    functions = [
        ng.functions.corefuncs.ArtificialFunction(name, block_dimension=d, rotation=rotation, expo=expo, zero_pen=True)
        for name in ["cigar", "sphere", "rastrigin", "hm", "deceptivemultimodal"]
        for rotation in [True]
        for expo in [1.0, 3.0, 5.0, 7.0, 9.0]
        for d in dims
    ]
    optims = ["QODE", "PSO", "SQOPSO", "DE", "CMA"]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 300, 400, 500, 600, 700, 800]:
                for nw in [1, 10, 50]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))


@ng.optimizers.registry.register
def illcondi(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    functions = [
        ng.functions.corefuncs.ArtificialFunction(name, block_dimension=50, rotation=rotation)
        for name in ["cigar", "ellipsoid"]
        for rotation in [True, False]
    ]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 1000, 10000]:
                yield Experiment(function, optim, budget=budget, num_workers=1, seed=next(seedg))


@ng.optimizers.registry.register
def illcondipara(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    functions = [
        ng.functions.corefuncs.ArtificialFunction(name, block_dimension=50, rotation=rotation)
        for name in ["cigar", "ellipsoid"]
        for rotation in [True, False]
    ]
    optims = get_optimizers("competitive", seed=next(seedg))
    optims = refactor_optims(optims)
    for function in functions:
        for budget in [100, 1000, 10000]:
            for optim in optims:
                xp = Experiment(function, optim, budget=budget, num_workers=50, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


@ng.optimizers.registry.register
def constrained_illconditioned_parallel(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    functions = [
        ng.functions.corefuncs.ArtificialFunction(name, block_dimension=50, rotation=rotation)
        for name in ["cigar", "ellipsoid"]
        for rotation in [True, False]
    ]
    for func in functions:
        func.parametrization.register_cheap_constraint(_Constraint("sum", as_bool=False))
    optims: tp.List[str] = ["DE", "CMA", "NGOpt"]
    optims = refactor_optims(optims)
    for function in functions:
        for budget in [400, 4000, 40000]:
            optims = get_optimizers("large", seed=next(seedg))
            for optim in optims:
                yield Experiment(function, optim, budget=budget, num_workers=1, seed=next(seedg))


@ng.optimizers.registry.register
def ranknoisy(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("progressive", seed=next(seedg)) + [
        "OptimisticNoisyOnePlusOne",
        "OptimisticDiscreteOnePlusOne",
        "NGOpt10",
    ]
    optims = ["SPSA", "TinySPSA", "TBPSA", "NoisyOnePlusOne", "NoisyDiscreteOnePlusOne"]
    optims = get_optimizers("basics", "noisy", "splitters", "progressive", seed=next(seedg))
    optims = refactor_optims(optims)
    for budget in [25000, 50000, 100000]:
        for optim in optims:
            for d in [20000, 200, 2000]:
                for name in ["cigar", "altcigar", "ellipsoid", "altellipsoid"]:
                    for noise_dissymmetry in [False, True]:
                        function = ng.functions.corefuncs.ArtificialFunction(
                            name=name,
                            rotation=False,
                            block_dimension=d,
                            noise_level=10,
                            noise_dissymmetry=noise_dissymmetry,
                            translation_factor=1.0,
                        )
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))


@ng.optimizers.registry.register
def noisy(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("progressive", seed=next(seedg)) + [
        "OptimisticNoisyOnePlusOne",
        "OptimisticDiscreteOnePlusOne",
    ]
    optims += ["NGOpt10", "Shiwa", "DiagonalCMA"] + sorted(
        x for x, y in ng.optimizers.registry.items() if ("SPSA" in x or "TBPSA" in x or "ois" in x or "epea" in x or "Random" in x)
    )
    optims = refactor_optims(optims)
    for budget in [25000, 50000, 100000]:
        for optim in optims:
            for d in [2, 20, 200, 2000]:
                for name in ["sphere", "rosenbrock", "cigar", "hm"]:
                    for noise_dissymmetry in [False, True]:
                        function = ng.functions.corefuncs.ArtificialFunction(
                            name=name,
                            rotation=True,
                            block_dimension=d,
                            noise_level=10,
                            noise_dissymmetry=noise_dissymmetry,
                            translation_factor=1.0,
                        )
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))


@ng.optimizers.registry.register
def paraalldes(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    for budget in [10, 100, 1000, 10000, 100000]:
        for optim in sorted(x for x, y in ng.optimizers.registry.items() if "DE" in x and "Tune" in x):
            for rotation in [False]:
                for d in [5, 20, 100, 500, 2500]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ng.functions.corefuncs.ArtificialFunction(
                                name=name,
                                rotation=rotation,
                                block_dimension=d,
                                useless_variables=d * u,
                                translation_factor=1.0,
                            )
                            yield Experiment(
                                function,
                                optim,
                                budget=budget,
                                seed=next(seedg),
                                num_workers=max(d, budget // 6),
                            )


@ng.optimizers.registry.register
def parahdbo4d(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    for budget in [25, 31, 37, 43, 50, 60]:
        for optim in refactor_optims(sorted(x for x, y in ng.optimizers.registry.items() if "BO" in x and "Tune" in x)):
            for rotation in [False]:
                for d in [20, 2000]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ng.functions.corefuncs.ArtificialFunction(
                                name=name,
                                rotation=rotation,
                                block_dimension=d,
                                useless_variables=d * u,
                                translation_factor=1.0,
                            )
                            yield Experiment(
                                function,
                                optim,
                                budget=budget,
                                seed=next(seedg),
                                num_workers=max(d, budget // 6),
                            )


@ng.optimizers.registry.register
def alldes(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    for budget in [10, 100, 1000, 10000, 100000]:
        for optim in refactor_optims(sorted(x for x, y in ng.optimizers.registry.items() if "DE" in x or "Shiwa" in x)):
            for rotation in [False]:
                for d in [5, 20, 100]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ng.functions.corefuncs.ArtificialFunction(
                                name=name,
                                rotation=rotation,
                                block_dimension=d,
                                useless_variables=d * u,
                                translation_factor=1.0,
                            )
                            yield Experiment(function, optim, budget=budget, seed=next(seedg))


@ng.optimizers.registry.register
def hdbo4d(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    for budget in [25, 31, 37, 43, 50, 60]:
        for optim in refactor_optims(get_optimizers("all_bo", seed=next(seedg))):
            for rotation in [False]:
                for d in [20]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ng.functions.corefuncs.ArtificialFunction(
                                name=name,
                                rotation=rotation,
                                block_dimension=d,
                                useless_variables=d * u,
                                translation_factor=1.0,
                            )
                            yield Experiment(function, optim, budget=budget, seed=next(seedg))


@ng.optimizers.registry.register
def spsa_benchmark(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("spsa", seed=next(seedg))
    optims += ["NGOpt", "NGOptRW"]
    optims = refactor_optims(optims)
    for budget in [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        for optim in optims:
            for rotation in [True, False]:
                for name in ["sphere", "sphere4", "cigar"]:
                    function = ng.functions.corefuncs.ArtificialFunction(
                        name=name, rotation=rotation, block_dimension=20, noise_level=10
                    )
                    yield Experiment(function, optim, budget=budget, seed=next(seedg))


@ng.optimizers.registry.register
def realworld(seed: OptionalInt = None) -> ExperimentIterator:
    funcs: tp.List[tp.Union[Experiment, rl.agents.TorchAgentFunction]] = [
        ng.functions.mlda.Clustering.from_mlda(name, num, rescale)
        for name, num in [("Ruspini", 5), ("German towns", 10)]
        for rescale in [True, False]
    ]
    funcs += [
        ng.functions.mlda.SammonMapping.from_mlda("Virus", rescale=False),
        ng.functions.mlda.SammonMapping.from_mlda("Virus", rescale=True),
        ng.functions.mlda.Landscape(transform)
        for transform in [None, "square", "gaussian"]
    ]
    funcs += [ARCoating()]
    funcs += [ng.functions.powersystems.PowerSystem(), ng.functions.powersystems.PowerSystem(13)]
    funcs += [ng.functions.stsp.STSP(), ng.functions.stsp.STSP(500)]
    funcs += [ng.functions.games.game.Game("war")]
    funcs += [ng.functions.games.game.Game("batawaf")]
    funcs += [ng.functions.games.game.Game("flip")]
    funcs += [ng.functions.games.game.Game("guesswho")]
    funcs += [ng.functions.games.game.Game("bigguesswho")]
    base_env = rl.envs.DoubleOSeven(verbose=False)
    random_agent = rl.agents.Agent007(base_env)
    modules = {"mono": rl.agents.Perceptron, "multi": rl.agents.DenseNet}
    agents = {
        a: rl.agents.TorchAgent.from_module_maker(base_env, m, deterministic=False)
        for a, m in modules.items()
    }
    env = base_env.with_agent(player_0=random_agent).as_single_agent()
    runner = rl.EnvironmentRunner(env.copy(), num_repetitions=100, max_step=50)
    for archi in ["mono", "multi"]:
        func = rl.agents.TorchAgentFunction(agents[archi], runner, reward_postprocessing=lambda x: 1 - x)
        funcs += [func]
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@ng.optimizers.registry.register
def aquacrop_fao(seed: OptionalInt = None) -> ExperimentIterator:
    funcs = [ng.functions.ac.NgAquacrop(i, 300.0 + 150.0 * np.cos(i)) for i in range(3, 7)]
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    optims = ["RBFGS", "LBFGSB", "MemeticDE"]
    optims = ["PCABO"]
    optims = ["PCABO", "NGOpt", "QODE"]
    optims = ["QOPSO"]
    optims = ["NGOpt"]
    optims = ["SQOPSO"]
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


@ng.optimizers.registry.register
def fishing(seed: OptionalInt = None) -> ExperimentIterator:
    funcs = [OptimizeFish(i) for i in [17, 35, 52, 70, 88, 105]]
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    optims += ["NGOpt", "NGOptRW", "ChainMetaModelSQP"]
    optims = ["NGOpt"]
    optims = ["PCABO"]
    optims = ["PCABO", "NGOpt", "QODE"]
    optims = ["QOPSO"]
    optims = ["SQOPSO"]
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600]:
        for algo in optims:
            for fu in funcs:
                xp = Experiment(fu, algo, budget, seed=next(seedg))
                xp.function.parametrization.real_world = True
                if not xp.is_incoherent:
                    yield xp


@ng.optimizers.registry.register
def rocket(seed: OptionalInt = None, seq: bool = False) -> ExperimentIterator:
    funcs = [Rocket(i) for i in range(17)]
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    optims += ["NGOpt", "NGOptRW", "ChainMetaModelSQP"]
    optims = ["RBFGS", "LBFGSB", "MemeticDE"]
    optims = ["CMA", "PSO", "QODE", "QRDE", "MetaModelPSO"]
    if seq:
        optims += ["RBFGS", "LBFGSB", "MemeticDE"]
    optims = ["NGOpt"]
    optims = ["PCABO"]
    optims = ["PCABO", "NGOpt", "QODE"]
    optims = ["QOPSO"]
    optims = ["SQOPSO"]
    optims = ["NGOpt", "QOPSO", "SOPSO", "QODE", "SODE", "CMA", "DiagonalCMA", "MetaModelOnePlusOne", "MetaModelDE"]
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600]:
        for num_workers in [1] if seq else [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        xp.function.parametrization.real_world = True
                        ng.functions.base.skip_ci(reason="Too slow")
                        if not xp.is_incoherent:
                            yield xp


@ng.optimizers.registry.register
def mono_rocket(seed: OptionalInt = None) -> ExperimentIterator:
    return rocket(seed, seq=True)


@ng.optimizers.registry.register
def mixsimulator(seed: OptionalInt = None) -> ExperimentIterator:
    funcs: tp.List[tp.Any] = [ng.functions.mixsimulator.OptimizeMix()]
    seedg = ng.functions.base.create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("basics", seed=next(seedg))
    optims = refactor_optims(optims)
    for budget in [20, 40, 80, 160]:
        for num_workers in [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@ng.optimizers.registry.register
def control_problem(seed: OptionalInt = None) -> ExperimentIterator:
    num_rollouts: int = 1
    seedg = ng.functions.base.create_seed_generator(seed)
    funcs = [
        Env(num_rollouts=num_rollouts, random_state=seed)
        for Env in [
            control.Swimmer,
            control.HalfCheetah,
            control.Hopper,
            control.Walker2d,
            control.Ant,
            control.Humanoid,
        ]
    ]
    sigmas = [0.1, 0.1, 0.1, 0.1, 0.01, 0.001]
    funcs2: tp.List[tp.Any] = []
    for sigma, func in zip(sigmas, funcs):
        f = func.copy()
        param: ng.p.Parameter = f.parametrization.copy()  # type: ignore
        for array in param:
            array.set_mutation(sigma=sigma)  # type: ignore
        param.set_name(f"sigma={sigma}")
        f.parametrization = param
        f.parametrization.freeze()
        funcs2.append(f)
    optims = get_optimizers("basics")
    optims = ["NGOpt", "PSO", "CMA"]
    optims = refactor_optims(optims)
    for budget in [50, 75, 100, 150, 200, 250, 300, 400, 500, 1000, 3000, 5000, 8000, 16000, 32000, 64000]:
        for algo in optims:
            for fu in funcs2:
                xp = Experiment(fu, algo, budget, num_workers=1, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


@ng.optimizers.registry.register
def neuro_control_problem(seed: OptionalInt = None) -> ExperimentIterator:
    num_rollouts: int = 1
    seedg = ng.functions.base.create_seed_generator(seed)
    funcs = [
        Env(num_rollouts=num_rollouts, intermediate_layer_dim=(50,), random_state=seed)
        for Env in [
            control.Swimmer,
            control.HalfCheetah,
            control.Hopper,
            control.Walker2d,
            control.Ant,
            control.Humanoid,
        ]
    ]
    optims = ["CMA", "NGOpt4", "DiagonalCMA", "NGOpt8", "MetaModel", "ChainCMAPowell"]
    optims = ["NGOpt", "CMA", "PSO"]
    optims = refactor_optims(optims)
    for budget in [50, 500, 5000, 10000, 20000, 35000, 50000, 100000, 200000]:
        for algo in optims:
            for fu in funcs:
                xp = Experiment(fu, algo, budget, num_workers=1, seed=next(seedg))
                xp.function.parametrization.real_world = True
                xp.function.parametrization.neural = True
                if not xp.is_incoherent:
                    yield xp


@ng.optimizers.registry.register
def olympus_surfaces(seed: OptionalInt = None) -> ExperimentIterator:
    funcs: tp.List[OlympusSurface] = []
    for kind in OlympusSurface.SURFACE_KINDS:
        for k in range(2, 5):
            for noise in ["GaussianNoise", "UniformNoise", "GammaNoise"]:
                funcs.append(OlympusSurface(kind, 10**k, noise, noise_scale=0.5 if k < 4 else 1))
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = get_optimizers("basics", "noisy", seed=next(seedg))
    optims = ["NGOpt", "CMA"]
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, int(budget), num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@ng.optimizers.registry.register
def olympus_emulators(seed: OptionalInt = None) -> ExperimentIterator:
    funcs: tp.List[OlympusEmulator] = []
    for dataset_kind in OlympusEmulator.DATASETS:
        for model_kind in ["BayesNeuralNet", "NeuralNet"]:
            funcs.append(OlympusEmulator(dataset_kind, model_kind))
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = get_optimizers("basics", "noisy", seed=next(seedg))
    optims = ["NGOpt", "CMA"]
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, int(budget), num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@ng.optimizers.registry.register
def topology_optimization(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    funcs = [TO(i) for i in [10, 20, 30, 40]]
    optims = ["CMA", "GeneticDE", "TwoPointsDE", "VoronoiDE", "DE", "PSO", "RandomSearch", "OnePlusOne"]
    optims = ["NGOpt"]
    optims = refactor_optims(optims)
    for budget in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960]:
        for optim in optims:
            for f in funcs:
                for nw in [1, 30]:
                    yield Experiment(f, optim, budget, num_workers=nw, seed=next(seedg))


@ng.optimizers.registry.register
def sequential_topology_optimization(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    funcs = [TO(i) for i in [10, 20, 30, 40]]
    optims = ["CMA", "GeneticDE", "TwoPointsDE", "VoronoiDE", "DE", "PSO", "RandomSearch", "OnePlusOne"]
    optims = ["NGOpt"]
    optims = refactor_optims(optims)
    for budget in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960]:
        for optim in optims:
            for f in funcs:
                for nw in [1, 30]:
                    yield Experiment(f, optim, budget, num_workers=nw, seed=next(seedg))


@ng.optimizers.registry.register
def simple_tsp(seed: OptionalInt = None, complex_tsp: bool = False) -> ExperimentIterator:
    funcs = [ng.functions.stsp.STSP(10**k, complex_tsp) for k in range(2, 6)]
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = [
        "RotatedTwoPointsDE",
        "DiscreteLenglerOnePlusOne",
        "DiscreteDoerrOnePlusOne",
        "DiscreteBSOOnePlusOne",
        "AdaptiveDiscreteOnePlusOne",
        "GeneticDE",
        "DE",
        "TwoPointsDE",
        "DiscreteOnePlusOne",
        "CMA",
        "MetaModel",
        "DiagonalCMA",
    ]
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@ng.optimizers.registry.register
def complex_tsp(seed: OptionalInt = None) -> ExperimentIterator:
    return simple_tsp(seed, complex_tsp=True)


@ng.optimizers.registry.register
def sequential_fastgames(seed: OptionalInt = None) -> ExperimentIterator:
    funcs = [ng.functions.games.game.Game(name) for name in ["war", "batawaf", "flip", "guesswho", "bigguesswho"]]
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = get_optimizers("noisy", "splitters", "progressive", seed=next(seedg))
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


@ng.optimizers.registry.register
def powersystems(seed: OptionalInt = None) -> ExperimentIterator:
    funcs: tp.List[tp.Union[ng.functions.OptimizeMix, ng.functions.powersystems.PowerSystem]] = []
    for dams in [3, 5, 9, 13]:
        funcs += [ng.functions.powersystems.PowerSystem(dams, depth=2, width=3)]
    seedg = ng.functions.base.create_seed_generator(seed)
    budgets = [3200, 6400, 12800]
    optims = get_optimizers("basics", "noisy", "splitters", "progressive", seed=next(seedg))
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


@ng.optimizers.registry.register
def mlda(seed: OptionalInt = None) -> ExperimentIterator:
    funcs: tp.List[ng.functions.ExperimentFunction] = [
        ng.functions.mlda.Clustering.from_mlda(name, num, rescale)
        for name, num in [("Ruspini", 5), ("German towns", 10)]
        for rescale in [True, False]
    ]
    funcs += [
        ng.functions.mlda.SammonMapping.from_mlda("Virus", rescale=False),
        ng.functions.mlda.SammonMapping.from_mlda("Virus", rescale=True),
        ng.functions.mlda.Perceptron.from_mlda(name)
        for name in ["quadratic", "sine", "abs", "heaviside"]
    ]
    funcs += [ng.functions.mlda.Landscape(transform) for transform in [None, "square", "gaussian"]]
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    xp = Experiment(funcs[0], algo, budget, num_workers=num_workers, seed=next(seedg))
                    xp.function.parametrization.real_world = True
                    if not xp.is_incoherent:
                        yield xp


@ng.optimizers.registry.register
def mldakmeans(seed: OptionalInt = None) -> ExperimentIterator:
    funcs: tp.List[ng.functions.ExperimentFunction] = [
        ng.functions.mlda.Clustering.from_mlda(name, num, rescale)
        for name, num in [("Ruspini", 5), ("German towns", 10), ("Ruspini", 50), ("German towns", 100)]
        for rescale in [True, False]
    ]
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = get_optimizers("splitters", "progressive", seed=next(seedg))
    optims += ["DE", "CMA", "PSO", "TwoPointsDE", "RandomSearch"]
    optims = ["QODE", "QRDE"]
    optims = ["NGOpt"]
    optims = refactor_optims(optims)
    for budget in [1000, 10000]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for func in funcs:
                        xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@ng.optimizers.registry.register
def image_similarity(
    seed: OptionalInt = None, with_pgan: bool = False, similarity: bool = True
) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = get_optimizers("structured_moo", seed=next(seedg))
    funcs: tp.List[ng.functions.ExperimentFunction] = [
        imagesxp.Image(loss=loss, with_pgan=with_pgan)
        for loss in imagesxp.imagelosses.registry.values()
        if loss.REQUIRES_REFERENCE == similarity
    ]
    optims = refactor_optims(optims)
    for budget in [100 * 5**k for k in range(3)]:
        for func in funcs:
            for algo in optims:
                xp = Experiment(func, algo, budget, num_workers=1, seed=next(seedg))
                ng.functions.base.skip_ci(reason="too slow")
                if not xp.is_incoherent:
                    yield xp


@ng.optimizers.registry.register
def image_similarity_pgan(seed: OptionalInt = None) -> ExperimentIterator:
    return image_similarity(seed, with_pgan=True)


@ng.optimizers.registry.register
def image_single_quality(seed: OptionalInt = None) -> ExperimentIterator:
    return image_similarity(seed, with_pgan=False, similarity=False)


@ng.optimizers.registry.register
def image_single_quality_pgan(seed: OptionalInt = None) -> ExperimentIterator:
    return image_similarity(seed, with_pgan=True, similarity=False)


@ng.optimizers.registry.register
def image_multi_similarity(
    seed: OptionalInt = None, cross_valid: bool = False, with_pgan: bool = False
) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = get_optimizers("structured_moo", seed=next(seedg))
    funcs: tp.List[ng.functions.ExperimentFunction] = [
        imagesxp.Image(loss=loss, with_pgan=with_pgan)
        for loss in imagesxp.imagelosses.registry.values()
        if loss.REQUIRES_REFERENCE
    ]
    base_values: tp.List[tp.Any] = [func(func.parametrization.sample().value) for func in funcs]
    if cross_valid:
        ng.functions.base.skip_ci(reason="Too slow")
        mofuncs: tp.List[tp.Any] = imagesxp.helpers.SpecialEvaluationExperiment.create_crossvalidation_experiments(
            funcs, pareto_size=25
        )
    else:
        mofuncs = [ng.functions.base.MultiExperiment(funcs, upper_bounds=base_values)]  # type: ignore
    optims = refactor_optims(optims)
    for budget in [100 * 5**k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                for mofunc in mofuncs:
                    xp = Experiment(mofunc, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp


@ng.optimizers.registry.register
def image_multi_similarity_pgan(seed: OptionalInt = None) -> ExperimentIterator:
    return image_multi_similarity(seed, with_pgan=True)


@ng.optimizers.registry.register
def image_multi_similarity_cv(seed: OptionalInt = None) -> ExperimentIterator:
    return image_multi_similarity(seed, cross_valid=True)


@ng.optimizers.registry.register
def image_multi_similarity_pgan_cv(seed: OptionalInt = None) -> ExperimentIterator:
    return image_multi_similarity(seed, cross_valid=True, with_pgan=True)


@ng.optimizers.registry.register
def image_quality_proxy(seed: OptionalInt = None, with_pgan: bool = False) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optims: tp.List[tp.Any] = get_optimizers("structured_moo", seed=next(seedg))
    iqa = imagesxp.Image(loss=imagesxp.imagelosses.Koncept512, with_pgan=with_pgan)
    blur = imagesxp.Image(loss=imagesxp.imagelosses.Blur, with_pgan=with_pgan)
    brisque = imagesxp.Image(loss=imagesxp.imagelosses.Brisque, with_pgan=with_pgan)
    optims = refactor_optims(optims)
    for budget in [100 * 5**k for k in range(3)]:
        for algo in optims:
            for func in [blur, brisque]:
                sfunc = imagesxp.helpers.SpecialEvaluationExperiment(func, evaluation=iqa)
                sfunc.add_descriptors(non_proxy_function=False)
                xp = Experiment(sfunc, algo, budget, num_workers=1, seed=next(seedg))
                yield xp


@ng.optimizers.registry.register
def image_quality_proxy_pgan(seed: OptionalInt = None) -> ExperimentIterator:
    return image_quality_proxy(seed, with_pgan=True)


@ng.optimizers.registry.register
def image_quality(
    seed: OptionalInt = None, cross_val: bool = False, with_pgan: bool = False, num_images: int = 1
) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optims: tp.List[tp.Any] = get_optimizers("structured_moo", seed=next(seedg))
    funcs: tp.List[ng.functions.ExperimentFunction] = [
        imagesxp.Image(loss=loss, with_pgan=with_pgan, num_images=num_images)
        for loss in (imagesxp.imagelosses.Koncept512, imagesxp.imagelosses.Blur, imagesxp.imagelosses.Brisque)
    ]
    if cross_val:
        mofuncs = imagesxp.helpers.SpecialEvaluationExperiment.create_crossvalidation_experiments(
            experiments=[funcs[0], funcs[2]], training_only_experiments=[funcs[1]], pareto_size=16
        )
    else:
        upper_bounds = [func(func.parametrization.value) for func in funcs]  # type: ignore
        mofuncs = [ng.functions.base.MultiExperiment(funcs, upper_bounds=upper_bounds)]  # type: ignore
    optims = refactor_optims(optims)
    for budget in [100 * 5**k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                for func in mofuncs:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp


@ng.optimizers.registry.register
def morphing_pgan_quality(seed: OptionalInt = None) -> ExperimentIterator:
    return image_quality(seed, with_pgan=True, num_images=2)


@ng.optimizers.registry.register
def image_quality_cv(seed: OptionalInt = None) -> ExperimentIterator:
    return image_quality(seed, cross_val=True)


@ng.optimizers.registry.register
def image_quality_pgan(seed: OptionalInt = None) -> ExperimentIterator:
    return image_quality(seed, with_pgan=True)


@ng.optimizers.registry.register
def image_quality_cv_pgan(seed: OptionalInt = None) -> ExperimentIterator:
    return image_quality(seed, cross_val=True, with_pgan=True)


@ng.optimizers.registry.register
def image_similarity_and_quality(
    seed: OptionalInt = None, cross_val: bool = False, with_pgan: bool = False
) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optims: tp.List[tp.Any] = get_optimizers("structured_moo", seed=next(seedg))
    func_iqa = imagesxp.Image(loss=imagesxp.imagelosses.Koncept512, with_pgan=with_pgan)
    func_blur = imagesxp.Image(loss=imagesxp.imagelosses.Blur, with_pgan=with_pgan)
    base_blur_value: float = func_blur(func_blur.parametrization.value)  # type: ignore
    optims = refactor_optims(optims)
    for func in [
        imagesxp.Image(loss=loss, with_pgan=with_pgan)
        for loss in imagesxp.imagelosses.registry.values()
        if loss.REQUIRES_REFERENCE
    ]:
        base_value: float = func(func.parametrization.value)  # type: ignore
        if cross_val:
            mofuncs = imagesxp.helpers.SpecialEvaluationExperiment.create_crossvalidation_experiments(
                training_only_experiments=[func, func_blur], experiments=[func_iqa], pareto_size=16
            )
        else:
            mofuncs = [ng.functions.base.MultiExperiment(
                [func, func_blur, func_iqa], upper_bounds=[base_value, base_blur_value, 100.0]
            )]
        for budget in [100 * 5**k for k in range(3)]:
            for algo in optims:
                for mofunc in mofuncs:
                    xp = Experiment(mofunc, algo, budget, num_workers=1, seed=next(seedg))
                    yield xp


@ng.optimizers.registry.register
def image_similarity_and_quality_cv(seed: OptionalInt = None) -> ExperimentIterator:
    return image_similarity_and_quality(seed, cross_val=True)


@ng.optimizers.registry.register
def image_similarity_and_quality_pgan(seed: OptionalInt = None) -> ExperimentIterator:
    return image_similarity_and_quality(seed, with_pgan=True)


@ng.optimizers.registry.register
def image_similarity_and_quality_cv_pgan(seed: OptionalInt = None) -> ExperimentIterator:
    return image_similarity_and_quality(seed, cross_val=True, with_pgan=True)


@ng.optimizers.registry.register
def double_o_seven(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    base_env = rl.envs.DoubleOSeven(verbose=False)
    random_agent = rl.agents.Agent007(base_env)
    modules = {"mono": rl.agents.Perceptron, "multi": rl.agents.DenseNet}
    agents = {
        a: rl.agents.TorchAgent.from_module_maker(base_env, m, deterministic=False)
        for a, m in modules.items()
    }
    env = base_env.with_agent(player_0=random_agent).as_single_agent()
    dde = ng.optimizers.DifferentialEvolution(crossover="dimension").set_name("DiscreteDE")
    optimizers: tp.List[tp.Any] = [
        "PSO",
        dde,
        "MetaTuneRecentering",
        "DiagonalCMA",
        "TBPSA",
        "SPSA",
        "RecombiningOptimisticNoisyDiscreteOnePlusOne",
        "MetaModelPSO",
    ]
    optimizers = ["NGOpt", "NGOptRW"]
    optimizers = refactor_optims(optimizers)
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


@ng.optimizers.registry.register
def multiobjective_example(
    seed: OptionalInt = None, hd: bool = False, many: bool = False
) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = get_optimizers("structure", "structured_moo", seed=next(seedg))
    optims += [
        ng.families.DifferentialEvolution(multiobjective_adaptation=False).set_name("DE-noadapt"),
        ng.families.DifferentialEvolution(crossover="twopoints", multiobjective_adaptation=False).set_name("TwoPointsDE-noadapt"),
    ]
    optims += ["DiscreteOnePlusOne", "DiscreteLenglerOnePlusOne"]
    optims = ["PymooNSGA2", "PymooBatchNSGA2", "LPCMA", "VLPCMA", "CMA"]
    optims = ["LPCMA", "VLPCMA", "CMA"]
    popsizes = [20, 40, 80]
    optims += [
        ng.families.EvolutionStrategy(recombination_ratio=recomb, only_offsprings=only, popsize=pop, offsprings=pop * 5)
        for only in [True, False]
        for recomb in [0.1, 0.5]
        for pop in popsizes
    ]
    optims = refactor_optims(optims)
    mofuncs: tp.List[ng.functions.base.MultiExperiment] = []
    dim: int = 2000 if hd else 7
    for name1, name2 in itertools.product(["sphere"], ["sphere", "hm"]):
        mofuncs.append(
            ng.functions.base.MultiExperiment(
                [ng.functions.corefuncs.ArtificialFunction(name1, block_dimension=dim),
                 ng.functions.corefuncs.ArtificialFunction(name2, block_dimension=dim)],
                upper_bounds=[100, 100],
            )
        )
        mofuncs.append(
            ng.functions.base.MultiExperiment(
                [ng.functions.corefuncs.ArtificialFunction(name1, block_dimension=dim - 1),
                 ng.functions.corefuncs.ArtificialFunction("sphere", block_dimension=dim - 1),
                 ng.functions.corefuncs.ArtificialFunction(name2, block_dimension=dim - 1)],
                upper_bounds=[100, 100, 100.0],
            )
        )
        if many:
            mofuncs[-2].add_descriptors(objectives="many")
            mofuncs[-1].add_descriptors(objectives="many")
    for mofunc in mofuncs:
        for optim in optims:
            for budget in [100, 200, 400, 800, 1600, 3200]:
                for nw in [1, 100]:
                    xp = Experiment(mofunc, optim, budget=budget, num_workers=nw, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp


@ng.optimizers.registry.register
def multiobjective_example_hd(seed: OptionalInt = None) -> ExperimentIterator:
    return multiobjective_example(seed, hd=True)


@ng.optimizers.registry.register
def multiobjective_example_many_hd(seed: OptionalInt = None) -> ExperimentIterator:
    return multiobjective_example(seed, hd=True, many=True)


@ng.optimizers.registry.register
def multiobjective_example_many(seed: OptionalInt = None) -> ExperimentIterator:
    return multiobjective_example(seed, many=True)


@ng.optimizers.registry.register
def pbt(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optimizers = [
        "CMA",
        "TwoPointsDE",
        "Shiwa",
        "OnePlusOne",
        "DE",
        "PSO",
        "NaiveTBPSA",
        "RecombiningOptimisticNoisyDiscreteOnePlusOne",
        "PortfolioNoisyDiscreteOnePlusOne",
    ]
    optimizers = refactor_optims(optimizers)
    for func in ng.functions.pbt.PBT.itercases():
        for optim in optimizers:
            for budget in [100, 400, 1000, 4000, 10000]:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))


@ng.optimizers.registry.register
def far_optimum_es(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optims = get_optimizers("es", "basics", seed=next(seedg))
    optims = refactor_optims(optims)
    for func in FarOptimumFunction.itercases():
        for optim in optims:
            for budget in [100, 400, 1000, 4000, 10000]:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))


@ng.optimizers.registry.register
def ceviche(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    instrum = ng.p.Array(shape=(40, 40), lower=0.0, upper=1.0).set_integer_casting()
    func = Experiment(imagesxp.photonics_ceviche, instrum.set_name("transition"))
    algos = [
        "DiagonalCMA",
        "PSO",
        "DE",
        "CMA",
        "OnePlusOne",
        "LognormalDiscreteOnePlusOne",
        "DiscreteLenglerOnePlusOne",
        "MetaModel",
        "MetaModelDE",
        "MetaModelDSproba",
        "MetaModelOnePlusOne",
        "MetaModelPSO",
        "MetaModelQODE",
        "MetaModelTwoPointsDE",
        "NeuralMetaModel",
        "NeuralMetaModelDE",
        "NeuralMetaModelTwoPointsDE",
        "RFMetaModel",
        "RFMetaModelDE",
        "RFMetaModelOnePlusOne",
        "RFMetaModelPSO",
        "RFMetaModelTwoPointsDE",
        "SVMMetaModel",
        "SVMMetaModelDE",
        "SVMMetaModelPSO",
        "SVMMetaModelTwoPointsDE",
        "RandRecombiningDiscreteLognormalOnePlusOne",
        "SmoothDiscreteLognormalOnePlusOne",
        "SmoothLognormalDiscreteOnePlusOne",
        "UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne",
        "SuperSmoothRecombiningDiscreteLognormalOnePlusOne",
        "SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne",
        "RecombiningDiscreteLognormalOnePlusOne",
        "RandRecombiningDiscreteLognormalOnePlusOne",
        "UltraSmoothDiscreteLognormalOnePlusOne",
        "ZetaSmoothDiscreteLognormalOnePlusOne",
        "SuperSmoothDiscreteLognormalOnePlusOne",
    ]
    for optim in algos:
        for budget in [20, 50, 100, 160, 240]:
            yield Experiment(func, optim, budget=budget, seed=next(seedg))


@ng.optimizers.registry.register
def multi_ceviche(
    seed: OptionalInt = None, c0: bool = False, precompute: bool = False, warmstart: bool = False
) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    algos = [
        "DiagonalCMA",
        "PSO",
        "DE",
        "CMA",
        "OnePlusOne",
        "LognormalDiscreteOnePlusOne",
        "DiscreteLenglerOnePlusOne",
        "MetaModel",
        "MetaModelDE",
        "MetaModelDSproba",
        "MetaModelOnePlusOne",
        "MetaModelPSO",
        "MetaModelQODE",
        "MetaModelTwoPointsDE",
        "NeuralMetaModel",
        "NeuralMetaModelDE",
        "NeuralMetaModelTwoPointsDE",
        "RFMetaModel",
        "RFMetaModelDE",
        "RFMetaModelOnePlusOne",
        "RFMetaModelPSO",
        "RFMetaModelTwoPointsDE",
        "SVMMetaModel",
        "SVMMetaModelDE",
        "SVMMetaModelPSO",
        "SVMMetaModelTwoPointsDE",
        "RandRecombiningDiscreteLognormalOnePlusOne",
        "SmoothDiscreteLognormalOnePlusOne",
        "SmoothLognormalDiscreteOnePlusOne",
        "UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne",
        "SuperSmoothRecombiningDiscreteLognormalOnePlusOne",
        "SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne",
        "RecombiningDiscreteLognormalOnePlusOne",
        "RandRecombiningDiscreteLognormalOnePlusOne",
        "UltraSmoothDiscreteLognormalOnePlusOne",
        "ZetaSmoothDiscreteLognormalOnePlusOne",
        "SuperSmoothDiscreteLognormalOnePlusOne",
    ]
    if not precompute:
        algos = ["RF1MetaModelLogNormal", "Neural1MetaModelLogNormal", "SVM1MetaModelLogNormal", "CMAL"]
    else:
        algos = ["UltraSmoothDiscreteLognormalOnePlusOne", "DiscreteLenglerOnePlusOne", "CMA", "CMAL"]
    algos = ["CMALS", "CMALYS", "CMALL"]
    algos = ["CLengler", "CMALS", "CMALYS", "CMALL", "CMAL"]
    algos = ["CMASL2", "CMASL3"]
    algos = [
        "DiagonalCMA",
        "CMAL3",
        "CMA",
        "CLengler",
        "CMALL",
        "CMALYS",
        "CMALS" "DiscreteLenglerOnePlusOne",
        "CMASL3",
        "CMASL2",
        "DSproba",
    ]
    algos += [
        "LognormalDiscreteOnePlusOne",
        "CMA",
        "DiscreteLenglerOnePlusOne",
        "SmoothDiscreteLognormalOnePlusOne",
        "SuperSmoothDiscreteLognormalOnePlusOne",
        "AnisotropicAdaptiveDiscreteOnePlusOne",
        "Neural1MetaModelE",
        "SVM1MetaModelE",
        "Quad1MetaModelE",
        "RF1MetaModelE",
        "UltraSmoothDiscreteLognormalOnePlusOne",
        "VoronoiDE",
        "UltraSmoothDiscreteLognormalOnePlusOne",
        "VoronoiDE",
        "RF1MetaModelLogNormal",
        "Neural1MetaModelLogNormal",
        "SVM1MetaModelLogNormal",
        "DSproba",
        "ImageMetaModelE",
        "ImageMetaModelOnePlusOne",
        "ImageMetaModelDiagonalCMA",
        "ImageMetaModelLengler",
        "ImageMetaModelLogNormal",
    ]
    algos = [a for a in algos if a in list(ng.optimizers.registry.keys())]
    for benchmark_type in [int(np.random.choice([0, 1, 2, 3]))]:
        if warmstart:
            try:
                suggestion = np.load(f"bestnp{benchmark_type}.npy")
            except Exception as e:
                print("Be careful! You need warmstart data for warmstarting :-)  use scripts/plot_ceviche.sh.")
                raise e
        else:
            suggestion = None
        shape: tp.Tuple[int, ...] = tuple(int(p) for p in list(imagesxp.photonics_ceviche(None, benchmark_type)))
        name: str = imagesxp.photonics_ceviche("name", benchmark_type) + str(shape)
        
        def pc(x: np.ndarray) -> float:
            return imagesxp.photonics_ceviche(x, benchmark_type)  # type: ignore

        def fpc(x: np.ndarray) -> tp.Tuple[float, np.ndarray]:
            loss, grad = imagesxp.photonics_ceviche(x.reshape(shape), benchmark_type, wantgrad=True)  # type: ignore
            return loss, grad.flatten()

        def epc(x: np.ndarray) -> float:
            return imagesxp.photonics_ceviche(x, benchmark_type, discretize=True)  # type: ignore

        def export_numpy(name: str, array: np.ndarray, fields: tp.Optional[np.ndarray] = None) -> None:
            from PIL import Image
            x_img = (255 * (1 - array)).astype("uint8")
            print(
                "Histogram",
                name,
                [100 * np.average(np.abs(np.round(10 * x_img.flatten()) - i) < 0.1) for i in range(11)],
            )
            if fields is not None:
                np.save(name + "fields.", fields)
                np.save(name + "savedarray", array)
            im = Image.fromarray(x_img)
            im.convert("RGB").save(f"{name}_{np.average(np.abs(array.flatten() - 0.5) < 0.35)}_{np.average(np.abs(array.flatten() - 0.5) < 0.45)}.png", mode="L")

        def cv(x: np.ndarray) -> float:
            return float(np.sum(np.clip(np.abs(x - np.round(x)) - 1e-3, 0.0, 50000000.0)))
            
        instrum = ng.p.Array(shape=shape, lower=0.0, upper=1.0)
        instrum2 = ng.p.Array(shape=shape, lower=0.0, upper=1.0)
        if c0:
            instrum_c0 = ng.p.Array(shape=shape, lower=0.0, upper=1.0)
            instrum_c0.set_name(name)
        else:
            instrum.set_name(name)
        instrum2.set_name(name)
        func = Experiment(pc, instrum)
        c0func = Experiment(pc, ng.p.Array(shape=shape, lower=0.0, upper=1.0))
        c0cfunc = Experiment(pc, ng.p.Array(shape=shape, lower=0.0, upper=1.0))
        eval_func = Experiment(epc, instrum2)
        optims_choice = [np.random.choice(algos)]
        budgets = (
            [int(np.random.choice([3, 20, 50, 90, 150, 250, 400, 800, 1600, 3200, 6400])),
             int(np.random.choice([12800, 25600, 51200, 102400, 204800, 409600]))]
            if not precompute
            else [int(np.random.choice([409600, 204800 + 102400, 204800])) - 102400]
        )
        if benchmark_type == 3:
            budgets = (
                [int(np.random.choice([3, 20, 50, 90, 150, 250, 400, 800, 1600, 3200, 6400])),
                 int(np.random.choice([12800, 25600, 51200, 102400]))]
                if not precompute
                else [int(np.random.choice([204800 + 51200, 204800])) - 102400]
            )
        for optim in optims_choice:
            for budget in budgets:
                if (np.random.rand() < 0.05 or precompute) and not warmstart:
                    from scipy import optimize as scipyoptimize
                    x0 = np.random.rand(np.prod(shape))
                    result = scipyoptimize.minimize(
                        fpc,
                        x0=x0,
                        method="L-BFGS-B",
                        tol=1e-9,
                        jac=True,
                        options={"maxiter": budget if not precompute else 102400},
                        bounds=[[0, 1] for _ in range(np.prod(shape))],
                    )
                    real_loss = epc(result.x.reshape(shape))
                    fake_loss, _ = fpc(result.x.reshape(shape))
                    initial_point = result.x.reshape(shape)
                    if budget > 100000 or np.random.rand() < 0.05:
                        export_numpy(f"pb{benchmark_type}_budget{budget if not precompute else 102400}_bfgs_{real_loss}_{fake_loss}", result.x.reshape(shape))
                if (c0 and np.random.choice([True, False, False, False])) and not precompute:
                    pen = bool(np.random.choice([True, False, False]) and not precompute)
                    pre_optim = ng.optimizers.registry[optim]
                    try:
                        optim2 = type(optim, pre_optim.__bases__, dict(pre_optim.__dict__))
                    except Exception:
                        optim2 = pre_optim
                    try:
                        optim2.name += "c0p"
                    except Exception:
                        optim2.__name__ += "c0p"
                    sfunc = imagesxp.helpers.SpecialEvaluationExperiment(c0func, evaluation=eval_func)
                    yield Experiment(sfunc, optim2, budget=budget, seed=next(seedg), constraint_violation=[cv], penalize_violation_at_test=False, suggestions=([suggestion] if warmstart else None))
                else:
                    def plot_epc(x: np.ndarray) -> float:
                        real_loss, fields = imagesxp.photonics_ceviche(x, benchmark_type, discretize=True, wantfields=True)  # type: ignore
                        if budget > 100000 or np.random.rand() < 0.05:
                            export_numpy(f"pb{benchmark_type}_{optim}_budget{budget}_{real_loss}", x.reshape(shape), fields)
                        return real_loss
                    plot_eval_func = Experiment(plot_epc, instrum2)
                    pfunc = imagesxp.helpers.SpecialEvaluationExperiment(func, evaluation=plot_eval_func)
                    yield Experiment(func if np.random.rand() < 0.0 else pfunc, optim, budget=budget, seed=next(seedg), suggestions=([suggestion] if warmstart else None))


@ng.optimizers.registry.register
def multi_ceviche_c0(seed: OptionalInt = None) -> ExperimentIterator:
    return multi_ceviche(seed, c0=True)


@ng.optimizers.registry.register
def multi_ceviche_c0_warmstart(seed: OptionalInt = None) -> ExperimentIterator:
    return multi_ceviche(seed, c0=True, warmstart=True)


@ng.optimizers.registry.register
def multi_ceviche_c0p(seed: OptionalInt = None) -> ExperimentIterator:
    return multi_ceviche(seed, c0=True, precompute=True)


@ng.optimizers.registry.register
def photonics(
    seed: OptionalInt = None, as_tuple: bool = False, small: bool = False, ultrasmall: bool = False, verysmall: bool = False
) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    divider: int = 2 if small else 1
    if ultrasmall or verysmall:
        divider = 4
    optims = get_optimizers("es", "basics", "splitters", seed=next(seedg))
    optims = ["MemeticDE", "PSO", "DE", "CMA", "OnePlusOne", "TwoPointsDE", "GeneticDE", "ChainMetaModelSQP", "MetaModelDE", "SVMMetaModelDE", "RFMetaModelDE", "RBFGS", "LBFGSB"]
    optims = refactor_optims(optims)
    for method in ["clipping", "tanh"]:
        for name in (
            ["bragg"]
            if ultrasmall
            else (
                ["cf_photosic_reference", "cf_photosic_realistic"]
                if verysmall
                else ["bragg", "chirped", "morpho", "cf_photosic_realistic", "cf_photosic_reference"]
            )
        ):
            func = Photonics(
                name,
                4 * ((60 // divider) // 4) if name == "morpho" else 80 // divider,
                bounding_method=method,
                as_tuple=as_tuple,
            )
            for budget in [10, 100, 1000]:
                for algo in optims:
                    xp = Experiment(func, algo, int(budget), num_workers=1, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp


@ng.optimizers.registry.register
def photonics2(seed: OptionalInt = None) -> ExperimentIterator:
    return photonics(seed, as_tuple=True)


@ng.optimizers.registry.register
def ultrasmall_photonics(seed: OptionalInt = None) -> ExperimentIterator:
    return photonics(seed, as_tuple=False, small=True, ultrasmall=True)


@ng.optimizers.registry.register
def ultrasmall_photonics2(seed: OptionalInt = None) -> ExperimentIterator:
    return photonics(seed, as_tuple=True, small=True, ultrasmall=True)


@ng.optimizers.registry.register
def verysmall_photonics(seed: OptionalInt = None) -> ExperimentIterator:
    return photonics(seed, as_tuple=False, small=True, verysmall=True)


@ng.optimizers.registry.register
def verysmall_photonics2(seed: OptionalInt = None) -> ExperimentIterator:
    return photonics(seed, as_tuple=True, small=True, verysmall=True)


@ng.optimizers.registry.register
def small_photonics(seed: OptionalInt = None) -> ExperimentIterator:
    return photonics(seed, as_tuple=False, small=True)


@ng.optimizers.registry.register
def small_photonics2(seed: OptionalInt = None) -> ExperimentIterator:
    return photonics(seed, as_tuple=True, small=True)


@ng.optimizers.registry.register
def adversarial_attack(seed: OptionalInt = None) -> ExperimentIterator:
    seedg = ng.functions.base.create_seed_generator(seed)
    optims: tp.List[tp.Any] = get_optimizers("structure", "structured_moo", seed=next(seedg))
    folder = os.environ.get("NEVERGRAD_ADVERSARIAL_EXPERIMENT_FOLDER", None)
    if folder is None:
        warnings.warn("Using random images, set variable NEVERGRAD_ADVERSARIAL_EXPERIMENT_FOLDER to specify a folder")
    optims = refactor_optims(optims)
    for func in imagesxp.ImageAdversarial.make_folder_functions(folder=folder):
        for budget in [100, 200, 300, 400, 1700]:
            for num_workers in [1]:
                for algo in optims:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp


def pbo_suite(seed: OptionalInt = None, reduced: bool = False) -> ExperimentIterator:
    # This function was already defined above.
    pass


@ng.optimizers.registry.register
def pbo_reduced_suite(seed: OptionalInt = None) -> ExperimentIterator:
    return pbo_suite(seed, reduced=True)


def causal_similarity(seed: OptionalInt = None) -> ExperimentIterator:
    # This function was already defined above.
    pass


def unit_commitment(seed: OptionalInt = None) -> ExperimentIterator:
    # This function was already defined above.
    pass


def team_cycling(seed: OptionalInt = None) -> ExperimentIterator:
    # This function was already defined above.
    pass

# End of annotated code.
