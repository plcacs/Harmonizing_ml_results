# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from .xpbase import Experiment as Experiment
from .xpbase import create_seed_generator
from .xpbase import registry as registry  # noqa
from .optgroups import get_optimizers

# register all experiments from other files
# pylint: disable=unused-import
from . import frozenexperiments  # noqa
from . import gymexperiments  # noqa

# pylint: disable=stop-iteration-return, too-many-nested-blocks, too-many-locals


def refactor_optims(x: tp.List[tp.Any]) -> tp.List[tp.Any]:  # type: ignore
    if False:  # np.random.randn() < 0.0:
        return list(
            np.random.choice(
                [
                    "NgIohTuned",
                    "NGOpt",
                    "NGOptRW",
                    "ChainCMASQP",
                    "PymooBIPOP",
                    "NLOPT_LN_SBPLX",
                    "QNDE",
                    "BFGSCMAPlus",
                    "ChainMetaModelSQP",
                    "BFGSCMA",
                    "BAR4",
                    "BFGSCMAPlus",
                    "LBFGSB",
                    "LQOTPDE",
                    "LogSQPCMA",
                ],
                4,
            )
        )
    # return ["RandomSearch", "OnePlusOne", "DE", "PSO"]
    list_optims = x
    algos: tp.Dict[str, tp.List[str]] = {}
    algos["aquacrop_fao"] = [
        "CMA",
        "CMandAS2",
        "DE",
        "MetaModel",
        "NGOpt10",
    ]
    algos["bonnans"] = [
        "AdaptiveDiscreteOnePlusOne",
        "DiscreteBSOOnePlusOne",
        "DiscreteLenglerFourthOnePlusOne",
        "DiscreteLenglerHalfOnePlusOne",
        "DiscreteLenglerOnePlusOne",
        "MemeticDE",
    ]
    algos["double_o_seven"] = [
        "DiagonalCMA",
        "DiscreteDE",
        "MetaTuneRecentering",
        "PSO",
        "RecombiningOptimisticNoisyDiscreteOnePlusOne",
        "TBPSA",
    ]
    algos["fishing"] = [
        "CMA",
        "CMandAS2",
        "ChainMetaModelSQP",
        "DE",
        "MetaModel",
        "NGOpt10",
    ]
    algos["mldakmeans"] = [
        "DE",
        "SplitCMA5",
        "SplitTwoPointsDE3",
        "SplitTwoPointsDE5",
        "SplitTwoPointsDEAuto",
        "TwoPointsDE",
    ]
    algos["mltuning"] = [
        "OnePlusOne",
        "RandomSearch",
    ]
    algos["mono_rocket"] = [
        "CMA",
        "CMandAS2",
        "DE",
        "MetaModel",
        "NGOpt10",
    ]
    algos["ms_bbob"] = [
        "ChainMetaModelSQP",
        "MetaModelOnePlusOne",
        "Powell",
        "QODE",
        "SQP",
        "TinyCMA",
    ]
    algos["multiobjective_example_hd"] = [
        "DiscreteLenglerOnePlusOne",
        "DiscreteOnePlusOne",
        "MetaNGOpt10",
        "ParametrizationDE",
        "RecES",
        "RecMutDE",
    ]
    algos["multiobjective_example_many_hd"] = [
        "DiscreteLenglerOnePlusOne",
        "DiscreteOnePlusOne",
        "MetaNGOpt10",
        "ParametrizationDE",
        "RecES",
        "RecMutDE",
    ]
    algos["multiobjective_example"] = [
        "CMA",
        "DE",
        "ParametrizationDE",
        "RecES",
        "RecMutDE",
    ]
    algos["naive_seq_keras_tuning"] = [
        "CMA",
        "DE",
        "HyperOpt",
        "OnePlusOne",
        "RandomSearch",
        "TwoPointsDE",
    ]
    algos["nano_naive_seq_mltuning"] = [
        "DE",
        "HyperOpt",
        "OnePlusOne",
        "RandomSearch",
        "TwoPointsDE",
    ]
    algos["nano_seq_mltuning"] = [
        "DE",
        "HyperOpt",
        "OnePlusOne",
        "RandomSearch",
        "TwoPointsDE",
    ]
    algos["oneshot_mltuning"] = [
        "DE",
        "OnePlusOne",
        "RandomSearch",
        "TwoPointsDE",
    ]
    algos["pbbob"] = [
        "CMAbounded",
        "DE",
        "MetaModelDE",
        "MetaModelOnePlusOne",
        "QODE",
        "QrDE",
    ]
    algos["pbo_reduced_suite"] = [
        "DiscreteLenglerOnePlusOne",
        "LognormalDiscreteOnePlusOne",
        "DiscreteLenglerOnePlusOneT",
        "DiscreteLenglerOnePlusOneT",
        "SADiscreteLenglerOnePlusOneExp09",
        "SADiscreteLenglerOnePlusOneExp09",
        "discretememetic",
    ]
    algos["reduced_yahdlbbbob"] = [
        "CMA",
        "DE",
        "MetaModelOnePlusOne",
        "OnePlusOne",
        "PSO",
        "RFMetaModelDE",
    ]
    algos["seq_keras_tuning"] = [
        "CMA",
        "DE",
        "HyperOpt",
        "OnePlusOne",
        "RandomSearch",
        "TwoPointsDE",
    ]
    algos["sequential_topology_optimization"] = [
        "CMA",
        "DE",
        "GeneticDE",
        "OnePlusOne",
        "TwoPointsDE",
        "VoronoiDE",
    ]
    algos["spsa_benchmark"] = [
        "CMA",
        "DE",
        "NaiveTBPSA",
        "OnePlusOne",
        "SPSA",
        "TBPSA",
    ]
    algos["topology_optimization"] = [
        "CMA",
        "DE",
        "GeneticDE",
        "OnePlusOne",
        "TwoPointsDE",
        "VoronoiDE",
    ]
    algos["yabbob"] = [
        "CMA",
        "ChainMetaModelSQP",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
    ]
    algos["yabigbbob"] = [
        "ChainMetaModelSQP",
        "MetaModel",
        "MetaModelDE",
        "NeuralMetaModel",
        "PSO",
        "TwoPointsDE",
    ]
    algos["yaboundedbbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yaboxbbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yamegapenbbob"] = [
        "ChainMetaModelSQP",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yamegapenboundedbbob"] = [
        "CMA",
        "ChainMetaModelSQP",
        "MetaModel",
        "MetaModelOnePlusOne",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yamegapenboxbbob"] = [
        "ChainMetaModelSQP",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yanoisybbob"] = [
        "TBPSA",
        "NoisyRL2",
        "NoisyRL3",
        "RecombiningOptimisticNoisyDiscreteOnePlusOne",
        "RBFGS",
        "MicroCMA",
        "NoisyDiscreteOnePlusOne",
        "RandomSearch",
        "RecombiningOptimisticNoisyDiscreteOnePlusOne",
        "SQP",
    ]
    algos["yaonepenbbob"] = [
        "CMandAS2",
        "ChainMetaModelSQP",
        "MetaModel",
        "NGOpt",
        "NeuralMetaModel",
        "Shiwa",
    ]
    algos["yaonepenboundedbbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yaonepenboxbbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yaonepennoisybbob"] = [
        "NoisyDiscreteOnePlusOne",
        "RandomSearch",
        "SQP",
        "TBPSA",
    ]
    algos["yaonepenparabbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelDE",
        "NeuralMetaModel",
        "RFMetaModel",
        "RFMetaModelDE",
    ]
    algos["yaonepensmallbbob"] = [
        "Cobyla",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yaparabbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelDE",
        "NeuralMetaModel",
        "RFMetaModel",
        "RFMetaModelDE",
    ]
    algos["yapenbbob"] = [
        "ChainMetaModelSQP",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yapenboundedbbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yapenboxbbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yapennoisybbob"] = [
        "NoisyDiscreteOnePlusOne",
        "RandomSearch",
        "SQP",
        "TBPSA",
    ]
    algos["yapenparabbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelDE",
        "NeuralMetaModel",
        "RFMetaModel",
        "RFMetaModelDE",
    ]
    algos["yapensmallbbob"] = [
        "Cobyla",
        "MetaModel",
        "MetaModelOnePlusOne",
        "OnePlusOne",
        "RFMetaModel",
        "RFMetaModelDE",
    ]
    algos["yasmallbbob"] = [
        "Cobyla",
        "MetaModelDE",
        "MetaModelOnePlusOne",
        "OnePlusOne",
        "PSO",
        "RFMetaModelDE",
    ]
    algos["yatinybbob"] = [
        "Cobyla",
        "DE",
        "MetaModel",
        "MetaModelDE",
        "MetaModelOnePlusOne",
        "TwoPointsDE",
    ]
    algos["yatuningbbob"] = [
        "Cobyla",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "RFMetaModelDE",
        "RandomSearch",
        "TwoPointsDE",
    ]

    benchmark = str(inspect.stack()[1].function)
    if benchmark in algos:
        list_algos = algos[benchmark][:5] + [
            "CSEC10",
            "NGOpt",
            "NLOPT_LN_SBPLX",
        ]
        return (
            list_algos
            if (
                "eras" in benchmark
                or "tial_instrum" in benchmark
                or "big" in benchmark
                or "lsgo" in benchmark
                or "rock" in benchmark
            )
            else list_algos
        )
    if benchmark in algos:
        list_algos = algos[benchmark]
        return (
            list_algos
            if (
                "eras" in benchmark
                or "tial_instrum" in benchmark
                or "big" in benchmark
                or "lsgo" in benchmark
                or "rock" in benchmark
            )
            else list_algos
        )
    return [
        "NgDS3",
        "NgIoh4",
        "NgIoh21",
        "NGOpt",
        "NGDSRW",
    ]

    def doint(s: str) -> int:
        return 7 + sum([ord(c) * i for i, c in enumerate(s)])

    import socket

    host = socket.gethostname()

    if "iscr" in benchmark or "pbo" in benchmark:
        list_optims += [
            a
            for a in [
                "DiscreteDE",
                "DiscreteOnePlusOne",
                "SADiscreteLenglerOnePlusOneExp09",
                "SADiscreteLenglerOnePlusOneExp099",
                "SADiscreteLenglerOnePlusOneExp09Auto",
                "SADiscreteLenglerOnePlusOneLinAuto",
                "SADiscreteLenglerOnePlusOneLin1",
                "SADiscreteLenglerOnePlusOneLin100",
                "SADiscreteOnePlusOneExp099",
                "SADiscreteOnePlusOneLin100",
                "SADiscreteOnePlusOneExp09",
                "PortfolioDiscreteOnePlusOne",
                "DiscreteLenglerOnePlusOne",
                "DiscreteLengler2OnePlusOne",
                "DiscreteLengler3OnePlusOne",
                "DiscreteLenglerHalfOnePlusOne",
                "DiscreteLenglerFourthOnePlusOne",
                "AdaptiveDiscreteOnePlusOne",
                "LognormalDiscreteOnePlusOne",
                "AnisotropicAdaptiveDiscreteOnePlusOne",
                "DiscreteBSOOnePlusOne",
                "DiscreteDoerrOnePlusOne",
                "DoubleFastGADiscreteOnePlusOne",
                "SparseDoubleFastGADiscreteOnePlusOne",
                "RecombiningPortfolioDiscreteOnePlusOne",
                "MultiDiscrete",
                "discretememetic",
                "SmoothDiscreteOnePlusOne",
                "SmoothPortfolioDiscreteOnePlusOne",
                "SmoothDiscreteLenglerOnePlusOne",
                "SuperSmoothDiscreteLenglerOnePlusOne",
                "UltraSmoothDiscreteLenglerOnePlusOne",
                "SmoothLognormalDiscreteOnePlusOne",
                "SmoothAdaptiveDiscreteOnePlusOne",
                "SmoothRecombiningPortfolioDiscreteOnePlusOne",
                "SmoothRecombiningDiscreteLanglerOnePlusOne",
                "UltraSmoothRecombiningDiscreteLanglerOnePlusOne",
                "SuperSmoothElitistRecombiningDiscreteLanglerOnePlusOne",
                "SuperSmoothRecombiningDiscreteLanglerOnePlusOne",
                "SmoothElitistRecombiningDiscreteLanglerOnePlusOne",
                "RecombiningDiscreteLanglerOnePlusOne",
                "DiscreteDE",
                "cGA",
                "NGOpt",
                "NgIoh4",
                "NgIoh5",
                "NgIoh6",
                "NGOptRW",
                "NgIoh7",
            ]
            if ("Smooth" in a or "Lognor" in a or "Recomb" in a)
        ]

    return [list_optims[doint(host) % len(list_optims)]]


def skip_ci(*, reason: str) -> None:
    if os.environ.get("NEVERGRAD_PYTEST", False):
        raise fbase.UnsupportedExperiment("Skipping CI: " + reason)


class _Constraint:
    def __init__(self, name: str, as_bool: bool) -> None:
        self.name = name
        self.as_bool = as_bool

    def __call__(self, data: np.ndarray) -> tp.Union[bool, float]:
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Unexpected inputs as np.ndarray, got {data}")
        if self.name == "sum":
            value = float(np.sum(data))
        elif self.name == "diff":
            value = float(np.sum(data[::2]) - np.sum(data[1::2]))
        elif self.name == "second_diff":
            value = float(2 * np.sum(data[1::2]) - 3 * np.sum(data[::2]))
        elif self.name == "ball":
            value = float(np.sum(np.square(data)) - len(data) - np.sqrt(len(data)))
        else:
            raise NotImplementedError(f"Unknown function {self.name}")
        return value > 0 if self.as_bool else value


@registry.register
def keras_tuning(
    seed: tp.Optional[int] = None,
    overfitter: bool = False,
    seq: bool = False,
    veryseq: bool = False,
) -> tp.Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ["OnePlusOne", "BO", "RandomSearch", "CMA", "DE", "TwoPointsDE", "HyperOpt", "PCABO", "Cobyla"]
    optims = [
        "OnePlusOne",
        "RandomSearch",
        "CMA",
        "DE",
        "TwoPointsDE",
        "HyperOpt",
        "Cobyla",
        "MetaModel",
        "MetaModelOnePlusOne",
        "RFMetaModel",
        "RFMetaModelOnePlusOne",
    ]
    optims = ["OnePlusOne", "RandomSearch", "Cobyla"]
    optims = ["DE", "TwoPointsDE", "HyperOpt", "MetaModelOnePlusOne"]
    optims = get_optimizers("oneshot", seed=next(seedg))  # type: ignore
    optims = [
        "MetaTuneRecentering",
        "MetaRecentering",
        "HullCenterHullAvgCauchyScrHammersleySearch",
        "LHSSearch",
        "LHSCauchySearch",
    ]
    optims = ["NGOpt", "NGOptRW", "QODE"]
    optims = ["NGOpt"]
    optims = ["PCABO", "NGOpt", "QODE"]
    optims = ["QOPSO"]
    optims = ["SQOPSO"]
    optims = refactor_optims(optims)
    datasets = ["kerasBoston", "diabetes", "auto-mpg", "red-wine", "white-wine"]
    optims = refactor_optims(optims)
    for dimension in [None]:
        for dataset in datasets:
            function = MLTuning(
                regressor="keras_dense_nn", data_dimension=dimension, dataset=dataset, overfitter=overfitter
            )
            for budget in [150, 500]:
                for num_workers in [1, budget // 4] if seq else [budget]:
                    if veryseq and num_workers > 1:
                        continue
                    for optim in optims:
                        xp = Experiment(
                            function, optim, num_workers=num_workers, budget=budget, seed=next(seedg)
                        )
                        skip_ci(reason="too slow")
                        xp.function.parametrization.real_world = True
                        xp.function.parametrization.hptuning = True
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def mltuning(
    seed: tp.Optional[int] = None,
    overfitter: bool = False,
    seq: bool = False,
    veryseq: bool = False,
    nano: bool = False,
) -> tp.Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ["OnePlusOne", "BO", "RandomSearch", "CMA", "DE", "TwoPointsDE", "PCABO", "HyperOpt", "Cobyla"]
    optims = [
        "OnePlusOne",
        "RandomSearch",
        "CMA",
        "DE",
        "TwoPointsDE",
        "HyperOpt",
        "Cobyla",
        "MetaModel",
        "MetaModelOnePlusOne",
        "RFMetaModel",
        "RFMetaModelOnePlusOne",
    ]
    optims = ["OnePlusOne", "RandomSearch", "Cobyla"]
    optims = ["DE", "TwoPointsDE", "HyperOpt", "MetaModelOnePlusOne"]
    optims = get_optimizers("oneshot", seed=next(seedg))  # type: ignore
    optims = [
        "MetaTuneRecentering",
        "MetaRecentering",
        "HullCenterHullAvgCauchyScrHammersleySearch",
        "LHSSearch",
        "LHSCauchySearch",
    ]
    optims = ["NGOpt", "NGOptRW", "QODE"]
    optims = ["NGOpt"]
    optims = ["PCABO"]
    optims = ["PCABO", "NGOpt", "QODE"]
    optims = ["QOPSO"]
    optims = ["SQOPSO"]
    optims = refactor_optims(optims)
    for dimension in [None, 1, 2, 3]:
        if dimension is None:
            datasets = ["diabetes", "auto-mpg", "red-wine", "white-wine"]
        else:
            datasets = ["artificialcos", "artificial", "artificialsquare"]
        for regressor in ["mlp", "decision_tree", "decision_tree_depth"]:
            for dataset in datasets:
                function = MLTuning(
                    regressor=regressor, data_dimension=dimension, dataset=dataset, overfitter=overfitter
                )
                for budget in [150, 500] if not nano else [80, 160]:
                    parallelization = [1, budget // 4] if seq else [budget]
                    for num_workers in parallelization:
                        if veryseq and num_workers > 1:
                            continue

                        for optim in optims:
                            xp = Experiment(
                                function, optim, num_workers=num_workers, budget=budget, seed=next(seedg)
                            )
                            skip_ci(reason="too slow")
                            xp.function.parametrization.real_world = True
                            xp.function.parametrization.hptuning = True
                            if not xp.is_incoherent:
                                yield xp


@registry.register
def naivemltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return mltuning(seed, overfitter=True)


@registry.register
def veryseq_keras_tuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return keras_tuning(seed, overfitter=False, seq=True, veryseq=True)


@registry.register
def seq_keras_tuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return keras_tuning(seed, overfitter=False, seq=True)


@registry.register
def naive_seq_keras_tuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return keras_tuning(seed, overfitter=True, seq=True)


@registry.register
def naive_veryseq_keras_tuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return keras_tuning(seed, overfitter=True, seq=True, veryseq=True)


@registry.register
def oneshot_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return mltuning(seed, overfitter=False, seq=False)


@registry.register
def seq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return mltuning(seed, overfitter=False, seq=True)


@registry.register
def nano_seq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return mltuning(seed, overfitter=False, seq=True, nano=True)


@registry.register
def nano_veryseq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return mltuning(seed, overfitter=False, seq=True, nano=True, veryseq=True)


@registry.register
def nano_naive_veryseq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return mltuning(seed, overfitter=True, seq=True, nano=True, veryseq=True)


@registry.register
def nano_naive_seq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return mltuning(seed, overfitter=True, seq=True, nano=True)


@registry.register
def naive_seq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return mltuning(seed, overfitter=True, seq=True)


@registry.register
def yawidebbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    total_xp_per_optim = 0
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation, translation_factor=tf)
        for name in ["cigar", "ellipsoid"]
        for rotation in [True, False]
        for tf in [0.1, 10.0]
    ]
    for i, func in enumerate(functions):
        func.parametrization.register_cheap_constraint(_Constraint("sum", as_bool=i % 2 == 0))
    assert len(functions) == 8
    names = ["hm", "rastrigin", "sphere", "doublelinearslope", "ellipsoid"]

    functions += [
        ArtificialFunction(
            name,
            block_dimension=d,
            rotation=rotation,
            noise_level=nl,
            split=split,
            translation_factor=tf,
            num_blocks=num_blocks,
        )
        for name in names
        for rotation in [True, False]
        for nl in [0.0, 100.0]
        for tf in [0.1, 10.0]
        for num_blocks in [1, 8]
        for d in [5, 70, 10000]
        for split in [True, False]
    ][::37]
    assert len(functions) == 21, f"{len(functions)} problems instead of 21. Yawidebbob should be standard."
    optims = ["NGOptRW", "NGOpt", "RandomSearch", "CMA", "DE", "DiscreteLenglerOnePlusOne"]
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

    assert total_xp_per_optim == 33, f"We have 33 single-obj xps per optimizer (got {total_xp_per_optim})."
    index = 0
    for nv in [200, 2000]:
        for arity in [2, 7, 37]:
            instrum = ng.p.TransitionChoice(range(arity), repetitions=nv)
            for name in ["onemax", "leadingones", "jump"]:
                index += 1
                if index % 4 != 0:
                    continue
                dfunc = ExperimentFunction(
                    corefuncs.DiscreteFunction(name, arity), instrum.set_name("transition")
                )
                dfunc.add_descriptors(arity=arity)
                for budget in [500, 1500, 5000]:
                    for nw in [1, 100]:
                        total_xp_per_optim += 1
                        for optim in optims:
                            yield Experiment(dfunc, optim, num_workers=nw, budget=budget, seed=next(seedg))
    assert total_xp_per_optim == 57, f"Including discrete, we check xps per optimizer (got {total_xp_per_optim})."

    mofuncs: tp.List[fbase.MultiExperiment] = []
    for name1, name2 in itertools.product(["sphere"], ["sphere", "hm"]):
        mofuncs.append(
            fbase.MultiExperiment(
                [
                    ArtificialFunction(name1, block_dimension=7),
                    ArtificialFunction(name2, block_dimension=7),
                ],
                upper_bounds=[100, 100],
            )
        )
        mofuncs.append(
            fbase.MultiExperiment(
                [
                    ArtificialFunction(name1, block_dimension=7),
                    ArtificialFunction("sphere", block_dimension=7),
                    ArtificialFunction(name2, block_dimension=7),
                ],
                upper_bounds=[100, 100, 100.0],
            )
        )
    index = 0
    for mofunc in mofuncs[::3]:
        for budget in [2000, 4000, 8000]:
            for nw in [1, 20, 100]:
                index += 1
                if index % 5 == 0:
                    total_xp_per_optim += 1
                    for optim in optims:
                        yield Experiment(mofunc, optim, budget=budget, num_workers=nw, seed=next(seedg))
    assert total_xp_per_optim == 71, f"We should have 71 xps per optimizer, not {total_xp_per_optim}."


@registry.register
def parallel_small_budget(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ["DE", "TwoPointsDE", "CMA", "NGOpt", "PSO", "OnePlusOne", "RandomSearch"]
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "multipeak"]
    names += ["sphere", "cigar", "ellipsoid", "altellipsoid"]
    names += ["deceptiveillcond", "deceptivemultimodal", "deceptivepath"]
    functions = [
        ArtificialFunction(name, block_dimension=d, rotation=rotation)
        for name in names
        for rotation in [True, False]
        for d in [2, 4, 8]
    ]
    budgets = [10, 50, 100, 200, 400]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in budgets:
                for nw in [2, 8, 16]:
                    for batch in [True, False]:
                        if nw < budget / 4:
                            xp = Experiment(
                                function,
                                optim,
                                num_workers=nw,
                                budget=budget,
                                batch_mode=batch,
                                seed=next(seedg),
                            )
                            if not xp.is_incoherent:
                                yield xp


@registry.register
def instrum_discrete(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ["DiscreteOnePlusOne", "NGOpt", "CMA", "TwoPointsDE", "DiscreteLenglerOnePlusOne"]
    optims = ["RFMetaModelOnePlusOne"]
    optims = ["FastGADiscreteOnePlusOne"]
    optims = ["DoubleFastGADiscreteOnePlusOne"]
    optims = ["DiscreteOnePlusOne"]
    optims = ["OnePlusOne"]
    optims = ["DiscreteLenglerOnePlusOne"]
    optims = ["NGOpt", "NGOptRW"]
    optims = refactor_optims(optims)
    for nv in [10, 50, 200, 1000, 5000]:
        for arity in [2, 3, 7, 30]:
            for instrum_str in ["Unordered", "Softmax", "Ordered"]:
                if instrum_str == "Softmax":
                    instrum: ng.p.Parameter = ng.p.Choice(range(arity), repetitions=nv)
                else:
                    assert instrum_str in ("Ordered", "Unordered")
                    instrum = ng.p.TransitionChoice(
                        range(arity), repetitions=nv, ordered=instrum_str == "Ordered"
                    )
                for name in ["onemax", "leadingones", "jump"]:
                    dfunc = ExperimentFunction(
                        corefuncs.DiscreteFunction(name, arity), instrum.set_name(instrum_str)
                    )
                    dfunc.add_descriptors(arity=arity)
                    dfunc.add_descriptors(nv=nv)
