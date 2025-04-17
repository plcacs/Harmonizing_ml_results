```python
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


def refactor_optims(x: tp.List[str]) -> tp.List[str]:
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

    # Below, we use the best in the records above.
    benchmark = str(inspect.stack()[1].function)
    # if "bbob" in benchmark and np.random.choice([True, False, False, False, False]):
    #    return ["DSproba" + str(i) for i in range(2, 10)]
    if benchmark in algos:  # and np.random.choice([True, False]):  # and np.random.randint(2) > 0 and False:
        list_algos = algos[benchmark][:5] + [
            "CSEC10",
            "NGOpt",
            "NLOPT_LN_SBPLX",
        ]
        return (
            list_algos  # [np.random.choice(list_algos)]
            if (
                "eras" in benchmark
                or "tial_instrum" in benchmark
                or "big" in benchmark
                or "lsgo" in benchmark
                or "rock" in benchmark
            )
            else list_algos  # list(np.random.choice(list_algos, 5))
        )
    if benchmark in algos:
        list_algos = algos[benchmark]
        return (
            list_algos  # [np.random.choice(list_algos)]
            if (
                "eras" in benchmark
                or "tial_instrum" in benchmark
                or "big" in benchmark
                or "lsgo" in benchmark
                or "rock" in benchmark
            )
            else list_algos  # list(np.random.choice(list_algos, 5))
        )
    return [
        "NgDS3",
        "NgIoh4",
        "NgIoh21",
        "NGOpt",
        "NGDSRW",
    ]

    # Here, we pseudo-randomly draw one optim in the provided list,
    # depending on the host (so that each host is using the same optim).
    #    list_optims = x
    #    list_optims = ["BAR", "BAR2", "BAR3"]
    #    list_optims = ["BAR", "BAR2", "BAR3", "BAR4", "NGOpt", "NGOptRW", "CMandAS2"]
    #    list_optims = ["QOTPDE", "LQOTPDE", "LQODE", "BAR4", "NGOpt", "CMandAS2"]
    #    list_optims = ["QOTPDE", "LQOTPDE", "LQODE"]
    #    list_optims = ["SPQODE", "SQOPSO", "DiagonalCMA"]
    #    list_optims = ["BAR", "BAR3", "BAR2", "BAR4", "SPQODE", "SQOPSO", "DiagonalCMA"]
    #    list_optims = ["QODE", "CMA", "SQOPSO", "RandomSearch", "OnePlusOne", "DE"]
    #    list_optims = ["AX", "SMAC3", "pysot"]
    #    # list_optims = ["DiagonalCMA"]
    #    list_optims = ["GeneticDE"]
    #    list_optims = [
    #        "NGOpt",
    #        "CMA",
    #        "DiagonalCMA",
    #        "GeneticDE",
    #        "SQOPSO",
    #        "QODE",
    #        "RandomSearch",
    #        "BFGS",
    #        "PSO",
    #        "DE",
    #        "MetaTuneRecentering",
    #        "MetaRecentering",
    #        "LhsDE",
    #        "HullCenterHullAvgCauchyScrHammersleySearch",
    #    ]
    #    list_optims = [
    #        "QOPSO",
    #        "OnePlusOne",
    #        "NaiveTBPSA",
    #        "LBFGSB",
    #        "LHSSearch",
    #        "DiscreteLenglerOnePlusOneT",
    #        "MetaModel",
    #        "MetaModelOnePlusOne",
    #        "LHSCauchySearch",
    #        "Cobyla",
    #        "CMA",
    #        "DiagonalCMA",
    #    ]
    def doint(s: str) -> int:  # Converting a string into an int.
        return 7 + sum([ord(c) * i for i, c in enumerate(s)])

    import socket

    host = socket.gethostname()

    if "iscr" in benchmark or "pbo" in benchmark:
        list_optims += [
            a
           