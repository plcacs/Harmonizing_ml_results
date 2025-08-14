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
    if False:
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
    list_optims = x
    algos: tp.Dict[str, tp.List[str]] = {}
    algos["aquacrop_fao"] = [
        "CMA",
        "CMandAS2",
        "DE",
        "MetaModel",
        "NGOpt10",
    ]
    # ... (rest of the algos dictionary remains the same)
    
    benchmark = str(inspect.stack()[1].function)
    if benchmark in algos:
        list_algos = algos[benchmark][:5] + ["CSEC10", "NGOpt", "NLOPT_LN_SBPLX"]
        return list_algos if ("eras" in benchmark or "tial_instrum" in benchmark or "big" in benchmark or "lsgo" in benchmark or "rock" in benchmark) else list_algos
    if benchmark in algos:
        list_algos = algos[benchmark]
        return list_algos if ("eras" in benchmark or "tial_instrum" in benchmark or "big" in benchmark or "lsgo" in benchmark or "rock" in benchmark) else list_algos
    return ["NgDS3", "NgIoh4", "NgIoh21", "NGOpt", "NGDSRW"]

def skip_ci(*, reason: str) -> None:
    """Only use this if there is a good reason for not testing the xp,
    such as very slow for instance (>1min) with no way to make it faster.
    This is dangerous because it won't test reproducibility and the experiment
    may therefore be corrupted with no way to notice it automatically.
    """
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
    optims = ["OnePlusOne", "RandomSearch", "CMA", "DE", "TwoPointsDE", "HyperOpt", "Cobyla"]
    optims = refactor_optims(optims)
    datasets = ["kerasBoston", "diabetes", "auto-mpg", "red-wine", "white-wine"]
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

# ... (rest of the functions with their type annotations)

@registry.register
def lsgo() -> tp.Iterator[Experiment]:
    optims = ["CMA", "NGOpt", "NGOptRW"]
    optims = refactor_optims(optims)
    for i in range(1, 16):
        for optim in optims:
            for budget in [120000, 600000, 3000000]:
                yield Experiment(lsgo_makefunction(i).instrumented(), optim, budget=budget)

@registry.register
def smallbudget_lsgo() -> tp.Iterator[Experiment]:
    optims = ["CMA", "NGOpt", "NGOptRW"]
    optims = refactor_optims(optims)
    for i in range(1, 16):
        for optim in optims:
            for budget in [1200, 6000, 30000]:
                yield Experiment(lsgo_makefunction(i).instrumented(), optim, budget=budget)
