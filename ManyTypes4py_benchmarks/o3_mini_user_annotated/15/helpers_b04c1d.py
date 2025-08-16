#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.optimization import multiobjective as mobj
from . import base


class SpecialEvaluationExperiment(base.ExperimentFunction):
    """Experiment which uses one experiment for the optimization,
    and another for the evaluation.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        experiment: base.ExperimentFunction,
        evaluation: base.ExperimentFunction,
        pareto_size: tp.Optional[int] = None,
        pareto_subset: str = "random",
        pareto_subset_tentatives: int = 30,
    ) -> None:
        self._experiment: base.ExperimentFunction = experiment
        self._evaluation: base.ExperimentFunction = evaluation
        self._pareto_size: tp.Optional[int] = pareto_size
        self._pareto_subset: str = pareto_subset
        self._pareto_subset_tentatives: int = pareto_subset_tentatives
        super().__init__(self._delegate_to_experiment, experiment.parametrization)
        self.add_descriptors(non_proxy_function=False)
        # remove multiobjective descriptors if singleobjective / no pareto subset
        if self._pareto_size is None:
            names = [name for name in self._descriptors if name.startswith("pareto_")]
            for name in names:
                del self._descriptors[name]

    def _delegate_to_experiment(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Loss:
        return self._experiment(*args, **kwargs)

    def copy(self) -> "SpecialEvaluationExperiment":
        """Creates a copy with new experiments and evaluations."""
        instance: SpecialEvaluationExperiment = super().copy()  # type: ignore
        for name in [
            "_experiment",
            "_evaluation",
        ]:
            setattr(instance, name, getattr(self, name).copy())
        return instance

    def compute_pseudotime(self, input_parameter: tp.Any, loss: tp.Loss) -> float:
        return self._experiment.compute_pseudotime(input_parameter, loss)

    def evaluation_function(self, *recommendations: p.Parameter) -> float:
        if self._pareto_size is not None and len(recommendations) > self._pareto_size:
            # select a subset
            hypervolume: mobj.HypervolumePareto = mobj.HypervolumePareto(upper_bounds=self.multiobjective_upper_bounds)  # type: ignore
            hypervolume.extend(recommendations)
            recommendations = tuple(
                hypervolume.pareto_front(
                    size=self._pareto_size,
                    subset=self._pareto_subset,
                    subset_tentatives=self._pareto_subset_tentatives,
                )
            )
        return min(self._evaluation.evaluation_function(recom) for recom in recommendations)

    @property
    def descriptors(self) -> tp.Dict[str, tp.Any]:
        """Description of the function parameterization, as a dict.
        This base class implementation provides function_class, noise_level, transform and dimension.
        """
        desc: tp.Dict[str, tp.Any] = dict(self._descriptors)
        desc.update(self._experiment.descriptors)
        # TODO descriptors for the evaluation function?
        return desc

    @classmethod
    def create_crossvalidation_experiments(
        cls,
        experiments: tp.List[base.ExperimentFunction],
        training_only_experiments: tp.Sequence[base.ExperimentFunction] = (),
        pareto_size: int = 12,
        pareto_subset_methods: tp.Sequence[str] = (
            "random",
            "loss-covering",
            "EPS",
            "domain-covering",
            "hypervolume",
        ),
    ) -> tp.List["SpecialEvaluationExperiment"]:
        """Returns a list of SpecialEvaluationExperiment corresponding to MOO cross-validation:
        Each experiment consists in optimizing all but one of the input ExperimentFunction's,
        and then considering that the score is the performance of the best solution in the
        approximate Pareto front for the excluded ExperimentFunction.

        Parameters
        ----------
        experiments: tp.List[base.ExperimentFunction]
            Iterable of experiment functions, used for creating the crossvalidation.
        training_only_experiments: tp.Sequence[base.ExperimentFunction]
            Iterable of experiment functions, used only as training functions in the crossvalidation 
            and never for test.
        pareto_size: int
            If provided, selects a subset of the full pareto front with the given maximum size.
        pareto_subset_methods: tp.Sequence[str]
            Methods for selecting the subset (see optimizer.pareto_front).
        """
        funcs: tp.List["SpecialEvaluationExperiment"] = []
        if "PYTEST_NEVERGRAD" in os.environ:
            pareto_subset_methods = ("random",)  # override for speed
        for pareto_subset in pareto_subset_methods:
            params: tp.Dict[str, tp.Any] = dict(pareto_size=pareto_size, pareto_subset=pareto_subset)
            for eval_xp in experiments:
                trainxps: tp.List[base.ExperimentFunction] = [xp for xp in experiments if xp != eval_xp]
                if training_only_experiments is not None:
                    trainxps += list(training_only_experiments)
                if len(trainxps) == 1:  # singleobjective
                    experiment: base.ExperimentFunction = trainxps[0]
                else:  # multiobjective
                    # uses origin as upper bound
                    param = eval_xp.parametrization
                    upper_bounds = [xp(*param.args, **param.kwargs) for xp in trainxps]
                    experiment = base.MultiExperiment(trainxps, upper_bounds)  # type: ignore
                funcs.append(cls(experiment, eval_xp, **params))
        return funcs
