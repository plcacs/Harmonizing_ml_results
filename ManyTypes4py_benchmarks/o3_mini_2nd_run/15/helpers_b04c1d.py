import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.optimization import multiobjective as mobj
from . import base


class SpecialEvaluationExperiment(base.ExperimentFunction):
    def __init__(
        self,
        experiment: base.ExperimentFunction,
        evaluation: base.ExperimentFunction,
        pareto_size: Optional[int] = None,
        pareto_subset: str = "random",
        pareto_subset_tentatives: int = 30,
    ) -> None:
        self._experiment = experiment
        self._evaluation = evaluation
        self._pareto_size = pareto_size
        self._pareto_subset = pareto_subset
        self._pareto_subset_tentatives = pareto_subset_tentatives
        super().__init__(self._delegate_to_experiment, experiment.parametrization)
        self.add_descriptors(non_proxy_function=False)
        if self._pareto_size is None:
            names = [name for name in self._descriptors if name.startswith("pareto_")]
            for name in names:
                del self._descriptors[name]

    def _delegate_to_experiment(self, *args: Any, **kwargs: Any) -> Any:
        return self._experiment(*args, **kwargs)

    def copy(self) -> "SpecialEvaluationExperiment":
        instance: SpecialEvaluationExperiment = super().copy()
        for name in ["_experiment", "_evaluation"]:
            setattr(instance, name, getattr(self, name).copy())
        return instance

    def compute_pseudotime(self, input_parameter: Any, loss: Any) -> Any:
        return self._experiment.compute_pseudotime(input_parameter, loss)

    def evaluation_function(self, *recommendations: Any) -> Any:
        if self._pareto_size is not None and len(recommendations) > self._pareto_size:
            hypervolume = mobj.HypervolumePareto(upper_bounds=self.multiobjective_upper_bounds)
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
    def descriptors(self) -> Dict[str, Any]:
        desc: Dict[str, Any] = dict(self._descriptors)
        desc.update(self._experiment.descriptors)
        return desc

    @classmethod
    def create_crossvalidation_experiments(
        cls: Type["SpecialEvaluationExperiment"],
        experiments: Sequence[base.ExperimentFunction],
        training_only_experiments: Sequence[base.ExperimentFunction] = (),
        pareto_size: int = 12,
        pareto_subset_methods: Tuple[str, ...] = (
            "random",
            "loss-covering",
            "EPS",
            "domain-covering",
            "hypervolume",
        ),
    ) -> List["SpecialEvaluationExperiment"]:
        funcs: List[SpecialEvaluationExperiment] = []
        if "PYTEST_NEVERGRAD" in os.environ:
            pareto_subset_methods = ("random",)
        for pareto_subset in pareto_subset_methods:
            params = {"pareto_size": pareto_size, "pareto_subset": pareto_subset}
            for eval_xp in experiments:
                trainxps: List[base.ExperimentFunction] = [xp for xp in experiments if xp != eval_xp]
                if training_only_experiments is not None:
                    trainxps += list(training_only_experiments)
                if len(trainxps) == 1:
                    experiment: base.ExperimentFunction = trainxps[0]
                else:
                    param = eval_xp.parametrization
                    upper_bounds = [xp(*param.args, **param.kwargs) for xp in trainxps]
                    experiment = base.MultiExperiment(trainxps, upper_bounds)
                funcs.append(cls(experiment, eval_xp, **params))
        return funcs