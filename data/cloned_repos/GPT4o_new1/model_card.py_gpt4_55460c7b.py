import os
import logging
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Callable
from allennlp.common.from_params import FromParams

from allennlp.models import Model
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)


def get_description(model_class: Callable[..., Model]) -> str:
    """
    Returns the model's description from the docstring.
    """
    return model_class.__doc__.split("# Parameters")[0].strip()


class ModelCardInfo(FromParams):
    def to_dict(self) -> Dict[str, Any]:
        """
        Only the non-empty attributes are returned, to minimize empty values.
        """
        info = {}
        for key, val in self.__dict__.items():
            if val:
                info[key] = val
        return info

    def __str__(self) -> str:
        display = ""
        for key, val in self.to_dict().items():
            display += "\n" + key.replace("_", " ").capitalize() + ": "
            display += "\n\t" + str(val).replace("\n", "\n\t") + "\n"
        if not display:
            display = super(ModelCardInfo, self).__str__()
        return display.strip()


@dataclass(frozen=True)
class Paper(ModelCardInfo):
    title: Optional[str] = None
    url: Optional[str] = None
    citation: Optional[str] = None


class ModelDetails(ModelCardInfo):
    def __init__(
        self,
        description: Optional[str] = None,
        short_description: Optional[str] = None,
        developed_by: Optional[str] = None,
        contributed_by: Optional[str] = None,
        date: Optional[str] = None,
        version: Optional[str] = None,
        model_type: Optional[str] = None,
        paper: Optional[Union[str, Dict, Paper]] = None,
        license: Optional[str] = None,
        contact: Optional[str] = None,
    ) -> None:
        self.description = description
        self.short_description = short_description
        self.developed_by = developed_by
        self.contributed_by = contributed_by
        self.date = date
        self.version = version
        self.model_type = model_type
        if isinstance(paper, Paper):
            self.paper = paper
        elif isinstance(paper, Dict):
            self.paper = Paper(**paper)
        else:
            self.paper = Paper(title=paper)
        self.license = license
        self.contact = contact


@dataclass(frozen=True)
class IntendedUse(ModelCardInfo):
    primary_uses: Optional[str] = None
    primary_users: Optional[str] = None
    out_of_scope_use_cases: Optional[str] = None


@dataclass(frozen=True)
class Factors(ModelCardInfo):
    relevant_factors: Optional[str] = None
    evaluation_factors: Optional[str] = None


@dataclass(frozen=True)
class Metrics(ModelCardInfo):
    model_performance_measures: Optional[str] = None
    decision_thresholds: Optional[str] = None
    variation_approaches: Optional[str] = None


@dataclass(frozen=True)
class Dataset(ModelCardInfo):
    name: Optional[str] = None
    url: Optional[str] = None
    processed_url: Optional[str] = None
    notes: Optional[str] = None


class EvaluationData(ModelCardInfo):
    def __init__(
        self,
        dataset: Optional[Union[str, Dict, Dataset]] = None,
        motivation: Optional[str] = None,
        preprocessing: Optional[str] = None,
    ) -> None:
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, Dict):
            self.dataset = Dataset(**dataset)
        else:
            self.dataset = Dataset(name=dataset)
        self.motivation = motivation
        self.preprocessing = preprocessing

    def to_dict(self) -> Dict[str, Any]:
        info = {}
        for key, val in self.__dict__.items():
            if val:
                info["evaluation_" + key] = val
        return info


class TrainingData(ModelCardInfo):
    def __init__(
        self,
        dataset: Optional[Union[str, Dict, Dataset]] = None,
        motivation: Optional[str] = None,
        preprocessing: Optional[str] = None,
    ) -> None:
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, Dict):
            self.dataset = Dataset(**dataset)
        else:
            self.dataset = Dataset(name=dataset)
        self.motivation = motivation
        self.preprocessing = preprocessing

    def to_dict(self) -> Dict[str, Any]:
        info = {}
        for key, val in self.__dict__.items():
            if val:
                info["training_" + key] = val
        return info


@dataclass(frozen=True)
class QuantitativeAnalyses(ModelCardInfo):
    unitary_results: Optional[str] = None
    intersectional_results: Optional[str] = None


@dataclass(frozen=True)
class ModelEthicalConsiderations(ModelCardInfo):
    ethical_considerations: Optional[str] = None


@dataclass(frozen=True)
class ModelCaveatsAndRecommendations(ModelCardInfo):
    caveats_and_recommendations: Optional[str] = None


class ModelUsage(ModelCardInfo):
    _storage_location = "https://storage.googleapis.com/allennlp-public-models/"
    _config_location = (
        "https://raw.githubusercontent.com/allenai/allennlp-models/main/training_config"
    )

    def __init__(
        self,
        archive_file: Optional[str] = None,
        training_config: Optional[str] = None,
        install_instructions: Optional[str] = None,
        overrides: Optional[Dict] = None,
    ) -> None:

        if archive_file and not archive_file.startswith("https:"):
            archive_file = os.path.join(self._storage_location, archive_file)

        if training_config and not training_config.startswith("https:"):
            training_config = os.path.join(self._config_location, training_config)

        self.archive_file = archive_file
        self.training_config = training_config
        self.install_instructions = install_instructions
        self.overrides = overrides


class ModelCard(ModelCardInfo):
    def __init__(
        self,
        id: str,
        registered_model_name: Optional[str] = None,
        model_class: Optional[Callable[..., Model]] = None,
        registered_predictor_name: Optional[str] = None,
        display_name: Optional[str] = None,
        task_id: Optional[str] = None,
        model_usage: Optional[Union[str, ModelUsage]] = None,
        model_details: Optional[Union[str, ModelDetails]] = None,
        intended_use: Optional[Union[str, IntendedUse]] = None,
        factors: Optional[Union[str, Factors]] = None,
        metrics: Optional[Union[str, Metrics]] = None,
        evaluation_data: Optional[Union[str, EvaluationData]] = None,
        training_data: Optional[Union[str, TrainingData]] = None,
        quantitative_analyses: Optional[Union[str, QuantitativeAnalyses]] = None,
        model_ethical_considerations: Optional[Union[str, ModelEthicalConsiderations]] = None,
        model_caveats_and_recommendations: Optional[
            Union[str, ModelCaveatsAndRecommendations]
        ] = None,
    ) -> None:

        assert id
        if not model_class and registered_model_name:
            try:
                model_class = Model.by_name(registered_model_name)
            except ConfigurationError:
                logger.warning("{} is not a registered model.".format(registered_model_name))

        if model_class:
            display_name = display_name or model_class.__name__
            model_details = model_details or get_description(model_class)
            if not registered_predictor_name:
                registered_predictor_name = model_class.default_predictor  # type: ignore

        if isinstance(model_usage, str):
            model_usage = ModelUsage(archive_file=model_usage)
        if isinstance(model_details, str):
            model_details = ModelDetails(description=model_details)
        if isinstance(intended_use, str):
            intended_use = IntendedUse(primary_uses=intended_use)
        if isinstance(factors, str):
            factors = Factors(relevant_factors=factors)
        if isinstance(metrics, str):
            metrics = Metrics(model_performance_measures=metrics)
        if isinstance(evaluation_data, str):
            evaluation_data = EvaluationData(dataset=evaluation_data)
        if isinstance(training_data, str):
            training_data = TrainingData(dataset=training_data)
        if isinstance(quantitative_analyses, str):
            quantitative_analyses = QuantitativeAnalyses(unitary_results=quantitative_analyses)
        if isinstance(model_ethical_considerations, str):
            model_ethical_considerations = ModelEthicalConsiderations(model_ethical_considerations)
        if isinstance(model_caveats_and_recommendations, str):
            model_caveats_and_recommendations = ModelCaveatsAndRecommendations(
                model_caveats_and_recommendations
            )

        self.id = id
        self.registered_model_name = registered_model_name
        self.registered_predictor_name = registered_predictor_name
        self.display_name = display_name
        self.task_id = task_id
        self.model_usage = model_usage
        self.model_details = model_details
        self.intended_use = intended_use
        self.factors = factors
        self.metrics = metrics
        self.evaluation_data = evaluation_data
        self.training_data = training_data
        self.quantitative_analyses = quantitative_analyses
        self.model_ethical_considerations = model_ethical_considerations
        self.model_caveats_and_recommendations = model_caveats_and_recommendations

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the `ModelCard` to a flat dictionary object. This can be converted to
        json and passed to any front-end.
        """
        info = {}
        for key, val in self.__dict__.items():
            if key != "id":
                if isinstance(val, ModelCardInfo):
                    info.update(val.to_dict())
                else:
                    if val is not None:
                        info[key] = val
        return info
