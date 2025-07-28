#!/usr/bin/env python3
"""
A specification for defining model cards as described in
[Model Cards for Model Reporting (Mitchell et al, 2019)]
(https://api.semanticscholar.org/CorpusID:52946140)

The descriptions of the fields and some examples
are taken from the paper.

The specification is provided to prompt model developers
to think about the various aspects that should ideally
be reported. The information filled should adhere to
the spirit of transparency rather than the letter; i.e.,
it should not be filled for the sake of being filled. If
the information cannot be inferred, it should be left empty.
"""
import os
import logging
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Callable, Type
from allennlp.common.from_params import FromParams
from allennlp.models import Model
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)

def get_description(model_class: Type[Any]) -> str:
    """
    Returns the model's description from the docstring.
    """
    return model_class.__doc__.split('# Parameters')[0].strip()

class ModelCardInfo(FromParams):
    def to_dict(self) -> Dict[str, Any]:
        """
        Only the non-empty attributes are returned, to minimize empty values.
        """
        info: Dict[str, Any] = {}
        for key, val in self.__dict__.items():
            if val:
                info[key] = val
        return info

    def __str__(self) -> str:
        display: str = ''
        for key, val in self.to_dict().items():
            display += '\n' + key.replace('_', ' ').capitalize() + ': '
            display += '\n\t' + str(val).replace('\n', '\n\t') + '\n'
        if not display:
            display = super(ModelCardInfo, self).__str__()
        return display.strip()

@dataclass(frozen=True)
class Paper(ModelCardInfo):
    """
    This provides information about the paper.

    # Parameters

    title : `str`
        The name of the paper.

    url : `str`
        A web link to the paper.

    citation : `str`
        The BibTex for the paper.
    """
    title: Optional[str] = None
    url: Optional[str] = None
    citation: Optional[str] = None

class ModelDetails(ModelCardInfo):
    """
    This provides the basic information about the model.

    # Parameters

    description : `str`
        A high-level overview of the model.
    short_description : `str`
        A one-line description of the model.
    developed_by : `str`
        Person/organization that developed the model.
    contributed_by : `str`
        Person that contributed the model to the repository.
    date : `str`
        The date on which the model was contributed.
    version : `str`
        The version of the model.
    model_type : `str`
        The type of the model; the basic architecture.
    paper : `Union[str, Dict, Paper]`
        The paper on which the model is based.
    license : `str`
        License information for the model.
    contact : `str`
        The email address to reach out to the relevant developers/contributors.
    """
    def __init__(self,
                 description: Optional[str] = None,
                 short_description: Optional[str] = None,
                 developed_by: Optional[str] = None,
                 contributed_by: Optional[str] = None,
                 date: Optional[str] = None,
                 version: Optional[str] = None,
                 model_type: Optional[str] = None,
                 paper: Optional[Union[str, Dict[str, Any], Paper]] = None,
                 license: Optional[str] = None,
                 contact: Optional[str] = None) -> None:
        self.description: Optional[str] = description
        self.short_description: Optional[str] = short_description
        self.developed_by: Optional[str] = developed_by
        self.contributed_by: Optional[str] = contributed_by
        self.date: Optional[str] = date
        self.version: Optional[str] = version
        self.model_type: Optional[str] = model_type
        if isinstance(paper, Paper):
            self.paper: Paper = paper
        elif isinstance(paper, Dict):
            self.paper = Paper(**paper)
        else:
            self.paper = Paper(title=paper)
        self.license: Optional[str] = license
        self.contact: Optional[str] = contact

@dataclass(frozen=True)
class IntendedUse(ModelCardInfo):
    """
    This determines what the model should and should not be used for.

    # Parameters

    primary_uses : `str`
        Details the primary intended uses of the model.
    primary_users : `str`
        The primary intended users.
    out_of_scope_use_cases : `str`
        Highlights the technology that the model might easily be confused with.
    """
    primary_uses: Optional[str] = None
    primary_users: Optional[str] = None
    out_of_scope_use_cases: Optional[str] = None

@dataclass(frozen=True)
class Factors(ModelCardInfo):
    """
    This provides a summary of relevant factors such as
    demographics, instrumentation used, etc. for which the
    model performance may vary.

    # Parameters

    relevant_factors : `str`
         The foreseeable salient factors.
    evaluation_factors : `str`
        Mentions the factors that are being reported.
    """
    relevant_factors: Optional[str] = None
    evaluation_factors: Optional[str] = None

@dataclass(frozen=True)
class Metrics(ModelCardInfo):
    """
    This lists the reported metrics and the reasons for choosing them.

    # Parameters

    model_performance_measures : `str`
        Which model performance measures were selected and the reasons for selecting them.
    decision_thresholds : `str`
        If decision thresholds are used, what are they, and the reasons for choosing them.
    variation_approaches : `str`
        How are the measurements calculated? Eg. standard deviation, variance.
    """
    model_performance_measures: Optional[str] = None
    decision_thresholds: Optional[str] = None
    variation_approaches: Optional[str] = None

@dataclass(frozen=True)
class Dataset(ModelCardInfo):
    """
    This provides basic information about the dataset.

    # Parameters

    name : `str`
        The name of the dataset.
    url : `str`
        A web link to the dataset information/datasheet.
    processed_url : `str`
        A web link to a downloadable version of the dataset.
    notes: `str`
        Any other notes on downloading/processing the data.
    """
    name: Optional[str] = None
    url: Optional[str] = None
    processed_url: Optional[str] = None
    notes: Optional[str] = None

class EvaluationData(ModelCardInfo):
    """
    This provides information about the evaluation data.

    # Parameters

    dataset : `Union[str, Dict, Dataset]`
        The dataset(s) used to evaluate the model.
    motivation : `str`
        The reasons for selecting the dataset(s).
    preprocessing : `str`
        How was the data preprocessed for evaluation?
    """
    def __init__(self,
                 dataset: Optional[Union[str, Dict[str, Any], Dataset]] = None,
                 motivation: Optional[str] = None,
                 preprocessing: Optional[str] = None) -> None:
        if isinstance(dataset, Dataset):
            self.dataset: Dataset = dataset
        elif isinstance(dataset, Dict):
            self.dataset = Dataset(**dataset)
        else:
            self.dataset = Dataset(name=dataset)
        self.motivation: Optional[str] = motivation
        self.preprocessing: Optional[str] = preprocessing

    def to_dict(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        for key, val in self.__dict__.items():
            if val:
                info['evaluation_' + key] = val
        return info

class TrainingData(ModelCardInfo):
    """
    This provides information about the training data.

    # Parameters

    dataset : `Union[str, Dict, Dataset]`
        The dataset(s) used to train the model.
    motivation : `str`
        The reasons for selecting the dataset(s).
    preprocessing : `str`
        How was the data preprocessed for training?
    """
    def __init__(self,
                 dataset: Optional[Union[str, Dict[str, Any], Dataset]] = None,
                 motivation: Optional[str] = None,
                 preprocessing: Optional[str] = None) -> None:
        if isinstance(dataset, Dataset):
            self.dataset: Dataset = dataset
        elif isinstance(dataset, Dict):
            self.dataset = Dataset(**dataset)
        else:
            self.dataset = Dataset(name=dataset)
        self.motivation: Optional[str] = motivation
        self.preprocessing: Optional[str] = preprocessing

    def to_dict(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        for key, val in self.__dict__.items():
            if val:
                info['training_' + key] = val
        return info

@dataclass(frozen=True)
class QuantitativeAnalyses(ModelCardInfo):
    """
    This provides disaggregated evaluation of how the
    model performed based on chosen metrics.

    # Parameters

    unitary_results : `str`
        The performance of the model with respect to each chosen factor.
    intersectional_results : `str`
        The performance of the model with respect to the intersection of the evaluated factors.
    """
    unitary_results: Optional[str] = None
    intersectional_results: Optional[str] = None

@dataclass(frozen=True)
class ModelEthicalConsiderations(ModelCardInfo):
    """
    This highlights any ethical considerations to keep
    in mind when using the model.
    """
    ethical_considerations: Optional[str] = None

@dataclass(frozen=True)
class ModelCaveatsAndRecommendations(ModelCardInfo):
    """
    This lists any additional concerns.
    """
    caveats_and_recommendations: Optional[str] = None

class ModelUsage(ModelCardInfo):
    """
    archive_file : `str`, optional
        The location of model's pretrained weights.
    training_config : `str`, optional
        A url to the training config.
    install_instructions : `str`, optional
        Any additional instructions for installations.
    overrides : `Dict`, optional
        Optional overrides for the model's architecture.
    """
    _storage_location: str = 'https://storage.googleapis.com/allennlp-public-models/'
    _config_location: str = 'https://raw.githubusercontent.com/allenai/allennlp-models/main/training_config'

    def __init__(self,
                 archive_file: Optional[str] = None,
                 training_config: Optional[str] = None,
                 install_instructions: Optional[str] = None,
                 overrides: Optional[Dict[str, Any]] = None) -> None:
        if archive_file and (not archive_file.startswith('https:')):
            archive_file = os.path.join(self._storage_location, archive_file)
        if training_config and (not training_config.startswith('https:')):
            training_config = os.path.join(self._config_location, training_config)
        self.archive_file: Optional[str] = archive_file
        self.training_config: Optional[str] = training_config
        self.install_instructions: Optional[str] = install_instructions
        self.overrides: Optional[Dict[str, Any]] = overrides

class ModelCard(ModelCardInfo):
    """
    The model card stores the recommended attributes for model reporting.

    # Parameters

    id : `str`
        Model's id.
    registered_model_name : `str`, optional
        The model's registered name.
    model_class : `type`, optional
        If given, the `ModelCard` will pull some default information from the class.
    registered_predictor_name : `str`, optional
        The registered name of the corresponding predictor.
    display_name : `str`, optional
        The pretrained model's display name.
    task_id : `str`, optional
        The id of the task for which the model was built.
    model_usage: `Union[ModelUsage, str]`, optional
    model_details : `Union[ModelDetails, str]`, optional
    intended_use : `Union[IntendedUse, str]`, optional
    factors : `Union[Factors, str]`, optional
    metrics : `Union[Metrics, str]`, optional
    evaluation_data : `Union[EvaluationData, str]`, optional
    training_data : `Union[TrainingData, str]`, optional
    quantitative_analyses : `Union[QuantitativeAnalyses, str]`, optional
    model_ethical_considerations : `Union[ModelEthicalConsiderations, str]`, optional
    model_caveats_and_recommendations : `Union[ModelCaveatsAndRecommendations, str]`, optional
    """
    def __init__(self,
                 id: str,
                 registered_model_name: Optional[str] = None,
                 model_class: Optional[Type[Model]] = None,
                 registered_predictor_name: Optional[str] = None,
                 display_name: Optional[str] = None,
                 task_id: Optional[str] = None,
                 model_usage: Optional[Union[ModelUsage, str]] = None,
                 model_details: Optional[Union[ModelDetails, str]] = None,
                 intended_use: Optional[Union[IntendedUse, str]] = None,
                 factors: Optional[Union[Factors, str]] = None,
                 metrics: Optional[Union[Metrics, str]] = None,
                 evaluation_data: Optional[Union[EvaluationData, str]] = None,
                 training_data: Optional[Union[TrainingData, str]] = None,
                 quantitative_analyses: Optional[Union[QuantitativeAnalyses, str]] = None,
                 model_ethical_considerations: Optional[Union[ModelEthicalConsiderations, str]] = None,
                 model_caveats_and_recommendations: Optional[Union[ModelCaveatsAndRecommendations, str]] = None) -> None:
        assert id
        if not model_class and registered_model_name:
            try:
                model_class = Model.by_name(registered_model_name)  # type: ignore
            except ConfigurationError:
                logger.warning('{} is not a registered model.'.format(registered_model_name))
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
            model_ethical_considerations = ModelEthicalConsiderations(ethical_considerations=model_ethical_considerations)
        if isinstance(model_caveats_and_recommendations, str):
            model_caveats_and_recommendations = ModelCaveatsAndRecommendations(caveats_and_recommendations=model_caveats_and_recommendations)
        self.id: str = id
        self.registered_model_name: Optional[str] = registered_model_name
        self.registered_predictor_name: Optional[str] = registered_predictor_name
        self.display_name: Optional[str] = display_name
        self.task_id: Optional[str] = task_id
        self.model_usage: Optional[Union[ModelUsage, str]] = model_usage
        self.model_details: Optional[Union[ModelDetails, str]] = model_details
        self.intended_use: Optional[Union[IntendedUse, str]] = intended_use
        self.factors: Optional[Union[Factors, str]] = factors
        self.metrics: Optional[Union[Metrics, str]] = metrics
        self.evaluation_data: Optional[Union[EvaluationData, str]] = evaluation_data
        self.training_data: Optional[Union[TrainingData, str]] = training_data
        self.quantitative_analyses: Optional[Union[QuantitativeAnalyses, str]] = quantitative_analyses
        self.model_ethical_considerations: Optional[Union[ModelEthicalConsiderations, str]] = model_ethical_considerations
        self.model_caveats_and_recommendations: Optional[Union[ModelCaveatsAndRecommendations, str]] = model_caveats_and_recommendations

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the `ModelCard` to a flat dictionary object.
        """
        info: Dict[str, Any] = {}
        for key, val in self.__dict__.items():
            if key != 'id':
                if isinstance(val, ModelCardInfo):
                    info.update(val.to_dict())
                elif val is not None:
                    info[key] = val
        return info
