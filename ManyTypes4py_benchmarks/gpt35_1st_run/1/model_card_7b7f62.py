from typing import Optional, Union, Dict, Any

def get_description(model_class: type) -> str:
    ...

class ModelCardInfo(FromParams):
    def to_dict(self) -> Dict[str, Any]:
        ...
    def __str__(self) -> str:
        ...

@dataclass(frozen=True)
class Paper(ModelCardInfo):
    title: Optional[str] = None
    url: Optional[str] = None
    citation: Optional[str] = None

@dataclass(frozen=True)
class ModelDetails(ModelCardInfo):
    description: Optional[str] = None
    short_description: Optional[str] = None
    developed_by: Optional[str] = None
    contributed_by: Optional[str] = None
    date: Optional[str] = None
    version: Optional[str] = None
    model_type: Optional[str] = None
    paper: Union[str, Dict, Paper] = None
    license: Optional[str] = None
    contact: Optional[str] = None

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

@dataclass(frozen=True)
class EvaluationData(ModelCardInfo):
    dataset: Union[str, Dict, Dataset] = None
    motivation: Optional[str] = None
    preprocessing: Optional[str] = None

@dataclass(frozen=True)
class TrainingData(ModelCardInfo):
    dataset: Union[str, Dict, Dataset] = None
    motivation: Optional[str] = None
    preprocessing: Optional[str] = None

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
    archive_file: Optional[str] = None
    training_config: Optional[str] = None
    install_instructions: Optional[str] = None
    overrides: Optional[Dict] = None

@dataclass(frozen=True)
class ModelCard(ModelCardInfo):
    id: str
    registered_model_name: Optional[str] = None
    model_class: Optional[type] = None
    registered_predictor_name: Optional[str] = None
    display_name: Optional[str] = None
    task_id: Optional[str] = None
    model_usage: Union[ModelUsage, str] = None
    model_details: Union[ModelDetails, str] = None
    intended_use: Union[IntendedUse, str] = None
    factors: Union[Factors, str] = None
    metrics: Union[Metrics, str] = None
    evaluation_data: Union[EvaluationData, str] = None
    training_data: Union[TrainingData, str] = None
    quantitative_analyses: Union[QuantitativeAnalyses, str] = None
    model_ethical_considerations: Union[ModelEthicalConsiderations, str] = None
    model_caveats_and_recommendations: Union[ModelCaveatsAndRecommendations, str] = None
