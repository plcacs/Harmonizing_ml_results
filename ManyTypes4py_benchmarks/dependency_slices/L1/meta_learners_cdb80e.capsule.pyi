from typing import Any

# === Internal dependency: fklearn.common_docstrings ===
def learner_pred_fn_docstring(f_name, shap=...): ...
def learner_return_docstring(model_name): ...

# === Internal dependency: fklearn.exceptions.exceptions ===
class MultipleTreatmentsError(Exception): ...
class MissingControlError(Exception): ...
class MissingTreatmentError(Exception): ...

# === Internal dependency: fklearn.training.pipeline ===
def build_pipeline(*learners, has_repeated_learners=...): ...

# === Internal dependency: fklearn.types ===
LearnerReturnType: Any

# === Third-party dependency: numpy ===
# Used symbols: ndarray, ones, where, zeros

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Third-party dependency: toolz ===
# Used symbols: curry