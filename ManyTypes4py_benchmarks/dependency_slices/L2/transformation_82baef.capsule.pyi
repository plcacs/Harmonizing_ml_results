from typing import Any

# === Internal dependency: fklearn.common_docstrings ===
def learner_pred_fn_docstring(f_name: str, shap: bool = ...) -> str: ...
def learner_return_docstring(model_name: str) -> str: ...

# === Internal dependency: fklearn.preprocessing.schema ===
def column_duplicatable(columns_to_bind: str) -> Callable: ...

# === Internal dependency: fklearn.training.utils ===
def log_learner_time(learner: UncurriedLearnerFnType, learner_name: str) -> UncurriedLearnerFnType: ...

# === Internal dependency: fklearn.types ===
LearnerLogType: Any
LearnerReturnType: Any

# === Third-party dependency: numpy ===
# Used symbols: array, digitize, empty, isnan, nan, random, searchsorted, sort, where

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, Series, qcut

# === Third-party dependency: sklearn.preprocessing ===
# Used symbols: StandardScaler

# === Third-party dependency: statsmodels.distributions ===
# Used symbols: empirical_distribution

# === Third-party dependency: toolz ===
# Used symbols: compose, curry, mapcat, merge