from typing import Any

# === Internal dependency: fklearn.common_docstrings ===
def learner_pred_fn_docstring(f_name, shap=...): ...
def learner_return_docstring(model_name): ...

# === Internal dependency: fklearn.preprocessing.schema ===
def column_duplicatable(columns_to_bind): ...

# === Internal dependency: fklearn.training.utils ===
def log_learner_time(learner, learner_name): ...

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