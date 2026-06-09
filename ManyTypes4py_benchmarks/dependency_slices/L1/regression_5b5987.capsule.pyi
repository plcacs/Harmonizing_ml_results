from typing import Any

# === Third-party dependency: catboost ===
# Used symbols: CatBoostRegressor, Pool, __version__

# === Internal dependency: fklearn.common_docstrings ===
def learner_pred_fn_docstring(f_name, shap=...): ...
def learner_return_docstring(model_name): ...

# === Internal dependency: fklearn.training.utils ===
def log_learner_time(learner, learner_name): ...
def expand_features_encoded(df, features): ...

# === Internal dependency: fklearn.types ===
LearnerReturnType: Any

# === Third-party dependency: lightgbm ===
# Used symbols: Dataset, __version__, train

# === Third-party dependency: numpy ===
# Used symbols: repeat

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Third-party dependency: shap ===
# Used symbols: TreeExplainer

# === Third-party dependency: sklearn ===
__version__: str

# === Third-party dependency: sklearn.gaussian_process ===
# Used symbols: GaussianProcessRegressor

# === Third-party dependency: sklearn.linear_model ===
# Used symbols: ElasticNet, LinearRegression

# === Third-party dependency: toolz ===
# Used symbols: assoc, curry, merge

# === Third-party dependency: xgboost ===
# re-export: from .core import DMatrix
# re-export: from .training import train
__version__: _py_version

# === Third-party dependency: xgboost.core ===
def _py_version() -> str: ...