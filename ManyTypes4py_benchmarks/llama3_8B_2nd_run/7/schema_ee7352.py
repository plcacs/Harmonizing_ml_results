import functools
import inspect
import pandas as pd
import toolz
from typing import Any, Callable, Dict, List, Optional, Union
from fklearn.types import LearnerLogType, LearnerReturnType

@toolz.curry
def feature_duplicator(
    df: pd.DataFrame, 
    columns_to_duplicate: Optional[List[str]], 
    columns_mapping: Optional[Dict[str, str]], 
    prefix: Optional[str], 
    suffix: Optional[str]
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]:
    """
    Duplicates some columns in the dataframe.

    ...

    Returns
    ----------
    increased_dataset : pandas.DataFrame
        A dataset with repeated columns
    """
    ...

def column_duplicatable(
    columns_to_bind: str
) -> Callable[[Callable], Callable]:
    """
    Decorator to prepend the feature_duplicator learner.

    ...

    Parameters
    ----------
    columns_to_bind: str
        Sets feature_duplicator's "columns_to_duplicate" parameter equal to the
        `columns_to_bind` parameter from the decorated learner
    """

    def _decorator(child: Callable) -> Callable:
        mixin = feature_duplicator

        def _init(*args: Any, **kwargs: Any) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]:
            mixin_spec = inspect.getfullargspec(mixin)
            mixin_named_args = set(mixin_spec.args) | set(mixin_spec.kwonlyargs)
            mixin_kwargs = {key: value for key, value in kwargs.items() if key in mixin_named_args}
            child_spec = inspect.getfullargspec(child)
            child_named_args = set(child_spec.args) | set(child_spec.kwonlyargs)
            child_kwargs = {key: value for key, value in kwargs.items() if key in child_named_args}
            child_arg_names = list(inspect.signature(child).parameters.keys())
            columns_to_bind_idx = child_arg_names.index(columns_to_bind)
            curry_is_ready = not child._should_curry(args, child_kwargs)
            if curry_is_ready:
                columns_to_duplicate = kwargs[columns_to_bind] if columns_to_bind in kwargs else args[columns_to_bind_idx]
                mixin_fn, mixin_df, mixin_log = mixin(args[0], **mixin_kwargs, columns_to_duplicate=columns_to_duplicate)
                child_fn, child_df, child_log = child(mixin_df, *args[1:], **child_kwargs)
                return (toolz.compose(child_fn, mixin_fn), child_df, {**mixin_log, **child_log})
            else:
                return functools.update_wrapper(functools.partial(_init, *args, **kwargs), child(*args[1:], **child_kwargs))
        callable_fn = functools.update_wrapper(_init, child)
        return callable_fn
    return _decorator
