import functools
import inspect
import pandas as pd
import toolz
from typing import Any, Callable, Dict, List, Optional, Tuple

from fklearn.types import LearnerLogType, LearnerReturnType

@toolz.curry
def feature_duplicator(
    df: pd.DataFrame,
    columns_to_duplicate: Optional[List[str]] = None,
    columns_mapping: Optional[Dict[str, str]] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]:
    """
    Duplicates some columns in the dataframe.

    When encoding features, a good practice is to save the encoded
    version in a different column rather than replacing the
    original values. The purpose of this function is to duplicate
    the column to be encoded, to be later replaced by the encoded
    values.

    The duplication method is used to preserve the original
    behaviour (replace).

    Parameters
    ----------
    df: pandas.DataFrame
        A Pandas' DataFrame with columns_to_duplicate columns

    columns_to_duplicate: list of str, optional
        List of columns names

    columns_mapping: Dict[str, str], optional
        Mapping of source columns to destination columns

    prefix: str, optional
        Prefix to add to columns to duplicate

    suffix: str, optional
        Suffix to add to columns to duplicate

    Returns
    -------
    Tuple containing:
        - Callable[[pd.DataFrame], pd.DataFrame]: Function to duplicate columns
        - pd.DataFrame: DataFrame with duplicated columns
        - LearnerLogType: Log information
    """
    if columns_to_duplicate:
        columns_final_mapping = (
            columns_mapping
            if columns_mapping is not None
            else {col: (prefix or "") + str(col) + (suffix or "") for col in columns_to_duplicate}
        )
    else:
        columns_final_mapping = {}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        categ_columns: Dict[str, pd.Series] = {
            dest_col: new_df[src_col] for src_col, dest_col in columns_final_mapping.items()
        }
        return new_df.assign(**categ_columns)

    p.__doc__ = feature_duplicator.__doc__
    
    log: LearnerLogType = {
        "feature_duplicator": {
            "columns_to_duplicate": columns_to_duplicate,
            "columns_mapping": columns_mapping,
            "prefix": prefix,
            "suffix": suffix,
            "columns_final_mapping": columns_final_mapping,
        }
    }
    
    return (p, p(df.copy()), log)

def column_duplicatable(
    columns_to_bind: str
) -> Callable[[Callable[..., LearnerReturnType]], Callable[..., LearnerReturnType]]:
    """
    Decorator to prepend the feature_duplicator learner.

    Identifies the columns to be duplicated and applies duplicator.

    Parameters
    ----------
    columns_to_bind: str
        Sets feature_duplicator's "columns_to_duplicate" parameter equal to the
        `columns_to_bind` parameter from the decorated learner

    Returns
    -------
    Callable: Decorated learner function
    """
    def _decorator(
        child: Callable[..., LearnerReturnType]
    ) -> Callable[..., LearnerReturnType]:
        mixin = feature_duplicator

        def _init(*args: Any, **kwargs: Any) -> LearnerReturnType:
            mixin_spec = inspect.getfullargspec(mixin)
            mixin_named_args = set(mixin_spec.args) | set(mixin_spec.kwonlyargs)
            mixin_kwargs: Dict[str, Any] = {key: value for key, value in kwargs.items() if key in mixin_named_args}
            
            child_spec = inspect.getfullargspec(child)
            child_named_args = set(child_spec.args) | set(child_spec.kwonlyargs)
            child_kwargs: Dict[str, Any] = {key: value for key, value in kwargs.items() if key in child_named_args}
            
            child_arg_names: List[str] = list(inspect.signature(child).parameters.keys())
            columns_to_bind_idx: int = child_arg_names.index(columns_to_bind)
            
            curry_is_ready: bool = not getattr(child, "_should_curry", lambda *a, **kw: False)(args, child_kwargs)
            
            if curry_is_ready:
                if columns_to_bind in kwargs:
                    columns_to_duplicate: Any = kwargs[columns_to_bind]
                else:
                    columns_to_duplicate = args[columns_to_bind_idx]
                
                mixin_fn, mixin_df, mixin_log = mixin(args[0], **mixin_kwargs, columns_to_duplicate=columns_to_duplicate)
                child_fn, child_df, child_log = child(mixin_df, *args[1:], **child_kwargs)
                
                combined_fn = toolz.compose(child_fn, mixin_fn)
                combined_log: LearnerLogType = {**mixin_log, **child_log}
                
                return (combined_fn, child_df, combined_log)
            else:
                return functools.update_wrapper(
                    functools.partial(_init, *args, **kwargs),
                    child(*args, **child_kwargs)
                )
        
        return functools.update_wrapper(_init, child)

    return _decorator
