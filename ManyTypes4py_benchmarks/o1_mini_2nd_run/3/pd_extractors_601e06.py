import collections.abc
from datetime import datetime
from itertools import chain, repeat
import pandas as pd
from toolz import curry
from numpy import nan
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

@curry
def evaluator_extractor(result: Optional[Dict[str, Any]], evaluator_name: str) -> pd.DataFrame:
    metric_value = result[evaluator_name] if result else nan
    return pd.DataFrame({evaluator_name: [metric_value]})

@curry
def combined_evaluator_extractor(result: Optional[Dict[str, Any]], base_extractors: List[Callable[[Optional[Dict[str, Any]]], pd.DataFrame]]) -> pd.DataFrame:
    return pd.concat([x(result) for x in base_extractors], axis=1)

@curry
def split_evaluator_extractor_iteration(
    split_value: Any,
    result: Optional[Dict[str, Any]],
    split_col: str,
    base_extractor: Callable[[Optional[Dict[str, Any]], ...], pd.DataFrame],
    eval_name: Optional[str] = None
) -> pd.DataFrame:
    if eval_name is None:
        eval_name = 'split_evaluator__' + split_col
    key = eval_name + '_' + str(split_value)
    return base_extractor(result.get(key, {})).assign(**{eval_name: split_value})

@curry
def split_evaluator_extractor(
    result: Optional[Dict[str, Any]],
    split_col: str,
    split_values: Iterable[Any],
    base_extractor: Callable[[Optional[Dict[str, Any]], ...], pd.DataFrame],
    eval_name: Optional[str] = None
) -> pd.DataFrame:
    return pd.concat(list(map(
        split_evaluator_extractor_iteration(
            result=result,
            split_col=split_col,
            base_extractor=base_extractor,
            eval_name=eval_name
        ),
        split_values
    )))

@curry
def temporal_split_evaluator_extractor(
    result: Optional[Dict[str, Any]],
    time_col: str,
    base_extractor: Callable[[Optional[Dict[str, Any]], ...], pd.DataFrame],
    time_format: str = '%Y-%m',
    eval_name: Optional[str] = None
) -> pd.DataFrame:
    if eval_name is None:
        eval_name = 'split_evaluator__' + time_col
    split_keys = [key for key in result.keys() if eval_name in key]
    split_values = []
    for key in split_keys:
        date = key.split(eval_name)[1][1:]
        try:
            datetime.strptime(date, time_format)
            split_values.append(date)
        except ValueError:
            pass
    return split_evaluator_extractor(result, time_col, split_values, base_extractor)

@curry
def learning_curve_evaluator_extractor(
    result: Optional[Dict[str, Any]],
    base_extractor: Callable[[Optional[Dict[str, Any]], ...], pd.DataFrame]
) -> pd.DataFrame:
    return base_extractor(result).assign(lc_period_end=result['lc_period_end'])

@curry
def reverse_learning_curve_evaluator_extractor(
    result: Optional[Dict[str, Any]],
    base_extractor: Callable[[Optional[Dict[str, Any]], ...], pd.DataFrame]
) -> pd.DataFrame:
    return base_extractor(result).assign(reverse_lc_period_start=result['reverse_lc_period_start'])

@curry
def stability_curve_evaluator_extractor(
    result: Optional[Dict[str, Any]],
    base_extractor: Callable[[Optional[Dict[str, Any]], ...], pd.DataFrame]
) -> pd.DataFrame:
    return base_extractor(result).assign(sc_period=result['sc_period'])

@curry
def repeat_split_log(
    split_log: Union[Iterable[Any], Any],
    results_len: int
) -> List[Any]:
    if isinstance(split_log, collections.abc.Iterable):
        n_repeat = results_len // len(split_log)
        return list(chain.from_iterable(zip(*repeat(split_log, n_repeat))))
    else:
        return split_log

@curry
def extract_base_iteration(
    result: Dict[str, Any],
    extractor: Callable[[Optional[Dict[str, Any]], ...], pd.DataFrame]
) -> pd.DataFrame:
    extracted_results = pd.concat(list(map(extractor, result['eval_results'])))
    repeat_fn = repeat_split_log(results_len=len(extracted_results))
    keys = result['split_log'].keys()
    assignments = {k: repeat_fn(result['split_log'][k]) for k in keys}
    return extracted_results.assign(fold_num=result['fold_num']).assign(**assignments)

@curry
def extract(
    validator_results: List[Dict[str, Any]],
    extractor: Callable[[Dict[str, Any]], pd.DataFrame]
) -> pd.DataFrame:
    return pd.concat(list(map(extract_base_iteration(extractor=extractor), validator_results)))

@curry
def extract_lc(
    validator_results: List[Dict[str, Any]],
    extractor: Callable[[Optional[Dict[str, Any]], ...], pd.DataFrame]
) -> pd.DataFrame:
    return extract(validator_results, learning_curve_evaluator_extractor(base_extractor=extractor))

@curry
def extract_reverse_lc(
    validator_results: List[Dict[str, Any]],
    extractor: Callable[[Optional[Dict[str, Any]], ...], pd.DataFrame]
) -> pd.DataFrame:
    return extract(validator_results, reverse_learning_curve_evaluator_extractor(base_extractor=extractor))

@curry
def extract_sc(
    validator_results: List[Dict[str, Any]],
    extractor: Callable[[Optional[Dict[str, Any]], ...], pd.DataFrame]
) -> pd.DataFrame:
    return extract(validator_results, stability_curve_evaluator_extractor(base_extractor=extractor))

@curry
def extract_param_tuning_iteration(
    iteration: int,
    tuning_log: Dict[int, Dict[str, Any]],
    base_extractor: Callable[[Dict[str, Any]], pd.DataFrame],
    model_learner_name: str
) -> pd.DataFrame:
    iter_df = base_extractor(tuning_log[iteration]['validator_log'])
    return iter_df.assign(**tuning_log[iteration]['train_log'][model_learner_name]['parameters'])

@curry
def extract_tuning(
    tuning_log: Dict[int, Dict[str, Any]],
    base_extractor: Callable[[Dict[str, Any]], pd.DataFrame],
    model_learner_name: str
) -> pd.DataFrame:
    iter_fn = extract_param_tuning_iteration(
        tuning_log=tuning_log,
        base_extractor=base_extractor,
        model_learner_name=model_learner_name
    )
    return pd.concat(list(map(iter_fn, range(len(tuning_log)))))

@curry
def permutation_extractor(
    results: Dict[str, Any],
    base_extractor: Callable[[Dict[str, Any]], pd.DataFrame]
) -> pd.DataFrame:
    df = pd.concat((base_extractor(r) for r in results['permutation_importance'].values()))
    df.index = results['permutation_importance'].keys()
    if 'permutation_importance_baseline' in results:
        baseline = base_extractor(results['permutation_importance_baseline'])
        baseline.index = ['baseline']
        df = pd.concat((df, baseline))
        for c in baseline.columns:
            df[c + '_delta_from_baseline'] = baseline[c].iloc[0] - df[c]
    return df
