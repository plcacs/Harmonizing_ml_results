import collections.abc
from datetime import datetime
from itertools import chain, repeat
import pandas as pd
from toolz import curry
from numpy import nan

@curry
def evaluator_extractor(result: str, evaluator_name: Any):
    metric_value = result[evaluator_name] if result else nan
    return pd.DataFrame({evaluator_name: [metric_value]})

@curry
def combined_evaluator_extractor(result: Union[pandas.DataFrame, list[dict], list[int]], base_extractors: Union[pandas.DataFrame, list[dict], list[int]]):
    return pd.concat([x(result) for x in base_extractors], axis=1)

@curry
def split_evaluator_extractor_iteration(split_value: Union[str, typing.Callable[str, str], typing.Mapping, None], result: dict, split_col: str, base_extractor: Union[str, bytes, dict], eval_name: Union[None, str, dict[str, typing.Any]]=None) -> Union[str, bool, dict]:
    if eval_name is None:
        eval_name = 'split_evaluator__' + split_col
    key = eval_name + '_' + str(split_value)
    return base_extractor(result.get(key, {})).assign(**{eval_name: split_value})

@curry
def split_evaluator_extractor(result: Union[str, None, bool, typing.Sequence[str]], split_col: Union[str, None, bool, typing.Sequence[str]], split_values: Union[str, None, bool, typing.Sequence[str]], base_extractor: Union[str, None, bool, typing.Sequence[str]], eval_name: Union[None, str, bool, typing.Sequence[str]]=None):
    return pd.concat(list(map(split_evaluator_extractor_iteration(result=result, split_col=split_col, base_extractor=base_extractor, eval_name=eval_name), split_values)))

@curry
def temporal_split_evaluator_extractor(result: Any, time_col: Union[str, pandas.DataFrame], base_extractor: Union[str, typing.Iterable[str]], time_format: typing.Text='%Y-%m', eval_name: Union[None, str]=None):
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
def learning_curve_evaluator_extractor(result: dict[str, typing.Any], base_extractor: Union[dict[str, typing.Any], typing.Mapping]) -> Union[float, str]:
    return base_extractor(result).assign(lc_period_end=result['lc_period_end'])

@curry
def reverse_learning_curve_evaluator_extractor(result: Union[dict, dict[str, typing.Any]], base_extractor: Union[dict, dict[str, typing.Any], typing.MutableMapping]) -> Union[str, typing.Type, float]:
    return base_extractor(result).assign(reverse_lc_period_start=result['reverse_lc_period_start'])

@curry
def stability_curve_evaluator_extractor(result: Union[dict, dict[str, int], str], base_extractor: Union[dict, dict[str, int], str]) -> Union[list[dict], dict[sqlalchemy.exdeclarative.DeclarativeMeta, pandas.DataFrame], numpy.ndarray]:
    return base_extractor(result).assign(sc_period=result['sc_period'])

@curry
def repeat_split_log(split_log: Union[Checkpoint, Frame, None, Frame], results_len: int) -> Union[list, Checkpoint, Frame, None, Frame]:
    if isinstance(split_log, collections.abc.Iterable):
        n_repeat = results_len // len(split_log)
        return list(chain.from_iterable(zip(*repeat(split_log, n_repeat))))
    else:
        return split_log

@curry
def extract_base_iteration(result: Union[list[str], pandas.DataFrame], extractor: Union[numpy.ndarray, list[str], pandas.Series]) -> Union[list[str], int, bytes]:
    extracted_results = pd.concat(list(map(extractor, result['eval_results'])))
    repeat_fn = repeat_split_log(results_len=len(extracted_results))
    keys = result['split_log'].keys()
    assignments = {k: repeat_fn(result['split_log'][k]) for k in keys}
    return extracted_results.assign(fold_num=result['fold_num']).assign(**assignments)

@curry
def extract(validator_results: Union[list, dict[str, object]], extractor: Union[list, dict[str, object]]):
    return pd.concat(list(map(extract_base_iteration(extractor=extractor), validator_results)))

@curry
def extract_lc(validator_results: Union[bool, pandas.DataFrame, None, pandas._libs.tslibs.Resolution], extractor: Union[bool, pandas.DataFrame, None, pandas._libs.tslibs.Resolution]) -> Union[list[int], str, int]:
    return extract(validator_results, learning_curve_evaluator_extractor(base_extractor=extractor))

@curry
def extract_reverse_lc(validator_results: Union[dict, dict[str, object], SqlFile], extractor: Union[dict, dict[str, object], SqlFile]) -> Union[set[int], ks.Series, bool]:
    return extract(validator_results, reverse_learning_curve_evaluator_extractor(base_extractor=extractor))

@curry
def extract_sc(validator_results: Union[bool, list, dict[str, typing.Any]], extractor: Union[bool, list, dict[str, typing.Any]]) -> Union[int, dict[str, dict[str, str]], typing.Mapping]:
    return extract(validator_results, stability_curve_evaluator_extractor(base_extractor=extractor))

@curry
def extract_param_tuning_iteration(iteration: bool, tuning_log: Union[str, list[str]], base_extractor: Union[str, None], model_learner_name: numpy.ndarray) -> Union[str, bytes, typing.Deque]:
    iter_df = base_extractor(tuning_log[iteration]['validator_log'])
    return iter_df.assign(**tuning_log[iteration]['train_log'][model_learner_name]['parameters'])

@curry
def extract_tuning(tuning_log: Union[str, list[dict[str, typing.Any]]], base_extractor: Union[str, typing.Callable[..., collections.abc.Awaitable], None], model_learner_name: Union[str, typing.Callable[..., collections.abc.Awaitable], None]):
    iter_fn = extract_param_tuning_iteration(tuning_log=tuning_log, base_extractor=base_extractor, model_learner_name=model_learner_name)
    return pd.concat(list(map(iter_fn, range(len(tuning_log)))))

@curry
def permutation_extractor(results: Union[pandas.DataFrame, dict], base_extractor: Union[pandas.DataFrame, typing.OrderedDict]):
    df = pd.concat((base_extractor(r) for r in results['permutation_importance'].values()))
    df.index = results['permutation_importance'].keys()
    if 'permutation_importance_baseline' in results:
        baseline = base_extractor(results['permutation_importance_baseline'])
        baseline.index = ['baseline']
        df = pd.concat((df, baseline))
        for c in baseline.columns:
            df[c + '_delta_from_baseline'] = baseline[c].iloc[0] - df[c]
    return df