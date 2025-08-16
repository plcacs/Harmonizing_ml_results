from typing import Callable, List
from toolz.curried import curry, take, first
from fklearn.tuning.utils import get_best_performing_log, get_avg_metric_from_extractor, get_used_features
from fklearn.types import ExtractorFnType, ListLogListType
StopFnType = Callable[[ListLogListType], bool]

def aggregate_stop_funcs(*stop_funcs: List[Callable[[ListLogListType], bool]]) -> Callable[[ListLogListType], bool]:
    def p(logs: ListLogListType) -> bool:
        return any([stop_fn(logs) for stop_fn in stop_funcs])
    return p

@curry
def stop_by_iter_num(logs: ListLogListType, iter_limit: int = 50) -> bool:
    return len(logs) >= iter_limit

@curry
def stop_by_no_improvement(logs: ListLogListType, extractor: ExtractorFnType, metric_name: str, early_stop: int = 3, threshold: float = 0.001) -> bool:
    if len(logs) < early_stop:
        return False
    limited_logs = list(take(early_stop, logs))
    curr_auc = get_avg_metric_from_extractor(limited_logs[-1], extractor, metric_name)
    return all([curr_auc - get_avg_metric_from_extractor(log, extractor, metric_name) <= threshold for log in limited_logs[:-1]])

@curry
def stop_by_no_improvement_parallel(logs: ListLogListType, extractor: ExtractorFnType, metric_name: str, early_stop: int = 3, threshold: float = 0.001) -> bool:
    if len(logs) < early_stop:
        return False
    log_list = [get_best_performing_log(log, extractor, metric_name) for log in logs]
    limited_logs = list(take(early_stop, log_list))
    curr_auc = get_avg_metric_from_extractor(limited_logs[-1], extractor, metric_name)
    return all([curr_auc - get_avg_metric_from_extractor(log, extractor, metric_name) <= threshold for log in limited_logs[:-1]])

@curry
def stop_by_num_features(logs: ListLogListType, min_num_features: int = 50) -> bool:
    return len(get_used_features(first(logs))) <= min_num_features

@curry
def stop_by_num_features_parallel(logs: ListLogListType, extractor: ExtractorFnType, metric_name: str, min_num_features: int = 50) -> bool:
    best_log = get_best_performing_log(first(logs), extractor, metric_name)
    return stop_by_num_features([best_log], min_num_features)
