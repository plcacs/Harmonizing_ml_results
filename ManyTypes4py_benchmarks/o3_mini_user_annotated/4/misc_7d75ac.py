#!/usr/bin/env python3
"""
Various tool function for Freqtrade and scripts
"""

import gzip
import logging
from collections.abc import Iterator, Mapping
from io import StringIO
from pathlib import Path
from typing import Any, Optional, TextIO, Union

import pandas as pd
import rapidjson

from freqtrade.enums import SignalTagType, SignalType

logger = logging.getLogger(__name__)


def dump_json_to_file(file_obj: TextIO, data: Any) -> None:
    """
    Dump JSON data into a file object.
    """
    rapidjson.dump(data, file_obj, default=str, number_mode=rapidjson.NM_NATIVE)


def file_dump_json(filename: Path, data: Any, is_zip: bool = False, log: bool = True) -> None:
    """
    Dump JSON data into a file.
    """
    if is_zip:
        if filename.suffix != ".gz":
            filename = filename.with_suffix(".gz")
        if log:
            logger.info(f'dumping json to "{filename}"')

        with gzip.open(filename, "wt", encoding="utf-8") as fpz:
            dump_json_to_file(fpz, data)
    else:
        if log:
            logger.info(f'dumping json to "{filename}"')
        with filename.open("w") as fp:
            dump_json_to_file(fp, data)

    logger.debug(f'done json to "{filename}"')


def json_load(datafile: TextIO) -> Any:
    """
    Load data with rapidjson using number_mode set to NM_NATIVE for speed.
    """
    return rapidjson.load(datafile, number_mode=rapidjson.NM_NATIVE)


def file_load_json(file: Path) -> Optional[Any]:
    if file.suffix != ".gz":
        gzipfile = file.with_suffix(file.suffix + ".gz")
    else:
        gzipfile = file
    # Try gzip file first, otherwise regular json file.
    if gzipfile.is_file():
        logger.debug(f"Loading historical data from file {gzipfile}")
        with gzip.open(gzipfile, "rt", encoding="utf-8") as datafile:
            pairdata = json_load(datafile)
    elif file.is_file():
        logger.debug(f"Loading historical data from file {file}")
        with file.open() as datafile:
            pairdata = json_load(datafile)
    else:
        return None
    return pairdata


def is_file_in_dir(file: Path, directory: Path) -> bool:
    """
    Helper function to check if file is in directory.
    """
    return file.is_file() and file.parent.samefile(directory)


def pair_to_filename(pair: str) -> str:
    for ch in ["/", " ", ".", "@", "$", "+", ":"]:
        pair = pair.replace(ch, "_")
    return pair


def deep_merge_dicts(source: dict[Any, Any],
                     destination: dict[Any, Any],
                     allow_null_overrides: bool = True) -> dict[Any, Any]:
    """
    Values from source override destination. The destination is returned (and modified).
    """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            deep_merge_dicts(value, node, allow_null_overrides)
        elif value is not None or allow_null_overrides:
            destination[key] = value
    return destination


def round_dict(d: Mapping[str, Any], n: int) -> dict[str, Any]:
    """
    Rounds float values in the dict to n digits after the decimal point.
    """
    return {k: (round(v, n) if isinstance(v, float) else v) for k, v in d.items()}


DictMap = Union[dict[str, Any], Mapping[str, Any]]


def safe_value_fallback(obj: DictMap, key1: str, key2: Optional[str] = None, default_value: Any = None) -> Any:
    """
    Search a value in obj; return it if it's not None.
    Then search key2 in obj and return that if it's not None, then use default_value.
    """
    if key1 in obj and obj[key1] is not None:
        return obj[key1]
    else:
        if key2 and key2 in obj and obj[key2] is not None:
            return obj[key2]
    return default_value


def safe_value_fallback2(dict1: DictMap, dict2: DictMap, key1: str, key2: str, default_value: Any = None) -> Any:
    """
    Search a value in dict1; return it if it's not None.
    Otherwise, fall back to dict2 and return key2 if it's not None.
    """
    if key1 in dict1 and dict1[key1] is not None:
        return dict1[key1]
    else:
        if key2 in dict2 and dict2[key2] is not None:
            return dict2[key2]
    return default_value


def plural(num: float, singular: str, plural: Optional[str] = None) -> str:
    return singular if (num == 1 or num == -1) else plural or singular + "s"


def chunks(lst: list[Any], n: int) -> Iterator[list[Any]]:
    """
    Split lst into chunks of size n.
    """
    for chunk in range(0, len(lst), n):
        yield lst[chunk:chunk + n]


def parse_db_uri_for_logging(uri: str) -> str:
    """
    Parse the DB URI and return the URI with the password censored if present.
    """
    parsed_db_uri = urlparse(uri)
    if not parsed_db_uri.netloc:
        return uri
    pwd = parsed_db_uri.netloc.split(":")[1].split("@")[0]
    return parsed_db_uri.geturl().replace(f":{pwd}@", ":*****@")


def dataframe_to_json(dataframe: pd.DataFrame) -> str:
    """
    Serialize a DataFrame to a JSON string.
    """
    return dataframe.to_json(orient="split")


def json_to_dataframe(data: str) -> pd.DataFrame:
    """
    Deserialize a JSON string into a DataFrame.
    """
    dataframe = pd.read_json(StringIO(data), orient="split")
    if "date" in dataframe.columns:
        dataframe["date"] = pd.to_datetime(dataframe["date"], unit="ms", utc=True)
    return dataframe


def remove_entry_exit_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove Entry and Exit signals from a DataFrame.
    """
    dataframe[SignalType.ENTER_LONG.value] = 0
    dataframe[SignalType.EXIT_LONG.value] = 0
    dataframe[SignalType.ENTER_SHORT.value] = 0
    dataframe[SignalType.EXIT_SHORT.value] = 0
    dataframe[SignalTagType.ENTER_TAG.value] = None
    dataframe[SignalTagType.EXIT_TAG.value] = None
    return dataframe


def append_candles_to_dataframe(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    Append the `right` DataFrame to the `left` DataFrame.
    """
    if left.iloc[-1]["date"] != right.iloc[-1]["date"]:
        left = pd.concat([left, right])
    left = left[-1500:] if len(left) > 1500 else left
    left.reset_index(drop=True, inplace=True)
    return left
