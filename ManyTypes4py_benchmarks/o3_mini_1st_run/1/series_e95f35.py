from typing import Any, Union
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from databricks.koalas.frame import DataFrame
from databricks.koalas.series import Series

def unpack_scalar(sdf: SparkDataFrame) -> Any:
    """
    Takes a dataframe that is supposed to contain a single row with a single scalar value,
    and returns this value.
    """
    l: pd.DataFrame = sdf.limit(2).toPandas()
    assert len(l) == 1, (sdf, l)
    row: pd.Series = l.iloc[0]
    l2: list[Any] = list(row)
    assert len(l2) == 1, (row, l2)
    return l2[0]


def first_series(df: Union[DataFrame, pd.DataFrame]) -> Series:
    """
    Takes a DataFrame and returns the first column of the DataFrame as a Series.
    """
    assert isinstance(df, (DataFrame, pd.DataFrame)), type(df)
    if isinstance(df, DataFrame):
        return df._kser_for(df._internal.column_labels[0])
    else:
        return df[df.columns[0]]