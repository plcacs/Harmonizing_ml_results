from io import BytesIO, StringIO
import pytest
import pandas as pd
import pandas._testing as tm
from typing import Union

def test_compression_roundtrip(compression: Union[str, None]) -> None:
    df: pd.DataFrame = pd.DataFrame([[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z'])
    ...

def test_read_zipped_json(datapath: callable) -> None:
    ...

def test_with_s3_url(compression: Union[str, None], s3_public_bucket, s3so) -> None:
    ...

def test_lines_with_compression(compression: Union[str, None]) -> None:
    ...

def test_chunksize_with_compression(compression: Union[str, None]) -> None:
    ...

def test_write_unsupported_compression_type() -> None:
    ...

def test_read_unsupported_compression_type() -> None:
    ...

def test_to_json_compression(compression_only: str, read_infer: bool, to_infer: bool, compression_to_extension: dict, infer_string: bool) -> None:
    ...

def test_to_json_compression_mode(compression: Union[str, None]) -> None:
    ...
