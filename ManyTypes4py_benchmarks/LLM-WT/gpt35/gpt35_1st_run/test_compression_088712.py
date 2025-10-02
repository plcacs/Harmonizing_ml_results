import gzip
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap
import time
import zipfile
from typing import Any, Dict, Union
import numpy as np
import pytest
import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal
from pandas.compat import is_platform_windows
import pandas._testing as tm
import pandas.io.common as icom

CompressionOptions = Union[str, Dict[str, Union[str, int]]]

def test_compression_size(obj: Any, method: str, compression_only: CompressionOptions):
    ...

def test_compression_size_fh(obj: Any, method: str, compression_only: CompressionOptions):
    ...

def test_dataframe_compression_defaults_to_infer(write_method: str, write_kwargs: Dict[str, Any], read_method: Any, compression_only: CompressionOptions, compression_to_extension: Dict[CompressionOptions, str]):
    ...

def test_series_compression_defaults_to_infer(write_method: str, write_kwargs: Dict[str, Any], read_method: Any, read_kwargs: Dict[str, Any], compression_only: CompressionOptions, compression_to_extension: Dict[CompressionOptions, str]):
    ...

def test_compression_warning(compression_only: CompressionOptions):
    ...

def test_compression_binary(compression_only: CompressionOptions):
    ...

def test_gzip_reproducibility_file_name():
    ...

def test_gzip_reproducibility_file_object():
    ...

def test_with_missing_lzma():
    ...

def test_with_missing_lzma_runtime():
    ...

def test_gzip_compression_level(obj: Any, method: str):
    ...

def test_xz_compression_level_read(obj: Any, method: str):
    ...

def test_bzip_compression_level(obj: Any, method: str):
    ...

def test_empty_archive_zip(suffix: str, archive: Any):
    ...

def test_ambiguous_archive_zip():
    ...

def test_ambiguous_archive_tar(tmp_path: Path):
    ...

def test_tar_gz_to_different_filename():
    ...

def test_tar_no_error_on_close():
    ...
