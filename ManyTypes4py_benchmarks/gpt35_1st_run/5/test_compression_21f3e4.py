from typing import List, Tuple, Union

def test_zip(parser_and_data: Tuple, compression: str) -> None:
def test_zip_error_multiple_files(parser_and_data: Tuple, compression: str) -> None:
def test_zip_error_no_files(parser_and_data: Tuple) -> None:
def test_zip_error_invalid_zip(parser_and_data: Tuple) -> None:
def test_compression(request, parser_and_data: Tuple, compression_only: str, buffer: bool, filename: Union[None, str], compression_to_extension: dict) -> None:
def test_infer_compression(all_parsers: List, csv1: str, buffer: bool, ext: Union[None, str]) -> None:
def test_compression_utf_encoding(all_parsers: List, csv_dir_path: str, utf_value: int, encoding_fmt: str) -> None:
def test_invalid_compression(all_parsers: List, invalid_compression: str) -> None:
def test_compression_tar_archive(all_parsers: List, csv_dir_path: str) -> None:
def test_ignore_compression_extension(all_parsers: List) -> None:
def test_writes_tar_gz(all_parsers: List) -> None:
