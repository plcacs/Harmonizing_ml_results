# === Third-party dependency: numpy ===
# Used symbols: array, asarray, dtype, empty, float64, int64, ndarray, uint8, zeros

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import to_timedelta
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._libs.byteswap ===
def read_float_with_byteswap(data: bytes, offset: int, byteswap: bool) -> float: ...
def read_double_with_byteswap(data: bytes, offset: int, byteswap: bool) -> float: ...
def read_uint16_with_byteswap(data: bytes, offset: int, byteswap: bool) -> int: ...
def read_uint32_with_byteswap(data: bytes, offset: int, byteswap: bool) -> int: ...
def read_uint64_with_byteswap(data: bytes, offset: int, byteswap: bool) -> int: ...

# === Internal dependency: pandas._libs.sas ===
class Parser:
    def __init__(self, parser: SAS7BDATReader) -> None: ...
    def read(self, nrows: int) -> None: ...
def get_subheader_index(signature: bytes) -> int: ...

# === Internal dependency: pandas._libs.tslibs.conversion ===
def cast_from_unit_vectorized(values: np.ndarray, unit: str, out_unit: str = ...) -> np.ndarray: ...

# === Internal dependency: pandas.errors ===
class EmptyDataError(ValueError): ...

# === Internal dependency: pandas.io.common ===
def get_handle(path_or_buf: FilePath | BaseBuffer, mode: str, *, encoding: str | None = ..., compression: CompressionOptions = ..., memory_map: bool = ..., is_text: Literal[False], errors: str | None = ..., storage_options: StorageOptions = ...) -> IOHandles[bytes]: ...
def get_handle(path_or_buf: FilePath | BaseBuffer, mode: str, *, encoding: str | None = ..., compression: CompressionOptions = ..., memory_map: bool = ..., is_text: Literal[True] = ..., errors: str | None = ..., storage_options: StorageOptions = ...) -> IOHandles[str]: ...
def get_handle(path_or_buf: FilePath | BaseBuffer, mode: str, *, encoding: str | None = ..., compression: CompressionOptions = ..., memory_map: bool = ..., is_text: bool = ..., errors: str | None = ..., storage_options: StorageOptions = ...) -> IOHandles[str] | IOHandles[bytes]: ...
def get_handle(path_or_buf: FilePath | BaseBuffer, mode: str, *, encoding: str | None = ..., compression: CompressionOptions | None = ..., memory_map: bool = ..., is_text: bool = ..., errors: str | None = ..., storage_options: StorageOptions | None = ...) -> IOHandles[str] | IOHandles[bytes]: ...

# === Internal dependency: pandas.io.sas.sas_constants ===
magic: Final
align_1_checker_value: Final
align_1_offset: Final
align_1_length: Final
u64_byte_checker_value: Final
align_2_offset: Final
align_2_length: Final
align_2_value: Final
endianness_offset: Final
endianness_length: Final
encoding_offset: Final
encoding_length: Final
date_created_offset: Final
date_created_length: Final
date_modified_offset: Final
date_modified_length: Final
header_size_offset: Final
header_size_length: Final
page_size_offset: Final
page_size_length: Final
page_bit_offset_x86: Final
page_bit_offset_x64: Final
subheader_pointer_length_x86: Final
subheader_pointer_length_x64: Final
page_type_offset: Final
page_type_length: Final
block_count_offset: Final
block_count_length: Final
subheader_count_offset: Final
subheader_count_length: Final
page_type_mask2: Final
page_data_type: Final
page_mix_type: Final
page_amd_type: Final
page_meta_types: Final
subheader_pointers_offset: Final
truncated_subheader_id: Final
compressed_subheader_id: Final
compressed_subheader_type: Final
text_block_size_length: Final
row_length_offset_multiplier: Final
row_count_offset_multiplier: Final
col_count_p1_multiplier: Final
col_count_p2_multiplier: Final
row_count_on_mix_page_offset_multiplier: Final
column_name_pointer_length: Final
column_name_text_subheader_offset: Final
column_name_text_subheader_length: Final
column_name_offset_offset: Final
column_name_offset_length: Final
column_name_length_offset: Final
column_name_length_length: Final
column_data_offset_offset: Final
column_data_length_offset: Final
column_data_length_length: Final
column_type_offset: Final
column_type_length: Final
column_format_text_subheader_index_offset: Final
column_format_text_subheader_index_length: Final
column_format_offset_offset: Final
column_format_offset_length: Final
column_format_length_offset: Final
column_format_length_length: Final
column_label_text_subheader_index_offset: Final
column_label_text_subheader_index_length: Final
column_label_offset_offset: Final
column_label_offset_length: Final
column_label_length_offset: Final
column_label_length_length: Final
rle_compression: Final
compression_literals: Final
encoding_names: Final
sas_date_formats: Final
sas_datetime_formats: Final

# === Internal dependency: pandas.io.sas.sasreader ===
class SASReader(Iterator['DataFrame'], ABC):
    def read(self, nrows: int | None = ...) -> DataFrame: ...
    def close(self) -> None: ...