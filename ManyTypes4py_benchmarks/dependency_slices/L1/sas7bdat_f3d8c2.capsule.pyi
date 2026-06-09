# === Third-party dependency: numpy ===
# Used symbols: array, asarray, dtype, empty, float64, int64, ndarray, uint8, zeros

# === Internal dependency: pandas ===
from pandas.core.api import Timestamp
from pandas.core.api import to_timedelta
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._libs.byteswap ===
def read_float_with_byteswap(data, offset, byteswap): ...
def read_double_with_byteswap(data, offset, byteswap): ...
def read_uint16_with_byteswap(data, offset, byteswap): ...
def read_uint32_with_byteswap(data, offset, byteswap): ...
def read_uint64_with_byteswap(data, offset, byteswap): ...

# === Internal dependency: pandas._libs.sas ===
class Parser:
    def __init__(self, parser): ...
    def read(self, nrows): ...
def get_subheader_index(signature): ...

# === Internal dependency: pandas._libs.tslibs.conversion ===
def cast_from_unit_vectorized(values, unit, out_unit=...): ...

# === Internal dependency: pandas.errors ===
class EmptyDataError(ValueError): ...

# === Internal dependency: pandas.io.common ===
def get_handle(path_or_buf, mode, *, encoding=..., compression=..., memory_map=..., is_text, errors=..., storage_options=...): ...
def get_handle(path_or_buf, mode, *, encoding=..., compression=..., memory_map=..., is_text=..., errors=..., storage_options=...): ...

# === Internal dependency: pandas.io.sas.sas_constants ===
magic = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc2\xea\x81`\xb3\x14\x11\xcf\xbd\x92\x08\x00\t\xc71\x8c\x18\x1f\x10\x11'
align_1_checker_value = b'3'
align_1_offset = 32
align_1_length = 1
u64_byte_checker_value = b'3'
align_2_offset = 35
align_2_length = 1
align_2_value = 4
endianness_offset = 37
endianness_length = 1
encoding_offset = 70
encoding_length = 1
date_created_offset = 164
date_created_length = 8
date_modified_offset = 172
date_modified_length = 8
header_size_offset = 196
header_size_length = 4
page_size_offset = 200
page_size_length = 4
page_bit_offset_x86 = 16
page_bit_offset_x64 = 32
subheader_pointer_length_x86 = 12
subheader_pointer_length_x64 = 24
page_type_offset = 0
page_type_length = 2
block_count_offset = 2
block_count_length = 2
subheader_count_offset = 4
subheader_count_length = 2
page_type_mask = 3840
page_type_mask2 = 61440 | page_type_mask
page_meta_type = 0
page_data_type = 256
page_mix_type = 512
page_amd_type = 1024
page_meta2_type = 16384
page_meta_types = [page_meta_type, page_meta2_type]
subheader_pointers_offset = 8
truncated_subheader_id = 1
compressed_subheader_id = 4
compressed_subheader_type = 1
text_block_size_length = 2
row_length_offset_multiplier = 5
row_count_offset_multiplier = 6
col_count_p1_multiplier = 9
col_count_p2_multiplier = 10
row_count_on_mix_page_offset_multiplier = 15
column_name_pointer_length = 8
column_name_text_subheader_offset = 0
column_name_text_subheader_length = 2
column_name_offset_offset = 2
column_name_offset_length = 2
column_name_length_offset = 4
column_name_length_length = 2
column_data_offset_offset = 8
column_data_length_offset = 8
column_data_length_length = 4
column_type_offset = 14
column_type_length = 1
column_format_text_subheader_index_offset = 22
column_format_text_subheader_index_length = 2
column_format_offset_offset = 24
column_format_offset_length = 2
column_format_length_offset = 26
column_format_length_length = 2
column_label_text_subheader_index_offset = 28
column_label_text_subheader_index_length = 2
column_label_offset_offset = 30
column_label_offset_length = 2
column_label_length_offset = 32
column_label_length_length = 2
rle_compression = b'SASYZCRL'
rdc_compression = b'SASYZCR2'
compression_literals = [rle_compression, rdc_compression]
encoding_names = {20: 'utf-8', 29: 'latin1', 30: 'latin2', 31: 'latin3', 32: 'latin4', 33: 'cyrillic', 34: 'arabic', 35: 'greek', ...}
sas_date_formats = ('DATE', 'DAY', 'DDMMYY', 'DOWNAME', 'JULDAY', 'JULIAN', 'MMDDYY', 'MMYY', ...)
sas_datetime_formats = ('DATETIME', 'DTWKDATX', 'B8601DN', 'B8601DT', 'B8601DX', 'B8601DZ', 'B8601LX', 'E8601DN', ...)

# === Internal dependency: pandas.io.sas.sasreader ===
class SASReader(Iterator['DataFrame'], ABC):
    def read(self, nrows=...): ...
    def close(self): ...