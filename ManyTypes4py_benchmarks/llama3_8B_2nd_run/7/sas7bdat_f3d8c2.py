class _Column:
    def __init__(self, col_id: int, name: str, label: str, format: str, ctype: str, length: int) -> None:
        self.col_id: int = col_id
        self.name: str = name
        self.label: str = label
        self.format: str = format
        self.ctype: str = ctype
        self.length: int = length

class SAS7BDATReader:
    def __init__(self, path_or_buf: Union[pathlib.Path, str, bytes, TextIO], 
                 index: Optional[str] = None, 
                 convert_dates: bool = True, 
                 blank_missing: bool = True, 
                 chunksize: Optional[int] = None, 
                 encoding: Optional[str] = None, 
                 convert_text: bool = True, 
                 convert_header_text: bool = True, 
                 compression: str = 'infer') -> None:
        # ... (rest of the method remains the same)

    def column_data_lengths(self) -> np.ndarray:
        return np.asarray(self._column_data_lengths, dtype=np.int64)

    def column_data_offsets(self) -> np.ndarray:
        return np.asarray(self._column_data_offsets, dtype=np.int64)

    def column_types(self) -> np.ndarray:
        return np.asarray(self._column_types, dtype=np.dtype('S1'))

    def _read_float(self, offset: int, width: int) -> float:
        # ... (rest of the method remains the same)

    def _read_uint(self, offset: int, width: int) -> int:
        # ... (rest of the method remains the same)

    def _process_rowsize_subheader(self, offset: int, length: int) -> None:
        # ... (rest of the method remains the same)

    def _process_columnsize_subheader(self, offset: int, length: int) -> None:
        # ... (rest of the method remains the same)

    def _process_subheader_counts(self, offset: int, length: int) -> None:
        pass

    def _process_columntext_subheader(self, offset: int, length: int) -> None:
        # ... (rest of the method remains the same)

    def _process_columnname_subheader(self, offset: int, length: int) -> None:
        # ... (rest of the method remains the same)

    def _process_columnattributes_subheader(self, offset: int, length: int) -> None:
        # ... (rest of the method remains the same)

    def _process_columnlist_subheader(self, offset: int, length: int) -> None:
        pass

    def _process_format_subheader(self, offset: int, length: int) -> None:
        # ... (rest of the method remains the same)

    def read(self, nrows: Optional[int] = None) -> DataFrame:
        # ... (rest of the method remains the same)

    def _chunk_to_dataframe(self) -> DataFrame:
        # ... (rest of the method remains the same)

    def _decode_string(self, b: bytes) -> str:
        return b.decode(self.encoding or self.default_encoding)

    def _convert_header_text(self, b: bytes) -> str:
        if self.convert_header_text:
            return self._decode_string(b)
        else:
            return b
