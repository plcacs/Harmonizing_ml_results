    def predict(self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs) -> Tuple[DataFrame, np.ndarray]:
    def encode_class_names(self, data_dictionary: Dict[str, DataFrame], dk: FreqaiDataKitchen, class_names: List[str]) -> None:
    def assert_valid_class_names(target_column: pd.Series, class_names: List[str]) -> None:
    def decode_class_names(self, class_ints: torch.Tensor) -> List[str]:
    def init_class_names_to_index_mapping(self, class_names: List[str]) -> None:
    def convert_label_column_to_int(self, data_dictionary: Dict[str, DataFrame], dk: FreqaiDataKitchen, class_names: List[str]) -> None:
    def get_class_names(self) -> List[str]:
    def train(self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs) -> Any:
