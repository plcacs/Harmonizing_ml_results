def feature_duplicator(df: pd.DataFrame, columns_to_duplicate: List[str] = None, columns_mapping: Dict[str, str] = None, prefix: str = None, suffix: str = None) -> Union[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
def column_duplicatable(columns_to_bind: str) -> Callable[[Callable], Callable]:
