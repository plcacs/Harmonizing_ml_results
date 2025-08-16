def boxplot(df: DataFrame, groupby: Union[str, List[str]], metrics: List[str], whisker_type: PostProcessingBoxplotWhiskerType, percentiles: Optional[Tuple[Union[int, float], Union[int, float]]] = None) -> DataFrame:
    def quartile1(series: Series) -> float:
    def quartile3(series: Series) -> float:
    def whisker_high(series: Series) -> Any:
    def whisker_low(series: Series) -> Any:
    def outliers(series: Series) -> List[Any]:
