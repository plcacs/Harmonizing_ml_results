    def __init__(self, series: pd.Series) -> None:
    def categories(self) -> pd.Index:
    def ordered(self) -> bool:
    def codes(self) -> pd.Series:
    def add_categories(self, new_categories: list, inplace: bool = False) -> None:
    def as_ordered(self, inplace: bool = False) -> None:
    def as_unordered(self, inplace: bool = False) -> None:
    def remove_categories(self, removals: list, inplace: bool = False) -> None:
    def remove_unused_categories(self) -> None:
    def rename_categories(self, new_categories: list, inplace: bool = False) -> None:
    def reorder_categories(self, new_categories: list, ordered: bool = None, inplace: bool = False) -> None:
    def set_categories(self, new_categories: list, ordered: bool = None, rename: bool = False, inplace: bool = False) -> None:
