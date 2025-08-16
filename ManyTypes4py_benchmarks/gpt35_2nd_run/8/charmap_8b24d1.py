from typing import TYPE_CHECKING, Literal, Optional, Iterable, TypeAlias

CategoryName = Literal['L', 'Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'M', 'Mn', 'Mc', 'Me', 'N', 'Nd', 'Nl', 'No', 'P', 'Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po', 'S', 'Sm', 'Sc', 'Sk', 'So', 'Z', 'Zs', 'Zl', 'Zp', 'C', 'Cc', 'Cf', 'Cs', 'Co', 'Cn']
Categories = Iterable[CategoryName]
CategoriesTuple = tuple[CategoryName, ...]

def charmap_file(fname: str = 'charmap') -> str:
    return storage_directory('unicode_data', unicodedata.unidata_version, f'{fname}.json.gz')

def charmap() -> dict[CategoryName, tuple[tuple[int, int]]]:
    ...

def intervals_from_codec(codec_name: str) -> IntervalSet:
    ...

def categories() -> CategoriesTuple:
    ...

def as_general_categories(cats: list[CategoryName], name: str = 'cats') -> CategoriesTuple:
    ...

def _category_key(cats: Optional[set[CategoryName]]) -> tuple[CategoryName]:
    ...

def _query_for_key(key: tuple[CategoryName]) -> tuple[tuple[int, int]]:
    ...

def query(*, categories: Optional[list[CategoryName]] = None, min_codepoint: Optional[int] = None, max_codepoint: Optional[int] = None, include_characters: str = '', exclude_characters: str = '') -> tuple[tuple[int, int]]:
    ...
