        from typing import List, Union

        from pydantic._internal._generics import replace_types

        replace_types(tuple[str, Union[List[str], float]], {str: int})
        #> tuple[int, Union[List[int], float]]
        