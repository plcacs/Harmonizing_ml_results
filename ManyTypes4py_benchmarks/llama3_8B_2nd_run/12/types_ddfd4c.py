        from typing import Annotated

        from pydantic.types import Strict

        StrictBool = Annotated[bool, Strict()]
        