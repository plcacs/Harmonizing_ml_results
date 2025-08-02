            from pydantic import BaseModel

            class MyModel(BaseModel, extra='allow'): ...
            