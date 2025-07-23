            import pydantic

            class MyModel(pydantic.BaseModel):
                foo: int = pydantic.Field(4)
            