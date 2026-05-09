            class Custom(Block):
                message: str

            Custom(message="Hello!").save("my-custom-message")

            loaded_block = await Custom.aload("my-custom-message")
            