from typing import TypedDict

class BuildDockerImageResult(TypedDict):
    image_name: str
    tag: str
    image: str
    image_id: str
    additional_tags: list

class PushDockerImageResult(TypedDict):
    image_name: str
    tag: str
    image: str
    additional_tags: list
