from os import PathLike
from typing import Union, Sequence, Tuple, List
import torch
from torch import Tensor

OnePath = Union[str, PathLike]
ManyPaths = Sequence[OnePath]
ImagesWithSize = Tuple[Tensor, Tensor]

class ImageLoader(Registrable):
    def __init__(self, *, size_divisibility: int = 0, pad_value: float = 0.0, device: Union[str, torch.device] = 'cpu') -> None:
    def __call__(self, filename_or_filenames: Union[OnePath, ManyPaths]) -> ImagesWithSize:
    def load(self, filename: str) -> Tensor:
    def _pack_image_list(self, images: List[Tensor], sizes: List[Tensor]) -> ImagesWithSize:

@ImageLoader.register('torch')
class TorchImageLoader(ImageLoader):
    def __init__(self, *, image_backend: str = None, resize: bool = True, normalize: bool = True, min_size: int = 800, max_size: int = 1333, pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225), size_divisibility: int = 32, **kwargs) -> None:
    def load(self, filename: str) -> Tensor:
