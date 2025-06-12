from os import PathLike
from typing import Union, Sequence, Tuple, List, cast, Optional
import torch
import torchvision
from torch import FloatTensor, IntTensor, Tensor
from allennlp.common.file_utils import cached_path
from allennlp.common.registrable import Registrable

OnePath = Union[str, PathLike]
ManyPaths = Sequence[OnePath]
ImagesWithSize = Tuple[FloatTensor, IntTensor]

class ImageLoader(Registrable):
    default_implementation = 'torch'

    def __init__(
        self,
        *,
        size_divisibility: int = 0,
        pad_value: float = 0.0,
        device: Union[str, torch.device] = 'cpu'
    ) -> None:
        self.size_divisibility = size_divisibility
        self.pad_value = pad_value
        self.device = device

    def __call__(
        self, filename_or_filenames: Union[OnePath, ManyPaths]
    ) -> ImagesWithSize:
        if not isinstance(filename_or_filenames, (list, tuple)):
            image, size = self([filename_or_filenames])
            return (cast(FloatTensor, image.squeeze(0)), cast(IntTensor, size.squeeze(0)))
        images: List[Tensor] = []
        sizes: List[IntTensor] = []
        for filename in filename_or_filenames:
            image = self.load(cached_path(filename)).to(self.device)
            size = cast(IntTensor, torch.tensor([image.shape[-2], image.shape[-1]], dtype=torch.int32, device=self.device))
            images.append(image)
            sizes.append(size)
        return self._pack_image_list(images, sizes)

    def load(self, filename: OnePath) -> FloatTensor:
        raise NotImplementedError()

    def _pack_image_list(
        self, images: List[Tensor], sizes: List[IntTensor]
    ) -> ImagesWithSize:
        size_tensor = torch.stack(sizes)
        max_size = size_tensor.max(0).values
        if self.size_divisibility > 1:
            max_size = (max_size + self.size_divisibility - 1) // self.size_divisibility * self.size_divisibility
        batched_shape = [len(images)] + list(images[0].shape[:-2]) + list(max_size)
        batched_images = images[0].new_full(batched_shape, self.pad_value)
        for image, batch_slice, size in zip(images, batched_images, size_tensor):
            batch_slice[..., :image.shape[-2], :image.shape[-1]].copy_(image)
        return (cast(FloatTensor, batched_images), cast(IntTensor, size_tensor)

@ImageLoader.register('torch')
class TorchImageLoader(ImageLoader):
    def __init__(
        self,
        *,
        image_backend: Optional[str] = None,
        resize: bool = True,
        normalize: bool = True,
        min_size: int = 800,
        max_size: int = 1333,
        pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        size_divisibility: int = 32,
        **kwargs
    ) -> None:
        super().__init__(size_divisibility=size_divisibility, **kwargs)
        if image_backend is not None:
            torchvision.set_image_backend(image_backend)
        self.resize = resize
        self.normalize = normalize
        self.min_size = min_size
        self.max_size = max_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def load(self, filename: OnePath) -> FloatTensor:
        image = torchvision.io.read_image(filename).float().to(self.device) / 256
        if self.normalize:
            mean = torch.as_tensor(self.pixel_mean, dtype=image.dtype, device=self.device).view(-1, 1, 1)
            std = torch.as_tensor(self.pixel_std, dtype=image.dtype, device=self.device).view(-1, 1, 1)
            image = (image - mean) / std
        if self.resize:
            min_size = min(image.shape[-2:])
            max_size = max(image.shape[-2:])
            scale_factor = self.min_size / min_size
            if max_size * scale_factor > self.max_size:
                scale_factor = self.max_size / max_size
            image = torch.nn.functional.interpolate(image[None], scale_factor=scale_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)[0]
        return image
