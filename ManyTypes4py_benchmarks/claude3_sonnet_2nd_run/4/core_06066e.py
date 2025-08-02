import itertools
import os
from pathlib import Path
import cv2
import nevergrad as ng
import nevergrad.common.typing as tp
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tr
from nevergrad.common import errors
from torchvision.models import resnet50
from .. import base
from . import imagelosses


class Image(base.ExperimentFunction):

    def __init__(self, problem_name: str = 'recovering', index: int = 0, 
                 loss: tp.Type[imagelosses.ImageLoss] = imagelosses.SumAbsoluteDifferences, 
                 with_pgan: bool = False, num_images: int = 1) -> None:
        """
        problem_name: the type of problem we are working on.
           recovering: we directly try to recover the target image.§
        index: the index of the problem, inside the problem type.
           For example, if problem_name is "recovering" and index == 0,
           we try to recover the face of O. Teytaud.
        """
        self.domain_shape: tp.Tuple[int, int, int] = (226, 226, 3)
        self.problem_name: str = problem_name
        self.index: int = index
        self.with_pgan: bool = with_pgan
        self.num_images: int = num_images
        assert problem_name == 'recovering'
        assert index == 0
        path: Path = Path(__file__).with_name('headrgb_olivier.png')
        image: PIL.Image.Image = PIL.Image.open(path).resize((self.domain_shape[0], self.domain_shape[1]))
        self.data: np.ndarray = np.asarray(image)[:, :, :3]
        if not with_pgan:
            assert num_images == 1
            array: ng.p.Array = ng.p.Array(init=128 * np.ones(self.domain_shape), mutable_sigma=True)
            array.set_mutation(sigma=35)
            array.set_bounds(lower=0, upper=255.99, method='clipping', full_range_sampling=True)
            max_size: ng.p.Scalar = ng.p.Scalar(lower=1, upper=200).set_integer_casting()
            array = ng.ops.mutations.Crossover(axis=(0, 1), max_size=max_size)(array).set_name('')
            super().__init__(loss(reference=self.data), array)
        else:
            self.pgan_model: torch.nn.Module = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', model_name='celebAHQ-512', pretrained=True, useGPU=False)
            self.domain_shape = (num_images, 512)
            initial_noise: np.ndarray = np.random.normal(size=self.domain_shape)
            self.initial: np.ndarray = np.random.normal(size=(1, 512))
            self.target: np.ndarray = np.random.normal(size=(1, 512))
            array = ng.p.Array(init=initial_noise, mutable_sigma=True)
            array.set_mutation(sigma=35.0)
            array = ng.ops.mutations.Crossover(axis=(0, 1))(array).set_name('')
            self._descriptors.pop('use_gpu', None)
            super().__init__(self._loss_with_pgan, array)
        assert self.multiobjective_upper_bounds is None
        self.add_descriptors(loss=loss.__class__.__name__)
        self.loss_function: imagelosses.ImageLoss = loss(reference=self.data)

    def _generate_images(self, x: np.ndarray) -> np.ndarray:
        """Generates images tensor of shape [nb_images, x, y, 3] with pixels between 0 and 255"""
        noise: torch.Tensor = torch.tensor(x.astype('float32'))
        return ((self.pgan_model.test(noise).clamp(min=-1, max=1) + 1) * 255.99 / 2).permute(0, 2, 3, 1).cpu().numpy()[:, :, :, [2, 1, 0]]

    @staticmethod
    def interpolate(base_image: np.ndarray, target: np.ndarray, k: int, num_images: int) -> np.ndarray:
        if num_images == 1:
            return target
        coef1: float = k / (num_images - 1)
        coef2: float = (num_images - 1 - k) / (num_images - 1)
        return coef1 * base_image + coef2 * target

    def _loss_with_pgan(self, x: np.ndarray, export_string: str = '') -> float:
        loss: float = 0.0
        factor: int = 1 if self.num_images < 2 else 10
        num_total_images: int = factor * self.num_images
        for i in range(num_total_images):
            base_i: int = i // factor
            base_image: np.ndarray = self.interpolate(self.initial, self.target, i, num_total_images)
            movability: float = 0.5
            if self.num_images > 1:
                movability = 4 * (0.25 - (i / (num_total_images - 1) - 0.5) ** 2)
            moving: np.ndarray = movability * np.sqrt(self.dimension) * np.expand_dims(x[base_i], 0) / (1e-10 + np.linalg.norm(x[base_i]))
            base_image = moving if self.num_images == 1 else base_image + moving
            image: np.ndarray = self._generate_images(base_image).squeeze(0)
            image = cv2.resize(image, dsize=(226, 226), interpolation=cv2.INTER_NEAREST)
            if export_string:
                cv2.imwrite(f'{export_string}_image{i}_{num_total_images}_{self.num_images}.jpg', image)
            assert image.shape == (226, 226, 3), f'{x.shape} != {(226, 226, 3)}'
            loss += self.loss_function(image)
        return loss

    def export_to_images(self, x: np.ndarray, export_string: str = 'export') -> None:
        self._loss_with_pgan(x, export_string=export_string)


class Normalize(nn.Module):

    def __init__(self, mean: tp.List[float], std: tp.List[float]) -> None:
        super().__init__()
        self.mean: torch.Tensor = torch.Tensor(mean)
        self.std: torch.Tensor = torch.Tensor(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class Resnet50(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.norm: Normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model: nn.Module = resnet50(pretrained=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.norm(x))


class TestClassifier(nn.Module):

    def __init__(self, image_size: int = 224) -> None:
        super().__init__()
        self.model: nn.Linear = nn.Linear(image_size * image_size * 3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.view(x.shape[0], -1))


class ImageAdversarial(base.ExperimentFunction):

    def __init__(self, classifier: nn.Module, image: torch.Tensor, label: int = 0, targeted: bool = False, epsilon: float = 0.05) -> None:
        """
        params : needs to be detailed
        """
        self.targeted: bool = targeted
        self.epsilon: float = epsilon
        self.image: torch.Tensor = image
        self.label: torch.Tensor = torch.Tensor([label])
        self.label = self.label.long()
        self.classifier: nn.Module = classifier
        self.classifier.eval()
        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.imsize: int = self.image.shape[1]
        array: ng.p.Array = ng.p.Array(init=np.zeros(self.image.shape), mutable_sigma=True).set_name('')
        array.set_mutation(sigma=self.epsilon / 10)
        array.set_bounds(lower=-self.epsilon, upper=self.epsilon, method='clipping', full_range_sampling=True)
        max_size: ng.p.Scalar = ng.p.Scalar(lower=1, upper=200).set_integer_casting()
        array = ng.p.mutation.Crossover(axis=(1, 2), max_size=max_size)(array)
        super().__init__(self._loss, array)
        self._descriptors.pop('label')

    def _loss(self, x: np.ndarray) -> float:
        output_adv: torch.Tensor = self._get_classifier_output(x)
        value: float = float(self.criterion(output_adv, self.label).item())
        return value * (1.0 if self.targeted else -1.0)

    def _get_classifier_output(self, x: np.ndarray) -> torch.Tensor:
        y: torch.Tensor = torch.Tensor(x)
        image_adv: torch.Tensor = torch.clamp(self.image + y, 0, 1)
        image_adv = image_adv.view(1, 3, self.imsize, self.imsize)
        return self.classifier(image_adv)

    def evaluation_function(self, *recommendations: tp.Any) -> float:
        """Returns wether the attack worked or not"""
        assert len(recommendations) == 1, 'Should not be a pareto set for a singleobjective function'
        x: np.ndarray = recommendations[0].value
        output_adv: torch.Tensor = self._get_classifier_output(x)
        _, pred = torch.max(output_adv, axis=1)
        actual: int = int(self.label)
        return float(pred != actual if self.targeted else pred == actual)

    @classmethod
    def make_folder_functions(cls, folder: tp.Optional[str], model: str = 'resnet50') -> tp.Iterator[base.ExperimentFunction]:
        """

        Parameters
        ----------
        folder: str or None
            folder to use for reference images. If None, 1 random image is created.
        model: str
            model name to use

        Yields
        ------
        ExperimentFunction
            an experiment function corresponding to 1 of the image of the provided folder dataset.
        """
        assert model in {'resnet50', 'test'}
        tags: tp.Dict[str, str] = {'folder': '#FAKE#' if folder is None else Path(folder).name, 'model': model}
        classifier: nn.Module = Resnet50() if model == 'resnet50' else TestClassifier()
        classifier.eval()
        transform: tr.Compose = tr.Compose([tr.Resize(256), tr.CenterCrop(224), tr.ToTensor()])
        if folder is None:
            x: torch.Tensor = torch.zeros(1, 3, 224, 224)
            _, pred = torch.max(classifier(x), axis=1)
            data_loader: tp.List[tp.Tuple[torch.Tensor, torch.Tensor]] = [(x, pred)]
        elif Path(folder).is_dir():
            ifolder: torchvision.datasets.ImageFolder = torchvision.datasets.ImageFolder(folder, transform=transform)
            data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(ifolder, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        else:
            raise ValueError(f'{folder} is not a valid folder.')
        for data, target in itertools.islice(data_loader, 0, 100):
            _, pred = torch.max(classifier(data), axis=1)
            if pred == target:
                func: ImageAdversarial = cls(classifier=classifier, image=data[0], label=int(target), targeted=False, epsilon=0.05)
                func.add_descriptors(**tags)
                yield func


class ImageFromPGAN(base.ExperimentFunction):
    """
    Creates face images using a GAN from pytorch GAN zoo trained on celebAHQ and optimizes the noise vector of the GAN

    Parameters
    ----------
    problem_name: str
        the type of problem we are working on.
    initial_noise: np.ndarray
        the initial noise of the GAN. It should be of dimension (1, 512). If None, it is defined randomly.
    use_gpu: bool
        whether to use gpus to compute the images
    loss: ImageLoss
        which loss to use for the images (default: Koncept512)
    mutable_sigma: bool
        whether the sigma should be mutable
    sigma: float
        standard deviation of the initial mutations
    """

    def __init__(self, initial_noise: tp.Optional[np.ndarray] = None, use_gpu: bool = False, 
                 loss: tp.Optional[imagelosses.ImageLoss] = None, mutable_sigma: bool = True, sigma: float = 35) -> None:
        if loss is None:
            loss = imagelosses.Koncept512()
        if not torch.cuda.is_available():
            use_gpu = False
        CI: bool = bool(os.environ.get('CIRCLECI', False)) or bool(os.environ.get('CI', False))
        if CI:
            raise errors.UnsupportedExperiment('ImageFromPGAN is not well supported in CircleCI')
        self.pgan_model: torch.nn.Module = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', model_name='celebAHQ-512', pretrained=True, useGPU=use_gpu)
        self.domain_shape: tp.Tuple[int, int] = (1, 512)
        if initial_noise is None:
            initial_noise = np.random.normal(size=self.domain_shape)
        assert initial_noise.shape == self.domain_shape, f'The shape of the initial noise vector was {initial_noise.shape}, it should be {self.domain_shape}'
        array: ng.p.Array = ng.p.Array(init=initial_noise, mutable_sigma=mutable_sigma)
        array.set_mutation(sigma=sigma)
        array = ng.ops.mutations.Crossover(axis=(0, 1))(array).set_name('')
        super().__init__(self._loss, array)
        self.loss_function: imagelosses.ImageLoss = loss
        self._descriptors.pop('use_gpu', None)
        self.add_descriptors(loss=loss.__class__.__name__)

    def _loss(self, x: np.ndarray) -> float:
        image: np.ndarray = self._generate_images(x)
        loss: float = self.loss_function(image)
        return loss

    def _generate_images(self, x: np.ndarray) -> np.ndarray:
        """Generates images tensor of shape [nb_images, x, y, 3] with pixels between 0 and 255"""
        noise: torch.Tensor = torch.tensor(x.astype('float32'))
        return ((self.pgan_model.test(noise).clamp(min=-1, max=1) + 1) * 255.99 / 2).permute(0, 2, 3, 1).cpu().numpy()
