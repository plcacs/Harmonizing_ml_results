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

    def __init__(self, problem_name: str = 'recovering', index: int = 0, loss: tp.Type[imagelosses.SumAbsoluteDifferences], with_pgan: bool = False, num_images: int = 1):
        ...

    def _generate_images(self, x: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    def interpolate(base_image: np.ndarray, target: np.ndarray, k: int, num_images: int) -> np.ndarray:
        ...

    def _loss_with_pgan(self, x: np.ndarray, export_string: str = '') -> float:
        ...

    def export_to_images(self, x: np.ndarray, export_string: str = 'export'):
        ...

class Normalize(nn.Module):

    def __init__(self, mean: list, std: list):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

class Resnet50(nn.Module):

    def __init__(self):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

class TestClassifier(nn.Module):

    def __init__(self, image_size: int = 224):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

class ImageAdversarial(base.ExperimentFunction):

    def __init__(self, classifier: nn.Module, image: np.ndarray, label: int = 0, targeted: bool = False, epsilon: float = 0.05):
        ...

    def _loss(self, x: np.ndarray) -> float:
        ...

    def _get_classifier_output(self, x: np.ndarray) -> torch.Tensor:
        ...

    def evaluation_function(self, *recommendations) -> float:
        ...

    @classmethod
    def make_folder_functions(cls, folder: str, model: str = 'resnet50'):
        ...

class ImageFromPGAN(base.ExperimentFunction):

    def __init__(self, initial_noise: np.ndarray = None, use_gpu: bool = False, loss: tp.Type[imagelosses.ImageLoss] = None, mutable_sigma: bool = True, sigma: float = 35):
        ...

    def _loss(self, x: np.ndarray) -> float:
        ...

    def _generate_images(self, x: np.ndarray) -> np.ndarray:
        ...
