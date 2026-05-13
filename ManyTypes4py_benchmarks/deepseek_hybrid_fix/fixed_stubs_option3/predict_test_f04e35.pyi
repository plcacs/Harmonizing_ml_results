import argparse
import csv
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import pytest
from allennlp.commands import main
from allennlp.commands.predict import Predict
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import JsonDict, push_python_path
from allennlp.data.dataset_readers import DatasetReader, TextClassificationJsonReader
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor, TextClassifierPredictor
from typing import Any, Dict, List, Optional, Union

class TestPredict(AllenNlpTestCase):
    classifier_model_path: pathlib.Path
    classifier_data_path: pathlib.Path
    tempdir: pathlib.Path
    infile: pathlib.Path
    outfile: pathlib.Path

    def setup_method(self) -> None: ...
    def test_add_predict_subparser(self) -> None: ...
    def test_works_with_known_model(self) -> None: ...
    def test_using_dataset_reader_works_with_known_model(self) -> None: ...
    def test_uses_correct_dataset_reader(self) -> None: ...
    def test_base_predictor(self) -> None: ...
    def test_batch_prediction_works_with_known_model(self) -> None: ...
    def test_fails_without_required_args(self) -> None: ...
    def test_can_specify_predictor(self) -> None: ...
    def test_can_specify_extra_args(self) -> None: ...
    def test_other_modules(self) -> None:
        self.infile: str
        self.outfile: str
        ...
    def test_alternative_file_formats(self) -> None: ...