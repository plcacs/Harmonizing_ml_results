import argparse
import csv
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
from typing import List, Optional, Dict, Any, Union

import pytest
from allennlp.commands.predict import Predict
from allennlp.common.testing import AllenNlpTestCase
from allennlp.predictors import Predictor
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model

class TestPredict(AllenNlpTestCase):
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

    def test_other_modules(self) -> None: ...

    def test_alternative_file_formats(self) -> None: ...