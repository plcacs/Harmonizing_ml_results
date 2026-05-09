import argparse
import copy
import json
import logging
import math
import os
import re
import shutil
from collections import OrderedDict, Counter
from typing import Optional, List, Dict, Any

@TrainerCallback.register('training_data_logger')
class TrainingDataLoggerOnBatchCallback(TrainerCallback):
    def on_batch(self, 
                 trainer: Trainer, 
                 batch_inputs: List[Dict[str, Any]], 
                 batch_outputs: List[Dict[str, Any]], 
                 batch_metrics: Dict[str, float], 
                 epoch: int, 
                 batch_number: int, 
                 is_training: bool, 
                 is_primary: bool = True, 
                 **kwargs) -> None:
        ...

@TrainerCallback.register('training_device_logger')
class TrainingDeviceLoggerOnBatchCallback(TrainerCallback):
    def on_batch(self, 
                 trainer: Trainer, 
                 batch_inputs: List[Dict[str, Any]], 
                 batch_outputs: List[Dict[str, Any]], 
                 batch_metrics: Dict[str, float], 
                 epoch: int, 
                 batch_number: int, 
                 is_training: bool, 
                 is_primary: bool = True, 
                 **kwargs) -> None:
        ...

@TrainerCallback.register('training_primary_check')
class TrainingPrimaryCheckCallback(TrainerCallback):
    def on_start(self, 
                 trainer: Trainer, 
                 is_primary: bool = True, 
                 **kwargs) -> None:
        ...

class TestTrain(AllenNlpTestCase):
    DEFAULT_PARAMS: Params = Params({'model': {'type': 'simple_tagger', 'text_field_embedder': {'token_embedders': {'tokens': {'type': 'embedding', 'embedding_dim': 5}}}, 'encoder': {'type': 'lstm', 'input_size': 5, 'hidden_size': 7, 'num_layers': 2}}, 'dataset_reader': {'type': 'sequence_tagging'}, 'train_data_path': SEQUENCE_TAGGING_DATA_PATH, 'validation_data_path': SEQUENCE_TAGGING_DATA_PATH, 'data_loader': {'batch_size': 2}, 'trainer': {'num_epochs': 2, 'optimizer': 'adam'}})

    def test_train_model(self) -> None:
        ...
