import copy
import pytest
import torch
from torch.testing import assert_allclose
from transformers import AutoModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.albert.configuration_albert import AlbertConfig
from transformers.models.albert.modeling_albert import AlbertEmbeddings
from allennlp.common import Params, FromParams
from allennlp.modules.transformer import TransformerEmbeddings, ImageFeatureEmbeddings, TransformerModule

PARAMS_DICT: dict = {'vocab_size': int, 'embedding_size': int, 'pad_token_id': int, 'max_position_embeddings': int, 'type_vocab_size': int, 'dropout': float}

@pytest.fixture
def params_dict() -> dict:
    return copy.deepcopy(PARAMS_DICT)

@pytest.fixture
def params(params_dict: dict) -> Params:
    return Params(params_dict)

@pytest.fixture
def transformer_embeddings(params: Params) -> TransformerEmbeddings:
    return TransformerEmbeddings.from_params(params.duplicate())

def test_can_construct_from_params(params_dict: dict, transformer_embeddings: TransformerEmbeddings) -> None:
    embeddings: torch.nn.Embedding = transformer_embeddings.embeddings
    assert embeddings.word_embeddings.num_embeddings == params_dict['vocab_size']
    assert embeddings.word_embeddings.embedding_dim == params_dict['embedding_size']
    assert embeddings.word_embeddings.padding_idx == params_dict['pad_token_id']
    assert embeddings.position_embeddings.num_embeddings == params_dict['max_position_embeddings']
    assert embeddings.position_embeddings.embedding_dim == params_dict['embedding_size']
    assert embeddings.token_type_embeddings.num_embeddings == params_dict['type_vocab_size']
    assert embeddings.token_type_embeddings.embedding_dim == params_dict['embedding_size']
    assert transformer_embeddings.layer_norm.normalized_shape[0] == params_dict['embedding_size']
    assert transformer_embeddings.dropout.p == params_dict['dropout']

def test_sanity() -> None:
    # ...

def test_forward_runs_with_inputs(transformer_embeddings: TransformerEmbeddings) -> None:
    # ...

def test_output_size(params: Params) -> None:
    # ...

def test_no_token_type_layer(params: Params) -> None:
    # ...

@pytest.mark.parametrize('pretrained_name', ['bert-base-cased', 'epwalsh/bert-xsmall-dummy'])
def test_loading_from_pretrained_module(pretrained_name: str) -> None:
    # ...

def test_loading_albert() -> None:
    # ...

def get_modules() -> tuple:
    # ...

@pytest.mark.parametrize('module_name, hf_module', get_modules())
def test_forward_against_huggingface_output(transformer_embeddings: TransformerEmbeddings, module_name: str, hf_module: BertEmbeddings) -> None:
    # ...

@pytest.fixture
def image_params_dict() -> dict:
    return {'feature_size': int, 'embedding_size': int, 'dropout': float}

@pytest.fixture
def image_params(image_params_dict: dict) -> Params:
    return Params(image_params_dict)

@pytest.fixture
def image_embeddings(image_params: Params) -> ImageFeatureEmbeddings:
    return ImageFeatureEmbeddings.from_params(image_params.duplicate())

def test_can_construct_image_embeddings_from_params(image_embeddings: ImageFeatureEmbeddings, image_params_dict: dict) -> None:
    # ...

def test_image_embedding_forward_runs_with_inputs(image_embeddings: ImageFeatureEmbeddings, image_params_dict: dict) -> None:
    # ...

def test_image_embeddings_sanity(image_params_dict: dict) -> None:
    # ...
