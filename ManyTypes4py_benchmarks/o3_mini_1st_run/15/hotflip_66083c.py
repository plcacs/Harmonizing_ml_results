from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import numpy
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, TokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.interpret.attackers import utils
from allennlp.interpret.attackers.attacker import Attacker
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.predictors.predictor import Predictor

DEFAULT_IGNORE_TOKENS: List[str] = ['@@NULL@@', '.', ',', ';', '!', '?', '[MASK]', '[SEP]', '[CLS]']

@Attacker.register('hotflip')
class Hotflip(Attacker):
    """
    Runs the HotFlip style attack at the word-level https://arxiv.org/abs/1712.06751.  We use the
    first-order taylor approximation described in https://arxiv.org/abs/1903.06620, in the function
    `_first_order_taylor()`.

    We try to re-use the embedding matrix from the model when deciding what other words to flip a
    token to.  For a large class of models, this is straightforward.  When there is a
    character-level encoder, however (e.g., with ELMo, any char-CNN, etc.), or a combination of
    encoders (e.g., ELMo + glove), we need to construct a fake embedding matrix that we can use in
    `_first_order_taylor()`.  We do this by getting a list of words from the model's vocabulary
    and embedding them using the encoder.  This can be expensive, both in terms of time and memory
    usage, so we take a `max_tokens` parameter to limit the size of this fake embedding matrix.
    This also requires a model to `have` a token vocabulary in the first place, which can be
    problematic for models that only have character vocabularies.
    """
    def __init__(self, predictor: Predictor, vocab_namespace: str = 'tokens', max_tokens: int = 5000) -> None:
        super().__init__(predictor)
        self.vocab = self.predictor._model.vocab
        self.namespace: str = vocab_namespace
        self.max_tokens: int = max_tokens
        self.invalid_replacement_indices: List[int] = []
        for i in self.vocab._index_to_token[self.namespace]:
            if not self.vocab._index_to_token[self.namespace][i].isalnum():
                self.invalid_replacement_indices.append(i)
        self.embedding_matrix: Optional[torch.Tensor] = None
        self.embedding_layer: Optional[Any] = None
        self.cuda_device: int = predictor.cuda_device

    def initialize(self) -> None:
        """
        Call this function before running attack_from_json(). We put the call to
        `_construct_embedding_matrix()` in this function to prevent a large amount of compute
        being done when __init__() is called.
        """
        if self.embedding_matrix is None:
            self.embedding_matrix = self._construct_embedding_matrix()

    def _construct_embedding_matrix(self) -> torch.Tensor:
        """
        For HotFlip, we need a word embedding matrix to search over. The below is necessary for
        models such as ELMo, character-level models, or for models that use a projection layer
        after their word embeddings.

        We run all of the tokens from the vocabulary through the TextFieldEmbedder, and save the
        final output embedding. We then group all of those output embeddings into an "embedding
        matrix".
        """
        embedding_layer = self.predictor.get_interpretable_layer()
        self.embedding_layer = embedding_layer
        if isinstance(embedding_layer, (Embedding, torch.nn.modules.sparse.Embedding)):
            return embedding_layer.weight
        all_tokens: List[str] = list(self.vocab._token_to_index[self.namespace])[:self.max_tokens]
        max_index: int = self.vocab.get_token_index(all_tokens[-1], self.namespace)
        self.invalid_replacement_indices = [i for i in self.invalid_replacement_indices if i < max_index]
        inputs: Dict[str, Dict[str, torch.Tensor]] = self._make_embedder_input(all_tokens)
        embedding_matrix: torch.Tensor = embedding_layer(inputs).squeeze()
        return embedding_matrix

    def _make_embedder_input(self, all_tokens: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        inputs: Dict[str, Dict[str, torch.Tensor]] = {}
        indexers: Dict[str, Any] = self.predictor._dataset_reader._token_indexers
        for indexer_name, token_indexer in indexers.items():
            if isinstance(token_indexer, SingleIdTokenIndexer):
                all_indices: List[int] = [self.vocab._token_to_index[self.namespace][token] for token in all_tokens]
                inputs[indexer_name] = {'tokens': torch.LongTensor(all_indices).unsqueeze(0)}
            elif isinstance(token_indexer, TokenCharactersIndexer):
                tokens: List[Token] = [Token(x) for x in all_tokens]
                max_token_length: int = max((len(x) for x in all_tokens))
                max_token_length = max(max_token_length, token_indexer._min_padding_length)
                indexed_tokens: Any = token_indexer.tokens_to_indices(tokens, self.vocab)
                padding_lengths: Dict[str, int] = token_indexer.get_padding_lengths(indexed_tokens)
                padded_tokens: Dict[str, Any] = token_indexer.as_padded_tensor_dict(indexed_tokens, padding_lengths)
                inputs[indexer_name] = {'token_characters': torch.LongTensor(padded_tokens['token_characters']).unsqueeze(0)}
            elif isinstance(token_indexer, ELMoTokenCharactersIndexer):
                elmo_tokens: List[List[int]] = []
                for token in all_tokens:
                    elmo_indexed_token: Dict[str, Any] = token_indexer.tokens_to_indices([Token(text=token)], self.vocab)
                    elmo_tokens.append(elmo_indexed_token['elmo_tokens'][0])
                inputs[indexer_name] = {'elmo_tokens': torch.LongTensor(elmo_tokens).unsqueeze(0)}
            else:
                raise RuntimeError('Unsupported token indexer:', token_indexer)
        return util.move_to_device(inputs, self.cuda_device)

    def attack_from_json(self, 
                         inputs: JsonDict, 
                         input_field_to_attack: str = 'tokens', 
                         grad_input_field: str = 'grad_input_1', 
                         ignore_tokens: Optional[List[str]] = None, 
                         target: Optional[JsonDict] = None) -> JsonDict:
        """
        Replaces one token at a time from the input until the model's prediction changes.
        `input_field_to_attack` is for example `tokens`, it says what the input field is
        called.  `grad_input_field` is for example `grad_input_1`, which is a key into a grads
        dictionary.
        """
        instance: Instance = self.predictor._json_to_instance(inputs)
        self.predictor._dataset_reader.apply_token_indexers(instance)
        if target is None:
            output_dict: JsonDict = self.predictor._model.forward_on_instance(instance)
        else:
            output_dict = target
        ListOfInstances: List[Dict[str, Any]] = self.predictor.predictions_to_labeled_instances(instance, output_dict)
        original_text_field: TextField = ListOfInstances[0][input_field_to_attack]
        original_tokens: List[Token] = deepcopy(original_text_field.tokens)
        final_tokens: List[Any] = []
        final_outputs: List[Any] = []
        for instance_dict in ListOfInstances:
            tokens, outputs = self.attack_instance(instance=instance_dict, 
                                                     inputs=inputs, 
                                                     input_field_to_attack=input_field_to_attack, 
                                                     grad_input_field=grad_input_field, 
                                                     ignore_tokens=ignore_tokens, 
                                                     target=target)
            final_tokens.append(tokens)
            final_outputs.append(outputs)
        return sanitize({'final': final_tokens, 'original': original_tokens, 'outputs': final_outputs})

    def attack_instance(self, 
                        instance: Instance, 
                        inputs: JsonDict, 
                        input_field_to_attack: str = 'tokens', 
                        grad_input_field: str = 'grad_input_1', 
                        ignore_tokens: Optional[List[str]] = None, 
                        target: Optional[JsonDict] = None) -> Tuple[List[Token], Dict[str, Any]]:
        if self.embedding_matrix is None:
            self.initialize()
        ignore_tokens = DEFAULT_IGNORE_TOKENS if ignore_tokens is None else ignore_tokens
        sign: int = -1 if target is None else 1
        fields_to_compare: Any = utils.get_fields_to_compare(inputs, instance, input_field_to_attack)
        text_field: TextField = instance[input_field_to_attack]
        grads, outputs = self.predictor.get_gradients([instance])
        flipped: List[int] = []
        for index, token in enumerate(text_field.tokens):
            if token.text in ignore_tokens:
                flipped.append(index)
        if 'clusters' in outputs:
            for cluster in outputs['clusters']:
                for mention in cluster:
                    for index in range(mention[0], mention[1] + 1):
                        flipped.append(index)
        while True:
            grad = grads[grad_input_field][0]
            grads_magnitude: List[float] = [g.dot(g) for g in grad]
            for index in flipped:
                grads_magnitude[index] = -1
            index_of_token_to_flip: int = int(numpy.argmax(grads_magnitude))
            if grads_magnitude[index_of_token_to_flip] == -1:
                break
            flipped.append(index_of_token_to_flip)
            text_field_tensors: Dict[str, torch.Tensor] = text_field.as_tensor(text_field.get_padding_lengths())
            input_tokens: List[int] = util.get_token_ids_from_text_field_tensors(text_field_tensors)
            original_id_of_token_to_flip: torch.Tensor = torch.tensor(input_tokens[index_of_token_to_flip])
            new_id: int = self._first_order_taylor(grad[index_of_token_to_flip], original_id_of_token_to_flip, sign)
            new_token: Token = Token(self.vocab._index_to_token[self.namespace][new_id])
            text_field.tokens[index_of_token_to_flip] = new_token
            instance.indexed = False
            grads, outputs = self.predictor.get_gradients([instance])
            for key, output in outputs.items():
                if isinstance(output, torch.Tensor):
                    outputs[key] = output.detach().cpu().numpy().squeeze()
                elif isinstance(output, list):
                    outputs[key] = output[0]
            labeled_instance: Dict[str, Any] = self.predictor.predictions_to_labeled_instances(instance, outputs)[0]
            has_changed: bool = utils.instance_has_changed(labeled_instance, fields_to_compare)
            if target is None and has_changed:
                break
            if target is not None and (not has_changed):
                break
        return (text_field.tokens, outputs)

    def _first_order_taylor(self, grad: numpy.ndarray, token_idx: torch.Tensor, sign: int) -> int:
        """
        Replaces the current token_idx with another token_idx to increase the loss. In particular, this
        function uses the grad, alongside the embedding_matrix to select the token that maximizes the
        first-order taylor approximation of the loss.
        """
        grad_tensor: torch.Tensor = util.move_to_device(torch.from_numpy(grad), self.cuda_device)
        if token_idx.size() != ():
            raise NotImplementedError('You are using a character-level indexer with no other indexers. This case is not currently supported for hotflip. If you would really like to see us support this, please open an issue on github.')
        if token_idx.item() >= self.embedding_matrix.size(0):
            inputs: Dict[str, Dict[str, torch.Tensor]] = self._make_embedder_input([self.vocab.get_token_from_index(token_idx.item())])
            word_embedding: torch.Tensor = self.embedding_layer(inputs)[0]
        else:
            word_embedding = torch.nn.functional.embedding(util.move_to_device(torch.LongTensor([token_idx.item()]), self.cuda_device), self.embedding_matrix)
        word_embedding = word_embedding.detach().unsqueeze(0)
        grad_tensor = grad_tensor.unsqueeze(0).unsqueeze(0)
        new_embed_dot_grad: torch.Tensor = torch.einsum('bij,kj->bik', (grad_tensor, self.embedding_matrix))
        prev_embed_dot_grad: torch.Tensor = torch.einsum('bij,bij->bi', (grad_tensor, word_embedding)).unsqueeze(-1)
        neg_dir_dot_grad: torch.Tensor = sign * (prev_embed_dot_grad - new_embed_dot_grad)
        neg_dir_dot_grad = neg_dir_dot_grad.detach().cpu().numpy()
        neg_dir_dot_grad[:, :, self.invalid_replacement_indices] = -numpy.inf
        best_at_each_step = neg_dir_dot_grad.argmax(2)
        return int(best_at_each_step[0].data[0])