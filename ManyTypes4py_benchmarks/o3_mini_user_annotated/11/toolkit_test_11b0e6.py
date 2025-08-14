import pytest
import torch
from torch import Tensor
from torch.testing import assert_allclose
from typing import Dict, Any, Tuple, List

from transformers import AutoModel
from transformers.models.albert.modeling_albert import AlbertEmbeddings

from allennlp.common import cached_transformers
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding, TokenEmbedder
from allennlp.modules.transformer import TransformerStack, TransformerEmbeddings, TransformerPooler
from allennlp.common.testing import AllenNlpTestCase


class TestTransformerToolkit(AllenNlpTestCase):
    def setup_method(self, method: Any) -> None:
        super().setup_method(method)
        self.vocab: Vocabulary = Vocabulary()
        # populate vocab.
        self.vocab.add_token_to_namespace("word")
        self.vocab.add_token_to_namespace("the")
        self.vocab.add_token_to_namespace("an")

    def test_create_embedder_using_toolkit(self) -> None:
        embedding_file: str = str(self.FIXTURES_ROOT / "embeddings/glove.6B.300d.sample.txt.gz")

        class TinyTransformer(TokenEmbedder):
            def __init__(self, vocab: Vocabulary, embedding_dim: int, hidden_size: int, intermediate_size: int) -> None:
                super().__init__()
                self.embeddings: Embedding = Embedding(
                    pretrained_file=embedding_file,
                    embedding_dim=embedding_dim,
                    projection_dim=hidden_size,
                    vocab=vocab,
                )

                self.transformer: TransformerStack = TransformerStack(
                    num_hidden_layers=4,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                )

            def forward(self, token_ids: torch.LongTensor) -> Tensor:
                x: Tensor = self.embeddings(token_ids)
                x = self.transformer(x)
                return x

        tiny: TinyTransformer = TinyTransformer(self.vocab, embedding_dim=300, hidden_size=80, intermediate_size=40)
        tiny.forward(torch.LongTensor([[0, 1, 2]]))

    def test_use_first_four_layers_of_pretrained(self) -> None:
        pretrained: str = "bert-base-cased"

        class SmallTransformer(TokenEmbedder):
            def __init__(self) -> None:
                super().__init__()
                self.embeddings: TransformerEmbeddings = TransformerEmbeddings.from_pretrained_module(
                    pretrained, relevant_module="bert.embeddings"
                )
                self.transformer: TransformerStack = TransformerStack.from_pretrained_module(
                    pretrained,
                    num_hidden_layers=4,
                    relevant_module="bert.encoder",
                    strict=False,
                )

            def forward(self, token_ids: torch.LongTensor) -> Tensor:
                x: Tensor = self.embeddings(token_ids)
                x = self.transformer(x)
                return x

        small: SmallTransformer = SmallTransformer()
        assert len(small.transformer.layers) == 4
        small(torch.LongTensor([[0, 1, 2]]))

    def test_use_selected_layers_of_bert_for_different_purposes(self) -> None:
        class MediumTransformer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embeddings: TransformerEmbeddings = TransformerEmbeddings.from_pretrained_module(
                    "bert-base-cased", relevant_module="bert.embeddings"
                )
                self.separate_transformer: TransformerStack = TransformerStack.from_pretrained_module(
                    "bert-base-cased",
                    relevant_module="bert.encoder",
                    num_hidden_layers=8,
                    strict=False,
                )
                self.combined_transformer: TransformerStack = TransformerStack.from_pretrained_module(
                    "bert-base-cased",
                    relevant_module="bert.encoder",
                    num_hidden_layers=4,
                    mapping={f"layer.{l}": f"layers.{i}" for (i, l) in enumerate(range(8, 12))},
                    strict=False,
                )

            def forward(
                self,
                left_token_ids: torch.LongTensor,
                right_token_ids: torch.LongTensor,
            ) -> Tensor:
                left: Tensor = self.embeddings(left_token_ids)
                left = self.separate_transformer(left)

                right: Tensor = self.embeddings(right_token_ids)
                right = self.separate_transformer(right)

                # combine the sequences in some meaningful way. here, we just add them.
                combined: Tensor = left + right

                return self.combined_transformer(combined)

        medium: MediumTransformer = MediumTransformer()
        assert len(medium.separate_transformer.layers) == 8
        assert len(medium.combined_transformer.layers) == 4

        pretrained_model: Any = cached_transformers.get("bert-base-cased", False)
        pretrained_layers: Dict[str, Any] = dict(pretrained_model.encoder.layer.named_modules())

        separate_layers: Dict[str, Any] = dict(medium.separate_transformer.layers.named_modules())
        assert_allclose(
            separate_layers["0"].intermediate.dense.weight.data,
            pretrained_layers["0"].intermediate.dense.weight.data,
        )

        combined_layers: Dict[str, Any] = dict(medium.combined_transformer.layers.named_modules())
        assert_allclose(
            combined_layers["0"].intermediate.dense.weight.data,
            pretrained_layers["8"].intermediate.dense.weight.data,
        )
        assert_allclose(
            combined_layers["1"].intermediate.dense.weight.data,
            pretrained_layers["9"].intermediate.dense.weight.data,
        )
        assert_allclose(
            combined_layers["2"].intermediate.dense.weight.data,
            pretrained_layers["10"].intermediate.dense.weight.data,
        )
        assert_allclose(
            combined_layers["3"].intermediate.dense.weight.data,
            pretrained_layers["11"].intermediate.dense.weight.data,
        )

    def test_combination_of_two_different_berts(self) -> None:
        # Regular BERT, but with AlBERT's special compressed embedding scheme

        class AlmostRegularTransformer(TokenEmbedder):
            def __init__(self) -> None:
                super().__init__()
                self.embeddings: Any = AutoModel.from_pretrained("albert-base-v2").embeddings
                self.transformer: TransformerStack = TransformerStack.from_pretrained_module(
                    "bert-base-cased", relevant_module="bert.encoder"
                )
                # We want to tune only the embeddings, because that's our experiment.
                self.transformer.requires_grad = False

            def forward(self, token_ids: torch.LongTensor, mask: torch.BoolTensor) -> Tensor:
                x: Tensor = self.embeddings(token_ids, mask)
                x = self.transformer(x)
                return x

        almost: AlmostRegularTransformer = AlmostRegularTransformer()
        assert len(almost.transformer.layers) == 12
        # type checking for instance of AlbertEmbeddings is retained.
        assert isinstance(almost.embeddings, AlbertEmbeddings)

    @pytest.mark.parametrize("model_name", ["bert-base-cased", "roberta-base"])
    def test_end_to_end(self, model_name: str) -> None:
        data: List[Tuple[str, str]] = [
            ("I'm against picketing", "but I don't know how to show it."),
            ("I saw a human pyramid once.", "It was very unnecessary."),
        ]
        tokenizer: Any = cached_transformers.get_tokenizer(model_name)
        batch: Dict[str, Tensor] = tokenizer.batch_encode_plus(data, padding=True, return_tensors="pt")  # type: ignore

        with torch.no_grad():
            huggingface_model: Any = cached_transformers.get(model_name, make_copy=False).eval()
            huggingface_output: Any = huggingface_model(**batch)

            embeddings: TransformerEmbeddings = TransformerEmbeddings.from_pretrained_module(model_name).eval()
            transformer_stack: TransformerStack = TransformerStack.from_pretrained_module(model_name).eval()
            pooler: TransformerPooler = TransformerPooler.from_pretrained_module(model_name).eval()
            batch["attention_mask"] = batch["attention_mask"].to(torch.bool)
            output: Any = embeddings(**batch)
            output = transformer_stack(output, batch["attention_mask"])

            assert_allclose(
                output.final_hidden_states,
                huggingface_output.last_hidden_state,
                rtol=0.0001,
                atol=1e-4,
            )

            output_final: Tensor = pooler(output.final_hidden_states)
            assert_allclose(output_final, huggingface_output.pooler_output, rtol=0.0001, atol=1e-4)