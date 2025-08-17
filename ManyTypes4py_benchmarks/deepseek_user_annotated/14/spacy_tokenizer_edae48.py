from typing import List, Optional, Dict, Any, Union

import spacy
from spacy.tokens import Doc, Token as SpacyToken
from spacy.vocab import Vocab

from allennlp.common.util import get_spacy_model
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("spacy")
class SpacyTokenizer(Tokenizer):
    def __init__(
        self,
        language: str = "en_core_web_sm",
        pos_tags: bool = True,
        parse: bool = False,
        ner: bool = False,
        keep_spacy_tokens: bool = False,
        split_on_spaces: bool = False,
        start_tokens: Optional[List[str]] = None,
        end_tokens: Optional[List[str]] = None,
    ) -> None:
        self._language = language
        self._pos_tags = pos_tags
        self._parse = parse
        self._ner = ner
        self._split_on_spaces = split_on_spaces

        self.spacy = get_spacy_model(self._language, self._pos_tags, self._parse, self._ner)

        if self._split_on_spaces:
            self.spacy.tokenizer = _WhitespaceSpacyTokenizer(self.spacy.vocab)

        self._keep_spacy_tokens = keep_spacy_tokens
        self._start_tokens = start_tokens or []
        self._start_tokens.reverse()
        self._is_version_3 = spacy.__version__ >= "3.0"
        self._end_tokens = end_tokens or []

    def _sanitize(self, tokens: List[SpacyToken]) -> List[Token]:
        if not self._keep_spacy_tokens:
            tokens = [
                Token(
                    token.text,
                    token.idx,
                    token.idx + len(token.text),
                    token.lemma_,
                    token.pos_,
                    token.tag_,
                    token.dep_,
                    token.ent_type_,
                )
                for token in tokens
            ]
        for start_token in self._start_tokens:
            tokens.insert(0, Token(start_token, 0))
        for end_token in self._end_tokens:
            tokens.append(Token(end_token, -1))
        return tokens

    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        if self._is_version_3:
            return [
                self._sanitize(_remove_spaces(tokens))
                for tokens in self.spacy.pipe(texts, n_process=-1)
            ]
        else:
            return [
                self._sanitize(_remove_spaces(tokens))
                for tokens in self.spacy.pipe(texts, n_threads=-1)
            ]

    def tokenize(self, text: str) -> List[Token]:
        return self._sanitize(_remove_spaces(self.spacy(text)))

    def _to_params(self) -> Dict[str, Any]:
        return {
            "type": "spacy",
            "language": self._language,
            "pos_tags": self._pos_tags,
            "parse": self._parse,
            "ner": self._ner,
            "keep_spacy_tokens": self._keep_spacy_tokens,
            "split_on_spaces": self._split_on_spaces,
            "start_tokens": self._start_tokens,
            "end_tokens": self._end_tokens,
        }


class _WhitespaceSpacyTokenizer:
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def __call__(self, text: str) -> Doc:
        words = text.split(" ")
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def _remove_spaces(tokens: List[SpacyToken]) -> List[SpacyToken]:
    return [token for token in tokens if not token.is_space]
